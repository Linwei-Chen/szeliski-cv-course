#!/usr/bin/env python3
"""E124 NeRF Ⅱ：体积渲染积分、位置编码取舍、1/r 无界场景玩具。

依赖：numpy/matplotlib。
边界：一维解析密度场 + 一维信号回归，不是完整 NeRF 训练；用它们验证公式与工程取舍。
教材：§14.6 p.906–908 的扩展/加速段；体积渲染公式与分层采样按原始 NeRF 展开，
      不冒充原书原文；原书只给结论，不给本 lab 的数值。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"


def sigma(z):
    return 40.0 * np.exp(-((z - 0.25) / 0.03) ** 2)


def color(z):
    return 0.2 + 0.8 * z


def render(samples):
    """分段常数体积渲染：返回 (C, A, depth, w)。samples 为升序深度。"""
    z = np.sort(np.asarray(samples, dtype=float)); n = z.size
    if n == 1:
        dz = np.array([1.0])
    else:
        mid = (z[1:] + z[:-1]) / 2
        dz = np.r_[mid[0], mid[1:] - mid[:-1], 1.0 - mid[-1]]
        dz = np.maximum(dz, 1e-12)
    s = sigma(z); a = 1 - np.exp(-s * dz)
    T = np.cumprod(np.r_[1.0, 1 - a[:-1]])
    w = T * a; A = float(w.sum())
    C = float((w * color(z)).sum())
    D = float((w * z).sum() / max(A, 1e-12))
    return C, A, D, w


def uniform(n):
    return (np.arange(n) + 0.5) / n


z_ref = uniform(200000)
C_ref, A_ref, D_ref, _ = render(z_ref)
analytic_A = float(1 - np.exp(-40.0 * 0.03 * np.sqrt(np.pi)))

Ns = [8, 16, 32, 64, 128, 256]
errs_C, errs_D, alphas = [], [], []
for n in Ns:
    C, A, D, _ = render(uniform(n))
    errs_C.append(abs(C - C_ref)); errs_D.append(abs(D - D_ref)); alphas.append(A)

# 分层采样：64 粗 + 64 细（细采样窗由粗权重均值±5σ 决定）
z_coarse = uniform(64); Cc, Ac, Dc, wc = render(z_coarse)
mu = float((wc * z_coarse).sum() / wc.sum())
sd = float(np.sqrt((wc * (z_coarse - mu) ** 2).sum() / wc.sum()))
lo, hi = max(0.0, mu - 5 * sd), min(1.0, mu + 5 * sd)
z_fine = np.linspace(lo, hi, 64)
z_adapt = np.sort(np.r_[z_coarse, z_fine])
C_adapt, A_adapt, D_adapt, _ = render(z_adapt)
C_u128, A_u128, D_u128, _ = render(uniform(128))
C_u64, A_u64, D_u64, _ = render(z_coarse)
C_adapt_err, D_adapt_err = abs(C_adapt-C_ref), abs(D_adapt-D_ref)
C_u128_err, D_u128_err = abs(C_u128-C_ref), abs(D_u128-D_ref)
C_u64_err, D_u64_err = abs(C_u64-C_ref), abs(D_u64-D_ref)

# 位置编码频率取舍：1D 带噪信号，训练 120 点，测试 600 点
rng = np.random.default_rng(7)
def target(x):
    return 0.5 + 0.4 * np.sin(2 * np.pi * x) + 0.2 * np.sin(10 * np.pi * x) + 0.08 * np.sin(30 * np.pi * x)
x_train = np.sort(rng.uniform(0, 1, 120)); y_train = target(x_train) + rng.normal(0, 0.03, x_train.size)
x_test = np.linspace(0, 1, 600); y_test = target(x_test)


def feat(x, L):
    fs = []
    for l in range(L):
        f = 2 ** l * np.pi
        fs += [np.sin(f * x), np.cos(f * x)]
    return np.stack(fs, -1) if fs else x[:, None]


class MLP:
    def __init__(self, din, h=64, seed=1):
        r = np.random.default_rng(seed); sizes = [din, h, h, 1]
        self.W = [r.normal(0, np.sqrt(2 / (a + b)), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]; self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.tanh(z) if i < len(self.W) - 1 else z)
        return hs

    def loss_grad(self, x, y):
        hs = self.forward(x); out = hs[-1]; mse = float(np.mean((out - y) ** 2))
        g = (out - y) * 2 / out.size
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1 - hs[i] ** 2)
        return mse, gW, gb

    def step(self, gW, gb, lr=0.01, b1=0.9, b2=0.999):
        self.t += 1
        for i in range(len(self.W)):
            for target_, g, m, v in ((self.W[i], gW[i], self.mW[i], self.vW[i]),
                                     (self.b[i], gb[i], self.mb[i], self.vb[i])):
                m[:] = b1 * m + (1 - b1) * g; v[:] = b2 * v + (1 - b2) * g * g
                target_ -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + 1e-8)

    def n_params(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))


def train_L(L, steps=900, seed=1):
    Xf = feat(x_train, L); net = MLP(Xf.shape[1], 64, seed)
    for _ in range(steps):
        mse, gW, gb = net.loss_grad(Xf, y_train[:, None]); net.step(gW, gb)
    pred_train = net.forward(Xf)[-1][:, 0]
    pred_test = net.forward(feat(x_test, L))[-1][:, 0]
    return (float(np.mean((pred_train - y_train) ** 2)),
            float(np.mean((pred_test - y_test) ** 2)), pred_test, net.n_params())


Ls = list(range(0, 8))
train_mse, test_mse, preds, params_L = [], [], {}, {}
for L in Ls:
    tr, te, pred, npar = train_L(L)
    train_mse.append(tr); test_mse.append(te); preds[L] = pred; params_L[L] = npar

# NeRF++ 1/r 参数化：r 均匀 → u=1/r 均匀；Δr = r² Δu
r_near, r_far = 1.0, 100.0
u_near, u_far = 1 / r_near, 1 / r_far
u = np.linspace(u_far, u_near, 4000)
r = 1 / u
res_ratio_10x = (10.0 / 1.0) ** 2
res_ratio_100x = (100.0 / 1.0) ** 2
# 用 64 个均匀 u 采样，量测相邻 r 间隔的远/近比
u64 = np.linspace(u_far, u_near, 64); r64 = 1 / u64
gaps = np.diff(np.sort(r64))
gap_ratio = float(gaps[-1] / gaps[0])

checks = [
 ("解析 alpha 与数值积分一致", abs(A_ref - analytic_A) < 1e-4, f"{A_ref:.6f} vs {analytic_A:.6f}"),
 ("256 样本积分误差小于 1e-4", errs_C[-1] < 1e-4 and errs_D[-1] < 1e-4, f"C {errs_C[-1]:.2e}, D {errs_D[-1]:.2e}"),
 ("积分误差随样本数单调下降", all(errs_C[i] > errs_C[i+1] for i in range(len(Ns)-1)), "C 误差 " + "→".join(f"{e:.1e}" for e in errs_C)),
 ("分层采样优于 128 均匀（颜色）", C_adapt_err < 0.25 * C_u128_err, f"{C_adapt_err:.2e} < 0.25×{C_u128_err:.2e}"),
 ("分层采样优于 128 均匀（深度）", D_adapt_err < 0.25 * D_u128_err, f"{D_adapt_err:.2e} < 0.25×{D_u128_err:.2e}"),
 ("位置编码 L=4 测试误差低于 1e-2", test_mse[4] < 1e-2, f"{test_mse[4]:.2e}"),
 ("位置编码 L=4 远优于裸坐标 L=0", test_mse[4] < 0.1 * test_mse[0], f"{test_mse[4]:.2e} < 0.1×{test_mse[0]:.2e}"),
 ("频率过高 L=7 测试误差反弹", test_mse[7] > 3 * test_mse[4], f"{test_mse[7]:.2e} > 3×{test_mse[4]:.2e}"),
 ("高 L 训练误差下降但测试变差", train_mse[7] < train_mse[2] and test_mse[7] > test_mse[4], f"train {train_mse[7]:.2e} < {train_mse[2]:.2e}; test {test_mse[7]:.2e} > {test_mse[4]:.2e}"),
 ("1/r 参数化把 10 倍距离变成 100 倍分辨率差", abs(res_ratio_10x - 100) < 1e-9 and abs(res_ratio_100x - 10000) < 1e-9, f"10×→{res_ratio_10x:.0f}×, 100×→{res_ratio_100x:.0f}×"),
 ("64 个均匀 u 样本的远/近间隔比超过 100", gap_ratio > 100, f"gaps {gaps[-1]:.3f} / {gaps[0]:.5f} = {gap_ratio:.0f}×"),
 ("位置编码维度随 L 线性增长", params_L[6] > params_L[2] and feat(x_train, 6).shape[1] == 12 and feat(x_train, 0).shape[1] == 1, f"L=6 输入 12 维（2L，不含原始坐标）, 参数 {params_L[6]} vs L=2 {params_L[2]}"),
]
n_pass = 0
print(f"[lab] E124 NeRF Ⅱ 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：一维解析密度场+一维信号回归；不是完整 NeRF 训练；三图程序生成")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 积分收敛与分层采样
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
axes[0].loglog(Ns, errs_C, "o-", color=BLUE, lw=2, label="颜色 C 误差")
axes[0].loglog(Ns, errs_D, "s--", color=GREEN, lw=2, label="深度误差")
axes[0].set_xlabel("每条射线样本数", fontsize=19); axes[0].set_ylabel("绝对误差（对数）", fontsize=19)
axes[0].tick_params(labelsize=17); axes[0].set_title("均匀积分：样本越多越准", fontsize=19); axes[0].legend(fontsize=17)
axes[1].plot(np.linspace(0,1,500), sigma(np.linspace(0,1,500)), color=ACCENT, lw=3, label="密度 σ(z)")
axes[1].scatter(z_coarse, np.full_like(z_coarse, 5), s=18, color=GRAY, label="粗采样 64", zorder=3)
axes[1].scatter(z_fine, np.full_like(z_fine, 2), s=18, color=GREEN, label="细采样 64", zorder=3)
axes[1].set_xlabel("沿射线的深度 z", fontsize=19); axes[1].set_ylabel("密度 / 采样位置", fontsize=19)
axes[1].tick_params(labelsize=17); axes[1].set_title("分层采样：把样本放到高密度区", fontsize=19); axes[1].legend(fontsize=17, loc="upper right")
fig.suptitle("体积渲染：先粗后细，把样本花在刀刃上", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e124_quadrature.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 位置编码 L 取舍
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
axes[0].plot(Ls, train_mse, "o-", color=BLUE, lw=2, label="训练 MSE")
axes[0].plot(Ls, test_mse, "s-", color=ACCENT, lw=2.5, label="测试 MSE")
axes[0].axvline(4, color=GRAY, ls=":", lw=1.5)
axes[0].set_yscale("log"); axes[0].set_xlabel("位置编码倍频程数 L", fontsize=19); axes[0].set_ylabel("MSE（对数）", fontsize=19)
axes[0].tick_params(labelsize=17); axes[0].set_title("L 太小欠拟合，太大过拟合", fontsize=19); axes[0].legend(fontsize=17)
axes[1].plot(x_test, y_test, color=GRAY, lw=2, label="真值")
axes[1].scatter(x_train, y_train, s=14, color=INK, alpha=0.6, label="带噪训练点")
for L, col in [(1, BLUE), (4, GREEN), (7, ACCENT)]:
    axes[1].plot(x_test, preds[L], color=col, lw=2, label=f"L={L}")
axes[1].set_xlabel("输入 x", fontsize=19); axes[1].set_ylabel("信号值", fontsize=19)
axes[1].tick_params(labelsize=17); axes[1].set_title("L=4 抓住细节，L=7 追噪声", fontsize=19)
axes[1].legend(fontsize=17, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
fig.suptitle("位置编码不是越多越好：训练误差和测试误差要分开看", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.subplots_adjust(bottom=0.30)
fig.savefig(FIG / "e124_encoding.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 1/r 参数化与深度分辨率
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
axes[0].plot(r, u, color=BLUE, lw=2.5)
axes[0].set_xscale("log"); axes[0].set_xlabel("欧氏距离 r（对数）", fontsize=19); axes[0].set_ylabel("反转球坐标 u = 1/r", fontsize=19)
axes[0].tick_params(labelsize=17); axes[0].set_title("远场景被压到原点附近", fontsize=19)
axes[1].loglog(r, r**2, color=ACCENT, lw=2.5)
axes[1].set_xlabel("欧氏距离 r（对数）", fontsize=19); axes[1].set_ylabel("同样 Δu 对应的 Δr（对数）", fontsize=19)
axes[1].tick_params(labelsize=17); axes[1].set_title("远处分辨率按 r² 变粗", fontsize=19)
fig.suptitle("NeRF++ 的 1/r 参数化：无界场景的取舍", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e124_unbounded.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks),
                C_ref=C_ref, A_ref=A_ref, D_ref=D_ref, analytic_A=analytic_A,
                Ns=Ns, errs_C=errs_C, errs_D=errs_D, alphas=alphas,
                C_adapt=C_adapt, D_adapt=D_adapt, C_u128=C_u128, D_u128=D_u128,
                C_adapt_err=C_adapt_err, D_adapt_err=D_adapt_err, C_u128_err=C_u128_err, D_u128_err=D_u128_err,
                C_u64_err=C_u64_err, D_u64_err=D_u64_err,
                C_u64=C_u64, D_u64=D_u64, adaptive_window=[lo, hi],
                Ls=Ls, train_mse=train_mse, test_mse=test_mse, params_L={str(k): v for k, v in params_L.items()},
                r_near=r_near, r_far=r_far, res_ratio_10x=res_ratio_10x, res_ratio_100x=res_ratio_100x,
                gap_ratio=gap_ratio, gap_min=float(gaps[0]), gap_max=float(gaps[-1]))
(OUT / "e124_nerf2_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for name in ("e124_quadrature.png", "e124_encoding.png", "e124_unbounded.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e124_nerf2_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
