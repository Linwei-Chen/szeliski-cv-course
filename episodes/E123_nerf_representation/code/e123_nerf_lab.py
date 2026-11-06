#!/usr/bin/env python3
"""E123 NeRF 表示：二维玩具辐射场 + 位置编码 + 一维体积渲染。

依赖：numpy/matplotlib。
边界：二维图像拟合不是完整 NeRF；一维体积渲染是同一套 alpha 合成公式的切片。
原书第 13 章 p.833 只有 NeRF 桥接，输入/位置编码/MLP 细节在第 14 章 §14.6 p.905–907。
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
rng = np.random.default_rng(123)
N = 64; xs = np.linspace(-1, 1, N); X, Y = np.meshgrid(xs, xs)
K = 6


def target_low():
    return np.stack([0.5 + 0.3*X, 0.5 + 0.3*Y, np.full_like(X, 0.5)], -1)


def target_high():
    check = ((np.floor((X + 1) * 12) + np.floor((Y + 1) * 12)) % 2)
    return np.stack([0.1 + 0.8*check,
                     0.1 + 0.8*(1-check),
                     0.5 + 0.2*np.sin(20*X)], -1)


def feat_raw(x, y):
    return np.stack([x, y], -1)


def feat_fourier(x, y, K=K):
    out = []
    for k in range(K):
        f = 2**k * np.pi
        out += [np.sin(f*x), np.cos(f*x), np.sin(f*y), np.cos(f*y)]
    return np.stack(out, -1)


class MLP:
    def __init__(self, din, h=64, seed=0):
        r = np.random.default_rng(seed)
        sizes = [din, h, h, 3]
        self.W = [r.normal(0, np.sqrt(2/(a+b)), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.mW = [np.zeros_like(w) for w in self.W]; self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]; self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.tanh(z) if i < len(self.W)-1 else 1/(1+np.exp(-z)))
        return hs

    def loss_grad(self, x, y):
        hs = self.forward(x); out = hs[-1]; mse = float(np.mean((out-y)**2))
        g = (out-y) * 2/out.size * out * (1-out)
        gW = [None]*len(self.W); gb = [None]*len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (1-hs[i]**2)
        return mse, gW, gb

    def step(self, gW, gb, lr=0.005, b1=0.9, b2=0.999):
        self.t += 1
        for i in range(len(self.W)):
            for target, g, m, v in ((self.W[i], gW[i], self.mW[i], self.vW[i]),
                                    (self.b[i], gb[i], self.mb[i], self.vb[i])):
                m[:] = b1*m + (1-b1)*g; v[:] = b2*v + (1-b2)*g*g
                mh = m/(1-b1**self.t); vh = v/(1-b2**self.t)
                target -= lr * mh/(np.sqrt(vh)+1e-8)

    def n_params(self):
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))


def train(feat, target, steps=1200, seed=0):
    Xf = feat(X.ravel(), Y.ravel()); Yf = target.reshape(-1, 3)
    net = MLP(Xf.shape[1], 64, seed); hist = []
    for _ in range(steps):
        mse, gW, gb = net.loss_grad(Xf, Yf); hist.append(mse); net.step(gW, gb)
    out = net.forward(Xf)[-1].reshape(N, N, 3)
    return net, hist, out


def render(sigma, color, z):
    dz = np.diff(np.r_[z, z[-1] + (z[-1]-z[-2])])
    alpha = 1 - np.exp(-sigma*dz)
    T = np.cumprod(np.r_[1.0, 1-alpha[:-1]])
    w = T*alpha
    return w, float(np.sum(w*color)), float(np.sum(w*z)), float(np.sum(w))


net_raw_low, hist_raw_low, out_raw_low = train(feat_raw, target_low(), steps=800, seed=2)
net_raw, hist_raw, out_raw = train(feat_raw, target_high(), steps=1200, seed=1)
net_f, hist_f, out_f = train(feat_fourier, target_high(), steps=1200, seed=1)
mse_raw_low = float(np.mean((out_raw_low-target_low())**2))
mse_raw = float(np.mean((out_raw-target_high())**2))
mse_f = float(np.mean((out_f-target_high())**2))
freqs = np.pi * 2**np.arange(K)
grid_params = 64**3*3
ratio = net_f.n_params()/grid_params

z = np.linspace(0, 1, 64)
sigma = 20*np.exp(-((z-0.25)/0.08)**2)
w, C, depth, A = render(sigma, np.ones_like(z), z)
w0, C0, depth0, A0 = render(np.zeros_like(z), np.ones_like(z), z)
sig2 = np.array([10.0, 0.0]); col2 = np.array([1.0, 0.0]); z2 = np.array([0.25, 0.75])
w2, C2, depth2, A2 = render(sig2, col2, z2)
manual = (1-np.exp(-10*0.5))*1.0

checks = [
 ("位置编码频率是倍频程", np.allclose(freqs, np.pi*2**np.arange(K)), "、".join(f"{f/np.pi:g}π" for f in freqs)),
 ("裸坐标 MLP 能拟合低频图像", mse_raw_low < 1e-4, f"MSE={mse_raw_low:.2e}"),
 ("裸坐标 MLP 拟合不了高频细节", mse_raw > 0.02, f"MSE={mse_raw:.4f}"),
 ("傅里叶特征显著改善高频细节", mse_f < 0.01 and mse_f < 0.25*mse_raw, f"{mse_f:.4f} < 0.25×{mse_raw:.4f}"),
 ("MLP 参数量小于 64³ 网格的 1%", ratio < 0.01, f"{net_f.n_params()} / {grid_params} = {100*ratio:.2f}%"),
 ("体积权重非负且总和 ≤1", w.min() >= 0 and A <= 1+1e-12, f"min={w.min():.2e}, Σw={A:.4f}"),
 ("常色射线的合成色等于色值×alpha", abs(C - A) < 1e-12, f"C={C:.6f}, A={A:.6f}"),
 ("近处高密度样本主导深度", depth < 0.35, f"depth={depth:.4f}"),
 ("空场景 alpha 接近 0", A0 < 1e-9, f"alpha={A0:.2e}"),
 ("两样本手算与代码一致", abs(C2-manual) < 1e-12, f"{C2:.6f} vs {manual:.6f}"),
]
n_pass = 0
print(f"[lab] E123 NeRF 表示开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：二维图像拟合+一维体积渲染切片；不是完整 NeRF 训练/渲染")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 目标/裸坐标/傅里叶
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
for ax, img, title in [(axes[0], target_high(), "目标图像"),
                       (axes[1], np.clip(out_raw, 0, 1), "裸坐标 MLP"),
                       (axes[2], np.clip(out_f, 0, 1), "傅里叶特征 MLP")]:
    ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=18, color=INK)
fig.suptitle("把图像写进 MLP：位置编码决定能不能看见细节", fontsize=21, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIG / "e123_nerf_fit.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 损失曲线与最终 MSE
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
axes[0].semilogy(hist_raw, label="裸坐标", color=BLUE, lw=2)
axes[0].semilogy(hist_f, label="傅里叶特征", color=GREEN, lw=2)
axes[0].set_xlabel("训练步数", fontsize=19); axes[0].set_ylabel("MSE（对数）", fontsize=19)
axes[0].tick_params(labelsize=17); axes[0].set_title("训练损失", fontsize=19); axes[0].legend(fontsize=19)
vals = [mse_raw, mse_f]; bars = axes[1].bar(["裸坐标", "傅里叶特征"], vals, color=[BLUE, GREEN], width=0.55, log=True)
axes[1].set_ylabel("最终 MSE（对数）", fontsize=19); axes[1].tick_params(labelsize=17)
axes[1].set_title("高频目标上的最终误差", fontsize=19)
axes[1].set_ylim(1e-4, max(vals) * 3.0)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x()+b.get_width()/2, v*1.4, f"{v:.4f}", ha="center", fontsize=19)
fig.suptitle("位置编码让 MLP 从“糊”变“清楚”", fontsize=21, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e123_spectral.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 体积渲染权重
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
axes[0].plot(z, sigma, color=ACCENT, lw=2.5)
axes[0].set_xlabel("沿射线的深度 z", fontsize=19); axes[0].set_ylabel("密度 σ(z)", fontsize=19)
axes[0].tick_params(labelsize=17); axes[0].set_title("一条射线上的密度", fontsize=19)
axes[1].plot(z, w, color=GREEN, lw=2.5, label="采样权重 w(z)")
axes[1].plot(z, np.cumsum(w), color=BLUE, lw=2, ls="--", label="累计 alpha")
axes[1].set_xlabel("沿射线的深度 z", fontsize=19); axes[1].set_ylabel("权重", fontsize=19)
axes[1].tick_params(labelsize=17); axes[1].set_title("alpha 合成：近处先占权重", fontsize=19)
axes[1].legend(fontsize=19)
fig.suptitle("体积渲染：密度→透明度→沿射线加权", fontsize=21, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e123_volume.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), grid=N, K=K,
                mse_raw_low=mse_raw_low, mse_raw=mse_raw, mse_f=mse_f,
                params_raw=net_raw.n_params(), params_fourier=net_f.n_params(),
                grid_params=grid_params, param_ratio=ratio,
                alpha=A, depth=depth, color_const=C, alpha_empty=A0,
                two_sample_C=C2, two_sample_manual=manual,
                freqs=[float(f) for f in freqs])
(OUT / "e123_nerf_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for name in ("e123_nerf_fit.png", "e123_spectral.png", "e123_volume.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e123_nerf_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
