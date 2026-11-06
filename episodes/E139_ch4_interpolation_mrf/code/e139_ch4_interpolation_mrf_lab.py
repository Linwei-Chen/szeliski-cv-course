#!/usr/bin/env python3
"""E139 第4章补遗 lab：RBF 散乱插值 + Tikhonov/TV 正则化 + 二值 MRF（ICM）。

边界：合成数据、三种教学基线；不实现图割/信念传播/dense CRF；
不是通用 benchmark，参数与分数只服务本集概念对照。
"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; GOLD = "#C9971C"; PURPLE = "#7A5A8E"
for cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(cand).exists():
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = PAPER
plt.rcParams["axes.facecolor"] = PAPER
plt.rcParams["savefig.facecolor"] = PAPER
plt.rcParams["text.color"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK
plt.rcParams["axes.edgecolor"] = INK
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 15

print(f"[lab] E139 第4章补遗实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] NumPy {np.__version__}")
print("[lab] 边界：合成数据、RBF/Tikhonov/TV/ICM 四基线；不实现图割/BP/dense CRF；不是通用 benchmark")

# ── 1. RBF 散乱插值 ─────────────────────────────────────────────
rng = np.random.default_rng(139)
def true_f(x, y):
    return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 0.3 * x

N = 60
X = rng.random((N, 2))
z = true_f(X[:, 0], X[:, 1]) + rng.normal(0, 0.0, N)
gx, gy = np.meshgrid(np.linspace(0, 1, 80), np.linspace(0, 1, 80))
Xq = np.c_[gx.ravel(), gy.ravel()]
zq = true_f(Xq[:, 0], Xq[:, 1])

def rbf_fit(X, z, Xq, c=0.25, lam=1e-6):
    K = np.exp(-cdist(X, X, "sqeuclidean") / c ** 2)
    w = np.linalg.solve(K + lam * np.eye(len(X)), z)
    Kq = np.exp(-cdist(Xq, X, "sqeuclidean") / c ** 2)
    return Kq @ w

def nearest_fit(X, z, Xq):
    return z[cdist(Xq, X).argmin(axis=1)]

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

pred_clean = rbf_fit(X, z, Xq, lam=1e-6)
pred_nearest = nearest_fit(X, z, Xq)
rmse_clean = rmse(pred_clean, zq)
rmse_nearest = rmse(pred_nearest, zq)
# 噪声数据：不验证就把噪声也插进去，验证集 RMSE 会爆炸；正则化把它压回来
z_noisy = true_f(X[:, 0], X[:, 1]) + rng.normal(0, 0.15, N)
pred_noisy = rbf_fit(X, z_noisy, Xq, lam=1e-10)
pred_reg = rbf_fit(X, z_noisy, Xq, lam=0.10)
rmse_noisy = rmse(pred_noisy, zq)
rmse_reg = rmse(pred_reg, zq)

# ── 2. 一维 Tikhonov vs TV ─────────────────────────────────────
rng2 = np.random.default_rng(140)
L = 256
t = np.arange(L)
clean = np.zeros(L)
clean[60:130] = 1.0
clean[130:190] = 0.3
clean[190:] = 0.8
y = clean + rng2.normal(0, 0.12, L)
D = sparse.diags([-np.ones(L - 1), np.ones(L - 1)], [0, 1], shape=(L - 1, L)).tocsc()
I = sparse.eye(L, format="csc")

def tikhonov(y, lam):
    A = (I + lam * (D.T @ D)).tocsc()
    return spsolve(A, y)

def tv_irls(y, lam=0.5, iters=30, eps=1e-4):
    x = y.copy()
    for _ in range(iters):
        d = D @ x
        w = 1.0 / np.sqrt(d * d + eps)
        A = (I + lam * (D.T @ sparse.diags(w) @ D)).tocsc()
        x = spsolve(A, y)
    return x

def edge_error(x):
    jumps = [60, 130, 190]
    errs = [abs((x[j] - x[j - 1]) - (clean[j] - clean[j - 1])) for j in jumps]
    return float(np.mean(errs))

def total_variation(x):
    return float(np.abs(np.diff(x)).sum())

x_tik = tikhonov(y, lam=5.0)
x_tv = tv_irls(y, lam=0.5)
mse_noisy = float(np.mean((y - clean) ** 2))
mse_tik = float(np.mean((x_tik - clean) ** 2))
mse_tv = float(np.mean((x_tv - clean) ** 2))
edge_noisy, edge_tik, edge_tv = edge_error(y), edge_error(x_tik), edge_error(x_tv)
tv_noisy, tv_tik, tv_tv = total_variation(y), total_variation(x_tik), total_variation(x_tv)

# ── 3. 二值 MRF（ICM）──────────────────────────────────────────
rng3 = np.random.default_rng(141)
H = W = 64
yy, xx = np.mgrid[0:H, 0:W]
true_label = (((xx - 32) ** 2 / 22 ** 2 + (yy - 30) ** 2 / 16 ** 2) <= 1).astype(np.int8)
obs = true_label.copy()
flip = rng3.random((H, W)) < 0.18
obs[flip] ^= 1

def mrf_energy(lab, obs, w_data=1.0, w_smooth=1.0):
    data = w_data * float((lab != obs).sum())
    smooth = w_smooth * float((lab[:, 1:] != lab[:, :-1]).sum() + (lab[1:, :] != lab[:-1, :]).sum())
    return data + smooth

def isolated_count(lab):
    pad = np.pad(lab, 1, mode="edge")
    same = ((pad[:-2, 1:-1] == lab) | (pad[2:, 1:-1] == lab) |
            (pad[1:-1, :-2] == lab) | (pad[1:-1, 2:] == lab))
    return int((~same).sum())

unary = obs.copy()
acc_unary = float((unary == true_label).mean())
energy_unary = mrf_energy(unary, obs)
iso_unary = isolated_count(unary)

lab = unary.copy()
energy_trace = [mrf_energy(lab, obs)]
for _ in range(10):
    changed = 0
    for r in range(H):
        for c in range(W):
            cur = lab[r, c]
            best = cur
            best_e = None
            for cand in (0, 1):
                lab[r, c] = cand
                e = mrf_energy(lab, obs)
                if best_e is None or e < best_e:
                    best_e = e; best = cand
            lab[r, c] = best
            if best != cur:
                changed += 1
    energy_trace.append(mrf_energy(lab, obs))
    if changed == 0:
        break
acc_mrf = float((lab == true_label).mean())
energy_mrf = mrf_energy(lab, obs)
iso_mrf = isolated_count(lab)
monotone = all(energy_trace[i] >= energy_trace[i + 1] for i in range(len(energy_trace) - 1))

checks = []
def check(name, ok, detail):
    checks.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")

check("干净 RBF 的 RMSE <0.05", rmse_clean < 0.05, f"{rmse_clean:.4f}")
check("RBF 比最近邻更准", rmse_clean < rmse_nearest, f"{rmse_clean:.4f} < {rmse_nearest:.4f}")
check("含噪精确插值明显变差", rmse_noisy > rmse_clean, f"{rmse_noisy:.4f} > {rmse_clean:.4f}")
check("正则化降低含噪验证误差", rmse_reg < rmse_noisy, f"{rmse_reg:.4f} < {rmse_noisy:.4f}")
check("Tikhonov MSE 低于含噪信号", mse_tik < mse_noisy, f"{mse_tik:.4f} < {mse_noisy:.4f}")
check("TV MSE 低于含噪信号", mse_tv < mse_noisy, f"{mse_tv:.4f} < {mse_noisy:.4f}")
check("TV 边缘误差低于 Tikhonov", edge_tv < edge_tik, f"{edge_tv:.4f} < {edge_tik:.4f}")
check("TV 的总变差低于含噪信号", tv_tv < tv_noisy, f"{tv_tv:.2f} < {tv_noisy:.2f}")
check("MRF 准确率高于逐点阈值", acc_mrf > acc_unary, f"{acc_mrf:.4f} > {acc_unary:.4f}")
check("MRF 能量低于逐点阈值", energy_mrf < energy_unary, f"{energy_mrf:.1f} < {energy_unary:.1f}")
check("MRF 孤立像素少于逐点阈值", iso_mrf < iso_unary, f"{iso_mrf} < {iso_unary}")
check("ICM 能量单调不升", monotone, f"{len(energy_trace)} 次迭代")

n_pass = int(sum(checks))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：RBF 散乱插值
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0))
ax = axes[0, 0]
sc = ax.scatter(X[:, 0], X[:, 1], c=z, s=38, cmap="viridis", edgecolor=INK, linewidth=0.4)
ax.set_title("60 个散乱样本（颜色=函数值）", fontsize=15)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("x", fontsize=15); ax.set_ylabel("y", fontsize=15)
ax = axes[0, 1]
im = ax.imshow(pred_clean.reshape(80, 80), origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
ax.set_title(f"RBF 重建：RMSE {rmse_clean:.3f}", fontsize=15)
ax.set_xticks([]); ax.set_yticks([])
ax = axes[1, 0]
ax.imshow(pred_nearest.reshape(80, 80), origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
ax.set_title(f"最近邻：RMSE {rmse_nearest:.3f}", fontsize=15)
ax.set_xticks([]); ax.set_yticks([])
ax = axes[1, 1]
names = ["干净\nRBF", "最近邻", "含噪\n未正则", "含噪\n正则"]
vals = [rmse_clean, rmse_nearest, rmse_noisy, rmse_reg]
cols = [GREEN, PURPLE, ACCENT, BLUE]
ax.bar(names, vals, color=cols)
ax.set_ylabel("验证 RMSE", fontsize=15); ax.set_title("噪声与正则化的作用", fontsize=15)
ax.tick_params(axis="x", labelsize=15)
ax.grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
fig.suptitle("散乱数据插值：先插值，再判断该不该插这么紧", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "e139_rbf.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：Tikhonov vs TV
fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.2), sharey=True)
for ax, sig, title, color in [
    (axes[0], y, f"含噪：MSE {mse_noisy:.4f}", ACCENT),
    (axes[1], x_tik, f"Tikhonov：MSE {mse_tik:.4f}，边缘误差 {edge_tik:.3f}", BLUE),
    (axes[2], x_tv, f"TV：MSE {mse_tv:.4f}，边缘误差 {edge_tv:.3f}", GREEN)]:
    ax.plot(t, clean, color=INK, lw=2.2, ls="--", label="真值")
    ax.plot(t, sig, color=color, lw=2.0, label="估计")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("位置", fontsize=15)
    ax.grid(True, color=GRID, alpha=0.45, lw=0.6)
axes[0].set_ylabel("信号值", fontsize=15)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=15, bbox_to_anchor=(0.5, 0.01))
fig.suptitle("同一份噪声：L2 把台阶磨圆，L1 把台阶留住", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0.10, 1, 0.93))
fig.savefig(FIG / "e139_tv.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：二值 MRF
fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.8))
for ax, img, title in [
    (axes[0], true_label, "真值"),
    (axes[1], obs, f"观测（翻转 18%）：{acc_unary:.3f}"),
    (axes[2], unary, f"逐点阈值：{acc_unary:.3f}"),
    (axes[3], lab, f"MRF/ICM：{acc_mrf:.3f}")]:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=15)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"MRF 平滑先验：准确率 {acc_unary:.3f} → {acc_mrf:.3f}，能量 {energy_unary:.0f} → {energy_mrf:.0f}", fontsize=16, y=1.03)
fig.tight_layout()
fig.savefig(FIG / "e139_mrf.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

for _name in ("e139_rbf.png", "e139_tv.png", "e139_mrf.png"):
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] RBF：干净 {rmse_clean:.4f}；最近邻 {rmse_nearest:.4f}；含噪未正则 {rmse_noisy:.4f}；含噪正则 {rmse_reg:.4f}")
print(f"[lab] 正则：MSE 含噪 {mse_noisy:.4f}/Tik {mse_tik:.4f}/TV {mse_tv:.4f}；边缘误差 Tik {edge_tik:.4f}/TV {edge_tv:.4f}；TV 总量 {tv_noisy:.2f}/{tv_tik:.2f}/{tv_tv:.2f}")
print(f"[lab] MRF：准确率 {acc_unary:.4f}→{acc_mrf:.4f}；能量 {energy_unary:.1f}→{energy_mrf:.1f}；孤立 {iso_unary}→{iso_mrf}；迭代 {len(energy_trace)-1} 次")

readings = dict(
    n_pass=n_pass, n_total=len(checks),
    rbf=dict(n_samples=N, grid=80, c=0.25, clean_lambda=1e-6, noisy_lambda=1e-10, reg_lambda=0.10,
             rmse=dict(clean=rmse_clean, nearest=rmse_nearest, noisy=rmse_noisy, regularized=rmse_reg)),
    regularization=dict(length=L, noise_sigma=0.12, tikhonov_lambda=5.0, tv_lambda=0.5, tv_iters=30,
                        mse=dict(noisy=mse_noisy, tikhonov=mse_tik, tv=mse_tv),
                        edge_error=dict(noisy=edge_noisy, tikhonov=edge_tik, tv=edge_tv),
                        total_variation=dict(noisy=tv_noisy, tikhonov=tv_tik, tv=tv_tv)),
    mrf=dict(height=H, width=W, flip_probability=0.18, w_data=1.0, w_smooth=1.0,
             accuracy=dict(unary=acc_unary, mrf=acc_mrf),
             energy=dict(unary=energy_unary, mrf=energy_mrf),
             isolated=dict(unary=iso_unary, mrf=iso_mrf),
             iterations=len(energy_trace) - 1, monotone=bool(monotone),
             energy_trace=[float(v) for v in energy_trace]),
    protocol=dict(synthetic=True, machine="local", general_benchmark=False,
                  implements="RBF interpolation + Tikhonov + TV-IRLS + binary MRF-ICM",
                  not_implemented=["graph cuts", "belief propagation", "dense CRF"]),
    versions=dict(numpy=np.__version__),
)
out = ROOT / "outputs" / "e139_ch4_interpolation_mrf_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
