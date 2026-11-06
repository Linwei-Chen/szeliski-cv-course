#!/usr/bin/env python3
"""E075 实验：最小二乘与 Lucas–Kanade 一维平移（程序生成，未用原书原图）。

三图：LK 迭代管线 / 平移最小二乘的质心解释 / 收敛曲线（小位移、大位移、金字塔）。
10 条数值断言；readings 落盘。
"""
import hashlib
import json
import pathlib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32"

# ── 1D 模板与目标 ──
N = 801
x = np.linspace(-4, 4, N)
dx = x[1] - x[0]
T = np.exp(-(x / 0.55) ** 2) + 0.35 * np.sin(7 * x) * np.exp(-(x / 2.2) ** 2)
T = (T - T.min()) / (T.max() - T.min())
gradT = np.gradient(T, dx)


def sample(arr, xq):
    return np.interp(xq, x, arr, left=0.0, right=0.0)


def lk_1d(I, p0=0.0, iters=30, Tarr=None, Garr=None, xg=None):
    Ta = T if Tarr is None else Tarr
    Ga = gradT if Garr is None else Garr
    xq = x if xg is None else xg
    p = float(p0); errs = []
    for _ in range(iters):
        warp = np.interp(xq - p, xq, Ta, left=0.0, right=0.0)
        g = np.interp(xq - p, xq, Ga, left=0.0, right=0.0)
        r = I - warp
        errs.append(float(np.mean(r ** 2)))
        dp = -float(np.dot(g, r) / (np.dot(g, g) + 1e-12))
        cur = errs[-1]
        for _ in range(6):
            warp2 = np.interp(xq - (p + dp), xq, Ta, left=0.0, right=0.0)
            if float(np.mean((I - warp2) ** 2)) <= cur:
                break
            dp *= 0.5
        p += dp
        if abs(dp) < 1e-6:
            break
    return p, errs


def blur1d(a, sigma_samples):
    if sigma_samples <= 0:
        return a.copy()
    r = int(3 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_samples) ** 2)
    k /= k.sum()
    return np.convolve(a, k, mode="same")


def pyramid_lk(I, levels=3):
    """由粗到细：先在大尺度（强模糊）上估，再逐级减小模糊细化。"""
    p = 0.0
    for sig in (0.24, 0.11, 0.0):
        Tb = blur1d(T, sig / dx)
        Ib = blur1d(I, sig / dx)
        p, _ = lk_1d(Ib, p0=p, iters=40, Tarr=Tb, Garr=np.gradient(Tb, dx), xg=x)
    return p


d_small, d_large = 0.15, 0.80
I_small = sample(T, x - d_small)
I_large = sample(T, x - d_large)
p_small, err_small = lk_1d(I_small, 0.0)
p_large, err_large = lk_1d(I_large, 0.0)
p_pyr = pyramid_lk(I_large, levels=4)

# ── 点集最小二乘 ──
rng = np.random.default_rng(7)
P = rng.uniform(-2, 2, (12, 2))
delta = np.array([0.42, -0.23])
Q = P + delta
p_trans = (Q - P).mean(axis=0)

A_true = np.array([[1.12, 0.18], [0.07, 0.93]])
t_true = np.array([0.15, -0.08])
Q_aff = P @ A_true.T + t_true
M = np.c_[P, np.ones(len(P))]
coef, *_ = np.linalg.lstsq(M, Q_aff, rcond=None)
A_hat, t_hat = coef[:2].T, coef[2]
aff_err = float(np.abs(A_hat - A_true).max() + np.abs(t_hat - t_true).max())

# 加权最小二乘：一个外点
Qw = Q.copy(); Qw[0] += np.array([6.0, -5.0])
w = np.ones(len(P)); w[0] = 0.02
p_unw = (Qw - P).mean(axis=0)
p_w = np.sum(w[:, None] * (Qw - P), axis=0) / w.sum()
w_err = float(np.linalg.norm(p_w - delta)); unw_err = float(np.linalg.norm(p_unw - delta))

# ── 旋转的 Gauss–Newton ──
theta_true = np.deg2rad(20)
Pc = P - P.mean(axis=0)
Qr = Pc @ np.array([[np.cos(theta_true), -np.sin(theta_true)],
                    [np.sin(theta_true), np.cos(theta_true)]]).T
theta = 0.0; rot_errs = []
for _ in range(30):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pred = Pc @ R.T
    r = (Qr - pred).ravel()
    J = np.zeros((2 * len(Pc), 1))
    J[0::2, 0] = (-s * Pc[:, 0] - c * Pc[:, 1])
    J[1::2, 0] = (c * Pc[:, 0] - s * Pc[:, 1])
    dp = float(np.linalg.solve(J.T @ J + 1e-12, J.T @ r)[0])
    theta += dp
    rot_errs.append(abs(theta - theta_true))
    if abs(dp) < 1e-9:
        break

# ── LM 阻尼：同一点集，比较步长 ──
Jt = np.ones((len(P), 1))
for lam in (0.0, 1.0):
    A = Jt.T @ Jt
    b = Jt.T @ (Q - P)[:, :1]
    step = np.linalg.solve(A + lam * np.diag(np.diag(A)), b)
    if lam == 0.0:
        step0 = float(np.linalg.norm(step))
    else:
        step1 = float(np.linalg.norm(step))

# ── 图 1：LK 管线 ──
fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
def box(x0, y0, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fffdf8", ec=color, lw=1.6))
    ax.text(x0 + w / 2, y0 + h - 0.055, title, ha="center", va="center", fontsize=16.5, color=color, fontweight="bold")
    ax.text(x0 + w / 2, y0 + h / 2 - 0.04, body, ha="center", va="center", fontsize=15, color=INK, linespacing=1.5)
box(0.03, 0.62, 0.27, 0.25, "模板与目标", "同一场景\n两次采样", BLUE)
box(0.36, 0.62, 0.27, 0.25, "算梯度", "模板对 x 求导\n得到局部斜率", BLUE)
box(0.69, 0.62, 0.28, 0.25, "解 2×2 方程", "梯度平方和\n梯度×残差", ACCENT)
box(0.22, 0.20, 0.27, 0.25, "更新位移", "p ← p + Δp\n往残差小的方向", GREEN)
box(0.56, 0.20, 0.28, 0.25, "检查收敛", "残差够小或\n步长够小就停", ACCENT)
for a, b in [((0.30, 0.745), (0.36, 0.745)), ((0.63, 0.745), (0.69, 0.745)),
             ((0.83, 0.62), (0.63, 0.45)), ((0.49, 0.325), (0.56, 0.325)),
             ((0.60, 0.45), (0.39, 0.62))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#7a5c2e",
                                 connectionstyle="arc3,rad=0.12"))
ax.text(0.5, 0.93, "Lucas–Kanade：用梯度把残差拧到零", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
ax.text(0.5, 0.075, "每一步都解一个很小的线性方程；循环到位移不再变", ha="center", va="center", fontsize=15, color=ACCENT)
fig.savefig(FIG / "e075_lk_pipeline.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 图 2：平移最小二乘 = 质心差 ──
fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.scatter(P[:, 0], P[:, 1], s=55, c=BLUE, label="图 A 的点", zorder=4)
ax.scatter(Qw[:, 0], Qw[:, 1], s=55, c=GREEN, marker="^", label="图 B 的点（含一个外点）", zorder=4)
for a, b in zip(P, Qw):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", color="#8a8f99", lw=1.1, alpha=0.8))
cp, cq = P.mean(axis=0), Qw.mean(axis=0)
ax.scatter(*cp, s=220, c=ACCENT, marker="X", zorder=6, label="A 的质心")
ax.scatter(*cq, s=220, c=ACCENT, marker="P", zorder=6, label="B 的质心")
ax.annotate("", xy=cq, xytext=cp, arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=3, mutation_scale=22))
ax.text((cp[0] + cq[0]) / 2 + 0.2, (cp[1] + cq[1]) / 2 - 0.45, "平均位移 = 质心差", fontsize=15, color=ACCENT, fontweight="bold")
ax.set_title("纯平移的最小二乘：所有位移取平均", fontsize=18, color=INK)
ax.set_xlabel("x", fontsize=14); ax.set_ylabel("y", fontsize=14)
ax.tick_params(labelsize=13); ax.grid(alpha=0.25)
ax.legend(fontsize=14.5, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "e075_translation_ls.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 图 3：收敛曲线 ──
fig, axs = plt.subplots(1, 2, figsize=(9.4, 5.6), dpi=150)
fig.patch.set_facecolor(PAPER)
for ax in axs:
    ax.set_facecolor(PAPER); ax.grid(alpha=0.25); ax.tick_params(labelsize=13)
axs[0].plot(range(1, len(err_small) + 1), err_small, color=GREEN, lw=2.2, label="小位移 0.15")
axs[0].plot(range(1, len(err_large) + 1), err_large, color=ACCENT, lw=2.2, label="大位移 0.80（直接 LK）")
axs[0].set_yscale("log"); axs[0].set_xlabel("迭代次数", fontsize=14); axs[0].set_ylabel("均方残差（对数）", fontsize=14)
axs[0].set_title("直接 LK：小位移收敛，大位移卡住", fontsize=16, color=INK)
axs[0].legend(fontsize=14.5)
labels = ["小位移\n直接 LK", "大位移\n直接 LK", "大位移\n多尺度"]
vals = [p_small - d_small, p_large - d_large, p_pyr - d_large]
axs[1].barh(labels, vals, color=[GREEN, ACCENT, BLUE], height=0.55)
axs[1].axvline(0, color=INK, lw=1)
for i, v in enumerate(vals):
    axs[1].text(v + (0.02 if v >= 0 else -0.02), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=13, color=INK)
axs[1].set_xlabel("估计误差", fontsize=14); axs[1].tick_params(labelsize=12)
axs[1].set_xlim(-1.0, 0.25)
axs[1].set_title("多尺度先把初值送到附近", fontsize=16, color=INK)
fig.suptitle("Lucas–Kanade 的收敛与初值", fontsize=18, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e075_lk_convergence.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

tests = [
    ("平移最小二乘=平均位移", np.allclose(p_trans, delta, atol=1e-12), f"估计={p_trans.round(3).tolist()}"),
    ("仿射最小二乘精确复原", aff_err < 1e-10, f"max误差={aff_err:.2e}"),
    ("加权最小二乘压住外点", w_err < 0.05 and unw_err > 0.5, f"加权={w_err:.4f} 未加权={unw_err:.3f}"),
    ("旋转 Gauss–Newton 收敛", abs(theta - theta_true) < np.deg2rad(0.1), f"误差={np.rad2deg(abs(theta-theta_true)):.4f}°"),
    ("旋转迭代次数≤10", len(rot_errs) <= 10, f"{len(rot_errs)} 次"),
    ("小位移 LK 收敛", abs(p_small - d_small) < 0.01, f"估计={p_small:.4f} 真值={d_small}"),
    ("大位移直接 LK 失败", abs(p_large - d_large) > 0.1, f"估计={p_large:.4f} 误差={abs(p_large-d_large):.3f}"),
    ("金字塔大位移收敛", abs(p_pyr - d_large) < 0.02, f"估计={p_pyr:.4f} 误差={abs(p_pyr-d_large):.4f}"),
    ("LM 阻尼缩短步长", step1 < step0, f"无阻尼:{step0:.3f} 加阻尼:{step1:.3f}"),
    ("三张图在盘", all((FIG / n).exists() for n in ("e075_lk_pipeline.png", "e075_translation_ls.png", "e075_lk_convergence.png")), "三张全在"),
]
lines = []
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
n_pass = sum(1 for _, ok, _ in tests if ok)
readings = dict(
    translation=[float(v) for v in p_trans], delta=[float(v) for v in delta],
    affine_error=aff_err, weighted_error=w_err, unweighted_error=unw_err,
    rotation_deg=float(np.rad2deg(theta)), rotation_error_deg=float(np.rad2deg(abs(theta-theta_true))),
    rotation_iters=len(rot_errs), lk_small=float(p_small), lk_large=float(p_large),
    lk_small_error=float(abs(p_small-d_small)), lk_large_error=float(abs(p_large-d_large)),
    pyramid_large=float(p_pyr), pyramid_error=float(abs(p_pyr-d_large)),
    lm_step0=step0, lm_step1=step1, n_pass=n_pass, n_total=len(tests),
)
(OUT / "e075_lk_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e075_lk_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
print(f"[lab] E075 最小二乘与 Lucas–Kanade 开始 {now}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines:
    print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e075_lk_readings.json")
print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests)-n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass != len(tests):
    raise SystemExit(1)
