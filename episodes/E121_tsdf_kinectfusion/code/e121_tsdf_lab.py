#!/usr/bin/env python3
"""E121 深度图融合与 TSDF（合成圆，2D 切片）。

依赖：numpy/scipy/matplotlib。
边界：2D 合成圆，不冒充真实 Kinect 扫描；TSDF 的截断/加权公式是讲者按通行定义整理，
原书 p.824 只给名称与用途，§13.5.1 给符号距离/隐式面定义，未展开完整 KinectFusion 实现。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.ndimage import map_coordinates
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"

R = 0.5; EXT = 1.2; N = 180; TAU = 0.12; WMAX = 20.0; CAM_D = 3.0
xs = np.linspace(-EXT, EXT, N); ys = xs.copy(); X, Y = np.meshgrid(xs, ys)
PHI = np.sqrt(X**2 + Y**2) - R
VIEWS = [0.0, np.pi/2, np.pi, 3*np.pi/2]


def measure_view(angle, noise=0.0, seed=121):
    """从一台相机沿投影方向量深度，返回射线方向的截断符号距离与有效掩码。"""
    C = np.array([CAM_D*np.cos(angle), CAM_D*np.sin(angle)])
    f = -C / CAM_D
    r = np.array([-f[1], f[0]])
    P = np.stack([X - C[0], Y - C[1]], axis=-1)
    depth = P[..., 0]*f[0] + P[..., 1]*f[1]
    dist = np.linalg.norm(P, axis=-1)
    ux = P[..., 0] / np.maximum(dist, 1e-12); uy = P[..., 1] / np.maximum(dist, 1e-12)
    b = 2 * (C[0]*ux + C[1]*uy); c = C @ C - R*R
    disc = b*b - 4*c
    has = disc >= 0
    t = (-b - np.sqrt(np.maximum(disc, 0))) / 2
    hit_x = C[0] + t*ux; hit_y = C[1] + t*uy
    d_surf = (hit_x - C[0])*f[0] + (hit_y - C[1])*f[1]
    sdf = d_surf - depth
    if noise > 0:
        rng = np.random.default_rng(seed)
        sdf = sdf + rng.normal(0, noise, sdf.shape)
    valid = (depth > 0) & has & (sdf >= -TAU) & (sdf <= TAU)
    return sdf, valid


def fuse(angles, noises=None, wmax=WMAX, seed=121):
    tsdf = np.full((N, N), TAU); weight = np.zeros((N, N))
    for k, ang in enumerate(angles):
        noise = 0.0 if noises is None else noises[k]
        sdf, valid = measure_view(ang, noise, seed=seed + k)
        sc = np.clip(sdf, -TAU, TAU)
        nw = weight + valid
        tsdf = np.where(valid, (tsdf*weight + sc) / np.maximum(nw, 1), tsdf)
        weight = np.minimum(nw, wmax)
    return tsdf, weight


def sample(field, x, y):
    rr = (y - ys[0]) / (ys[1] - ys[0]); cc = (x - xs[0]) / (xs[1] - xs[0])
    return map_coordinates(field, [rr, cc], order=1, mode="nearest")


def crossings(tsdf, weight, nang=72, front_only=False):
    """径向采样，取从内（负）到外（正）的零交叉；返回 (角度, 半径或 None)。"""
    out = []
    for th in np.linspace(-np.pi, np.pi, nang, endpoint=False):
        if front_only and not (-np.pi/2 <= th <= np.pi/2):
            continue
        rs = np.linspace(0.25, 0.75, 501)
        f = sample(tsdf, rs*np.cos(th), rs*np.sin(th))
        ww = sample(weight, rs*np.cos(th), rs*np.sin(th))
        hit = None
        for i in range(len(rs) - 1):
            if ww[i] > 0.5 and ww[i+1] > 0.5 and f[i] <= 0 <= f[i+1] and (f[i] < 0 or f[i+1] > 0):
                den = f[i] - f[i+1]
                hit = rs[i] if den == 0 else rs[i] + (rs[i+1] - rs[i]) * f[i] / den
                break
        out.append((th, hit))
    return out


def radii(cs):
    return np.array([r for _, r in cs if r is not None])


def update_repeat(field, weight, sdf, wmax=5, times=10):
    f = float(field); w = float(weight)
    for _ in range(times):
        nw = min(w + 1, wmax)
        f = (f*w + np.clip(sdf, -TAU, TAU)) / (w + 1)
        w = nw
    return f, w


tsdf_one, w_one = fuse([VIEWS[0]])
tsdf_four, w_four = fuse(VIEWS)
c_one = crossings(tsdf_one, w_one); c_four = crossings(tsdf_four, w_four)
r_one = radii(c_one); r_four = radii(c_four)
cov_one, cov_four = len(r_one), len(r_four)
err_one = float(np.mean(np.abs(r_one - R)))
err_four = float(np.mean(np.abs(r_four - R)))
mid = N // 2
f_center = float(PHI[mid, mid]); f_far = float(PHI[0, 0])
max_trunc = float(np.max(np.abs(tsdf_four)))
surface_w = float(sample(w_four, np.array([0.0]), np.array([-R]))[0])
corner_w = float(sample(w_four, np.array([1.1]), np.array([1.1]))[0])
active_frac = float(np.mean(w_four > 0))

# 同一视角多帧带噪融合：20 组独立噪声取平均，20 帧平均应比单帧稳
single_stds, five_stds = [], []
for trial in range(20):
    tsdf_n1, w_n1 = fuse([VIEWS[0]], [0.02], seed=500 + 10*trial)
    tsdf_n20, w_n20 = fuse([VIEWS[0]]*20, [0.02]*20, seed=700 + 10*trial)
    single_stds.append(float(np.std(radii(crossings(tsdf_n1, w_n1, front_only=True)))))
    five_stds.append(float(np.std(radii(crossings(tsdf_n20, w_n20, front_only=True)))))
std_n1 = float(np.mean(single_stds)); std_n20 = float(np.mean(five_stds))

# 权重截断：同一点重复观测 10 次，权重封顶 5
cap_f, cap_w = update_repeat(TAU, 0.0, 0.03, wmax=5, times=10)

# 符号翻转对照：翻转已观测体素的符号，零交叉跑到截断带外沿
flip = tsdf_one.copy(); flip[w_one > 0] *= -1
r_flip = radii(crossings(flip, w_one))
flip_err = float(np.mean(np.abs(r_flip - R))) if len(r_flip) else 9.99

checks = [
 ("精确 SDF：圆心为负、远处为正", f_center < 0 < f_far, f"{f_center:.3f} / {f_far:.3f}"),
 ("TSDF 被截断在 ±τ 内", max_trunc <= TAU + 1e-9, f"max|TSDF|={max_trunc:.3f}"),
 ("单视角只覆盖前半圈", 30 <= cov_one <= 36 and cov_four == 72, f"{cov_one}/72 vs {cov_four}/72"),
 ("四视角覆盖整圈", cov_four == 72, f"{cov_four}/72"),
 ("四视角零交叉半径误差 <0.005", err_four < 0.005, f"{err_four:.6f}"),
 ("表面体素至少被一条视线看到", surface_w >= 1.0, f"weight={surface_w:.1f}"),
 ("远处未观测体素权重为 0", corner_w == 0.0, f"weight={corner_w:.1f}"),
 ("20 帧带噪融合比单帧稳", std_n20 < 0.8*std_n1, f"{std_n20:.4f} < 0.8×{std_n1:.4f}"),
 ("权重封顶：重复 10 次仍为 5", cap_w == 5, f"w={cap_w:.0f}"),
 ("符号翻转把零交叉推到带外", flip_err > 0.04, f"半径误差={flip_err:.4f}"),
]
n_pass = 0
print(f"[lab] E121 TSDF/深度图融合开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：2D 合成圆；TSDF 截断/加权公式=讲者按通行定义整理，不是原书完整算法")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 三切片：精确 SDF、单视角、四视角
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
for ax, field, title, cs in [
    (axes[0], PHI, "精确符号距离（真值）", None),
    (axes[1], tsdf_one, "单视角 TSDF：只看到前半圈", c_one),
    (axes[2], tsdf_four, "四视角融合：整圈都看到", c_four)]:
    im = ax.imshow(field, extent=[-EXT, EXT, -EXT, EXT], origin="lower",
                   cmap="RdBu_r", vmin=-TAU, vmax=TAU)
    ax.contour(X, Y, PHI, levels=[0], colors=INK, linewidths=1.1, linestyles="--")
    if cs:
        rr = [(r*np.cos(t), r*np.sin(t)) for t, r in cs if r is not None]
        if rr:
            ax.scatter([p[0] for p in rr], [p[1] for p in rr], s=18, c=GREEN, zorder=3)
    ax.set_title(title, fontsize=17, color=INK)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
cbar = fig.colorbar(im, ax=axes, shrink=0.75)
cbar.set_label("TSDF 值", fontsize=17); cbar.ax.tick_params(labelsize=15)
fig.suptitle("TSDF 切片：从单视角到四视角融合（绿点=零交叉）", fontsize=20, color=INK)
fig.savefig(FIG / "e121_tsdf_slices.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 覆盖角与噪声
fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
ax = axes[0]
for rr, lab, color in [(r_one, "单视角", BLUE), (r_four, "四视角", GREEN)]:
    th = np.linspace(-np.pi, np.pi, len(rr), endpoint=False) if len(rr) == 72 else \
         np.array([t for t, r in c_one if r is not None])
    ax.scatter(th*180/np.pi, rr, s=20, label=lab, color=color)
ax.axhline(R, color=INK, lw=1, ls="--", label="真值半径")
ax.set_xlabel("方位角 / 度", fontsize=18); ax.set_ylabel("零交叉半径", fontsize=18)
ax.tick_params(labelsize=18); ax.set_title("覆盖：单视角半圈，四视角整圈", fontsize=18)
ax.set_ylim(0.48, 0.52)
ax.legend(fontsize=18, loc="lower right")
ax = axes[1]
bars = ax.bar(["单帧", "20 帧平均"], [std_n1, std_n20], color=[BLUE, GREEN], width=0.55)
for b, v in zip(bars, [std_n1, std_n20]):
    ax.text(b.get_x()+b.get_width()/2, v+0.001, f"{v:.4f}", ha="center", fontsize=18)
ax.set_ylabel("零交叉半径标准差", fontsize=18); ax.tick_params(labelsize=18)
ax.set_title("带噪观测：平均 20 帧更稳", fontsize=18)
ax.set_ylim(0, 0.042)
fig.suptitle("融合的两个收益：覆盖更全、噪声更小", fontsize=21, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e121_fusion.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 KinectFusion 循环
fig, ax = plt.subplots(figsize=(12.4, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
ax.axis("off")
boxes = [
    (0.2, 1.5, "深度帧\nDepth frame", BLUE),
    (2.5, 1.5, "位姿跟踪\nICP-like", ACCENT),
    (4.8, 1.5, "TSDF 融合\n加权平均", GREEN),
    (7.1, 1.5, "提面/渲染\nraycast", GRAY),
]
for x, y, text, color in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 1.8, 1.0, boxstyle="round,pad=0.08",
                                fc="white", ec=color, lw=2.2))
    ax.text(x+0.9, y+0.5, text, ha="center", va="center", fontsize=17, color=INK)
for x in [2.05, 4.35, 6.65]:
    ax.add_patch(FancyArrowPatch((x, 2.0), (x+0.38, 2.0), arrowstyle="-|>",
                                 mutation_scale=18, lw=2, color=INK))
ax.add_patch(FancyArrowPatch((8.0, 1.45), (8.0, 0.55), arrowstyle="-", lw=2, color=ACCENT))
ax.add_patch(FancyArrowPatch((8.0, 0.55), (1.1, 0.55), arrowstyle="-", lw=2, color=ACCENT))
ax.add_patch(FancyArrowPatch((1.1, 0.55), (1.1, 1.45), arrowstyle="-|>",
                             mutation_scale=18, lw=2, color=ACCENT))
ax.text(4.5, 0.30, "下一帧：用当前模型再做跟踪（循环）", ha="center", fontsize=17, color=ACCENT)
ax.text(5.0, 3.25, "KinectFusion 的主循环", ha="center", fontsize=22, color=INK)
ax.text(5.0, 2.85, "跟踪给位姿，融合改模型，渲染供下一帧对齐", ha="center", fontsize=17, color=GRAY)
fig.savefig(FIG / "e121_pipeline.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), tau=TAU, grid=N,
                f_center=f_center, f_far=f_far, max_trunc=max_trunc,
                cov_one=cov_one, cov_four=cov_four,
                err_one=err_one, err_four=err_four,
                surface_weight=surface_w, corner_weight=corner_w, active_frac=active_frac,
                std_noise_single=std_n1, std_noise_fused=std_n20, noise_trials=20,
                cap_weight=cap_w, flip_err=flip_err, n_cross_flip=len(r_flip))
(OUT / "e121_tsdf_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for name in ("e121_tsdf_slices.png", "e121_fusion.png", "e121_pipeline.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e121_tsdf_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
