#!/usr/bin/env python3
"""E082 运动估计开篇：对应、孔径、无纹理与多运动（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261102)

R = 14
BASE = 150
PATCH = 100
P0 = 25


def texture(n=150, sigma=1.2):
    t = gaussian_filter(RNG.normal(0, 1, (n, n)), sigma)
    t = (t - t.min()) / (t.max() - t.min())
    return 0.05 + 0.90 * t


def ssd(a, b):
    d = a - b
    return float(np.mean(d * d))


def search_shift(img, patch, r=R):
    """在 img 中搜 patch 的平移；返回 (best_sx,best_sy), best_err, err_map。"""
    scores = np.zeros((2 * r + 1, 2 * r + 1))
    for iy, sy in enumerate(range(-r, r + 1)):
        for ix, sx in enumerate(range(-r, r + 1)):
            cand = img[P0 + sy:P0 + sy + PATCH, P0 + sx:P0 + sx + PATCH]
            scores[iy, ix] = ssd(cand, patch)
    iy, ix = np.unravel_index(np.argmin(scores), scores.shape)
    return (int(ix - r), int(iy - r)), float(scores[iy, ix]), scores


# —— 图块 1：角点/纹理丰富，位移唯一 ——
base_c = texture()
patch_c = base_c[P0:P0 + PATCH, P0:P0 + PATCH]
true_c = (7, -5)
base_c_shift = np.roll(base_c, shift=(true_c[1], true_c[0]), axis=(0, 1)) + RNG.normal(0, 0.002, base_c.shape)
corner_best, corner_err, corner_map = search_shift(base_c_shift, patch_c)
vals = np.sort(corner_map.ravel())
corner_ratio = float(vals[1] / max(vals[0], 1e-12))

# —— 图块 2：竖直边缘，沿边滑动有孔径歧义 ——
edge_img = np.zeros((BASE, BASE))
edge_img[:, :BASE // 2] = 0.20
edge_img[:, BASE // 2:] = 0.82
patch_e = edge_img[P0:P0 + PATCH, P0:P0 + PATCH]
true_e = (5, 0)
edge_shift = np.roll(edge_img, shift=(true_e[1], true_e[0]), axis=(0, 1))
edge_best, edge_err, edge_map = search_shift(edge_shift, patch_e)
plateau = int(np.sum(edge_map <= edge_map.min() + 1e-9))

# —— 图块 3：平坦区，完全没有信息 ——
flat_img = np.full((BASE, BASE), 0.5)
patch_f = flat_img[P0:P0 + PATCH, P0:P0 + PATCH]
flat_best, flat_err, flat_map = search_shift(flat_img, patch_f)
flat_std = float(flat_map.std())

# —— 两层运动：一个全局位移不够 ——
N = 220
bg0 = texture(N, 1.0)
fg = texture(70, 0.8)
y0, x0 = 78, 76
y1, x1 = y0 + 3, x0 - 6
frame0 = bg0.copy()
frame0[y0:y0 + 70, x0:x0 + 70] = fg
frame1 = np.roll(bg0, 4, axis=1)
frame1[y1:y1 + 70, x1:x1 + 70] = fg


def search_region(a, b, r=10):
    best = (1e9, 0, 0)
    for sy in range(-r, r + 1):
        for sx in range(-r, r + 1):
            aa = a[20:200, 20:200]
            bb = b[20 + sy:200 + sy, 20 + sx:200 + sx]
            e = ssd(aa, bb)
            if e < best[0]: best = (e, sx, sy)
    return best


glob_err, glob_dx, glob_dy = search_region(frame0, frame1)
# 前景局部：只在 50×50 核心区比
core0 = frame0[y0 + 10:y0 + 60, x0 + 10:x0 + 60]
best_fg = (1e9, 0, 0)
for sy in range(-10, 11):
    for sx in range(-10, 11):
        cand = frame1[y0 + 10 + sy:y0 + 60 + sy, x0 + 10 + sx:x0 + 60 + sx]
        e = ssd(cand, core0)
        if e < best_fg[0]: best_fg = (e, sx, sy)
fg_err, fg_dx, fg_dy = best_fg
# 背景局部：避开前景
bg_patch = frame0[20:80, 150:210]
best_bg = (1e9, 0, 0)
for sy in range(-10, 11):
    for sx in range(-10, 11):
        cand = frame1[20 + sy:80 + sy, 150 + sx:210 + sx]
        e = ssd(cand, bg_patch)
        if e < best_bg[0]: best_bg = (e, sx, sy)
bg_err, bg_dx, bg_dy = best_bg

# —— 图 1：三种图块与 SSD 曲面 ——
fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.0), dpi=150)
fig.patch.set_facecolor(PAPER)
items = [(patch_c, corner_map, "角点：唯一谷底", "位移唯一"),
         (patch_e, edge_map, "竖直边缘：一条谷", "沿边滑动都像对"),
         (patch_f, flat_map, "平坦区：没有谷", "怎么移都差不多")]
for j, (patch, smap, title, note) in enumerate(items):
    ax = axes[0, j]; ax.set_facecolor(PAPER)
    ax.imshow(patch, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=15, color=INK); ax.axis("off")
    ax = axes[1, j]; ax.set_facecolor(PAPER)
    im = ax.imshow(smap, cmap="viridis", origin="lower")
    ax.set_title(note, fontsize=14, color=ACCENT); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("同一个位移问题：三种图块给出三种答案", fontsize=17, color=INK, y=0.99)
fig.text(0.5, 0.01, "程序生成｜SSD 曲面越尖，位移越确定；谷越平，越不可信", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(FIG / "e082_ambiguity.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 2：孔径问题 ——
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), dpi=150)
fig.patch.set_facecolor(PAPER)
ax = axes[0]; ax.set_facecolor(PAPER)
ax.imshow(patch_e, cmap="gray", vmin=0, vmax=1, extent=[0, PATCH, PATCH, 0])
for yy in (18, 34, 50, 66, 82):
    ax.add_patch(FancyArrowPatch((82, yy), (82, yy + 12), arrowstyle="-|>", mutation_scale=15, color=ACCENT, lw=2))
ax.text(50, 8, "沿边滑动：画面几乎不变", ha="center", fontsize=14, color=ACCENT,
        bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.9, pad=2))
ax.set_title("竖直边缘只给出法向约束", fontsize=15, color=INK); ax.axis("off")
ax = axes[1]; ax.set_facecolor(PAPER)
ax.imshow(edge_map, cmap="viridis", origin="lower")
ax.axvline(R + true_e[0], color=ACCENT, lw=2, ls="--")
ax.set_title("SSD 谷是一条线，不是一个点", fontsize=15, color=INK)
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("孔径问题：边缘告诉你「怎么动」，但没告诉你「沿边滑多少」", fontsize=17, color=INK, y=0.99)
fig.text(0.5, 0.01, f"程序生成｜平台上 {plateau} 个位移与最优值几乎相同", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(FIG / "e082_aperture.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 3：两层运动 ——
fig = plt.figure(figsize=(9.6, 5.2), dpi=150)
fig.patch.set_facecolor(PAPER)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])
for j, (img, lab) in enumerate(((frame0, "第 0 帧"), (frame1, "第 1 帧"))):
    ax = fig.add_subplot(gs[0, j]); ax.set_facecolor(PAPER)
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(lab, fontsize=15, color=INK); ax.axis("off")
ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor(PAPER)
ax.imshow(frame0, cmap="gray", vmin=0, vmax=1, alpha=0.35)
ax.add_patch(Rectangle((x0, y0), 70, 70, fill=False, ec=ACCENT, lw=1.6, ls="--"))
ax.add_patch(FancyArrowPatch((x0 + 38, y0 + 34), (x0 + 18, y0 + 44), arrowstyle="-|>", mutation_scale=30, color=ACCENT, lw=3.4))
ax.add_patch(FancyArrowPatch((x0 + 38, y0 + 58), (x0 + 50, y0 + 58), arrowstyle="-|>", mutation_scale=30, color=GREEN, lw=3.4))
ax.text(x0 - 2, y0 - 8, "前景 (-6,+3)", fontsize=14, color=ACCENT, fontweight="bold")
ax.text(x0 + 2, y0 + 82, "背景 (+4,0)", fontsize=14, color=GREEN, fontweight="bold")
ax.set_title("两种运动，一个全局平移无法同时解释", fontsize=14, color=INK); ax.axis("off")
fig.suptitle("多运动：全局估计会把两层折中成一个错误答案", fontsize=17, color=INK, y=0.99)
fig.text(0.5, 0.01, f"程序生成｜全局误差 {glob_err:.4f}｜前景局部 {fg_err:.4f}｜背景局部 {bg_err:.4f}", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(FIG / "e082_layers.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

tests = [
    ("角点位移唯一命中", corner_best == true_c, f"best={corner_best}/真值 {true_c}"),
    ("角点次优明显更大", corner_ratio > 1.5, f"次优/最优={corner_ratio:.2f}"),
    ("边缘法向位移正确", edge_best[0] == true_e[0], f"dx={edge_best[0]}/真值 {true_e[0]}"),
    ("边缘存在平台", plateau >= 21, f"平台 {plateau} 个位移"),
    ("平坦区无位移信息", flat_std < 1e-3, f"SSD 标准差={flat_std:.2e}"),
    ("平坦区误差近零", flat_err < 1e-4, f"最小 SSD={flat_err:.2e}"),
    ("前景局部运动正确", (fg_dx, fg_dy) == (-6, 3), f"best=({fg_dx},{fg_dy})"),
    ("背景局部运动正确", (bg_dx, bg_dy) == (4, 0), f"best=({bg_dx},{bg_dy})"),
    ("全局估计明显更差", glob_err > 5 * max(fg_err, bg_err, 1e-12), f"全局 {glob_err:.4f}/局部 max {max(fg_err,bg_err):.4f}"),
    ("三图在盘", all((FIG / n).exists() for n in ("e082_ambiguity.png", "e082_aperture.png", "e082_layers.png")), "三张全在"),
]
lines = []; n_pass = 0
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass += bool(ok)
readings = dict(R=R, patch=PATCH, corner_best=list(corner_best), corner_true=list(true_c), corner_err=corner_err,
                corner_ratio=corner_ratio, edge_best=list(edge_best), edge_err=edge_err, plateau=plateau,
                flat_err=flat_err, flat_std=flat_std, glob_err=glob_err, glob_shift=[glob_dx, glob_dy],
                fg_err=fg_err, fg_shift=[fg_dx, fg_dy], bg_err=bg_err, bg_shift=[bg_dx, bg_dy],
                n_pass=n_pass, n_total=len(tests))
(OUT / "e082_motion_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e082_motion_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E082 运动估计开篇 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e082_motion_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests) - n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass != len(tests): raise SystemExit(1)
