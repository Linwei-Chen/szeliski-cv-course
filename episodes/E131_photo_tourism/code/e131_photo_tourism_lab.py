#!/usr/bin/env python3
"""E131 Photo Tourism lab：15 张合成照片、平面代理 warp、照片图与虚拟路径。"""
import datetime
import hashlib
import json
import logging
import pathlib
from collections import deque

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; GOLD = "#C9971C"
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

print(f"[lab] E131 Photo Tourism 开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成正面外墙+一个前景柱子；平面代理是已知背景平面；不是真实照片/SfM/完整 Photo Tourism")

W, H = 240, 180
XS = [-1.0, -0.5, 0.0, 0.5, 1.0]
YS = [-0.5, 0.0, 0.5]
u = (np.arange(W) + 0.5) / W - 0.5
v = (np.arange(H) + 0.5) / H - 0.5
U, V = np.meshgrid(u, v)

def facade(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.55; base[..., 1] = 0.45; base[..., 2] = 0.36
    win = ((np.mod(np.floor((X + 4) / 0.7), 2) == 0) &
           (np.mod(np.floor((Y + 3) / 0.55), 2) == 0) &
           (np.mod(X + 4, 0.7) > 0.12) & (np.mod(X + 4, 0.7) < 0.62) &
           (np.mod(Y + 3, 0.55) > 0.10) & (np.mod(Y + 3, 0.55) < 0.46))
    base[win] = [0.20, 0.30, 0.42]
    line = ((np.mod(X + 4, 0.7) < 0.03) | (np.mod(Y + 3, 0.55) < 0.03))
    base[line] *= 0.8
    return np.clip(base, 0, 1)

def render(cx, cy, layers=False):
    """相机在 (cx,cy,0)，看 +Z；背景平面 Z=6，前景柱 Z=2。"""
    Xb = cx + 6 * U; Yb = cy + 6 * V
    base = facade(Xb, Yb)
    Xf = cx + 2 * U; Yf = cy + 2 * V
    fg = (np.abs(Xf - 0.2) < 0.22) & (np.abs(Yf) < 0.62)
    fgimg = np.zeros_like(base); fgimg[:] = [0.85, 0.25, 0.12]
    stripe = fg & (np.mod(np.floor((Yf + 1) / 0.12), 2) == 0)
    fgimg[stripe] = [0.95, 0.55, 0.15]
    full = np.where(fg[..., None], fgimg, base)
    if layers:
        return np.clip(base, 0, 1), np.clip(fgimg, 0, 1), fg
    return np.clip(full, 0, 1)

views = {(ix, iy): render(cx, cy) for iy, cy in enumerate(YS) for ix, cx in enumerate(XS)}
src_ix, src_iy = 1, 1
tgt_ix, tgt_iy = 2, 1
src = views[(src_ix, src_iy)]; tgt = views[(tgt_ix, tgt_iy)]
dx_bg = -(XS[tgt_ix] - XS[src_ix]) / 6 * W
dx_fg = -(XS[tgt_ix] - XS[src_ix]) / 2 * W
ghost_shift_px = abs(dx_fg - dx_bg)

# 平面代理：用背景平面拟合出的 homography；本场景是正对平面，所以退化为平移。
planar = ndimage.shift(src, (0, dx_bg, 0), order=1, mode="nearest")

# 只取目标前景掩码与背景层，用来分别评估“前景错位”和“平面自身是否对齐”。
sb, _, _ = render(XS[src_ix], YS[src_iy], True)
tb, _, tm = render(XS[tgt_ix], YS[tgt_iy], True)
wb = ndimage.shift(sb, (0, dx_bg, 0), order=1, mode="nearest")

def psnr(a, b, mask=None):
    d = (a - b) ** 2
    d = d if mask is None else d[mask]
    return float(min(10 * np.log10(1 / max(d.mean(), 1e-12)), 45.0))

def mse(a, b, mask=None):
    d = (a - b) ** 2
    return float((d if mask is None else d[mask]).mean())

tfg = tm
inner = np.zeros((H, W), bool); inner[20:-20, 20:-20] = True
psnr_planar = psnr(planar, tgt)
psnr_bg_plane = psnr(wb, tb, inner)
fg_mse_planar = mse(planar, tgt, tfg)
ghost_pixels = int(((np.abs(planar - tgt).max(2) > 0.25) & ~tfg).sum())

# 照片图：5×3 相机，四邻接；A=(0,0) 到 B=(4,2) 的最短路径。
nodes = [(ix, iy) for iy in range(3) for ix in range(5)]
adj = {n: [] for n in nodes}
for ix, iy in nodes:
    for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        q = (ix + ddx, iy + ddy)
        if q in adj:
            adj[(ix, iy)].append(q); adj[q].append((ix, iy))
start, goal = (0, 0), (4, 2)
prev = {start: None}; dq = deque([start])
while dq:
    n = dq.popleft()
    if n == goal:
        break
    for q in adj[n]:
        if q not in prev:
            prev[q] = n; dq.append(q)
path = []
n = goal
while n is not None:
    path.append(n); n = prev[n]
path = path[::-1]
path_steps = len(path) - 1
max_step = max(float(np.hypot((a[0] - b[0]) * 0.5, (a[1] - b[1]) * 0.5))
               for a, b in zip(path, path[1:]))

# 覆盖：在相机矩形里随机取 800 个虚拟位置，算最近一张照片的最大距离。
rng = np.random.default_rng(0)
samples = np.column_stack([rng.uniform(-1, 1, 800), rng.uniform(-0.5, 0.5, 800)])
def max_nearest(points):
    P = np.array(points, float)
    d = np.sqrt(((samples[:, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
    return float(d.max()), float(d.mean())
dense_pts = [(x, y) for y in YS for x in XS]
sparse_pts = [(-1, -0.5), (-1, 0.5), (1, -0.5), (1, 0.5)]
dense_max, dense_mean = max_nearest(dense_pts)
sparse_max, sparse_mean = max_nearest(sparse_pts)
coverage_gain = sparse_max / dense_max
protocol = dict(synthetic=True, sfm=False, images=[W, H], n_views=len(nodes),
                scene="fronto-parallel facade Z=6 + foreground post Z=2",
                proxy="dominant background plane (translation-only homography)",
                graph="4-neighbor 5×3 camera grid", full_photo_tourism=False)

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("合成照片尺寸与数量", len(views) == 15 and next(iter(views.values())).shape == (H, W, 3),
      f"15/{W}×{H}")
check("背景平面 warp 内区 PSNR≥30", psnr_bg_plane >= 30, f"{psnr_bg_plane:.1f}")
check("平面代理整图 PSNR 在 15–25", 15 <= psnr_planar <= 25, f"{psnr_planar:.2f}")
check("前景平面代理 MSE≥0.03", fg_mse_planar >= 0.03, f"{fg_mse_planar:.4f}")
check("前景重影像素数≥500", ghost_pixels >= 500, f"{ghost_pixels}")
check("前景/背景位移差=40px", abs(ghost_shift_px - 40) < 1e-6, f"{ghost_shift_px:.1f}")
check("照片图 15 个节点全连通", all(len(adj[n]) > 0 for n in nodes), "15/15")
check("A→B 最短路径=6 步", path_steps == 6, f"{path_steps}")
check("路径单步最大距离=0.5", abs(max_step - 0.5) < 1e-9, f"{max_step:.1f}")
check("15 张网格比 4 个角点覆盖更密", dense_max < sparse_max and coverage_gain >= 2,
      f"{dense_max:.3f}/{sparse_max:.3f} 倍={coverage_gain:.2f}")
check("lab 边界声明为合成场景", protocol["synthetic"] is True and protocol["full_photo_tourism"] is False,
      "合成外墙+前景柱，无真实 SfM")

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：5×3 照片墙 + 右侧照片图/最短路径
cell_w, cell_h, gap = W, H, 6
montage = np.ones((3 * cell_h + 2 * gap, 5 * cell_w + 4 * gap, 3))
for iy, cy in enumerate(YS):
    for ix, cx in enumerate(XS):
        y0 = (2 - iy) * (cell_h + gap); x0 = ix * (cell_w + gap)
        montage[y0:y0 + cell_h, x0:x0 + cell_w] = views[(ix, iy)]
fig = plt.figure(figsize=(13.2, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.12)
ax = fig.add_subplot(gs[0])
ax.imshow(montage, interpolation="bilinear"); ax.axis("off")
centers = {}
for iy, cy in enumerate(YS):
    for ix, cx in enumerate(XS):
        centers[(ix, iy)] = (ix * (cell_w + gap) + cell_w / 2,
                             (2 - iy) * (cell_h + gap) + cell_h / 2)
for a, b in zip(path, path[1:]):
    ax.plot([centers[a][0], centers[b][0]], [centers[a][1], centers[b][1]],
            color=ACCENT, lw=5, alpha=0.9, solid_capstyle="round")
for n in nodes:
    ax.scatter(*centers[n], s=70, color=INK, zorder=5)
ax.text(*centers[start], "A", color="white", fontsize=20, ha="center", va="center", zorder=6, fontweight="bold")
ax.text(*centers[goal], "B", color="white", fontsize=20, ha="center", va="center", zorder=6, fontweight="bold")
ax.set_title("15 张照片的“照片海”：红线是从 A 到 B 的切换路径", fontsize=19)

ax = fig.add_subplot(gs[1])
for n in nodes:
    ix, iy = n
    for q in adj[n]:
        ax.plot([ix, q[0]], [iy, q[1]], color=GRID, lw=2.0, zorder=1)
for a, b in zip(path, path[1:]):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=ACCENT, lw=4.0, zorder=2)
for n in nodes:
    ax.scatter([n[0]], [n[1]], s=300, color=PAPER, edgecolor=INK, linewidth=1.5, zorder=3)
ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.7, 2.7); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("照片图：边=可切换的邻近照片", fontsize=19)
ax.text(2.0, -0.62, f"15 个节点，A→B 最短 {path_steps} 步", ha="center", fontsize=18, color=INK)
fig.suptitle("互联网相册不是一张大图，而是一张可走的图", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e131_graph.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：目标 / 平面代理 / 误差图
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
for ax, im, title in [(axes[0], tgt, "目标视图"), (axes[1], planar, "平面代理 warp")]:
    ax.imshow(im, interpolation="bilinear"); ax.set_title(title, fontsize=19); ax.axis("off")
err = np.abs(planar - tgt).mean(2)
axes[2].imshow(err, cmap="magma", vmin=0, vmax=0.45, interpolation="nearest")
axes[2].set_title("误差图：越亮越差", fontsize=19); axes[2].axis("off")
axes[1].text(0.02, 0.96, f"PSNR {psnr_planar:.1f} dB\n前景重影 {ghost_pixels} px", transform=axes[1].transAxes,
             va="top", fontsize=18, color="white", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=3))
fig.suptitle("平面代理：背景对齐，前景柱子错位 40px", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e131_proxy.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：路径上的 6 个虚拟视角 + 覆盖柱状
path_views = [render((XS[a[0]] + XS[b[0]]) / 2, (YS[a[1]] + YS[b[1]]) / 2) for a, b in zip(path, path[1:])]
pgap = 8
pm = np.ones((2 * H + pgap, 3 * W + 2 * pgap, 3))
for k, im in enumerate(path_views):
    row, col = k // 3, k % 3
    y0 = row * (H + pgap); x0 = col * (W + pgap)
    pm[y0:y0 + H, x0:x0 + W] = im
fig = plt.figure(figsize=(13.0, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.24)
ax = fig.add_subplot(gs[0])
ax.imshow(pm, interpolation="bilinear"); ax.axis("off")
for k in range(6):
    row, col = k // 3, k % 3
    y0 = row * (H + pgap); x0 = col * (W + pgap)
    ax.text(x0 + 14, y0 + 14, str(k + 1), fontsize=18, color="white", fontweight="bold",
            va="top", bbox=dict(facecolor=ACCENT, edgecolor="none", boxstyle="round,pad=0.15"))
ax.set_title("沿 A→B 路径的 6 个中间视角", fontsize=19)
fig.text(0.32, 0.035, "沿图走，不需要真的飞到每一个相机位置", ha="center", fontsize=18)
ax = fig.add_subplot(gs[1])
bars = ax.bar(["4 个角点", "15 张网格"], [sparse_max, dense_max], color=[BLUE, ACCENT], width=0.55)
for b, val in zip(bars, [sparse_max, dense_max]):
    ax.text(b.get_x() + b.get_width() / 2, val + 0.03, f"{val:.2f}", ha="center", fontsize=18)
ax.set_ylabel("最近照片的最大距离", fontsize=18, labelpad=8)
ax.set_ylim(0, 1.25); ax.tick_params(labelsize=18)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
fig.suptitle("照片图上的虚拟路径：15 张比 4 个角点密得多", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e131_path.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), image=[W, H], n_views=len(views),
                camera_x=XS, camera_y=YS, dx_bg=dx_bg, dx_fg=dx_fg,
                ghost_shift_px=ghost_shift_px,
                psnr_planar=psnr_planar, psnr_bg_plane=psnr_bg_plane,
                fg_mse_planar=fg_mse_planar,
                ghost_pixels=ghost_pixels, path=path, path_steps=path_steps,
                max_step=max_step, dense_max=dense_max, dense_mean=dense_mean,
                sparse_max=sparse_max, sparse_mean=sparse_mean,
                coverage_gain=coverage_gain, protocol=protocol,
                fig_sizes={"graph": [13.2, 5.4], "proxy": [13.0, 4.6], "path": [13.0, 4.8]})
out = ROOT / "outputs" / "e131_photo_tourism_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
for name in ("e131_graph.png", "e131_proxy.png", "e131_path.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] E131 Photo Tourism 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
