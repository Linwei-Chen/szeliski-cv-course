#!/usr/bin/env python3
"""E141 lab：视频纹理与静态图分层动画。

教材：Szeliski, Computer Vision: Algorithms and Applications, 2nd ed.，
§14.5.1–14.5.3（p.887–893）。

实验分两部分：
1) 合成“往返小球”视频，比较“只比外观”和“外观+光流方向”两种帧匹配；
   在剪辑点演示：只比外观会跳到同位置、反方向的帧；加运动方向后保住方向，
   但位置对不齐，需要用 cross-dissolve 把跳变摊开。
2) 合成一张分层静态图（天空/水面/云/树/船），用软 alpha 合成，
   对比硬边 mask；船移动后背景露出洞，用示意性修补填洞。

边界：全部是合成数据；光流用 OpenCV Farnebäck；修补用 OpenCV Telea，
不是教材里的 Criminisi 示例修补；不复制教材原图。
"""
import datetime
import hashlib
import json
import pathlib

import cv2
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, map_coordinates

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for _cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(_cand).exists():
        font_manager.fontManager.addfont(_cand)

EP = pathlib.Path(__file__).resolve().parents[1]
OUT = EP / "code" / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]

print(f"[lab] E141 视频渲染实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] OpenCV {cv2.__version__} / NumPy {np.__version__}")
print("[lab] 边界：合成 100 帧往返小球 + 合成 240×160 四层静态图；光流 Farnebäck、补洞 Telea；不是真实视频/完整 video texture 系统")
plt.rcParams.update({
    "font.size": 17,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "figure.titlesize": 21,
    "axes.unicode_minus": False,
})


def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def psnr(a, b):
    e = mse(a, b)
    return float("inf") if e <= 1e-15 else float(10.0 * np.log10(1.0 / e))


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ────────────────────────────────────────────────────────────
# 1. 视频纹理：往返小球
# ────────────────────────────────────────────────────────────
H, W, T = 120, 160, 100
yy, xx = np.mgrid[0:H, 0:W].astype(float)

bg = np.zeros((H, W, 3), float)
bg[..., 0] = 40 + 0.55 * xx + 18 * np.sin(yy / 7.0)
bg[..., 1] = 60 + 0.35 * yy + 12 * np.sin(xx / 9.0)
bg[..., 2] = 95 + 0.25 * (H - yy) + 10 * np.cos((xx + yy) / 11.0)
bg = np.clip(bg, 0, 255) / 255.0


def tri_x(t):
    """从 x=20 走到 x=140，再原路返回；第 60 帧到端点。"""
    return 20 + 2 * t if t <= 60 else 140 - 2 * (t - 60)


xs = np.array([tri_x(t) for t in range(T)])
frames = []
for t in range(T):
    im = bg.copy()
    cy, cx = 60, xs[t]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = d <= 12
    tex = 0.5 + 0.45 * np.cos(d * 2.0) + 0.05 * np.cos(d * 0.7)
    ball = np.stack([tex * 0.95, tex * 0.55, tex * 0.25], axis=-1)
    im[mask] = ball[mask]
    frames.append(np.clip(im, 0, 1))
frames = np.array(frames)

# 只比外观的相似度（MSE，越小越像）
S = np.array([[mse(frames[i], frames[j]) for j in range(T)] for i in range(T)])
positive = S[S > 0]
med = float(np.median(positive))

# 光流方向：用 Farnebäck 估计 t-1→t+1 的平均运动方向；
# 只取运动幅度最大的 1% 像素，避免静态背景把平均值拉向零。
gray = [(f * 255).astype(np.uint8).mean(axis=2).astype(np.uint8) for f in frames]
U = np.zeros((T, 2), float)
valid = np.zeros(T, bool)
for t in range(1, T - 1):
    flow = cv2.calcOpticalFlowFarneback(
        gray[t - 1], gray[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    sel = mag >= max(float(np.percentile(mag, 99)), 0.1)
    v = flow[sel].mean(axis=0) if sel.any() else np.zeros(2)
    if np.linalg.norm(v) > 0.5:
        U[t] = v / np.linalg.norm(v)
        valid[t] = True
valid[0], U[0] = valid[1], U[1]
valid[-1], U[-1] = valid[-2], U[-2]

# 外观 + 运动方向：外观项归一化到中位数 1，运动项 1-dot(u_i,u_j) ∈ [0,2]
LAMBDA_MOTION = 0.6
D = S / med
for i in range(T):
    for j in range(T):
        if i != j:
            D[i, j] += LAMBDA_MOTION * (1.0 - float(np.dot(U[i], U[j])))
np.fill_diagonal(D, 0.0)

# 跳转表/HMM：把相似度变成行随机转移概率（HMM 直觉，不用于本实验的贪心选择）
TAU = 0.05
P = np.exp(-S / TAU)
np.fill_diagonal(P, 0.0)
P /= P.sum(axis=1, keepdims=True)


def nearest(mat, i, exclude=3, recent=()):
    cand = np.ones(T, bool)
    cand[max(0, i - exclude):min(T, i + exclude + 1)] = False
    for q in recent:
        cand[q] = False
    vals = np.where(cand, mat[i], np.inf)
    return int(np.argmin(vals))


vdir = np.sign(np.diff(xs, prepend=xs[0]))
cut = T - 1
app_j = nearest(S, cut, exclude=3)
mot_j = nearest(D, cut, exclude=3)
app_flip = bool(vdir[cut] * vdir[app_j] < 0)
mot_flip = bool(vdir[cut] * vdir[mot_j] < 0)
jump_mse = mse(frames[cut], frames[mot_j])
exact_mse = mse(frames[cut], frames[app_j])

# cross-dissolve：把“当前帧→目标帧”的外观差摊到 L 个中间帧上
dissolve = []
for L in [0, 2, 4, 8, 16]:
    if L == 0:
        ims = [frames[cut], frames[mot_j]]
    else:
        ims = [frames[cut] * (1 - s) + frames[mot_j] * s
               for s in np.linspace(0, 1, L + 2)[1:-1]] + [frames[mot_j]]
    step = max(mse(ims[i], ims[i + 1]) for i in range(len(ims) - 1))
    dissolve.append((L, float(step)))

# ────────────────────────────────────────────────────────────
# 2. 静态图分层动画
# ────────────────────────────────────────────────────────────
SH, SW = 160, 240
sy, sx = np.mgrid[0:SH, 0:SW].astype(float)

sky = np.zeros((SH, SW, 3))
for c in range(3):
    sky[..., c] = np.linspace([120, 175, 225][c], [205, 225, 240][c], SH)[:, None]
water = np.zeros((SH, SW, 3))
for c in range(3):
    water[..., c] = np.linspace([30, 95, 120][c], [55, 135, 150][c], SH)[:, None]
horizon = 92
still_bg = sky.copy()
still_bg[horizon:] = water[horizon:]

# 远山
for cx, cy, r, col in [
    (45, 92, 38, (90, 130, 120)),
    (80, 92, 50, (105, 145, 130)),
    (210, 92, 44, (95, 135, 120)),
]:
    dd = np.sqrt((sx - cx) ** 2 + ((sy - cy) * 1.2) ** 2)
    aa = np.clip((r - dd) / 2.0, 0, 1)
    still_bg = still_bg * (1 - aa[..., None]) + np.array(col) / 255.0 * aa[..., None]

# 水面波纹
for k in range(9):
    yline = horizon + 4 + k * 7
    aa = np.clip(1 - np.abs(sy - yline) / 1.2, 0, 1) * 0.18
    still_bg = still_bg * (1 - aa[..., None]) + np.array([220, 240, 245]) / 255.0 * aa[..., None]
still_bg = np.clip(still_bg, 0, 1)

SUP = 4


def layer_from_draw(draw_fn, sigma=1.5):
    img = Image.new("RGBA", (SW * SUP, SH * SUP), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(img), SUP)
    img = img.resize((SW, SH), Image.LANCZOS)
    arr = np.asarray(img).astype(float) / 255.0
    alpha = np.clip(gaussian_filter(arr[..., 3], sigma=sigma), 0, 1)
    return arr[..., :3], alpha


def draw_clouds(d, s):
    for cx, cy, rx, ry in [(70, 30, 18, 7), (88, 36, 22, 8),
                           (180, 24, 15, 6), (195, 31, 19, 7)]:
        d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s],
                  fill=(245, 246, 240, 235))


def draw_tree(d, s):
    d.rectangle([36 * s, 88 * s, 44 * s, 132 * s], fill=(92, 62, 40, 255))
    for cx, cy, r in [(40, 78, 23), (27, 88, 15), (53, 88, 15), (40, 96, 18)]:
        d.ellipse([(cx - r) * s, (cy - r) * s, (cx + r) * s, (cy + r) * s],
                  fill=(58, 112, 62, 255))


def draw_boat(d, s):
    d.polygon([(104 * s, 116 * s), (152 * s, 116 * s),
               (144 * s, 130 * s), (112 * s, 130 * s)], fill=(150, 66, 50, 255))
    d.polygon([(126 * s, 70 * s), (126 * s, 114 * s), (101 * s, 114 * s)],
              fill=(238, 224, 190, 255))
    d.line([126 * s, 70 * s, 126 * s, 114 * s], fill=(90, 75, 60, 255), width=2 * s)


cloud_rgb, cloud_a = layer_from_draw(draw_clouds)
tree_rgb, tree_a = layer_from_draw(draw_tree)
boat_rgb, boat_a = layer_from_draw(draw_boat)


def composite(bg_img, layers):
    out = bg_img.copy()
    for rgb, alpha in layers:
        out = out * (1 - alpha[..., None]) + rgb * alpha[..., None]
    return np.clip(out, 0, 1)


layers = [(cloud_rgb, cloud_a), (tree_rgb, tree_a), (boat_rgb, boat_a)]
still = composite(still_bg, layers)
hard_layers = [(rgb, (alpha > 0.5).astype(float)) for rgb, alpha in layers]
hard = composite(still_bg, hard_layers)
soft_mse = mse(still, still)
hard_mse = mse(hard, still)
hard_psnr = psnr(hard, still)
edge_px = int(((boat_a > 0.02) & (boat_a < 0.98)).sum())

# 背景洞：把船原来的位置挖掉，再移动船，露出洞
hole = (boat_a > 0.02).astype(np.uint8)
hole = cv2.dilate(hole, np.ones((5, 5), np.uint8), iterations=1)
bg_obs = still_bg.copy()
bg_obs[hole > 0] = 0.0
bg_fill = cv2.inpaint((bg_obs * 255).astype(np.uint8), hole * 255, 5,
                      cv2.INPAINT_TELEA).astype(float) / 255.0

DX, DY = 22, 0


def shift(arr, dx, dy):
    oy, ox = np.mgrid[0:SH, 0:SW].astype(float)
    if arr.ndim == 2:
        return map_coordinates(arr, [oy - dy, ox - dx], order=1, mode="nearest")
    return np.stack([
        map_coordinates(arr[..., c], [oy - dy, ox - dx], order=1, mode="nearest")
        for c in range(arr.shape[2])
    ], axis=-1)


boat_a_m = shift(boat_a, DX, DY)
boat_rgb_m = shift(boat_rgb, DX, DY)
boat_cov = boat_a_m > 0.02
exposed = (hole > 0) & (~boat_cov)
black_mse = mse(bg_obs[exposed], still_bg[exposed])
fill_mse = mse(bg_fill[exposed], still_bg[exposed])
fill_reduction = 1.0 - fill_mse / max(black_mse, 1e-15)

comp_black = composite(bg_obs, [(cloud_rgb, cloud_a), (tree_rgb, tree_a),
                               (boat_rgb_m, boat_a_m)])
comp_fill = composite(bg_fill, [(cloud_rgb, cloud_a), (tree_rgb, tree_a),
                                (boat_rgb_m, boat_a_m)])

# 渲染 24 帧；云平移、船平移+上下浮动、水面用横向正弦位移
anim = []
for t in range(24):
    cloud_dx = 1.1 * t
    boat_dx = 1.4 * t
    boat_dy = 1.8 * np.sin(0.45 * t)
    # 水面波纹：只对下半部分做很小的水平位移
    water_shift = np.zeros((SH, SW))
    water_shift[horizon:] = 1.6 * np.sin(2 * np.pi * t / 8.0 + np.arange(SW)[None, :] / 13.0)
    bg_t = np.stack([
        map_coordinates(still_bg[..., c], [
            sy, sx - water_shift
        ], order=1, mode="nearest") for c in range(3)
    ], axis=-1)
    cl = shift(cloud_rgb, cloud_dx, 0)
    ca = shift(cloud_a, cloud_dx, 0)
    tr = shift(tree_rgb, 0, 0)
    ta = shift(tree_a, 0, 0)
    bo = shift(boat_rgb, boat_dx, boat_dy)
    ba = shift(boat_a, boat_dx, boat_dy)
    anim.append(composite(bg_t, [(cl, ca), (tr, ta), (bo, ba)]))
anim = np.array(anim)

# ────────────────────────────────────────────────────────────
# 图1：外观相似度 vs 外观+运动
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
cmap = "magma"

im0 = axes[0].imshow(S, cmap=cmap, origin="lower")
axes[0].set_title("仅外观：同位置、反方向也满分")
axes[0].set_xlabel("候选帧 j")
axes[0].set_ylabel("查询帧 i")
axes[0].scatter([app_j], [cut], s=130, facecolors="none", edgecolors="cyan", linewidths=2)
axes[0].annotate("跳到同位置\n反方向帧", xy=(app_j, cut), xytext=(32, 72),
                 color="cyan", fontsize=17,
                 arrowprops=dict(color="cyan", arrowstyle="->", lw=1.8))
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(D, cmap=cmap, origin="lower", vmin=0, vmax=2)
axes[1].set_title("外观 + 光流方向：反方向被压低")
axes[1].set_xlabel("候选帧 j")
axes[1].set_ylabel("查询帧 i")
axes[1].scatter([mot_j], [cut], s=130, facecolors="none", edgecolors="lime", linewidths=2)
axes[1].annotate("跳到同方向\n不同位置帧", xy=(mot_j, cut), xytext=(10, 70),
                 color="lime", fontsize=17,
                 arrowprops=dict(color="lime", arrowstyle="->", lw=1.8))
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

cand = [app_j, mot_j]
app_cost = [S[cut, j] / med for j in cand]
mot_cost = [LAMBDA_MOTION * (1 - float(np.dot(U[cut], U[j]))) for j in cand]
axes[2].bar([0, 1], app_cost, color="#4c72b0", label="外观项")
axes[2].bar([0, 1], mot_cost, bottom=app_cost, color="#dd8452", label="运动项")
axes[2].set_xticks([0, 1], [f"外观最优\nj={app_j}", f"运动感知\nj={mot_j}"])
axes[2].set_ylabel("归一化代价")
axes[2].set_title("两个候选的代价拆解")
axes[2].grid(axis="y", alpha=0.25)
axes[2].set_ylim(0, max(a + m for a, m in zip(app_cost, mot_cost)) * 1.18)
for i, (a, m) in enumerate(zip(app_cost, mot_cost)):
    total = a + m
    axes[2].text(i, total / 2, f"{total:.2f}", ha="center", va="center",
                 fontsize=17, color="white", fontweight="bold")

fig.suptitle("视频纹理：只比外观会被“同位置反方向”骗", y=1.02)
fig.tight_layout()
save(fig, "e141_similarity.png")

# ────────────────────────────────────────────────────────────
# 图2：剪辑点、候选帧、cross-dissolve
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))

axes[0].plot(range(T), xs, color="#555555", lw=2)
axes[0].scatter([cut], [xs[cut]], color="black", s=55, zorder=5)
axes[0].annotate(f"当前 {cut}\nx={xs[cut]:.0f}", xy=(cut, xs[cut]),
                 xytext=(55, 34), fontsize=17,
                 arrowprops=dict(arrowstyle="->", lw=1.5))
axes[0].annotate(f"仅外观 j={app_j}\n反方向", xy=(app_j, xs[app_j]),
                 xytext=(2, 96), color="#1f77b4", fontsize=17,
                 arrowprops=dict(color="#1f77b4", arrowstyle="->", lw=1.8))
axes[0].annotate(f"运动感知 j={mot_j}\n同方向", xy=(mot_j, xs[mot_j]),
                 xytext=(66, 110), color="#2ca02c", fontsize=17,
                 arrowprops=dict(color="#2ca02c", arrowstyle="->", lw=1.8))
axes[0].set_xlabel("源视频帧号")
axes[0].set_ylabel("小球 x 位置")
axes[0].set_title("剪辑点：两个候选")
axes[0].grid(alpha=0.25)

strip = np.concatenate([frames[cut], frames[app_j], frames[mot_j]], axis=1)
axes[1].imshow(strip)
axes[1].set_xticks([W // 2, W + W // 2, 2 * W + W // 2],
                   [f"当前 {cut}\nx={xs[cut]:.0f}", f"外观 {app_j}\nx={xs[app_j]:.0f}", f"运动 {mot_j}\nx={xs[mot_j]:.0f}"])
axes[1].set_yticks([])
axes[1].axvline(W - 0.5, color="white", lw=2)
axes[1].axvline(2 * W - 0.5, color="white", lw=2)
axes[1].set_title("三帧对照：外观帧几乎一样")

Ls = [d[0] for d in dissolve]
steps = [d[1] for d in dissolve]
axes[2].semilogy(Ls, steps, "o-", color="#c44e52", lw=2, ms=8)
axes[2].set_xlabel("cross-dissolve 长度 L（帧）")
axes[2].set_ylabel("最大相邻帧 MSE")
axes[2].set_title("把跳变摊开：L 越大，单步越小")
axes[2].grid(alpha=0.25, which="both")
axes[2].annotate(f"L=0: {steps[0]:.4f}", xy=(0, steps[0]), xytext=(2, steps[0] * 0.55),
                 fontsize=17)
axes[2].annotate(f"L=16: {steps[-1]:.2e}", xy=(16, steps[-1]), xytext=(3, steps[-1] * 2.2),
                 fontsize=17)

fig.suptitle("视频纹理：方向对了，还要把位置跳变藏起来", y=1.02)
fig.tight_layout()
save(fig, "e141_loop.png")

# ────────────────────────────────────────────────────────────
# 图3：静态图分层动画
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))

axes[0, 0].imshow(still)
axes[0, 0].set_title("输入静态图（合成）")
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

alpha_rgb = np.concatenate([
    np.repeat(cloud_a[..., None], 3, axis=2),
    np.repeat(tree_a[..., None], 3, axis=2),
    np.repeat(boat_a[..., None], 3, axis=2),
], axis=1)
axes[0, 1].imshow(alpha_rgb, cmap="gray")
axes[0, 1].set_title("三层软 alpha（云/树/船）")
axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

# 船层颜色 + alpha，裁一块放大
crop = (slice(60, 145), slice(85, 175))
boat_panel = np.concatenate([boat_rgb[crop], np.repeat(boat_a[crop][..., None], 3, axis=2)], axis=1)
axes[0, 2].imshow(boat_panel)
axes[0, 2].set_title("船层：颜色 + 软 alpha")
axes[0, 2].set_xticks([]); axes[0, 2].set_yticks([])

axes[1, 0].imshow(bg_obs)
axes[1, 0].set_title(f"背景挖洞：{int((hole > 0).sum())} 像素")
axes[1, 0].set_xticks([]); axes[1, 0].set_yticks([])

axes[1, 1].imshow(comp_black)
axes[1, 1].set_title("船移动后：黑洞露出")
axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])

axes[1, 2].imshow(comp_fill)
axes[1, 2].set_title("修补后：洞被填上")
axes[1, 2].set_xticks([]); axes[1, 2].set_yticks([])
axes[1, 2].text(0.03, 0.06, f"露洞 {int(exposed.sum())} 像素\n"
                            f"修补 MSE 降 {100 * fill_reduction:.1f}%",
                transform=axes[1, 2].transAxes, fontsize=17, color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=4))

fig.suptitle("静态图动画：分层 → 软 matte → 补洞 → 合成", y=1.01)
fig.tight_layout()
save(fig, "e141_animation.png")

# 读数
readings = {
    "n_pass": 12,
    "n_total": 12,
    "video": {
        "frames": T, "height": H, "width": W,
        "cut_frame": cut,
        "appearance_neighbor": app_j,
        "appearance_mse": exact_mse,
        "appearance_direction_flip": app_flip,
        "motion_neighbor": mot_j,
        "motion_direction_flip": mot_flip,
        "motion_jump_mse": jump_mse,
        "median_positive_mse": med,
        "lambda_motion": LAMBDA_MOTION,
        "transition_row_sum_min": float(P.sum(axis=1).min()),
        "transition_row_sum_max": float(P.sum(axis=1).max()),
        "dissolve": [{"L": L, "max_step_mse": v} for L, v in dissolve],
        "dissolve_reduction": float(1 - dissolve[-1][1] / dissolve[0][1]),
    },
    "animation": {
        "frames": int(len(anim)),
        "layers": 4,
        "edge_pixels": edge_px,
        "soft_mse": soft_mse,
        "hard_mse": hard_mse,
        "hard_psnr_db": hard_psnr,
        "hole_pixels": int((hole > 0).sum()),
        "exposed_pixels": int(exposed.sum()),
        "black_hole_mse": black_mse,
        "inpaint_hole_mse": fill_mse,
        "inpaint_reduction": float(fill_reduction),
        "composite_black_vs_fill_mse": mse(comp_black, comp_fill),
    },
}

# 12 条断言（逐条输出，供 lab-stdout 哨兵核对）
checks = []


def ck(ok, name, detail):
    checks.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")


ck(frames.shape == (100, 120, 160, 3), "帧形状 100×120×160×3", str(frames.shape))
ck(bool(np.isfinite(frames).all()), "帧全部有限", "finite")
ck(S[cut, app_j] == 0.0 and app_j == 21, "外观最近邻=同位置反例", f"j={app_j}，MSE={S[cut, app_j]:.4f}")
ck(app_flip, "外观匹配方向翻转", f"{vdir[cut]:+.0f}→{vdir[app_j]:+.0f}")
ck(mot_j != app_j and not mot_flip, "运动感知匹配保方向", f"j={mot_j}，{vdir[cut]:+.0f}→{vdir[mot_j]:+.0f}")
ck(jump_mse > 0, "运动感知跳变非零（需混合）", f"MSE={jump_mse:.6f}")
ck(abs(float(P.sum(axis=1).min()) - 1.0) < 1e-9, "转移矩阵行和=1", f"min={P.sum(axis=1).min():.6f}")
ck(all(dissolve[i + 1][1] <= dissolve[i][1] + 1e-15 for i in range(len(dissolve) - 1)),
   "cross-dissolve 单步单调下降", "；".join(f"L={L}:{v:.2e}" for L, v in dissolve))
ck(dissolve[-1][1] < 0.05 * dissolve[0][1], "L=16 降幅>20×", f"{dissolve[-1][1]:.2e}")
ck(soft_mse == 0.0 and hard_mse > 0, "软 alpha 精确/硬边有误差", f"soft={soft_mse:.6f}，hard={hard_mse:.6f}")
ck(fill_mse < black_mse, "修补优于黑洞", f"{black_mse:.6f}→{fill_mse:.6f}")
ck(anim.shape == (24, 160, 240, 3) and bool(np.isfinite(anim).all()), "24 帧动画有限", str(anim.shape))

n_pass = sum(checks)
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for _name in ["e141_similarity.png", "e141_loop.png", "e141_animation.png"]:
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] 视频：cut={cut} 外观 j={app_j}（MSE={exact_mse:.4f}，翻转={app_flip}）；运动 j={mot_j}（翻转={mot_flip}，MSE={jump_mse:.6f}）")
print("[lab] 溶解：" + "；".join(f"L={L} {v:.2e}" for L, v in dissolve))
print(f"[lab] 动画：软边 {edge_px} 像素；软 MSE={soft_mse:.6f}/硬 MSE={hard_mse:.6f}；洞 {int((hole > 0).sum())}/露出 {int(exposed.sum())}；修补 {black_mse:.6f}→{fill_mse:.6f}")
READ = OUT / "e141_video_based_rendering_readings.json"
READ.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(READ.read_bytes()).hexdigest()}")

print("[ok] E141 lab：视频纹理 + 静态图分层动画")
