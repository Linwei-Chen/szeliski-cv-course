#!/usr/bin/env python3
"""E115 单目深度沙盘：地面先验、已知尺寸与尺度歧义（合成单视图，不训练网络）。"""
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
H, W = 240, 320; F = 300.0; CX, CY = W / 2.0, H // 2
yy, xx = np.mgrid[0:H, 0:W]
dx = (xx - CX) / F; dy = (yy - CY) / F; dz = np.ones_like(dx)
H1, WALL1, R1, C1 = 1.5, 12.0, 0.5, np.array([0.7, 0.2, 7.0])

def render(scale=1.0):
    h = H1 * scale; wall_z = WALL1 * scale; checker = 0.5 * scale
    brick_x = 0.8 * scale; brick_y = 0.4 * scale
    depth = np.full((H, W), np.inf, np.float32)
    color = np.ones((H, W, 3), np.float32) * np.array([0.72, 0.82, 0.92], np.float32)
    obj = np.zeros((H, W), np.uint8)
    # 地面：Y=+h（图像 y 向下），t=h/dy
    dy_safe = np.where(dy == 0, 1.0, dy)
    t = np.where(dy > 0, h / dy_safe, -1.0)
    ok = (t > 0) & (t < wall_z)
    X = np.where(ok, t * dx, 0.0); Z = np.where(ok, t, 0.0)
    chk = ((np.floor(X / checker) + np.floor(Z / checker)) % 2).astype(int)
    fc = np.where(chk[..., None] == 0, np.array([0.80, 0.76, 0.68]), np.array([0.55, 0.58, 0.63]))
    depth[ok] = t[ok]; color[ok] = fc[ok]; obj[ok] = 1
    # 后墙 Z=wall_z
    Xw = wall_z * dx; Yw = wall_z * dy
    ok = (np.abs(Xw) < 6 * scale) & (Yw > -2.5 * scale) & (Yw < 1.5 * scale) & (wall_z < depth)
    br = ((np.floor(Xw / brick_x) + np.floor(Yw / brick_y)) % 2).astype(int)
    wc = np.where(br[..., None] == 0, np.array([0.72, 0.60, 0.52]), np.array([0.58, 0.47, 0.42]))
    depth[ok] = wall_z; color[ok] = wc[ok]; obj[ok] = 2
    # 悬浮球
    C = C1 * scale; r = R1 * scale
    a = dx * dx + dy * dy + 1.0
    b = -2 * (dx * C[0] + dy * C[1] + C[2])
    c = C @ C - r * r
    disc = b * b - 4 * a * c
    ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    ok = (disc > 0) & (ts > 0) & (ts < depth)
    depth[ok] = ts[ok]; obj[ok] = 3
    P = np.stack([ts * dx - C[0], ts * dy - C[1], ts - C[2]], axis=-1) / r
    L = np.array([0.3, -0.5, 0.8]); L = L / np.linalg.norm(L)
    lam = np.clip(0.45 + 0.55 * np.maximum(0, (P * L).sum(-1)), 0, 1)
    sc = np.stack([0.95 * lam, 0.45 * lam, 0.15 * lam], axis=-1)
    color[ok] = sc[ok]
    # 地面上的方块
    lo = np.array([-1.35, 0.8, 4.15]) * scale; hi = np.array([-0.65, 1.5, 4.85]) * scale
    tmin = np.full((H, W), -np.inf); tmax = np.full((H, W), np.inf)
    for d, li, hi_i in [(dx, lo[0], hi[0]), (dy, lo[1], hi[1]), (dz, lo[2], hi[2])]:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = li / d; t2 = hi_i / d
        tmin = np.maximum(tmin, np.minimum(t1, t2)); tmax = np.minimum(tmax, np.maximum(t1, t2))
    ok = (tmax >= tmin) & (tmin > 0) & (tmin < depth)
    depth[ok] = tmin[ok]; color[ok] = np.array([0.20, 0.42, 0.78]); obj[ok] = 4
    return np.clip(color, 0, 1), depth, obj

img1, depth1, obj1 = render(1.0)
img2, depth2, obj2 = render(2.0)
valid = np.isfinite(depth1); valid2 = np.isfinite(depth2); common = valid & valid2
valid_frac = float(valid.mean())
floor = obj1 == 1; wall = obj1 == 2; sphere = obj1 == 3; cube = obj1 == 4

# 地面先验：Z = f*h/(y-cy)
Zg = np.full((H, W), np.inf, np.float32)
Zg[CY + 1:, :] = F * H1 / (yy[CY + 1:, :] - CY)
def region_mae(mask):
    mm = mask & np.isfinite(Zg)
    return float(np.abs(Zg[mm] - depth1[mm]).mean()) if mm.any() else float("nan")
floor_mae = region_mae(floor); wall_mae = region_mae(wall); sphere_mae = region_mae(sphere); cube_mae = region_mae(cube)

# 已知球半径的尺寸线索
sphere_area = int(sphere.sum()); r_px = float(np.sqrt(sphere_area / np.pi))
z_size = float(F * R1 / r_px); z_center = float(C1[2]); size_err = abs(z_size - z_center)

# 尺度歧义：整体放大 2 倍，图像不变、深度整体 ×2
img_diff = float(np.abs(img1 - img2).max())
depth_ratio = float(np.median(depth2[common] / depth1[common]))
depth_corr = float(np.corrcoef(depth1[common], depth2[common])[0, 1])
# 相机高度加倍，地面先验整体 ×2
Zg2 = np.full((H, W), np.inf, np.float32)
Zg2[CY + 1:, :] = F * (2 * H1) / (yy[CY + 1:, :] - CY)
mm = floor & np.isfinite(Zg)
zg_ratio = float(np.median(Zg2[mm] / Zg[mm]))

# 图1 场景与真值
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
axes[0].imshow(img1); axes[0].set_title("单张合成图：地面 + 后墙 + 悬浮球 + 方块", fontsize=20, color=INK); axes[0].axis("off")
dvis = np.where(valid, depth1, np.nan)
o = axes[1].imshow(dvis, cmap="viridis", vmin=2, vmax=12); axes[1].set_title("真实深度 Z（米）", fontsize=20, color=INK); axes[1].axis("off")
cb = fig.colorbar(o, ax=axes[1], fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=19)
fig.suptitle("单目深度：一张图，多个深度层", fontsize=22, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "e115_scene.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 地面先验的捷径与失败
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.7))
Zg_vis = np.where(np.isfinite(Zg), np.clip(Zg, 0, 20), np.nan)
o = axes[0].imshow(Zg_vis, cmap="magma", vmin=0, vmax=20); axes[0].set_title("地面先验预测的深度", fontsize=20, color=INK); axes[0].axis("off")
axes[0].text(160, 45, "地平线以上\n先验无定义", ha="center", va="center", fontsize=15, color=GRAY)
cb = fig.colorbar(o, ax=axes[0], fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=19)
names = ["地面", "后墙", "悬浮球"]; vals = [floor_mae, wall_mae, sphere_mae]
bars = axes[1].bar(names, vals, color=[GREEN, ACCENT, BLUE], width=0.55)
axes[1].set_ylabel("平均绝对误差 / 米", fontsize=18); axes[1].set_title("同一先验，换个区域就崩", fontsize=20, color=INK)
axes[1].tick_params(labelsize=18); axes[1].grid(axis="y", alpha=0.2)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f}", ha="center", fontsize=18, color=INK)
axes[1].set_ylim(0, max(vals) * 1.18)
fig.suptitle("地面位置是一个强先验，也是一个封闭世界捷径", fontsize=21, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "e115_shortcut.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 尺度歧义
fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
axes[0, 0].imshow(img1); axes[0, 0].set_title("原场景", fontsize=20, color=INK); axes[0, 0].axis("off")
axes[0, 1].imshow(img2); axes[0, 1].set_title("整体放大 2 倍（图像相同）", fontsize=20, color=INK); axes[0, 1].axis("off")
for ax, d, title in [(axes[1, 0], depth1, "深度 1×"), (axes[1, 1], depth2, "深度 2×")]:
    o = ax.imshow(np.where(np.isfinite(d), d, np.nan), cmap="viridis", vmin=2, vmax=24)
    ax.set_title(title, fontsize=20, color=INK); ax.axis("off")
    cb = fig.colorbar(o, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=19)
fig.suptitle("单张图看不出绝对尺度：图像一样，深度可以整体差 2 倍", fontsize=22, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "e115_scale.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

assertions = [
    ("单图与深度图尺寸正确", img1.shape == (H, W, 3) and depth1.shape == (H, W), f"{img1.shape}"),
    ("有效深度占比超过 0.7", valid_frac > 0.7, f"{valid_frac:.3f}"),
    ("地面像素足够多", int(floor.sum()) > 20000, f"{int(floor.sum())} 像素"),
    ("地面先验在地面精确", floor_mae < 0.01, f"{floor_mae:.3f} 米"),
    ("地面先验在后墙失效", wall_mae > 10, f"{wall_mae:.1f} 米"),
    ("地面先验在悬浮球失效", sphere_mae > 10, f"{sphere_mae:.1f} 米"),
    ("已知球半径的尺寸线索可用", size_err < 0.1, f"{z_size:.2f} vs {z_center:.2f} 米"),
    ("整体放大 2 倍图像不变", img_diff < 1e-6, f"{img_diff:.1e}"),
    ("整体放大 2 倍深度翻倍", abs(depth_ratio - 2.0) < 1e-6, f"{depth_ratio:.1f} 倍"),
    ("相对深度顺序完全一致", depth_corr > 0.999999, f"{depth_corr:.6f}"),
]
n_pass = sum(bool(ok) for _, ok, _ in assertions)
readings = {
    "n_pass": n_pass, "n_total": len(assertions), "H": H, "W": W, "C": 3, "F": F,
    "valid_frac": valid_frac, "valid_frac_3": round(valid_frac, 3),
    "n_floor": int(floor.sum()), "n_wall": int(wall.sum()), "n_sphere": int(sphere.sum()), "n_cube": int(cube.sum()),
    "floor_mae": floor_mae, "floor_mae_2": round(floor_mae, 2),
    "wall_mae": wall_mae, "wall_mae_1": round(wall_mae, 1),
    "sphere_mae": sphere_mae, "sphere_mae_1": round(sphere_mae, 1),
    "cube_mae": cube_mae, "cube_mae_2": round(cube_mae, 2),
    "sphere_area": sphere_area, "r_px": r_px, "z_size": z_size, "z_size_2": round(z_size, 2),
    "z_center": z_center, "size_err": size_err,
    "img_diff": img_diff, "depth_ratio": depth_ratio, "depth_corr": depth_corr,
    "zg_ratio": zg_ratio,
}
print(f"[lab] E115 单目深度开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] {W}x{H}｜f {F:.0f}｜相机高 {H1} 米｜有效 {valid_frac:.3f}｜地面 {int(floor.sum())}｜墙 {int(wall.sum())}｜球 {int(sphere.sum())}｜方块 {int(cube.sum())}")
for name, ok, detail in assertions:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] 地面先验 MAE：地面 {floor_mae:.2f}｜后墙 {wall_mae:.1f}｜悬浮球 {sphere_mae:.1f}｜方块 {cube_mae:.2f}")
print(f"[lab] 尺寸线索：球面积 {sphere_area} 像素，r_px {r_px:.2f}，Z_est {z_size:.2f} vs 中心 {z_center:.2f} 米")
print(f"[lab] 尺度歧义：图像最大差 {img_diff:.1e}｜深度比 {depth_ratio:.1f}｜相关 {depth_corr:.6f}｜相机高度比 {zg_ratio:.1f}")
for p in sorted(FIG.glob("e115_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e115_mono_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e115_mono_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e115_mono_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(assertions)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(assertions)-n_pass} / 共 {len(assertions)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(assertions): raise SystemExit(1)
