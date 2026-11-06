#!/usr/bin/env python3
"""E111 视差即深度：合成立体对 + 块匹配 + 深度反演。"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
H, W = 480, 640; F = 800.0; CX, CY = 320.0, 240.0; B = 0.2
DEPTHS = [1.0, 2.0, 4.0, 8.0]
DISP = [F * B / z for z in DEPTHS]
rng = np.random.default_rng(111)

def tex(w, h, seed, base):
    r = np.random.default_rng(seed)
    lo = r.random((max(4, h // 8), max(4, w // 8), 3)).astype(np.float32)
    im = Image.fromarray((lo * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    a = np.asarray(im).astype(np.float32) / 255
    a = 0.7 * a + 0.3 * np.array(base, dtype=np.float32)
    return np.clip(a, 0, 1)

left = np.zeros((H, W, 3), np.float32); gt = np.full((H, W), np.nan, np.float32)
bg = tex(W, H, 1, (0.35, 0.45, 0.55))
left[:] = bg
gt[:] = DISP[3]
layers = [
    (200, 360, 100, 380, 0, (0.85, 0.45, 0.35)),
    (380, 540, 60, 420, 1, (0.35, 0.75, 0.45)),
    (560, 630, 120, 360, 2, (0.45, 0.45, 0.85)),
]
for x0, x1, y0, y1, idx, base in layers:
    t = tex(x1 - x0, y1 - y0, 10 + idx, base)
    left[y0:y1, x0:x1] = t
    gt[y0:y1, x0:x1] = DISP[idx]

def shifted(img, d, fill):
    out = np.empty_like(img); out[:] = fill[:, None, :]
    if d <= 0:
        out[:] = img
    else:
        out[:, :W - d] = img[:, d:]
        out[:, W - d:] = img[:, -d:][:, ::-1]
    return out

right = shifted(bg, int(round(DISP[3])), bg[:, -1])
for x0, x1, y0, y1, idx, _ in sorted(layers, key=lambda z: -z[4]):
    d = int(round(DISP[idx]))
    right[y0:y1, x0 - d:x1 - d] = left[y0:y1, x0:x1]

left_u8 = (np.clip(left, 0, 1) * 255).astype(np.uint8)
right_u8 = (np.clip(right, 0, 1) * 255).astype(np.uint8)
Image.fromarray(left_u8).save(OUT / "e111_left.png")
Image.fromarray(right_u8).save(OUT / "e111_right.png")

# 块匹配 SAD：窗口 9×9，搜索 0–170
gl = left.mean(axis=2); gr = right.mean(axis=2)
WIN = 9; DMAX = 170
best = np.full((H, W), np.inf, np.float32); est = np.zeros((H, W), np.float32)
for d in range(DMAX + 1):
    diff = np.full((H, W), 1.0, np.float32)
    valid_d = np.zeros((H, W), bool)
    if d < W:
        diff[:, d:] = np.abs(gl[:, d:] - gr[:, :W - d])
        valid_d[:, d:] = True
    cost = uniform_filter(diff, size=WIN, mode="nearest")
    upd = (cost < best) & valid_d
    best[upd] = cost[upd]; est[upd] = d
# 右图深度标签 + 遮挡掩码（左图背景点投到右图后被前景挡住）
right_gt = np.full((H, W), DEPTHS[3], np.float32)
gt_depth = np.full((H, W), DEPTHS[3], np.float32)
for x0, x1, y0, y1, idx, _ in sorted(layers, key=lambda z: -z[4]):
    d = int(round(DISP[idx]))
    right_gt[y0:y1, x0 - d:x1 - d] = DEPTHS[idx]
    gt_depth[y0:y1, x0:x1] = DEPTHS[idx]
dmap = np.zeros((H, W), np.float32)
for idx in range(4):
    dmap[np.abs(gt - DISP[idx]) < 1e-6] = DISP[idx]
xs = np.arange(W)[None, :] - dmap
xrc = np.clip(xs, 0, W - 1).astype(int)
occl = (xs < 0) | (np.take_along_axis(right_gt, xrc, axis=1) < gt_depth)
# 有效区：GT 恒定窗口 + 边界内缩 + 非遮挡
valid = np.zeros((H, W), bool)
for idx in range(4):
    m = np.abs(gt - DISP[idx]) < 1e-6
    valid |= binary_erosion(m, structure=np.ones((WIN, WIN), bool))
valid[:8, :] = False; valid[-8:, :] = False; valid[:, :DMAX] = False; valid[:, -8:] = False
valid &= ~occl
err = np.abs(est - gt)
mae = float(err[valid].mean()); acc1 = float((err[valid] <= 1).mean() * 100)

# 深度反演与边界/内部误差
def depth_from_disp(d): return F * B / np.maximum(d, 1e-6)
centers = [(280, 240), (460, 240), (595, 240), (100, 30)]
center_d = [float(est[y, x]) for x, y in centers]
center_gt = [float(gt[y, x]) for x, y in centers]
center_depth = [depth_from_disp(d) for d in center_d]
center_rel = [abs(z - zt) / zt * 100 for z, zt in zip(center_depth, [1, 2, 4, 8])]
# 边界带（距深度跳变 10px 内）与内部
near_edge = np.zeros((H, W), bool)
for idx in range(4):
    m = np.abs(gt - DISP[idx]) < 1e-6
    near_edge |= (binary_erosion(m, structure=np.ones((WIN, WIN), bool)) != m)
edge_err = float(err[near_edge & (gt > 0)].mean()); inter_err = float(err[valid].mean())
# 解析关系
prod = np.array([d * z for d, z in zip(DISP, DEPTHS)])
prod_dev = float(np.max(np.abs(prod - F * B)))
ratio_near_far = float(DISP[0] / DISP[3])
d_B2 = float(F * (2 * B) / DEPTHS[0]); d_f2 = float((2 * F) * B / DEPTHS[0])

# 图1 几何俯视 + 图像坐标
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.25, 1]})
ax1.set_xlim(-0.2, 1.25); ax1.set_ylim(-0.25, 1.8); ax1.axis("off"); ax1.set_aspect("equal")
cL, cR = 0.0, 0.5; P = np.array([0.9, 1.5])
for cx0, name in [(cL, "左相机"), (cR, "右相机")]:
    ax1.plot([cx0, cx0], [-0.06, 0.06], color=INK, lw=2)
    ax1.plot(cx0, 0, "o", color=BLUE, ms=10)
    ax1.text(cx0, -0.19, name, ha="center", fontsize=15, color=INK)
ax1.plot([cL, cR], [0, 0], "-", color=INK, lw=1.5)
ax1.annotate("", xy=(cR, -0.05), xytext=(cL, -0.05), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4))
ax1.text((cL + cR) / 2, -0.085, "基线 B", fontsize=15, color=ACCENT, ha="center")
ax1.plot([cL, P[0]], [0, P[1]], color=BLUE, lw=1.6)
ax1.plot([cR, P[0]], [0, P[1]], color=GREEN, lw=1.6)
ax1.plot(*P, "o", color=ACCENT, ms=9); ax1.text(P[0] + 0.02, P[1] + 0.03, "P", fontsize=17, color=ACCENT)
ax1.annotate("", xy=(1.18, P[1]), xytext=(1.18, 0), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4))
ax1.text(1.2, P[1] / 2, "Z", fontsize=17, color=ACCENT)
ax1.set_title("俯视：同一 P 的两条视线", fontsize=16, color=INK)

ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
for yy, name, u in [(0.72, "左图像", 0.66), (0.32, "右图像", 0.36)]:
    ax2.add_patch(plt.Rectangle((0.08, yy - 0.09), 0.84, 0.18, facecolor="#eef2f7", edgecolor=GRAY, lw=1.2))
    ax2.plot(u, yy, "o", color=ACCENT, ms=10)
    ax2.text(0.5, yy + 0.13, name, fontsize=15, color=INK, ha="center")
    ax2.plot([u, u], [0.28, 0.76], "--", color=GRAY, lw=1.0)
ax2.annotate("", xy=(0.66, 0.52), xytext=(0.36, 0.52), arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.8))
ax2.text(0.51, 0.56, "d", fontsize=18, color=ACCENT, ha="center")
ax2.text(0.5, 0.08, "同一个 P：uL 与 uR 的差就是视差", fontsize=15, color=INK, ha="center")
ax2.set_title("图像坐标：d = uL − uR", fontsize=16, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e111_geometry.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 视差-深度曲线
fig, ax = plt.subplots(figsize=(9.8, 4.8))
zs = np.linspace(0.6, 10, 500); ds = F * B / zs
ax.plot(zs, ds, color=BLUE, lw=2.2)
for z, d in zip(DEPTHS, DISP):
    ax.plot(z, d, "o", color=ACCENT, ms=8); ax.text(z + 0.15, d + 4, f"Z={z:g}, d={d:.0f}", fontsize=14.5, color=INK)
ax.set_xlabel("深度 Z / 场景单位", fontsize=15); ax.set_ylabel("视差 d / 像素", fontsize=15)
ax.set_title("视差与深度成反比：越近越大，越远越小", fontsize=16, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(alpha=0.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e111_curve.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 左右图 + 真值/估计视差
fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.4))
for ax, im, title in [(axes[0, 0], left_u8, "左图"), (axes[0, 1], right_u8, "右图")]:
    ax.imshow(im); ax.set_title(title, fontsize=16, color=INK); ax.axis("off")
im1 = axes[1, 0].imshow(gt, cmap="viridis"); axes[1, 0].set_title("真值视差", fontsize=16, color=INK); axes[1, 0].axis("off")
im2 = axes[1, 1].imshow(est, cmap="viridis"); axes[1, 1].set_title("块匹配视差", fontsize=16, color=INK); axes[1, 1].axis("off")
for ax, im in [(axes[1, 0], im1), (axes[1, 1], im2)]:
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=15)
fig.tight_layout(); fig.savefig(FIG / "e111_stereo.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks = [
    ("d·Z 恒定", prod_dev < 1e-9, f"max dev {prod_dev:.2e}"),
    ("视差随深度递减", all(DISP[i] > DISP[i + 1] for i in range(3)), "→".join(f"{d:.0f}" for d in DISP)),
    ("近远比等于深度比", abs(ratio_near_far - 8.0) < 1e-9, f"{ratio_near_far:.1f}"),
    ("基线加倍视差加倍", abs(d_B2 / DISP[0] - 2) < 1e-9, f"{d_B2:.0f}"),
    ("焦距加倍视差加倍", abs(d_f2 / DISP[0] - 2) < 1e-9, f"{d_f2:.0f}"),
    ("块匹配平均误差小", mae < 1.0, f"{mae:.3f} px"),
    ("1px 内准确率高", acc1 > 90, f"{acc1:.1f}%"),
    ("中心深度误差小", max(center_rel) < 5, f"max {max(center_rel):.2f}%"),
    ("边界误差更大", edge_err > inter_err, f"{edge_err:.2f} vs {inter_err:.2f}"),
    ("三图在盘", all((FIG / f"e111_{n}.png").exists() for n in ("geometry", "curve", "stereo")), "geometry/curve/stereo"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "baseline": B, "focal": F, "width": W, "height": H,
    "z_near": DEPTHS[0], "z_far": DEPTHS[-1], "d_near": DISP[0], "d_far": DISP[-1],
    "prod_dev": prod_dev, "ratio_near_far": ratio_near_far, "d_B2": d_B2, "d_f2": d_f2,
    "mae": mae, "mae_3": round(mae, 3), "acc1": acc1, "acc1_1": round(acc1, 1),
    "center_rel_max": max(center_rel), "center_rel_max_2": round(max(center_rel), 2),
    "edge_err": edge_err, "edge_err_2": round(edge_err, 2), "inter_err": inter_err, "inter_err_2": round(inter_err, 2),
    "center_d": center_d, "center_depth": center_depth,
}
print(f"[lab] E111 视差即深度开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] 基线 {B}｜焦距 {F:.0f}｜图像 {W}x{H}｜深度 {DEPTHS}｜视差 {[round(d,1) for d in DISP]}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] d·Z={F*B:.1f}（max dev {prod_dev:.2e}）｜近远比 {ratio_near_far:.1f}｜B×2={d_B2:.0f}｜f×2={d_f2:.0f}")
print(f"[lab] 块匹配 MAE {mae:.3f} px｜1px 内 {acc1:.1f}%｜中心深度最大误差 {max(center_rel):.2f}%")
print(f"[lab] 边界误差 {edge_err:.2f} px vs 内部 {inter_err:.2f} px")
for p in sorted(FIG.glob("e111_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e111_disparity_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e111_disparity_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e111_disparity_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
