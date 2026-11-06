#!/usr/bin/env python3
"""E114 PSM-Net：4D 成本体积 + 固定 3D 聚合沙盘（合成双目，不训练网络）。

口径：本实验不是 PSM-Net 复现，也不训练任何网络。它只验证三件事——
① PSM-Net 式 4D 成本体积的形状与内存量级；
② 沿视差维做 3D 聚合，能不能把弱纹理区的错误视差拉回来；
③ 没有训练出尖锐代价峰时，softmax 视差回归会失败。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.ndimage import gaussian_filter, uniform_filter, map_coordinates

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
rng = np.random.default_rng(20261102)
H, W, D = 192, 320, 64
yy, xx = np.mgrid[0:H, 0:W]

# ── 合成场景：背景斜平面 + 近处矩形 + 圆形 + 人为弱纹理块 ──
gt = np.full((H, W), 12.0, np.float32)
gt += 0.02 * (xx - W / 2)
rect = (xx > 80) & (xx < 230) & (yy > 55) & (yy < 155)
gt[rect] = 38.0 + 0.05 * (yy[rect] - H / 2)
cx, cy, rad = 270, 70, 32
circ = (xx - cx) ** 2 + (yy - cy) ** 2 < rad * rad
gt[circ] = 25.0 + 0.03 * (xx[circ] - cx)

base = np.stack([gaussian_filter(rng.normal(size=(H, W)), 3) for _ in range(3)], axis=-1)
fine = np.stack([gaussian_filter(rng.normal(size=(H, W)), 0.7) for _ in range(3)], axis=-1)
left = 0.7 * base + 0.5 * fine
left = (left - left.min()) / (left.max() - left.min())
left += 0.08 * np.sin(xx / 4.0)[..., None]
weak = (xx > 130) & (xx < 175) & (yy > 95) & (yy < 140)
left[weak] = 0.55 + 0.01 * rng.normal(size=(int(weak.sum()), 3))
left = np.clip(left + 0.01 * rng.normal(size=left.shape), 0, 1).astype(np.float32)

X = xx + gt; Y = yy.astype(np.float32)
right = np.zeros_like(left)
for c in range(3):
    right[..., c] = map_coordinates(left[..., c], [Y, X], order=1, mode="nearest")
valid = (X >= 0) & (X < W - 1)
right = np.clip(right + 0.01 * rng.normal(size=right.shape), 0, 1).astype(np.float32)
valid[:4, :] = False; valid[-4:, :] = False; valid[:, :D] = False

Image.fromarray((left * 255).astype(np.uint8)).save(OUT / "e114_left.png")
Image.fromarray((right * 255).astype(np.uint8)).save(OUT / "e114_right.png")

# ── PSM-Net 式 4D 体积：每个视差层拼接左特征与右特征（这里特征=RGB） ──
V = np.full((H, W, D, 6), np.nan, np.float32)
for d in range(D):
    if d == 0:
        V[:, :, d, :3] = left; V[:, :, d, 3:] = right
    else:
        V[:, d:, d, :3] = left[:, d:]; V[:, d:, d, 3:] = right[:, :W - d]
valid_vox = np.isfinite(V[..., 0])
diff = np.abs(V[..., :3] - V[..., 3:])
diff = np.where(valid_vox[..., None], diff, 0.0)
cost = diff.mean(axis=-1)
cost[~valid_vox] = np.nan
valid_frac = float(valid_vox.sum() / cost.size)
Cfill = np.where(valid_vox, cost, 1.0).astype(np.float32)

raw = np.argmin(Cfill, axis=2).astype(np.float32)
C2 = uniform_filter(Cfill, size=(5, 5, 1), mode="nearest")
p2 = np.argmin(C2, axis=2).astype(np.float32)
C3 = uniform_filter(Cfill, size=(3, 3, 3), mode="nearest")
C3 = uniform_filter(C3, size=(3, 3, 3), mode="nearest")
p3 = np.argmin(C3, axis=2).astype(np.float32)

# 全局 softmax 回归（PSM-Net 式，但这里没有训练，代价峰不尖）
T_global = 0.05
logits = -C3 / T_global
logits -= logits.max(axis=2, keepdims=True)
P = np.exp(logits); P /= P.sum(axis=2, keepdims=True)
soft_global = (P * np.arange(D)).sum(axis=2).astype(np.float32)
# 局部 softmax 回归（只在最小值 ±2 内做，作为对照；不是 PSM-Net 原式）
d0 = np.argmin(C3, axis=2)
soft_local = np.zeros_like(raw)
T_local = 0.005
for y in range(H):
    for x in range(W):
        a = max(0, int(d0[y, x]) - 2); b = min(D, int(d0[y, x]) + 3)
        c = C3[y, x, a:b]
        pp = np.exp(-(c - c.min()) / T_local); pp /= pp.sum()
        soft_local[y, x] = float((pp * np.arange(a, b)).sum())

grad = np.hypot(*np.gradient(gt))
weak_m = weak & valid
bound_m = (grad > 1.0) & valid
flat_m = (~weak) & (grad <= 1.0) & valid

def metrics(pred, mask):
    e = np.abs(pred - gt)[mask]
    return float(e.mean()), float((e < 1.0).mean()), float(np.percentile(e, 90))

raw_mae, raw_acc, raw_p90 = metrics(raw, valid)
p2_mae, p2_acc, p2_p90 = metrics(p2, valid)
p3_mae, p3_acc, p3_p90 = metrics(p3, valid)
sg_mae, sg_acc, sg_p90 = metrics(soft_global, valid)
sl_mae, sl_acc, sl_p90 = metrics(soft_local, valid)
raw_weak = metrics(raw, weak_m)[0]; p2_weak = metrics(p2, weak_m)[0]; p3_weak = metrics(p3, weak_m)[0]
raw_bound = metrics(raw, bound_m)[0]; p3_bound = metrics(p3, bound_m)[0]
p2_bound = metrics(p2, bound_m)[0]

# ── PSM-Net 内存量级（论文裁剪 256×512、1/4 特征图、D=192、左右各 32 通道） ──
psm_h, psm_w, psm_d, psm_c = 256, 512, 192, 64
feat_h, feat_w = psm_h // 4, psm_w // 4
psm_bytes = psm_d * feat_h * feat_w * psm_c * 4
naive_bytes = psm_d * psm_h * psm_w * psm_c * 4
batch_bytes = psm_bytes * 12
psm_mib = psm_bytes / 2 ** 20
naive_gib = naive_bytes / 2 ** 30
batch_gib = batch_bytes / 2 ** 30
size_ratio = naive_bytes / psm_bytes

# 3D 3×3×3 与 2D 3×3 的参数量（32→32 通道）
p3d = 3 * 3 * 3 * 32 * 32 + 32
p2d = 3 * 3 * 32 * 32 + 32
param_ratio = p3d / p2d

# 图1 三类方法
fig, ax = plt.subplots(figsize=(10.0, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
ax.text(0.2, 4.95, "端到端立体匹配的三条路线", fontsize=19, color=INK, weight="bold")
cols = [
    ("传统管线 + 学习", ["学匹配代价", "学 SGM 转移代价"], "#eef2f7", BLUE),
    ("2D 端到端", ["DispNet-C 相关层", "编码器—解码器"], "#eef7ee", GREEN),
    ("3D 端到端", ["显式视差维 D", "3D 卷积正则化"], "#f8ece8", ACCENT),
]
for i, (title, lines, bg, edge) in enumerate(cols):
    x = 0.3 + i * 3.25
    ax.add_patch(Rectangle((x, 1.2), 2.9, 3.35, facecolor=bg, edgecolor=edge, lw=2.0))
    ax.text(x + 1.45, 4.05, title, ha="center", va="center", fontsize=16, color=INK, weight="bold")
    for j, line in enumerate(lines):
        ax.text(x + 1.45, 3.3 - j * 0.7, line, ha="center", va="center", fontsize=14.5, color=INK)
ax.text(0.2, 0.55, "教材第 12.6 节，原书第 778–780 页；程序生成，未用原书原图", fontsize=13.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e114_taxonomy.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 4D 体积与 3D CNN
fig, ax = plt.subplots(figsize=(10.0, 5.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
ax.text(0.2, 5.0, "PSM-Net：4D 成本体积 + 3D 卷积", fontsize=18, color=INK, weight="bold")
def box(x, y, w, h, text, fc="#eef2f7", ec=BLUE, fs=14):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)
box(0.2, 3.5, 1.35, 0.85, "左图")
box(0.2, 2.35, 1.35, 0.85, "右图")
box(1.85, 3.5, 1.75, 0.85, "CNN+SPP", fs=13.5)
box(1.85, 2.35, 1.75, 0.85, "CNN+SPP", fs=13.5)
box(3.95, 2.2, 2.3, 2.15, "4D 体积\nD×H/4×W/4×64", "#f8ece8", ACCENT, 14.5)
box(6.55, 2.2, 2.05, 2.15, "3D 卷积\n3×3×3\n堆叠沙漏", "#eef7ee", GREEN, 14)
box(8.9, 2.2, 0.95, 2.15, "视差\n回归", fs=14)
for a, b in [((1.55, 3.92), (1.85, 3.92)), ((1.55, 2.77), (1.85, 2.77)),
             ((3.6, 3.92), (3.95, 3.7)), ((3.6, 2.77), (3.95, 3.0)),
             ((6.25, 3.27), (6.55, 3.27)), ((8.6, 3.27), (8.9, 3.27))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="->", color=ACCENT, lw=2, mutation_scale=15))
ax.text(3.95, 1.75, "每个视差 d：左特征 ‖ 右移 d 的右特征", fontsize=12.5, color=GRAY)
ax.text(0.2, 0.75, "裁剪 256×512、D=192、64 通道：1/4 分辨率 384 MiB；全分辨率 6 GiB。", fontsize=13.5, color=INK)
ax.text(0.2, 0.3, "论文第 3.3–3.5 节；程序生成，未用原书原图或论文原图", fontsize=12.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e114_volume.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 实验结果
fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5))
panels = [(axes[0, 0], left, "左图（框内为弱纹理块）", None),
          (axes[0, 1], gt, "真值视差", "viridis"),
          (axes[1, 0], raw, "逐像素 argmin：弱纹理处成片出错", "viridis"),
          (axes[1, 1], p3, "固定 3D 聚合后：弱纹理被拉回", "viridis")]
for ax, im, title, cmap in panels:
    if cmap is None:
        ax.imshow(im); ax.add_patch(Rectangle((129, 94), 47, 47, fill=False, edgecolor=ACCENT, lw=2.5))
    else:
        o = ax.imshow(im, cmap=cmap, vmin=8, vmax=44); cb = fig.colorbar(o, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=20, color=INK); ax.axis("off")
fig.suptitle(f"同一张合成双目：3D 聚合把弱纹理块的误差从 {raw_weak:.2f} 降到 {p3_weak:.2f} 像素", fontsize=22, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIG / "e114_lab.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

assertions = [
    ("4D 成本体积形状", V.shape == (H, W, D, 6), f"{V.shape}"),
    ("有效体素占比超过 0.85", valid_frac > 0.85, f"{valid_frac:.3f}"),
    ("PSM-Net 裁剪体积 384 MiB", abs(psm_mib - 384.0) < 1e-9, f"{psm_mib:.0f}"),
    ("朴素全分辨率体积 6 GiB、16 倍", abs(naive_gib - 6.0) < 1e-9 and abs(size_ratio - 16.0) < 1e-9, f"{naive_gib:.0f} / {size_ratio:.0f}"),
    ("批量 12 张体积 4.5 GiB", abs(batch_gib - 4.5) < 1e-9, f"{batch_gib:.1f}"),
    ("3D 卷积参数量约为 2D 的三倍", param_ratio > 2.9, f"{param_ratio:.2f} 倍"),
    ("3D 聚合降低总体误差", p3_mae < raw_mae, f"{raw_mae:.2f} → {p3_mae:.2f}"),
    ("3D 聚合降低弱纹理误差", p3_weak < raw_weak, f"{raw_weak:.2f} → {p3_weak:.2f}"),
    ("3D 聚合降低九成位数误差", p3_p90 < raw_p90, f"{raw_p90:.2f} → {p3_p90:.2f}"),
    ("未训练全局 softmax 回归更差", sg_mae > p3_mae, f"{sg_mae:.2f} vs {p3_mae:.2f}"),
]
n_pass = sum(bool(ok) for _, ok, _ in assertions)
readings = {
    "n_pass": n_pass, "n_total": len(assertions),
    "H": H, "W": W, "D": D, "C": 6,
    "valid_frac": valid_frac, "valid_frac_3": round(valid_frac, 3),
    "psm_bytes": psm_bytes, "psm_mib": psm_mib,
    "naive_bytes": naive_bytes, "naive_gib": naive_gib, "size_ratio": size_ratio,
    "batch_bytes": batch_bytes, "batch_gib": batch_gib,
    "p3d": p3d, "p2d": p2d, "param_ratio": param_ratio,
    "raw_mae": raw_mae, "raw_mae_2": round(raw_mae, 2), "raw_acc": raw_acc, "raw_p90": raw_p90, "raw_p90_2": round(raw_p90, 2),
    "p2_mae": p2_mae, "p2_mae_2": round(p2_mae, 2), "p2_acc": p2_acc, "p2_p90": p2_p90,
    "p3_mae": p3_mae, "p3_mae_2": round(p3_mae, 2), "p3_acc": p3_acc, "p3_p90": p3_p90, "p3_p90_2": round(p3_p90, 2),
    "raw_weak": raw_weak, "raw_weak_2": round(raw_weak, 2),
    "p2_weak": p2_weak, "p2_weak_2": round(p2_weak, 2),
    "p3_weak": p3_weak, "p3_weak_2": round(p3_weak, 2),
    "raw_bound": raw_bound, "raw_bound_2": round(raw_bound, 2),
    "p2_bound": p2_bound, "p2_bound_2": round(p2_bound, 2),
    "p3_bound": p3_bound, "p3_bound_2": round(p3_bound, 2),
    "soft_global_mae": sg_mae, "soft_global_mae_2": round(sg_mae, 2), "soft_global_acc": sg_acc,
    "soft_local_mae": sl_mae, "soft_local_mae_2": round(sl_mae, 2), "soft_local_acc": sl_acc,
    "n_weak": int(weak_m.sum()), "n_bound": int(bound_m.sum()), "n_valid": int(valid.sum()),
}
print(f"[lab] E114 PSM-Net 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] {W}x{H}｜D {D}｜通道 6｜弱纹理 {int(weak_m.sum())} 像素｜有效 {int(valid.sum())} 像素")
for name, ok, detail in assertions:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] 原始 argmin {raw_mae:.2f}｜2D 聚合 {p2_mae:.2f}｜3D 聚合 {p3_mae:.2f}｜弱纹理 {raw_weak:.2f}→{p3_weak:.2f}｜边界 {raw_bound:.2f}→{p3_bound:.2f}")
print(f"[lab] 2D 弱纹理 {p2_weak:.2f} vs 3D 弱纹理 {p3_weak:.2f}（固定核的诚实对照）")
print(f"[lab] 全局 softmax {sg_mae:.2f}｜局部 softmax {sl_mae:.2f}｜整数 argmin {p3_mae:.2f}")
print(f"[lab] PSM-Net 裁剪体积 {psm_mib:.0f} MiB｜朴素全分辨率 {naive_gib:.0f} GiB（{size_ratio:.0f} 倍）｜批量 12 张 {batch_gib:.1f} GiB")
print(f"[lab] 3D 参数 {p3d}｜2D 参数 {p2d}｜比值 {param_ratio:.2f}")
for p in sorted(FIG.glob("e114_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e114_psmnet_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e114_psmnet_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e114_psmnet_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(assertions)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(assertions)-n_pass} / 共 {len(assertions)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(assertions): raise SystemExit(1)
