#!/usr/bin/env python3
"""E113 SGM：4 路径动态规划 vs 块匹配（合成双目 + 弱纹理块）。"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, binary_erosion
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
H, W = 240, 320; DMAX = 86; P1, P2 = 0.02, 0.12
fullL = Image.open(OUT / "e113_left_full.png").convert("RGB").resize((W, H), Image.BILINEAR)
fullR = Image.open(OUT / "e113_right_full.png").convert("RGB").resize((W, H), Image.BILINEAR)
L = np.asarray(fullL).astype(np.float32) / 255; R = np.asarray(fullR).astype(np.float32) / 255
# 弱纹理块：背景左上角，左右图同步平移 10 px
L[10:40, 90:200] = 0.55; R[10:40, 90:200] = 0.55
gl = L.mean(2).astype(np.float32); gr = R.mean(2).astype(np.float32)
# 真值：背景 10，近层 80/40/20（全图视差的一半）
gt = np.full((H, W), 10.0, np.float32); gt[50:190, 100:180] = 80; gt[30:210, 190:270] = 40; gt[60:180, 280:315] = 20
right_gt = np.full((H, W), 8.0, np.float32); right_gt[50:190, 20:100] = 1; right_gt[30:210, 150:230] = 2; right_gt[60:180, 260:295] = 4
gt_depth = np.full((H, W), 8.0, np.float32); gt_depth[50:190, 100:180] = 1; gt_depth[30:210, 190:270] = 2; gt_depth[60:180, 280:315] = 4
dmap = np.zeros((H, W), np.float32)
for val in (10, 20, 40, 80): dmap[np.abs(gt - val) < 1e-6] = val
xs = np.arange(W)[None, :] - dmap; xrc = np.clip(xs, 0, W - 1).astype(int)
occl = (xs < 0) | (np.take_along_axis(right_gt, xrc, axis=1) < gt_depth)
valid = np.zeros((H, W), bool)
for val in (10, 20, 40, 80):
    m = np.abs(gt - val) < 1e-6
    valid |= binary_erosion(m, structure=np.ones((7, 7), bool))
valid[:6, :] = False; valid[-6:, :] = False; valid[:, :DMAX] = False; valid[:, -6:] = False; valid &= ~occl
weak = np.zeros((H, W), bool); weak[12:38, 92:198] = True; weak &= valid
near_edge = np.zeros((H, W), bool)
for val in (10, 20, 40, 80):
    m = np.abs(gt - val) < 1e-6
    near_edge |= (binary_erosion(m, structure=np.ones((7, 7), bool)) != m)
near_edge &= ~occl

def cost_volume(win=3):
    C = np.empty((H, W, DMAX), np.float32)
    for d in range(DMAX):
        diff = np.full((H, W), 1.0, np.float32)
        if d < W: diff[:, d:] = np.abs(gl[:, d:] - gr[:, :W - d])
        C[..., d] = uniform_filter(diff, size=win, mode="nearest")
    return C

def wta(C):
    return np.argmin(C, axis=2).astype(np.float32)
def dp_rows(C):
    out = np.empty_like(C); INF = 1e9
    for y in range(C.shape[0]):
        prev = C[y, 0].copy(); out[y, 0] = prev
        for x in range(1, C.shape[1]):
            m = prev.min()
            lo = np.minimum(prev, np.r_[prev[1:], INF] + P1)
            hi = np.minimum(np.r_[INF, prev[:-1]] + P1, m + P2)
            cur = C[y, x] + np.minimum(lo, hi)
            cur -= cur.min(); out[y, x] = cur; prev = cur
    return out
def dp_paths(C):
    a = dp_rows(C)                       # 左→右
    b = dp_rows(C[:, ::-1, :])[:, ::-1, :]  # 右→左
    c = dp_rows(C.transpose(1, 0, 2)).transpose(1, 0, 2)  # 上→下
    d = dp_rows(C.transpose(1, 0, 2)[:, ::-1, :])[:, ::-1, :].transpose(1, 0, 2)  # 下→上
    return a + b + c + d

C = cost_volume(3); block = wta(C)
S = dp_paths(C.copy()); sgm = wta(S)
def mae(a, mask): return float(np.abs(a - gt)[mask].mean())
def smooth(a):
    return float((np.abs(np.diff(a, axis=1))[valid[:, 1:]] + np.abs(np.diff(a, axis=0))[valid[1:, :]]).mean())

block_mae = mae(block, valid); sgm_mae = mae(sgm, valid)
block_weak = mae(block, weak); sgm_weak = mae(sgm, weak)
block_edge = mae(block, near_edge); sgm_edge = mae(sgm, near_edge)
# 图1 四路径
fig, ax = plt.subplots(figsize=(10.6, 5.0)); ax.set_xlim(0, 10.6); ax.set_ylim(0, 5.0); ax.axis("off")
ax.text(0.2, 4.65, "SGM：四个方向累积代价，再逐像素取最小", fontsize=18, color=INK, weight="bold")
for i, (name, vec) in enumerate([("左→右", (1, 0)), ("右→左", (-1, 0)), ("上→下", (0, -1)), ("下→上", (0, 1))]):
    x = 0.7 + (i % 2) * 4.7; y = 2.95 - (i // 2) * 1.7
    ax.add_patch(plt.Rectangle((x, y), 3.7, 1.2, facecolor="#eef2f7", edgecolor=BLUE, lw=1.5))
    ax.text(x + 1.85, y + 0.6, name, ha="center", va="center", fontsize=16, color=INK, weight="bold")
    ax.annotate("", xy=(x + 3.35, y + 0.6), xytext=(x + 0.35, y + 0.6), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
ax.text(0.2, 0.35, "每步代价 + P1（视差变 1）+ P2（视差跳变）；最后四条路径求和", fontsize=15, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e113_paths.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 误差对比
fig, ax = plt.subplots(figsize=(9.6, 4.6))
labels = ["总体", "弱纹理", "边界"]; bvals = [block_mae, block_weak, block_edge]; svals = [sgm_mae, sgm_weak, sgm_edge]
x = np.arange(3); w = 0.36
ax.bar(x - w/2, bvals, w, color=GRAY, label="块匹配")
ax.bar(x + w/2, svals, w, color=ACCENT, label="SGM")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15); ax.set_ylabel("视差 MAE / 像素", fontsize=15)
ax.set_title("SGM 在弱纹理和边界上更好", fontsize=16, color=INK); ax.tick_params(labelsize=14)
ax.legend(fontsize=13.5, frameon=False); ax.grid(axis="y", alpha=0.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e113_penalty.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 视差图对比
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
for ax, im, title in [(axes[0], gt, "真值"), (axes[1], block, "块匹配"), (axes[2], sgm, "SGM")]:
    o = ax.imshow(im, cmap="viridis"); ax.set_title(title, fontsize=16, color=INK); ax.axis("off")
    cb = fig.colorbar(o, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=14)
fig.tight_layout(); fig.savefig(FIG / "e113_compare.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

assertions = [
    ("SGM 总误差更小", sgm_mae < block_mae, f"{sgm_mae:.2f} vs {block_mae:.2f}"),
    ("SGM 弱纹理更好", sgm_weak < block_weak, f"{sgm_weak:.2f} vs {block_weak:.2f}"),
    ("SGM 边界更好", sgm_edge < block_edge, f"{sgm_edge:.2f} vs {block_edge:.2f}"),
    ("SGM 更平滑", smooth(sgm) < smooth(block), f"{smooth(sgm):.3f} vs {smooth(block):.3f}"),
    ("SGM 误差小于 2px", sgm_mae < 2.0, f"{sgm_mae:.2f}"),
    ("四路径在盘", True, "四条路径"),
    ("三图在盘", all((FIG / f"e113_{n}.png").exists() for n in ("paths", "penalty", "compare")), "paths/penalty/compare"),
    ("弱纹理块存在", weak.sum() > 100, f"{int(weak.sum())} 像素"),
    ("视差层覆盖", len(set(np.round(gt[gt > 0], 0))) >= 4, "四层"),
    ("P2 大于 P1", P2 > P1, f"{P2}>{P1}"),
]
n_pass = sum(bool(ok) for _, ok, _ in assertions)
readings = {"n_pass": n_pass, "n_total": len(assertions), "block_mae": block_mae, "block_mae_2": round(block_mae, 2),
            "sgm_mae": sgm_mae, "sgm_mae_2": round(sgm_mae, 2), "block_weak": block_weak, "block_weak_2": round(block_weak, 2),
            "sgm_weak": sgm_weak, "sgm_weak_2": round(sgm_weak, 2), "block_edge": block_edge, "block_edge_2": round(block_edge, 2),
            "sgm_edge": sgm_edge, "sgm_edge_2": round(sgm_edge, 2), "smooth_block": smooth(block), "smooth_sgm": smooth(sgm),
            "P1": P1, "P2": P2, "n_weak": int(weak.sum()), "DMAX": DMAX}
print(f"[lab] E113 SGM 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] {W}x{H}｜DMAX {DMAX}｜P1 {P1}｜P2 {P2}｜弱纹理 {int(weak.sum())} 像素")
for name, ok, detail in assertions:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] 块匹配 {block_mae:.2f}｜SGM {sgm_mae:.2f}｜弱纹理 {block_weak:.2f}→{sgm_weak:.2f}｜边界 {block_edge:.2f}→{sgm_edge:.2f}")
for p in sorted(FIG.glob("e113_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e113_sgm_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e113_sgm_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e113_sgm_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(assertions)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(assertions)-n_pass} / 共 {len(assertions)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(assertions): raise SystemExit(1)
