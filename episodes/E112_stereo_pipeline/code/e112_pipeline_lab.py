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
Image.fromarray(left_u8).save(OUT / "e112_left.png")
Image.fromarray(right_u8).save(OUT / "e112_right.png")

# 块匹配 SAD：窗口 9×9，搜索 0–170

# ── E112：四步流水线（代价→聚合→WTA→亚像素/交叉检查/补洞） ──
gl = left.mean(axis=2).astype(np.float32); gr = right.mean(axis=2).astype(np.float32)
WIN_LIST = [3, 5, 9, 15, 21]; DMAX = 170
def cost_map(ref, tgt, darr, win, metric="SAD", direction="left"):
    darr = np.asarray(darr, dtype=np.float32)
    xs = np.arange(W)[None, :] - darr if direction == "left" else np.arange(W)[None, :] + darr
    valid = (xs >= 0) & (xs < W)
    tgt_s = np.take_along_axis(tgt, np.clip(xs, 0, W - 1).astype(int), axis=1)
    diff = ref - tgt_s
    diff = np.abs(diff) if metric == "SAD" else diff * diff
    diff = np.where(valid, diff, 1.0)
    return uniform_filter(diff, size=win, mode="nearest"), valid
def wta(ref, tgt, win, metric="SAD", direction="left"):
    best = np.full((H, W), np.inf, np.float32); est = np.zeros((H, W), np.float32)
    for d in range(DMAX + 1):
        c, vd = cost_map(ref, tgt, d, win, metric, direction)
        upd = (c < best) & vd
        best[upd] = c[upd]; est[upd] = d
    return est, best
est_sad, best_sad = wta(gl, gr, 9, "SAD", "left")
est_ssd, _ = wta(gl, gr, 9, "SSD", "left")

# 亚像素：抛物线拟合 c(d-1),c(d),c(d+1)
c0, _ = cost_map(gl, gr, est_sad, 9, "SAD", "left")
cm, _ = cost_map(gl, gr, np.maximum(est_sad - 1, 0), 9, "SAD", "left")
cp, _ = cost_map(gl, gr, np.minimum(est_sad + 1, DMAX), 9, "SAD", "left")
den = cm - 2 * c0 + cp
delta = np.where(np.abs(den) > 1e-8, 0.5 * (cm - cp) / np.maximum(den, 1e-8), 0.0)
delta = np.clip(delta, -0.5, 0.5); est_sub = est_sad + delta

# 左右一致性：右图匹配左图
est_R, _ = wta(gr, gl, 9, "SAD", "right")
xR = np.arange(W)[None, :] - est_sad
dR = np.take_along_axis(est_R, np.clip(xR, 0, W - 1).astype(int), axis=1)
bad = (xR < 0) | (np.abs(est_sad - dR) > 1.0)
# 补洞：每行用最近有效值线性插值
est_fill = est_sad.copy()
for y in range(H):
    good = np.where(~bad[y])[0]
    if len(good) > 1:
        est_fill[y] = np.interp(np.arange(W), good, est_sad[y, good])

# 有效区（GT 恒定 + 非遮挡 + 边界内缩）
right_gt = np.full((H, W), DEPTHS[3], np.float32)
gt_depth = np.full((H, W), DEPTHS[3], np.float32)
for x0, x1, y0, y1, idx, _ in sorted(layers, key=lambda z: -z[4]):
    d = int(round(DISP[idx])); right_gt[y0:y1, x0 - d:x1 - d] = DEPTHS[idx]; gt_depth[y0:y1, x0:x1] = DEPTHS[idx]
dmap = np.zeros((H, W), np.float32)
for idx in range(4): dmap[np.abs(gt - DISP[idx]) < 1e-6] = DISP[idx]
xs2 = np.arange(W)[None, :] - dmap; xrc = np.clip(xs2, 0, W - 1).astype(int)
occl = (xs2 < 0) | (np.take_along_axis(right_gt, xrc, axis=1) < gt_depth)
valid = np.zeros((H, W), bool)
for idx in range(4):
    m = np.abs(gt - DISP[idx]) < 1e-6
    valid |= binary_erosion(m, structure=np.ones((9, 9), bool))
valid[:8, :] = False; valid[-8:, :] = False; valid[:, :DMAX] = False; valid[:, -8:] = False
valid &= ~occl
near_edge = np.zeros((H, W), bool)
for idx in range(4):
    m = np.abs(gt - DISP[idx]) < 1e-6
    near_edge |= (binary_erosion(m, structure=np.ones((9, 9), bool)) != m)
near_edge &= ~occl

def mae(a): return float(np.abs(a - gt)[valid].mean())
win_rows = []
for win in WIN_LIST:
    e, _ = wta(gl, gr, win, "SAD", "left")
    edge = float(np.abs(e - gt)[near_edge].mean())
    win_rows.append((win, mae(e), edge))
sad_mae = mae(est_sad); ssd_mae = mae(est_ssd)
sub_mae = mae(est_sub); int_mae = sad_mae
occl_bad = float(bad[occl].mean() * 100); clear_bad = float(bad[~occl & valid].mean() * 100)
invalid_before = float(bad.mean() * 100)
fill_cover = float((bad & (est_fill != est_sad)).sum() / max(bad.sum(), 1) * 100)
invalid_after = float(((bad) & (~np.isfinite(est_fill))).mean() * 100)
acc1 = float((np.abs(est_sad - gt)[valid] <= 1).mean() * 100)
# 采样点的 DSI 最小位置
sample = [(280, 240), (460, 240), (595, 240), (100, 30)]
dsi_ok = all(abs(est_sad[y, x] - gt[y, x]) <= 1 for x, y in sample)

# 图1 流水线
fig, ax = plt.subplots(figsize=(12.6, 4.2)); ax.set_xlim(0, 12.6); ax.set_ylim(0, 4.2); ax.axis("off")
ax.text(0.2, 3.75, "稠密立体匹配四步：代价 → 聚合 → 视差 → 优化", fontsize=18, color=INK, weight="bold")
steps = [("① 代价", "C0(x,y,d)\nSAD / SSD", BLUE), ("② 聚合", "C = w * C0\n窗口/自适应", GREEN),
         ("③ 视差", "d* = argmin C\nWTA", ACCENT), ("④ 优化", "亚像素 δ\n交叉检查/补洞", GRAY)]
for i, (t, sub, col) in enumerate(steps):
    x = 0.3 + i * 3.1
    ax.add_patch(plt.Rectangle((x, 1.3), 2.7, 1.7, facecolor="#eef2f7", edgecolor=col, lw=1.8))
    ax.text(x + 1.35, 2.55, t, ha="center", fontsize=17, color=INK, weight="bold")
    ax.text(x + 1.35, 1.9, sub, ha="center", fontsize=15, color=INK)
    if i < 3:
        ax.annotate("", xy=(x + 3.05, 2.15), xytext=(x + 2.75, 2.15), arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))
fig.tight_layout(); fig.savefig(FIG / "e112_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 窗口大小取舍
fig, ax = plt.subplots(figsize=(9.8, 4.8))
ws = [r[0] for r in win_rows]; ms = [r[1] for r in win_rows]; es = [r[2] for r in win_rows]
ax.plot(ws, ms, "-o", color=BLUE, lw=2, label="内部 MAE")
ax.plot(ws, es, "-s", color=ACCENT, lw=2, label="边界 MAE")
ax.set_xlabel("聚合窗口边长 / 像素", fontsize=15); ax.set_ylabel("视差 MAE / 像素", fontsize=15)
ax.set_title("窗口越大内部越稳，边界越糊", fontsize=16, color=INK); ax.tick_params(labelsize=14)
ax.grid(alpha=0.2); ax.legend(fontsize=13.5, frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e112_window.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 整数/亚像素/交叉检查/补洞
fig, axes = plt.subplots(2, 2, figsize=(12.6, 6.2))
for ax, im, title in [(axes[0,0], est_sad, "整数 WTA"), (axes[0,1], est_sub, "亚像素"),
                      (axes[1,0], bad.astype(float), "交叉检查无效"), (axes[1,1], est_fill, "补洞后")]:
    imobj = ax.imshow(im, cmap="viridis"); ax.set_title(title, fontsize=16, color=INK); ax.axis("off")
    cb = fig.colorbar(imobj, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=15)
fig.tight_layout(); fig.savefig(FIG / "e112_refine.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks = [
    ("SAD 不差于 SSD", sad_mae <= ssd_mae + 1e-9, f"{sad_mae:.3f} vs {ssd_mae:.3f}"),
    ("小窗口内部误差小", win_rows[0][1] < 0.15, f"{win_rows[0][1]:.3f}"),
    ("大窗口边界更差", win_rows[2][2] > win_rows[0][2], f"{win_rows[2][2]:.3f} vs {win_rows[0][2]:.3f}"),
    ("整数真值下亚像素不劣化", sub_mae <= int_mae + 0.05, f"{sub_mae:.3f} vs {int_mae:.3f}"),
    ("交叉检查抓遮挡", occl_bad > clear_bad, f"{occl_bad:.1f}% vs {clear_bad:.1f}%"),
    ("补洞覆盖无效像素", fill_cover > 95, f"{fill_cover:.1f}%"),
    ("WTA 1px 内准确率高", acc1 > 90, f"{acc1:.1f}%"),
    ("采样点 DSI 最小正确", dsi_ok, "四点全对"),
    ("四层全覆盖", len(set(np.round(gt[gt > 0], 1))) >= 4, "四层"),
    ("三图在盘", all((FIG / f"e112_{n}.png").exists() for n in ("pipeline", "window", "refine")), "pipeline/window/refine"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "sad_mae": sad_mae, "sad_mae_3": round(sad_mae, 3),
    "ssd_mae": ssd_mae, "ssd_mae_3": round(ssd_mae, 3), "sub_mae": sub_mae, "sub_mae_4": round(sub_mae, 4),
    "int_mae": int_mae, "int_mae_3": round(int_mae, 3), "acc1": acc1, "acc1_1": round(acc1, 1),
    "occl_bad": occl_bad, "occl_bad_1": round(occl_bad, 1), "clear_bad": clear_bad, "clear_bad_1": round(clear_bad, 1),
    "invalid_before": invalid_before, "invalid_before_1": round(invalid_before, 1),
    "invalid_after": invalid_after, "invalid_after_1": round(invalid_after, 1), "fill_cover": fill_cover, "fill_cover_1": round(fill_cover, 1),
    "win3_mae": win_rows[0][1], "win9_mae": win_rows[2][1], "win21_mae": win_rows[4][1],
    "win3_edge": win_rows[0][2], "win9_edge": win_rows[2][2], "win21_edge": win_rows[4][2], "win_rows": [[w, round(m, 3), round(e, 3)] for w, m, e in win_rows],
}
print(f"[lab] E112 立体匹配流水线开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] 图像 {W}x{H}｜深度 {DEPTHS}｜视差 {[round(d,1) for d in DISP]}｜窗口 {WIN_LIST}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] SAD9 {sad_mae:.3f} px｜SSD9 {ssd_mae:.3f} px｜亚像素 {sub_mae:.4f} px｜1px 内 {acc1:.1f}%")
print(f"[lab] 交叉检查遮挡 {occl_bad:.1f}% vs 非遮挡 {clear_bad:.1f}%｜补洞覆盖 {fill_cover:.1f}%（原无效 {invalid_before:.1f}%）")
print(f"[lab] 窗口：3/9/21 内部 {win_rows[0][1]:.3f}/{win_rows[2][1]:.3f}/{win_rows[4][1]:.3f}；9/21 边界 {win_rows[2][2]:.3f}/{win_rows[4][2]:.3f}")
for p in sorted(FIG.glob("e112_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e112_pipeline_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e112_pipeline_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e112_pipeline_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
