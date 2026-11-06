#!/usr/bin/env python3
"""E138 第7章补遗 lab：边缘检测（Canny）、轮廓提取（findContours）、颜色聚类分割（k-means）。

边界：合成图、三种经典基线、单机运行；不实现 snakes/level sets/normalized cuts，
也不声称这是通用分割 benchmark。k-means 只作为 mean-shift 同族“找模式”的可运行替身。
"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; GOLD = "#C9971C"; PURPLE = "#7A5A8E"
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
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 15

print(f"[lab] E138 第7章补遗实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] OpenCV {cv2.__version__} / NumPy {np.__version__}")
print("[lab] 边界：合成图、三种经典基线；不实现 snakes/level sets/normalized cuts；不是通用 benchmark")

rng = np.random.default_rng(138)
H, W = 270, 360
yy, xx = np.mgrid[0:H, 0:W]

# 浅灰背景 + 轻微光照梯度 + 噪声；三个颜色分明的形状
base = 0.78 + 0.06 * (xx / W) - 0.03 * (yy / H)
img = np.stack([base, base, base], axis=-1)
img += rng.normal(0, 0.012, img.shape)
rect = (xx >= 60) & (xx < 150) & (yy >= 60) & (yy < 150)
circ = ((xx - 240) ** 2 + (yy - 90) ** 2) <= 42 ** 2
tri = (yy >= 155) & (yy <= 230) & (np.abs(xx - 240) <= (yy - 155) * 0.7)
img[rect] = [0.80, 0.20, 0.18]
img[circ] = [0.18, 0.62, 0.28]
img[tri] = [0.20, 0.35, 0.78]
img = np.clip(img, 0, 1)
rgb = (img * 255).astype(np.uint8)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.2)

gt = np.zeros((H, W), np.uint8)
for m in (rect, circ, tri):
    gt[m] = 255
gt_boundary = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
d_gt = cv2.distanceTransform((~gt_boundary).astype(np.uint8), cv2.DIST_L2, 3)

LO, HI = 50, 150
LO2, HI2 = 180, 350
edges_lo = cv2.Canny(blur, LO, HI)
edges_hi = cv2.Canny(blur, LO2, HI2)

def edge_scores(edge):
    eb = edge > 0
    d_e = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    tp_edge = int((eb & (d_gt <= 2)).sum())
    tp_gt = int((gt_boundary & (d_e <= 2)).sum())
    precision = tp_edge / max(int(eb.sum()), 1)
    recall = tp_gt / max(int(gt_boundary.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return dict(count=int(eb.sum()), precision=float(precision), recall=float(recall), f1=float(f1))

score_lo = edge_scores(edges_lo)
score_hi = edge_scores(edges_hi)

# 轮廓：把低阈值边缘闭合后取外轮廓，按面积过滤
closed = cv2.morphologyEx(edges_lo, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = [c for c in contours if cv2.contourArea(c) > 500]
contours = sorted(contours, key=cv2.contourArea, reverse=True)
gt_areas = {"矩形": int(rect.sum()), "圆": int(circ.sum()), "三角": int(tri.sum())}
contour_areas = [float(cv2.contourArea(c)) for c in contours]
circularities = [float(4 * np.pi * cv2.contourArea(c) / max(cv2.arcLength(c, True) ** 2, 1e-9)) for c in contours]
# 面积最接近的 GT 形状配对（只为读数与图注，不参与分割真值）
pairs = []
for c, a, cir in zip(contours, contour_areas, circularities):
    name = min(gt_areas, key=lambda k: abs(gt_areas[k] - a))
    pairs.append(dict(name=name, area=a, gt_area=gt_areas[name], circularity=cir,
                      rel_err=abs(a - gt_areas[name]) / gt_areas[name]))

# 分割：Lab 颜色 k-means（K=4：三个形状 + 背景）；作为 mean-shift 同族基线
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
feat = lab.reshape(-1, 3)
K = 4
cv2.setRNGSeed(138)
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)
compactness, labels, centers = cv2.kmeans(feat, K, None, crit, 8, cv2.KMEANS_PP_CENTERS)
label_img = labels.reshape(H, W)
gt_label = np.zeros((H, W), np.int32)
gt_label[rect] = 1; gt_label[circ] = 2; gt_label[tri] = 3
mapping = {int(k): int(np.bincount(gt_label[label_img == k]).argmax()) for k in range(K) if (label_img == k).any()}
pred = np.zeros_like(gt_label)
for k, v in mapping.items():
    pred[label_img == k] = v
acc = float((pred == gt_label).mean())
class_names = {1: "矩形", 2: "圆", 3: "三角"}
ious = {}
for t in (1, 2, 3):
    p = pred == t; g = gt_label == t
    ious[class_names[t]] = float((p & g).sum() / max((p | g).sum(), 1))
mean_iou = float(np.mean(list(ious.values())))
fg_frac = float((gt > 0).mean())

checks = []
def check(name, ok, detail):
    checks.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")

check("合成图前景占比在 0.15–0.60", 0.15 <= fg_frac <= 0.60, f"{fg_frac:.3f}")
check("低阈值边缘数多于高阈值", score_lo["count"] > score_hi["count"], f"{score_lo['count']} vs {score_hi['count']}")
check("低阈值边缘 F1 ≥0.90", score_lo["f1"] >= 0.90, f"{score_lo['f1']:.3f}")
check("高阈值 F1 低于低阈值", score_hi["f1"] < score_lo["f1"], f"{score_hi['f1']:.3f} < {score_lo['f1']:.3f}")
check("高阈值召回低于低阈值", score_hi["recall"] < score_lo["recall"], f"{score_hi['recall']:.3f} < {score_lo['recall']:.3f}")
check("闭合边缘提取到 3 个轮廓", len(contours) == 3, f"{len(contours)} 个")
check("三个轮廓面积误差都 <10%", all(p["rel_err"] < 0.10 for p in pairs), "；".join(f"{p['name']} {p['rel_err']:.3f}" for p in pairs))
check("圆的圆度 ≥0.85", max(p["circularity"] for p in pairs if p["name"] == "圆") >= 0.85, f"{max(p['circularity'] for p in pairs if p['name']=='圆'):.3f}")
check("圆度：圆 > 矩形", max(p["circularity"] for p in pairs if p["name"] == "圆") > max(p["circularity"] for p in pairs if p["name"] == "矩形"), "圆 vs 矩形")
check("k-means 像素准确率 ≥0.95", acc >= 0.95, f"{acc:.4f}")
check("k-means 平均 IoU ≥0.90", mean_iou >= 0.90, f"{mean_iou:.4f}")
check("三个类别 IoU 都 ≥0.90", all(v >= 0.90 for v in ious.values()), "；".join(f"{k} {v:.3f}" for k, v in ious.items()))

n_pass = int(sum(checks))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：输入 / Sobel / Canny 低阈值 / Canny 高阈值
gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
mag = np.hypot(gx, gy)
mag = (mag / max(mag.max(), 1e-9) * 255).astype(np.uint8)
fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.1))
axes = axes.ravel()
axes[0].imshow(rgb); axes[0].set_title("输入：三个形状", fontsize=16)
axes[1].imshow(mag, cmap="magma"); axes[1].set_title("梯度幅值 |∇I|", fontsize=16)
axes[2].imshow(edges_lo, cmap="gray"); axes[2].set_title(f"Canny 低阈值 {LO}/{HI}：{score_lo['count']} 像素", fontsize=15)
axes[3].imshow(edges_hi, cmap="gray"); axes[3].set_title(f"Canny 高阈值 {LO2}/{HI2}：{score_hi['count']} 像素", fontsize=15)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"边缘检测：低阈值 F1={score_lo['f1']:.2f}，高阈值 F1={score_hi['f1']:.2f}", fontsize=17, y=0.98)
fig.tight_layout(rect=(0, 0.035, 1, 0.95))
fig.savefig(FIG / "e138_edges.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：轮廓叠加 + 面积对比 + 圆度
fig = plt.figure(figsize=(12.0, 4.6))
ax0 = fig.add_subplot(1, 3, 1)
over = rgb.copy()
cv2.drawContours(over, contours, -1, (194, 78, 42), 3)
ax0.imshow(over); ax0.set_title("闭合边缘 → 3 个外轮廓", fontsize=15)
ax0.set_xticks([]); ax0.set_yticks([])
ax1 = fig.add_subplot(1, 3, 2)
names = [p["name"] for p in pairs]
xpos = np.arange(len(pairs)); w = 0.36
ax1.bar(xpos - w / 2, [p["gt_area"] for p in pairs], w, color=GRID, label="真值面积")
ax1.bar(xpos + w / 2, [p["area"] for p in pairs], w, color=BLUE, label="轮廓面积")
ax1.set_xticks(xpos); ax1.set_xticklabels(names, fontsize=15)
ax1.set_ylabel("像素数", fontsize=15); ax1.set_title("轮廓面积与真值接近", fontsize=15)
ax1.legend(fontsize=15, frameon=False); ax1.grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
ax2 = fig.add_subplot(1, 3, 3)
colors = [GOLD if p["name"] == "圆" else PURPLE for p in pairs]
ax2.bar(names, [p["circularity"] for p in pairs], color=colors)
ax2.set_ylim(0, 1.05); ax2.set_ylabel("4πA / P²", fontsize=15)
ax2.set_title("圆度：圆最接近 1", fontsize=15)
ax2.grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
fig.suptitle("轮廓提取：链接、闭合、再用面积/圆度筛选", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))
fig.savefig(FIG / "e138_contours.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：输入 / 真值 / k-means / 每类 IoU
fig, axes = plt.subplots(1, 4, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
axes[0].imshow(rgb); axes[0].set_title("输入", fontsize=15)
axes[1].imshow(gt_label, cmap="tab20b", vmin=0, vmax=3); axes[1].set_title("真值标签", fontsize=15)
axes[2].imshow(pred, cmap="tab20b", vmin=0, vmax=3); axes[2].set_title(f"k-means（K={K}）", fontsize=15)
for ax in axes[:3]:
    ax.set_xticks([]); ax.set_yticks([])
axes[3].bar(list(ious), list(ious.values()), color=[GREEN, GOLD, BLUE])
axes[3].set_ylim(0, 1.05); axes[3].set_title("每类 IoU", fontsize=15)
axes[3].tick_params(axis="x", labelsize=14)
axes[3].grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
fig.suptitle(f"颜色聚类分割：像素准确率 {acc*100:.1f}%，平均 IoU {mean_iou:.2f}", fontsize=17, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "e138_segmentation.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

for _name in ("e138_edges.png", "e138_contours.png", "e138_segmentation.png"):
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] 边缘：低阈值 {LO}/{HI} → {score_lo['count']} 像素，P={score_lo['precision']:.3f} R={score_lo['recall']:.3f} F1={score_lo['f1']:.3f}")
print(f"[lab] 边缘：高阈值 {LO2}/{HI2} → {score_hi['count']} 像素，P={score_hi['precision']:.3f} R={score_hi['recall']:.3f} F1={score_hi['f1']:.3f}")
print(f"[lab] 轮廓：{len(contours)} 个；面积 " + "；".join(f"{p['name']} {p['area']:.0f}/真值 {p['gt_area']}" for p in pairs))
print(f"[lab] 圆度：" + "；".join(f"{p['name']} {p['circularity']:.3f}" for p in pairs))
print(f"[lab] 分割：准确率 {acc:.4f}；IoU " + "；".join(f"{k} {v:.4f}" for k, v in ious.items()) + f"；平均 {mean_iou:.4f}")

readings = dict(
    n_pass=n_pass, n_total=len(checks),
    image=dict(height=H, width=W, foreground_fraction=fg_frac,
               gt_areas=gt_areas, gt_boundary_pixels=int(gt_boundary.sum())),
    edges=dict(low=dict(thresholds=[LO, HI], **score_lo),
               high=dict(thresholds=[LO2, HI2], **score_hi)),
    contours=dict(count=len(contours), pairs=pairs),
    segmentation=dict(k=K, method="k-means on Lab color", accuracy=acc, iou=ious,
                      mean_iou=mean_iou, mapping=mapping),
    protocol=dict(synthetic=True, machine="local", general_benchmark=False,
                  implements="Canny + findContours + k-means",
                  not_implemented=["snakes", "level sets", "normalized cuts"]),
    versions=dict(opencv=cv2.__version__, numpy=np.__version__),
)
out = ROOT / "outputs" / "e138_edges_contours_segmentation_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
