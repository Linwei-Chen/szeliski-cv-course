#!/usr/bin/env python3
"""E061 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 6.26–6.31+Table 6.2 均不复制）。

图 1 e061_detection_metrics.png（deck 页 4，f004 配图）：检测度量流水线——
  提议框→IoU 量框（交集÷并集）→NMS 去重→查准/查全→P-R 曲线→AP/mAP；
  左下附 IoU 两框重叠示意。对应原书 p.380–381。
图 2 e061_rcnn_family.png（deck 页 10，f010 配图）：两阶段检测器谱系——
  R-CNN（2000 提议/缩放 224/SVM）→Fast R-CNN（共享卷积/FC 双头）→
  Faster R-CNN（RPN 锚框）→FPN（自顶向下多尺度）→DETR（Transformer/二部匹配）。
  对应原书 p.382–384。
图 3 e061_single_stage.png（deck 页 14，f014 配图）：单阶段检测器对照——
  SSD/YOLO/RetinaNet（焦点损失）/YOLOv4（neck+免费礼包/特殊礼包），
  下方速度-精度示意。对应原书 p.384–385。
断言体系（E057–E060 判例同构+readings 体系同构）：
①图 1 流水线七词在档（提议框/IoU/NMS/查准/查全/P-R 曲线/AP）；
②图 1 IoU 示意词在档（交集/并集）；
③图 2 五节点词在档（R-CNN/Fast R-CNN/Faster R-CNN/FPN/DETR）+RPN；
④图 3 四检测器词在档（SSD/YOLO/RetinaNet/YOLOv4）+焦点损失/neck；
⑤主文本 fontsize≥11.5pt（11.5/12/12.5/14 档）；
⑥「程序生成，未用原书原图」×3 图各一；
⑦图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak 零命中）；
⑧三张 PNG 落盘且计数=3。
readings 体系：[PASS] 断言行+readings.json 叶值存档+源码/readings/PNG sha 自证行。
输出 code/outputs/figures/{e061_detection_metrics,e061_rcnn_family,e061_single_stage}.png
（6.6×4.2 inch @dpi150）。
红线 5：可见层无内部记号——图内不渲染集号/内部编号/节记号；只标概念+原书页。
"""
import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E061_general_object_detection"
FIG_DIR = EP / "code" / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EP / "code" / "outputs"

PAPER = "#faf7f0"
INK = "#2f3540"
MUT = "#5a6472"
ACC = "#8a3324"
BLU = "#3c4f6b"
GOLD = "#b08a3e"
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")


def _box(ax, x, y, w, h, ec=INK, fc="#ffffff", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))


def _arrow(ax, p1, p2, color=GOLD, lw=1.3, ms=11):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def collect_texts(fig):
    out = []
    for ax in fig.axes:
        out.extend(t for t in ax.texts if t.get_text().strip())
        for patch in ax.patches:
            if hasattr(patch, "get_text") and patch.get_text().strip():
                out.append(patch)
    return out


# ── 图 1：检测度量流水线 ──
def fig_detection_metrics():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "检测度量流水线：从框到 AP", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    steps = [("提议框", "一批候选"), ("IoU 量框", "交集÷并集"), ("NMS 去重", "一目标一框"),
             ("查准/查全", "逐阈值扫"), ("P-R 曲线", "曲线下面积")]
    x0, w, gap = 0.28, 1.10, 0.20
    for i, (name, sub) in enumerate(steps):
        x = x0 + i * (w + gap)
        _box(ax, x, 2.62, w, 0.86, ec=(ACC if i == 1 else INK),
             fc=("#f3e9d2" if i == 1 else "#ffffff"))
        ax.text(x + w / 2, 3.18, name, fontsize=12.5, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, 2.86, sub, fontsize=11.5, color=MUT,
                ha="center", va="center")
        if i < len(steps) - 1:
            _arrow(ax, (x + w + 0.01, 3.05), (x + w + gap - 0.01, 3.05), ms=9)
    ax.text(5.95, 2.25, "AP / mAP", fontsize=12.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.text(5.95, 1.98, "各类平均", fontsize=11.5, color=MUT,
            ha="center", va="center")

    # 左下：IoU 两框示意
    ax.text(1.05, 1.80, "IoU：两框重叠", fontsize=12, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.30, 0.42), 1.10, 0.86, fc="#e6ebf2", ec=BLU, lw=1.6))
    ax.add_patch(Rectangle((0.78, 0.66), 1.10, 0.86, fc="#f3e9d2", ec=GOLD, lw=1.6,
                           alpha=0.85))
    ax.text(1.09, 0.96, "交集", fontsize=11.5, color=ACC, ha="center", va="center",
            bbox=dict(fc="#ffffff", ec="none", alpha=0.85, pad=1.5))
    ax.text(2.45, 1.53, "并集=两块相加再减交集", fontsize=11.5, color=MUT,
            ha="left", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.380–381，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：两阶段检测器谱系 ──
def fig_rcnn_family():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.02, "两阶段谱系：从提议到二部匹配", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    nodes = [("R-CNN", "2000 提议\n缩放 224\nSVM 分类"),
             ("Fast\nR-CNN", "共享卷积\nFC 双头\n分类+精修"),
             ("Faster\nR-CNN", "RPN 锚框\n替代选择性\n搜索"),
             ("FPN", "自顶向下\n多尺度\n语义融合"),
             ("DETR", "编解码\n二部匹配\n无 NMS")]
    x0, w, gap, y = 0.22, 1.16, 0.14, 2.05
    for i, (name, sub) in enumerate(nodes):
        x = x0 + i * (w + gap)
        ec = ACC if i in (0, 4) else INK
        _box(ax, x, y, w, 1.55, ec=ec, fc=("#f3e9d2" if i == 4 else "#ffffff"))
        ax.text(x + w / 2, y + 1.18, name, fontsize=12, color=INK,
                ha="center", va="center", fontweight="bold", linespacing=1.1)
        ax.text(x + w / 2, y + 0.50, sub, fontsize=11.5, color=MUT,
                ha="center", va="center", linespacing=1.35)
        if i < len(nodes) - 1:
            _arrow(ax, (x + w + 0.005, y + 0.78), (x + w + gap - 0.005, y + 0.78), ms=8)
    ax.text(3.3, 1.55, "两阶段：先找候选，再分类精修；FPN 补多尺度，DETR 换 Transformer",
            fontsize=11.5, color=BLU, ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.382–384，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 3：单阶段检测器对照 ──
def fig_single_stage():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.02, "单阶段：一张网扫过去", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    cards = [("SSD", "多尺度\n默认框"), ("YOLO", "一次回归\n位置+类别"),
             ("RetinaNet", "焦点损失\n盯难样本"), ("YOLOv4", "neck 增强\n两类技巧")]
    x0, w, gap = 0.28, 1.32, 0.18
    for i, (name, sub) in enumerate(cards):
        x = x0 + i * (w + gap)
        _box(ax, x, 2.42, w, 1.04, ec=(ACC if i == 2 else INK),
             fc=("#f3e9d2" if i == 2 else "#ffffff"))
        ax.text(x + w / 2, 3.22, name, fontsize=12.5, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, 2.78, sub, fontsize=11.5, color=MUT,
                ha="center", va="center", linespacing=1.35)
    # 下方速度-精度示意
    ax.annotate("", xy=(5.55, 0.48), xytext=(0.55, 0.48),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.annotate("", xy=(0.55, 1.78), xytext=(0.55, 0.48),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(3.05, 0.28, "速度 →", fontsize=11.5, color=MUT, ha="center", va="center")
    ax.text(0.20, 1.13, "精度 →", fontsize=11.5, color=MUT, ha="center",
            va="center", rotation=90)
    pts = [(1.25, 0.92, "YOLO"), (2.15, 1.18, "SSD"),
           (3.10, 1.30, "RetinaNet"), (4.30, 1.52, "YOLOv4")]
    for px, py, name in pts:
        ax.plot([px], [py], marker="o", ms=7, color=ACC)
        ax.text(px, py + 0.13, name, fontsize=11.5, color=INK, ha="center", va="bottom")
    ax.text(3.3, 1.78, "越靠右上：又快又准", fontsize=11.5, color=BLU,
            ha="right", va="center")
    ax.text(3.3, 0.06,
            "程序生成，未用原书原图 · 对应原书 p.384–385，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).resolve()
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    fig1 = fig_detection_metrics()
    p1 = FIG_DIR / "e061_detection_metrics.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    t1 = [t.get_text() for t in collect_texts(fig1)]
    for w in ("提议框", "IoU 量框", "NMS 去重", "查准/查全", "P-R 曲线", "AP / mAP"):
        assert w in t1, f"[FAIL] 图1 缺词 {w}"
    for w in ("交集", "并集=两块相加再减交集"):
        assert any(w in s for s in t1), f"[FAIL] 图1 缺 IoU 词 {w}"
    fs1 = [t.get_fontsize() for t in collect_texts(fig1)]

    fig2 = fig_rcnn_family()
    p2 = FIG_DIR / "e061_rcnn_family.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    t2 = [t.get_text() for t in collect_texts(fig2)]
    t2_flat = [s.replace("\n", " ") for s in t2]
    for w in ("R-CNN", "Fast R-CNN", "Faster R-CNN", "FPN", "DETR", "RPN 锚框"):
        assert any(w in s for s in t2_flat), f"[FAIL] 图2 缺词 {w}"
    fs2 = [t.get_fontsize() for t in collect_texts(fig2)]

    fig3 = fig_single_stage()
    p3 = FIG_DIR / "e061_single_stage.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    t3 = [t.get_text() for t in collect_texts(fig3)]
    for w in ("SSD", "YOLO", "RetinaNet", "YOLOv4", "焦点损失", "neck 增强"):
        assert any(w in s for s in t3), f"[FAIL] 图3 缺词 {w}"
    fs3 = [t.get_fontsize() for t in collect_texts(fig3)]

    for fs, label in ((fs1, "图1"), (fs2, "图2"), (fs3, "图3")):
        assert min(fs) >= 11.5, f"[FAIL] {label} 最小字号 {min(fs)}<11.5"
    for texts, label in ((t1, "图1"), (t2, "图2"), (t3, "图3")):
        assert sum("程序生成，未用原书原图" in s for s in texts) == 1, f"[FAIL] {label} 注记数"
    n_texts = 0
    for texts in (t1, t2, t3):
        for s in texts:
            n_texts += 1
            if FIG_MARK.search(s):
                leaked.append(s)

    pngs = sorted(FIG_DIR.glob("e061_*.png"))
    assert len(pngs) == 3, f"[FAIL] PNG 数 {len(pngs)}≠3"

    rows = [
        "图1 度量流水线词在档　| 提议框/IoU/NMS/查准查全/P-R 曲线/AP-mAP 全命中",
        "图1 IoU 两框示意词在档　| 交集/并集 全命中",
        "图2 谱系五节点词在档　| R-CNN/Fast R-CNN/Faster R-CNN/FPN/DETR/RPN 全命中",
        "图3 单阶段四检测器词在档　| SSD/YOLO/RetinaNet/YOLOv4/焦点损失/neck 全命中",
        "主文本字号≥11.5pt+程序生成注记在档×3　| "
        f"图一 最小 {min(fs1)}pt/图二 最小 {min(fs2)}pt/图三 最小 {min(fs3)}pt",
        "图内文字内部记号零泄漏（红线 5）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e061_detection_metrics.png",
        "FG2 程序生成图落盘　| e061_rcnn_family.png",
        "FG3 程序生成图落盘　| e061_single_stage.png",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"steps": 5, "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"nodes": 5, "min_fs": min(fs2), "max_fs": max(fs2)},
        "fig3": {"detectors": 4, "min_fs": min(fs3), "max_fs": max(fs3)},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rd = OUT_DIR / "e061_general_object_detection_readings.json"
    rd.write_text(json.dumps(readings, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rd.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rd.read_bytes()).hexdigest()}")
    for png in pngs:
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(rows) - n_pass} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n_pass != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
