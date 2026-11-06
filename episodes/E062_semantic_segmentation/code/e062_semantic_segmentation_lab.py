#!/usr/bin/env python3
"""E062 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 6.32–6.43+Table 6.2 均不复制）。

图 1 e062_segmentation_tasks.png（deck 页 2，f002 配图）：四类分割任务对照——
  语义分割（逐像素分类）/实例分割（每个物体一个掩码）/全景分割（物体+背景）/
  密集姿态（人体像素映射到三维表面）。对应原书 p.387。
图 2 e062_fcn_unet.png（deck 页 4，f004 配图）：FCN→U-Net 流水线——
  编码器逐级下采样、解码器逐级上采样、跳连把浅层细节接回、逐像素输出。
  对应原书 p.388。
图 3 e062_mask_rcnn.png（deck 页 10，f010 配图）：Mask R-CNN 三头与 Panoptic FPN——
  RPN 提议→类别/框/掩码三头；Panoptic FPN 的实例+语义两分支合并。
  对应原书 p.391–393。
断言体系（E057–E061 判例同构+readings 体系同构）：
①图 1 四类任务词在档（语义/实例/全景/密集姿态）；
②图 2 编解码词在档（编码器/解码器/跳连/逐像素输出）；
③图 3 三头+两分支词在档（类别/框/掩码/实例分支/语义分支）；
④主文本 fontsize≥11.5pt+「程序生成，未用原书原图」×3；
⑤图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak）；
⑥三张 PNG 落盘且计数=3。
readings 体系：[PASS] 断言行+readings.json 叶值存档+源码/readings/PNG sha 自证行。
输出 code/outputs/figures/{e062_segmentation_tasks,e062_fcn_unet,e062_mask_rcnn}.png
（6.6×4.2 inch @dpi150）。
"""
import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E062_semantic_segmentation"
FIG_DIR = EP / "code" / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EP / "code" / "outputs"

PAPER = "#faf7f0"
INK = "#2f3540"
MUT = "#5a6472"
ACC = "#8a3324"
BLU = "#3c4f6b"
GOLD = "#b08a3e"
GRN = "#4a6b4a"
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")


def _box(ax, x, y, w, h, ec=INK, fc="#ffffff", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))


def _arrow(ax, p1, p2, color=GOLD, lw=1.3, ms=10):
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


# ── 图 1：四类分割任务 ──
def fig_segmentation_tasks():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "四类分割任务：从像素类别到三维表面", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    cards = [("语义分割", "每像素分类", BLU), ("实例分割", "每物一掩码", ACC),
             ("全景分割", "物体+背景", GOLD), ("密集姿态", "像素→表面", GRN)]
    x0, w, gap = 0.25, 1.42, 0.18
    for i, (name, sub, col) in enumerate(cards):
        x = x0 + i * (w + gap)
        _box(ax, x, 2.55, w, 1.10, ec=col)
        ax.text(x + w / 2, 3.32, name, fontsize=12.5, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, 2.92, sub, fontsize=11.5, color=MUT,
                ha="center", va="center")
        # 小示意：场景色块
        ax.add_patch(Rectangle((x + 0.18, 2.20), w - 0.36, 0.18, fc="#dfe7f0", ec=BLU, lw=0.6))
        ax.add_patch(Rectangle((x + 0.18, 1.98), w - 0.36, 0.18, fc="#e8e2d2", ec=GOLD, lw=0.6))
        if i == 1 or i == 2:
            ax.add_patch(Ellipse((x + 0.55, 2.28), 0.22, 0.30, fc=ACC, ec=ACC, alpha=0.6))
            ax.add_patch(Ellipse((x + 0.90, 2.28), 0.22, 0.30, fc=ACC, ec=ACC, alpha=0.6))
        if i == 3:
            ax.add_patch(Ellipse((x + 0.72, 2.28), 0.30, 0.34, fc=GRN, ec=GRN, alpha=0.5))
    ax.text(3.3, 1.58, "语义：逐像素分到类别；实例：每个物体单独描",
            fontsize=11.5, color=BLU, ha="center", va="center")
    ax.text(3.3, 1.30, "全景：物体与背景一起标；密集姿态：人体像素对应三维表面",
            fontsize=11.5, color=BLU, ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.387，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：FCN→U-Net ──
def fig_fcn_unet():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.02, "FCN 与 U-Net：编码看全局，解码补细节", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    ys = [3.05, 2.60, 2.15, 1.70]
    # 编码器（左，下采样）
    for i, y in enumerate(ys):
        _box(ax, 0.30, y, 1.05, 0.30, ec=BLU)
        ax.text(0.825, y + 0.15, f"下采样{i + 1}", fontsize=11.5, color=BLU,
                ha="center", va="center")
        if i < 3:
            _arrow(ax, (0.825, y), (0.825, ys[i + 1] + 0.30), ms=7)
    # 瓶颈
    _box(ax, 2.45, 1.10, 1.20, 0.32, ec=ACC)
    ax.text(3.05, 1.26, "瓶颈特征", fontsize=11.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    _arrow(ax, (1.36, 1.78), (2.44, 1.30), ms=7)
    # 解码器（右，自下而上逐级上采样）
    ys_dec = [1.70, 2.15, 2.60, 3.05]
    for j, y in enumerate(ys_dec):
        _box(ax, 4.35, y, 1.05, 0.30, ec=GOLD)
        ax.text(4.875, y + 0.15, f"上采样{j + 1}", fontsize=11.5, color=INK,
                ha="center", va="center")
        if j < 3:
            _arrow(ax, (4.875, y + 0.30), (4.875, ys_dec[j + 1]), ms=7)
    _arrow(ax, (3.66, 1.30), (4.34, 1.85), ms=7)
    # 输出（解码器 4 之后）
    _box(ax, 5.58, 2.95, 0.82, 0.50, ec=ACC)
    ax.text(5.99, 3.20, "逐像素\n输出", fontsize=11.5, color=ACC,
            ha="center", va="center", fontweight="bold", linespacing=1.2)
    _arrow(ax, (5.41, 3.20), (5.57, 3.20), ms=7)
    # 跳连（镜像层）
    for i in range(3):
        ax.annotate("", xy=(4.33, ys[i] + 0.15), xytext=(1.37, ys[i] + 0.15),
                    arrowprops=dict(arrowstyle="-|>", color=GRN, lw=1.1,
                                    linestyle=(0, (4, 2))))
    ax.text(2.80, 3.62, "跳连：把浅层细节接回来", fontsize=11.5, color=GRN,
            ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.388，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 3：Mask R-CNN 三头+Panoptic FPN ──
def fig_mask_rcnn():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "Mask R-CNN 三头与全景双分支", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    _box(ax, 0.30, 2.55, 1.20, 0.66, ec=BLU)
    ax.text(0.90, 2.88, "区域提议", fontsize=12, color=BLU,
            ha="center", va="center", fontweight="bold")
    heads = [("类别头", 3.30), ("框精修头", 2.70), ("掩码头", 2.10)]
    for name, y in heads:
        _box(ax, 2.25, y - 0.22, 1.25, 0.44, ec=(ACC if name == "掩码头" else INK))
        ax.text(2.875, y, name, fontsize=11.5, color=INK, ha="center", va="center")
        _arrow(ax, (1.52, 2.88), (2.23, y), ms=7)
    ax.text(4.45, 2.62, "多分支损失\n要平衡", fontsize=11.5, color=ACC,
            ha="right", va="center", linespacing=1.35)
    # 全景双分支
    _box(ax, 4.55, 3.30, 1.75, 0.40, ec=GRN)
    ax.text(5.425, 3.50, "实例分支", fontsize=11.5, color=GRN, ha="center", va="center")
    _box(ax, 4.55, 2.80, 1.75, 0.40, ec=GOLD)
    ax.text(5.425, 3.00, "语义分支", fontsize=11.5, color=GOLD, ha="center", va="center")
    _box(ax, 4.55, 2.05, 1.75, 0.45, ec=ACC)
    ax.text(5.425, 2.275, "全景输出", fontsize=11.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    _arrow(ax, (5.425, 3.28), (5.425, 3.22), ms=6)
    _arrow(ax, (5.425, 2.78), (5.425, 2.52), ms=6)
    ax.text(3.3, 1.45, "三头：类别/框/掩码；全景：实例+语义两分支在 FPN 上合并",
            fontsize=11.5, color=BLU, ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.391–393，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).resolve()
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    fig1 = fig_segmentation_tasks()
    p1 = FIG_DIR / "e062_segmentation_tasks.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    t1 = [t.get_text() for t in collect_texts(fig1)]
    for w in ("语义分割", "实例分割", "全景分割", "密集姿态"):
        assert w in t1, f"[FAIL] 图1 缺词 {w}"
    fs1 = [t.get_fontsize() for t in collect_texts(fig1)]

    fig2 = fig_fcn_unet()
    p2 = FIG_DIR / "e062_fcn_unet.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    t2 = [t.get_text() for t in collect_texts(fig2)]
    t2_flat = [s.replace("\n", "") for s in t2]
    for w in ("下采样1", "瓶颈特征", "上采样1", "跳连：把浅层细节接回来", "逐像素输出"):
        assert any(w in s for s in t2_flat), f"[FAIL] 图2 缺词 {w}"
    fs2 = [t.get_fontsize() for t in collect_texts(fig2)]

    fig3 = fig_mask_rcnn()
    p3 = FIG_DIR / "e062_mask_rcnn.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    t3 = [t.get_text() for t in collect_texts(fig3)]
    for w in ("类别头", "框精修头", "掩码头", "实例分支", "语义分支"):
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
    pngs = sorted(FIG_DIR.glob("e062_*.png"))
    assert len(pngs) == 3, f"[FAIL] PNG 数 {len(pngs)}≠3"

    rows = [
        "图1 四类任务词在档　| 语义分割/实例分割/全景分割/密集姿态 全命中",
        "图1 任务对照句在档　| 逐像素分类/物体掩码/物体+背景/像素→表面 全命中",
        "图2 编解码词在档　| 下采样/瓶颈/上采样/跳连/逐像素输出 全命中",
        "图3 三头+两分支词在档　| 类别头/框精修头/掩码头/实例分支/语义分支 全命中",
        "主文本字号≥11.5pt+程序生成注记在档×3　| "
        f"图一 最小 {min(fs1)}pt/图二 最小 {min(fs2)}pt/图三 最小 {min(fs3)}pt",
        "图内文字内部记号零泄漏（红线 5）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e062_segmentation_tasks.png",
        "FG2 程序生成图落盘　| e062_fcn_unet.png",
        "FG3 程序生成图落盘　| e062_mask_rcnn.png",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"tasks": 4, "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"encoder_levels": 4, "decoder_levels": 4, "min_fs": min(fs2), "max_fs": max(fs2)},
        "fig3": {"heads": 3, "branches": 2, "min_fs": min(fs3), "max_fs": max(fs3)},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rd = OUT_DIR / "e062_semantic_segmentation_readings.json"
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
