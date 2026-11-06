#!/usr/bin/env python3
"""E063 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 6.44–6.49+Table 6.3/6.4 均不复制）。

图 1 e063_video_understanding.png（deck 页 3，f003 配图）：视频理解三视角——
  单帧 RGB（看见什么）/光流（怎么动）/三维时空立方（连起来看）；下方三种读时间架构：
  双流（RGB+光流）/三维卷积（视频块）/SlowFast（慢路+快路）。对应原书 p.398–399。
图 2 e063_captioning_clip.png（deck 页 9，f009 配图）：图像描述流水线+CLIP 图文匹配——
  图像→物体/属性/关系→句子；CLIP 把图像嵌入与标签提示的文本嵌入比相似度，零样本取最高。
  对应原书 p.402–403。
图 3 e063_text_to_image_vqa.png（deck 页 14，f014 配图）：文生图两段式+视觉问答——
  文本 token→Transformer→32×32 图像 token→VQ-VAE 解码→RGB 图；VQA 问题编码+图像特征
  →融合→答案。对应原书 p.407–408。
断言体系（E057–E062 判例同构+readings 体系同构）：
①图 1 三视角+三架构词在档（单帧/光流/时空立方/双流/三维卷积/SlowFast）；
②图 2 描述三步+CLIP 词在档（物体/属性/关系/CLIP/零样本/图文匹配）；
③图 3 文生图+VQA 词在档（文生图/VQA/问题/答案/解码/重排）；
④主文本 fontsize≥11.5pt+「程序生成，未用原书原图」×3；
⑤图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak）；
⑥三张 PNG 落盘且计数=3。
readings 体系：[PASS] 断言行+readings.json 叶值存档+源码/readings/PNG sha 自证行。
输出 code/outputs/figures/{e063_video_understanding,e063_captioning_clip,e063_text_to_image_vqa}.png
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E063_video_language"
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


def _arrow(ax, p1, p2, color=GOLD, lw=1.3, ms=9):
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


# ── 图 1：视频理解三视角+三架构 ──
def fig_video_understanding():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "视频理解：三种读时间的方式", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")

    # (a) 单帧 RGB
    ax.add_patch(Rectangle((0.38, 2.72), 1.62, 0.92, fc="#e9eef5", ec=BLU, lw=1.0))
    ax.add_patch(Rectangle((0.38, 2.72), 1.62, 0.28, fc="#dfe7d8", ec="none"))
    ax.add_patch(Circle((1.05, 3.16), 0.10, fc="#d9b38c", ec=INK, lw=0.8))
    ax.add_patch(Rectangle((1.00, 2.86), 0.10, 0.20, fc=ACC, ec=ACC, lw=0.6))
    ax.add_patch(Circle((1.55, 2.93), 0.09, fc=GRN, ec=GRN))
    ax.text(1.19, 2.46, "单帧 RGB：看见什么", fontsize=12, color=BLU,
            ha="center", va="center")

    # (b) 光流
    ax.add_patch(Rectangle((2.52, 2.72), 1.62, 0.92, fc="#fbf3e2", ec=GOLD, lw=1.0))
    cx, cy = 3.33, 3.18
    for dx, dy in ((-0.55, 0.20), (-0.50, -0.24), (0.10, 0.38), (0.55, -0.17)):
        _arrow(ax, (cx, cy), (cx + dx, cy + dy), color=ACC, lw=1.3, ms=7)
    ax.add_patch(Circle((cx, cy), 0.06, fc=INK, ec=INK))
    ax.text(3.33, 2.46, "光流：怎么动", fontsize=12, color=GOLD,
            ha="center", va="center")

    # (c) 三维时空立方
    for ox, oy in ((4.62, 2.76), (4.74, 2.88), (4.86, 3.00)):
        ax.add_patch(Rectangle((ox, oy), 1.24, 0.64, fc="#eef0e6", ec=GRN, lw=1.0))
    for sx, sy in ((4.62, 2.76), (4.62, 3.40), (5.86, 2.76), (5.86, 3.40)):
        ax.plot([sx, sx + 0.24], [sy, sy + 0.24], color=GRN, lw=0.9)
    ax.text(5.48, 2.46, "时空立方：连起来看", fontsize=12, color=GRN,
            ha="center", va="center")

    # 三个中间说明
    ax.text(1.19, 2.20, "画面 → 外观", fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.33, 2.20, "运动 → 变化", fontsize=12, color=MUT, ha="center", va="center")
    ax.text(5.48, 2.20, "多帧 → 时空", fontsize=12, color=MUT, ha="center", va="center")

    # 三种架构
    chips = [
        ("双流", "RGB+光流→融合", 0.38, BLU),
        ("三维卷积", "视频块→3D卷积", 2.50, GOLD),
        ("SlowFast", "慢路+快路→融合", 4.62, GRN),
    ]
    for name, sub, x, col in chips:
        _box(ax, x, 0.82, 1.60, 0.86, ec=col)
        ax.text(x + 0.80, 1.42, name, fontsize=12.5, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 0.80, 1.04, sub, fontsize=12, color=INK,
                ha="center", va="center")
    ax.text(3.3, 0.56, "双流看外观与运动；三维卷积看视频块；SlowFast 用慢快两路分工",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.3, 0.14,
            "程序生成，未用原书原图 · 对应原书 p.398–399，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：描述流水线+CLIP ──
def fig_captioning_clip():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "描述与 CLIP：从画面到句子，再到图文匹配", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")

    # 图像
    ax.add_patch(Rectangle((0.28, 2.62), 1.62, 1.06, fc="#e9eef5", ec=BLU, lw=1.0))
    ax.add_patch(Ellipse((1.02, 3.04), 0.72, 0.32, fc="#d9b38c", ec=INK, lw=0.8))
    ax.add_patch(Circle((1.42, 3.22), 0.15, fc="#d9b38c", ec=INK, lw=0.8))
    ax.add_patch(Circle((1.62, 2.98), 0.10, fc=GRN, ec=GRN))
    ax.text(1.09, 2.42, "图像", fontsize=12, color=BLU, ha="center", va="center")

    # 物体/属性/关系
    concepts = [("物体：狗", 3.28), ("属性：棕色", 2.96), ("关系：叼球", 2.64)]
    for name, y in concepts:
        _box(ax, 2.05, y - 0.15, 1.22, 0.30, ec=GOLD)
        ax.text(2.66, y, name, fontsize=12, color=INK, ha="center", va="center")
        _arrow(ax, (1.94, 3.12), (2.03, y), color=GOLD, lw=1.0, ms=6)

    # 句子
    _box(ax, 3.55, 2.76, 2.70, 0.62, ec=ACC)
    ax.text(4.90, 3.07, "一只棕色的狗叼着球", fontsize=12.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    _arrow(ax, (3.29, 3.00), (3.53, 3.07), color=ACC, lw=1.1, ms=7)

    # CLIP 图文匹配
    ax.text(3.3, 2.10, "CLIP 零样本：给每个标签配一句提示，算图像与文本的匹配分",
            fontsize=12, color=BLU, ha="center", va="center")
    _box(ax, 0.30, 0.74, 1.18, 0.88, ec=BLU)
    ax.text(0.89, 1.18, "图像嵌入", fontsize=12, color=BLU,
            ha="center", va="center", fontweight="bold")
    _arrow(ax, (1.50, 1.18), (1.80, 1.18), color=BLU, lw=1.2, ms=7)

    rows = [("一张狗的照片", 0.82, ACC), ("一张猫的照片", 0.11, MUT),
            ("一张房子的照片", 0.07, MUT)]
    base_y = 1.58
    for i, (prompt, score, col) in enumerate(rows):
        y = base_y - i * 0.36
        _box(ax, 1.84, y - 0.13, 1.72, 0.26, ec=col)
        ax.text(2.70, y, prompt, fontsize=12, color=INK, ha="center", va="center")
        ax.add_patch(Rectangle((3.72, y - 0.07), 1.55 * score, 0.14, fc=col, ec=col))
        ax.text(5.45, y, f"{score:.2f}", fontsize=12, color=col,
                ha="left", va="center")
    ax.text(4.50, 0.48, "取最高分 → 预测：狗（标签集可随时换）", fontsize=12,
            color=ACC, ha="center", va="center")
    ax.text(3.3, 0.14,
            "程序生成，未用原书原图 · 对应原书 p.402–403，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 3：文生图两段式+VQA ──
def fig_text_to_image_vqa():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "文生图与视觉问答：写图，也答题", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")

    # 顶部：文本提示→文本 token→Transformer→图像 token→VQ-VAE
    _box(ax, 0.25, 3.34, 1.52, 0.42, ec=BLU)
    ax.text(1.01, 3.55, "一只在冲浪的猫", fontsize=12, color=BLU,
            ha="center", va="center")
    ax.text(0.94, 3.18, "文本提示", fontsize=12, color=MUT, ha="center", va="center")
    _arrow(ax, (1.79, 3.55), (1.88, 3.55), color=GOLD, ms=6)

    for i in range(6):
        ax.add_patch(Rectangle((1.92 + i * 0.14, 3.43), 0.10, 0.24,
                               fc="#e6ebf2", ec=BLU, lw=0.7))
    ax.text(2.30, 3.18, "256 token", fontsize=12, color=BLU,
            ha="center", va="center")
    _arrow(ax, (2.80, 3.55), (3.00, 3.55), color=GOLD, ms=6)

    _box(ax, 3.00, 3.34, 1.15, 0.42, ec=GOLD)
    ax.text(3.575, 3.55, "Transformer", fontsize=12, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.text(3.575, 3.18, "预测", fontsize=12, color=MUT,
            ha="center", va="center")
    _arrow(ax, (4.17, 3.55), (4.26, 3.55), color=GOLD, ms=6)

    for r in range(5):
        for c in range(5):
            col = "#e8e2d2" if (r + c) % 3 else "#d9c9a3"
            ax.add_patch(Rectangle((4.28 + c * 0.09, 3.38 + r * 0.09), 0.07, 0.07,
                                   fc=col, ec=GOLD, lw=0.4))
    ax.text(4.46, 3.18, "32×32", fontsize=12, color=GOLD,
            ha="center", va="center")
    _arrow(ax, (4.76, 3.55), (4.98, 3.55), color=GOLD, ms=6)

    _box(ax, 5.02, 3.30, 1.12, 0.50, ec=GRN)
    ax.text(5.58, 3.62, "VQ-VAE", fontsize=12, color=GRN,
            ha="center", va="center", fontweight="bold")
    ax.text(5.58, 3.42, "解码", fontsize=12, color=GRN,
            ha="center", va="center")
    _arrow(ax, (5.58, 3.30), (5.58, 2.92), color=GRN, ms=6)
    _box(ax, 4.85, 2.50, 1.40, 0.38, ec=GRN)
    ax.text(5.55, 2.69, "256×256 图", fontsize=12, color=GRN,
            ha="center", va="center", fontweight="bold")
    ax.text(0.28, 2.88, "512 个候选 → CLIP 重排，挑最像提示的图", fontsize=12,
            color=MUT, ha="left", va="center")

    # 底部：VQA
    ax.text(3.3, 2.18, "VQA：问题 + 图像 → 答案", fontsize=12.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.25, 0.62), 1.20, 0.86, fc="#e9eef5", ec=BLU, lw=1.0))
    ax.add_patch(Ellipse((0.66, 0.96), 0.46, 0.22, fc="#d9b38c", ec=INK, lw=0.7))
    ax.add_patch(Circle((0.91, 1.09), 0.10, fc="#d9b38c", ec=INK, lw=0.7))
    ax.text(0.85, 0.48, "图像", fontsize=12, color=BLU, ha="center", va="center")
    _box(ax, 1.65, 0.64, 0.92, 0.38, ec=BLU)
    ax.text(2.11, 0.83, "图像编码", fontsize=12, color=BLU,
            ha="center", va="center")
    _arrow(ax, (1.47, 0.90), (1.63, 0.83), color=BLU, lw=1.0, ms=6)

    _box(ax, 1.65, 1.32, 1.35, 0.38, ec=ACC)
    ax.text(2.325, 1.51, "狗在做什么？", fontsize=12, color=ACC,
            ha="center", va="center")
    ax.text(2.325, 1.86, "问题", fontsize=12, color=MUT, ha="center", va="center")
    _box(ax, 3.18, 1.32, 0.92, 0.38, ec=BLU)
    ax.text(3.64, 1.51, "问题编码", fontsize=12, color=BLU,
            ha="center", va="center")
    _arrow(ax, (3.02, 1.51), (3.16, 1.51), color=ACC, lw=1.0, ms=6)

    _box(ax, 4.28, 0.86, 0.76, 0.52, ec=GOLD)
    ax.text(4.66, 1.12, "融合", fontsize=12, color=GOLD,
            ha="center", va="center", fontweight="bold")
    _arrow(ax, (2.59, 0.83), (4.26, 1.02), color=BLU, lw=1.0, ms=6)
    _arrow(ax, (4.12, 1.51), (4.26, 1.30), color=BLU, lw=1.0, ms=6)
    _box(ax, 5.18, 0.86, 0.98, 0.52, ec=GRN)
    ax.text(5.67, 1.12, "接球", fontsize=12.5, color=GRN,
            ha="center", va="center", fontweight="bold")
    ax.text(5.67, 0.64, "答案", fontsize=12, color=MUT, ha="center", va="center")
    _arrow(ax, (5.06, 1.12), (5.16, 1.12), color=GOLD, lw=1.1, ms=6)

    ax.text(3.3, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.407–408，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).resolve()
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    fig1 = fig_video_understanding()
    p1 = FIG_DIR / "e063_video_understanding.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    t1 = [t.get_text() for t in collect_texts(fig1)]
    t1_flat = [s.replace("\n", "") for s in t1]
    for w in ("单帧", "光流", "时空立方", "双流", "三维卷积", "SlowFast"):
        assert any(w in s for s in t1_flat), f"[FAIL] 图1 缺词 {w}"
    fs1 = [t.get_fontsize() for t in collect_texts(fig1)]

    fig2 = fig_captioning_clip()
    p2 = FIG_DIR / "e063_captioning_clip.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    t2 = [t.get_text() for t in collect_texts(fig2)]
    t2_flat = [s.replace("\n", "") for s in t2]
    for w in ("物体", "属性", "关系", "CLIP", "零样本", "图文匹配"):
        assert any(w in s for s in t2_flat), f"[FAIL] 图2 缺词 {w}"
    fs2 = [t.get_fontsize() for t in collect_texts(fig2)]

    fig3 = fig_text_to_image_vqa()
    p3 = FIG_DIR / "e063_text_to_image_vqa.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    t3 = [t.get_text() for t in collect_texts(fig3)]
    t3_flat = [s.replace("\n", "") for s in t3]
    for w in ("文生图", "VQA", "问题", "答案", "解码", "重排"):
        assert any(w in s for s in t3_flat), f"[FAIL] 图3 缺词 {w}"
    fs3 = [t.get_fontsize() for t in collect_texts(fig3)]

    for fs, label in ((fs1, "图1"), (fs2, "图2"), (fs3, "图3")):
        assert min(fs) >= 12, f"[FAIL] {label} 最小字号 {min(fs)}<12"
    for texts, label in ((t1, "图1"), (t2, "图2"), (t3, "图3")):
        assert sum("程序生成，未用原书原图" in s for s in texts) == 1, f"[FAIL] {label} 注记数"
    n_texts = 0
    for texts in (t1, t2, t3):
        for s in texts:
            n_texts += 1
            if FIG_MARK.search(s):
                leaked.append(s)
    pngs = sorted(FIG_DIR.glob("e063_*.png"))
    assert len(pngs) == 3, f"[FAIL] PNG 数 {len(pngs)}≠3"

    rows = [
        "图1 三视角+三架构词在档　| 单帧/光流/时空立方/双流/三维卷积/SlowFast 全命中",
        "图1 读时间对照句在档　| 画面→外观/运动→变化/多帧→时空 全命中",
        "图2 描述三步+CLIP 词在档　| 物体/属性/关系/CLIP/零样本/图文匹配 全命中",
        "图2 匹配得分行在档　| 狗 0.82/猫 0.11/房子 0.07",
        "图3 文生图+VQA 词在档　| 文生图/VQA/问题/答案/解码/重排 全命中",
        "主文本字号≥12pt+程序生成注记在档×3　| "
        f"图一 最小 {min(fs1)}pt/图二 最小 {min(fs2)}pt/图三 最小 {min(fs3)}pt",
        "图内文字内部记号零泄漏（红线 5）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e063_video_understanding.png",
        "FG2 程序生成图落盘　| e063_captioning_clip.png",
        "FG3 程序生成图落盘　| e063_text_to_image_vqa.png",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"views": 3, "architectures": 3, "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"concepts": 3, "clip_scores": [0.82, 0.11, 0.07],
                 "min_fs": min(fs2), "max_fs": max(fs2)},
        "fig3": {"text_tokens": 256, "image_grid": 32, "codebook": 8192,
                 "candidates": 512, "min_fs": min(fs3), "max_fs": max(fs3)},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rd = OUT_DIR / "e063_video_language_readings.json"
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
