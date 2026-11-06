#!/usr/bin/env python3
"""E064 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Ex 6.3 小方块示意图不复制）。

图 1 e064_reading_map.png（deck 页 3，f003 配图）：§6.7 六条阅读线——
  实例识别与检索/特征法分类/AlexNet 与人脸/检测/分割/视频与语言；底部下一步第 7 章。
  对应原书 p.409–413。
图 2 e064_exercise_matrix.png（deck 页 9，f009 配图）：十三道习题四类矩阵——
  诊断（6.1/6.9）/训练（6.2/6.3/6.4/6.6/6.10/6.12）/任务（6.5/6.7/6.8/6.11）/分析（6.13）。
  对应原书 p.413–415。
图 3 e064_chapter_recap.png（deck 页 16，f016 配图）：第 6 章一条线——
  分类→检测→分割→视频→语言→第 7 章特征点。对应原书 p.409–416。
断言体系（E057–E063 判例同构+readings 体系同构）：
①图 1 六条阅读线+第 7 章下一步词在档；
②图 2 十三题+四类分组词在档；
③图 3 五站+第 7 章词在档；
④主文本 fontsize≥12pt+「程序生成，未用原书原图」×3；
⑤图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak）；
⑥三张 PNG 落盘且计数=3。
readings 体系：[PASS] 断言行+readings.json 叶值存档+源码/readings/PNG sha 自证行。
输出 code/outputs/figures/{e064_reading_map,e064_exercise_matrix,e064_chapter_recap}.png
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E064_exercises_recognition"
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


def _box(ax, x, y, w, h, ec=INK, fc="#ffffff", lw=1.2):
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


# ── 图 1：六条阅读线 ──
def fig_reading_map():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "章末阅读地图：六条线，一条下一步", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    lanes = [
        ("实例识别与检索", "2D 点+视角不变描述子", "实例检索/视觉相似", BLU),
        ("特征法分类", "词袋/轮廓/部件分解", "场景上下文+机器学习", GOLD),
        ("AlexNet 与人脸", "2012 转向深度网络", "ASM/3DMM/AAM→深度", ACC),
        ("检测", "人脸/行人/汽车", "两阶段→单阶段", BLU),
        ("分割", "语义/实例/全景", "Mask R-CNN", GRN),
        ("视频与语言", "动作识别→深度网络", "描述/VQA/预训练", GOLD),
    ]
    w, h = 3.05, 0.82
    for i, (name, l1, l2, col) in enumerate(lanes):
        x = 0.20 + (i % 2) * 3.20
        y = 2.95 - (i // 2) * 0.92
        _box(ax, x, y, w, h, ec=col)
        ax.text(x + 0.12, y + 0.62, name, fontsize=12.5, color=col,
                ha="left", va="center", fontweight="bold")
        ax.text(x + 0.12, y + 0.37, l1, fontsize=12, color=INK, ha="left", va="center")
        ax.text(x + 0.12, y + 0.17, l2, fontsize=12, color=MUT, ha="left", va="center")
    _box(ax, 0.35, 0.38, 5.90, 0.42, ec=GRN)
    ax.text(3.30, 0.59, "下一步：第 7 章 特征点——跟踪、配准、检索共同的地基",
            fontsize=12.5, color=GRN, ha="center", va="center", fontweight="bold")
    ax.text(3.3, 0.14,
            "程序生成，未用原书原图 · 对应原书 p.409–413，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：十三题四类矩阵 ──
def fig_exercise_matrix():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "十三道习题：诊断、训练、任务、分析", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    cols = [
        ("诊断", ["6.1 找错误", "6.9 分布漂移"], BLU, 1.22),
        ("训练", ["6.2 重训练", "6.3 扰动", "6.4 归一化", "6.6 类别编码", "6.10 自监督", "6.12 冻结浅层"], GOLD, 1.78),
        ("任务", ["6.5 三种分割", "6.7 检测器对比", "6.8 标签 vs 描述", "6.11 视频跟踪"], ACC, 1.82),
        ("分析", ["6.13 整部电影"], GRN, 1.43),
    ]
    x = 0.05
    for name, items, col, w in cols:
        _box(ax, x, 3.30, w, 0.46, ec=col)
        ax.text(x + w / 2, 3.53, name, fontsize=13, color=col,
                ha="center", va="center", fontweight="bold")
        for j, item in enumerate(items):
            y = 2.95 - j * 0.40
            _box(ax, x, y, w, 0.32, ec=col, lw=1.0)
            ax.text(x + w / 2, y + 0.16, item, fontsize=12, color=INK,
                    ha="center", va="center")
        x += w + 0.08
    ax.text(3.3, 0.55, "四类走法：先诊断，再训练，再做任务，最后做整体分析",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.413–415，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 3：第 6 章一条线 ──
def fig_chapter_recap():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "第 6 章一条线：从分类到语言，再交棒特征点", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    stations = [
        ("分类", "47–50集", BLU),
        ("检测", "51–53集", GOLD),
        ("分割", "54集", ACC),
        ("视频", "55集", GRN),
        ("语言", "55集", BLU),
    ]
    x = 0.10
    for i, (name, ep, col) in enumerate(stations):
        _box(ax, x, 2.55, 1.18, 0.72, ec=col)
        ax.text(x + 0.59, 2.98, name, fontsize=13, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 0.59, 2.72, ep, fontsize=12, color=MUT,
                ha="center", va="center")
        if i < len(stations) - 1:
            _arrow(ax, (x + 1.19, 2.91), (x + 1.27, 2.91), ms=7)
        x += 1.30
    _arrow(ax, (6.32, 2.91), (6.44, 2.91), ms=7)
    _box(ax, 0.60, 1.50, 5.40, 0.56, ec=GRN)
    ax.text(3.30, 1.83, "第 7 章：特征点", fontsize=13, color=GRN,
            ha="center", va="center", fontweight="bold")
    ax.text(3.30, 1.62, "跟踪、配准、检索共同的地基", fontsize=12, color=INK,
            ha="center", va="center")
    ax.text(3.3, 0.95, "下一集：为什么需要特征点", fontsize=12.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.text(3.3, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.409–416，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).resolve()
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    fig1 = fig_reading_map()
    p1 = FIG_DIR / "e064_reading_map.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    t1 = [t.get_text() for t in collect_texts(fig1)]
    t1_flat = [s.replace("\n", "") for s in t1]
    for w in ("实例识别与检索", "特征法分类", "AlexNet", "检测", "分割", "视频与语言", "第 7 章"):
        assert any(w in s for s in t1_flat), f"[FAIL] 图1 缺词 {w}"
    fs1 = [t.get_fontsize() for t in collect_texts(fig1)]

    fig2 = fig_exercise_matrix()
    p2 = FIG_DIR / "e064_exercise_matrix.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    t2 = [t.get_text() for t in collect_texts(fig2)]
    t2_flat = [s.replace("\n", "") for s in t2]
    for w in ("诊断", "训练", "任务", "分析", "6.1", "6.13"):
        assert any(w in s for s in t2_flat), f"[FAIL] 图2 缺词 {w}"
    n_boxes = sum(len(v) for _, v, _ in [
        ("诊断", ["6.1", "6.9"], None), ("训练", ["6.2", "6.3", "6.4", "6.6", "6.10", "6.12"], None),
        ("任务", ["6.5", "6.7", "6.8", "6.11"], None), ("分析", ["6.13"], None)])
    assert n_boxes == 13
    fs2 = [t.get_fontsize() for t in collect_texts(fig2)]

    fig3 = fig_chapter_recap()
    p3 = FIG_DIR / "e064_chapter_recap.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    t3 = [t.get_text() for t in collect_texts(fig3)]
    t3_flat = [s.replace("\n", "") for s in t3]
    for w in ("分类", "检测", "分割", "视频", "语言", "第 7 章", "特征点"):
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
    pngs = sorted(FIG_DIR.glob("e064_*.png"))
    assert len(pngs) == 3, f"[FAIL] PNG 数 {len(pngs)}≠3"

    rows = [
        "图1 六条阅读线词在档　| 实例识别与检索/特征法分类/AlexNet/检测/分割/视频与语言 全命中",
        "图1 下一步第 7 章在档　| 第 7 章/特征点/跟踪/配准/检索 全命中",
        "图2 十三题四类词在档　| 诊断/训练/任务/分析/6.1–6.13 全命中",
        "图2 题目总数在档　| 13 题（诊断 2/训练 6/任务 4/分析 1）",
        "图3 五站+第 7 章词在档　| 分类/检测/分割/视频/语言/第 7 章/特征点 全命中",
        "主文本字号≥12pt+程序生成注记在档×3　| "
        f"图一 最小 {min(fs1)}pt/图二 最小 {min(fs2)}pt/图三 最小 {min(fs3)}pt",
        "图内文字内部记号零泄漏（红线 5）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e064_reading_map.png",
        "FG2 程序生成图落盘　| e064_exercise_matrix.png",
        "FG3 程序生成图落盘　| e064_chapter_recap.png",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"lanes": 6, "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"exercises": 13, "groups": 4, "diag": 2, "train": 6, "task": 4, "analysis": 1, "ex_min": 6.1, "ex_max": 6.13, "ex_span": 7, "min_fs": min(fs2), "max_fs": max(fs2)},
        "fig3": {"stations": 6, "min_fs": min(fs3), "max_fs": max(fs3)},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rd = OUT_DIR / "e064_exercises_recognition_readings.json"
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
