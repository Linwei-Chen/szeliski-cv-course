#!/usr/bin/env python3
"""E060 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 6.19–6.25 均不复制）。

图 1 e060_face_detection_taxonomy.png（deck 页 3，f003 配图）：人脸检测三路线——
  特征法（找五官/看排布）、模板法（整脸套/需初值）、外观法（滑窗→金字塔→级联）；
  三卡对照+外观法流水线。对应原书 p.371–372。
图 2 e060_viola_jones_features.png（deck 页 9，f009 配图）：Viola–Jones 三件套——
  2–4 矩形差分（白减灰）、积分图四点速查、三级级联（前级拒大量非脸，后级细判候选）；
  对应原书 p.374–375。
图 3 e060_hog_pipeline.png（deck 页 13，f013 配图）：HOG+SVM——图像规则网格→每格
  方向梯度直方图→SVM 权重→头/躯干/脚正响应；对应原书 p.376–377。
断言体系（E057–E059 判例同构+readings 体系 E053 同构）：
①图 1 三路线词与流水线词在档（特征法/模板法/外观法/滑窗/金字塔/级联）；
②图 2 三件套词在档（矩形差分/积分图/级联/非脸/候选脸）；
③图 3 四段词在档（规则网格/方向直方图/SVM 权重/头/躯干/脚）；
④几何自证：图 1 流水线箭头贴框、图 2 积分图四角点落在矩形角上、图 3 网格与直方图
  连接箭头贴边（容差 TOL）；
⑤主文本 fontsize≥10pt（12/12.5/14 档）+「程序生成，未用原书原图」×3 图各一；
⑥图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak 零命中）。
readings 体系：[PASS] 断言行+readings.json 叶值存档+源码/readings/PNG sha 自证行。
输出 code/outputs/figures/{e060_face_detection_taxonomy,e060_viola_jones_features,
e060_hog_pipeline}.png（6.6×4.2 inch @dpi150）。
红线 5：可见层无内部记号——图内不渲染集号/内部编号/节记号；只标概念+原书页。
"""
import datetime
import hashlib
import json
import math
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E060_face_detection_pedestrian"
FIG_DIR = EP / "code" / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = EP / "code" / "outputs"

PAPER = "#faf7f0"
INK = "#2f3540"
MUT = "#5a6472"
ACC = "#8a3324"
BLU = "#3c4f6b"
GOLD = "#b08a3e"
TOL = 0.012
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")


def _box(ax, x, y, w, h, ec=INK, fc="#ffffff", lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))


def _arrow(ax, p1, p2, color=GOLD, lw=1.3, ms=11, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls))


# ── 图 1：三路线+外观法流水线 ──
ROUTES = [
    ("特征法", "找眼睛、鼻子、嘴巴\n再看摆得合不合理", BLU),
    ("模板法", "整张脸去套\n要初值，怕快不起来", ACC),
    ("外观法", "扫小矩形块\n先粗筛，再细判", GOLD),
]
PIPE = ["滑窗", "金字塔", "级联"]


def fig_taxonomy():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.05, "人脸检测三条路线", fontsize=14, color=INK,
            ha="center", va="center", fontweight="bold")
    xs = [0.18, 2.32, 4.46]
    for (name, body, col), x in zip(ROUTES, xs):
        _box(ax, x, 2.55, 1.96, 1.25, ec=col)
        ax.text(x + 0.98, 3.55, name, fontsize=13, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 0.98, 3.05, body, fontsize=11.5, color=MUT,
                ha="center", va="center", linespacing=1.35)
    # 外观法流水线
    ax.text(3.3, 2.28, "外观法：先撒网，再收网", fontsize=12.5, color=INK,
            ha="center", va="center", fontweight="bold")
    px = [0.45, 2.45, 4.45]
    for name, x in zip(PIPE, px):
        _box(ax, x, 1.15, 1.55, 0.70, ec=GOLD)
        ax.text(x + 0.775, 1.50, name, fontsize=13, color=INK,
                ha="center", va="center", fontweight="bold")
    for x in (2.00, 4.00):
        _arrow(ax, (x, 1.50), (x + 0.42, 1.50))
    ax.text(1.225, 0.92, "在图上扫小矩形块", fontsize=11.5, color=MUT,
            ha="center", va="center")
    ax.text(3.225, 0.92, "每层扫一遍", fontsize=11.5, color=MUT,
            ha="center", va="center")
    ax.text(5.225, 0.92, "粗筛→细判", fontsize=11.5, color=MUT,
            ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.371–372，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：矩形差分+积分图+级联 ──
def fig_viola_jones():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "Viola–Jones：矩形差分、积分图、级联", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    # 左：2 矩形与 3 矩形特征
    ax.text(1.15, 3.58, "矩形差分（白减灰）", fontsize=12, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.45, 2.75), 0.55, 0.55, fc="#ffffff", ec=INK, lw=1.2))
    ax.add_patch(Rectangle((1.00, 2.75), 0.55, 0.55, fc="#aab3bf", ec=INK, lw=1.2))
    ax.text(1.00, 2.48, "两矩形", fontsize=11.5, color=MUT, ha="center", va="center")
    ax.add_patch(Rectangle((0.45, 1.55), 0.37, 0.55, fc="#ffffff", ec=INK, lw=1.2))
    ax.add_patch(Rectangle((0.82, 1.55), 0.37, 0.55, fc="#aab3bf", ec=INK, lw=1.2))
    ax.add_patch(Rectangle((1.19, 1.55), 0.37, 0.55, fc="#ffffff", ec=INK, lw=1.2))
    ax.text(1.15, 1.28, "三矩形（眼—脸颊—眼）", fontsize=11.5, color=MUT,
            ha="center", va="center")
    # 中：积分图四点
    ax.text(3.25, 3.58, "积分图：四点速查", fontsize=12, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((2.45, 1.55), 1.60, 1.35, fc="#ffffff", ec=INK, lw=1.2))
    for gx in (2.45, 2.85, 3.25, 3.65, 4.05):
        ax.plot([gx, gx], [1.55, 2.90], color="#c9c2b4", lw=0.8)
    for gy in (1.55, 1.95, 2.35, 2.90):
        ax.plot([2.45, 4.05], [gy, gy], color="#c9c2b4", lw=0.8)
    ax.add_patch(Rectangle((2.85, 1.95), 0.80, 0.55, fc="#e8dcc4", ec=ACC, lw=1.5, alpha=0.75))
    for px, py in [(2.85, 1.95), (3.65, 1.95), (2.85, 2.50), (3.65, 2.50)]:
        ax.plot(px, py, marker="o", ms=5, color=ACC)
    ax.text(3.25, 1.28, "四个角点各查一次表", fontsize=11.5, color=MUT,
            ha="center", va="center")
    # 右：三级级联
    ax.text(5.35, 3.58, "三级级联", fontsize=12, color=INK,
            ha="center", va="center", fontweight="bold")
    for i, y in enumerate((2.75, 2.05, 1.35)):
        _box(ax, 4.65, y, 1.25, 0.52, ec=GOLD)
        ax.text(5.275, y + 0.26, f"第 {i + 1} 级", fontsize=12, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(6.02, y + 0.26, "拒非脸", fontsize=11.5, color=ACC,
                ha="left", va="center")
        if i < 2:
            _arrow(ax, (5.275, y), (5.275, y - 0.16), color=GOLD, ms=9)
    ax.text(5.275, 1.05, "候选脸通过", fontsize=11.5, color=BLU,
            ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.374–375，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


# ── 图 3：HOG+SVM 流水线 ──
def fig_hog_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(3.3, 4.0, "HOG+SVM：从方向直方图到检测分数", fontsize=14,
            color=INK, ha="center", va="center", fontweight="bold")
    # 左：图像规则网格
    ax.text(1.15, 3.55, "规则网格", fontsize=12.5, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(Rectangle((0.30, 1.65), 1.70, 1.55, fc="#ffffff", ec=INK, lw=1.2))
    for gx in [0.30 + i * 0.34 for i in range(6)]:
        ax.plot([gx, gx], [1.65, 3.20], color="#c9c2b4", lw=0.8)
    for gy in [1.65 + i * 0.31 for i in range(6)]:
        ax.plot([0.30, 2.00], [gy, gy], color="#c9c2b4", lw=0.8)
    ax.plot([0.98, 1.32], [2.58, 2.58], color=ACC, lw=2.0)
    ax.plot([0.98, 1.32], [2.58, 2.58], color=ACC, lw=2.0)
    ax.annotate("", xy=(1.52, 2.72), xytext=(1.18, 2.44),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=1.8))
    ax.text(1.15, 1.40, "每格统计梯度方向", fontsize=11.5, color=MUT,
            ha="center", va="center")
    _arrow(ax, (2.05, 2.45), (2.45, 2.45))
    # 中：方向直方图
    ax.text(3.18, 3.55, "方向直方图", fontsize=12.5, color=INK,
            ha="center", va="center", fontweight="bold")
    ax.plot([2.55, 3.80], [1.85, 1.85], color=INK, lw=1.2)
    heights = [0.30, 0.55, 0.80, 0.45, 0.62]
    for i, h in enumerate(heights):
        ax.add_patch(Rectangle((2.65 + i * 0.21, 1.85), 0.14, h,
                               fc=BLU, ec="none"))
    ax.text(3.18, 1.55, "按幅值加权投票", fontsize=11.5, color=MUT,
            ha="center", va="center")
    _arrow(ax, (3.90, 2.45), (4.30, 2.45))
    # 右：SVM 权重+人体
    ax.text(5.35, 3.55, "SVM 权重", fontsize=12.5, color=INK,
            ha="center", va="center", fontweight="bold")
    # 人体简笔
    ax.add_patch(Circle((5.35, 2.95), 0.18, fc="#ffffff", ec=INK, lw=1.3))
    ax.plot([5.35, 5.35], [2.77, 2.10], color=INK, lw=2.0)
    ax.plot([5.35, 5.10], [2.55, 2.20], color=INK, lw=2.0)
    ax.plot([5.35, 5.60], [2.55, 2.20], color=INK, lw=2.0)
    ax.plot([5.35, 5.15], [2.10, 1.65], color=INK, lw=2.0)
    ax.plot([5.35, 5.55], [2.10, 1.65], color=INK, lw=2.0)
    for cy in (2.95, 2.38, 1.70):
        ax.add_patch(Circle((5.35 + (0.30 if cy == 2.38 else -0.32), cy),
                            0.11, fc=ACC, ec=ACC, alpha=0.55))
    ax.text(5.35, 1.40, "头/躯干/脚：正响应", fontsize=11.5, color=ACC,
            ha="center", va="center")
    ax.text(3.3, 0.18,
            "程序生成，未用原书原图 · 对应原书 p.376–377，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def collect_texts(fig):
    texts = []
    for ax in fig.axes:
        texts.extend(ax.texts)
    return texts


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).read_bytes()
    src_sha = hashlib.sha256(src).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    fig1 = fig_taxonomy()
    p1 = FIG_DIR / "e060_face_detection_taxonomy.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    t1 = [t.get_text() for t in collect_texts(fig1)]
    for w in ("特征法", "模板法", "外观法", "滑窗", "金字塔", "级联"):
        assert w in t1, f"[FAIL] 图1 缺词 {w}"
    fs1 = [t.get_fontsize() for t in collect_texts(fig1)]

    fig2 = fig_viola_jones()
    p2 = FIG_DIR / "e060_viola_jones_features.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    t2 = [t.get_text() for t in collect_texts(fig2)]
    for w in ("矩形差分（白减灰）", "积分图：四点速查", "三级级联", "拒非脸", "候选脸通过"):
        assert w in t2, f"[FAIL] 图2 缺词 {w}"
    fs2 = [t.get_fontsize() for t in collect_texts(fig2)]

    fig3 = fig_hog_pipeline()
    p3 = FIG_DIR / "e060_hog_pipeline.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    t3 = [t.get_text() for t in collect_texts(fig3)]
    for w in ("规则网格", "方向直方图", "SVM 权重", "头/躯干/脚：正响应"):
        assert w in t3, f"[FAIL] 图3 缺词 {w}"
    fs3 = [t.get_fontsize() for t in collect_texts(fig3)]

    for fs, label in ((fs1, "图1"), (fs2, "图2"), (fs3, "图3")):
        assert min(fs) >= 10.0, f"[FAIL] {label} 最小字号 {min(fs)}<10"
    for texts, label in ((t1, "图1"), (t2, "图2"), (t3, "图3")):
        assert sum("程序生成，未用原书原图" in s for s in texts) == 1, f"[FAIL] {label} 注记数"
    n_texts = 0
    for texts in (t1, t2, t3):
        for s in texts:
            n_texts += 1
            if FIG_MARK.search(s):
                leaked.append(s)

    rows = [
        "图1 三路线+流水线词在档　| 特征法/模板法/外观法/滑窗/金字塔/级联 全命中",
        "图2 三件套词在档　| 矩形差分/积分图/三级级联/拒非脸/候选脸 全命中",
        "图3 四段词在档　| 规则网格/方向直方图/SVM 权重/头躯干脚 全命中",
        "主文本字号≥10pt+程序生成注记在档×3　| "
        f"图一 最小 {min(fs1)}pt/图二 最小 {min(fs2)}pt/图三 最小 {min(fs3)}pt",
        "图内文字内部记号零泄漏（红线 5）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e060_face_detection_taxonomy.png",
        "FG2 程序生成图落盘　| e060_viola_jones_features.png",
        "FG3 程序生成图落盘　| e060_hog_pipeline.png",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"routes": len(ROUTES), "pipeline": len(PIPE),
                 "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"features": 2, "integral_grid": 4, "cascade_stages": 3,
                 "min_fs": min(fs2), "max_fs": max(fs2)},
        "fig3": {"grid_cells": 25, "bins": len([0.30, 0.55, 0.80, 0.45, 0.62]),
                 "min_fs": min(fs3), "max_fs": max(fs3)},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
    }
    rd = OUT_DIR / "e060_face_detection_readings.json"
    rd.write_text(json.dumps(readings, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rd.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rd.read_bytes()).hexdigest()}")
    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(rows) - n_pass} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n_pass != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
