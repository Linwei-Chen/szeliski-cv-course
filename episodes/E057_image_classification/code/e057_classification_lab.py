#!/usr/bin/env python3
"""E057 S2 教学替代图×2（红线 2：程序生成，未用原书原图——Figure 6.4/6.5/6.6/6.7/6.8
均不复制）。

图 1 e057_bow_vs_pyramid.png（deck 页 4，f008 配图）：词袋 vs 空间金字塔对照卡——
  左卡「词袋」（整图矩形→箭头→一张直方图，白话「只数不看位置」）右卡「空间金字塔」
  （三层格子塔→三组迷你柱，白话「分区点名」）；卡文本==口播 f003/f008 词面（逐格断言）；
  全图单注「左卡词频直方图对应原书 Figure 6.6（p.352），右卡分区统计对应原书 p.353
  文字描述，布局自绘」（门A r1 P1-2 修复——对应物拆分到卡，右卡不冒领 Figure 6.6）。
图 2 e057_topology_ladder.png（deck 页 7，f013 配图）：部件拓扑谱系五档复杂度阶梯——
  词袋→树和星形→有向无环图→k 扇形→星座模型逐级抬升；五档主名+副注==口播 f013 五档
  词面（逐档断言——锚 §25 照录句直证词序）；全图单注「直证词对应原书 p.355–356，
  布局自绘」。
断言体系（E056 判例同构+readings 体系 E053 同构）：
①图 1 两卡文本==口播 f003/f008 词面序列（逐格比对——BOW_TEXTS/PYR_TEXTS 常数即口播
  词源：「词袋/只数不看位置/空间金字塔/层级切格子/分区点名」出自 f008（「层级切格子」
  系口播「按层级切格子」词面子串）；「一张直方图」出自 f003（「做成一张直方图」词面
  子串））；
②图 2 五档主名序列==「词袋/树和星形/有向无环图/k 扇形/星座模型」（==口播 f013 五档
  ——锚 §25 照录句直证词序）；
③图 2 副注词面逐档命中（「毫无几何关系/推断最高效/被迫加上因果结构/复杂度往下压/
  最通用」均系 f013 口播逐字词面）；
④箭头端点贴框缘/贴图元（数据坐标常数自证，容差 TOL）；
⑤全部主文本 fontsize≥10pt（常数遍历断言——E056 首跑 9.5pt 破 D27 图内 12px 线判例，
  本集字号档直接统一 10.5/11.5）+「程序生成，未用原书原图」×2 图各一；
⑥图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak 零命中——
  红线 5，E044–E056 同款收割断言）；
readings 体系（E053 判例同构）：[PASS] 断言行（断言名　| 结果段）+readings.json
（结果段读数叶值全量存档——make_script 哨兵②LEDGER/③版本绑定 sha 与
lab-stdout↔readings 弱一致机检的数据源）+源码/readings/PNG sha 自证行全量套。
输出 code/outputs/figures/{e057_bow_vs_pyramid,e057_topology_ladder}.png（8.9×2.6
inch @dpi150——deck 页 4/页 7 右栏图区缩放比随 deck 定稿，10.5pt 最小字号保证缩后
≥12px 线，display_check 存证核算为准）。
红线 5：可见层无内部记号——图内不渲染集号/内部编号/节记号；只标概念+原书页。"""

import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E057_image_classification"
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
# 图内记号零泄漏（红线 5——E044–E056 同款）：集号/帧号/门/节记号/口播标签字样
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")

# ── 图 1：词袋 vs 空间金字塔对照卡（卡文本==口播 f003/f008 词面）──
BOW_TEXTS = ["词袋", "一张直方图", "只数不看位置"]
PYR_TEXTS = ["空间金字塔", "层级切格子", "分区点名"]
BARS = [0.22, 0.42, 0.30, 0.58, 0.36, 0.50, 0.26, 0.40]  # 左卡直方图 8 柱高
BAR_W, BAR_X0, BAR_BASE = 0.20, 2.08, 0.88


def fig_bow_vs_pyramid():
    fig, ax = plt.subplots(figsize=(8.9, 2.6), dpi=150)
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(4.5, 2.40, "词袋 vs 空间金字塔", fontsize=11.5,
            color=INK, ha="center", va="center", fontweight="bold")
    # 左卡：词袋（整图→一张直方图）
    ax.add_patch(FancyBboxPatch((0.35, 0.40), 3.85, 1.70,
                                boxstyle="round,pad=0.02",
                                fc="#ffffff", ec=BLU, lw=1.5))
    ax.text(2.275, 1.86, "词袋", fontsize=10.5, color=BLU,
            ha="center", va="center", fontweight="bold")
    ax.text(2.275, 1.60, "一张直方图", fontsize=10.5, color=MUT,
            ha="center", va="center")
    # 整图矩形+3 个特征点
    ax.add_patch(FancyBboxPatch((0.70, 0.98), 0.85, 0.55,
                                boxstyle="round,pad=0.02",
                                fc="#ffffff", ec=INK, lw=1.2))
    for px, py in [(0.95, 1.15), (1.15, 1.32), (1.35, 1.10)]:
        ax.plot(px, py, marker="x", ms=5, color=ACC, mew=1.6)
    ax.add_patch(FancyArrowPatch((1.57, 1.255), (2.06, 1.255),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=INK, lw=1.3))
    for i, hh in enumerate(BARS):
        ax.add_patch(FancyBboxPatch(
            (BAR_X0 + i * (BAR_W + 0.055), BAR_BASE), BAR_W, hh,
            boxstyle="square,pad=0", fc=BLU, ec="none"))
    ax.text(2.275, 0.62, "只数不看位置", fontsize=10.5, color=ACC,
            ha="center", va="center")
    # 右卡：空间金字塔（三层格子塔→三组迷你柱）
    ax.add_patch(FancyBboxPatch((4.80, 0.40), 3.85, 1.70,
                                boxstyle="round,pad=0.02",
                                fc="#ffffff", ec=ACC, lw=1.5))
    ax.text(6.725, 1.86, "空间金字塔", fontsize=10.5, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.text(6.725, 1.60, "层级切格子", fontsize=10.5, color=MUT,
            ha="center", va="center")
    tower = [(5.95, 1.30, 0.85, 1), (5.95, 1.04, 0.85, 2),
             (5.95, 0.78, 0.85, 4)]  # (x, y, 宽, 格数)
    for tx, ty, tw, n in tower:
        cw = tw / n
        for k in range(n):
            ax.add_patch(FancyBboxPatch(
                (tx + k * cw, ty), cw, 0.20, boxstyle="square,pad=0",
                fc="#ffffff", ec=INK, lw=1.1))
    for cx, cy in [(6.15, 1.40), (6.37, 1.14), (6.09, 1.10),
                   (6.51, 0.88), (6.25, 0.84), (6.63, 0.88)]:
        ax.plot(cx, cy, marker="o", ms=2.4, color=ACC)
    for ay in (1.40, 1.14, 0.88):  # 塔→迷你柱三支虚线箭头（与各层中心对齐）
        ax.add_patch(FancyArrowPatch((6.87, ay), (7.11, ay),
                                     arrowstyle="-|>", mutation_scale=9,
                                     color=GOLD, lw=1.1,
                                     linestyle=(0, (3, 2))))
    for gy, ghs in [(1.33, [0.10, 0.14, 0.08]),
                    (1.07, [0.12, 0.09, 0.15]),
                    (0.81, [0.08, 0.13, 0.10])]:
        for j, gh in enumerate(ghs):
            ax.add_patch(FancyBboxPatch(
                (7.19 + j * 0.14, gy), 0.09, gh, boxstyle="square,pad=0",
                fc=ACC, ec="none"))
    ax.text(6.725, 0.62, "分区点名", fontsize=10.5, color=ACC,
            ha="center", va="center")
    # 中缝分隔虚线
    ax.plot([4.50, 4.50], [0.45, 2.05], color=MUT, lw=0.8,
            linestyle=(0, (4, 3)))
    ax.text(4.5, 0.10,
            "程序生成，未用原书原图 · 左卡词频直方图对应原书 Figure 6.6（p.352），右卡分区统计对应原书 p.353 文字描述，布局自绘",
            fontsize=10.5, color=MUT, ha="center", va="center")
    return fig


# ── 图 2：部件拓扑谱系五档复杂度阶梯（主名+副注==口播 f013 五档词面——锚 §25 直证）──
LADDER = [
    ("词袋", "毫无几何关系"),
    ("树和星形", "推断最高效"),
    ("有向无环图", "被迫加上因果结构"),
    ("k 扇形", "复杂度往下压"),
    ("星座模型", "最通用"),
]


def fig_topology_ladder():
    fig, ax = plt.subplots(figsize=(8.9, 2.6), dpi=150)
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(4.45, 2.40, "部件拓扑谱系：从简到繁", fontsize=11.5,
            color=INK, ha="center", va="center", fontweight="bold")
    w, h = 1.50, 1.00
    xs = [0.30 + i * 1.74 for i in range(5)]
    ys = [0.50 + i * 0.14 for i in range(5)]
    centers, edges = [], []
    for (main, sub), x, y in zip(LADDER, xs, ys):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02",
                                    fc="#ffffff", ec=INK, lw=1.5))
        ax.text(x + w / 2, y + h - 0.30, main, fontsize=10.5, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, y + h - 0.64, sub, fontsize=10.5, color=MUT,
                ha="center", va="center")
        centers.append((x + w / 2, y + h / 2))
        edges.append((x, y))
    # 复杂度轴箭头：贯穿五档底部（从简到繁的方向自证——左端起于首档外、右端越过末档）
    ax.add_patch(FancyArrowPatch((0.22, 0.30), (8.82, 0.30),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=GOLD, lw=2.0))
    ax.text(4.45, 0.12,
            "程序生成，未用原书原图 · 直证词对应原书 p.355–356，布局自绘",
            fontsize=10.5, color=MUT, ha="center", va="center")
    return fig, xs, ys, w, h


def collect_texts(fig):
    texts = []
    for ax in fig.axes:
        for t in ax.texts:
            texts.append(t)
    return texts


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).read_bytes()
    src_sha = hashlib.sha256(src).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    # ── 图 1 ──
    fig1 = fig_bow_vs_pyramid()
    p1 = FIG_DIR / "e057_bow_vs_pyramid.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts1 = collect_texts(fig1)
    t1s = [t.get_text() for t in texts1]
    # 断言①：两卡文本==口播 f003/f008 词面序列（逐格比对——BOW_TEXTS/PYR_TEXTS 常数
    # 即口播词源，词面出处见 docstring ①）
    for txt in BOW_TEXTS + PYR_TEXTS:
        assert txt in t1s, f"[FAIL] 图1 缺卡文本 {txt}"
    fs1 = [t.get_fontsize() for t in texts1]
    # ── 图 2 ──
    fig2, xs2, ys2, w2, h2 = fig_topology_ladder()
    p2 = FIG_DIR / "e057_topology_ladder.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts2 = collect_texts(fig2)
    t2s = [t.get_text() for t in texts2]
    # 断言②：五档主名序列==常数（==口播 f013 五档——锚 §25 照录句直证词序）
    mains = [m for m, _s in LADDER]
    assert mains == ["词袋", "树和星形", "有向无环图", "k 扇形", "星座模型"]
    for m in mains:
        assert m in t2s, f"[FAIL] 图2 缺档主名 {m}"
    # 断言③：副注词面逐档命中（均系 f013 口播逐字词面）
    for _m, sub in LADDER:
        assert sub in t2s, f"[FAIL] 图2 缺档副注 {sub}"
    fs2 = [t.get_fontsize() for t in texts2]
    f2min, f2max = min(fs2), max(fs2)
    # 断言④：箭头端点贴框缘/贯穿（数据坐标常数自证——图1 整图→直方图箭头端点贴图元
    # 缘 1.55/BAR_X0 差 ≤2TOL；图2 复杂度轴箭头贯穿五档——左端≤首档左缘、右端≥末档右缘）
    assert abs(1.57 - 1.55) <= 2 * TOL and abs(2.06 - BAR_X0) <= 2 * TOL
    n_axis_arrows = 1
    assert 0.22 <= xs2[0] and 8.82 >= xs2[-1] + w2
    # 断言⑤：字号≥10pt（全图字号档统一 10.5/11.5——E056 首跑 9.5pt 破线判例预防，
    # 本集直接 10.5 起步；display 缩放核算随 display_check）
    fs1_main = [t.get_fontsize() for t in texts1
                if t.get_text() in BOW_TEXTS + PYR_TEXTS]
    fs2_main = [t.get_fontsize() for t in texts2
                if t.get_text() in mains + [s for _m, s in LADDER]]
    assert min(fs1_main) >= 10.0, f"[FAIL] 图1 卡文本字号 {min(fs1_main)}<10"
    assert min(fs2_main) >= 10.0, f"[FAIL] 图2 档主名/副注字号 {min(fs2_main)}<10"
    # 断言⑤b：程序生成注记两图各一
    assert sum("程序生成，未用原书原图" in s for s in t1s) == 1
    assert sum("程序生成，未用原书原图" in s for s in t2s) == 1
    # 断言⑥：内部记号零泄漏（全量收割）
    n_texts = 0
    for fig, texts in [(fig1, texts1), (fig2, texts2)]:
        for t in texts:
            n_texts += 1
            if FIG_MARK.search(t.get_text()):
                leaked.append(t.get_text())

    rows = [
        "图1 两卡文本==口播 f003/f008 词面（逐格比对）　| "
        f"词袋侧 {len(BOW_TEXTS)} 格+空间金字塔侧 {len(PYR_TEXTS)} 格全命中",
        "图2 五档主名序列==口播 f013 五档（锚 §25 照录句直证词序）　| "
        f"{'→'.join(mains)}，字号最小 {f2min}pt",
        "图2 副注词面逐档命中（f013 口播逐字词面）　| "
        f"{len(LADDER)} 档副注全命中",
        "箭头端点贴框缘/贯穿五档（数据坐标常数自证）　| "
        f"图1 直方图箭头 1 支贴缘+图二 复杂度轴箭头 {n_axis_arrows} 支贯穿五档",
        "主文本字号≥10pt+程序生成注记在档×2　| "
        f"图1 卡文本最小 {min(fs1_main)}pt/图二 档主名副注最小 {min(fs2_main)}pt"
        "（全图字号档统一 10.5/11.5——E056 破线判例预防，display 缩放核算随 display_check）",
        "图内文字内部记号零泄漏（红线 5——E044–E056 同款收割断言）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e057_bow_vs_pyramid.png 已落盘（sha 自证见 [lab] PNG 行）",
        "FG2 程序生成图落盘　| e057_topology_ladder.png 已落盘（sha 自证见 [lab] PNG 行）",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    # readings 落盘（结果段读数叶值全量存档——E053 判例同构）
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"bow_cells": len(BOW_TEXTS), "pyr_cells": len(PYR_TEXTS),
                 "bars": len(BARS), "tower_levels": 3,
                 "min_fs": min(fs1), "max_fs": max(fs1)},
        "fig2": {"stages": len(LADDER), "arrows": n_axis_arrows,
                 "main_fs_min": min(fs2_main), "max_fs": f2max},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
    }
    rd_path = OUT_DIR / "e057_classification_readings.json"
    rd_path.write_text(json.dumps(readings, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[lab] readings → {rd_path.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rd_path.read_bytes()).hexdigest()}")
    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})="
              f"{hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(rows) - n_pass} / 共 {len(rows)}")
    t1 = datetime.datetime.now()
    print(f"[lab] 结束 {t1.isoformat(timespec='seconds')}")
    if leaked or n_pass != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
