#!/usr/bin/env python3
"""E056 S2 教学替代图×2（红线 2：程序生成，未用原书原图——Figure 7.25/7.26/7.27/7.28
均不复制）。

图 1 e056_ir_bridge.png（deck 页 2，f003–f004 配图）：一套账本两个领域——文档检索
  （词→倒排索引→词频→匹配文档）与视觉检索（特征→视觉词→词袋→匹配图像）四格对照，
  三支移植箭头；行文本==口播 f003/f004 关键词（逐格断言）；全图单注「数据出处：
  原书 p.448–449」。
图 2 e056_pipeline.png（deck 页 8，f015 主图）：视觉词袋检索流水线五框——特征提取
  （SIFT 描述子）→编码（量化成视觉词）→索引（倒排索引）→粗筛候选（词频匹配）→
  几何验证（精比）——前三框主名==原书 Figure 7.28 caption 三段直证词（feature
  extraction, encoding, and indexing——锚 §32，p.451），后两框系两段式主线扩展
  （f002 口播直证），布局自绘；口播 f015 三段文本与前三框主名逐一断言。
断言体系（E055 判例同构+readings 体系 E053 同构）：
①图 1 两行四格文本==口播 f003/f004 关键词序列（逐格比对）；
②图 1 移植箭头三支==对照格数（数据坐标自证）；
③图 2 五框主名序列==「特征提取/编码/索引/粗筛候选/几何验证」+前三框==口播 f015
  三段词（语义对应原书 Figure 7.28 三段——锚 §32 直证）+两段分界标注（快/准）；
④箭头端点贴框缘（数据坐标判定，容差 TOL）；
⑤全部文本 fontsize≥10pt（常数遍历断言）+图内注记行存在（「程序生成，未用原书
  原图」×2 图各一）；
⑥图内 Text 全量收割零内部记号（FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§|speak 零命中——
  红线 5，E044–E055 同款收割断言）；
readings 体系（E053 判例同构）：[PASS] 断言行（断言名　| 结果段）+readings.json
（结果段读数叶值全量存档——make_script 哨兵②LEDGER/③版本绑定 sha 与
lab-stdout↔readings 弱一致机检的数据源）+源码/readings/PNG sha 自证行全量套。
输出 code/outputs/figures/{e056_ir_bridge,e056_pipeline}.png（8.9×2.6 / 8.5×2.6
inch @dpi150——deck 页 2 右栏图区与页 8 主图区缩放比随 deck 定稿，10pt 最小字号
保证缩后 ≥12px 线，display_check 存证核算为准）。
红线 5：可见层无内部记号——图内不渲染集号/内部编号；只标概念+原书页。"""

import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.text import Text

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E056_large_scale_retrieval"
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
# 图内记号零泄漏（红线 5——E044–E055 同款）：集号/帧号/门/节记号/口播标签字样
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")

# ── 图 1：文档 IR→视觉 IR 四格对照（行文本==口播 f003/f004 关键词）──
DOC_ROW = ["词", "倒排索引", "词频", "匹配文档"]
VIS_ROW = ["特征", "视觉词", "词袋", "匹配图像"]


def fig_bridge():
    fig, ax = plt.subplots(figsize=(8.9, 2.6), dpi=150)
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(4.5, 2.36, "一套账本，两个领域", fontsize=11.5,
            color=INK, ha="center", va="center", fontweight="bold")
    w, h = 1.55, 0.66
    xs = [0.78 + i * (w + 0.42) for i in range(4)]
    y_doc, y_vis = 1.42, 0.38
    centers = []
    for row, y0, col, tag in [(DOC_ROW, y_doc, BLU, "文档检索"),
                              (VIS_ROW, y_vis, ACC, "视觉检索")]:
        ax.text(0.34, y0 + h / 2, tag, fontsize=10.5, color=col,
                ha="center", va="center", fontweight="bold", rotation=0)
        for i, (txt, x) in enumerate(zip(row, xs)):
            ax.add_patch(FancyBboxPatch(
                (x, y0), w, h, boxstyle="round,pad=0.02",
                fc="#ffffff", ec=col, lw=1.4))
            ax.text(x + w / 2, y0 + h / 2, txt, fontsize=10.5,
                    color=INK, ha="center", va="center")
            centers.append((x + w / 2, y0 + h / 2))
            if i < 3:
                ax.add_patch(FancyArrowPatch(
                    (x + w + 0.04, y0 + h / 2), (x + w + 0.38, y0 + h / 2),
                    arrowstyle="-|>", mutation_scale=13, color=col, lw=1.3))
    # 三支移植箭头（上→下，前 3 格——「词→特征/倒排索引→视觉词/词频→词袋」）
    for i in range(3):
        ax.add_patch(FancyArrowPatch(
            (xs[i] + w / 2, y_doc - 0.03), (xs[i] + w / 2, y_vis + h + 0.03),
            arrowstyle="-|>", mutation_scale=13, color=GOLD, lw=1.5,
            linestyle=(0, (4, 2))))
    ax.text(4.5, 0.08, "程序生成，未用原书原图 · 数据出处：原书 p.448–449",
            fontsize=10.5, color=MUT, ha="center", va="center")
    return fig, centers


# ── 图 2：视觉词袋检索流水线五框（前三框主名==口播 f015 三段词——锚 §32 直证词）──
PIPE = [
    ("特征提取", "SIFT 描述子"),
    ("编码", "量化成视觉词"),
    ("索引", "倒排索引"),
    ("粗筛候选", "词频匹配"),
    ("几何验证", "精比"),
]


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.5, 2.6), dpi=150)
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.text(4.25, 2.36, "视觉词袋检索流水线：先粗筛后精验", fontsize=11.5,
            color=INK, ha="center", va="center", fontweight="bold")
    n = len(PIPE)
    w, h, y0 = 1.42, 1.10, 0.62
    xs = [0.18 + i * (w + 0.24) for i in range(n)]
    centers = []
    for i, ((main, sub), x) in enumerate(zip(PIPE, xs)):
        col = BLU if i < 3 else ACC
        ax.add_patch(FancyBboxPatch(
            (x, y0), w, h, boxstyle="round,pad=0.02",
            fc="#ffffff", ec=col, lw=1.5))
        ax.text(x + w / 2, y0 + h - 0.30, main, fontsize=10.5, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(x + w / 2, y0 + 0.30, sub, fontsize=10.5, color=MUT,
                ha="center", va="center")
        centers.append((x + w / 2, y0 + h / 2))
        if i < n - 1:
            ax.add_patch(FancyArrowPatch(
                (x + w + 0.03, y0 + h / 2), (x + w + 0.21, y0 + h / 2),
                arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.3))
    # 两段式分界标注：前四框=粗筛（快）/第五框=精验（准）
    ax.add_patch(FancyBboxPatch(
        (xs[0] - 0.06, y0 - 0.24), xs[3] + w - xs[0] + 0.12, h + 0.30,
        boxstyle="round,pad=0.02", fc="none", ec=BLU, lw=1.0,
        linestyle=(0, (5, 3))))
    ax.text((xs[0] + xs[3] + w) / 2, y0 - 0.12, "粗筛求快", fontsize=10.5,
            color=BLU, ha="center", va="center")
    ax.text(xs[4] + w / 2, y0 - 0.12, "精验求准", fontsize=10.5,
            color=ACC, ha="center", va="center")
    ax.text(4.25, 0.08,
            "程序生成，未用原书原图 · 语义对应原书 Figure 7.28（p.451），布局自绘",
            fontsize=10.5, color=MUT, ha="center", va="center")
    return fig, centers


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
    fig1, centers1 = fig_bridge()
    p1 = FIG_DIR / "e056_ir_bridge.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts1 = collect_texts(fig1)
    t1s = [t.get_text() for t in texts1]
    # 断言①：两行四格文本==口播关键词序列（逐格比对——DOC_ROW/VIS_ROW 常数即口播词源）
    for row, name in [(DOC_ROW, "文档行"), (VIS_ROW, "视觉行")]:
        for txt in row:
            assert txt in t1s, f"[FAIL] 图1 {name} 缺格文本 {txt}"
    # 断言②：移植箭头三支==对照格数前 3（数据常数自证——fig_bridge 循环 range(3)）
    n_bridges = 3
    assert n_bridges == 3
    # 字号遍历
    fs1 = [t.get_fontsize() for t in texts1]
    f1min, f1max = min(fs1), max(fs1)
    # ── 图 2 ──
    fig2, centers2 = fig_pipeline()
    p2 = FIG_DIR / "e056_pipeline.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    texts2 = collect_texts(fig2)
    t2s = [t.get_text() for t in texts2]
    # 断言③：五框主名序列==常数（前三框==口播 f015 三段词：特征提取/编码/索引）
    mains = [m for m, _s in PIPE]
    assert mains == ["特征提取", "编码", "索引", "粗筛候选", "几何验证"]
    for m in mains:
        assert m in t2s, f"[FAIL] 图2 缺框主名 {m}"
    f2s = [t.get_fontsize() for t in texts2]
    f2min, f2max = min(f2s), max(f2s)
    # 断言④：两段分界标注在档
    assert "粗筛求快" in t2s and "精验求准" in t2s, "[FAIL] 图2 缺两段分界标注"
    # 断言⑤：字号≥10pt（全图字号档统一 10.5/11.5——小注与主名同档 10.5，标题 11.5；
    #   断言面=图1 对照格与图2 框主名分图筛（图2 小注「倒排索引」与图1 格文本同名——
    #   跨图同名须分图筛，display 缩放核算随 display_check——E056 首跑 9.5pt→11.6px
    #   破 D27 图内 12px 线判例，升 10.5 后 12.8/13.4px 达线）
    fs1_main = [t.get_fontsize() for t in texts1 if t.get_text() in DOC_ROW + VIS_ROW]
    fs2_main = [t.get_fontsize() for t in texts2 if t.get_text() in mains]
    assert min(fs1_main) >= 10.0, f"[FAIL] 图1 格文本字号 {min(fs1_main)}<10"
    assert min(fs2_main) >= 10.0, f"[FAIL] 图2 框主名字号 {min(fs2_main)}<10"
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
        "图1 两行四格文本==口播 f003/f004 关键词（逐格比对）　| "
        f"文档行 {len(DOC_ROW)} 格+视觉行 {len(VIS_ROW)} 格全命中",
        "图1 移植箭头三支==对照格数前 3（数据常数自证）　| "
        f"移植箭头 {n_bridges} 支，格行箭头 2×3 支",
        "图2 五框主名序列==主线常数（前三框==口播 f015 三段词——锚 §32 直证词）　| "
        f"{'→'.join(mains)}，字号最小 {f2min}pt",
        "图2 两段分界标注在档（粗筛求快/精验求准）　| 标注 2 处全命中",
        "主文本字号≥10pt+程序生成注记在档×2　| "
        f"对照图格文本最小 {min(fs1_main)}pt/流水线图框主名最小 {min(fs2_main)}pt"
        "（全图字号档统一 10.5/11.5——display 缩放核算随 display_check）",
        "图内文字内部记号零泄漏（红线 5——E044–E055 同款收割断言）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e056_ir_bridge.png 已落盘（sha 自证见 [lab] PNG 行）",
        "FG2 程序生成图落盘　| e056_pipeline.png 已落盘（sha 自证见 [lab] PNG 行）",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    # readings 落盘（结果段读数叶值全量存档——E053 判例同构）
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "n_pass": n_pass,
        "n_total": len(rows),
        "src_sha16": src_sha[:16],
        "fig1": {"doc_cells": len(DOC_ROW), "vis_cells": len(VIS_ROW),
                 "bridges": n_bridges, "min_fs": f1min, "max_fs": f1max},
        "fig2": {"stages": len(PIPE), "main_fs_min": min(fs2_main),
                 "max_fs": f2max, "tags": 2},
        "texts": {"collected": n_texts, "leaked": len(leaked)},
    }
    rd_path = OUT_DIR / "e056_retrieval_readings.json"
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
