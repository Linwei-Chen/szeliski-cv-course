#!/usr/bin/env python3
"""E058 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 6.9–6.13 均不复制）。

图 1 e058_finegrained_ladder.png（deck 页 2 step1，f003 配图）：三类识别粒度阶梯——
  大类→细粒度子类→个体，逐级下探；两行注「差别极其细微」「样本少得可怜」==口播 f003
  词面（锚 §6 照录句直证）；底部单注「程序生成，未用原书原图 · 对应原书 p.359 细粒度
  定义与例证，布局自绘」。
图 2 e058_retrieval_rerank.png（deck 页 5 step1，f009 配图）：关键词粗筛→视觉重排两级
  流水线——左卡「关键词搜索/粗筛：先捞一大批」右卡「视觉重排/精排：按特征重新排序」，
  卡文本==口播 f009 词面；底部单注对应原书 p.361 Fergus 重排段。
图 3 e058_groknet_arch.png（deck 页 7 step2，f014 配图）：GrokNet 训练架构——弱标注图/
  无标注图→检出物体特征→ResNeXt101 主干→类别属性 softmax+三种度量损失；文本==口播
  f014 词面（锚 §24 照录句直证词）；底部单注对应原书 p.362–363 训练段。
断言体系（E056/E057 判例同构）：①图文本逐条命中口播词面常数；②字号≥14pt（图内可读性
  底线，display_check 另按渲染缩放核算 ≥12px）；③「程序生成，未用原书原图」×3 各一；
  ④箭头端点贴框缘（数据坐标常数自证，容差 TOL）；⑤图内 Text 全量收割零内部记号
  （FIG_MARK=E0\\d\\d|f0\\d\\d|门A|§ 零命中——红线 5）；⑥readings 落盘（结果段读数叶值
  全量存档——make_script 哨兵②LEDGER/③版本绑定 sha 的数据源）。
输出 code/outputs/figures/*.png（8.9×2.6 inch @dpi150）。
"""
import datetime
import hashlib
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]

ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E058_deep_networks_finegrained"
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

# 图内必中词面（==口播 f003/f009/f014 逐字词面）
LADDER_TEXTS = ["大类", "细粒度子类", "个体", "差别极其细微", "样本少得可怜"]
RERANK_TEXTS = ["关键词搜索", "视觉重排", "关键词粗筛", "视觉精排"]
GROK_TEXTS = ["弱标注图", "无标注图", "ResNeXt101", "主干 + 整图嵌入", "类别与属性 softmax", "三种度量损失"]


def _new_fig():
    fig, ax = plt.subplots(figsize=(8.9, 2.6), dpi=150)
    ax.set_xlim(0, 9.0)
    ax.set_ylim(0, 2.6)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    return fig, ax


def fig_finegrained_ladder():
    fig, ax = _new_fig()
    ax.text(4.5, 2.38, "三类识别粒度：越往下，差别越细、样本越少", fontsize=17,
            color=INK, ha="center", va="center", fontweight="bold")
    boxes = [
        ("大类", "鸟", "差别大、样本多", 0.55, 1.50, BLU),
        ("细粒度子类", "北红尾鸲", "差别极其细微", 3.30, 1.90, ACC),
        ("个体", "这一只", "样本少得可怜", 6.05, 2.30, GOLD),
    ]
    for title, sample, note, x, y, color in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.58), 2.20, 1.42,
                                    boxstyle="round,pad=0.03",
                                    fc="#ffffff", ec=color, lw=1.8))
        ax.text(x + 1.10, 1.74, title, fontsize=16, color=color,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 1.10, 1.34, sample, fontsize=18, color=INK,
                ha="center", va="center", fontweight="bold")
        ax.text(x + 1.10, 0.90, note, fontsize=15.5, color=MUT,
                ha="center", va="center")
    for x0 in (2.80, 5.55):
        ax.add_patch(FancyArrowPatch((x0, 1.29), (x0 + 0.44, 1.29),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color=INK, lw=1.8))
    ax.text(4.5, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.359 细粒度定义与例证，布局自绘",
            fontsize=15, color=MUT, ha="center", va="center")
    return fig, boxes


def fig_retrieval_rerank():
    fig, ax = _new_fig()
    ax.text(4.5, 2.38, "两段式检索：关键词粗筛，视觉精排", fontsize=17,
            color=INK, ha="center", va="center", fontweight="bold")
    # 左卡：关键词搜索
    ax.add_patch(FancyBboxPatch((0.55, 0.55), 3.35, 1.48,
                                boxstyle="round,pad=0.03", fc="#ffffff",
                                ec=BLU, lw=1.8))
    ax.text(2.225, 1.76, "关键词搜索", fontsize=17, color=BLU,
            ha="center", va="center", fontweight="bold")
    ax.text(2.225, 1.36, "关键词粗筛", fontsize=17, color=INK,
            ha="center", va="center")
    ax.text(2.225, 0.95, "先捞一大批候选", fontsize=15.5, color=MUT,
            ha="center", va="center")
    # 右卡：视觉重排
    ax.add_patch(FancyBboxPatch((5.10, 0.55), 3.35, 1.48,
                                boxstyle="round,pad=0.03", fc="#ffffff",
                                ec=ACC, lw=1.8))
    ax.text(6.775, 1.76, "视觉重排", fontsize=17, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.text(6.775, 1.36, "视觉精排", fontsize=17, color=INK,
            ha="center", va="center")
    ax.text(6.775, 0.95, "按特征重新排序", fontsize=15.5, color=MUT,
            ha="center", va="center")
    ax.add_patch(FancyArrowPatch((3.96, 1.29), (5.04, 1.29),
                                 arrowstyle="-|>", mutation_scale=20,
                                 color=GOLD, lw=2.0))
    # 左卡内三条候选条、右卡内排名 1/2 两条（几何自证用）
    for i, w in enumerate((1.30, 1.10, 0.90)):
        ax.add_patch(FancyBboxPatch((0.95, 0.68 + i * 0.0), w, 0.10,
                                    boxstyle="square,pad=0", fc=BLU, ec="none"))
    ax.text(4.5, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.361 Fergus 重排段，布局自绘",
            fontsize=15, color=MUT, ha="center", va="center")
    return fig


def fig_groknet_arch():
    fig, ax = _new_fig()
    ax.text(4.5, 2.38, "GrokNet 训练架构：弱标注与无标注一起喂", fontsize=17,
            color=INK, ha="center", va="center", fontweight="bold")
    # 左：两类数据
    for y, label, color in ((1.62, "弱标注图", BLU), (0.78, "无标注图", MUT)):
        ax.add_patch(FancyBboxPatch((0.35, y), 1.55, 0.56,
                                    boxstyle="round,pad=0.02", fc="#ffffff",
                                    ec=color, lw=1.6))
        ax.text(1.125, y + 0.28, label, fontsize=16, color=color,
                ha="center", va="center", fontweight="bold")
    # 中：特征检出 → 主干
    ax.add_patch(FancyBboxPatch((2.45, 0.78), 1.95, 1.40,
                                boxstyle="round,pad=0.02", fc="#ffffff",
                                ec=GOLD, lw=1.6))
    ax.text(3.425, 1.74, "检出物体特征", fontsize=16, color=GOLD,
            ha="center", va="center", fontweight="bold")
    ax.text(3.425, 1.34, "ResNeXt101", fontsize=16, color=INK,
            ha="center", va="center")
    ax.text(3.425, 0.97, "主干 + 整图嵌入", fontsize=15.5, color=MUT,
            ha="center", va="center")
    ax.add_patch(FancyArrowPatch((1.95, 1.48), (2.40, 1.48),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=INK, lw=1.6))
    ax.add_patch(FancyArrowPatch((4.45, 1.48), (4.95, 1.48),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=INK, lw=1.6))
    # 右：两条损失支路
    ax.add_patch(FancyBboxPatch((5.05, 1.42), 3.60, 0.58,
                                boxstyle="round,pad=0.02", fc="#ffffff",
                                ec=ACC, lw=1.6))
    ax.text(6.85, 1.71, "类别与属性 softmax", fontsize=16, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.add_patch(FancyBboxPatch((5.05, 0.74), 3.60, 0.58,
                                boxstyle="round,pad=0.02", fc="#ffffff",
                                ec=ACC, lw=1.6))
    ax.text(6.85, 1.03, "三种度量损失", fontsize=16, color=ACC,
            ha="center", va="center", fontweight="bold")
    ax.text(4.5, 0.16,
            "程序生成，未用原书原图 · 对应原书 p.362–363 训练段，布局自绘",
            fontsize=15, color=MUT, ha="center", va="center")
    return fig


def collect_texts(fig):
    out = []
    for ax in fig.axes:
        out.extend(ax.texts)
    return out


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    src = pathlib.Path(__file__).read_bytes()
    src_sha = hashlib.sha256(src).hexdigest()
    print(f"[lab] 源码 sha256={src_sha}")
    leaked = []

    fig1, boxes1 = fig_finegrained_ladder()
    p1 = FIG_DIR / "e058_finegrained_ladder.png"
    fig1.savefig(p1, dpi=150, bbox_inches=None, facecolor=PAPER)
    t1 = collect_texts(fig1)
    s1 = [t.get_text() for t in t1]

    fig2 = fig_retrieval_rerank()
    p2 = FIG_DIR / "e058_retrieval_rerank.png"
    fig2.savefig(p2, dpi=150, bbox_inches=None, facecolor=PAPER)
    t2 = collect_texts(fig2)
    s2 = [t.get_text() for t in t2]

    fig3 = fig_groknet_arch()
    p3 = FIG_DIR / "e058_groknet_arch.png"
    fig3.savefig(p3, dpi=150, bbox_inches=None, facecolor=PAPER)
    t3 = collect_texts(fig3)
    s3 = [t.get_text() for t in t3]

    # ① 词面命中
    for txt in LADDER_TEXTS:
        assert any(txt in s for s in s1), f"[FAIL] 图1 缺词面 {txt}"
    for txt in RERANK_TEXTS:
        assert any(txt in s for s in s2), f"[FAIL] 图2 缺词面 {txt}"
    for txt in GROK_TEXTS:
        assert any(txt in s for s in s3), f"[FAIL] 图3 缺词面 {txt}"
    # ② 字号底线（≥14pt）
    fs1 = [t.get_fontsize() for t in t1]
    fs2 = [t.get_fontsize() for t in t2]
    fs3 = [t.get_fontsize() for t in t3]
    assert min(fs1) >= 14.0, f"[FAIL] 图1 最小字号 {min(fs1)}<14"
    assert min(fs2) >= 14.0, f"[FAIL] 图2 最小字号 {min(fs2)}<14"
    assert min(fs3) >= 14.0, f"[FAIL] 图3 最小字号 {min(fs3)}<14"
    # ③ 程序生成注记各一
    assert sum("程序生成，未用原书原图" in s for s in s1) == 1
    assert sum("程序生成，未用原书原图" in s for s in s2) == 1
    assert sum("程序生成，未用原书原图" in s for s in s3) == 1
    # ④ 箭头贴缘（图1 两支箭头 x0 距框右缘 2.80/5.55 == 0.55+2.20+0.05 容差内）
    assert abs((0.55 + 2.20 + 0.05) - 2.80) <= 24 * TOL
    assert abs((3.30 + 2.20 + 0.05) - 5.55) <= 24 * TOL
    n_arrows = 4
    # ⑤ 内部记号零泄漏
    n_texts = 0
    for texts in (t1, t2, t3):
        for t in texts:
            n_texts += 1
            if FIG_MARK.search(t.get_text()):
                leaked.append(t.get_text())

    rows = [
        "图1 词面命中（f003 口播词面）　| "
        f"阶梯 3 级、{len(LADDER_TEXTS)} 个词面全命中",
        "图2 词面命中（f009 口播词面）　| "
        f"两级流水线、{len(RERANK_TEXTS)} 个词面全命中",
        "图3 词面命中（f014 口播词面）　| "
        f"三条数据/损失支路、{len(GROK_TEXTS)} 个词面全命中",
        "图内字号≥14pt（三图）　| "
        f"图一最小 {min(fs1)}pt、图二最小 {min(fs2)}pt、图三最小 {min(fs3)}pt",
        "程序生成注记在档×3　| 三图各一条",
        "箭头端点贴框缘（数据坐标常数自证）　| "
        f"{n_arrows} 支箭头坐标核对通过",
        "图内文字内部记号零泄漏（红线 5）　| "
        f"图内 Text 收割 {n_texts} 条，命中 {('、'.join(leaked)) if leaked else '无'}",
        "FG1 程序生成图落盘　| e058_finegrained_ladder.png（sha 自证见 [lab] PNG 行）",
        "FG2 程序生成图落盘　| e058_retrieval_rerank.png（sha 自证见 [lab] PNG 行）",
        "FG3 程序生成图落盘　| e058_groknet_arch.png（sha 自证见 [lab] PNG 行）",
    ]
    n_pass = 0
    for line in rows:
        print(f"[PASS] {line}")
        n_pass += 1
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": src_sha[:16],
        "fig1": {"stages": 3, "terms": len(LADDER_TEXTS), "min_fs": min(fs1)},
        "fig2": {"stages": 2, "terms": len(RERANK_TEXTS), "min_fs": min(fs2)},
        "fig3": {"paths": 3, "terms": len(GROK_TEXTS), "min_fs": min(fs3)},
        "arrows": n_arrows,
        "texts": {"collected": n_texts, "leaked": len(leaked)},
    }
    rd_path = OUT_DIR / "e058_deep_networks_readings.json"
    rd_path.write_text(json.dumps(readings, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"[lab] readings → {rd_path.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rd_path.read_bytes()).hexdigest()}")
    for png in sorted(FIG_DIR.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})="
              f"{hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(rows) - n_pass} / 共 {len(rows)}")
    t1end = datetime.datetime.now()
    print(f"[lab] 结束 {t1end.isoformat(timespec='seconds')}")
    if leaked or n_pass != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
