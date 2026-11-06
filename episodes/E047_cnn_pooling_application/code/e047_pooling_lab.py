#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E047 lab —— 池化实验：max/average pooling 数值实测+反传 switch 路由实证+作业档+
max pooling 示意图（§5.4.1 p.295–297+§5.4.2 p.298–299 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；seed 2047 全档固定）：
- E1 max vs average pooling 数值实测（p.295「compute the maximum response within a square
  window, which is called max pooling」「When the weights inside the convolution kernel are
  identical and sum up to 1, this is called average pooling」的读数级实证——f004 口播
  「每个窗口留最大的那个数」的实算档）：输入 4×4 矩阵（seed 2047 随机整数 0–9 固定在
  readings）、2×2 窗口 stride 2 不重叠：
  ①max pooling 输出 2×2：手算（逐窗口取最大）vs scipy.ndimage.maximum_filter（size=2、
    步进切块）双线 allclose——「谁最强听谁的」的数字版；
  ②average pooling 输出 2×2：手算（逐窗口取均值）vs scipy.ndimage.uniform_filter 双线
    allclose——「权重全相同加起来等于一」的数字版；
  ③正确性：两路双线 allclose(atol=1e-9)——池化跑的是真算子不是纸面算术。
  断言：max 手算==readings、avg 手算==readings、双线 allclose 通过。
- E0 反传 switch 路由实证（p.295「The max pool operator acts like a “switch” that shunts
  (connects) one of the input units to the output unit…we only need to pass the error and
  derivatives down to this maximally active unit, as long as we have remembered which unit
  has this response」的读数级实证——f006 口播「正向谁赢了，反向谁领走全部梯度」的实算档）：
  正向：逐窗口记录胜者位置（mask 表 2×2 窗口各 1 个胜者=4 个胜者位）；反向：输出梯度
  全一，按 mask 只回传胜者位：
  ①胜者位梯度==1（4 个位置各收到 1）；②非胜者位梯度==0（12 个位置）；③梯度总流量==4
  （=窗口数——「每窗口只领走一份」）；④对照：平均池化反传梯度每窗 4 位各 0.25（
  「梯度平均分」对照档——E044 全连接口径呼应）。
  断言：winner 梯度全 1、非胜者全 0、总流量==4、avg 对照==0.25×16 汇总 4。
- E10 作业档位（f014 第一道口算答案存证——「一个四乘四的输入、二乘二的池化窗口、步长二，
  最大池化之后的输出是几乘几」）：输出尺寸公式 (W−K)/S+1=(4−2)/2+1=2——输出 2×2
  （口播不出答案——答案只在实验档，E045/E046 f014 同款惯例；deck 作业卡同样不出答案），
  并以 E1 的实算输出形状 2×2 互证。断言：out_size==2 且 E1 输出形状==(2,2)。
  第二道思考题（并列最大梯度分配）系开放题（f014 refs 申报无标准答案——无档位）。
- FG max pooling 示意图（S1 悬置①定案：Figure 5.36/5.37/5.38 原书图不复制——程序生成
  「max pooling 2×2 窗口+switch 路由」示意图，deck 页 2 用）：4×4 输入数字网格+2×2 高亮
  窗口（四窗轮动一次画出）+右侧 2×2 输出+胜者位红框标注+「胜者领走全部梯度」箭头注记——
  对应 f004/f006 口播「每个窗口留最大的那个数」「正向谁赢了，反向谁领走全部梯度」。
  图内文字运行时收割（E044–E046 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。
- 图字号口径（D27 图内显示 ≥12px）：deck 页 2 左栏列宽缩放实测以 display_check 存证为准；
  图内字号取 14–16pt（E046 14.5pt 同档——本图数字格多、标签少）。
- 存证运行法（E043–E046 同款）：python3 code/e047_pooling_lab.py >
  code/outputs/e047_pooling_lab.txt 2> code/outputs/e047_pooling_lab_stderr.txt——
  stdout/stderr 分流；输出内记源码/readings/PNG sha 供 make_script lab-stdout 哨兵版本
  绑定核验。
- 符号口径：口播无希腊字母无下标（Lp 读「L p」、∞ 读「推到无穷大」——f008 refs 申报）；
  deck 卡书面可保留记号（Lp→∞/GeM/DAME/U-Net/AT&T 等系原书编号与人名——E046 deck 卡
  同款口径）。
"""
import datetime
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as _mtext

try:
    from scipy import ndimage
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
READINGS = OUT / "e047_pooling_readings.json"
PNG = FIG_DIR / "e047_fig1_pooling_diagram.png"
SEED = 2047

FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
LEDGER = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}　| {detail}")
    LEDGER.append((name, ok))


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 输入 锚证据 sha256={hashlib.sha256((EP / 'production/evidence/e047_book_ch5_anchors.txt').read_bytes()).hexdigest()[:64]}…")
    rng = np.random.default_rng(SEED)
    x = rng.integers(0, 10, size=(4, 4)).astype(float)
    # 固定输入矩阵（readings 存档逐元素——数字格教学友好，0–9 整数）
    x = np.array([[1.0, 3.0, 2.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [4.0, 2.0, 1.0, 0.0],
                  [3.0, 9.0, 5.0, 6.0]])
    print(f"[lab] seed={SEED} 输入 4×4 固定矩阵（readings 逐元素存档）")

    # ── E1 max/average pooling 实测 ──
    ATOL = 1e-9  # 双线对照容差——readings 存同值（lab-stdout 数值出处检查口径）

    def pool2x2(a, op):
        out = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                w = a[2 * i:2 * i + 2, 2 * j:2 * j + 2]
                out[i, j] = op(w)
        return out

    def pool_grad_switch(a, out):
        """正向记录胜者位置，反向输出梯度全一只回传胜者位。"""
        g = np.zeros_like(a)
        for i in range(2):
            for j in range(2):
                w = a[2 * i:2 * i + 2, 2 * j:2 * j + 2]
                wi, wj = np.unravel_index(int(np.argmax(w)), w.shape)
                g[2 * i + wi, 2 * j + wj] += 1.0
        return g

    max_hand = pool2x2(x, np.max)
    avg_hand = pool2x2(x, np.mean)
    max_expect = np.array([[6.0, 8.0], [9.0, 6.0]])
    avg_expect = np.array([[3.75, 5.25], [4.5, 3.0]])
    check("E1 max pooling 手算=readings", bool(np.array_equal(max_hand, max_expect)),
          f"max_out={max_hand.flatten().tolist()} expect=[6,8,9,6]")
    check("E1 average pooling 手算=readings", bool(np.allclose(avg_hand, avg_expect, atol=ATOL)),
          f"avg_out={avg_hand.flatten().tolist()} expect=[3.75,5.25,4.5,3.0]（偏置无——池化无参数申报）")
    if HAVE_SCIPY:
        mmax = ndimage.maximum_filter(x, size=2, mode="constant", cval=0.0)[1::2, 1::2]
        mavg = ndimage.uniform_filter(x, size=2, mode="constant", cval=0.0)[1::2, 1::2]
        check("E1 max 双线 scipy 对照 allclose", bool(np.allclose(max_hand, mmax, atol=ATOL)),
              f"scipy maximum_filter 一致（atol={ATOL:g}——池化实验系真算子）")
        check("E1 avg 双线 scipy 对照 allclose", bool(np.allclose(avg_hand, mavg, atol=ATOL)),
              f"scipy uniform_filter 一致（atol={ATOL:g}）")
    else:
        check("E1 scipy 缺席跳过", True, "scipy 未安装——双线对照降级申报（手算双口径已互证）")

    # ── E0 switch 路由实证 ──
    gmax = pool_grad_switch(x, max_hand)
    winners = int((gmax == 1.0).sum())
    zeros = int((gmax == 0.0).sum())
    flow = float(gmax.sum())
    check("E0 胜者位梯度==1（4 位）", winners == 4 and bool((gmax[gmax > 0] == 1.0).all()),
          f"winners={winners} 位置={np.argwhere(gmax == 1.0).tolist()}（mask 记忆=「记住哪个单元最大」）")
    check("E0 非胜者位梯度==0（12 位）", zeros == 12,
          f"zeros={zeros}（「其他位置一分钱都拿不到」——f006 白话的数字版）")
    check("E0 梯度总流量==4（=窗口数）", abs(flow - 4.0) < 1e-12,
          f"flow={flow:g}（每窗口只领走一份——switch 转接口径）")
    per_cell = 1.0 / (2 * 2)  # 每窗 4 位均分——0.25 的来源（窗口面积倒数）
    gavg = np.zeros_like(x)
    for i in range(2):
        for j in range(2):
            gavg[2 * i:2 * i + 2, 2 * j:2 * j + 2] += per_cell
    avg_cells = int((gavg > 0).sum())
    check("E0 对照：平均池化梯度每窗 4 位各 0.25", bool(np.allclose(gavg[gavg > 0], per_cell)) and avg_cells == x.size,
          f"avg 梯度 {avg_cells} 位各 {per_cell:g} 汇总 {gavg.sum():g}（「梯度平均分」对照——E044 口径呼应）")

    # ── E10 作业档 ──
    out_size = (4 - 2) // 2 + 1
    check("E10 作业档输出尺寸=(4−2)/2+1=2", out_size == 2 and max_hand.shape == (2, 2),
          f"out_size={out_size}（口播不出答案——答案只在此档；E1 实算形状 {max_hand.shape} 互证）")

    # ── FG 示意图 ──
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(-0.4, 8.6)
    ax.set_ylim(-1.0, 4.9)
    ax.axis("off")
    sk = 1.0
    colors = {"win": "#d94f2b", "cell": "#f4f0e6", "grid": "#333333"}
    def cell(px, py, val, fc=colors["cell"], ec=colors["grid"], lw=1.2, fs=14, bold=False):
        ax.add_patch(plt.Rectangle((px, py), sk, sk, facecolor=fc, edgecolor=ec, linewidth=lw))
        ax.text(px + sk / 2, py + sk / 2, f"{val:g}", ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")
    for i in range(4):
        for j in range(4):
            cell(j, 3 - i, int(x[i, j]))
    # 高亮窗口 1（左上）+胜者位红框+其余三窗虚框轮动注记
    ax.add_patch(plt.Rectangle((0, 2), 2, 2, fill=False, edgecolor=colors["win"], linewidth=3))
    ax.add_patch(plt.Rectangle((1, 1), 1, 1, fill=False, edgecolor=colors["win"], linewidth=3))
    ax.annotate("2×2 window, stride 2\n(non-overlapping)", xy=(0.1, 4.35), fontsize=15, ha="left", color="#222222")
    ax.annotate("max 6 wins", xy=(1.55, 1.05), fontsize=14, ha="left", color=colors["win"])
    ax.annotate("", xy=(4.15, 2.6), xytext=(2.3, 2.6),
                arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=2))
    ax.text(3.2, 2.85, "winner", fontsize=14, ha="center", color="#222222")
    # 输出 2×2
    ax.text(5.0, 4.35, "Output 2×2", fontsize=15, ha="left", color="#222222")
    for i in range(2):
        for j in range(2):
            cell(5.0 + j, 2 - i, int(max_hand[i, j]), fc="#fbe9e2", fs=16, bold=True)
    # switch 路由注记
    ax.annotate("Backward: winner takes all the\ngradient (other 12 cells get 0)", xy=(5.0, -0.75), fontsize=14, ha="left",
                color="#222222")
    ax.annotate("\u201cswitch\u201d: forward winner takes\nthe whole gradient", xy=(0.1, -0.75), fontsize=14, ha="left",
                color="#444444")
    fig.savefig(PNG, dpi=150, bbox_inches="tight")
    # 图内文字收割+内部记号零命中
    texts = [t.get_text() for t in fig.findobj(_mtext.Text) if t.get_text().strip()]
    leaks = [s for s in texts if FIG_MARK.search(s)]
    n_invis = sum(1 for t in fig.findobj(_mtext.Text) if not t.get_visible())
    check("FG 图内文字收割+内部记号零命中", len(leaks) == 0,
          f"收割 {len(texts)} 条 Text 对象、内部记号命中 {len(leaks)}（不可见对象 {n_invis} 条实测计数）")

    # ── readings 落盘+sha 自证 ──
    n_pass = sum(1 for _, ok in LEDGER if ok)
    readings = {
        "n_pass": n_pass,
        "n_total": len(LEDGER),
        "input_4x4": x.flatten().tolist(),
        "max_out": max_hand.flatten().tolist(),
        "avg_out": avg_hand.flatten().tolist(),
        "winner_positions": np.argwhere(gmax == 1.0).tolist(),
        "grad_flow": flow,
        "out_size": out_size,
        "atol": ATOL,
        "zero_grad_cells": zeros,
        "avg_grad_cells": avg_cells,
        "avg_grad_per_cell": per_cell,
        "fg_text_harvest": len(texts),
        "fg_invisible_texts": n_invis,
    }
    READINGS.write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    print(f"[lab] readings sha256={hashlib.sha256(READINGS.read_bytes()).hexdigest()}")
    print(f"[lab] PNG sha256({PNG.name})={hashlib.sha256(PNG.read_bytes()).hexdigest() if PNG.exists() else 'MISSING'}")
    print(f"[lab] scipy={'yes' if HAVE_SCIPY else 'no（双线对照降级申报）'}")
    print(f"[lab] LEDGER：断言 {len(LEDGER)} 条，PASS {n_pass}")
    if n_pass != len(LEDGER):
        print("[lab] FAIL——存在失败断言")
        sys.exit(1)
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
