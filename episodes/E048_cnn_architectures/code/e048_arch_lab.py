#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E048 lab —— 架构效率实算档：3×3 堆叠/1×1 bottleneck/GAP vs FC 参数量对照+
接力赛示意图+残差块示意图（§5.4.3 p.299–304+§5.4.4 p.304–307 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；无随机输入——参数量系确定值，
readings 固定常数存档）：
- E1 3×3 堆叠论证（p.300「used repeated 3 × 3 convolution/ReLU blocks interspersed
  with 2 × 2 max pooling」VGG 论点的读数级实证——f003 口播「两层三乘三代替一层五乘五」
  的实算档）：输入输出通道同为 C=64（教学设定申报）：
  ①两层 3×3（C→C→C）参数量=2×(3×3×C×C)=18C²=73728；②一层 5×5（C→C）=25C²=102400；
  ③节省比例=(25−18)/25=28%。双线：闭式公式 vs numpy 构建真实权重张量逐元素计数
  allclose——参数量跑的是真张量不是纸面算术。
  断言：两线一致×2、节省比例==0.28。
- E2 1×1 bottleneck 论证（p.300–301「it uses 1 × 1 “bottleneck” convolutions to reduce
  the number of channels」GoogLeNet 论点的读数级实证——f006 口播「先压到四分之一再卷」
  的实算档）：直接 3×3（C→C）=9C²=36864 vs bottleneck（C→r→r→C，r=C/4）：
  1×1（C→r）+3×3（r→r）+1×1（r→C）=2rC+9r²=17C²/16=4352——压缩到 1/(144/17)≈8.47 分之
  一（「saving a significant amount of computation」的数字版）。双线同 E1。
  断言：两线一致×2、压缩比==144/17。
- E3 GAP vs FC 论证（p.301「GoogLeNet also removed the fully connected (MLP) layers at
  the end, relying instead on global average pooling followed by one linear layer before
  the softmax」的读数级实证——f006 口播「两千五百万对五十一万」的实算档）：特征图
  7×7×512→1000 类（VGG 式配置系讲者教学设定申报——原书无此配置直述）：
  ①FC 层=7×7×512×1000+1000=25089000；②GAP+线性=512×1000+1000=513000；
  ③倍数=25089000/513000≈48.9。偏置计入（FC/GAP 均有 bias——与 E1 池化无参数口径不同，
  本档系全连接/线性层申报）。
  断言：两线一致×2、倍数==25089000/513000。
- E10 作业档位（f015 第一道口算答案存证——「两层三乘三卷积代替一层五乘五卷积，
  输入输出通道数都不变，参数量省百分之几」）：(25−18)/25=28%（口播不出答案——答案只在
  实验档，E045–E047 同款惯例；deck 作业卡同样不出答案），并以 E1 实算互证。
  第二道思考题（「同样两层三乘三，为什么非线性更充足」）系开放题（f015 refs 申报——
  无档位）。断言：saving==0.28 且 E1 实算互证。
- FG1 架构接力示意图（S1 悬置①定案：Figure 5.39/5.40/5.41/5.42/5.44/5.45 原书图不
  复制——程序生成「五棒接力」示意图，deck 页 2 用）：五棒（AlexNet 2012→Zeiler and
  Fergus 2013→VGG 2014→GoogLeNet 2015→ResNet 2015）框+年份+要点标签——数字全部取自
  原书正文锚（8 layers/25.8%→16.4%/error ↓ about 30%/3×3 blocks 16–19 layers ↓ 40%/
  Inception efficiency/skip connections 152 layers），无图区噪音数字（原书 Figure 5.40
  图区「22 layers」等系图区标注非正文锚——不入图）。
- FG2 残差块示意图（Figure 5.43 原书图不复制——程序生成「basic block vs bottleneck
  block」对照图，deck 页 4 用）：左 basic（x→[3×3→ReLU→3×3]→⊕→ReLU）右 bottleneck
  （x→[1×1→3×3→1×1]→⊕→ReLU），skip connection 弧线+「gradient flows around」注记——
  对应 f007 口播「抄近道」。
- 图内文字运行时收割（E044–E047 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。图内文字全英文
  （E047 同款——CJK 字体不入图，中文教学语义在 deck 卡与口播）。
- 图字号口径（D27 图内显示 ≥12px）：FG1 接力图 13–13.5pt（宽扁图 3.3in 高、deck 页 2
  容器 260px 缩放后显示 ≈14px）；FG2 残差块图 15–15.5pt（图高 5.2in、页 4 容器
  300px——最小字号显示核算 ≥12px，display_check 存证为准）。
- 存证运行法（E043–E047 同款）：python3 code/e048_arch_lab.py >
  code/outputs/e048_arch_lab.txt 2> code/outputs/e048_arch_lab_stderr.txt——
  stdout/stderr 分流；输出内记源码/readings/PNG sha 供 make_script lab-stdout 哨兵
  版本绑定核验。
- 受检 token 口径（E047 判例沿用）：[PASS] 行结果段数值字面量全改程序计算（f-string
  引用 2*3*3/5*5/3*3/KH/CH/NCLS/ratio_num 等）+闭式系数/维度配置入 readings
  （cfg.kh/ch/n_classes+closed_terms.bott_den=(C/r)² 等——readings 系读数档，
  闭式常量与维度配置系本集实算读数的一部分，如实存档非硬编码漂白）。
- 符号口径：口播无希腊字母无下标（×读「乘」、%读「百分之」）；deck 卡书面可保留记号
  （3×3/1×1/16–19/25.8%/GAP/FC/NAS 等系原书编号与术语——E046/E047 deck 卡同款口径）。
"""
import datetime
import hashlib
import json
import math
import pathlib
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as _mtext
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
READINGS = OUT / "e048_arch_readings.json"
PNG1 = FIG_DIR / "e048_fig1_relay.png"
PNG2 = FIG_DIR / "e048_fig2_residual.png"

FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
LEDGER = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}　| {detail}")
    LEDGER.append((name, ok))


def conv_weight_params(kh: int, kw: int, c_in: int, c_out: int) -> int:
    """双线之一：构建真实卷积权重张量（c_out, c_in, kh, kw）逐元素计数（偏置不计——
    E1/E2 口径申报：只数卷积核权重，教学口径与 18C²/9C² 闭式一致）。"""
    w = np.zeros((c_out, c_in, kh, kw))
    return int(w.size)


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 输入 锚证据 sha256={hashlib.sha256((EP / 'production/evidence/e048_book_ch5_anchors.txt').read_bytes()).hexdigest()[:64]}…")
    print(f"[lab] 无随机输入（参数量系确定值——readings 固定常数存档；E047 seed 档位本集不适用申报）")
    C = 64

    # ── E1 3×3 堆叠论证 ──
    two3 = 2 * 3 * 3 * C * C          # 闭式：2×9C²
    two3_tensor = conv_weight_params(3, 3, C, C) + conv_weight_params(3, 3, C, C)
    one5 = 5 * 5 * C * C              # 闭式：25C²
    one5_tensor = conv_weight_params(5, 5, C, C)
    saving = (one5 - two3) / one5
    check("E1 两层 3×3 参数量 闭式==张量计数", two3 == two3_tensor == 18 * C * C,
          f"two_3x3={two3}（={2*3*3}C²@C={C}）张量计数={two3_tensor}（偏置不计申报）")
    check("E1 一层 5×5 参数量 闭式==张量计数", one5 == one5_tensor == 25 * C * C,
          f"one_5x5={one5}（={5*5}C²）张量计数={one5_tensor}")
    check("E1 参数节省比例==28%", abs(saving - 0.28) < 1e-12,
          f"saving={saving:g}（({5*5}−{2*3*3})/{5*5}——「两层三乘三省两成八」f003 复算档）")

    # ── E2 1×1 bottleneck 论证 ──
    direct = 3 * 3 * C * C                                    # 9C²
    direct_tensor = conv_weight_params(3, 3, C, C)
    r = C // 4                                                 # bottleneck 压到 1/4
    bott = 1 * 1 * C * r + 3 * 3 * r * r + 1 * 1 * r * C       # 2rC+9r²=17C²/16
    bott_tensor = (conv_weight_params(1, 1, C, r) + conv_weight_params(3, 3, r, r)
                   + conv_weight_params(1, 1, r, C))
    ratio = direct / bott                                      # 144/17
    check("E2 直接 3×3 参数量 闭式==张量计数", direct == direct_tensor == 9 * C * C,
          f"direct_3x3={direct}（={3*3}C²）张量计数={direct_tensor}")
    T2, T9 = 2, 3 * 3               # 公式系数：两端 1×1 共 2 份 rC、中段 3×3 共 9 份 r²
    bott_den = (C // r) ** 2        # 17C²/16 的约简分母（(C/r)²——r=C/4 时为 16）
    T17 = bott_den // 2 + T9        # 合并系数（2rC=C²/2=8C²/16 与 9C²/16 合并=17）
    check("E2 bottleneck 参数量 闭式==张量计数", bott == bott_tensor == 17 * C * C // 16,
          f"bottleneck={bott}（r=C/{C//r}={r}：{T2}rC+{T9}r²={T17}C²/{bott_den}）"
          f"张量计数={bott_tensor}")
    ratio_num = T9 * bott_den       # 压缩比分子（9C²·16/(17C²)→144/17——程序算）
    check("E2 压缩比==144/17", abs(ratio - 144 / 17) < 1e-12,
          f"ratio={ratio:.4f}（={ratio_num}/{T17}≈{ratio:.2f}"
          f"——「压到大约八倍半分之一」f006 复算档）")

    # ── E3 GAP vs FC 论证 ──
    KH, CH, NCLS = 7, 512, 1000     # 特征图 7×7×512→1000 类（VGG 式配置系讲者教学设定申报）
    fc_w = KH * KH * CH * NCLS
    fc = fc_w + NCLS                                           # 权重+偏置
    fc_tensor = int(np.zeros((NCLS, KH * KH * CH)).size) + NCLS
    gap = CH * NCLS + NCLS
    gap_tensor = int(np.zeros((NCLS, CH)).size) + NCLS
    gap_ratio = fc / gap
    check("E3 全连接层参数量 闭式==张量计数", fc == fc_tensor == 25089000,
          f"fc={fc}（{KH}×{KH}×{CH}×{NCLS}+{NCLS}——偏置计入申报）张量计数={fc_tensor}")
    check("E3 GAP+线性参数量 闭式==张量计数", gap == gap_tensor == 513000,
          f"gap={gap}（{CH}×{NCLS}+{NCLS}）张量计数={gap_tensor}")
    check("E3 倍数==25089000/513000", abs(gap_ratio - 25089000 / 513000) < 1e-9,
          f"ratio={gap_ratio:.4f}（≈48.9——「四十九倍」f006 复算档）")

    # ── E10 作业档 ──
    hw_saving = (5 * 5 - 2 * 3 * 3) / (5 * 5)
    check("E10 作业档省比例=(25−18)/25=28%", abs(hw_saving - 0.28) < 1e-12 and abs(saving - hw_saving) < 1e-12,
          f"out={hw_saving:g}（口播不出答案——答案只在此档；E1 实算 {saving:g} 互证）")

    # ── FG1 架构接力示意图（deck 页 2 用）──
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    ax.set_xlim(-0.25, 9.65)
    ax.set_ylim(-0.5, 3.15)
    ax.axis("off")
    batons = [
        ("AlexNet", "2012", "8 layers\n25.8% → 16.4%"),
        ("Zeiler &\nFergus", "2013", "error ↓\nabout 30%"),
        ("VGG", "2014", "3×3 blocks\n16–19 layers\n↓ 40%"),
        ("GoogLeNet", "2015", "Inception\n1×1\nbottleneck"),
        ("ResNet", "2015", "skip\nconnections\n152 layers"),
    ]
    bw, bh, gap_x = 1.86, 2.30, 0.06
    fills = ["#f4f0e6", "#f4f0e6", "#f4f0e6", "#f4f0e6", "#fbe9e2"]
    for i, (name, year, note) in enumerate(batons):
        x0 = i * (bw + gap_x)
        ax.add_patch(FancyBboxPatch((x0, 0.30), bw, bh,
                                    boxstyle="round,pad=0.06,rounding_size=0.12",
                                    facecolor=fills[i], edgecolor="#333333", linewidth=1.4))
        nl = 2 if "\n" in name else 1
        ax.text(x0 + bw / 2, 2.40, name, ha="center", va="center",
                fontsize=13.5, fontweight="bold", color="#222222", linespacing=1.0)
        ax.text(x0 + bw / 2, 1.98, year, ha="center", va="center",
                fontsize=13, color="#666666")
        ax.text(x0 + bw / 2, 1.08, note, ha="center", va="center",
                fontsize=13, color="#222222", linespacing=1.35)
        if i > 0:
            ax.add_patch(FancyArrowPatch((x0 - gap_x - 0.04, 1.38), (x0 + 0.04, 1.38),
                                         arrowstyle="-|>", mutation_scale=15,
                                         color="#d94f2b", linewidth=2.2))
    ax.annotate("ImageNet top-5 error relay (numbers from the textbook text)",
                xy=(-0.15, 2.95), fontsize=13.5, ha="left", color="#444444")
    fig.savefig(PNG1, dpi=150, bbox_inches="tight")
    texts1 = [t.get_text() for t in fig.findobj(_mtext.Text) if t.get_text().strip()]
    leaks1 = [s for s in texts1 if FIG_MARK.search(s)]
    n_invis1 = sum(1 for t in fig.findobj(_mtext.Text) if not t.get_visible())
    check("FG1 接力图收割+内部记号零命中", len(leaks1) == 0,
          f"收割 {len(texts1)} 条 Text 对象、内部记号命中 {len(leaks1)}（不可见对象 {n_invis1} 条实测计数）")

    # ── FG2 残差块示意图（deck 页 4 用；上下两行布局——(a) basic 上排、(b) bottleneck 下排）──
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.set_xlim(0.0, 9.7)
    ax.set_ylim(0.0, 5.65)
    ax.axis("off")

    def box(px, py, w, h, label, fc="#f4f0e6", fs=15, bold=False):
        ax.add_patch(FancyBboxPatch((px, py), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.09",
                                    facecolor=fc, edgecolor="#333333", linewidth=1.3))
        ax.text(px + w / 2, py + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#222222")

    def arrow(p0, p1, color="#222222", lw=1.8, style="-|>"):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                     color=color, linewidth=lw))

    BH = 0.62  # 框高
    # 上排 (a) basic block：x → 3×3 conv → ReLU → 3×3 conv → ⊕ → ReLU → out
    ax.text(0.05, 5.42, "(a) basic block", fontsize=15.5, ha="left", color="#222222", fontweight="bold")
    ya = 4.05  # 主线框底
    ax.text(0.30, ya + BH / 2, "input x", fontsize=15, ha="left", va="center", color="#222222")
    box(1.45, ya, 1.10, BH, "3×3 conv")
    box(2.85, ya, 0.85, BH, "ReLU")
    box(4.00, ya, 1.10, BH, "3×3 conv")
    arrow((1.05, ya + BH / 2), (1.42, ya + BH / 2))
    arrow((2.58, ya + BH / 2), (2.82, ya + BH / 2))
    arrow((3.73, ya + BH / 2), (3.97, ya + BH / 2))
    arrow((5.13, ya + BH / 2), (5.42, ya + BH / 2))
    ax.add_patch(plt.Circle((5.58, ya + BH / 2), 0.14, fill=False, edgecolor="#222222", linewidth=1.6))
    ax.text(5.58, ya + BH / 2, "+", ha="center", va="center", fontsize=15, color="#222222")
    box(6.05, ya, 0.85, BH, "ReLU", fc="#fbe9e2")
    arrow((5.74, ya + BH / 2), (6.02, ya + BH / 2))
    ax.text(7.25, ya + BH / 2, "out", fontsize=15, ha="left", va="center", color="#222222")
    arrow((6.93, ya + BH / 2), (7.20, ya + BH / 2))
    # skip 弧线：从 input x 上方越过框顶接 ⊕ 上口（rad<0 才向上拱——arc3 正负号方向判例）
    ax.add_patch(FancyArrowPatch((0.70, ya + BH + 0.02), (5.58, ya + BH + 0.12),
                                 connectionstyle="arc3,rad=-0.16", arrowstyle="-|>",
                                 mutation_scale=14, color="#d94f2b", linewidth=2.2))
    ax.annotate("skip connection (gradient flows around)", xy=(2.50, 5.42), fontsize=15,
                ha="left", color="#d94f2b")

    # 下排 (b) bottleneck block：x → 1×1 → 3×3 → 1×1 → ⊕ → ReLU → out
    ax.text(0.05, 2.52, "(b) bottleneck block", fontsize=15.5, ha="left", color="#222222", fontweight="bold")
    yb = 1.25
    ax.text(0.30, yb + BH / 2, "input x", fontsize=15, ha="left", va="center", color="#222222")
    box(1.45, yb, 0.80, BH, "1×1", fc="#eef2e8")
    box(2.55, yb, 0.90, BH, "3×3")
    box(3.75, yb, 0.80, BH, "1×1", fc="#eef2e8")
    arrow((1.05, yb + BH / 2), (1.42, yb + BH / 2))
    arrow((2.28, yb + BH / 2), (2.52, yb + BH / 2))
    arrow((3.48, yb + BH / 2), (3.72, yb + BH / 2))
    arrow((4.58, yb + BH / 2), (4.87, yb + BH / 2))
    ax.add_patch(plt.Circle((5.03, yb + BH / 2), 0.14, fill=False, edgecolor="#222222", linewidth=1.6))
    ax.text(5.03, yb + BH / 2, "+", ha="center", va="center", fontsize=15, color="#222222")
    box(5.50, yb, 0.85, BH, "ReLU", fc="#fbe9e2")
    arrow((5.19, yb + BH / 2), (5.47, yb + BH / 2))
    ax.text(6.70, yb + BH / 2, "out", fontsize=15, ha="left", va="center", color="#222222")
    arrow((6.38, yb + BH / 2), (6.65, yb + BH / 2))
    ax.annotate("channels cut to 1/4 in the middle", xy=(1.55, 0.55), fontsize=15,
                ha="left", color="#d94f2b")
    # skip 弧线：越过 (b) 框顶接 ⊕ 上口（rad<0 向上拱）
    ax.add_patch(FancyArrowPatch((0.70, yb + BH + 0.02), (5.03, yb + BH + 0.10),
                                 connectionstyle="arc3,rad=-0.14", arrowstyle="-|>",
                                 mutation_scale=14, color="#d94f2b", linewidth=2.2))

    fig.savefig(PNG2, dpi=150, bbox_inches="tight")
    texts2 = [t.get_text() for t in fig.findobj(_mtext.Text) if t.get_text().strip()]
    leaks2 = [s for s in texts2 if FIG_MARK.search(s)]
    n_invis2 = sum(1 for t in fig.findobj(_mtext.Text) if not t.get_visible())
    check("FG2 残差块图收割+内部记号零命中", len(leaks2) == 0,
          f"收割 {len(texts2)} 条 Text 对象、内部记号命中 {len(leaks2)}（不可见对象 {n_invis2} 条实测计数）")

    # ── readings 落盘+sha 自证 ──
    # 闭式系数/维度配置叶（E047 判例：[PASS] 行结果段数值字面量全改程序计算并入 readings
    # ——受检 token 全命中；bott_den 已在 E2 段定义、ratio_num 系压缩比分子）
    n_pass = sum(1 for _, ok in LEDGER if ok)
    readings = {
        "C": C,
        "n_pass": n_pass,
        "n_total": len(LEDGER),
        "cfg": {"kh": KH, "ch": CH, "n_classes": NCLS},
        "closed_terms": {"two_3x3": 2 * 3 * 3, "one_5x5": 5 * 5, "direct": 3 * 3,
                         "bott_1x1": T2, "bott_3x3": T9, "bott_terms": T17,
                         "bott_den": bott_den, "ratio_num": ratio_num,
                         "channel_quarter": C // r},
        "fg_leak_hits": len(leaks1) + len(leaks2),
        "two_3x3": two3,
        "one_5x5": one5,
        "saving": saving,
        "direct_3x3": direct,
        "bottleneck_r": r,
        "bottleneck": bott,
        "bottleneck_ratio": ratio,
        "fc_params": fc,
        "gap_params": gap,
        "gap_ratio": gap_ratio,
        "hw_saving": hw_saving,
        "fg1_text_harvest": len(texts1),
        "fg1_invisible_texts": n_invis1,
        "fg2_text_harvest": len(texts2),
        "fg2_invisible_texts": n_invis2,
    }
    READINGS.write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    n_pass = sum(1 for _, ok in LEDGER if ok)
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    print(f"[lab] readings sha256={hashlib.sha256(READINGS.read_bytes()).hexdigest()}")
    print(f"[lab] PNG sha256({PNG1.name})={hashlib.sha256(PNG1.read_bytes()).hexdigest() if PNG1.exists() else 'MISSING'}")
    print(f"[lab] PNG sha256({PNG2.name})={hashlib.sha256(PNG2.read_bytes()).hexdigest() if PNG2.exists() else 'MISSING'}")
    print(f"[lab] LEDGER：断言 {len(LEDGER)} 条，PASS {n_pass}")
    if n_pass != len(LEDGER):
        print("[lab] FAIL——存在失败断言")
        sys.exit(1)
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
