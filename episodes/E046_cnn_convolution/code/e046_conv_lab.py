#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E046 lab —— 卷积实验：参数量/计算量实测+权值共享 vs 全连接同任务对照+作业档+三进两出示意图
（§5.4 前半 p.291–294 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；seed 2042 全档固定）：
- E1 参数量/计算量实测（p.292 Figure 5.34 caption「the total number of learnable parameters
  in each convolutional layer is S 2 × C1 × C2」「the amount of computation (multiply-adds)
  … is W HS 2 C1 C2」的读数级实证——f005 复算账可现场对）：C1=6 输入通道、C2=4 输出通道、
  S=3 窗口、输入 10×10（输出 8×8）：
  ①参数量：核权重张量 (C2,C1,3,3).size=9×6×4=216（口播「二百一十六」），偏置 (C2,)=4
    （式 5.76 的 +b(c2)——偏置独立申报不计入 9×6×4 权重账）；
  ②计算量：六重嵌套循环真卷积逐次计数乘加=输出像素 8×8 × 窗口 9 × C1 6 × C2 4=13824
    （宽×高×九乘六乘四——「宽×高」系输出层像素位置口径，原书 caption「applied at each of
    the W × H pixels in a given layer」）；偏置加法 8×8×4=256 次另计不入乘加账（申报口径）；
  ③正确性：嵌套循环结果 vs sliding_window_view+einsum 向量化参考 allclose(atol=1e-9)——
    计数实验跑的是真卷积不是纸面算术。
  断言：weights==216、macs==13824、allclose 通过。
- E0 权值共享 vs 全连接同任务对照（p.293「This means that far fewer weights are needed
  than in fully connected layers」的读数级实证——f010「这个差距有多大」白话的实算档）：
  同任务=同输入输出形状（功能不同：卷积共享权重、全连接逐点连线——申报口径）：
  输入 10×10×6=600 单元 → 输出 8×8×4=256 单元；
  全连接权重=600×256=153600（口播量级白话「十几万根连线」）；卷积核权重=216（同 E1）；
  倍数=153600/216=711.1（deck 页 6 卡引用值）。
  断言：fc==153600、conv==216、ratio>100。
- E10 作业档位（f014 第一道口算答案存证——「5×5 窗口、3 输入通道、8 输出通道，口算可学
  权重再用小实验程序验证」）：核权重张量 (8,3,5,5).size=25×3×8=600（口播不出答案——
  答案只在实验档，E045 f014 同款惯例；deck 作业卡同样不出答案），偏置 (8,)=8 另计申报。
  断言：weights==600。第二道思考题系开放题（f014 refs 申报无标准答案——无档位）。
- FG 三进两出示意图（S1 悬置①定案：Figure 5.33/5.34/5.35 原书图不复制——程序生成
  「卷积窗口×通道堆叠」示意图，deck 页 2 用）：三张输入通道堆叠+每张同步高亮 3×3 窗口+
  两套卷积核堆叠+两张输出特征图——对应 f004 口播「三张输入通道进，两张输出通道出，
  窗口在每张通道上同步滑动」。图内文字运行时收割（E044/E045 同款对象遍历）：fig.findobj(Text)
  收割，FIG_MARK（E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。
- 图字号口径（D27 图内显示 ≥12px）：deck 页 2 右栏列宽缩放实测以 display_check 存证为准；
  图内字号取 13–16pt（E044 12.5–14.5pt/E045 12.5–15pt 同档——本图标签少、字号可再大一档）。
- 存证运行法（E043/E044/E045 同款）：python3 code/e046_conv_lab.py >
  code/outputs/e046_conv_lab.txt 2> code/outputs/e046_conv_lab_stderr.txt——stdout/stderr
  分流；输出内记源码/readings/PNG sha 供 make_script lab-stdout 哨兵版本绑定核验。
- 符号口径：口播无希腊字母（式 5.76 白话读法「把所有输入通道、窗口里每个偏移位置加起来」
  ——f007 refs 申报）；deck 卡书面可保留式号与记号（5.76/1×1/153600 等数字卡系 lab 实测
  值引用——E045 deck 卡引用 lab 数字同款）。
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
from matplotlib.patches import Polygon

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2042
LEDGER = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}　| {detail}")
    LEDGER.append((name, ok))
    if not ok:
        print(f"[FAIL-ABORT] {name}", file=sys.stderr)


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    anc = EP / "production" / "evidence" / "e046_book_ch5_anchors.txt"
    print(f"[lab] 输入 锚证据 sha256={hashlib.sha256(anc.read_bytes()).hexdigest()[:40]}…")
    src = pathlib.Path(__file__).read_bytes()
    print(f"[lab] 源码 sha256={hashlib.sha256(src).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    R = {}

    # ── E1 参数量/计算量实测（p.292 Figure 5.34 caption 两式口径）──
    rng = np.random.default_rng(SEED)
    c1, hw, s, c2 = 6, 10, 3, 4
    x = rng.standard_normal((c1, hw, hw))
    W = rng.standard_normal((c2, c1, s, s))
    b = rng.standard_normal(c2)
    conv_weights, conv_bias = int(W.size), int(b.size)

    def conv2d_counting(x, W, b):
        """六重嵌套循环真卷积（互相关形态——偏移加到坐标上，p.292 口径）+乘加计数。"""
        cc, h, _ = x.shape
        d, _, kh, kw = W.shape
        oh, ow = h - kh + 1, h - kw + 1
        out = np.zeros((d, oh, ow))
        mac = 0
        for di in range(d):
            for i in range(oh):
                for j in range(ow):
                    acc = 0.0
                    for ci in range(cc):
                        for k in range(kh):
                            for l in range(kw):
                                acc += W[di, ci, k, l] * x[ci, i + k, j + l]
                                mac += 1
                    out[di, i, j] = acc + b[di]   # 偏置加法另计（申报口径：不入乘加账）
        return out, mac

    out_loop, macs = conv2d_counting(x, W, b)
    # 向量化参考（sliding_window_view+einsum）——计数实验跑的是真卷积的正确性对照
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(x, (s, s), axis=(1, 2))          # (C1, oh, ow, s, s)
    ref = np.einsum("cijhw,dchw->dij", win, W) + b[:, None, None]
    ok_close = bool(np.allclose(out_loop, ref, atol=1e-9))
    oh = ow = hw - s + 1
    macs_formula = oh * ow * s * s * c1 * c2
    R["e1_params_macs"] = {
        "c1": c1, "c2": c2, "kernel": s, "input_hw": hw, "output_hw": oh,
        "conv_weights": conv_weights, "conv_bias": conv_bias,
        "macs": macs, "macs_formula": macs_formula, "allclose": ok_close, "seed": SEED,
    }
    check("E1 核权重=9×6×4=216（Figure 5.34 caption S²×C1×C2 口径；偏置 4 另计）",
          conv_weights == 216 and conv_bias == 4,
          f"conv_weights={conv_weights} conv_bias={conv_bias}")
    check("E1 乘加计数=8×8×9×6×4=13824（嵌套循环真卷积逐次计数=公式宽×高×九乘六乘四）",
          macs == macs_formula == 13824,
          f"macs={macs} macs_formula={macs_formula}")
    check("E1 嵌套循环 vs 向量化参考 allclose（atol=1e-9——计数实验系真卷积）", ok_close,
          f"allclose={ok_close}")

    # ── E0 权值共享 vs 全连接同任务对照（p.293 far fewer weights 读数级实证）──
    in_units, out_units = c1 * hw * hw, c2 * oh * ow
    fc_weights = in_units * out_units
    ratio = round(fc_weights / conv_weights, 1)
    R["e0_fc_vs_conv"] = {
        "input_units": in_units, "output_units": out_units,
        "fc_weights": fc_weights, "conv_weights": conv_weights, "ratio": ratio,
    }
    check("E0 同任务全连接权重=600×256=153600（口播量级白话「十几万根连线」的实算档）",
          fc_weights == 153600, f"fc_weights={fc_weights}")
    check("E0 同任务对照倍数=153600/216=711.1（>100 量级——far fewer weights 实证）",
          fc_weights == 153600 and conv_weights == 216 and ratio == 711.1 and ratio > 100,
          f"fc={fc_weights} conv={conv_weights} ratio={ratio}")

    # ── E10 作业档位（f014 第一道口算答案存证）──
    kw10, kh10, c1_10, c2_10 = 5, 5, 3, 8
    W10 = np.zeros((c2_10, c1_10, kh10, kw10))
    hw_weights = int(W10.size)
    hw_bias = c2_10
    R["e10_homework"] = {
        "kw": kw10, "kh": kh10, "c1": c1_10, "c2": c2_10,
        "weights": hw_weights, "bias": hw_bias,
    }
    check("E10 作业档核权重=25×3×8=600（f014 第一道口算答案——口播不出答案，只在此档兑现）",
          hw_weights == 600, f"weights={hw_weights} bias={hw_bias}（偏置另计申报）")

    # ── FG 三进两出示意图（S1 悬置①——程序生成，deck 页 2 左栏 f004 下用）──
    # 版面口径：deck 页 2 左栏=两帧口播+本图（右栏双卡）——图容器 max-height 260px、
    # PNG 1560×639（figsize 10.4×4.26@150dpi，幅面比 2.44=xlim/ylim 无失真）——图内
    # 最小字号 14.5pt 显示核算 12.3px≥12（D27 图内线，display_check 存证）；版面结构
    # 判例同 E045 页 9：动结构不硬压尺寸。
    fig, ax = plt.subplots(figsize=(10.4, 4.26))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def slab(x0, y0, w, h, sk, fc, ec="#333333", lw=1.2, z=2):
        """平行四边形板（等距感通道板）。"""
        pts = [(x0, y0), (x0 + w, y0), (x0 + w + sk, y0 + h), (x0 + sk, y0 + h)]
        ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))
        return pts

    def grid_lines(x0, y0, w, h, sk, n, z=3):
        for i in range(1, n):
            t = i / n
            ax.plot([x0 + w * t, x0 + w * t + sk], [y0, y0 + h],
                    color="#9aa5b1", lw=0.7, zorder=z)
            ax.plot([x0 + sk * t, x0 + w + sk * t], [y0 + h * t, y0 + h * t],
                    color="#9aa5b1", lw=0.7, zorder=z)

    def window(x0, y0, w, h, sk, n, k0, z=5):
        """3×3 窗口高亮（自第 k0 列、第 1 行起共 3×3 格——同一相对位置=同步滑动）。"""
        cw, ch = w / n, h / n
        wx0, wy0 = x0 + cw * k0, y0 + ch
        sk3 = sk * 3 / n
        pts = [(wx0, wy0), (wx0 + cw * 3, wy0),
               (wx0 + cw * 3 + sk3, wy0 + ch * 3), (wx0 + sk3, wy0 + ch * 3)]
        ax.add_patch(Polygon(pts, closed=True, facecolor="#f6bcbc", alpha=0.75,
                             edgecolor="#c62828", lw=2.2, zorder=z))

    # 三张输入通道（堆叠偏移 0.7——后通道窗口露出量足够读出「同步滑动」）；
    # z 序=后通道整体盖前通道，窗口只盖本通道板面（z=2+idx*3 系板/格线/窗口三件套的逐层递进）
    for idx in range(3):
        x0, y0 = 0.5 + 0.7 * idx, 0.75 + 0.7 * idx
        zb = 2 + idx * 3
        slab(x0, y0, 3.4, 2.0, 0.9, "#dae8fc" if idx == 2 else "#c3daf5",
             lw=1.6 if idx == 2 else 1.0, z=zb)
        grid_lines(x0, y0, 3.4, 2.0, 0.9, 5, z=zb + 1)
        window(x0, y0, 3.4, 2.0, 0.9, 5, 1, z=zb + 2)
    ax.text(0.9, 5.9, "Input: 3 channels", fontsize=16.5, fontweight="bold")
    ax.text(0.5, 0.12, "3×3 window slides in sync on every channel",
            fontsize=14.5, color="#c62828")

    # 两套卷积核（每套 3 层堆叠的小 3×3 板）
    for ky, klabel in [(3.6, "Kernel set A"), (0.85, "Kernel set B")]:
        for idx in range(3):
            x0, y0 = 7.0 + 0.35 * idx, ky + 0.35 * idx
            slab(x0, y0, 1.4, 1.0, 0.38, "#ffe6cc" if idx == 2 else "#ffd9b3",
                 lw=1.4 if idx == 2 else 0.9, z=2 + idx)
            grid_lines(x0, y0, 1.4, 1.0, 0.38, 3, z=3 + idx)
        ax.text(7.0, ky + 1.95, klabel, fontsize=15.5, fontweight="bold")
    # （每核堆 3 层=3 输入通道的说明系 deck 页 2 卡与 figcaption 覆盖域——图内不重复，底部让给红字窗口说明）

    # 两张输出特征图（标签右对齐收进画布内——左对齐会越界裁切）
    for oy, olabel in [(3.7, "Output channel 1"), (0.75, "Output channel 2")]:
        slab(12.2, oy, 1.9, 1.5, 0.55, "#d5e8d4", lw=1.5, z=2)
        grid_lines(12.2, oy, 1.9, 1.5, 0.55, 3, z=3)
        ax.text(15.45, oy + 2.33, olabel + " (feature map)", fontsize=14.5,
                ha="right")  # 板上方独立高度（与核堆标签错行——压幅后同高相撞判例）

    # 箭头：输入堆→两套核→两张输出
    arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=1.8)
    ax.annotate("", xy=(7.0, 4.75), xytext=(6.5, 3.3), arrowprops=arrow_kw, zorder=6)
    ax.annotate("", xy=(7.0, 1.7), xytext=(6.5, 2.6), arrowprops=arrow_kw, zorder=6)
    ax.annotate("", xy=(12.15, 4.4), xytext=(9.6, 4.55), arrowprops=arrow_kw, zorder=6)
    ax.annotate("", xy=(12.15, 1.5), xytext=(9.6, 1.5), arrowprops=arrow_kw, zorder=6)

    ax.set_title("Multi-channel convolution: 3 in, 2 out", fontsize=17.5, pad=8)
    texts = [t.get_text() for t in fig.findobj(_mtext.Text) if t.get_text().strip()]
    FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
    hits = [t for t in texts if FIG_MARK.search(t)]
    invisible = sum(1 for t in fig.findobj(_mtext.Text)
                    if not t.get_visible() or not t.get_text().strip())
    fig_dir = OUT / "figures"   # 图入 figures/ 子目录——共享 probe 的 PNG sha 自证行按该目录打印（E044/E045 同款）
    fig_dir.mkdir(exist_ok=True)
    png = fig_dir / "e046_fig1_conv_diagram.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    R["fig1"] = {"text_objects": len(texts), "invisible_objects": invisible,
                 "mark_hits": len(hits), "png": "e046_fig1_conv_diagram.png"}
    check("FG 图内文字收割（对象遍历）+内部记号零命中", len(hits) == 0,
          f"收割 {len(texts)} 条 Text 对象、内部记号命中 {len(hits)}（不可见对象 {invisible} 条实测计数）")

    # ── readings 落盘+sha+LEDGER ──
    rp = OUT / "e046_conv_readings.json"
    rp.write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    n_pass = sum(1 for _, ok in LEDGER if ok)
    print(f"[lab] LEDGER：断言 {len(LEDGER)} 条，PASS {n_pass}")
    png_sha = hashlib.sha256(png.read_bytes()).hexdigest()
    print(f"[lab] PNG sha256({png.name})={png_sha}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}（起 {t0.isoformat(timespec='seconds')}）")
    if n_pass != len(LEDGER):
        sys.exit(1)


if __name__ == "__main__":
    main()
