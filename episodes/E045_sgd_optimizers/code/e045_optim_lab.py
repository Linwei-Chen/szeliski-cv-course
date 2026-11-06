#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E045 lab —— 优化器实验：He init 方差实测+学习率三档+窄谷动量+IIR 稳态（§5.3.6+初始化推导 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；seed 2042 全档固定）：
- E0 He init 方差实测（p.283 式 5.59–5.61 推导链的「可验算」档）：5 层 ReLU 网络（每层
  256 单元、偏置 0），输入 256 维 N(0,1) 4000 样本。两种初始化对照：
  ①He 初始化 W~N(0, 2/n_l)（式 5.61 V_l=2/n_l）——理论逐层方差比≈1（ReLU 截半恰好被
    2/n 补偿：Var[s]=n·(2/n)·Var[x]=2Var[x]→Var[y]=Var[x]）；
  ②Glorot 口径 W~N(0, 1/n_l)（fan-in 倒数——原书申报其分析假设线性激活，对 ReLU 不
    匹配）——理论逐层方差比 0.5，5 层后≈0.031（深层激活衰减实证）。
  读数：各层输出均方 E[y²]/输入均方 E[x²] 比值（逐层+终层——原书式 5.60 左端 E[y²] 的
  二阶矩口径；方差口径首跑判例：He 比值≈0.68=1−2/π 系统偏移，因 ReLU 输出均值非零）。
  断言：he 各层∈[0.75,1.15]（期望意义保持；单实现有限 n 偏差逐层随机游走——首跑判例
  0.82–1.03，原书口径「stay about the same」）、glorot 终层<0.1。
- E1 学习率三档（p.288「The step size parameter α is often called the learning rate…must
  be carefully adjusted…avoiding overshooting and exploding gradients」的读数级实证）：
  二次损失 L(w)=(w−3)²/2（梯度=w−3，w0=0，60 步）三档：
  α=0.1（太小——每步走 10%，60 步 |w−3|=3·0.9^60≈0.005 尚未走完）；
  α=1.0（恰好——w1=0+1.0·3=3.0 一步精确到谷底）；
  α=2.2（太大——|w−3| 每步×1.2 发散，20 步 |w−3|≈3·1.2^20≈115）。
  断言：best 档 err_1<1e-9、div 档 err_20>10。左图三条 |w−3| 轨迹（log y）——「步长决定命运」。
- E2 窄谷动量对比（p.289 momentum 式 5.74–5.75+ρ∈[0.9,0.99]「good memory」+「Regular
  gradient descent is prone to stalling when the current solution reaches a flat spot」的
  读数级实证）：扁谷地形 L(w)=0.0173·w1²+1.0·w2²（程序生成教学地形，非原书地形——
  w1 方向平坦=「flat spot」向、w2 方向陡=窄谷壁），w0=(30,1)，α=0.05，400 步，ρ=0
  （裸 SGD）vs ρ=0.9（momentum）。
  理论：SGD w1 衰减率 1−α·H1=0.99827/步→400 步 ≈15.0；momentum 等效增益 α/(1−ρ)=0.5
  →衰减率≈0.9734/步→400 步 ≈6e−4。断言：w1_mom<0.1、w1_sgd>10（动量穿平坦方向快
  百倍量级）、momentum w2 残余<0.01（振铃衰减包络 |λ|=√ρ=0.949/步）。右图 |w1| 轨迹。
- E2b IIR 稳态位移验证（p.289 脚注 32 定量口径「a recursive formula such as (5.74)…
  converges in the limit to a value of g/(1 − ρ), so α needs to be correspondingly
  adjusted」——现场可验算）：恒定梯度场 g=−3（L=(w−3)²/2 在 w0=0 处），纯递推
  v_{t+1}=ρv_t+g、w_{t+1}=w_t−αv_t（式 5.74–5.75 形态，ρ=0.9、α=0.05、200 步）。
  稳态每步位移=α·|g|/(1−ρ)=1.5（理论）；裸 SGD（同梯度同 α）每步位移=α·|g|=0.15。
  断言：稳态位移与理论差<1e-9、比值=1/(1−ρ)=10——「动量把步长等效放大 1/(1−ρ) 倍，
  所以 α 要相应调小」的原书定量申报实算。
- 图内文字运行时收割（E044 E6 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。
- 存证运行法（E043/E044 同款）：python3 code/e045_optim_lab.py > code/outputs/e045_optim_lab.txt
  2> code/outputs/e045_optim_lab_stderr.txt——stdout/stderr 分流；输出内记源码/readings/
  PNG sha 供 make_script lab-stdout 哨兵版本绑定核验。
- E10 作业档位（f014 第一道手算答案存证）：稳态位移公式代值 g=−2、α=0.1、ρ=0.9→
  α·|g|/(1−ρ)=2.0，与 IIR 递推实测对拍<1e-9（E044 E10 同款惯例——口播手算答案可现场对）。
- 符号口径：口播无希腊字母无上下标（α 读「学习率」、ρ 读「动量系数」、g/(1−ρ) 读
  「梯度除以一减动量系数」）；式号读「式子 5 点七二」等；deck 卡书面保留式号与记号
  （t+i 下标系原书排印原样——锚 §判例申报，卡上如出现须申报读法）。
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
    anc = EP / "production" / "evidence" / "e045_book_ch5_anchors.txt"
    print(f"[lab] 输入 锚证据 sha256={hashlib.sha256(anc.read_bytes()).hexdigest()[:40]}…")
    src = pathlib.Path(__file__).read_bytes()
    print(f"[lab] 源码 sha256={hashlib.sha256(src).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    R = {}

    # ── E0 He init 方差实测 ──
    rng = np.random.default_rng(SEED)
    n_in, n_unit, n_layer, n_sample = 256, 256, 5, 4000
    x = rng.standard_normal((n_sample, n_in))

    def forward_variances(scale: float):
        v = x.copy()
        v_in_ms = (v ** 2).mean()               # E[x²] 二阶矩口径——原书式 5.60 左端 E[y²] 的口径
        ratios = []                              # （ReLU 输出均值非零，方差口径含 1−2/π≈0.68 系统偏移——
        for _ in range(n_layer):                 #   首版实测判例申报：Var 口径 He 比值≈0.68=E[y²]−E[y]²）
            W = rng.standard_normal((v.shape[1], n_unit)) * np.sqrt(scale / v.shape[1])
            v = np.maximum(v @ W, 0.0)          # ReLU，偏置 0
            ratios.append((v ** 2).mean() / v_in_ms)
        return ratios

    he = forward_variances(2.0)                  # 式 5.61：V_l=2/n_l
    glo = forward_variances(1.0)                 # fan-in 倒数（Glorot 口径，对 ReLU 不匹配）
    R["e0_he_init"] = {
        "he_ratios": [round(float(r), 6) for r in he],
        "glorot_ratios": [round(float(r), 6) for r in glo],
        "he_final": round(float(he[-1]), 6),
        "glorot_final": round(float(glo[-1]), 6),
        "n_layer": n_layer, "n_unit": n_unit, "n_sample": n_sample, "seed": SEED,
    }
    # 原书口径「stay about the same」——期望意义 E[y²]=E[x²]；单实现有限 n 的 Var[s] 偏差逐层
    # 随机游走累积（首跑判例：5 层实现 0.82–1.03），断言窗 [0.75,1.15] 申报量级保持
    ok_he = all(0.75 <= r <= 1.15 for r in he)
    ok_glo = glo[-1] < 0.1
    check("E0 He init 逐层均方比∈[0.75,1.15]（式 5.60 E[y²] 口径——量级保持，原书「about the same」）", ok_he,
          f"he_ratios={[round(float(r),3) for r in he]}")
    check("E0 Glorot 口径 5 层后激活均方<0.1（对 ReLU 不匹配→深层衰减）", ok_glo,
          f"glorot_final={glo[-1]:.6f}（深层衰减实测——理论每层减半）")

    # ── E1 学习率三档 ──
    def gd_track(alpha: float, steps: int):
        w, track = 0.0, []
        for t in range(1, steps + 1):
            g = w - 3.0                          # dL/dw, L=(w−3)²/2
            w = w - alpha * g
            track.append(abs(w - 3.0))
        return track

    tr_slow = gd_track(0.1, 60)
    tr_best = gd_track(1.0, 60)
    tr_div = gd_track(2.2, 60)
    R["e1_lr"] = {
        "slow": {"alpha": 0.1, "err60": round(float(tr_slow[-1]), 9)},
        "best": {"alpha": 1.0, "w1": 3.0, "err1": round(float(tr_best[0]), 12)},
        "div": {"alpha": 2.2, "err20": round(float(tr_div[19]), 6)},
        "n_steps": 60, "w0": 0.0, "w_star": 3.0,
    }
    check("E1 α=0.1 慢档（60 步尚未走完——每步走 10%）", 1e-4 < tr_slow[-1] < 0.02,
          f"err60={tr_slow[-1]:.3e}（理论 3·0.9^60≈5.4e-3——在降但未到）")
    check("E1 α=1.0 恰好档（一步精确到谷底）", tr_best[0] < 1e-9,
          f"err1={tr_best[0]:.1e}（w1=0+1.0×3=3.0）")
    check("E1 α=2.2 发散档（|w−3| 每步×1.2——overshooting/exploding 实证）",
          tr_div[19] > 10.0, f"err20={tr_div[19]:.1f}（发散实测——每步放大约二成）")

    # ── E2 窄谷动量对比 ──
    H1, H2 = 2 * 0.0173, 2 * 1.0                 # 对角 Hessian（L=0.0173w1²+1.0w2²）
    alpha, rho, steps = 0.05, 0.9, 400
    w0 = np.array([30.0, 1.0])

    def ravine(rho: float):
        w = w0.copy(); v = np.zeros(2)
        w1s, w2s = [], []
        for _ in range(steps):
            g = np.array([H1 * w[0], H2 * w[1]])
            v = rho * v + g
            w = w - alpha * v
            w1s.append(w[0]); w2s.append(w[1])
        return np.array(w1s), np.array(w2s)

    w1s0, w2s0 = ravine(0.0)
    w1s9, w2s9 = ravine(rho)
    mom_resid = float(np.abs(w2s9[-50:]).max())
    R["e2_momentum"] = {
        "w1_sgd_400": round(float(w1s0[-1]), 6),
        "w1_mom_400": round(float(w1s9[-1]), 9),
        "w2_mom_residual": round(mom_resid, 12),  # 全精度存档——stdout .2e 展示 9.20e-09 须同值命中（round 9 位会把 9.2e-9 舍成 9e-9，2.2% 差超 1% 容差判例）
        "w2_envelope_theory": round(0.9 ** 0.5, 6),  # 振铃包络理论值 √ρ——stdout「≈0.95」读法同值命中（0.95 vs 0.948683 差 0.14%<1% 容差）
        "alpha": alpha, "rho": rho, "steps": steps,
        "w0": [30.0, 1.0], "terrain": "L=0.0173·w1²+1.0·w2²（程序生成教学地形，非原书）",
    }
    check("E2 动量穿平坦方向（400 步 |w1| 由 30 →<0.1）", abs(w1s9[-1]) < 0.1,
          f"w1_mom={w1s9[-1]:.2e}")
    check("E2 裸 SGD 平坦方向停滞（400 步 |w1| 仍>10——flat spot 停滞实证）",
          abs(w1s0[-1]) > 10.0, f"w1_sgd={w1s0[-1]:.3f}（理论衰减率 0.99827/步）")
    check("E2 momentum w2 振铃衰减（残余<0.01）", mom_resid < 0.01,
          f"residual={mom_resid:.2e}（包络 |λ|=√ρ≈0.95/步）")

    # ── E2b IIR 稳态位移（脚注 32 定量口径） ──
    g_const = -3.0
    alpha_b, rho_b, steps_b = 0.05, 0.9, 400   # 400 步：瞬态 0.9^400≈1.4e-18 远离 1e-9 断言线（200 步首跑判例）
    w, v = 0.0, 0.0
    disp = []
    for _ in range(steps_b):
        v = rho_b * v + g_const
        w_new = w - alpha_b * v
        disp.append(abs(w_new - w))
        w = w_new
    steady = float(np.mean(disp[-20:]))
    theory = alpha_b * abs(g_const) / (1 - rho_b)
    bare = alpha_b * abs(g_const)
    R["e2b_iir"] = {
        "steady_displacement": round(steady, 12),
        "steady_theory": round(theory, 12),
        "bare_displacement": round(bare, 12),
        "ratio": round(steady / bare, 9),
        "theory_ratio": round(1.0 / (1 - rho_b), 9),
        "g": -3.0, "alpha": alpha_b, "rho": rho_b, "steps": steps_b,
    }
    check("E2b IIR 稳态位移=α·|g|/(1−ρ)（脚注 32 g/(1−ρ) 口径）", abs(steady - theory) < 1e-9,
          f"实测 {steady:.6f} vs 理论 {theory:.6f}")
    check("E2b 等效步长放大=1/(1−ρ)（动量 vs 裸 SGD 同梯度同 α）",
          abs(steady / bare - 1.0 / (1 - rho_b)) < 1e-9,
          f"比值 {steady / bare:.1f} vs 理论 {1.0 / (1 - rho_b):.1f}——α 需相应调小的原书申报实算")

    # ── E10 作业档位（f014 第一道手算答案存证）：稳态位移手算——恒定梯度 g=−2、
    #    α=0.1、ρ=0.9，稳态每步位移=α·|g|/(1−ρ)=2.0（脚注 32 公式代值）+IIR 递推实测对拍 ──
    g_h, a_h, r_h = -2.0, 0.1, 0.9
    w_h, v_h = 0.0, 0.0
    for _ in range(600):
        v_h = r_h * v_h + g_h
        w_h = w_h - a_h * v_h
    d_prev = None
    for _ in range(3):
        v_h = r_h * v_h + g_h
        w_new = w_h - a_h * v_h
        d_prev = abs(w_new - w_h)
        w_h = w_new
    hand_ans = a_h * abs(g_h) / (1 - r_h)
    R["e10_homework"] = {
        "hand_answer": round(hand_ans, 12),
        "iir_measured": round(d_prev, 12),
        "params": {"g": g_h, "alpha": a_h, "rho": r_h},
    }
    check("E10 手算稳态位移 vs IIR 递推实测（f014 作业第一道答案）", abs(d_prev - hand_ans) < 1e-9,
          f"手算 {hand_ans:.6f} vs 实测 {d_prev:.6f}（g=−2、α=0.1、ρ=0.9——α·|g|/(1−ρ) 代值）")

    # ── 图：e045_fig1_optim_compare.png（双面板，程序生成——未用原书原图） ──
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.1))
    axL.semilogy(range(1, 61), np.maximum(tr_slow, 1e-12), "-", color="#c46a2b",
                 label="learning rate 0.1")
    axL.semilogy(range(1, 61), np.maximum(tr_best, 1e-12), "-", color="#3d7a4e",
                 label="learning rate 1.0")
    axL.semilogy(range(1, 61), np.maximum(tr_div, 1e-12), "-", color="#a03838",
                 label="learning rate 2.2")
    # 字号适配（D27 图内显示 ≥12px：deck 右栏列宽/图高双约束缩放后取最小档 12.5pt 实测见 display_check 存证）
    for ax in (axL, axR):
        ax.tick_params(labelsize=12.5)
    axL.set_xlabel("step", fontsize=13)
    axL.set_ylabel("distance to optimum", fontsize=13)
    axL.set_title("Step size decides the fate", fontsize=13.5)
    axL.legend(fontsize=13)
    axR.semilogy(range(1, steps + 1), np.abs(w1s0) + 1e-12, "-", color="#a03838",
                 label="plain gradient descent")
    axR.semilogy(range(1, steps + 1), np.abs(w1s9) + 1e-12, "-", color="#3d5a8a",
                 label="momentum 0.9")
    axR.set_xlabel("step", fontsize=13)
    axR.set_ylabel("flat-axis weight magnitude", fontsize=13)
    axR.set_title("Momentum crosses the flat direction", fontsize=13.5)
    axR.legend(fontsize=13)
    fig.suptitle("Optimization experiments: step size and momentum", fontsize=15)
    # 图内文字收割（红线 5：内部记号零泄漏）——保存前对同一图对象遍历（E044 E6 同款）
    texts = [t.get_text() for t in fig.findobj(_mtext.Text) if t.get_text().strip()]
    FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
    hits = [t for t in texts if FIG_MARK.search(t)]
    invisible = sum(1 for t in fig.findobj(_mtext.Text)
                    if not t.get_visible() or not t.get_text().strip())
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig_dir = OUT / "figures"   # 图入 figures/ 子目录——共享 probe 的 PNG sha 自证行按该目录打印（E044 同款结构）
    fig_dir.mkdir(exist_ok=True)
    png = fig_dir / "e045_fig1_optim_compare.png"
    fig.savefig(png, dpi=170)
    plt.close(fig)
    R["fig1"] = {"text_objects": len(texts), "invisible_objects": invisible,
                 "mark_hits": len(hits), "png": "e045_fig1_optim_compare.png"}
    check("E6 图内文字收割（对象遍历）+内部记号零命中", len(hits) == 0,
          f"收割 {len(texts)} 条 Text 对象、内部记号命中 {len(hits)}（不可见对象 {invisible} 条实测计数）")

    # ── readings 落盘+sha+LEDGER ──
    OUT.mkdir(exist_ok=True)
    rp = OUT / "e045_optim_readings.json"
    n_pass = sum(1 for _, ok in LEDGER if ok)
    R["n_pass"] = n_pass
    R["n_total"] = len(LEDGER)
    rp.write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(LEDGER)} 条，PASS {n_pass}")
    png_sha = hashlib.sha256(png.read_bytes()).hexdigest()
    print(f"[lab] PNG sha256({png.name})={png_sha}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}（起 {t0.isoformat(timespec='seconds')}）")
    if n_pass != len(LEDGER):
        sys.exit(1)


if __name__ == "__main__":
    main()
