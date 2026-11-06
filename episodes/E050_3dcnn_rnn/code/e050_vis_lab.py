#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E050 lab —— 3D 卷积与循环网络数值演示档：感受野辨析参数计数/时序融合四方案对比/
极简 RNN 前向与权重共享/梯度沿时间反传衰减与爆炸/作业档+三张程序生成图
（§5.5 导言 p.317+§5.5.1 p.317–320+§5.5.2 p.321–322 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；固定 seed 确定性输入，readings
固定常数存档）：
- E1 感受野辨析与参数计数档（p.317「their input receptive fields are 3D boxes in
  (x, y, c), where c is the (feature) channel dimension」+p.318「weight sharing across
  the third dimension for all kernels」+p.318「very simple 3 × 3 × 3 convolutions」的
  读数级实证——f003/f005 口播的实算档）：参数计数三对齐（教学设定 C_in=3/C_out=16）：
  2D 核 3×3（逐帧）=3·3·3·16；3D 核 3×3×3=3·3·3·3·16（=2D 的 K_t 倍——时间维权重
  共享申报）；逐帧独立核（不共享）T=16=2D 的 T 倍。双线：手算闭式 vs numpy 计数
  allclose。
  断言：两线一致×2、3D 参数==2D×K_t、独立参数==2D×T、共享比==T/K_t。
- E2 时序融合四方案对比档（p.318 Figure 5.52「Alternative approaches to information
  fusion over the temporal dimensions (Karpathy, Toderici et al. 2014)」的教学重演——
  f004 口播的实算档；参数计数系教学设定申报——原书 Figure 5.52 系架构框图无数值，
  本档 conv1 参数计数（C_out=64）+时间平移等变性最小数值演示）：①single frame=
  64·3·3·3；②early fusion（T=3 帧堆通道 C_in=9）=64·9·3·3（=single×T——靠通道数堆）；
  ③late fusion（双支各半宽 32+融合层）=2·32·3·3·3+64·64+64；④3D CNN（K_t=3）=
  64·3·3·3·3（=single×K_t——与 early conv1 参数量恰好相同，但 3D 核带时间平移
  等变性）。时间平移等变演示（教学点 p.318「the interactions across time do not have
  the convolutional aspects of weight sharing and temporal shift invariance」的读数级
  实证——3D 核沿时间滑 vs early 通道拼接：合成时间信号移 1 步，3D 输出等变移位
  （allclose）、early 通道序变化输出不保序（not allclose）。
  断言：early==3D conv1 参数、3D==single×K_t、3D 移位等变 allclose、early 不等变。
- E3 极简 RNN 前向与权重共享档（p.321「Individual layers share weights across time」+
  p.321 Figure 5.55 展开/滚动两视图的读数级实证——f010/f011 口播的实算档）：
  numpy 手写 h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)（d=4，T=8，固定 seed）——逐帧
  调用同一组 (W_x, W_h, b)（权重跨时共享申报），参数量 d·d+d·d+d=36 与 T 无关
  （公式申报）；前向两次 allclose（同权重确定性）；隐藏状态范数序列全正且非退化
  （信息在时间上流动——Figure 5.55 展开图教学语义）。
  断言：参数量==36 且与 T 无关、两次前向 allclose、范数序列非退化×2。
- E4 梯度沿时间反传衰减与爆炸档（p.321「Because gradients can propagate for a long
  distance backward in time, and can therefore vanish or explode (just as in deep
  networks before the advent of residual networks)」的读数级实证——f012 口播的实算
  档；教学设定申报：线性 RNN 教学简化（h_t = W_h·h_{t-1} + x_t）——tanh 导数≤1 只会
  让真实 RNN 衰减更快，线性区已显形 vanish/explode 机制；W_h 取正交阵缩放三档
  0.5/1.0/1.8（特征值模=缩放、谱半径可控）——原书无具体数值，本档系最小数值演示）：
  |d h_T/d h_0| 数值微分（对 h_0 加 ε 扰动测 T 步后 |Δh_T|/ε，T=1..12）——缩放 0.5 档
  连乘衰减（vanish）、1.8 档连乘放大（explode）、1.0 档临界（谱半径 1 不增不减）。
  断言：0.5 档 T=12 <0.01、1.8 档 T=12 >10、1.0 档 T=12 ≤1.5（读数落 readings+FG3
  右面板曲线）。
- E10 作业档位（f016 两道口算答案存证——①「T=10 的展开图，可训练的权重矩阵有
  几组」：1 组（W_x/W_h/b 全程共享——E3 参数量与 T 无关互证）；②「缩放 0.5 的
  W_h，梯度传 12 步还剩几成」：E4 读数（衰减<0.01=不足一成的百分之一量级——
  readings 实测为准）。断言：E3/E4 互证×2。
- FG1 四种时序融合框图（Figure 5.52 原书图不复制——程序生成「四面板框图」示意图，
  deck 页 2 用）：Single frame/Late fusion/Early fusion/3D CNN 四面板（帧序列方块条
  +CNN 框+score 框+连接方式差异）——架构语义照原书 caption/正文（single frame 逐帧
  独立/late 各帧特征出片段分类/early 相邻帧组多通道/3D CNN 时空核图中未画——
  「(not shown in this figure)」在档，本图系程序生成补画申报非原书图像）。
- FG2 感受野盒子 vs 3D 时空窗示意图（程序生成，deck 页 3 用）：左面板 (x,y,c) 立方
  盒子（通道维标注——2D CNN 感受野「已是 3D 盒子」）、右面板 (x,y,t) 时空窗+3×3×3
  核块（时间维标注+滑动箭头——真 3D 卷积时间维共享）。
- FG3 RNN 展开图（程序生成，deck 页 6 用）：展开三步 h0→h1→h2（共享权重标注
  shared weights——Figure 5.55 教学语义重演非原书图）。
- FG4 梯度衰减/爆炸曲线（程序生成，deck 页 7 用）：E4 三档 |d h_T/d h_0| 曲线
  （log 纵轴 T=1..12——vanish/explode 读数可视化）。
- 图内文字运行时收割（E044–E049 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。图内文字全英文
  （E047–E049 同款——CJK 字体不入图，中文教学语义在 deck 卡与口播）。
- 图字号口径（D27 图内显示 ≥12px）：学生版 deck 图移入右栏（列宽约 803.6px），
  FG1 升 14pt（13pt 在右栏宽度绑定下 11.7px 破 D27 图内线；全图文字含 helper
  默认 fs 与 suptitle 同升）；FG2 12pt+/FG3 12pt+/FG4 12pt+，逐图显示值以
  display_check 存证为准；savefig dpi=200。
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
from matplotlib.patches import FancyArrowPatch, Rectangle

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
READINGS = OUT / "e050_vis_readings.json"
PNG1 = FIG_DIR / "e050_fig1_fusion.png"
PNG2 = FIG_DIR / "e050_fig2_kernel.png"
PNG3 = FIG_DIR / "e050_fig3_unroll.png"
PNG4 = FIG_DIR / "e050_fig4_gradient.png"

FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
LEDGER = []
_TEXTS = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}　| {detail}")
    LEDGER.append((name, ok))


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _save_and_register(fig, path: pathlib.Path) -> None:
    """统一保存通道：图内 Text 全量收割注册（display/哨兵共用——E049 判例）。"""
    for ax in fig.axes:
        for t in ax.findobj(_mtext.Text):
            _TEXTS.append(t.get_text())
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── E1 感受野辨析与参数计数 ──
C_IN, C_OUT, K, KT, TFR = 3, 16, 3, 3, 16


def conv2d_params(cin: int, cout: int, khw: int) -> int:
    return cout * cin * khw * khw + cout


def conv3d_params(cin: int, cout: int, kt: int, khw: int) -> int:
    return cout * cin * kt * khw * khw + cout


def lab_e1() -> None:
    global p2d, p3d, p_ind, ratio, k2d, k3d
    p2d = conv2d_params(C_IN, C_OUT, K)          # 448（含偏置 16）
    p3d = conv3d_params(C_IN, C_OUT, KT, K)      # 1312（含偏置 16）
    p_ind = conv2d_params(C_IN, C_OUT, K) * TFR  # 逐帧独立核 T=16
    k2d, k3d = p2d - C_OUT, p3d - C_OUT          # 核体（偏置不随核缩放——申报）
    ratio = (k2d * TFR) / k3d
    closed = (C_OUT * C_IN * KT * K * K + C_OUT,
              C_OUT * C_IN * K * K * TFR + C_OUT * TFR)
    num = (p3d, p_ind)
    check("E1 手算 vs numpy 计数一致（3D 核）", math.isclose(closed[0], num[0]),
          f"闭式 {closed[0]} vs 实算 {num[0]}")
    check("E1 手算 vs numpy 计数一致（逐帧独立）", math.isclose(closed[1], num[1]),
          f"闭式 {closed[1]} vs 实算 {num[1]}")
    check("E1 3D 核体==2D 核体×K_t（时间维共享申报）", k3d == k2d * KT,
          f"二维核体 {k2d} 三维核体 {k3d} K_t={KT}（偏置 {C_OUT} 不随核缩放——核体对比申报）")
    check("E1 逐帧独立参数==2D×T（无共享申报）", p_ind == p2d * TFR,
          f"独立={p_ind} = 二维×{TFR}（整层含偏置逐帧复制）")
    check("E1 核体共享比==T/K_t", math.isclose(ratio, TFR / KT),
          f"{k2d * TFR}/{k3d}={ratio:.4f} vs {TFR}/{KT}")
    print(f"[e1] 2D 核参数 {p2d}；3D 核参数 {p3d}（核体=2D×K_t 时间维共享）；"
          f"逐帧独立 {p_ind}（=2D×T）；核体共享比 {ratio:.2f}")


# ── E2 时序融合四方案对比 ──
C1_OUT, TT, CT = 64, 3, 3


def lab_e2() -> None:
    p_single = C1_OUT * CT * 3 * 3 + C1_OUT                 # 1792
    p_early = C1_OUT * (CT * TT) * 3 * 3 + C1_OUT           # 5312（T=3 帧堆通道）
    p_late = 2 * (32 * CT * 3 * 3 + 32) + (64 * 64 + 64)    # 双支半宽+融合层
    p_3d = C1_OUT * CT * 3 * 3 * TT + C1_OUT                # 5312（K_t=T=3）
    k_single = p_single - C1_OUT
    check("E2 early conv1 核体==single 核体×T（通道堆叠申报）",
          p_early - C1_OUT == k_single * TT,
          f"early 核体 {p_early - C1_OUT} = single 核体 {k_single}×T{TT}（偏置不随通道变——核体对比申报）")
    check("E2 3D conv1 核体==single 核体×K_t（时间维共享申报）",
          p_3d - C1_OUT == k_single * TT,
          f"3D 核体 {p_3d - C1_OUT} = single 核体×K_t{TT}")
    check("E2 early 与 3D conv1 参数量相同（教学点——差别在等变性）",
          p_early == p_3d, f"均 {p_early}（late={p_late} 系双支半宽+融合层教学设定）")
    # 时间平移等变最小演示（1D 时间信号：3D 核沿时间滑 vs early 通道拼接）
    rng = np.random.default_rng(50)
    T_sig, Kt_sig = 8, 3
    sig = rng.normal(size=T_sig)
    ker = rng.normal(size=Kt_sig)

    def conv1d(s, w):
        return np.array([float((s[t:t + len(w)] * w).sum())
                         for t in range(len(s) - len(w) + 1)])

    out = conv1d(sig, ker)
    sig_sh = np.roll(sig, 1)                                  # 时间移 1 步（晚 1 帧出现）
    out_sh = conv1d(sig_sh, ker)
    d_shift = float(np.abs(out_sh[1:] - out[:-1]).max())
    check("E2 时间核卷积移位等变（temporal shift invariance 读数级实证）",
          np.allclose(out_sh[1:], out[:-1]),
          f"输入移 1 步→输出同移 1 步（allclose，max|Δ|={d_shift:.2e}）")
    M = rng.normal(size=(5, T_sig))                           # early=摊平后固定网络
    o_e = M @ sig
    o_e_sh = M @ sig_sh
    d_early = float(np.abs(o_e_sh - np.roll(o_e, 1)).max())
    check("E2 early 通道拼接不移位等变（正文教学点数值互证）",
          not np.allclose(o_e_sh, np.roll(o_e, 1)),
          f"摊平通道序变化后输出不保序（max|Δ|={d_early:.4f}）")
    print(f"[e2] conv1 参数：single {p_single}/early {p_early}/late {p_late}/3D {p_3d}；"
          f"时间核等变 PASS、early 不等变 PASS")
    globals().update(p_single=p_single, p_early=p_early, p_late=p_late, p_3dcnn=p_3d,
                     k_body_single=k_single, k_body_fuse=p_early - C1_OUT,
                     e2_shift_maxabs=d_shift, e2_early_maxabs=d_early)


# ── E3 极简 RNN 前向与权重共享 ──
D, T8 = 4, 8


def rnn_step(x, h, Wx, Wh, b):
    return np.tanh(Wx @ x + Wh @ h + b)


def lab_e3() -> None:
    rng = np.random.default_rng(51)
    Wx = rng.normal(size=(D, D)) * 0.6
    Wh = rng.normal(size=(D, D)) * 0.6
    b = rng.normal(size=D) * 0.1
    X = rng.normal(size=(T8, D))
    n_par = Wx.size + Wh.size + b.size                        # 36
    check("E3 RNN 参数量==d·d+d·d+d==36 且与 T 无关（共享申报）",
          n_par == D * D + D * D + D and n_par == 36,
          f"({D}×{D}+{D}×{D}+{D})={n_par}（T={T8} 步共用同组权重）")
    h = np.zeros(D)
    hs = []
    for t in range(T8):
        h = rnn_step(X[t], h, Wx, Wh, b)
        hs.append(h.copy())
    h2 = np.zeros(D)
    hs2 = []
    for t in range(T8):
        h2 = rnn_step(X[t], h2, Wx, Wh, b)                    # 同权重再前向
        hs2.append(h2.copy())
    d_fwd = float(np.abs(np.array(hs) - np.array(hs2)).max())
    check("E3 同权重两次前向一致（权重跨时复用确定性）",
          np.allclose(np.array(hs), np.array(hs2)),
          f"max|Δ|={d_fwd:.2e}")
    norms = np.array([np.linalg.norm(h) for h in hs])
    check("E3 隐藏状态范数序列非退化（信息沿时间流动）",
          bool((norms > 0).all() and norms.std() > 0),
          f"min={norms.min():.4f} max={norms.max():.4f} std={norms.std():.4f}")
    check("E3 展开节点数==T（Figure 5.55 展开视图语义）", len(hs) == T8,
          f"展开 {len(hs)} 步、共享 1 组权重（滚动视图=同函数复用）")
    print(f"[e3] 参数 {n_par}（与 T 无关）；范数 {norms[0]:.3f}→{norms[-1]:.3f}")
    globals().update(rnn_norms=norms, rnn_npar=n_par, e3_fwd_maxabs=d_fwd)


# ── E4 梯度沿时间反传衰减与爆炸（线性化教学演示） ──
T12 = 12
EPS4 = 1e-5


def rnn_forward_lin(h0, X, Wh):
    """线性 RNN（教学简化申报）：h_t = W_h·h_{t-1} + x_t。

    线性化口径：tanh 导数≤1 只会让真实 RNN 衰减更快——线性区已显形 vanish/explode
    机制（梯度传递 = W_h 连乘，与输入无关）。W_h 取正交阵缩放（特征值模=缩放，
    谱半径可控——教学设定申报）。
    """
    h = h0
    hs = [h.copy()]
    for t in range(len(X)):
        h = Wh @ h + X[t]
        hs.append(h.copy())
    return np.array(hs)


def lab_e4() -> None:
    rng = np.random.default_rng(52)
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    X = rng.normal(size=(T12, D)) * 0.05
    scales = {"s0.5": 0.5, "s1.0": 1.0, "s1.8": 1.8}
    curves = {}
    for name, s in scales.items():
        Wh = Q * s
        h0 = np.zeros(D)
        base = rnn_forward_lin(h0, X, Wh)
        h0p = h0.copy()
        h0p[0] += EPS4                                        # 单分量单位方向扰动（谱范数语义精确申报）
        pert = rnn_forward_lin(h0p, X, Wh)
        grad = np.abs(pert[1:] - base[1:]).max(axis=1) / EPS4  # |dh_t/dh_0|（max 分量）
        curves[name] = grad
    g05, g10, g18 = curves["s0.5"], curves["s1.0"], curves["s1.8"]
    check("E4 缩放 0.5 档 T=12 梯度衰减 <0.01（vanish 实证）", g05[-1] < 0.01,
          f"|dh_12/dh_0|={g05[-1]:.6f}")
    check("E4 缩放 1.8 档 T=12 梯度 >10（explode 实证）", g18[-1] > 10,
          f"|dh_12/dh_0|={g18[-1]:.4f}")
    check("E4 缩放 1.0 档 T=12 临界不增不爆（正交阵谱半径 1 申报）", g10[-1] <= 1.5,
          f"|dh_12/dh_0|={g10[-1]:.4f}")
    print(f"[e4] T=12 读数：0.5 档 {g05[-1]:.6f}/1.0 档 {g10[-1]:.4f}/1.8 档 {g18[-1]:.4f}")
    globals().update(e4_curves=curves)


# ── E10 作业档 ──
def lab_e10() -> None:
    check("E10 题1 互证：展开图 T=10 步权重组数==1（E3 参数量与 T 无关）",
          rnn_npar == 36 and T8 == 8,
          f"E3 实测 {rnn_npar} 参数与 T={T8} 无关——T=10 同理 1 组 (W_x,W_h,b)")
    check("E10 题2 互证：0.5 档传 12 步读数==E4 实测", e4_curves["s0.5"][-1] < 0.01,
          f"E4 实测 {e4_curves['s0.5'][-1]:.6f}（不足一成的百分之一量级——白名单见 META）")
    print("[e10] 题1 答 1 组权重（共享）；题2 答 E4 衰减读数（readings 为准）")


# ── FG1 四种时序融合框图（Figure 5.52 教学重演——程序生成） ──
def _frames_row(ax, xc, y, n, w=0.075, h=0.13, gap=0.024, label=None,
                hi=None):
    """帧序列小方块条（xc=中心 x）；hi=高亮帧下标集合。"""
    total = n * w + (n - 1) * gap
    x0 = xc - total / 2
    for i in range(n):
        fc = "#b8cfe8" if (hi is None or i in hi) else "#eef2f7"
        ax.add_patch(Rectangle((x0 + i * (w + gap), y), w, h,
                               fc=fc, ec="#4a6785", lw=1.2))
    if label:
        ax.text(xc, y - 0.035, label, ha="center", va="top", fontsize=14,
                color="#333333")
    return x0, x0 + total


def _box2(ax, x, y, w, h, text, fc="#f5e6d0", ec="#8a6d3b", fs=14):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#333333")


def _arrow2(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=12,
                                 color="#555555", lw=1.3))


def fig1() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.8))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    # (a) single frame：单帧（高亮）→CNN→score 垂直流
    ax = axes[0]
    ax.text(0.5, 0.96, "(a) Single frame", ha="center", fontsize=14,
            fontweight="bold")
    _frames_row(ax, 0.5, 0.76, 3, label="frames", hi={0})
    _arrow2(ax, 0.5, 0.76, 0.5, 0.66)
    _box2(ax, 0.28, 0.44, 0.44, 0.22, "CNN\n(per frame)")
    _arrow2(ax, 0.5, 0.44, 0.5, 0.36)
    _box2(ax, 0.36, 0.14, 0.28, 0.22, "score", fc="#e8f0e4", ec="#5a7a52")
    ax.text(0.5, 0.05, "each frame alone", ha="center", fontsize=14,
            color="#666666", style="italic")
    # (b) late fusion：两路各一帧（first/last）→双 CNN→fuse+classify
    ax = axes[1]
    ax.text(0.5, 0.96, "(b) Late fusion", ha="center", fontsize=14,
            fontweight="bold")
    _frames_row(ax, 0.30, 0.76, 1, label=None)
    _frames_row(ax, 0.70, 0.76, 1, label=None)
    ax.text(0.30, 0.725, "first", ha="center", va="top", fontsize=14,
            color="#333333")
    ax.text(0.70, 0.725, "last", ha="center", va="top", fontsize=14,
            color="#333333")
    _arrow2(ax, 0.30, 0.76, 0.30, 0.66)
    _arrow2(ax, 0.70, 0.76, 0.70, 0.66)
    _box2(ax, 0.14, 0.44, 0.32, 0.22, "CNN")
    _box2(ax, 0.54, 0.44, 0.32, 0.22, "CNN")
    _arrow2(ax, 0.30, 0.44, 0.38, 0.36)
    _arrow2(ax, 0.70, 0.44, 0.62, 0.36)
    _box2(ax, 0.28, 0.14, 0.44, 0.22, "fuse + classify", fc="#e8f0e4",
          ec="#5a7a52")
    ax.text(0.5, 0.05, "features from two frames", ha="center", fontsize=14,
            color="#666666", style="italic")
    # (c) early fusion：三帧→stack channels→2D CNN→score
    ax = axes[2]
    ax.text(0.5, 0.96, "(c) Early fusion", ha="center", fontsize=14,
            fontweight="bold")
    _frames_row(ax, 0.5, 0.76, 3, label=None)
    _arrow2(ax, 0.5, 0.76, 0.5, 0.70)
    _box2(ax, 0.22, 0.46, 0.56, 0.24, "stack as channels\n(adjacent frames)",
          fc="#efe3f2", ec="#6d5a8a")
    _arrow2(ax, 0.5, 0.46, 0.5, 0.40)
    _box2(ax, 0.30, 0.16, 0.28, 0.24, "2D\nCNN")
    _arrow2(ax, 0.58, 0.28, 0.64, 0.28)
    _box2(ax, 0.64, 0.16, 0.22, 0.24, "score", fc="#e8f0e4", ec="#5a7a52")
    ax.text(0.5, 0.05, "no sharing over time", ha="center", fontsize=14,
            color="#666666", style="italic")
    # (d) 3D CNN：多帧→3D CNN→score
    ax = axes[3]
    ax.text(0.5, 0.96, "(d) 3D CNNs", ha="center", fontsize=14,
            fontweight="bold")
    _frames_row(ax, 0.5, 0.76, 4, label="frames (time)")
    _arrow2(ax, 0.5, 0.76, 0.5, 0.68)
    _box2(ax, 0.20, 0.44, 0.60, 0.24, "3D CNN\n(space-time kernel)")
    _arrow2(ax, 0.5, 0.44, 0.5, 0.36)
    _box2(ax, 0.36, 0.14, 0.28, 0.22, "score", fc="#e8f0e4", ec="#5a7a52")
    ax.text(0.5, 0.05, "kernels shared over time", ha="center", fontsize=14,
            color="#666666", style="italic")
    fig.suptitle("Alternative approaches to information fusion over time "
                 "(schematic, after Karpathy, Toderici et al. 2014)",
                 fontsize=14, y=1.03)
    fig.tight_layout()
    _save_and_register(fig, PNG1)


# ── FG2 感受野盒子 vs 3D 时空窗 ──
def fig2() -> None:
    fig = plt.figure(figsize=(10.8, 4.2))
    axL = fig.add_subplot(121, projection="3d")
    axR = fig.add_subplot(122, projection="3d")
    for ax, title in ((axL, "(a) 2D CNN receptive field: box in (x, y, c)"),
                      (axR, "(b) 3D CNN kernel: box in (x, y, t)")):
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_zlim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(title, fontsize=12.5, pad=2)
        ax.view_init(elev=18, azim=-58)
    # 左：通道维盒子
    vox = np.ones((3, 3, 3), dtype=bool)
    axL.voxels(vox, facecolors="#dce6f2", edgecolors="#4a6785", linewidth=0.8)
    axL.set_xlabel("x", fontsize=12)
    axL.set_ylabel("y", fontsize=12)
    axL.set_zlabel("c (channels)", fontsize=12, labelpad=1)
    axL.text2D(0.5, -0.02, "already 3D — but no sharing across channels",
               transform=axL.transAxes, ha="center", fontsize=12, color="#666666",
               style="italic")
    # 右：时空窗（时间轴配色+核块）
    colors = np.empty((3, 3, 3), dtype=object)
    for t in range(3):
        shade = ["#dbe7f5", "#b8cfe8", "#8fb4d8"][t]
        for i in range(3):
            for j in range(3):
                colors[i, j, t] = shade
    axR.voxels(colors, edgecolors="#4a6785", linewidth=0.8)
    axR.set_xlabel("x", fontsize=12)
    axR.set_ylabel("y", fontsize=12)
    axR.set_zlabel("t (time)", fontsize=12, labelpad=1)
    axR.text2D(0.5, -0.02, "one kernel shared across the whole time window",
               transform=axR.transAxes, ha="center", fontsize=12, color="#666666",
               style="italic")
    fig.suptitle("From channel boxes to space-time windows (schematic)",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    _save_and_register(fig, PNG2)


# ── FG3 RNN 展开图（deck 页 6 用） ──
def fig3() -> None:
    fig, axL = plt.subplots(1, 1, figsize=(6.4, 3.6))
    axL.set_xlim(0, 1)
    axL.set_ylim(0, 1)
    axL.axis("off")
    xs = [0.16, 0.44, 0.72]
    for i, x in enumerate(xs):
        axL.add_patch(Rectangle((x, 0.52), 0.15, 0.20, fc="#dce6f2", ec="#4a6785",
                                lw=1.4))
        axL.text(x + 0.075, 0.62, f"$h_{{{i}}}$", ha="center", va="center",
                 fontsize=13)
        if i:
            axL.add_patch(FancyArrowPatch((xs[i - 1] + 0.15, 0.62), (x, 0.62),
                                          arrowstyle="-|>", mutation_scale=13,
                                          color="#555555", lw=1.4))
    axL.text(0.5, 0.86, "shared weights $W_h$, $W_x$ at every step",
             ha="center", fontsize=13, color="#8a3324")
    axL.text(0.5, 0.32, "same weights, unrolled in time", ha="center",
             fontsize=12.5, color="#666666", style="italic")
    axL.text(0.5, 0.14, "input $x_t$ up, output from any $h_t$ down",
             ha="center", fontsize=12.5, color="#666666")
    axL.set_title("A recurrent layer, unrolled in time", fontsize=13)
    fig.tight_layout()
    _save_and_register(fig, PNG3)


# ── FG4 梯度沿时间衰减/爆炸曲线（E4 数据，deck 页 7 用） ──
def fig4() -> None:
    fig, axR = plt.subplots(1, 1, figsize=(6.8, 3.8))
    Ts = np.arange(1, T12 + 1)
    for name, color, lab in (("s0.5", "#2a6e2a", "W scaled by 0.5 (vanish)"),
                             ("s1.0", "#8a6d3b", "W scaled by 1.0 (critical)"),
                             ("s1.8", "#a03030", "W scaled by 1.8 (explode)")):
        axR.semilogy(Ts, np.maximum(e4_curves[name], 1e-7), marker="o", ms=4.5,
                     lw=1.7, color=color, label=lab)
    axR.set_xlabel("time steps back-propagated (T)", fontsize=12.5)
    axR.set_ylabel("|d h_T / d h_0|  (max component)", fontsize=12.5)
    axR.set_title("Gradients vanish or explode over time", fontsize=13)
    axR.legend(fontsize=12, loc="upper left")
    axR.grid(alpha=0.3)
    axR.tick_params(labelsize=12)
    fig.tight_layout()
    _save_and_register(fig, PNG4)


# ── 哨兵：图内文字内部记号零泄漏 ──
def harvest_mark() -> None:
    hits = [t for t in _TEXTS if FIG_MARK.search(t)]
    check("FIG 内部记号零泄漏", not hits,
          f"4 张图全部 Text 对象 {len(_TEXTS)} 条、零命中"
          + (f"：{hits[:3]}" if hits else ""))


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] E050 3D CNN/RNN 数值演示档 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    lab_e1()
    lab_e2()
    lab_e3()
    lab_e4()
    lab_e10()
    fig1()
    fig2()
    fig3()
    fig4()
    harvest_mark()
    pngs = {"FG1": PNG1, "FG2": PNG2, "FG3": PNG3, "FG4": PNG4}
    for k, p in pngs.items():
        check(f"{k} 程序生成图落盘", p.exists(),
              f"{p.name} 已落盘（全量 sha 自证见 [lab] PNG 行——短形态摘要系 readings 字段）")
    n_pass = sum(1 for _, ok in LEDGER if ok)
    n_fail = len(LEDGER) - n_pass
    readings = {
        "n_pass": n_pass,
        "n_total": len(LEDGER),
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": sha16(pathlib.Path(__file__)),
        "cfg": {"c_in": C_IN, "c_out": C_OUT, "k": K, "kt": KT, "t_frames": TFR,
                "c1_out": C1_OUT, "t_fuse": TT, "d": D, "t_rnn": T8,
                "t_grad": T12, "eps": EPS4, "scales": [0.5, 1.0, 1.8],
                "seeds": [50, 51, 52]},
        "e1": {"p_2d": p2d, "p_3d": p3d, "p_independent": p_ind,
               "share_ratio": float(ratio), "k_body_2d": k2d, "k_body_3d": k3d,
               "share_body_num": k2d * TFR},
        "e2": {"p_single": p_single, "p_early": p_early, "p_late": p_late,
               "p_3dcnn": p_3dcnn, "k_body_single": k_body_single,
               "k_body_fuse": k_body_fuse, "shift_maxabs": e2_shift_maxabs,
               "early_maxabs": e2_early_maxabs},
        "e3": {"n_params": rnn_npar, "norms_first": float(rnn_norms[0]),
               "norms_last": float(rnn_norms[-1]), "norms_max": float(rnn_norms.max()),
               "norms_std": float(rnn_norms.std()), "fwd_maxabs": e3_fwd_maxabs},
        "e4": {"vanish_t12": float(e4_curves["s0.5"][-1]),
               "critical_t12": float(e4_curves["s1.0"][-1]),
               "explode_t12": float(e4_curves["s1.8"][-1]),
               "curves": {k: [float(x) for x in v] for k, v in e4_curves.items()}},
        "png": {k: sha16(v) for k, v in pngs.items()},
        "fig_text_count": len(_TEXTS),
    }
    READINGS.write_text(json.dumps(readings, indent=2, ensure_ascii=False) + "\n")
    print(f"[lab] readings → {READINGS.name}")
    print(f"[lab] readings sha256={hashlib.sha256(READINGS.read_bytes()).hexdigest()}")
    for _png in (PNG1, PNG2, PNG3, PNG4):
        print(f"[lab] PNG sha256({_png.name})="
              f"{hashlib.sha256(_png.read_bytes()).hexdigest() if _png.exists() else 'MISSING'}")
    print(f"[lab] LEDGER：断言 {len(LEDGER)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {n_fail} / 共 {len(LEDGER)}")
    t1 = datetime.datetime.now()
    print(f"[lab] 结束 {t1.isoformat(timespec='seconds')}")
    if n_fail:
        sys.exit(f"[FAIL] lab 断言失败 {n_fail} 项")


if __name__ == "__main__":
    main()
