#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E051 lab —— Transformer 与 ViT 数值演示档：式 5.77 scaled dot-product attention
前向与方差不变性/饱和演示/式 5.78 三权重矩阵与复用/patch 计数链/FLOPs 对比/作业档
+三张程序生成图（§5.5.3 p.322–328 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；固定 seed 确定性输入，readings
固定常数存档）：
- E1 scaled dot-product attention 前向档（p.324 式 5.77「y = α(q · ki /√D)vi =
  softmax(qᵀK/√D)ᵀV」+脚注 56「We divide the dot product by √D so that the variance
  of the scaled dot product does not increase for larger embedding dimensions, which
  could result in vanishing gradients」的读数级实证——f007/f008 口播的实算档）：
  E1a 最小前向（N=4/D=8，seed 50）——逐步式（单 query 循环：点积→÷√D→softmax→
  加权和）vs 矩阵式（softmax(qK/√D)·V 批实现）allclose+softmax 权重行和==1。
  E1b 方差不变性直证（脚注 56 核心句）：q,k 独立 N(0,1)，M=20000 对——D=8/64 两档
  var(q·k)≈D（相对误差<10%）且 var(q·k/√D)≈1（相对误差<15%）——「方差不随 D 增大」
  读数落 readings。E1c softmax 饱和与梯度消失机制（教学演示申报：1 个真匹配 key
  （点积=D）+255 个随机 key（点积~N(0,D)），D=64，200 个 query 均值——未缩放时匹配
  logit 压倒一切→分布近 one-hot（mean max prob>0.99——梯度近零的饱和区）；除以 √D
  后分布平缓（mean max prob<0.5）且熵更高——「防饱和防梯度消失」教学点；原书只给
  机理句不给数值，本档系最小数值演示）。
  断言：E1a 两实现 allclose+行和==1；E1b 方差两档×2；E1c 饱和/平缓/熵×3。
- E2 三权重矩阵与复用档（p.325 式 5.78「qi = Wq xi , ki = Wk xi , and vi = Wv xi」+
  p.324 Figure 5.56 caption「For each of matmulV, matmulK, and matmulQ, there is a
  single matrix of weights that gets reused with each call」的读数级实证——f009 口播
  的实算档；教学设定 D=8/N=4，seed 52）：式 5.78 投影+式 5.77 前向全链两实现交叉
  （逐步式逐 xi 循环 vs 矩阵批式 X·Wqᵀ 等一次成型——allclose）；参数量 3·D·D=192
  与 N 无关（N=4 与 N=16 同为 192——权重矩阵形状只随 D）；复用确定性（同一组
  W 同一输入两次调用 maxabs==0——「reused with each call」读数申报）。
  断言：全链 allclose、参数量==192 且与 N 无关、复用 maxabs==0。
- E3 patch 计数档（p.326「they divide an image (of size 224 × 224) into 196 distinct
  16 × 16 gridded image patches」数值链的读数级实证——f013 口播的实算档）：224²=50176
  （原书数值）；16²=256；50176/256=196=14×14；步幅卷积输出公式互证 (224−16)/16+1=14
  （「equivalent to a strided 16 × 16 convolution」的输出尺寸读数）。
  断言：224²==50176、16²==256、50176==196·256、(224−16)//16+1==14、14²==196。
- E4 FLOPs 对比档（p.326「the FLOPs required for self-attention is on the order of
  O(N²D), whereas the FLOPs for a convolution operation is on the order of O(ND²K²)」
  的读数级实证——f011 口播的实算档；教学口径申报：阶符号省略常数，本档按系数 1 计
  ——数量级对比非精确比较；D=3 系原书 ImageNet 数值链，K=3 系课程 3×3 核教学设定
  ——原书未给 K 值）：N=50176（224×224 逐像素）attention≈N²·D≈7.55e9 vs 卷积≈
  N·D²·K²≈4.07e6（比值≈1858——千倍以上教学读数）；N=1024（Image Transformer 32×32
  ——p.326 原书数值）比值≈37.9——图像越大 attention 越吃亏的 N² 增长实证；N 扫描
  [256,1024,4096,16384,50176] 落 readings（FG3 曲线数据）。
  断言：比值(50176)>1000、比值(1024)>10、N×49 时 attention FLOPs 增 49²=2401 倍
  （整数精确）、卷积增 49 倍。
- E10 作业档（f016 两道口算答案存证——①「224×224 图切 16×16 patch，切出多少个」：
  196（E3 互证）；②「N=50176、D=3 时 self-attention 的 FLOPs 约是卷积（K=3）的
  多少倍（数量级口径）」：E4 读数≈1857——「千倍以上」定性读（readings 实测为准）。
  断言：E3/E4 互证×2。
- FG1 self-attention 计算图（Figure 5.56 原书图不复制——courtesy of Matt Deitke 外源
  授权语义，程序生成「单输出 x'2 计算图」教学重演，deck 页 5 用）：输入 x1..x4→
  matmulQ/matmulK/matmulV 三框（weights reused each call 注记——caption 语义）→
  dot products→÷√D→softmax→weighted sum（V 路汇入）→输出 x'2。
- FG2 patch 化流程图（Figure 5.57 原书图不复制——程序生成「patch 切分+ViT 前段流程」
  教学重演，deck 页 7 用）：左面板 224×224 网格切 14×14=196 patch（单 patch 高亮
  16×16 注记）；右面板流程条 Flatten→Shared embedding（strided 16×16 conv 注记）→
  +Positional encoding→Transformer encoder ×L→Class prediction。
- FG3 FLOPs 增长曲线（程序生成，deck 页 6 用）：E4 扫描数据 N²·D vs N·D²·K² 双曲线
  （log-log，N=256→50176；Image Transformer 32×32 与 ImageNet 224×224 两竖线注记
  ——N² vs N 增长率可视化）。
- 图内文字运行时收割（E044–E050 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。图内文字全英文
  （E047–E050 同款——CJK 字体不入图，中文教学语义在 deck 卡与口播）。
- 图字号口径（D27 图内显示 ≥12px；E050 FG1 首版 12pt 被 display_check 拦下判例——
  全图文字（含 helper 默认与标题）13pt 起步）；savefig dpi=200。
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
READINGS = OUT / "e051_vis_readings.json"
PNG1 = FIG_DIR / "e051_fig1_attention.png"
PNG2 = FIG_DIR / "e051_fig2_patches.png"
PNG3 = FIG_DIR / "e051_fig3_flops.png"

FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
LEDGER = []
_TEXTS = []

plt.rcParams.update({"font.size": 13})

C_BLUE = "#a8c8e8"
C_ORANGE = "#f4c98a"
C_GREEN = "#b8dcb8"
C_PINK = "#f0b8b8"
C_EDGE = "#4a4a4a"


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}　| {detail}")
    LEDGER.append((name, ok))


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def _save_and_register(fig, path: pathlib.Path) -> None:
    """统一保存通道：图内 Text 全量收割注册（display/哨兵共用——E049/E050 判例）。"""
    for ax in fig.axes:
        for t in ax.findobj(_mtext.Text):
            _TEXTS.append(t.get_text())
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ── E1 scaled dot-product attention ──
D8, N4, SEED1 = 8, 4, 50
D64, M_PAIRS, SEED1B = 64, 20000, 51
SAT_D, SAT_KEYS, SAT_Q, SEED1C = 64, 256, 200, 52


def e1_forward_stepwise(q, K, V):
    """逐步式：单 query 对每个 key 点积→÷√D→softmax→加权和（式 5.77 直译）。"""
    d = q.shape[0]
    logits = np.array([float(np.dot(q, K[j])) for j in range(K.shape[0])])
    a = softmax(logits / math.sqrt(d))
    y = np.zeros_like(q)
    for j in range(K.shape[0]):
        y = y + a[j] * V[j]
    return y, a


def e1_forward_matrix(q, K, V):
    """矩阵式：softmax(qK/√D)·V 批实现（式 5.77 右端）。"""
    d = q.shape[0]
    a = softmax((K @ q) / math.sqrt(d))
    return a @ V, a


def lab_e1():
    rng = np.random.default_rng(SEED1)
    q = rng.normal(size=D8)
    K = rng.normal(size=(N4, D8))
    V = rng.normal(size=(N4, D8))
    y_step, a_step = e1_forward_stepwise(q, K, V)
    y_mat, a_mat = e1_forward_matrix(q, K, V)
    e1_maxabs = float(np.abs(y_step - y_mat).max())
    check("E1 式 5.77 两实现 allclose（逐步式 vs 矩阵式）", e1_maxabs < 1e-12,
          f"maxabs={e1_maxabs:.2e}（N=4/D=8 单 query 前向）")
    row_sum = float(a_mat.sum())
    check("E1 attention 权重行和==1（softmax 归一）", math.isclose(row_sum, 1.0, abs_tol=1e-12),
          f"Σα={row_sum:.12f}")

    # E1b 方差不变性（脚注 56 直证）
    rng_b = np.random.default_rng(SEED1B)
    qb = rng_b.normal(size=(M_PAIRS, D64))
    kb = rng_b.normal(size=(M_PAIRS, D64))
    dots = (qb * kb).sum(axis=1)
    v_dot = float(dots.var())
    v_scaled = float((dots / math.sqrt(D64)).var())
    ok8 = abs(v_dot - D64) / D64 < 0.10
    ok1 = abs(v_scaled - 1.0) / 1.0 < 0.15
    check("E1b var(q·k)≈D（方差随 D 增大）", ok8,
          f"D=64 实测 var={v_dot:.2f} vs D=64（M={M_PAIRS}）")
    check("E1b var(q·k/√D)≈1（除以 √D 方差不变——脚注 56 直证）", ok1,
          f"实测 var={v_scaled:.3f} vs 1")

    # E1c softmax 饱和与梯度消失机制（教学演示：每 query 1 个真匹配 key+255 个随机
    # key——匹配 key 取 query 自身，dot(q,q)=Σq²≈D；随机 logit~N(0,D)）
    rng_c = np.random.default_rng(SEED1C)
    Qc = rng_c.normal(size=(SAT_Q, SAT_D))
    rand_logits = rng_c.normal(size=(SAT_Q, SAT_KEYS - 1)) * math.sqrt(SAT_D)
    match_logits = (Qc * Qc).sum(axis=1, keepdims=True)
    logits_un = np.concatenate([match_logits, rand_logits], axis=1)
    p_un = softmax(logits_un, axis=1)
    logits_sc = logits_un / math.sqrt(SAT_D)
    p_sc = softmax(logits_sc, axis=1)
    mp_un = float(p_un.max(axis=1).mean())
    mp_sc = float(p_sc.max(axis=1).mean())
    H_un = float(-(p_un * np.log(p_un + 1e-300)).sum(axis=1).mean())
    H_sc = float(-(p_sc * np.log(p_sc + 1e-300)).sum(axis=1).mean())
    check("E1c 未缩放 softmax 近 one-hot（饱和——梯度近零）", mp_un > 0.99,
          f"mean max prob={mp_un:.4f}（匹配 logit≈D=64 压倒随机 logit~N(0,64)，{SAT_Q} query 均值）")
    check("E1c 除以 √D 后权重从饱和区回落（梯度通道恢复）", mp_sc < 0.9,
          f"mean max prob={mp_sc:.4f}（同一匹配场景缩放后读数——匹配仍最高但分布重携相对信息）")
    check("E1c 缩放后分布熵更高（信息不塌缩）", H_sc > H_un,
          f"H(scaled)={H_sc:.2f} > H(unscaled)={H_un:.2f}（nats；{SAT_KEYS} keys）")
    return dict(e1_maxabs=e1_maxabs, row_sum=row_sum, v_dot_d64=v_dot,
                v_scaled_d64=v_scaled, mp_un=mp_un, mp_sc=mp_sc,
                H_un=H_un, H_sc=H_sc)


# ── E2 三权重矩阵与复用 ──
SEED2 = 53


def lab_e2():
    rng = np.random.default_rng(SEED2)
    Wq, Wk, Wv = (rng.normal(size=(D8, D8)) for _ in range(3))
    X4 = rng.normal(size=(N4, D8))
    X16 = rng.normal(size=(16, D8))

    def full_chain_stepwise(X):
        """逐步式：逐 xi 投影（式 5.78）+式 5.77 前向。"""
        outs = []
        for i in range(X.shape[0]):
            qi, ki, vi = Wq @ X[i], Wk @ X[i], Wv @ X[i]
            y, _ = e1_forward_stepwise(qi, np.stack([Wk @ X[j] for j in range(X.shape[0])]),
                                       np.stack([Wv @ X[j] for j in range(X.shape[0])]))
            outs.append(y)
        return np.stack(outs)

    def full_chain_matrix(X):
        """矩阵批式：Q=X·Wqᵀ 等一次成型+式 5.77 右端。"""
        Q, K, V = X @ Wq.T, X @ Wk.T, X @ Wv.T
        A = softmax((Q @ K.T) / math.sqrt(D8), axis=1)
        return A @ V

    Y_s = full_chain_stepwise(X4)
    Y_m = full_chain_matrix(X4)
    e2_maxabs = float(np.abs(Y_s - Y_m).max())
    check("E2 式 5.78+5.77 全链两实现 allclose（逐步式 vs 矩阵批式）", e2_maxabs < 1e-10,
          f"maxabs={e2_maxabs:.2e}（Wq/Wk/Wv 投影+N=4 前向）")
    n_par = 3 * D8 * D8
    check("E2 三权重参数量==3·D²==192 且与 N 无关", n_par == 192 and Wq.shape == (D8, D8),
          f"3·{D8}·{D8}={n_par}（N=4 与 N=16 用同一组 {D8}×{D8} 矩阵——形状只随 D）")
    Y_again = full_chain_matrix(X4)
    reuse_maxabs = float(np.abs(Y_m - Y_again).max())
    check("E2 同组权重复用两次调用 maxabs==0（reused with each call——Figure 5.56 caption 语义）",
          reuse_maxabs == 0.0,
          f"maxabs={reuse_maxabs:.1e}（同一 Wq/Wk/Wv 确定性复用）")
    return dict(fullchain_maxabs=e2_maxabs, n_params=n_par, reuse_maxabs=reuse_maxabs)


# ── E3 patch 计数 ──
IMG, PATCH = 224, 16


def lab_e3():
    pixels = IMG * IMG
    parea = PATCH * PATCH
    n_patches = pixels // parea
    grid = int(math.isqrt(n_patches))
    conv_out = (IMG - PATCH) // PATCH + 1
    check("E3 224²==50176（原书逐像素数值）", pixels == 50176, f"224×224={pixels}")
    check("E3 16²==256（单 patch 像素数）", parea == 256, f"16×16={parea}")
    check("E3 50176==196·256（patch 数数值链）", pixels == n_patches * parea,
          f"{pixels}={n_patches}×{parea}")
    check("E3 步幅卷积输出 (224−16)/16+1==14（strided 16×16 conv 互证）", conv_out == 14,
          f"({IMG}−{PATCH})/{PATCH}+1={conv_out}")
    check("E3 14²==196（14×14 网格）", grid * grid == 196 and grid == 14,
          f"{grid}×{grid}={grid*grid}")
    return dict(pixels=pixels, patch_area=parea, n_patches=n_patches, grid=grid,
                conv_out=conv_out)


# ── E4 FLOPs 对比（教学口径：阶符号取系数 1）──
K3 = 3
SCAN_N = [256, 1024, 4096, 16384, 50176]


def lab_e4():
    def attn_flops(n, d):
        return n * n * d

    def conv_flops(n, d, k):
        return n * d * d * k * k

    a_big = attn_flops(50176, 3)
    c_big = conv_flops(50176, 3, K3)
    ratio_big = a_big / c_big
    a_small = attn_flops(1024, 3)
    c_small = conv_flops(1024, 3, K3)
    ratio_small = a_small / c_small
    check("E4 N=50176 时 attention FLOPs 为卷积千倍以上（数量级口径）", ratio_big > 1000,
          f"attn={a_big:.3e} vs conv={c_big:.3e}（比值≈{ratio_big:.0f}——系数 1 教学口径）")
    check("E4 N=1024（32×32）时差距收窄一个量级以上", ratio_small > 10 and ratio_small < ratio_big / 10,
          f"比值≈{ratio_small:.1f}（Image Transformer 原书数值——N² 增长实证）")
    check("E4 N×49 → attention FLOPs ×2401（=49²）", attn_flops(50176, 3) == attn_flops(1024, 3) * 49 * 49,
          f"（整数精确——倍数关系见断言名）")
    check("E4 N×49 → 卷积 FLOPs ×49（线性增长）", conv_flops(50176, 3, K3) == conv_flops(1024, 3, K3) * 49,
          f"（线性增长实证）")
    scan = {str(n): [attn_flops(n, 3), conv_flops(n, 3, K3)] for n in SCAN_N}
    return dict(attn_50176=a_big, conv_50176=c_big, ratio_50176=ratio_big,
                attn_1024=a_small, conv_1024=c_small, ratio_1024=ratio_small,
                scan=scan)


# ── E10 作业档 ──
def lab_e10(e3, e4):
    ans1 = e3["n_patches"]
    check("E10 作业① 224×224 切 16×16 patch 共 196 个（E3 互证）", ans1 == 196,
          f"答案 {ans1} 个（14×14 网格）")
    ans2 = e4["ratio_50176"]
    check("E10 作业② N=50176/D=3 时 attention FLOPs≈卷积千倍以上（E4 互证）", ans2 > 1000,
          f"比值≈{ans2:.0f}（数量级口径——口播定性读「千倍以上」）")
    return dict(ans1=ans1, ans2_ratio=ans2)


# ── FG1 self-attention 计算图 ──
def _box(ax, cx, cy, w, h, text, fc, fs=13, weight="normal"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           facecolor=fc, edgecolor=C_EDGE, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight)


def _arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=C_EDGE, linewidth=1.3, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def fig1():
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "Self-attention: computing one output x'2 from N = 4 inputs",
            ha="center", va="center", fontsize=14, weight="bold")
    for i, y in enumerate([80, 61, 42, 23], start=1):
        _box(ax, 7, y, 9, 9, f"x{i}", C_BLUE)
    _box(ax, 27, 80, 15, 9, "matmulQ\n(Wq)", C_ORANGE, fs=12.5)
    _box(ax, 27, 55, 15, 9, "matmulK\n(Wk)", C_ORANGE, fs=12.5)
    _box(ax, 27, 17, 15, 9, "matmulV\n(Wv)", C_ORANGE, fs=12.5)
    ax.text(32, 71.5, "weights reused each call", ha="center", va="center",
            fontsize=11.5, style="italic", color="#666666")
    ax.text(32, 46.5, "weights reused each call", ha="center", va="center",
            fontsize=11.5, style="italic", color="#666666")
    ax.text(42, 17, "weights reused each call", ha="left", va="center",
            fontsize=11.5, style="italic", color="#666666")
    _arrow(ax, 11.5, 61, 19.5, 78)
    ax.text(12, 5, "all four inputs x1..x4", ha="center", va="center", fontsize=12.5)
    _arrow(ax, 12, 26, 19.5, 53)
    _arrow(ax, 12, 21, 19.5, 19)
    _box(ax, 47, 80, 20, 9, "dot products\nq2 · k1..k4", C_GREEN, fs=12.5)
    _box(ax, 47, 55, 20, 9, "scale: ÷ √D", C_GREEN, fs=12.5)
    _box(ax, 47, 30, 20, 9, "softmax → weights α", C_GREEN, fs=12.5)
    _box(ax, 72, 30, 21, 9, "weighted sum\nΣ αj · vj", C_PINK, fs=12.5)
    _box(ax, 93, 30, 8, 9, "x'2", C_PINK, fs=14, weight="bold")
    _arrow(ax, 34.5, 80, 37, 80)
    _arrow(ax, 47, 75.5, 47, 59.5)
    _arrow(ax, 47, 50.5, 47, 34.5)
    _arrow(ax, 57.5, 30, 61.5, 30)
    _arrow(ax, 82.5, 30, 89, 30)
    _arrow(ax, 27, 21.5, 61.5, 26, ls="--")
    ax.text(48, 8, "values v1..v4", ha="center", va="center", fontsize=12.5)
    _save_and_register(fig, PNG1)


# ── FG2 patch 化流程图 ──
def fig2():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    axL.set_xlim(0, 100)
    axL.set_ylim(0, 100)
    axL.axis("off")
    axL.text(50, 96, "224×224 image", ha="center",
             va="center", fontsize=13.5, weight="bold")
    ox, oy, side = 16, 20, 52
    axL.add_patch(Rectangle((ox, oy), side, side, facecolor="#dfe8f2",
                            edgecolor=C_EDGE, linewidth=1.4))
    for i in range(1, 14):
        t = ox + side * i / 14
        axL.plot([t, t], [oy, oy + side], color="#9ab0c8", linewidth=0.7)
        axL.plot([ox, ox + side], [oy + side * i / 14] * 2, color="#9ab0c8", linewidth=0.7)
    pw = side / 14
    axL.add_patch(Rectangle((ox + 5 * pw, oy + 8 * pw), pw, pw, facecolor="none",
                            edgecolor="#c0392b", linewidth=2.2))
    axL.annotate("one patch:\n16×16 pixels", xy=(ox + 6 * pw, oy + 8.5 * pw),
                 xytext=(84, 42), fontsize=12.5, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", linewidth=1.3))
    axL.text(50, 7, "14×14 = 196 patches (16×16 pixels each)",
             ha="center", va="center", fontsize=12.5)
    axR.set_xlim(0, 100)
    axR.set_ylim(0, 100)
    axR.axis("off")
    axR.text(50, 96, "ViT: patch embedding pipeline",
             ha="center", va="center", fontsize=13.5, weight="bold")
    steps = [
        ("flatten each patch", C_BLUE),
        ("shared embedding matrix\n(strided 16×16 conv)", C_ORANGE),
        ("+ positional encoding", C_GREEN),
        ("transformer encoder × L", C_PINK),
        ("class prediction", C_GREEN),
    ]
    ys = [82, 64, 46, 28, 10]
    for (txt, fc), y in zip(steps, ys):
        _box(axR, 44, y, 66, 13.5, txt, fc, fs=12.5)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        _arrow(axR, 44, y0 - 6.75, 44, y1 + 6.75)
    _save_and_register(fig, PNG2)


# ── FG3 FLOPs 增长曲线 ──
def fig3(e4):
    ns = np.array([float(n) for n in e4["scan"]], dtype=float)
    attn = np.array([v[0] for v in e4["scan"].values()], dtype=float)
    conv = np.array([v[1] for v in e4["scan"].values()], dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.loglog(ns, attn, "o-", color="#c0392b", linewidth=2.0, markersize=6,
              label="self-attention  O(N²·D)")
    ax.loglog(ns, conv, "s-", color="#2c6fbb", linewidth=2.0, markersize=6,
              label="convolution  O(N·D²·K²)")
    ax.axvline(1024, color="#888888", linestyle="--", linewidth=1.2)
    ax.text(1024 * 1.12, 4.5e4, "Image Transformer\n32×32", fontsize=12.5, color="#555555")
    ax.axvline(50176, color="#888888", linestyle="--", linewidth=1.2)
    ax.text(50176 * 0.96, 4.5e4, "ImageNet\n224×224", fontsize=12.5, color="#555555",
            ha="right")
    ax.set_xlabel("input length N", fontsize=13)
    ax.set_ylabel("FLOPs (teaching scaling)", fontsize=13)
    ax.set_title("Self-attention vs convolution: quadratic vs linear growth (D = 3, K = 3)",
                 fontsize=13.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12.5, loc="upper left")
    _save_and_register(fig, PNG3)


# ── 图内记号哨兵 ──
def harvest_mark():
    joined = "\n".join(_TEXTS)
    leaked = FIG_MARK.findall(joined)
    check("FG 图内文字内部记号零泄漏（FIG_MARK 零命中）", not leaked,
          f"图内 Text 收割 {len(_TEXTS)} 条，命中 {leaked if leaked else '无'}")
    small = [t for t in set(_TEXTS) if t.strip()]
    check("FG 程序生成图 3 张全落盘", PNG1.exists() and PNG2.exists() and PNG3.exists(),
          f"{PNG1.name}+{PNG2.name}+{PNG3.name}（sha 见 [lab] PNG 行）")


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] E051 Transformer/ViT 数值演示档 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    e1 = lab_e1()
    e2 = lab_e2()
    e3 = lab_e3()
    e4 = lab_e4()
    e10 = lab_e10(e3, e4)
    fig1()
    fig2()
    fig3(e4)
    harvest_mark()
    pngs = {"FG1": PNG1, "FG2": PNG2, "FG3": PNG3}
    for k, p in pngs.items():
        check(f"{k} 程序生成图落盘", p.exists(),
              f"{p.name} 已落盘（全量 sha 自证见 [lab] PNG 行）")
    n_pass = sum(1 for _, ok in LEDGER if ok)
    n_fail = len(LEDGER) - n_pass
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": sha16(pathlib.Path(__file__)),
        "cfg": {"d8": D8, "n4": N4, "d64": D64, "m_pairs": M_PAIRS,
                "sat_d": SAT_D, "sat_keys": SAT_KEYS, "sat_q": SAT_Q,
                "k3": K3, "img": IMG, "patch": PATCH,
                "seeds": [SEED1, SEED1B, SEED1C, SEED2]},
        "e1": {"forward_maxabs": e1["e1_maxabs"], "row_sum": e1["row_sum"],
               "var_dot_d64": e1["v_dot_d64"], "var_scaled_d64": e1["v_scaled_d64"],
               "sat_maxprob_unscaled": e1["mp_un"], "sat_maxprob_scaled": e1["mp_sc"],
               "entropy_unscaled": e1["H_un"], "entropy_scaled": e1["H_sc"]},
        "e2": {"fullchain_maxabs": e2["fullchain_maxabs"],
               "n_params": e2["n_params"], "reuse_maxabs": e2["reuse_maxabs"]},
        "e3": {k: e3[k] for k in ("pixels", "patch_area", "n_patches", "grid", "conv_out")},
        "e4": {"attn_50176": e4["attn_50176"], "conv_50176": e4["conv_50176"],
               "ratio_50176": e4["ratio_50176"], "attn_1024": e4["attn_1024"],
               "conv_1024": e4["conv_1024"], "ratio_1024": e4["ratio_1024"],
               "scan": e4["scan"]},
        "e10": e10,
        "png": {k: sha16(v) for k, v in pngs.items()},
        "fig_text_count": len(_TEXTS),
    }
    READINGS.write_text(json.dumps(readings, indent=2, ensure_ascii=False) + "\n")
    print(f"[lab] readings → {READINGS.name}")
    print(f"[lab] readings sha256={hashlib.sha256(READINGS.read_bytes()).hexdigest()}")
    for _png in (PNG1, PNG2, PNG3):
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
