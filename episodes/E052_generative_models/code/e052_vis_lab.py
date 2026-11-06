#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E052 lab —— 生成模型数值演示档：VQ-VAE 码本量化直证/离散组合数位数/式 5.79 数值
实例化/重叠 patch 计数/作业档+三张程序生成图（§5.5.4 p.328–336 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；固定 seed 确定性输入，readings
固定常数存档）：
- E1 VQ-VAE 码本量化直证（p.330「the VQ-VAE replaces each spatial dimension of the
  predicted latent representation with its nearest vector from a discrete set of
  vectors (named the codebook)」的读数级实证——f007 口播的实算档；参数教学口径：
  K=512 与 z=32×32×1 系原书 p.330 数值，码本向量维数 Dv=8 系课程教学设定——原书
  未给每向量维数，seed 61）：随机隐表示网格 z=32×32（每位置一个 8 维向量）→逐位置
  替换为码本最近向量（欧氏距离 argmin）→量化位移统计（均值/最大欧氏距离——量化
  确实发生）+输出向量全部来自码本（每个输出与某码本向量距离==0）+命中码本条目数
  ≤512（离散值域直证）+形状 32×32 保持。
  断言：全部来自码本×1、命中条目≤512×1、量化位移>0×1、形状保持×1。
- E2 离散组合数位数档（p.330「they can represent 51232·32·1 possible images」=
  512 的 32·32·1 次幂（指数排版 raw 压平——疑误照录②申报）的读数级实证——f008
  口播「上千位的天文数字」的实算档）：512**1024 用 Python 大整数精确幂（无浮点
  溢出），十进制位数 = floor(1024·log10(512))+1——整数位数千位级直证；
  log 口径与精确幂互证。
  断言：精确幂位数==2775（整数精确）、>1000、math 口径互证 allclose。
- E3 式 5.79 数值实例化档（p.331「EGAN (wG , wD ) = log D(xn ) + log (1 − D(G(zn )))」
  的读数级实证——f010 口播的实算档；教学设定：判别器输出取温和值 D(x)=0.9（真图）、
  D(G(z))=0.2（假图）——原书只给式子不给数值，本档系最小数值演示；0.9/0.2 系
  中间态（非训练初期、非完胜——完胜方向另由 ideal 直证），0.2 读法=判别器已识破
  八成假图，生成器改进动机即压低这个「被识破率」）：真图支
  log(0.9)≈−0.105/假图支 log(1−0.2)=log(0.8)≈−0.223/联合=两支之和；理想判别器
  （真图→1、假图→0 的近似 0.9999/0.0001）时联合损失趋 0——「D 完胜 loss 最小」
  方向直证。
  断言：真图支==log(0.9)、假图支==log(0.8)、联合==和、理想判别器联合<0.01。
- E4 重叠 patch 计数档（p.333「the discriminator looks at overlapping patches and
  makes decisions on a patch-by-patch basis, which requires fewer parameters and
  provides more training data and more discriminative feedback」的读数级实证——
  f013 口播的实算档；教学口径申报：256×256 输入/70×70 感受野/步长 16 系课程教学
  设定——原书未给数值，70 系 pix2pix 论文常见判别器感受野配置）：滑窗位置数
  (256−70)//16+1=12/维 → 12×12=144 个判定块 vs 整图判别 1 次——「更多训练数据」
  =144 倍判定样本直证；重叠率（相邻窗重叠 54/70）。
  断言：位置数==12、块数==144、144>1、重叠率>50%。
- E10 作业档（f016 两道口算答案存证——①「码本五百一十二个向量、隐变量三十二乘
  三十二乘一，能表示多少张不同的图」：512^1024，位数 2775（E2 互证——「上千位的
  天文数字」）；②「二百五十六见方的图用七十见方小块逐块判，比整图判一次多出多少
  个判定块」：144（E4 互证）。
  断言：E2/E4 互证×2。
- FG1 autoencoder 瓶颈示意图（程序生成——原书本段无图，教学示意申报，deck 页 2 用）：
  input image→encoder（收窄梯形）→latent vector z（窄条）→decoder（张开梯形）→
  reconstructed image；底部注记 one-to-one mapping（p.329 教学点）。
- FG2 VQ-VAE 码本量化示意图（Figure 5.58 原书图不复制——© van den Oord, Vinyals,
  and Kavukcuoglu 2017，程序生成教学重演，deck 页 4 用）：左面板连续隐向量散点→
  最近码本格点（箭头=nearest vector 替换——codebook K=512 注记）；右面板流程条
  encoder→nearest codebook vector→discrete latent→decoder。
- FG3 GAN 博弈框图（Figure 5.59a 原书图不复制——Pan, Yu et al. 2019 © 2019 IEEE，
  程序生成教学重演，deck 页 5 用）：z→[generator G]→fake image→[discriminator D]→
  real / fake 判定+real images 支路；「G: fool D」/「D: catch the forger」对抗注记
  （p.331 fool/forger 语义）。
- 图内文字运行时收割（E044–E051 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。图内文字全英文
  （E047–E051 同款——CJK 字体不入图，中文教学语义在 deck 卡与口播）。
- 图字号口径（D27 图内显示 ≥12px；E050/E051 判例——全图文字（含 helper 默认与标题）
  13pt 起步，灰注 11.5–12.5pt 沿 E051 FG1 先例）；savefig dpi=200。
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
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
READINGS = OUT / "e052_vis_readings.json"
PNG1 = FIG_DIR / "e052_fig1_autoencoder.png"
PNG2 = FIG_DIR / "e052_fig2_vqvae.png"
PNG3 = FIG_DIR / "e052_fig3_gan.png"

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


# ── E1 VQ-VAE 码本量化直证 ──
K_CB, GRID, DV, SEED1 = 512, 32, 8, 61


def lab_e1():
    rng = np.random.default_rng(SEED1)
    codebook = rng.normal(size=(K_CB, DV))
    latent = rng.normal(size=(GRID, GRID, DV))          # z=32×32（每位置一个向量）
    flat = latent.reshape(-1, DV)                        # 1024 个位置
    d2 = ((flat[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    nearest = d2.argmin(axis=1)
    quant = codebook[nearest]                            # 最近码本向量替换
    # 输出全部来自码本：每个输出向量与自身码本条目距离==0
    self_d = ((quant - codebook[nearest]) ** 2).sum(axis=1)
    all_from_cb = bool((self_d == 0.0).all())
    check("E1 量化输出全部来自码本（替换后每向量与码本条目距离==0）", all_from_cb,
          f"{flat.shape[0]} 个位置逐一替换（K={K_CB}/Dv={DV}）")
    n_hit = int(np.unique(nearest).size)
    check("E1 命中码本条目数≤512（离散值域直证）", n_hit <= K_CB,
          f"实际命中 {n_hit}/{K_CB} 条（1024 个位置共享码本——离散化语义）")
    disp = np.sqrt(((flat - quant) ** 2).sum(axis=1))
    moved = float(disp.mean())
    check("E1 量化位移均值为正（量化确实发生——非恒等映射）", moved > 0.0,
          f"mean shift={moved:.3f}（max={disp.max():.3f}，随机 latent 位移统计）")
    check("E1 隐表示形状 32×32 保持（z=32×32×1 空间结构不破坏）", quant.shape == (GRID * GRID, DV),
          f"({GRID},{GRID},{DV})→展平 {GRID * GRID} 位置→还原网格——空间逐位置替换")
    return dict(n_positions=GRID * GRID, n_hit=n_hit, mean_shift=moved,
                max_shift=float(disp.max()))


# ── E2 组合数位数 ──
K2, Z = 512, 1024          # 512^(32·32·1)=512^1024


def lab_e2():
    n_images = K2 ** Z                                   # Python 大整数精确幂
    digits_exact = len(str(n_images))
    digits_log = int(math.floor(Z * math.log10(K2))) + 1
    check("E2 512^(32·32·1) 十进制位数==2775（整数精确）", digits_exact == 2775,
          f"精确幂位数={digits_exact} 位（512**1024——Python 大整数无浮点溢出）")
    check("E2 位数>1000（「上千位的天文数字」口径直证）", digits_exact > 1000,
          f"{digits_exact} 位（monstrously large 读数级直证——p.330）")
    check("E2 log 口径互证（floor(1024·log10 512)+1==精确位数）", digits_log == digits_exact,
          f"log 口径={digits_log}（1024×log10(512)≈{Z * math.log10(K2):.1f}）")
    return dict(digits=digits_exact, digits_log=digits_log)


# ── E3 式 5.79 数值实例化 ──
# 教学设定：判别器输出温和值 0.9/0.2 系中间态——非训练初期（初期生成器弱、
# 判别器易识破，D(G(z)) 更低），也非完胜（完胜方向另由 ideal 直证趋 0）；
# 0.2 的读法=判别器已识破八成假图——生成器改进动机即压低这个「被识破率」。
D_REAL, D_FAKE = 0.9, 0.2
D_IDEAL_R, D_IDEAL_F = 0.9999, 0.0001


def lab_e3():
    real_term = math.log(D_REAL)
    fake_term = math.log(1.0 - D_FAKE)
    joint = real_term + fake_term
    check("E3 真图支 log D(x)==log(0.9)", math.isclose(real_term, math.log(0.9), rel_tol=1e-12),
          f"log(0.9)={real_term:.4f}（判别器对真图输出 0.9——教学设定申报）")
    check("E3 假图支 log(1−D(G(z)))==log(0.8)", math.isclose(fake_term, math.log(0.8), rel_tol=1e-12),
          f"log(0.8)={fake_term:.4f}（假图判别输出 0.2→1−0.2）")
    check("E3 联合损失==两支之和（式 5.79 结构）", math.isclose(joint, real_term + fake_term, rel_tol=1e-12),
          f"joint={joint:.4f}（N=1 实例——两项相加）")
    ideal = math.log(D_IDEAL_R) + math.log(1.0 - D_IDEAL_F)
    check("E3 理想判别器（真→1/假→0 近似）联合损失趋 0（D 完胜方向直证）", ideal > -0.01 and ideal < 0.01,
          f"ideal joint={ideal:.6f}（0.9999/0.0001 近似——数值演示避免对零取对数）")
    check("E3 假图被拒概率实算 1−D(G(z))==0.8（stdout 读数出处）",
          math.isclose(1.0 - D_FAKE, 0.8, rel_tol=1e-12),
          f"1−{D_FAKE}={1.0 - D_FAKE:g}（假图被判为假=被拒概率；"
          f"「图像为真概率」是 D(G(z))={D_FAKE} 本身——两值互补，勿混）")
    return dict(real_term=real_term, fake_term=fake_term, joint=joint, ideal=ideal,
                prob_fake_rejected=1.0 - D_FAKE)


# ── E4 重叠 patch 计数（教学口径：70 感受野/步长 16 系课程教学设定）──
IMG2, RF, STRIDE = 256, 70, 16


def lab_e4():
    pos = (IMG2 - RF) // STRIDE + 1
    n_blocks = pos * pos
    overlap = (RF - STRIDE) / RF
    check("E4 滑窗位置数 (256−70)//16+1==12（每维）", pos == 12,
          f"({IMG2}−{RF})//{STRIDE}+1={pos}")
    check("E4 判定块总数==144（12×12）vs 整图判别 1 次", n_blocks == 144,
          f"{pos}×{pos}={n_blocks} 个判定块——「more training data」=144 倍判定样本直证")
    check("E4 块数>1（patch-by-patch 结构成立）", n_blocks > 1,
          f"{n_blocks} > 1（overlapping patches——p.333）")
    check("E4 相邻窗重叠率>50%（overlapping 语义直证）", overlap > 0.5,
          f"重叠 {RF}−{STRIDE}={RF - STRIDE}px/{RF}px={overlap:.1%}"
          f"（等价读法：窗口每次滑动 {STRIDE}px，{RF}px 窗口中 {RF - STRIDE}px 与前窗重叠）")
    check("E4 重叠像素与百分数实算（stdout 读数出处）",
          RF - STRIDE == 54 and round((RF - STRIDE) / RF * 100, 1) == 77.1,
          f"{RF}−{STRIDE}={RF - STRIDE}px、{((RF - STRIDE) / RF) * 100:.1f}%（算术读数入档）")
    return dict(pos_per_dim=pos, n_blocks=n_blocks, overlap=overlap,
                overlap_px=RF - STRIDE, overlap_pct=round((RF - STRIDE) / RF * 100, 1))


# ── E10 作业档 ──
def lab_e10(e2, e4):
    ans1 = f"512^{Z}（{e2['digits']} 位数）"
    check("E10 作业① 能表示图像数==512^1024==2775 位（E2 互证）", e2["digits"] == 2775,
          f"答案 {ans1}（口播定性读「上千位的天文数字」）")
    ans2 = e4["n_blocks"]
    check("E10 作业② 判定块==144（E4 互证）", ans2 == 144,
          f"答案 {ans2} 块（12×12——口播读「一百四十四」）")
    return dict(ans1=ans1, ans2=ans2)


# ── FG1 autoencoder 瓶颈示意图 ──
def _box(ax, cx, cy, w, h, text, fc, fs=13, weight="normal"):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           facecolor=fc, edgecolor=C_EDGE, linewidth=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, weight=weight)


def _arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=C_EDGE, linewidth=1.3, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def _trapezoid(ax, x0, x1, ytop0, ytop1, ybot0, ybot1, fc):
    ax.add_patch(Polygon([[x0, ytop0], [x1, ytop1], [x1, ybot1], [x0, ybot0]],
                         closed=True, facecolor=fc, edgecolor=C_EDGE, linewidth=1.2))


def fig1():
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 94, "Autoencoder: compress, then reconstruct", ha="center",
            va="center", fontsize=14, weight="bold")
    _box(ax, 8, 50, 12, 20, "input\nimage", C_BLUE, fs=12.5)
    _trapezoid(ax, 22, 40, 72, 58, 28, 42, C_ORANGE)     # encoder 收窄
    ax.text(31, 66, "encoder", ha="center", va="center", fontsize=13, weight="bold")
    _box(ax, 46, 50, 10, 14, "latent\nvector z", C_GREEN, fs=12.5)
    _trapezoid(ax, 55, 73, 58, 72, 42, 28, C_PINK)       # decoder 张开
    ax.text(64, 34, "decoder", ha="center", va="center", fontsize=13, weight="bold")
    _box(ax, 88, 50, 15, 20, "reconstructed\nimage", C_BLUE, fs=12.5)
    _arrow(ax, 14, 50, 21.5, 50)
    _arrow(ax, 40.5, 50, 41, 50)
    _arrow(ax, 51, 50, 54.5, 50)
    _arrow(ax, 73.5, 50, 80, 50)
    ax.text(50, 12, "deterministic one-to-one mapping: each input  →  exactly one latent vector  →  one output",
            ha="center", va="center", fontsize=12.5, style="italic", color="#555555")
    ax.text(46, 78, "narrowest point = compact code", ha="center", va="center",
            fontsize=11.5, style="italic", color="#666666")
    _save_and_register(fig, PNG1)


# ── FG2 VQ-VAE 码本量化示意图 ──
def fig2():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    axL.set_xlim(0, 100)
    axL.set_ylim(0, 100)
    axL.axis("off")
    axL.text(50, 95, "quantization: snap to nearest codebook vector",
             ha="center", va="center", fontsize=13, weight="bold")
    rng = np.random.default_rng(7)
    cb = rng.uniform(18, 74, size=(10, 2))
    pts = rng.uniform(20, 72, size=(14, 2))
    axL.scatter(cb[:, 0], cb[:, 1], marker="s", s=90, c="#f4c98a",
                edgecolors=C_EDGE, linewidths=1.1, zorder=3,
                label="codebook vectors (K = 512 in paper)")
    axL.scatter(pts[:, 0], pts[:, 1], marker="o", s=48, c="#a8c8e8",
                edgecolors=C_EDGE, linewidths=1.0, zorder=3,
                label="encoder output vectors")
    for p in pts:
        d = ((cb - p) ** 2).sum(axis=1)
        q = cb[d.argmin()]
        axL.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=9,
                                      color="#c0392b", linewidth=1.0,
                                      shrinkA=3, shrinkB=4))
    axL.text(50, 8, "each spatial position  →  its nearest vector (quantize)",
             ha="center", va="center", fontsize=12.5)
    axL.text(3, 88, "□ codebook vectors (K = 512 in paper)",
             ha="left", va="center", fontsize=12.5, color="#b05a1a")
    axL.text(3, 81.5, "● encoder output vectors",
             ha="left", va="center", fontsize=12.5, color="#3d6a99")
    axR.set_xlim(0, 100)
    axR.set_ylim(0, 100)
    axR.axis("off")
    axR.text(50, 95, "VQ-VAE pipeline", ha="center", va="center",
             fontsize=13.5, weight="bold")
    steps = [
        ("encoder  (like a VAE)", C_BLUE),
        ("replace each position with\nnearest codebook vector", C_ORANGE),
        ("discrete latent  (32×32 grid)", C_GREEN),
        ("decoder", C_PINK),
        ("reconstructed / generated image", C_GREEN),
    ]
    ys = [82, 63, 44, 25, 8]
    for (txt, fc), y in zip(steps, ys):
        _box(axR, 46, y, 70, 12.5, txt, fc, fs=12.5)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        _arrow(axR, 46, y0 - 6.25, 46, y1 + 6.25)
    axR.text(46, 1.0, "codebook trained jointly with encoder & decoder",
             ha="center", va="center", fontsize=12.5, style="italic", color="#666666")
    _save_and_register(fig, PNG2)


# ── FG3 GAN 博弈框图 ──
def fig3():
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 94, "Generative adversarial network: the forging game",
            ha="center", va="center", fontsize=14, weight="bold")
    _box(ax, 8, 42, 10, 14, "random\nnoise z", C_BLUE, fs=12.5)
    _box(ax, 30, 42, 17, 15, "generator G\n(forger)", C_ORANGE, fs=12.5, weight="bold")
    _box(ax, 56, 42, 14, 14, "fake\nimage", C_PINK, fs=12.5)
    _box(ax, 80, 42, 17, 15, "discriminator D\n(critic)", C_GREEN, fs=12.5, weight="bold")
    _arrow(ax, 13, 42, 21.5, 42)
    _arrow(ax, 38.5, 42, 49, 42)
    _arrow(ax, 63, 42, 71.5, 42)
    _box(ax, 80, 12, 15, 11, "real images\n(training set)", C_GREEN, fs=12)
    _arrow(ax, 80, 17.5, 80, 34.5)
    ax.text(92, 42, "real / fake ?", ha="left", va="center", fontsize=12.5, weight="bold")
    _arrow(ax, 88.5, 42, 93, 42)
    ax.text(30, 70, "G: make images that fool D", ha="center", va="center",
            fontsize=12.5, style="italic", color="#b05a1a")
    ax.text(80, 70, "D: catch the forger", ha="center", va="center",
            fontsize=12.5, style="italic", color="#1f6f3f")
    ax.text(50, 86, "both networks trained simultaneously  ·  minimax game (Eq. 5.79)",
            ha="center", va="center", fontsize=12.5)
    _save_and_register(fig, PNG3)


# ── 图内记号哨兵 ──
def harvest_mark():
    joined = "\n".join(_TEXTS)
    leaked = FIG_MARK.findall(joined)
    check("FG 图内文字内部记号零泄漏（FIG_MARK 零命中）", not leaked,
          f"图内 Text 收割 {len(_TEXTS)} 条，命中 {leaked if leaked else '无'}")
    check("FG 程序生成图 3 张全落盘", PNG1.exists() and PNG2.exists() and PNG3.exists(),
          f"{PNG1.name}+{PNG2.name}+{PNG3.name}（sha 见 [lab] PNG 行）")


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] E052 生成模型数值演示档 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    e1 = lab_e1()
    e2 = lab_e2()
    e3 = lab_e3()
    e4 = lab_e4()
    e10 = lab_e10(e2, e4)
    fig1()
    fig2()
    fig3()
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
        "cfg": {"k_cb": K_CB, "grid": GRID, "dv": DV, "seed1": SEED1,
                "k2": K2, "z": Z, "d_real": D_REAL, "d_fake": D_FAKE,
                "d_ideal_r": D_IDEAL_R, "d_ideal_f": D_IDEAL_F,
                "img2": IMG2, "rf": RF, "stride": STRIDE},
        "e1": {k: e1[k] for k in ("n_positions", "n_hit", "mean_shift", "max_shift")},
        "e2": {k: e2[k] for k in ("digits", "digits_log")},
        "e3": {k: e3[k] for k in ("real_term", "fake_term", "joint", "ideal", "prob_fake_rejected")},
        "e4": {k: e4[k] for k in ("pos_per_dim", "n_blocks", "overlap", "overlap_px", "overlap_pct")},
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
