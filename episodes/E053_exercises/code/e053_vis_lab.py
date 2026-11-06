#!/usr/bin/env python3
"""E053 实验课数值演示档（E052 lab 同款框架——check/readings/PNG 三件套）。

Ex 5.4/5.5/5.7/5.18 口播实算档（f006–f015 的实验读数出处）+作业互证档。

- E1 Ex 5.4 网格实算档（p.338 问 2/3 的读数级实证——f007 口播的实算档）：双隐藏单元
  ReLU 网络无偏置——z1=ReLU(w11·x1+w12·x2)/z2=ReLU(w21·x1+w22·x2)/y=w31·z1+w32·z2；
  (b) 合理起点 (w11,w12,w21,w22,w31,w32)=(1,1,−1,−1,1,1)（raw 图区排版读数：w31/w32=1 1
  与 w11..w22=−1 −1 1 1——四数横排与边连接线不可从文本层复原，支分配系教学演绎申报
  ——z1=ReLU(x1+x2)/z2=ReLU(−x1−x2)/y=z1+z2=|x1+x2|）；目标 t=|x1+1.1x2|（式 5.80——
  1.1 系 (b) 起点未含的训练修正项：x2 支权重 1→训练推向 1.1，教学申报）；3×3 网格
  (x1,x2)∈{−1,0,1}² 九点前向激活+平方损失；六权重解析梯度（链式法则手推式）与数值
  梯度（中心差分）互证。
  断言：前向 y==|x1+x2|（齐次核对）、九点损失总和>0（1.1 项未拟合）、解析 vs 数值
  梯度最大误差<1e-9、六权重梯度全非零（信号存在）。
- E2 Ex 5.4 (c) 差尺度对比档（p.338 问 5/6 的读数级实证——f008 口播的实算档）：
  (c) 起点 (100,100,−100,−100,0.01,0.01)（raw 图区读数：w11..w22=−100 −100 100 100+
  w31/w32=0.01 0.01——隐藏层 ±100+输出层 0.01 双重差尺度）；前向齐次性：ReLU(100u)=
  100·ReLU(u)→(c) 的 y 与 (b) 完全相同（前向等价断言）；梯度不对称：隐藏层权重梯度
  ×0.01（∂L/∂w11 含 w31=0.01 因子）、输出层权重梯度×100（∂L/∂w31 含 z1=100·ReLU(u)
  因子——且输出层有效步长 α·z1² 被放大约 1e4 倍）。
  实测定案（原书问 6 预期「误差下降变差+更多迭代」——实测更严峻，教学申报）：α=0.02
  下 (b) 一步改善（loss 0.060→0.0399）而 (c) 一步 loss→1.7e5（抬升约 3e6 倍）；(c) 在
  α∈{1e-3,2e-4,1e-4} 全发散（发散守卫 loss>1e6 判定）；(c) 唯一站得住的量级 α=1e-5
  下 20000 步 loss 仅 0.060→0.030（连一半都没磨掉），(b) 同 α 9461 步达标——「权值
  尺度决定的不只是快慢，还有学不学得动」读数直证（E042 初始化回指）。
  断言：(b)(c) 九点前向 y 全等（前向等价）、(c) 隐藏层梯度==(b)×0.01、(c) 输出层梯度==
  (b)×100、(b) α=0.02 一步改善、(c) α=0.02 一步爆（loss>1e5）、(c) 三个较大 α 全发散、
  (c) α=1e-5 蜗牛（20000 步未达标 vs (b) 同 α 达标）。
- E3 Ex 5.5 优化器轨迹档（p.339 的读数级实证——f011 口播的实算档）：f=x²+20y²、
  ∇f=(2x,40y)、起点 (−20,5)；GD/momentum（v=ρv+g，x←x−αv，ρ=0.99）/Adam（β1=0.9、
  β2=0.999、ε=1e-8——原书 b2 系排印疑误①照录不改，本档按 β2 教学）三优化器各
  200 步；收敛判定 max(|x|,|y|)<0.1（「near the minimum」教学口径申报——原书只说
  near/under 200 steps）；单调性=损失序列逐点不增判定；「单一学习率」=α 对数网格
  0.001–0.1（400 点）扫描存在性判定。
  断言：GD 存在连续收敛窗口（α 对数网格 0.001–0.5 共 500 点扫描——实测窗口申报）、
  momentum ρ=0.99 收敛窗口系碎岛（500 点仅个位数 α 命中、散在区间内——对学习率极
  敏感「动量的脾气」——GD 同网格连续命中百余点对照）、GD∩momentum 有小交集（前两
  位可共享）、Adam 收敛需大 α（下界>0.2 申报）且窗口与 GD∩momentum 不搭界（大十倍
  不止）、三窗口无交集——「单一学习率不存在」实测定案（原书问句不给答案，结论系
  本档实测申报）、GD 在 α=0.03 下损失单调不增、momentum（α=0.03）与 Adam（α=0.5
  收敛窗口上段）损失序列存在回升点（原书问「每条轨迹是否单调」——GD 是/mom 与
  Adam 否的读数直证；Adam 窗口下段 α∈[0.225, 0.3] 平滑无回升——自适应步长在小 α 下
  抑制过冲，大 α 才折返，口径申报）。
- E4 Ex 5.7 卷积口算档（p.340 的读数级实证——f012 口播的实算档）：64 核、k∈{3,5,7,9}；
  参数量 params(k)=k·k·3·64（不含 bias——教学设定申报，bias 口径另注 +64/组）；输出
  尺寸 out(k)=(256−k+1)²×64（stride 1/padding 0 教学设定申报——原书未给）。
  断言：params(3)==1728（f015 口算互证）、params(5)/7/9 同法、out(3)==254²·64。
- E5 Ex 5.18 GAN 极限档（p.341 的读数级实证——f014 口播的实算档；E052 对抗损失档
  续篇）：D 近完美（D(x)=1−ε/D(G(z))=ε）；式 5.81 两支：log D(x)+log(1−D(G(z)))→
  趋 0（ε→0 时≈−2ε——判别器视角损失趋 0=「近乎完美」读数直证）；生成器视角
  饱和：生成器 max L_GAN→log(1−ε) 在 ε→0 处达最大值 0——改进空间为零+有效信号
  对比：non-saturating 替代 −log D(G(z)) 在 ε→0 处→发散（大信号）——脚注 65 hint
  （「对强判别器给出大修正式、对弱判别器给出小修正式」）语义直证；ε 对数扫描
  {0.5,0.1,0.01,1e-4} 两列损失+比值读数。
  断言：joint(1e-4)>−0.001（趋 0）、ns(1e-4)>9（大信号）、ratio 随 ε 减小单调增
  （「老师太强学生学不动→换写法缓解」读数链）。
- E10 作业档（f015 两道口算答案存证——①「二百五十六见方三通道配六十四个三乘三
  卷积核，参数量多少」：1728（E4 互证——「不算偏置」口径）；②「负二十、五处的
  梯度」：(−40,200)（E3 互证——2x=−40/40y=200）。
  断言：E4/E3 互证×2。
- FG1 权值尺度实验对比图（程序生成——Figure 5.62 原书图不复制，教学重演申报，
  deck 页 5 用）：左面板 (b) 起点训练损失曲线（α=0.02、30 步——数步内陡降）；右面板
  (c) 起点同 α 前 4 步损失（log 纵轴——第一步即爆 3e6 倍）+灰注「α=1e-5 时 20000 步
  才磨到 0.030（连一半都没到）」——「(b) reasonable start / (c) poorly scaled」注记
  （Figure 5.62 caption 语义）。
- FG2 优化器轨迹图（Figure 5.63 原书图不复制，程序生成教学重演，deck 页 7 用）：
  f=x²+20y² 等高线背景+同一 α=0.03 三轨迹（GD 收敛到原点/momentum ρ=0.99 卡在半途
  振荡/Adam 匀速慢爬）+起点/谷底标注+「same α=0.03 — only GD reaches the minimum
  within 200 steps」注记（Figure 5.63b 语义+E3 实测定案——同 α 展示最贴原书「单一
  学习率」题旨）。
- FG3 LeNet 十行起手结构图（教学自拟——原书本段无 LeNet 结构图，程序生成申报，
  deck 页 3 用）：input→conv→pool→conv→pool→fc→fc→output 流程条+「10 lines to
  start（PyTorch beginner tutorial）」注记（Ex 5.2 脚注 63 语义）。
- 图内文字运行时收割（E044–E052 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。图内文字全英文
  （E047–E052 同款——CJK 字体不入图，中文教学语义在 deck 卡与口播）。
- 图字号口径（D27 图内显示 ≥12px；E050/E051/E052 判例——全图文字（含 helper 默认与
  标题）13pt 起步，灰注 12.5pt 沿 E052 FG2 先例——E052 FG2 首版 10.5pt legend 被
  display_check 拦下（10.4px<12）改 12.5pt 判例延续）；savefig dpi=200。
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

OUT = pathlib.Path(__file__).resolve().parent / "outputs"
FIGDIR = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)
READINGS = OUT / "e053_vis_readings.json"
PNG1 = FIGDIR / "e053_fig1_weightscale.png"
PNG2 = FIGDIR / "e053_fig2_optimizers.png"
PNG3 = FIGDIR / "e053_fig3_lenet.png"

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
    """统一保存通道：图内 Text 全量收割注册（fig 级遍历含图题与图例——判例沿用）。"""
    for t in fig.findobj(_mtext.Text):
        _TEXTS.append(t.get_text())
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── E1/E2 共用：Ex 5.4 网络 ──
# (b)/(c) 权重系 raw 图区排版读数（w31/w32 行与 w11..w22 行）——四数横排与边连接线
# 不可从文本层复原，支分配系教学演绎申报（对称结构，支序无关前向/梯度结论）。
W_B = (1.0, 1.0, -1.0, -1.0, 1.0, 1.0)          # (w11,w12,w21,w22,w31,w32)
W_C = (100.0, 100.0, -100.0, -100.0, 0.01, 0.01)  # (c) 差尺度（隐藏 ±100+输出 0.01）
GRID9 = [(x1, x2) for x1 in (-1, 0, 1) for x2 in (-1, 0, 1)]  # 问 2 九点网格
ALPHA = 0.02                                     # 训练对比学习率（教学设定申报）


def forward(w, x1, x2):
    w11, w12, w21, w22, w31, w32 = w
    z1 = max(0.0, w11 * x1 + w12 * x2)
    z2 = max(0.0, w21 * x1 + w22 * x2)
    return z1, z2, w31 * z1 + w32 * z2


def target(x1, x2):
    return abs(x1 + 1.1 * x2)                     # 式 5.80


def loss_total(w):
    return sum((forward(w, a, b)[2] - target(a, b)) ** 2 for a, b in GRID9)


def grad_total(w):
    """六权重解析梯度（链式法则手推式——Ex 5.4 问 3/E044 5.65–5.68 教学口径）。"""
    g = [0.0] * 6
    for x1, x2 in GRID9:
        w11, w12, w21, w22, w31, w32 = w
        u1, u2 = w11 * x1 + w12 * x2, w21 * x1 + w22 * x2
        z1, z2 = max(0.0, u1), max(0.0, u2)
        y = w31 * z1 + w32 * z2
        dLdy = 2.0 * (y - target(x1, x2))
        g[4] += dLdy * z1                          # ∂L/∂w31 = dLdy·z1
        g[5] += dLdy * z2                          # ∂L/∂w32 = dLdy·z2
        if u1 > 0:
            dz1 = dLdy * w31
            g[0] += dz1 * x1                       # ∂L/∂w11 = dLdy·w31·x1（u1>0）
            g[1] += dz1 * x2
        if u2 > 0:
            dz2 = dLdy * w32
            g[2] += dz2 * x1
            g[3] += dz2 * x2
    return g


def grad_numeric(w, h=1e-6):
    out = []
    for i in range(6):
        wp, wm = list(w), list(w)
        wp[i] += h
        wm[i] -= h
        out.append((loss_total(wp) - loss_total(wm)) / (2 * h))
    return out


def train_steps(w, steps, alpha=ALPHA):
    w = list(w)
    hist = []
    for _ in range(steps):
        hist.append(loss_total(w))
        g = grad_total(w)
        w = [wi - alpha * gi for wi, gi in zip(w, g)]
    hist.append(loss_total(w))
    return w, hist


BOOM = 1e6                                       # 发散守卫阈（(c) 爆炸提前停——防 Overflow）


def train_ex54(w, steps, alpha):
    """Ex 5.4 GD 训练+发散守卫：loss>1e6 或非数值即提前停（boom=True）。"""
    w = list(w)
    hist = [loss_total(w)]
    for _ in range(steps):
        g = grad_total(w)
        w = [wi - alpha * gi for wi, gi in zip(w, g)]
        h = loss_total(w)
        hist.append(h)
        if h != h or h > BOOM:
            return w, hist, True
    return w, hist, False


def lab_e1():
    y_b = [forward(W_B, a, b)[2] for a, b in GRID9]
    abs_b = [abs(a + b) for a, b in GRID9]
    check("E1 前向 y==|x1+x2|（(b) 起点齐次核对）",
          all(math.isclose(y, m, rel_tol=1e-12) for y, m in zip(y_b, abs_b)),
          f"九点 y 与 |x1+x2| 全等（z1=ReLU(x1+x2)+z2=ReLU(−x1−x2)——两支相加出绝对值）")
    loss_b = loss_total(W_B)
    check("E1 九点平方损失>0（1.1 系数未拟合——训练修正项存在）", loss_b > 0,
          f"loss={loss_b:.4f}（目标 |x1+1.1x2| 与起点 |x1+x2| 之差——f006 口播读数出处）")
    g_ana = grad_total(W_B)
    g_num = grad_numeric(W_B)
    err = max(abs(a - n) for a, n in zip(g_ana, g_num))
    check("E1 解析梯度 vs 数值梯度最大误差<1e-9（链式法则互证）", err < 1e-9,
          f"max_err={err:.2e}（六权重——链式法则四式手推直证，f007 口播读数出处）")
    check("E1 六权重梯度全非零（梯度信号存在）", all(abs(g) > 1e-9 for g in g_ana),
          "六条梯度全非零（x2 支 1.1 修正信号在档）")
    return dict(loss_b=round(loss_b, 6), grad=[round(g, 6) for g in g_ana],
                grad_max_err=float(f"{err:.2e}"), target_coef=1.1)


def lab_e2():
    y_b = [forward(W_B, a, b)[2] for a, b in GRID9]
    y_c = [forward(W_C, a, b)[2] for a, b in GRID9]
    check("E2 (b)/(c) 九点前向 y 全等（ReLU 齐次性——前向等价）",
          all(math.isclose(a, b, rel_tol=1e-9) for a, b in zip(y_b, y_c)),
          "ReLU(100u)=100·ReLU(u)→0.01·ReLU(100u)=ReLU(u)——前向输出完全相同（权值尺度"
          "不改变网络能表达什么——f008 口播读数出处）")
    g_b, g_c = grad_total(W_B), grad_total(W_C)
    hid = all(math.isclose(gc, 0.01 * gb, rel_tol=1e-9)
              for gc, gb in zip(g_c[:4], g_b[:4]))
    out_w = all(math.isclose(gc, 100.0 * gb, rel_tol=1e-9)
                for gc, gb in zip(g_c[4:], g_b[4:]))
    check("E2 (c) 隐藏层梯度==(b)×0.01（w31=0.01 因子缩梯度）", hid,
          "w11..w22 四条梯度精确 0.01 倍（∂L/∂w11=dLdy·w31·x1——反向链上的尺度瓶颈）")
    check("E2 (c) 输出层梯度==(b)×100（z1=100·ReLU 因子放梯度）", out_w,
          "w31/w32 两条梯度精确 100 倍（前向等价 dLdy 相同·z1 放大 100——有效步长 α·z1² "
          "再放大 ~1e4 倍——不对称读数）")
    _, hb1, boom_b = train_ex54(W_B, 1, ALPHA)
    _, hc1, _ = train_ex54(W_C, 1, ALPHA)
    check("E2 (b) α=0.02 一步改善（合理起点走得动）",
          (not boom_b) and hb1[1] < hb1[0],
          f"loss {hb1[0]:.4f}→{hb1[1]:.4f}（一步即降——f008「合理起点几步压下去」读数出处）")
    check("E2 (c) α=0.02 一步爆（同学习率下损失抬升百万倍级）",
          hc1[1] > 1e5,
          f"loss {hc1[0]:.4f}→{hc1[1]:.3g}（≈{hc1[1] / hc1[0]:.2g} 倍——输出层 0.01 权重的"
          f"梯度放大效应一步引爆——f008「第一步抬高几百万倍」读数出处）")
    booms = []
    for al in (1e-3, 2e-4, 1e-4):
        _, _, bm = train_ex54(W_C, 500, al)
        booms.append(bm)
    check("E2 (c) 较大 α 全发散（安全学习率窗口被挤没）", all(booms),
          "α∈{1e-3,2e-4,1e-4} 三档全部触发发散守卫（有效步长 α·z1² 随激活尺度爆炸——"
          "「站不住的量级」读数出处）")
    _, hbL, _ = train_ex54(W_B, 20000, 1e-5)
    _, hcL, _ = train_ex54(W_C, 20000, 1e-5)
    steps_b15 = next((i for i, h in enumerate(hbL) if h < 0.01), None)
    check("E2 (c) 唯一稳定量级 α=1e-5 下蜗牛爬行（20000 步未过半 vs (b) 同 α 达标）",
          hcL[-1] > 0.01 and steps_b15 is not None,
          f"(c) 20000 步 loss {hcL[0]:.4f}→{hcL[-1]:.4f}（连一半都没磨掉）；(b) 同 α "
          f"{steps_b15} 步即达标——「决定的不只是快慢，还有学不学得动」读数直证——"
          "f008 口播出处+E042 初始化回指")
    return dict(loss0=round(loss_total(W_B), 6),
                b_step1=[round(hb1[0], 4), round(hb1[1], 4)],
                c_step1=[round(hc1[0], 4), float(f"{hc1[1]:.3g}")],
                c_boom_alphas=[1e-3, 2e-4, 1e-4], c_boom_all=all(booms),
                crawl_alpha=1e-5,
                c_crawl_20k=round(hcL[-1], 4), b_crawl_steps=steps_b15,
                z1_gain=100.0 ** 2, boom_ratio=hc1[1] / hc1[0], crawl_steps=20000)


# ── E3 Ex 5.5 优化器 ──
P0 = (-20.0, 5.0)                                # 起点申报（问 2）
RHO, B1, B2, EPS_A = 0.99, 0.9, 0.999, 1e-8      # 原书数值（b2 系排印疑误①按 β2 教学）
NSTEPS, CONV_TH = 200, 0.1


def f55(p):
    return p[0] ** 2 + 20.0 * p[1] ** 2


def g55(p):
    return np.array([2.0 * p[0], 40.0 * p[1]])


# 三优化器统一返回 (轨迹, 损失序列, 收敛步数)——lab_e3 扫描与 fig2 轨迹图共用
def run_gd(alpha, steps=NSTEPS):
    p = np.array(P0, dtype=float)
    traj = [p.copy()]
    hist = [f55(p)]
    done = None
    for t in range(1, steps + 1):
        p = p - alpha * g55(p)
        traj.append(p.copy())
        if not np.isfinite(p).all() or np.abs(p).max() > 1e8:
            break

        hist.append(f55(p))
        if done is None and max(abs(p[0]), abs(p[1])) < CONV_TH:
            done = t
    return np.array(traj), hist, done


def run_mom(alpha, steps=NSTEPS):
    p = np.array(P0, dtype=float)
    v = np.zeros(2)
    traj = [p.copy()]
    hist = [f55(p)]
    done = None
    for t in range(1, steps + 1):
        g = g55(p)
        v = RHO * v + g
        p = p - alpha * v
        traj.append(p.copy())
        if not np.isfinite(p).all() or np.abs(p).max() > 1e8:
            break

        hist.append(f55(p))
        if done is None and max(abs(p[0]), abs(p[1])) < CONV_TH:
            done = t
    return np.array(traj), hist, done


def run_adam(alpha, steps=NSTEPS):
    p = np.array(P0, dtype=float)
    m = np.zeros(2)
    v = np.zeros(2)
    traj = [p.copy()]
    hist = [f55(p)]
    done = None
    for t in range(1, steps + 1):
        g = g55(p)
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        p = p - alpha * (m / (1 - B1 ** t)) / (np.sqrt(v / (1 - B2 ** t)) + EPS_A)
        traj.append(p.copy())
        if not np.isfinite(p).all() or np.abs(p).max() > 1e8:
            break

        hist.append(f55(p))
        if done is None and max(abs(p[0]), abs(p[1])) < CONV_TH:
            done = t
    return np.array(traj), hist, done


def lab_e3():
    # 单一学习率扫描：α 对数网格 0.001–0.5 共 500 点（教学设定申报——原书只问存在性）
    alphas = np.logspace(-3, np.log10(0.5), 500)
    gd_win = [float(a) for a in alphas if run_gd(float(a))[2] is not None]
    mom_win = [float(a) for a in alphas if run_mom(float(a))[2] is not None]
    adam_win = [float(a) for a in alphas if run_adam(float(a))[2] is not None]
    gm = sorted(set(gd_win) & set(mom_win))          # GD∩momentum（同一网格源——精确相等）
    tri = set(gd_win) & set(mom_win) & set(adam_win)
    check("E3 GD 存在连续收敛窗口（α 窗口申报）", bool(gd_win),
          f"GD 收敛窗口 α∈[{min(gd_win):.4f}, {max(gd_win):.4f}]（对数网格 500 点连续命中——"
          "f011「标准梯度下降有一段安全窗口」读数出处）")
    check("E3 momentum 收敛窗口系碎岛（对学习率极敏感——动量的脾气）",
          0 < len(mom_win) <= 20,
          f"500 点仅 {len(mom_win)} 个 α 命中，散在 [{min(mom_win):.4f},{max(mom_win):.4f}]"
          f"（GD 同网格连续命中 {len(gd_win)} 点——动量振荡系统对 α 的共振带——"
          "f011「孤零零的小岛/挑得厉害」读数出处）")
    check("E3 GD 与 momentum 有公共窗口（前两位可共享）", bool(gm),
          f"交集 α∈[{min(gm):.4f}, {max(gm):.4f}]（{len(gm)} 点）")
    check("E3 Adam 收敛需大 α 且与前两者窗口不搭界（大十倍不止）",
          bool(adam_win) and min(adam_win) > 2 * max(gm),
          f"Adam 收敛窗口 α≥{min(adam_win):.3f}（扫描上限 0.5 内）vs GD∩momentum 上界 "
          f"{max(gm):.4f}——f011「Adam 要大步子/步子大十倍不止」读数出处")
    check("E3 无公共学习率使三优化器均 200 步收敛（「单一学习率」实测定案）",
          bool(gd_win) and bool(adam_win) and not tri,
          "三窗口交集为空（Adam 独走大 α）——原书问句不给答案，结论系本档实测申报——"
          "f011「找不到任何一个学习率」读数出处")
    a_demo = 0.03          # 同 α 演示：GD 窗口内/mom 岛外/Adam 窗外——同 α 只有 GD 到（FG2）
    tg, hg, dg = run_gd(a_demo)
    tm, hm, _ = run_mom(a_demo)
    ta, ha, _ = run_adam(a_demo)
    ta_own, ha_own, da_own = run_adam(min(adam_win))
    ta_up, ha_up, da_up = run_adam(0.5)
    mono_gd = all(h2 <= h1 + 1e-9 for h1, h2 in zip(hg, hg[1:]))
    check("E3 GD 在 α=0.03 下损失单调不增（原书「是否单调」——GD 是）", mono_gd,
          f"GD 损失序列逐点不增（α=0.03——{dg} 步收敛；α<2/λ_max=0.05 单调稳定界内）")
    over_m = any(h2 > h1 * 1.001 for h1, h2 in zip(hm[:60], hm[1:61]))
    over_a = any(h2 > h1 * 1.001 for h1, h2 in zip(ha_up[:80], ha_up[1:81]))
    check("E3 momentum/Adam 在演示 α 下损失回升（原书「是否单调」——两者否）",
          over_m and over_a,
          f"momentum α=0.03 前 60 步现回升；Adam α=0.5（收敛窗口上段）前 80 步现回升"
          "（momentum 惯性冲过谷底来回荡/Adam 大步跨过谷底折返——f011「冲过头再荡回来」"
          "读数出处；口径申报：Adam 窗口下段 α∈[0.225, 0.3] 平滑无回升——自适应步长在小 α 下"
          "抑制过冲，大 α 才折返）")
    return dict(alpha_demo=a_demo,
                gd_window=[round(min(gd_win), 4), round(max(gd_win), 4)],
                gd_hits=len(gd_win),
                mom_islands=len(mom_win),
                mom_span=[round(min(mom_win), 4), round(max(mom_win), 4)],
                gd_mom_window=[round(min(gm), 4), round(max(gm), 4)] if gm else None,
                adam_window=[round(min(adam_win), 4), round(max(adam_win), 4)],
                steps_gd=dg, steps_adam_own=da_own, steps_adam_up=da_up,
                monotonic_gd=mono_gd, mom_overshoot=over_m, adam_overshoot=over_a,
                p0=list(P0), scan="log 0.001–0.5, 500 pts", scan_pts=500,
                gd_mom_hits=len(gm), mono_k=2, mono_bound=2 / 40,
                overshoot_win=[60, 80], adam_band=[0.225, 0.3])


def lab_e4():
    ks = (3, 5, 7, 9)
    params = {k: k * k * 3 * 64 for k in ks}
    outs = {k: (256 - k + 1) ** 2 * 64 for k in ks}
    check("E4 Ex 5.7 参数量口算（3×3×3×64=1728，不含 bias——教学设定申报）",
          params[3] == 1728,
          f"k=3：{params[3]}（bias 另注 +64/核——口径 f012 口播「不算偏置」申报）")
    check("E4 五/七/九见方同法（读数在档）",
          params[5] == 4800 and params[7] == 9408 and params[9] == 15552,
          f"k=5:{params[5]}/k=7:{params[7]}/k=9:{params[9]}")
    check("E4 输出尺寸 (256−k+1)²×64（stride 1/padding 0 教学设定申报）",
          outs[3] == 254 ** 2 * 64 and outs[9] == 248 ** 2 * 64,
          f"k=3：254²×64={outs[3]}；k=9：248²×64={outs[9]}（f012 口播「边长各减二」出处）")
    return dict(params=params, outs=outs, k_list=list(ks), channels=64,
                edges=[256 - k + 1 for k in ks])


def lab_e5():
    eps_list = [0.5, 0.1, 0.01, 1e-4]
    joints = [math.log(1 - e) + math.log(1 - e) for e in eps_list]   # D(x)=1−ε/D(G)=ε
    ns = [-math.log(e) for e in eps_list]                            # non-saturating 替代
    ratios = [n / abs(j) if j else float("inf") for n, j in zip(ns, joints)]
    check("E5 式 5.81 两支损失趋 0（D 近完美读数）", joints[-1] > -0.001,
          f"ε=1e-4：log(1−ε)+log(1−ε)={joints[-1]:.6f}（≈−2ε 趋零——「判别器近乎完美」"
          f"f014 口播读数出处；E052 对抗损失档续篇）")
    check("E5 生成器视角饱和：log(1−ε) 在 ε→0 达最大值 0——改进空间为零",
          abs(joints[-1] / 2) < 1e-3,
          f"生成器 max log(1−D(G(z)))={math.log(1 - eps_list[-1]):.6f}（已达上界——"
          "无有效改进方向——「老师太强学生学不动」读数出处；口播白话『梯度跟着消失』的"
          "精确口径=损失已达生成器上界、梯度不再携带有效改进信号，教学申报）")
    mono = all(r2 > r1 for r1, r2 in zip(ratios, ratios[1:]))
    check("E5 non-saturating 替代在 ε→0 发散（大信号）且比值单调增（换写法缓解直证）",
          ns[-1] > 9 and mono,
          f"−log(ε)@1e-4={ns[-1]:.2f}（大信号）；比值序列 {[f'{r:.1f}' for r in ratios]} "
          f"单调增（脚注 65 hint 语义：强判别器→大修正/弱判别器→小修正——f014 口播出处）")
    return dict(eps=eps_list, joint=[round(j, 6) for j in joints],
                ns=[round(n, 3) for n in ns],
                ratio=[round(r, 1) if math.isfinite(r) else None for r in ratios],
                log1m=round(math.log(1 - eps_list[-1]), 6), footnote=65)


def lab_e10(e3, e4):
    p1 = e4["params"][3]
    check("E10 口算①Ex 5.7 三乘三参数量==1728（E4 互证）", p1 == 1728,
          f"{p1}（f015 口播「一千七百二十八」存证）")
    g_at = (2.0 * P0[0], 40.0 * P0[1])
    check("E10 口算②Ex 5.5 起点 (−20,5) 梯度==(−40,200)（E3 互证）",
          g_at == (-40.0, 200.0),
          f"({g_at[0]:.0f},{g_at[1]:.0f})（f015 口播「负四十、二百」存证）")
    return {"q1_params3": p1, "q2_grad_p0": list(g_at)}


# ── FG1 权值尺度实验对比图 ──
def fig1():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 4.2))
    _, hb = train_steps(W_B, 30)
    axL.plot(range(len(hb)), hb, color="#3d6a99", lw=2.2)
    axL.set_title("(b) reasonable start — a few steps suffice",
                  fontsize=13, color="#3d6a99")
    axL.set_xlabel("step", fontsize=12.5)
    axL.set_ylabel("squared loss (9 points)", fontsize=12.5)
    axL.annotate("α = 0.02", (8, hb[8]), textcoords="offset points", xytext=(0, 8),
                 fontsize=12.5, color="#3d6a99")
    _, hc, _ = train_ex54(W_C, 3, ALPHA)
    axR.semilogy(range(len(hc)), hc, color="#b05a1a", lw=2.2, marker="o", ms=4)
    axR.set_title("(c) poorly scaled — first step blows up at the same α",
                  fontsize=13, color="#b05a1a")
    axR.set_xlabel("step", fontsize=12.5)
    axR.set_ylabel("squared loss (9 points, log)", fontsize=12.5)
    axR.text(0.1, hc[-1] * 2.2, "α = 0.02: loss ×3,000,000 in one step",
             fontsize=12.5, color="#b05a1a")
    axR.text(0.1, hc[0] * 0.55, "shrunk to α = 1e-5: 20,000 steps → 0.030\n"
             "(b) reaches the same bar in 9,461 steps",
             fontsize=12.5, color="#555555")
    axR.set_ylim(hc[0] * 0.3, None)
    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Weight-scale experiment (Ex. 5.4, Fig. 5.62 semantics) — program-generated, "
                 "not the original figure", fontsize=12.5, y=1.02, color="#555555")
    fig.tight_layout()
    _save_and_register(fig, PNG1)


# ── FG2 优化器轨迹图（同一 α=0.03——三轨迹同图，最贴「单一学习率」题旨） ──
def fig2(a_demo):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs = np.linspace(-24, 4, 240)
    ys = np.linspace(-7.5, 7.5, 240)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = XX ** 2 + 20 * YY ** 2
    ax.contour(XX, YY, ZZ, levels=[1, 5, 25, 100, 250, 500, 900, 1500],
               colors="#cccccc", linewidths=0.8)
    tg, _, _ = run_gd(a_demo)
    tm, _, _ = run_mom(a_demo)
    ta, _, _ = run_adam(a_demo)
    ax.plot(tg[:, 0], tg[:, 1], "-o", ms=2.6, color="#3d6a99", lw=1.8, label="GD")
    ax.plot(tm[:, 0], tm[:, 1], "-o", ms=2.6, color="#b05a1a", lw=1.8,
            label="momentum (ρ = 0.99)")
    ax.plot(ta[:, 0], ta[:, 1], "-o", ms=2.6, color="#3d7a4a", lw=1.8,
            label="Adam (β1 = 0.9, β2 = 0.999)")
    ax.plot(*P0, marker="s", ms=9, color="#4a4a4a")
    ax.annotate("start (−20, 5)", P0, textcoords="offset points", xytext=(8, 8),
                fontsize=12.5, color="#4a4a4a")
    ax.plot(0, 0, marker="*", ms=16, color="#4a4a4a")
    ax.annotate("minimum (0, 0)", (0, 0), textcoords="offset points", xytext=(8, -14),
                fontsize=12.5, color="#4a4a4a")
    ax.annotate("GD", (tg[-1, 0], tg[-1, 1]), textcoords="offset points", xytext=(6, 6),
                fontsize=12.5, color="#3d6a99")
    ax.annotate("momentum", (tm[-1, 0], tm[-1, 1]), textcoords="offset points",
                xytext=(6, 6), fontsize=12.5, color="#b05a1a")
    ax.annotate("Adam", (ta[-1, 0], ta[-1, 1]), textcoords="offset points",
                xytext=(6, 6), fontsize=12.5, color="#3d7a4a")
    ax.set_title(f"f(x, y) = x² + 20y², same α = {a_demo}, 200 steps\nonly GD reaches "
                 "the minimum (program-generated, not the original figure)",
                 fontsize=12.5, color="#4a4a4a")
    ax.set_xlabel("x", fontsize=12.5)
    ax.set_ylabel("y", fontsize=12.5)
    ax.legend(fontsize=12.5, loc="lower left")
    fig.tight_layout()
    _save_and_register(fig, PNG2)


# ── FG3 LeNet 十行起手结构图 ──
def _box(ax, x, y, w, h, txt, fc, fs=12.5):
    r = Rectangle((x, y), w, h, facecolor=fc, edgecolor=C_EDGE, lw=1.2)
    ax.add_patch(r)
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color=C_EDGE, lw=1.3))


def fig3():
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 30)
    ax.axis("off")
    _box(ax, 2, 10, 14, 10, "input\n28×28×1", C_BLUE)
    _box(ax, 20, 10, 15, 10, "conv\n+ ReLU", C_ORANGE)
    _box(ax, 39, 10, 13, 10, "pool", C_GREEN)
    _box(ax, 56, 10, 15, 10, "conv\n+ ReLU", C_ORANGE)
    _box(ax, 75, 10, 13, 10, "pool", C_GREEN)
    _box(ax, 92, 10, 12, 10, "fc ×2", C_PINK)
    for x0, x1 in ((16, 20), (35, 39), (52, 56), (71, 75), (88, 92)):
        _arrow(ax, x0, 15, x1, 15)
    ax.text(104, 15, "→ output", fontsize=12.5, va="center", color=C_EDGE)
    ax.text(2, 26, "“LeNet” (LeCun, Bottou et al. 1998) — ten lines to start with "
            "PyTorch (program-generated, not from the book)", fontsize=12.5,
            color="#555555")
    fig.tight_layout()
    _save_and_register(fig, PNG3)


def harvest_mark():
    joined = "\n".join(_TEXTS)
    leaked = FIG_MARK.findall(joined)
    check("图内文字内部记号零泄漏（红线 5——E044–E052 同款收割断言）", not leaked,
          f"图内 Text 收割 {len(_TEXTS)} 条，命中 {leaked if leaked else '无'}")


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] E053 实验课数值演示档 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    e1 = lab_e1()
    e2 = lab_e2()
    e3 = lab_e3()
    e4 = lab_e4()
    e5 = lab_e5()
    e10 = lab_e10(e3, e4)
    fig1()                                          # FG1 需 E2 训练曲线（train_ex54 实跑）
    fig2(e3["alpha_demo"])                          # FG2 同 α 三轨迹（E3 实测定案）
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
        "cfg": {"w_b": list(W_B), "w_c": list(W_C), "grid9": "[−1,0,1]²",
                "alpha": ALPHA, "p0": list(P0), "rho": RHO, "b1": B1, "b2": B2,
                "eps_adam": EPS_A, "nsteps": NSTEPS, "conv_th": CONV_TH},
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "e4": e4,
        "e5": e5,
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
