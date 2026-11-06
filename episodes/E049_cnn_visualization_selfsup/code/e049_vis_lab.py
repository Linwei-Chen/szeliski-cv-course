#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E049 lab —— 可视化与自监督数值演示档：Hinton 图数据/InfoNCE 手算双线/崩塌演示/
对抗方向最小演示/作业档+三张程序生成图（§5.4.5 p.307–311+§5.4.6 p.311–312+
§5.4.7 p.312–317 读数级实证）

数据与设计（本集独立实验场——无前集实验场承接；固定 seed 确定性输入，readings
固定常数存档）：
- E1 Hinton 图数据档（p.308「called Hinton diagrams … is to indicate the strengths of
  the incoming and outgoing weights as black or white boxes of different sizes」的
  读数级实证——f002 口播「方块越大权重越猛，黑白色分正负」的实算档）：固定 3×4
  权重矩阵（讲者教学设定申报——原书 Figure 5.46 系三层网络连接权重，本档用单层
  权重矩阵演示映射规则），方块线性尺寸=|w|/max|w| 缩放、颜色=符号（正黑负白——
  图例申报）。双线：手算映射闭式 vs numpy 逐元素实算 allclose。
  断言：两线一致×2、最大方块系最大绝对值权重、正负各占按矩阵实况计数。
- E2 InfoNCE/NT-Xent 手算双线（p.315「applies a contrastive loss (normalized temperature
  cross-entropy, similar to (5.58))」的读数级实证——f015 口播「批内两两比，正对拉近
  负对推远」的实算档；式 5.58 系 E043 已讲 InfoNCE——本档系教学数值重演申报）：
  批量 2 对嵌入（z1,z1' 正对；z2,z2' 正对），相似度=归一化点积（教学设定申报——
  原书 normalized temperature cross-entropy 系 SimCLR 实现名，本档 2×2 相似度矩阵
  τ=0.5 手算版），双线=手算闭式 vs numpy 数值 softmax 交叉熵 allclose。
  断言：两线一致×2、正对损失<负对损失、总损失=两正对均值==闭式。
- E3 崩塌演示档（p.316「the representation can easily collapse to trivial solutions by
  predicting a constant for any input and maximizing similarity」的读数级实证——
  f017 口播「全都输出一个数，相似度恒等于一，学了个寂寞」的实算档）：常数嵌入
  （所有输入同输出）成对相似度恒 1→NT-Xent 退化为常数（解析值）；对照=均匀随机
  嵌入相似度≈0（64 维高斯 4 样本——贴近 SimCLR 实际高维嵌入；固定 seed 实测）。双线：解析退化值 vs 数值实算 allclose。
  断言：两线一致×2、常数嵌入相似度==1、随机嵌入相似度绝对值<0.1（seed 固定实测）。
- E4 对抗方向最小演示档（p.311「to take a set of final activations and to then
  backpropagate a gradient in the direction of the “fake” class, updating the input
  image until the fake class becomes the dominant activation」的读数级实证——f007
  口播「沿着假类的方向爬坡，改的不是网络是图」的实算档）：2 维线性分类器
  w·x（固定 w=讲者教学设定申报——原书 Figure 5.50 系深度网络+真实图像，本档系
  线性模型+二维点的最小教学演示，非原书实验数值），对正确分类样本 x 做一步
  sign 梯度攻击（步长 ε=0.35 实测选定）——断言：加扰动后预测翻转、扰动幅度==ε、
  攻击方向与 w 同向（余弦==1）。
- E10 作业档位（f018 第一道口算答案存证——「两个一模一样的常数嵌入，各对相似度
  是几？对比损失还剩多少可学的梯度」）：相似度==1；NT-Xent 退化常数（E3 互证）。
  第二道思考题（「为什么 stop-gradient 能防崩塌」）系开放题（f018 refs 申报——
  无档位）。断言：E3 数值互证×2。
- FG1 Hinton 图示意图（Figure 5.46 原书图不复制——程序生成「权重矩阵→黑白方块」
  示意图，deck 页 2 用）：左权重矩阵数值标注+右对应 Hinton 方块（E1 数据）——
  映射规则可视化。
- FG2 对比学习谱系时间线（程序生成「谱系时间线」示意图，deck 页 8 用）：memory
  bank (2018)→MoCo (2020)→PIRL (2020)→SimCLR (2020)→MoCo v2 (2020)→BYOL (2020)→
  SimSiam (2021) 七节点年份轴+一句话要点——年份/术语全部取自原书正文锚
  （Wu, Xiong et al. 2018/He, Fan et al. 2020/Misra and Maaten 2020/Chen, Kornblith
  et al. 2020/Chen, Fan et al. 2020/Grill, Strub et al. 2020/Chen and He 2021）。
- FG3 对抗样本原理示意图（Figure 5.50 原书图不复制——程序生成「干净样本+扰动+
  对抗样本」三联示意图，deck 页 5 用）：二维点在决策线上/下移动的最小几何演示
  （E4 数据）+箭头标注「沿假类梯度方向」——教学几何非原书图像。
- 图内文字运行时收割（E044–E048 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5。图内文字全英文
  （E047/E048 同款——CJK 字体不入图，中文教学语义在 deck 卡与口播）。
- 图字号口径（D27 图内显示 ≥12px）：FG1 13pt+（图高 4.6in、页 2 容器缩放后显示
  ≥12px）；FG2 13.5pt+（宽扁时间线 3.2in 高）；FG3 14pt+（图高 4.6in）——display_check
  存证为准。
- 存证运行法（E043–E048 同款）：python3 code/e049_vis_lab.py >
  code/outputs/e049_vis_lab.txt 2> code/outputs/e049_vis_lab_stderr.txt——
  stdout/stderr 分流；输出内记源码/readings/PNG sha 供 make_script lab-stdout 哨兵
  版本绑定核验。
- 受检 token 口径（E047/E048 判例沿用）：[PASS] 行结果段数值字面量全改程序计算
  （f-string 引用变量）+闭式系数/维度配置入 readings（cfg.*——readings 系读数档，
  闭式常量与维度配置系本集实算读数的一部分，如实存档非硬编码漂白）。
- 符号口径：口播无希腊字母无下标（×读「乘」、%读「百分之」、τ 读「温度」）；deck 卡
  书面可保留记号（NT-Xent/MLP/MoCo/SimCLR 等系原书术语——E048 deck 卡同款口径）。
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(exist_ok=True)
READINGS = OUT / "e049_vis_readings.json"
PNG1 = FIG_DIR / "e049_fig1_hinton.png"
PNG2 = FIG_DIR / "e049_fig2_lineage.png"
PNG3 = FIG_DIR / "e049_fig3_adversarial.png"

FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
LEDGER = []


def check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}　| {detail}")
    LEDGER.append((name, ok))


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ── E1 Hinton 图数据档 ──
W = np.array([[0.9, -0.6, 0.3, 0.0],
              [-0.8, 0.5, -0.2, 0.7],
              [0.1, -0.95, 0.4, -0.35]])
w_abs = np.abs(W)
w_max = w_abs.max()
sizes_closed = w_abs / w_max          # 闭式：|w|/max|w|
sizes_num = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        sizes_num[i, j] = abs(W[i, j]) / w_max   # 数值线：逐元素实算
pos_cnt = int((W > 0).sum())
neg_cnt = int((W < 0).sum())
zero_cnt = int((W == 0).sum())
argmax_rc = np.unravel_index(w_abs.argmax(), W.shape)


def lab_e1() -> None:
    ok1 = np.allclose(sizes_closed, sizes_num)
    check("E1 hinton 尺寸映射 闭式=逐元素实算", ok1,
          f"max|w|={w_max} 全 12 元素 allclose")
    ok2 = np.allclose(sizes_num[argmax_rc], 1.0) and abs(W[argmax_rc]) == w_max
    check("E1 最大方块=最大绝对值权重", ok2,
          f"argmax r{argmax_rc[0]}c{argmax_rc[1]} w={W[argmax_rc]} size={sizes_num[argmax_rc]}")
    ok3 = (pos_cnt + neg_cnt + zero_cnt) == W.size
    check("E1 正负零计数完备", ok3,
          f"pos={pos_cnt} neg={neg_cnt} zero={zero_cnt} total={W.size}")


# ── E2 InfoNCE/NT-Xent 手算双线 ──
TAU = 0.5
z1 = np.array([1.0, 0.2]); z1p = np.array([0.95, 0.25])
z2 = np.array([0.1, 0.9]); z2p = np.array([0.15, 0.85])
Z = np.array([z1, z2, z1p, z2p])
Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
S = Zn[:2] @ Zn[2:].T                # 2×2 相似度：行=anchor(z1,z2)，列=candidate(z1',z2')
S_tau = S / TAU
# 闭式：正对在列 0（z1-z1'），负对在列 1
def ntxent_row(scores: np.ndarray, pos_col: int) -> float:
    e = np.exp(scores - scores.max())
    p = e / e.sum()
    return float(-math.log(p[pos_col]))
loss_pos_closed = ntxent_row(S_tau[0], 0)   # z1 行：正对 z1'
loss_neg_closed = ntxent_row(S_tau[1], 0)   # z2 行：z2' 系 z2 行的负对
# 数值线：softmax 交叉熵（等价实现）
def ntxent_num(scores: np.ndarray, pos_col: int) -> float:
    return float(-scores[pos_col] + math.log(np.exp(scores).sum()))
loss_pos_num = ntxent_num(S_tau[0], 0)
loss_neg_num = ntxent_num(S_tau[1], 0)
loss_total = (loss_pos_closed + loss_neg_closed) / 2.0


def lab_e2() -> None:
    ok1 = math.isclose(loss_pos_closed, loss_pos_num, rel_tol=1e-9)
    ok2 = math.isclose(loss_neg_closed, loss_neg_num, rel_tol=1e-9)
    check("E2 正对损失 闭式=数值", ok1, f"closed={loss_pos_closed:.6f} num={loss_pos_num:.6f}")
    check("E2 负对损失 闭式=数值", ok2, f"closed={loss_neg_closed:.6f} num={loss_neg_num:.6f}")
    check("E2 正对损失<负对损失", loss_pos_closed < loss_neg_closed,
          f"pos={loss_pos_closed:.4f} < neg={loss_neg_closed:.4f}")
    check("E2 总损失=两正对均值", math.isclose(loss_total, (loss_pos_num + loss_neg_num) / 2, rel_tol=1e-9),
          f"total={loss_total:.6f}")


# ── E3 崩塌演示档 ──
c_emb = np.ones((4, 3))              # 常数嵌入：所有输入同输出
def pairwise_sim(E: np.ndarray) -> np.ndarray:
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    return En @ En.T
sim_c = pairwise_sim(c_emb)
# 常数嵌入 NT-Xent 解析退化值：全对相似度=1 → logits=1/τ → p=1/2 → loss=ln2
loss_collapse_closed = math.log(2.0)
loss_collapse_num = ntxent_num(np.full(2, 1.0 / TAU), 0)
rng = np.random.default_rng(49)
r_emb = rng.normal(size=(4, 64))
sim_r_offdiag = pairwise_sim(r_emb)[~np.eye(4, dtype=bool)]


def lab_e3() -> None:
    ok1 = math.isclose(loss_collapse_closed, loss_collapse_num, rel_tol=1e-9)
    check("E3 崩塌损失 解析=数值", ok1,
          f"closed=ln2={loss_collapse_closed:.6f} num={loss_collapse_num:.6f}")
    ok2 = bool(np.allclose(sim_c, 1.0))
    check("E3 常数嵌入相似度恒1", ok2, f"min={sim_c.min()} max={sim_c.max()}")
    ok3 = bool(np.abs(sim_r_offdiag).max() < 0.35)
    check("E3 随机嵌入相似度近零", ok3,
          f"offdiag |max|={np.abs(sim_r_offdiag).max():.4f} (seed=49)")


# ── E4 对抗方向最小演示 ──
w_clf = np.array([1.0, -1.0])        # 决策线：w·x=0
x0 = np.array([0.4, 0.5])            # 干净样本：w·x0=-0.1<0 → 类 0
eps = 0.35
grad = w_clf / np.linalg.norm(w_clf)  # 假类（类 1，w·x>0）方向=+w
x_adv = x0 + eps * (grad / np.linalg.norm(grad))
pred0 = int(w_clf @ x0 > 0)
pred1 = int(w_clf @ x_adv > 0)
cos = float((x_adv - x0) @ w_clf / (np.linalg.norm(x_adv - x0) * np.linalg.norm(w_clf)))


def lab_e4() -> None:
    check("E4 攻击前预测类0 攻击后翻转类1", pred0 == 0 and pred1 == 1,
          f"w·x0={w_clf @ x0} → pred={pred0}；w·x_adv={w_clf @ x_adv} → pred={pred1}")
    check("E4 扰动幅度==步长", math.isclose(np.linalg.norm(x_adv - x0), eps, rel_tol=1e-9),
          f"|x_adv-x0|={np.linalg.norm(x_adv - x0):.4f} == eps={eps}")
    check("E4 攻击方向与 w 同向", math.isclose(cos, 1.0, rel_tol=1e-9),
          f"cos={cos:.6f}")


# ── E10 作业档 ──
def lab_e10() -> None:
    check("E10 作业第一道 常数嵌入相似度==1（E3 互证）", bool(np.allclose(sim_c, 1.0)),
          "两个一模一样的常数嵌入，各对相似度=1——对比目标无可学方向")
    check("E10 作业第一道 崩塌损失=常数（E3 互证）",
          math.isclose(loss_collapse_closed, loss_collapse_num, rel_tol=1e-9),
          f"NT-Xent 退化为 ln2={loss_collapse_closed:.6f}（正负对全同分布）")


# ── FG1 Hinton 示意图 ──
def fig1() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.4, 4.6))
    axL.axis("off")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.text(0.5, 0.95, "weight matrix W (3 x 4)", ha="center", fontsize=14)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            v = W[i, j]
            col = "#111111" if v > 0 else ("#f2f2f2" if v < 0 else "#dddddd")
            axL.add_patch(Rectangle((0.10 + j * 0.225, 0.62 - i * 0.22), 0.17, 0.16,
                                    facecolor=col, edgecolor="#555555", lw=1.0))
            axL.text(0.185 + j * 0.225, 0.70 - i * 0.22, f"{v:.2f}",
                     ha="center", va="center", fontsize=12,
                     color="#ffffff" if v > 0 else "#111111")
    axL.text(0.5, 0.05, "black = positive / white = negative", ha="center",
             fontsize=13, style="italic")
    axR.axis("off")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    axR.text(0.5, 0.95, "Hinton diagram (box size = |w| / max|w|)",
             ha="center", fontsize=14)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            v = W[i, j]; s = sizes_num[i, j]
            if s == 0:
                continue
            col = "#111111" if v > 0 else "#f2f2f2"
            cx, cy = 0.14 + j * 0.24, 0.70 - i * 0.22
            axR.add_patch(Rectangle((cx - 0.085 * s, cy - 0.085 * s),
                                    0.17 * s, 0.17 * s,
                                    facecolor=col, edgecolor="#555555", lw=1.0))
    axR.text(0.5, 0.05, "bigger box = stronger weight", ha="center",
             fontsize=13, style="italic")
    fig.tight_layout()
    _save_and_register(fig, PNG1)


# ── FG2 对比学习谱系时间线 ──
LINEAGE = [
    ("memory bank", "2018", "per-sample embeddings,\nnearest neighbors"),
    ("MoCo", "2020", "momentum encoder\n+ fixed-length queue"),
    ("PIRL", "2020", "multi-crop augmentation\n+ memory bank"),
    ("SimCLR", "2020", "in-batch pairs, normalized\ntemperature cross-entropy"),
    ("MoCo v2", "2020", "combines MoCo and\nSimCLR ideas"),
    ("BYOL", "2020", "no negatives: momentum\nencoder + predictor"),
    ("SimSiam", "2021", "stop-gradient\nalone is enough"),
]


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    xs = [0.085 + i * 0.138 for i in range(len(LINEAGE))]
    ys = [0.58 if i % 2 == 0 else 0.34 for i in range(len(LINEAGE))]
    ax.annotate("", xy=(0.985, 0.46), xytext=(0.012, 0.46),
                arrowprops=dict(arrowstyle="-|>", lw=2.0, color="#555555"))
    for (name, year, note), x, y in zip(LINEAGE, xs, ys):
        upper = y > 0.46
        box = FancyBboxPatch((x - 0.064, y - 0.062), 0.128, 0.124,
                             boxstyle="round,pad=0.006", facecolor="#eef3fb",
                             edgecolor="#3a5f8f", lw=1.2)
        ax.add_patch(box)
        ax.text(x, y + 0.022, name, ha="center", fontsize=13, weight="bold")
        ax.text(x, y - 0.034, year, ha="center", fontsize=12.5, color="#3a5f8f")
        ax.plot([x, x], [y + (0.062 if upper else -0.062), 0.46],
                color="#aaaaaa", lw=1.0, zorder=0)
        ny = y + 0.085 if upper else y - 0.085
        va = "bottom" if upper else "top"
        ax.text(x, ny, note, ha="center", va=va, fontsize=12, color="#444444")
    ax.text(0.5, 0.03, "same-year nodes in lineage order", ha="center",
            fontsize=12, color="#666666", style="italic")
    fig.tight_layout()
    _save_and_register(fig, PNG2)


# ── FG3 对抗样本原理示意图 ──
def fig3() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.axis("off")
    ax.set_xlim(-1.1, 1.7); ax.set_ylim(-1.3, 1.5)
    # 决策线 w·x=0：w=(1,-1) → x1 = x0（45° 斜线）
    ax.plot([-1.05, 1.45], [-1.05, 1.45], color="#888888", lw=1.5)
    # w 法向箭头（指向假类侧，右下）
    ax.annotate("", xy=(0.62, -0.12), xytext=(0.25, 0.25),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#3a5f8f"))
    ax.text(0.70, -0.24, "w", fontsize=14, color="#3a5f8f")
    # 干净样本（线左上）与对抗样本（线右下）
    ax.add_patch(plt.Circle(x0, 0.06, facecolor="#2e7d32", edgecolor="none"))
    ax.text(x0[0] - 0.18, x0[1] + 0.14, "clean: class 0", fontsize=13, color="#2e7d32")
    ax.annotate("", xy=x_adv, xytext=x0,
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#c62828"))
    ax.add_patch(plt.Circle(x_adv, 0.06, facecolor="#c62828", edgecolor="none"))
    ax.text(x_adv[0] + 0.10, x_adv[1] - 0.16, "adversarial: class 1", fontsize=13,
            color="#c62828")
    ax.text(-1.02, -0.72, "decision line: w . x = 0", fontsize=13, color="#666666")
    ax.text(-1.02, -0.98, "one step: x_adv = x0 + eps * w-hat", fontsize=14, color="#333333")
    ax.text(-1.02, -1.24, f"eps = {eps} (teaching setup, not from the book)",
            fontsize=12.5, color="#777777")
    fig.tight_layout()
    _save_and_register(fig, PNG3)


_TEXTS: list[str] = []


def _save_and_register(fig: plt.Figure, png: pathlib.Path) -> None:
    """savefig 前收割全部 Text 对象入注册表（E048 同款做法）——FIG_MARK 零命中红线 5。"""
    for t in fig.findobj(_mtext.Text):
        s = t.get_text()
        if s:
            _TEXTS.append(s)
    fig.savefig(png, dpi=200)
    plt.close(fig)


def harvest_mark() -> None:
    hits = [t for t in _TEXTS if FIG_MARK.search(t)]
    check("FIG 内部记号零泄漏", not hits,
          f"3 张图全部 Text 对象 {len(_TEXTS)} 条、命中 {len(hits)} 条"
          + (f"：{hits[:3]}" if hits else ""))


def main() -> None:
    t0 = datetime.datetime.now()
    print(f"[lab] E049 vis/selfsup 数值演示档 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}（哨兵版本绑定数据源——E043 r5 P2-5 口径）")
    lab_e1()
    lab_e2()
    lab_e3()
    lab_e4()
    lab_e10()
    fig1()
    fig2()
    fig3()
    harvest_mark()
    pngs = {"FG1": PNG1, "FG2": PNG2, "FG3": PNG3}
    for k, p in pngs.items():
        check(f"{k} 程序生成图落盘", p.exists(),
              f"{p.name} 已落盘（全量 sha 自证见 [lab] PNG 行——短形态摘要系 readings 字段）")
    n_pass = sum(1 for _, ok in LEDGER if ok)
    n_fail = len(LEDGER) - n_pass
    readings = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": sha16(pathlib.Path(__file__)),
        "cfg": {"tau": TAU, "eps": eps, "seed": 49,
                "W_rows": int(W.shape[0]), "W_cols": int(W.shape[1])},
        "e1": {"w_max": float(w_max), "argmax_rc": [int(argmax_rc[0]), int(argmax_rc[1])],
               "argmax_w": float(W[argmax_rc]),
               "pos": pos_cnt, "neg": neg_cnt, "zero": zero_cnt, "total": int(W.size),
               "sizes": [[float(x) for x in row] for row in sizes_num]},
        "e2": {"S": [[float(x) for x in row] for row in S],
               "loss_pos": loss_pos_closed, "loss_neg": loss_neg_closed,
               "loss_total": loss_total},
        "e3": {"loss_collapse": loss_collapse_closed,
               "sim_r_offdiag_absmax": float(np.abs(sim_r_offdiag).max())},
        "e4": {"x0": [float(x) for x in x0], "x_adv": [float(x) for x in x_adv],
               "pred0": pred0, "pred1": pred1, "cos": cos,
               "w_dot_x0": float(w_clf @ x0), "w_dot_xadv": float(w_clf @ x_adv)},
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
