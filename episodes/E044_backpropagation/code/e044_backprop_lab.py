#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E044 lab —— 反向传播实验：全链链式法则梯度+示意图核验+作业档位（§5.3.5 后半 读数级实证）

数据与集谱（承接 E042/E043 同一实验场）：
- 主实验场：seed 2042，一维曲线回归——40 点 x∼U[−3,3]，y=sin(x)+N(0,0.05)；固定随机
  ReLU 基 12 个（W1,b1 与 E042/E043 同种子同分布生成——「上一集固定随机的基，本集把
  梯度流回它们的每一根」）。网络=单隐层：x →(加权和 w_k x+b_k)→ ReLU →(线性 c)→ 输出，
  恰是 Figure 5.31 结构的最小版（一个中间层+误差回传）。
- E1 全链链式法则梯度 vs 数值差分（讲者实验，本集主实验——四式+回传式全套实算）：
  能量 E(θ)=mean‖yhat−y‖²（均值口径，E043 E1 同款系数申报：原书式 5.63 的单样本形式
  ∂E_n/∂y=y−t 为原书系数约定，本 lab 不复刻原书系数、只验证梯度结构）。
  链式法则解析梯度（对应原书四式 5.65–5.68+回传式 5.70，单隐层版）：
  ①输出层误差 δ=2(yhat−y)/N（式 5.63 形态：预测减目标×均值系数）；
  ②∂E/∂c=[H,1]ᵀδ（式 5.66：输入×误差——输出层的输入就是 H）；
  ③隐层单元误差 e_k=δ·c_k·1[S_k>0]（式 5.70 单隐层版：下游误差 c_k δ 加权×激活斜率，
     ReLU 的斜率=1[S>0]——「闸门」：死单元斜率 0，误差传不过去）；
  ④∂E/∂W1_k=e_k·x（式 5.66：输入×误差）；∂E/∂b1_k=e_k（式 5.67：误差本身）。
  中心差分 vs 解析式逐分量「绝对差 <1e-8 或相对 <1e-5」（37 参数：c 13+W1 12+b1 12；判据口径申报见 E1 代码注记——近零梯度分量纯相对判据被分母放大判例）——「梯度流回每一根
  旋钮」的实算；隐层死单元个数程序计数入 readings（E042 ReLU die 呼应读数）。
- E9 示意图核验档（f009 deck 图数字出处——图内数字现场可验算）：下游误差 [0.6,0.2,0.4]、
  权重 [0.5,1.0,0.25]、激活斜率 0.8——加权和=0.5×0.6+1.0×0.2+0.25×0.4、本层误差=0.8×加权和
  断言入 readings e9_diagram（数值叶值 numeric 机检同值命中）。
- E10 作业档位（f014 第一道手算答案存证）：2 输入→2 ReLU（w/b 手工指定）→1 线性输出，
  单样本 L2；解析手算 ∂E/∂w_(1,1)=e_1·x_1 与数值差分 rel<1e-8 入 readings e10_homework；
  单元 2 设计为死单元（加权和<0，ReLU 输出 0，误差 0——f010 直觉段呼应读数）。
- 图内文字运行时收割（E043 E6 同款对象遍历）：fig.findobj(Text) 收割，FIG_MARK
  （E0\\d\\d|f0\\d\\d|门A|§|speak）零命中断言——内部记号零泄漏红线 5；不可见对象构成
  实测计数（Tick.label2 身份实测——E043 r6 P1-1 口径）。
- 存证运行法（E043 同款）：python3 code/e044_backprop_lab.py > code/outputs/e044_backprop_lab.txt
  2> code/outputs/e044_backprop_lab_stderr.txt——stdout/stderr 分流；输出内记源码/readings/
  PNG sha 供 make_script lab-stdout 哨兵版本绑定核验。
- 符号口径：θ=(c, W1, b1) 全部 37 参数参与链式回传（E043 只训 c——本集「梯度流回基参数」
  =新内容申报）；δ 系数 2/N 系均值口径（E043 E1 同款，不复刻原书单样本系数）。
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
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(exist_ok=True)

T0 = datetime.datetime.now()
print(f"[lab] 开始 {T0.isoformat(timespec='seconds')}")
print(f"[lab] 输入 锚证据 sha256={hashlib.sha256((EP / 'production/evidence/e044_book_ch5_anchors.txt').read_bytes()).hexdigest()[:16]}…")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}"
      f"（哨兵版本绑定数据源——E043 r5 P2-5 口径）")

plt.rcParams["font.family"] = ["Songti SC", "STSong", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18})

check_log = []


def check(name, cond, detail):
    check_log.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}　| {detail}")


# E6 运行时收割器（E043 同款对象遍历+不可见构成实测）
ALL_TEXTS = []
E6_PER_FIG = []
E6_INV_LABEL2 = 0
E6_INV_OTHER = 0


def harvest_texts(fig):
    label2_ids = set()
    for _ax in fig.axes:
        for _axis in (_ax.xaxis, _ax.yaxis):
            for _tk in list(_axis.get_major_ticks()) + list(_axis.get_minor_ticks()):
                label2_ids.add(id(_tk.label2))
    n_all = n_vis = 0
    for t in fig.findobj(_mtext.Text):
        s = t.get_text()
        if s and s.strip():
            ALL_TEXTS.append(s)
            n_all += 1
            if t.get_visible():
                n_vis += 1
            elif id(t) in label2_ids:
                globals()["E6_INV_LABEL2"] += 1
            else:
                globals()["E6_INV_OTHER"] += 1
    E6_PER_FIG.append((n_all, n_vis))


FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|\bspeak\b")

# ════════════════════════════════════════════════════════════
# 数据：seed 2042 sin 曲线 40 点（E042/E043 同场）
# ════════════════════════════════════════════════════════════
rng = np.random.default_rng(2042)
X = np.sort(rng.uniform(-3, 3, 40))
Y = np.sin(X) + rng.normal(0, 0.05, 40)

K = 12
W1 = rng.normal(0, 1.0, K)
b1 = rng.uniform(-1.5, 1.5, K)
C = rng.normal(0, 0.5, K + 1)          # 输出层权重（含偏置列）——起点随机，非训练结果


def forward(W1_, b1_, C_):
    """前向：返回 yhat 与缓存（S/H——§8 工程段呼应：反向传播需要前向激活）。"""
    S = W1_[None, :] * X[:, None] + b1_[None, :]      # 40×12 加权和
    H = np.maximum(0.0, S)                            # ReLU 激活
    H1 = np.column_stack([H, np.ones_like(X)])        # 增偏置列
    yhat = H1 @ C_
    return yhat, (S, H, H1)


def loss(W1_, b1_, C_):
    yhat, _ = forward(W1_, b1_, C_)
    return float(np.mean((yhat - Y) ** 2))


def analytic_grad(W1_, b1_, C_):
    """链式法则解析梯度（docstring 四式+回传式）。"""
    yhat, (S, H, H1) = forward(W1_, b1_, C_)
    delta = 2.0 * (yhat - Y) / len(Y)                 # 输出层误差（预测减目标×均值系数）
    gC = H1.T @ delta                                 # 式 5.66（输出层）
    e = (delta[:, None] * C_[None, :12]) * (S > 0)    # 式 5.70 单隐层版（下游加权×ReLU 斜率）
    gW1 = e.T @ X                                     # 式 5.66（输入×误差）
    gb1 = e.sum(axis=0)                               # 式 5.67（误差本身，40 样本求和）
    return gC, gW1, gb1, e, S


def numeric_grad(eps=1e-6):
    """中心差分（全部 37 参数）。"""
    gW1 = np.zeros_like(W1)
    gb1 = np.zeros_like(b1)
    gC = np.zeros_like(C)
    for k in range(K):
        for which in ("w", "b"):
            base = W1 if which == "w" else b1
            plus, minus = base.copy(), base.copy()
            plus[k] += eps
            minus[k] -= eps
            lp = loss(W1, b1, C) if which == "w" else loss(W1, b1, C)
            if which == "w":
                lp, lm = loss(plus, b1, C), loss(minus, b1, C)
                gW1[k] = (lp - lm) / (2 * eps)
            else:
                lp, lm = loss(W1, plus, C), loss(W1, minus, C)
                gb1[k] = (lp - lm) / (2 * eps)
    for k in range(len(C)):
        cp, cm = C.copy(), C.copy()
        cp[k] += eps
        cm[k] -= eps
        gC[k] = (loss(W1, b1, cp) - loss(W1, b1, cm)) / (2 * eps)
    return gC, gW1, gb1


# ════════════════════════════════════════════════════════════
# E1 全链链式法则 vs 数值差分（37 参数）
# ════════════════════════════════════════════════════════════
gC, gW1, gb1, e_mat, S_mat = analytic_grad(W1, b1, C)
nC, nW1, nb1 = numeric_grad()
# 判据=绝对/相对混合（判据口径申报：基偏置存在近零梯度分量（如 b1[7]=6.7e-06，该单元仅
# 极少样本激活），纯相对判据被分母放大（绝对差 2.5e-10 却报 rel 3.4e-05 判例）——统一改
# 「绝对差 <1e-8 或相对 <1e-5」，三侧同判据）
def grad_ok(ga, gn):
    ad = float(np.max(np.abs(ga - gn)))
    rl = float(np.max(np.abs(ga - gn) / (np.abs(gn) + 1e-12)))
    return (ad < 1e-8) or (rl < 1e-5), ad, rl


okC, adC, relC = grad_ok(gC, nC)
okW, adW, relW = grad_ok(gW1, nW1)
okB, adB, relB = grad_ok(gb1, nb1)
dead = int((S_mat.max(axis=0) <= 0).sum())
print(f"[e1] 全链梯度 37 参数（c {len(C)}+W1 {K}+b1 {K}）——式 5.66/5.67/5.70 单隐层全套")
check("E1 解析链式梯度 vs 数值差分（输出层 c）", okC, f"max_abs={adC:.3e} max_rel={relC:.3e}")
check("E1 解析链式梯度 vs 数值差分（基权重 W1——式 5.66 输入×误差）", okW, f"max_abs={adW:.3e} max_rel={relW:.3e}")
check("E1 解析链式梯度 vs 数值差分（基偏置 b1——式 5.67 误差本身）", okB, f"max_abs={adB:.3e} max_rel={relB:.3e}")
check("E1 死单元读数（E042 ReLU die 呼应）", dead >= 0, f"死单元 {dead}/{K}（起点随机 ReLU 基，max(S)≤0 判）")

# ════════════════════════════════════════════════════════════
# E9 示意图核验档（f009 图上数字出处——现场可验算）
# ════════════════════════════════════════════════════════════
e_down = np.array([0.6, 0.2, 0.4])
w_conn = np.array([0.5, 1.0, 0.25])
slope = 0.8
weighted = float(w_conn @ e_down)
e_this = float(slope * weighted)
print(f"[e9] 示意图核验：加权和={weighted}（0.5×0.6+1.0×0.2+0.25×0.4）、本层误差={e_this}（0.8×加权和）")
check("E9 加权和=0.5×0.6+1.0×0.2+0.25×0.4", abs(weighted - 0.6) < 1e-12, f"weighted={weighted}")
check("E9 本层误差=斜率×加权和（式 5.70 白话实算）", abs(e_this - 0.48) < 1e-12, f"e_this={e_this}")

# ════════════════════════════════════════════════════════════
# E10 作业档位（f014 第一道手算答案——2 输入→2 ReLU→1 输出）
# ════════════════════════════════════════════════════════════
xh = np.array([0.5, -0.5])
Wh = np.array([[0.8, 0.4], [-0.6, 0.2]])
bh = np.array([0.1, -0.1])
ch = np.array([1.0, -1.0, 0.3])
th = 1.0
sh = Wh @ xh + bh                       # [0.3, -0.5]
hh = np.maximum(0.0, sh)                # [0.3, 0.0]——单元 2 死单元
yhat_h = float(ch[:2] @ hh + ch[2])
loss_h = float((yhat_h - th) ** 2)
delta_h = 2.0 * (yhat_h - th)           # 单样本均值口径系数 2/1
e1_h = float(delta_h * ch[0] * (sh[0] > 0))
w11_grad = float(e1_h * xh[0])          # 手算：∂E/∂w11 = e1·x1
eps = 1e-7


def loss_hw(w11):
    W2 = Wh.copy()
    W2[0, 0] = w11
    s2_ = W2 @ xh + bh
    h2_ = np.maximum(0.0, s2_)
    yh2 = float(ch[:2] @ h2_ + ch[2])
    return (yh2 - th) ** 2


w11_num = (loss_hw(w11=Wh[0, 0] + eps) - loss_hw(w11=Wh[0, 0] - eps)) / (2 * eps)
rel10 = abs(w11_grad - w11_num) / (abs(w11_num) + 1e-12)
print(f"[e10] 作业小例：前向 yhat={yhat_h:.4f}（单元 2 死单元 h=0）、手算 ∂E/∂w11={w11_grad}、"
      f"数值差分={w11_num:.6e}")
check("E10 手算梯度 vs 数值差分（f014 作业第一道答案）", rel10 < 1e-8,
      f"解析 {w11_grad} vs 数值 {w11_num:.3e}（rel={rel10:.2e}）")
check("E10 死单元读数（单元 2 加权和<0——f010 直觉段呼应）", hh[1] == 0.0 and sh[1] < 0,
      f"s2={sh[1]}（ReLU 输出 0、误差 0——梯度断流实证）")

# ════════════════════════════════════════════════════════════
# 图1：双向流水线示意图（E9 数字标注——程序生成示意，非原书图）
# 版式 v2（overflow_probe 页 6 压线整改+图文对齐整改）：v1 画成三行一对一、圆圈 e=
#   错标下游值——与 f009 口播「每到一个单元，先按连接权重把下游误差加权汇总，再乘
#   斜率」矛盾（式 5.69/5.70 是**一个单元**收三条下游误差：加权汇总→乘斜率）。
#   v2：焦点单元（中间层中座）收三条红虚线，e=0.48 落焦点圆内=式 5.70 两步的产出；
#   连接权重入算账面板（箭头不标——配对由加权和表达式显式给出，避免 v1 权重标签与
#   面板文字重叠判例）；下层三单元=层次背景（阴影框=Figure 5.31 口径，口播已申报）。
#   画布 10×5in=1500×750（列宽绑死显示缩放整改：deck 右栏 782px/1500 → 12.5pt 显示
#   ≈13.2 CSS px；v1 1725×930 时 782/1725→11.8px 破 D27 图内线）。
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

xin = (0.8, 2.6)                       # 输入节点
mid = [(4.3, 3.5), (4.3, 2.6), (4.3, 1.7)]   # 中间层三单元（阴影框；中座=焦点单元）
focal = mid[1]
down = [(8.6, 4.1), (8.6, 2.75), (8.6, 1.4)]  # 下游三单元
box = plt.Rectangle((3.6, 0.9), 1.5, 3.3, fill=True, facecolor="#fff3cd",
                    edgecolor="#c8a415", lw=1.8, zorder=1)
ax.add_patch(box)
ax.text(4.3, 4.55, "中间层", fontsize=13.5, ha="center", color="#6b5626")
ax.text(8.6, 4.72, "下游", fontsize=13.5, ha="center", color="#6b5626")

# 前向（绿实线）：输入→中间层三个单元；焦点单元→下游三单元
for (mx, my) in mid:
    ax.annotate("", xy=(mx - 0.45, my), xytext=(xin[0] + 0.42, xin[1]),
                arrowprops=dict(arrowstyle="-|>", color="#1a7f37", lw=2.2), zorder=2)
for (dx, dy) in down:
    ax.annotate("", xy=(dx - 0.45, dy), xytext=(focal[0] + 0.45, focal[1]),
                arrowprops=dict(arrowstyle="-|>", color="#1a7f37", lw=2.2), zorder=2)
    # 反向（红虚线）：下游误差按原路传回焦点单元（式 5.69 的 w_ki e_k 汇总）
    ax.annotate("", xy=(focal[0] + 0.45, focal[1] + 0.14), xytext=(dx - 0.45, dy + 0.14),
                arrowprops=dict(arrowstyle="-|>", color="#c62828", lw=1.8,
                                linestyle=(0, (5, 3)), alpha=0.9), zorder=3)

# 圆与误差标注：下游三圆 e=下游误差；焦点圆 e=本层误差（红=式 5.70 产出）；
#   上下两座=层次背景不标数值（口播只讲焦点单元的账）
for (dx, dy), ev in zip(down, e_down):
    ax.add_patch(plt.Circle((dx, dy), 0.34, facecolor="#d5e8d4", edgecolor="#333", lw=1.5, zorder=4))
    ax.text(dx, dy, f"e={ev}", fontsize=12.5, ha="center", va="center", zorder=5)
for (mx, my) in (mid[0], mid[2]):
    ax.add_patch(plt.Circle((mx, my), 0.34, facecolor="#e8daef", edgecolor="#333", lw=1.5, zorder=4))
ax.add_patch(plt.Circle(focal, 0.34, facecolor="#e8daef", edgecolor="#c62828", lw=2.0, zorder=4))
ax.text(focal[0], focal[1], f"e={e_this}", fontsize=12.5, ha="center", va="center",
        zorder=5, color="#c62828")

ax.add_patch(plt.Circle(xin, 0.34, facecolor="#dae8fc", edgecolor="#333", lw=1.5, zorder=4))
ax.text(xin[0], xin[1], "输入", fontsize=12.5, ha="center", va="center", zorder=5)
ax.text(2.35, 3.6, "前向：激活", fontsize=14.5, color="#1a7f37", ha="center")

# 算账面板（右下）：式 5.69（加权和）→式 5.70（乘斜率）两步；配对由表达式显式给出
ax.text(6.3, 1.15, "反向流（本层算账）", fontsize=14.5, color="#c62828", ha="left")
ax.text(6.3, 0.80, f"连接权重 {w_conn[0]}、{w_conn[1]}、{w_conn[2]}", fontsize=14, ha="left")
ax.text(6.3, 0.45,
        f"加权和 {w_conn[0]}×{e_down[0]}+{w_conn[1]}×{e_down[1]}+{w_conn[2]}×{e_down[2]} = {weighted}",
        fontsize=14, ha="left")
ax.text(6.3, 0.10, f"× 斜率 {slope} → 本层误差 = {e_this}", fontsize=14, ha="left")
ax.text(0.3, 0.15, "程序生成示意（非原书图），结构借原书 Figure 5.31 口径",
        fontsize=12.5, ha="left", color="#555")

ax.set_title("双向流水线：前向算激活（绿实线），反向传误差（红虚线）", fontsize=18)
fig.tight_layout()
fig.savefig(FIG / "e044_fig1_backprop_diagram.png", dpi=150)
harvest_texts(fig)
plt.close(fig)

# ════════════════════════════════════════════════════════════
# 图内文字收割断言（E043 E6 口径：对象遍历+内部记号零命中+不可见构成实测）
# ════════════════════════════════════════════════════════════
hits = [s for s in ALL_TEXTS if FIG_MARK.search(s)]
check("E6 图内文字收割（对象遍历）+内部记号零命中", len(ALL_TEXTS) > 0 and not hits,
      f"收割 {len(ALL_TEXTS)} 条 Text 对象、内部记号命中 {len(hits)}"
      + (f"：{hits[:3]}" if hits else ""))

# ════════════════════════════════════════════════════════════
# readings 落盘 + 存证自证
# ════════════════════════════════════════════════════════════
readings = {
    "data": {"n": int(len(X)), "k": K, "seed": 2042,
             "note": "E042/E043 同场（seed 2042 sin 40 点+ReLU 基 12）——本集 θ=(c,W1,b1) 全部 37 参数参与链式回传"},
    "e1_numeric_vs_analytic": {
        "rel_c": relC, "rel_w1": relW, "rel_b1": relB,
        "abs_c": adC, "abs_w1": adW, "abs_b1": adB,
        "n_params": int(len(C) + 2 * K), "dead_units": dead,
        "eps": 1e-6,
        "note": "式 5.66/5.67/5.70 单隐层解析 vs 中心差分；dead_units=起点随机基 max(S)≤0 计数"
                "（E042 ReLU die 呼应读数——f010 口播「误差为零，权重梯度也为零」实据）"},
    "e9_diagram": {
        "e_down": [0.6, 0.2, 0.4], "w_conn": [0.5, 1.0, 0.25], "slope": 0.8,
        "weighted": weighted, "e_this": e_this,
        "note": "f009 示意图数字出处——加权和与本层误差实算断言；图内数字现场可验算"},
    "e10_homework": {
        "x": [0.5, -0.5], "w11_grad_analytic": w11_grad, "w11_grad_numeric": float(w11_num),
        "rel": float(rel10), "yhat": yhat_h, "loss": loss_h,
        "dead_unit2": {"s": float(sh[1]), "h": 0.0},
        "note": "f014 作业第一道答案存证：∂E/∂w11=e1·x1（式 5.66）；单元 2 死单元=∇断流实证（f010 呼应）"},
    "e6_figtext": {
        "per_fig_all": E6_PER_FIG, "inv_label2": E6_INV_LABEL2, "inv_other": E6_INV_OTHER,
        "note": "收割口径=E043 E6（findobj(Text) 对象遍历；visible 标志为真计数；不可见按 "
                "Tick.label2 身份实测、其余只计数不归因——E043 r6 P1-1 口径）"},
}
(OUT / "e044_backprop_readings.json").write_text(
    json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e044_backprop_readings.json').read_bytes()).hexdigest()}")

n_pass = sum(1 for _, ok, _ in check_log if ok)
print(f"[lab] LEDGER：断言 {len(check_log)} 条，PASS {n_pass}")
T1 = datetime.datetime.now()
for fpath in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({fpath.name})={hashlib.sha256(fpath.read_bytes()).hexdigest()}")
print(f"[lab] 结束 {T1.isoformat(timespec='seconds')}（起 {T0.isoformat(timespec='seconds')}）")
if n_pass != len(check_log):
    sys.exit(1)
