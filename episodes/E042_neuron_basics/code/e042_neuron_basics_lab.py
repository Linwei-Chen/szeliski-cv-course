#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E042 lab —— 神经元最小单元实验：加权求和+激活函数（§5.3.1/§5.3.2 读数级实证）

数据与集谱：
- 主实验场：seed 2042，一维曲线回归——40 点 x∼U[−3,3]，y=sin(x)+N(0,0.05)。
  选 sin 是因为它「直线拟合永远学不会」（讲者白话——原书无此句，实验实证支撑）；40 点
  小样本保证一维散点图可读。
- E1 线性塌缩复合实证（**讲者实验，锚 §4.2/§5 申报：原书 5.3.1–5.3.2 无「stacked linear
  layers collapse to linear」显式论证句**——线性层复合仍为线性属线性代数事实
  （W₂(W₁x+b₁)+b₂=W_eff x+b_eff，矩阵乘法结合律），本 lab 以实验实证呈现，不冒充原书）：
  固定随机第一层 W₁,b₁（12 单元），第二层由线性最小二乘闭式最优——最优两层**线性**网络
  的能量=单层 OLS 能量（rel<1e-9）——没有弯，两层白搭。
- E2 ReLU 基拟合（讲者实验）：固定随机方向的 ReLU 基 φ_k(x)=max(0, w_k x+b_k)（k=1..12，
  w_k,b_k 固定不训练——训练循环是 E043 的事），输出系数 c 由线性最小二乘解（**E041 工具
  闭环：上集的线性最小二乘就是本集神经网络的最后一层求解器**——门A r1 P1-3：本场景
  ΦᵀΦ 奇异，正规方程逆不存在，实现用 np.linalg.lstsq（SVD 最小范数解）；解不唯一但均满足
  正规方程、能量唯一）——非线性拟合成功，E_relu ≪ E_ols。
- E3 线性基对照（同场塌缩实锤）：同样的 12 个「基」但激活=恒等（线性基 φ_k(x)=w_k x+b_k），
  最小二乘解——E_lin=E_ols（rel<1e-9）：把弯拿掉，两层退化回一层。
- E4 ReLU die 数值演示（原书 p.273 Karpathy 警告照录锚 §3）：**手工配置一个死亡单元演示
  死亡状态的后果（门A r1 P2-8 措辞降档：未复现「大梯度致死」过程）**——取 w=-1.0、b=-3.5，
  全数据 pre=wx+b<0 → 输出恒 0；∂E/∂w=Σ1_{pre>0}(h−y)x 按定义恒 0——
  「死了：永远不激活，永远没梯度」。原书警告全文（含尾句 learning rate）见锚 v6 §3。
- E5 矩阵形式一致性（式 5.47 层加权和 s=Wx+单元级偏置（式 5.45））：逐单元循环
  si=wiᵀxi+bi→yi=h(si) vs 层矩阵 s=Wx+b→h(s) 逐元一致（rel<1e-12）——「把一排神经元
  打包成矩阵乘法」的实算。
- 符号口径：一维输入 x 标量（E1–E4）；E5 用 d=5 维输入 x∈R⁵、W∈R^{k×d}（真矩阵乘法——
  门A r1 P1-4 整改）；最小二乘解=np.linalg.lstsq（SVD 最小范数解，E036/E041 同款；
  线性基场景 ΦᵀΦ 奇异——列由 x 与 1 张成秩≤2——解不唯一、能量唯一，锚 §5e 申报）。
主锚：§5.3.1/§5.3.2（e042_book_ch5_anchors.txt §1–§3）；承接 p.237 导言（§4.1）；
移交 5.3.4 p.280（§4.3）。
输出：code/outputs/e042_neuron_basics_lab.txt（stdout 重定向）+e042_neuron_basics_readings.json+figures 三图。
"""
import json
import math
import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]  # 仓库根（code→E042→episodes→根）
EP = ROOT / "episodes/E042_neuron_basics"
OUT = EP / "code/outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 存证自证（E042 门A r3 P0-1/P1-6 整改）：起止时间+三 PNG sha256 由本输出直接打印——
# lab 存证 e042_lab.txt 由本输出重定向生成，结果行不再手填
import datetime
import hashlib
T0 = datetime.datetime.now()
print(f"[lab] 开始 {T0.isoformat(timespec='seconds')}")
print(f"[lab] 输入 锚证据 sha256={hashlib.sha256((EP / 'production/evidence/e042_book_ch5_anchors.txt').read_bytes()).hexdigest()[:16]}…")

plt.rcParams["font.family"] = ["Songti SC", "STSong", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18})

# ════════════════════════════════════════════════════════════
# 数据：seed 2042 sin 曲线 40 点（讲者白话：直线拟合学不会的形状——实验实证）
# ════════════════════════════════════════════════════════════
rng = np.random.default_rng(2042)
X = np.sort(rng.uniform(-3, 3, 40))
Y = np.sin(X) + rng.normal(0, 0.05, 40)

check_log = []


def check(name, cond, detail):
    check_log.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}　| {detail}")


def ols_fit(Phi, y):
    """最小二乘闭式（np.linalg.lstsq——E036/E041 同款数值稳定实现）。返回 (coef, E)。"""
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    r = Phi @ coef - y
    return coef, float(r @ r)


# ════════════════════════════════════════════════════════════
# E1：线性塌缩复合实证（讲者实验——最优两层线性网络=单层 OLS）
# ════════════════════════════════════════════════════════════
K = 12
W1 = rng.normal(0, 1.0, K)          # 第一层权重（固定随机，不训练——训练循环是 E043）
b1 = rng.uniform(-1.5, 1.5, K)      # 第一层偏置（随机铺开拐点位置）
Phi_lin = W1[None, :] * X[:, None] + b1[None, :]      # 12 个线性基 φ_k(x)=w_k x+b_k
c_lin, E_lin = ols_fit(Phi_lin, Y)
# 单层 OLS 基线：Φ=[x,1]
Phi_ols = np.column_stack([X, np.ones_like(X)])
c_ols, E_ols = ols_fit(Phi_ols, Y)
rel_collapse = abs(E_lin - E_ols) / E_ols
check("E1 线性塌缩复合实证（讲者实验·锚 §4.2 申报）：最优两层线性网络的能量=单层 OLS 能量——没有弯，两层白搭（矩阵复合 W₂(W₁x+b₁)+b₂ 化简为线性映射，原书无此句、双标申报）",
      rel_collapse < 1e-9,
      f"两层线性（12 单元最优第二层）E={E_lin:.6f} vs 单层 OLS E={E_ols:.6f} rel={rel_collapse:.2e}")

# ════════════════════════════════════════════════════════════
# E2：ReLU 基拟合曲线（讲者实验——E041 线性最小二乘（lstsq/SVD 最小范数解）=本集输出层求解器）
# ════════════════════════════════════════════════════════════
Phi_relu = np.maximum(0.0, W1[None, :] * X[:, None] + b1[None, :])   # 12 个 ReLU 基
c_relu, E_relu = ols_fit(Phi_relu, Y)
# 拟合优度：E_relu 应远小于 E_ols（sin 是弯的）
ratio = E_ols / E_relu
check("E2 ReLU 基拟合曲线（讲者实验）：固定随机 ReLU 基 12 个+最小二乘输出层（E041 工具闭环）——非线性拟合成功",
      E_relu < E_ols * 0.5,
      f"E_relu={E_relu:.6f} vs E_ols={E_ols:.6f}（线性能量/非线性能量={ratio:.1f} 倍）")

# ════════════════════════════════════════════════════════════
# E3：线性基对照（同场塌缩实锤——把弯拿掉两层退化回一层）
# ════════════════════════════════════════════════════════════
# （E3=E1 同一场：Phi_lin 即线性基拟合——E_lin 与 E_ols 的 rel 已在 E1 断言；
#   此处补 E_relu vs E_lin 的显著性对照申报，同场三能量并列入 readings）
check("E3 线性基对照（同场）：同样 12 个基、激活=恒等——能量回到单层 OLS（弯没了两层=一层）",
      E_lin > E_relu * 5.0 and rel_collapse < 1e-9,
      f"E_lin={E_lin:.6f}（≈E_ols）vs E_relu={E_relu:.6f}——ReLU 基把能量压到线性基的 {E_relu/E_lin:.1%}")

# ════════════════════════════════════════════════════════════
# E4：ReLU die 数值演示（原书 p.273 Karpathy 警告照录——锚 §3）
# ════════════════════════════════════════════════════════════
w_die, b_die = -1.0, -3.5           # 手工配置死亡单元：全数据 pre=wx+b<0 → 死区（演示后果，非复现致死过程）
pre = w_die * X + b_die
dead = (pre <= 0).all()
h_die = np.maximum(0.0, pre)
dEdw = float(np.sum((h_die - Y) * (pre > 0) * X))    # dE/dw = Σ 1_{pre>0}·(h−y)·x
check("E4 ReLU die 死亡状态演示（原书 p.273 Karpathy 警告照录；手工配置演示后果——未复现致死过程）：单元在全部数据上 pre=wx+b<0 → 输出恒 0 且 ∂E/∂w=0——永远不激活、永远没梯度",
      dead and abs(dEdw) < 1e-12,
      f"wx+b 最大值={pre.max():.3f}（<0 全数据）输出恒 0；∂E/∂w={dEdw:.2e}")

# ════════════════════════════════════════════════════════════
# E5：矩阵形式一致性（式 5.47 层加权和+单元级偏置（式 5.45）——逐单元循环 vs 层矩阵）
# ════════════════════════════════════════════════════════════
rng2 = np.random.default_rng(20425)  # E5 独立种子（不扰动主实验场随机序列）
d_in = 5                              # d=5 维输入（门A r1 P1-4 整改：真矩阵乘法，非 5 个一维样本广播）
x_vec = rng2.normal(0, 1.0, d_in)
W_mat = rng2.normal(0, 1.0, (K, d_in))
b_vec = rng2.normal(0, 1.0, K)
# 逐单元（式 5.45/5.46 循环）：si=wiᵀx+bi
y_loop = np.zeros(K)
for j in range(K):
    s_j = W_mat[j] @ x_vec + b_vec[j]
    y_loop[j] = np.maximum(0.0, s_j)
# 层矩阵（式 5.47 层加权和 s=Wx，不含偏置——偏置为单元级、式 5.45）：s=Wx+b→h(s)
S_mat = W_mat @ x_vec + b_vec
Y_mat = np.maximum(0.0, S_mat)
rel_mat = float(np.abs(y_loop - Y_mat).max() / max(np.abs(Y_mat).max(), 1e-300))
check("E5 矩阵形式一致性（式 5.47 层加权和 s=Wx+单元级偏置（式 5.45））：逐单元循环 si=wiᵀx+bi→h(si) vs 层矩阵 s=Wx+b→h(s) 逐元一致（d=5 维输入×12 单元，真矩阵乘法）",
      rel_mat < 1e-12, f"max_rel={rel_mat:.2e}（d=5 输入×12 单元，W∈R^{{12×5}}；"
                        f"本机平台实测恰好为零——浮点累加顺序依赖平台，断言阈值 1e-12 覆盖非零情形）")

# ════════════════════════════════════════════════════════════
# 图 1：神经元结构图（程序生成——原书 Figure 5.23 不用，红线 2）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
# 输入节点（三个——门A r1 P0-3 整改：三路输入各乘权重，汇入求和）
for i, (yy, lab_, sub_) in enumerate([(4.6, "x\u2081", "\u2081"), (3.0, "x\u2082", "\u2082"),
                                      (1.4, "x\u2083", "\u2083")]):
    ax.annotate(f"${lab_}$", xy=(1.0, yy), fontsize=20, ha="center", va="center")
    ax.annotate(f"× w{sub_}", xy=(2.9, yy), fontsize=17, ha="center", va="center")
    ax.annotate("", xy=(4.6, 3.0 if i == 1 else 3.0 + (0.35 if i == 0 else -0.35)),
                xytext=(3.7, yy),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="k"))
ax.annotate("输入", xy=(1.0, 0.55), fontsize=17, ha="center", va="center")
ax.annotate("权重", xy=(2.9, 0.55), fontsize=17, ha="center", va="center")
# 偏置箭头接入求和
ax.annotate("$+\\,b$", xy=(4.35, 4.9), fontsize=19, ha="center", va="center")
ax.annotate("偏置", xy=(5.6, 4.9), fontsize=17, ha="center", va="center")
ax.annotate("", xy=(5.4, 3.5), xytext=(5.4, 4.75),
            arrowprops=dict(arrowstyle="->", lw=2.0, color="k"))
# 求和
ax.annotate("$\\Sigma$", xy=(5.4, 3.0), fontsize=26, ha="center", va="center")
ax.annotate("加权和", xy=(5.4, 2.1), fontsize=17, ha="center", va="center")
# 激活
ax.annotate("$h$", xy=(7.0, 3.0), fontsize=22, ha="center", va="center", color="tab:red")
ax.annotate("激活函数", xy=(7.0, 2.1), fontsize=17, ha="center", va="center", color="tab:red")
# 输出
ax.annotate("$y$", xy=(9.1, 3.0), fontsize=22, ha="center", va="center")
ax.annotate("输出", xy=(9.1, 2.1), fontsize=17, ha="center", va="center")
for x0_, x1_ in [(6.1, 6.5), (7.5, 8.6)]:
    ax.annotate("", xy=(x1_, 3.0), xytext=(x0_, 3.0),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="k"))
ax.set_title("一个神经元：加权和 + 激活")
fig.tight_layout()
fig.savefig(FIG / "e042_neuron_anatomy.png")

# ════════════════════════════════════════════════════════════
# 图 2：线性塌缩 vs ReLU 弯（拟合对比——f009/f010 用）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
ax.scatter(X, Y, s=26, color="0.35", zorder=4)   # 灰散点=数据
xs = np.linspace(-3, 3, 300)
ax.plot(xs, c_ols[0] * xs + c_ols[1], "r-", lw=2.2)          # 红实线=单层线性拟合
Phi_relu_xs = np.maximum(0.0, W1[None, :] * xs[:, None] + b1[None, :])
ax.plot(xs, Phi_relu_xs @ c_relu, color="tab:purple", lw=2.2)  # 紫实线=ReLU 基网络
ax.set_title("直线拟合学不会的弯")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout()
fig.savefig(FIG / "e042_relu_vs_linear.png")

# ════════════════════════════════════════════════════════════
# 图 3：激活函数族（程序生成——原书 Figure 5.26 不用，红线 2）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
t = np.linspace(-3, 3, 300)
ax.plot(t, np.maximum(0, t), "tab:red", lw=2.4)      # 红实线=ReLU
ax.plot(t, np.where(t >= 0, t, 0.1 * t), "tab:orange", lw=2.0, ls="--")  # 橙虚线=leaky ReLU（负段斜率 0.1，讲者示例参数）
ax.plot(t, 1 / (1 + np.exp(-t)), "tab:blue", lw=2.0, ls="-.")  # 蓝点划线=sigmoid
ax.plot(t, np.tanh(t), "tab:green", lw=2.0, ls=":")  # 绿点线=tanh
ax.axhline(0, color="0.5", lw=0.7, ls=":")
ax.set_ylim(-1.3, 3.2)
ax.set_title("四条代表曲线：折角弯与柔弯")
ax.set_xlabel("y（加权和）"); ax.set_ylabel("h(y)")
fig.tight_layout()
fig.savefig(FIG / "e042_activation_family.png")

# ════════════════════════════════════════════════════════════
# E6：图内文字运行时扫描（E041 E8 同款机制——内部记号零泄漏）
# ════════════════════════════════════════════════════════════
# 收割=matplotlib 对象运行时遍历（门A r2 P2-6 整改：annotate(f"...") 等 f-string/动态文本
# 在源码正则下漏收——改为从 figure/axes 对象直接取渲染文本，机制上不漏；图 2/图 3 图例已取消
# （E041 判例），legend 分支保留作回归护栏；_nolegend_（占位，不出图）不计入）
harvest = []
for _num in plt.get_fignums():
    _fig = plt.figure(_num)
    for _ax in _fig.axes:
        for _t in list(_ax.texts):
            harvest.append(_t.get_text())
        harvest.append(_ax.get_title())
        harvest.append(_ax.get_xlabel())
        harvest.append(_ax.get_ylabel())
        _lg = _ax.get_legend()
        if _lg is not None:
            for _t in _lg.get_texts():
                harvest.append(_t.get_text())
        for _t in list(_ax.get_xticklabels()) + list(_ax.get_yticklabels()):
            harvest.append(_t.get_text())
harvest = [s for s in harvest if s and s != "_nolegend_"]
seen, uniq = set(), []
for s_ in harvest:
    if s_ not in seen:
        seen.add(s_)
        uniq.append(s_)
harvest = uniq
bad = [s for s in harvest if re.search(r"E0\d\d|f0\d\d|门[ABC]|run\d|§\d", s)]
n_hit = len(bad)
check(f"E6 图内文字运行时收割（matplotlib 对象遍历 {len(harvest)} 条，含标题/轴标/注记/刻度；图例已取消——E041 判例，线型说明进 figcaption）：内部记号零命中",
      n_hit == 0, f"命中 {n_hit} 条" + (f"：{bad[:3]}" if bad else ""))

# ════════════════════════════════════════════════════════════
# readings.json + LEDGER
# ════════════════════════════════════════════════════════════
readings = {
    "data": {"seed": 2042, "n": 40, "x_range": [-3, 3], "noise": 0.05,
             "note": "一维 sin 曲线回归——「直线拟合学不会的形状」实验场（讲者白话）"},
    "e1_linear_collapse": {
        "K": K, "E_two_layer_linear": E_lin, "E_ols": E_ols,
        "rel": rel_collapse,
        "note": "讲者实验（锚 §4.2 申报：原书无 stacked-linear-collapse 显式句）——固定随机第一层、"
                "第二层最小二乘最优的两层线性网络，能量=单层 OLS（rel<1e-9）。线性层复合仍为线性=矩阵结合律事实"},
    "e2_relu_fit": {
        "E_relu": E_relu, "E_ols": E_ols, "ratio_ols_over_relu": ratio,
        "note": "固定随机 ReLU 基 12 个（w_k,b_k 不训练——训练循环 E043），输出系数由线性最小二乘解"
                "（E041 闭环：上集工具=本集神经网络最后一层求解器；np.linalg.lstsq SVD 最小范数解——"
                "ΦᵀΦ 奇异解不唯一、能量唯一——门A r1 P1-3）"},
    "e3_linear_basis_same_field": {
        "E_lin": E_lin, "E_relu": E_relu,
        "note": "同场三能量并列：线性基 12 个（激活=恒等）能量回到 E_ols（E1 同一场实算）——"
                "弯没了两层=一层；ReLU 基把能量压到线性基的 %.1f%%" % (E_relu / E_lin * 100)},
    "e4_relu_die": {
        "w": w_die, "b": b_die, "pre_max": float(pre.max()),
        "dEdw": dEdw,
        "note": "原书 p.273 Karpathy 警告（锚 §3 照录）死亡状态演示——手工配置（w=-1.0,b=-3.5）演示后果：输出恒 0+梯度恒 0（未复现大梯度致死过程——门A r1 P2-8 措辞降档）"},
    "e5_matrix_form": {
        "max_rel": rel_mat,
        "note": "式 5.47 层加权和 s=Wx（不含偏置——偏置为单元级、式 5.45）+逐单元偏置 b→h(s)，与逐单元循环逐元一致（d=5 维输入×12 单元，W∈R^{12×5} 真矩阵乘法——门A r1 P1-4 整改）；"
            "max_rel=0.0 为本机 numpy 平台实测，浮点累加顺序（gemv vs dot）依赖平台，非零情形由断言阈值 rel<1e-12 覆盖（r2 P2-9 申报）"},
    "e6_figtext": {"harvest": len(harvest), "hit": n_hit},
}
(OUT / "e042_neuron_basics_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))

n_pass = sum(1 for _, ok, _ in check_log if ok)
print(f"\n断言 {len(check_log)} 条，PASS {n_pass}")
print("LEDGER")
for name, ok, _ in check_log:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] {png.name} sha256={hashlib.sha256(png.read_bytes()).hexdigest()}")
print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
# r5 P2-6：任一断言 FAIL 即非零退出——存证链「# exit=」不再依赖打印约定（gen_lab_evidence
# 从 LEDGER 判定写 exit，双层一致：退出码与 LEDGER 行均由同一 check_log 派生）
if n_pass < len(check_log):
    sys.exit(1)
