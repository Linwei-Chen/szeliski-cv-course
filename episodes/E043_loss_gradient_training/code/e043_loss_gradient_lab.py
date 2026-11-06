#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E043 lab —— 网络怎么学实验：损失梯度+梯度下降迭代（§5.3.4/§5.3.5 前半 读数级实证）

数据与集谱（承接 E042 同一实验场）：
- 主实验场：seed 2042，一维曲线回归——40 点 x∼U[−3,3]，y=sin(x)+N(0,0.05)；固定随机
  ReLU 基 φ_k(x)=max(0, w_k x+b_k)（k=1..12，w_k,b_k 与 E042 同种子同分布生成——
  「上一集固定随机的基，本集接上训练循环」）。
- E1 数值梯度 vs 解析梯度（讲者实验，支撑 5.63「预测减目标」直觉；系数口径申报：本 lab
  能量 E(c)=mean‖Φc−y‖²（均值口径），解析梯度 ∇E=2/N·Φᵀ(Φc−y)；原书式 5.63 的单样本
  形式 ∂E_n/∂y=y−t 为原书系数约定（锚 §3.3 照录），本 lab 不复刻原书系数、只验证
  「梯度里出现预测减目标因子 (y−t)」与数值差分一致性）：
  中心差分 vs 解析式逐分量 rel<1e-5——「梯度=每动一点点能量变多少」的实算。
- E2 softmax+交叉熵梯度=预测减目标（式 5.62 实证，讲者实验）：3 类小例（s 随机、t=
  one-hot），E=−log p_t，数值差分 ∂E/∂s_k vs (p−t)_k rel<1e-5——「softmax 的梯度
  就是预测概率减目标」的实算（式 5.62 原书直觉句照录锚 §3.3）。
- E3 梯度下降迭代逼近闭式解（讲者实验，最小直线拟合场）：**实验场划界申报——GD 演示用
  二参数直线场 E(w1,w2)=mean(w1·x̃+w2−y)²（x 标准化 x̃=(x−μ)/σ，讲者实验配置：良态条件数
  使迭代收敛可演示；ReLU 基 13 维场的 Hessian 病态 λ_min≈0（含死单元列+近共线基），纯 GD
  数千步内到不了闭式解——病态问题原书 5.3.6 预处理/优化变体区段处理，E045 移交）**。
  η=稳定界 2/λ_max 的四分之一（收缩因子 1−ηλ_max=0.5，逐步下降可见；r1 P1-2 整改申报：
  一半 0.5 时因子恰为 0、第一步即落极小点，迭代演示退化，故取四分之一）；报告 rel<1e-6
  首达步位（steps_to_conv）。2000 步 E_gd vs 闭式 lstsq E_ols rel<1e-6：「一步到位的解
  （E036/E041 工具），一步步走也能到（本集训练循环）」。全网络反向传播（层间误差回传）
  =E044（5.3.5 后半）。
- E4 收敛前能量严格下降（r2 P1-2 整改：旧「全部 2000 步不增」断言在浮点底噪上空真——
  max_increase≈2.78e-17=E≈0.139 处一个 ulp；断言域改为前 steps_to_conv 步每步真实变好
  diffs<0 全部，其后能量停在浮点底噪上——口播同步改为「前十来步每一步都在变好」）。
- E5 大步长发散实证：η=稳定界 2/λ_max 的 1.5 倍时能量 50 步内爆炸（>1e3×E_0 或非有限）——
  「步长怎么选」原书在 5.3.6 Training and optimization（E045 区段，锚 §4 移交申报），
  本集只演示后果；η 取谱半径实算（稳定界 2/λ_max 的倍数，讲者实验配置）；readings 拆
  cross_1000x_step/nonfinite_step 双步位字段（r1 P1-3 整改——旧 diverged_at_step 曾被
  E6 源码扫描循环的 i 变量泄漏污染，报 340 实为 lab 源码行数+1，与本实验无关）。
- E6 图内文字运行时收割（r3 P1-2 整改：fig.findobj(Text) 对象遍历——标题/批注/图例/刻度全部
  Text 对象，三图 savefig 前收割；旧实现为源码正则静态扫描，声明「对象遍历」与实现不符+计数
  17 含 4 条非绘图行+漏图3 x 刻度文字 cats——内部记号零泄漏红线 5；r4 P2-3：计数口径分
  全部/visible 标志两档入 readings（per_fig_all/per_fig_visible 逐图两档），断言扫全部=更严不漏检；
  r7 P2-4 改口：visible=get_visible() 标志为真（含视图外未绘制刻度），不申报为「图上可见」；
  r5 P2-2 立项不可见构成实测、r6 P1-1 更正归因：不可见对象以「对象 is Tick.label2」身份
  实测计数（matplotlib 每个 Tick 有 label1 显示侧/label2 对侧默认隐藏同文本——v6 的
  「set_xlim/semilogy/分类轴替换遗留孤儿副本」归因系未实测推断且与计数结构矛盾，删除；
  「身份比对（文本集比对）」「逐图三列」等未落地表述同步清除，以实现为准）。
- F1a 等高线几何核验（r3 P1-1 整改：旧图1 ZZ[i,j]=E(gu[i],gv[j]) 行=x 配 contourf(gu,gv,ZZ)
  使等高线转置——最优点偏离白星、批注「垂直等高线」不成立）：①ZZ 按行=y/列=x 约定
  （meshgrid）重造后网格 argmin 与闭式解距离 ≤ 一个格距；②起点解析梯度与等高线数值法向
  （np.gradient）夹角 <2°；set_aspect("equal") 等数据比例；图1 轨迹自身首达步数入 readings
  fig1_traj（r3 P2-1——deck caption 引本值，不再错挂 E3 零点起跑读数）。
- F1b 图1 版面核验（r4 P2-2 立项，r5 P1-2/P2-1 强化，r6 P2-1/P2-2 收口：equal 比例下轴区压成
  方形，旧批注 xytext 越出 xlim/压起点标记——两处带底批注均安放轴区右侧图幅留白区；断言
  get_aspect()==1.0+恰两条带底批注（len==2——r5 P2-1：旧断言无条数要求，框全删仍空真）+背景框
  get_bbox_patch() 外扩含 pad 的显示 bbox 全在图幅内+与轴区窗口不相交（框不遮图内任何元素）
  +不压起点方块/轨迹圆/白星标记（中心±半径显示矩形相交判定——r5 P2-1：旧只测标记中心点）
  +callout 引线真实存在（annotate arrowprops——r5 P1-2：v5 申报 callout 版式但未给 arrowprops，
  箭头不存在，申报失实）且除各自指向目标外不触任何标记；r6 P2-1：收敛簇豁免集几何判定实时
  计算入 readings converged_idx/converged_dist_px——申报只引字段不手写数字（手抄数值三犯教训）；
  r6 P2-2：检查扩展到无框轴内文字（「起点」）——越图幅/越轴区（左右/底）/压标题/压带底框/
  压标记全零断言；顶边外溢允许（角点标记距轴区顶仅 0.2 数据单位<17pt 字高≈0.7 单位，全含轴区
  与避开轨迹折线几何不相容——外溢入轴区-标题间隙且不压标题，量测申报见断言 detail）。
- E7/E8/E8b 作业档位实证（r2 P1-3 整改：讲者布置的 0.5/1.0 复算档位给 lab 断言存证，替代
  numeric 白名单申报）：E7 η=0.5=2/λ_max 的一半，收缩因子 1−ηλ_max=0，一步到谷底
  rel<1e-12；E8 η=2/λ_max 实算（r3 P2-6：旧硬编码 1.0 与 2/λ_max 有 2e-16 级浮点差异，
  改实算消歧）1−ηλ_max=−1（|·|=1）临界——参数每步翻转到谷底另一侧（相邻步 (w−w*) 点积
  全部 <0，翻转次数程序计数入存证——r4 P1-1 整改：旧输出串硬编码「19 次翻转」非计算值且
  与实际 20 对不符）且能量恒等于出发值（20 步最大漂移<1e-12）——「不收敛也不爆炸」；
  E8b 字面档位（r4 P2-4）：η=1.0（口播「一点零」的字面输入）与 E8 同为临界——存证单独
  覆盖口播值（readings e8_literal）——存证层同值记录；numeric 机检为存档同值命中、未做字段
  绑定（r5 P2-3 申报更正：旧「语义绑定本字段、不再靠 e2 巧合命中」超出机检实际能力）。
- fig2_eta_tiers（r2 P2-4 整改：deck 图2 caption 三档 η=0.0002/0.25/1.5 的读数出处
  申报进 readings；0.0002 为讲者实验配置「慢但收敛」档，无独立断言）。
- 符号口径：c∈R^12 输出层权重（E042 记号 c_relu 延续）；b 输出层偏置并入 c（Φ 增列 1）
  ——与 E042 ols_fit 同构；E2 独立小例与主实验场无关（独立种子）。
- 存证运行法（r5 P2-4）：python3 code/e043_loss_gradient_lab.py > code/outputs/e043_loss_gradient_lab.txt
  2> code/outputs/e043_loss_gradient_lab_stderr.txt——stdout/stderr 分流（旧申报只写 stdout
  重定向而存证混入 stderr 警告行，命令与存证不符）；输出内记源码/readings/PNG sha 供
  make_script lab-stdout 哨兵③版本绑定核验（r5 P2-5）。
主锚：§5.3.4/§5.3.5 前半（e043_book_ch5_anchors.txt §1/§3）；承接 E042 f013 移交；
移交 5.3.6 p.287（E045）。
输出：code/outputs/e043_loss_gradient_lab.txt（stdout 重定向）+e043_loss_gradient_readings.json+figures 三图。
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]  # 仓库根（code→E043→episodes→根）
EP = ROOT / "episodes/E043_loss_gradient_training"
OUT = EP / "code/outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as _mtext
import matplotlib.ticker as _mticker
import re

# E6 运行时收割器（r3 P1-2 整改：fig.findobj(Text) 对象遍历——含标题/批注/刻度全部
# Text 对象；旧实现为源码正则静态扫描，声明「对象遍历」与实现不符+计数含非绘图行+漏刻度文字）
ALL_TEXTS = []
E6_PER_FIG = []      # 每图 (全部条数, visible 标志为真条数——含视图外未绘制刻度，r7 P2-4 改口)
E6_INV_LABEL2 = 0    # 不可见·Tick.label2 对侧隐藏标签（r6 P1-1：以「对象 is tick.label2」身份实测）
E6_INV_OTHER = 0     # 不可见·其余（只计数不归因——r6 P1-1 教训：未经实测的归因即失实申报）


def harvest_texts(fig):
    # 不可见对象构成实测（r5 P2-2 立项，r6 P1-1 更正：v6 的文本集比对+「孤儿副本」归因
    # 与结构不符——get_xticklabels() 只返回 label1，比对集合里本就没有 label2，
    # 「0 命中→孤儿」推断不成立。本版按 matplotlib Tick 双标签结构做对象身份实测：
    # 每个 Tick 有 label1（显示侧）与 label2（对侧默认隐藏、同文本），不可见对象
    # is tick.label2 计数入 E6_INV_LABEL2，其余只计数不归因（E6_INV_OTHER）。
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

# 存证自证（E042 同款）：起止时间+三 PNG sha256 由本输出直接打印
import datetime
import hashlib
T0 = datetime.datetime.now()
print(f"[lab] 开始 {T0.isoformat(timespec='seconds')}")
print(f"[lab] 输入 锚证据 sha256={hashlib.sha256((EP / 'production/evidence/e043_book_ch5_anchors.txt').read_bytes()).hexdigest()[:16]}…")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}"
      f"（哨兵③版本绑定数据源——r5 P2-5）")

plt.rcParams["font.family"] = ["Songti SC", "STSong", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 17, "axes.titlesize": 18})

# ════════════════════════════════════════════════════════════
# 数据：seed 2042 sin 曲线 40 点（与 E042 同场——上一集的实验场接上训练）
# ════════════════════════════════════════════════════════════
rng = np.random.default_rng(2042)
X = np.sort(rng.uniform(-3, 3, 40))
Y = np.sin(X) + rng.normal(0, 0.05, 40)

K = 12
W1 = rng.normal(0, 1.0, K)
b1 = rng.uniform(-1.5, 1.5, K)
Phi_relu = np.maximum(0.0, W1[None, :] * X[:, None] + b1[None, :])   # 12 个 ReLU 基（固定不训练）
Phi = np.column_stack([Phi_relu, np.ones_like(X)])                    # 并入偏置列 → 13 维设计矩阵
N, D = Phi.shape

check_log = []


def check(name, cond, detail):
    check_log.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}　| {detail}")


def energy(c):
    r = Phi @ c - Y
    return float(r @ r / N)          # E(c)=mean‖Φc−y‖²（均值口径——E1 申报）


def grad(c):
    return 2.0 / N * (Phi.T @ (Phi @ c - Y))   # 解析梯度（E=mean 口径）


c_ols, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
E_ols = energy(c_ols)

# ════════════════════════════════════════════════════════════
# E1：数值梯度 vs 解析梯度（中心差分）
# ════════════════════════════════════════════════════════════
rng_num = np.random.default_rng(4301)
c0 = rng_num.normal(0, 0.3, D)
eps = 1e-6
g_num = np.zeros(D)
for j in range(D):
    cp, cm = c0.copy(), c0.copy()
    cp[j] += eps
    cm[j] -= eps
    g_num[j] = (energy(cp) - energy(cm)) / (2 * eps)
g_ana = grad(c0)
rel_g = float(np.abs(g_num - g_ana).max() / np.abs(g_ana).max())
check("E1 数值梯度 vs 解析梯度（讲者实验，支撑 5.63 直觉；系数口径申报见 docstring）：中心差分与解析式 ∇E=2/N·Φᵀ(Φc−y) 逐分量一致——梯度里出现预测减目标因子 (Φc−y)",
      rel_g < 1e-5,
      f"max_rel={rel_g:.2e}（D={D} 维中心差分 eps=1e-6）")

# ════════════════════════════════════════════════════════════
# E2：softmax+交叉熵梯度=预测减目标（式 5.62 实证，3 类小例独立种子）
# ════════════════════════════════════════════════════════════
rng3 = np.random.default_rng(4302)
s = rng3.normal(0, 1.0, 3)
t_onehot = np.array([0.0, 0.0, 1.0])       # 目标=第 3 类（one-hot）
e_s = np.exp(s - s.max())
p = e_s / e_s.sum()


def ce_loss(sv):
    es = np.exp(sv - sv.max())
    pv = es / es.sum()
    return float(-np.log(pv[2]))


g_ce_num = np.zeros(3)
eps2 = 1e-6
for j in range(3):
    sp, sm = s.copy(), s.copy()
    sp[j] += eps2
    sm[j] -= eps2
    g_ce_num[j] = (ce_loss(sp) - ce_loss(sm)) / (2 * eps2)
g_ce_ana = p - t_onehot                    # 式 5.62：∂E_n/∂s_nk = p_nk − t̃_nk
rel_ce = float(np.abs(g_ce_num - g_ce_ana).max() / np.abs(g_ce_num).max())
check("E2 softmax+交叉熵梯度=预测减目标（式 5.62 实证，讲者实验）：数值差分 vs (p−t) 逐分量一致——原书直觉句「预测类概率与真实类别之差」的实算",
      rel_ce < 1e-5,
      f"p={np.round(p, 4).tolist()} t=onehot(第3类) max_rel={rel_ce:.2e}（3 类小例 eps=1e-6）")

# ════════════════════════════════════════════════════════════
# E3：梯度下降迭代逼近闭式解（最小直线拟合场——划界申报见 docstring）
# ════════════════════════════════════════════════════════════
x_t = (X - X.mean()) / X.std()               # 标准化输入（讲者实验配置，良态条件数）
A = np.column_stack([x_t, np.ones_like(x_t)])  # 设计矩阵 [x̃, 1]——两参数 w1,w2
E_line = lambda w: float(((A @ w - Y) ** 2).mean())
grad_line = lambda w: 2.0 / len(X) * (A.T @ (A @ w - Y))
H_line = 2.0 / len(X) * (A.T @ A)
LAM_MAX = float(np.linalg.eigvalsh(H_line).max())
# η=稳定界 2/λ_max 的四分之一（收缩因子 1−ηλ_max=0.5，逐步下降肉眼可见；
# 一半 0.5 时因子恰为 0、第一步即落极小点——演示退化，见 docstring 申报。门A r1 P1-2 整改）
ETA = round(0.5 / LAM_MAX, 4)
STEPS = 2000
w_ols, *_ = np.linalg.lstsq(A, Y, rcond=None)
E_ols_line = E_line(w_ols)
w_gd = np.zeros(2)
E_curve = [E_line(w_gd)]
steps_to_conv = None                          # rel<1e-6 首达步位（门A r1 P1-2 整改）
for step_i in range(1, STEPS + 1):
    w_gd = w_gd - ETA * grad_line(w_gd)
    E_now = E_line(w_gd)
    E_curve.append(E_now)
    if steps_to_conv is None and abs(E_now - E_ols_line) / E_ols_line < 1e-6:
        steps_to_conv = step_i
E_gd = E_curve[-1]
rel_gd = abs(E_gd - E_ols_line) / E_ols_line
check("E3 梯度下降迭代逼近闭式解（讲者实验，最小直线拟合场——实验场划界申报见 docstring）：w←w−η∇E（η=四分之一稳定界）——能量逐步下降到与闭式最小二乘一致（rel<1e-6）：一步到位的解，一步一步走也能到",
      rel_gd < 1e-6,
      f"E_gd={E_gd:.8f} vs E_ols={E_ols_line:.8f} rel={rel_gd:.2e}（η={ETA}，rel<1e-6 首达第 {steps_to_conv} 步/共 {STEPS} 步；λ_max={LAM_MAX:.3f}）")

# ════════════════════════════════════════════════════════════
# E4：收敛前能量严格下降（r2 P1-2 整改：旧「全部 2000 步不增」断言在浮点底噪上空真——
#     max_increase=2.78e-17 恰为 E≈0.139 处一个 ulp，收敛后能量停在谷底偶有一 ulp 抖动；
#     断言域改为前 steps_to_conv 步每步真实变好）
# ════════════════════════════════════════════════════════════
arr = np.array(E_curve)
diffs = arr[1:] - arr[:-1]
strict_descent = bool((diffs[:steps_to_conv] < 0).all())
check("E4 收敛前能量严格下降（讲者实验·观察申报，r2 P1-2 整改：断言域=前 steps_to_conv 步每步真实变好，避开浮点底噪空真）：合适步长 η（四分之一稳定界）下收敛前每步能量严格下降，其后停在谷底（浮点底噪）",
      strict_descent,
      f"前 {steps_to_conv} 步单步降幅全部 <0（最大步降幅 {diffs[:steps_to_conv].min():.2e}）；全程最大单步增幅={diffs.max():.2e}（≈E≈0.139 处一个 ulp 的底噪）；E 从 {arr[0]:.4f} 降到 {arr[-1]:.8f}")

# ════════════════════════════════════════════════════════════
# E5：大步长发散实证（步长怎么选=5.3.6/E045 主题，本集只演示后果）
# ════════════════════════════════════════════════════════════
ETA_BIG = round(1.5 * 2.0 / LAM_MAX, 4)   # 稳定界 2/λ_max 的 1.5 倍（讲者实验配置——r1 P2-6 文案更正）
w_big = np.zeros(2)
E_big0 = E_line(w_big)
cross_1000x_step = None                   # 首个 E>1e3×E_0 的步位（r1 P1-3 字段拆分）
nonfinite_step = None                     # 首个非有限步位（循环上限/判据公开申报）
E_big_last = E_big0
for step_big in range(1, 2001):
    w_big = w_big - ETA_BIG * grad_line(w_big)
    Eb_big = E_line(w_big)
    if cross_1000x_step is None and np.isfinite(Eb_big) and Eb_big > 1e3 * E_big0:
        cross_1000x_step = step_big
    if not np.isfinite(Eb_big):
        nonfinite_step = step_big
        break
    E_big_last = Eb_big
check("E5 大步长发散实证（讲者实验）：η=稳定界的 1.5 倍时能量 50 步内爆炸（>1e3×E_0 或非有限）——步长过敏感发散；步长怎么选原书在 5.3.6（E045 移交，锚 §4 申报）",
      cross_1000x_step is not None and cross_1000x_step <= 50,
      f"E_0={E_big0:.4f} → 第 {cross_1000x_step} 步冲破 1000×E_0（爆炸前最后 E={E_big_last:.3e}；非有限步位={nonfinite_step}，循环上限 2000 步/判据=非有限即停）")

# ════════════════════════════════════════════════════════════
# E7/E8：作业二档位实证（r2 P1-3 整改：讲者布置的 0.5/1.0 复算档位给 lab 存证，
#         替代 numeric 白名单申报——0.5 一步到谷底/1.0 临界能量恒定）
# ════════════════════════════════════════════════════════════
ETA_HALF = round(1.0 / LAM_MAX, 4)        # η=0.5：收缩因子 1−ηλ_max=1−0.5×2=0——第一步即落极小点
w_half = -ETA_HALF * grad_line(np.zeros(2))
E_half_1 = E_line(w_half)
rel_half = abs(E_half_1 - E_ols_line) / E_ols_line
check("E7 作业档位实证（讲者实验，r2 P1-3 整改）：η=0.5（收缩因子 1−ηλ_max=0）恰一步到谷底——E 与闭式解一致",
      rel_half < 1e-12,
      f"η={ETA_HALF} 一步后 E={E_half_1:.10f} vs E_ols={E_ols_line:.10f} rel={rel_half:.2e}")

ETA_CRIT = 2.0 / LAM_MAX                  # η=稳定界本身 2/λ_max（浮点 ≈0.9999999999999998——r3 P2-6 整改：
                                          # 旧硬编码 1.0 与 2/λ_max 存在 2e-16 级差异，改实算消歧）
w_crit = np.zeros(2)
E_crit0 = E_line(w_crit)
crit_drift = 0.0
n_flip = 0                                # 相邻步 (w−w*) 点积 <0 计数——r4 P1-1 整改：旧输出串硬编码
pairs = 0                                 # 「19 次翻转」既非程序计算值、又与实际 20 对相邻步不符
w_prev = w_crit.copy()
for _ in range(20):
    w_crit = w_crit - ETA_CRIT * grad_line(w_crit)
    crit_drift = max(crit_drift, abs(E_line(w_crit) - E_crit0))
    pairs += 1
    if float((w_prev - w_ols) @ (w_crit - w_ols)) < 0:
        n_flip += 1
    w_prev = w_crit.copy()
check("E8 临界档位实证（讲者实验，r2 P1-3 整改；r3 P2-6 补参数振荡断言；r4 P1-1 翻转计数改程序计算入存证）：η=2/λ_max（恰为稳定界）下参数每步翻转到谷底另一侧（相邻步 (w−w*) 点积全部 <0），能量恒等于出发值——不收敛也不爆炸",
      crit_drift < 1e-12 and n_flip == pairs == 20,
      f"η={ETA_CRIT:.17g} 走 20 步能量最大漂移={crit_drift:.2e}（E 恒={E_crit0:.4f}）；相邻步 (w−w*) 点积全 <0：{n_flip}/{pairs} 次翻转")

# E8b 字面档位实证（r4 P2-4）：口播「一点零」的学员输入是字面 η=1.0，与 E8 的 2/λ_max 实算值
# 相差 2e-16——存证只跑实算值时，口播数值在 readings 无同值绑定（numeric 只能靠 e2 巧合命中）。
# 补一次字面 η=1.0 的 20 步运行，读数入 readings e8_literal。
ETA_LIT = 1.0
w_lit = np.zeros(2)
E_lit0 = E_line(w_lit)
lit_drift = 0.0
n_flip_lit = 0
pairs_lit = 0
w_prev = w_lit.copy()
for _ in range(20):
    w_lit = w_lit - ETA_LIT * grad_line(w_lit)
    lit_drift = max(lit_drift, abs(E_line(w_lit) - E_lit0))
    pairs_lit += 1
    if float((w_prev - w_ols) @ (w_lit - w_ols)) < 0:
        n_flip_lit += 1
    w_prev = w_lit.copy()
check("E8b 字面档位实证（讲者实验，r4 P2-4）：η=1.0（口播「一点零」的字面输入——学员复现口径）与 E8 同为临界：参数每步翻转到谷底另一侧、能量恒等于出发值——口播读数与存证同值记录（存证层绑定；numeric 机检为存档同值命中、无字段级绑定——r5 P2-3 申报更正）",
      lit_drift < 1e-12 and n_flip_lit == pairs_lit == 20,
      f"η={ETA_LIT} 走 20 步能量最大漂移={lit_drift:.2e}；相邻步 (w−w*) 点积全 <0：{n_flip_lit}/{pairs_lit} 次翻转")

# ════════════════════════════════════════════════════════════
# 图 1：梯度下降路径（2D 权重子空间等高线+下降轨迹——程序生成）
# ════════════════════════════════════════════════════════════
# 直线拟合场 (w1,w2)：能量等高线+梯度下降轨迹（二维全空间——无切片近似）
def e_plane(w1, w2):
    return E_line(np.array([w1, w2]))


# 等高线网格：matplotlib contourf(X, Y, Z) 约定 Z[行=y 索引, 列=x 索引]——
# 旧实现 ZZ[i,j]=E(gu[i],gv[j])（行=x）配 contourf(gu,gv,ZZ) 使等高线转置
# （最优点偏离白星、批注「垂直等高线」不成立——r3 P1-1 判例：meshgrid 按 xy 约定重造）
gu = np.linspace(-1.1, 3.3, 41)           # x 跨度 4.4=gv 跨度（set_aspect("equal") 无变形）
gv = np.linspace(-2.0, 2.4, 41)
UU, VV = np.meshgrid(gu, gv)              # UU[i,j]=gu[j]（列=x）、VV[i,j]=gv[i]（行=y）
ZZ = np.zeros_like(UU)
for i in range(ZZ.shape[0]):
    for j in range(ZZ.shape[1]):
        ZZ[i, j] = e_plane(UU[i, j], VV[i, j])

# r3 P1-1 断言①：等高线网格 argmin 与闭式解距离 ≤ 一个格距（转置判例回归锚）
ij_min = np.unravel_index(int(np.argmin(ZZ)), ZZ.shape)
argmin_pt = np.array([gu[ij_min[1]], gv[ij_min[0]]])   # 行=y/列=x 约定还原坐标
grid_pitch = max(gu[1] - gu[0], gv[1] - gv[0])
# r3 P1-1 断言②：起点处解析负梯度与等高线数值法向（ZZ 网格 np.gradient）夹角 <2°
gy, gx = np.gradient(ZZ, gv, gu)          # [行方向 d/dy, 列方向 d/dx]
uv0 = np.array([3.0, 2.2])
i0 = int(np.argmin(np.abs(gu - uv0[0])))
j0 = int(np.argmin(np.abs(gv - uv0[1])))
normal_fd = np.array([gx[j0, i0], gy[j0, i0]])
g_ana = grad_line(uv0)
cosang = float(normal_fd @ g_ana / (np.linalg.norm(normal_fd) * np.linalg.norm(g_ana)))
angle_deg = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
check("F1a 等高线几何核验（讲者实验，r3 P1-1 整改）：ZZ 按行=y/列=x 约定构造——网格 argmin 坐标与闭式解距离 ≤ 一个格距；起点处解析梯度与等高线数值法向（np.gradient）夹角 <2°——图内批注「垂直等高线」成立",
      np.linalg.norm(argmin_pt - w_ols) <= grid_pitch and angle_deg < 2.0,
      f"argmin=({argmin_pt[0]:.3f}, {argmin_pt[1]:.3f}) vs w_ols=({w_ols[0]:.3f}, {w_ols[1]:.3f}) 距离={np.linalg.norm(argmin_pt - w_ols):.4f}（格距 {grid_pitch:.3f}）；起点梯度-法向夹角={angle_deg:.3f}°")

# 图1 轨迹（与 E3 同 η，起点 (3.0,2.2)——轨迹自身首达步数写入 readings（r3 P2-1：
# 旧 caption「11 步」错挂 E3 零点起跑读数，与图内轨迹不同源）
eta_p = ETA
traj = [uv0.copy()]
fig1_steps = None
for step_i in range(1, 601):
    uv0 = uv0 - eta_p * grad_line(uv0)
    traj.append(uv0.copy())
    if fig1_steps is None and abs(E_line(uv0) - E_ols_line) / E_ols_line < 1e-6:
        fig1_steps = step_i
traj = np.array(traj)

fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
cs = ax.contourf(gu, gv, ZZ, levels=18, cmap="viridis", alpha=0.85)
ax.contour(gu, gv, ZZ, levels=10, colors="k", linewidths=0.4, alpha=0.35)
ax.plot(traj[:, 0], traj[:, 1], "-", color="crimson", lw=1.6)
ax.plot(traj[:12:2, 0], traj[:12:2, 1], "o", color="crimson", ms=5.5)
ax.plot(traj[0, 0], traj[0, 1], "s", color="crimson", ms=10)
# 负梯度方向箭头（起点处，垂直等高线——F1a 断言核验）
gd0 = grad_line(traj[0])
gd0 = gd0 / np.linalg.norm(gd0) * 0.9
ax.annotate("", xy=(traj[0, 0] - gd0[0], traj[0, 1] - gd0[1]),
            xytext=(traj[0, 0], traj[0, 1]),
            arrowprops=dict(arrowstyle="->", lw=2.4, color="gold"))
# 批注版面（r4 P2-2 立项，r5 P1-2 整改：v5 申报「callout 版式」但 annotate 未给 arrowprops
# ——箭头根本不存在，申报失实；v6 两处批注补 arrowprops 引线成真：框在轴区右侧图幅留白区，
# 引线分别指向起点处金色箭头尖/白星（各自指向目标，F1b 豁免判定））；衬底 alpha 0.55→0.8
# （框在白色图幅边缘上，0.55 衬底透白对比不足）；「最优」框同步移右缘下槽（r5 P2-1：17pt
# 7 字框宽约 248px 与轴区同量级，框内安放有压白星/越轴区风险——移出轴区后框不遮图内元素）
ax.annotate("每步沿负梯度\n（垂直等高线）", xy=(traj[0, 0] - gd0[0], traj[0, 1] - gd0[1]),
            xytext=(1.06, 0.70), textcoords="axes fraction",
            fontsize=16, color="gold", ha="left", va="center",
            bbox=dict(boxstyle="round", fc="black", alpha=0.8, ec="none"),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="gold", alpha=0.9))
ax.plot(w_ols[0], w_ols[1], "*", color="white", ms=17, mec="k")
ax.annotate("起点", xy=(traj[0, 0], traj[0, 1]), xytext=(traj[0, 0] - 0.55, traj[0, 1] - 0.15),
            fontsize=17, color="crimson", ha="right")
ax.annotate(f"最优（闭式解）\n({w_ols[0]:.2f}, {w_ols[1]:.2f})", xy=(w_ols[0], w_ols[1]),
            xytext=(1.06, 0.22), textcoords="axes fraction",
            fontsize=16, color="white", ha="left", va="center",
            bbox=dict(boxstyle="round", fc="black", alpha=0.8, ec="none"),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="white", alpha=0.9))
ax.set_xlabel("权重 w₁（斜率）", fontsize=17)
ax.set_ylabel("权重 w₂（截距）", fontsize=17)
ax.set_title("梯度下降：沿负梯度一步步走向能量最低点", fontsize=18, pad=12)
# pad=12（r6 P2-2：默认 pad 下「起点」17pt 文字 bbox 顶（CJK 上伸部≈0.88em）与标题 bbox 底
# 相交 1-2px——标题上移让出轴区-标题间隙，实测间隙入 readings unboxed_title_gap_px）
ax.set_aspect("equal")                    # r3 P1-1：等数据比例——屏幕上梯度方向垂直等高线
ax.set_xlim(gu[0], gu[-1]); ax.set_ylim(gv[0], gv[-1])
# 刻度统一（r5 目视整改）：小轴区大字号下 MaxNLocator 自动抽稀出「0.0/2.5」x 与「2/0/−2」y
# 混杂步长且小数/整数风格不一——统一整数步长 1（−1..3/−2..2 各 5 档，17pt×5≈175px<轴区 219px 不重叠）
ax.xaxis.set_major_locator(_mticker.MultipleLocator(1.0))
ax.yaxis.set_major_locator(_mticker.MultipleLocator(1.0))
plt.tight_layout()

# F1b 图1 版面核验（r4 P2-2 立项，r5 P1-2/P2-1 强化）：等比例生效+恰两条带底批注（len==2）+
# 背景框（get_bbox_patch 外扩含 pad）bbox 全在图幅内+与轴区窗口不相交（框安放轴区外，
# 不遮图内任何元素）+不压标记（中心±半径显示矩形相交判定，覆盖起点方块 ms=10/轨迹圆
# ms=5.5/白星 ms=17——r5 P2-1：旧只测标记中心点且无条数要求）+callout 引线真实存在
# （arrow_patch 非 None——r5 P1-2：v5 申报 callout 但无 arrowprops）且除各自指向目标
# （「每步沿负梯度」→金色箭头尖非标记；「最优」→白星本身）外不触任何标记
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_fig_w, _fig_h = fig.get_size_inches() * fig.dpi
_PX_PT = fig.dpi / 72.0
_anns = [t for t in fig.findobj(_mtext.Text) if t.get_text().strip() and t.get_bbox_patch() is not None]
_axes_bb = ax.get_window_extent(_renderer)


def _rect(bb):
    return (bb.x0, bb.y0, bb.x1, bb.y1)


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _marker_rect(c_disp, ms):
    r = ms * _PX_PT / 2.0
    return (c_disp[0] - r, c_disp[1] - r, c_disp[0] + r, c_disp[1] + r)


_start_rect = _marker_rect(ax.transData.transform(traj[0]), 10)
_circle_rects = [_marker_rect(ax.transData.transform(pt), 5.5) for pt in traj[2:12:2]]
_star_rect = _marker_rect(ax.transData.transform(w_ols), 17)
# 收敛簇豁免（r5 P2-1 引入，r6 P2-1 申报更正：豁免集由几何判定实时计算并写入 readings
# converged_idx/converged_dist_px——凡圆心矩形与星标矩形相交的轨迹圆，η=0.25 逐半收缩
# 使末段轨迹圆视觉上已并入白星，星目标引线的 bbox 必然与其重叠；豁免仅对「最优」引线
# 生效——「每步沿负梯度」引线不豁免。点位与距离只引 readings 字段，不手写数字）
_converged, _converged_idx, _converged_dist = [], [], []
for _ci, _mr in enumerate(_circle_rects):
    if _overlap(_mr, _star_rect):
        _converged.append(_mr)
        _converged_idx.append(2 + 2 * _ci)      # circles=traj[2:12:2] → 轨迹全局步位
        _c_disp = np.array(ax.transData.transform(traj[2 + 2 * _ci]))
        _s_disp = np.array(ax.transData.transform(w_ols))
        _converged_dist.append(float(np.hypot(*(_c_disp - _s_disp))))
_bbox_overflow = _axes_int = _marker_hit = _arrow_missing = _arrow_hit = 0
for _t in _anns:
    _ext = _rect(_t.get_bbox_patch().get_window_extent(_renderer))   # r5 P2-1：含 pad 外扩
    if not (0 <= _ext[0] and _ext[2] <= _fig_w and 0 <= _ext[1] and _ext[3] <= _fig_h):
        _bbox_overflow += 1
    if _overlap(_ext, _rect(_axes_bb)):                              # 框遮轴区=安放失败
        _axes_int += 1
    for _mr in [_start_rect, *_circle_rects, _star_rect]:
        if _overlap(_ext, _mr):
            _marker_hit += 1
    _ap = getattr(_t, "arrow_patch", None)                           # r5 P1-2：引线必须真实存在
    if _ap is None:
        _arrow_missing += 1
    else:
        _ap_ext = _rect(_ap.get_window_extent(_renderer))
        # 目标豁免：「最优」引线终点=白星本身（含视觉并入星标的收敛簇轨迹圆）；
        # 「每步沿负梯度」终点=金色箭头尖（非标记），不豁免
        _targets = [_star_rect, *_converged] if _t.get_text().startswith("最优") else []
        for _mr in [_start_rect, *_circle_rects, _star_rect]:
            if _mr not in _targets and _overlap(_ap_ext, _mr):
                _arrow_hit += 1
# 无框轴内文字检查（r6 P2-2）：「起点」等无框可见非刻度文字——不越图幅、不越轴区
# （左右/底）、不压标题/带底框/任何标记；顶边外溢允许（角点标记距轴区顶仅 0.2 数据单位
# <17pt 字高≈0.7 单位，全含轴区与避开轨迹折线几何不相容——外溢入轴区-标题间隙），
# 但外溢部分不得压标题（起点 bbox 与标题 bbox 不相交——下方断言核验）
_skip_ids = {id(ax.title), id(ax.xaxis.label), id(ax.yaxis.label)} | {id(_t) for _t in _anns}
for _ax2 in fig.axes:
    for _axis in (_ax2.xaxis, _ax2.yaxis):
        for _tk in list(_axis.get_major_ticks()) + list(_axis.get_minor_ticks()):
            _skip_ids.add(id(_tk.label1))
            _skip_ids.add(id(_tk.label2))
_boxed_bbs = [_rect(_t.get_bbox_patch().get_window_extent(_renderer)) for _t in _anns]
_title_bb = _rect(ax.title.get_window_extent(_renderer))
_axes_r = _rect(_axes_bb)
_unboxed = [t for t in fig.findobj(_mtext.Text)
            if t.get_text().strip() and t.get_visible()
            and t.get_bbox_patch() is None and id(t) not in _skip_ids]
_uf_out = _uf_axes = _uf_title = _uf_ann = _uf_marker = 0
_title_gap = None                          # 标题底−文字顶净空 px（x 相交者；正值=有净空，负=相交）
for _t2 in _unboxed:
    _e2 = _rect(_t2.get_window_extent(_renderer))
    if not (0 <= _e2[0] and _e2[2] <= _fig_w and 0 <= _e2[1] and _e2[3] <= _fig_h):
        _uf_out += 1
    if _e2[0] < _axes_r[0] or _e2[2] > _axes_r[2] or _e2[1] < _axes_r[1]:
        _uf_axes += 1                       # 左/右/底越轴区=FAIL；顶边外溢入标题间隙允许（申报）
    if _overlap(_e2, _title_bb):
        _uf_title += 1
    if _e2[2] > _title_bb[0] and _e2[0] < _title_bb[2]:
        _g = _title_bb[1] - _e2[3]          # display 坐标 y 向上：标题底更高，减文字顶=净空
        _title_gap = _g if _title_gap is None else min(_title_gap, _g)
    for _b in _boxed_bbs:
        if _overlap(_e2, _b):
            _uf_ann += 1
    for _mr in [_start_rect, *_circle_rects, _star_rect]:
        if _overlap(_e2, _mr):
            _uf_marker += 1
check("F1b 图1 版面核验（讲者实验，r4 P2-2 立项；r5 P1-2/P2-1 强化；r6 P2-1/P2-2 收口）：set_aspect(\"equal\") 生效；恰两条带底批注，背景框（含 pad 外扩）bbox 全在图幅内、与轴区窗口不相交（框安放轴区右侧留白区不遮图内元素）、不压起点方块/轨迹圆/白星标记（中心±半径矩形相交判定）；两处 callout 引线真实存在（arrowprops——v5 申报失实整改）且除指向目标（白星+收敛簇豁免集入 readings f1a_geometry.converged_idx/converged_dist_px——申报只引字段）外不触标记；无框轴内文字（起点）越图幅/越轴区（左右/底）/压标题/压框/压标记全零（顶边外溢入轴区-标题间隙允许——角点标记距轴顶 0.2 单位<17pt 字高≈0.7 单位，全含轴区与避轨迹几何不相容，外溢不压标题）",
      float(ax.get_aspect()) == 1.0 and len(_anns) == 2 and _bbox_overflow == 0
      and _axes_int == 0 and _marker_hit == 0 and _arrow_missing == 0 and _arrow_hit == 0
      and _uf_out == 0 and _uf_axes == 0 and _uf_title == 0 and _uf_ann == 0 and _uf_marker == 0,
      f"aspect={ax.get_aspect()}；批注框 越界 {_bbox_overflow}/2、压轴区 {_axes_int}、压标记 {_marker_hit}；引线 缺 {_arrow_missing}、误触标记 {_arrow_hit}；无框文字 {len(_unboxed)} 条 越图幅 {_uf_out}、越轴区(左右/底) {_uf_axes}、压标题 {_uf_title}、压框 {_uf_ann}、压标记 {_uf_marker}、文字顶-标题底净空 {(_title_gap is not None and f'{_title_gap:.1f}px') or '无 x 相交'}")
ax_aspect_val = float(ax.get_aspect())
bbox_overflow, anns_count, marker_hit = _bbox_overflow, len(_anns), _marker_hit
axes_int_val, arrow_missing_val, arrow_hit_val = _axes_int, _arrow_missing, _arrow_hit
unboxed_count, unboxed_out, unboxed_axes = len(_unboxed), _uf_out, _uf_axes
unboxed_title, unboxed_ann, unboxed_marker = _uf_title, _uf_ann, _uf_marker
unboxed_gap = None if _title_gap is None else round(float(_title_gap), 4)
harvest_texts(fig)
plt.savefig(FIG / "e043_fig1_gd_path.png")
plt.close(fig)

# ════════════════════════════════════════════════════════════
# 图 2：不同步长的能量下降曲线（三条线——线型/颜色含义进 deck figcaption+口播）
# ════════════════════════════════════════════════════════════
def gd_curve(eta, steps):
    w = np.zeros(2)
    es = [E_line(w)]
    for _ in range(steps):
        w = w - eta * grad_line(w)
        es.append(E_line(w))
    return np.array(es)


es_slow = gd_curve(0.0002, 2000)   # 太慢档：收缩因子 1−0.0002×2=0.9996（能量超额每步 ×0.9992）——2000 步远未到底（讲者实验配置；r3 P2-6 注释值更正）
es_fit = gd_curve(ETA, 2000)
es_big = gd_curve(ETA_BIG, 2000)

fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
ax.semilogy(range(len(es_fit)), es_fit, color="tab:blue", lw=2.2)
ax.semilogy(range(len(es_slow)), es_slow, color="tab:green", lw=2.2, linestyle="--")
ax.semilogy(range(len(es_big)), np.minimum(es_big, 1e8), color="tab:red", lw=2.2, linestyle=":")
ax.set_xlabel("迭代步数", fontsize=17)
ax.set_ylabel("能量 E（对数轴）", fontsize=17)
ax.set_title("步长决定下降：太慢、合适、发散", fontsize=18)
ax.set_ylim(1e-2, 1e8)
plt.tight_layout()
harvest_texts(fig)
plt.savefig(FIG / "e043_fig2_eta_curves.png")
plt.close(fig)

# ════════════════════════════════════════════════════════════
# 图 3：交叉熵梯度=预测减目标（式 5.62 直觉——p 条形+目标虚线+差值）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
cats = ["类别 1", "类别 2", "类别 3"]
bars = ax.bar(cats, p, color=["#9ecae1", "#9ecae1", "#4292c6"], width=0.55)
for k in range(3):
    ax.plot([k - 0.325, k + 0.325], [t_onehot[k], t_onehot[k]], color="crimson", lw=2.5,
            linestyle="--")
    ax.annotate(f"p−t={p[k] - t_onehot[k]:+.3f}", xy=(k, max(p[k], t_onehot[k]) + 0.06),
                fontsize=16, ha="center", color="dimgray")
ax.set_ylim(0, 1.25)
ax.set_ylabel("概率", fontsize=17)
ax.set_title("式 5.62：梯度=预测概率减目标（红虚线）", fontsize=18)
plt.tight_layout()
harvest_texts(fig)
plt.savefig(FIG / "e043_fig3_p_minus_t.png")
plt.close(fig)

# ════════════════════════════════════════════════════════════
# E6：图内文字运行时收割（matplotlib 对象遍历 fig.findobj(Text)——内部记号零泄漏红线 5）
# ════════════════════════════════════════════════════════════
hits = sum(1 for s in ALL_TEXTS if FIG_MARK.search(s))
vis_total = sum(v for _, v in E6_PER_FIG)
inv_total = len(ALL_TEXTS) - vis_total
check("E6 图内文字运行时收割（matplotlib 对象遍历 fig.findobj(Text)：标题/批注/图例/刻度全部 Text 对象，三图 savefig 前收割——内部记号零泄漏红线 5；r3 P1-2 整改：旧实现为源码正则静态扫描，声明「对象遍历」与实现不符+计数 17 含 4 条非绘图行+漏图3 x 刻度文字 cats；r4 P2-3：计数口径分全部/visible 标志两档申报，扫全部=更严不漏检；r7 P2-4 改口：visible=get_visible() 标志为真，含视图外未绘制刻度（MultipleLocator 向两端各外扩一档的刻度 label1 visible=True 但不在视图内、不绘制），不等于图上实际呈现数；r5 P2-2 立项+r6 P1-1 更正：不可见对象构成以「对象 is Tick.label2」身份实测计数（label2=每刻度对侧默认隐藏的同文本标签），v6「孤儿副本」归因系未实测推断已删除，其余不可见对象只计数不归因）",
      hits == 0,
      f"运行时收割 {len(ALL_TEXTS)} 条 Text 对象（visible 标志为真 {vis_total} 条，含视图外未绘制"
      f"刻度；不可见 {inv_total}=Tick label2 对侧隐藏 {E6_INV_LABEL2}+其他 {E6_INV_OTHER}）命中 {hits}")

# ════════════════════════════════════════════════════════════
# readings.json（backfill 读数存档）
# ════════════════════════════════════════════════════════════
readings = {
    "data": {"n": 40, "x_range": [-3, 3], "noise": 0.05, "seed": 2042},
    "e1_numeric_vs_analytic": {"max_rel": rel_g, "eps": 1e-6, "D": int(D)},
    "e2_ce_grad_p_minus_t": {"p": [float(x) for x in p], "t": [1.0 if k == 2 else 0.0 for k in range(3)],
                             "n_classes": 3, "max_rel": rel_ce},
    "e3_gd_vs_ols": {"eta": ETA, "steps": STEPS, "steps_to_conv": steps_to_conv, "E_gd": E_gd, "E_ols": E_ols_line, "rel": rel_gd, "lam_max": LAM_MAX, "w_gd": [float(x) for x in w_gd], "w_ols": [float(x) for x in w_ols]},
    "e4_monotone": {"strict_descent_steps": steps_to_conv, "max_increase": float(diffs.max()),
                    "max_single_decrease": float(diffs.min()),
                    "ulp_note": "max_increase≈2.78e-17=E≈0.139 处一个 ulp 的浮点底噪（r2 P1-2 整改申报）",
                    "E_start": float(arr[0]), "E_end": float(arr[-1])},
    "e5_divergence": {"eta_big": ETA_BIG, "E_0": E_big0, "E_last": float(E_big_last),
                      "cross_1000x_step": cross_1000x_step, "nonfinite_step": nonfinite_step,
                      "threshold_1000x": 1000, "loop_limit": 2000, "stop_rule": "非有限即停"},
    "e7_half_step": {"eta": ETA_HALF, "E_after_1": float(E_half_1), "rel": float(rel_half),
                     "claim": "一步到谷底（收缩因子 1−ηλ_max=0）"},
    "e8_critical": {"eta": ETA_CRIT, "E_const": float(E_crit0), "max_drift": float(crit_drift),
                    "steps": 20, "n_flip": n_flip, "pairs": pairs, "osc_flip": bool(n_flip == pairs),
                    "claim": "临界能量恒定（参数谷底两侧来回跳）——r4 P1-1：翻转计数改程序计算入存证"},
    "e8_literal": {"eta": ETA_LIT, "E_const": float(E_lit0), "max_drift": float(lit_drift),
                   "steps": 20, "n_flip": n_flip_lit, "pairs": pairs_lit,
                   "claim": "口播「一点零」的字面输入 η=1.0（ETA_LIT 变量引用——r5 P2-3）——与 e8_critical 的 "
                            "2/λ_max 实算值相差 2e-16，存证单独覆盖口播值（r4 P2-4 立项；r5 P2-3 申报更正："
                            "numeric 机检为存档同值命中，未做字段绑定）"},
    "fig2_eta_tiers": {"etas": [0.0002, ETA, ETA_BIG],
                       "E_end": [float(es_slow[-1]), float(es_fit[-1]), "发散（clip 到 1e8 绘图）"],
                       "labels": ["太慢", "合适", "发散"],
                       "note": "deck 页7 图2 caption 三档 η 的存证（r2 P2-4）；E_end 对应 2000 步终点"},
    "e6_figtext": {"scanned_all": len(ALL_TEXTS), "scanned_visible": vis_total,
                   "inv_total": inv_total,
                   "inv_label2": E6_INV_LABEL2, "inv_other": E6_INV_OTHER,
                   "per_fig_all": [n for n, _ in E6_PER_FIG],
                   "per_fig_visible": [v for _, v in E6_PER_FIG],
                   "hits": hits,
                   "method": "fig.findobj(Text) 运行时收割（r3 P1-2——含刻度/批注/标题全部 Text 对象；"
                             "r4 P2-3 计数分全部/visible 标志两档；r7 P2-4 改口：scanned_visible/"
                             "per_fig_visible=get_visible() 标志为真计数，含视图外未绘制刻度"
                             "（MultipleLocator 向两端各外扩一档的刻度 label1 visible=True 但不在"
                             "视图内、不绘制），不等于图上实际呈现数；r5 P2-2 立项+r6 P1-1 更正：不可见构成以"
                             "「对象 is Tick.label2」身份实测（label2=每刻度对侧默认隐藏的同文本标签），"
                             "v6 孤儿副本归因删除，其余只计数不归因）"},
    "fig1_traj": {"start": [3.0, 2.2], "eta": ETA, "steps_to_conv": fig1_steps,
                  "note": "图1 轨迹自身首达步数（rel<1e-6）——deck 图1 caption 引本值，与 E3 零点起跑的 "
                          "steps_to_conv 不同源（r3 P2-1 整改：旧 caption「11 步」错挂 E3 读数）"},
    "f1a_geometry": {"argmin": [float(argmin_pt[0]), float(argmin_pt[1])],
                     "argmin_dist": float(np.linalg.norm(argmin_pt - w_ols)),
                     "grid_pitch": float(grid_pitch), "grad_angle_deg": angle_deg,
                     "aspect": float(ax_aspect_val), "bbox_overflow": bbox_overflow,
                     "anns_bboxed": anns_count, "marker_hit": marker_hit,
                     "axes_intersect": axes_int_val, "arrow_missing": arrow_missing_val,
                     "arrow_hit": arrow_hit_val,
                     "converged_idx": _converged_idx, "converged_dist_px": _converged_dist,
                     "unboxed_texts": unboxed_count, "unboxed_out": unboxed_out,
                     "unboxed_axes_violation": unboxed_axes, "unboxed_title_hit": unboxed_title,
                     "unboxed_ann_hit": unboxed_ann, "unboxed_marker_hit": unboxed_marker,
                     "unboxed_title_gap_px": unboxed_gap,
                     "note": "F1a/F1b 断言读数（r3 P1-1+r4 P2-2 立项+r5 P1-2/P2-1 强化+r6 P2-1 "
                             "converged_idx/converged_dist_px 豁免集入存证（全局步位=traj 下标，"
                             "距离=圆心距星心显示 px）+r6 P2-2 无框文字检查字段）——make_script "
                             "lab-stdout 数值命中机检数据源；申报只引字段不手写数字"},
}
n_pass = sum(1 for _, ok, _ in check_log if ok)
readings["n_pass"] = n_pass
readings["n_total"] = len(check_log)
(OUT / "e043_loss_gradient_readings.json").write_text(
    json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e043_loss_gradient_readings.json').read_bytes()).hexdigest()}"
      f"（哨兵③版本绑定数据源——r5 P2-5）")
print(f"[lab] LEDGER：断言 {len(check_log)} 条，PASS {n_pass}")
T1 = datetime.datetime.now()
for fpath in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({fpath.name})={hashlib.sha256(fpath.read_bytes()).hexdigest()}")
print(f"[lab] 结束 {T1.isoformat(timespec='seconds')}（起 {T0.isoformat(timespec='seconds')}）")
if n_pass != len(check_log):
    sys.exit(1)
