#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E041 lab —— 非线性最小二乘与 LM 算法实验（附录 A.3 读数级实证）

数据与集谱：
- 直线主线：seed 2038 同配方 16 真内点（E038/E039/E040 同款，真值 m=0.5,b=1,σ=0.15；本集不用外点——
  LM 机制演示与第 4 章会师都在纯内点上做，外点话题已由 E038–E040 承担）。
  非线性参数化 (θ,b)：y = x·tanθ + b（倾角-截距）——与线性 (m,b) 是同一能量面的重参数化
  （m=tanθ），最优解严格不变，故收敛解可与线性 OLS（本 lab 同配方现算）逐元对照（第 4 章闭环；
  E036 为 seed 2036/8 点/σ=0.5 的另一数据集，不可混用——E7 对照口径=同配方现算 OLS）。
  注意：不用法向 (θ,ρ) 参数化（其残差=正交距离，与竖直残差 OLS 不是同一最优化问题——
  探测实证入 readings.line_vs_normal_param，本 lab 实算两值，见 E9 探测块）。
- 圆拟合演示场：seed 2041，12 点 r=2 圆弧（φ∈[−0.9,0.9]）+N(0,0.03) 噪声，残差=几何距离
  √((x−a)²+(y−c)²)−r。失灵主因申报=远起点+强非线性（最优参数附近存在大半径退化直线谷，
  JᵀJ 病态）——非「大残差问题」（σ=0.03，最优点处残差能量≈0.019，是小残差；原书 p.931
  收尾段 large residual methods 句是「初值远离二次收敛区时换牛顿型」的指路，不作为本场分类依据
  ——round1 P1-6 申报纠正）。能量地形两盆地：真参数处能量参照 E≈0.0207+大半径退化直线谷
  （E≈0.7766）——远起点 (3,3,2)/(4,0,3) 侧的 LM 也滑入直线谷（E6 far_starts 实算申报）。
- 残差与雅可比符号口径（round1 P1-7 统一申报）：
  本 lab 全部使用**残差雅可比** J_r=∂r/∂p（非 ∂f/∂p）。直线 r=Y−X·tanθ−b=x′−f（与 A.46 的
  rᵢ=x′ᵢ−f 同号）；圆 r=dist−r_target=f−x′（与 A.46 反号，两场口径不同各自自洽）。
  预测能量统一写 E_pred=Σ‖r+J_r∆p‖²——与 A.46 的 Σ‖J∆p−r_A46‖² 逐项等价（等距同构：符号
  内外翻转）。GN 步最小化 E_pred ⇒ E_pred≤E_pred(0)=E 当前能量恒成立（**结构恒真项**，非实测）。
- LM 实现：Madsen–Nielsen–Tingleff 2004 Method 1 增益比方案——ρ=(E−E_real)/(E−E_pred)，
  E_pred=Σ‖r+J∆p‖² 即 (A.46) 预测能量（原书 p.931 λ 升降判据「as predicted by (A.46)」的
  标准化实现）；ρ>0 接受且 λ×max(1/3, 1−(2ρ−1)³)、ν=2；ρ≤0 拒绝且 λ×ν。
  **超出 MNT 2004 的部分=讲者实现约定，单独申报**（round1 P2-4+round2 P2-3）：ν 上限 64（MNT 中 ν 无上界）、
  λ₀=1e-3 绝对值（MNT 为 τ·max diag(C)）、终止条件（‖d‖<1e-12 且 na>2，或 λ>1e14，或 n_iter=400 上限截断——
  far_starts 分起点：(3,3,2)=400 轮上限截断、(4,0,3)=155 轮步长停滞（‖d‖<tol_d 触发退出，stop_reason 命名
  为「步长停滞」非「收敛」——停于大半径渐近谷，非收敛到极小，round4 P2-2）；E7 噪声判定阈值 ‖d‖≤1e-8
  为实现约定（介于 tol_d=1e-12 与有效步长之间，区分收敛末端舍入拒绝与真实坏步，每条噪声轮存
  norm_d/e_gap 入 readings 供复核，round4 P2-3））、lam_min 参数已删（round2）。
  λ 因子数值=文献方案，方向与判据=原书。
- 「直线谷」性质申报（round2 P2-3）：远起点侧参数发散至 1e5–1e6 量级、E 逼近下确界≈0.7766 的**渐近谷**
  （非有限极小点）——退化方向上 E 单调趋近渐近值，GN/LM 均不收敛到极小。
主锚：A.3 p.930–931（e041_book_a3_anchors.txt §1）；衔接 p.945 非线性句（§2）；p.235 第 5 章起始（§4）。
输出：code/outputs/e041_nonlinear_lstsq_lab.txt（stdout 重定向）+e041_nonlinear_lstsq_readings.json+figures 三图。
"""
import json
import math
import pathlib
import re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]  # 仓库根（code→E041→episodes→根）
EP = ROOT / "episodes/E041_nonlinear_lstsq"
OUT = EP / "code/outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Songti SC", "STSong", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 17
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 17
plt.rcParams["axes.titlesize"] = 18

checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("　| " + detail if detail else ""))

# ── E8 准备：出图前收割全部运行时文字（E040 判例机检延续——round3 P2-3 版含 xlabel/ylabel）──
RUNTIME_FIGTEXTS: list[tuple[str, list[str]]] = []

def harvest_figtexts(tag, fig, ax):
    texts = [t.get_text() for t in fig.texts]
    texts.append(fig._suptitle.get_text() if fig._suptitle else "")
    texts.append(ax.get_title())
    texts += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
    texts += [t.get_text() for t in ax.texts]
    texts += [ax.get_xlabel(), ax.get_ylabel()]
    leg = ax.get_legend()
    if leg:
        texts += [t.get_text() for t in leg.get_texts()]
    RUNTIME_FIGTEXTS.append((tag, [s for s in texts if s]))

MARK_RE = re.compile(r"E0\d\d|run\s*\d+|runN|round\s*\d+|门[ABC]|f0\d\d|§")

# ════════════════════════════════════════════════════════════
# 数据重放
# ════════════════════════════════════════════════════════════
rng = np.random.default_rng(2038)
xs = np.sort(rng.uniform(0.0, 4.0, 16))
ys = 1 + 0.5 * xs + rng.normal(0, 0.15, 16)
X, Y = xs, ys

A_lin = np.column_stack([X, np.ones_like(X)])
m_ols, b_ols = np.linalg.lstsq(A_lin, Y, rcond=None)[0]
E_OLS_LINE = float(np.sum((Y - A_lin @ np.array([m_ols, b_ols])) ** 2))  # 同配方现算 OLS 最优能量

def resid_line(p):
    th, b = p
    return Y - X * np.tan(th) - b            # = x′−f（A.46 同号）

def energy_line(p):
    r = resid_line(p)
    return float(r @ r)                       # A.45：Σ‖r‖²（原书口径，无 ½）

def jac_line(p):
    th = p[0]
    return np.column_stack([-X / np.cos(th) ** 2, -np.ones_like(X)])   # ∂r/∂p

rng2 = np.random.default_rng(2041)
phi = np.linspace(-0.9, 0.9, 12)
xc = 2 * np.cos(phi) + rng2.normal(0, 0.03, 12)
yc = 2 * np.sin(phi) + rng2.normal(0, 0.03, 12)
A_TRUE, C_TRUE, R_TRUE = 0.0, 0.0, 2.0

def resid_circ(p):
    return np.sqrt((xc - p[0]) ** 2 + (yc - p[1]) ** 2) - p[2]   # = f−x′（与 A.46 反号，自洽）

def energy_circ(p):
    r = resid_circ(p)
    return float(r @ r)

def jac_circ(p):
    d = np.sqrt((xc - p[0]) ** 2 + (yc - p[1]) ** 2)
    return np.column_stack([-(xc - p[0]) / d, -(yc - p[1]) / d, -np.ones_like(xc)])  # ∂r/∂p，第 3 列恒 −1（对 r 线性）

E_CIRC_TRUE = energy_circ(np.array([A_TRUE, C_TRUE, R_TRUE]))   # 真参数处能量参照（非最优值——LM 收敛可更低）

# ════════════════════════════════════════════════════════════
# E1：Jacobian 解析 vs 数值中心差分（原书 p.930「Jacobian」定义验证；
#     本 lab J 为残差雅可比 ∂r/∂p——round1 P1-7 口径，断言名同步）
# ════════════════════════════════════════════════════════════
p_e1 = np.array([0.7, 0.4])
J_an = jac_line(p_e1)
eps = 1e-7
J_num = np.zeros_like(J_an)
for k in range(2):
    e = np.zeros(2); e[k] = eps
    J_num[:, k] = (resid_line(p_e1 + e) - resid_line(p_e1 - e)) / (2 * eps)
e1_max_rel = float(np.abs(J_an - J_num).max() / np.abs(J_an).max())
check("E1 Jacobian 解析（残差雅可比 ∂r/∂p）=中心差分验证（原书 p.930 定义句）",
      e1_max_rel < 1e-8, f"max_rel={e1_max_rel:.3e}")

# ════════════════════════════════════════════════════════════
# E2：A.46 泰勒预测失灵双形态（round1 P1-2 恒真/实测拆分申报）
#   【结构恒真项（非实测，勿冒充）】
#   (i)  GN 步最小化 E_pred ⇒ E_pred≤E（当前能量）恒成立——对任何问题任何起点。
#   (ii) 直线 (θ,b) 场：J 的列空间 {x/cos²θ,1}={x,1} 与线性 (m,b) 相同 ⇒ GN 一步预测能量
#        E_pred 恒等于 OLS 最优能量（下方以 rel 读数申报，不作为实证断言条件）。
#   【实测项（断言条件只留这些）】
#   E2a：走完 GN 全步的实测能量 E_real 远大于 E_pred（远起点实测失准）；
#   E2b：实测 E_after>E_before（能量上升——原书 may not always decrease 直接实证）。
# ════════════════════════════════════════════════════════════
p_far = np.array([1.2, 3.0])
J0 = jac_line(p_far); r0 = resid_line(p_far)
d_gn_line = np.linalg.solve(J0.T @ J0, -J0.T @ r0)
E0_line = energy_line(p_far)
E_pred_line = float(np.sum((J0 @ d_gn_line + r0) ** 2))
E_real_line = energy_line(p_far + d_gn_line)
e2a_ols_rel = abs(E_pred_line - E_OLS_LINE) / E_OLS_LINE   # 结构恒真项申报读数（预期≈0）
# round2 P2-2：断言条件只留实测倍数（E_real/E_pred>10）——E_pred<E0 与 E_pred<E_real 均为结构必然
# （E_pred=OLS 全局最小 ⇒ E_pred≤E_real 恒真；申报见 structural_identity_e2a）
check("E2a A.46 预测失灵（直线远起点·实测项：剩余比 >10）：走完全步实测 E_real 远高于预测——线性近似把剩余路当成一步跨完",
      E_real_line / E_pred_line > 10,
      f"E0={E0_line:.1f} 预测剩余={E_pred_line:.4f} 实测剩余={E_real_line:.1f}"
      f"（剩余比 实测/预测={E_real_line/E_pred_line:.1f} 倍——唯一实测断言；恒真申报：E_pred vs OLS 最优 rel={e2a_ols_rel:.2e}）")

# 圆 GN 重放到首个上升步（i=3，即第 4 步）
def gn_traj_circ(p0, n_iter=30):
    p = np.array(p0, dtype=float)
    Es = [energy_circ(p)]
    ps = [p.copy()]
    steps = []                      # (i, d, E_before, E_after)
    rises = 0
    for i in range(n_iter):
        J = jac_circ(p); r = resid_circ(p)
        d = np.linalg.solve(J.T @ J, -J.T @ r)
        E_after = energy_circ(p + d)
        steps.append((i, d.copy(), Es[-1], E_after))
        if E_after > Es[-1] + 1e-12:
            rises += 1
        p = p + d
        Es.append(E_after); ps.append(p.copy())
    return p, Es, ps, steps, rises

p_gn_c, Es_gn_c, Ps_gn_c, steps_gn_c, rises_gn_c = gn_traj_circ([3.0, 3.0, 2.0])
# 上行取步前位置（round1 P2-10：死代码已删）
p_at = None
p_tmp = np.array([3.0, 3.0, 2.0])
for i, d, Eb, Ea in steps_gn_c:
    if Ea > Eb + 1e-12:
        p_at = p_tmp.copy(); d_blow = d.copy(); E_before_blow = Eb; E_after_blow = Ea
        i_blow = i
        break
    p_tmp = p_tmp + d
J_at = jac_circ(p_at); r_at = resid_circ(p_at)
E_pred_blow = float(np.sum((J_at @ d_blow + r_at) ** 2))
# round1 P1-3：爆升前三步实测存档（能量在降但参数已甩到 1e5 量级）
pre_blow = [{"step": i, "E": Es_gn_c[i], "p": list(map(float, Ps_gn_c[i]))} for i in range(i_blow)]
check("E2b A.46 预测失灵（圆 GN 第 4 步·实测项）：实测能量上升——原书 p.931「update may not always decrease」直接实证（恒真申报：E_pred≤E_before 结构必然而非实测）",
      E_after_blow > E_before_blow,
      f"E_before={E_before_blow:.4f} 预测={E_pred_blow:.4f}（≤E_before 恒真）实测={E_after_blow:.3e}")

# ════════════════════════════════════════════════════════════
# E3：圆 GN (3,3,2) 上升步与不收敛（round1 P1-3：前三步轨迹入 readings）
# ════════════════════════════════════════════════════════════
check("E3 圆 GN (3,3,2)：多次上升步+30 轮不达真参数处能量——线性化在远处不可靠",
      rises_gn_c >= 10 and Es_gn_c[-1] > E_CIRC_TRUE * 10,
      f"上升步={rises_gn_c} E末={Es_gn_c[-1]:.4f} vs 真参数处能量参照={E_CIRC_TRUE:.4f} 轮数={len(Es_gn_c)-1}")

# ════════════════════════════════════════════════════════════
# E4：A.47 步长减半 line search（在爆升步位置找首个下降 α）——含步长读数（round1 P2-3）
# ════════════════════════════════════════════════════════════
norm_d_gn = float(np.linalg.norm(d_blow))
alpha = 1.0
halves = 0
alpha_star = None
while alpha > 1e-9:
    if energy_circ(p_at + alpha * d_blow) < E_before_blow:
        alpha_star = alpha
        break
    alpha /= 2
    halves += 1
alpha_step_len = alpha_star * norm_d_gn
check("E4 A.47 line search：α 从 1 逐次减半找到首个下降步（α*<1）——原书减半机制实证",
      alpha_star is not None and alpha_star < 1,
      f"α*={alpha_star:.3e} 减半次数={halves} ‖d_GN‖={norm_d_gn:.3e} 实走步长=α*·‖d‖={alpha_step_len:.3e}（该位置 E_before={E_before_blow:.4f}）")

# ════════════════════════════════════════════════════════════
# E5：A.49 阻尼步对照（同一位置：GN 全步爆升，λ 逐级 ×10 找到稳降阻尼步）——含步长读数
# ════════════════════════════════════════════════════════════
JTJ_at = J_at.T @ J_at
lam_e5 = 1e-3
dE_e5 = None
lam_e5_star = None
d_damped_e5 = None
while lam_e5 <= 1e8:
    d = np.linalg.solve(JTJ_at + lam_e5 * np.diag(np.diag(JTJ_at)), -J_at.T @ r_at)
    E_new = energy_circ(p_at + d)
    if E_new < E_before_blow:
        dE_e5 = E_before_blow - E_new
        lam_e5_star = lam_e5
        d_damped_e5 = d.copy()
        break
    lam_e5 *= 10
norm_d_damped = float(np.linalg.norm(d_damped_e5))
# round2 P2-4：病态读数——核实 f006「拧紧弹簧」比喻（阻尼压制近零空间方向）
evals_at, evecs_at = np.linalg.eigh(JTJ_at)
cond_jtj = float(evals_at.max() / evals_at.min())
v_min = evecs_at[:, 0]                      # 最小特征值方向（近零空间）
proj_gn = float(abs(v_min @ d_blow))
proj_damped = float(abs(v_min @ d_damped_e5))
check("E5 A.49 阻尼步：同一爆升位置加对角阻尼后实测下降——阻尼让泰勒近似在可靠范围内使用",
      dE_e5 is not None and dE_e5 > 0,
      f"λ*={lam_e5_star:.0e} ΔE=+{dE_e5:.4f} ‖d_阻尼‖={norm_d_damped:.3e}（对照：GN 全步 ΔE={E_before_blow-E_after_blow:+.3e}）"
      f" 病态：cond(JᵀJ)={cond_jtj:.2e} λmin={evals_at.min():.2e} 最小特征方向投影 |v·d|：GN={proj_gn:.3e} vs 阻尼={proj_damped:.3e}")

# ════════════════════════════════════════════════════════════
# LM（MNT 2004 Method 1，增益比 ρ 判据=A.46 预测对比；超 MNT 约定申报见 docstring）
# ════════════════════════════════════════════════════════════
def lm_mnt(resid, energy, jac, p0, lam0=1e-3, n_iter=400, lam_max=1e14, tol_d=1e-12):
    p = np.array(p0, dtype=float).copy()
    E = energy(p)
    Es = [E]; lams = [lam0]; lam = lam0; nu = 2.0
    na = nr = 0
    acc_seq = []
    steps = []   # 每轮明细（round3 P1-2：lam 前值/步长/ρ/是否接受——区分有效坏步与收敛末端噪声）
    stop_reason = "n_iter 上限截断"
    for i in range(n_iter):
        J = jac(p); r = resid(p)
        JTJ = J.T @ J
        d = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), -J.T @ r)
        E_real = energy(p + d)
        E_pred = float(np.sum((J @ d + r) ** 2))   # A.46 预测能量（残差雅可比口径，见 docstring）
        den = E - E_pred
        rho = (E - E_real) / den if den > 1e-300 else (math.inf if E - E_real > 0 else -math.inf)
        lam_before = lam
        if rho > 0:
            p = p + d; E = E_real; na += 1
            lam = lam * max(1.0 / 3.0, 1.0 - (2 * rho - 1) ** 3); nu = 2.0
            acc_seq.append(True)
        else:
            lam = lam * nu; nu = min(2 * nu, 64.0); nr += 1
            acc_seq.append(False)
        steps.append({"lam_before": lam_before, "lam_after": lam, "norm_d": float(np.linalg.norm(d)),
                      "rho": rho if math.isfinite(rho) else None, "accepted": acc_seq[-1],
                      "e_gap": float(E - E_real)})
        Es.append(E); lams.append(lam)
        if lam > lam_max:
            stop_reason = "lam>lam_max 发散"
            break
        if na > 2 and np.linalg.norm(d) < tol_d:
            stop_reason = "norm_d<tol_d 步长停滞"
            break
    return p, Es, lams, na, nr, acc_seq, steps, stop_reason

# ════════════════════════════════════════════════════════════
# E6：圆 LM 近起点收敛真参数邻域 + 远起点滑谷实算申报（round1 P1-4/P1-12/P2-5）
# ════════════════════════════════════════════════════════════
p_lm_c, Es_lm_c, lams_lm_c, na_c, nr_c, acc_seq_c, steps_lm_c, stop_lm_c = lm_mnt(resid_circ, energy_circ, jac_circ, [1.5, 1.5, 1.5])
a_lm, c_lm, r_lm = p_lm_c
# 远起点 LM 实算（round1 P1-4：口播须申报「远起点 LM 也掉谷」——本块给出读数）
far_starts = {}
for fs in ([3.0, 3.0, 2.0], [4.0, 0.0, 3.0]):
    pf, Esf, lamsf, naf, nrf, _, _, stopf = lm_mnt(resid_circ, energy_circ, jac_circ, fs)
    far_starts[str([float(v) for v in fs])] = {
        "E_final": Esf[-1], "iters": len(Esf) - 1, "a": float(pf[0]), "c": float(pf[1]), "r": float(pf[2]),
        "slide_valley": bool(Esf[-1] > 0.5), "stop_reason": stopf,
    }
far_slide_count = sum(v["slide_valley"] for v in far_starts.values())
check("E6 圆 LM（近起点 1.5,1.5,1.5，MNT 增益比）：收敛真参数邻域——E 末≈真参数处能量参照，圆心近 (0,0)、半径近 2；远起点 LM 滑入直线谷（far_starts 实算申报）",
      abs(Es_lm_c[-1] - E_CIRC_TRUE) < 0.1 * E_CIRC_TRUE and abs(a_lm) < 0.1 and abs(c_lm) < 0.1 and abs(r_lm - 2) < 0.1
      and all(v["slide_valley"] for v in far_starts.values()),
      f"轮数={len(Es_lm_c)-1} 接受{na_c}/拒绝{nr_c} E末={Es_lm_c[-1]:.6f} (a={a_lm:.4f},c={c_lm:.4f},r={r_lm:.4f})"
      f" 远起点滑谷={['%.4f@%s' % (v['E_final'], k) for k, v in far_starts.items()]}（滑谷 {far_slide_count} 起点，"
      f"退出原因 {{{', '.join('%s:%s' % (k, v['stop_reason']) for k, v in far_starts.items())}}}）")

# ════════════════════════════════════════════════════════════
# E7：直线主线 LM 三起点会师线性 OLS（第 4 章闭环；对照=同配方现算 OLS——round1 P2-6 口径）
# ════════════════════════════════════════════════════════════
starts = [(1.2, 3.0), (0.05, 3.0), (1.35, -2.0)]
line_runs = []
max_rel_all = 0.0
for p0 in starts:
    p_lm, Es_l, lams_l, na_l, nr_l, _, steps_l, stop_l = lm_mnt(resid_line, energy_line, jac_line, p0)
    m_lm, b_lm = float(np.tan(p_lm[0])), float(p_lm[1])
    rel = max(abs(m_lm - m_ols) / abs(m_ols), abs(b_lm - b_ols) / abs(b_ols))
    max_rel_all = max(max_rel_all, rel)
    # round3 P1-2：升 λ 分有效轮（‖d‖>1e-8，真实坏步/低增益）与噪声轮（收敛末端 ‖d‖<tol 的舍入拒绝）
    lam_raises_eff = sum(1 for s in steps_l if s["lam_after"] > s["lam_before"] * 1.001 and s["norm_d"] > 1e-8)
    noise_rejects = sum(1 for s in steps_l if s["lam_after"] > s["lam_before"] * 1.001 and s["norm_d"] <= 1e-8)
    noise_steps = [{"norm_d": s["norm_d"], "e_gap": s["e_gap"], "lam_ratio": s["lam_after"] / s["lam_before"]}
                   for s in steps_l if s["lam_after"] > s["lam_before"] * 1.001 and s["norm_d"] <= 1e-8]
    line_runs.append({"p0": list(p0), "iters": len(Es_l) - 1, "accept": na_l, "reject": nr_l,
                      "m": m_lm, "b": b_lm, "max_rel": rel, "lam_raises": lam_raises_eff,
                      "noise_rejects": noise_rejects, "stop_reason": stop_l,
                      "E_final": Es_l[-1], "lam_traj_head": lams_l[:8], "noise_steps": noise_steps})
check("E7 直线主线 LM 三起点会师：换算 (m,b)=（tanθ, b）与同配方现算线性 OLS 逐元一致（<1e-9）——直线场快速收敛，升 λ 只出现在收敛末端舍入拒绝（noise_rejects 申报，非自适应双向实测；双向实测以 E6 圆场为证）",
      max_rel_all < 1e-9,
      f"max_rel={max_rel_all:.2e}（OLS m={m_ols:.10f} b={b_ols:.10f}）；"
      + "; ".join(f"起点{r['p0']} {r['iters']}轮 有效升λ{r['lam_raises']}次+噪声拒绝{r['noise_rejects']}次（{r['stop_reason']}）" for r in line_runs))
check("E7b 直线场有效升 λ=0 一致性（round5 P2-2：E7 断言名「升 λ 只出现在收敛末端舍入拒绝」的数据守卫，独立于会师断言）",
      all(r["lam_raises"] == 0 for r in line_runs),
      "有效升λ按起点序=" + repr([r["lam_raises"] for r in line_runs])
      + "（全 0 则 E7 断言名陈述成立；若非 0 须重审 E7 断言名与 f011 refs 口径）")

# ════════════════════════════════════════════════════════════
# E9（round1 P2-5 纳入实算）：法向 (θ,ρ) 参数化 vs 倾角-截距 (θ,b) 非同一问题——探测代码
# ════════════════════════════════════════════════════════════
def resid_norm(p):
    th, rho = p
    return -np.sin(th) * X + np.cos(th) * Y - rho     # 正交距离（竖直残差的另一问题）

def jac_norm(p):
    th = p[0]
    return np.column_stack([-np.cos(th) * X - np.sin(th) * Y, -np.ones_like(X)])

pn = np.array([np.arctan(0.5), np.cos(np.arctan(0.5)) * 1.0])
for _ in range(200):
    Jn = jac_norm(pn); rn = resid_norm(pn)
    dn = np.linalg.solve(Jn.T @ Jn, -Jn.T @ rn)
    pn = pn + dn
    if np.linalg.norm(dn) < 1e-14:
        break
E_normal = float(resid_norm(pn) @ resid_norm(pn))
th_star = np.arctan(m_ols)
rho_star = math.cos(th_star) * b_ols
rr = resid_norm(np.array([th_star, rho_star]))
E_normal_at_ols_converted = float(rr @ rr)

# ════════════════════════════════════════════════════════════
# 图 1：直线 (θ,b) 能量等高线 + GN/LM 轨迹 + OLS 会师点（deck 页4；线型说明进 figcaption——round2 P0-B）
# （round1 P0-4+round2 P2-5：figsize (6.0,3.0)=900×450 全 17pt；round4 P2-1 校准：页4 显示高限 265px（display_check v5：265/450=0.5889→17pt→13.35px——round3 P1-1/P2-5 口播加长次生整改 270→265 后未同步））
# （round1 P2-9：网格扩到含起点 (1.2,3.0)）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
th_grid = np.linspace(0.25, 1.25, 220)
b_grid = np.linspace(-0.4, 3.2, 220)
TT, BB = np.meshgrid(th_grid, b_grid)
ZZ = np.zeros_like(TT)
for i in range(TT.shape[0]):
    r = Y[None, :] - X[None, :] * np.tan(TT[i])[:, None] - BB[i][:, None]
    ZZ[i] = (r * r).sum(axis=1)
levels = np.logspace(np.log10(0.31), np.log10(80), 18)
cs = ax.contourf(TT, BB, ZZ, levels=levels, cmap="viridis_r", alpha=0.85, extend="max")
cl = ax.contour(TT, BB, ZZ, levels=[0.5, 1, 2, 4, 8, 16, 32, 64], colors="k", linewidths=0.6, alpha=0.55)
ax.clabel(cl, fontsize=17, fmt="%g", inline=True)
# GN 轨迹（远起点 1.2,3.0）
p_g = np.array([1.2, 3.0]); traj_gn = [p_g.copy()]
for _ in range(30):
    J = jac_line(p_g); r = resid_line(p_g)
    d = np.linalg.solve(J.T @ J, -J.T @ r)
    p_g = p_g + d; traj_gn.append(p_g.copy())
    if np.linalg.norm(d) < 1e-12: break
tg = np.array(traj_gn)
ax.plot(tg[:, 0], tg[:, 1], "--o", color="black", lw=2.2, ms=5, markerfacecolor="none", zorder=5)
p_l, Es_l1, lams_l1, na_1, nr_1, _, _, _ = lm_mnt(resid_line, energy_line, jac_line, [1.2, 3.0])
# 重放 LM 轨迹
p_l2 = np.array([1.2, 3.0]); lam_l = 1e-3; nu_l = 2.0; traj_lm = [p_l2.copy()]
E_l = energy_line(p_l2)
for _ in range(200):
    J = jac_line(p_l2); r = resid_line(p_l2)
    JTJ = J.T @ J
    d = np.linalg.solve(JTJ + lam_l * np.diag(np.diag(JTJ)), -J.T @ r)
    E_new = energy_line(p_l2 + d)
    E_pred = float(np.sum((J @ d + r) ** 2))
    den = E_l - E_pred
    rho = (E_l - E_new) / den if den > 1e-300 else (math.inf if E_l - E_new > 0 else -math.inf)
    if rho > 0:
        p_l2 = p_l2 + d; E_l = E_new; lam_l = lam_l * max(1/3, 1-(2*rho-1)**3); nu_l = 2.0
    else:
        lam_l = lam_l * nu_l; nu_l = min(2*nu_l, 64.0)
    traj_lm.append(p_l2.copy())
    if np.linalg.norm(d) < 1e-12 and rho > 0: break
tl = np.array(traj_lm)
ax.plot(tl[:, 0], tl[:, 1], "r-o", lw=2.2, ms=4, zorder=4)
ax.plot(np.arctan(m_ols), b_ols, marker="*", ms=19, color="yellow", markeredgecolor="k", ls="none", zorder=6)
ax.set_xlabel("倾角 θ（弧度）")
ax.set_ylabel("截距 b")
ax.set_title("换成倾角-截距写法后，只能迭代着解", pad=32)
fig.tight_layout()
harvest_figtexts("图1", fig, ax)
fig.savefig(FIG / "e041_line_traj.png")
plt.close(fig)

# ════════════════════════════════════════════════════════════
# 图 2：A.46 泰勒预测 vs 实测（爆升步的 α 扫描，对数纵轴）——deck 页6 CSS .p5fix 显示高限 248px（round3 溢出整改后）：
# (6.0,3.0)=900×450→scale=min(0.733,248/450=0.551)=0.551→17pt≈12.49px（display_check v5 口径）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 3.0), dpi=150)
etas = np.linspace(0.02, 1.0, 60)
E_real_scan = [energy_circ(p_at + eta * d_blow) for eta in etas]
E_pred_scan = [float(np.sum((J_at @ (eta * d_blow) + r_at) ** 2)) for eta in etas]
l1, = ax.plot(etas, E_real_scan, "o-", ms=3.5, color="tab:red")
l2, = ax.plot(etas, E_pred_scan, "s--", ms=3.5, color="tab:blue")
l3 = ax.axhline(E_before_blow, color="gray", ls=":", lw=1.4)
ax.set_yscale("log")
ax.set_ylim(1e0, 1e13)
ax.set_xlabel("步长比例（1.0 = 高斯-牛顿全步）")
ax.set_ylabel("能量（对数轴）")
ax.set_title("泰勒近似只在脚下的小范围内可信", pad=32)
fig.tight_layout()
harvest_figtexts("图2", fig, ax)
fig.savefig(FIG / "e041_taylor_blow.png")
plt.close(fig)

# ════════════════════════════════════════════════════════════
# 图 3：圆 LM 收敛过程（E 序列 + λ 轨迹，双轴对数）——deck 页7 CSS .p6fix 显示高限 230px（round3 溢出整改后）：
# figsize (6.0,2.7)=900×405→scale=min(0.733,230/405=0.568)=0.568→17pt≈12.87px（display_check v5 口径）
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.0, 2.7), dpi=150)
it_axis = np.arange(len(Es_lm_c))
ax.plot(it_axis, Es_lm_c, "o-", ms=4, color="tab:purple")
ax.set_yscale("log")
ax.set_xlabel("迭代轮数")
ax.set_ylabel("能量 E（对数轴）")
ax2 = ax.twinx()
ax2.plot(it_axis, lams_lm_c, "s--", ms=4, color="tab:orange")
ax2.set_yscale("log")
ax2.set_ylabel("阻尼 λ（对数轴）")
ax.axhline(E_CIRC_TRUE, color="gray", ls=":", lw=1.4)
ax.set_title("LM 收敛：下降不及预期 λ 放大，达到预期 λ 收紧", pad=36)
fig.tight_layout()
harvest_figtexts("图3", fig, ax)
fig.savefig(FIG / "e041_lm_converge.png")
plt.close(fig)

# ════════════════════════════════════════════════════════════
# E8：图内文字运行时扫描（红线 5——PNG 内嵌字不得泄漏内部记号）
# ════════════════════════════════════════════════════════════
hits = []
total_texts = 0
for tag, texts in RUNTIME_FIGTEXTS:
    for s in texts:
        total_texts += 1
        if MARK_RE.search(s):
            hits.append((tag, s))
check("E8 图内文字运行时扫描（含 xlabel/ylabel/图例/脚注）：内部记号零命中",
      not hits, f"收割 {total_texts} 条字符串，命中 {len(hits)}")

# ════════════════════════════════════════════════════════════
# readings 落盘
# ════════════════════════════════════════════════════════════
readings = {
    "e1_jac_max_rel": e1_max_rel,
    "e2a_line_pred_underestimates_remaining": {
        "E0": E0_line, "E_pred": E_pred_line, "E_real": E_real_line, "E_ols": E_OLS_LINE,
        "p0": [1.2, 3.0],
        "ratio_remaining_real_over_pred": E_real_line / E_pred_line,
        "structural_identity_e2a": {
            "kind": "恒真项申报（非实测）",
            "note": "直线 (θ,b) 的 J 列空间 {x/cos²θ,1}={x,1} 与线性 (m,b) 相同 ⇒ GN 一步预测能量恒等于 OLS 最优能量；"
                    "对照读数 |E_pred−E_ols|/E_ols=%.2e（结构必然，非发现）" % e2a_ols_rel,
            "e_pred_vs_ols_rel": e2a_ols_rel,
        },
        "note": "实测项：走完 GN 全步 E_real 远高于预测——预测把剩余能量低估 190.8 倍（剩余比=实测剩余/预测剩余；"
                "非下降量之比——round1 P0-1 口径纠正：下降量之比仅 1.09）",
    },
    "e2b_circle_pred_down_real_blow": {
        "E_before": E_before_blow, "E_pred": E_pred_blow, "E_real": E_after_blow,
        "step_index": i_blow, "p_at": list(map(float, p_at)),
        "pre_blow_steps": pre_blow,
        "structural_identity_e2b": {
            "kind": "恒真项申报（非实测）",
            "note": "GN 步最小化 E_pred ⇒ E_pred≤E_before 恒成立；实测项=E_after>E_before（上升）",
        },
        "note": "实测：能量上升——原书 p.931 may not always decrease 直接实证；前三步能量在降但参数已甩到 1e5 量级（大半径退化区）",
    },
    "e3_circle_gn": {"rise_steps": rises_gn_c, "E_final": Es_gn_c[-1], "iters": len(Es_gn_c) - 1,
                      "E_circ_true": E_CIRC_TRUE, "p0": [3.0, 3.0, 2.0],
                      "note": "E_circ_true=真参数处能量参照（非全局最优值——LM 收敛 E 可更低）"},
    "e4_line_search": {"alpha_star": alpha_star, "halves": halves, "E_before": E_before_blow,
                        "norm_d_gn": norm_d_gn, "alpha_step_len": alpha_step_len},
    "e5_damped_step": {"lam_star": lam_e5_star, "dE": dE_e5, "E_before": E_before_blow,
                       "cond_jtj": cond_jtj, "lambda_min": float(evals_at.min()),
                       "proj_gn_vmin": proj_gn, "proj_damped_vmin": proj_damped,
                       "note": "病态读数（round2 P2-4）：cond(JᵀJ)/最小特征值/d_GN 与 d_阻尼在最小特征向量上的投影比——核实 f006「拧紧弹簧」比喻（阻尼压掉近零空间方向）。"
                               "round3 P2-3 量级申报：λmin≈5e-17 已处机器精度量级（双精度 eps≈2.2e-16），cond≈4.8e17 为「数值奇异」的下界性质读数——具体数值是舍入噪声，"
                               "只证明 cond≳1e16；‖d_GN‖=6.39e5（口播六十四万）由近奇异矩阵求解决定，量级可信、末位依赖舍入（实际算出的步长如实引用）",
                        "norm_d_damped": norm_d_damped},
    "e6_circle_lm": {"iters": len(Es_lm_c) - 1, "accept": na_c, "reject": nr_c,
                      "acc_seq": acc_seq_c, "lams": lams_lm_c,
                      "a": float(a_lm), "c": float(c_lm), "r": float(r_lm),
                      "E_final": Es_lm_c[-1], "E_circ_true": E_CIRC_TRUE,
                      "p0": [1.5, 1.5, 1.5],
                      "far_starts": far_starts,
                      "note": "近起点收敛真参数邻域；远起点 (3,3,2)/(4,0,3) 侧 LM 滑入大半径退化直线谷（far_starts 实算申报，"
                              "round1 P2-5 由离线探测纳入 lab）——远起点退出口径分起点（round4 P2-2）：(3,3,2) iters=400"
                              "=n_iter 上限属截断、(4,0,3) iters=155 步长停滞（‖d‖<tol_d 退出）——两者都停于直线谷：参数发散至 "
                              f"1e5–{max(abs(v['a']) for v in far_starts.values()):.1e}、E 逼近下确界≈{list(far_starts.values())[0]['E_final']:.4f} 的渐近谷非有限极小点（round2 P2-3 申报，上界由 far_starts 实测生成 round5 P2-2）——本集口播申报起点依然要紧，"
                              "盆地几何不展开"},
    "e7_line_lm": {"ols_m": float(m_ols), "ols_b": float(b_ols),
                    "max_rel_max": max_rel_all, "runs": line_runs,
                    "note": "换算 (m,b)=(tanθ, b)——与同配方现算线性 OLS 逐元一致（E036 为另一数据集不可混用——round1 P2-6 口径）；"
                            f"直线场快速收敛，有效升 λ 0 次；升 λ 仅见于收敛末端舍入拒绝（noise_rejects 按起点序 {[r['noise_rejects'] for r in line_runs]}，‖d‖≤1e-8 判定，明细存 runs[*].noise_steps），不作自适应双向证据——双向实测以 e6 圆场为证"},
    "line_vs_normal_param": {"normal_gn_E": E_normal, "ols_converted_E": E_normal_at_ols_converted,
                              "method": "lab 实算（round1 P2-5）：法向 GN 200 轮收敛能量 vs OLS 解换算到法向参数化的能量",
                              "note": "设计决策入档：法向 (θ,ρ) 残差=正交距离与竖直 OLS 非同一问题（E_normal<E_ols_converted 实证），"
                                      "改用倾角-截距 (θ,b) 保证会师"},
}
(OUT / "e041_nonlinear_lstsq_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))

n_pass = sum(1 for _, ok, _ in checks if ok)
print(f"\n断言 {len(checks)} 条，PASS {n_pass}")
print("LEDGER")
for name, ok, detail in checks:
    print(("[PASS] " if ok else "[FAIL] ") + name)
raise SystemExit(0 if n_pass == len(checks) else 1)
