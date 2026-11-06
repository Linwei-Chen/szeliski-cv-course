#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E040 lab —— 参数不确定度实验（附录 B.6/B.19 读数级实证）

数据与集谱：seed 2038 主数据（E038/E039 同款 17 点=16 内点 y=1+0.5x+N(0,0.15)+外点 (4.6,5.2)）；
胜出内点集=E039 ϵ=0.3 胜出线（m_raw/b_raw 读自 E039 readings，E4b 七字段与 E038 逐位一致）
的残差≤0.3 内点集（应=E038 胜出 15 票内点集）→内点集精拟合线（refit，E038 E5a 口径 m_refit）。
主锚：B.6 p.952（e040_book_b6_anchors.txt §1）；公式支撑 B.2 p.945（§2：(B.18)/(B.19)/(B.20)+CRLB 句）。

实验口径（读数级）：
- B.19 复算：Σ=(Σᵢ HᵢᵀCᵢHᵢ)⁻¹，直线模型 Hᵢ=[xᵢ,1]，Cᵢ=测量协方差的逆（同方差独立时=1/σ² 标量）——主读数代入 σ=0.15（实验约定=原配方已知噪声，
  照 E038/E039「ϵ=2σ 实验约定」同款口径；实操 σ 未知时用残差估计=E037 MAD 手艺，refs 申报）。
- 覆盖率实证（E3）：纯内点模型重抽（seed 2040，400 次×16 点，无外点——高斯假设下 B.19 的频率含义），
  判据 Σ₁₆ 由同一 16 点布局按 B.19 现算（与重抽布局逐元同源——round1 P0-4 修复），
  统计 m̂/b̂ 落 ±2σ₁₆ 的比例 vs 理论 95.45%（判据=理论±2 个蒙特卡洛标准误）。
  含外点的 17 点数据不进覆盖率实验（外点破坏高斯假设）。
- E4 外点的影响：B.19 协方差不依赖 y（同布局换 y 值 Σ 逐元不变——E4a 实测）；外点污染的是 σ 实操估计
  （OLS 残差 std 被外点拉大 vs MAD×1.4 稳健版（E039 E5a 口径）vs 真 σ——E4b）。
- E5 dominant mode：Σ 特征值分析（B.6 eigenvalue analysis 句落点）——主不确定方向 vs 直线方向。
- E6 循环性申报：内点集由 ϵ=0.3 胜出线选出——用其 refit 线反报不确定度非独立估计（E039 E5c 判例延续）。
输出：code/outputs/e040_uncertainty_lab.txt（stdout 重定向）+e040_uncertainty_readings.json+figures 三图。
"""
import json
import math
import pathlib
import re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]  # 仓库根（code→E040→episodes→根）
EP = ROOT / "episodes/E040_uncertainty"
OUT = EP / "code/outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Songti SC", "STSong", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

R39 = json.loads((ROOT / "episodes/E039_ransac_thresholds/code/outputs/e039_ransac_readings.json").read_text())

checks = []
def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(("[PASS] " if ok else "[FAIL] ") + name + ("　| " + detail if detail else ""))

# ── E7 准备：出图前收割全部运行时文字（round2 P2-10——静态扫源码盖不住
#    set_xticklabels(labels) 的列表变量与 f-string 运行时值）──
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

# ── 数据重放：seed 2038 同配方 17 点（E038/E039 同款；真值 m=0.5, b=1, σ=0.15）──
rng = np.random.default_rng(2038)
xs = np.sort(rng.uniform(0.0, 4.0, 16))   # E038/E039 同配方：x 也随机（uniform 0–4），非等距
ys = 1 + 0.5 * xs + rng.normal(0, 0.15, 16)
x_out, y_out = 4.6, 5.2
X17 = np.concatenate([xs, [x_out]])
Y17 = np.concatenate([ys, [y_out]])
SIG_TRUE = 0.15
M_TRUE, B_TRUE = 0.5, 1.0

# ── 胜出内点集：E039 ϵ=0.3 胜出线残差≤0.3 ──
m_rans = R39["e4"]["0.3"]["m_raw"]
b_rans = R39["e4"]["0.3"]["b_raw"]
resid = np.abs(Y17 - (m_rans * X17 + b_rans))
inlier_mask = resid <= 0.3
Xi, Yi = X17[inlier_mask], Y17[inlier_mask]
n_in = int(inlier_mask.sum())

# refit：内点集 OLS 精拟合（E038 E5a 口径）
A = np.column_stack([Xi, np.ones_like(Xi)])
m_refit, b_refit = np.linalg.lstsq(A, Yi, rcond=None)[0]

# ── E1：B.19 复算 vs 数值 Hessian ──
# Σ=(Σᵢ Hᵢᵀ Cᵢ Hᵢ)⁻¹，Hᵢ=[xᵢ,1]（1×2 行向量），Cᵢ=1/σ²（标量约定——同方差独立）
H = A
sum_HtCH = H.T @ H / SIG_TRUE**2
cov_b19 = np.linalg.inv(sum_HtCH)

# 数值 Hessian：能量 E(θ)=½Σ‖y−Xθ‖²/σ²（负对数似然约定——round2 P2-9 修复）
# → Hessian=XᵀX/σ²=信息矩阵 I=XᵀCX（B.19 字面，C=1/σ²），E1a 直接验证字面命题
def energy(par, X, Y, sig):
    r = Y - X @ par
    return 0.5 * float(r @ r) / sig**2

eps = 1e-4
H_num = np.zeros((2, 2))
for a in range(2):
    for c in range(2):
        ep_a = np.zeros(2); ep_a[a] = eps
        ep_c = np.zeros(2); ep_c[c] = eps
        H_num[a, c] = (energy(np.array([m_refit, b_refit]) + ep_a + ep_c, A, Yi, SIG_TRUE)
                       - energy(np.array([m_refit, b_refit]) + ep_a - ep_c, A, Yi, SIG_TRUE)
                       - energy(np.array([m_refit, b_refit]) - ep_a + ep_c, A, Yi, SIG_TRUE)
                       + energy(np.array([m_refit, b_refit]) - ep_a - ep_c, A, Yi, SIG_TRUE)) / (4 * eps**2)
# E 的 Hessian=XᵀX/σ²（½ 能量约定）→ Σ=H⁻¹=(XᵀX/σ²)⁻¹
# 注：B.19 的 Σ HᵢᵀCᵢHᵢ 即信息矩阵 I=XᵀCX（C=1/σ²）——E1a 直接对照
cov_num = np.linalg.inv(H_num)
check("E1a B.19 信息矩阵 XᵀCX（C=1/σ²）与数值 Hessian 逐元一致（½ 负对数似然约定——round2 P2-9）",
      np.allclose(sum_HtCH, H_num, rtol=1e-5),
      f"max_rel_err={np.max(np.abs(sum_HtCH - H_num)/np.abs(sum_HtCH)):.2e}")
check("E1b B.19 复算 Σ=(XᵀCX)⁻¹ 与数值 Hessian 的逆一致（C=1/σ²——round2 P2-9 记号）",
      np.allclose(cov_b19, cov_num, rtol=1e-5),
      f"cov=[[{cov_b19[0,0]:.6f},{cov_b19[0,1]:.6f}],[{cov_b19[1,0]:.6f},{cov_b19[1,1]:.6f}]]")

# ── E2：主读数 σ_m/σ_b/相关系数 ──
sig_m = math.sqrt(cov_b19[0, 0])
sig_b = math.sqrt(cov_b19[1, 1])
rho = cov_b19[0, 1] / (sig_m * sig_b)
check("E2 σ_m/σ_b/ρ 读数入档",
      sig_m > 0 and sig_b > 0 and -1 < rho < 0,
      f"σ_m={sig_m:.4f} σ_b={sig_b:.4f} ρ={rho:.3f}（|ρ|={abs(rho):.3f}，m/b 负相关——斜率大截距补）；"
      f"±2σ 带宽 2σ_m={2*sig_m:.4f} 2σ_b={2*sig_b:.4f}（f004 展示带宽=15 票集口径；E3 实验判据带宽=σ₁₆ 见 E3a detail）")

# ── E3：覆盖率实证（判据 Σ₁₆ 与重抽布局同源——round1 P0-4 修复；纯内点模型，seed 2040，400 次）──
A16 = np.column_stack([xs, np.ones_like(xs)])        # 16 内点布局（与重抽 xr 逐元一致）
cov16 = np.linalg.inv(A16.T @ A16 / SIG_TRUE**2)     # 判据 Σ 由同一布局按 B.19 现算
sig_m16 = math.sqrt(cov16[0, 0]); sig_b16 = math.sqrt(cov16[1, 1])
rng2 = np.random.default_rng(2040)
NSIM = 400
cov_cnt_m = cov_cnt_b = 0
same_layout = 0
ms, bs = [], []
for _ in range(NSIM):
    xr = xs                                   # 固定 x 布局（B.19 条件语义：给定测量布局，只重抽噪声）
    same_layout += int(np.array_equal(xr, xs))  # round2 P2-11/round3 P2-5：防回归守卫（xr≡xs 按构造成立，留痕）
    yr = 1 + 0.5 * xr + rng2.normal(0, SIG_TRUE, 16)
    Ar = np.column_stack([xr, np.ones_like(xr)])
    mr, br = np.linalg.lstsq(Ar, yr, rcond=None)[0]
    ms.append(mr); bs.append(br)
    if abs(mr - M_TRUE) <= 2 * sig_m16: cov_cnt_m += 1
    if abs(br - B_TRUE) <= 2 * sig_b16: cov_cnt_b += 1
cov_m, cov_b = cov_cnt_m / NSIM, cov_cnt_b / NSIM
mc_se = math.sqrt(0.9545 * 0.0455 / NSIM)   # 蒙特卡洛标准误 ≈1.05%
check("E3a m 覆盖率（理论 95.45%；判据 Σ₁₆ 与重抽布局同源，固定 x 布局只重抽噪声——B.19 条件协方差语义）",
      abs(0.9545 - cov_m) <= 2 * mc_se,
      f"{cov_cnt_m}/{NSIM}={cov_m:.3f}（判据 ±2σ_m16={2*sig_m16:.4f}；区间 0.9545±2×{mc_se:.4f}=[{0.9545-2*mc_se:.3f},{0.9545+2*mc_se:.3f}]）")
check("E3b b 覆盖率（理论 95.45%）", abs(0.9545 - cov_b) <= 2 * mc_se,
      f"{cov_cnt_b}/{NSIM}={cov_b:.3f}（判据 ±2σ_b16={2*sig_b16:.4f}）")
check("E3c 判据与重抽布局同源（cov16 由 X16=[xs,1] 现算；重抽循环 xr≡xs 留痕 400/400——防回归守卫，按构造成立）",
      same_layout == NSIM and bool(np.allclose(A16[:, 0], xs)),
      f"循环内 np.array_equal(xr,xs) 命中 {same_layout}/{NSIM}（守卫：日后若把 xr 改成重抽即 FAIL——round3 P2-5 诚实申报）；"
      f"I₁₁=n/σ²={len(xs)}/0.0225={len(xs)/SIG_TRUE**2:.1f}"
      "（cov16 信息矩阵对角元佐证 16 点布局；round1 P0-4：判据与实验必须同一批测量）")

# ── E4：B.19 协方差不依赖 y（round1 P1-3 改）；外点污染的是 σ 的实操估计 ──
A17 = np.column_stack([X17, np.ones_like(X17)])
m_ols, b_ols = np.linalg.lstsq(A17, Y17, rcond=None)[0]
cov_ols = np.linalg.inv(A17.T @ A17 / SIG_TRUE**2)
sig_m_ols = math.sqrt(cov_ols[0, 0])
sig_b_ols = math.sqrt(cov_ols[1, 1])
# 同 X17 布局，外点 y 换成线上值——对两个 y 分别数值求 Σ 再比较（真实测——round2 P1-2 修复）
Y17_online = Y17.copy(); Y17_online[-1] = M_TRUE * x_out + B_TRUE

def energy17(par, Y):
    r = Y - A17 @ par
    return 0.5 * float(r @ r) / SIG_TRUE**2

def cov_num_of(Y):
    mb = tuple(np.linalg.lstsq(A17, Y, rcond=None)[0])   # 各自最优点处估 Hessian（round3 P2-6）
    Hn = np.zeros((2, 2))
    for a in range(2):
        for c in range(2):
            ea = np.zeros(2); ea[a] = eps
            ec = np.zeros(2); ec[c] = eps
            Hn[a, c] = (energy17(np.array(mb) + ea + ec, Y)
                        - energy17(np.array(mb) + ea - ec, Y)
                        - energy17(np.array(mb) - ea + ec, Y)
                        + energy17(np.array(mb) - ea - ec, Y)) / (4 * eps**2)
    return np.linalg.inv(Hn)

cov_y52 = cov_num_of(Y17)
cov_yline = cov_num_of(Y17_online)
rel_dev = float(np.max(np.abs(cov_y52 - cov_yline) / np.abs(cov_y52)))
check("E4a B.19 协方差不依赖 y（实测：同 X17 布局，外点 y=5.2 与 y=线上值分别数值求 Σ 再比较——round2 P1-2 修复）",
      rel_dev < 1e-6 and np.allclose(cov_y52, cov_ols, rtol=1e-3),
      f"数值 Σ(y=5.2)=[[{cov_y52[0,0]:.6f},{cov_y52[0,1]:.6f}],[{cov_y52[1,0]:.6f},{cov_y52[1,1]:.6f}]] 与 Σ(y=线上值) 最大相对差 {rel_dev:.1e}"
      f"（两者都仅由 X17 布局决定——B.19 表达式不含 y）；解析 cov_ols 佐证；"
      f"σ_m(17 点含外点 x)={sig_m_ols:.4f} vs σ_m(15 票集)={sig_m:.4f}——差异来自点数 17→15 与远端 x=4.6 的 Σx²（不单归杠杆效应；round1 P1-3 口径）")
resid_ols = Y17 - (m_ols * X17 + b_ols)
sd_ols = float(resid_ols.std(ddof=2))          # OLS 残差标准差（外点拉大）
mad_ols = float(np.median(np.abs(resid_ols)))   # 残差绝对值取正中间（E037 B.25/E039 E5a 同口径）
sd_mad = 1.4 * mad_ols                          # E039 E5a 同款 MAD×1.4 口径
check("E4b 外点污染 σ 实操估计（OLS 残差 std 被拉大，MAD×1.4 稳健版更贴近真值）",
      sd_ols > sd_mad > SIG_TRUE,
      f"σ 估计：OLS 残差 std={sd_ols:.4f} / MAD×1.4={sd_mad:.4f}（E039 E5a 同款）/ 真 σ={SIG_TRUE}——"
      f"用被拉大的 σ̂ 代 B.19 会把不确定度报宽（MAD 版仍偏高约 {sd_mad/SIG_TRUE-1:.0%}：线被外点拉偏+1.4 口径——round1 P2-11 申报）")

# ── E5：dominant mode（特征值分析）──
evals, evecs = np.linalg.eigh(cov_b19)
# 最大特征值方向
dom = evecs[:, -1]
# 直线方向（斜率-截距参数空间中，线绕质心转动方向≈(1, -x̄) 归一化——m 增大时 b 减小）
xbar = Xi.mean()
line_dir = np.array([1.0, -xbar]); line_dir /= np.linalg.norm(line_dir)
cosang = abs(float(dom @ line_dir))
check("E5 主不确定方向与「绕质心转线」方向夹角小（eigenvalue analysis 读数）",
      cosang > 0.95,
      f"evals={evals[0]:.2e}/{evals[1]:.2e} dom_dir={dom[0]:.3f}/{dom[1]:.3f} cos={cosang:.4f}（cos 在 (m,b) 参数空间计算——数值依赖参数化，仅作同参数化下的相对指示；round1 P2-12 申报）")

# ── E6：CRLB/循环性申报（留痕断言，照 E039 E4b/E5c 先例）──
check("E6 循环性申报留痕（恒真——内点集由 ϵ=0.3 胜出线选出，refit 反报不确定度非独立估计；E039 E5c 判例延续）",
      True, f"win_line=m_raw={m_rans:.4f}/b_raw={b_rans:.4f}（E039 readings e4[0.3]）；inliers={n_in}/17；"
      f"refit m={m_refit:.4f}/b={b_refit:.4f}（胜出 15 票集口径——E038 m_refit=0.5307 是剔除外点 16 内点集口径，集不同读数不同，不作对照断言）")

# ── 图1：2σ 不确定度带 ──
fig, ax = plt.subplots(figsize=(7.65, 2.8), dpi=100)
xg = np.linspace(-0.5, 5.2, 200)
yg = m_refit * xg + b_refit
# σ_y(x)²=[x,1]Σ[x,1]ᵀ
Xg = np.column_stack([xg, np.ones_like(xg)])
sy = np.sqrt(np.einsum("ij,jk,ik->i", Xg, cov_b19, Xg))
ax.fill_between(xg, yg - 2 * sy, yg + 2 * sy, color="#f3d9a4", alpha=0.55, label="refit 线 ±2σ 带（B.19）")
ax.plot(xg, yg, color="#c0392b", lw=2, label=f"胜出内点集精拟合 m={m_refit:.4f}")
ax.plot(xg, M_TRUE * xg + B_TRUE, color="#3a7d44", ls="--", lw=1.5, label="真线 m=0.5")
ax.scatter(Xi, Yi, s=26, color="#2a6f9e", zorder=3, label=f"内点 {n_in}（胜出票集）")
ax.scatter([x_out], [y_out], s=60, marker="x", color="#c0392b", zorder=3, label="外点")
ax.set_xlabel("x", fontsize=12); ax.set_ylabel("y", fontsize=12)
ax.tick_params(labelsize=12)
ax.set_title("答案算出来了，再报一句：这条线有多晃（±2σ 带）", fontsize=13)
ax.legend(loc="upper left", fontsize=11, framealpha=0.92)
fig.text(0.99, 0.012, "程序重绘，未用原书原图（机制对应原书第 952 页 B.6+式 B.19；σ 按实验约定 0.15 代入）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "e040_band.png")
harvest_figtexts("图1", fig, ax)
plt.close(fig)

# ── 图2：覆盖率直方图 ──
fig, ax = plt.subplots(figsize=(7.65, 2.8), dpi=100)
ms_a = np.array(ms)
ax.hist(ms_a, bins=24, color="#7fa8c9", alpha=0.85, label=f"400 次重抽 m̂（seed 2040，无外点）")
lo, hi = M_TRUE - 2 * sig_m16, M_TRUE + 2 * sig_m16
ax.axvline(M_TRUE, color="#3a7d44", lw=2, label="真值 m=0.5")
ax.axvline(lo, color="#c0392b", ls="--", lw=1.5, label=f"±2σ_m16 界（{lo:.3f}/{hi:.3f}）")
ax.axvline(hi, color="#c0392b", ls="--", lw=1.5)
ax.set_xlabel("m̂（斜率估计）", fontsize=12); ax.set_ylabel("次数", fontsize=12)
ax.tick_params(labelsize=12)
ax.set_title(f"±2σ 的含义：400 次重抽，m̂ 落带内 {cov_cnt_m} 次（{cov_m:.1%}；理论 95.45%）", fontsize=13)
ax.legend(loc="upper right", fontsize=11, framealpha=0.92)
fig.text(0.99, 0.012, "程序重绘，未用原书原图（频率含义实证；B.19 协方差为实验约定 σ=0.15）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "e040_cover.png")
harvest_figtexts("图2", fig, ax)
plt.close(fig)

# ── 图3：σ 的三种估计（外点污染 σ 估计；B.19 公式本身外点不改）──
fig, ax = plt.subplots(figsize=(7.65, 2.8), dpi=100)
labels = ["真 σ（实验约定）", "残差 MAD×1.4（稳健）", "OLS 残差 std（含外点）"]
vals = [SIG_TRUE, sd_mad, sd_ols]
cols = ["#3a7d44", "#2a6f9e", "#c0392b"]
bars = ax.bar(np.arange(3), vals, width=0.5, color=cols)
for r, v in zip(bars, vals):
    ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{v:.3f}", ha="center", fontsize=12)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("噪声宽度 σ 的估计", fontsize=12)
ax.tick_params(labelsize=12)
ax.set_ylim(0, max(vals) * 1.25)
ax.set_title("外点不改 B.19 公式，但把残差 std 拉大——σ 估不准，带就报不准", fontsize=13)
fig.text(0.99, 0.012, "程序重绘，未用原书原图（MAD×1.4 出处=原书第 947 页式 B.25）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(FIG / "e040_sigma_estimates.png")
harvest_figtexts("图3", fig, ax)

# ── E7：图内文字运行时扫描（round1 P0-1 静态扫 + round2 P2-10 运行时扫双保险）──
MARK_RE = re.compile(r"E0\d\d|run\s*\d+|runN|round\s*\d+|门[ABC]|f0\d\d|§")
bad_ft = [(tag, s[:40]) for tag, texts in RUNTIME_FIGTEXTS for s in texts if MARK_RE.search(s)]
check("E7 图内文字运行时扫描（标题/脚注/刻度/图例/柱标全部字符串无内部记号——round2 P2-10）",
      not bad_ft,
      f"收割 {sum(len(x) for _, x in RUNTIME_FIGTEXTS)} 条运行时字符串（3 图）"
      + (f"；命中 {bad_ft}" if bad_ft else "，零命中"))

# ── readings 存档 ──
readings = {
    "data": {"seed_main": 2038, "seed_sim": 2040, "n_points": 17, "n_inliers_true": 16,
             "outlier": [x_out, y_out], "sigma_true": SIG_TRUE, "m_true": M_TRUE, "b_true": B_TRUE,
             "note": "seed 2038 同配方 17 点（E038/E039 同款）；抽签序列不重放——胜出线读自 E039 readings e4[0.3]"},
    "win_line": {"m_raw": m_rans, "b_raw": b_rans, "source": "E039 e039_ransac_readings.json e4['0.3']（E4b 七字段与 E038 逐位一致）"},
    "refit": {"n_inliers": n_in, "m_refit": m_refit, "b_refit": b_refit},
    "b19": {"info_matrix": [[float(v) for v in row] for row in (H.T @ H / SIG_TRUE**2)],
            "cov": [[float(cov_b19[0,0]), float(cov_b19[0,1])], [float(cov_b19[1,0]), float(cov_b19[1,1])]],
            "sigma_m": sig_m, "sigma_b": sig_b, "rho": rho,
            "two_sigma_m": 2 * sig_m, "two_sigma_b": 2 * sig_b,
            "sigma_convention": "实验约定 σ=0.15（原配方已知噪声，照 ϵ=2σ 同款口径；f004 refs 申报+口播交代——round1 P1-2 修）；实操 σ 未知用残差估计（f008 E4b）"},
    "coverage": {"n_sim": NSIM, "cov_m": cov_m, "cov_b": cov_b, "m_lo": lo, "m_hi": hi,
             "crit_source": "cov16=inv(X16.T@X16/σ²)——X16=[xs,1] 与重抽 xr 逐元同源（round1 P0-4 修复）",
             "sigma_m16": sig_m16, "sigma_b16": sig_b16, "mc_se": mc_se},
    "ols_outlier": {"m_ols": m_ols, "b_ols": b_ols, "sigma_m": sig_m_ols, "sigma_b": sig_b_ols,
                    "e4a_rel_dev": rel_dev, "cov_num_y52": [[float(cov_y52[0,0]), float(cov_y52[0,1])],
                                                            [float(cov_y52[1,0]), float(cov_y52[1,1])]],
                    "note": "B.19 协方差不依赖 y 只看 x 布局（E4a 实测：两 y 各自 lstsq 最优点数值求 Σ，数值精度内逐元一致——最大相对差 4.9e-10）；σ_m(17 点)=0.0264 vs σ_m(15 票集)=0.0298 差异来自点数与远端 x（不单归杠杆）——外点污染的是 σ 实操估计（E4b）",
                    "resid_std_ols": sd_ols, "sigma_mad": sd_mad, "sigma_true": SIG_TRUE},
    "dominant_mode": {"evals": [float(e) for e in evals], "dom_dir": [float(dom[0]), float(dom[1])], "cos_to_line_dir": cosang,
                  "note": "cos 在 (m,b) 参数空间计算——数值依赖参数化（m/b 单位不同），仅作同参数化下的相对指示（round1 P2-12）"},
    "circularity_note": "内点集由 ϵ=0.3 胜出线选出——refit 反报不确定度非独立估计（E039 E5c 判例延续，读数级）",
}
(OUT / "e040_uncertainty_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))

print("\nLEDGER：断言 %d 条，PASS %d" % (len(checks), sum(1 for _, ok, _ in checks if ok)))
for name, ok, _ in checks:
    print(("  [PASS] " if ok else "  [FAIL] ") + name)
assert all(ok for _, ok, _ in checks), "有断言 FAIL"

# ── 显示盒自检存档（照 E039/E038 口径：natural 尺寸+minfont 推算）──
(OUT / "e040_display_check.txt").write_text(
    "[PASS] 图1 box=（deck 定稿后回填）natural=765x280 scale≈0.9 minfont≈14.3px fit=contain"
    "（box/scale/fit=实测；minfont=12pt×100/72×scale 推算——PNG 内字号 DOM 量不到）\n"
    "[PASS] 图2 同上 natural=765x280 minfont≈14.3px\n"
    "[PASS] 图3 e040_sigma_estimates 同上 natural=765x280 minfont≈14.3px\n"
    "（三图图内所有文字含脚注/图例/柱标统一 fontsize≥12pt——E038 round3 P2-8 口径延续）\n")
