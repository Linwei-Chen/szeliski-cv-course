#!/usr/bin/env python3
"""E010 实验 · 应用大盘点：把「工程方法论」跑成代码（讲者设计实验）

原书 p.8 说：做视觉要先有问题定义，再用两类测试数据检验——
合成数据（验证正确性、分析噪声敏感性）+ 真实数据（贴近最终使用）。
本实验（讲者设计，参数为演示设定）把这句话兑现成一个最小「机器质检台」：

演示一（对应书中 machine inspection，印刷 p.5）：
  合成一个零件边缘剖面（暗背景 0 → 亮零件 1，线性斜坡过渡，总宽 2px），
  用 50% 交叉点线性内插做亚像素边缘定位，对噪声水平 σ 做 Monte-Carlo 扫描：
  - σ=0 时应对任意真值位置零偏、零误差（验证正确性）
  - 误差随 σ 单调增大（噪声敏感性）
  - 给定公差 t=0.25px，误判率随 σ 上升

演示二（对应 p.10 机器人测距例子）：
  真实距离 d ~ N(3.0, 0.5)。损失不对称：低估（把障碍估近）每米代价 1，
  高估（把障碍估远）每米代价 4。最优估计 = 距离分布的 q=1/(1+4)=0.2 分位数，
  故意低于均值——「宁可低估，不要高估」的数学形态。
  同时验证对调配置 4:1（最优 = 0.8 分位数，均值上方）与对称损失 1:1（最优 = 均值）。

输出：outputs/inspection_demo.png（三面板）+ 全程 [verify] 行 + 汇总。
"""

import os
import sys
from math import erf, exp, pi, sqrt
from statistics import NormalDist

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

# 中文字体（macOS 优先 PingFang SC，找不到就退化为系统默认）
for f in ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]:
    if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# 自检账本
CHECKS = {"pass": 0, "fail": 0}


def verify(ok, msg):
    CHECKS["pass" if ok else "fail"] += 1
    print(f"[verify] {msg} [{'PASS' if ok else 'FAIL'}]")


# ----------------------------------------------------------------------
# 共用：边缘剖面模型
# ----------------------------------------------------------------------
def edge_profile(x, x0=40.0, w=2.0):
    """1D 零件边缘：x<x0-w/2 背景 0，x>x0+w/2 零件 1，线性斜坡（ramp）过渡，
    过渡区总宽 w。50% 交叉点恰好落在 x0——亚像素定位的前提。"""
    return np.clip((x - x0) / w + 0.5, 0.0, 1.0)


def estimate_edge(y, x, x0_nominal=40.0, halfwin=8):
    """50% 交叉点亚像素内插：在名义位置附近取窗，找亮度越过
    （窗内最大+最小）/2 的上升沿交叉点，线性内插出亚像素位置。
    线性斜坡 + 零噪声 ⇒ 对任意真值位置精确恢复；
    噪声下交叉点抖动 std ≈ σ/局部斜率——正是要演示的噪声敏感性。"""
    lo, hi = int(x0_nominal - halfwin), int(x0_nominal + halfwin)
    xs, ys = x[lo:hi + 1], y[lo:hi + 1]
    mid = 0.5 * (ys.max() + ys.min())
    above = ys >= mid
    idx = int(np.argmax(above))
    if idx == 0:
        return float(xs[0])
    y1, y0 = ys[idx], ys[idx - 1]
    if y1 == y0:
        return float(xs[idx])
    t = (mid - y0) / (y1 - y0)
    return float(xs[idx - 1] + t * (xs[idx] - xs[idx - 1]))


# ----------------------------------------------------------------------
# 演示一：合成质检台 —— 亚像素边缘定位 vs 噪声
# ----------------------------------------------------------------------
rng = np.random.default_rng(20261025)

N = 128
x = np.arange(N, dtype=float)
X0_TRUE = 40.0          # 真实边缘位置（像素）
TOL = 0.25              # 质检公差（像素）：|估计-真值|>0.25 判不合格
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3]
M_TRIALS = 2000         # 每个 σ 的 Monte-Carlo 次数

base = edge_profile(x, x0=X0_TRUE)

# 零噪声理想情况：整数与非整数真值都应精确无偏
for x0_check in (40.0, 40.3):
    est0 = estimate_edge(edge_profile(x, x0=x0_check), x, x0_nominal=x0_check)
    verify(abs(est0 - x0_check) < 1e-9, f"σ=0、真值 {x0_check}px 时定位精确无偏（差 {abs(est0-x0_check):.2e}px）")

err_std, err_bias, fail_rate = [], [], []
for s in SIGMAS:
    est = np.empty(M_TRIALS)
    for m in range(M_TRIALS):
        y = base + (rng.normal(0.0, s, N) if s > 0 else 0.0)
        est[m] = estimate_edge(y, x)
    e = est - X0_TRUE
    err_std.append(float(np.std(e)))
    err_bias.append(float(np.mean(e)))
    fail_rate.append(float(np.mean(np.abs(e) > TOL)))

print(f"[demo1] 亚像素边缘定位（真值 {X0_TRUE}px，公差 ±{TOL}px，每档 {M_TRIALS} 次 MC）")
for s, sd, bi, fr in zip(SIGMAS, err_std, err_bias, fail_rate):
    print(f"  σ={s:<4} 误差std={sd:.4f}px  偏置={bi:+.4f}px  误判率={fr*100:5.2f}%")

verify(err_bias[0] < 1e-9 and err_std[0] < 1e-9, "σ=0 时偏置与误差均为零（合成数据验证正确性）")
mono = all(err_std[i + 1] > err_std[i] for i in range(len(SIGMAS) - 1))
verify(mono, "误差 std 随 σ 逐档单调上升（噪声敏感性，5 档）")
verify(fail_rate[0] == 0.0, "σ=0 误判率为 0（好零件零误杀）")
verify(all(fail_rate[i + 1] > fail_rate[i] for i in range(len(SIGMAS) - 1)),
       "误判率随 σ 逐档上升（" + " → ".join(f"{f*100:.1f}%" for f in fail_rate) + "）")
verify(err_std[-1] > 10 * err_std[1], "σ=0.3 的误差 std 是 σ=0.05 的 10 倍以上（噪声放大肉眼可见）")

# ----------------------------------------------------------------------
# 演示二：不对称损失 —— 「宁可低估，不要高估」的数学形态
#   最优估计 = 距离分布的 q = c_under/(c_under+c_over) 分位数（解析可证）
# ----------------------------------------------------------------------
D_MEAN, D_STD = 3.0, 0.5          # 真实距离分布 N(3.0, 0.5) 米
GRID = np.linspace(1.0, 5.0, 4001)


def phi(z):
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / sqrt(2 * pi)


def Phi(z):
    return np.array([0.5 * (1 + erf(zz / sqrt(2))) for zz in z])


def expected_loss(chat, c_under, c_over):
    """E[L(chat, d)]，d ~ N(D_MEAN, D_STD)：
    低估（chat<d）每米罚 c_under，高估（chat>d）每米罚 c_over。
    解析式：E[(X)^+] = m·Φ(m/σ) + σ·φ(m/σ)，X ~ N(m, σ)。"""
    zz = (np.asarray(chat, dtype=float) - D_MEAN) / D_STD
    under = (D_MEAN - chat) * Phi(-zz) + D_STD * phi(zz)   # E[(d-chat)^+]
    over = (chat - D_MEAN) * Phi(zz) + D_STD * phi(zz)     # E[(chat-d)^+]
    return c_under * under + c_over * over


def optimal_estimate(c_under, c_over):
    """网格最优估计 vs 解析分位数 q = c_under/(c_under+c_over)。"""
    q = c_under / (c_under + c_over)
    d_analytic = float(NormalDist(D_MEAN, D_STD).inv_cdf(q))
    el = expected_loss(GRID, c_under, c_over)
    d_star = float(GRID[int(np.argmin(el))])
    return d_star, d_analytic, el


print(f"[demo2] 不对称损失（d ~ N({D_MEAN}, {D_STD}²)，网格 {len(GRID)} 点）")

d_a, an_a, el_asym = optimal_estimate(1.0, 4.0)   # 低估:高估 = 1:4（讲者设定）
print(f"  1:4（低估:高估）最优 = {d_a:.3f} m（解析 {an_a:.3f} m = 0.2 分位数）"
      f"，E[L] = {el_asym.min():.4f}，若用均值则 E[L] = {float(expected_loss(np.array([D_MEAN]), 1.0, 4.0)[0]):.4f}")
d_b, an_b, _ = optimal_estimate(4.0, 1.0)         # 对调：4:1
print(f"  4:1（对调）      最优 = {d_b:.3f} m（解析 {an_b:.3f} m = 0.8 分位数）")
d_s, an_s, el_sym = optimal_estimate(1.0, 1.0)    # 对称
print(f"  1:1（对称）      最优 = {d_s:.3f} m（解析 {an_s:.3f} m = 均值）")

verify(abs(d_a - an_a) < 0.02, f"1:4 最优估计 = 0.2 分位数（网格 {d_a:.3f} vs 解析 {an_a:.3f}，差 < 0.02 m）")
verify(d_a < D_MEAN, f"1:4 最优估计 {d_a:.2f} m 刻意低于均值 3.00 m（安全侧偏置）")
verify(float(expected_loss(np.array([d_a]), 1.0, 4.0)[0]) < float(expected_loss(np.array([D_MEAN]), 1.0, 4.0)[0]),
       "1:4 按分位数出招的期望风险严格低于按均值出招")
verify(abs(d_b - an_b) < 0.02, f"对调 4:1 后最优估计 = 0.8 分位数（网格 {d_b:.3f} vs 解析 {an_b:.3f}）——作业预期")
verify(d_b > D_MEAN, f"对调 4:1 后最优估计 {d_b:.2f} m 移到均值上方（偏向另一侧，作业预期）")
verify(abs(d_s - an_s) < 0.02, f"对称损失 1:1 最优估计回到均值本身（网格 {d_s:.3f} vs 解析 {an_s:.3f}）")

# ----------------------------------------------------------------------
# 三面板图（标签全部由参数生成，改参数不再失配）
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
INK, ORANGE, SAND, GRAY = "#1e3a5f", "#e8590c", "#c9a227", "#8a8377"

# 面板1：含噪边缘剖面 + 定位结果
ax = axes[0]
ax.plot(x, base, color=INK, lw=2.5, label=f"干净边缘（真值 {X0_TRUE}px）")
yn = base + rng.normal(0.0, 0.3, N)
ax.plot(x, yn, color=GRAY, lw=1.0, alpha=0.8, label="σ=0.3 的一次仿真")
est_n = estimate_edge(yn, x)
ax.axvline(X0_TRUE, color=INK, ls="--", lw=1.4)
ax.axvline(est_n, color=ORANGE, ls="-", lw=1.8)
ax.plot([est_n], [0.5], marker="o", color=ORANGE, ms=9, zorder=5,
        label=f"50% 交叉估计 {est_n:.2f}px")
ax.set_title("① 合成质检台：一次含噪仿真", fontsize=14, color=INK)
ax.set_xlabel("像素位置")
ax.set_ylabel("亮度")
ax.set_ylim(-0.35, 1.45)
ax.legend(fontsize=9, loc="upper left")
ax.text(0.02, -0.24, f"线性斜坡过渡，总宽 2px ｜ 真值 {X0_TRUE:.1f}px", fontsize=9,
        color=GRAY, transform=ax.transAxes)

# 面板2：误差 std 与误判率 vs σ
ax = axes[1]
ax.semilogy(SIGMAS[1:], err_std[1:], marker="o", color=INK, lw=2.2, label="误差 std（px）")
ax.axhline(TOL, color=ORANGE, ls="--", lw=1.6, label=f"质检公差 ±{TOL}px")
ax.set_title("② 噪声敏感性：误差随 σ 单调上升", fontsize=14, color=INK)
ax.set_xlabel("噪声 σ")
ax.set_ylabel("定位误差（px，对数轴）")
ax2 = ax.twinx()
ax2.plot(SIGMAS[1:], [f * 100 for f in fail_rate[1:]], marker="s", color=SAND, lw=2.0,
         label="误判率（%）")
ax2.set_ylabel("误判率（%）", color="#8a6d1b")
ax2.tick_params(axis="y", labelcolor="#8a6d1b")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
ax.text(0.02, -0.24, f"每档 {M_TRIALS} 次 Monte-Carlo；σ=0 档误差恒为 0（对数轴省略）",
        fontsize=9, color=GRAY, transform=ax.transAxes)

# 面板3：不对称损失的期望风险曲线（1:4 讲者设定）
ax = axes[2]
ax.plot(GRID, el_asym, color=ORANGE, lw=2.4, label=f"损失 {1.0:.0f}:{4.0:.0f}（低估:高估，讲者设定）")
ax.plot(GRID, el_sym, color=INK, lw=2.0, ls="--", label="对称损失 1:1")
ax.axvline(d_a, color=ORANGE, ls=":", lw=1.8)
ax.axvline(D_MEAN, color=GRAY, ls=":", lw=1.4)
q_a = 1.0 / (1.0 + 4.0)   # 1:4 配置的分位数（讲者设定）
ax.annotate(f"最优 {d_a:.2f} m\n({q_a:.1f} 分位数)",
            xy=(d_a, el_asym.min()), xytext=(d_a - 0.72, el_asym.min() + 0.75),
            fontsize=10, color=ORANGE)
ax.annotate(f"均值 {D_MEAN:.2f} m\n(对称最优)", xy=(D_MEAN, float(el_sym[int(np.argmin(el_sym))])),
            xytext=(D_MEAN + 0.55, el_sym.min() + 0.28), fontsize=10, color=INK)
ax.set_title("③ 宁可低估：不对称损失的最优解", fontsize=14, color=INK)
ax.set_xlabel("机器人报出的距离（m）")
ax.set_ylabel("期望损失 E[L]")
ax.legend(fontsize=10, loc="upper right")

fig.suptitle("E010 实验 · 讲者设计：把「合成测试数据 + 不对称风险」跑成代码（演示一对应第 8 页，演示二对应第 10 页）",
             fontsize=16, color=INK, y=1.02)
fig.tight_layout()
out_png = os.path.join(OUT, "inspection_demo.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"[demo] 图已存 {out_png}")

total = CHECKS["pass"] + CHECKS["fail"]
print(f"汇总：{CHECKS['pass']}/{total} PASS" + (" —— 全部通过" if CHECKS["fail"] == 0 else " —— 存在失败！"))
sys.exit(0 if CHECKS["fail"] == 0 else 1)
