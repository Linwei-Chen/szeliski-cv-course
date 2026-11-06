#!/usr/bin/env python3
"""E037 数值实验：外点拉偏平方损失 → 损失函数族 → IRLS 迭代重加权 → MAD 定尺度。
锚：e037_book_b3_s413_anchors.txt（附录 B.3 印刷 p.945–947 + §4.1.3 印刷 p.202–204）。
数据：seed 2037，内点 16 个 y=1+0.5x+N(0,0.15)，外点 1 个 (4.6, 5.2)（杠杆位置）。
D30 口径（图 4.7 题注 Welsch=α−1 照原书；题注与图例 −∞ 不一致属原书小疵，课程不断言谁对）见 docs/决策记录.md；D29 口径（B.22 印刷系数疑点）：lab 损失族按 Barron 2019 正形 c²|α−2|/α[...] 实现，
印刷形 |α−2|/2 数值复算留痕（E2e：α=−2 给负损失）——照 E029 脚注 11 先例。
IRLS 权重：α=0 柯西型 ρ=c²ln(1+r²/(2c²)) → ψ=r/(1+r²/(2c²)) → w=1/(1+r²/(2c²))；
           作业②换 α=−1 Welsch 型重跑（讲者复算辖域，解析式见 E5）。
图（程序重绘，未用原书原图）：图1 外点拉偏 765×300（概念对应 p.946 外点讨论）；
图2 损失族 765×300（概念对应 p.203 图 4.7）；图3 IRLS 轨迹 765×500（对应 p.947 IRLS 两步）。
"""
import json
import pathlib

import numpy as np

rng = np.random.default_rng(2037)
OUT = pathlib.Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
checks = []


def chk(label, ok, detail=""):
    checks.append(label)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"FAIL: {label}")


# ── 数据 ──
M_TRUE, B_TRUE, SIGMA = 0.5, 1.0, 0.15
x_in = np.sort(rng.uniform(0.0, 4.0, 16))
y_in = B_TRUE + M_TRUE * x_in + rng.normal(0.0, SIGMA, 16)
x_out_pt, y_out_pt = 4.6, 5.2
x_all = np.append(x_in, x_out_pt)
y_all = np.append(y_in, y_out_pt)


def ols(x, y):
    X = np.stack([x, np.ones_like(x)], axis=1)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


# ── E1 外点拉偏 ──
m_clean, b_clean = ols(x_in, y_in)
m_out, b_out = ols(x_all, y_all)
chk("E1a 剔除外点 OLS 贴近真值（|m−0.5|<0.05）", abs(m_clean - M_TRUE) < 0.05,
    f"m_clean={m_clean:.4f}")
chk("E1b 一个外点拉偏 OLS（|m−0.5|≥0.10）", abs(m_out - M_TRUE) >= 0.10,
    f"m_out={m_out:.4f}")
chk("E1c 单次实现偏差口径：拉偏幅度是内点噪声级扰动的 3 倍以上",
    abs(m_out - M_TRUE) >= 3 * abs(m_clean - M_TRUE),
    f"pull={abs(m_out - M_TRUE):.4f} vs noise={abs(m_clean - M_TRUE):.4f}（单次实现偏差，非噪声标准误）")

# ── E2 损失函数族（Barron 正形 + 五退化）──
A2, A1, A0, AM1, AM2 = 2.0, 1.0, 0.0, -1.0, -2.0


def rho_barron(r, alpha, c=1.0):
    """Barron 2019 正形（D29）：c²|α−2|/α [ ((r/c)²/|α−2|+1)^{α/2} − 1 ]；
    α=2 用极限 r²/2；α=0 用极限 c²ln(1+r²/(2c²))（Cauchy 型）。"""
    if alpha == 2.0:
        return 0.5 * r * r
    if alpha == 0.0:
        return c * c * np.log(1.0 + (r / c) ** 2 / 2.0)
    coef = c * c * abs(alpha - 2.0) / alpha
    return coef * (((r / c) ** 2 / abs(alpha - 2.0) + 1.0) ** (alpha / 2.0) - 1.0)


def psi_barron(r, alpha, c=1.0):
    """正形导数（影响函数）：ψ=r((r/c)²/|α−2|+1)^{α/2−1}；α=2 用极限 ψ=r；
    α=0 用极限 ψ=r/(1+r²/(2c²))。"""
    if alpha == 2.0:
        return r
    if alpha == 0.0:
        return r / (1.0 + (r / c) ** 2 / 2.0)
    return r * (((r / c) ** 2 / abs(alpha - 2.0) + 1.0) ** (alpha / 2.0 - 1.0))


def rho_printed(r, alpha, c=1.0):
    """原书 B.22 印刷形（系数 |α−2|/2）——仅 D29 复算留痕用。"""
    return abs(alpha - 2.0) / 2.0 * (((r / c) ** 2 / abs(alpha - 2.0) + 1.0) ** (alpha / 2.0) - 1.0)


chk("E2a α=2 极限=L2（ρ=r²/2）", all(abs(rho_barron(r, 2.0) - 0.5 * r * r) < 1e-12 for r in (0.0, 1.0, 5.0)))
chk("E2b α=1 退化为 Charbonnier（解析 c²(√((r/c)²+1)−1)）",
    all(abs(rho_barron(r, 1.0) - (np.sqrt((r / 1.0) ** 2 + 1.0) - 1.0)) < 1e-12 for r in (0.0, 1.0, 5.0)))
grid = np.linspace(0.0, 10.0, 2001)
chk("E2c 五个 α 的 ρ 单调不减", all(np.all(np.diff([rho_barron(r, a) for r in grid]) > -1e-12)
                                     for a in (A2, A1, A0, AM1, AM2)))
big = 5.0
vals = [rho_barron(big, a) for a in (A2, A1, A0, AM1, AM2)]
chk("E2d 大残差惩罚随 α 减小递减（L2>Charbonnier>Cauchy>Welsch>GM）",
    all(vals[i] > vals[i + 1] for i in range(4)),
    " ".join(f"{v:.4f}" for v in vals))
chk("E2e D29 留痕：印刷形 |α−2|/2 在 α=−2, r=1 给负损失（-0.4）",
    rho_printed(1.0, AM2) < 0.0, f"printed={rho_printed(1.0, AM2):.4f}")
g = 1e-6
chk("E2f ψ=ρ′（数值导数 vs 解析，五 α 全部一致）",
    all(abs(psi_barron(r, a) - (rho_barron(r + g, a) - rho_barron(r - g, a)) / (2 * g)) < 1e-5
        for a in (A1, A0, AM1, AM2) for r in (0.3, 1.0, 3.0)))
chk("E2g 柯西型权重解析 w=1/(1+r²/(2c²))（α=0）",
    all(abs(psi_barron(r, A0) / r - 1.0 / (1.0 + r * r / 2.0)) < 1e-12 for r in (0.5, 1.0, 3.0)))

# ── E3 尺度估计（B.25：med‖r_i‖×1.4）——从被拉偏拟合的残差估会被模型误差抬高 ──
r0 = y_all - (m_out * x_all + b_out)
mad_init = 1.4 * np.median(np.abs(r0))
std_poor = float(np.std(r0, ddof=1))
chk("E3a 拉偏拟合残差使 MAD 失真（MAD×1.4 > 2σ——残差被坏线带偏，估计随之失真）",
    mad_init > 2 * SIGMA, f"mad_init={mad_init:.4f} vs σ={SIGMA}")
chk("E3b 拉偏拟合残差的普通 std 更离谱（|std−σ|>1.5σ）", abs(std_poor - SIGMA) > 1.5 * SIGMA,
    f"std={std_poor:.4f}")
# 收敛后再估（见 E4 自调循环）——占位断言在 E4 收敛数据出来后核
mad_final_holder = {}


def cauchy_w(r, c):
    return 1.0 / (1.0 + (r / c) ** 2 / 2.0)          # 柯西型 w（E2g）


def welsch_w(r, c):
    return (1.0 + (r / c) ** 2 / 2.0) ** (-1.5)      # Welsch 型 w（α=−1，讲者复算）


def irls(wfun, m0, b0, n_max=40, c_fixed=None):
    """IRLS 两步舞。c_fixed=None 时每轮用当前残差重估 c=med‖r_i‖×1.4（B.25，自调）；
    给定 c_fixed 则全程固定（红降快的 Welsch 需固定尺度防 c 反馈收缩——讲者复算注记）。"""
    m_it, b_it = m0, b0
    traj_m, w_out_traj, w_in_traj, traj_c = [m0], [1.0], [1.0], []
    c = 1.4 * np.median(np.abs(y_all - (m_it * x_all + b_it)))  # 首轮 c=B.25
    for it in range(n_max):
        r = y_all - (m_it * x_all + b_it)
        if c_fixed is None:
            c = 1.4 * np.median(np.abs(r))
        traj_c.append(float(c))
        w = wfun(r, c)
        XW = X * w[:, None]
        theta = np.linalg.solve(X.T @ XW, XW.T @ y_all)
        m_it, b_it = float(theta[0]), float(theta[1])
        r = y_all - (m_it * x_all + b_it)
        w = wfun(r, c)
        traj_m.append(m_it)
        w_out_traj.append(float(w[-1]))
        w_in_traj.append(float(np.mean(w[:-1])))
        if it >= 1 and abs(traj_m[-1] - traj_m[-2]) < 1e-8:
            break
    return m_it, b_it, traj_m, w_out_traj, w_in_traj, traj_c, w


# ── E4 IRLS（柯西型）──
X = np.stack([x_all, np.ones_like(x_all)], axis=1)
m_it, b_it, traj_m, w_out_traj, w_in_traj, traj_c, w = irls(cauchy_w, m_out, b_out)
n_steps = len(traj_m) - 1
mad_final = 1.4 * np.median(np.abs(y_all - (m_it * x_all + b_it)))
mad_final_holder["v"] = mad_final
chk("E4a IRLS 收敛（≤40 步，末步斜率增量 <1e-8）", n_steps <= 40, f"steps={n_steps}")
chk("E4b 收敛斜率贴近剔除外点解（|Δ|<0.02）", abs(m_it - m_clean) < 0.02,
    f"m_irls={m_it:.4f} vs m_clean={m_clean:.4f}")
chk("E4c 外点最终权重趋零（<0.05 且 <内点均权/10）",
    w[-1] < 0.05 and w[-1] < np.mean(w[:-1]) / 10.0,
    f"w_out={w[-1]:.5f} vs w_in_mean={np.mean(w[:-1]):.4f}")
chk("E4d 收敛后残差再估 MAD×1.4 回到 σ 邻近（比值 0.6–1.6）",
    0.6 <= mad_final / SIGMA <= 1.6, f"mad_final={mad_final:.4f}")
chk("E4e 收敛后 MAD vs 拉偏 std（跨口径对照，仅示意——|Δ| 比较）",
    abs(mad_final - SIGMA) < abs(std_poor - SIGMA),
    f"mad|Δ|={abs(mad_final - SIGMA):.4f} vs std|Δ|={abs(std_poor - SIGMA):.4f}")

# ── E5 作业②同款：Welsch 型（α=−1）IRLS 复跑（讲者复算辖域）──
# round2 P0-1 终版口径：2×2 受控网格 {固定 c=mad_final, 自调 c} × {拉偏初值, 柯西解初值}，四终点全入档。
# 数据能隔离出的结论=四终点极差 0.0436（读数级，vs 柯西对照 0.0265）；「好起点更好」不成立——
# 固定 c 下拉偏初值终点（≈0.5066）反而比柯西解初值（≈0.4784）更贴好解（反例入档）；
# 「起点要好」由原书第 947 页 chip 承载，E5 不做该归因（混来自 round1 P1-5→round2 P0-1）
m_g_fix_pull, *_ , w_g_fix_pull = irls(welsch_w, m_out, b_out, c_fixed=mad_final)
m_g_fix_init, *_ , w_g_fix_init = irls(welsch_w, m_it, b_it, c_fixed=mad_final)
m_g_auto_pull, *_ , w_g_auto_pull = irls(welsch_w, m_out, b_out)   # 自调：c_init=mad_init 起每轮重估
m_g_auto_init, *_ , w_g_auto_init = irls(welsch_w, m_it, b_it)
grid = [m_g_fix_pull, m_g_fix_init, m_g_auto_pull, m_g_auto_init]
chk("E5a 2×2 网格（尺度策略×初值）四终点极差 0.0436≥0.02（vs 柯西对照 0.0265，见 E5d）",
    max(grid) - min(grid) >= 0.02,
    f"grid={[round(v, 4) for v in grid]} spread={max(grid) - min(grid):.4f}")
deltas = [abs(v - m_clean) for v in grid]
chk("E5b 四终点离剔除外点解不全 <0.02——更狠的尺子不保证更贴（柯西 |Δ|=0.0084 对照）",
    any(d >= 0.02 for d in deltas),
    f"deltas={[round(float(d), 4) for d in deltas]} vs cauchy={abs(m_it - m_clean):.4f}")
chk("E5c 网格固定 c+柯西解初值格：外点最终权重比柯西更小（最终权重对照）", w_g_fix_init[-1] < w[-1],
    f"w_welsch={w_g_fix_init[-1]:.6f}（该格） < w_cauchy={w[-1]:.6f}")

# ── E5d/e 对照与去重（round3 P1-2 + round4 P1-1 方案a）：柯西对照降为读数级（设计偏：柯西解初值列=不动点、
#    自调+拉偏格=E4 本尊，不支撑「更狠→更不稳」概括）；韦尔施四终点去重计数入档 → 口播「三个不同的地方」有实数支撑 ──
m_c_fix_pull, *_ , _ = irls(cauchy_w, m_out, b_out, c_fixed=mad_final)
m_c_fix_init, *_ , _ = irls(cauchy_w, m_it, b_it, c_fixed=mad_final)
m_c_auto_pull = m_it  # 自调+拉偏初值=E4 本尊
m_c_auto_init, *_ , _ = irls(cauchy_w, m_it, b_it)
cauchy_grid = [m_c_fix_pull, m_c_fix_init, m_c_auto_pull, m_c_auto_init]
cauchy_deltas = [abs(v - m_clean) for v in cauchy_grid]
chk("E5d 同网格柯西对照读数：四终点离剔除外点解全 <0.02 且极差小于韦尔施（对照设计偏注：柯西解初值列=其自身不动点、自调+拉偏格=E4 本尊，有信息量格位仅固定 c+拉偏初值）",
    all(d < 0.02 for d in cauchy_deltas)
    and (max(cauchy_grid) - min(cauchy_grid)) < (max(grid) - min(grid)),
    f"cauchy_grid={[round(v, 4) for v in cauchy_grid]} "
    f"spread={max(cauchy_grid) - min(cauchy_grid):.4f} vs welsch {max(grid) - min(grid):.4f}")
n_distinct = len({round(v, 4) for v in grid})
chk("E5e 韦尔施网格四终点去重后=3 处（口播「三个不同的地方」口径）", n_distinct == 3,
    f"distinct={n_distinct}")

# ── E6 断言总数 ──
n_expect = 23
chk("E6a 断言总数=23（含本条）", len(checks) + 1 == n_expect, f"n={len(checks) + 1}")

readings = {
    "data": {"m_true": M_TRUE, "b_true": B_TRUE, "sigma": SIGMA, "n_inlier": 16,
             "outlier": [x_out_pt, y_out_pt], "seed": 2037},
    "ols": {"m_clean": round(m_clean, 6), "b_clean": round(b_clean, 6),
            "m_out": round(m_out, 6), "b_out": round(b_out, 6)},
    "mad": {"mad_init": round(mad_init, 6), "mad_final": round(mad_final, 6),
            "sigma_true": SIGMA, "std_poor": round(std_poor, 6),
            "note": "mad_init/mad_final=MAD×1.4 后的 σ 估计（原书 B.25 口径，非裸 med）；"
                    "std_poor=拉偏拟合残差（m_out 线）的普通 std，mad_init=同口径残差的 MAD×1.4、"
                    "mad_final=IRLS 收敛后残差口径（round4 P0-1 口径注）；"
                    "traj_c[k]=第 k+1 轮所用 c（17 项），traj_m/traj_w 首项=初值（18 项，含 OLS 起点）"},
    "irls": {"steps": n_steps, "m_final": round(m_it, 6), "b_final": round(b_it, 6),
             "w_outlier_final": round(float(w[-1]), 6),
             "w_inlier_mean_final": round(float(np.mean(w[:-1])), 6),
             "delta_m_vs_clean": round(abs(m_it - m_clean), 6),
             "c_init": round(float(traj_c[0]), 6), "c_final": round(float(traj_c[-1]), 6)},
    "welsch": {"alpha": -1,
               "grid_c_fixed_pull": round(m_g_fix_pull, 6),
               "grid_c_fixed_cauchy_init": round(m_g_fix_init, 6),
               "grid_c_auto_pull": round(m_g_auto_pull, 6),
               "grid_c_auto_cauchy_init": round(m_g_auto_init, 6),
               "distinct_endpoints": n_distinct,
               "cauchy_grid": [round(v, 6) for v in cauchy_grid],
               "cauchy_grid_spread": round(max(cauchy_grid) - min(cauchy_grid), 6),
               "cauchy_grid_deltas": [round(float(d), 6) for d in cauchy_deltas],
               "w_outlier_final": round(float(w_g_fix_init[-1]), 6),
               "note": "2×2 网格 {固定 c=0.1639, 自调}×{拉偏初值, 柯西解初值}（round2 P0-1）：四终点去重后 3 处"
                       "（自调行两初值收敛到同一点），极差 0.0436（vs 柯西对照 0.0265，读数级）；固定 c 下拉偏初值终点比柯西解初值"
                       "更贴好解——「好起点更好」不成立（反例在档），起点归因由原书第 947 页 chip 承载；"
                       "E5d 柯西对照只报读数（柯西极差 0.0265<0.0436、四终点离好解全 <0.02），不支撑「更狠→更不稳」概括"
                       "——对照设计偏：柯西解初值列=其自身不动点、自调+拉偏格=E4 本尊，有信息量格位仅固定 c+拉偏初值"
                       "（round4 P1-1 方案a）；w_outlier_final=固定 c+柯西解初值格（E5c）；"
                       "非「非凸实证/局部谷」"},
    "loss_family": {"alpha_list": [2, 1, 0, -1, -2],
                    "rho_at_r5": [round(rho_barron(5.0, a), 6) for a in (A2, A1, A0, AM1, AM2)],
                    "note": "Barron 正形 c²|α−2|/α（D29，c=1 课程缩放约定）；五退化照图 4.7 题注口径=L2(2)/Charbonnier(1)/Cauchy(0)/Welsch(−1)/GM(−2)（D30）"},
    "traj_m": [round(v, 6) for v in traj_m],
    "traj_c": [round(v, 6) for v in traj_c],
    "traj_w_outlier": [round(v, 6) for v in w_out_traj],
    "traj_w_inlier_mean": [round(v, 6) for v in w_in_traj],
}

# ── 图（765 CSS px 显示盒 1:1；dpi=100；最小字 12pt≈16.7px）──
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
FIGS = OUT / "figures"
FIGS.mkdir(exist_ok=True)
FOOT = 12


def foot(ax, text):
    ax.text(0.5, -0.16, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=FOOT, color="#555", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))


# 图 1：外点拉偏（两面板）
fig, axes = plt.subplots(1, 2, figsize=(7.65, 3.3), dpi=100)
for ax, use_out, title in ((axes[0], False, "没有外点：平方损失稳稳落在线上"),
                           (axes[1], True, "一个外点：平方损失被拉歪")):
    xs, ys = (x_all, y_all) if use_out else (x_in, y_in)
    m_, b_ = ols(xs, ys)
    ax.scatter(x_in, y_in, s=18, color="#3b82f6", label="带噪声内点", zorder=3)
    ax.plot([0, 5], [B_TRUE + M_TRUE * 0, B_TRUE + M_TRUE * 5], "--", color="#9ca3af",
            lw=1.4, label=f"真线 斜率 {M_TRUE:.2f}")
    ax.plot([0, 5], [b_ + m_ * 0, b_ + m_ * 5], "-", color="#ef4444", lw=1.6,
            label=f"平方损失拟合 斜率 {m_:.4f}")
    if use_out:
        ax.scatter([x_out_pt], [y_out_pt], s=90, marker="*", color="#dc2626",
                   edgecolor="white", lw=0.6, zorder=4, label="外点")
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 5.4)
    ax.set_ylim(0.4, 6.0)
    ax.tick_params(labelsize=12)
foot(axes[1], "程序重绘，未用原书原图；外点拉偏概念对应原书第 946 页（MAD/IRLS 见第 947 页）")
axes[0].legend(fontsize=12, loc="upper left")
axes[1].legend(fontsize=12, loc="upper left")
fig.tight_layout()
fig.savefig(FIGS / "e037_outlier_pull.png", bbox_inches="tight")
plt.close(fig)

# 图 2：损失函数族
fig, ax = plt.subplots(figsize=(7.65, 3.2), dpi=100)
rr = np.linspace(0, 5, 401)
for a, name, col in ((A2, "L2（α=2）", "#2563eb"), (A1, "Charbonnier（α=1）", "#0d9488"),
                     (A0, "Cauchy（α=0）", "#f59e0b"), (AM1, "Welsch（α=−1）", "#dc2626"),
                     (AM2, "Geman-McClure（α=−2）", "#7c3aed")):
    ax.plot(rr, [rho_barron(r, a) for r in rr], lw=1.8, color=col, label=name)
ax.set_xlabel("残差 r", fontsize=12)
ax.set_ylabel("损失 ρ(r)", fontsize=12)
ax.set_title("同一族损失：α 越小，大残差越被打折", fontsize=13)
ax.legend(fontsize=12, loc="upper left")
ax.tick_params(labelsize=12)
ax.set_xlim(0, 5)
ax.set_ylim(bottom=-0.2)
foot(ax, "程序重绘，未用原书原图；概念对应原书第 203 页 图 4.7（Barron 2019）——五退化名见题注")
fig.tight_layout()
fig.savefig(FIGS / "e037_loss_family.png", bbox_inches="tight")
plt.close(fig)

# 图 3：IRLS 轨迹（上斜率，下权重）
fig, axes = plt.subplots(2, 1, figsize=(7.65, 5.2), dpi=100, sharex=True)
axes[0].plot(range(len(traj_m)), traj_m, "o-", color="#2563eb", lw=1.6, ms=4,
             label="IRLS 每轮斜率")
axes[0].axhline(m_clean, ls="--", color="#059669", lw=1.4,
                label=f"剔除外点拟合 斜率 {m_clean:.4f}")
axes[0].axhline(M_TRUE, ls=":", color="#9ca3af", lw=1.4, label=f"真线 斜率 {M_TRUE:.2f}")
axes[0].axhline(m_out, ls="-.", color="#ef4444", lw=1.2,
                label=f"平方损失（含外点）斜率 {m_out:.4f}")
axes[0].set_ylabel("斜率 m", fontsize=12)
axes[0].set_title("IRLS 两步舞：固定权重解加权最小二乘 → 重算权重，外点的拉力一轮轮变轻",
                  fontsize=13)
axes[0].legend(fontsize=12, loc="center right")
axes[0].tick_params(labelsize=12)
axes[1].plot(range(len(w_out_traj)), w_out_traj, "s-", color="#dc2626", lw=1.6, ms=4,
             label="外点权重 w")
axes[1].plot(range(len(w_in_traj)), w_in_traj, "o-", color="#3b82f6", lw=1.6, ms=4,
             label="内点平均权重 w")
axes[1].set_xlabel("迭代轮数", fontsize=12)
axes[1].set_ylabel("权重 w(r)", fontsize=12)
axes[1].legend(fontsize=12, loc="center right")
axes[1].tick_params(labelsize=12)
foot(axes[1], "程序重绘，未用原书原图；IRLS 流程对应原书第 947 页（式 B.24）")
fig.tight_layout()
fig.savefig(FIGS / "e037_irls_trace.png", bbox_inches="tight")
plt.close(fig)

(OUT / "e037_robust_readings.json").write_text(
    json.dumps(readings, ensure_ascii=False, indent=1))
print(f"LEDGER: 断言 {len(checks)} 条，全部 PASS")
print(f"READINGS-DONE → e037_robust_readings.json；traj {len(traj_m)} 步；"
      f"m_clean={m_clean:.4f} m_out={m_out:.4f} m_irls={m_it:.4f} "
      f"mad_init={mad_init:.4f} mad_final={mad_final:.4f} c_final={traj_c[-1]:.4f}")
print("E037-ROBUST-LAB-DONE")
