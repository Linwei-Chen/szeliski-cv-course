#!/usr/bin/env python3.12
"""E006 贝叶斯思维：先验、后验与 MAP —— 实验演示

场景延续 E005：两台传感器估同一个真值 3.0
  A：σ=1（准），读出 yA    B：σ=4（糊），读出 yB
本集加入**高斯先验** N(x0, σ0²)：我们知道真值大致在 x0 附近、不确定度 σ0。
MAP 估计 = 信息加权平均（测量一票 + 先验一票）：
  x_MAP = (C_A·yA + C_B·yB + C_0·x0) / (C_A + C_B + C_0)，  C = 1/σ²
这正是式 B.20 的直接推广（式 B.28：能量 = 数据项 + 先验项）。
后验协方差 = 1/(C_A + C_B + C_0)（式 B.19 的 H=I 特例加先验项）。

固定随机种子，所有 [verify] 行可直接核对。
"""
import numpy as np

X_TRUE = 3.0          # 真值
SIGMA_A = 1.0         # 传感器 A：准
SIGMA_B = 4.0         # 传感器 B：糊
X0 = 3.5              # 先验中心："真值大概在 3.5 附近"
SIGMA0_DEFAULT = 2.0  # 先验默认强度（作业里改成 0.5）
SEED = 5              # 与 E005 同种子 → 同一批读数（yA=2.1981、yB=−2.2974），叙事连续
N_MC = 20000

rng = np.random.default_rng(SEED)

# 一次性测量（与 E005 同一种子生成方式，但独立种子）
y_a = X_TRUE + SIGMA_A * rng.standard_normal()
y_b = X_TRUE + SIGMA_B * rng.standard_normal()


def weighted_mean(ys, cs):
    """信息加权平均：Σcᵢyᵢ / Σcᵢ（式 B.20 / 含先验的 B.28 高斯特例）"""
    cs = np.asarray(cs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    return float((cs * ys).sum() / cs.sum())


def energy_curve(xs, y, sigma):
    """单个高斯测量的负对数似然（能量）：½((x−y)/σ)²"""
    return 0.5 * ((xs - y) / sigma) ** 2


def main():
    c_a = 1.0 / SIGMA_A ** 2
    c_b = 1.0 / SIGMA_B ** 2

    print("[verify] 真值 x* =", X_TRUE)
    print(f"[verify] 传感器 A 读数 yA = {y_a:.4f}（σ={SIGMA_A}）")
    print(f"[verify] 传感器 B 读数 yB = {y_b:.4f}（σ={SIGMA_B}）")

    # ---------- 无先验：E005 的 MLE / 信息加权平均（式 B.20） ----------
    mle = weighted_mean([y_a, y_b], [c_a, c_b])
    var_mle = 1.0 / (c_a + c_b)
    print(f"[verify] 无先验 MLE（式 B.20）：x̂ = {mle:.4f}，后验 std = {np.sqrt(var_mle):.4f}")

    # ---------- MAP：加入高斯先验（式 B.28 = 数据能量 + 先验能量） ----------
    print(f"[verify] 先验：N(x0={X0}, σ0={SIGMA0_DEFAULT})，C0 = 1/σ0² = {1/SIGMA0_DEFAULT**2:.4f}")
    c0 = 1.0 / SIGMA0_DEFAULT ** 2
    x_map = weighted_mean([y_a, y_b, X0], [c_a, c_b, c0])
    var_map = 1.0 / (c_a + c_b + c0)
    print(f"[verify] MAP（式 B.28 高斯特例）：x_MAP = {x_map:.4f}，后验 std = {np.sqrt(var_map):.4f}")
    print(f"[verify] 先验把估计从 {mle:.4f} 拉向 {X0}：位移 = {x_map - mle:+.4f}")

    # ---------- 先验强度扫描：σ0 从无穷大（无先验）到 0.5 ----------
    print("[verify] 先验强度扫描（σ0 → x_MAP → 后验 std）")
    sweep = []
    for s0 in [np.inf, 4.0, 2.0, 1.0, 0.5]:
        c0_s = 0.0 if np.isinf(s0) else 1.0 / s0 ** 2
        xm = weighted_mean([y_a, y_b, X0], [c_a, c_b, c0_s])
        sd = np.sqrt(1.0 / (c_a + c_b + c0_s))
        sweep.append((s0, xm, sd))
        name = "∞（无先验=MLE）" if np.isinf(s0) else f"{s0:.1f}"
        print(f"[verify]   σ0={name:>14s} → x_MAP = {xm:.4f}，后验 std = {sd:.4f}")

    # ---------- 网格积分验证后验（独立于解析公式：对未归一化 exp(−E) 数值积分） ----------
    xs_g = np.linspace(-4.2, 8.2, 200001)
    e_tot_g = (energy_curve(xs_g, y_a, SIGMA_A)
               + energy_curve(xs_g, y_b, SIGMA_B)
               + energy_curve(xs_g, X0, SIGMA0_DEFAULT))
    w_g = np.exp(-e_tot_g)
    w_g /= w_g.sum()
    g_mean = float((w_g * xs_g).sum())
    g_std = float(np.sqrt((w_g * (xs_g - g_mean) ** 2).sum()))
    print(f"[verify] 网格积分(N=200001, exp(−E) 未归一化后验)：mean = {g_mean:.4f}（解析 {x_map:.4f}），"
          f"std = {g_std:.4f}（解析 {np.sqrt(var_map):.4f}）")

    # ---------- 作业答案核对：先验中心 x0 = 0（错误先验），σ0 扫描 ----------
    print("[verify] 作业核对：先验中心 x0=0 时 MAP（σ0 → x_MAP，ML 基线见上）")
    for s0 in [4.0, 2.0, 1.0, 0.5]:
        c0_s = 1.0 / s0 ** 2
        xm0 = weighted_mean([y_a, y_b, 0.0], [c_a, c_b, c0_s])
        print(f"[verify]   σ0={s0:>4.1f} → x_MAP = {xm0:.4f}")

    # ---------- 图：能量分解 + 先验强度扫描 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    DARK = "#0b1220"
    C_DATA = "#4da3ff"   # 数据能量
    C_PRIOR = "#f2a33c"  # 先验能量
    C_TOTAL = "#2fe0a8"  # 总能量
    C_TEXT = "#f0ead9"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    fig.patch.set_facecolor(DARK)

    # 左图：三条能量曲线（σ0 = 默认值）
    xs = np.linspace(-4.2, 8.2, 400)
    e_data = energy_curve(xs, y_a, SIGMA_A) + energy_curve(xs, y_b, SIGMA_B)
    e_prior = energy_curve(xs, X0, SIGMA0_DEFAULT)
    e_total = e_data + e_prior
    ax1.plot(xs, e_data, color=C_DATA, lw=2.4, label="数据能量（似然碗，½ 加权平方残差，式 B.13）")
    ax1.plot(xs, e_prior, color=C_PRIOR, lw=2.4, label=f"先验能量 ½((x−{X0:.1f})/{SIGMA0_DEFAULT:.0f})²（式 B.28）")
    ax1.plot(xs, e_total, color=C_TOTAL, lw=2.8, label="总能量 = 相加 → MAP")
    ax1.axvline(mle, color=C_DATA, lw=1.2, ls=":", alpha=0.9)
    ax1.axvline(x_map, color=C_TOTAL, lw=1.6, ls="-", alpha=0.9)
    ax1.axvline(X0, color=C_PRIOR, lw=1.2, ls=":", alpha=0.9)
    ax1.annotate(f"MLE {mle:.2f}", xy=(mle, 2.2), xytext=(-3.9, 8.0),
                 color=C_DATA, fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=C_DATA, lw=1.2))
    ax1.annotate(f"先验中心 {X0:.1f}", xy=(X0, 0.15), xytext=(5.3, 5.0),
                 color=C_PRIOR, fontsize=11,
                 arrowprops=dict(arrowstyle="->", color=C_PRIOR, lw=1.2))
    ax1.annotate(f"MAP {x_map:.2f}", xy=(x_map + 0.06, 1.6), xytext=(3.6, 12.0),
                 color=C_TOTAL, fontsize=12, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_TOTAL, lw=1.4))
    ax1.set_xlim(-4.2, 8.2)
    ax1.set_ylim(-0.6, 26)
    ax1.set_xlabel("待估的未知量 x", color="white")
    ax1.set_ylabel("能量（负对数似然）", color="white")
    ax1.set_title("MAP = 数据能量 + 先验能量 的谷底（式 B.28）", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor=DARK, labelcolor="white", edgecolor="#334155", loc="upper center", fontsize=10)
    ax1.set_facecolor(DARK)
    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # 右图：先验强度扫描（σ0 → x_MAP，误差棒 = 后验 std）
    labels = ["∞\n(=MLE)", "4.0", "2.0\n(本集)", "1.0", "0.5"]
    xs_map = [s[1] for s in sweep]
    xs_sd = [s[2] for s in sweep]
    xpos = np.arange(len(sweep))
    bars = ax2.bar(xpos, xs_map, yerr=xs_sd, width=0.62,
                   color=["#3b4b63", "#3b6b8f", C_TOTAL, "#3b6b8f", "#3b4b63"],
                   edgecolor="white", linewidth=1.1,
                   error_kw=dict(ecolor="white", elinewidth=1.4, capsize=5, capthick=1.4))
    ax2.axhline(X_TRUE, color="#ff7a45", lw=1.6, ls="--")
    ax2.annotate(f"真值 {X_TRUE:.1f}", xy=(0.5, 3.02), xytext=(0.5, 4.18),
                 color="#ff7a45", fontsize=11, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#ff7a45", lw=1.3))
    ax2.axhline(X0, color=C_PRIOR, lw=1.6, ls=":")
    ax2.text(-0.42, X0 + 0.06, f"先验中心 {X0:.1f}", color=C_PRIOR, fontsize=11)
    for xi, v in zip(xpos, xs_map):
        ax2.text(xi + 0.16, v + 0.10, f"{v:.2f}", color="white", fontsize=11)
    ax2.set_xticks(xpos, labels, color="white")
    ax2.set_xlabel("先验强度 σ0（越小 = 先验越强）", color="white")
    ax2.set_ylabel("MAP 估计（误差棒 = 后验 std）", color="white")
    ax2.set_title("先验越强，谷底越靠向先验、后验越窄（式 B.19）", color="white")
    ax2.set_ylim(0, 4.4)
    ax2.tick_params(colors="white")
    ax2.set_facecolor(DARK)
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    fig.tight_layout()
    out = "outputs/bayes_map_demo.png"
    import pathlib
    pathlib.Path("outputs").mkdir(exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=DARK, bbox_inches="tight")
    print(f"[verify] 图已保存：{out}")

    return {
        "yA": y_a, "yB": y_b, "mle": mle, "x_map": x_map, "sweep": sweep,
        "var_map": var_map,
    }


if __name__ == "__main__":
    main()
