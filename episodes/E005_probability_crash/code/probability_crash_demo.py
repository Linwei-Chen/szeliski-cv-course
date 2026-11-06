#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probability_crash_demo.py —— E005《概率速通：噪声的语言》实验

实验（两台传感器估同一个真值）：
  未知量的真值 X_TRUE = 3.0。两台传感器各给一个带高斯噪声的读数：
    传感器 A：噪声标准差 SIGMA_A = 1.0（准）
    传感器 B：噪声标准差 SIGMA_B = 4.0（糊）
  噪声模型即原书式 B.7：n_i ~ N(0, Σ_i)（这里取各向同性 σ_i²I）。

  面板一（似然的形状）：
    对每台传感器画负对数似然 E_A(x) = ½((x−y_A)/σ_A)²、E_B(x) = ½((x−y_B)/σ_B)²
    ——这正是原书式 B.13 去掉常数后的单测量能量；联合能量 E = E_A + E_B
    对应式 B.6 的似然连乘取负对数（exp 下的乘积在对数下变求和）。
    联合能量的谷底就是最大似然解，解析式恰为原书式 B.20（H_i = I 的信息加权平均）：
        x* = (y_A/σ_A² + y_B/σ_B²) / (1/σ_A² + 1/σ_B²)
    预期：谷底被"准"的传感器 A 拉得离 y_A 很近——式 B.13 里 C_i = Σ_i⁻¹ 是残差的权重
    （原书 p.943：信息多、约束强）。

  面板二（方差的账，蒙特卡洛 N_MC 次重复）：
    每次重复重新抽一对 (y_A, y_B)，分别用「简单平均」和「信息加权平均（式 B.20）」估计真值，
    画估计误差直方图并报标准差。理论值（讲者推导，可解析验证）：
        简单平均方差 = (σ_A² + σ_B²)/4 = 17/4 = 4.25      → std ≈ 2.062
        加权平均方差 = (C_A + C_B)⁻¹ = 16/17 ≈ 0.941（原书式 B.19 取 H_i = I）
                                                    → std ≈ 0.970
    预期：加权平均的误差散布明显更窄（约 2.1 倍），且两种估计的误差均值都 ≈ 0
    ——零均值噪声（E[n] = 0）经线性运算传递后依然零均值：期望的线性性。

输出：
  code/outputs/probability_crash_demo.png       （似然曲线 + 蒙特卡洛误差直方图）
  同时复制到 slides/figures/ 供 deck.html 引用。
"""
import pathlib
import shutil

import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.facecolor"] = "#0b1220"

import matplotlib.pyplot as plt

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "slides" / "figures"

X_TRUE = 3.0        # 真值（讲者设定）
SIGMA_A = 1.0       # 传感器 A：准
SIGMA_B = 4.0       # 传感器 B：糊
N_MC = 20000        # 蒙特卡洛重复次数
SEED = 5            # 第 5 集，用 5 号种子

BLUE, GREEN, ORANGE, GRAY = "#1e3a5f", "#009E73", "#D55E00", "#8a8378"


def nll(y: float, sigma: float, xs: np.ndarray) -> np.ndarray:
    """单测量负对数似然（去掉常数 k）：½((x−y)/σ)²，即式 B.13 的单测量能量。"""
    return 0.5 * ((xs - y) / sigma) ** 2


def weighted_mean(y_a: float, y_b: float) -> float:
    """式 B.20（H_i = I）：x = (Σ C_i)⁻¹ Σ C_i ỹ_i，C_i = 1/σ_i²。"""
    c_a, c_b = 1.0 / SIGMA_A**2, 1.0 / SIGMA_B**2
    return (c_a * y_a + c_b * y_b) / (c_a + c_b)


def main() -> None:
    rng = np.random.default_rng(SEED)

    # 头条一对读数：从种子里取前两抽（A 先、B 后），报真实值
    y_a = X_TRUE + rng.normal(0.0, SIGMA_A)
    y_b = X_TRUE + rng.normal(0.0, SIGMA_B)
    x_w = weighted_mean(y_a, y_b)

    # 蒙特卡洛：N_MC 次重复，各算法的估计误差
    y_a_mc = X_TRUE + rng.normal(0.0, SIGMA_A, N_MC)
    y_b_mc = X_TRUE + rng.normal(0.0, SIGMA_B, N_MC)
    err_plain = 0.5 * (y_a_mc + y_b_mc) - X_TRUE
    err_weight = np.array([weighted_mean(a, b) - X_TRUE for a, b in zip(y_a_mc, y_b_mc)])

    # 理论值（讲者推导）
    var_plain = (SIGMA_A**2 + SIGMA_B**2) / 4.0            # = 17/4
    var_weight = 1.0 / (1.0 / SIGMA_A**2 + 1.0 / SIGMA_B**2)  # = 16/17，式 B.19

    print(f"[verify] 真值 x* = {X_TRUE}")
    print(f"[verify] 传感器 A 读数 yA = {y_a:.4f}（σ={SIGMA_A}）")
    print(f"[verify] 传感器 B 读数 yB = {y_b:.4f}（σ={SIGMA_B}）")
    print(f"[verify] 式 B.20 信息加权平均 x̂ = (16·yA + yB)/17 = {x_w:.4f}")
    print(f"[verify] 理论：简单平均方差 = (1+16)/4 = {var_plain:.4f}，std = {var_plain**0.5:.4f}")
    print(f"[verify] 理论：加权平均方差 = (1+1/16)^-1 = 16/17 = {var_weight:.4f}，std = {var_weight**0.5:.4f}")
    print(f"[verify] 理论窄化倍数 = {var_plain**0.5 / var_weight**0.5:.2f}")
    print(f"[verify] MC(N={N_MC}) 简单平均：mean_err = {err_plain.mean():+.4f}，std = {err_plain.std():.4f}")
    print(f"[verify] MC(N={N_MC}) 加权平均：mean_err = {err_weight.mean():+.4f}，std = {err_weight.std():.4f}")

    # ---------- 画图 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.4), facecolor="#0b1220")

    # 面板一：负对数似然曲线
    xs = np.linspace(-4.2, 8.2, 800)
    e_a = nll(y_a, SIGMA_A, xs)
    e_b = nll(y_b, SIGMA_B, xs)
    e_j = e_a + e_b
    ax1.plot(xs, e_a, color="#7ea4cc", lw=2.6, label=f"A 的能量 ½(x−{y_a:.2f})²（σ=1，陡）")
    ax1.plot(xs, e_b, color=ORANGE, lw=2.6, label=f"B 的能量 ½((x{y_b:+.2f})/4)²（σ=4，缓）")
    ax1.plot(xs, e_j, color=GREEN, lw=3.2, ls="--", label="联合能量 = 两者相加（式 B.6→B.13）")
    ax1.axvline(y_a, color="#7ea4cc", lw=1.4, ls=":", alpha=0.8)
    ax1.axvline(y_b, color=ORANGE, lw=1.4, ls=":", alpha=0.8)
    ax1.axvline(x_w, color=GREEN, lw=2.0, alpha=0.9)
    ax1.text(y_b, 20.8, "B 的读数", color=ORANGE, fontsize=10, ha="center")
    ax1.text(y_a, 26.5, "A 的读数", color="#7ea4cc", fontsize=10, ha="center")
    ax1.annotate("联合谷底 = 式 B.20\n信息加权平均", xy=(x_w + 0.06, 7.5), xytext=(3.5, 20.5),
                 fontsize=11, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax1.set_title("似然翻成能量：谷底被「准」的传感器拉近", color="white", fontsize=13.5, fontweight="bold")
    ax1.set_xlabel("待估的未知量 x", color="white", fontsize=11.5)
    ax1.set_ylabel("负对数似然（能量）", color="white", fontsize=11.5)
    ax1.legend(fontsize=9.5, facecolor="#0b1220", labelcolor="white", loc="upper left")
    ax1.set_xlim(-4.2, 8.2)
    ax1.set_ylim(-0.6, 30)

    # 面板二：蒙特卡洛误差直方图
    bins = np.linspace(-8, 8, 61)
    ax2.hist(err_plain, bins=bins, color=ORANGE, alpha=0.55,
             label=f"简单平均：std = {err_plain.std():.2f}")
    ax2.hist(err_weight, bins=bins, color=GREEN, alpha=0.75,
             label=f"加权平均（式 B.20）：std = {err_weight.std():.2f}")
    ax2.axvline(0.0, color="white", lw=1.6, ls="--", alpha=0.85)
    ax2.annotate("两种估计的误差均值都 ≈ 0\n（零均值噪声的期望）", xy=(0.3, 520), xytext=(3.4, 980),
                 fontsize=10.5, color="#f0ead9",
                 arrowprops=dict(arrowstyle="->", color="#f0ead9", lw=1.3))
    ax2.set_title(f"重复 {N_MC} 次的估计误差：加权更窄（理论 2.06 vs 0.97）",
                  color="white", fontsize=13.5, fontweight="bold")
    ax2.set_xlabel("估计误差 = 估计值 − 真值 3.0", color="white", fontsize=11.5)
    ax2.set_ylabel("次数", color="white", fontsize=11.5)
    ax2.legend(fontsize=10.5, facecolor="#0b1220", labelcolor="white", loc="upper left")

    for ax in (ax1, ax2):
        ax.tick_params(colors="white", labelsize=10)
        for s in ax.spines.values():
            s.set_color("#444444")
        ax.grid(True, color="#8a8378", alpha=0.30, lw=0.7)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "probability_crash_demo.png"
    fig.savefig(out_png, dpi=170, facecolor="#0b1220", bbox_inches="tight")
    shutil.copyfile(out_png, FIG_DIR / "probability_crash_demo.png")
    print(f"[verify] 图已保存：{out_png} → slides/figures/")


if __name__ == "__main__":
    main()
