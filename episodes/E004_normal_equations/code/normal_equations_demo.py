#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""normal_equations_demo.py —— E004《矩阵求导与最小二乘的灵魂》实验

实验一（四方对照，一个答案）：
  同一组 4 个含噪数据点拟合 y = m·x + b，用四种方法解同一个最小二乘问题：
    1) 正规方程直接解   (AᵀA)x = Aᵀb           （原书式 A.30，p.928）
    2) Cholesky 分解    C = AᵀA = RᵀR，两次三角回代（原书式 A.31–A.33）
    3) np.linalg.lstsq（内部 QR/SVD）          （原书式 A.34 的思路）
    4) 梯度下降（上一集的方法，步长 0.85×2/λmax(C)）
  预期：前三种给出几乎相同的解（互差 ~1e-12），梯度下降多走几十步也汇聚到同一点。
  结论：条条大路通正规方程；差别只在步数与数值稳定性。

实验二（病态问题，QR/SVD 更稳）：
  把数据点的 x 全体平移 +1e5（列 [x,1] 几乎共线，AᵀA 条件数飙升到 ~1e19），
  噪声调小到 1e-3。分别用 float32 / float64 求解正规方程与 lstsq，
  与 float64 lstsq 参考解对比「斜率 m」的偏差。此时 b 已被病态放大到没有意义
  （x 离原点太远，外推回 x=0 的截距误差爆炸——这本身就是一堂课），所以只看 m。
  预期 2×2：float32 正规方程 = 灾难（m 偏差 O(10)）；float32 lstsq = 尚可（1e-4）；
  float64 正规方程 = 轻微失真（1e-7）；float64 lstsq = 参考答案。
  这正是原书 p.928–929 的建议：病态时"加正则化/先验，或改用 QR/SVD"。
  另验证：把 x 减去 1e5（中心化）后条件数立刻回到 ~14 —— 病的是坐标系，不是数据。

输出：
  code/outputs/normal_equations_demo.png        （实验一：拟合线 + 偏差柱状图）
  code/outputs/condition_number_demo.png        （实验二：m 偏差柱状图 + 飞线示意）
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
from scipy.linalg import solve_triangular

OUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "slides" / "figures"

M_STAR, B_STAR = 0.8, 1.5          # 真直线
NOISE_MAIN = 0.30                  # 实验一噪声幅度
NOISE_ILL = 1e-3                   # 实验二噪声幅度（病态时调小，让舍入误差可见）
X_SCATTER = np.array([0.0, 1.0, 2.0, 3.0])
X_CROWD_SHIFT = 1e5                # 实验二全体 x 平移量
GD_MAX_STEPS = 400
GD_TOL = 1e-10        # 单步位移阈值：步长不再挪动有效数字时就停

BLUE, GREEN, ORANGE, RED, GRAY = "#1e3a5f", "#009E73", "#D55E00", "#CC79A7", "#8a8378"


def make_design_matrix(x: np.ndarray) -> np.ndarray:
    """每行 [x_i, 1]，即原书 a_i 行向量，堆成 A。"""
    return np.column_stack([x, np.ones_like(x)])


def noisy_points(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return M_STAR * x + B_STAR + rng.normal(0.0, sigma, size=x.shape)


def fit_normal_eq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    C = A.T @ A
    d = A.T @ b
    return np.linalg.solve(C, d)


def fit_cholesky(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """原书式(A.31)–(A.33)：C = RᵀR；先解 Rᵀz = d 再解 R x = z。
    numpy 的 cholesky 返回下三角 L，对应原书的 Rᵀ。"""
    C = A.T @ A
    d = A.T @ b
    L = np.linalg.cholesky(C)                       # L = Rᵀ
    z = solve_triangular(L, d, lower=True)          # Rᵀ z = d（前代）
    return solve_triangular(L.T, z, lower=False)    # R x = z（回代）


def fit_lstsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x


def fit_gradient_descent(A: np.ndarray, b: np.ndarray):
    """上一集的下山法：步长 0.85 × 2/λmax(C)。返回 (解, 用了几步)。"""
    C = A.T @ A
    d = A.T @ b
    lam_max = max(np.linalg.eigvalsh(C))
    alpha = 0.85 * 2.0 / lam_max
    p = np.zeros(2)
    for k in range(1, GD_MAX_STEPS + 1):
        g = C @ p - d
        p = p - alpha * g
        if np.linalg.norm(alpha * g) < GD_TOL:
            return p, k
    return p, GD_MAX_STEPS


def report(name: str, x: np.ndarray) -> None:
    print(f"    {name:<14} m = {x[0]:+.10f},  b = {x[1]:+.10f}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # ---------- 实验一：四方对照 ----------
    print("[实验一] 四种方法解同一个直线拟合（4 个含噪点）")
    y = noisy_points(X_SCATTER, NOISE_MAIN, rng)
    A = make_design_matrix(X_SCATTER)

    x_ne = fit_normal_eq(A, y)
    x_ch = fit_cholesky(A, y)
    x_ls = fit_lstsq(A, y)
    x_gd, gd_steps = fit_gradient_descent(A, y)

    ref = x_ls
    diffs = {
        "正规方程": np.linalg.norm(x_ne - ref),
        "Cholesky": np.linalg.norm(x_ch - ref),
        "lstsq(QR/SVD)": 0.0,
        "梯度下降": np.linalg.norm(x_gd - ref),
    }
    report("正规方程", x_ne)
    report("Cholesky", x_ch)
    report("lstsq(QR/SVD)", x_ls)
    report("梯度下降", x_gd)
    print(f"[verify] 与 lstsq 参考解的最大互差 = {max(diffs.values()):.2e}"
          f"（梯度下降走了 {gd_steps} 步，其余 1 步到位）")
    C = A.T @ A
    print(f"[verify] 分散数据 cond(AᵀA) = {np.linalg.cond(C):.1f}")

    # 图 1：左=数据与四条拟合线；右=偏差柱状图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="#0b1220")
    xs = np.linspace(X_SCATTER.min() - 0.3, X_SCATTER.max() + 0.3, 50)
    ax1.scatter(X_SCATTER, y, s=90, marker="x", color=ORANGE, linewidths=3, zorder=3,
                label="含噪数据")
    styles = [
        ("正规方程 $(A^TA)x=A^Tb$", BLUE, "-"),
        ("Cholesky 回代", GREEN, "--"),
        ("lstsq(QR/SVD)", RED, ":"),
        ("梯度下降（%d 步）" % gd_steps, "#56B4E9", "-."),
    ]
    for (label, color, ls), xhat in zip(styles, [x_ne, x_ch, x_ls, x_gd]):
        ax1.plot(xs, xhat[0] * xs + xhat[1], color=color, ls=ls, linewidth=2.2, label=label)
    ax1.set_title(f"四种方法拟合同一组数据（拟合线完全重叠）\n"
                  f"一致解 m={x_ls[0]:.6f}, b={x_ls[1]:.6f}",
                  color="white", fontsize=12.5)
    ax1.set_xlabel("x", color="white")
    ax1.set_ylabel("y", color="white")
    leg = ax1.legend(fontsize=9.5, facecolor="#0b1220", labelcolor="white",
                     loc="upper left")
    for t in leg.get_texts():
        t.set_color("white")
    ax1.tick_params(colors="white")
    for sp in ax1.spines.values():
        sp.set_color("#444444")

    names = list(diffs.keys())
    vals = [max(d, 1e-16) for d in diffs.values()]
    bars = ax2.bar(names, vals, color=[BLUE, GREEN, RED, "#56B4E9"])
    ax2.set_yscale("log")
    ax2.set_ylim(1e-16, 1e-5)
    ax2.set_title("与 lstsq 参考解的偏差（对数轴）", color="white", fontsize=13)
    ax2.set_ylabel("‖x_方法 − x_lstsq‖", color="white")
    ax2.tick_params(colors="white")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v * 3, f"{v:.1e}",
                 ha="center", color="white", fontsize=10)
    for sp in ax2.spines.values():
        sp.set_color("#444444")
    fig.suptitle("实验一：条条大路通正规方程——四个方法，一个答案",
                 color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for out in (OUT_DIR / "normal_equations_demo.png", FIG_DIR / "normal_equations_demo.png"):
        fig.savefig(out, bbox_inches="tight", facecolor="#0b1220")
        print(f"saved: {out}")
    plt.close(fig)

    # ---------- 实验二：病态问题 ----------
    print("\n[实验二] 病态数据（x 全体平移 +1e5，噪声 1e-3）下的数值稳定性")
    x_crowd = X_SCATTER + X_CROWD_SHIFT
    y_ill = noisy_points(x_crowd, NOISE_ILL, rng)
    A_ill = make_design_matrix(x_crowd)
    C_ill = A_ill.T @ A_ill
    print(f"[verify] 拥挤数据 cond(AᵀA) = {np.linalg.cond(C_ill):.3e}"
          f"（分散数据 {np.linalg.cond(C):.1f}，放大 {np.linalg.cond(C_ill) / np.linalg.cond(C):.1e} 倍）")

    ref64 = fit_lstsq(A_ill.astype(np.float64), y_ill.astype(np.float64))
    cases = [
        ("正规方程\nfloat32", fit_normal_eq(A_ill.astype(np.float32), y_ill.astype(np.float32)).astype(np.float64), ORANGE),
        ("lstsq(QR/SVD)\nfloat32", fit_lstsq(A_ill.astype(np.float32), y_ill.astype(np.float32)).astype(np.float64), BLUE),
        ("正规方程\nfloat64", fit_normal_eq(A_ill, y_ill), ORANGE),
        ("lstsq(QR/SVD)\nfloat64", fit_lstsq(A_ill, y_ill), BLUE),
    ]
    m_errs = {name: abs(float(x[0] - ref64[0])) for name, x, _ in cases}
    for name, x, _ in cases:
        report(name.replace("\n", " "), x)
    print(f"[verify] 斜率偏差基准 = float64 lstsq 参考解（m={ref64[0]:+.6f}）："
          + ", ".join(f"{k.replace(chr(10), ' ')} {v:.1e}" for k, v in m_errs.items()))
    floor = NOISE_ILL / np.sqrt(np.sum((x_crowd - x_crowd.mean()) ** 2))
    print(f"[verify] 参考解自身离真值 {M_STAR} 偏 {abs(ref64[0] - M_STAR):.1e}"
          f"（≈ 统计地板 σ/√Sxx = {floor:.1e}——这是噪声的账，不是算法的锅）")
    print(f"[verify] 截距 b 全部失真（参考解 b={ref64[1]:+.2f}，真值 {B_STAR}）——"
          f"x 在 1e5 开外，外推回 x=0 的截距本就不可辨识，这是病态的另一面")
    C_shiftback = make_design_matrix(x_crowd - X_CROWD_SHIFT)
    C_centered = make_design_matrix(x_crowd - x_crowd.mean())
    print(f"[verify] 把 x 减去平移量（原点挪回数据外）：cond(AᵀA) = {np.linalg.cond(C_shiftback.T @ C_shiftback):.1f}"
          f"（与原始数据完全相同）")
    print(f"[verify] 真正的中心化（x 减去均值）：cond(AᵀA) = {np.linalg.cond(C_centered.T @ C_centered):.2f}"
          f" —— 比原始数据的 14.1 还健康")

    # 图 2：左=斜率偏差柱状图（2×2）；右=飞线示意
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="#0b1220")
    names2 = [name for name, _, _ in cases]
    vals2 = [max(m_errs[name], 1e-16) for name in names2]
    colors2 = [c for _, _, c in cases]
    bars = ax1.bar(names2, vals2, color=colors2, width=0.62)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-9, 1e3)
    ax1.set_title(f"病态数据（cond≈{np.linalg.cond(C_ill):.0e}）：斜率 m 的偏差\n（基准 = float64 lstsq 参考解）",
                  color="white", fontsize=12)
    ax1.set_ylabel("|m_方法 − m_参考|", color="white")
    ax1.tick_params(colors="white", labelsize=9.5)
    for bar, v in zip(bars, vals2):
        y_pos = max(v * 3, 3e-9)
        ax1.text(bar.get_x() + bar.get_width() / 2, y_pos,
                 f"{v:.1e}" if v > 1e-15 else "参考",
                 ha="center", color="white", fontsize=10)
    for sp in ax1.spines.values():
        sp.set_color("#444444")

    # 右图：把病态数据放大看，float32 正规方程的"拟合线"直接飞了
    xw = np.array([X_CROWD_SHIFT - 0.5, X_CROWD_SHIFT + 3.5])
    xd = xw - X_CROWD_SHIFT
    ax2.scatter(x_crowd - X_CROWD_SHIFT, y_ill, s=110, marker="x", color="white",
                linewidths=3, zorder=3, label="含噪数据（噪声 1e-3）")
    ax2.plot(xd, ref64[0] * xw + ref64[1], color=BLUE, linewidth=2.4,
             label=f"lstsq float64（m={ref64[0]:.4f}）")
    x_ne32 = cases[0][1]
    ax2.plot(xd, x_ne32[0] * xw + x_ne32[1], color=ORANGE, linewidth=2.4, ls="--",
             label=f"正规方程 float32（m={x_ne32[0]:.1f}，飞了）")
    ax2.set_title("同一批数据放大 1e5 倍看：float32 正规方程的线斜率错了 25 倍",
                  color="white", fontsize=12)
    ax2.set_xlabel("x − 100000", color="white")
    ax2.set_ylabel("y", color="white")
    leg = ax2.legend(fontsize=9.5, facecolor="#0b1220", labelcolor="white", loc="upper left")
    for t in leg.get_texts():
        t.set_color("white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_color("#444444")
    fig.suptitle("实验二：病态时舍入误差被条件数放大——QR/SVD 更稳（第 928–929 页）",
                 color="white", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for out in (OUT_DIR / "condition_number_demo.png", FIG_DIR / "condition_number_demo.png"):
        fig.savefig(out, bbox_inches="tight", facecolor="#0b1220")
        print(f"saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
