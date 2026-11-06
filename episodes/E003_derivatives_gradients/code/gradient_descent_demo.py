#!/usr/bin/env python3
"""gradient_descent_demo.py —— E003 配套实验：在一个碗状误差曲面上做梯度下降。

运行: python3 gradient_descent_demo.py
输出: outputs/gradient_descent_demo.png（等高线轨迹 + 误差下降曲线，拷贝一份到 slides/figures/）
依赖: numpy, matplotlib（无其他外部依赖；无版权问题）

原理（与正片一致）：
  误差函数取椭圆碗 E(p) = 1/2 (p−p*)ᵀ A (p−p*)，解析梯度 ∇E(p) = A(p−p*)。
  因为 E 是 ½dᵀAd 形式，海森矩阵就是 A，梯度下降步长的稳定边界是 α < 2/λmax(A)
  （λmax 是 A 的最大特征值）：稳步收敛、贴边来回跳、越过边界直接发散，三条轨迹一次看全。
  脚本同时用中心差分数值梯度对照解析梯度，验证实现正确。
  数据是理想合成碗（E* = 0），所以收敛轨迹的误差能降到机器精度量级。
"""
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

P_STAR = np.array([2.0, 1.0])            # 碗底（真实最优参数）
A = np.array([[1.0, 0.35],
              [0.35, 0.6]])              # 对称正定 => 碗状
LAM_MAX = float(np.linalg.eigvalsh(A).max())
ALPHA_OK = 0.85 * 2.0 / LAM_MAX          # 稳定收敛
ALPHA_BIG = 2.15 * 2.0 / LAM_MAX         # 越过 2/λmax，发散


def energy(p: np.ndarray) -> float:
    d = p - P_STAR
    return 0.5 * d @ A @ d


def grad_analytic(p: np.ndarray) -> np.ndarray:
    return A @ (p - P_STAR)


def grad_numeric(p: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """中心差分数值梯度：独立于解析式，用来对照验证。"""
    g = np.zeros(2)
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        g[k] = (energy(p + e) - energy(p - e)) / (2 * h)
    return g


def gradient_descent(p0: np.ndarray, alpha: float, steps: int = 60) -> list:
    path = [p0.copy()]
    for _ in range(steps):
        p = path[-1] - alpha * grad_analytic(path[-1])
        path.append(p.copy())
        if np.linalg.norm(p - P_STAR) < 1e-6:
            break
    return path


def main() -> None:
    starts = {"起点甲": np.array([0.2, 3.2]),
              "起点乙": np.array([4.6, 2.6]),
              "起点丙": np.array([0.5, 0.2])}
    # 发散轨迹只跑 12 步：教学上看到「来回反弹、冲出图外」即可，否则坐标轴被拉到 1e31
    settings = [("稳步下山", ALPHA_OK, "起点甲", "#56B4E9", 60),
                ("再换起点", ALPHA_OK, "起点乙", "#009E73", 60),
                ("步长太大", ALPHA_BIG, "起点丙", "#D55E00", 12)]

    paths, curves = [], []
    for _, alpha, sk, _, steps in settings:
        path = gradient_descent(starts[sk], alpha, steps)
        paths.append(path)
        curves.append([energy(p) for p in path])

    # 数值梯度 vs 解析梯度（同一随机点，两条路径互相验证）
    rng = np.random.default_rng(0)
    diff = 0.0
    for _ in range(200):
        p = rng.uniform(-3, 5, 2)
        diff = max(diff, np.abs(grad_numeric(p) - grad_analytic(p)).max())
    print(f"[verify] A 特征值 = {np.linalg.eigvalsh(A)}")
    print(f"[verify] 稳定步长边界 2/λmax = {2.0 / LAM_MAX:.3f}（实验分别用 "
          f"{ALPHA_OK:.3f} 和 {ALPHA_BIG:.3f}）")
    print(f"[verify] 数值梯度 vs 解析梯度 最大差 = {diff:.2e}")
    for (name, alpha, sk, _, _), path in zip(settings, paths):
        end = path[-1]
        status = "已收敛" if np.linalg.norm(end - P_STAR) < 1e-3 else "发散"
        print(f"[verify] {name}: 步长 {alpha:.3f}, 终点 ({end[0]:.4f}, {end[1]:.4f}), 距碗底 "
              f"{np.linalg.norm(end - P_STAR):.2e} -> {status}")

    # 画图：左 = 等高线与轨迹，右 = 误差下降曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2), dpi=150)
    xs = np.linspace(-1.0, 5.0, 300)
    ys = np.linspace(-2.0, 4.0, 300)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.array([[energy(np.array([x, y])) for x in xs] for y in ys])
    levels = np.geomspace(zz.min() + 1e-3, zz.max(), 14)
    ax1.contour(xx, yy, zz, levels=levels, colors="#3a4256", linewidths=0.8)
    for (name, alpha, sk, color, _), path in zip(settings, paths):
        pts = np.array(path)
        ax1.plot(pts[:, 0], pts[:, 1], "-o", color=color, ms=3.5, lw=1.8,
                 label=f"{name} (α={alpha:.2f})")
        ax1.plot(*pts[0], "x", color=color, ms=11, mew=2.5)
    ax1.plot(*P_STAR, "*", color="#f5e6c8", ms=20, mec="#0b1220")
    ax1.annotate("碗底 = 最优参数", P_STAR, textcoords="offset points",
                 xytext=(12, 10), color="#f5e6c8", fontsize=12)
    ax1.set_xlim(-1.0, 5.0)
    ax1.set_ylim(-2.0, 4.0)
    ax1.annotate("✗ 步长太大：几步就冲出图外", xy=(3.6, 3.1), color="#D55E00",
                 fontsize=13, ha="right")
    ax1.set_title("三条梯度下降轨迹（等高线 = 误差地形）", color="w", fontsize=15)
    ax1.set_xlabel("参数 m", color="#c9d1e3")
    ax1.set_ylabel("参数 b", color="#c9d1e3")
    ax1.tick_params(colors="#8b94a8")
    leg = ax1.legend(facecolor="#141b2b", edgecolor="#3a4256", fontsize=10)
    for t in leg.get_texts():
        t.set_color("#e6e9f2")

    for (name, _, _, color, _), curve in zip(settings, curves):
        ax2.semilogy(np.arange(len(curve)), curve, color=color, lw=2.2, label=name)
    ax2.axhline(1e-6, color="#8b94a8", ls="--", lw=1)
    ax2.set_title("误差随迭代下降（步长太大：越走越高）", color="w", fontsize=15)
    ax2.set_xlabel("迭代步数", color="#c9d1e3")
    ax2.set_ylabel("误差 E（对数轴，理想碗 E*=0）", color="#c9d1e3")
    ax2.tick_params(colors="#8b94a8")
    leg2 = ax2.legend(facecolor="#141b2b", edgecolor="#3a4256", fontsize=10)
    for t in leg2.get_texts():
        t.set_color("#e6e9f2")

    fig.patch.set_facecolor("#0b1220")
    for a in (ax1, ax2):
        a.set_facecolor("#0b1220")
    fig.tight_layout()

    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "gradient_descent_demo.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="#0b1220")
    slides_fig = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures" / "gradient_descent_demo.png"
    slides_fig.write_bytes(out_png.read_bytes())
    print(f"saved: {out_png}")
    print(f"saved: {slides_fig}")


if __name__ == "__main__":
    main()
