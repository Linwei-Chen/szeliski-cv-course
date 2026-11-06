#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inverse_problem_demo.py —— E009 实验：画一张标准内克尔立方三联图。

标准内克尔立方 = 12 条棱的透明线框立方体（左下、右上两个错位正方形 + 四条连线）。
同一组线条有两种立体解读，都是凸立方体，区别只在观察方位：
  读法 A：从左下方看 —— 左下的面在前，可见「前面 + 底面 + 左侧面」三个面。
  读法 B：从右上方看 —— 右上的面在前，可见「前面 + 顶面 + 右侧面」三个面。
两个面板着色方式完全对称，线条一模一样，只有「哪三个面被点亮」不同。

只依赖 matplotlib（不查任何 3D 库）；不联网。
输出: code/outputs/necker_cube.png，并同步到 slides/figures/ 供幻灯片引用。
用法: python3 episodes/E009_what_is_computer_vision/code/inverse_problem_demo.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 中文字体（macOS 自带；缺省 DejaVu Sans 无汉字）
plt.rcParams["font.family"] = ["PingFang SC", "Songti SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#1e3a5f"
ORANGE = "#e8590c"
PAPER = "#faf7f2"

# 立方体 8 个投影顶点：左下正方形 ABCD；右上正方形 EFGH = ABCD 沿对角线平移 (0.7, 0.7)
S1 = {"A": (-0.85, -0.85), "B": (0.35, -0.85), "C": (0.35, 0.35), "D": (-0.85, 0.35)}
S2 = {"E": (-0.15, -0.15), "F": (1.05, -0.15), "G": (1.05, 1.05), "H": (-0.15, 1.05)}
V = {**S1, **S2}
SQUARE1 = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
SQUARE2 = [("E", "F"), ("F", "G"), ("G", "H"), ("H", "E")]
CONNECT = [("A", "E"), ("B", "F"), ("C", "G"), ("D", "H")]
EDGES = SQUARE1 + SQUARE2 + CONNECT          # 共 12 条棱

# 读法 A（左下的面在前）可见三面：前面 ABCD + 底面 ABFE + 左侧面 ADHE
FACES_A = [
    [("A", "B", "C", "D"), 0.50],
    [("A", "B", "F", "E"), 0.30],
    [("A", "D", "H", "E"), 0.30],
]
# 读法 B（右上的面在前）可见三面：前面 EFGH + 顶面 DCGH + 右侧面 BCGF
FACES_B = [
    [("E", "F", "G", "H"), 0.50],
    [("D", "C", "G", "H"), 0.30],
    [("B", "C", "G", "F"), 0.30],
]


def draw_wireframe(ax, lw=3.4, color=BLUE):
    for a, b in EDGES:
        (x1, y1), (x2, y2) = V[a], V[b]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                solid_capstyle="round", zorder=3)


def panel(ax, title, subtitle):
    ax.set_xlim(-1.45, 1.65)
    ax.set_ylim(-1.45, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=30, color="#1e1e26", pad=46, fontweight="bold")
    ax.text(0.1, -1.32, subtitle, fontsize=19, color="#8a8377", ha="center")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.6), dpi=150)
    fig.patch.set_facecolor(PAPER)

    # 左：歧义线框（两个读法都说得通）
    ax = axes[0]
    panel(ax, "同一组十二条边", "左下的面在前，还是右上的面在前？")
    draw_wireframe(ax)
    ax.text(0.1, 0.1, "?", fontsize=44, color=ORANGE, ha="center",
            va="center", fontweight="bold", zorder=4)

    # 中：读法 A —— 从左下方看，左下的面在前（可见三面以 A 为公共角）
    ax = axes[1]
    panel(ax, "读法 A：左下的面在前", "从左下方看，底面、侧面一并可见 → 凸立方")
    for pts, alpha in FACES_A:
        ax.add_patch(Polygon([V[p] for p in pts], closed=True,
                             facecolor=BLUE, alpha=alpha, edgecolor="none", zorder=1))
    draw_wireframe(ax, lw=3.8)

    # 右：读法 B —— 同样的线条，从右上方看，右上的面在前（可见三面以 G 为公共角）
    ax = axes[2]
    panel(ax, "读法 B：右上的面在前", "从右上方看，顶面、侧面一并可见 → 朝向翻转")
    for pts, alpha in FACES_B:
        ax.add_patch(Polygon([V[p] for p in pts], closed=True,
                             facecolor=ORANGE, alpha=alpha, edgecolor="none", zorder=1))
    draw_wireframe(ax, lw=3.8)

    fig.suptitle("内克尔立方：一张图，两种世界 —— 视觉是逆问题",
                 fontsize=24, color=BLUE, y=1.00, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "necker_cube.png"
    fig.savefig(out, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)

    slides_fig = Path(__file__).resolve().parent.parent / "slides" / "figures"
    slides_fig.mkdir(parents=True, exist_ok=True)
    (slides_fig / "necker_cube.png").write_bytes(out.read_bytes())
    print(f"[ok] 已生成 {out}")
    print(f"[ok] 已同步 {slides_fig / 'necker_cube.png'}")


if __name__ == "__main__":
    main()
