#!/usr/bin/env python3
"""rotate_demo.py —— E001 配套实验：手写旋转矩阵，转动一张图。

运行: python3 rotate_demo.py
输出: outputs/rotation_demo.png（前后对比图，同时拷贝一份到 slides/figures/ 供幻灯片引用）
依赖: numpy, matplotlib（无其他外部依赖；图像为程序合成，无版权问题）

原理（与正片一致）：
  对输出图的每个像素位置 p_out，用旋转矩阵的逆 R^-1 反查它在原图的位置 p_in，
  再做双线性插值取色 —— 这就是几何变换的「逆映射」标准做法（详见 E030）。
"""
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

W = H = 240          # 合成图像尺寸
ANGLE_DEG = 30.0     # 旋转角


def make_image() -> np.ndarray:
    """合成一张结构丰富的图：双色棋盘 + 圆 + 十字线。"""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), dtype=np.float32)
    board = ((xx // 30 + yy // 30) % 2 == 0)
    img[board] = (0.16, 0.19, 0.27)          # 深蓝格
    img[~board] = (0.55, 0.62, 0.72)         # 浅灰格
    r = np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2)
    img[(r > 40) & (r < 60)] = (0.18, 0.83, 0.75)   # 青色圆环
    img[(np.abs(xx - 120) < 4) | (np.abs(yy - 120) < 4)] = (0.96, 0.62, 0.07)  # 橙十字
    return img


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """逆映射旋转：对输出每个像素，用 R^-1 找回原图坐标并双线性插值。"""
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])          # 正向旋转矩阵（本集主角）
    Rinv = R.T                                # 旋转矩阵的逆 = 转置（正交矩阵性质）

    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w]
    # 输出坐标 -> 以中心为原点 -> R^-1 反查原图坐标
    dx, dy = xx - cx, yy - cy
    src = Rinv @ np.stack([dx.ravel(), dy.ravel()])   # 2 x N
    sx, sy = src[0].reshape(h, w) + cx, src[1].reshape(h, w) + cy

    # 双线性插值（越界填 0）
    x0 = np.clip(np.floor(sx).astype(int), 0, w - 2)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 2)
    fx = np.clip(sx - x0, 0, 1)[..., None]
    fy = np.clip(sy - y0, 0, 1)[..., None]
    out = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
           + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    outside = (sx < 0) | (sx > w - 1) | (sy < 0) | (sy > h - 1)
    out[outside] = 0
    return out


def main() -> None:
    img = make_image()
    rot = rotate(img, ANGLE_DEG)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), dpi=150)
    axes[0].imshow(img)
    axes[0].set_title("input image", fontsize=11)
    axes[1].imshow(rot)
    axes[1].set_title(f"rotated by $R$({ANGLE_DEG:.0f}$\\degree$)", fontsize=11)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "rotation_demo.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="#0b1220")

    # 同步一份给幻灯片引用
    slides_fig = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures" / "rotation_demo.png"
    slides_fig.write_bytes(out_png.read_bytes())

    # 打印矩阵本体，方便对照正片
    th = np.deg2rad(ANGLE_DEG)
    print(f"R({ANGLE_DEG:.0f}°) = [[cos, -sin], [sin, cos]]")
    print(f"          = [[{np.cos(th):.3f}, {-(np.sin(th)):.3f}], [{np.sin(th):.3f}, {np.cos(th):.3f}]]")
    print(f"saved: {out_png}")
    print(f"saved: {slides_fig}")


if __name__ == "__main__":
    main()
