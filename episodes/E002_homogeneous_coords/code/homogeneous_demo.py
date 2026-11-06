#!/usr/bin/env python3
"""homogeneous_demo.py —— E002 配套实验：齐次坐标把「平移」变成乘法。

运行: python3 homogeneous_demo.py
输出: outputs/homogeneous_demo.png（三写法对比图，同时拷贝一份到 slides/figures/ 供幻灯片引用）
依赖: numpy, matplotlib（无其他外部依赖；图像为程序合成，无版权问题）

原理（与正片一致，原书 §2.1.1 印刷 p.40 式 2.14–2.15）：
  方法A（老写法）：绕中心旋转 = 先减中心 → 旋转 → 再加回去，2×2 矩阵 + 手工加减法。
  方法B（齐次）：  M = T(c)·R(θ)·T(−c) 一条 3×3 矩阵链，逆映射直接用 inv(M)。
  方法C（长链）：  平移·旋转·缩放·再平移 四个 3×3 矩阵连乘，一次作用。
  逆映射（详见 E030）：对输出每个像素，用 M 的逆反查它在原图的位置，再双线性插值取色。
"""
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

W = H = 240          # 合成图像尺寸
ANGLE_DEG = 30.0     # 旋转角（作业：改成 -30 再跑一遍）
SCALE = 1.25         # 长链里的缩放
OFFSET = (-28.0, 34.0)  # 长链最后一步的平移


def make_image() -> np.ndarray:
    """合成一张结构丰富的图：双色棋盘 + 圆环 + 十字线（与 E001 同款）。"""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3), dtype=np.float64)
    board = ((xx // 30 + yy // 30) % 2 == 0)
    img[board] = (0.16, 0.19, 0.27)          # 深蓝格
    img[~board] = (0.55, 0.62, 0.72)         # 浅灰格
    r = np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2)
    img[(r > 40) & (r < 60)] = (0.18, 0.83, 0.75)   # 青色圆环
    img[(np.abs(xx - 120) < 4) | (np.abs(yy - 120) < 4)] = (0.96, 0.62, 0.07)  # 橙十字
    return img


def rot2d(angle_deg: float) -> np.ndarray:
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


def hom(tx: float = 0.0, ty: float = 0.0, a: float = 0.0, s: float = 1.0) -> np.ndarray:
    """一步构造齐次变换矩阵：先绕原点旋转 a、再缩放 s、再平移 (tx, ty)。"""
    R = s * rot2d(a)
    return np.array([[R[0, 0], R[0, 1], tx],
                     [R[1, 0], R[1, 1], ty],
                     [0.0, 0.0, 1.0]])


def warp_naive(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """方法A（老写法）：逆映射 = R^-1 反查，先减中心、旋转、再加回去。

    与方法B独立实现：不经过齐次矩阵 M，不调用 np.linalg.inv。
    注意：图像坐标系 y 轴向下，正角度在屏幕上看是顺时针。
    """
    Rinv = rot2d(angle_deg).T                # 旋转矩阵的逆 = 转置（正交矩阵性质）
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx - cx, yy - cy                # 第1步：减去中心
    src = Rinv @ np.stack([dx.ravel(), dy.ravel()])   # 第2步：旋转（乘法）
    sx = src[0].reshape(h, w) + cx           # 第3步：加回中心
    sy = src[1].reshape(h, w) + cy
    return bilinear(img, sx, sy)


def warp_homogeneous(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """方法B/C（齐次）：对输出每个像素，用 inv(M) 反查原图的齐次坐标。"""
    Minv = np.linalg.inv(M)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    p = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)])   # 3 x N 齐次坐标
    src = Minv @ p                                           # 一次矩阵乘法
    sx = (src[0] / src[2]).reshape(h, w)                     # 除以 w 回到普通坐标
    sy = (src[1] / src[2]).reshape(h, w)
    return bilinear(img, sx, sy)


def bilinear(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """双线性插值取色（越界填 0），与 E001 的 rotate_demo 相同。"""
    h, w = img.shape[:2]
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
    c = np.array([(W - 1) / 2, (H - 1) / 2])

    # 方法A：老写法（先乘再加）
    out_a = warp_naive(img, ANGLE_DEG)
    # 方法B：齐次矩阵链 T(c)·R(θ)·T(-c)
    M_chain = hom(+c[0], +c[1]) @ hom(a=ANGLE_DEG) @ hom(-c[0], -c[1])
    out_b = warp_homogeneous(img, M_chain)
    # 方法C：四矩阵长链 平移·旋转·缩放·再平移
    M_long = hom(+c[0] + OFFSET[0], +c[1] + OFFSET[1]) @ hom(s=SCALE) \
        @ hom(a=-15.0) @ hom(-c[0], -c[1])
    out_c = warp_homogeneous(img, M_long)

    # 一致性验证：方法A 与 方法B 的 8bit 输出是否逐像素一致
    a8 = np.round(np.clip(out_a, 0, 1) * 255).astype(np.uint8)
    b8 = np.round(np.clip(out_b, 0, 1) * 255).astype(np.uint8)
    diff = int(np.abs(a8.astype(int) - b8.astype(int)).max())
    print(f"[verify] 方法A vs 方法B 最大像素差 = {diff}（0 表示逐像素一致）")
    print("[M] T(c)·R(30°)·T(-c) =")
    for row in M_chain:
        print("   ", np.array2string(row, precision=4, suppress_small=True))

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), dpi=150)
    ax1, ax2, ax3 = axes
    ax1.imshow(out_a); ax1.set_title("方法A：老写法（先乘再加）", fontsize=15, color="w")
    ax2.imshow(out_b); ax2.set_title("方法B：齐次矩阵链，一次乘完", fontsize=15, color="w")
    ax3.imshow(out_c); ax3.set_title("长链：T₂·S·R(−15°)·T₁", fontsize=15, color="w")
    for a in axes:
        a.axis("off")
    out_dir = pathlib.Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / "homogeneous_demo.png"
    fig.savefig(out_png, bbox_inches="tight", facecolor="#0b1220")

    # 同步一份给幻灯片引用
    slides_fig = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures" / "homogeneous_demo.png"
    slides_fig.write_bytes(out_png.read_bytes())
    print(f"saved: {out_png}")
    print(f"saved: {slides_fig}")


if __name__ == "__main__":
    main()
