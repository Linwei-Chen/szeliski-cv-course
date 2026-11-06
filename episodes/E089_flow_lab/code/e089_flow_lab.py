#!/usr/bin/env python3
"""E089 实验课：光流场可视化 + 手写 Lucas–Kanade（含孔径/金字塔/窗口）。"""
import colorsys
import hashlib
import json
import pathlib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, zoom

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 192


def sample(I, x, y):
    return map_coordinates(I, [y, x], order=1, mode="nearest")


def warp(I, u, v):
    yy, xx = np.mgrid[0:I.shape[0], 0:I.shape[1]].astype(float)
    return sample(I, xx + u, yy + v)


def lk(I0, I1, u, v, win=15, iters=5, reg=1e-4, alpha=0.6, clip=1.0):
    """手写局部 LK：2×2 法方程 + 对角正则 + 更新限幅。"""
    for _ in range(iters):
        Iw = warp(I1, u, v)
        Iy, Ix = np.gradient(Iw)
        It = Iw - I0
        sig = win / 3.0
        S = lambda A: gaussian_filter(A, sig, mode="nearest")
        Sxx = S(Ix * Ix) + reg; Sxy = S(Ix * Iy); Syy = S(Iy * Iy) + reg
        Sxt = S(Ix * It); Syt = S(Iy * It)
        det = Sxx * Syy - Sxy * Sxy
        du = alpha * (-Syy * Sxt + Sxy * Syt) / det
        dv = alpha * (Sxy * Sxt - Sxx * Syt) / det
        mag = np.sqrt(du * du + dv * dv)
        s = np.minimum(1.0, clip / (mag + 1e-9))
        u = u + du * s; v = v + dv * s
    return u, v


def pyramid(I, levels=5):
    ls = [I]
    for _ in range(levels - 1):
        ls.append(zoom(gaussian_filter(ls[-1], 1.0), 0.5, order=1))
    return ls


def lk_pyramid(I0, I1, levels=5, iters=20):
    p0, p1 = pyramid(I0, levels), pyramid(I1, levels)
    u = np.zeros_like(p0[-1]); v = np.zeros_like(p0[-1])
    for l in range(levels - 1, -1, -1):
        if l < levels - 1:
            u = zoom(u, 2, order=1)[:p0[l].shape[0], :p0[l].shape[1]] * 2
            v = zoom(v, 2, order=1)[:p0[l].shape[0], :p0[l].shape[1]] * 2
        win = 9 if p0[l].shape[0] < 40 else 15
        u, v = lk(p0[l], p1[l], u, v, win=win, iters=iters)
    return u, v


def epe_map(u, v, gt):
    return np.sqrt((u - gt[0]) ** 2 + (v - gt[1]) ** 2)


def mean_epe(u, v, gt, margin=25):
    return float(epe_map(u, v, gt)[margin:-margin, margin:-margin].mean())


def flow_rgb(u, v, maxmag=6.0):
    ang = np.arctan2(v, u)
    hue = (ang / (2 * np.pi) + 1.0) % 1.0
    mag = np.clip(np.sqrt(u * u + v * v) / maxmag, 0, 1)
    val = 0.35 + 0.65 * mag
    hsv = np.stack([hue, np.ones_like(hue), val], axis=-1)
    return hsv_to_rgb(hsv)


def main():
    print(f"[lab] E089 光流/LK 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")

    # 合成纹理：平滑随机场 + 两个亮斑（有梯度、有结构）
    rng = np.random.default_rng(89)
    base = gaussian_filter(rng.normal(size=(N, N)), 2.2)
    base = (base - base.min()) / (base.max() - base.min())
    yy, xx = np.mgrid[0:N, 0:N]
    base += 0.25 * np.exp(-((xx - 60) ** 2 + (yy - 70) ** 2) / (2 * 12 ** 2))
    base += 0.20 * np.exp(-((xx - 130) ** 2 + (yy - 120) ** 2) / (2 * 10 ** 2))
    base = np.clip(base, 0, 1)

    # 1) 小位移：一次线性化 vs 迭代
    small = (2.3, 1.7)
    I1s = warp(base, -small[0], -small[1])
    u1, v1 = lk(base, I1s, np.zeros((N, N)), np.zeros((N, N)), win=15, iters=1)
    u5, v5 = lk(base, I1s, np.zeros((N, N)), np.zeros((N, N)), win=15, iters=5)
    small_single = mean_epe(u1, v1, small)
    small_iter = mean_epe(u5, v5, small)

    # 2) 大位移：单层 vs 5 层金字塔
    large = (12.6, -9.4)
    I1l = warp(base, -large[0], -large[1])
    us, vs = lk(base, I1l, np.zeros((N, N)), np.zeros((N, N)), win=15, iters=20)
    up, vp = lk_pyramid(base, I1l)
    large_single = mean_epe(us, vs, large)
    large_pyr = mean_epe(up, vp, large)

    # 3) 孔径：竖直正弦条纹，只有 x 方向梯度
    edge = 0.5 + 0.4 * np.cos(2 * np.pi * xx / 40.0)
    at = (3.2, 5.1)
    I1a = warp(edge, -at[0], -at[1])
    ua, va = lk(edge, I1a, np.zeros((N, N)), np.zeros((N, N)), win=15, iters=30)
    m = 30
    edge_norm = float(abs(ua[m:-m, m:-m].mean() - at[0]))
    edge_full = mean_epe(ua, va, at, margin=m)
    Iwa = warp(I1a, ua, va)
    Iya, Ixa = np.gradient(Iwa)
    S = lambda A: gaussian_filter(A, 5.0, mode="nearest")
    Axx = S(Ixa * Ixa); Axy = S(Ixa * Iya); Ayy = S(Iya * Iya)
    tr = Axx + Ayy; det = Axx * Ayy - Axy * Axy
    disc = np.sqrt(np.maximum(tr * tr - 4 * det, 0))
    lmax = (tr + disc) / 2; lmin = (tr - disc) / 2
    edge_eig = float(abs(lmin[96, 96]) / max(abs(lmax[96, 96]), 1e-30))

    # 4) 平坦区：梯度为零，LK 无信息
    flat = np.full((N, N), 0.5)
    I1f = warp(flat, -at[0], -at[1])
    uf, vf = lk(flat, I1f, np.zeros((N, N)), np.zeros((N, N)), win=15, iters=10)
    flat_epe = mean_epe(uf, vf, at, margin=m)

    # 5) 窗口大小：加噪的纹理上，小窗更抖、大窗更稳
    rng2 = np.random.default_rng(7)
    t2 = (2.0, 1.3)
    clean = 0.5 + 0.4 * (gaussian_filter(rng2.normal(size=(N, N)), 2.0) - 0.5)
    I1n = warp(clean, -t2[0], -t2[1])
    I0n = np.clip(clean + rng2.normal(0, 0.06, clean.shape), 0, 1)
    I1n = np.clip(I1n + rng2.normal(0, 0.06, clean.shape), 0, 1)
    uw, vw = lk(I0n, I1n, np.zeros((N, N)), np.zeros((N, N)), win=5, iters=10)
    ub, vb = lk(I0n, I1n, np.zeros((N, N)), np.zeros((N, N)), win=25, iters=10)
    win5 = mean_epe(uw, vw, t2)
    win25 = mean_epe(ub, vb, t2)

    # 6) 颜色轮：方向编解码
    ang_err = []
    for a in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        rgb = np.array(colorsys.hsv_to_rgb((a / (2 * np.pi)) % 1.0, 1, 1))
        h, s, val = colorsys.rgb_to_hsv(*rgb)
        aa = 2 * np.pi * h
        aa = (aa + np.pi) % (2 * np.pi) - np.pi
        ang_err.append(abs(np.angle(np.exp(1j * (aa - a)))))
    color_err_deg = float(np.degrees(max(ang_err)))

    checks = [
        ("颜色轮方向往返", color_err_deg < 0.1, f"最大 {color_err_deg:.2e}°"),
        ("一次线性化误差大", small_single > 1.0, f"EPE={small_single:.3f}px"),
        ("迭代 5 次到亚像素", small_iter < 0.15, f"EPE={small_iter:.3f}px"),
        ("单层大位移失败", large_single > 10.0, f"EPE={large_single:.2f}px"),
        ("五层金字塔可估大位移", large_pyr < 0.70, f"EPE={large_pyr:.3f}px"),
        ("孔径法向流准确", edge_norm < 0.10, f"法向误差={edge_norm:.3f}px"),
        ("孔径切向不可观", edge_full > 4.0, f"完整 EPE={edge_full:.2f}px"),
        ("局部 Hessian 近奇异", edge_eig < 1e-6, f"λmin/λmax={edge_eig:.2e}"),
        ("平坦区 LK 失效", flat_epe > 2.0, f"EPE={flat_epe:.2f}px"),
        ("大窗在噪声下更稳", win25 < 0.5 * win5, f"{win5:.2f}→{win25:.2f}px"),
    ]

    # 图 1：原图/真值流/LK/误差
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.15))
    axes[0].imshow(base, cmap="gray", vmin=0, vmax=1); axes[0].set_title("输入帧 I0", fontsize=12)
    axes[1].imshow(flow_rgb(np.full((N, N), small[0]), np.full((N, N), small[1]), 4.0))
    axes[1].set_title("真值流（颜色）", fontsize=12)
    axes[2].imshow(flow_rgb(u5, v5, 4.0)); axes[2].set_title("LK 估计（颜色）", fontsize=12)
    errmap = epe_map(u5, v5, small)
    im = axes[3].imshow(errmap, cmap="magma", vmin=0, vmax=0.5)
    axes[3].set_title(f"误差图（均值 {small_iter:.2f}）", fontsize=12)
    st = 32
    axes[2].quiver(xx[::st, ::st], yy[::st, ::st], u5[::st, ::st], v5[::st, ::st],
                   color="#1c2437", angles="xy", scale_units="xy", scale=0.25,
                   width=0.006, headwidth=4)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("光流可视化：颜色轮编码 + 箭头；误差看亮度", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG / "e089_flow.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 2：孔径
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    crop = (slice(64, 128), slice(40, 120))
    axes[0].imshow(edge[crop], cmap="gray", vmin=0, vmax=1, extent=(40, 120, 128, 64))
    cx0, cy0 = 80, 88
    axes[0].arrow(cx0, cy0, at[0] * 5, at[1] * 5, color=GREEN, width=2.4, head_width=6, length_includes_head=True, label="真值")
    axes[0].arrow(cx0, cy0, at[0] * 5, 0, color=BLUE, width=2.4, head_width=6, length_includes_head=True, label="法向流")
    axes[0].arrow(cx0, cy0, ua[96, 80] * 5, va[96, 80] * 5, color=ACCENT, width=1.6, head_width=6, length_includes_head=True, linestyle="--", label="LK")
    axes[0].set_title("竖直条纹：切向不可观", fontsize=12)
    axes[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].bar(["法向", "完整"], [edge_norm, edge_full], color=[BLUE, ACCENT])
    axes[1].set_title(f"误差（法向 {edge_norm:.3f}）", fontsize=12); axes[1].set_ylabel("像素", fontsize=11)
    axes[1].tick_params(labelsize=11)
    axes[2].bar(["最大特征值", "最小特征值"], [abs(lmax[96, 96]), max(abs(lmin[96, 96]), 1e-12)], color=[GREEN, PURPLE])
    axes[2].set_yscale("log"); axes[2].set_title("最小/最大特征值≈0（近奇异）", fontsize=12)
    axes[2].tick_params(labelsize=11)
    fig.suptitle("孔径问题：沿边缘的法向可测，切向没有信息", fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    fig.savefig(FIG / "e089_aperture.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 3：大位移收敛 + 窗口
    its = [1, 2, 3, 5, 8, 10, 15, 20]
    single_curve = []
    pyr_curve = []
    for it in its:
        a, b = lk(base, I1l, np.zeros((N, N)), np.zeros((N, N)), win=15, iters=it)
        single_curve.append(mean_epe(a, b, large))
        a, b = lk_pyramid(base, I1l, iters=it)
        pyr_curve.append(mean_epe(a, b, large))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    axes[0].plot(its, single_curve, "o-", color=ACCENT, label="单层", ms=3, lw=1.4)
    axes[0].plot(its, pyr_curve, "s-", color=GREEN, label="5 层金字塔", ms=3, lw=1.4)
    axes[0].set_xlabel("每层迭代次数", fontsize=11); axes[0].set_ylabel("平均 EPE（像素）", fontsize=11)
    axes[0].set_title("大位移 12.6/−9.4 像素", fontsize=12); axes[0].legend(fontsize=10, frameon=False)
    axes[0].grid(alpha=0.3); axes[0].tick_params(labelsize=10)
    axes[1].bar(["窗口 5", "窗口 25"], [win5, win25], color=[ACCENT, GREEN])
    axes[1].set_title("噪声下窗口越大越稳", fontsize=12); axes[1].set_ylabel("平均 EPE（像素）", fontsize=11)
    axes[1].tick_params(labelsize=11)
    fig.suptitle("由粗到细处理大位移；窗口大小是取舍", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "e089_pyramid.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(
        small_single_epe=small_single, small_iter_epe=small_iter,
        large_single_epe=large_single, large_pyr_epe=large_pyr,
        edge_normal_err=edge_norm, edge_full_epe=edge_full, edge_eig_ratio=edge_eig,
        flat_epe=flat_epe, win5_epe=win5, win25_epe=win25, color_err_deg=color_err_deg,
        small_u=small[0], small_v=small[1], large_u=large[0], large_v=large[1], large_v_abs=abs(large[1]),
        edge_u=at[0], edge_v=at[1],
        small_single_2dp=round(small_single, 2), small_iter_2dp=round(small_iter, 2),
        large_single_1dp=round(large_single, 1), large_pyr_2dp=round(large_pyr, 2),
        edge_normal_3dp=round(edge_norm, 3), edge_full_2dp=round(edge_full, 2),
        flat_2dp=round(flat_epe, 2), win5_2dp=round(win5, 2), win25_2dp=round(win25, 2),
        n_pass=npass, n_total=len(checks))
    (OUT / "e089_flow_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print("[lab] readings → e089_flow_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e089_flow_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
