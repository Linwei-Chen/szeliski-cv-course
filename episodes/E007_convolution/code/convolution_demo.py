#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E007 卷积到底是什么：加权平均的数学化 —— 实验脚本（原书 §3.2，式 3.12–3.23）

四组演示：
1. 骰子伏笔（讲者补充，回收 E005 f003）：两枚独立骰子点数之和的分布 = 两个分布的卷积
2. 一维卷积三兄弟：带噪阶跃边缘过 box-5 / tent-3 / gaussian-5——压噪声、磨圆边缘（低通直觉）
3. 冲激响应：δ * h 给出 h 本形（式 3.14–3.15，核因此叫冲激响应），δ ⊗ h 给出翻转镜像（相关，式 3.12–3.13）
4. unsharp masking（式 3.22）：g_sharp = f + γ(f − h_blur * f)，模糊丢的高频补回去

另打印：三枚骰子作业③验证、可分离滤波操作数对比（K² vs 2K，式 3.20）与"外积重构 + 两趟一维 = 直接 2D"（对称核 + 非对称 v≠h 核）数值验证。
所有随机性固定种子，[verify] 行可直接核对。
产出三张图（outputs/*.png），deck 引用前需 cp 到 ../slides/figures/：
convolution_demo1.png（骰子+一维去噪）、convolution_impulse.png（冲激响应）、convolution_unsharp.png（锐化三联）。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]

DARK = "#0b1220"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(7)


def style_ax(ax):
    ax.set_facecolor(DARK)
    for s in ax.spines.values():
        s.set_color("#3b4a63")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.title.set_fontweight("bold")


def conv1d(f, h):
    """一维卷积（式 3.14 的 1D 版），居中裁剪、补零，输出与 f 等长。"""
    full = np.convolve(f, h)
    k = (len(h) - 1) // 2
    return full[k:k + len(f)]


def corr1d(f, h):
    """一维相关（式 3.12 的 1D 版：核不翻转），居中裁剪。"""
    return np.correlate(f, h, mode="same")


def separable2d(img, v, h):
    """先横后竖两趟一维卷积（式 3.20 的实现）。"""
    tmp = np.apply_along_axis(lambda r: conv1d(r, h), 1, img)
    return np.apply_along_axis(lambda c: conv1d(c, v), 0, tmp)


def naive_conv2d(img, K2):
    """直接 2D 卷积（居中窗口、翻转核、补零），供数值对照。"""
    k = K2.shape[0] // 2
    pad = np.pad(img, k)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (pad[i:i + 2 * k + 1, j:j + 2 * k + 1][::-1, ::-1] * K2).sum()
    return out


# ---------------- 核定义（图 3.14 同族） ----------------
BOX5 = np.ones(5) / 5                      # 方块核（滑动平均，p.125）
TENT3 = np.array([1, 2, 1]) / 4            # 帐篷/双线性核（p.126，图 3.14b）
GAUSS5 = np.array([1, 4, 6, 4, 1]) / 16    # 近高斯核（p.126，图 3.14c）
H_ASYM = np.array([0.5, 0.3, 0.2])         # 非对称核：演示冲激响应的翻转差


def main():
    print("[verify] ===== 卷积是什么：四组演示（§3.2） =====")

    # ---- 演示 3：冲激响应 ----
    delta = np.zeros(11); delta[5] = 1.0
    conv_h = conv1d(delta, H_ASYM)
    corr_h = corr1d(delta, H_ASYM)
    ok_conv = np.allclose(conv_h[4:7], H_ASYM)
    ok_flip = np.allclose(corr_h[4:7], H_ASYM[::-1])
    print(f"[verify] 冲激 δ（原点=1）过卷积：响应 = {conv_h[4:7].tolist()}，即 h = {H_ASYM.tolist()} 本形 → {ok_conv}（核因此叫冲激响应）")
    print(f"[verify]            过相关：响应 = {corr_h[4:7].tolist()}，即翻转镜像 {H_ASYM[::-1].tolist()} → {ok_flip}（式 3.14 与 3.12 的差别）")

    # ---- 演示 1：骰子和 = 分布的卷积（讲者补充，回收 E005 伏笔） ----
    die = np.ones(6) / 6
    two = np.convolve(die, die)
    expect = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) / 36
    print(f"[verify] 单枚骰子 P(k) = 1/6；两枚独立骰子之和 = die*die = {np.round(two*36).astype(int).tolist()}/36（概率和 = {two.sum():.4f}）")
    print(f"[verify] 与 36 种等可能组合逐一对照一致 → {np.allclose(two, expect)}；众数 7 出现 6/36 ≈ {two.max():.4f}")
    three = np.round(np.convolve(two, die) * 216).astype(int)
    print(f"[verify] 作业③ die*die 再卷一次 die：三枚之和的有序组合数 = {three.tolist()}（总 {three.sum()}）")
    print(f"[verify]   众数搬到 10 与 11 并列，各 {three[7]}/216 ≈ {three[7]/216:.4f}")

    # ---- 演示 2：一维去噪（低通直觉） ----
    n = 80
    x = np.arange(n)
    step = np.where(x >= 40, 3.0, 1.0)
    noisy = step + RNG.normal(0, 0.55, n)
    g_box, g_tent, g_gauss = conv1d(noisy, BOX5), conv1d(noisy, TENT3), conv1d(noisy, GAUSS5)
    flat = slice(15, 32)
    raw_n = float(np.std(noisy[flat] - step[flat]))
    print(f"[verify] 带噪阶跃（σ_noise=0.55）：平坦区噪声 std = {raw_n:.3f}")
    for name, g in (("box-5", g_box), ("tent-3", g_tent), ("gauss-5", g_gauss)):
        nz = float(np.std(g[flat] - step[flat]))
        print(f"[verify]   {name} → 噪声 std {nz:.3f}（压噪）")
    slope_raw = float(np.abs(np.diff(step[38:43])).max())
    print(f"[verify]   代价：台阶边缘变缓——理想台阶高差 {slope_raw:.1f}，box-5 后最大单步跳变 ≈ {np.abs(np.diff(g_box[37:44])).max():.2f}（图中可见）")

    # ---- 演示 4：unsharp masking（式 3.22） ----
    N = 200
    yy, xx = np.mgrid[0:N, 0:N]
    img = (xx / (N - 1) * 0.35 + 0.25)
    img += ((xx - 70) ** 2 + (yy - 80) ** 2 < 38 ** 2) * 0.45
    img += ((xx > 120) & (xx < 170) & (yy > 120) & (yy < 170)) * 0.30
    blur2d = separable2d(img, GAUSS5, GAUSS5)
    GAMMA = 0.7
    sharp = img + GAMMA * (img - blur2d)
    col = 100
    e0 = float(np.abs(np.diff(img[:, col])).max())
    e1 = float(np.abs(np.diff(blur2d[:, col])).max())
    e2 = float(np.abs(np.diff(sharp[:, col])).max())
    print(f"[verify] unsharp masking（式 3.22）：g = f + γ(f − h_blur*f)，γ = {GAMMA}")
    print(f"[verify]   采样列边缘跳变：原图 {e0:.3f} → 模糊 {e1:.3f} → 锐化 {e2:.3f}（高频补回，超过原图属正常过冲）")

    # ---- 可分离：操作数与数值验证 ----
    print("[verify] 可分离滤波（式 3.20）：K×K 直接卷 K² 次乘加/像素 → 先横后竖 2K 次")
    for K in (3, 5, 11, 21):
        print(f"[verify]   K = {K:2d}：K² = {K*K:3d} → 2K = {2*K:3d}（省 {100*(1-2/K):.0f}%）")
    kernel2d = np.outer(GAUSS5, GAUSS5)
    g_sep = separable2d(img, GAUSS5, GAUSS5)
    g_direct = naive_conv2d(img, kernel2d)
    print(f"[verify] 近高斯核 2D = 外积 v·hᵀ（式 3.20）：每行是 vᵢ·h 的倍数（行和 = v）→ {np.allclose(kernel2d.sum(axis=1), GAUSS5)}")
    print(f"[verify] 对称核两趟一维（先横后竖）与直接 2D 卷积逐像素一致 → {np.allclose(g_sep, g_direct)}（最大差 {np.abs(g_sep-g_direct).max():.2e}）")
    v_a, h_a = TENT3, np.array([-1.0, 0.0, 1.0]) / 2.0
    g_sep_a = separable2d(img, v_a, h_a)
    g_dir_a = naive_conv2d(img, np.outer(v_a, h_a))
    print(f"[verify] 非对称核（v≠h，Sobel 族 K = v·hᵀ）两趟 ≡ 直接 2D → {np.allclose(g_sep_a, g_dir_a)}（最大差 {np.abs(g_sep_a-g_dir_a).max():.2e}）")

    make_fig1(die, two, x, noisy, g_box, g_tent, g_gauss)
    make_fig_impulse(delta, conv_h, corr_h)
    make_fig_unsharp(img, blur2d, sharp)
    print("[verify] 图已保存：outputs/convolution_demo1.png, outputs/convolution_impulse.png, outputs/convolution_unsharp.png（deck 引用前 cp 到 ../slides/figures/）")


def make_fig1(die, two, x, noisy, g_box, g_tent, g_gauss):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.4), facecolor=DARK)
    ax = axes[0]
    style_ax(ax)
    ks = np.arange(2, 13)
    ax.bar(ks, two, width=0.6, color="#2fe0a8", label="两枚之和 = die * die")
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 0.20)
    ax.set_xlabel("点数之和"); ax.set_ylabel("概率")
    ax.set_title("两枚独立骰子之和的分布 = 卷积（讲者补充）", fontsize=11)
    ax.legend(facecolor=DARK, labelcolor="white", fontsize=9, loc="upper right")
    ax.text(0.03, 0.78, "[1 2 3 4 5 6 5 4 3 2 1] / 36", transform=ax.transAxes,
            color="#2fe0a8", fontsize=10, family="monospace")

    ax = axes[1]
    style_ax(ax)
    ax.plot(x, noisy, color="#8a93a6", lw=1.0, alpha=0.9, label="带噪阶跃（σ=0.55）")
    ax.plot(x, g_box, color="#f4b64c", lw=2.0, label="box-5（滑动平均）")
    ax.plot(x, g_tent, color="#4cc9f0", lw=2.0, label="tent-3（双线性）")
    ax.plot(x, g_gauss, color="#2fe0a8", lw=2.0, label="gauss-5（近高斯）")
    ax.axvline(40, color="#e8590c", lw=1.2, ls="--", alpha=0.8)
    ax.set_xlabel("位置"); ax.set_ylabel("信号值")
    ax.set_title("一维卷积：压噪声，边缘变缓（低通直觉）", fontsize=11)
    ax.legend(facecolor=DARK, labelcolor="white", fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "convolution_demo1.png", dpi=170, facecolor=DARK)
    plt.close(fig)


def make_fig_impulse(delta, conv_h, corr_h):
    """冲激响应单图：δ*h 本形 vs δ⊗h 镜像（deck S4 引用 convolution_impulse.png）。"""
    fig, ax = plt.subplots(figsize=(6.3, 4.8), facecolor=DARK)
    style_ax(ax)
    xs = np.arange(len(delta))
    ax.bar(xs, conv_h, width=0.42, color="#2fe0a8", label=r"$\delta \ast h$（卷积）：h 本形")
    ax.bar(xs + 0.5, corr_h, width=0.42, color="#f4b64c", label=r"$\delta \star h$（相关）：镜像")
    ax.axvline(5, color="#e8590c", lw=1.2, ls="--", alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i - 5) if abs(i - 5) <= 4 else "" for i in xs], fontsize=8)
    ax.set_xlabel("相对冲激的偏移"); ax.set_ylabel("响应")
    ax.set_title("冲激一试便知（h = [0.5, 0.3, 0.2]）", fontsize=11)
    ax.legend(facecolor=DARK, labelcolor="white", fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "convolution_impulse.png", dpi=170, facecolor=DARK)
    plt.close(fig)


def make_fig_unsharp(img, blur2d, sharp):
    """unsharp 三联单图：原图 → 模糊 → 锐化（deck S11 引用 convolution_unsharp.png）。"""
    fig = plt.figure(figsize=(6.3, 3.43), facecolor=DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(DARK); ax.axis("off")
    ax.set_title("unsharp masking（式 3.22，γ = 0.7）：原图 → 模糊 → 锐化",
                 color="white", fontsize=11, fontweight="bold", pad=10)
    for k, (im, lab) in enumerate([(img, "原图 f"), (blur2d, "模糊 h*f"), (sharp, "锐化 g")]):
        axw = fig.add_axes([0.06 + k * 0.315, 0.12, 0.26, 0.68])
        axw.imshow(im, cmap="magma", vmin=0, vmax=1.15)
        axw.set_xticks([]); axw.set_yticks([])
        for s in axw.spines.values():
            s.set_color("#3b4a63")
        axw.set_title(lab, color="white", fontsize=10, pad=6)
    fig.savefig(OUT / "convolution_unsharp.png", dpi=170, facecolor=DARK)
    plt.close(fig)


if __name__ == "__main__":
    main()
