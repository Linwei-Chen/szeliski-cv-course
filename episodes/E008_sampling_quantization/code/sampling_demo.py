#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E008 图像即函数：采样、量化与混叠 —— 配套实验
原书：Szeliski 2nd ed. §2.3（印刷 pp.79–100）与 §3.4（印刷 pp.143–148）

五个演示：
  1) 双正弦混叠：f=3/4 与 f=5/4 按 fs=2 采样逐点相同（Figure 2.25 重绘），
     方盒宽=1（两个采样间距）滤波后幅值 +30% / −18%（图注数字；原书称 "100% fill factor
     box filter"，但严格按 100% 占空比宽 0.5 应为 +78%/+47%——deck 已加讲者注说明约定），
     且红波采样恰为蓝波采样的 −0.6 倍（图注 "inverted lower magnitude version"）。
  2) 啁啾图 4× 降采样三滤镜对比（Figure 2.26 复现）：25% 方盒 / 100% 方盒 / 9-tap。
     原书未给 9-tap 系数，本集用 Blackman 加窗 sinc（讲者选择，画面已标注）。
  3) Table 3.1（p.145）六个核的频响闭式曲线，闭式 vs np.fft 逐点对照。
  4) 量化条带：渐变图量化到 2/4/8/16 级（讲者演示；量化噪声见原书噪声清单 p.83）。
  5) DCT 8×8 分块压缩示意（式 3.62–3.63；JPEG 用 8×8，Figure 3.25）。

运行：python3 sampling_demo.py   （输出 PNG 到 code/outputs/，[verify] 行打印核对结果）
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

C_BLUE, C_RED, C_GREEN, C_ORANGE, C_GRAY = "#1f6fb4", "#c9401f", "#4a8a2e", "#e8912d", "#8a8378"

# [verify] 断言助手：每条核对带阈值判定，末尾汇总 PASS/FAIL
CHECKS = {"pass": 0, "fail": 0}


def verify(ok, msg):
    CHECKS["pass" if ok else "fail"] += 1
    print(f"[verify] {msg} [{'PASS' if ok else 'FAIL'}]")


def box_filter_1d(x, width_px):
    """沿轴做宽度为 width_px 的均匀方盒平均（reflect 补边）。x: (n,) 或 (h, w) 按行。"""
    k = np.ones(width_px) / width_px
    pad = width_px // 2
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, xp)


# ---------------------------------------------------------------- 演示一：双正弦混叠
def demo1_alias():
    fs = 2.0                      # 采样率（每单位长度 2 个样本）
    f1, f2 = 0.75, 1.25           # 蓝波 / 红波频率（原书 Figure 2.25）
    # 相位选取：x1 = +sin(2π f1 t)、x2 = −sin(2π f2 t) ⇒ 采样点重合（可解析证明）
    t = np.linspace(-0.7, 5.3, 24000)
    x1 = +np.sin(2 * np.pi * f1 * t)
    x2 = -np.sin(2 * np.pi * f2 * t)
    n = np.arange(10)             # 采样时刻 t_n = n/fs = 0, 0.5, ..., 4.5
    ts = n / fs
    s1 = +np.sin(2 * np.pi * f1 * ts)
    s2 = -np.sin(2 * np.pi * f2 * ts)
    ident = np.max(np.abs(s1 - s2))

    # 方盒宽=1（两个采样间距）→ 频响 H(f)=sinc(f)=sin(πf)/(πf)。
    # 讲者注：原书图注（p.85）称 "100% fill factor box filter" 并给出 30%/−18%，但严格按
    # 100% 占空比（宽=1 个采样间距=0.5）算出的是 +78%/+47% 且不倒相——图注数字对应宽 1，
    # deck slide 7 已加讲者注说明该约定。
    H1 = np.sinc(f1)              # np.sinc(x) = sin(πx)/(πx)
    H2 = np.sinc(f2)
    # 数值旁证：细网格上方盒平均后再采样
    dx = t[1] - t[0]
    half = int(round(0.5 / dx))
    y1 = np.array([x1[np.searchsorted(t, ti - 0.5):np.searchsorted(t, ti + 0.5)].mean() for ti in ts])
    y2 = np.array([x2[np.searchsorted(t, ti - 0.5):np.searchsorted(t, ti + 0.5)].mean() for ti in ts])
    ratio = np.max(np.abs(y2 + 0.6 * y1))     # 图注：红 = 蓝 的倒相缩小版 → −0.6 倍

    verify(ident < 1e-10, f"演示一 采样同一性 max|s1-s2| = {ident:.2e} （阈值 1e-10）")
    verify(abs(H1 - 0.3001) < 1e-3 and abs(H2 + 0.1801) < 1e-3,
           f"演示一 H(0.75)={H1:+.4f}（图注 30%）、H(1.25)={H2:+.4f}（图注 -18%）")
    verify(ratio < 1e-3, f"演示一 数值盒平均后 max|y2+0.6*y1| = {ratio:.2e}（倒相比例 -0.6）")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, sig1, sig2, ttl in (
        (axes[0], x1, x2, "采样前：两根不同的波"),
        (axes[1], None, None, "100% 方盒后：幅值 30% 与 −18%，仍难分"),
    ):
        if sig1 is None:
            sig1, sig2 = H1 * np.sin(2 * np.pi * f1 * t), -H2 * np.sin(2 * np.pi * f2 * t)
            d1, d2 = H1 * s1, H2 * s2
        else:
            d1, d2 = s1, s2
        ax.plot(t, sig1, color=C_BLUE, lw=2.0, label=r"$f=\frac{3}{4}$（蓝）")
        ax.plot(t, sig2, color=C_RED, lw=1.6, ls=(0, (4, 2)), label=r"$f=\frac{5}{4}$（红）")
        ax.plot(ts, d1, "o", color="black", ms=6, zorder=5, label="采样点")
        ax.plot(ts, d2, "o", color="black", ms=6, zorder=5)
        ax.axhline(0, color="#bbb", lw=0.8)
        ax.set_title(ttl, fontsize=12)
        ax.set_ylim(-1.35, 1.35)
        ax.set_xlabel("t", fontsize=10)
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    axes[1].text(0.03, 0.05, r"蓝 $\times 0.30$　红 $\times(-0.18)$", transform=axes[1].transAxes,
                 fontsize=10, color="#333")
    fig.suptitle("Figure 2.25（第 84–85 页）示意重绘：同一组采样，两个名字（alias）", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sampling_alias.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 演示二：啁啾三滤镜
def demo2_chirp():
    W, Hr = 1024, 240
    freq = np.linspace(0.02, 0.48, W)          # cycles/px，沿横向递增（chirp，上限低于细网格奈奎斯特 0.5）
    phase = 2 * np.pi * np.cumsum(freq)
    img = 0.5 + 0.5 * np.tile(np.sin(phase), (Hr, 1))

    ds = 4
    b25 = img[:, ::ds]                                          # 25% 填充 ≈ 点采样
    c100 = box_filter_1d(img, ds)[:, ::ds]                      # 100% 填充：4px 方盒
    # 9-tap 截断 sinc，截止 = 输出奈奎斯特 0.125 cycles/px（讲者选择；原书未给系数）
    k = np.arange(9) - 4
    fc = 1.0 / (2 * ds)
    h = 2 * fc * np.sinc(2 * fc * k)
    h /= h.sum()
    d9 = np.apply_along_axis(lambda r: np.convolve(r, h, mode="same"), 1, img)[:, ::ds]

    # 假纹度量：输出右半区（原频率 > 输出奈奎斯特）行均值信号的振荡幅度（std）
    def fake_amp(a):
        return float(a[:, a.shape[1] // 2:].mean(axis=0).std())

    p25, p100, p9 = fake_amp(b25), fake_amp(c100), fake_amp(d9)
    verify(p25 > p100 > p9 and p9 < 0.02,
           f"演示二 右半区假纹振荡幅度：25%盒 {p25:.3f} / 100%盒 {p100:.3f} / 9-tap {p9:.4f}（阈值：递减且 9-tap<0.02）")

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 2.9))
    for ax, a, ttl in (
        (axes[0], img[::8, :], "原图：频率左疏右密"),
        (axes[1], b25[::2, :], "25% 方盒（感光面小）"),
        (axes[2], c100[::2, :], "100% 方盒"),
        (axes[3], d9[::2, :], "9-tap 截断 sinc（讲者选择）"),
    ):
        ax.imshow(a, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")
    fig.suptitle("Figure 2.26（第 85–86 页）复现：4× 降采样——低质量滤镜把高频混叠成可见假纹，9-tap 彻底清除",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chirp_downsample.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 演示三：Table 3.1 频响曲线
def demo3_kernels():
    w = np.linspace(-np.pi, np.pi, 2401)
    H = {
        "box-3":     (1 + 2 * np.cos(w)) / 3,
        "box-5":     (1 + 2 * np.cos(w) + 2 * np.cos(2 * w)) / 5,
        "linear":    (1 + np.cos(w)) / 2,
        "binomial":  (1 + np.cos(w)) ** 2 / 4,
    }
    Hd = {
        "Sobel |·|": np.abs(np.sin(w)),
        "corner":    (1 - np.cos(w)) / 2,
    }
    # 闭式 vs np.fft 逐点对照（ω = 2πm/N）
    N = 64
    kernels = {
        "box-3":    np.ones(3) / 3,
        "box-5":    np.ones(5) / 5,
        "linear":   np.array([1, 2, 1]) / 4,
        "binomial": np.array([1, 4, 6, 4, 1]) / 16,
        "Sobel":    np.array([-1, 0, 1]) / 2,
        # 讲者注：Table 3.1（p.145）印 Kernel ½[−1 2 −1]，但其 Transform ½(1−cosω) 对应核
        # ¼[−1 2 −1]——原书表自身差因子 2。按 Transform 自洽取 ¼（与表 Kernel 同号）。
        "corner":   np.array([-1, 2, -1]) / 4,
    }
    closed = {
        "box-3": lambda m: (1 + 2 * np.cos(2 * np.pi * m / N)) / 3,
        "box-5": lambda m: (1 + 2 * np.cos(2 * np.pi * m / N) + 2 * np.cos(4 * np.pi * m / N)) / 5,
        "linear": lambda m: (1 + np.cos(2 * np.pi * m / N)) / 2,
        "binomial": lambda m: (1 + np.cos(2 * np.pi * m / N)) ** 2 / 4,
        "Sobel": lambda m: np.sin(2 * np.pi * m / N),
        "corner": lambda m: (1 - np.cos(2 * np.pi * m / N)) / 2,
    }
    worst = 0.0
    for name, kern in kernels.items():
        # 先补零到 N、再把核中心卷绕到索引 0，对应原书 Σ_k h(k)e^{-jkω} 的中心约定
        padded = np.zeros(N)
        padded[:len(kern)] = kern
        centered = np.roll(padded, -(len(kern) // 2))
        fft_H = np.fft.fft(centered)
        m = np.arange(N)
        # Sobel 的 DFT 含线性相位（−j sinω），原书表中给的是幅值 sin ω，故只对照幅值；
        # 对称核频响为实数，额外做带符号实部 vs 闭式对照（可检出 box 旁瓣的负号）
        if name == "Sobel":
            err = np.max(np.abs(np.abs(fft_H) - np.abs(closed[name](m))))
            verify(err < 1e-12, f"演示三 {name:9s} 幅值谱 vs |闭式| 最大偏差 = {err:.2e}（线性相位，虚部承载符号）")
        else:
            err = np.max(np.abs(np.abs(fft_H) - np.abs(closed[name](m))))
            err_re = np.max(np.abs(fft_H.real - closed[name](m)))
            sym_im = np.max(np.abs(np.imag(fft_H)))
            verify(err < 1e-12 and err_re < 1e-12,
                   f"演示三 {name:9s} 幅值/带符号实部 vs 闭式 最大偏差 = {err:.2e}/{err_re:.2e}，虚部 max {sym_im:.1e}")
        worst = max(worst, err)
    h0 = [H[k][np.argmin(np.abs(w))] for k in ("box-3", "box-5", "linear", "binomial")]
    verify(all(abs(v - 1) < 1e-12 for v in h0),
           f"演示三 四个平滑核 H(0) = {['%.3f' % v for v in h0]}（阈值：|H(0)−1|<1e-12）")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    styles = dict(box3=None)
    for name, hh in H.items():
        axes[0].plot(w, hh, lw=2.0, label=name)
    axes[0].axvline(np.pi, color=C_RED, ls="--", lw=1.2)
    axes[0].axhline(0, color="#bbb", lw=0.8)
    axes[0].text(np.pi - 0.08, 0.92, r"$\omega=\pi$（最高频）", color=C_RED, fontsize=10, ha="right")
    axes[0].annotate("box-3 在 ω=π 处 −1/3：反相残留 → 振铃",
                     xy=(3.05, -0.31), xytext=(0.35, -0.30),
                     fontsize=9.5, color="#333",
                     arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    axes[0].set_title(r"平滑四核：$H(\omega)$（Table 3.1，第 145 页）", fontsize=12)
    axes[0].set_xlabel(r"$\omega$", fontsize=10)
    axes[0].legend(fontsize=9, loc="center right")
    for name, hh in Hd.items():
        axes[1].plot(w, hh, lw=2.0, label=name)
    axes[1].axhline(0, color="#bbb", lw=0.8)
    axes[1].set_title(r"差分两核：先放大后衰减 / 只增不减", fontsize=12)
    axes[1].set_xlabel(r"$\omega$", fontsize=10)
    axes[1].legend(fontsize=9, loc="upper left")
    fig.suptitle("闭式与 np.fft 逐点一致（最大偏差 %.1e）｜原书读表：box 不均匀压高频→振铃；binomial 仍漏高频→降采样混叠；Sobel 细边缘吃亏" % worst,
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kernel_freqresp.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 演示四：量化条带
def demo4_quantize():
    h, w = 96, 480
    grad = np.tile(np.linspace(0.0, 1.0, w), (h, 1))
    levels = [2, 4, 8, 16]
    qs = []
    for L in levels:
        q = np.round(grad * (L - 1)) / (L - 1)
        qs.append(q)
        verify(len(np.unique(q)) == L, f"演示四 L={L:2d} 级：图像唯一灰阶级数 = {len(np.unique(q))}（阈值：恰为 L）")

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9))
    panels = ((grad, "连续渐变"), (qs[0], "2 级"), (qs[1], "4 级"), (qs[2], "8 级"), (qs[3], "16 级"))
    for ax, (a, ttl) in zip(axes.ravel(), panels):
        ax.imshow(a, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")
    axes.ravel()[5].axis("off")
    fig.suptitle("量化条带（讲者演示；量化噪声第 83 页、ADC 位数第 83–84 页）：级数越少，条带越粗",
                 fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "quantize_banding.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return grad


# ---------------------------------------------------------------- 演示五：DCT 8×8 分块
def dct_matrix(N):
    """正交归一 DCT 矩阵（与式 3.62 同一核 cos(π/N·(i+½)k)，差归一化因子——讲者注）。"""
    k = np.arange(N)[:, None]
    i = np.arange(N)[None, :]
    C = np.cos(np.pi / N * (i + 0.5) * k)
    C[0] *= np.sqrt(1 / N)
    C[1:] *= np.sqrt(2 / N)
    return C


def demo5_dct():
    N = 8
    C = dct_matrix(N)
    ortho = np.max(np.abs(C @ C.T - np.eye(N)))
    # 合成 96×96 图块：平滑渐变 + 圆盘 + 轻噪
    n = 96
    yy, xx = np.mgrid[0:n, 0:n]
    img = 0.35 + 0.3 * xx / n + 0.35 * ((xx - 62) ** 2 + (yy - 38) ** 2 < 18 ** 2)
    rng = np.random.default_rng(7)
    noisy = np.clip(img + 0.05 * rng.standard_normal((n, n)), 0, 1)

    def block_dct_compress(a, keep):
        rec = np.zeros_like(a)
        err_acc = []
        for r in range(0, n, N):
            for c in range(0, n, N):
                B = a[r:r + N, c:c + N]
                F = C @ B @ C.T
                idx = np.argsort(-np.abs(F), axis=None)
                mask = np.zeros((N, N), bool)
                mask.ravel()[idx[:keep]] = True
                Fq = np.where(mask, F, 0.0)
                rec[r:r + N, c:c + N] = C.T @ Fq @ C
                err_acc.append(np.mean((rec[r:r + N, c:c + N] - B) ** 2))
        return np.clip(rec, 0, 1), float(np.sqrt(np.mean(err_acc)))

    full, e_full = block_dct_compress(noisy, 64)
    k16, e16 = block_dct_compress(noisy, 16)
    k6, e6 = block_dct_compress(noisy, 6)
    k3, e3 = block_dct_compress(noisy, 3)
    verify(ortho < 1e-12, f"演示五 DCT 矩阵正交性 max|CC^T−I| = {ortho:.2e}（阈值 1e-12）")
    verify(e_full < 1e-12, f"演示五 全 64 系数重构 RMSE = {e_full:.2e}（阈值 1e-12）")
    verify(e16 < e6 < e3 and e16 < 0.05,
           f"演示五 每 8×8 块保留 16/6/3 系数：RMSE = {e16:.3f}/{e6:.3f}/{e3:.3f}（阈值：递增且 16 系数<0.05）")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.8))
    for ax, a, ttl in (
        (axes[0][0], noisy, "原图块（含噪声）"),
        (axes[0][1], k16, "每块保留 16/64 系数"),
        (axes[1][0], k6, "每块保留 6/64 系数"),
        (axes[1][1], k3, "每块保留 3/64 系数"),
    ):
        ax.imshow(a, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")
    fig.suptitle("JPEG 式 8×8 分块 DCT（式 3.62–3.63，第 147–148 页）：保留低频、丢高频；丢弃越多，块效应越明显",
                 fontsize=10.5, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dct_compress.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    demo1_alias()
    demo2_chirp()
    demo3_kernels()
    demo4_quantize()
    demo5_dct()
    n_all = CHECKS["pass"] + CHECKS["fail"]
    print(f"[verify] 汇总：{CHECKS['pass']}/{n_all} PASS" + ("" if CHECKS["fail"] == 0 else f"，{CHECKS['fail']} FAIL ←→ 有失败项，检查输出"))
    print("DONE ->", OUT)
