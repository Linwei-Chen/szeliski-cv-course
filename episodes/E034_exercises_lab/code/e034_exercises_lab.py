#!/usr/bin/env python3
"""E034 实验课（§3.8 选解）S2 数值实验：5 组实验+区分力注入。

选题台账：本集不新增布置；兑现 Ch3 台账 7 题号（Ex 3.9/3.10/3.12/3.13/3.18/3.21/3.28），
Ex 3.5 第 4 步与 Ex 3.25 继续有效（理由见 status.s2_selection），Ex 4.3 属第 4 章。
跨集复用：E023/E024/E025/E031 实验代码经 ast 逐函数加载（不触发其实验正文），
加载缺失即断言失败——对照物是前面各集跑过门审的同一实现。
红线 2：本节原书无图表——3 张程序生成图均标「程序生成，本节原书无图表」。
"""
import ast
import pathlib
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

HERE = pathlib.Path(__file__).resolve().parent
COURSE = HERE.parents[1]          # episodes/
OUT = HERE / "outputs"
FIG = HERE.parent / "slides" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
results, R, FAILS = [], [], []


def check(rid, label, ok, detail=""):
    results.append((rid, label, ok, detail))
    if not ok:
        FAILS.append(rid)
    print(f"  [{'PASS' if ok else 'FAIL'}] {rid} {label}" + (f"  {detail}" if detail else ""))


def reading(label, value):
    R.append((label, value))
    print(f"  [读数] {label}: {value}")


def savefig(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_funcs(path, names, assigns=(), extra=None):
    """按 ast 抽取兄弟集代码中的指定函数/常量定义执行加载（不触发其实验正文）。"""
    src = pathlib.Path(path).read_text()
    keep = []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(ast.get_source_segment(src, node))
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in assigns:
                    keep.append(ast.get_source_segment(src, node))
    g = {"np": np}
    if extra:
        g.update(extra)
    exec(compile("\n\n".join(keep), path, "exec"), g)
    missing = [n for n in tuple(names) + tuple(assigns) if n not in g]
    assert not missing, f"函数加载失败: {missing} <- {path}"
    return g


E023 = load_funcs(COURSE / "E023_linfilter" / "code" / "e023_linfilter.py",
                  ["correlate2d", "convolve2d", "pad_image"])
E024 = load_funcs(COURSE / "E024_separable" / "code" / "e024_separable.py",
                  ["pad_along_axis", "sep_two_pass"])
E025 = load_funcs(COURSE / "E025_nonlinear" / "code" / "e025_nonlinear.py",
                  ["pad_reflect", "window_view", "median_filter", "gaussian_kernel1d",
                   "gaussian_filter_sep", "bilateral_filter"])
E031 = load_funcs(COURSE / "E031_wavelets_blending" / "code" / "e031_wavelets_blending.py",
                  ["lift_analyze", "lift_synth", "wavelet2d", "iwavelet2d"],
                  extra={"convolve": np.convolve})
print("跨集加载：E023 correlate2d/convolve2d/pad_image；E024 pad_along_axis/sep_two_pass；"
      "E025 median/gaussian/bilateral 全套；E031 lifting 小波全套——加载成功")

rng = np.random.default_rng(42)
BIN5 = np.array([1, 4, 6, 4, 1]) / 16.0
MODES6 = ("zero", "constant", "clamp", "wrap", "mirror", "extend")


def psnr(a, b, peak=1.0):
    return float(10 * np.log10(peak ** 2 / np.mean((a - b) ** 2)))


def blur_axis_clamp(im, ker, axis):
    """逐轴 clamp padding + 一维相关（DC 保持：clamp 补全下 Σk·1=Σk=1）。"""
    p = len(ker) // 2
    pad = [(p, p) if a == axis else (0, 0) for a in range(im.ndim)]
    g = np.pad(im, pad, mode="edge")
    out = np.zeros_like(im, dtype=float)
    n = im.shape[axis]
    for i, kv in enumerate(ker):
        sl = [slice(None)] * im.ndim
        sl[axis] = slice(i, i + n)
        out = out + kv * g[tuple(sl)]
    return out


def gauss_pyr_clamp(img, levels):
    pyr = [img.astype(float)]
    for _ in range(levels):
        im = blur_axis_clamp(blur_axis_clamp(pyr[-1], BIN5, 1), BIN5, 0)
        pyr.append(im[::2, ::2])
    return pyr


def expand_clamp(im):
    up = np.zeros((im.shape[0] * 2, im.shape[1] * 2))
    up[::2, ::2] = im
    return blur_axis_clamp(blur_axis_clamp(up, 2 * BIN5, 1), 2 * BIN5, 0)


def lap_pyr_clamp(img, levels):
    g = gauss_pyr_clamp(img, levels)
    lap = [g[i] - expand_clamp(g[i + 1]) for i in range(levels)]
    lap.append(g[-1])
    return lap


def collapse_clamp(lap):
    cur = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        cur = expand_clamp(cur) + lap[i]
    return cur


def blur_axis_zero(im, ker, axis):
    """逐轴 zero padding + 一维相关（对照口径：DC 在边界亏空）。"""
    p = len(ker) // 2
    pad = [(p, p) if a == axis else (0, 0) for a in range(im.ndim)]
    g = np.pad(im, pad, mode="constant")
    n = g.shape[axis] - len(ker) + 1
    out = np.zeros([n if a == axis else im.shape[a] for a in range(im.ndim)])
    for i, kv in enumerate(ker):
        sl = [slice(None)] * im.ndim
        sl[axis] = slice(i, i + n)
        out = out + kv * g[tuple(sl)]
    return out


def synth_base(N=128):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    img = 0.5 + 0.12 * np.sin(2 * np.pi * xx / 32.0) + 0.10 * np.sin(2 * np.pi * yy / 24.0)
    img += 0.25 * ((xx - 84) ** 2 + (yy - 40) ** 2 < 18 ** 2)      # 圆盘
    img += 0.15 * (((xx // 6) + (yy // 6)) % 2) * (xx > 96)        # 右侧条纹块
    return np.clip(img, 0, 1)


print("口径注记（S2 标头，refs 引用一律指向本存档）：")
print("· 选题台账：Ch2=E020 两道手算题（Y'601/YCbCr、Bayer 绿双倍）为手算/概念题不上机；"
      "Ch3 台账 9 题号（Ex3.5 第 4 步/3.9/3.10/3.12/3.13/3.18/3.21/3.25/3.28，四源一致），"
      "本集兑现 7 题号（3.9/3.10/3.12/3.13/3.18/3.21/3.28），Ex3.5 第 4 步与 Ex3.25 继续有效，"
      "Ex4.3 属第 4 章。本集不新增布置。")
print("· 卷积/相关口径：E023 correlate2d=相关（式 3.12），E024 sep_two_pass=卷积口径（核翻转），"
      "对照时核取向已对齐；PSNR 峰值=1.0（0–1 浮点口径）；噪声为加性高斯（rng 种子 42）。")
print("· 金字塔口径：binomial 核 1/16(1,4,6,4,1)（Burt–Adelson），mask/边界一律 clamp padding"
      "（DC 保持——Σw≡1 的前提，见 #5 裁决）；小波=2-tap lifting（周期延拓，E031 同栈）。")
print("· 红线 2：本节原书无图表——3 张程序生成图（steerable 响应/去噪对比/标签融合）图内均标"
      "「程序生成，本节原书无图表」。")

# ══ #1 Ex 3.9+3.10（L114/L132；E023/E024 布置）══
print("\n#1 Ex 3.9 padding + Ex 3.10 可分离滤波（E023/E024 布置兑现）")
f_img = rng.uniform(0, 1, (5, 7))


def my_pad(f, mode, C=0.0, p=1):
    """按 Ex 3.9 题意独立重写六模式（对照物=E023 pad_image）。"""
    M, N = f.shape
    if mode == "zero":
        return np.pad(f, p, constant_values=0.0)
    if mode == "constant":
        return np.pad(f, p, constant_values=C)
    if mode == "clamp":
        ii = np.clip(np.arange(-p, M + p), 0, M - 1)
        jj = np.clip(np.arange(-p, N + p), 0, N - 1)
        return f[np.ix_(ii, jj)]
    if mode == "wrap":
        ii = np.arange(-p, M + p) % M
        jj = np.arange(-p, N + p) % N
        return f[np.ix_(ii, jj)]
    if mode == "mirror":
        idx_i = np.arange(-p, M + p)
        ii = np.where(idx_i < 0, -idx_i, np.where(idx_i > M - 1, 2 * (M - 1) - idx_i, idx_i))
        idx_j = np.arange(-p, N + p)
        jj = np.where(idx_j < 0, -idx_j, np.where(idx_j > N - 1, 2 * (N - 1) - idx_j, idx_j))
        return f[np.ix_(ii, jj)]
    if mode == "extend":
        g = np.pad(f, p, mode="edge")
        for axis, n in ((0, M), (1, N)):
            lo = np.take(g, [p] * p, axis=axis)
            lo_m = np.take(g, list(range(2 * p, p, -1)), axis=axis)
            hi = np.take(g, [n - 1 + p] * p, axis=axis)
            hi_m = np.take(g, list(range(n - 2 + p, n - 2, -1)), axis=axis)
            sl = [slice(None)] * 2
            sl[axis] = slice(0, p)
            g[tuple(sl)] = 2 * lo - lo_m
            sl[axis] = slice(g.shape[axis] - p, g.shape[axis])
            g[tuple(sl)] = 2 * hi - hi_m
        return g
    raise ValueError(mode)


bad = [m for m in MODES6 if not np.array_equal(my_pad(f_img, m, 0.25, 2),
                                               E023["pad_image"](f_img, m, 0.25, 2))]
check("1-1", "六模式 padding 独立实现 vs E023 pad_image 逐位一致（p=2，C=0.25）",
      not bad, f"不一致: {bad}" if bad else "6/6")
pp2 = 2
kk_i = np.clip(np.arange(-pp2, f_img.shape[0] + pp2), 0, f_img.shape[0] - 1)
ll_j = np.clip(np.arange(-pp2, f_img.shape[1] + pp2), 0, f_img.shape[1] - 1)
padded_formula = f_img[np.ix_(kk_i, ll_j)]
check("1-2", "Ex 3.9 replication 公式 k=max(0,min(M−1,i))、l=max(0,min(N−1,j)) 生成的整块 padding "
             "==clamp padding（本实现+E023 实现+np.pad edge 三方逐位）",
      np.array_equal(padded_formula, np.pad(f_img, pp2, mode="edge"))
      and np.array_equal(padded_formula, my_pad(f_img, "clamp", 0.0, pp2))
      and np.array_equal(padded_formula, E023["pad_image"](f_img, "clamp", 0.0, pp2)))

K1 = np.array([1, 2, 1]) / 4.0
K2 = np.array([-1.0, 0, 1])
K2D_SYM = np.outer(BIN5, BIN5)
K2D_ASY = np.outer(K1, K2)


def my_two_pass_corr(f, k1, k2, mode, C=0.0):
    """可分离两趟（相关口径）：先横向 k2 再纵向 k1，padding 用 E023 pad_image。
    要求 len(k1)==len(k2)；纵向直接对横向结果做 valid 相关（padding 已含在横向 pad 里）。"""
    p = len(k2) // 2
    gp = E023["pad_image"](f.astype(float), mode, C, p)      # (M+2p, N+2p)
    tmp = np.stack([np.convolve(row, k2[::-1], mode="valid") for row in gp])   # (M+2p, N)
    return np.stack([np.convolve(col, k1[::-1], mode="valid") for col in tmp.T]).T  # (M, N)


ok13, worst = True, 0.0
for mode in MODES6:
    for (ky, kx, K2D) in ((BIN5, BIN5, K2D_SYM), (K1, K2, K2D_ASY)):
        pd = K2D.shape[0] // 2
        direct = E023["correlate2d"](E023["pad_image"](f_img, mode, 0.25, pd), K2D)
        mine = my_two_pass_corr(f_img, ky, kx, mode, 0.25)
        e = float(np.max(np.abs(direct - mine)))
        worst = max(worst, e)
        ok13 &= e < 1e-12
check("1-3a", "我的两趟（相关口径）vs E023 correlate2d 直插逐位等价：binomial 外积+Sobel 可分离核 [1,2,1]/4⊗[−1,0,1]，六模式全图",
      ok13, f"最大误差 {worst:.2e}")
d_const = float(np.max(np.abs(
    E023["convolve2d"](E023["pad_image"](f_img, "constant", 0.25, 1), np.outer(K2, K2))
    - E024["sep_two_pass"](f_img, K2, "constant", 0.25))))
reading("边界补全可分离性发现", f"constant-C 补全不可分离：E024 逐轴各补一次 vs 直插双轴一次补全，"
        f"角部差 {d_const:.2e}（=C 本身）；zero/mirror 可分离故全等——E024 对照限 zero/mirror")
ok13b, worstb = True, 0.0
for mode, Cb in (("zero", 0.0), ("mirror", 0.0)):
    for k1d in (BIN5, K2):
        conv_direct = E023["convolve2d"](E023["pad_image"](f_img, mode, Cb, len(k1d) // 2),
                                         np.outer(k1d, k1d))
        e024out = E024["sep_two_pass"](f_img, k1d, mode, Cb)
        e = float(np.max(np.abs(conv_direct - e024out)))
        worstb = max(worstb, e)
        ok13b &= e < 1e-12
check("1-3b", "E024 sep_two_pass（卷积口径）vs E023 convolve2d 直插逐位一致（zero/mirror ×两核；"
              "constant-C 见上条发现：逐轴补全不可分离，不作等价对照）",
      ok13b, f"最大误差 {worstb:.2e}")
f512 = rng.uniform(0, 1, (512, 512))
t0 = time.perf_counter()
for _ in range(3):
    E023["correlate2d"](np.pad(f512, 2, mode="edge"), K2D_SYM)
t_direct = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(3):
    blur_axis_clamp(blur_axis_clamp(f512, BIN5, 1), BIN5, 0)
t_two = time.perf_counter() - t0
reading("512² binomial 5×5 耗时（3 次取和）", f"直接二维 {t_direct:.3f}s vs 两趟 {t_two:.3f}s "
        f"（比值 {t_direct / max(t_two, 1e-9):.2f}×）")
check("1-4", "可分离两趟不慢于直接二维（5×5 核，O(K²)→O(2K)）", t_two < t_direct,
      f"{t_two:.3f} < {t_direct:.3f}")

# ══ #2 Ex 3.13 steerable filters（L171；E024 布置）══
print("\n#2 Ex 3.13 steerable filters（E024 布置兑现：一阶两基+二阶三基+特征定位）")
sig, rad = 1.4, 4
ys, xs = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(float)
g0 = np.exp(-(xs ** 2 + ys ** 2) / (2 * sig ** 2))
KX = (-xs / sig ** 2) * g0        # ∂/∂x 高斯（一阶基 1）
KY = (-ys / sig ** 2) * g0        # ∂/∂y 高斯（一阶基 2）
KXX = (xs ** 2 - sig ** 2) / sig ** 4 * g0
KYY = (ys ** 2 - sig ** 2) / sig ** 4 * g0
KXY = (xs * ys / sig ** 4) * g0


def corr_same(f, k):
    return E023["correlate2d"](np.pad(f, rad, mode="edge"), k)


def k1_theta(th):
    """θ 方向导数解析核：−∂/∂(x cosθ+y sinθ) 高斯。"""
    c, s = np.cos(th), np.sin(th)
    return (-(xs * c + ys * s) / sig ** 2) * g0


err_k = max(float(np.max(np.abs(k1_theta(np.deg2rad(t))
                                    - (np.cos(np.deg2rad(t)) * KX + np.sin(np.deg2rad(t)) * KY))))
            for t in (0, 30, 45, 60, 90))
check("2-1", "一阶 steer 核恒等式：θ 方向解析核=cosθ·kx+sinθ·ky（θ∈{0,30,45,60,90}°）",
      err_k < 1e-15, f"err={err_k:.1e}")
f_r = rng.uniform(0, 1, (64, 64))
err_r = 0.0
for t in (0, 30, 45, 60, 90):
    th = np.deg2rad(t)
    direct = corr_same(f_r, k1_theta(th))
    comb = np.cos(th) * corr_same(f_r, KX) + np.sin(th) * corr_same(f_r, KY)
    err_r = max(err_r, float(np.max(np.abs(direct - comb))))
check("2-2", "一阶响应恒等式 R(θ)=cosθ·Rx+sinθ·Ry（随机图，全图）", err_r < 1e-12, f"err={err_r:.1e}")
th45 = np.deg2rad(45.0)
direct2 = corr_same(f_r, np.cos(th45) ** 2 * KXX + 2 * np.cos(th45) * np.sin(th45) * KXY
                    + np.sin(th45) ** 2 * KYY)
comb2 = (np.cos(th45) ** 2 * corr_same(f_r, KXX)
         + 2 * np.cos(th45) * np.sin(th45) * corr_same(f_r, KXY)
         + np.sin(th45) ** 2 * corr_same(f_r, KYY))
check("2-3", "二阶三基 steer 恒等式 R₂(45°)=cos²θ·Rxx+2cosθsinθ·Rxy+sin²θ·Ryy",
      float(np.max(np.abs(direct2 - comb2))) < 1e-12,
      f"err={float(np.max(np.abs(direct2 - comb2))):.1e}")
N2 = 96
yy2, xx2 = np.mgrid[0:N2, 0:N2].astype(float)
phi = np.deg2rad(45.0)
d_perp = (xx2 - N2 / 2) * np.sin(phi) - (yy2 - N2 / 2) * np.cos(phi)
line45 = np.exp(-d_perp ** 2 / (2 * 2.5 ** 2))          # 45° 高斯剖面亮线
thetas = np.deg2rad(np.arange(0, 180, 5))
resp = np.array([np.max(np.abs(corr_same(line45, k1_theta(t)))) for t in thetas])
peak_deg = int(np.rad2deg(thetas[int(np.argmax(resp))]))
reading("45° 线方向响应峰值角度（线法向）", f"{peak_deg}°（理论 135°）")
check("2-4", "45° 线的 steerable 方向响应峰值=线法向 135°±5°", 130 <= peak_deg <= 140, f"峰值 {peak_deg}°")
d1 = (xx2 - 48) * np.sin(phi) - (yy2 - 48) * np.cos(phi)
d2 = (xx2 - 48) * np.sin(np.deg2rad(135)) - (yy2 - 48) * np.cos(np.deg2rad(135))
cross = np.exp(-d1 ** 2 / (2 * 2.5 ** 2)) + np.exp(-d2 ** 2 / (2 * 2.5 ** 2))   # 45°+135° 十字


def peak_dir(pt):
    rs = {int(np.rad2deg(t)): float(np.abs(corr_same(cross, k1_theta(t)))[pt])
          for t in np.deg2rad(np.arange(0, 180, 5))}
    return rs, max(rs, key=rs.get)


rs45, pk45 = peak_dir((57, 55))     # 45° 臂旁 1.4px（臂线上梯度恒零，峰值在臂侧）
rs135, pk135 = peak_dir((41, 57))   # 135° 臂旁 1.4px
rs_c = {int(np.rad2deg(t)): float(np.abs(corr_same(cross, k1_theta(t)))[48, 48])
        for t in np.deg2rad(np.arange(0, 180, 5))}
reading("十字交点：45° 臂峰值方向", f"{pk45}°（理论=线法向 135°）")
reading("十字交点：135° 臂峰值方向", f"{pk135}°（理论=线法向 45°）")
reading("十字交点中心（对称点）全向最大响应",
        f"{max(rs_c.values()):.2e}（两臂峰值的 "
        f"{max(rs_c.values()) / max(max(rs45.values()), max(rs135.values())):.1e} 倍）")
check("2-5", "十字交点特征：45° 臂峰值方向=135°±5°、135° 臂峰值方向=45°±5°（两方向并存即交点指纹）",
      130 <= pk45 <= 140 and 40 <= pk135 <= 50, f"{pk45}°/{pk135}°")
bands = [corr_same(line45, k1_theta(t)) for t in np.deg2rad((0, 45, 90))]
check("2-6", "多带输出账（题面 multi-banded image：0°+90° 两条一阶基带）：0°/45°/90° 三方向响应带与输入同尺寸",
      all(b.shape == line45.shape for b in bands), f"{[b.shape for b in bands]}")

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax[0].imshow(line45, cmap="gray")
ax[0].set_title("合成 45° 亮线（输入）")
ax[1].plot(np.rad2deg(thetas), resp, marker="o", ms=3)
ax[1].axvline(135, ls="--", lw=1, color="crimson")
ax[1].set_xlabel("方向 θ（度）")
ax[1].set_ylabel("方向响应幅值（最大值）")
ax[1].set_title("一阶可导向滤波方向响应（峰值 135°）")
fig.suptitle("可导向滤波方向定位实验（程序生成，本节原书无图表）", fontsize=11)
fig.tight_layout()
savefig(fig, "e034_steerable_response.png")

# ══ #3 Ex 3.12+3.28（L165/L367；E025/E028 布置）══
print("\n#3 Ex 3.12 软化/锐化/去噪 + Ex 3.28 去噪对比（E025/E028 布置兑现）")
base = synth_base()
noisy = {s: np.clip(base + rng.normal(0, s, base.shape), 0, 1) for s in (0.04, 0.10)}
res_rows = []
for s in (0.04, 0.10):
    pn = psnr(base, noisy[s])
    row = {"噪声": pn,
           "高斯": psnr(base, E025["gaussian_filter_sep"](noisy[s], 1.0, 3)),
           "中值": psnr(base, E025["median_filter"](noisy[s], 3)),
           "双边(σr匹配)": psnr(base, E025["bilateral_filter"](noisy[s], 1.8, s, radius=4)),
           "双边(σr×2)": psnr(base, E025["bilateral_filter"](noisy[s], 1.8, 2 * s, radius=4))}
    res_rows.append(row)
    reading(f"σ={s} 各法 PSNR(dB)", {k: round(v, 2) for k, v in row.items()})
check("3-1", "两档噪声下三种滤波器 PSNR 全部高于含噪图（3 法×2 档）",
      all(row[k] > row["噪声"] for row in res_rows for k in ("高斯", "中值", "双边(σr匹配)")))
d_b = abs(res_rows[1]["双边(σr匹配)"] - res_rows[1]["双边(σr×2)"])
reading("bilateral 噪声估计敏感性（σ=0.10，σr 匹配 vs ×2 失配）", f"ΔPSNR={d_b:.2f} dB")
check("3-2", "题面问句裁决：bilateral 性能依赖噪声水平估计（σr 失配 PSNR 可测变化>0.05dB），"
              "Gaussian/median 无噪声参数不依赖", d_b > 0.05, f"Δ={d_b:.2f}dB")

# ══ #4 Ex 3.21（L265；E028 布置，两任务选一=去噪 coring）══
print("\n#4 Ex 3.21 小波 coring 去噪（E028 布置兑现；复用 E031 lifting 栈）")
yy4, xx4 = np.mgrid[0:128, 0:128].astype(float)
base_s = np.clip(0.40 + 0.25 * np.exp(-((xx4 - 45) ** 2 + (yy4 - 40) ** 2) / (2 * 16 ** 2))
                 + 0.30 * np.exp(-((xx4 - 90) ** 2 + (yy4 - 85) ** 2) / (2 * 20 ** 2)), 0, 1)
reading("coring 实验输入", "平滑合成图（双亮斑+缓变）——coring 适合低频为主信号；高频丰富图"
        "（#3 条纹图）上阈值会伤结构，此设计选择在此申报")
nz = np.clip(base_s + rng.normal(0, 0.10, base_s.shape), 0, 1)
det, low = E031["wavelet2d"](nz, levels=2)
w_noise = rng.standard_normal((128, 128))
det_w, _ = E031["wavelet2d"](w_noise, levels=1)
G2 = float(np.std(det_w[0]["HH"]))
reading("HH 带噪声增益 G₂（讲者复算：白噪声过行列两级 lifting 高通）",
        f"{G2:.4f}（理论 1.5=[√1.5]²，行核 [−½,1,−½]×列核 [−½,1,−½]）")
sig_hat = float(np.median(np.abs(det[0]["HH"])) / (0.6745 * G2))
reading("噪声水平稳健估计 σ̂（HH 带 MAD/(0.6745·G₂)）", f"{sig_hat:.4f}（真值 0.10）")
check("4-1", "σ̂ MAD 估计（含带内增益归正）与真值偏差<0.03", abs(sig_hat - 0.10) < 0.03,
      f"σ̂={sig_hat:.4f}")


def core_c0(w_, T):
    """分段线性 C⁰ 软阈值（原书「piecewise linear or other C⁰ function」）。"""
    return np.sign(w_) * np.maximum(np.abs(w_) - T, 0.0)


det_c = [{k: (core_c0(v, sig_hat) if k != "LL" else v) for k, v in d.items()}
         for d in det]   # LL=低通子带不 coring（原书 coring 只压小系数细节带）
rec = np.clip(E031["iwavelet2d"](det_c, low), 0, 1)
p_noisy, p_cored = psnr(base_s, nz), psnr(base_s, rec)
p_bil = psnr(base_s, E025["bilateral_filter"](nz, 1.8, 0.10, radius=4))
reading("coring 前后 PSNR(dB)", f"含噪 {p_noisy:.2f} → coring {p_cored:.2f}（+{p_cored - p_noisy:.2f}）；"
        f"同输入 bilateral {p_bil:.2f}")
check("4-2", "coring 重构 PSNR 较含噪图提升>1dB", p_cored - p_noisy > 1.0,
      f"+{p_cored - p_noisy:.2f}dB")
check("4-3", "3.28 汇聚读数：coring 与 bilateral（平滑图同输入）均有效（均>含噪图），孰优以读数为准（不预设）",
      p_cored > p_noisy and p_bil > p_noisy, f"coring {p_cored:.2f} vs bilateral {p_bil:.2f}")

fig, ax = plt.subplots(figsize=(7.6, 3.8))
names = ["含噪", "高斯", "中值", "双边(σr匹配)", "双边(σr×2)", "小波 coring"]
g_s = float(psnr(base_s, E025["gaussian_filter_sep"](nz, 1.0, 3)))
m_s = float(psnr(base_s, E025["median_filter"](nz, 3)))
b2_s = float(psnr(base_s, E025["bilateral_filter"](nz, 1.8, 0.20, radius=4)))
reading("平滑合成图六法 PSNR(dB)（#4 出图读数，σ=0.10；双边×2=σr 0.20）",
        f"含噪 {p_noisy:.2f}/高斯 {g_s:.2f}/中值 {m_s:.2f}/"
        f"双边匹配 {p_bil:.2f}/双边×2 {b2_s:.2f}/coring {p_cored:.2f}")
vals = [p_noisy, g_s, m_s, p_bil, b2_s, p_cored]
bars = ax.bar(names, vals, color=["#999999", "#4c72b0", "#4c72b0", "#dd8452", "#c44e52", "#55a868"])
ax.bar_label(bars, fmt="%.2f", fontsize=9)
ax.set_ylabel("PSNR（dB，峰值 1.0）")
ax.set_title("去噪方法对比：平滑合成图 σ=0.10（程序生成，本节原书无图表）")
ax.set_ylim(min(vals) - 1.5, max(vals) + 0.8)
fig.tight_layout()
savefig(fig, "e034_denoise_compare.png")

# ══ #5 Ex 3.18（L225；E031 布置变体：整数标签图融合 3 图）══
print("\n#5 Ex 3.18 金字塔融合：整数标签图三图融合+总权重裁决（E031 布置兑现）")
N5 = 128
yy5, xx5 = np.mgrid[0:N5, 0:N5].astype(float)
A1 = np.clip(0.45 + 0.2 * (((xx5 // 8) % 2) * ((yy5 // 8) % 2)), 0, 1)      # 棋盘块
A2 = np.clip(0.35 + 0.4 * np.exp(-((xx5 - 40) ** 2 + (yy5 - 88) ** 2) / (2 * 14 ** 2)), 0, 1)
A3 = np.clip(0.30 + 0.3 * np.sin(2 * np.pi * yy5 / 16.0) * (xx5 / N5), 0, 1)
labels = np.ones((N5, N5), int)
labels[:, 43:86] = 2
labels[:, 86:] = 3
LV = 2
laps = {i: lap_pyr_clamp(src, LV) for i, src in ((1, A1), (2, A2), (3, A3))}
masks = {i: gauss_pyr_clamp((labels == i).astype(float), LV) for i in (1, 2, 3)}
sw_max_dev = max(float(np.max(np.abs(sum(masks[i][lv] for i in (1, 2, 3)) - 1.0)))
                 for lv in range(LV + 1))
check("5-1", "clamp padding 下 mask 高斯金字塔逐层 Σw≡1（max|Σw−1|<1e-12，LV+1 层全查）",
      sw_max_dev < 1e-12, f"max偏差 {sw_max_dev:.1e}")
m_all1 = gauss_pyr_clamp(np.ones((N5, N5)), LV)
out_deg = collapse_clamp([laps[1][i] * m_all1[i] for i in range(LV + 1)])
e_deg = float(np.max(np.abs(out_deg - A1)))
check("5-2", "退化完全重构：全标签=1（mask≡1）时融合输出==A1（<1e-9）", e_deg < 1e-9, f"err={e_deg:.1e}")
out5 = collapse_clamp([sum(masks[i][lv] * laps[i][lv] for i in (1, 2, 3)) for lv in range(LV + 1)])
blk = {"区1(中心)": (8, 24), "区2(中心)": (60, 70), "区3(中心)": (104, 120)}
errs5 = {}
for name, (r0, r1) in blk.items():
    src = {1: A1, 2: A2, 3: A3}[labels[r0, r0]]
    errs5[name] = float(np.max(np.abs(out5[r0:r1, r0:r1] - src[r0:r1, r0:r1])))
check("5-3", "三图融合：各区域内部块逐位恢复原图（max err<1e-9；LV=2 级联核支撑半径 R2=6px，块距标签边界≥17px）",
      max(errs5.values()) < 1e-9, " ".join(f"{k}={v:.1e}" for k, v in errs5.items()))


def gauss_pyr_zero(img, levels):
    pyr = [img.astype(float)]
    for _ in range(levels):
        im = blur_axis_zero(blur_axis_zero(pyr[-1], BIN5, 1), BIN5, 0)
        pyr.append(im[::2, ::2])
    return pyr


masks_zero = {i: gauss_pyr_zero((labels == i).astype(float), LV) for i in (1, 2, 3)}
swz_lv = {lv: sum(masks_zero[i][lv] for i in (1, 2, 3)) for lv in range(1, LV + 1)}
sw_min = min(float(sw.min()) for sw in swz_lv.values())
n_def = int(np.sum(swz_lv[1] < 1 - 1e-6))
reading("zero padding 变体模糊级 Σw（level 1–2）",
        f"min={sw_min:.4f}；level1（{swz_lv[1].shape[0]}×{swz_lv[1].shape[1]}）亏空像素 "
        f"{n_def}/{swz_lv[1].size}（{100 * n_def / swz_lv[1].size:.1f}%）——边界带")
check("5-4", "原书问句裁决：Σw 是否恒 1 取决于边界 padding——clamp 口径 math just works out"
              "（5-1 已证）；zero 口径边界亏空需 renormalize（亏空像素>0）", n_def > 0,
      f"min Σw={sw_min:.4f}")
tr = {lv: int(np.sum((masks[1][lv][masks[1][lv].shape[0] // 2] > 0.1)
                     & (masks[1][lv][masks[1][lv].shape[0] // 2] < 0.9)))
      for lv in (1, 2)}
reading("mask 过渡带宽度（level1/2 中行，0.1<m<0.9）", f"{tr} 列（binomial 5-tap，E031 同口径）")

fig, ax = plt.subplots(2, 3, figsize=(11.4, 7.0))
for a, (im, t, cm) in zip(ax.flat, [(A1, "源图 1（棋盘块）", "gray"),
                                    (A2, "源图 2（亮斑）", "gray"),
                                    (A3, "源图 3（渐变条纹）", "gray"),
                                    (labels, "整数标签图（1/2/3）", "viridis"),
                                    (out5, "拉普拉斯金字塔融合输出", "gray")]):
    a.imshow(im, cmap=cm)
    a.set_title(t, fontsize=10)
    a.set_xticks([])
    a.set_yticks([])
err_img = np.zeros((N5, N5))
for i in (1, 2, 3):
    m = masks[i][0] > 0.999
    err_img[m] = np.abs(out5[m] - {1: A1, 2: A2, 3: A3}[i][m])
ax.flat[5].imshow(err_img, cmap="magma", vmin=0, vmax=1e-9)
ax.flat[5].set_title("区域内部恢复误差（≤1e-9）", fontsize=10)
ax.flat[5].set_xticks([])
ax.flat[5].set_yticks([])
fig.suptitle("整数标签图三图金字塔融合（程序生成，本节原书无图表）", fontsize=12)
fig.tight_layout()
savefig(fig, "e034_label_blend.png")

# ══ 区分力注入（不计数；每条证明机检能拦）══
print("\n区分力注入（6 条，不计数）")
D = []
bad_two = my_two_pass_corr(f_img, K1, K1, "clamp", 0.0)     # D1：第二趟误用第一核
d1 = float(np.max(np.abs(bad_two - E023["correlate2d"](
    E023["pad_image"](f_img, "clamp", 0.0, K2D_ASY.shape[0] // 2), K2D_ASY))))
D.append(("D1 两趟第二趟误用 k1（应 k2）→ 等价断言必挂", d1 > 1e-6, f"err={d1:.2e}"))
d2 = float(np.max(np.abs(np.pad(f_img, 2, mode="symmetric") - np.pad(f_img, 2, mode="reflect"))))
D.append(("D2 mirror（reflect 不复制边缘）≠symmetric（复制边缘）→ 1-1 类断言必挂", d2 > 0, f"差={d2:.2e}"))
th = np.deg2rad(45.0)
nl = np.abs(corr_same(f_r, KX)) + np.abs(corr_same(f_r, KY))
ln = np.abs(corr_same(f_r, k1_theta(th)))
D.append(("D3 非线性组合 |Rx|+|Ry| ≠ R(45°)（steer 是线性组合性质）→ 2-2 类断言必挂",
          float(np.max(np.abs(nl - ln))) > 1e-6, f"err={float(np.max(np.abs(nl - ln))):.2e}"))
D.append(("D4 PSNR 基准错（peak=0.5）→ 全体读数平移 6.02dB（注入自证：偏移量实测见右；"
          "3-1/3-2 为相对断言不受平移影响，与 D1/D2/D3/D5/D6 的既有断言必挂型不同）",
          abs((psnr(base, nz) - psnr(base, nz, 0.5)) - 6.0206) < 0.01,
          f"Δ={psnr(base, nz) - psnr(base, nz, 0.5):.3f}"))
det_bad = [{k: core_c0(-v, sig_hat) for k, v in d.items()} for d in det]
rec_bad = np.clip(E031["iwavelet2d"](det_bad, low), 0, 1)
D.append(("D5 coring 反向（系数符号翻转）PSNR 必降 → 4-2 类断言必挂",
          psnr(base_s, rec_bad) < p_noisy, f"{psnr(base_s, rec_bad):.2f} < {p_noisy:.2f}"))
m_wrong = gauss_pyr_clamp((labels == 2).astype(float), LV)   # D6：标签图取错列（全零 mask）
out_wrong = collapse_clamp([laps[1][i] * m_wrong[i] for i in range(LV + 1)])
D.append(("D6 mask 取错（标签图错列→全零 mask）→ 5-2 类退化重构断言必挂",
          float(np.max(np.abs(out_wrong - A1))) > 1e-6,
          f"err={float(np.max(np.abs(out_wrong - A1))):.2e}"))
for name, ok, det_ in D:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {det_}")
if not all(ok for _, ok, _ in D):
    FAILS.append("注入")

n_fail = len([r for r in results if not r[2]])
print(f"\n汇总：{len(results) - n_fail} 条 PASS / {n_fail} 条 FAIL；"
      f"注入 {sum(1 for _, ok, _ in D if ok)}/{len(D)} 生效"
      f"（D1–D3/D5/D6 挂既有断言，D4 自证偏移）")
if FAILS:
    print("FAIL 明细：", FAILS)
    raise SystemExit(1)
print("RUN3-DONE")
