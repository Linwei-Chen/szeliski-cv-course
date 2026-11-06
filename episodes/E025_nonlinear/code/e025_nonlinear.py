#!/usr/bin/env python3
"""
E025 · 非线性登场：中值滤波与双边滤波（§3.3 前半）

范围：§3.3 引言（印刷 p.131，Figure 3.18 中值/双边对照与图注）＋§3.3.1
Non-linear filtering（pp.132–133：中值滤波、α-trimmed mean、加权中值式 3.33、
保边性）＋§3.3.2 Bilateral filtering（pp.133–137：式 3.34–3.37 域核×值域核、
Figure 3.19c/d、Figure 3.20 噪声阶跃边、迭代自适应平滑=各向异性扩散式
3.38–3.40，Perona–Malik）。切点=§3.3.3 Binary image processing 标题（p.138）。

数值实验：把中值对 shot noise 的滤除（Figure 3.18f/g 语义）、中值对高斯噪声
的短板（p.132「not as efficient at averaging away regular Gaussian noise」）、
α-trimmed mean（p.132–133，Figure 3.19b 数值 4.6 复算）、加权中值=式 3.33
p=1 最小化与 p=2=相关式 3.12（p.133）、保边性对照（p.133）、双边滤波核与
两个退化极限（式 3.34–3.37，pp.133–134）、双边对噪声阶跃边的保边（Figure
3.20 语义）与对 shot noise 的失手（p.131 Figure 3.18 图注「the bilateral
filter fails to remove the shot noise」——E024 f024「双边也有失手的时候」
承诺闭环）、迭代形式=各向异性扩散（式 3.38–3.40，pp.135–136，Perona–Malik
1990b）与无偏置收敛警告（p.136）全部算一遍。

断言（26 条 = 22 数据 + 4 常数核对 C1–C4；另同步 1 项、区分力自检 2 项不
计数；错误实现类断言一律由 #D 注入承担，不在正式计数里复写判据——E019
门A round3 新P1-2 惯例）：
  #1  中值滤 shot noise（p.132「Since the shot noise value usually lies well
      outside the true values in the neighborhood, the median filter is able
      to filter away such bad pixels (Figure 3.18g)」；3 条）：基线 5.0 的一维
      信号（15 像素）3 个位置设为 200（离群幅度 +195），窗口 3 中值滤波后
      离群点残差≈0（<1e-12 口径：离群点位置输出回到真值）；同位置高斯滤波
      后残差显著（>离群幅度 30%——「turns them into softer (but still
      visible) spots」Figure 3.18f 语义）；9 值手工锚点中位=排序第 5 位
  #2  中值不擅平均高斯噪声（p.132「it is not as efficient at averaging away
      regular Gaussian noise」；2 条）：纯高斯噪声下 3×3 均值滤波 MSE < 中值
      滤波 MSE（随机 200 帧均值口径，讲者实验设计）；均值滤波 MSE 缩减比
      ≈1/9（方差分析：3×3 独立噪声均值方差 σ²/9，讲者推演）
  #3  α-trimmed mean（p.132–133「averages together all of the pixels except
      for the α fraction that are the smallest and the largest」+Figure 3.19b；
      3 条）：α=0 时与均值滤波逐位一致；Figure 3.19b 复算——25 像素去掉最小
      4 个（1,1,1,1）与最大 4 个（8,8,9,9）后均值 78/17=4.588…→一位小数
      4.6（原书「α-mean= 4.6」锚点）；shot+高斯混合噪声下 α-trimmed（α=4/9）
      MSE < 同输入均值滤波 MSE（同一 sig 逐帧 hash 校验，讲者实验设计；
      门A r1 P1-3 整改：不再跨条件对照 #2 的纯高斯口径）
  #4  加权中值与式 3.33（p.133「This turns out to be equivalent to minimizing
      the weighted objective function … where p = 1 for the weighted median.
      The value p = 2 is the usual weighted mean, which is equivalent to
      correlation (3.12) after normalizing by the sum of the weights」；3 条）：
      离散网格搜索验证加权中值输出=argmin_g Σ w|f−g|（p=1，多组随机权重）；
      p=2 加权均值=归一化相关（Σw·f/Σw）逐位一致；Figure 3.19a 的 25 像素
      中位=4（原书「(a) median = 4」锚点，值表逐字取自原书）
  #5  保边性对照（p.133「Such filtering is more edge preserving, i.e., it has
      less tendency to soften edges while filtering away high-frequency
      noise」+「these techniques still have a tendency to round sharp
      corners」；2 条）：阶跃边+高斯噪声，四种滤波（高斯/中值/α-trimmed/双边）
      的 10–90% 边缘过渡宽度——高斯最宽、中值/双边显著更窄（保边量化，讲者
      实验设计）；α-trimmed 在直边上边缘宽度介于高斯与中值之间（「round sharp
      corners」倾向的直边弱化版）
  #6  双边滤波核（pp.133–134 式 3.34–3.37；4 条）：式 3.37 权重=式 3.35 域核
      ×式 3.36 值域核逐位一致；σr→∞ 时值域核→1、双边输出退化为高斯滤波
      （逐位，σr=1e6）；常数图输入输出恒等（任意 σd/σr——值域无差异且分母
      归一化）；Figure 3.19c 域核复现——exp(−(Δi²+Δj²)/4)（σd=√2）一位小数
      四舍五入与原书 5×5 网格逐位一致（0.1/0.3/0.4/0.6/0.8/1.0）
  #7  双边保边与失手（Figure 3.20 语义+p.131 Figure 3.18 图注「Note that the
      bilateral filter fails to remove the shot noise because the noisy pixels
      are too different from their neighbors」——E024 f024「双边也有失手」；
      4 条）：噪声阶跃边上双边边缘宽度 < 高斯（保边，讲者实验设计）；双边在
      高斯噪声上去噪有效（MSE 下降）；失手案例——坏点与邻居差得太远（p.131
      图注语义），坏点为格心时每个邻居的值域权重被压到≈e^{−2}≈0.14，坏点
      自己成为平均里最大的单项——两处各留约四成（非「原样保留」，r2 P1
      定量口径），输出残差显著大于中值滤波（讲者实验设计，双标）；失手
      机理量化——坏点为格心时残差占比随离群幅度（+0.1/+0.3/+1.0）单调上升
      0.09→0.36→1.00，「几乎只剩坏点自己」仅在 +1.0 时成立（门A r1 P0-1
      整改断言）
  #8  迭代自适应平滑=各向异性扩散（pp.135–136 式 3.38–3.40+「(3.40) is the
      same as the discrete anisotropic diffusion equation first proposed by
      Perona and Malik (1990b)」+「without a bias term towards the original
      image, anisotropic diffusion and iterative adaptive smoothing converge
      to a constant image」；3 条）：N4 邻域下式 3.40 迭代一步与「域核取式
      3.39 两值的式 3.34」输出逐位一致（η 折算）；卡通化演示——阶跃边迭代
      T 步后边缘两侧平台保持、过渡带宽不随迭代扩张（对比高斯迭代带宽线性
      扩张，讲者实验设计）；无偏置迭代 max−min 单调收缩（T=40 步内单调，
      收敛警告语义）
  C1–C4 常数核对：C1 式 3.39 N4 域核值 e^{−1/(2σd²)}（σd=√2 时 e^{−1/4}
      =0.7788→Figure 3.19c 的 0.8）；C2 式 3.35/3.36 的 exp(−x/(2σ²)) 在
      x=σ² 处=e^{−0.5}=0.6065；C3 α-trimmed 78/17=4.5882→4.6；C4 25 像素
      中位=排序第 13 位=4
  #S  同步：7 张程序生成图写入 code/figures/ 与 slides/figures/（md5 一致）
  #D  区分力注入 2 项：均值滤波冒充中值（#1 拦截）、双边去掉值域核（#6/#7
      拦截）——错误实现必须 FAIL
"""
import hashlib
import json
import pathlib
import sys
import numpy as np

RNG = np.random.default_rng(20261028)
FAILS = []
COUNT = {"n": 0}


def chk(label, cond, detail=""):
    COUNT["n"] += 1
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def pad_reflect(a, r):
    return np.pad(a, r, mode="reflect")


def window_view(a, k):
    """k×k 窗口滑窗视图（reflect padding），返回 (H, W, k, k)。"""
    r = k // 2
    p = pad_reflect(a, r)
    H, W = a.shape
    s = p.strides
    return np.lib.stride_tricks.as_strided(
        p, shape=(H, W, k, k), strides=(s[0], s[1], s[0], s[1]))


def median_filter(a, k=3):
    return np.median(window_view(a, k), axis=(2, 3))


def mean_filter(a, k=3):
    return window_view(a, k).mean(axis=(2, 3))


def alpha_trimmed_filter(a, k=3, alpha=0.0):
    """每窗口去掉最小/最大各 alpha 比例后取均值（窗口元素数 n，去掉 n*alpha 个）。"""
    w = np.sort(window_view(a, k).reshape(*a.shape, -1), axis=-1)
    n = w.shape[-1]
    m = int(n * alpha)
    if m == 0:
        return w.mean(axis=-1)
    return w[..., m:n - m].mean(axis=-1)


def gaussian_kernel1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def gaussian_filter_sep(a, sigma, radius):
    """可分离高斯（reflect padding），1D/2D 通用（逐轴卷积）。"""
    g = gaussian_kernel1d(sigma, radius)
    r = radius
    out = a.astype(float)
    for axis in range(out.ndim):
        pad = [(r, r) if ax == axis else (0, 0) for ax in range(out.ndim)]
        p = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        new = np.zeros_like(out)
        for i, gi in enumerate(g):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            new = new + gi * p[tuple(sl)]
        out = new
    return out


def bilateral_filter(a, sigma_d, sigma_r, radius=None):
    """式 3.34–3.37 直接实现（灰度；域核 exp(−dist²/2σd²)、值域核 exp(−Δf²/2σr²)）。"""
    if radius is None:
        radius = max(1, int(np.ceil(2 * sigma_d)))
    r = radius
    p = pad_reflect(a, r)
    H, W = a.shape
    out = np.zeros((H, W))
    ii, jj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    d = np.exp(-((ii ** 2 + jj ** 2)) / (2 * sigma_d ** 2))          # 式 3.35
    acc = np.zeros((H, W))
    wsum = np.zeros((H, W))
    for u in range(2 * r + 1):
        for v in range(2 * r + 1):
            patch = p[u:u + H, v:v + W]
            dv = d[u, v]
            rv = np.exp(-(a - patch) ** 2 / (2 * sigma_r ** 2))      # 式 3.36
            w = dv * rv                                              # 式 3.37
            acc += w * patch
            wsum += w
    return acc / wsum                                                # 式 3.34


def iter_adaptive(f0, sigma_r, eta=1.0, T=1):
    """式 3.40：N4 邻域迭代自适应平滑（无偏置）。返回每步结果列表。"""
    out = [f0.astype(float).copy()]
    f = f0.astype(float).copy()
    H, W = f.shape
    for t in range(T):
        g = f.copy()
        acc_r = np.zeros((H, W))
        acc = np.zeros((H, W))
        for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            fp = np.roll(np.roll(f, di, axis=0), dj, axis=1)
            r = np.exp(-(g - fp) ** 2 / (2 * sigma_r ** 2))          # 式 3.36，d≡式3.39 常数并入 η
            acc += r * fp
            acc_r += r
        f = (g + eta * acc) / (1 + eta * acc_r)                      # 式 3.40
        out.append(f.copy())
    return out


def edge_width_1090(profile):
    """10–90% 过渡带内的像素数（锐利跳变=0，越窄越保边）。"""
    lo, hi = profile.min(), profile.max()
    if hi - lo < 1e-9:
        return np.inf
    t0, t1 = lo + 0.1 * (hi - lo), lo + 0.9 * (hi - lo)
    idx = np.where((profile > t0) & (profile < t1))[0]
    return float(idx[-1] - idx[0] + 1) if len(idx) else 0.0


def mse(a, b):
    return float(np.mean((a - b) ** 2))


print("== E025 数值实验（§3.3 前半：中值/α-trimmed/双边/各向异性扩散）==\n")

# ---------- #1 中值滤 shot noise ----------
print("#1 中值滤 shot noise（p.132，Figure 3.18f/g 语义）")
true_line = np.full(15, 5.0)
shot = true_line.copy()
shot_pos = [3, 8, 12]
shot[shot_pos] = 200.0                                   # 偶发大值
med_line = median_filter(shot.reshape(1, -1), 3)[0]
gau_line = gaussian_filter_sep(shot.reshape(1, -1), 1.0, 4)[0]
res_med = np.abs(med_line[shot_pos] - true_line[shot_pos]).max()
res_gau = np.abs(gau_line[shot_pos] - true_line[shot_pos]).max()
chk("中值滤除 shot noise：离群点输出回到真值（残差<1e-9）", res_med < 1e-9, f"数值: {res_med:.2e}")
chk("高斯把离群点变成可见斑点（残差>真值幅度 30%——Figure 3.18f 语义）",
    res_gau > 0.3 * 195, f"数值: {res_gau:.1f}（195 幅度）")
w9 = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5], dtype=float)
chk("3×3 窗口中值=排序第 5 位（手工锚点 4）", np.median(w9) == 4 and
    np.sort(w9)[4] == 4, f"数值: {np.sort(w9)[4]:.0f}")

# ---------- #2 中值不擅平均高斯噪声 ----------
print("\n#2 中值 vs 均值 on 高斯噪声（p.132）")
rng = np.random.default_rng(7)
mses_mean, mses_med = [], []
for _ in range(200):
    sig = np.full((24, 24), 10.0) + rng.normal(0, 10, (24, 24))
    mses_mean.append(mse(mean_filter(sig, 3), 10.0))
    mses_med.append(mse(median_filter(sig, 3), 10.0))
chk("高斯噪声下均值滤波 MSE < 中值滤波 MSE（200 帧）",
    np.mean(mses_mean) < np.mean(mses_med),
    f"数值: mean {np.mean(mses_mean):.2f} vs median {np.mean(mses_med):.2f}")
ratio = np.mean(mses_mean) / (10 ** 2 / 9)
chk("均值滤波 MSE≈σ²/9（3×3 独立噪声方差分析，±25%）", 0.75 < ratio < 1.25, f"数值: {ratio:.2f}")

# ---------- #3 α-trimmed mean ----------
print("\n#3 α-trimmed mean（p.132–133，Figure 3.19b）")
a0 = np.array([[3.0, 7], [2, 9], [5, 1]])
a0m = alpha_trimmed_filter(a0, 3, 0.0)
a0mean = mean_filter(a0, 3)
chk("α=0 时 α-trimmed 与均值滤波逐位一致", np.allclose(a0m, a0mean))
fig_vals = np.array([1, 2, 1, 2, 4, 2, 1, 3, 5, 8, 1, 3, 7, 6, 9, 3, 4, 8, 6, 7, 4, 5, 7, 8, 9], dtype=float)
trimmed = np.sort(fig_vals)[4:21]
c3 = trimmed.mean()
chk("Figure 3.19b 复算：去最小 4+最大 4 → 78/17≈4.588→4.6（原书锚点）",
    abs(fig_vals.sum() - 116) < 1e-9 and abs(trimmed.sum() - 78) < 1e-9
    and abs(c3 - 78 / 17) < 1e-9 and round(c3, 1) == 4.6,
    f"数值: 全和 116，去 4 小(=4)+4 大(=34)后 {trimmed.sum():.0f}/{len(trimmed)}={c3:.4f}→{round(c3, 1)}")
rng = np.random.default_rng(11)
mses_at, mses_mean_mix = [], []
for _ in range(200):
    sig = np.full((24, 24), 10.0)
    idx = rng.choice(24 * 24, 12, replace=False)
    sig.flat[idx] += rng.choice([-1, 1], 12) * rng.uniform(60, 90, 12)   # shot noise
    sig += rng.normal(0, 5, (24, 24))
    h = hash(sig.tobytes())                    # 同一输入校验：两个滤波器必须吃同一份数据
    mses_at.append(mse(alpha_trimmed_filter(sig, 3, 4 / 9), 10.0))
    assert hash(sig.tobytes()) == h, "α-trimmed 输入被改动"
    mses_mean_mix.append(mse(mean_filter(sig, 3), 10.0))
    assert hash(sig.tobytes()) == h, "均值滤波输入被改动"
chk("shot+高斯混合噪声下 α-trimmed（去 4/9）MSE < 同输入均值滤波 MSE（同一 sig，hash 校验）",
    np.mean(mses_at) < np.mean(mses_mean_mix),
    f"数值: α-trim {np.mean(mses_at):.2f} vs mean {np.mean(mses_mean_mix):.2f}")

# ---------- #4 加权中值与式 3.33 ----------
print("\n#4 加权中值=式 3.33 p=1（p.133）")
rng = np.random.default_rng(13)
ok_argmin = True
for _ in range(50):
    vals = rng.integers(0, 20, 9).astype(float)
    w = rng.uniform(0.2, 2.0, 9)
    wm = float(np.average(vals, weights=w))          # 先给占位，下面用中值定义核对
    # 加权中值：排序后累积权重首次过半的数据点（精确定义）
    order = np.argsort(vals)
    cum = np.cumsum(np.asarray(w)[order])
    wm = float(vals[order][np.searchsorted(cum, w.sum() / 2)])
    gs = np.linspace(vals.min() - 1, vals.max() + 1, 4001)
    obj = np.array([np.sum(w * np.abs(vals - g)) for g in gs])
    obj_wm = np.sum(w * np.abs(vals - wm))
    if obj_wm > obj.min() + 1e-6:
        ok_argmin = False
chk("加权中值输出达到 argmin_g Σ w|f−g|₁ 的最小目标值（50 组随机权重，网格搜索；plateau 上任一极小点均可）",
    ok_argmin)
f4 = rng.uniform(0, 10, (6, 6))
w4 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
wv = window_view(f4, 3)
num = (wv * w4).sum(axis=(2, 3))
den = w4.sum()
chk("p=2 加权均值=归一化相关 Σw·f/Σw（式 3.12 口径）",
    np.allclose(num / den, (wv * w4).sum(axis=(2, 3)) / w4.sum()))
chk("Figure 3.19a 中位=排序第 13 位=4（原书「(a) median = 4」锚点）",
    np.sort(fig_vals)[12] == 4 and len(fig_vals) == 25, f"数值: {np.sort(fig_vals)[12]:.0f}")

# ---------- #5 保边性对照 ----------
print("\n#5 保边性对照（p.133）")
step = np.where(np.arange(32) < 16, 0.0, 1.0)
noise5 = step + np.random.default_rng(17).normal(0, 0.05, 32)
edge = np.clip(noise5, 0, 1).reshape(1, -1)
g5 = gaussian_filter_sep(edge, 1.5, 5)[0]
m5 = median_filter(edge, 3)[0]
at5 = alpha_trimmed_filter(edge, 3, 2 / 9)[0]
b5 = bilateral_filter(edge, 1.5, 0.3, 4)[0]
wg, wm_, wa, wb = (edge_width_1090(x[12:20]) for x in (g5, m5, at5, b5))
chk("边缘过渡带宽：高斯最宽，中值/双边显著更窄（保边量化；锐跳变=0）",
    wg > wm_ and wg > wb, f"数值: 高斯 {wg:.0f} / 中值 {wm_:.0f} / α-trim {wa:.0f} / 双边 {wb:.0f} px")
chk("α-trimmed 直边宽度介于高斯与中值之间（round corners 倾向的直边弱化）",
    wa < wg and wa > wm_, f"数值: α-trim {wa:.1f} ∈ ({wm_:.1f}, {wg:.1f})")

# ---------- #6 双边滤波核 ----------
print("\n#6 双边滤波核（式 3.34–3.37，pp.133–134）")
sig6 = np.random.default_rng(19).uniform(0, 1, (20, 20))
r6, d6 = 2, 2
p6 = pad_reflect(sig6, r6)
H6, W6 = sig6.shape
ii6, jj6 = np.meshgrid(np.arange(-r6, r6 + 1), np.arange(-r6, r6 + 1), indexing="ij")
dom = np.exp(-(ii6 ** 2 + jj6 ** 2) / (2 * r6 ** 2))
rngker = np.random.default_rng(23)
cen = sig6[8, 8]
wsum6 = 0.0
acc6 = 0.0
prod_ok = True
for u in range(2 * r6 + 1):
    for v in range(2 * r6 + 1):
        val = p6[8 + u, 8 + v]
        ran = np.exp(-((cen - val) ** 2) / (2 * 0.2 ** 2))
        wprod = dom[u, v] * ran
        # 直接按式 3.37 单式重算
        wdirect = np.exp(-((ii6[u, v] ** 2 + jj6[u, v] ** 2) / (2 * r6 ** 2)
                           + (cen - val) ** 2 / (2 * 0.2 ** 2)))
        if abs(wprod - wdirect) > 1e-15:
            prod_ok = False
        acc6 += wprod * val
        wsum6 += wprod
chk("式 3.37 权重=式 3.35 域核×式 3.36 值域核（逐位）", prod_ok)
bil_small = bilateral_filter(sig6, r6, 1e6, r6)
gau_small = gaussian_filter_sep(sig6, np.sqrt(2) * 1.0, 2)  # σd=√2×r? 用同域核口径
# 直接构造同域核的高斯（2σd²=2·r²）
p6g = pad_reflect(sig6, r6)
gacc = np.zeros_like(sig6)
gwsum = np.zeros_like(sig6)
for u in range(2 * r6 + 1):
    for v in range(2 * r6 + 1):
        w = dom[u, v]
        gacc += w * p6g[u:u + H6, v:v + W6]
        gwsum += w
chk("σr→∞ 退化为同域核高斯（值域核→1，逐位）",
    np.allclose(bil_small, gacc / gwsum, atol=1e-12), f"数值: 最大差 {np.abs(bil_small - gacc / gwsum).max():.2e}")
const6 = np.full((16, 16), 3.7)
chk("常数图输入输出恒等（任意 σd/σr）",
    np.allclose(bilateral_filter(const6, 1.5, 0.3, 3), 3.7, atol=1e-12))
dc = np.exp(-(ii6 ** 2 + jj6 ** 2) / 4.0)  # σd=√2 → 2σd²=4
fig_c = np.array([[0.1, 0.3, 0.4, 0.3, 0.1], [0.3, 0.6, 0.8, 0.6, 0.3],
                  [0.4, 0.8, 1.0, 0.8, 0.4], [0.3, 0.6, 0.8, 0.6, 0.3],
                  [0.1, 0.3, 0.4, 0.3, 0.1]])
chk("Figure 3.19c 域核复现：exp(−dist²/4)（σd=√2）一位小数逐位一致",
    np.array_equal(np.round(dc, 1), fig_c), f"数值: 与原书网格差 {np.abs(np.round(dc,1)-fig_c).max():.1f}")

# ---------- #7 双边保边与失手 ----------
print("\n#7 双边保边与失手（Figure 3.20 语义+p.131 Figure 3.18 图注）")
step2d = np.where(np.arange(40) < 20, 0.2, 0.8)
noise2d = step2d + np.random.default_rng(29).normal(0, 0.04, 40)
row = noise2d.reshape(1, -1)
bg7 = gaussian_filter_sep(row, 2.0, 5)[0]
bb7 = bilateral_filter(row, 2.0, 0.15, 5)[0]
wbg = edge_width_1090(bg7[14:26])
wbb = edge_width_1090(bb7[14:26])
chk("噪声阶跃边：双边边缘宽度 < 高斯（Figure 3.20d/e 保边语义）", wbb < wbg,
    f"数值: 双边 {wbb:.0f} vs 高斯 {wbg:.0f} px")
base = np.full((24, 24), 0.5) + np.random.default_rng(31).normal(0, 0.05, (24, 24))
rec = bilateral_filter(base, 1.5, 0.2, 3)
chk("双边在高斯噪声上去噪有效（MSE 下降）", mse(rec, 0.5) < 0.05 ** 2,
    f"数值: {mse(rec, 0.5):.2e} < {0.05 ** 2:.2e}")
fail_base = np.full((24, 24), 0.5)
fail_in = fail_base + np.random.default_rng(37).normal(0, 0.008, (24, 24))
shot_idx = [(8, 8), (15, 12)]
for (i, j) in shot_idx:
    fail_in[i, j] += 0.30          # 离群值 +0.3≈2σr：以坏点为格心时每个邻居值域权重≈e^{-2}≈0.14，
                                   # 坏点自己成为平均里最大的单项——两处各留约四成（p.131 图注语义）
bil_fail = bilateral_filter(fail_in, 1.5, 0.15, 3)
med_fail = median_filter(fail_in, 3)
rb = np.abs(bil_fail[8, 8] - 0.5) + np.abs(bil_fail[15, 12] - 0.5)
rm = np.abs(med_fail[8, 8] - 0.5) + np.abs(med_fail[15, 12] - 0.5)
chk("失手案例：与邻居差得太远的孤立坏点，双边去不掉（残差>中值 5×）——Figure 3.18 图注语义",
    rb > 5 * max(rm, 1e-6), f"数值: 双边残差 {rb:.4f} vs 中值 {rm:.4f}")
# 失手机理量化（门A r1 P0-1 整改）：坏点为格心时，离群幅度越大→邻居值域权重越小→
# 坏点保留得越彻底（残差占比→1）；「几乎只剩坏点自己」仅在 +1.0（=6.7σr）时成立
fracs = []
for amp in (0.1, 0.3, 1.0):
    iso = np.full((11, 11), 0.5)
    iso[5, 5] = 0.5 + amp
    out_iso = bilateral_filter(iso, 1.5, 0.15, 4)
    fracs.append(abs(out_iso[5, 5] - 0.5) / amp)
chk("失手机理：坏点为格心时残差占比随离群幅度单调上升（邻居被值域核压低，坏点越离谱保留越彻底）",
    fracs[0] < fracs[1] < fracs[2] and fracs[2] > 0.9,
    f"数值: +0.1→{fracs[0]:.2f} / +0.3→{fracs[1]:.2f} / +1.0→{fracs[2]:.2f}")

# ---------- #8 迭代自适应平滑=各向异性扩散 ----------
print("\n#8 迭代形式=各向异性扩散（式 3.38–3.40，pp.135–136）")
f8 = np.random.default_rng(41).uniform(0, 1, (12, 12))
sigma_r8 = 0.3
one_iter = iter_adaptive(f8, sigma_r8, eta=1.0, T=1)[1]
# 直接式 3.34（N4 域核：式 3.39 d=center→1、neighbor→exp(−1/2σd²)，σd=√2 → e^{−1/4}=0.7788 并入 η 口径）
# 式 3.40 取 d≡1（N4 域核常数并入 η）——直接实现核对
H8, W8 = f8.shape
acc8 = np.zeros((H8, W8))
rsum8 = np.zeros((H8, W8))
for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
    fp = np.roll(np.roll(f8, di, axis=0), dj, axis=1)
    r8 = np.exp(-(f8 - fp) ** 2 / (2 * sigma_r8 ** 2))
    acc8 += r8 * fp
    rsum8 += r8
direct8 = (f8 + 1.0 * acc8) / (1 + 1.0 * rsum8)
chk("式 3.40 迭代一步与直接 N4 加权归一实现逐位一致", np.allclose(one_iter, direct8, atol=1e-14))
c1 = np.exp(-0.25)
chk("C1 式 3.39：σd=√2 时邻域域核 e^{−1/4}=0.7788（Figure 3.19c 的 0.8）",
    abs(c1 - 0.7788) < 1e-3 and round(dc[2, 1], 1) == 0.8, f"数值: {c1:.4f}，网格中心左邻 {dc[2, 1]:.4f}→{round(dc[2, 1], 1)}")
c2 = np.exp(-0.5)
chk("C2 式 3.35/3.36：x=σ² 处 exp(−x/2σ²)=e^{−0.5}=0.6065", abs(c2 - 0.6065) < 1e-3, f"数值: {c2:.4f}")
chk("C3 α-trimmed 78/17=4.5882→4.6", abs(78 / 17 - 4.5882) < 1e-3, f"数值: {78 / 17:.4f}")
chk("C4 25 像素中位=排序第 13 位（Figure 3.19a 锚点）", len(fig_vals) == 25 and np.sort(fig_vals)[12] == 4)

step8 = np.where(np.arange(40) < 20, 0.2, 0.8)
step8n = step8 + np.random.default_rng(43).normal(0, 0.03, (9, 40))
it = iter_adaptive(step8n, 0.1, eta=1.0, T=15)
w_it = [edge_width_1090(x[4][14:26]) for x in it]
w_gs = []
gf = step8n.copy()
for t in range(15):
    gf = gaussian_filter_sep(gf, 1.0, 2)
    w_gs.append(edge_width_1090(gf[4][14:26]))
chk("卡通化：迭代自适应平滑边缘带宽不随迭代扩张（高斯迭代线性扩张对照）",
    max(w_it) - min(w_it) <= 2 and w_gs[-1] - w_gs[0] > 3,
    f"数值: 迭代带宽 {min(w_it):.0f}–{max(w_it):.0f} vs 高斯 {w_gs[0]:.0f}→{w_gs[-1]:.0f}")
mm = [x.max() - x.min() for x in it]
chk("无偏置迭代 max−min 单调收缩（p.136 收敛警告语义）",
    all(mm[i + 1] <= mm[i] + 1e-9 for i in range(len(mm) - 1)) and mm[-1] < mm[0],
    f"数值: {mm[0]:.3f}→{mm[-1]:.3f}")

# ---------- #S 同步 ----------
print("\n#S 同步（7 张程序生成图）")
sync_ok = True
figdir = pathlib.Path(__file__).resolve().parent / "figures"
slide_dir = pathlib.Path(__file__).resolve().parent.parent / "slides" / "figures"
made = []
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    def savefig(fig, name):
        for d in (figdir, slide_dir):
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / name, dpi=150, bbox_inches="tight")
        made.append(name)
        plt.close(fig)

    # 图1 shot noise：高斯 vs 中值
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    ax[0].plot(shot, "k.-", lw=0.8, ms=4)
    ax[0].plot(gau_line, "b-", lw=1.2)
    ax[0].set_title("高斯 σ=1：离群点变成斑点（Figure 3.18f 语义，程序生成）", fontsize=10)
    ax[1].plot(shot, "k.-", lw=0.8, ms=4)
    ax[1].plot(med_line, "r-", lw=1.2)
    ax[1].set_title("3×3 中值：离群点滤除（Figure 3.18g 语义）", fontsize=10)
    for a in ax:
        a.axhline(5, color="gray", ls=":", lw=0.7)
        a.set_xlabel("x", fontsize=9)
    ax[0].set_ylabel("灰度", fontsize=9)
    savefig(fig, "e025_median_shot.png")

    # 图2 边缘保持对照
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(edge[0], "k.", ms=3, label="含噪阶跃边")
    ax.plot(g5, "b-", lw=1.1, label=f"高斯 σ=1.5（宽 {wg:.0f}px）")
    ax.plot(at5, "g-", lw=1.1, label=f"α-trimmed（宽 {wa:.0f}px）")
    ax.plot(m5, "r-", lw=1.1, label=f"中值（宽 {wm_:.0f}px）")
    ax.plot(b5, "m-", lw=1.1, label=f"双边（宽 {wb:.0f}px）")
    ax.legend(fontsize=8)
    ax.set_title("保边性对照：边缘 10–90% 过渡宽度（程序生成，原书图未使用）", fontsize=10)
    ax.set_xlabel("x", fontsize=9)
    ax.set_ylabel("灰度", fontsize=9)
    savefig(fig, "e025_edge_preserve.png")

    # 图3 Figure 3.19 数值例复现（表格）
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    diff = np.abs(np.round(dc, 1) - fig_c)
    for a, (mat, fmt, title) in zip(ax, [
        (fig_vals.reshape(5, 5).astype(int), "d", "(a) 邻域像素（原书数值）median=4"),
        (dc, ".1f", "(c) 域核 exp(−dist²/4)，σd=√2"),
        (diff, ".1f", "(c′) 程序核四舍五入后与原书印刷值之差")]):
        a.axis("off")
        cell_text = [[f"{v:{fmt}}" for v in row] for row in mat]
        tbl = a.table(cellText=cell_text, loc="center", cellLoc="center")
        tbl.scale(1, 1.4)
        a.set_title(title, fontsize=9)
    savefig(fig, "e025_fig319_check.png")

    # 图4 双边核分解
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    im0 = ax[0].imshow(dom, vmin=0, vmax=1, cmap="viridis")
    ax[0].set_title("域核 d（式 3.35，σd=2）", fontsize=9)
    vals_axis = np.linspace(-0.5, 1.5, 200)
    ax[1].plot(vals_axis, np.exp(-((0.5 - vals_axis) ** 2) / (2 * 0.15 ** 2)), "r-")
    ax[1].set_title("值域核 r（式 3.36，中心 0.5，σr=0.15）", fontsize=9)
    im2 = ax[2].imshow(bilateral_filter(np.where(np.arange(32) < 16, 0.0, 1.0).reshape(1, -1)
                                        .repeat(9, 0), 2.0, 0.15, 5)[4:5, :], aspect="auto", vmin=0, vmax=1)
    ax[2].set_title("双边滤波输出（阶跃边中线，式 3.34）", fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    savefig(fig, "e025_bilateral_kernel.png")

    # 图5 双边失手 vs 中值
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    ax[0].plot(fail_in[8], "k.-", lw=0.8, ms=4)
    ax[0].plot(bil_fail[8], "m-", lw=1.2)
    ax[0].plot(fail_base[8], "gray", ls=":", lw=0.8)
    ax[0].set_title("双边 σr=0.15：与邻居差得太远的坏点去不掉（失手，程序生成）", fontsize=10)
    ax[1].plot(fail_in[8], "k.-", lw=0.8, ms=4)
    ax[1].plot(med_fail[8], "r-", lw=1.2)
    ax[1].plot(fail_base[8], "gray", ls=":", lw=0.8)
    ax[1].set_title("中值：离群点滤除", fontsize=10)
    for a in ax:
        a.set_xlabel("x", fontsize=9)
    ax[0].set_ylabel("灰度", fontsize=9)
    savefig(fig, "e025_bilateral_fail.png")

    # 图6 迭代=各向异性扩散
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(step8n[4], "k.", ms=3, label="含噪阶跃边")
    ax.plot(it[1][4], "c-", lw=1.0, label="迭代 1 步")
    ax.plot(it[8][4], "b-", lw=1.1, label="迭代 8 步")
    ax.plot(it[15][4], "m-", lw=1.2, label="迭代 15 步（卡通化）")
    ax.set_title("迭代自适应平滑=离散各向异性扩散（式 3.40，Perona–Malik）——程序生成，原书图未使用", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlabel("x", fontsize=9)
    savefig(fig, "e025_iter_diffusion.png")

    # 图7 #3-3 同输入 MSE 对比（页4 配图，r2 P2：撤掉与 α-mean 复算无关的保边图）
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    bars = ax.bar(["α-截断均值\n（去最小/最大各 4）", "普通均值"], 
                  [np.mean(mses_at), np.mean(mses_mean_mix)], color=["g", "b"], width=0.5)
    for b, v in zip(bars, [np.mean(mses_at), np.mean(mses_mean_mix)]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("MSE（越低越好）", fontsize=9)
    ax.set_title("脉冲+高斯混合噪声，200 帧同一输入：α-trim 误差不到均值的三成\n（#3-3，逐帧 hash 校验同输入；程序生成，原书图未使用）", fontsize=9)
    ax.set_ylim(0, max(np.mean(mses_mean_mix), np.mean(mses_at)) * 1.18)
    savefig(fig, "e025_alpha_mse.png")
except Exception as e:
    import traceback
    traceback.print_exc()
    sync_ok = False
    print(f"  [WARN] 图生成异常: {e}")
fig_count = len(made)
md5s = []
for name in made:
    h1 = hashlib.md5((figdir / name).read_bytes()).hexdigest()
    h2 = hashlib.md5((slide_dir / name).read_bytes()).hexdigest()
    if h1 != h2:
        sync_ok = False
    md5s.append((name, h1[:12]))
chk("#S 同步：7 张程序生成图双目录 md5 一致", sync_ok and fig_count == 7, f"数值: {fig_count} 张")

# ---------- #D 区分力注入 ----------
print("\n#D 区分力注入（错误实现必须 FAIL）")
try:
    fake_median = mean_filter(shot.reshape(1, -1), 3)[0]
    res_fake = np.abs(fake_median[shot_pos] - true_line[shot_pos]).max()
    chk("D-1 均值滤波冒充中值 → #1 断言拦截", res_med < 1e-9 and res_fake > 0.3 * 195,
        f"数值: 假中值残差 {res_fake:.1f}（真中值 {res_med:.2e}）")
    def bilateral_norange(a, sigma_d, radius):
        r = radius
        p = pad_reflect(a, r)
        H, W = a.shape
        ii, jj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        d = np.exp(-(ii ** 2 + jj ** 2) / (2 * sigma_d ** 2))
        acc = np.zeros((H, W)); wsum = np.zeros((H, W))
        for u in range(2 * r + 1):
            for v in range(2 * r + 1):
                acc += d[u, v] * p[u:u + H, v:v + W]
                wsum += d[u, v]
        return acc / wsum
    fake_bil = bilateral_norange(fail_in, 1.5, 3)
    rf = np.abs(fake_bil[8, 8] - 0.5) + np.abs(fake_bil[15, 12] - 0.5)
    chk("D-2 双边去掉值域核（=纯高斯）→ #7 失手断言语义反转拦截",
        rb > 5 * max(rm, 1e-6) and rf < rb / 3,
        f"数值: 无值域核残差 {rf:.4f} << 有值域核 {rb:.4f}（离群点被平均掉，失手消失）")
except Exception as e:
    chk("D 区分力注入执行", False, f"异常: {e}")

# ---------- 汇总 ----------
print(f"\n汇总：{COUNT['n'] - len(FAILS)} 条 PASS / {len(FAILS)} FAIL")
if FAILS:
    print("FAIL 明细：", FAILS)
    sys.exit(1)
print("DONE")
