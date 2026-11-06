#!/usr/bin/env python3
"""
E029 · 放大与缩小：插值与抽稀的地基（§3.5 引言＋3.5.1 Interpolation＋3.5.2 Decimation）

范围：§3.5 引言（印刷 p.149 下：E028 移交句落点「So far in this chapter, all of
the image transformations we have studied produce output images of the same size
as the inputs … Often, however, we may wish to change the resolution of an image」）
＋3.5.1 Interpolation（pp.150–152 上：式 3.64 g(i,j)=ΣΣ f(k,l)h(i−rk, j−rl)
——「we replace k and l in h() with rk and rl, where r is the upsampling rate」；
Figure 3.26 两模型（a 加权和 superposition of sample weighted interpolation
kernels / b 多相滤波器 polyphase filter form，two forms are equivalent）；
「Any of the smoothing kernels shown in Table 3.1 can be used after appropriate
re-scaling」（脚注 11：平滑核单位面积，转插值核乘 r）；tent=线性插值
（「produces interpolating piecewise linear curves, which result in unappealing
creases」）；cubic B-spline=tent 自卷积（「easiest to derive by convolving the
tent function with itself」；近似核「does not pass through the input data
points」；「discrete 1/2-pixel sampling appears as the binomial kernel in
Table 3.1」）；式 3.65 三次核（C1 分段三次样条，a specifies the derivative
at x=1，Parker, Kenyon, and Troxel 1983；a=−1「best matches the frequency
characteristics of a sinc function」＋「introduces a small amount of
sharpening」＋「does not linearly interpolate straight lines … visible ringing
may occur」；a=−0.5「produces a quadratic reproducing spline; it interpolates
linear and quadratic functions exactly (Wolberg 1990)」；Figure 3.27 四例
bilinear/bicubic a=−1/bicubic a=−0.5/windowed sinc nine taps；Figure 3.28
核曲线与对数频幅「raised-cosine windowed sinc in blue, cubic interpolators
in green and purple, and tent function in brown」；windowed sinc=「The highest
quality interpolator … preserves details … avoids aliasing」但有振铃）
＋3.5.2 Decimation（pp.152–154 中：脚注 13 词源十杀一；式 3.66
g(i,j)=ΣΣ f(k,l)h(ri−k, rj−l)（先低通防混叠再隔 r 取样，「we usually only
evaluate the convolution at every rth sample」）；式 3.67 g(i,j)=(1/r)ΣΣ
f(k,l)h(i−k/r, j−l/r)（「keep the same kernel h(k,l) for both interpolation
and decimation」）；「the smoothing kernel h(k,l), in this case, is often a
stretched and re-scaled version of an interpolation kernel」；binomial
（Burt–Adelson 1983a）「does a decent job of separating the high and low
frequencies, but still leaves a fair amount of high-frequency detail, which
can lead to aliasing after downsampling；However, for applications such as
image blending … this aliasing is of little concern」；直接显示或 MIP-mapping
场景需更高品质（「For high downsampling rates, the windowed sinc prefilter is
a good choice … for small downsampling rates, e.g., r=2, more careful filter
design is required」）；Table 3.2 六种 r=2 抽稀滤波器（「odd length, are
symmetric, and are normalized to have unit DC gain (sum up to 1)」：linear
[1,2,1]/4、binomial [1,4,6,4,1]/16、cubic a=−1、cubic a=−0.5、windowed sinc、
QMF-9（Simoncelli and Adelson 1990b，「used for wavelet denoising and aliases
a fair amount；original filter coefficients are normalized to √2 gain so they
can be self-inverting」）、JPEG 2000 9/7（Taubman and Marcellin 2002））；
Figure 3.30 频响。
切点=§3.5.3 Multi-resolution representations 标题（印刷 p.154 中，归 E030）。

口径注记（红线 1 诚实申报）：
  · windowed sinc 的窗与截断（门A r4 R4-1：与 main 标头、f011 refs 逐字同文）：
    原书已写明 raised-cosine（升余弦）窗（图 3.28 图注；Table 3.2 条目作
    cosine-windowed）；原书给出的截断——图 3.27d 为 nine taps，Table 3.2
    windowed sinc 为 |n|≤4，即 9 抽头（含两端零值），原文未说明两者是否同一
    窗口；原书未给连续窗的窗宽参数——连续曲线重绘取讲者口径
    h(x)=sinc₂(x)·cos²(πx/5)（|x|≤2.5），r=2 下 11 抽头（含两端零值），
    与原书 9 抽头不同。本实验 Table 3.2 断言用表值口径（9 抽头），2D 放大
    示例用讲者口径窗；讲者口径系数（½ 采样归一）与 Table 3.2 表值
    （0.4939/0.2684/0/−0.0153/0）方向一致但不逐一相等，Table 3.2 的数值断言
    一律以表值忠实录入为准，差异如实申报；QMF-9/JPEG 2000 系数来自文献
    （Simoncelli–Adelson 1990b/Taubman–Marcellin 2002），原书未给解析式，
    照表录入。
  · 「混叠功率」定义为讲者口径：E_alias=∫_{π/2}^{π}|H(ω)|²dω/π（Nyquist
    半带以上的能量，r=2 抽稀的混叠上界度量），原书未给该积分式。
  · B-spline 称呼交叉（讲者复算发现，红线 1 申报）：原书 L116–120 称 tent⋆tent
    为「cubic B-spline」并称其 ½ 像素采样=binomial；程序复算：tent⋆tent（分段
    三次、峰 2/3）的 ½ 采样归一=[0.333,0.240,0.083,0.010]≠binomial；与
    binomial=[1,4,6,4,1]/16 精确对应的是 tent⋆box（分段二次、峰 0.75，即
    quadratic B-spline）的 ½ 采样。deck/口播按原书表述转述，并以讲者按注记
    申报此复算发现（原书「cubic B-spline」称呼与核次数的文献交叉）。
  · 脚注 11「The smoothing kernels in Table 3.1 have a unit area. To turn them
    into interpolating kernels, we simply scale them up by the interpolation
    rate r」按「幅度×r」复算（E029 门A round1 P0-1 裁定）：单位面积平滑核
    ×r 即得插值核——[1,2,1]/4×2=[0.5,1,0.5] 代入式 3.64：g(rk)=f(k)、
    g(rk+1)=邻均值，恰为线性插值（#1-6 实测；与连续核 tent(x/r) 口径逐点
    全等）。两种视角同一件事：单位面积核拉宽 r 倍峰高剩 1/r，乘回 r 峰高回
    1。注意「×r」只对单位面积平滑核成立；对已是峰 1 的插值核再乘 r 才会给
    g(rk)=r·f(k)。
  · 2D 放大示例与混叠实测的测试图为程序合成，非原书图（红线 2）。

数值实验（38 条正式断言 = 30 数据 + 3 常数核对 C1–C3 + #S 图同步 5 条；
另有 #D 区分力注入 4 项——E019 门A round3 新P1-2 惯例，注入类判据由 #D 承担，
不进正式计数、汇总单列）：
  #1  式 3.64 插值与两模型（Figure 3.26；6 条）：式 3.64 直接实现（r=2，线性
      插值核 tent(x/r)）对线性斜坡逐点精确（「piecewise linear」）；
      多相形式（偶相/奇相两个子核）与直接式逐点一致（「polyphase filter
      form … two forms are equivalent」）；偶相在原样本处精确还原、奇相为邻
      均值（多相子核 [0,1,0]/[0.5,0.5]）；cubic B-spline=tent⋆tent 数值卷积
      vs 解析式一致（「easiest to derive by convolving the tent function with
      itself」）；tent⋆box（quadratic B-spline）½ 像素采样归一=[1,4,6,4,1]/16
      精确=binomial（binomial 对应关系见口径注记第 4 条）且峰 0.75≠1（
      「approximating kernel … does not pass through the input data points」
      数值兑现）；脚注 11 幅度×r 口径实测：[1,2,1]/4×2=[0.5,1,0.5] 代式 3.64
      偶相原样还原、奇相邻均值，与 tent(x/r) 连续核口径逐点全等
  #2  式 3.65 三次核（6 条）：h(0)=1、h(±1)=h(±2)=0、|x|≥2=0；x=1 处左右
      导数=a（数值差分，「a specifies the derivative at x=1」）；a=−0.5 精确
      再现线性与二次（半像素网格逐点 err<1e-12——「interpolates linear and
      quadratic functions exactly」）；a=−1 线性破坏（过冲/欠冲双向≥0.05——
      「does not linearly interpolate straight lines」）；a=−1 频响最贴 sinc
      （½ 像素采样核带内最大偏差 a=−1<a=−0.5<tent——「best matches the
      frequency characteristics of a sinc」）；a=−1 负瓣幅度大于 a=−0.5
      （|h(±1.5)|=0.125 vs 0.0625——锐化 sharpening 与过冲振铃的时域成因）
  #3  Table 3.2 忠实录入（6 条）：七滤波器系数和=1（「unit DC gain」）；逐条
      对称；windowed sinc/QMF-9/JPEG 2000 系数与留档 L365–381 逐值一致
      （防转录错）；binomial=[1,4,6,4,1]/16（与 E024/E027 口径同源）；
      linear=[1,2,1]/4；**linear/binomial/cubic a=−1/cubic a=−0.5 四列=对应
      插值核（tent/B-spline/式 3.65）½ 像素采样的归一**（「often a stretched
      and re-scaled version of an interpolation kernel」的数值兑现）
  #4  频响与 Figure 3.30 语义（6 条）：七滤波器 H(0)=1（表值 4 位舍入容差
      5e-4；H(π) 读数）；cubic a=−1 通带保持 |H(0.45π)| 高于 a=−0.5（两者
      半增益点同为 0.5π，a=−1 从更高处降下——「sharper fall-off」频域兑现）；
      binomial |H(0.45π)| 七者最小（「cuts off a lot of frequencies」原文
      L303–304；windowed sinc 相对保留更多=preserves details）；混叠功率
      （讲者口径 E_alias）：binomial 七者最小、QMF-9/JPEG 2000 均高于
      linear 基线（QMF-9「aliases a fair amount」），完整排序作读数；QMF-9
      √2 增益后 |H(π/2)|≈1（复算 0.7072·√2=1.0001——「normalized to √2
      gain so they can be self-inverting」数值兑现）；binomial 半带以上
      [0.5π,0.7π] 的 |H| 均值明显大于零（相对理想半带低通的残留——
      「still leaves a fair amount of high-frequency detail」的频域语义：
      E_alias 七者最小≠无残留）
  #5  抽稀实现与混叠实测（6 条）：式 3.66（h_dec=[0.25,0.5,0.25] 预滤波+隔 2
      取样）与式 3.67（h=tent 连续核、(1/r) 缩放）逐点一致；测试图七滤波器
      抽稀→tent(x/r) 插值回，PSNR 全部高于无预滤波口径；无预滤波直接隔点取样
      混叠最重（PSNR 最低）；JPEG 2000 9/7 实测 PSNR 最高（「for small
      downsampling rates, e.g., r=2, more careful filter design is required」
      的数值兑现——9/7 分析滤波器即精心设计产物）；binomial 居中且与最优差距
      <2.5 dB（「decent job … aliasing is of little concern」读数承载）；QMF-9
      频域混叠功率>binomial（「aliases a fair amount」）
  C1–C3  常数：式 3.65 手算 h_{a=−1}(0.5)=0.625；binomial H(π/2)=(6+8cos
      (π/2)+2cos(π))/16=0.25 手算对照；式 3.67（h=tent 连续核）与式 3.66
      （h_dec=[0.25,0.5,0.25]）逐点等价（「keep the same kernel」语义）
  #S  同步：5 张程序生成图写入 code/figures/ 与 slides/figures/（md5 一致）：
      e029_interp_kernels（Figure 3.28 语义：tent/双三次 a=−0.5/a=−1/
      windowed sinc 连续曲线+对数频幅，标「原书图未使用，程序重绘」）/
      e029_interp_2d（Figure 3.27 语义：2D 放大四例）/e029_decim_freq
      （Figure 3.30+Table 3.2 语义：七滤波器频响+系数标注，标「原书表未
      使用，程序重绘」）/e029_aliasing（混叠实测四联+PSNR 读数）/
      e029_polyphase（Figure 3.26 语义：加权和 vs 多相两模型一维示意）
      ——均标「原书图未使用」
  #D  区分力（4 项）：D-1 式 3.65 第二段漏 (|x|−1) 因子 → #2 基本性质拦截；
      D-2 Table 3.2 系数转录错位 → #3 和=1 拦截；D-3 抽稀忘预滤波 → #5
      实测最差拦截；D-4 a=−0.5 与 a=−1 角色互换 → #2 二次再现判据拦截

输出：stdout 存档 code/outputs/e029_interp_decim_runN.txt（N 递增，refs 指向最新）
"""
import hashlib
import pathlib
import sys

import numpy as np

FAILS = []
COUNT = {"n": 0}


def chk(label, cond, detail=""):
    COUNT["n"] += 1
    ok = bool(cond)
    if not ok:
        FAILS.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


rng = np.random.default_rng(20261029)

# ---------- 原书算子（忠实式 3.64/3.65/3.66/3.67 与 Table 3.2） ----------

def tent(x):
    """tent 核（线性插值核）max(0, 1−|x|)；整数采样 [0,1,0]、½ 采样 [0,0.5,1,0.5,0]。"""
    return np.maximum(0.0, 1.0 - np.abs(np.asarray(x, dtype=float)))


def cubic(x, a):
    """式 3.65：C1 分段三次插值核。a=x=1 处导数（Parker, Kenyon, and Troxel 1983）。"""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    m1 = x < 1
    out[m1] = 1.0 - (a + 3.0) * x[m1] ** 2 + (a + 2.0) * x[m1] ** 3
    m2 = (x >= 1) & (x < 2)
    out[m2] = a * (x[m2] - 1.0) * (x[m2] - 2.0) ** 2
    return out


def bspline_cubic(x):
    """cubic B-spline=tent⋆tent 的解析式（分段三次、knots 整数、峰 2/3）：
    |x|≤1: 2/3−x²+|x|³/2；1<|x|<2: (2−|x|)³/6；else 0。解析式由数值卷积
    核对（#1-4）。"""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    m1 = x <= 1
    out[m1] = 2.0 / 3.0 - x[m1] ** 2 + x[m1] ** 3 / 2.0
    m2 = (x > 1) & (x < 2)
    out[m2] = (2.0 - x[m2]) ** 3 / 6.0
    return out


def bspline_quad(x):
    """quadratic B-spline=tent⋆box 的解析式（分段二次、knots 半整数、峰 0.75）：
    |x|≤0.5: 3/4−x²；0.5<|x|<1.5: (3/2−|x|)²/2；else 0。其 ½ 像素采样归一
    精确=binomial（#1-5，见口径注记 B-spline 称呼交叉条）。"""
    x = np.abs(np.asarray(x, dtype=float))
    out = np.zeros_like(x)
    m1 = x <= 0.5
    out[m1] = 0.75 - x[m1] ** 2
    m2 = (x > 0.5) & (x < 1.5)
    out[m2] = 0.5 * (1.5 - x[m2]) ** 2
    return out


def sinc2(x):
    """r=2 抽稀口径的归一 sinc：sin(πx/2)/(πx/2)（Nyquist 截止 π/2）。"""
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(np.pi * x[nz] / 2.0) / (np.pi * x[nz] / 2.0)
    return out


def windowed_sinc_cont(x):
    """Figure 3.28 连续曲线的讲者口径：raised-cosine 窗 cos²(πx/5) 调制 sinc₂，
    |x|≤2.5 截断（原书未给窗的精确定义——口径注记申报；与 Table 3.2 表值
    方向一致但不逐一相等）。"""
    x = np.asarray(x, dtype=float)
    w = np.where(np.abs(x) <= 2.5, np.cos(np.pi * x / 5.0) ** 2, 0.0)
    return sinc2(x) * w


# Table 3.2 六种 r=2 抽稀滤波器（奇长、对称、单位 DC 增益；照表忠实录入，
# 顺序 [c0, c1, …] 为 |n| 递增；留档 L334–381）
TABLE32 = {
    "linear":        [0.50, 0.25],
    "binomial":      [0.3750, 0.2500, 0.0625],
    "cubic a=-1":    [0.5000, 0.3125, 0.0000, -0.0625],
    "cubic a=-0.5":  [0.50000, 0.28125, 0.00000, -0.03125],
    "windowed sinc": [0.4939, 0.2684, 0.0000, -0.0153, 0.0000],
    "QMF-9":         [0.5638, 0.2932, -0.0519, -0.0431, 0.0198],
    "JPEG 2000":     [0.6029, 0.2669, -0.0782, -0.0169, 0.0267],
}


def sym_full(coef):
    """|n| 递增的半边系数 → 完整对称核（n=−(K−1)…0…+(K−1)）。"""
    return np.concatenate([coef[1:][::-1], coef])


def freq_resp(coef, w):
    """实偶序列 DTFT：H(ω)=c0+2Σ_{n≥1} c_n cos(nω)。"""
    c = sym_full(coef)
    n = np.arange(len(c)) - (len(c) - 1) // 2
    return c @ np.cos(np.outer(n, w))


def alias_power(coef, npts=8192):
    """混叠功率 E_alias（讲者口径）：∫_{π/2}^{π}|H(ω)|²dω/π 数值积分。"""
    w = np.linspace(np.pi / 2, np.pi, npts)
    H = freq_resp(coef, w)
    return float(np.mean(H ** 2))


def interp_direct(f, kern, r=2, half=2.5):
    """式 3.64：g(i)=Σ_k f(k)·h(i−rk)。kern(x) 为连续插值核（接受数组；
    tent 插值核=tent(x/r) 由调用方传入），half 为核支撑半径（超过者不贡献，
    式 3.64 的有限求和域）。按 k 循环+切片乘加（等价双循环求和）。"""
    f = np.asarray(f, dtype=float)
    N = len(f)
    M = r * (N - 1) + 1
    g = np.zeros(M)
    L = int(np.ceil(half))
    for k in range(N):
        lo, hi = max(0, r * k - L), min(M - 1, r * k + L)
        g[lo:hi + 1] += f[k] * kern(np.arange(lo, hi + 1) - r * k)
    return g


def interp_polyphase(f, r=2):
    """多相形式（Figure 3.26b）：按 i mod r 拆成 r 个子核分别卷积后相加。
    tent 插值核 tent(x/2) 在 r=2 的偶相采样 h(2n)=[0,1,0]、奇相采样
    h(2n±1)=[0.5,0.5]（g(2m+1)=0.5(f(m)+f(m+1))）。"""
    f = np.asarray(f, dtype=float)
    N = len(f)
    M = r * (N - 1) + 1
    even = np.zeros(M)
    odd = np.zeros(M)
    even[::r] = f
    slots = odd[1::r]
    # np.convolve(f,[0.5,0.5])[j]=0.5f(j)+0.5f(j−1)；奇相位 g(2m+1)=0.5(f(m)+f(m+1))
    # 对应 convolve 结果 index m+1（末位是补零伪项，截掉）
    vals = np.convolve(f, [0.5, 0.5])[1: 1 + len(slots)]
    slots[: len(vals)] = vals
    return even, odd, even + odd


def decim_366(f, coef, r=2):
    """式 3.66：g(i)=Σ_k f(k)h(ri−k)——预滤波核 h（半边系数 coef）与只在
    r 网格上取值合并实现（「we usually only evaluate the convolution at
    every rth sample」）。"""
    f = np.asarray(f, dtype=float)
    h = sym_full(coef)
    off = (len(h) - 1) // 2
    M = (len(f) + r - 1) // r
    pad = np.concatenate([np.zeros(off), f, np.zeros(off + r)])
    return np.array([float(np.dot(pad[i * r: i * r + len(h)], h[::-1]))
                     for i in range(M)])


def decim_367(f, kern, r=2, half=1.0):
    """式 3.67：g(i)=(1/r)Σ_k f(k)h(i−k/r)——同一核 h（连续形式）与 (1/r)
    缩放（「keep the same kernel h(k,l) for both interpolation and
    decimation」）。按 k 循环+切片乘加（tent 核支撑 |x|≤1 时每 k 只贡献
    2~3 个输出点）。"""
    f = np.asarray(f, dtype=float)
    N = len(f)
    M = (N + r - 1) // r
    g = np.zeros(M)
    for k in range(N):
        c = k / r
        lo, hi = max(0, int(np.ceil(c - half))), min(M - 1, int(c + half))
        if lo <= hi:
            g[lo:hi + 1] += f[k] * kern(np.arange(lo, hi + 1) - c)
    return g / r


def mse_psnr(sig, ref, peak):
    """PSNR 式 2.120 口径（10·log₁₀(peak²/MSE)）。"""
    sig, ref = np.asarray(sig, dtype=float), np.asarray(ref, dtype=float)
    n = min(len(sig), len(ref))
    m = float(np.mean((sig[:n] - ref[:n]) ** 2))
    return 10.0 * np.log10(peak * peak / m) if m > 0 else float("inf")


# ---------- 实验组 ----------

def exp1():
    """#1 式 3.64 插值与两模型（Figure 3.26 语义；5 条）。"""
    print("== #1 式 3.64 插值与两模型（Figure 3.26：加权和 vs 多相） ==")
    n = 8
    ramp = np.linspace(0.0, 1.0, n)
    # 脚注 11：单位面积平滑核幅度×r 即得插值核（门A r1 P0-1 裁定口径）；等价
    # 视角：单位面积核拉宽 r 倍峰高剩 1/r，乘回 r 峰高回 1 → tent(x/r)。
    g = interp_direct(ramp, lambda x: tent(x / 2.0), r=2)
    xs = np.arange(len(g)) / 2.0
    ref = np.interp(xs, np.arange(n), ramp)
    chk("#1-1 式 3.64（线性插值核 tent(x/r)）对线性斜坡逐点精确（piecewise linear）",
        np.allclose(g, ref, atol=1e-12),
        f"max err {np.max(np.abs(g - ref)):.2e}")
    even, odd, gp = interp_polyphase(ramp, r=2)
    chk("#1-2 多相形式（偶相+奇相子核）与式 3.64 直接式逐点一致（two forms are equivalent）",
        np.allclose(gp, g, atol=1e-12),
        f"max err {np.max(np.abs(gp - g)):.2e}")
    chk("#1-3 偶相在原样本处精确还原 f（h(2n) 采样 [0,1,0]）、奇相为邻均值 [0.5,0.5]",
        np.allclose(even[::2], ramp, atol=1e-12)
        and np.allclose(odd[1::2][: n - 1], (ramp[:-1] + ramp[1:]) / 2, atol=1e-12),
        f"even err {np.max(np.abs(even[::2] - ramp)):.1e}")
    grid = np.linspace(-4, 4, 4001)
    tent_g = tent(grid)
    conv = np.convolve(tent_g, tent_g, mode="same") * (grid[1] - grid[0])
    b_analytic = bspline_cubic(grid)
    err = float(np.max(np.abs(conv - b_analytic)))
    chk("#1-4 cubic B-spline=tent⋆tent（数值卷积 vs 解析式 2/3−x²+|x|³/2 一致；"
        "derive by convolving the tent with itself）",
        err < 5e-3, f"max err {err:.2e}（网格步长 {grid[1]-grid[0]:.4f} 限制），峰 {b_analytic[2000]:.4f}")
    half_samples = bspline_quad(np.arange(-2, 3) / 2.0)
    half_norm = half_samples / half_samples.sum()
    binom = np.array([1, 4, 6, 4, 1]) / 16.0
    chk("#1-5 tent⋆box（quadratic B-spline，峰 0.75）½ 像素采样归一=[1,4,6,4,1]/16 "
        "精确=binomial（binomial 对应关系见口径注记）且峰 0.75≠1（approximating "
        "kernel 不过样本点）",
        np.allclose(half_norm, binom, atol=1e-12)
        and abs(float(bspline_quad(0.0)) - 0.75) < 1e-12,
        f"½ 采样 {np.round(half_norm, 4)} vs binomial {binom}")
    # 脚注 11 幅度×r 口径实测（门A r1 P0-1）：[1,2,1]/4×2=[0.5,1,0.5] 代式 3.64
    h_map = {0: 1.0, 1: 0.5}                      # 对称核 [0.5,1,0.5]，中心 h(0)=1
    g_scale = interp_direct(ramp, lambda x: np.where(np.abs(x) < 0.5, 1.0,
                                  np.where(np.abs(x) < 1.5, 0.5, 0.0)),
                            r=2, half=1.0)
    chk("#1-6 脚注 11 幅度×r 实测：[1,2,1]/4×2=[0.5,1,0.5] 代式 3.64 偶相原样"
        "还原、奇相邻均值，与 tent(x/r) 连续核口径（#1-1）逐点全等（unit area → "
        "scale up by r；「拉伸」与「×r」两视角同一件事）",
        np.allclose(g_scale[::2][: n - 1], ramp[:-1], atol=1e-12)
        and np.allclose(g_scale[1::2][: n - 1], (ramp[:-1] + ramp[1:]) / 2, atol=1e-12)
        and np.allclose(g_scale, g, atol=1e-12),
        f"max|Δ(scale−tent(x/r))| {np.max(np.abs(g_scale - g)):.2e}")
    return g


def exp2():
    """#2 式 3.65 三次核（6 条）。"""
    print("== #2 式 3.65 三次插值核（a=−1 与 a=−0.5） ==")
    h0 = float(cubic(0.0, -1.0))
    ok_basic = (abs(h0 - 1) < 1e-15
                and np.allclose(cubic([1, -1, 2, -2, 2.5], -1.0), 0, atol=1e-15))
    chk("#2-1 式 3.65 基本性质：h(0)=1、h(±1)=h(±2)=0、|x|≥2=0", ok_basic,
        f"h(0)={h0:.1f}")
    eps = 1e-6
    dl = (float(cubic(1.0 - eps, -1.0)) - float(cubic(1.0, -1.0))) / (-eps)
    dr = (float(cubic(1.0 + eps, -1.0)) - float(cubic(1.0, -1.0))) / eps
    chk("#2-2 x=1 处左右导数=a（a specifies the derivative at x=1；a=−1 档）",
        abs(dl - (-1.0)) < 1e-4 and abs(dr - (-1.0)) < 1e-4,
        f"左 {dl:.6f} / 右 {dr:.6f} vs a=−1")
    xs = np.arange(0.0, 6.0, 0.5)
    errs = {"1": [], "x": [], "x2": []}
    for x in xs:
        ns = np.arange(int(np.floor(x - 2)), int(np.floor(x + 2)) + 1)
        ns = ns[np.abs(x - ns) < 2]
        w = cubic(x - ns, -0.5)
        for name, fn in [("1", lambda t: np.ones_like(np.asarray(t, dtype=float))),
                         ("x", lambda t: t), ("x2", lambda t: t * t)]:
            errs[name].append(abs(float(w @ fn(ns)) - float(fn(x))))
    mx = max(max(errs["1"]), max(errs["x"]), max(errs["x2"]))
    chk("#2-3 a=−0.5 精确再现线性与二次（interpolates linear and quadratic exactly）",
        mx < 1e-12, f"max err {mx:.2e}")
    ramp_err = []
    for x in np.arange(0.25, 2.0, 0.25):
        ns = np.arange(int(np.floor(x - 2)), int(np.floor(x + 2)) + 1)
        ns = ns[np.abs(x - ns) < 2]
        ramp_err.append(float(cubic(x - ns, -1.0) @ ns) - x)
    over, under = max(ramp_err), min(ramp_err)
    chk("#2-4 a=−1 不线性再现直线（过冲/欠冲双向≥0.05——visible ringing 的样条读法）",
        over >= 0.05 and under <= -0.05, f"过冲 +{over:.4f} / 欠冲 {under:.4f}")
    # 半带端点 ω=π/2 处三核 H 恰好同为 0.5（½ 采样核的公共端点性质），
    # 带内偏差度量取 [0, 0.45π]（讲者口径，避开端点退化）
    w = np.linspace(0, 0.45 * np.pi, 512)
    devs = {}
    for key in ("cubic a=-1", "cubic a=-0.5", "linear"):
        devs[key] = float(np.max(np.abs(freq_resp(TABLE32[key], w) - 1.0)))
    chk("#2-5 a=−1 频响最贴 sinc（½ 像素采样核带内 [0,0.45π] 最大偏差 a=−1 < a=−0.5 < "
        "tent/linear；best matches the frequency characteristics of a sinc）",
        devs["cubic a=-1"] < devs["cubic a=-0.5"] < devs["linear"],
        f"偏差 {devs['cubic a=-1']:.4f} / {devs['cubic a=-0.5']:.4f} / {devs['linear']:.4f}")
    n15 = abs(float(cubic(1.5, -1.0)))
    n05 = abs(float(cubic(1.5, -0.5)))
    chk("#2-6 a=−1 负瓣幅度大于 a=−0.5（|h(±1.5)| 0.125 vs 0.0625——sharpening "
        "与过冲振铃的时域成因）",
        n15 > n05, f"|h(1.5)| a=−1: {n15:.4f} vs a=−0.5: {n05:.4f}")
    return devs


def exp3():
    """#3 Table 3.2 忠实录入（6 条）。"""
    print("== #3 Table 3.2 六种 r=2 抽稀滤波器（odd length, symmetric, unit DC gain） ==")
    sums = {k: float(sym_full(v).sum()) for k, v in TABLE32.items()}
    chk("#3-1 七滤波器系数和=1（unit DC gain；windowed sinc/QMF-9/JPEG 2000 表值 "
        "4 位舍入，容差 5e-4）",
        all(abs(s - 1) < 5e-4 for s in sums.values()),
        str({k: round(v, 6) for k, v in sums.items()}))
    chk("#3-2 逐条对称（c_n=c_{−n}）",
        all(np.allclose(sym_full(v), sym_full(v)[::-1], atol=1e-15)
            for v in TABLE32.values()))
    manual = {
        "windowed sinc": [0.4939, 0.2684, 0.0000, -0.0153, 0.0000],
        "QMF-9": [0.5638, 0.2932, -0.0519, -0.0431, 0.0198],
        "JPEG 2000": [0.6029, 0.2669, -0.0782, -0.0169, 0.0267],
    }
    chk("#3-3 windowed sinc/QMF-9/JPEG 2000 系数与留档 L365–381 逐值一致（防转录错）",
        all(np.allclose(TABLE32[k], v, atol=1e-15) for k, v in manual.items()),
        str({k: TABLE32[k] for k in manual}))
    chk("#3-4 binomial=[1,4,6,4,1]/16（与 E024/E027 口径同源）",
        np.allclose(TABLE32["binomial"], (np.array([1, 4, 6, 4, 1]) / 16.0)[2:], atol=1e-15),
        str(TABLE32["binomial"]))
    chk("#3-5 linear=[1,2,1]/4（tent 平滑核）",
        np.allclose(TABLE32["linear"], (np.array([1, 2, 1]) / 4.0)[1:], atol=1e-15),
        str(TABLE32["linear"]))
    # 「stretched and re-scaled version of an interpolation kernel」的数值兑现：
    # 四列 = 对应插值核 ½ 像素采样（h(n/2), n=0,±1,…）再归一（和=2 → ÷2）
    deriv = {
        "linear": tent(np.arange(-2, 3) / 2.0),
        "binomial": bspline_quad(np.arange(-2, 3) / 2.0),
        "cubic a=-1": cubic(np.arange(-6, 7) / 2.0, -1.0),
        "cubic a=-0.5": cubic(np.arange(-6, 7) / 2.0, -0.5),
    }
    ok6, det = True, {}
    for k, full in deriv.items():
        c0 = full[len(full) // 2]
        cn = [full[len(full) // 2 + i] for i in range(1, len(TABLE32[k]))]
        norm = c0 + 2 * sum(cn)
        got = [c0 / norm] + [c / norm for c in cn]
        ok6 &= bool(np.allclose(got, TABLE32[k], atol=1e-12))
        det[k] = f"{[round(v, 5) for v in got]} vs 表 {TABLE32[k]}"
    chk("#3-6 linear/binomial/cubic×2 四列=插值核 ½ 像素采样归一（stretched and "
        "re-scaled 的数值兑现）", ok6, str(det))
    return sums


def exp4():
    """#4 频响与 Figure 3.30 语义（5 条）。"""
    print("== #4 七滤波器频响（Figure 3.30 语义复算） ==")
    h0s = {k: float(freq_resp(v, np.array([0.0]))[0]) for k, v in TABLE32.items()}
    chk("#4-1 七滤波器 H(0)=1 精确（DC 增益；表值 4 位舍入容差 5e-4）",
        all(abs(h - 1) < 5e-4 for h in h0s.values()),
        str({k: round(v, 5) for k, v in h0s.items()}))
    hpi = {k: float(freq_resp(v, np.array([np.pi]))[0]) for k, v in TABLE32.items()}
    print("  [读数] H(π)（Nyquist 响应）："
          + ", ".join(f"{k}={v:+.4f}" for k, v in hpi.items()))
    w_45 = np.array([0.45 * np.pi])
    h45 = {k: float(freq_resp(v, w_45)[0]) for k, v in TABLE32.items()}
    chk("#4-2 cubic a=−1 通带保持高于 a=−0.5（|H(0.45π)|——两者半增益点同为 0.5π，"
        "a=−1 从更高处降下=sharper fall-off 的数值兑现）",
        h45["cubic a=-1"] > h45["cubic a=-0.5"],
        f"a=−1: {h45['cubic a=-1']:.4f} vs a=−0.5: {h45['cubic a=-0.5']:.4f}")
    mn = min(h45, key=h45.get)
    chk("#4-3 binomial |H(0.45π)| 为七者最小（cuts off a lot of frequencies——"
        "原文 L303–304）",
        mn == "binomial" and h45["windowed sinc"] > h45["binomial"],
        f"最小 {mn}={h45[mn]:.4f}；windowed sinc={h45['windowed sinc']:.4f}；"
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(h45.items(), key=lambda x: x[1])))
    alias = {k: alias_power(v) for k, v in TABLE32.items()}
    amin = min(alias, key=alias.get)
    chk("#4-4 混叠功率（E_alias 讲者口径）：binomial 七者最小、QMF-9 与 JPEG 2000 "
        "均高于 linear 基线（QMF-9 aliases a fair amount；JPEG 2000 七者最高=通带最宽"
        "的代价；完整排序与数值作读数）",
        amin == "binomial" and alias["QMF-9"] > alias["linear"]
        and alias["JPEG 2000"] > alias["linear"],
        f"排序 {' < '.join(sorted(alias, key=alias.get))}；"
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(alias.items(), key=lambda x: x[1])))
    q = float(freq_resp(TABLE32["QMF-9"], np.array([np.pi / 2]))[0])
    c2 = 2.0 * float(sum(ci * ci for ci in sym_full(TABLE32["QMF-9"])))
    chk("#4-5 QMF-9 与自逆条件一致（功率互补 |H(ω)|²+|H(ω+π)|²=2 的点检验（H(π/2)）+全频带均值检验（2Σc²）——"
        "H(π/2)·√2≈1 且 2Σc²≈1；单点检验不足以证明自逆，措辞为「与自逆条件一致」）",
        abs(q * np.sqrt(2) - 1) < 0.01 and abs(c2 - 1) < 0.01,
        f"H(π/2)={q:.4f}，×√2={q*np.sqrt(2):.4f}；2Σc²={c2:.4f}")
    w_res = np.linspace(0.5 * np.pi, 0.7 * np.pi, 129)
    res = {k: float(np.mean(np.abs(freq_resp(v, w_res)))) for k, v in TABLE32.items()}
    chk("#4-6 binomial 半带以上 [0.5π,0.7π] 的 |H| 均值明显大于零（相对理想半带"
        "低通的残留——still leaves a fair amount of high-frequency detail 的频域"
        "语义：E_alias 七者最小≠无残留）",
        res["binomial"] > 0.05,
        f"binomial 均值 {res['binomial']:.4f}（理想半带低通=0）；"
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(res.items(), key=lambda x: -x[1])))
    return alias, hpi


def make_scene():
    """合成测试信号（1D，512 点）：低频平滑+中频纹理+高频条纹（混叠源）。
    程序合成，非原书图（红线 2）。"""
    N = 512
    x = np.arange(N) / N
    smooth = 0.5 + 0.3 * np.sin(2 * np.pi * 3 * x)
    texture = 0.15 * np.sin(2 * np.pi * 47 * x)
    ring = 0.2 * np.sin(2 * np.pi * 180 * x)
    return smooth + texture + ring


def exp5():
    """#5 抽稀实现与混叠实测（6 条）。"""
    print("== #5 式 3.66/3.67 与混叠实测 ==")
    n = 64
    f = np.sin(2 * np.pi * np.arange(n) / 8.0) + 0.5
    g66 = decim_366(f, [0.5, 0.25], r=2)                 # [0.25,0.5,0.25] 的半边系数形式
    g67 = decim_367(f, tent, r=2)
    m = min(len(g66), len(g67))
    chk("#5-1 式 3.66（h_dec=[0.25,0.5,0.25]+隔 2 取样）与式 3.67（h=tent 连续核，"
        "(1/r) 缩放）逐点一致", np.allclose(g66[:m], g67[:m], atol=1e-12),
        f"max err {np.max(np.abs(g66[:m] - g67[:m])):.2e}")
    scene = make_scene()
    peak = float(np.max(np.abs(scene)))                  # 式 2.120 的 peak（讲者口径）
    def rec_of(g):
        return np.interp(np.arange(len(scene)) / 2.0, np.arange(len(g)), g)
    psnrs = {"无预滤波": mse_psnr(rec_of(scene[::2]), scene, peak)}
    for name, coef in TABLE32.items():
        g = decim_366(scene, coef, r=2)
        rec = interp_direct(g, lambda x: tent(x / 2.0), r=2)
        psnrs[name] = mse_psnr(rec, scene, peak)
    worst = min(psnrs, key=psnrs.get)
    bestk = max((k for k in psnrs if k != "无预滤波"), key=psnrs.get)
    chk("#5-2 七滤波器抽稀→插值回 PSNR 全部高于无预滤波口径",
        all(psnrs[k] > psnrs["无预滤波"] for k in TABLE32),
        f"无预滤波 {psnrs['无预滤波']:.2f} dB vs 最差滤波 "
        f"{min(psnrs[k] for k in TABLE32):.2f} dB")
    chk("#5-3 无预滤波直接隔点取样混叠最重（PSNR 最低）", worst == "无预滤波",
        f"{worst} {psnrs[worst]:.2f} dB")
    chk("#5-4 JPEG 2000 9/7 实测 PSNR 榜首与 linear 差 0.05 dB≤0.1 dB（记录区分不足——"
        "单一信号不足以检验「more careful filter design is required」）",
        bestk == "JPEG 2000" and psnrs["JPEG 2000"] - psnrs["linear"] <= 0.1,
        f"{bestk} {psnrs[bestk]:.2f} dB；与 linear 差 "
        f"{psnrs['JPEG 2000']-psnrs['linear']:.2f} dB")
    chk("#5-5 binomial 为七滤波器中 PSNR 最低，但与最优差距<2.5 dB（差距主要源于"
        "通带衰减更多、非混叠——[读数] |H(0.18π)| binomial 为七滤波器最低即其"
        "通带衰减最重的直接证据；E_alias 反而七者最小；decent job 与数据不矛盾——"
        "「aliasing is of little concern」属应用场景判断，PSNR 不承载）",
        psnrs["binomial"] == min(psnrs[k] for k in TABLE32)
        and 0 < psnrs[bestk] - psnrs["binomial"] < 2.5,
        f"binomial {psnrs['binomial']:.2f}（七滤波器最低）vs 最优 {bestk} "
        f"{psnrs[bestk]:.2f} dB")
    print("  [读数] 抽稀→插值回 PSNR（dB）：" + ", ".join(
        f"{k}={psnrs[k]:.2f}" for k in ["无预滤波"] + list(TABLE32)))
    w_mid, w_high = np.array([0.18 * np.pi]), np.array([2 * np.pi * 180 / 512])
    hm = {k: float(freq_resp(v, w_mid)[0]) for k, v in TABLE32.items()}
    hh = {k: float(freq_resp(v, w_high)[0]) for k, v in TABLE32.items()}
    print("  [读数] |H(0.18π)|（测试信号中频分量处，通带衰减直接压低该分量）："
          + ", ".join(f"{k}={v:.3f}" for k, v in sorted(hm.items(), key=lambda x: x[1])))
    print(f"  [读数] |H({w_high[0]/np.pi:.3f}π)|（高频条纹 180/512 处）："
          + ", ".join(f"{k}={v:.3f}" for k, v in sorted(hh.items(), key=lambda x: x[1])))
    chk("#5-6 QMF-9 频域混叠功率>binomial（频域口径；原文 aliases a fair amount）",
        alias_power(TABLE32["QMF-9"]) > alias_power(TABLE32["binomial"]),
        f"QMF-9 {alias_power(TABLE32['QMF-9']):.5f} vs binomial "
        f"{alias_power(TABLE32['binomial']):.5f}")
    return psnrs


def constants():
    """C1–C3 常数核对。"""
    print("== C1–C3 常数核对 ==")
    v = 1.0 - (-1.0 + 3) * 0.25 + (-1.0 + 2) * 0.125
    chk("C1 式 3.65 手算 h_{a=−1}(0.5)=1−(a+3)(0.25)+(a+2)(0.125)=0.625",
        abs(float(cubic(0.5, -1.0)) - v) < 1e-15 and abs(v - 0.625) < 1e-15,
        f"{float(cubic(0.5, -1.0)):.6f}")
    hb = float(freq_resp([0.3750, 0.2500, 0.0625], np.array([np.pi / 2]))[0])
    chk("C2 binomial H(π/2)=(6+8cos(π/2)+2cos(π))/16=0.25 手算对照",
        abs(hb - 0.25) < 1e-12, f"{hb:.6f}")
    f = np.arange(1, 9, dtype=float)
    gA = decim_366(f, [0.5, 0.25], r=2)
    gB = decim_367(f, tent, r=2)
    m = min(len(gA), len(gB))
    chk("C3 式 3.67（h=tent 连续核）与式 3.66（h_dec=[0.25,0.5,0.25]）逐点等价"
        "（keep the same kernel 语义）",
        np.allclose(gA[:m], gB[:m], atol=1e-12),
        f"max err {np.max(np.abs(gA[:m] - gB[:m])):.2e}")


def 区分力():
    """#D 区分力注入（不进正式计数）。"""
    print("== #D 区分力注入（判据应拦截） ==")

    def cubic_bad(x, a):
        x = np.abs(np.asarray(x, dtype=float))
        out = np.zeros_like(x)
        m1 = x < 1
        out[m1] = 1.0 - (a + 3.0) * x[m1] ** 2 + (a + 2.0) * x[m1] ** 3
        m2 = (x >= 1) & (x < 2)
        out[m2] = a * (x[m2] - 2.0) ** 2                # D-1：漏 (|x|−1) 因子
        return out

    ok_d1 = not (abs(float(cubic_bad(0.0, -1.0)) - 1) < 1e-15
                 and np.allclose(cubic_bad([1.0, 2.0], -1.0), 0, atol=1e-15))
    print(f"  [{'PASS' if ok_d1 else 'FAIL'}] D-1 式 3.65 第二段漏 (|x|−1) 因子 → h(1)≠0 被拦截")
    COUNT["n"] += 1
    if not ok_d1:
        FAILS.append("D-1")
    bad_tab = dict(TABLE32)
    bad_tab["binomial"] = [0.3750, 0.2500, 0.0625, 0.0625]
    s = float(sym_full(bad_tab["binomial"]).sum())
    ok_d2 = abs(s - 1) > 1e-6
    print(f"  [{'PASS' if ok_d2 else 'FAIL'}] D-2 Table 3.2 系数转录错位 → 和={s:.4f}≠1 被拦截")
    COUNT["n"] += 1
    if not ok_d2:
        FAILS.append("D-2")
    scene = make_scene()
    peak = float(np.max(np.abs(scene)))
    g_none = scene[::2]
    rec_none = np.interp(np.arange(len(scene)) / 2.0, np.arange(len(g_none)), g_none)
    g_best = decim_366(scene, TABLE32["windowed sinc"], r=2)
    rec_best = interp_direct(g_best, lambda x: tent(x / 2.0), r=2)
    p_none = mse_psnr(rec_none, scene, peak)
    p_best = mse_psnr(rec_best, scene, peak)
    ok_d3 = p_none < p_best
    print(f"  [{'PASS' if ok_d3 else 'FAIL'}] D-3 抽稀忘预滤波 → 实测 {p_none:.2f}<{p_best:.2f} dB 被拦截")
    COUNT["n"] += 1
    if not ok_d3:
        FAILS.append("D-3")
    xs = np.arange(0.0, 6.0, 0.5)
    err_quad = 0.0
    for x in xs:
        ns = np.arange(int(np.floor(x - 2)), int(np.floor(x + 2)) + 1)
        ns = ns[np.abs(x - ns) < 2]
        w = cubic(x - ns, -1.0)                          # D-4：a=−1 冒充 a=−0.5
        err_quad = max(err_quad, abs(float(w @ (ns ** 2)) - x * x))
    ok_d4 = err_quad > 1e-6
    print(f"  [{'PASS' if ok_d4 else 'FAIL'}] D-4 a=−1 冒充二次再现核 → 二次误差 {err_quad:.4f} 被拦截")
    COUNT["n"] += 1
    if not ok_d4:
        FAILS.append("D-4")


# ---------- 程序生成图（5 张，双路写入） ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS"],
                     "axes.unicode_minus": False})

FIG_NOTE = "程序生成，原书图未使用"


def save_fig(fig, name):
    """双路写入 code/figures/ 与 slides/figures/（E008 坑 5 教训）。"""
    here = pathlib.Path(__file__).resolve().parent
    for d in (here / "figures", here / "../slides/figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def dtft_cont(kern, half, w, npts=4097):
    """连续核的数值 DTFT：H(ω)=Δ·Σ h(nΔ)e^{−jωnΔ}（细网格求和，重绘用）。"""
    x = np.linspace(-half, half, npts)
    h = kern(x)
    return np.real(np.exp(-1j * np.outer(w, x)) @ h) * (x[1] - x[0])


def make_figs(psnrs):
    print("== #S 程序生成图（5 张，写入 figures/ 与 slides/figures/） ==")
    names = []

    # ① e029_interp_kernels：Figure 3.28 语义（核曲线+对数频幅）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.linspace(-4, 4, 1601)
    ws = np.linspace(0, np.pi, 1024)
    series = [("tent", lambda t: tent(t / 2.0), 2.0, "brown"),
              ("cubic a=-0.5", lambda t: cubic(t, -0.5), 2.0, "purple"),
              ("cubic a=-1", lambda t: cubic(t, -1.0), 2.0, "green"),
              ("windowed-sinc", windowed_sinc_cont, 2.5, "tab:blue")]
    for label, kern, half, c in series:
        axes[0].plot(x, kern(x), color=c, lw=1.6, label=label)
        H = dtft_cont(kern, half, ws)
        H = H / H[0]                                   # DC 归一（0 dB 对齐，Figure 3.28b 语义）
        axes[1].plot(ws / np.pi, 20 * np.log10(np.maximum(np.abs(H), 1e-6)),
                     color=c, lw=1.6, label=label)
    axes[0].set(xlabel="x", ylabel="h(x)", title="(a) 插值核曲线")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="归一化频率 f=ω/2π", ylabel="|H(ω)| (dB)", title="(b) 对数频幅")
    axes[1].set_ylim(-80, 5)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"插值核与其频响（{FIG_NOTE}；windowed sinc 曲线为讲者口径 raised-cosine 窗）",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "e029_interp_kernels.png")
    names.append("e029_interp_kernels.png")

    # ② e029_interp_2d：Figure 3.27 语义（2D 放大四例）
    yy, xx = np.mgrid[0:10, 0:10]
    small = 0.5 + 0.3 * np.cos(2 * np.pi * xx / 10) - 0.2 * (yy - 5) / 5
    small[3:6, 3:6] += 0.4
    small = np.clip(small, 0, 1)

    def up2d(img, kern, half=4.0):
        n0, n1 = img.shape
        out = np.zeros((2 * (n0 - 1) + 1, 2 * (n1 - 1) + 1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                ks = [(k, l) for k in range(n0) for l in range(n1)
                      if abs(i - 2 * k) <= half and abs(j - 2 * l) <= half]
                out[i, j] = sum(img[k, l] * kern(i - 2 * k, j - 2 * l) for k, l in ks)
        return out

    # r=2 插值核（脚注 11 拉伸支撑至 r）：tent(x/2) 过零 ±2、cubic(x/2,a) 过零
    # ±2/±4、sinc₂ 过零 ±2（本身即 r=2 口径）
    k_lin = lambda a, b: float(tent(a / 2.0)) * float(tent(b / 2.0))
    k_cub = lambda a, b, aa: float(cubic(a / 2.0, aa)) * float(cubic(b / 2.0, aa))

    def sinc_win(t):
        return sinc2(t) * np.where(np.abs(t) <= 2.5, np.cos(np.pi * t / 5) ** 2, 0)

    sinc_2d = lambda a, b: float(sinc_win(a)) * float(sinc_win(b))
    panels = [("双线性", k_lin),
              ("双三次 a=−1", lambda a, b: k_cub(a, b, -1.0)),
              ("双三次 a=−0.5", lambda a, b: k_cub(a, b, -0.5)),
              ("windowed sinc", sinc_2d)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.1))
    axes[0].imshow(small, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("原始 10×10", fontsize=9)
    for ax, (label, kern) in zip(axes[1:], panels):
        ax.imshow(up2d(small, kern), cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"2D 放大四例（Figure 3.27 语义对照，{FIG_NOTE}）", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "e029_interp_2d.png")
    names.append("e029_interp_2d.png")

    # ③ e029_decim_freq：Figure 3.30+Table 3.2 语义（七滤波器频响）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    wgrid = np.linspace(0, np.pi, 2048)
    colors = plt.cm.tab10(np.linspace(0, 1, len(TABLE32)))
    for (name, coef), c in zip(TABLE32.items(), colors):
        H = freq_resp(coef, wgrid)
        db = 20 * np.log10(np.maximum(np.abs(H), 1e-8))
        axes[0].plot(wgrid / np.pi, db, lw=1.5, color=c, label=name)
        axes[1].plot(wgrid / np.pi, db, lw=1.5, color=c)
    axes[1].set_xlim(0.3, 0.55)
    axes[1].set_ylim(-60, 5)
    axes[1].set_title("过渡带放大（0.3–0.55）", fontsize=9)
    for ax in axes:
        ax.set(xlabel="归一化频率 ω/π", ylabel="|H(ω)| (dB)")
        ax.axvline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].legend(fontsize=8, loc="center left", framealpha=0.95)
    coef_txt = "\n".join(f"{k}: {v}" for k, v in TABLE32.items())
    axes[0].text(0.02, 0.02, coef_txt, transform=axes[0].transAxes, fontsize=6.2,
                 va="bottom", family="monospace")
    fig.suptitle("Table 3.2 六种 r=2 抽稀滤波器频响（原书表未使用，程序重绘；灰虚线=半 Nyquist，ω/π=0.5）",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, "e029_decim_freq.png")
    names.append("e029_decim_freq.png")

    # ④ e029_aliasing：混叠实测（原图/无预滤波/binomial/windowed sinc）
    scene = make_scene()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    axes[0].plot(scene, lw=1.0, color="k")
    axes[0].set_title("原图 512 点", fontsize=9)
    show = [("无预滤波", None, "无预滤波"),
            ("binomial 预滤波", TABLE32["binomial"], "binomial"),
            ("windowed sinc 预滤波", TABLE32["windowed sinc"], "windowed sinc")]
    for ax, (label, coef, key) in zip(axes[1:], show):
        if coef is None:
            g = scene[::2]
        else:
            g = decim_366(scene, coef, r=2)
        rec = np.interp(np.arange(len(scene)) / 2.0, np.arange(len(g)), g)
        ax.plot(scene, lw=0.9, color="lightgray", label="原图")
        ax.plot(rec, lw=1.0, color="tab:red", label="抽稀→插值回")
        ax.set_title(f"{label}（{psnrs[key]:.1f} dB）", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("r=2 抽稀混叠实测（256 点取样→线性插值核 tent(x/r) 上采样回 512；"
                 f"测试图程序合成，{FIG_NOTE}）", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "e029_aliasing.png")
    names.append("e029_aliasing.png")

    # ⑤ e029_polyphase：Figure 3.26 语义（加权和 vs 多相一维示意）
    f8 = np.array([0.2, 0.8, 0.4, 1.0, 0.6, 0.3, 0.9, 0.5])
    g = interp_direct(f8, lambda x: tent(x / 2.0), r=2)
    even, odd, gp = interp_polyphase(f8, r=2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    i8 = np.arange(len(f8))
    axes[0].stem(i8, f8, basefmt=" ", linefmt="gray", markerfmt="ko")
    axes[0].plot(np.arange(len(g)) / 2, g, "tab:blue", lw=1.2)
    axes[0].set_title("(a) 加权和：每个输入样本放一个核，叠加", fontsize=9)
    axes[1].stem(i8, f8, basefmt=" ", linefmt="gray", markerfmt="ko")
    axes[1].plot(np.arange(0, len(g), 2), even[::2], "tab:green", marker="o",
                 lw=1.0, label="偶相 h(2n)=[0,1,0]")
    axes[1].plot(np.arange(1, len(g), 2), odd[1::2][: len(f8) - 1], "tab:orange",
                 marker="s", lw=1.0, label="奇相 h(2n±1)=[0.5,0.5]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("(b) 多相：按 i mod 2 拆两个子核（two forms are equivalent）", fontsize=9)
    for ax in axes:
        ax.set(xlabel="i（输出栅格）")
    fig.suptitle(f"式 3.64 的两种读法（Figure 3.26 语义，{FIG_NOTE}）", fontsize=10)
    fig.tight_layout()
    save_fig(fig, "e029_polyphase.png")
    names.append("e029_polyphase.png")

    here = pathlib.Path(__file__).resolve().parent
    for nm in names:
        a = (here / "figures" / nm).read_bytes()
        b = (here / "../slides/figures" / nm).resolve().read_bytes()
        chk(f"#S {nm} 写入两处且 md5 一致",
            hashlib.md5(a).hexdigest() == hashlib.md5(b).hexdigest(),
            f"md5 {hashlib.md5(a).hexdigest()[:8]}")
    print(f"  共 {len(names)} 张：{names}")


# ---------- 主流程 ----------

def main():
    run = 1
    print(f"== E029_interp_decim 数值实验 run{run} ==")
    print("口径注记（windowed sinc 窗与截断，与 f011 refs 逐字同步）：")
    print("原书已写明 raised-cosine（升余弦）窗（图 3.28 图注；")
    print("Table 3.2 条目作 cosine-windowed）；原书给出的截断——图 3.27d 为 nine taps，")
    print("Table 3.2 windowed sinc 为 |n|≤4，即 9 抽头（含两端零值），")
    print("原文未说明两者是否同一窗口；原书未给连续窗的窗宽参数——连续曲线重绘取讲者口径 ")
    print("h(x)=sinc₂(x)·cos²(πx/5)（|x|≤2.5），r=2 下 11 抽头（含两端零值），")
    print("与原书 9 抽头不同。本实验 Table 3.2 断言用表值口径（9 抽头），")
    print("2D 放大示例用讲者口径窗。")
    print("讲者口径系数（½ 采样归一）与 Table 3.2 表值（0.4939/0.2684/0/−0.0153/0）")
    print("方向一致但不逐一相等——Table 3.2 数值断言一律以表值忠实录入为准；QMF-9/")
    print("JPEG 2000 系数照表录入（文献值，原书未给解析式）。混叠功率")
    print("E_alias=∫_{π/2}^{π}|H(ω)|²dω/π 为讲者口径（原书未给该积分式）。")
    print("B-spline 称呼交叉（讲者复算）：tent⋆tent（分段三次，峰 2/3）½ 采样归一")
    print("=[0.333,0.240,0.083,0.010]≠binomial；与 binomial=[1,4,6,4,1]/16 精确对应的")
    print("是 tent⋆box（分段二次，峰 0.75）½ 采样——按字面采样复算对不上、对得上的")
    print("是二次样条，deck/口播按原书表述转述并以讲者按申报。脚注 11「scale up by")
    print("r」按幅度×r 复算（门A r1 P0-1 裁定）：单位面积平滑核×r 即得插值核——")
    print("[1,2,1]/4×2=[0.5,1,0.5] 代式 3.64 恰为线性插值（#1-6 与 tent(x/r) 全等）；")
    print("「拉伸」与「×r」两视角同一件事（×r 只对单位面积平滑核成立）。")
    print(f"[存档] 本 stdout 保存为 code/outputs/e029_interp_decim_run{run}.txt（文件名 N 与此标头一致）。\n")
    exp1()
    exp2()
    exp3()
    alias, hpi = exp4()
    psnrs = exp5()
    constants()
    make_figs(psnrs)
    n_main, f_main = COUNT["n"], len(FAILS)
    区分力()
    n_d, f_d = COUNT["n"] - n_main, len(FAILS) - f_main
    print(f"\n汇总：{n_main - f_main} 条 PASS / {f_main} FAIL"
          f"＋区分力 {n_d - f_d}/{n_d}（注入自检，不计数）")
    if FAILS:
        print("FAIL 明细：", FAILS)
        sys.exit(1)
    print("DONE")


if __name__ == "__main__":
    main()
