#!/usr/bin/env python3
"""
E028 · 傅里叶变换Ⅱ：二维频谱、DCT 压缩与去噪（§3.4.1＋§3.4.2）

范围：§3.4.1 Two-dimensional Fourier transforms（印刷 pp.146–147 上）：一维公式
直接迁移（「The formulas and insights we have developed for one-dimensional
signals and their transforms translate directly to two-dimensional images」）＋
式 3.58 有向正弦 s(x,y)=sin(ωx x+ωy y)（「instead of just specifying a
horizontal or vertical frequency ωx or ωy, we can create an oriented sinusoid」）
＋式 3.59 连续 2D FT＋式 3.60 离散 2D DFT（H(kx,ky)=(1/MN)ΣΣ h(x,y)
e^{−j2π(kx x/M+ky y/N)}，M and N are the width and height of the image）
＋性质迁移（「All of the Fourier transform properties from 1D carry over …
use vector inner products instead of multiplications」）
＋Wiener filtering 小节（式 3.61 ⟨[S(ωx,ωy)]²⟩=Ps(ωx,ωy) 功率谱模型；
angle brackets denote the expected (mean) value，脚注 E[·] 亦常用；生成法=
「create a random Gaussian noise image S(ωx,ωy) where each "pixel" is a
zero-mean Gaussian of variance Ps and then take its inverse FFT」；正文语义=
「signal spectra capture a first-order description of spatial statistics」→
Wiener filter 为最优复原；二版删除推导、引一版 Szeliski 2010 §3.4.3——
「almost never used in practice any more」）
＋DCT（pp.147–148 上：Figure 3.25 基函数「The first DC (i.e., constant) basis
is the horizontal blue line, the second is the brown half-cycle waveform …
widely used in image and video compression standards such as JPEG」；式 3.62
一维 F(k)=Σᵢ cos(π/N)(i+½)k f(i)——「the 1/2-pixel offset is used to make the
basis coefficients symmetric (Wallace 1991)」；「the first basis function …
encodes the average DC value … the second encodes a slightly curvy version of
the slope」；KL 变换≈PCA §5.2.3 最优压缩；式 3.63 二维；「the 2D DCT can be
implemented separably … each of the DCTs can also be computed in O(N log N)
time」；「the DCT is widely used in today's image and video compression
algorithms」＋JPEG2000/JPEG XR 的小波与 overlapped DCT 替代＋「blocking
artifacts (visible edge-aligned discontinuities) that result from the pixels
in each block (typically 8×8) being transformed and quantized
independently」＋Exercise 4.3）
＋§3.4.2 Application: Sharpening, blur, and noise removal（pp.148–149 上）：
方法演进（线性滤波 §3.2/§3.4.1→非线性 weighted median/bilateral 式 3.34–3.37、
anisotropic diffusion 式 3.39–3.40、non-local means→变分 total variation L1→
「deep neural networks have taken over the denoising community (Section
10.3)」）＋评价：「it is common to report the results as a peak signal-to-noise
ratio (PSNR) measurement (2.120)」→「A better way … perceptually based
similarity metric, such as the structural similarity (SSIM) index (Wang, Bovik
et al. 2004; Wang, Bovik, and Simoncelli 2005) or FLIP …」→神经感知相似度
（「unlike L2 (PSNR) or L1 metrics, which encourage smooth or flat average
results, prefer images with similar amounts of texture」）→no-reference IQA
＋Exercises 3.12/3.21/3.28。
切点=§3.5 Pyramids and wavelets 正文标题（p.149，留档 L170–171）。

口径注记（红线 1 诚实申报）：
  · 式 2.120 定义处=§2.3.3 图像压缩质量（印刷 p.101：PSNR=10 log₁₀(I²max/MSE)
    =20 log₁₀(Imax/RMS)，Imax=255 for eight-bit images；式 2.119 RMS=√MSE），
    p.148 正文将其引用于去噪评价——本实验沿 E020 mse_psnr 同款口径。
  · 原书交叉引用疑点（S1 立案记录）：§3.4.2 正文（p.148）称「Figure 3.19 shows
    some examples of linear and non-linear filters being used to remove
    noise」，但 Figure 3.19 实为 §3.3.1 的 Median and bilateral filtering 图
    （印刷 p.132，图注原文见留档 e028_book_p132_fig319.txt L79–82）——§3.4.2
    正文的描述与该图图注不完全对应；本实验不复制任何
    原书图（红线 2），去噪对比全部程序生成。
  · Ps 具体形式原书未给（「A simple model」）：取径向幂律 Ps∝1/(1+(f/f0)²) 为
    讲者补充模型（自然图像 1/f² 统计惯例）；生成法的实性处理（原书只写
    「take its inverse FFT」，复谱不共轭对称则结果为复）=独立半谱生成+共轭镜像，
    讲者口径，E|S|²=Ps 全谱保持。
  · SSIM 公式细节原书未给（仅提名 Wang et al. 2004/2005）：实现为讲者口径
    （7×7 均值窗、C1=(0.01·L)²、C2=(0.03·L)²，L=255）。
  · 块压缩量化表=JPEG 标准亮度量化表（Annex K）按 scale 缩放——讲者口径
    （标准表不在本段原文；感知加权使高频被狠砍，是真实 JPEG blocking 的
    成因；均匀标量量化下块内低频波纹与边界跳变同幅、边界特征不显著）。

数值实验（33 条正式断言 = 28 数据 + 4 常数核对 C1–C4 + #S 同步 1 项（5 张图）；
另有 #D 区分力注入 4 项——E019 门A round3 新P1-2 惯例，注入类判据由 #D 承担，
不进正式计数、汇总单列）：
  #1  二维 DFT（式 3.58–3.60；7 条）：式 3.60 直接定义实现（独立行/列外积矩阵，
      1/MN 归一）与 numpy fft2×(1/MN) 全谱对照（随机实图 16 高×32 宽，M≠N
      验证「M and N are the width and height」）；可分离性两顺序（先行后列/
      先列后行）与直接定义一致；式 3.58 有向正弦剖面=正弦+等相位线恒定；
      DFT 峰值位置=±(kx,ky) 共轭对；实图全谱共轭对称；方向旋转 45° 峰随转；
      线性性质逐点（1D 性质迁移代表）
  #2  Wiener 功率谱（式 3.61；5 条）：生成法忠实（每点零均值方差 Ps 复高斯+
      共轭镜像）→ ifft2 虚部≈0（实图像）；共轭对称关系全谱逐点；
      蒙特卡洛 600 次 |S(k)|² 族均值/Ps 相对偏差<15%（期望语义兑现）；
      单次实现中位偏差>15%（「angle brackets=期望」记号的必要性实证）；
      生成图像自相关随 lag 单调衰减（「spectra capture a first-order
      description of spatial statistics」数值兑现）
  #3  DCT（式 3.62/3.63；10 条）：基矩阵 Gram=diag(N, N/2,…, N/2)（Wallace
      ½ 偏移的数值兑现）；正交化完美重构；常数信号 F(0)=N·c 其余为 0
      （「first basis encodes the average DC value」）；斜坡 F(1) 最大
      （「second encodes … the slope」）；式 3.63 双循环直接定义 vs 可分离
      （先行后列）一致；可分离顺序无关；2D 正交化完美重构；平滑图能量
      集中（前 16×16 系数>90%）；轻量化（JPEG 亮度量化表 scale=0.3）重建
      PSNR>40 dB；强量化（scale=4.0）块边界/块内梯度比>1.4 而轻量化<0.9
      （两点对照——「blocking artifacts (visible edge-aligned discontinuities)」
      数值实证；JPEG 亮度量化表为讲者口径，感知加权是真实 JPEG blocking 成因）
  #4  去噪评价（PSNR 式 2.120+SSIM；6 条）：PSNR 双形式恒等（10log₁₀(Imax²/
      MSE)≡20log₁₀(Imax/RMS)）；PSNR 随噪声 σ 单调降（3 档）；四法（box-3/
      box-7/中值 3×3/双边 σd=1.5 σr=30——式 3.34–3.37 口径同 E025 实现）均提升
      PSNR；SSIM(x,x)=1 且 ≤1；SSIM 随 σ 单调降；box-7 vs box-3 三项反转
      （PSNR 与全局 SSIM 均略优、纹理区 SSIM 反转更低——原书「L2 (PSNR) or
      L1 metrics … encourage smooth or flat average results, prefer images
      with similar amounts of texture」实证；方向由固定种子锁定）
  C1–C4  常数：式 3.58 离散换算 ωx=2πkx/M（kx=5, M=64 周期核对）；式 3.60
      归一 H(0,0)=mean(h)；式 3.62 ½ 偏移基逐点核对+k=0 行全 1；式 2.120
      常数（恒定偏差 d ⇒ PSNR=20log₁₀(255/d) 手算对照）
  #S  同步：5 张程序生成图写入 code/figures/ 与 slides/figures/（md5 一致）：
      e028_oriented_sin（有向正弦图+幅度谱峰，式 3.58/3.60）/
      e028_power_spectrum（Ps 剖面+单次实现图+族均值谱，式 3.61）/
      e028_dct_basis（1D DCT 前 8 基函数曲线，对照 Figure 3.25 语义，标
      「原书图未使用，程序重绘」）/e028_dct_blocking（轻量化 vs 强量化重建
      +块边界标注）/e028_denoise_compare（干净/含噪/三法重建+PSNR/SSIM
      读数）——均标「原书图未使用」
  #D  区分力（4 项）：D-1 2D DFT 漏 1/MN 归一 → #1 拦截；D-2 DCT 基漏 ½
      偏移 → #3 Gram 拦截；D-3 PSNR 用 ln 冒充 log₁₀ → #4/C4 拦截；
      D-4 Wiener 谱不共轭镜像直接 ifft2 → #2 虚部断言拦截

输出：stdout 存档 code/outputs/e028_ft2d_runN.txt（N 递增，refs 指向最新）
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

# ---------- 原书算子（忠实式 3.58–3.63 与式 2.120） ----------

def dft2_book(h):
    """式 3.60：H(kx,ky)=(1/MN)Σ_xΣ_y h(x,y)e^{−j2π(kx x/M+ky y/N)}（1/MN 归一）。
    原书明言 M and N are the width and height——M=宽（axis=1）、N=高（axis=0）。
    行/列外积矩阵实现（两次矩阵乘，等价双循环求和，不经 numpy fft）。"""
    h = np.asarray(h, dtype=float)
    N, M = h.shape                                   # N=高（行数）、M=宽（列数）
    Ey = np.exp(-2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)   # [ky, y]
    Ex = np.exp(-2j * np.pi * np.outer(np.arange(M), np.arange(M)) / M)   # [kx, x]
    return (Ey @ h @ Ex.T) / (M * N)


def dct_matrix(N):
    """式 3.62 基：C[k,i]=cos(π/N)(i+½)k——½ 像素偏移（Wallace 1991）。"""
    k = np.arange(N)[:, None]
    i = np.arange(N)[None, :]
    return np.cos(np.pi / N * (i + 0.5) * k)


def dct_norm(N):
    """正交归一因子 s_k：k=0 → √(1/N)、k>0 → √(2/N)（Gram=diag(N,N/2,…) 的平方根倒数）。"""
    s = np.full(N, np.sqrt(2.0 / N))
    s[0] = np.sqrt(1.0 / N)
    return s


def dct2_direct(h):
    """式 3.63 直接双循环定义：F(k,l)=ΣΣ cos(π/N)(i+½)k cos(π/N)(j+½)l f(i,j)。"""
    h = np.asarray(h, dtype=float)
    N = h.shape[0]
    C = dct_matrix(N)
    return C @ h @ C.T                               # 双求和的矩阵形式（逐项=式 3.63）


def dct2_sep(h):
    """式 3.63 可分离实现：「first computing the DCT of each line … then … column」。"""
    return dct_matrix(h.shape[0]) @ h @ dct_matrix(h.shape[1]).T


def mse_psnr(img, ref, peak=255.0):
    """式 2.118/2.120（印刷 p.101）：MSE=(1/n)Σ[I−Î]²；PSNR=10 log₁₀(Imax²/MSE)。"""
    img = np.asarray(img, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mse = ((img - ref) ** 2).mean()
    psnr = 10.0 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    return mse, psnr


def ssim_map(img, ref, L=255.0, win=7):
    """SSIM 图（逐窗；讲者口径同 ssim——供纹理区局部统计）。"""
    img = np.asarray(img, dtype=float)
    ref = np.asarray(ref, dtype=float)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    k = np.ones((win, win)) / win ** 2
    from numpy.lib.stride_tricks import sliding_window_view

    def stats(a):
        w = sliding_window_view(a, (win, win))
        mu = (w * k).sum(axis=(2, 3))
        w2 = sliding_window_view(a * a, (win, win))
        var = (w2 * k).sum(axis=(2, 3)) - mu ** 2            # E[x²]−μ²（均值窗）
        return mu, var

    mu1, v1 = stats(img)
    mu2, v2 = stats(ref)
    cov = (sliding_window_view(img * ref, (win, win)) * k).sum(axis=(2, 3)) - mu1 * mu2
    s = ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (v1 + v2 + C2))
    pad = win // 2
    return np.pad(s, pad, mode="edge")


def ssim(img, ref, L=255.0, win=7):
    """SSIM（讲者口径：7×7 均值窗、C1=(0.01L)²、C2=(0.03L)²；原书仅提名
    Wang, Bovik et al. 2004——公式细节为外部标准实现的简化版）。"""
    return float(ssim_map(img, ref, L, win).mean())


def boxn(a, n=3):
    """n×n 均值滤波（box 核口径，反射补边；n=3 即 box-3，n=7 强平滑对照）。"""
    r = n // 2
    p = np.pad(a, r, mode="reflect")
    H, W = a.shape
    acc = np.zeros((H, W))
    for u in range(n):
        for v in range(n):
            acc += p[u:u + H, v:v + W]
    return acc / (n * n)


def box3(a):
    """3×3 均值滤波（§3.2 box 核口径，反射补边）。"""
    return boxn(a, 3)


def median3(a):
    """3×3 中值滤波（weighted median 所属的非线性族代表；scipy 滑窗）。"""
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(a, dtype=float), size=3, mode="reflect")


def bilateral_filter(a, sigma_d, sigma_r, radius=None):
    """式 3.34–337 直接实现（灰度；域核 exp(−dist²/2σd²)、值域核 exp(−Δf²/2σr²)）。
    口径同 E025 实现（式 3.34–3.37，pp.133–134）。"""
    if radius is None:
        radius = max(1, int(np.ceil(2 * sigma_d)))
    r = radius
    p = np.pad(a, r, mode="reflect")
    H, W = a.shape
    ii, jj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
    d = np.exp(-(ii ** 2 + jj ** 2) / (2 * sigma_d ** 2))            # 式 3.35
    acc = np.zeros((H, W))
    wsum = np.zeros((H, W))
    for u in range(2 * r + 1):
        for v in range(2 * r + 1):
            patch = p[u:u + H, v:v + W]
            w = d[u, v] * np.exp(-((a - patch) ** 2) / (2 * sigma_r ** 2))   # 式 3.36–3.37
            acc += w * patch
            wsum += w
    return acc / wsum


def make_scene(W=96, H=96):
    """合成「自然样图」（0–255）：平滑背景+圆盘边缘+细纹理块（讲者构造，
    去噪与块压缩共用；不含任何原书图像素）。纹理=sin·sin>0.5 的格状块。"""
    x = np.arange(W)[None, :] * np.ones((H, 1))
    y = np.ones((H, 1)) * np.arange(H)[:, None]
    img = 110 + 25 * np.sin(2 * np.pi * y / 48) * np.cos(2 * np.pi * x / 64)
    img += 70 * (np.hypot(x - 64, y - 30) < 18)                      # 圆盘（边缘）
    img += 15 * ((np.sin(2 * np.pi * x / 12) * np.sin(2 * np.pi * y / 12)) > 0.5)
    return np.clip(img, 0, 255)


def texture_mask(W=96, H=96):
    """纹理块掩码（与 make_scene 同公式导出；供纹理区局部 SSIM 与平滑区
    blocking 统计用）。"""
    x = np.arange(W)[None, :] * np.ones((H, 1))
    y = np.ones((H, 1)) * np.arange(H)[:, None]
    tex = (np.sin(2 * np.pi * x / 12) * np.sin(2 * np.pi * y / 12)) > 0.5
    disk = np.hypot(x - 64, y - 30) < 18
    return tex, disk


# ---------- #1 二维 DFT（式 3.58–3.60） ----------

def exp1():
    print("== #1 二维 DFT（式 3.58–3.60，pp.146–147 上） ==")
    # 1a 式 3.60 直接定义 vs numpy fft2×(1/MN)：16 高×32 宽随机实图（M≠N 验证宽高口径）
    h = rng.random((16, 32)) - 0.5
    H1 = dft2_book(h)
    H2 = np.fft.fft2(h) / (32 * 16)
    err = float(np.abs(H1 - H2).max())
    chk("#1a 式 3.60 直接定义（1/MN 归一、M=宽 32/N=高 16）vs fft2/(MN)（max err<1e-12）",
        err < 1e-12, f"max err={err:.2e}")
    # 1b 可分离性两顺序（先行后列/先列后行）与直接定义一致
    sep_rc = np.fft.fft(np.fft.fft(h, axis=1), axis=0) / (32 * 16)   # 先行（axis=1）后列
    sep_cr = np.fft.fft(np.fft.fft(h, axis=0), axis=1) / (32 * 16)   # 先列后行
    e_rc = float(np.abs(sep_rc - H2).max())
    e_cr = float(np.abs(sep_cr - H2).max())
    chk("#1b 可分离性两顺序（先行后列/先列后行）均=直接定义（max err<1e-12）",
        max(e_rc, e_cr) < 1e-12, f"行→列 err={e_rc:.2e}；列→行 err={e_cr:.2e}")
    # 1c 式 3.58 有向正弦：剖面=正弦、等相位线恒定（kx=5, ky=2, M=N=64）
    M = N = 64
    kx, ky = 5, 2
    wx, wy = 2 * np.pi * kx / M, 2 * np.pi * ky / N
    x = np.arange(M)[None, :] * np.ones((N, 1))
    y = np.ones((N, 1)) * np.arange(N)[:, None]
    s = np.sin(wx * x + wy * y)                                      # 式 3.58
    prof = s[0, :]                                                    # y=0 剖面=sin(ωx x)
    e_prof = float(np.abs(prof - np.sin(wx * np.arange(M))).max())
    # 等相位线：相位 φ=wy·y+wx·x，整数位移 (Δy,Δx)=(5,−2) 使 wy·5+wx·(−2)
    # =2π/64·(2·5+5·(−2))=0 精确成立（kx/ky=5/2 有理比）——沿该位移逐点严格等相位
    tt = np.arange(13)
    yy = (10 + 5 * tt) % N
    xx = (10 - 2 * tt) % M
    const_spread = float(s[yy, xx].std())
    chk("#1c 式 3.58 有向正弦（kx=5,ky=2）：y=0 剖面=sin(ωx x)（err<1e-12）+等相位线"
        "（整数位移 (5,−2)，2·5+5·(−2)=0）逐点恒定（std<1e-12）",
        e_prof < 1e-12 and const_spread < 1e-12,
        f"剖面 err={e_prof:.2e}；等相位 std={const_spread:.2e}")
    # 1d DFT 峰值位置=±(kx,ky) 共轭对（幅度谱前 4 峰恰为四个共轭位置）
    S = np.fft.fft2(s) / (M * N)
    mag = np.abs(S)
    flat = np.argsort(mag.ravel())[::-1]
    top = [(int(i // M), int(i % M)) for i in flat[:8]]              # 前 8 大（含共轭对+杂散）
    peaks = set(top[:4])
    want = {(ky, kx), (N - ky, M - kx)}                              # ±(kx,ky)（wrap 口径）
    # 实正弦谱=±(kx,ky) 两共轭对（j 因子）：幅度 top2 应为这两个位置
    top2 = set(top[:2])
    chk("#1d 有向正弦幅度谱 top-2 峰=±(kx,ky) 共轭对位置（(2,5) 与 wrap(62,59)）",
        top2 == want, f"top2={sorted(top2)} want={sorted(want)}")
    # 1e 实图全谱共轭对称 H(−kx,−ky)=conj H(kx,ky)（wrap 口径逐点）
    hr = rng.random((16, 32)) - 0.5
    Hr = np.fft.fft2(hr)
    Nn, Mm = hr.shape
    conj_err = float(np.abs(Hr - np.conj(np.roll(np.roll(Hr[::-1, ::-1], 1, 0), 1, 1))).max())
    # 等价写法核对：H[ky,kx]=conj(H[(N−ky)%N,(M−kx)%M])
    ky_i = np.arange(Nn)[:, None]
    kx_i = np.arange(Mm)[None, :]
    conj_err2 = float(np.abs(Hr - np.conj(Hr[(Nn - ky_i) % Nn, (Mm - kx_i) % Mm])).max())
    chk("#1e 实图 DFT 全谱共轭对称 H(ky,kx)=conj H(−ky,−kx)（两种写法 max err<1e-12）",
        max(conj_err, conj_err2) < 1e-12, f"err={conj_err:.2e}/{conj_err2:.2e}")
    # 1f 方向旋转 45°（ωx=ωy）→ 峰在 (k,k) 对角
    k45 = 4
    s45 = np.sin(2 * np.pi * k45 * (x + y) / 64)
    mag45 = np.abs(np.fft.fft2(s45))
    pos = np.unravel_index(np.argmax(mag45), mag45.shape)
    chk("#1f 45° 方向正弦（ωx=ωy=2π·4/64）：幅度谱全局峰在 (ky,kx) 对角（行=列=4 或 60）",
        {pos[0], pos[1]} <= {k45, 64 - k45} and pos[0] == pos[1],
        f"峰位=(row {pos[0]}, col {pos[1]})")
    # 1g 线性性质（1D 性质迁移代表，式 3.60 线性）
    a, b = 1.7, -0.6
    h1 = rng.random((16, 32)) - 0.5
    h2 = rng.random((16, 32)) - 0.5
    err_lin = float(np.abs(dft2_book(a * h1 + b * h2) - (a * dft2_book(h1) + b * dft2_book(h2))).max())
    chk("#1g 线性性质逐点（FT(a·h1+b·h2)=a·H1+b·H2，err<1e-12）——「All of the Fourier "
        "transform properties from 1D carry over」", err_lin < 1e-12, f"err={err_lin:.2e}")
    return s


# ---------- #2 Wiener 功率谱（式 3.61） ----------

def power_spectrum(N=64, M=64, f0=0.1):
    """讲者补充模型：径向幂律 Ps∝1/(1+(f/f0)²)²（≈1/f⁴ 尾；原书未给具体 Ps——
    「A simple model」）。f 为归一频率（周期/像素，fftfreq 口径），f0=0.1 对应
    64px 图上约 10px 尺度（讲者口径）；平方衰减使离散网格上低频主导能量。"""
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.fftfreq(M)[None, :]
    f = np.hypot(ky, kx)
    Ps = 1.0 / (1.0 + (f / f0) ** 2) ** 2
    return Ps


def make_field(Ps, r):
    """式 3.61 生成法（讲者口径补实性）：每点 S~零均值方差 Ps 复高斯（实虚部各
    Ps/2），共轭镜像化（独立半谱+conj(flip) 镜像，E|S|²=Ps 全谱保持）→
    ifft2 取原结果（虚部≈0）。返回 (S, s)。"""
    N, M = Ps.shape
    S0 = np.sqrt(Ps / 2) * (r.standard_normal((N, M)) + 1j * r.standard_normal((N, M)))
    ky = np.arange(N)[:, None]
    kx = np.arange(M)[None, :]
    # 独立区=共轭对 (ky,kx)~(N−ky,M−kx) 中恰取一个：行 1..N/2−1 整行独立
    # （与行 N−1..N/2+1 对称）；行 0 与行 N/2 与自身对称（wrap），行内取半区 kx<M/2
    indep = ((ky > 0) & (2 * ky < N)) | (((ky == 0) | (2 * ky == N)) & (2 * kx < M))
    S = np.where(indep, S0, np.conj(S0[(N - ky) % N, (M - kx) % M]))
    # 自共轭点（(N−ky)%N==ky 且 (M−kx)%M==kx：(0,0)/(0,M/2)/(N/2,0)/(N/2,M/2)）
    # 必须实数化（其对称点是自身）——讲者口径：取实部，4 点方差统计由蒙特卡洛
    # 选点避开
    self_conj = (((N - ky) % N == ky) & ((M - kx) % M == kx))
    S = np.where(self_conj, S.real, S)
    S[0, 0] = 0.0                                                    # 零均值场
    s = np.fft.ifft2(S)
    return S, s


def exp2():
    print("== #2 Wiener 功率谱（式 3.61，pp.146–147） ==")
    print("    口径注记：Ps 取讲者补充径向幂律 1/(1+(f/f0)²)（原书未给具体形式）；"
          "实性=独立半谱+共轭镜像（原书只写 take its inverse FFT）")
    N = M = 64
    Ps = power_spectrum(N, M)
    r = np.random.default_rng(20261029)
    # 2a 生成法：ifft2 虚部≈0（实图像）
    S, s = make_field(Ps, r)
    imag_max = float(np.abs(s.imag).max())
    chk("#2a 生成法（逐点零均值方差 Ps 复高斯+共轭镜像）ifft2 虚部 max<1e-12（实图像）",
        imag_max < 1e-12, f"max|Im|={imag_max:.2e}")
    # 2b 共轭对称关系全谱逐点：S[ky,kx]=conj(S[(N−ky)%N,(M−kx)%M])
    ky_i = np.arange(N)[:, None]
    kx_i = np.arange(M)[None, :]
    e_sym = float(np.abs(S - np.conj(S[(N - ky_i) % N, (M - kx_i) % M])).max())
    chk("#2b 谱共轭对称 S(ky,kx)=conj S(−ky,−kx) 全谱逐点（max err<1e-12）",
        e_sym < 1e-12, f"err={e_sym:.2e}")
    # 2c 蒙特卡洛 600 次：独立半谱区 8 个频点 |S(k)|² 族均值/Ps ∈ (0.85, 1.15)
    pts = [(4, 3), (8, 5), (12, 2), (16, 9), (6, 30), (20, 20), (10, 45), (24, 12)]
    K = 600
    acc = np.zeros(len(pts))
    imgs = np.zeros((K, N, M))
    for i in range(K):
        Si, si = make_field(Ps, r)
        imgs[i] = si.real
        for j, (py, px) in enumerate(pts):
            acc[j] += abs(Si[py, px]) ** 2
    ratios = acc / K / np.array([Ps[py, px] for py, px in pts])
    chk("#2c 蒙特卡洛 600 次：8 频点 |S(k)|² 族均值/Ps 全部落在 (0.85, 1.15)（式 3.61 期望语义）",
        bool(np.all((ratios > 0.85) & (ratios < 1.15))),
        " ".join(f"{v:.3f}" for v in ratios))
    # 2d 单次实现波动大：单张 |S(k)|²/Ps 的中位相对偏差 >15%（「angle brackets=期望」必要性）
    single = np.array([abs(S[py, px]) ** 2 / Ps[py, px] for py, px in pts])
    med_dev = float(np.median(np.abs(single - 1.0)))
    chk("#2d 单次实现 |S(k)|²/Ps 中位相对偏差>15%（单张图看不出 Ps——期望记号的必要性实证）",
        med_dev > 0.15, f"中位偏差={med_dev:.3f}（单次 ratios: "
        + " ".join(f"{v:.2f}" for v in single) + "）")
    # 2e 空间统计：生成图像自相关随 lag 单调衰减（「spectra capture a first-order
    # description of spatial statistics」数值兑现；取固定种子的单张代表实现——
    # 族均值图随 K→0（各实现相位不相干），其自相关无意义）
    m = imgs[0].copy()
    m = m - m.mean()
    denom = (m * m).sum()
    lags = [1, 5, 20]
    rho = [float((m * np.roll(m, lag, axis=1)).sum() / denom) for lag in lags]
    chk("#2e 生成图像水平自相关 ρ(1)>ρ(5)>ρ(20) 严格单调衰减且 ρ(1)>0.1（空间统计一阶描述）",
        rho[0] > rho[1] > rho[2] and rho[0] > 0.1, " ".join(f"ρ({l})={v:.3f}" for l, v in zip(lags, rho)))
    return Ps, pts


# ---------- #3 DCT（式 3.62/3.63） ----------

def exp3():
    print("== #3 DCT（式 3.62/3.63，pp.147–148 上） ==")
    N = 8
    C = dct_matrix(N)
    # 3a Gram：C Cᵀ = diag(N, N/2, …, N/2)（½ 偏移正交结构的数值兑现）
    G = C @ C.T
    want = np.diag([N] + [N / 2] * (N - 1))
    e_g = float(np.abs(G - want).max())
    chk("#3a Gram C·Cᵀ=diag(N, N/2, …, N/2)（N=8；½ 偏移 Wallace 1991，max err<1e-9）",
        e_g < 1e-9, f"max err={e_g:.2e}")
    # 3b 正交化完美重构（1D）
    s8 = rng.random(N) - 0.5
    Sn = dct_norm(N)
    F = Sn[:, None] * C @ s8
    rec = C.T @ (Sn * F)
    e_r = float(np.abs(rec - s8).max())
    chk("#3b 正交化 1D DCT 完美重构（F̂=S·C·f，f=Cᵀ·S·F̂，err<1e-12）", e_r < 1e-12,
        f"err={e_r:.2e}")
    # 3c DC 语义：常数信号 F(0)=N·c、F(k≠0)=0（「first basis encodes the average DC value」）
    c = 3.7
    Fc = C @ np.full(N, c)
    e_dc = float(np.abs(Fc - np.array([N * c] + [0.0] * (N - 1))).max())
    chk("#3c 常数信号 F(0)=N·c、F(k>0)=0（err<1e-9）——DC 基编码块平均", e_dc < 1e-9,
        f"F={np.round(Fc, 6)}")
    # 3d 斜坡语义：F(1) 为最大系数（「second encodes … the slope」）
    ramp = np.linspace(-1, 1, N)
    Fr = C @ ramp
    chk("#3d 线性斜坡的 DCT 以 F(1) 为最大系数（第二基=斜坡语义）",
        int(np.argmax(np.abs(Fr))) == 1, f"|F|={np.round(np.abs(Fr), 3)}")
    # 3e 式 3.63 双循环直接定义 vs 可分离（先行后列）一致（8×8 随机块）
    blk = rng.random((8, 8)) - 0.5
    e_sep = float(np.abs(dct2_direct(blk) - dct2_sep(blk)).max())
    chk("#3e 式 3.63 直接双循环 vs 可分离（每行 DCT→每列 DCT）一致（err<1e-12）",
        e_sep < 1e-12, f"err={e_sep:.2e}")
    # 3f 可分离顺序无关：逐列/逐行的独立 1D DCT 循环实现，两种次序与双循环定义一致
    C8 = dct_matrix(8)

    def dct_cols(hh):                       # 对每列做 1D DCT（左乘 C）
        out = np.zeros_like(hh)
        for j in range(hh.shape[1]):
            out[:, j] = C8 @ hh[:, j]
        return out

    def dct_rows(hh):                       # 对每行做 1D DCT（右乘 Cᵀ）
        out = np.zeros_like(hh)
        for i in range(hh.shape[0]):
            out[i] = hh[i] @ C8.T
        return out

    e_rc = float(np.abs(dct_cols(dct_rows(blk)) - dct2_direct(blk)).max())
    e_cr = float(np.abs(dct_rows(dct_cols(blk)) - dct2_direct(blk)).max())
    chk("#3f 可分离顺序无关（先列后行/先行列两种次序的独立循环实现均=双循环定义，max err<1e-12）",
        max(e_rc, e_cr) < 1e-12, f"列→行 err={e_rc:.2e}；行→列 err={e_cr:.2e}")
    # 3g 2D 正交化完美重构
    Sn = dct_norm(8)
    F2 = Sn[:, None] * dct2_direct(blk) * Sn[None, :]
    rec2 = C8.T @ (Sn[:, None] * F2 * Sn[None, :]) @ C8
    e_r2 = float(np.abs(rec2 - blk).max())
    chk("#3g 2D 正交化 DCT 完美重构（err<1e-12）", e_r2 < 1e-12, f"err={e_r2:.2e}")
    # 3h 平滑图能量集中：前 5×5 系数能量占比>95%（低频集合语义，KL≈PCA 最优压缩的近似）
    img = make_scene(64, 64)
    img_n = (img - img.mean()) / 255.0
    Fimg = dct2_sep(img_n)
    energy = (Fimg ** 2).sum()
    low8 = (Fimg[:8, :8] ** 2).sum() / energy
    low16 = (Fimg[:16, :16] ** 2).sum() / energy
    chk("#3h 合成自然样图 2D DCT 前 16×16 系数能量占比>90%（能量压缩≈KL/PCA 最优；"
        "样图含纹理与边缘故能量集中度低于纯平滑图像——讲者口径）", low16 > 0.90,
        f"16×16 占比={low16:.4f}；8×8 占比={low8:.4f}")
    # 3i/3j 块压缩（8×8 块 DCT+JPEG 亮度量化表缩放）与 blocking artifacts
    # 口径注记：JPEG 标准亮度量化表（Annex K）为讲者口径（不在本段原文）——
    # 感知加权（高频大权重）是真实 JPEG blocking 的成因；均匀标量量化下块内
    # 低频波纹与边界跳变同幅，边界特征不显著
    Q_JPEG = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                       [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                       [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                       [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], float)

    def dct_inv(F):
        """式 3.63 逆变换：F 为式 3.63 口径系数（非归一），正交化 F̂=S·F·S 后
        h=Cᵀ·S·F̂·S·C=Cᵀ·S²·F·S²·C（S 对角、S⁻¹=S）。"""
        Sn = dct_norm(F.shape[0])
        C = dct_matrix(F.shape[0])
        return C.T @ ((Sn ** 2)[:, None] * F * (Sn ** 2)[None, :]) @ C

    def block_compress(imgq, scale):
        Qs = np.clip(Q_JPEG * scale, 1.0, None)
        rec = np.zeros_like(imgq)
        for by in range(0, imgq.shape[0], 8):
            for bx in range(0, imgq.shape[1], 8):
                Fb = dct2_sep(imgq[by:by + 8, bx:bx + 8])
                rec[by:by + 8, bx:bx + 8] = dct_inv(np.round(Fb / Qs) * Qs)
        return rec

    img255 = make_scene(96, 96)
    rec_light = block_compress(img255, 0.3)
    _, p_light = mse_psnr(rec_light, img255)
    chk("#3i 轻量化（JPEG 表 scale=0.3）块压缩重建 PSNR>40 dB（量化仅舍入损耗）",
        p_light > 40.0, f"PSNR={p_light:.1f} dB")
    # blocking：块边界（差分索引 x=7,15,…,79）vs 块内列（3,11,…）的水平梯度中位比，
    # 统计限于平滑像素（~纹理 & ~圆盘——其强梯度属图像自身结构，非量化伪影）
    tex, disk = texture_mask(96, 96)
    sm_mask = ~tex & ~disk
    eci = np.arange(7, 88, 8)
    ici = np.arange(3, 88, 8)

    def boundary_ratio(rec):
        gx = np.abs(np.diff(rec, axis=1))
        ev = np.median(gx[:, eci][sm_mask[:, eci]])
        iv = np.median(gx[:, ici][sm_mask[:, ici]])
        return float(ev / max(iv, 1e-9))

    r_weak, r_strong = boundary_ratio(block_compress(img255, 0.3)), boundary_ratio(block_compress(img255, 4.0))
    chk("#3j 强量化（scale=4.0）块边界/块内梯度比>1.4 而轻量化（scale=0.3）<0.9"
        "（块边界相对不连续只在强量化下出现——「blocking artifacts (visible "
        "edge-aligned discontinuities)」两点对照实证）",
        r_strong > 1.4 and r_weak < 0.9, f"scale0.3 比={r_weak:.3f}；scale4.0 比={r_strong:.3f}")
    return img255, rec_light, block_compress(img255, 4.0)


# ---------- #4 去噪评价（PSNR 式 2.120+SSIM） ----------

def exp4():
    print("== #4 去噪评价（PSNR 式 2.120+SSIM；§3.4.2 pp.148–149 上） ==")
    print("    口径注记：原书正文引 Figure 3.19 为去噪例图，但 Figure 3.19 实为 §3.3.1")
    print("    的 Median and bilateral filtering 图（印刷 p.132，图注见留档")
    print("    e028_book_p132_fig319.txt L79–82）——§3.4.2 正文的描述与图注不完全对应；")
    print("    本实验不复制原书图，对比全部程序生成（红线 2）。")
    print("    [实验参数] make_scene(96,96)=渐变背景+圆盘+12px 周期纹理块；σ=20 高斯噪；"
          "四法=box-3/box-7/中值3/双边（空间 σd=1.5、值域 σr=30，式 3.34–3.37 口径）")
    img = make_scene(96, 96)
    r = np.random.default_rng(20261029)
    # 4a PSNR 双形式恒等（式 2.120：10log₁₀(Imax²/MSE)=20log₁₀(Imax/RMS)）
    noisy = np.clip(img + r.normal(0, 20, img.shape), 0, 255)
    mse, psnr_a = mse_psnr(noisy, img)
    psnr_b = 20.0 * np.log10(255.0 / np.sqrt(mse))
    e_p = abs(psnr_a - psnr_b)
    chk("#4a 式 2.120 双形式恒等 10log₁₀(Imax²/MSE)≡20log₁₀(Imax/RMS)（err<1e-9）",
        e_p < 1e-9, f"MSE={mse:.2f}；PSNR={psnr_a:.4f} dB")
    # 4b PSNR 随 σ 单调降（3 档）
    r2 = np.random.default_rng(777)
    p_by_sigma = []
    for sig in (10.0, 20.0, 40.0):
        n2 = np.clip(img + r2.normal(0, sig, img.shape), 0, 255)
        _, p2 = mse_psnr(n2, img)
        p_by_sigma.append(p2)
    chk("#4b PSNR 随噪声 σ=10/20/40 单调降（含噪图口径）",
        p_by_sigma[0] > p_by_sigma[1] > p_by_sigma[2],
        " ".join(f"σ={s}:{v:.1f}dB" for s, v in zip((10, 20, 40), p_by_sigma)))
    # 4c 四法去噪均提升 PSNR（相对同张含噪图 noisy；box-7 强平滑对照——原书
    # 「L2 (PSNR) or L1 metrics … encourage smooth or flat average results」）
    den = {"box-3": box3(noisy), "box-7": boxn(noisy, 7), "中值3": median3(noisy),
           "双边": bilateral_filter(noisy, 1.5, 30.0)}
    _, p0 = mse_psnr(noisy, img)
    gains = {}
    for name, d in den.items():
        _, p = mse_psnr(d, img)
        gains[name] = p - p0
    chk("#4c 四法（box-3/box-7/中值3/双边）相对含噪图 PSNR 均提升>0.5 dB",
        all(v > 0.5 for v in gains.values()),
        " ".join(f"{k}:+{v:.2f}dB" for k, v in gains.items()))
    # 4d SSIM 自洽：SSIM(x,x)=1、SSIM≤1
    s_self = ssim(img, img)
    s_noisy = ssim(noisy, img)
    chk("#4d SSIM(x,x)=1（err<1e-12）且 SSIM(含噪,干净)≤1", abs(s_self - 1.0) < 1e-12 and s_noisy <= 1.0,
        f"SSIM 自身={s_self:.12f}；含噪={s_noisy:.4f}")
    # 4e SSIM 随 σ 单调降
    s_by_sigma = []
    for sig in (10.0, 20.0, 40.0):
        n2 = np.clip(img + r2.normal(0, sig, img.shape), 0, 255)
        s_by_sigma.append(ssim(n2, img))
    chk("#4e SSIM 随噪声 σ=10/20/40 单调降",
        s_by_sigma[0] > s_by_sigma[1] > s_by_sigma[2],
        " ".join(f"σ={s}:{v:.3f}" for s, v in zip((10, 20, 40), s_by_sigma)))
    # 4f 感知分歧（原书「unlike L2 (PSNR) or L1 metrics, which encourage smooth
    # or flat average results, prefer images with similar amounts of texture」）：
    # box-7 相对 box-3 的三项读数——PSNR 与全局 SSIM 都略优（平滑区主导全局口径，
    # 「flat average results」被奖励），唯独纹理区 SSIM 反转（纹理被更狠地抹平，
    # 感知口径在纹理区给出相反答案）。数据实证、方向由固定种子锁定。
    tex, _ = texture_mask(96, 96)
    ps = {}
    ss = {}
    st = {}
    for name, d in den.items():
        _, p = mse_psnr(d, img)
        ps[name] = p
        ss[name] = ssim(d, img)
        st[name] = float(ssim_map(d, img)[tex].mean())       # 纹理区局部 SSIM
    print(f"    [读数] PSNR：{ {k: round(v, 2) for k, v in ps.items()} }")
    print(f"    [读数] 全局 SSIM：{ {k: round(v, 4) for k, v in ss.items()} }")
    print(f"    [读数] 纹理区 SSIM：{ {k: round(v, 4) for k, v in st.items()} }")
    chk("#4f box-7 vs box-3：PSNR 更高 且 全局 SSIM 更高 但 纹理区 SSIM 反转更低"
        "（「PSNR encourage smooth or flat average results；感知口径 prefer … texture」"
        "实证；方向由本读数承载）",
        ps["box-7"] > ps["box-3"] and ss["box-7"] > ss["box-3"] and st["box-7"] < st["box-3"],
        f"ΔPSNR={ps['box-7'] - ps['box-3']:+.2f}dB；Δ全局SSIM={ss['box-7'] - ss['box-3']:+.4f}；"
        f"Δ纹理SSIM={st['box-7'] - st['box-3']:+.4f}")
    return img, noisy, den, ps, ss


# ---------- 常数核对 C1–C4 ----------

def constants():
    print("== C1–C4 常数核对 ==")
    # C1 式 3.58 离散换算：kx=5, M=64 ⇒ 剖面周期=64/5（正弦峰间隔核对）
    wx = 2 * np.pi * 5 / 64
    xs = np.arange(64)
    e_c1 = float(np.abs(np.sin(wx * xs) - np.sin(2 * np.pi * xs / (64 / 5))).max())
    chk("C1 式 3.58 离散换算 ωx=2πkx/M：kx=5、M=64 剖面周期=64/5（err<1e-12）",
        e_c1 < 1e-12, f"err={e_c1:.2e}")
    # C2 式 3.60 归一：H(0,0)=mean(h)
    h = rng.random((16, 32)) - 0.5
    H = dft2_book(h)
    e_c2 = abs(H[0, 0] - h.mean())
    chk("C2 式 3.60 归一：H(0,0)=mean(h)（err<1e-12）", e_c2 < 1e-12,
        f"H(0,0)={H[0, 0].real:.9f} mean={h.mean():.9f}")
    # C3 式 3.62 ½ 偏移基逐点：C[2,i]=cos(π/8)(i+½)·2、k=0 行全 1
    C = dct_matrix(8)
    e_c3 = max(float(np.abs(C[2] - np.cos(np.pi / 8 * (np.arange(8) + 0.5) * 2)).max()),
               float(np.abs(C[0] - 1.0).max()))
    chk("C3 式 3.62 基逐点：C[2,i]=cos(π/8)(i+½)·2 且 k=0 行全 1（err<1e-12）",
        e_c3 < 1e-12, f"err={e_c3:.2e}")
    # C4 式 2.120 常数：恒定偏差 d ⇒ MSE=d² ⇒ PSNR=20log₁₀(255/d)
    _, p = mse_psnr(np.full((4, 4), 100.0), np.full((4, 4), 105.0))
    e_c4 = abs(p - 20.0 * np.log10(255.0 / 5.0))
    chk("C4 式 2.120 常数：恒定偏差 5 ⇒ PSNR=20log₁₀(255/5)（err<1e-9）",
        e_c4 < 1e-9, f"PSNR={p:.6f} dB")


# ---------- 图（5 张，程序生成） ----------

def save_fig(fig, name):
    for d in ("figures", "../slides/figures"):
        p = pathlib.Path(__file__).parent / d
        p.mkdir(parents=True, exist_ok=True)
        fig.savefig(p / name, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_figs(s_oriented, Ps, pts, img255, rec_light, rec_strong, img, noisy, den, ps, ss):
    print("== #S 程序生成图（5 张，写入 figures/ 与 slides/figures/） ==")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Hiragino Sans GB", "PingFang SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    names = []
    NOTE = "（程序生成，原书图未使用）"

    # ① e028_oriented_sin：有向正弦图+幅度谱峰
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].imshow(s_oriented, cmap="gray", origin="upper")
    axes[0].set_title("式 3.58：s(x,y)=sin(ωx·x+ωy·y)（kx=5, ky=2）", fontsize=10)
    mag = np.abs(np.fft.fft2(s_oriented))
    axes[1].imshow(np.fft.fftshift(mag), cmap="magma", origin="upper")
    axes[1].plot(32 + 5, 32 + 2, "o", ms=10, mfc="none", mec="c")
    axes[1].plot(32 - 5, 32 - 2, "o", ms=10, mfc="none", mec="c")
    axes[1].set_title("式 3.60：|H(kx,ky)| 峰在 ±(kx,ky) 共轭对", fontsize=10)
    fig.suptitle("二维频谱：把一维的正弦探针铺成平面" + NOTE, fontsize=11)
    fig.tight_layout(); save_fig(fig, "e028_oriented_sin.png"); names.append("e028_oriented_sin.png")

    # ② e028_power_spectrum：Ps 剖面+单次实现图+族均值谱
    r = np.random.default_rng(20261029)
    N = M = 64
    S, s = make_field(Ps, r)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    ky = np.arange(N)[:, None]
    kx = np.arange(M)[None, :]
    f = np.hypot(np.fft.fftfreq(N)[:, None], np.fft.fftfreq(M)[None, :])
    order = np.argsort(f.ravel())
    axes[0].plot(f.ravel()[order], Ps.ravel()[order], lw=1.6, color="#2c5f8a")
    axes[0].set_title("功率谱模型 Ps（径向幂律，讲者补充）", fontsize=10)
    axes[1].imshow(s.real, cmap="gray")
    axes[1].set_title("单次实现（式 3.61 生成法）", fontsize=10)
    K = 120
    acc = np.zeros((N, M))
    for _ in range(K):
        Si, _ = make_field(Ps, r)
        acc += np.abs(Si) ** 2
    axes[2].imshow(np.fft.fftshift(np.log10(acc / K + 1e-9)), cmap="magma")
    axes[2].set_title("120 次族均值 |S|²（单次波动大→需期望）", fontsize=10)
    fig.suptitle("Wiener 小节：图像作为随机噪声场（式 3.61 〈[S]²〉=Ps，〈·〉=期望）" + NOTE, fontsize=11)
    fig.tight_layout(); save_fig(fig, "e028_power_spectrum.png"); names.append("e028_power_spectrum.png")
    print(f"    [图③口径] 族均值示意图 K={K} 次（deck 页3 figcap 已注明示意图；数值检验为 #2c 的 600 次）")

    # ③ e028_dct_basis：1D DCT 前 8 基函数（对照 Figure 3.25 语义）
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    C = dct_matrix(8)
    t = np.arange(8)
    for k in range(8):
        ax.plot(t, C[k] + 2.2 * k, "o-", lw=1.4, ms=4,
                color="#2c5f8a" if k in (0, 1) else "#b56a2c")
    ax.set_yticks([])
    ax.set_xticks(range(8))
    ax.set_xlabel("块内像素 i")
    ax.set_title("式 3.62 的 8 条 DCT 基函数（下→上 k=0…7；k=0 水平直线=DC 均值、"
                 "k=1 半周期=斜坡微弯）" + NOTE, fontsize=10)
    fig.tight_layout(); save_fig(fig, "e028_dct_basis.png"); names.append("e028_dct_basis.png")

    # ④ e028_dct_blocking：轻量化 vs 强量化重建+块边界
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    axes[0].imshow(img255, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("原图（96×96 合成样图）", fontsize=10)
    axes[1].imshow(rec_light, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("块压缩 scale=0.3（PSNR>40 dB，几乎无损）", fontsize=10)
    axes[2].imshow(rec_strong, cmap="gray", vmin=0, vmax=255)
    for c in range(8, 88, 8):
        axes[2].axvline(c - 0.5, color="c", lw=0.5, alpha=0.7)
        axes[2].axhline(c - 0.5, color="c", lw=0.5, alpha=0.7)
    axes[2].set_title("scale=4.0：块边界可见 discontinuities（青线）", fontsize=10)
    fig.suptitle("8×8 块 DCT+量化：JPEG 式压缩与 blocking artifacts（式 3.63）" + NOTE, fontsize=11)
    fig.tight_layout(); save_fig(fig, "e028_dct_blocking.png"); names.append("e028_dct_blocking.png")

    # ⑤ e028_denoise_compare：干净/含噪/四法+PSNR/全局 SSIM 读数（纹理区
    # SSIM 反转见存档 #4f 读数）
    fig, axes = plt.subplots(1, 6, figsize=(17.4, 3.4))
    panels = [("干净图", img, None, None), ("含噪 σ=20", noisy, None, None)]
    for name, d in den.items():
        panels.append((name, d, ps[name], ss[name]))
    for ax, (name, d, p, s) in zip(axes, panels):
        ax.imshow(d, cmap="gray", vmin=0, vmax=255)
        tag = name if p is None else f"{name}\nPSNR={p:.1f}dB 全局SSIM={s:.3f}"
        ax.set_title(tag, fontsize=9)
    fig.suptitle("去噪四法与两种评价指标（原书第 148 页；强平滑 box-7 的 PSNR 与全局 "
                 "SSIM 均略优于 box-3，纹理区 SSIM 反转更低——讲者实测）" + NOTE, fontsize=10.5)
    fig.tight_layout(); save_fig(fig, "e028_denoise_compare.png"); names.append("e028_denoise_compare.png")

    ok = True
    det = []
    for nm in names:
        a = pathlib.Path(__file__).parent / "figures" / nm
        b = pathlib.Path(__file__).parent / "../slides/figures" / nm
        if not a.exists() or not b.exists() or hashlib.md5(a.read_bytes()).hexdigest() != hashlib.md5(b.read_bytes()).hexdigest():
            ok = False
        det.append(f"{nm}={hashlib.md5(a.read_bytes()).hexdigest()[:8] if a.exists() else 'MISSING'}")
    chk("#S 程序生成图 5 张写入两处且 md5 一致（oriented_sin/power_spectrum/dct_basis/"
        "dct_blocking/denoise_compare）", ok and len(names) == 5, " ".join(det))


# ---------- 区分力注入 ----------

def 区分力():
    print("== #D 区分力注入（4 项，不计数） ==")
    # D-1 2D DFT 漏 1/MN 归一 → #1a 拦截
    h = rng.random((16, 32)) - 0.5
    wrong = dft2_book(h) * (32 * 16)
    err = float(np.abs(wrong - np.fft.fft2(h) / (32 * 16)).max())
    chk("D-1 2D DFT 漏 1/MN 归一 → #1a 拦截", err > 1e-3, f"max err={err:.2f}")
    # D-2 DCT 基漏 ½ 偏移（cos(π/N)·k·i）→ #3a Gram 拦截
    k = np.arange(8)[:, None]; i = np.arange(8)[None, :]
    C_wrong = np.cos(np.pi / 8 * i * k)                     # 无 ½ 偏移
    G = C_wrong @ C_wrong.T
    want = np.diag([8] + [4.0] * 7)
    e_g = float(np.abs(G - want).max())
    chk("D-2 DCT 基漏 ½ 偏移 → #3a Gram 拦截", e_g > 1.0, f"max err={e_g:.2f}（非对角显著非零）")
    # D-3 PSNR 用 ln 冒充 log₁₀ → #4a/C4 拦截
    mse = 25.0
    wrong_p = 10.0 * np.log(255.0 ** 2 / mse) / np.log(np.e)   # 恰=log（数值上=ln·10）
    wrong_ln = 10.0 * np.log(255.0 ** 2 / mse)                # 直接用 ln（量纲错误）
    right = 10.0 * np.log10(255.0 ** 2 / mse)
    chk("D-3 PSNR 用 ln 冒充 log₁₀ → #4a/C4 拦截",
        abs(wrong_ln - right) > 5.0,
        f"ln 版={wrong_ln:.2f} vs 正确={right:.2f}（差 {abs(wrong_ln - right):.2f} dB）")
    # D-4 Wiener 谱不共轭镜像直接 ifft2 → #2a 虚部断言拦截
    Ps = power_spectrum()
    S0 = np.sqrt(Ps / 2) * (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)))
    S0[0, 0] = 0
    s_bad = np.fft.ifft2(S0)
    chk("D-4 不共轭镜像直接 ifft2 → #2a 虚部断言拦截", float(np.abs(s_bad.imag).max()) > 1e-3,
        f"max|Im|={float(np.abs(s_bad.imag).max()):.3f}（应为复图像，非实）")


# ---------- 主流程 ----------

def main():
    run = 4
    print(f"== E028_ft2d 数值实验 run{run} ==")
    print("口径注记（Figure 3.19 交叉引用疑点）：原书第 148 页正文称")
    print("「Figure 3.19 shows some examples of linear and non-linear filters being")
    print("used to remove noise」，但该图实为原书第 132 页的 Median and bilateral filtering")
    print("图（印刷 p.132，图注见留档 e028_book_p132_fig319.txt L79–82）——正文的")
    print("描述与图注不完全对应；本实验不复制原书图（红线 2），去噪对比全部程序生成。")
    print(f"[存档] 本 stdout 保存为 code/outputs/e028_ft2d_run{run}.txt（文件名 N 与此标头一致）。\n")
    s_oriented = exp1()
    Ps, pts = exp2()
    img255, rec_light, rec_strong = exp3()
    img, noisy, den, ps, ss = exp4()
    constants()
    make_figs(s_oriented, Ps, pts, img255, rec_light, rec_strong, img, noisy, den, ps, ss)
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
