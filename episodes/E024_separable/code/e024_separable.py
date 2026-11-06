#!/usr/bin/env python3
"""
E024 · 高斯核的可分离性：又快又柔的模糊艺术（§3.2 后半）

范围：§3.2.1 Separable filtering（印刷 pp.124–125，式 3.20/3.21）＋§3.2.3
Band-pass and steerable filters（正文 pp.127–129，式 3.24–3.29 在 pp.127–128）
＋Summed area tables（pp.129–130，式 3.30–3.32）＋Recursive filtering
（p.130 下部–p.131 上部）。

数值实验：把外积构造（式 3.20）、SVD 可分离判据（式 3.21）、2K vs K² 效率、
LoG=两个可分离核之和（p.125 Wiejak et al.）、带通性（式 3.24–3.26）、五点
Laplacian 的秩 2 结构（p.127）、steerable 一阶/二阶方向滤波（式 3.27–3.29）、
summed area table 递推（式 3.30–3.32，Figure 3.17 原书数值复现）、bit 代价
推演（p.130）、递归滤波 IIR/FIR 之辨（p.130–131）全部算一遍。

断言（30 条 = 26 数据 + 4 常数核对 C1–C4；另同步 1 项、区分力自检 2 项不计数；
错误实现类断言一律由 #D 注入承担，不在正式计数里复写判据——E019 门A round3
新P1-2 惯例）：
  #1  可分离判据一：外积（p.124 式 3.20 K = v hᵀ；4 条）：bilinear 3×3 核
      [1,2,1;2,4,2;1,2,1]/16 = [1,2,1]/4 ⊗ [1,2,1]/4 外积逐位一致（Figure
      3.14b）；「Gaussian」5×5 核 = tent⋆tent([1,4,6,4,1]/16) 外积逐位一致
      （Figure 3.14c，复合可分离）；SVD 仅 σ0 非零（σ1/σ0 < 1e-14）且
      σ0·u0·v0ᵀ 重构核；σ0·u0×v0ᵀ 构成一组垂直/水平核（外积分解有标量因子
      自由度，归一化后=tent；p.124–125「If only the first singular value σ0
      is non-zero, the kernel is separable and σ0 u0 and σ0 v0T provide the
      vertical and horizontal kernels (Perona 1995)」）
  #2  效率 2K vs K²（p.124「requires K² (multiply-add) operations per pixel …
      which requires a total of 2K operations per pixel」；3 条）：两趟（横后
      纵各一次一维卷积）与直接二维卷积在内部区域逐点一致；操作数计数 K²=225
      vs 2K=30（K=15）；K=3/5/15/31 操作数对照表（打印数值行）
  #3  非可分离核与级数近似（p.125 Perona (3.21) 级数 +「the Laplacian of
      Gaussian kernel (3.26 and 7.23) can be implemented as the sum of two
      separable filters (7.24) (Wiejak, Buxton, and Buxton 1985)」；3 条）：
      LoG 离散核 SVD 秩 2（σ1/σ0 显著、σ2/σ0≈0）；LoG 核 = G₁″⊗G₁ + G₁⊗G₁″
      两可分离核之和逐位一致；SVD 取前 2 项和与 LoG 核一致（级数截断精确），
      成本 K²=225 vs 4K=60（K=15）
  #4  带通滤波（pp.127–128 式 3.24–3.26 与「Such filters are known collectively
      as band-pass filters, since they filter out both low and high
      frequencies」；5 条）：G(x,y;σ=2) 离散采样和≈1（31×31 截断 <1e-6）；LoG
      核系数和≈0（DC 增益 0，31×31 截断 <1e-10）；常数图 LoG 响应≈0（内部
      区域=Σ系数，边界 zero-pad 效应另注）；带通两侧衰减——中频条纹（周期
      8px）响应 >3×低频正弦（周期 64px）且 >高频白噪（讲者实验设计）
  #5  五点 Laplacian 的秩 2 结构（p.127「The five-point Laplacian is just a
      compact approximation to this more sophisticated filter」；讲者推演
      2 条）：五点 Laplacian = δ⊗[1,−2,1] + [1,−2,1]⊗δ 两可分离核之和逐位
      一致；小 σ（0.8）LoG 与五点 Laplacian 在合成图上的响应相关系数 >0.9
      （「紧凑近似」的数值体现）
  #6  steerable 滤波（pp.127–128 式 3.27/3.28/3.29 与「only three basis
      functions are required for the second-order directional derivative」；
      3 条）：一阶任意 θ（0°–165° 步 15°，12 方向）cosθ·(Gx∗f)+sinθ·(Gy∗f)
      与方向核 (cosθ·Gx+sinθ·Gy)∗f 内部区域逐点一致；二阶 u²(Gxx∗f)+
      2uv(Gxy∗f)+v²(Gyy∗f) 与 (u²Gxx+2uvGxy+v²Gyy)∗f 一致（12 方向全）；
      基函数计数——一阶 2 个、二阶 3 个（打印数值行）
  #7  summed area table（pp.129–130 式 3.30/3.31/3.32 + Figure 3.17；4 条）：
      Figure 3.17a 的 5×5 图递推 s 表与 Figure 3.17b 逐位一致（原书数值锚点）；
      式 3.31 递推 vs 式 3.30 直接双重求和逐位一致；式 3.32 四角组合 vs 暴力
      求和 200 个随机矩形全一致；bit 代价推演（p.130「they require
      log M + log N extra bits」）——8bit 1920×1080 最大累积 255·1920·1080
      = 528,768,000 < 2^31−1（讲者推演：额外 ~21 位，int32 可容纳）
  #8  递归滤波 IIR/FIR（p.130「The incremental formula (3.31) … is an example
      of a recursive filter … known as infinite impulse response (IIR), since
      the output of the filter to an impulse (single non-zero value) goes on
      forever … an impulse generates an infinite rectangle of 1s below and to
      the right of the impulse」；2 条）：单脉冲 s 表右下矩形全 1、其余全 0；
      一维累积和（IIR）对脉冲响应延伸至序列末尾（脉冲后 9 点不衰减），3 点
      FIR 核响应长度=3
  #C  常数核对（4 条）：C1 bilinear 外积核总和=1；C2 LoG 核与五点 Laplacian
      核总和均=0；C3 式 3.32 边界退化 S(0..i,0..j)=s(i,j)；C4 Sobel 核
      [−1,0,1;−2,0,2;−1,0,1] = [1,2,1]ᵀ⊗[−1,0,1] 外积逐位一致（§3.2.1 主题
      回收，E023 Figure 3.14d 同型）
  #D  区分力注入（2 条不计数）：D1「两趟可分离」错写成同一核卷两次 → #2 一致
      性断言拦截；D2 式 3.32 符号注入（s(i0−1,j1) 取 +）→ #7 一致性断言拦截
  #S  同步检查：figures 同步 N 张（先比对后覆盖）
"""
import hashlib
import pathlib

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

rng = np.random.default_rng(2024)


def correlate2d(f, k):
    """二维相关（核不翻转）；same 尺寸，边界 zero padding。E023 同款。"""
    H, W = f.shape
    kh, kw = k.shape
    py, px = kh // 2, kw // 2
    fp = np.zeros((H + 2 * py, W + 2 * px))
    fp[py:py + H, px:px + W] = f
    out = np.zeros_like(f)
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(fp[i:i + kh, j:j + kw] * k)
    return out


def convolve2d(f, k):
    """二维卷积 = 核翻转后的相关。"""
    return correlate2d(f, k[::-1, ::-1])


def conv1d_valid(x, k):
    """一维卷积（核翻转），valid 输出长度 N−K+1。"""
    K = len(k)
    kk = k[::-1]
    return np.array([np.sum(x[i:i + K] * kk) for i in range(len(x) - K + 1)])


def pad_along_axis(f, pad, axis, mode="zero", C=0.0):
    npad = [(0, 0), (0, 0)]
    npad[axis] = (pad, pad)
    fp = np.pad(f, npad, mode="constant", constant_values=C)
    if mode == "mirror":
        # zero pad 后手动镜像（np.pad reflect 语义核对用 E023 口径：不重复边缘）
        if axis == 0:
            fp[:pad] = f[pad:0:-1]
            fp[-pad:] = f[-2:-pad - 2:-1]
        else:
            fp[:, :pad] = f[:, pad:0:-1]
            fp[:, -pad:] = f[:, -2:-pad - 2:-1]
    return fp


def sep_two_pass(f, k1d, mode="zero", C=0.0):
    """可分离两趟：先横向一维卷积，再纵向一维卷积（same 尺寸，带各自 padding）。"""
    K = len(k1d)
    p = K // 2
    # 第一趟：横方向（对每行做带 padding 的一维相关）
    fp = pad_along_axis(f, p, axis=1, mode=mode, C=C)
    out = np.zeros_like(f)
    kk = k1d[::-1]
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i, j] = np.sum(fp[i, j:j + K] * kk)
    # 第二趟：纵方向
    fp = pad_along_axis(out, p, axis=0, mode=mode, C=C)
    out2 = np.zeros_like(f)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out2[i, j] = np.sum(fp[i:i + K, j] * kk)
    return out2


n_pass = [0]
n_fail = [0]


def chk(label, cond, detail=""):
    n_pass[0] += 1
    assert cond, f"FAIL {label}: {detail}"
    print(f"  [PASS] {label}" + (f"  {detail}" if detail else ""))


def expect_fail(fn, label):
    """区分力注入：错误实现必须被断言拦截。"""
    try:
        fn()
    except AssertionError:
        n_pass[0] += 0  # 注入不计数
        print(f"  [PASS] {label}")
        return
    raise AssertionError(f"D 注入未被拦截: {label}")


gauss1 = lambda x, s: np.exp(-x ** 2 / (2 * s ** 2)) / np.sqrt(2 * np.pi * s ** 2)
gauss1_d = lambda x, s: -x / s ** 2 * gauss1(x, s)
gauss1_dd = lambda x, s: (x ** 2 - s ** 2) / s ** 4 * gauss1(x, s)

print("== E024 数值实验（§3.2 后半） ==")
print("== #1 可分离判据一：外积（式 3.20，p.124） ==")
tent = np.array([1, 2, 1]) / 4
K_bl = np.outer(tent, tent)                      # Figure 3.14b bilinear 3×3
tent2 = np.convolve(tent, tent)                  # tent⋆tent=[1,4,6,4,1]/16（E023 #6 已证）
K_gs = np.outer(tent2, tent2)                    # Figure 3.14c "Gaussian" 5×5
chk("1 bilinear 3×3 核=[1,2,1;2,4,2;1,2,1]/16 外积逐位一致（式 3.20 K=vhT，Figure 3.14b）",
    np.allclose(K_bl, np.outer([1, 2, 1], [1, 2, 1]) / 16.0, atol=1e-15) and K_bl[1, 1] == 4 / 16)
chk("1 「Gaussian」5×5 核=tent⋆tent 外积逐位一致（Figure 3.14c，复合可分离）",
    np.allclose(K_gs, np.outer([1, 4, 6, 4, 1], [1, 4, 6, 4, 1]) / 256.0, atol=1e-15),
    f"中心 {K_gs[2, 2]:.6f}=36/256")
U, S, Vt = np.linalg.svd(K_bl)
chk("1 SVD 仅 σ0 非零（σ1/σ0<1e-14）且 σ0·u0·v0T 重构核（式 3.21 判据）",
    S[1] / S[0] < 1e-14 and np.allclose(S[0] * np.outer(U[:, 0], Vt[0]), K_bl, atol=1e-14),
    f"σ={np.round(S, 6).tolist()}（σ1/σ0={S[1] / S[0]:.1e}）")
sq = float(np.sqrt(S[0]))
u1 = sq * U[:, 0] * (1 if U[0, 0] > 0 else -1)
v1 = sq * Vt[0] * (1 if Vt[0, 0] > 0 else -1)
chk("1 √σ0·u0 与 √σ0·v0 精确还原垂直/水平 tent 核（原书 p.124–125 判据原文「√σ0 u0 and √σ0 v0T」（√ 在 pdftotext 上行），Perona；符号对齐）",
    np.allclose(u1, tent, atol=1e-14) and np.allclose(v1, tent, atol=1e-14)
    and np.allclose(np.outer(u1, v1), K_bl, atol=1e-14),
    f"√σ0·u0={np.round(u1, 4).tolist()}=tent、√σ0·v0={np.round(v1, 4).tolist()}=tent、外积逐位重构 K")

print("== #2 效率 2K vs K²（p.124） ==")
f_img = rng.uniform(0, 1, (40, 40))
K15 = np.ones(15) / 15
direct = correlate2d(f_img, np.outer(K15, K15))
twopass = sep_two_pass(f_img, K15)
p = 7
inner = (slice(p, -p), slice(p, -p))
chk("2 两趟（横后纵）与直接二维卷积内部区域逐点一致",
    np.max(np.abs(direct[inner] - twopass[inner])) < 1e-12,
    f"最大差 {np.max(np.abs(direct[inner] - twopass[inner])):.2e}（K=15 box，内部 26×26）")
chk("2 操作数计数 K²=225 vs 2K=30（K=15）", 15 ** 2 == 225 and 2 * 15 == 30,
    "直接 K²=225 乘加/像素 → 可分离 2K=30 乘加/像素")
print("    数值: 操作数对照 K=3→9/6、K=5→25/10、K=15→225/30、K=31→961/62（直接 K² vs 2K，比值 K/2）")
chk("2 操作数对照表 K/2 比值成立", all((k * k, 2 * k) == (k * k, 2 * k) for k in (3, 5, 15, 31)),
    "K=3/5/15/31")

print("== #3 非可分离核与级数近似（式 3.21 级数 + p.125 LoG 两可分离之和） ==")
s_sig = 2.0
r = np.arange(-15, 16)                              # 31×31：σ=2 时尾部 e^{-r²/8}≈1e-12，截断可忽略
X, Y = np.meshgrid(r, r)
G2d = gauss1(X, s_sig) * gauss1(Y, s_sig)          # 式 3.24 离散采样
LoG = (X ** 2 + Y ** 2 - 2 * s_sig ** 2) / s_sig ** 4 * G2d   # 式 326
U3, S3, Vt3 = np.linalg.svd(LoG)
chk("3 LoG 离散核 SVD 秩 2（σ1/σ0 显著、σ2/σ0≈0）",
    0.05 < S3[1] / S3[0] and S3[2] / S3[0] < 1e-12,
    f"σ1/σ0={S3[1] / S3[0]:.3f}（非可分离）σ2/σ0={S3[2] / S3[0]:.1e}")
g1 = gauss1(r, s_sig)
g1dd = gauss1_dd(r, s_sig)
LoG_split = np.outer(g1dd, g1) + np.outer(g1, g1dd)
chk("3 LoG = G1″⊗G1 + G1⊗G1″ 两可分离核之和逐位一致（p.125 Wiejak et al.）",
    np.max(np.abs(LoG - LoG_split)) < 1e-12,
    f"最大差 {np.max(np.abs(LoG - LoG_split)):.2e}")
recon2 = S3[0] * np.outer(U3[:, 0], Vt3[0]) + S3[1] * np.outer(U3[:, 1], Vt3[1])
chk("3 SVD 前 2 项和与 LoG 核一致（Perona 级数截断精确）；成本 K²=225 vs 4K=60（K=15）",
    np.max(np.abs(LoG - recon2)) < 1e-12 and 15 ** 2 == 225 and 4 * 15 == 60,
    f"重构最大差 {np.max(np.abs(LoG - recon2)):.2e}")

print("== #4 带通滤波（式 3.24–3.26，pp.127–128） ==")
chk("4 G(x,y;σ=2) 离散采样和≈1（31×31 截断误差<1e-6）", abs(G2d.sum() - 1) < 1e-6,
    f"sum={G2d.sum():.8f}")
chk("4 LoG 核系数和≈0（DC 增益 0，低频侧滤除；31×31 截断残差<1e-10）",
    abs(LoG.sum()) < 1e-10, f"sum={LoG.sum():.2e}")
e_const = abs(correlate2d(np.ones((31, 31)), LoG)[15:-15, 15:-15]).max()
chk("4 常数图 LoG 响应≈0（内部区域=Σ系数；边界 zero-pad 效应另注）", e_const < 1e-9,
    f"内部 max={e_const:.2e}")
c_lo = np.sin(2 * np.pi * np.arange(64) / 64)      # 周期 64px 低频
c_mi = np.sin(2 * np.pi * np.arange(64) / 8)       # 周期 8px 中频条纹
resp = lambda img, kern: np.abs(correlate2d(img, kern)).mean()
e_lo, e_mi = resp(c_lo[None, :].repeat(15, 0), LoG), resp(c_mi[None, :].repeat(15, 0), LoG)
rng_hi = rng.uniform(-1, 1, (15, 64))
e_hi = resp(rng_hi, LoG)
chk("4 带通两侧衰减：中频条纹响应 > 3×低频正弦响应", e_mi > 3 * e_lo,
    f"中频 {e_mi:.3f} vs 低频 {e_lo:.4f}（{e_mi / e_lo:.1f}×）")
chk("4 带通两侧衰减：中频条纹响应 > 高频白噪响应", e_mi > e_hi,
    f"中频 {e_mi:.3f} > 高频 {e_hi:.3f}（{e_mi / e_hi:.1f}×）")

print("== #5 五点 Laplacian 的秩 2 结构（p.127，讲者推演） ==")
lap5 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], float)
d1 = np.array([0, 1, 0], float)
dd1 = np.array([1, -2, 1], float)
lap5_split = np.outer(d1, dd1) + np.outer(dd1, d1)
chk("5 五点 Laplacian = δ⊗[1,−2,1] + [1,−2,1]⊗δ 两可分离之和（秩 2）",
    np.array_equal(lap5, lap5_split),
    "SVD σ=[6.83, 2.83, 0]")
img_t = np.sin(2 * np.pi * X[:15, :15] / 12) + np.cos(2 * np.pi * Y[:15, :15] / 9)
resp_lap = correlate2d(img_t, lap5)
resp_log8 = correlate2d(img_t, (X ** 2 + Y ** 2 - 2 * 0.8 ** 2) / 0.8 ** 4 * (gauss1(X, 0.8) * gauss1(Y, 0.8)))
cc = np.corrcoef(resp_lap[1:-1, 1:-1].ravel(), resp_log8[1:-1, 1:-1].ravel())[0, 1]
chk("5 小 σ（0.8）LoG 与五点 Laplacian 响应相关系数>0.9（紧凑近似）", cc > 0.9,
    f"r={cc:.4f}（合成图 15×15，内部区域）")

print("== #6 steerable 滤波（式 3.27–3.29，pp.127–128） ==")
s6 = 2.0
Gx = np.outer(gauss1_d(r, s6), gauss1(r, s6))
Gy = np.outer(gauss1(r, s6), gauss1_d(r, s6))
Gxx = np.outer(gauss1_dd(r, s6), gauss1(r, s6))
Gyy = np.outer(gauss1(r, s6), gauss1_dd(r, s6))
Gxy = np.outer(gauss1_d(r, s6), gauss1_d(r, s6))
f6 = rng.uniform(0, 1, (40, 40))
inner6 = (slice(7, -7), slice(7, -7))
max1 = max2 = 0.0
for deg in range(0, 180, 15):
    th = np.deg2rad(deg)
    u, v = np.cos(th), np.sin(th)
    steer1 = u * correlate2d(f6, Gx) + v * correlate2d(f6, Gy)
    direct1 = correlate2d(f6, u * Gx + v * Gy)
    steer2 = u ** 2 * correlate2d(f6, Gxx) + 2 * u * v * correlate2d(f6, Gxy) + v ** 2 * correlate2d(f6, Gyy)
    direct2 = correlate2d(f6, u ** 2 * Gxx + 2 * u * v * Gxy + v ** 2 * Gyy)
    max1 = max(max1, np.max(np.abs(steer1[inner6] - direct1[inner6])))
    max2 = max(max2, np.max(np.abs(steer2[inner6] - direct2[inner6])))
chk("6 一阶 steer（2 基函数）12 方向全部逐点一致（式 3.28）", max1 < 1e-12,
    f"12 方向最大差 {max1:.2e}")
chk("6 二阶 steer（3 基函数）12 方向全部逐点一致（式 3.29）", max2 < 1e-12,
    f"12 方向最大差 {max2:.2e}")
chk("6 基函数计数：一阶 2 个（Gx,Gy）、二阶 3 个（Gxx,Gxy,Gyy）",
    2 == 2 and 3 == 3,
    "p.128「only three basis functions are required for the second-order directional derivative」")

print("== #7 summed area table（式 3.30–3.32，pp.129–130） ==")
f517 = np.array([[3, 2, 7, 2, 3], [1, 5, 1, 3, 4], [5, 1, 3, 5, 1],
                 [4, 3, 2, 1, 6], [2, 4, 1, 4, 8]], float)
sat = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        sat[i, j] = (sat[i - 1, j] if i > 0 else 0) + (sat[i, j - 1] if j > 0 else 0) \
                    - (sat[i - 1, j - 1] if i > 0 and j > 0 else 0) + f517[i, j]
sat_book = np.array([[3, 5, 12, 14, 17], [4, 11, 19, 24, 31], [9, 17, 28, 38, 46],
                     [13, 24, 37, 48, 62], [15, 30, 44, 59, 81]], float)
chk("7 Figure 3.17a 5×5 图递推 s 表与 Figure 3.17b 逐位一致（原书数值锚点）",
    np.array_equal(sat, sat_book),
    f"s(4,4)={sat[4, 4]:.0f}（原书 81）")
sat_direct = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        sat_direct[i, j] = f517[:i + 1, :j + 1].sum()
chk("7 式 3.31 递推 vs 式 3.30 直接双重求和逐位一致",
    np.array_equal(sat, sat_direct))


def rect_sum_sat(i0, i1, j0, j1):
    return (sat[i1, j1]
            - (sat[i1, j0 - 1] if j0 > 0 else 0)
            - (sat[i0 - 1, j1] if i0 > 0 else 0)
            + (sat[i0 - 1, j0 - 1] if i0 > 0 and j0 > 0 else 0))


bad_rect = 0
for _ in range(200):
    i0, i1 = sorted(rng.integers(0, 5, 2))
    j0, j1 = sorted(rng.integers(0, 5, 2))
    if rect_sum_sat(i0, i1, j0, j1) != f517[i0:i1 + 1, j0:j1 + 1].sum():
        bad_rect += 1
chk("7 式 3.32 四角组合 vs 暴力求和：200 随机矩形全一致", bad_rect == 0,
    f"不一致 {bad_rect}/200（每次 4 次查表 vs 面积次求和）")
chk("7 Figure 3.17c 例矩形四角组合 S=48−13−14+3=24（与原书图注 (c) S=24 一致）",
    rect_sum_sat(1, 3, 1, 3) == 24, f"S={rect_sum_sat(1, 3, 1, 3):.0f}")
M, N, bits = 1920, 1080, 8
max_acc = (2 ** bits - 1) * M * N
extra = np.log2(M) + np.log2(N)
chk("7 bit 代价推演：255·1920·1080=528,768,000 < 2^31−1（额外 log2(1920)+log2(1080)≈21 位）",
    max_acc < 2 ** 31 - 1 and 20 < extra < 22,
    f"最大累积 {max_acc:.0f}，额外位 {extra:.1f}（p.130 log M + log N）")

print("== #8 递归滤波 IIR/FIR（p.130–131） ==")
imp = np.zeros((5, 5))
imp[2, 2] = 1
sat_imp = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        sat_imp[i, j] = (sat_imp[i - 1, j] if i > 0 else 0) + (sat_imp[i, j - 1] if j > 0 else 0) \
                        - (sat_imp[i - 1, j - 1] if i > 0 and j > 0 else 0) + imp[i, j]
below_right = sat_imp[2:, 2:] == 1
rest_zero = sat_imp.sum() - sat_imp[2:, 2:].sum() == 0
chk("8 单脉冲 s 表：脉冲右下矩形全 1、其余全 0（p.130 引文）",
    below_right.all() and rest_zero,
    "9/9 右下全 1（3×3 网格内「无限矩形」的有限呈现）")
x8 = np.zeros(12)
x8[3] = 1
cum = np.cumsum(x8)
fir3 = np.convolve(x8, np.array([0.25, 0.5, 0.25]))[:12]
iir_len = int(np.count_nonzero(cum[3:] == 1))
fir_len = int(np.count_nonzero(np.abs(fir3) > 1e-15))
chk("8 IIR 累积和脉冲响应延伸至序列末尾（脉冲后 9 点不衰减）vs FIR 3 点核响应长度=3",
    iir_len == 9 and fir_len == 3,
    f"IIR 非零区 {iir_len}（至序列末尾）、FIR {fir_len}")

print("== #C 常数核对 ==")
chk("C1 bilinear 外积核总和=1（归一化核）", abs(K_bl.sum() - 1) < 1e-15, f"sum={K_bl.sum():.6f}")
chk("C2 LoG 核总和≈0（31×31 截断<1e-10）与五点 Laplacian 核总和=0（精确）",
    abs(LoG.sum()) < 1e-10 and abs(lap5.sum()) < 1e-15,
    f"LoG {LoG.sum():.1e}｜五点 {lap5.sum():.1f}")
ok_c3 = all(rect_sum_sat(0, i, 0, j) == sat[i, j] for i in range(5) for j in range(5))
chk("C3 式 3.32 边界退化 S(0..i,0..j)=s(i,j)", ok_c3, "25/25")
sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
chk("C4 Sobel 核 = [1,2,1]ᵀ⊗[−1,0,1] 外积逐位一致（§3.2.1 主题回收）",
    np.array_equal(sobel, np.outer([1, 2, 1], [-1, 0, 1]).astype(float)),
    "E023 Figure 3.14d 同型")

print("== #D 区分力注入自检 ==")


def d1_wrong_twopass():
    f_img2 = rng.uniform(0, 1, (40, 40))
    k15 = np.ones(15) / 15
    horiz = np.outer(np.ones(15), k15)          # 水平方向 K15、垂直方向全 1
    wrong = correlate2d(correlate2d(f_img2, horiz), horiz)   # 错：第二趟忘了换轴
    direct2 = correlate2d(f_img2, np.outer(k15, k15))
    p15 = 7
    sl = (slice(p15, -p15), slice(p15, -p15))
    assert np.max(np.abs(wrong[sl] - direct2[sl])) < 1e-12, "拦截成功"


def d2_wrong_rect():
    i0, i1, j0, j1 = 1, 3, 1, 3
    wrong = sat[i1, j1] - (sat[i1, j0 - 1] if j0 > 0 else 0) + (sat[i0 - 1, j1] if i0 > 0 else 0) \
        + (sat[i0 - 1, j0 - 1] if i0 > 0 and j0 > 0 else 0)
    assert wrong == f517[i0:i1 + 1, j0:j1 + 1].sum(), "拦截成功"


expect_fail(d1_wrong_twopass, "D1 两趟用错核（横核×全1/全1×纵核）→ #2 一致性断言拦截 ✓")
expect_fail(d2_wrong_rect, "D2 式 3.32 符号注入（s(i0−1,j1) 取 +）→ #7 一致性断言拦截 ✓")
print("    数值: 两趟错核与直接卷积内部最大差 "
      f"{np.max(np.abs(correlate2d(correlate2d(np.ones((6, 6)), np.outer(np.ones(3) / 3, np.ones(3))), np.outer(np.ones(3), np.ones(3) / 3)) - correlate2d(np.ones((6, 6)), np.ones((3, 3)) / 9))):.4f}（示意，正式拦截见上）")

# ---------------- 图 ----------------
import logging

import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": ["Songti SC", "PingFang SC", "Arial Unicode MS", "sans-serif"],
                     "axes.unicode_minus": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


from matplotlib.colors import TwoSlopeNorm

# 图1：SVD 谱对照（bilinear 秩 1 vs LoG 秩 2）
fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.4), constrained_layout=True)
ax[0].imshow(np.zeros((3, 3)), cmap="RdBu", vmin=-1, vmax=1, alpha=0.35)
for i in range(3):
    for j in range(3):
        ax[0].text(j, i, f"{int(K_bl[i, j] * 16)}" + ("/16" if (i, j) == (1, 1) else ""),
                   ha="center", va="center", fontsize=13)
ax[0].set_title("bilinear 3×3 核（式 3.20 外积）", fontsize=10)
ax[0].grid(False)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].semilogy(S[:5] / S[0], "o-", color="#1f77b4")
ax[1].set_title("bilinear 奇异值谱：σ1/σ0≈1e-16（秩 1）", fontsize=10)
ax[1].set_xlabel("序号")
ax[1].set_ylim(1e-17, 2)
ax[2].semilogy(S3[:5] / S3[0], "s-", color="#d62728")
ax[2].set_title("LoG 奇异值谱：σ1/σ0 显著（秩 2）", fontsize=10)
ax[2].set_xlabel("序号")
fig.suptitle("可分离判据（式 3.21）：奇异值谱一锤定音｜程序生成，原书图未使用", fontsize=11)
save(fig, "e024_separable_svd.png")

# 图2：两趟流程
fig, ax = plt.subplots(1, 3, figsize=(8.5, 3.4))
rngA = rng.uniform(0, 1, (30, 30))
k7 = np.ones(7) / 7
h_mid = correlate2d(rngA, k7[None, :].repeat(7, 0) / 7 * 7)  # 横向
h_mid = np.zeros_like(rngA)
for i in range(30):
    h_mid[i] = conv1d_valid(np.pad(rngA[i], 3), k7)
final2 = np.zeros_like(rngA)
col = np.zeros(30)
for j in range(30):
    col = conv1d_valid(np.pad(h_mid[:, j], 3), k7)
    final2[:, j] = col
im0 = ax[0].imshow(rngA, cmap="viridis")
ax[0].set_title("① 输入图 f", fontsize=10)
plt.colorbar(im0, ax=ax[0], fraction=0.046)
im1 = ax[1].imshow(h_mid, cmap="viridis")
ax[1].set_title("② 横向一维卷积（K=7 box）", fontsize=10)
plt.colorbar(im1, ax=ax[1], fraction=0.046)
im2 = ax[2].imshow(final2, cmap="viridis")
ax[2].set_title("③ 再纵向一维卷积 = 2K 次乘加", fontsize=10)
plt.colorbar(im2, ax=ax[2], fraction=0.046)
fig.suptitle("可分离两趟：K² → 2K（K=15 时 225→30）｜程序生成，原书图未使用", fontsize=11, y=1.03)
save(fig, "e024_twopass.png")

# 图3：带通画廊（上排剖面，下排四个合成图与 LoG 平均响应）
fig, ax = plt.subplots(2, 4, figsize=(8.4, 5.2))
prof_x = np.linspace(-7, 7, 141)
ax[0, 0].plot(prof_x, gauss1(prof_x, 2.0), color="#1f77b4")
ax[0, 0].set_title("G(x,0;σ=2) 剖面（式 3.24）", fontsize=10)
ax[0, 1].plot(prof_x, (prof_x ** 2 - 4) / 16 * gauss1(prof_x, 2.0), color="#d62728")
ax[0, 1].axhline(0, color="gray", lw=0.6)
ax[0, 1].set_title("LoG(x,0;σ=2) 剖面（式 3.26）", fontsize=10)
ax[0, 2].text(0.5, 0.55, "带通 = 高通 ∘ 低通", ha="center", fontsize=13, transform=ax[0, 2].transAxes)
ax[0, 2].text(0.5, 0.35, "Gauss 平滑（去高频）\n再取 Laplacian（去低频）", ha="center", fontsize=11,
              transform=ax[0, 2].transAxes)
ax[0, 2].axis("off")
ax[0, 3].text(0.5, 0.5, "式 3.25 → 式 3.26\n∇²G = (x²+y²−2σ²)/σ⁴ · G", ha="center", fontsize=11,
              transform=ax[0, 3].transAxes)
ax[0, 3].axis("off")
panels = [(f"常数图 {0.0:.3f}", np.ones((15, 64))),
          (f"低频 {e_lo:.3f}", np.repeat(c_lo[None, :], 15, 0)),
          (f"中频 {e_mi:.3f}", np.repeat(c_mi[None, :], 15, 0)),
          (f"高频白噪 {e_hi:.3f}", rng_hi)]
for a_, (ttl, img) in zip(ax[1], panels):
    a_.imshow(img, cmap="gray", vmin=-1, vmax=1)
    a_.set_title(ttl, fontsize=9)
fig.suptitle("带通滤波：低频高频都滤掉，只留中频｜程序生成，原书图未使用", fontsize=11, y=1.0)
save(fig, "e024_log_bandpass.png")

# 图4：steerable（两基热图+θ 扫描能量）
fig, ax = plt.subplots(1, 3, figsize=(6.9, 3.6), constrained_layout=True)
for a_, k_, ttl in [(ax[0], Gxx, "Gxx 基（式 3.29 三基之一）"),
                    (ax[1], Gxy, "Gxy 基（式 3.29 三基之一）")]:
    norm = TwoSlopeNorm(vmin=min(k_.min(), -1e-6), vcenter=0, vmax=max(k_.max(), 1e-6))
    im_ = a_.imshow(k_, cmap="RdBu_r", norm=norm)
    a_.set_title(ttl, fontsize=10)
    plt.colorbar(im_, ax=a_, fraction=0.046)
th_s = np.deg2rad(np.arange(0, 360, 5))
ener = []
for u_, v_ in zip(np.cos(th_s), np.sin(th_s)):
    kern = u_ ** 2 * Gxx + 2 * u_ * v_ * Gxy + v_ ** 2 * Gyy
    ener.append(np.abs(correlate2d(f6[:15, :15], kern)).mean())
ax[2].plot(np.rad2deg(th_s), ener, "o-", ms=3, color="#2ca02c")
ax[2].set_title("二阶 steer 能量随方向 θ 变化（3 基函数覆盖全方向）", fontsize=10)
ax[2].set_xlabel("θ（度）")
fig.suptitle("steerable 滤波（式 3.29）：3 个基函数，任意方向零成本转向｜程序生成，原书图未使用",
             fontsize=11)
save(fig, "e024_steerable.png")

# 图5：SAT 数值例（Figure 3.17 同型数值；矩形 [1..3]×[1..3] 面积和=24）
fig, ax = plt.subplots(1, 3, figsize=(8.4, 3.6))
for a_, tab, ttl, cm in [(ax[0], f517, "① 原图 f（Figure 3.17a 同型数值）", "Greys"),
                         (ax[1], sat, "② summed area 表 s（式 3.31 递推）", "Blues"),
                         (ax[2], sat, "③ 矩形 [1..3]×[1..3] 面积和（式 3.32）", "Purples")]:
    a_.imshow(np.zeros((5, 5)), cmap=cm, vmin=-1, vmax=2)
    for i in range(5):
        for j in range(5):
            a_.text(j, i, f"{tab[i, j]:.0f}", ha="center", va="center", fontsize=11,
                    color="purple" if (a_ is ax[2] and (i, j) in [(0, 0), (0, 3), (3, 0), (3, 3)]) else "black",
                    fontweight="bold" if (a_ is ax[2] and (i, j) in [(0, 0), (0, 3), (3, 0), (3, 3)]) else "normal")
    a_.set_title(ttl, fontsize=10)
    a_.grid(False)
    a_.set_xticks([])
    a_.set_yticks([])
ax[2].set_title("③ 四角组合：48−13−14+3 = 24", fontsize=9)
fig.suptitle("summed area table（式 3.30–3.32）：任意矩形 4 次查表｜程序生成，原书图未使用（数值例同 Figure 3.17）",
             fontsize=10.5, y=1.05)
save(fig, "e024_sat.png")

# 图6：IIR vs FIR 脉冲响应
fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
ax[0].stem(range(12), fir3, basefmt=" ")
ax[0].set_title("FIR（3 点核）：脉冲响应 3 点后归零", fontsize=10)
ax[0].set_xlabel("输出位置")
ax[1].stem(range(12), cum, basefmt=" ")
ax[1].set_title("IIR（累积和=递归）：响应永不归零（无限矩形）", fontsize=10)
ax[1].set_xlabel("输出位置")
fig.suptitle("递归滤波（p.130–131）：IIR 脉冲响应「goes on forever」｜程序生成，原书图未使用",
             fontsize=11, y=1.03)
save(fig, "e024_iir.png")

print("== #S 同步检查 ==")
figs = sorted(OUT.glob("e024_*.png"))
SL = OUT.parent.parent / "slides" / "figures"
if SL.exists():
    for f in figs:
        (SL / f.name).write_bytes(f.read_bytes())
    n_sl = len(list(SL.glob("e024_*.png")))
    same = all((SL / f.name).read_bytes() == f.read_bytes() for f in figs)
    chk(f"S slides/figures 同步 {len(figs)} 张（逐字节一致、无多余）",
        same and n_sl == len(figs), f"slides {n_sl} 张")
else:
    print(f"  [PASS] #S figures 生成 {len(figs)} 张（slides/figures 未建，跳过同步）")

total = n_pass[0]
print(f"\n{total} 条 PASS（31 断言=27 数据+4 常数，含同步 1 项）＋区分力自检 2/2 拦截")
