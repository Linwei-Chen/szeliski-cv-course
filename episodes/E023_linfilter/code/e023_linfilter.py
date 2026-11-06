#!/usr/bin/env python3
"""
E023 · 线性滤波与卷积上桌（§3.2 前半：节引言 pp.119–124 + 3.2.2 Examples of
linear filtering pp.125–127 上部；§3.2.1 Separable filtering 与 §3.2.3 Band-pass
and steerable filters 归 E024）

数值实验：把卷积/相关之辨（式 3.14/3.15）、LSI 两原理（式 3.16/3.17）、矩阵
视角（式 3.19 + Figure 3.12）、六种边界 padding（p.123–124 + Figure 3.13）、
box 滤波与滑动窗口（p.125–126）、tent 自卷积="Gaussian" 核（p.126）、锐化
式 3.22 与暗房式 3.23 的线性/非线性之辨、Sobel/corner 初瞥（Figure 3.14d–e）
全部算一遍。

断言（34 条 = 30 数据 + 4 常数核对 C1–C4；另同步 1 项、区分力自检 2 项不计数；
错误实现类断言一律由 #D 注入承担，不在正式计数里复写判据——E019 门A round3
新P1-2 惯例）：
  #1  卷积与相关之辨（p.120 式 3.14/3.15 与「the kernel function, h, convolved
      with an impulse signal, δ(i, j) … reproduces itself, h ∗ δ = h, whereas
      correlation produces the reflected signal. (Try this yourself to verify
      that it is so.)」；3 条）：δ 卷积 h 复现 h 本尊；δ 相关 h 得反射；非对称
      核下卷积与相关输出最大差>0 且=|h−flip(h)| 最大差
  #2  LSI 两原理（p.122 式 3.16 叠加原理、式 3.17 平移不变与「the operator
      'behaves the same everywhere'」；3 条）：零底画布上叠加 max 差 0.00e+00；
      输入平移 (2,2) 输出同平移（逐点 0.00e+00）；随机核卷积=显式固定加权求和
  #3  矩阵视角（p.122 式 3.19 g=Hf 与 Figure 3.12 一维稀疏矩阵；2 条）：Hf 与
      直接卷积逐点一致；H 每行非零元个数=核宽 K
  #4  边界 padding 六模式（p.123–124 列表 +「zero padding darkens the edges,
      clamp (replication) padding propagates border values inward, mirror
      (reflection) padding preserves colors near the borders. Extension
      padding (not shown) keeps the border pixels fixed」；6 条）：均匀图上
      zero 边缘环被外部 0 拉暗、constant C=60 渗入较温和、clamp/wrap/mirror/
      extend 四种无偏（=100 精确）；亮框图上 zero<mirror<clamp 排序（边界值
      向内传播）；normalized zero（RGBA 软 alpha 归一）边缘环精确回到内部值
      且=「只对图内窗口取均值」（角点 160/边中 140 实测）；extend 边缘环=
      行内一维平滑（垂直方向不引入外部污染）、角点精确保持原值
  #5  box 滤波与滑动窗口（p.125–126「The simplest filter to implement is the
      moving average or box filter … equivalent to convolving the image with a
      kernel of all ones and then scaling」「a more efficient implementation is
      to slide a moving window across each scanline … adding the newest pixel
      and subtracting the oldest pixel from the running sum」；3 条）：全 1 核
      卷积再缩放=均值核；1D 滑窗 running sum 与直接卷积逐点一致；二维两趟
      滑窗（横扫+纵扫）与直接二维卷积逐点一致
  #6  tent 自卷积（p.126「Convolving the linear tent function with itself yields
      the cubic approximating spline, which is called the 'Gaussian' kernel」；
      2 条）：[1,2,1]/4 ⋆ [1,2,1]/4 = [1,4,6,4,1]/16（Figure 3.14c 系数）；
      bilinear 3×3 核=两个线性 tent 的外积
  #7  锐化式 3.22（p.126 式 3.22 gsharp = f + γ(f − hblur ∗ f)；4 条）：γ=0
      恒等；对 f 线性（叠加 max 差 0.00e+00）；锐化增强边缘（γ=1.0 梯度能量
      >原图）；γ=0/0.5/1.0 梯度能量单调增
  #8  暗房式 3.23 非线性（p.126 式 3.23 gunsharp = f(1 − γhblur ∗ f) 与
      「This is no longer a linear filter but it still works well」；3 条）：
      同 γ 同图 3.23 与 3.22 输出最大差>0（讲者按字面实现：f 逐点乘
      (1−γ·hblur∗f)，暗房两片负片叠印=透射率相乘）；违反叠加原理（差>0，
      no longer a linear filter）；对 [0,1] 图像 3.22 输出逐点 ≥ 3.23 且严格
      本实验如实报告；乘性 3.23 输出恒非负（透射率非负），线性 3.22 有
      负值振铃——不把原书「still works well」夸大为能量增强）
  #9  Sobel 与 corner 初瞥（p.126–127「a simple 3 × 3 edge extractor called the
      Sobel operator … effectively emphasizes vertical edges」+「The simple
      corner detector … looks for simultaneous horizontal and vertical second
      derivatives … responds not only to the corners of the square, but also
      along diagonal edges」；4 条）：Sobel 竖边响应>横边响应；Sobel=竖直
      tent(1,2,1) 与水平中心差分(−1,0,1) 的外积；corner 核对角点有响应且对
      轴对齐直边响应为 0（外积可分性）；corner 核沿 45° 对角边也有响应
  #C  常数核对（4 项，不调被测函数）：Sobel 核每行和=0（差分性质）；corner
      核总和=0；式 3.22 在 γ=0 化简为恒等；非对称核 flip(h)≠h（式 3.14 与
      3.12 的 f 偏移符号相反）
  #S  同步检查：figures 与 slides/figures 先比对 md5（漂移判 FAIL）再覆盖同步
  #D  区分力注入（2 项，不计数）：D1 用相关冒充卷积→#1 断言拦截；D2 用 clamp
      冒充 mirror→#4 断言拦截
"""
import hashlib
import pathlib

import numpy as np

np.set_printoptions(precision=3, suppress=True, linewidth=100)
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "code" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


# ── 基础实现 ─────────────────────────────────────────────────────────────────
def correlate2d(f, k):
    """相关（式 3.12）：输出 (i,j)=Σ f(i+k, j+l) h(k,l)，'valid' 尺寸。"""
    K = k.shape[0]
    H, W = f.shape
    out = np.zeros((H - K + 1, W - K + 1))
    for u in range(K):
        for v in range(K):
            out += k[u, v] * f[u:u + out.shape[0], v:v + out.shape[1]]
    return out


def convolve2d(f, k):
    """卷积（式 3.14）：核翻转 180° 后做相关。"""
    return correlate2d(f, np.flip(k))


def pad_image(f, mode, C=0.0, p=1):
    """六种边界 padding（p.123–124 列表），p=每侧扩展宽度。"""
    if mode == "zero":
        return np.pad(f, p, constant_values=0.0)
    if mode == "constant":
        return np.pad(f, p, constant_values=C)
    if mode == "clamp":
        return np.pad(f, p, mode="edge")
    if mode == "wrap":
        return np.pad(f, p, mode="wrap")
    if mode == "mirror":
        return np.pad(f, p, mode="reflect")   # 原书 p.123：reflect pixels across the image edge（不复制边缘像素）
    if mode == "extend":
        # extend：以边缘像素为基准减去镜像信号 f̃(−k)=2f(0)−f(k)（逐轴可分离）
        g = f.astype(float)
        for axis in range(g.ndim):
            n = g.shape[axis]
            pads = [(p, p) if a == axis else (0, 0) for a in range(g.ndim)]
            g = np.pad(g, pads, mode="reflect")  # 借 reflect 定形
            lo_edge = np.take(g, [p] * p, axis=axis)                    # f(0)
            lo_mirr = np.take(g, list(range(2 * p, p, -1)), axis=axis)  # f(k), k=p..1
            hi_edge = np.take(g, [n - 1 + p] * p, axis=axis)            # f(n−1)
            hi_mirr = np.take(g, list(range(n - 2 + p, n - 2, -1)), axis=axis)  # f(n−1−k), k=1..p
            Sl = [slice(None)] * g.ndim
            Sl[axis] = slice(0, p)
            g[tuple(Sl)] = 2 * lo_edge - lo_mirr
            Sl[axis] = slice(g.shape[axis] - p, g.shape[axis])
            g[tuple(Sl)] = 2 * hi_edge - hi_mirr
        return g
    raise ValueError(mode)


BOX3 = np.full((3, 3), 1 / 9.0)
SOBEL3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
CORNER3 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float)
TENT = np.array([1, 2, 1], float) / 4.0
MODES = ("zero", "constant", "clamp", "wrap", "mirror", "extend")


def blur(f, mode, C=0.0):
    """3×3 box 模糊 = pad 后 valid 相关。"""
    return correlate2d(pad_image(f, mode, C, 1), BOX3)


def sobel_mag(f):
    gx = correlate2d(pad_image(f, "mirror"), SOBEL3)
    gy = correlate2d(pad_image(f, "mirror"), SOBEL3.T)
    return np.hypot(gx, gy)


def grad_energy(f):
    return float(sobel_mag(f).mean())


def sharp22(f, gamma):
    hb = correlate2d(pad_image(f, "mirror"), BOX3)
    return f + gamma * (f - hb)          # 式 3.22：gsharp = f + γ(f − hblur∗f)


def sharp23(f, gamma):
    hb = correlate2d(pad_image(f, "mirror"), BOX3)
    return f * (1 - gamma * hb)          # 式 3.23：讲者按字面实现（透射率相乘）


n_chk = 0


def chk(cond, label):
    global n_chk
    assert cond, f"[FAIL] {label}"
    n_chk += 1
    print(f"  [PASS] {label}")


print("== E023 数值实验（§3.2 前半） ==")

# ── #1 卷积与相关之辨 ────────────────────────────────────────────────────────
rng = np.random.default_rng(23)
h_asym = np.array([[1, 2, 4], [2, 8, 4], [1, 2, 1]], float)  # 非对称核
h_asym /= h_asym.sum()
delta = np.zeros((9, 9)); delta[4, 4] = 1.0
print("== #1 卷积与相关之辨（式 3.14/3.15，p.120） ==")
chk(np.allclose(convolve2d(delta, h_asym)[2:5, 2:5], h_asym),
    "#1 δ 卷积 h 复现 h 本尊（h∗δ=h）")
chk(np.allclose(correlate2d(delta, h_asym)[2:5, 2:5], np.flip(h_asym)),
    "#1 δ 相关 h 得反射（correlation produces the reflected signal）")
d_cc = float(np.abs(convolve2d(delta, h_asym) - correlate2d(delta, h_asym)).max())
chk(d_cc > 0, "#1 非对称核下卷积≠相关（输出差>0）")
print(f"    数值: 非对称核 δ 响应 卷积 vs 相关 最大差 {d_cc:.4f}"
      f"（=|h−flip(h)| 最大差 {np.abs(h_asym - np.flip(h_asym)).max():.4f}）")

# ── #2 LSI 两原理 ────────────────────────────────────────────────────────────
print("== #2 LSI 两原理（式 3.16/3.17，p.122） ==")
k_r = rng.normal(size=(3, 3)); k_r /= np.abs(k_r).sum()
fA = rng.random((16, 16)); fB = rng.random((16, 16))
FA = np.zeros((40, 40)); FA[8:24, 8:24] = fA
FB = np.zeros((40, 40)); FB[8:24, 8:24] = fB
GA, GB, GS = correlate2d(FA, k_r), correlate2d(FB, k_r), correlate2d(FA + FB, k_r)
d_sup = float(np.abs(GS - GA - GB).max())
chk(d_sup < 1e-12, "#2 叠加原理（式 3.16）：h∘(fA+fB)=h∘fA+h∘fB")
F2 = np.zeros((40, 40)); F2[10:26, 10:26] = fA   # 输入整体平移 (2,2)
G2 = correlate2d(F2, k_r)
d_sh = float(np.abs(G2[10:24, 10:24] - GA[8:22, 8:22]).max())
chk(d_sh < 1e-12, "#2 平移不变（式 3.17）：输入平移 (2,2)，输出同样平移")
print(f"    数值: 叠加最大差 {d_sup:.2e}；平移最大差 {d_sh:.2e}（behaves the same everywhere）")
f_t = rng.random((12, 12))
explicit = np.zeros_like(correlate2d(f_t, k_r))
for i in range(explicit.shape[0]):
    for j in range(explicit.shape[1]):
        s = 0.0
        for a in range(3):
            for b in range(3):
                s += f_t[i + a, j + b] * k_r[a, b]
        explicit[i, j] = s
chk(np.abs(explicit - correlate2d(f_t, k_r)).max() < 1e-12,
    "#2 线性滤波=小邻域固定加权组合（显式双循环=相关实现）")

# ── #3 矩阵视角（式 3.19 + Figure 3.12） ────────────────────────────────────
print("== #3 矩阵视角（式 3.19 g=Hf，Figure 3.12，p.122） ==")
h1 = np.array([1, 2, 1], float) / 4.0
f1d = rng.random(10)
H = np.zeros((8, 10))
for row in range(8):
    H[row, row:row + 3] = h1
g_H = H @ f1d
g_d = np.convolve(f1d, h1, mode="valid")
chk(np.abs(g_H - g_d).max() < 1e-15, "#3 一维滤波 g=Hf 与直接卷积逐点一致（Figure 3.12 同型）")
chk(all(int((H[row] != 0).sum()) == 3 for row in range(8)),
    "#3 H 每行非零元个数=核宽 K（Figure 3.12 的稀疏带状结构）")
print(f"    数值: 8×10 稀疏 H 每行非零元 3 个（=核宽 K），Hf 与卷积最大差 {np.abs(g_H - g_d).max():.2e}")

# ── #4 边界 padding 六模式 ───────────────────────────────────────────────────
print("== #4 边界 padding（p.123–124 列表 + Figure 3.13 断言） ==")
U = np.full((8, 8), 100.0)                      # 均匀图：分离出纯 padding 效应
B = np.full((8, 8), 100.0)                      # 亮框图：边界值 180、内部 100
B[0, :] = B[-1, :] = 180.0; B[:, 0] = B[:, -1] = 180.0
ringU = {m: blur(U, m, 60.0 if m == "constant" else 0.0)[0, :].mean() for m in MODES}
ringB = {m: blur(B, m, 60.0 if m == "constant" else 0.0)[0, :].mean() for m in MODES}
chk(ringU["zero"] < 100.0, "#4 zero padding 把边缘拉暗（darkens the edges）")
chk(ringU["constant"] > ringU["zero"], "#4 constant C=60 渗入比 zero 温和（边缘环更接近内部）")
chk(all(abs(ringU[m] - 100.0) < 1e-9 for m in ("clamp", "wrap", "mirror", "extend")),
    "#4 均匀图上 clamp/wrap/mirror/extend 边缘环无偏（=100 精确）")
chk(ringB["zero"] < ringB["mirror"] < ringB["clamp"],
    "#4 亮框图排序：zero 拉暗 < mirror 保色 < clamp 边界值向内传播")
print("    数值: 均匀图边缘环 zero {:.2f}、constant60 {:.2f}；亮框图边缘环 ".format(ringU["zero"], ringU["constant"]) +
      "、".join(f"{m} {ringB[m]:.1f}" for m in MODES))
# normalized zero：RGBA 软 alpha 归一 = 只对图内窗口取均值
a_p = pad_image(np.ones_like(U), "zero")
nzU = correlate2d(pad_image(U, "zero"), BOX3) / correlate2d(a_p, BOX3)
nzB = correlate2d(pad_image(B, "zero"), BOX3) / correlate2d(pad_image(np.ones_like(B), "zero"), BOX3)
chk(np.abs(nzU[0, :] - 100.0).max() < 1e-9 and
    abs(nzB[0, 0] - 160.0) < 1e-9 and np.abs(nzB[0, 2:6] - 140.0).max() < 1e-9,
    "#4 normalized zero（除以软 alpha）边缘环=只对图内窗口取均值（角点 160/边中 140）")
print(f"    数值: normalized zero 均匀图边缘环 {nzU[0, :].mean():.1f}；亮框图角点 {nzB[0, 0]:.1f}、边中 {nzB[0, 2:6].mean():.1f}")
# extend：垂直方向不引入外部污染，边界环=行内一维平滑，角点精确保持
R = rng.random((10, 10))
ext_r = blur(R, "extend")
box_row0 = np.convolve(R[0, :], np.full(3, 1 / 3.0), mode="same")
d_ext = float(np.abs(ext_r[0, 1:-1] - box_row0[1:-1]).max())
c_ext = float(abs(ext_r[0, 0] - R[0, 0]))
chk(d_ext < 1e-12 and c_ext < 1e-12,
    "#4 extend 边缘环=行内一维平滑（无外部污染）、角点精确保持原值（keeps the border pixels fixed）")
print(f"    数值: extend 边缘环 vs 行内平滑最大差 {d_ext:.2e}；角点 (0,0) 保持差 {c_ext:.2e}")

# ── #5 box 与滑动窗口 ────────────────────────────────────────────────────────
print("== #5 box 滤波与滑动窗口（p.125–126） ==")
f_img = rng.random((20, 20))
sums = correlate2d(f_img, np.full((5, 5), 1.0))
chk(np.abs(sums / 25.0 - correlate2d(f_img, np.full((5, 5), 1 / 25.0))).max() < 1e-12,
    "#5 全 1 核卷积再缩放=均值核（kernel of all ones and then scaling）")
K = 15
f1 = rng.random(40)
direct1 = np.convolve(f1, np.ones(K) / K, mode="valid")
rs = np.zeros(40 - K + 1)
s = f1[:K].sum()
for j in range(len(rs)):
    rs[j] = s / K
    if j + K < 40:
        s += f1[j + K] - f1[j]      # 加最新、减最旧
d_rs = float(np.abs(rs - direct1).max())
chk(d_rs < 1e-12, "#5 滑窗 running sum 与直接卷积逐点一致（1D，K=15）")


def box2d_running(f, K):
    """二维 box=横向 running sum 一趟 + 纵向 running sum 一趟。"""
    nrow, ncol = f.shape
    rowsum = np.zeros((nrow, ncol - K + 1))
    for i in range(nrow):
        s = f[i, :K].sum()
        for j in range(rowsum.shape[1]):
            rowsum[i, j] = s
            if j + K < ncol:
                s += f[i, j + K] - f[i, j]
    out = np.zeros((nrow - K + 1, rowsum.shape[1]))
    for j in range(rowsum.shape[1]):
        s = rowsum[:K, j].sum()
        for i in range(out.shape[0]):
            out[i, j] = s
            if i + K < nrow:
                s += rowsum[i + K, j] - rowsum[i, j]
    return out / (K * K)


direct2 = correlate2d(f_img, np.full((5, 5), 1 / 25.0))
d_2d = float(np.abs(box2d_running(f_img, 5) - direct2).max())
chk(d_2d < 1e-12, "#5 二维两趟滑窗（横扫+纵扫）与直接二维卷积逐点一致")
print(f"    数值: 1D running sum 最大差 {d_rs:.2e}；2D 两趟最大差 {d_2d:.2e}；"
      f"K=15 时每像素 直接 K²={K * K} 次乘加 → 每趟滑窗 2 次加减（与 K 无关）")

# ── #6 tent 自卷积 ───────────────────────────────────────────────────────────
print("== #6 tent 自卷积='Gaussian' 核（p.126） ==")
tent_star = np.convolve(TENT, TENT)
target = np.array([1, 4, 6, 4, 1], float) / 16.0
chk(np.abs(tent_star - target).max() < 1e-15,
    "#6 tent⋆tent=[1,4,6,4,1]/16（Figure 3.14c 系数）")
print(f"    数值: [1,2,1]/4⋆[1,2,1]/4 与 [1,4,6,4,1]/16 最大差 {np.abs(tent_star - target).max():.2e}")
K_bi = np.outer(TENT, TENT)
bilinear = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], float) / 16.0
chk(np.abs(K_bi - bilinear).max() < 1e-15,
    "#6 bilinear 3×3 核=两个线性 tent 的外积（p.126 bilinear）")

# ── #7 锐化式 3.22 ───────────────────────────────────────────────────────────
print("== #7 锐化式 3.22（p.126） ==")
img_f = rng.random((24, 24))
chk(np.abs(sharp22(img_f, 0.0) - img_f).max() < 1e-15, "#7 γ=0 → 恒等")
fA2, fB2 = rng.random((12, 12)), rng.random((12, 12))
d_lin = float(np.abs(sharp22(fA2 + fB2, 0.7) - sharp22(fA2, 0.7) - sharp22(fB2, 0.7)).max())
chk(d_lin < 1e-12, "#7 式 3.22 对 f 线性（叠加成立）")
e0 = grad_energy(img_f)
e1 = grad_energy(sharp22(img_f, 1.0))
e05 = grad_energy(sharp22(img_f, 0.5))
chk(e1 > e0, "#7 锐化增强边缘（γ=1.0 梯度能量>原图）")
chk(e0 < e05 < e1, "#7 γ=0/0.5/1.0 梯度能量单调增")
print(f"    数值: 梯度能量 γ=0 {e0:.4f} → 0.5 {e05:.4f} → 1.0 {e1:.4f}（提升 {e1 / e0:.2f}×）；线性叠加最大差 {d_lin:.2e}")

# ── #8 暗房式 3.23 非线性 ────────────────────────────────────────────────────
print("== #8 暗房式 3.23 非线性（p.126） ==")
d_2322 = float(np.abs(sharp23(img_f, 1.0) - sharp22(img_f, 1.0)).max())
chk(d_2322 > 1e-6, "#8 同 γ 同图 3.23 与 3.22 输出不同（最大差>0）")
d_nl = float(np.abs(sharp23(fA2 + fB2, 0.7) - sharp23(fA2, 0.7) - sharp23(fB2, 0.7)).max())
chk(d_nl > 1e-6, "#8 式 3.23 违反叠加（no longer a linear filter）")
g22_min = float(sharp22(img_f, 1.0).min())
g23_min = float(sharp23(img_f, 1.0).min())
chk(g23_min >= 0.0 and g22_min < 0.0,
    "#8 乘性 3.23 无负值振铃（输出≥0），线性 3.22 有负值振铃（透射率非负：暗房比喻自洽）")
print(f"    数值: 3.23 vs 3.22 最大差 {d_2322:.4f}；叠加违反量 {d_nl:.4f}；"
      f"输出最小值 3.22 {g22_min:.4f} < 0 ≤ 3.23 {g23_min:.4f}（γ=1）")

# ── #9 Sobel 与 corner 初瞥 ──────────────────────────────────────────────────
print("== #9 Sobel 与 corner 初瞥（Figure 3.14d–e，p.126–127） ==")
edge_v = np.zeros((12, 12)); edge_v[:, 6:] = 1.0    # 竖直边缘（左暗右亮）
edge_h = np.zeros((12, 12)); edge_h[6:, :] = 1.0    # 水平边缘
resp_v = correlate2d(pad_image(edge_v, "mirror"), SOBEL3)
resp_h = correlate2d(pad_image(edge_h, "mirror"), SOBEL3)
sv = float(np.abs(resp_v[5:7, 4:8]).max())
sh_ = float(np.abs(resp_h[4:8, 5:7]).max())
chk(sv > sh_, "#9 Sobel 强调竖直边缘（竖边响应>横边响应）")
decomp = np.outer(np.array([1.0, 2.0, 1.0]), np.array([-1.0, 0.0, 1.0]))
chk(np.abs(decomp - SOBEL3).max() < 1e-15,
    "#9 Sobel=竖直 tent(1,2,1) 与水平中心差分(−1,0,1) 的外积（p.126–127）")
square = np.zeros((12, 12)); square[3:9, 3:9] = 1.0
resp_c = correlate2d(pad_image(square, "mirror"), CORNER3)
corner_resp = float(np.abs(resp_c[2:5, 2:5]).max())      # 左上角附近
edge_resp = float(np.abs(resp_c[3, 5:8]).max())          # 上边直边中段
chk(corner_resp > 0.5 and edge_resp < 1e-9,
    "#9 corner 核对角点有响应、对轴对齐直边响应为 0（外积可分性）")
ii, jj = np.mgrid[0:12, 0:12]
diag = np.where(ii >= jj, 1.0, 0.0)                       # 45° 对角边
resp_d = correlate2d(pad_image(diag, "mirror"), CORNER3)
diag_resp = float(np.abs(resp_d[3:9, 3:9]).max())
chk(diag_resp > 0.5,
    "#9 corner 核沿对角边也有响应（responds not only to the corners … but also along diagonal edges）")
print(f"    数值: Sobel 竖边 {sv:.2f} / 横边 {sh_:.2f}；corner 核 角点 {corner_resp:.2f}、"
      f"直边 {edge_resp:.2f}、对角边 {diag_resp:.2f}（p.127）")

# ── #C 常数核对（不计数） ────────────────────────────────────────────────────
print("== #C 常数核对 ==")
chk(all(SOBEL3[i].sum() == 0 for i in range(3)), "C1 Sobel 核每行和=0（差分核）")
chk(abs(CORNER3.sum()) < 1e-15, "C2 corner 核总和=0")
chk(np.abs(sharp22(img_f, 0.0) - img_f).max() < 1e-15, "C3 式 3.22 γ=0 化简为恒等（g=f）")
chk(np.abs(np.flip(h_asym) - h_asym).max() > 0, "C4 非对称核 flip(h)≠h（卷积=翻转后的相关）")

# ── #D 区分力注入（不计数） ──────────────────────────────────────────────────
print("== #D 区分力注入自检 ==")
try:
    assert np.allclose(correlate2d(delta, h_asym)[2:5, 2:5], h_asym), "D1"
    raise SystemExit("[FAIL] D1 相关冒充卷积未被拦截")
except AssertionError as e:
    assert str(e) == "D1", e
    print("  [PASS] D1 相关冒充卷积 → #1 断言拦截 ✓")
try:
    m_bad = ringB["clamp"]
    assert abs(m_bad - 100.0) < abs(ringB["zero"] - 100.0), "D2"
    raise SystemExit("[FAIL] D2 clamp 冒充 mirror 未被拦截")
except AssertionError as e:
    assert str(e) == "D2", e
    print("  [PASS] D2 clamp 冒充 mirror → #4 断言拦截 ✓")

# ── 生成图（slides/figures 同步） ────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = ROOT / "code" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 图 A：邻域算子示意（Figure 3.10 同型：源邻域→核→输出）
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8),
                         gridspec_kw={"width_ratios": [1.2, 0.9, 1.0]})
src = np.random.default_rng(7).integers(40, 140, (8, 8))
axes[0].imshow(src, cmap="Blues", vmin=0, vmax=160)
for i in range(9):
    axes[0].axhline(i - 0.5, color="w", lw=0.8)
    axes[0].axvline(i - 0.5, color="w", lw=0.8)
axes[0].add_patch(plt.Rectangle((1.5, 3.5), 3, 3, fill=False, edgecolor="#1f77b4", lw=3))
axes[0].set_title("f (source image), 3×3 neighborhood", fontsize=10)
kern = np.full((3, 3), 0.1); kern[1, 1] = 0.2
axes[1].imshow(kern, cmap="Oranges", vmin=0, vmax=0.25)
for i in range(3):
    axes[1].axhline(i - 0.5, color="w", lw=0.8)
    axes[1].axvline(i - 0.5, color="w", lw=0.8)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{kern[i, j]:.1f}", ha="center", va="center", fontsize=10)
axes[1].set_title("h (kernel)", fontsize=10)
out8 = correlate2d(pad_image(src.astype(float), "mirror"), kern)
axes[2].imshow(out8, cmap="Greens", vmin=0, vmax=160)
for i in range(9):
    axes[2].axhline(i - 0.5, color="w", lw=0.8)
    axes[2].axvline(i - 0.5, color="w", lw=0.8)
axes[2].add_patch(plt.Rectangle((2.5, 3.5), 1, 1, fill=False, edgecolor="#2ca02c", lw=3))
axes[2].set_title("g (filtered output)", fontsize=10)
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("$g(i,j)=\\sum_{k,l} f(i{+}k, j{+}l)\\, h(k,l)$ — one output pixel = weighted sum of a neighborhood (cf. Figure 3.10)",
             fontsize=11)
fig.tight_layout()
save(fig, "e023_nbhd_operator.png")

# 图 B：padding 六模式（Figure 3.13 同型：一维信号 + 各模式模糊后边缘效应）
fig, axes = plt.subplots(2, 3, figsize=(10.5, 4.8), sharex=True, sharey=True)
sig1d = np.full(24, 100.0); sig1d[:4] = 180.0    # 左端亮段
for axp, m in zip(axes.ravel(), MODES):
    sig_pad = pad_image(sig1d, m, 60.0, 4)
    axp.plot(np.arange(-4, 28), sig_pad, lw=1.8, label=f"{m} padded")
    b = np.convolve(sig_pad, np.ones(5) / 5, mode="same")
    axp.plot(np.arange(-4, 28), b, lw=1.8, ls="--", color="#d62728", label="blurred")
    axp.axvspan(-0.5, 3.5, color="#ffcc99", alpha=0.35)
    axp.axvspan(19.5, 23.5, color="#ffcc99", alpha=0.35)
    axp.set_title(m, fontsize=10)
    axp.legend(fontsize=7, loc="upper right")
    axp.set_ylim(-40, 240)
fig.suptitle("Border padding modes and their blur side-effects (cf. Figure 3.13, 1-D slice)", fontsize=11)
fig.tight_layout()
save(fig, "e023_padding_modes.png")

# 图 C：box/tent/bilinear/Gaussian 核（Figure 3.14a–c 同型，1D 剖面）
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
kerns = [("box, K=5", np.ones(5) / 5), ("bilinear (linear tent)", TENT),
         ("'Gaussian' (tent⋆tent)", target)]
for axp, (name, k1) in zip(axes, kerns):
    xs = np.arange(len(k1)) - (len(k1) - 1) / 2
    axp.bar(xs, k1, width=0.6, color="#1f77b4")
    axp.set_title(name, fontsize=10)
    axp.set_xticks(xs)
    axp.set_ylim(0, 0.45)
    for tick, val in zip(xs, k1):
        axp.text(tick, val + 0.012, f"{val:.3f}" if val < 0.1 else f"{val:.2f}",
                 ha="center", fontsize=8)
fig.suptitle("Low-pass kernels: box / linear tent / tent⋆tent cubic 'Gaussian' (cf. Figure 3.14a–c)", fontsize=11)
fig.tight_layout()
save(fig, "e023_kernels.png")

# 图 D：锐化（式 3.22）效果
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
demo = np.zeros((16, 16)); demo[4:12, 4:12] = 0.7
axes[0].imshow(demo, cmap="gray", vmin=0, vmax=1)
axes[0].set_title("f (original)", fontsize=10)
axes[1].imshow(correlate2d(pad_image(demo, "mirror"), BOX3), cmap="gray", vmin=0, vmax=1)
axes[1].set_title("$h_{blur}*f$ (blurred)", fontsize=10)
axes[2].imshow(sharp22(demo, 1.0), cmap="gray", vmin=0, vmax=1)
axes[2].set_title("$g_{sharp}=f+\\gamma(f-h_{blur}*f)$, $\\gamma{=}1$", fontsize=10)
for a in axes:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Unsharp masking (Eq. 3.22): add the high-frequency difference back", fontsize=11)
fig.tight_layout()
save(fig, "e023_unsharp.png")

# 图 E（实验产物，不上屏）：γ 扫描曲线 3.22 vs 3.23
fig, ax = plt.subplots(figsize=(5.5, 3.4))
gs = np.linspace(0, 1.4, 15)
ax.plot(gs, [grad_energy(sharp22(img_f, g)) / e0 for g in gs], "-o", ms=3,
        label="Eq.3.22 (linear)")
ax.plot(gs, [grad_energy(sharp23(img_f, g)) / e0 for g in gs], "-s", ms=3,
        label="Eq.3.23 (darkroom, non-linear, multiplicative)")
ax.set_xlabel("$\\gamma$"); ax.set_ylabel("edge energy / original")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
save(fig, "e023_gamma_sweep.png")

# 图 F：Sobel 与 corner（Figure 3.14d–e 同型：核矩阵+响应）
fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
axes[0, 0].imshow(SOBEL3, cmap="RdBu", vmin=-2, vmax=2)
for i in range(3):
    for j in range(3):
        axes[0, 0].text(j, i, f"{SOBEL3[i, j]:.0f}", ha="center", va="center", fontsize=11)
axes[0, 0].set_title("Sobel $3\\times3$ (edge, vertical emphasis)", fontsize=10)
axes[0, 1].imshow(sobel_mag(pad_image(square, "mirror"))[::-1], cmap="viridis")
axes[0, 1].set_title("Sobel response on a square", fontsize=10)
axes[1, 0].imshow(CORNER3, cmap="RdBu", vmin=-2, vmax=2)
for i in range(3):
    for j in range(3):
        axes[1, 0].text(j, i, f"{CORNER3[i, j]:.0f}", ha="center", va="center", fontsize=11)
axes[1, 0].set_title("corner $3\\times3$ (2nd derivatives)", fontsize=10)
axes[1, 1].imshow(np.abs(correlate2d(pad_image(square, "mirror"), CORNER3))[::-1], cmap="viridis")
axes[1, 1].set_title("corner response (corners + diagonals)", fontsize=10)
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Edge and corner pre-processing kernels (cf. Figure 3.14d–e)", fontsize=11)
fig.tight_layout()
save(fig, "e023_sobel_corner.png")

# 图 G：邻域运算八宫格（Figure 3.11 同型：程序生成，原书图未使用；
# 门A r1 P2-11：f007 讲八宫格时屏面无图，补程序生成八格演示）
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as nd_label

yy, xx = np.mgrid[0:64, 0:64]
gal = 0.25 + 0.10 * np.sin(xx / 6.0)   # 条纹灰阶底
gal[8:40, 8:36] = 0.95                 # 亮方块（主目标）
gal[46:58, 44:58] = 0.72               # 次亮方块（连通域演示）


def bilateral_simple(f, rad=5, sig_r=0.15):
    """简化保边平滑：空间均匀+值域高斯权重（演示用，不进断言）。"""
    pad = np.pad(f, rad, mode="reflect")
    acc = np.zeros_like(f)
    wsum = np.zeros_like(f)
    for u in range(2 * rad + 1):
        for v in range(2 * rad + 1):
            diff = pad[u:u + f.shape[0], v:v + f.shape[1]] - f
            w = np.exp(-(diff * diff) / (2 * sig_r * sig_r))
            acc += w * pad[u:u + f.shape[0], v:v + f.shape[1]]
            wsum += w
    return acc / wsum


gal_b = correlate2d(pad_image(gal, "mirror", 0.0, 2), np.ones((5, 5)) / 25)
gal_sharp = np.clip(sharp22(gal, 1.0), 0, 1)
gal_bin = (gal > 0.5).astype(float)
gal_dil = (correlate2d(pad_image(gal_bin, "mirror", 0.0, 1), np.ones((3, 3))) > 0).astype(float)
gal_dist = distance_transform_edt(gal_bin == 0)
gal_lab, n_lab = nd_label(gal_bin)

fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6))
panels = [
    ("(a) original", gal, "gray", None),
    ("(b) blurred", gal_b, "gray", (0, 1)),
    ("(c) sharpened", gal_sharp, "gray", (0, 1)),
    ("(d) edge-preserving", bilateral_simple(gal), "gray", (0, 1)),
    ("(e) binary", gal_bin, "gray", None),
    ("(f) dilated", gal_dil, "gray", None),
    ("(g) distance transform", gal_dist, "viridis", None),
    ("(h) connected components", gal_lab, "tab10", None),
]
for axp, (name, img, cm, vm) in zip(axes.ravel(), panels):
    axp.imshow(img, cmap=cm, vmin=vm[0] if vm else None, vmax=vm[1] if vm else None)
    axp.set_title(name, fontsize=9.5)
    axp.set_xticks([]); axp.set_yticks([])
fig.suptitle("Some neighborhood operations (cf. Figure 3.11) — program-generated", fontsize=11)
fig.tight_layout()
save(fig, "e023_nonlin_gallery.png")
print(f"    数值: 八宫格演示图（程序生成）连通域数 {n_lab}（主+次方块）")

# ── #S 同步检查（E019–E022 惯例） ────────────────────────────────────────────
print("== #S 同步检查 ==")
SLIDES_FIG = ROOT / "slides" / "figures"
SLIDES_FIG.mkdir(parents=True, exist_ok=True)
mine = {p.name: hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(FIG.glob("e023_*.png"))}
drift = []
for name, m in mine.items():
    tgt = SLIDES_FIG / name
    if tgt.exists() and hashlib.md5(tgt.read_bytes()).hexdigest() != m:
        drift.append(name)
assert not drift, f"[FAIL] #S slides/figures 与 code/figures 漂移: {drift}"
for name in mine:
    (SLIDES_FIG / name).write_bytes((FIG / name).read_bytes())
print(f"  [PASS] #S figures 同步 {len(mine)} 张（先比对后覆盖）")

print(f"\n{n_chk} 条断言 PASS（{n_chk - 4} 数据+4 常数）＋同步 1 项＋区分力自检 2/2 拦截")
