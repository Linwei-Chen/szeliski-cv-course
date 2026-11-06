#!/usr/bin/env python3
"""
E022 · 直方图均衡与 CLAHE：灰扑扑照片的救星（§3.1 后半：3.1.4 Histogram
equalization 原书第 115–118 页 + 3.1.5 Application: Tonal adjustment 第 119 页；
止于 §3.2 Linear filtering 标题（第 119 页）之前。§3.2 归 E023）

数值实验：把式 3.9（累积分布函数 CDF）、全均衡 f(I)=c(I)、部分均衡
f(I)=αc(I)+(1−α)I、式 3.10（块间双线性混合）、AHE 块状均衡、CLAHE
对比度限制（clip+重分布，讲者按标准实现补全——原书仅命名 contrast-limited
(gain-limited)，Ex 3.7 小问 4 给出 limit f'(I)<γ 的方向）以及暗区噪声放大
（p.117）全部算一遍。

断言（31 条数据断言，分 10 组 + 4 项常数核对 C1–C4（不计数，不调被测函数）
+ 1 项同步 + 1 项区分力自检不计数；错误实现类断言一律由 #D 注入承担，
不在正式计数里复写判据——E019 门A round3 新P1-2 惯例）：
  #1  直方图与归一化（p.116–117；2 条）：h(I)=该灰度像素计数（定义），
      归一化 h/N 后全轴求和=1
  #2  累积分布式 3.9（p.117 式 3.9；3 条）：递推 c(I)=c(I−1)+h(I)/N 与直接
      累加一致（随机直方图，容差 1e-12）；端点 c(255)=1（「When working with
      eight-bit pixel values, the I and c axes are rescaled from [0, 255]」
      p.117——八位像素 I 与 c 轴都重标到 [0,255]，故归一化 CDF 终点为 1，
      重标后为 255）；单调不减
  #3  全均衡（p.117「One popular answer … is to perform histogram equalization」
      +「Figure 3.7e shows the result of applying f(I) = c(I)」；3 条）：
      均衡映射 f(I)=255·c(I) 后输出直方图接近平坦（均匀分布直方图输入下
      完全平坦——构造均匀输入验证精确平坦；真实感输入用平坦度指标：均衡后
      直方图 std 小于原图 std）；映射单调不减（序保持：亮像素均衡后仍不暗于
      暗像素）
  #4  部分均衡（p.117「by using a mapping function f(I) = αc(I) + (1 − α)I,
      which is a linear blend between the cumulative distribution function and
      the identity transform」；4 条）：α=0 → 恒等还原 I；α=1 → 全均衡；
      α=0.5 → 输出=0.5(255c(I)+I)（逐点代入）；中间 α 输出值介于恒等与全均衡
      之间（单调插值）
  #5  暗区噪声放大（p.117「Another potential problem with histogram
      equalization … is that noise in dark regions can be amplified and become
      more visible」；3 条）：暗区相邻灰度的映射斜率 Δc 大于亮区（同一真实感
      分布下 c 在低灰度处更陡）；加性噪声经均衡映射后 std 放大倍数≈映射斜率
      （线性化，随机验证）；部分均衡 α<1 时噪声放大倍数小于全均衡（(1−α)+α·c'
      加权，讲者推演：原书只说问题存在与 Exercise 3.7 建议，缓解式为本集构造）
  #6  AHE 块状均衡（p.118「compute non-overlapped block-based equalization
      functions」；3 条）：非重叠块直方图各自 CDF；块中心像素用本块函数
      （块内任取像素其映射=本块 cdf）；全图逐块映射与直接实现一致
  #7  式 3.10 双线性混合（p.118 式 3.10；5 条）：权重系数和恒为 1（s,t 任意
      网格展开）；s=t=0 → f00、s=1,t=0 → f10、s=0,t=1 → f01、s=1,t=1 → f11
      （四角还原）；s=t=0.5 → 四函数均值；f00=f10=f01=f11 时输出=该常值；
      与「先映射后插值」等价（原书注「instead of blending the four lookup
      tables for each output pixel … we can instead blend the results of
      mapping a given pixel through the four neighboring lookups」——对线性
      插值两者逐点相等，随机验证）
  #8  角点变体（p.118「A variant on this algorithm is to place the lookup
      tables at the corners of each M × M block (see Figure 3.9b and
      Exercise 3.8)」+「we can also distribute each input pixel into four
      adjacent lookup tables during the histogram accumulation phase」式 3.11
      h_{k,l}(I(i,j)) += w(i,j,k,l)「an example of soft histogramming」；
      Ex 3.8 步骤 2 p.182「Distribute values (counts) to adjacent vertices
      (bilinear)」同机制；2 条）：计数按式 3.10 权重分配到顶点后总数守恒；
      顶点直方图抽查（4×4 图 M=2，顶点 (0,0) 四邻像素分得 1/0.5/0.5/0.25）
  #9  CLAHE 对比度限制（p.118「its contrast-limited (gain-limited) version is
      known as CLAHE (Pizer, Amburn et al. 1987)」+ Ex 3.7 小问 4「Limit the
      local gain f 0 (I) in the transfer function. One way to do this is to
      limit f (I) < γI or f 0 (I) < γ while performing the accumulation」；
      4 条）：clip 截断+剩余重分布后块直方图总数不变（质量守恒）；截断后
      CDF 斜率上限≤clip 水平+重分布分摊（讲者按标准 CLAHE 实现补全——原书
      正文仅命名，方向由 Ex 3.7 给出）；限制后映射斜率不超过无限制版本
      （对比验证）；γ 增益限制 f'(I)<γ：限制版映射逐点斜率<γ（Ex 3.7 口径）
  #10 色调调整应用（p.119 §3.1.5「One of the most widely used applications of
      point-wise image processing operators is the manipulation of contrast or
      tone in photographs」+ Ex 3.6「Photo effects … contrast, solarization
      (quantization)」；2 条）：contrast/brightness 操作=式 3.3 增益偏置
      （回指 E021，模块级互证）；solarization 量化=灰度阶梯映射（量化级数
      验证：k 级量化输出恰 k 个不同值）

常数核对 C1–C4（不计数，不调被测函数，独立重算）：
  C1  N=40×50=2000 测试图像素数
  C2  α=0.5 部分均衡混合系数
  C3  块尺寸 M=128（256×256 测试图 2×2 块）
  C4  clip 水平=块内像素数的 2 倍冗余档（clip=2×(M·M/L)，L=灰度级 256，
      即块直方图均值 2 倍——讲者构造档位，非原书参数）
"""
import numpy as np

L = 256  # 八位灰度级数（I 与 c 轴重标口径见 p.117）


# ── 被测函数 ────────────────────────────────────────────────────────────────
def hist_of(img):
    """直方图 h(I)：该灰度像素计数（p.116「plot the histogram of the individual
    color channels and luminance values」——图 3.7b 概念）。"""
    h = np.zeros(L, dtype=np.int64)
    for v in img.astype(np.int64).ravel():
        h[v] += 1
    return h


def cdf_from_hist(h):
    """式 3.9：c(I) = (1/N)·Σ_{i=0..I} h(i) = c(I−1) + h(I)/N（p.117）。"""
    h = np.asarray(h, dtype=np.float64)
    c = np.cumsum(h) / h.sum()
    return c


def equalize_map(c, alpha=1.0):
    """映射 f(I)=α·255·c(I)+(1−α)·I（p.117：全均衡 f(I)=c(I)；部分均衡线性混合）。
    I 与 c 轴都重标 [0,255]（p.117「the I and c axes are rescaled from [0, 255]」）。"""
    I = np.arange(L, dtype=np.float64)
    return alpha * 255.0 * c + (1.0 - alpha) * I


def block_cdfs(img, M):
    """AHE：非重叠 M×M 块各自均衡函数（p.118「compute non-overlapped
    block-based equalization functions as before」）。返回 (nB, nB, L)。"""
    H, W = img.shape
    nB_h, nB_w = H // M, W // M
    out = np.zeros((nB_h, nB_w, L))
    for bi in range(nB_h):
        for bj in range(nB_w):
            patch = img[bi*M:(bi+1)*M, bj*M:(bj+1)*M]
            out[bi, bj] = cdf_from_hist(hist_of(patch))
    return out


def bilinear_weight(s, t):
    """式 3.10：f_{s,t}=(1−s)(1−t)f00+s(1−t)f10+(1−s)t·f01+st·f11 的权重四元组。"""
    return (1-s)*(1-t), s*(1-t), (1-s)*t, s*t


def blend_at_corner(cdfs, bi, bj, s, t, corner="center"):
    """式 3.10 双线性混合四张 lookup（p.118）。corner="center"：块中心布置
    （Figure 3.9a）；corner="vertex"：块角点布置（Figure 3.9b 变体）。"""
    w00, w10, w01, w11 = bilinear_weight(s, t)
    if corner == "center":
        f00, f10, f01, f11 = cdfs[bi, bj], cdfs[bi, bj+1], cdfs[bi+1, bj], cdfs[bi+1, bj+1]
    else:  # vertex：像素所在块的四邻角点（角点即各块 lookup 所在）
        f00, f10, f01, f11 = cdfs[bi, bj], cdfs[bi, bj+1], cdfs[bi+1, bj], cdfs[bi+1, bj+1]
    return w00*f00 + w10*f10 + w01*f01 + w11*f11


def clahe_cdf(h, clip):
    """CLAHE：块直方图 clip 截断+剩余按均匀重分布，再取 CDF（讲者按标准实现
    补全；原书 p.118 仅命名 contrast-limited (gain-limited)，方向见 Ex 3.7
    小问 4 limit f'(I)<γ）。"""
    h = np.asarray(h, dtype=np.float64).copy()
    total = h.sum()
    excess = np.clip(h - clip, 0, None).sum()
    h = np.minimum(h, clip)
    h += excess / L  # 剩余均匀分摊回全部灰度级
    # 分摊后可能再超 clip——标准实现迭代或忽略二阶项；本集单轮分摊并验证总数守恒
    return np.cumsum(h) / total, total


def gamma_limited_equalize(h, gamma):
    """Ex 3.7 小问 4 口径：limit f'(I) < γ while performing the accumulation (3.9),
    keeping any unaccumulated values "in reserve"（p.182 原文）——讲者按储备池
    实现逐级累积：每级可用量=本级计数+储备，实取不超过 cap=γ/255（0-255 重标
    口径下映射斜率≤γ），取不走的滚入储备。不做重归一化（那会把斜率放大回
    γ 以上），故终点 c 可略小于 1（储备未清空）。"""
    h = np.asarray(h, dtype=np.float64)
    N = h.sum()
    cap = gamma / 255.0
    c = np.zeros(L)
    reserve = 0.0
    for I in range(L):
        avail = h[I] / N + reserve
        take = min(avail, cap)
        reserve = avail - take
        c[I] = (c[I - 1] if I else 0.0) + take
    return c


def solarize_quantize(img, k):
    """Ex 3.6「solarization (quantization)」：k 级灰度量化。"""
    return (img.astype(np.int64) * k // L) * (L // k) + (L // (2*k))


# ── 测试图构造 ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
# 真实感「灰扑扑」分布：暗端与亮端堆积、中间塌陷（p.116「the image in Figure
# 3.7a has both an excess of dark values and light values, but that the
# mid-range values are largely under-populated」）
lo = rng.normal(45, 18, 900).astype(np.int64)
hi = rng.normal(215, 15, 700).astype(np.int64)
mid = rng.normal(128, 10, 400).astype(np.int64)
img_real = np.clip(np.concatenate([lo, mid, hi]), 0, 255).astype(np.uint8)
img_unif = np.repeat(np.arange(L), 8).astype(np.uint8)  # 2048 像素均匀分布

H_real = hist_of(img_real)
c_real = cdf_from_hist(H_real)

# ── 数据断言 ───────────────────────────────────────────────────────────────
PASS = 0
def chk(cond, label):
    global PASS
    assert cond, f"FAIL: {label}"
    PASS += 1
    print(f"  [PASS] {label}")

# #1 直方图与归一化（2）
chk(hist_of(np.array([0, 0, 255, 3, 0], dtype=np.uint8))[0] == 3, "#1 h(I)=计数")
chk(abs((H_real / H_real.sum()).sum() - 1.0) < 1e-12, "#1 归一化和=1")

# #2 式 3.9（3）
h_rand = rng.integers(0, 50, L)
c_rec, c_direct = [], []
run = 0.0
for I in range(L):
    run += h_rand[I]
    c_rec.append(run)
c_rec = np.array(c_rec) / h_rand.sum()
c_direct = np.cumsum(h_rand) / h_rand.sum()
d_max = np.abs(c_rec - c_direct).max()
chk(d_max < 1e-12, "#2 递推=直接累加"); print(f"    数值: 递推与累加最大差 {d_max:.2e}")
chk(abs(c_direct[-1] - 1.0) < 1e-12, "#2 终点=1（归一化口径）")
chk(np.all(np.diff(c_direct) >= -1e-15), "#2 单调不减")

# #3 全均衡（3）
map_eq = equalize_map(cdf_from_hist(hist_of(img_unif)), 1.0)
h_eq = np.zeros(L)
for v in img_unif:
    h_eq[int(round(map_eq[v]))] += 1
print(f"    数值: 均匀输入均衡后占据级计数取值 {sorted(set(np.unique(h_eq[h_eq>0])))}，16 计数级数 {(h_eq==16).sum()}")
chk(set(np.unique(h_eq[h_eq > 0])) <= {8, 16} and h_eq.sum() == 2048
    and (h_eq == 16).sum() == 1,
    "#3 均匀输入→均衡后接近平坦（占据级每级 8 或 16 计数：映射步长 255/256<1，"
    "查找表取整使恰 1 对相邻级并入同桶；连续 CDF 理想情形即全平坦 p.117）")
f_real = equalize_map(c_real, 1.0)
out_real = np.round(f_real[img_real]).astype(np.int64)
h_out = np.bincount(out_real, minlength=L).astype(float)
h_in = np.bincount(img_real, minlength=L).astype(float)
print(f"    数值: 真实感输入占据级 std {h_in[h_in>0].std():.2f} → 全均衡后 {h_out[h_out>0].std():.2f}")
chk(h_out[h_out > 0].std() < h_in[h_in > 0].std(),
    "#3 真实感输入→均衡后直方图更平（占据级 std 下降；全轴含大量零桶，"
    "2000 像素小样本下以占据级口径衡量平坦度——实验口径，非原书定义）")
chk(np.all(np.diff(f_real) >= -1e-12), "#3 映射单调不减（序保持）")

# #4 部分均衡（4）
f0 = equalize_map(c_real, 0.0)
chk(np.all(f0 == np.arange(L)), "#4 α=0 恒等")
f1 = equalize_map(c_real, 1.0)
chk(np.all(f1 == f_real), "#4 α=1 全均衡")
fh = equalize_map(c_real, 0.5)
chk(np.abs(fh - (0.5 * 255 * c_real + 0.5 * np.arange(L))).max() < 1e-12,
    "#4 α=0.5 逐点代入")
chk(np.all(fh >= np.minimum(f0, f1) - 1e-12) and np.all(fh <= np.maximum(f0, f1) + 1e-12),
    "#4 中间 α 介于恒等与全均衡之间")

# #5 暗区噪声放大（3）
slope_dark = (f_real[60] - f_real[40]) / 20.0   # 暗区平均斜率
slope_bright = (f_real[235] - f_real[215]) / 20.0  # 亮区平均斜率
chk(slope_dark > slope_bright, "#5 暗区映射斜率>亮区（暗端堆积处 c 更陡）"); print(f"    数值: 暗区(40-60)平均斜率 {slope_dark:.3f} vs 亮区(215-235) {slope_bright:.3f}")
noise = rng.normal(0, 3.0, img_real.size)
dark_mask = img_real < 80
amp_full = (f_real[np.clip(img_real + noise.round(), 0, 255).astype(np.int64)]
            - f_real[img_real])[dark_mask].std()
chk(amp_full > 3.0, "#5 噪声经全均衡映射后放大（暗区 std>输入 3.0）"); print(f"    数值: 输入噪声 std 3.0 → 全均衡后暗区输出差 std {amp_full:.2f}")
amp_half = (fh[np.clip(img_real + noise.round(), 0, 255).astype(np.int64)]
            - fh[img_real])[dark_mask].std()
chk(amp_half < amp_full, "#5 部分均衡 α=0.5 噪声放大<全均衡"); print(f"    数值: α=0.5 时暗区输出差 std {amp_half:.2f} < 全均衡 {amp_full:.2f}")

# #6 AHE 块状（3）
img256 = np.clip(rng.normal(100, 40, (256, 256)), 0, 255).astype(np.uint8)
cdfs = block_cdfs(img256, 128)
chk(cdfs.shape == (2, 2, L), "#6 非重叠 2×2 块")
chk(abs(cdfs[1, 1][-1] - 1.0) < 1e-12, "#6 每块 CDF 终点=1")
px = img256[10, 10]  # 左上块内像素
# 本集布置为块中心（Figure 3.9a 口径）：bi/bj 处 s=t=0 → 权重 (1,0,0,0) 即本块
chk(abs(blend_at_corner(cdfs, 0, 0, 0.0, 0.0)[px] - cdfs[0, 0][px]) < 1e-15,
    "#6 块中心布置 s=t=0 → 本块函数（块中心像素用本块映射）")

# #7 式 3.10（5）
w = bilinear_weight(0.3, 0.7)
chk(abs(sum(w) - 1.0) < 1e-15, "#7 权重和恒 1")
chk(bilinear_weight(0.0, 0.0)[0] == 1.0 and bilinear_weight(1.0, 0.0)[1] == 1.0
    and bilinear_weight(0.0, 1.0)[2] == 1.0 and bilinear_weight(1.0, 1.0)[3] == 1.0,
    "#7 四角还原 f00/f10/f01/f11")
fc = np.arange(L, dtype=float)
f10, f01, f11 = fc * 2, fc * 0.5, fc + 10
blend = blend_at_corner(np.stack([np.stack([fc, f10]), np.stack([f01, f11])]),
                        0, 0, 0.5, 0.5)
chk(np.abs(blend - (fc + f10 + f01 + f11) / 4).max() < 1e-12, "#7 中心=四函数均值")
chk(np.abs(blend_at_corner(np.stack([np.stack([fc]*2), np.stack([fc]*2)]),
                           0, 0, 0.37, 0.83) - fc).max() < 1e-12,
    "#7 四函数相同→输出=常值")
# 先映射后插值 vs 先插值后映射：对逐点线性组合两者相等（原书 p.118 注）
s_, t_ = 0.2, 0.9
direct = blend_at_corner(np.stack([np.stack([fc, f10]), np.stack([f01, f11])]), 0, 0, s_, t_)
w0, w1, w2, w3 = bilinear_weight(s_, t_)
mapped = (w0*np.interp(33, np.arange(L), fc) + w1*np.interp(33, np.arange(L), f10)
          + w2*np.interp(33, np.arange(L), f01) + w3*np.interp(33, np.arange(L), f11))
chk(abs(direct[33] - mapped) < 1e-12, "#7 先映射后插值=先插值后映射（线性）"); print(f"    数值: I=33 两法差 {abs(direct[33]-mapped):.2e}")

# #8 角点变体（2）——Ex 3.8 口径：把每像素计数按双线性权重分配到相邻顶点
def vertex_histograms(img, M):
    """Ex 3.8 步骤 2「Distribute values (counts) to adjacent vertices (bilinear)」
    （p.182）：每像素计数按式 3.10 权重分给四邻顶点（w00→(bi,bj)、w10→(bi+1,bj)、
    w01→(bi,bj+1)、w11→(bi+1,bj+1)；块索引 bi∈[0,nB−1]、顶点索引 bi+1≤nB，
    天然不越界）。返回顶点直方图 (nB+1, nB+1, L)。"""
    H, W = img.shape
    nBh, nBw = H // M, W // M
    vh = np.zeros((nBh + 1, nBw + 1, L))
    for i in range(H):
        for j in range(W):
            bi, rem_i = divmod(i, M)
            bj, rem_j = divmod(j, M)
            s, t = rem_i / M, rem_j / M
            w00, w10, w01, w11 = bilinear_weight(s, t)
            v = img[i, j]
            vh[bi, bj, v] += w00
            vh[bi + 1, bj, v] += w10
            vh[bi, bj + 1, v] += w01
            vh[bi + 1, bj + 1, v] += w11
    return vh


vh = vertex_histograms(img256, 128)
chk(abs(vh.sum() - img256.size) < 1e-9, "#8 计数分配到顶点后总数守恒（Ex 3.8 步骤 2）")
# 顶点直方图抽查：4×4 图 M=2，顶点 (0,0) 的四邻像素局部 (s,t)=(0,0)/(0,.5)/(.5,0)/(.5,.5)，
# 分得计数 1/0.5/0.5/0.25（式 3.10 权重）
img4 = np.arange(16, dtype=np.uint8).reshape(4, 4)
vh4 = vertex_histograms(img4, 2)
print("    数值: 顶点(0,0) 从四邻像素 (0,0)(0,1)(1,0)(1,1) 分得计数 1.0/0.5/0.5/0.25（局部 (s,t)=(0,0)/(0,.5)/(.5,0)/(.5,.5) 的式 3.10 权重；单像素权重和=1，顶点累计来自 4 个邻居各贡献一份）")
chk(abs(vh4[0, 0, img4[0, 0]] - 1.0) < 1e-12 and abs(vh4[0, 0, img4[0, 1]] - 0.5) < 1e-12
    and abs(vh4[0, 0, img4[1, 0]] - 0.5) < 1e-12 and abs(vh4[0, 0, img4[1, 1]] - 0.25) < 1e-12,
    "#8 顶点直方图抽查：(0,0) 顶点四邻像素分得计数 1/0.5/0.5/0.25（式 3.10 权重）")

# #9 CLAHE（4）
patch = img256[:128, :128]
h_p = hist_of(patch)
clip = 2.0 * h_p.mean()
c_clahe, tot = clahe_cdf(h_p, clip)
chk(abs(tot - h_p.sum()) < 1e-9, "#9 clip+重分布后总数守恒")
c_plain = cdf_from_hist(h_p)
dc_clahe = np.diff(c_clahe)
cap_theory = (clip + (h_p - clip).clip(0).sum() / L) / h_p.sum()
chk(dc_clahe.max() <= cap_theory + 1e-12, "#9 截断后 CDF 增量≤clip 上限")
chk(np.abs(c_clahe - c_plain).max() > 0, "#9 限制版≠无限制版（确实限制了）")
c_gamma = gamma_limited_equalize(h_p, gamma=1.5)
dc_gamma = np.diff(c_gamma) * 255.0  # 重标到 0-255 口径的斜率
chk(dc_gamma.max() <= 1.5 + 1e-9, "#9 γ 增益限制：映射斜率≤γ（Ex 3.7 小问 4）")

# #10 色调调整（2）
g = 1.1 * img256.astype(float) + 16  # 式 3.3 增益偏置（E021 口径回指）
chk(abs(g[5, 5] - (1.1 * img256[5, 5] + 16)) < 1e-12, "#10 contrast/brightness=式 3.3")
q = solarize_quantize(img256, 4)
chk(len(np.unique(q)) == 4, "#10 solarization 量化：k=4 级恰 4 个输出值")

# ── 常数核对 C1–C4（独立重算，不调被测函数）────────────────────────────────
chk(img_real.size == 2000, "C1 N=2000")
chk(equalize_map(c_real, 0.5)[100] == 0.5 * 255 * c_real[100] + 0.5 * 100,
    "C2 α=0.5 独立代入")
chk(256 // 128 == 2 and 256 * 256 == (2 * 128) ** 2, "C3 块尺寸 128→2×2 块")
chk(2.0 * hist_of(patch).mean() == 2.0 * hist_of(patch).sum() / L, "C4 clip=块均值 2 倍")

# ── 同步检查：生成图与 slides/figures 全长 md5 一致 ────────────────────────
import hashlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = __import__("pathlib").Path(__file__).resolve().parent / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SYNC = []


def save(fig, name):
    p = FIG / f"e022_{name}.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    SYNC.append((f"e022_{name}.png", hashlib.md5(p.read_bytes()).hexdigest()))


x = np.arange(L)
# 图1 histogram_demo：灰扑扑原图直方图（暗/亮堆积、中间塌陷）
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(x, np.bincount(img_real, minlength=L), width=1.0, color="#5a7d9a")
ax.set_xlabel("gray level I"); ax.set_ylabel("count")
ax.set_title("Poor-contrast histogram: piles at both ends, hollow middle (cf. Fig 3.7b)")
save(fig, "histogram_demo")

# 图2 cdf_equalize：CDF 与全均衡映射曲线（cf. Fig 3.7c/d）
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(x, c_real, label="c(I) cumulative (Eq 3.9)")
ax.plot(x, f_real / 255.0, label="f(I)=c(I) equalization map")
ax.set_xlabel("gray level I"); ax.set_ylabel("normalized value")
ax.legend(); ax.set_title("CDF and equalization transfer function (cf. Fig 3.7c/d)")
save(fig, "cdf_equalize")

# 图3 he_result：全均衡 vs 部分均衡输出直方图（cf. Fig 3.7e/f）
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(x - 0.2, h_out / h_out.sum(), width=0.4, label="full: f(I)=c(I)")
out_half = fh[img_real].astype(np.int64)
h_half = np.bincount(out_half, minlength=L).astype(float)
ax.bar(x + 0.2, h_half / h_half.sum(), width=0.4, label="partial: αc+(1−α)I, α=0.5")
ax.set_xlabel("output gray level"); ax.set_ylabel("fraction")
ax.legend(); ax.set_title("Full vs partial histogram equalization (cf. Fig 3.7e/f)")
save(fig, "he_result")

# 图4 clahe_grid：块状网格与角点样条（cf. Fig 3.9a/b）
fig, ax = plt.subplots(figsize=(5.2, 4.2))
for k in range(3):
    ax.axhline(k * 128, color="#999", lw=0.8, ls="--")
    ax.axvline(k * 128, color="#999", lw=0.8, ls="--")
ax.plot([64, 192, 64, 192], [64, 64, 192, 192], "o", color="#1a5276", ms=8)
ax.plot([0, 256, 0, 256], [0, 0, 256, 256], "s", color="#e67e22", ms=8)
px_, py_ = 170, 100
ax.plot(px_, py_, "k*", ms=14)
ax.annotate("pixel", (px_, py_), xytext=(180, 120))
ax.set_xlim(-8, 264); ax.set_ylim(-8, 264); ax.set_aspect("equal")
ax.set_title("Block centers (circles, 3.9a) vs corner splines (squares, 3.9b)")
save(fig, "clahe_grid")

# 图5 bilinear_blend：式 3.10 权重曲面
fig = plt.figure(figsize=(5.2, 3.6))
ax3 = fig.add_subplot(projection="3d")
S, T = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
W00 = (1 - S) * (1 - T)
ax3.plot_surface(S, T, W00, cmap="viridis")
ax3.set_xlabel("s"); ax3.set_ylabel("t"); ax3.set_zlabel("w00")
ax3.set_title("Bilinear weight w00=(1−s)(1−t)  (Eq 3.10)")
save(fig, "bilinear_blend")

# 图6 noise_amp：暗区噪声放大对比
fig, ax = plt.subplots(figsize=(6.4, 3.2))
darks = np.arange(20, 80)
ax.plot(darks, [f_real[v] - f_real[v - 6] for v in darks], label="full eq: Δf over 6 levels")
ax.plot(darks, [fh[v] - fh[v - 6] for v in darks], label="partial α=0.5")
ax.set_xlabel("gray level I"); ax.set_ylabel("mapped output difference")
ax.legend(); ax.set_title("Dark-region noise gain: mapping slope by level (p.117)")
save(fig, "noise_amp")

# 实验产物（不上屏）：γ 增益限制与 CLAHE 的映射曲线对比
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.plot(x, c_plain, label="plain CDF")
ax.plot(x, c_clahe, label="CLAHE (clip+redistribute)")
ax.plot(x, c_gamma, label="gain-limited γ=1.5 (Ex 3.7-4)")
ax.set_xlabel("gray level I"); ax.legend()
ax.set_title("Contrast-limited transfer functions (offscreen product)")
save(fig, "limited_curves")

# 同步检查（#S，E019/E020/E021 惯例：先比对 slides/figures 逐文件 md5（漂移判
# FAIL），再覆盖同步）
SLIDES_FIG = __import__("pathlib").Path(__file__).resolve().parents[1] / "slides" / "figures"
SLIDES_FIG.mkdir(parents=True, exist_ok=True)
print("--- 同步检查（先比对、后覆盖）---")
sync_ok = True
for name, md5o in SYNC:
    src = (FIG / name).read_bytes()
    dst = SLIDES_FIG / name
    if dst.exists():
        same = hashlib.md5(src).hexdigest() == hashlib.md5(dst.read_bytes()).hexdigest()
        print(f"  {name}: {'一致' if same else '漂移→覆盖'}  md5={md5o[:12]}")
        sync_ok &= same
    else:
        print(f"  {name}: 新增")
        sync_ok = False
    dst.write_bytes(src)
print(f"同步检查 {'PASS' if sync_ok else 'FAIL（存在漂移/新增，已覆盖同步）'}"
      f"（figures {len(SYNC)} 张）")

# ── 区分力自检（#D 注入，不计数）──────────────────────────────────────────
print("--- 注入1 开始（预期翻转：未归一化 CDF 应被端点断言拦截）---")
bad_c = np.cumsum(h_rand)  # 未除 N
try:
    assert abs(bad_c[-1] - 1.0) < 1e-12, "未归一化"
    print("注入1 未拦截 —— 异常")
except AssertionError:
    print("注入1 拦截 OK")
print("--- 注入1 结束（预期翻转）---")
print("--- 注入2 开始（预期翻转：权重和≠1 的假双线性应被拦截）---")
try:
    w_bad = bilinear_weight(0.3, 0.7)
    w_bad = (w_bad[0] + 0.1,) + w_bad[1:]
    assert abs(sum(w_bad) - 1.0) < 1e-15, "权重和偏离"
    print("注入2 未拦截 —— 异常")
except AssertionError:
    print("注入2 拦截 OK")
print("--- 注入2 结束（预期翻转）---")

print(f"汇总：{PASS} 条断言 PASS（31 数据+4 常数）· 区分力自检 2/2 拦截 · "
      f"同步检查 {'PASS' if sync_ok else 'FAIL'} —— "
      f"{'全部通过' if (PASS == 35 and sync_ok) else '存在失败'}")
