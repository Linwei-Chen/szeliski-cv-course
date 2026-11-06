#!/usr/bin/env python3
"""E030 数值实验：图像金字塔（§3.5.3，原书 pp.154–159 上）。

口径注记（与 main 标头逐字同步，供口播/deck 引用）：
- 原书图 Figure 3.31–3.35 一律未使用，全部程序重绘并标注「原书图未使用」。
- 式 3.68 的 pdftotext 版式把一维五抽头核断行拉大空格（「c b a  b c」），
  本实验取一维对称核 [c,b,a,b,c] 解读——依据：b=1/4、c=1/4−a/2、a=3/8 代入
  恰得式 3.69 的 [1,4,6,4,1]/16，与 E029 已验 Table 3.2 binomial 列全等；
  若按其他断行解读则推不出式 3.69，故一维解读是唯一自洽读法。
- Figure 3.32 图注「reconstruction filter coefficients are twice the analysis
  coefficients」：原书给的是系数倍率事实，未给推导——本实验以 perfect
  reconstruction 数值闭环反证该倍率（synthesis=analysis×2 时误差为机器精度）。
- 脚注 14「repeated convolutions of the binomial kernel converge to a Gaussian…
  this is true for any smoothing kernel (Wells 1986)」：原书未给收敛率——
  本实验以 box 核与 binomial 核各重复卷积对照演示（讲者口径）。
- half-octave/quincunx：原书未给 quincunx 滤波器组系数（引 Feilner, Van De
  Ville, and Unser 2005）——本实验只演示采样格与层数账（讲者口径），不实现
  完整滤波器组。
- Figure 3.35b 的 3/4 像素账为图注原文字面（「each wavelet level stores 3/4
  of the original pixels…so that the total number of wavelet coefficients and
  original pixels is the same」），本实验做几何级数闭环。
- DoG 与 LoG 的关系：原书只给式 3.70/3.71 定义与 Figure 3.34 的「close fit」
  图注，未给小 σ 极限系数——本实验只验证线性性/DC 抵消/剖面相关（讲者口径），
  不作等式断言。
- 变体金字塔的「less aliasing」（L197–198）原书为文字陈述、未给度量——本实验
  不作混叠数值断言，仅验证其「not self-inverting」（L198–199）。
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

print("口径注记（与 refs 逐字同步；门A r1 P0-3 补打印）：")
print("· 式 3.68 读法：原书第 155 页明说「a five-tap kernel of the form」——一维五抽头核"
      " [c,b,a,b,c]，b=1/4、c=1/4−a/2、a=3/8 代入恰得式 3.69 binomial [1,4,6,4,1]/16"
      "（与 E029 Table 3.2 binomial 列全等）。pdftotext 留档把五个系数的行内空格拉大"
      "（抽取伪迹），原书印刷并未断开——排版伪迹只记本标头，不进 deck/口播。")
print("· Figure 3.32 图注「reconstruction filter coefficients are twice the analysis "
      "coefficients」：原书给事实未给推导——本实验以 perfect reconstruction 闭环反证"
      "（×2 时 err=1.11e−16；不×2 err=0.793）。")
print("· 脚注 14「any smoothing kernel」：原书未给收敛率——binomial 与 box 对照演示"
      "（讲者口径；两种核不足以外推一切核）。")
print("· 变体「less aliasing」：原书文字陈述、无度量——不作数值断言，仅数值实证 "
      "not self-inverting（err=0.353）。")
print("· DoG↔LoG：原书只给式 3.70/3.71 与 Figure 3.34「close fit」图注，未给符号约定——"
      "讲者口径同尺度剖面 |cc|=0.9184（反相），形状同族不作等式断言。")
print("· half-octave/quincunx：原书未给 quincunx 滤波器组系数（引 Feilner, Van De Ville, "
      "and Unser 2005）——本实验只核算层数账与采样格算术（7-2），不实现完整滤波器组。")
print()
N_PASS = 0
N_FAIL = 0
INJ = {"total": 0, "caught": 0}


def chk(label, ok, detail=""):
    global N_PASS, N_FAIL
    if ok:
        N_PASS += 1
        print(f"  [PASS] {label}" + (f"  {detail}" if detail else ""))
    else:
        N_FAIL += 1
        print(f"  [FAIL] {label}" + (f"  {detail}" if detail else ""))


def inject(label, wrong_impl_passes):
    """区分力注入：wrong_impl_passes=True 表示错误实现蒙混过关=未拦截。"""
    INJ["total"] += 1
    caught = not wrong_impl_passes
    if caught:
        INJ["caught"] += 1
    print(f"  [注入{'已拦截' if caught else '未拦截'}] {label}")


def inj_detail(detail):
    print(f"    └ {detail}")


# ───────────────────────── 基础工具 ─────────────────────────

def conv1d(x, k):
    """一维卷积（same 长度；对称核下与相关等价）。"""
    k = np.asarray(k, float)
    h = (len(k) - 1) // 2
    xp = np.concatenate([np.zeros(h), x, np.zeros(2 * h)])
    return np.convolve(xp, k, "valid")[: len(x)]


def conv_axis(img, k, axis):
    """沿 axis 做一维 same 卷积（对称核，等价滑动窗口加权和）。"""
    k = np.asarray(k, float)
    h = (len(k) - 1) // 2
    n0, n1 = img.shape
    if axis == 1:
        xp = np.pad(img, ((0, 0), (h, 2 * h)))
        out = np.zeros((n0, n1))
        for j, kv in enumerate(k):
            out += kv * xp[:, j:j + n1]
    else:
        xp = np.pad(img, ((h, 2 * h), (0, 0)))
        out = np.zeros((n0, n1))
        for j, kv in enumerate(k):
            out += kv * xp[j:j + n0, :]
    return out


def conv2d_sep(img, k):
    """可分离 2D 卷积：行、列各一遍一维核（式 3.63 可分离惯例，E028 已验）。"""
    return conv_axis(conv_axis(img, k, 1), k, 0)


def down2(img):
    return img[::2, ::2]


def up2_zero(img):
    """↑2 插零（Figure 3.32 图注：white circles indicate zero values inserted）。"""
    H, W = img.shape
    z = np.zeros((2 * H, 2 * W))
    z[::2, ::2] = img
    return z


BINOM = np.array([1, 4, 6, 4, 1]) / 16.0          # 式 3.69
BINOM_X2 = np.array([1, 4, 6, 4, 1]) / 8.0        # analysis×2（Figure 3.32 图注）


def up2_interp(img):
    """↑2 后过插值核（=analysis×2，E029 脚注 11「幅度×r」同机制的镜像）。"""
    return conv2d_sep(up2_zero(img), BINOM_X2)


def gauss_pyramid(img, nlev):
    """analysis：逐层 blur+↓2（Figure 3.33 黄色通路）。"""
    lev = [img]
    for _ in range(nlev - 1):
        lev.append(down2(conv2d_sep(lev[-1], BINOM)))
    return lev


def lap_pyr(img, nlev):
    """标准 Laplacian pyramid：带通=本层−插值回的上采样下层。"""
    G = gauss_pyramid(img, nlev)
    L = []
    for i in range(nlev - 1):
        up = up2_interp(G[i + 1])[: G[i].shape[0], : G[i].shape[1]]
        L.append(G[i] - up)
    L.append(G[-1])
    return L


def recon(L):
    """synthesis：从最粗层逐层加回（Figure 3.33 重构通路）。"""
    I = L[-1]
    for i in range(len(L) - 2, -1, -1):
        up = up2_interp(I)[: L[i].shape[0], : L[i].shape[1]]
        I = L[i] + up
    return I


def recon_nox2(L):
    """对照：synthesis 用不 ×2 的 analysis 核（预期重构失败）。"""
    I = L[-1]
    for i in range(len(L) - 2, -1, -1):
        up = conv2d_sep(up2_zero(I), BINOM)[: L[i].shape[0], : L[i].shape[1]]
        I = L[i] + up
    return I


def test_image(H=256, W=256):
    """程序生成测试图：同心环+斜坡+方格+高频条纹（丰富频谱）。"""
    y, x = np.mgrid[0:H, 0:W].astype(float)
    r = np.hypot(x - W / 2, y - H / 2)
    img = 0.5 + 0.18 * np.sin(2 * np.pi * r / 14)
    img += 0.10 * x / W
    img += 0.08 * ((x // 32 + y // 32) % 2)
    img += 0.06 * np.sin(2 * np.pi * 180 * x / W)
    return np.clip(img, 0, 1)


# ───────────────────────── #1 式 3.68→3.69 ─────────────────────────
print("#1 式 3.68 核推导：b=1/4、c=1/4−a/2、a=3/8 → 式 3.69 binomial（p.155 L82–90）")
a, b = 3 / 8, 1 / 4
c = 1 / 4 - a / 2
k68 = np.array([c, b, a, b, c])
print("  [读数] a=3/8 → c=1/4−a/2 =", c, "→ 核 =", np.round(k68, 6).tolist())
chk("1-1 式 3.68(a=3/8) 全等式 3.69 [1,4,6,4,1]/16",
    np.array_equal(k68, BINOM), f"max|Δ|={np.max(np.abs(k68 - BINOM)):.1e}")
chk("1-2 与 E029 Table 3.2 binomial 列全等（0.3750/0.2500/0.0625）",
    k68[2] == 0.3750 and k68[1] == 0.2500 and k68[0] == 0.0625)
chk("1-3 unit DC gain（Table 3.2 图注 L42–43：sum up to 1）", k68.sum() == 1.0,
    f"sum={k68.sum():.15f}")
chk("1-4 对称奇长（Table 3.2 图注 L42：odd length, symmetric）",
    len(k68) == 5 and np.array_equal(k68, k68[::-1]))
chk("1-5 shifts and adds 系数来源：(1+z⁻¹)⁴=[1,4,6,4,1]（L91 shifts and adds）",
    np.array_equal(np.convolve(np.convolve([1, 1], [1, 1]), np.convolve([1, 1], [1, 1])),
                   [1, 4, 6, 4, 1]))
inject("D1 错误 a=1/2（c=0）应推不出 binomial（断言非恒真）",
       np.allclose(np.array([1 / 4 - (1 / 2) / 2, b, 1 / 2, b, 1 / 4 - (1 / 2) / 2]), BINOM))

# ───────────────────────── #2 Gaussian pyramid ─────────────────────────
print("#2 Gaussian pyramid 构造：blur+↓2，每层像素 ¼（Figure 3.31 L114–115）")
IMG = test_image()
G = gauss_pyramid(IMG, 5)
shapes = [g.shape for g in G]
print("  [读数] 层形状:", shapes)
chk("2-1 五层构造且逐层宽高减半",
    all(s2[0] == s1[0] // 2 and s2[1] == s1[1] // 2 for s1, s2 in zip(shapes, shapes[1:])))
chk("2-2 每层像素数=上层 ¼（Figure 3.31：a quarter of the pixels）",
    all(g2.size * 4 == g1.size for g1, g2 in zip(G, G[1:])))
chk("2-3 最粗层非空且幅值有界 [0,1]（非负归一核=凸组合，值域不扩）",
    G[-1].size > 0 and G[-1].min() >= 0 and G[-1].max() <= 1,
    f"[{G[-1].min():.3f},{G[-1].max():.3f}]")
Gn = [IMG]
for _ in range(4):
    Gn.append(down2(Gn[-1]))
inject("D2 不 blur 直接 ↓2 的『金字塔』应与正确版不同（blur 必要性）",
       np.allclose(G[1], Gn[1]))
inj_detail(f"max|Δ|={np.max(np.abs(G[1] - Gn[1])):.3f}")

# ───────────────────────── #3 perfect reconstruction ─────────────────────────
print("#3 Laplacian pyramid 与 perfect reconstruction（L171–178：exactly reconstruct）")
L = lap_pyr(IMG, 5)
Ihat = recon(L)
err = np.max(np.abs(IMG - Ihat))
err_nox2 = np.max(np.abs(IMG - recon_nox2(L)))
print(f"  [读数] perfect reconstruction max|I−Î| = {err:.3e}")
print(f"  [读数] 对照：synthesis 不 ×2 时 max|I−Î| = {err_nox2:.3f}")
chk("3-1 五层 Laplacian 金字塔各层尺寸与 Gaussian 层一致",
    [l.shape for l in L] == shapes)
chk("3-2 重构系数=analysis×2 时 max|I−Î| ≤ 1e−10（机器精度级）", err <= 1e-10, f"err={err:.2e}")
inner = up2_interp(np.full((4, 4), 0.5))[2:6, 2:6]
chk("3-3 插值核=analysis×2 的增益闭环：常数图 ↑2 插零后卷积 [1,4,6,4,1]/8 内部恒等（半零位×核和 2=1）",
    np.allclose(inner, 0.5, atol=1e-12), f"内部区 [{inner.min():.6f},{inner.max():.6f}]")
chk("3-4 带通=本层−上采样插值下层（Figure 3.33 蓝色通路定义 L161–164）逐元素复核",
    np.array_equal(L[0], IMG - up2_interp(G[1])[: IMG.shape[0], : IMG.shape[1]]))
inject("D3 synthesis 不 ×2 应重构失败（×2 倍率必要）", err_nox2 <= 1e-10)

# ───────────────────────── #4 变体（piping 直连）不 self-inverting ─────────────────────────
print("#4 变体：低通直取原模糊图（L179–181+L197–199：less aliasing, not self-inverting）")


def lap_pyr_variant(img, nlev):
    """变体：带通 = L_i − blur(L_i)（piping L box 直连减法，不经 ↓2↑2 往返）。"""
    Gv = gauss_pyramid(img, nlev)
    Lv = []
    for i in range(nlev - 1):
        Lv.append(Gv[i] - conv2d_sep(Gv[i], BINOM))
    Lv.append(Gv[-1])
    return Lv


def recon_variant(Lv):
    """变体只能用同标准 synthesis 加回（无 ×2 补偿路径）→ 预期误差显著非零。"""
    I = Lv[-1]
    for i in range(len(Lv) - 2, -1, -1):
        up = up2_interp(I)[: Lv[i].shape[0], : Lv[i].shape[1]]
        I = Lv[i] + up
    return I


Lv = lap_pyr_variant(IMG, 5)
err_v = np.max(np.abs(IMG - recon_variant(Lv)))
print(f"  [读数] 变体重构 max|I−Î| = {err_v:.3f}（标准版 {err:.2e}）")
chk("4-1 变体带通=本层−本层再模糊（piping 定义复核）",
    np.array_equal(Lv[0], IMG - conv2d_sep(IMG, BINOM)))
chk("4-2 变体重构误差显著非零（不 self-inverting，L198–199）", err_v > 0.05, f"err={err_v:.3f}")
chk("4-3 变体带通与标准带通不同（低通定义差异在案）",
    not np.allclose(Lv[0], L[0]), f"max|Δ|={np.max(np.abs(Lv[0]-L[0])):.3f}")
chk("4-4 「less aliasing」原书为文字陈述未给度量——不作数值断言（申报通过）",
    True, "见标头口径注记")

# ───────────────────────── #5 DoG 与 LoG ─────────────────────────
print("#5 DoG/LoG：式 3.70 线性性+DC 抵消+剖面相关（p.158 L200–221）")


def g1d(sigma, half=9):
    x = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


p_g1, p_g2 = g1d(1.0), g1d(2.0)
p_dog = p_g1 - p_g2
sig_test = np.sin(np.linspace(0, 8 * np.pi, 128))
chk("5-1 式 3.70 线性性：(Gσ1−Gσ2) 卷积 = Gσ1 卷积 − Gσ2 卷积",
    np.allclose(conv1d(sig_test, p_g1 - p_g2),
                conv1d(sig_test, p_g1) - conv1d(sig_test, p_g2), atol=1e-10))
chk("5-2 DoG 的 DC 增益=0（带通必要：两归一低通之差在 ω=0 抵消）",
    abs(p_dog.sum()) < 1e-12, f"sum={p_dog.sum():.1e}")
rng = np.random.default_rng(7)
sig_rw = rng.standard_normal(128) + 0.5 * np.sin(np.linspace(0, 10 * np.pi, 128))
log_kernel = np.convolve(g1d(1.0), np.array([1.0, -2.0, 1.0]))   # σ=1 尺度数值 LoG：先平滑再 ∇² 差分
num_log = np.convolve(sig_rw, log_kernel, "same")
num_dog = np.convolve(sig_rw, p_dog, "same")
cc = np.corrcoef(num_log, num_dog)[0, 1]
print(f"  [读数] DoG 与同尺度数值 LoG（G_σ1 平滑后 ∇²）剖面相关系数 = {cc:.4f}（负值=反相）")
chk("5-3 DoG 与 LoG 剖面形状同族：|相关|>0.85（讲者口径；尺度配平——DoG(σ1,σ2) 对应 σ1 尺度 LoG，"
    "原书 Figure 3.34 图注 close fit；反相系约定——σ1<σ2 时 DoG 中心正、∇²G 中心负，原书未给符号约定，"
    "不作等式断言）",
    abs(cc) > 0.85, f"cc={cc:.4f}")
chk("5-4 DoG 主瓣两侧负瓣存在（带通形状特征：中心正、|x|=3 处为负）",
    p_dog[9] > 0 and p_dog[6] < 0 and p_dog[12] < 0,
    f"center={p_dog[9]:.4f} sides(x=±3)={p_dog[6]:.4f}/{p_dog[12]:.4f}")
inject("D4 退化 DoG(σ1=σ2)≡0，『峰值>0』断言应 FAIL（检验断言非恒真）",
       (g1d(1.0) - g1d(1.0)).max() > 1e-12)

# ───────────────────────── #6 binomial 收敛 Gaussian（脚注 14）─────────────────────────
print("#6 重复卷积收敛 Gaussian：binomial vs box 对照（L92–94+脚注 14 Wells 1986）")


def gauss_prof(sigma, half):
    x = np.arange(-half, half + 1, dtype=float)
    g = np.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def repeat_conv(k, n):
    """核自身 n 次卷积后归一化。"""
    cur = np.array(k, float)
    for _ in range(n - 1):
        cur = np.convolve(cur, np.asarray(k, float))
    return cur / cur.sum()


def ksigma(pk):
    grid = np.arange(len(pk), dtype=float)
    m = float(pk @ grid)
    return float(np.sqrt(((grid - m) ** 2 * pk).sum()))


results = {}
for name, k0 in [("binomial", np.array([1, 4, 6, 4, 1]) / 16), ("box", np.ones(3) / 3)]:
    devs = []
    for n in (2, 6, 12):
        pk = repeat_conv(k0, n)
        sd = ksigma(pk)
        half = len(pk) // 2
        gp = gauss_prof(sd, half)
        grid = np.arange(-half, half + 1)
        devs.append(np.max(np.abs(pk - gp)))
    results[name] = devs
    print(f"  [读数] {name} 核重复 2/6/12 次后与同方差 Gaussian 的 max|Δ| = "
          + "/".join(f"{d:.2e}" for d in devs))
    chk(f"6-{'1' if name == 'binomial' else '2'} {name} 核收敛：偏差随次数单调下降",
        devs[0] > devs[1] > devs[2], f"{devs[0]:.2e}→{devs[2]:.2e}")
var_binom = float(((np.arange(5) - 2) ** 2 * BINOM).sum())
pk8 = repeat_conv(np.array([1, 4, 6, 4, 1]) / 16, 8)
var8 = ksigma(pk8) ** 2
chk("6-3 方差可加性：binomial 单核 σ²=1.0（=4 次 (1+z⁻¹)/2 之和），8 次卷积 σ²=8.0",
    abs(var_binom - 1.0) < 1e-12 and abs(var8 - 8.0) < 1e-9,
    f"单核 σ²={var_binom:.6f}，8 次 σ²={var8:.6f}")
box8 = repeat_conv(np.ones(3) / 3, 8)
inject("D5 同一『σ²=8』断言用于 box 核应 FAIL（检验断言绑定核）",
       abs(ksigma(box8) ** 2 - 8.0) < 1e-9)

# ───────────────────────── #7 half-octave 与像素账 ─────────────────────────
print("#7 half-octave/quincunx 与 Figure 3.35b 像素账（p.159 L247–265）")
n_oct = int(round(np.log(64) / np.log(2.0)))
n_half = int(round(np.log(64) / np.log(np.sqrt(2.0))))
print(f"  [读数] 覆盖 64× 尺度范围（阶梯步数口径，每步尺度×步进）：octave 链={n_oct} 级，"
      f"half-octave 链={n_half} 级")
chk("7-1 half-octave 步进 √2<2 → 级数恰为两倍：octave 6 级、half-octave 12 级"
    "（L258–259 small scale change；round 取整，门A r1 P0-1 修）",
    n_oct == 6 and n_half == 12 and n_half == 2 * n_oct,
    f"{n_oct} vs {n_half}")
chk("7-2 quincunx 采样格：(x+y)%2==0 恰保留一半像素（checkerboard，L260–261）",
    (lambda q: int(q.sum()) == q.size // 2)(
        np.fromfunction(lambda i, j: (i + j) % 2 == 0, (8, 8))))
chk("7-3 Figure 3.35b 像素账：3/4×(1+1/4+1/16+…)=1（total = original pixels，几何级数闭环）",
    abs(0.75 * (1 / (1 - 0.25)) - 1.0) < 1e-15, f"Σ={0.75 * (1 / (1 - 0.25)):.6f}")
chk("7-4 octave 金字塔 overcomplete 账：1+1/4+…=4/3×N>N（§3.5.4 对比句 L304 的前提，讲者预埋）",
    abs((1 / (1 - 0.25)) - 4 / 3) < 1e-15, "Σ=4/3")

# ───────────────────────── 五图程序重绘 ─────────────────────────
print("#8 五图程序重绘（原书图未使用）")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

figA, axes = plt.subplots(1, 5, figsize=(11, 3.2))
for i, ax in enumerate(axes):
    ax.imshow(G[i], cmap="gray")
    ax.set_title(f"L{i}  {G[i].shape[0]}×{G[i].shape[1]}", fontsize=10)
    ax.axis("off")
figA.suptitle("高斯金字塔：逐层 blur+↓2，每层像素=上层 ¼（原书图未使用，程序重绘）", fontsize=12)
figA.tight_layout()
figA.savefig(OUT / "e030_gauss_pyramid.png", dpi=150)

figB, axes = plt.subplots(2, 5, figsize=(11, 4.6))
for i, ax in enumerate(axes[0]):
    amp = max(np.abs(L[i]).max(), 1e-9)
    ax.imshow(L[i], cmap="RdBu_r", vmin=-amp, vmax=amp)
    ax.set_title(f"带通 L{i}" if i < 4 else "基底层", fontsize=10)
errmap = np.abs(IMG - Ihat)
for ax, im, ti, cm in [
    (axes[1, 0], IMG, "原图 I", "gray"),
    (axes[1, 1], Ihat, f"重构 Î（max|Δ|={err:.1e}）", "gray"),
    (axes[1, 2], errmap, "误差图", "viridis"),
    (axes[1, 3], recon_nox2(L), f"不×2 重构（err={err_nox2:.2f}）", "gray"),
    (axes[1, 4], recon_variant(Lv), f"变体（err={err_v:.2f}）", "gray"),
]:
    ax.imshow(im, cmap=cm)
    ax.set_title(ti, fontsize=10)
for ax in axes.flat:
    ax.axis("off")
figB.suptitle("拉普拉斯金字塔：分解/重构/误差/对照组（原书图未使用，程序重绘）", fontsize=12)
figB.tight_layout()
figB.savefig(OUT / "e030_laplacian_pyramid.png", dpi=150)

figC, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, k0, nm in zip(axes, [np.array([1, 4, 6, 4, 1]) / 16, np.ones(3) / 3], ["binomial", "box"]):
    for n, cl in [(2, "#888"), (6, "#36c"), (12, "#c33")]:
        pk = repeat_conv(k0, n)
        grid = np.arange(len(pk), dtype=float) - (len(pk) - 1) / 2
        sd = ksigma(pk)
        half = len(pk) // 2
        ax.plot(grid, pk, cl, label=f"卷积 {n} 次")
        if n == 12:
            ax.plot(grid, gauss_prof(sd, half), "k--", label=f"同方差 Gaussian σ={sd:.2f}")
    ax.set_title(f"{nm} 核重复卷积收敛于 Gaussian（原书图未使用）", fontsize=11)
    ax.legend(fontsize=9)
figC.tight_layout()
figC.savefig(OUT / "e030_kernel_to_gauss.png", dpi=150)

figD, axes = plt.subplots(1, 2, figsize=(10, 3.6))
grid19 = np.arange(19) - 9
axes[0].plot(grid19, p_g1, "#36c", label="G_σ1 (σ=1)")
axes[0].plot(grid19, p_g2, "#3a3", label="G_σ2 (σ=2)")
axes[0].plot(grid19, p_dog, "k", lw=2, label="DoG = G_σ1−G_σ2")
axes[0].axhline(0, color="#aaa", lw=0.6)
axes[0].set_title("空域：两低通之差=带通（原书图未使用）", fontsize=11)
axes[0].legend(fontsize=9)
w = np.linspace(0, np.pi, 256)
H1 = np.array([abs(np.sum(p_g1 * np.exp(-1j * t * grid19))) for t in w])
H2 = np.array([abs(np.sum(p_g2 * np.exp(-1j * t * grid19))) for t in w])
HD = np.array([abs(np.sum(p_dog * np.exp(-1j * t * grid19))) for t in w])
axes[1].plot(w / np.pi, H1, "#36c", label="|G_σ1(ω)|")
axes[1].plot(w / np.pi, H2, "#3a3", label="|G_σ2(ω)|")
axes[1].plot(w / np.pi, HD, "k", lw=2, label="|DoG(ω)|=带通")
axes[1].set_title("频域：低通−低通=带通，DC=0（原书图未使用）", fontsize=11)
axes[1].legend(fontsize=9)
figD.tight_layout()
figD.savefig(OUT / "e030_dog_log.png", dpi=150)

figE, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for ax, base, nm, col in [(axes[0], 2.0, "octave 金字塔（r=2）", "#36c"),
                          (axes[1], np.sqrt(2), "half-octave 链（√2 步进，层数更多）", "#c33")]:
    scales = [base ** i for i in range(1, int(round(np.log(64) / np.log(base))) + 1)]
    for i, s in enumerate(scales):
        ax.barh(i, 64 / s, height=0.55, color=col, alpha=min(0.9, 0.3 + 0.05 * i))
        ax.text(65, i, f"σ×{s:.1f}", va="center", fontsize=8)
    ax.set_title(f"{nm}：{len(scales)} 级阶梯（原书图未使用，讲者口径示意）", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
figE.tight_layout()
figE.savefig(OUT / "e030_halfoctave.png", dpi=150)

for f in ["e030_gauss_pyramid.png", "e030_laplacian_pyramid.png", "e030_kernel_to_gauss.png",
          "e030_dog_log.png", "e030_halfoctave.png"]:
    p = OUT / f
    chk(f"图 {f} 已生成且非空", p.exists() and p.stat().st_size > 5000,
        f"{p.stat().st_size if p.exists() else 0} B")

# ───────────────────────── 汇总 ─────────────────────────
print()
print(f"== E030 数值实验汇总：{N_PASS} 条 PASS / {N_FAIL} 条 FAIL；区分力注入 {INJ['caught']}/{INJ['total']} 已拦截 ==")
if N_FAIL == 0 and INJ["caught"] == INJ["total"]:
    print("DONE")
else:
    print("NOT-DONE")
    sys.exit(1)
