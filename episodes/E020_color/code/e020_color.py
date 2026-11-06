"""
E020 · 颜色从哪来：三原色、色度图与压缩收官（§2.3.2 Color 原书第 87–98 页 +
§2.3.3 Compression 第 98–101 页（r5 门A r2 P1-3 修订：p.101 含式 2.119 RMS/式 2.120 PSNR）；
§2.3 开头与 §2.3.1 采样混叠 E008 已讲，回指不重讲）

数值实验：把 CIE XYZ 变换（式 2.104）、色度坐标（式 2.105）、L*a*b*（式 2.106–2.108）、
BT.709（式 2.109）、Y′601/709（式 2.113–2.114）、JPEG YCbCr（式 2.116）、
gamma 压缩（式 2.111–2.112）与 MSE/PSNR（式 2.118 p.100/式 2.120 p.101）全部算一遍。

断言（27 条数据断言，分 6 组（#1 含 #1d' 批量回归 1 条、#3 含 #3a' 小 t 线性段 1 条、
#6 含 #6e Bayer 布局 1 条）+ 6 项常数核对 C1–C6（不计数，不调被测函数）+ 1 项同步
+ 1 项区分力自检不计数；
错误实现类断言一律由 #D 注入承担，不在正式计数里复写判据——E019 门A round3
新P1-2 惯例）：
  #1  XYZ 与色度（p.90 式 2.104/2.105；5 条，含 #1d' 批量回归）：常用形式矩阵（省去 1/0.17697）三行
      和均为 1 ⇒ RGB(1,1,1)→XYZ=(1,1,1)（Y=1，「second row adds up to one」）；
      官方形式（×1/0.17697）纯红 Y=1（「normalized so that the Y value corresponding
      to pure red is 1」）；色度坐标 x+y+z=1（任意样本）；等能白 E 的色度=(1/3,1/3)
      （矩阵转置错误实现由 #D②注入拦截；除以 X 而非 X+Y+Z 由 #D⑦拦截）
  #2  匹配函数与负红（p.89 Figure 2.29 口径；4 条）：高斯近似 x̄ȳz̄ 全程非负
      （「all three spectra now have only positive values」，p.90）；ȳ 峰位在
      550–560nm（「the ȳ(λ) curve matches that of the luminance」，p.90）；r̄ḡb̄
      （由官方矩阵逆变换回 RGB 基）在蓝绿区 500–540nm 存在负值——负红现象
      （「a negative amount of red light has to be added」，p.89）；三原色锚点：
      r̄(700)、ḡ(546.1)、b̄(435.8) 分别为各自曲线在对应原色波长附近的显著峰
      （匹配函数整体为负由 #D②拦截——转置矩阵使 r̄=x̄ 恒非负）
  #3  L*a*b*（pp.91–92 式 2.106–2.108；5 条）：f(t) 在分段点 t=δ³ 处两分支连续
      （δ=6/29，「a finite-slope approximation to the cube root」）+#3a' 小 t 区取
      线性段值（区分漏分段实现）；白点 L*=116·f(1)=116
      且 a*=b*=0（式 2.106 为 116f(Y/Yn) 无 −16，忠实原书公式；原书文字称
      「0...100 scale」p.91——标准 CIELAB 以 −16 修正至 100，讲者按口径差披露）；
      L*(Y=Yn/4)=116·(1/4)^(1/3)=73.1；L* 随 Y 单调递增
      （116 错系数由 #D⑧拦截；漏分段由 #D③拦截）
  #4  BT.709 与 luma（p.92 式 2.109、p.96 式 2.113/2.114；4 条）：式 2.109 第二行和
      =1.000000（白点 (1,1,1)→Y=1）；Y′601 单通道直读 (1,0,0)→0.299、
      (0,1,0)→0.587、(0,0,1)→0.114；灰色 (0.5,0.5,0.5) 的 Y′601=Y′709=0.5
      （行和 1 的推论）；Y′709 对绿通道系数最大（0.7154>0.2125>0.0721）
      （601/709 系数交换错误实现由 #D①注入拦截）
  #5  JPEG YCbCr 与作业①算例（p.97 式 2.116；4 条）：RGB(255,128,64) →
      Y′=158.68、Cb=74.57、Cr=196.70（容差 ±0.15；取整 159/75/197 为作业①参考答案，
      讲者算例口径）；往返 YCbCr→RGB 重建误差 <0.5；灰色 R=G=B ⇒ Cb=Cr=128
      （色差为零）；Cb 随蓝通道单调递增、Cr 随红通道单调递增
      （+128 偏移丢失由 #D⑥拦截）
  #6  gamma 压缩与 MSE/PSNR（pp.94–95 式 2.111/2.112、p.100 式 2.118 与 p.101 式 2.120；4 条
      + #6e Bayer 布局 1 条）：
      (V^(1/2.2))^(2.2)=V（0–1 均匀采样 401 点，容差 1e-9）；V=0.5 经 gamma 2.2
      后 0.217（显式下凸，γ≠1 由 #D④拦截）；噪声压暗区（Figure 2.32 定量版）：
      传输加固定绝对噪声 δ=0.02，与线性传输相比，gamma 通道把暗区 Y=0.1 的
      接收端绝对误差从 0.020 压到 0.013（相对误差 20%→12.9%，同样压小）——
      「noise … squashed in the dark regions」（p.95）；MSE/PSNR：常量图偏移 e 的
      MSE=e²、PSNR=10·log10(255²/e²)（式 2.120，原书第 101 页——r5 门A r2 P1-3 核实归原书，容差 1e-6）
  #C  常数核对（6 项 C1–C6，不计数、无区分力）：C1 CIE 三原色波长 700.0/546.1/435.8nm
      （p.89 源码常数复核）；C2 γ≈2.2 与 1/γ=0.45（pp.94–95，原书版式为分数 1/γ）；
      C3 Y′601/709 系数 0.299/0.587/0.114 与 0.2125/0.7154/0.0721（p.96）；
      C4 式 2.116 矩阵亮度行与 Y′601 一致（JPEG 亮度行=Y′601 的口径互证）；
      C5 色度图 xy 锚点表防错（CIE 官方表誊录校验）；C6 三原色 xy 与公开标准值一致
  #S  同步检查：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步
      （重跑至一致；stdout 的 md5 供门C 冷启动四方比对——E019 惯例）
  #D  区分力自检（不计数）：8 种错误实现 monkeypatch 进模块全局
      （① Y′601 系数交换 0.299↔0.114、② 式 2.104 矩阵转置、③ f(t) 漏分段、
      ④ gamma 恒等 γ=1、⑤ Bayer pattern 错位 GRBG→自定义错排、⑥ YCbCr +128
      偏移丢失、⑦ 色度除以 X 而非 X+Y+Z、⑧ L* 系数 116→100），
      重跑真实 assert_* 函数，验证每例至少一条断言实际翻转为 FAIL——
      区分力由「注入 → 真实断言 FAIL」直接证明（E019 门A round2/round4 惯例）

图（8 张：3 张程序重绘原书图概念 + 5 张讲者自制图表；原书图均未使用）：
  e020_fig2_28.png   Figure 2.28（p.88）重绘：加色 RGB 三圆（青品黄白）与
                     减色 CMY 三圆（红绿蓝黑）
  e020_cones.png     讲者示意（非原书图）：S/M/L 三种视锥细胞敏感曲线（高斯形态，
                     呼应 p.88 tri-stimulus）
  e020_fig2_29.png   Figure 2.29（p.89）重绘：(a) r̄ḡb̄ 匹配函数（含蓝绿区负红）
                     (b) x̄ȳz̄（非负；匹配曲线为讲者多段高斯近似合成，非 CIE 原表）
  e020_fig2_30.png   Figure 2.30a（pp.90–91）重绘：色度图（光谱轨迹+紫线+E 白点+
                     RGB 原色三角；轨迹锚定 CIE 1931 官方 xy 表、按波长渐变着色）
  e020_macadam.png   Figure 2.30b（p.91）概念重绘：色度图上叠等可辨椭圆——
                     位置/大小/方向为讲者示意（非 MacAdam 原始数据），呼应
                     「color regions of equal discriminability … basis of the Lab」
  e020_fig2_31.png   Figure 2.31（p.94）重绘：(a) Bayer CFA 布局 (b) 插值后像素
                     （大写=实测、小写=猜值，rGb/Rgb/rgB 图案）
  e020_fig2_32.png   Figure 2.32（p.95）重绘：gamma 压缩 (a) Y→Y′ 曲线
                     (b) 传输噪声在暗区被压扁（数值演示）
  e020_fig2_34.png   呼应 Figure 2.34（p.100）：同一程序生成图以 JPEG quality
                     90/25/8 实压三联对比（块效应随质量下降增强；原书图为
                     实拍照片，本集用程序生成图替代，未使用原书图）

运行：`python3 code/e020_color.py`
期望输出 `汇总：27/27 数据断言 PASS · 常数核对 6/6 · 区分力自检 8/8 拦截 · 同步检查 PASS —— 全部通过`。
"""
import hashlib
import io
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse
from PIL import Image

# 中文字体（同 E014–E019：macOS 优先 Hiragino Sans GB）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
GREEN, PURPLE, RED = "#2a7a4a", "#6b4fa0", "#b03030"
RESULTS = []
CONST_RESULTS = []
EP = pathlib.Path(__file__).resolve().parent.parent          # episodes/E020_color
FIGDIR = EP / "code" / "outputs" / "figures"
SLIDEFIG = EP / "slides" / "figures"


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))


def verify_const(name, ok, detail=""):
    """常数核对（不计数）：纯数值换算/文献常数复核，不调用被测函数、无区分力，
    故不计入数据断言总数（E019 门A round3 新P1-2 惯例）。"""
    CONST_RESULTS.append((name, bool(ok)))
    print(f"[const-{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------- 被测实现（可注入错误）
M_XYZ_COMMON = np.array([[0.49, 0.31, 0.20],
                         [0.17697, 0.81240, 0.01063],
                         [0.00, 0.01, 0.99]])           # p.90 式(2.104) 显示矩阵
K_OFFICIAL = 1.0 / 0.17697                              # 官方形式的前导因子

# p.89：CIE 三原色波长
LAM_R, LAM_G, LAM_B = 700.0, 546.1, 435.8

# pp.94–95：gamma
GAMMA = 2.2
GAMMA1 = 0.45

# p.96：luma 系数
Y601 = (0.299, 0.587, 0.114)
Y709 = (0.2125, 0.7154, 0.0721)

# p.91：L*a*b*
DELTA = 6.0 / 29.0
LSTAR_K = 116.0


def rgb_to_xyz_common(rgb):
    """式(2.104) 常用形式（省去前导分数，p.90）：第二行和为 1，(1,1,1)→Y=1。"""
    return M_XYZ_COMMON @ np.asarray(rgb, dtype=float)


def rgb_to_xyz_official(rgb):
    """式(2.104) 官方形式：×1/0.17697，纯红的 Y=1。"""
    return K_OFFICIAL * (M_XYZ_COMMON @ np.asarray(rgb, dtype=float))


def chromaticity(xyz):
    """式(2.105)（p.90）：x=X/(X+Y+Z) 等，三者和为 1。
    支持 (3,) 向量与 (3,N) 批量（按列归一化；axis=0 对向量退化为标量和）。"""
    xyz = np.asarray(xyz, dtype=float)
    s = xyz.sum(axis=0)
    return xyz / s


def f_lab(t):
    """式(2.107)（p.91）：t^(1/3)（t>δ³），否则 t/(3δ²)+2δ/3；δ=6/29。"""
    t = np.asarray(t, dtype=float)
    d3 = DELTA ** 3
    out = np.where(t > d3, np.cbrt(np.maximum(t, 1e-300)), t / (3 * DELTA ** 2) + 2 * DELTA / 3)
    return out if out.ndim else float(out)


def lab_from_xyz(xyz, white=(0.95047, 1.0, 1.08883)):
    """式(2.106)/(2.108)（pp.91–92）：L*=116f(Y/Yn)、a*=500[f(X/Xn)−f(Y/Yn)]、
    b*=200[f(Y/Yn)−f(Z/Zn)]。white 取 D65 参考白（讲者按：p.97 口径 BT.709 参考
    白为 D65，XYZ 白点用 ASTM D65 常用值）。"""
    w = np.asarray(white, dtype=float)
    x, y, z = np.asarray(xyz, dtype=float) / w
    fx, fy, fz = f_lab(x), f_lab(y), f_lab(z)
    L = LSTAR_K * fy
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return L, a, b


def y601(rgb):
    """式(2.113)（p.96）：Y′601 = 0.299R′+0.587G′+0.114B′。"""
    r, g, b = Y601
    R, G, B = np.asarray(rgb, dtype=float)
    return r * R + g * G + b * B


def y709(rgb):
    """式(2.114)（p.96）：Y′709 = 0.2125R′+0.7154G′+0.0721B′。"""
    r, g, b = Y709
    R, G, B = np.asarray(rgb, dtype=float)
    return r * R + g * G + b * B


M_YCBCR = np.array([[0.299, 0.587, 0.114],
                    [-0.168736, -0.331264, 0.5],
                    [0.5, -0.418688, -0.081312]])       # p.97 式(2.116)
YCBCR_OFFSET = np.array([0.0, 128.0, 128.0])


def rgb_to_ycbcr(rgb):
    """式(2.116)（p.97）：JPEG 全 8 位范围，Cb/Cr 加 128 偏移。"""
    return M_YCBCR @ np.asarray(rgb, dtype=float) + YCBCR_OFFSET


def ycbcr_to_rgb(ycc):
    """式(2.116) 逆变换（讲者按：矩阵可逆，往返重建）。"""
    return np.linalg.inv(M_YCBCR) @ (np.asarray(ycc, dtype=float) - YCBCR_OFFSET)


def gamma_encode(v, g=None):
    """式(2.112)（p.95）：Y′ = Y^(1/γ)（发送端反向预映射）。
    g 缺省运行时读全局 GAMMA（保证 #D④ patch 全局可生效——默认参数会在 def 时
    绑定，patch 不进函数，E020 门A 前自检发现）。"""
    g = GAMMA if g is None else g
    return np.power(np.asarray(v, dtype=float), 1.0 / g)


def gamma_decode(v, g=None):
    """式(2.111)（pp.94–95）：B = V^γ（接收端）；g 同上运行时读全局。"""
    g = GAMMA if g is None else g
    return np.power(np.asarray(v, dtype=float), g)


def mse_psnr(img, ref, peak=255.0):
    """式(2.118)（p.100）：MSE=(1/n)Σ[I(x)−Î(x)]²；PSNR=10·log10(peak²/MSE)。"""
    img = np.asarray(img, dtype=float)
    ref = np.asarray(ref, dtype=float)
    mse = ((img - ref) ** 2).mean()
    psnr = 10.0 * np.log10(peak ** 2 / mse) if mse > 0 else float("inf")
    return mse, psnr


def bayer_pattern(h=4, w=4):
    """Bayer RGGB/GRBG 布局（p.93）：左上 G、棋盘。返回 'G'/'R'/'B' 字符阵。"""
    pat = np.empty((h, w), dtype="<U1")
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == 0:
                pat[i, j] = "G"
            elif i % 2 == 0:
                pat[i, j] = "R"      # 偶行奇列 = R
            else:
                pat[i, j] = "B"      # 奇行奇列 = B
    return pat


# --------------------------------------------------------------- CIE 匹配函数（讲者多段高斯近似）
LAM = np.linspace(380.0, 800.0, 841)                    # p.90–91：380nm→800nm 扫描


def gsig(x, mu, tau):
    return np.exp(-((x - mu) / tau) ** 2)


def cmf_xyz(lam=None):
    """x̄ȳz̄ 高斯近似（讲者合成，Wyman 型多段高斯形态；图内标注近似口径）。
    仅用于形态演示与色度图绘制，不冒充 CIE 官方数值表。"""
    lam = LAM if lam is None else np.asarray(lam, dtype=float)
    # 讲者多段高斯近似（对齐 CIE 1931 大形：x̄ 蓝副峰 444+红尾 680、ȳ 厚红尾、z̄ 高蓝峰 445）
    xb = (0.36 * gsig(lam, 444.0, 26.0) + 1.05 * gsig(lam, 600.0, 40.0)
          + 0.016 * gsig(lam, 680.0, 45.0) - 0.04 * gsig(lam, 508.0, 26.0))
    yb = 1.0 * gsig(lam, 555.0, 44.0) + 0.10 * gsig(lam, 640.0, 35.0)
    zb = 1.45 * gsig(lam, 445.0, 20.0) + 0.55 * gsig(lam, 468.0, 45.0)
    return np.stack([np.clip(xb, 0, None), np.clip(yb, 0, None), np.clip(zb, 0, None)])


def cmf_rgb(lam=None):
    """r̄ḡb̄：由 x̄ȳz̄ 经式(2.104) 官方形式矩阵的逆变换回到 RGB 原色基
    （负红由逆变换自然涌现，非手工合成）。"""
    lam = LAM if lam is None else np.asarray(lam, dtype=float)
    xyz = cmf_xyz(lam)
    return np.linalg.inv(K_OFFICIAL * M_XYZ_COMMON) @ xyz


def wavelength_color(lam):
    """380–800nm 单色光的近似显示色（讲者示意，用于色度图轨迹渐变着色）。"""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    # 分段插值：紫→蓝→青→绿→黄→橙→红
    stops = [(380, (0.45, 0.0, 0.65)), (430, (0.25, 0.1, 0.9)), (480, (0.05, 0.45, 0.95)),
             (520, (0.05, 0.85, 0.35)), (560, (0.85, 0.9, 0.05)), (600, (0.98, 0.65, 0.0)),
             (650, (0.95, 0.2, 0.1)), (700, (0.8, 0.05, 0.05)), (800, (0.6, 0.0, 0.0))]
    xs = np.array([s[0] for s in stops], dtype=float)
    cs = np.array([s[1] for s in stops], dtype=float)
    out = np.empty((lam.size, 3))
    for k in range(3):
        out[:, k] = np.interp(lam, xs, cs[:, k])
    return np.clip(out, 0, 1)


# --------------------------------------------------------------- 断言
def assert_xyz_chromaticity():
    # 常用形式：三行和均为 1，(1,1,1)→XYZ=(1,1,1)
    row_sums = M_XYZ_COMMON.sum(axis=1)
    verify("#1a 常用形式矩阵三行和均为 1（p.90「second row adds up to one」）",
           np.allclose(row_sums, 1.0, atol=1e-9), f"行和={row_sums.round(6).tolist()}")
    xyz_w = rgb_to_xyz_common((1, 1, 1))
    verify("#1b RGB(1,1,1) → XYZ=(1,1,1)，Y=1（p.90「maps to a Y value of 1」）",
           np.allclose(xyz_w, 1.0, atol=1e-9), f"XYZ={xyz_w.round(6).tolist()}")
    # 官方形式：纯红 Y=1
    xyz_r = rgb_to_xyz_official((1, 0, 0))
    verify("#1c 官方形式纯红 Y=1（p.90「Y value corresponding to pure red is 1」）",
           abs(xyz_r[1] - 1.0) < 1e-9, f"Y_red={xyz_r[1]:.6f}")
    # 色度和为 1；E 白点=(1/3,1/3)
    s = rgb_to_xyz_common((0.3, 0.7, 0.4))
    c = chromaticity(s)
    verify("#1d 色度坐标 x+y+z=1 且 E=(1/3,1/3)（p.90 式 2.105）",
           abs(c.sum() - 1.0) < 1e-12 and np.allclose(chromaticity((1, 1, 1))[:2], 1 / 3, atol=1e-12),
           f"x+y+z={c.sum():.12f}，E=({chromaticity((1, 1, 1))[0]:.6f},{chromaticity((1, 1, 1))[1]:.6f})")
    batch = chromaticity(cmf_xyz(np.linspace(400.0, 680.0, 281)))
    verify("#1d' 批量色度：(3,N) 输入逐列和为 1（回归：批量按列归一化）",
           batch.shape == (3, 281) and np.allclose(batch.sum(axis=0), 1.0, atol=1e-12),
           f"shape={batch.shape}，列和范围=[{batch.sum(axis=0).min():.12f},{batch.sum(axis=0).max():.12f}]")


def assert_negative_red():
    xb_yb_zb = cmf_xyz()
    lam_peak_y = LAM[np.argmax(xb_yb_zb[1])]
    verify("#2a 高斯近似 x̄ȳz̄ 全程非负（p.90「now have only positive values」）",
           all((c >= -1e-12).all() for c in xb_yb_zb))
    verify("#2b ȳ 峰位在 550–560nm（p.90「ȳ(λ) curve matches … luminance」）",
           550 <= lam_peak_y <= 560, f"ȳ峰={lam_peak_y:.0f}nm")
    rgb_curves = cmf_rgb()
    seg = (LAM >= 500) & (LAM <= 540)
    verify("#2c r̄ 在蓝绿区 500–540nm 存在负值——负红（p.89「negative amount of red」）",
           (rgb_curves[0][seg] < -1e-3).any(),
           f"r̄ 最小={rgb_curves[0].min():.3f}（λ={LAM[np.argmin(rgb_curves[0])]:.0f}nm）")
    r, g, b = rgb_curves
    pk_r = LAM[np.argmax(r)]; pk_g = LAM[np.argmax(g)]; pk_b = LAM[np.argmax(b)]
    verify("#2d 主峰位与三原色次序一致：r̄ 红区/ḡ 绿区/b̄ 蓝区（p.89 三原色 700.0/546.1/435.8 口径）",
           590 <= pk_r <= 650 and 520 <= pk_g <= 560 and 425 <= pk_b <= 470 and pk_b < pk_g < pk_r,
           f"r̄峰={pk_r:.0f} ḡ峰={pk_g:.0f} b̄峰={pk_b:.0f}nm")


def assert_lab():
    d3 = DELTA ** 3
    lo, hi = f_lab(d3 - 1e-9), f_lab(d3 + 1e-9)
    verify("#3a f(t) 在分段点 δ³ 处连续（p.91 式 2.107，δ=6/29）",
           abs(lo - hi) < 1e-6, f"δ³={d3:.6f}，左右={lo:.8f}/{hi:.8f}")
    t_small = d3 / 2
    lin_val = t_small / (3 * DELTA ** 2) + 2 * DELTA / 3
    verify("#3a' 小 t 区取线性段值：f(δ³/2)=线性段公式（≠t^(1/3)，区分漏分段实现）",
           abs(f_lab(t_small) - lin_val) < 1e-12,
           f"f({t_small:.6f})={float(f_lab(t_small)):.8f} 线性={lin_val:.8f} cbrt={t_small**(1/3):.8f}")
    L100, a0, b0 = lab_from_xyz((0.95047, 1.0, 1.08883))
    verify("#3b 白点处 L*=116·f(1)=116 且 a*=b*=0（式 2.106/2.108 直算）",
           abs(L100 - 116.0) < 1e-9 and abs(a0) < 1e-9 and abs(b0) < 1e-9,
           f"L*={L100:.4f} a*={a0:.2e} b*={b0:.2e}"
           "（讲者按：原书式 2.106 为 116f(Y/Yn) 无 −16，白点直算 116；"
           "原书文字称 0..100 scale（p.91）——标准 CIELAB 以 −16 修正至 100，"
           "本集断言忠实原书公式并披露该口径差）")
    Lq, _, _ = lab_from_xyz((0.95047, 0.25, 1.08883))
    verify("#3c Y 减到 1/4 时 L*=73.1（=116·(1/4)^(1/3)，0–100 标尺）",
           abs(Lq - 116.0 * (0.25 ** (1 / 3))) < 1e-9, f"L*={Lq:.4f}")
    ys = np.linspace(0.01, 1.0, 50)
    Ls = np.array([lab_from_xyz((0.95047, y, 1.08883))[0] for y in ys])
    verify("#3d L* 随 Y 单调递增（p.91 lightness 语义）", (np.diff(Ls) > 0).all(),
           f"min ΔL={np.diff(Ls).min():.4f}")


def assert_bt709_luma():
    y_row = np.array([0.212671, 0.715160, 0.072169])
    verify("#4a 式 2.109 第二行和=1.000000（p.92 矩阵；白点 Y=1）",
           abs(y_row.sum() - 1.0) < 1e-9, f"和={y_row.sum():.6f}")
    verify("#4b Y′601 单通道直读：R→0.299、G→0.587、B→0.114（p.96 式 2.113）",
           abs(y601((1, 0, 0)) - 0.299) < 1e-12 and abs(y601((0, 1, 0)) - 0.587) < 1e-12
           and abs(y601((0, 0, 1)) - 0.114) < 1e-12,
           f"{y601((1,0,0)):.3f}/{y601((0,1,0)):.3f}/{y601((0,0,1)):.3f}")
    verify("#4c 灰 (0.5,0.5,0.5)：Y′601=Y′709=0.5（行和 1 的推论）",
           abs(y601((0.5, 0.5, 0.5)) - 0.5) < 1e-12 and abs(y709((0.5, 0.5, 0.5)) - 0.5) < 1e-12,
           f"Y601={y601((0.5,0.5,0.5)):.4f} Y709={y709((0.5,0.5,0.5)):.4f}")
    verify("#4d Y′709 绿通道系数最大（0.7154>0.2125>0.0721，p.96 式 2.114）",
           Y709[1] > Y709[0] > Y709[2], f"{Y709}")


def assert_ycbcr_homework():
    rgb = np.array([255.0, 128.0, 64.0])
    y, cb, cr = rgb_to_ycbcr(rgb)
    verify("#5a 作业①算例 RGB(255,128,64)→Y′=158.68、Cb=74.57、Cr=196.70（±0.15）",
           abs(y - 158.677) < 0.15 and abs(cb - 74.570) < 0.15 and abs(cr - 196.704) < 0.15,
           f"Y′={y:.2f} Cb={cb:.2f} Cr={cr:.2f}（取整 159/75/197）")
    back = ycbcr_to_rgb((y, cb, cr))
    verify("#5b YCbCr 往返重建误差 <0.5（式 2.116 可逆）",
           np.abs(back - rgb).max() < 0.5, f"最大误差={np.abs(back-rgb).max():.4f}")
    y_g, cb_g, cr_g = rgb_to_ycbcr((100.0, 100.0, 100.0))
    verify("#5c 灰色 R=G=B ⇒ Cb=Cr=128（色差为零，p.97 偏移口径）",
           abs(cb_g - 128) < 1e-9 and abs(cr_g - 128) < 1e-9, f"Cb={cb_g:.4f} Cr={cr_g:.4f}")
    cbs = [rgb_to_ycbcr((128.0, 128.0, bb))[1] for bb in (32.0, 128.0, 224.0)]
    crs = [rgb_to_ycbcr((rr, 128.0, 128.0))[2] for rr in (32.0, 128.0, 224.0)]
    verify("#5d Cb 随蓝通道、Cr 随红通道单调递增（色差语义）",
           cbs[0] < cbs[1] < cbs[2] and crs[0] < crs[1] < crs[2],
           f"Cb={cbs[0]:.1f}→{cbs[2]:.1f}，Cr={crs[0]:.1f}→{crs[2]:.1f}")


def assert_gamma_psnr():
    vs = np.linspace(0.0, 1.0, 401)
    err = np.abs(gamma_decode(gamma_encode(vs)) - vs).max()
    verify("#6a (V^(1/2.2))^(2.2)=V（0–1 采样 401 点，容差 1e-9；pp.94–95 式 2.111/2.112）",
           err < 1e-9, f"最大误差={err:.2e}")
    mid = float(gamma_decode(0.5))
    verify("#6b V=0.5 经 γ=2.2 后 B≈0.217（显式下凸，式 2.111）",
           abs(mid - 0.5 ** 2.2) < 1e-12 and mid < 0.5, f"B={mid:.4f}")
    # 噪声压暗区（Figure 2.32 定量版）：δ=0.02，暗区 Y=0.1
    d = 0.02
    y_dark, y_bright = 0.1, 0.9
    lin_dark_err, lin_bright_err = d, d
    enc = gamma_encode(np.array([y_dark, y_bright]))
    dec = gamma_decode(enc + d)
    g_dark_err, g_bright_err = abs(dec[0] - y_dark), abs(dec[1] - y_bright)
    verify("#6c gamma 通道压暗区误差：Y=0.1 绝对 0.020→0.013、相对 20%→12.9%（p.95 图 2.32 定量版）",
           g_dark_err < lin_dark_err and g_dark_err / y_dark < lin_dark_err / y_dark,
           f"绝对 {g_dark_err:.4f}<{lin_dark_err}，相对 {g_dark_err/y_dark*100:.1f}%<{lin_dark_err/y_dark*100:.0f}%")
    mse, psnr = mse_psnr(np.full((4, 4), 200.0), np.full((4, 4), 200.0 - 5.0))
    expect = 10 * np.log10(255 ** 2 / 25.0)
    verify("#6d 常量偏移 5 的 MSE=25、PSNR=34.15dB（p.100 式 2.118）",
           abs(mse - 25.0) < 1e-9 and abs(psnr - expect) < 1e-9,
           f"MSE={mse:.2f} PSNR={psnr:.4f}dB")


def assert_bayer_pattern():
    pat = bayer_pattern()
    verify("#6e Bayer 布局：(0,0)=G、(0,1)=R、(1,0)=B、绿占一半（p.93 Figure 2.31a）",
           pat[0, 0] == "G" and pat[0, 1] == "R" and pat[1, 0] == "B"
           and (pat == "G").sum() == pat.size // 2,
           f"绿 {int((pat=='G').sum())}/{pat.size}")


# --------------------------------------------------------------- 区分力自检（#D）
def assert_discrimination():
    """8 种错误实现 monkeypatch 进模块全局，重跑真实 assert_*，每例至少一条断言 FAIL。"""
    print("--- 区分力自检：注入 8 种错误实现（不计数）---")
    cases = [
        ("① Y′601 系数交换 0.299↔0.114", "Y601", (0.114, 0.587, 0.299),
         [assert_bt709_luma, assert_ycbcr_homework]),
        ("② 式 2.104 矩阵转置", "M_XYZ_COMMON", None,
         [assert_xyz_chromaticity, assert_negative_red]),
        ("③ f(t) 漏分段（恒 t^(1/3)）", "f_lab", None,
         [assert_lab]),
        ("④ gamma 恒等 γ=1", "GAMMA", 1.0,
         [assert_gamma_psnr]),
        ("⑤ Bayer 错位（奇行 R/B 互换）", "bayer_pattern", None,
         [assert_bayer_pattern]),
        ("⑥ YCbCr +128 偏移丢失", "YCBCR_OFFSET", np.zeros(3),
         [assert_ycbcr_homework]),
        ("⑦ 色度除以 X 而非 X+Y+Z", "chromaticity", None,
         [assert_xyz_chromaticity]),
        ("⑧ L* 系数 116→100", "LSTAR_K", 100.0,
         [assert_lab]),
    ]
    ok_all = True
    for title, target, value, runners in cases:
        saved = globals()[target]
        n_before = len(RESULTS)          # 每个 case 一次：注入期记录全部不进正式计数
        try:
            if title.startswith("②"):
                globals()[target] = saved.T.copy()          # 转置注入
            elif title.startswith("③"):
                globals()[target] = lambda t: float(np.cbrt(np.asarray(t, dtype=float)))
            elif title.startswith("⑤"):
                def bad_pattern(h=4, w=4):
                    p = np.empty((h, w), dtype="<U1")
                    for i in range(h):
                        for j in range(w):
                            if (i + j) % 2 == 0:
                                p[i, j] = "G"
                            elif i % 2 == 1:               # 错排：奇行奇列= R、偶行奇列= B
                                p[i, j] = "R"
                            else:
                                p[i, j] = "B"
                    return p
                globals()[target] = bad_pattern
            elif title.startswith("⑦"):
                globals()[target] = lambda xyz: np.asarray(xyz, dtype=float) / np.asarray(xyz, dtype=float)[0]
            else:
                globals()[target] = value
            n_fail = 0
            for run in runners:
                try:
                    run()
                except Exception:
                    pass
            n_fail = sum(1 for _, okk in RESULTS[n_before:] if not okk)
            del RESULTS[n_before:]          # 注入期断言记录不进正式计数
            verdict = n_fail > 0
            ok_all &= verdict
            print(f"[{'PASS' if verdict else 'FAIL'}] 注入{title} → {n_fail} 条断言翻转 FAIL"
                  + ("" if verdict else "（未检出——区分力不足！）"))
        finally:
            globals()[target] = saved
        # 清掉注入期间产生的断言记录
    return ok_all


# --------------------------------------------------------------- 图
def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[fig] {name}  {out.stat().st_size}B")


def fig_additive_subtractive():
    """Figure 2.28（p.88）重绘：加色 RGB / 减色 CMY（原书图未使用）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), facecolor=PAPER)
    n = 400
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    centers = {"R": (0.35, 0.62), "G": (0.65, 0.62), "B": (0.5, 0.38)}
    rads = 0.24
    mask = {k: ((xs - cx) ** 2 + (ys - cy) ** 2) < rads ** 2 for k, (cx, cy) in centers.items()}
    mR, mG, mB = mask["R"], mask["G"], mask["B"]
    # 加色：黑底，R+G=Y、R+B=M、G+B=C、RGB=W
    ax = axes[0]
    img = np.zeros((n, n, 3))
    img[mR, 0] = 1; img[mG, 1] = 1; img[mB, 2] = 1
    ax.imshow(img, extent=[0, 1, 0, 1])
    ax.set_title("加色 additive：红+绿=黄…全叠=白（原书图未使用·程序重绘）",
                 fontsize=13, color=INK)
    # 减色：白底，颜料吸收 → C+M=B、C+Y=G、M+Y=R、CMY=K
    ax = axes[1]
    img = np.ones((n, n, 3))
    onlyC = mG & ~mR & ~mB; onlyM = mR & ~mG & ~mB; onlyY = mB & ~mR & ~mG
    CM = mG & mR & ~mB; CY = mG & mB & ~mR; MY = mR & mB & ~mG; CMY = mR & mG & mB
    img[onlyC] = (0, 1, 1); img[onlyM] = (1, 0, 1); img[onlyY] = (1, 1, 0)
    img[CM] = (0, 0, 1); img[CY] = (0, 1, 0); img[MY] = (1, 0, 0); img[CMY] = (0, 0, 0)
    ax.imshow(img, extent=[0, 1, 0, 1])
    ax.set_title("减色 subtractive：青+黄=绿…全叠=黑（原书图未使用·程序重绘）",
                 fontsize=13, color=INK)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "e020_fig2_28.png")


def fig_cones():
    """讲者示意：S/M/L 视锥敏感曲线（非原书图，呼应 p.88 tri-stimulus）。"""
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=PAPER)
    lam = np.linspace(400, 700, 301)
    S = 0.95 * gsig(lam, 440, 30)
    M = 0.95 * gsig(lam, 535, 42)
    L = 0.95 * gsig(lam, 565, 45)
    ax.plot(lam, S, color=BLUE, lw=3, label="S（短波·蓝）")
    ax.plot(lam, M, color=GREEN, lw=3, label="M（中波·绿）")
    ax.plot(lam, L, color=RED, lw=3, label="L（长波·红）")
    ax.set_xlabel("波长 λ（nm）", fontsize=12, color=INK)
    ax.set_ylabel("相对敏感度（示意形态）", fontsize=12, color=INK)
    ax.set_title("三种视锥细胞 = 眼睛只问三个问题（示意形态，原书图未使用）",
                 fontsize=13.5, color=INK)
    ax.legend(fontsize=12, frameon=False)
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color("#c9c2b2")
    fig.tight_layout()
    _save(fig, "e020_cones.png")


def fig_matching_functions():
    """Figure 2.29（p.89）重绘：r̄ḡb̄（负红）+ x̄ȳz̄（原书图未使用；曲线为讲者多段
    高斯近似合成，非 CIE 官方数值表）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), facecolor=PAPER)
    rgb_c = cmf_rgb()
    xyz_c = cmf_xyz()
    ax = axes[0]
    ax.axhline(0, color="#9a938344", lw=1)
    ax.plot(LAM, rgb_c[0], color=RED, lw=2.4, label="$\\bar{r}$")
    ax.plot(LAM, rgb_c[1], color=GREEN, lw=2.4, label="$\\bar{g}$")
    ax.plot(LAM, rgb_c[2], color=BLUE, lw=2.4, label="$\\bar{b}$")
    neg = LAM[rgb_c[0] < -1e-3]
    ax.axvspan(neg.min(), neg.max(), color=ORANGE, alpha=0.16)
    ax.annotate("蓝绿区负红\n（红要加到对侧）", xy=(neg.min(), rgb_c[0].min()),
                xytext=(560, rgb_c[0].min() * 0.9), fontsize=14,
                color=BROWN, ha="left")
    for lam_v, c, lab in [(LAM_R, RED, "700.0"), (LAM_G, GREEN, "546.1"), (LAM_B, BLUE, "435.8")]:
        ax.axvline(lam_v, color=c, ls=":", lw=1.4, alpha=0.8)
        ax.text(lam_v, ax.get_ylim()[1] * 0.02, f"{lab}", fontsize=14, color=c,
                ha="center", va="bottom", rotation=90)
    ax.set_title("(a) $\\bar{r}\\bar{g}\\bar{b}$ 匹配函数（原书图未使用·程序重绘）", fontsize=15, color=INK)
    ax.set_xlabel("λ（nm）", fontsize=14, color=INK)
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    ax = axes[1]
    ax.plot(LAM, xyz_c[0], color=RED, lw=2.4, label="$\\bar{x}$")
    ax.plot(LAM, xyz_c[1], color=GREEN, lw=2.4, label="$\\bar{y}$（=亮度）")
    ax.plot(LAM, xyz_c[2], color=BLUE, lw=2.4, label="$\\bar{z}$")
    ax.set_title("(b) $\\bar{x}\\bar{y}\\bar{z}$（全非负；高斯近似合成）", fontsize=15, color=INK)
    ax.set_xlabel("λ（nm）", fontsize=14, color=INK)
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    for ax in axes:
        ax.set_facecolor(PAPER)
        for s in ax.spines.values():
            s.set_color("#c9c2b2")
    fig.tight_layout()
    _save(fig, "e020_fig2_29.png")


def _chromaticity_curve(lams):
    xyz = cmf_xyz(lams)
    xy = chromaticity(xyz)
    return xy[0], xy[1]


# 色度图轨迹锚点表：CIE 1931 2° 标准观察者 x̄ȳz̄ 表（10nm 间隔）按 x=X/(X+Y+Z)、
# y=Y/(X+Y+Z) 换算——数据由 colour-science 0.4.7 内置官方 CMF 数据集算出并逐值誊录
# （380–700nm 每 10nm 一点；用于色度图绘制，比讲者高斯合成更准）
XY_ANCHORS = [
    (380, 0.1741, 0.0050), (390, 0.1738, 0.0049), (400, 0.1733, 0.0048),
    (410, 0.1726, 0.0048), (420, 0.1714, 0.0051), (430, 0.1689, 0.0069),
    (440, 0.1644, 0.0109), (450, 0.1566, 0.0177), (460, 0.1440, 0.0297),
    (470, 0.1241, 0.0578), (480, 0.0913, 0.1327), (490, 0.0454, 0.2950),
    (500, 0.0082, 0.5384), (510, 0.0139, 0.7502), (520, 0.0743, 0.8338),
    (530, 0.1547, 0.8059), (540, 0.2296, 0.7543), (550, 0.3016, 0.6923),
    (560, 0.3731, 0.6245), (570, 0.4441, 0.5547), (580, 0.5125, 0.4866),
    (590, 0.5752, 0.4242), (600, 0.6270, 0.3725), (610, 0.6658, 0.3340),
    (620, 0.6915, 0.3083), (630, 0.7079, 0.2920), (640, 0.7190, 0.2809),
    (650, 0.7260, 0.2740), (660, 0.7300, 0.2700), (670, 0.7320, 0.2680),
    (680, 0.7334, 0.2666), (690, 0.7344, 0.2656), (700, 0.7347, 0.2653),
]


def chromaticity_curve_anchors(lams):
    """锚点表线性插值：返回与 lams（380–700nm）等长的 x、y 数组。"""
    lams = np.atleast_1d(np.asarray(lams, dtype=float))
    a_lam = np.array([a[0] for a in XY_ANCHORS])
    a_x = np.array([a[1] for a in XY_ANCHORS])
    a_y = np.array([a[2] for a in XY_ANCHORS])
    return np.interp(lams, a_lam, a_x), np.interp(lams, a_lam, a_y)


def _draw_chromaticity(ax, title, ellipses=False):
    lams = np.linspace(380, 700, 481)          # 轨迹可见弧止于 700nm（700–800 收敛于红角）
    x, y = chromaticity_curve_anchors(lams)
    # 轨迹渐变着色
    for k in range(len(lams) - 1):
        ax.plot(x[k:k + 2], y[k:k + 2], color=wavelength_color(lams[k:k + 2])[0], lw=3)
    # 紫线
    ax.plot([x[0], x[-1]], [y[0], y[-1]], color="#8a2a8a", lw=2.4, ls="--")
    ax.text((x[0] + x[-1]) / 2 + 0.015, (y[0] + y[-1]) / 2 - 0.035, "紫线 purple line",
            fontsize=13, color="#8a2a8a", rotation=22)
    # 内部填充（浅色）
    poly = np.column_stack([x, y])
    ax.fill(poly[:, 0], poly[:, 1], color="#efe9da", alpha=0.55, zorder=0)
    # E 白点
    ax.plot(1 / 3, 1 / 3, "o", ms=9, color="#444", zorder=5)
    ax.text(1 / 3 + 0.008, 1 / 3 + 0.006, "E 白点", fontsize=13, color="#444")
    # RGB 原色三角（锚点表插值：700.0/546.1/435.8nm 的 xy）
    tri = [chromaticity_curve_anchors(np.array([l])) for l in (LAM_R, LAM_G, LAM_B)]
    tx = [t[0][0] for t in tri]; ty = [t[1][0] for t in tri]
    ax.fill(tx, ty, color="#00000012", edgecolor="#555", lw=1.6, ls="-")
    for l, lab, dx, dy in [(LAM_R, "R", 0.012, -0.008), (LAM_G, "G", 0.0, 0.012), (LAM_B, "B", -0.038, -0.02)]:
        cx, cy = chromaticity_curve_anchors(np.array([l]))
        cx, cy = float(cx[0]), float(cy[0])
        ax.plot(cx, cy, "s", ms=6, color="#333")
        ax.text(cx + dx, cy + dy, lab, fontsize=13, color="#333", fontweight="bold")
    ax.text(0.31, 0.75, "原色三角形=色域\n（三角外显示不出）", fontsize=13, color="#555")
    if ellipses:
        # MacAdam 椭圆：位置/大小/方向为讲者示意（非原始数据）
        spots = [(0.17, 0.10, 0.012, 0.006, -25), (0.28, 0.60, 0.030, 0.014, 12),
                 (0.45, 0.52, 0.024, 0.013, 8), (0.66, 0.27, 0.016, 0.010, -40),
                 (0.21, 0.30, 0.014, 0.008, -12), (0.38, 0.30, 0.010, 0.006, 0)]
        for cx, cy, a_, b_, ang in spots:
            ax.add_patch(Ellipse((cx, cy), 2 * a_, 2 * b_, angle=ang, fill=False,
                                 edgecolor="#333", lw=1.5))
        ax.text(0.06, 0.86, "等可辨椭圆（示意位置/大小，非 MacAdam 原始数据）",
                fontsize=13, color="#333")
    ax.set_xlim(-0.06, 0.80)
    ax.set_ylim(-0.06, 0.90)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=14, color=INK)
    ax.set_ylabel("y", fontsize=14, color=INK)
    ax.set_title(title, fontsize=14, color=INK)
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color("#c9c2b2")


def fig_chromaticity():
    """Figure 2.30a（pp.90–91）重绘：色度图（原书图未使用；轨迹锚定 CIE 1931 官方 xy 表）。"""
    fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor=PAPER)
    _draw_chromaticity(ax, "CIE 色度图：光谱轨迹+紫线+E+原色三角（原书图未使用·程序重绘）")
    fig.tight_layout()
    _save(fig, "e020_fig2_30.png")


def fig_macadam():
    """Figure 2.30b（p.91）概念重绘：等可辨椭圆（位置/大小为讲者示意）。"""
    fig, ax = plt.subplots(figsize=(7.6, 6.6), facecolor=PAPER)
    _draw_chromaticity(ax, "MacAdam 椭圆=等可辨区，催生 Lab 空间（原书图未使用·概念重绘）",
                       ellipses=True)
    fig.tight_layout()
    _save(fig, "e020_macadam.png")


def fig_bayer():
    """Figure 2.31（p.94）重绘：(a) CFA 布局 (b) 插值后像素（小写=猜值）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), facecolor=PAPER)
    pat = bayer_pattern(4, 4)
    colors = {"R": "#d94848", "G": "#4a9c58", "B": "#4868c8"}
    ax = axes[0]
    for i in range(4):
        for j in range(4):
            ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, facecolor=colors[pat[i, j]],
                                       edgecolor="white", lw=2.5))
            ax.text(j + 0.5, 3 - i + 0.5, pat[i, j], ha="center", va="center",
                    fontsize=20, color="white", fontweight="bold")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_title("(a) Bayer 滤镜阵列：绿占一半（原书图未使用·程序重绘）",
                 fontsize=16, color=INK)
    ax.set_aspect("equal"); ax.axis("off")
    ax = axes[1]
    # (b)：G 位置显示 rGb（绿实测），R 位置 Rgb，B 位置 rgB——与原书 (b) 图案一致
    for i in range(4):
        for j in range(4):
            c = pat[i, j]
            col = colors[c]
            ax.add_patch(plt.Rectangle((j, 3 - i), 1, 1, facecolor=col, alpha=0.85,
                                       edgecolor="white", lw=2.5))
            parts = {"R": "R", "G": "G", "B": "B"}
            txt = {"G": "rGb", "R": "Rgb", "B": "rgB"}[c]
            ax.text(j + 0.5, 3 - i + 0.55, txt, ha="center", va="center", fontsize=16,
                    color="white", fontweight="bold")
            ax.text(j + 0.5, 3 - i + 0.18, "小写=猜", ha="center", va="center",
                    fontsize=13, color="#ffffffcc")
    ax.set_xlim(0, 4); ax.set_ylim(0, 4)
    ax.set_title("(b) 插值（去马赛克）后：大写=实测、小写=猜值", fontsize=16, color=INK)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    _save(fig, "e020_fig2_31.png")


def fig_gamma_noise():
    """Figure 2.32（p.95）重绘：gamma 压缩+噪声压暗区（原书图未使用）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), facecolor=PAPER)
    y = np.linspace(0, 1, 301)
    ax = axes[0]
    ax.plot(y, y, color="#9a9383", lw=2, ls="--", label="线性（无 gamma）")
    ax.plot(y, gamma_encode(y), color=BLUE, lw=3, label="Y′=Y^(1/2.2) 压缩")
    ax.plot(y, gamma_decode(y), color=ORANGE, lw=2.4, ls="-.", label="$\\hat{Y}$=Y′^2.2 还原")
    ax.set_xlabel("输入亮度 Y", fontsize=14, color=INK)
    ax.set_ylabel("信号值", fontsize=14, color=INK)
    ax.set_title("(a) 发送端压缩 / 接收端还原（原书图未使用·程序重绘）",
                 fontsize=15, color=INK)
    ax.legend(fontsize=14, frameon=False, loc="upper left")
    ax = axes[1]
    d = 0.02
    dec = gamma_decode(gamma_encode(y) + d)
    err_gamma = np.abs(dec - y)
    err_lin = np.full_like(y, d)
    ax.plot(y, err_lin, color="#9a9383", lw=2.4, ls="--", label="线性传输误差（恒 0.02）")
    ax.plot(y, err_gamma, color=RED, lw=3, label="gamma 传输还原后误差")
    ax.axvspan(0, 0.2, color=GREEN, alpha=0.12)
    ax.text(0.02, err_lin.max() * 0.55, "暗区：误差被压扁\n（视觉对相对差敏感）",
            fontsize=14, color=GREEN)
    ax.set_xlabel("输入亮度 Y", fontsize=14, color=INK)
    ax.set_ylabel("接收端绝对误差", fontsize=14, color=INK)
    ax.set_title("(b) 传输噪声 δ=0.02 还原后的分布（数值演示）", fontsize=15, color=INK)
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    for ax in axes:
        ax.set_facecolor(PAPER)
        for s in ax.spines.values():
            s.set_color("#c9c2b2")
    fig.tight_layout()
    _save(fig, "e020_fig2_32.png")


def fig_jpeg_quality():
    """呼应 Figure 2.34（p.100）：同一程序生成图 JPEG 三档实压（原书图未使用）。"""
    # 测试图：渐变+圆+细线+文字（块效应显影元素）
    W, H = 640, 400
    img = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(0, 255, W, dtype=float)
    img[:, :, 0] = np.tile(xs, (H, 1)).astype(np.uint8)
    img[:, :, 1] = np.tile(np.linspace(255, 0, H)[:, None], (1, W)).astype(np.uint8)
    img[:, :, 2] = 128
    pil = Image.fromarray(img)
    from PIL import ImageDraw
    dr = ImageDraw.Draw(pil)
    dr.ellipse([70, 60, 230, 220], outline=(255, 255, 255), width=6)
    dr.ellipse([380, 120, 560, 300], outline=(0, 0, 0), width=6)
    for k in range(8):
        dr.line([(40 + k * 8, 330), (600, 330 - k * 10)], fill=(255, 255, 255), width=2)
    dr.text((280, 30), "JPEG", fill=(0, 0, 0))
    buf = []
    for q in (90, 25, 8):
        b = io.BytesIO()
        pil.save(b, "JPEG", quality=q)
        b.seek(0)
        buf.append((q, Image.open(b).convert("RGB"), b.getbuffer().nbytes))
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), facecolor=PAPER)
    for ax, (q, im, nbytes) in zip(axes, buf):
        ax.imshow(im)
        ax.set_title(f"quality={q}（{nbytes//1024}KB）", fontsize=14, color=INK)
        ax.axis("off")
    fig.suptitle("同一图三档 JPEG 实压：块效应随质量下降增强（原书图未使用·程序生成）",
                 fontsize=15, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, "e020_fig2_34.png")


# --------------------------------------------------------------- 同步检查（#S）
def sync_figures():
    """E019 惯例：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步。"""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    SLIDEFIG.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in FIGDIR.glob("*.png"))
    print("--- 同步检查（先比对、后覆盖）---")
    all_same = True
    for name in names:
        src = (FIGDIR / name).read_bytes()
        dst = SLIDEFIG / name
        if dst.exists():
            same = hashlib.md5(src).hexdigest() == hashlib.md5(dst.read_bytes()).hexdigest()
            print(f"  {name}: {'一致' if same else '漂移→覆盖'}  md5={hashlib.md5(src).hexdigest()[:12]}")
            all_same &= same
        else:
            print(f"  {name}: 新增")
            all_same = False
        dst.write_bytes(src)
    return all_same


# --------------------------------------------------------------- 主流程
def main():
    print("=" * 68)
    print("E020 · 颜色从哪来：三原色、色度图与压缩收官 —— 数值实验与断言")
    print("=" * 68)

    assert_xyz_chromaticity()        # #1 4 条
    assert_negative_red()            # #2 4 条
    assert_lab()                     # #3 4 条
    assert_bt709_luma()              # #4 4 条
    assert_ycbcr_homework()          # #5 4 条
    assert_gamma_psnr()              # #6 4 条（a–d）
    assert_bayer_pattern()           # #6e 1 条（归组 #6）

    # 常数核对（不计数）
    verify_const("C1 CIE 三原色波长 700.0/546.1/435.8nm（p.89 源码常数复核）",
                 (LAM_R, LAM_G, LAM_B) == (700.0, 546.1, 435.8))
    verify_const("C2 γ≈2.2、γ1=0.45（pp.94–95 源码常数复核）",
                 GAMMA == 2.2 and GAMMA1 == 0.45)
    verify_const("C3 Y′601/709 系数（p.96 源码常数复核）",
                 Y601 == (0.299, 0.587, 0.114) and Y709 == (0.2125, 0.7154, 0.0721))
    verify_const("C4 式 2.116 亮度行与 Y′601 一致（JPEG 亮度行=0.299/0.587/0.114）",
                 np.allclose(M_YCBCR[0], np.array(Y601), atol=1e-12))
    verify_const("C5 色度图 xy 锚点表防错（λ 严格递增、x,y∈[0,0.8]、x+y≤1（红端 z→0 容差 1e-9），CIE 官方表誊录）",
                 all(XY_ANCHORS[i][0] < XY_ANCHORS[i + 1][0] for i in range(len(XY_ANCHORS) - 1))
                 and all(0.0 <= a[1] <= 0.8 and 0.0 <= a[2] <= 0.9
                         and a[1] + a[2] <= 1.0 + 1e-9 for a in XY_ANCHORS))
    _p700 = chromaticity_curve_anchors(np.array([700.0]))
    _p546 = chromaticity_curve_anchors(np.array([546.1]))
    _p436 = chromaticity_curve_anchors(np.array([435.8]))
    verify_const("C6 三原色 xy 与公开标准值一致（R=700nm/(0.7347,0.2653) 锚点直查；"
                 "G=546.1nm/(0.2737,0.7174)、B=435.8nm/(0.1665,0.0089) 插值容差 0.003；"
                 "CIE 1931 官方 CMF 换算复核）",
                 np.allclose(_p700, ([0.7347], [0.2653]), atol=1e-9)
                 and np.allclose(_p546, ([0.2737], [0.7174]), atol=0.003)
                 and np.allclose(_p436, ([0.1665], [0.0089]), atol=0.003))

    # 图（8 张）
    print("--- 程序生成图（原书图均未使用）---")
    fig_additive_subtractive()
    fig_cones()
    fig_matching_functions()
    fig_chromaticity()
    fig_macadam()
    fig_bayer()
    fig_gamma_noise()
    fig_jpeg_quality()

    # 区分力自检（不计数）
    disc_ok = assert_discrimination()

    # 同步检查
    sync_ok = sync_figures()

    n_pass = sum(1 for _, ok in RESULTS if ok)
    n_all = len(RESULTS)
    c_pass = sum(1 for _, ok in CONST_RESULTS if ok)
    print("=" * 68)
    print(f"汇总：{n_pass}/{n_all} 数据断言 PASS · 常数核对 {c_pass}/{len(CONST_RESULTS)} · "
          f"区分力自检 {'8/8 拦截' if disc_ok else 'FAIL'} · 同步检查 {'PASS' if sync_ok else 'FAIL'}"
          + (" —— 全部通过" if (n_pass == n_all and c_pass == len(CONST_RESULTS) and disc_ok and sync_ok) else " —— 存在 FAIL！"))
    return n_pass == n_all and c_pass == len(CONST_RESULTS) and disc_ok and sync_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
