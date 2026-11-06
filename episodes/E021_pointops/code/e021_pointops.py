#!/usr/bin/env python3
"""
E021 · 图像就是函数：点运算与灰度映射（§3.1 前半：节引言 原书第 109–110 页 +
3.1.1 Pixel transforms 第 111–112 页 + 3.1.2 Color transforms 第 112–113 页 +
3.1.3 Compositing and matting 第 113–115 页上部；止于 3.1.4 Histogram
equalization 标题（第 115 页下部）之前。3.1.4–3.1.5 归 E022）

数值实验：把式 3.1/3.2（像素变换定义）、式 3.3/3.4（增益偏置与空间可变）、
式 3.5（叠加原理）、式 3.6（线性混合/cross-dissolve）、式 3.7（伽马校正）、
式 3.8（over 算子）以及色度守恒提亮（回指式 2.105/2.117，E020 断言体系）
与 3×3 color twist 全部算一遍。

断言（29 条数据断言，分 8 组 + 4 项常数核对 C1–C4（不计数，不调被测函数）
+ 1 项同步 + 1 项区分力自检不计数；错误实现类断言一律由 #D 注入承担，
不在正式计数里复写判据——E019 门A round3 新P1-2 惯例）：
  #1  像素变换定义（p.111 式 3.1/3.2；2 条）：逐像素独立——翻转输入图中一个
      像素，输出仅该像素变化（「each output pixel's value depends on only the
      corresponding input pixel value」，p.109–111）；1×1 卷积等价：单通道
      gain/bias 与权重 w、偏置 b 的 1×1 卷积逐像素相等（p.111 脚注 1
      「In convolutional neural networks (Section 5.4), such operations are
      sometimes called 1 × 1 convolutions」）
  #2  增益与偏置（p.112 式 3.3；3 条）：f=100、a=1.1、b=16 → g=126.0
      （参数取自 p.110 Figure 3.2 图注「additive offset, b = 16」「multiplicative
      gain, a = 1.1」）；a 控对比：输入差 Δf 经增益后放大 a 倍（Δg=a·Δf，b 消去）；
      b 控亮度：输出平移量恰为 b、与 a 无关（「The parameters a > 0 and b are
      often called the gain and bias parameters; sometimes these parameters are
      said to control contrast and brightness, respectively」p.112）
  #3  空间可变增益（p.112 式 3.4；2 条）：渐变密度滤镜逐点 g=a(y)·f 且
      a(底部)=1.0 处不变（「simulating the graded density filter used by
      photographers to selectively darken the sky」p.112）；渐晕径向衰减场
      中心 1.0、随半径单调降（讲者推演口径：原书仅说「modeling vignetting in
      an optical system」未给公式，衰减形式为本集构造，自然渐晕 cos⁴ 见 E019）
  #4  叠加原理（p.112 式 3.5；4 条）：纯乘性增益 h(f)=af 满足
      h(f0+f1)=h(f0)+h(f1)（随机数组，容差 1e-9；「Multiplicative gain … is a
      linear operation, as it obeys the superposition principle」p.112）；
      空间可变增益同样满足（原书「both global and spatially varying」p.112）；
      含非零偏置的 h(f)=af+b 不满足——差恰为 −b（讲者推演：原书线性的主语是
      multiplicative gain，af+b 整体是仿射）；平方反例：
      (f0+f1)²−f0²−f1²=2f0f1≠0（「Operators such as image squaring … are not
      linear」p.112）
  #5  over 算子与合成（p.114 式 3.8；5 条）——编号兑现 shots.json f022 refs
      承诺「数值实验 e021_pointops.py #5 验证」：α=1 → C=F（「Pixels within the
      object are fully opaque (α = 1)」p.114）；α=0 → C=B（「pixels fully
      outside the object are transparent (α = 0)」p.114）；α=0.5 → 数值例
      B=100、F=200 → C=150 + 随机数组容差验证（式 3.8 直接代入）；预乘等价：
      premultiplied 形式 C=(1−α)B+(αF) 与纯 RGB 形式 C=(1−α)B+αF 数值相等
      （「it is convenient to represent the foreground colors in pre-multiplied
      form, i.e., to store (and manipulate) the αF values directly」p.114；
      背景为不透明底图（Figure 3.5 场景）故 α_B=1）；羽化过渡：α matte 取
      0→1 渐变时 C 从 B 平滑单调过渡到 F（「Pixels on the boundary of the
      object vary smoothly between these two extremes, which hides the
      perceptual visible jaggies」p.114）
  #6  线性混合与 cross-dissolve（p.112 式 3.6；4 条）：端点还原 α=0→f0、
      α=1→f1；α=0.5 → 两图均值；swap 恒等 g(α)+g(1−α)=f0+f1；α 从 0→1
      输出单调（「By varying α from 0 → 1, this operator can be used to
      perform a temporal cross-dissolve」p.112）
  #7  伽马校正（p.112 式 3.7；4 条）：往返 (f^(1/γ))^γ=f（0–1 均匀 401 点，
      容差 1e-9）；f=0.5 → 0.5^(1/2.2)=0.7297（中间调提亮，显式数值锚点）；
      与 E020 式 2.111/2.112 互证：加载 E020 模块 gamma_encode/decode，
      encode(0.5)=0.7297 与本集同式、decode(0.5)=0.5^2.2=0.2176 下凸（E020
      断言 #6 同源，γ=2.2 跨集一致）；γ 大小与暗部提升：0.5^(1/1.2)=0.5612
      < 0.5^(1/2.2)=0.7297（γ 越大中间调提亮越多；γ=1.2 取自 p.110 Figure 3.2
      图注「gamma (partially) linearized (γ = 1.2)」），且暗部相对提升远大于
      亮部——0.05→0.2561（5.12 倍）vs 0.8→0.9035（1.13 倍）
  #8  色彩变换（pp.112–113 §3.1.2；5 条）：三通道同加改变色度——色度坐标
      xy 位移 >0.02、色度分量和=1（式 2.105 口径不变量），且 Y 恰增 d（亮度
      行和=1 的推论，回指 E020 #1「second row adds up to one」；「adding the
      same value to each color channel not only increases the apparent
      intensity of each pixel, it can also affect the pixel's hue and
      saturation」p.113）；色度守恒提亮：各通道等比 ×k
      → xy 不变（容差 1e-12）且 Y×k 精确（「chromaticity coordinates (2.105)
      or even simpler color ratios (2.117) can first be computed and then used
      after manipulating (e.g., brightening) the luminance Y to re-compute a
      valid RGB image with the same hue and saturation」p.113——xy 不变 + Y×k
      在线性 RGB 系统中等价于各通道同乘 k，讲者推演口径）；颜色比率 (2.117)
      双口径：同加后 R:G、G:B 比率相对改变 >5%，等比缩放后不变（容差 1e-12）；
      对角 color twist ≡ 逐通道独立缩放（「color balancing … can be performed
      either by multiplying each channel with a different scale factor or by
      the more complex process of mapping to XYZ … which can be written down
      using a linear 3 × 3 color twist transform matrix」p.113）；一般 twist
      矩阵行和=1 ⇒ 中性灰 (v,v,v) 映射后仍为灰（讲者推演：白平衡保灰的实用
      性质，原书未述此性质）
  #C  常数核对（4 项 C1–C4，不计数、无区分力）：C1 γ=2.2（p.112「a gamma
      value of γ ≈ 2.2 is a reasonable fit for most digital cameras」源码常数
      复核）；C2 Figure 3.2 图注参数 b=16、a=1.1、γ=1.2（p.110 图注誊录）；
      C3 twist 矩阵行和=1 与 E020 XYZ common 矩阵亮度行和=1.00000（矩阵构造
      约定复核，回指 E020 #1）；C4 演示样例 RGB 常数 (255,128,64)（E020 #5
      同款样例跨集同例）与 (200,100,50)（×1.2 后 (240,120,60) 仍在 8 位界内）
  #S  同步检查：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步
      （重跑至一致；stdout 的 md5 供门C 冷启动四方比对——E019/E020 惯例）
  #D  区分力自检（不计数）：8 种错误实现 monkeypatch 进模块全局
      （① gain_bias a/b 交换、② linear_blend α 前后景颠倒、
      ③ over_composite 前景背景颠倒 C=αB+(1−α)F、④ premult_over 丢背景衰减
      C=B+αF、⑤ gamma_correct 方向反 g=f^γ、⑥ 色度除以 R 而非 R+G+B、
      ⑦ twist 矩阵行和 1→1.1（灰不动点破坏）、⑧ 点算子混入 3×3 邻域均值
      （点过程→邻域算子概念混淆）），重跑真实 assert_*，每例至少一条断言
      FAIL 才算拦截。

期望输出 `汇总：29/29 数据断言 PASS · 常数核对 4/4 · 区分力自检 8/8 拦截 ·
同步检查 PASS —— 全部通过`。

运行：python3 code/e021_pointops.py（在 episodes/E021_pointops/ 下）
产物：code/outputs/figures/*.png（8 张，先落 outputs 再同步 slides/figures）
"""

import hashlib
import importlib.util
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw

for _f in ["PingFang SC", "Hiragino Sans GB", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
GREEN, PURPLE, RED = "#2a7a4a", "#6b4fa0", "#b03030"
RESULTS = []
CONST_RESULTS = []
EP = pathlib.Path(__file__).resolve().parent.parent          # episodes/E021_pointops
FIGDIR = EP / "code" / "outputs" / "figures"
SLIDEFIG = EP / "slides" / "figures"


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))
    return bool(ok)


def verify_const(name, ok, detail=""):
    """常数核对（不计数）：纯常数/构造约定复核，不调用被测函数、无区分力，
    故不计入数据断言总数（E019 门A round3 新P1-2 惯例）。"""
    CONST_RESULTS.append((name, bool(ok)))
    print(f"[const-{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------- 常数（源码誊录自原书）
GAMMA = 2.2                # p.112：「a gamma value of γ ≈ 2.2 is a reasonable fit
                           #        for most digital cameras」（式 3.7 的 γ）
FIG_B = 16                 # p.110 Figure 3.2 图注：「additive offset, b = 16」
FIG_A = 1.1                # p.110 Figure 3.2 图注：「multiplicative gain, a = 1.1」
FIG_GAMMA = 1.2            # p.110 Figure 3.2 图注：「gamma (partially) linearized (γ = 1.2)」
SAMPLE_RGB = (255, 128, 64)   # E020 #5 同款样例（橙色，跨集同例）
BRIGHTEN_RGB = (200, 100, 50) # ×1.2 → (240, 120, 60)，8 位界内
# E020 式 2.104（p.90）common 形式 XYZ 矩阵的第二行 = 亮度行（行和=1）
Y_ROW = np.array([0.17697, 0.81240, 0.01063])
# 讲者构造的白平衡 twist 矩阵（行和=1；演示 p.113「linear 3 × 3 color twist
# transform matrix」；构造约定见 C3）
M_TWIST = np.array([[1.10, -0.12, 0.02],
                    [-0.05, 1.05, 0.00],
                    [0.00, -0.15, 1.15]])


# --------------------------------------------------------------- 被测实现（可注入错误）
def gain_bias(f, a, b):
    """式 3.3（p.112）：g = a·f + b。标量或 ndarray 均可。"""
    return a * np.asarray(f, dtype=float) + b


def spatial_gain_bias(f, a_map, b_map):
    """式 3.4（p.112）：g = a(x)·f + b(x)。a_map/b_map 与 f 同形。"""
    return a_map * np.asarray(f, dtype=float) + b_map


def square_op(f):
    """平方算子（p.112 反例：「image squaring … not linear」）。"""
    return np.asarray(f, dtype=float) ** 2


def linear_blend(f0, f1, alpha):
    """式 3.6（p.112）：g = (1−α)·f0 + α·f1。"""
    return (1.0 - alpha) * np.asarray(f0, dtype=float) + alpha * np.asarray(f1, dtype=float)


def over_composite(B, F, alpha):
    """式 3.8（p.114）：C = (1−α)·B + α·F（over 算子，Porter-Duff 1984）。"""
    return (1.0 - alpha) * np.asarray(B, dtype=float) + alpha * np.asarray(F, dtype=float)


def premult_over(B, F_premult, alpha):
    """预乘形式 over（p.114）：前景存 αF，C = (1−α)·B + (αF)。
    背景为不透明底图（Figure 3.5 场景，α_B=1）。"""
    return (1.0 - alpha) * np.asarray(B, dtype=float) + np.asarray(F_premult, dtype=float)


def gamma_correct(f, gamma=None):
    """式 3.7（p.112）：g = [f]^(1/γ)。gamma 缺省运行时读全局 GAMMA
    （保证 #D⑤ patch 全局可生效——默认参数会在 def 时绑定）。"""
    g = GAMMA if gamma is None else gamma
    return np.asarray(f, dtype=float) ** (1.0 / g)


def conv1x1(f, w, b):
    """1×1 卷积（p.111 脚注 1）：单通道 gain/bias 的卷积表达。"""
    return w * np.asarray(f, dtype=float) + b


def chromaticity_xy(rgb):
    """RGB 演示口径的色度坐标（式 2.105 的「分量/总和」定义用于 RGB 三分量；
    CIE XYZ 版本已在 E020 #1 验证）。rgb 为长度 3 序列或 (…,3) 数组。"""
    rgb = np.asarray(rgb, dtype=float)
    return rgb / rgb.sum(axis=-1, keepdims=True)


def luminance_y(rgb):
    """亮度 Y（8 位或浮点 RGB）：E020 common 矩阵亮度行（行和=1，p.90 回指）。"""
    return Y_ROW @ np.asarray(rgb, dtype=float)


def color_twist(rgb, mat):
    """3×3 color twist（p.113）：M @ rgb（列向量约定）。"""
    return np.asarray(mat, dtype=float) @ np.asarray(rgb, dtype=float)


def _load_e020():
    """加载 E020 模块（#7c 跨集互证；模块级仅常数/函数定义与 rcParams，无副作用）。"""
    p = EP.parent / "E020_color" / "code" / "e020_color.py"
    spec = importlib.util.spec_from_file_location("e020_color", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- 断言组
RNG = np.random.default_rng(20261104)   # 固定种子，冷启动确定性


def assert_pixel_transform():
    """#1 像素变换定义（p.111 式 3.1/3.2 + 脚注 1；2 条）。"""
    f = RNG.uniform(0, 255, size=(8, 8))
    # #1a 逐像素独立：翻转一个输入像素，输出仅该像素变化（p.109「each output
    # pixel's value depends on only the corresponding input pixel value」）
    g = gain_bias(f, FIG_A, FIG_B)
    f2 = f.copy()
    f2[3, 5] += 10.0
    g2 = gain_bias(f2, FIG_A, FIG_B)
    diff = g2 - g
    ok = (np.count_nonzero(np.abs(diff) > 1e-12) == 1
          and abs(diff[3, 5] - FIG_A * 10.0) < 1e-9)
    verify("#1a 逐像素独立：翻转一个输入像素输出仅该像素变化（Δg=a·Δf=11.0）",
           ok, f"变化像素数={np.count_nonzero(np.abs(diff) > 1e-12)}（应为 1），Δg={diff[3, 5]:.4f}")
    # #1b 1×1 卷积等价（p.111 脚注 1）
    ok = np.allclose(conv1x1(f, FIG_A, FIG_B), gain_bias(f, FIG_A, FIG_B), atol=1e-9)
    verify("#1b 单通道 gain/bias ≡ 1×1 卷积（p.111 脚注 1，全图逐像素相等）", ok)


def assert_gain_bias():
    """#2 增益与偏置（p.112 式 3.3；3 条）。"""
    # #2a 显式数值锚点（参数取自 p.110 Figure 3.2 图注）
    ok = abs(gain_bias(100.0, FIG_A, FIG_B) - 126.0) < 1e-9
    verify(f"#2a 式 3.3 数值：f=100, a={FIG_A}, b={FIG_B} → g=126.0", ok,
           f"g={float(gain_bias(100.0, FIG_A, FIG_B)):.4f}")
    # #2b a 控对比：输入差放大 a 倍
    f1, f2 = np.full((5, 5), 60.0), np.full((5, 5), 100.0)
    dg = gain_bias(f2, FIG_A, FIG_B) - gain_bias(f1, FIG_A, FIG_B)
    ok = np.allclose(dg, FIG_A * 40.0, atol=1e-9)
    verify("#2b 增益控对比：Δf=40 → Δg=a·Δf=44.0（b 消去）", ok, f"Δg={dg.ravel()[0]:.4f}")
    # #2c b 控亮度：加 b 使输出平移恰为 +b，且平移量与 a 无关（式 3.3 的可分离性）
    d1 = gain_bias(f1, FIG_A, FIG_B) - gain_bias(f1, FIG_A, 0.0)
    d2 = gain_bias(f1, 2.0 * FIG_A, FIG_B) - gain_bias(f1, 2.0 * FIG_A, 0.0)
    ok = np.allclose(d1, FIG_B, atol=1e-9) and np.allclose(d2, FIG_B, atol=1e-9)
    verify("#2c 偏置控亮度：加 b 使输出平移恰 +16，且在 a=1.1 与 a=2.2 下平移量相同（与 a 无关）",
           ok, f"平移(a=1.1)={d1.ravel()[0]:.1f}, 平移(a=2.2)={d2.ravel()[0]:.1f}")


def assert_spatial_gain():
    """#3 空间可变增益（p.112 式 3.4；2 条）。"""
    h, w = 61, 81                                   # 奇数采样：中心格恰为 r=0
    f = np.tile(np.linspace(40, 220, w), (h, 1))
    # #3a 渐变密度滤镜：a(y) 从顶部 0.5 渐变到底部 1.0（压暗天空，p.112）
    ys = np.linspace(0.0, 1.0, h)[:, None]          # 0=顶 1=底
    a_map = 0.5 + 0.5 * np.broadcast_to(ys, (h, w))
    g = spatial_gain_bias(f, a_map, 0.0)
    ok = (np.allclose(g, a_map * f, atol=1e-9)
          and np.allclose(g[-1], f[-1], atol=1e-9)           # 底部 a=1.0 不变
          and np.allclose(g[0], 0.5 * f[0], atol=1e-9))      # 顶部（天空）压半
    verify("#3a 渐变滤镜：g=a(y)·f 逐点成立；a=1.0 底部不变、顶部天空压暗至一半", ok)
    # #3b 渐晕径向衰减场（讲者推演口径：形式为本集构造，cos⁴ 见 E019）
    xs = np.linspace(-1, 1, w)[None, :]
    ys2 = np.linspace(-1, 1, h)[:, None]
    r = np.sqrt((xs * (w / h)) ** 2 + ys2 ** 2) / np.sqrt((w / h) ** 2 + 1.0)
    a_vig = 1.0 - 0.4 * r ** 2
    ok = (abs(a_vig[h // 2, w // 2] - 1.0) < 1e-12          # 中心无衰减
          and a_vig[0, 0] < a_vig[h // 2, w // 4] < 1.0     # 角落 < 边中 < 中心
          and np.all(np.diff(a_vig[:, w // 2][h // 2:]) <= 1e-12))   # 沿半径单调降
    verify("#3b 渐晕场（讲者推演形式）：中心 a=1.0、沿半径单调衰减至角落", ok,
           f"中心={a_vig[h // 2, w // 2]:.2f}, 角落={a_vig[0, 0]:.3f}")


def assert_superposition():
    """#4 叠加原理（p.112 式 3.5；4 条）。"""
    f0 = RNG.uniform(0, 100, size=(6, 6))
    f1 = RNG.uniform(0, 100, size=(6, 6))
    a_map = 0.8 + 0.4 * RNG.uniform(0, 1, size=(6, 6))
    # #4a 纯乘性增益满足叠加
    lin = lambda x: FIG_A * x                     # noqa: E731（被测：纯增益）
    ok = np.allclose(lin(f0 + f1), lin(f0) + lin(f1), atol=1e-9)
    verify("#4a 纯增益 h(f)=af 满足叠加原理（式 3.5，容差 1e-9）", ok)
    # #4b 空间可变增益同样满足（p.112「both global and spatially varying」）
    ok = np.allclose(a_map * (f0 + f1), a_map * f0 + a_map * f1, atol=1e-9)
    verify("#4b 空间可变增益 a(x)·f 同样满足叠加（原书明说 both）", ok)
    # #4c 含非零偏置不满足——差恰为 −b（讲者推演：af+b 是仿射）
    aff = lambda x: FIG_A * x + FIG_B             # noqa: E731
    resid = aff(f0 + f1) - aff(f0) - aff(f1)
    ok = np.allclose(resid, -FIG_B, atol=1e-9)
    verify("#4c 含偏置 h(f)=af+b 违反叠加：残差恒为 −b=−16（讲者推演）", ok,
           f"残差={resid.ravel()[0]:.4f}")
    # #4d 平方反例（p.112「image squaring … are not linear」）
    resid = square_op(f0 + f1) - square_op(f0) - square_op(f1)
    ok = np.allclose(resid, 2.0 * f0 * f1, atol=1e-9) and np.abs(resid).max() > 1.0
    verify("#4d 平方算子违反叠加：残差=2f0f1（最大 1 万量级，非线性实锤）", ok,
           f"max 残差={np.abs(resid).max():.1f}")


def assert_over():
    """#5 over 算子与合成（p.114 式 3.8；5 条）——兑现 f022 refs 承诺。"""
    B = RNG.uniform(20, 180, size=(6, 6))
    F = RNG.uniform(100, 255, size=(6, 6))
    # #5a α=1 → C=F
    ok = np.allclose(over_composite(B, F, 1.0), F, atol=1e-12)
    verify("#5a α=1 → C=F（物体内部全不透明，p.114）", ok)
    # #5b α=0 → C=B
    ok = np.allclose(over_composite(B, F, 0.0), B, atol=1e-12)
    verify("#5b α=0 → C=B（物体外部全透明，p.114）", ok)
    # #5c α=0.5 直接代入（f022 承诺的数值直觉）
    ok = abs(over_composite(100.0, 200.0, 0.5) - 150.0) < 1e-12
    ok = ok and np.allclose(over_composite(B, F, 0.5), 0.5 * B + 0.5 * F, atol=1e-12)
    verify("#5c α=0.5：B=100、F=200 → C=150（式 3.8 直接代入）+ 随机数组验证", ok)
    # #5d 预乘等价（p.114「store (and manipulate) the αF values directly」）
    ok = np.allclose(premult_over(B, 0.3 * F, 0.3), over_composite(B, F, 0.3), atol=1e-12)
    verify("#5d 预乘等价：C=(1−α)B+(αF) 与 C=(1−α)B+αF 数值相等（α=0.3，不透明背景）", ok)
    # #5e 羽化过渡单调（p.114「vary smoothly … hides the … jaggies」）
    alphas = np.linspace(0.0, 1.0, 21)
    col = over_composite(np.full(21, 100.0), np.full(21, 200.0), alphas)
    ok = np.all(np.diff(col) > 0) and col[0] == 100.0 and col[-1] == 200.0
    verify("#5e 羽化：α matte 0→1 渐变时 C 从 B 单调平滑过渡到 F（无跳变）", ok)


def assert_linear_blend():
    """#6 线性混合与 cross-dissolve（p.112 式 3.6；4 条）。"""
    f0 = RNG.uniform(0, 100, size=(5, 5))
    f1 = RNG.uniform(100, 255, size=(5, 5))
    # #6a 端点还原
    ok = (np.allclose(linear_blend(f0, f1, 0.0), f0, atol=1e-12)
          and np.allclose(linear_blend(f0, f1, 1.0), f1, atol=1e-12))
    verify("#6a 端点还原：α=0 → f0、α=1 → f1", ok)
    # #6b α=0.5 均值
    ok = np.allclose(linear_blend(f0, f1, 0.5), 0.5 * (f0 + f1), atol=1e-12)
    verify("#6b α=0.5 → 两图逐像素均值", ok)
    # #6c swap 恒等：g(α)+g(1−α)=f0+f1
    g_a = linear_blend(f0, f1, 0.3)
    g_b = linear_blend(f0, f1, 0.7)
    ok = np.allclose(g_a + g_b, f0 + f1, atol=1e-9)
    verify("#6c swap 恒等：g(0.3)+g(0.7)=f0+f1（α 与 1−α 互补）", ok)
    # #6d dissolve 单调（p.112「By varying α from 0 → 1 … temporal cross-dissolve」）
    seq = [float(linear_blend(40.0, 200.0, a)) for a in np.linspace(0, 1, 9)]
    ok = all(b > a for a, b in zip(seq, seq[1:]))
    verify("#6d α 0→1 序列输出严格单调（cross-dissolve 平滑过渡）", ok,
           f"序列={[round(v, 1) for v in seq]}")


def assert_gamma():
    """#7 伽马校正（p.112 式 3.7；4 条）。"""
    # #7a 往返恒等
    xs = np.linspace(0.0, 1.0, 401)
    rt = gamma_correct(xs) ** GAMMA                       # (f^(1/γ))^γ
    ok = np.allclose(rt, xs, atol=1e-9)
    verify("#7a 往返：(f^(1/γ))^γ=f（0–1 均匀 401 点，容差 1e-9）", ok)
    # #7b 显式数值锚点
    ok = abs(float(gamma_correct(0.5)) - 0.5 ** (1.0 / GAMMA)) < 1e-12
    verify(f"#7b f=0.5 → 0.5^(1/{GAMMA})={0.5 ** (1.0 / GAMMA):.4f}（中间调提亮）", ok)
    # #7c 与 E020 式 2.111/2.112 互证（γ=2.2 跨集一致）
    e020 = _load_e020()
    v_enc = float(e020.gamma_encode(0.5))
    v_dec = float(e020.gamma_decode(0.5))
    ok = (abs(v_enc - float(gamma_correct(0.5))) < 1e-12     # 同式：encode=f^(1/2.2)
          and abs(v_dec - 0.5 ** GAMMA) < 1e-9               # decode=f^2.2=0.2176 下凸
          and e020.GAMMA == GAMMA)
    verify("#7c E020 互证：encode(0.5)=0.7297 同式；decode(0.5)=0.5^2.2=0.2176 下凸；γ 均为 2.2",
           ok, f"E020 encode={v_enc:.4f}, decode={v_dec:.4f}")
    # #7d γ 大小与暗部提升（γ=1.2 取自 p.110 Figure 3.2 图注）
    g12 = float(gamma_correct(0.5, FIG_GAMMA))
    lift_dark = float(gamma_correct(0.05)) / 0.05
    lift_bright = float(gamma_correct(0.8)) / 0.8
    ok = g12 < float(gamma_correct(0.5)) and lift_dark > 3.0 > lift_bright
    verify("#7d γ=1.2 提亮 0.5612 < γ=2.2 提亮 0.7297；暗部提升 5.12 倍 ≫ 亮部 1.13 倍",
           ok, f"0.05→{float(gamma_correct(0.05)):.4f}（{lift_dark:.2f}×），0.8→{float(gamma_correct(0.8)):.4f}（{lift_bright:.2f}×）")


def assert_color_transforms():
    """#8 色彩变换（pp.112–113 §3.1.2；5 条）。"""
    rgb = np.array(SAMPLE_RGB, dtype=float)               # (255,128,64)
    d = 40.0
    xy0 = chromaticity_xy(rgb)[:2]                        # (x, y)：前两个归一化分量
    xy_add = chromaticity_xy(rgb + d)[:2]
    # #8a 同加改变色度；Y 恰增 d（亮度行和=1）
    shift = float(np.linalg.norm(xy_add - xy0))
    y0 = float(luminance_y(rgb))
    y_add = float(luminance_y(rgb + d))
    ok = (shift > 0.02 and abs(y_add - (y0 + d)) < 1e-9
          and abs(float(chromaticity_xy(rgb).sum()) - 1.0) < 1e-12)
    verify("#8a 三通道同加 40：xy 位移 0.0512>0.02（色相/饱和度被影响）；Y 恰增 40（行和=1）；"
           "色度分量和=1（式 2.105 口径，E020 #1 同款不变量）",
           ok, f"位移={shift:.4f}, ΔY={y_add - y0:.4f}")
    # #8b 色度守恒提亮：等比 ×k → xy 不变、Y×k（p.113 原书路径的线性等价形式）
    rgb2 = np.array(BRIGHTEN_RGB, dtype=float)            # (200,100,50)
    k = 1.2
    xy_b = chromaticity_xy(rgb2 * k)[:2]
    ok = (float(np.linalg.norm(xy_b - chromaticity_xy(rgb2)[:2])) < 1e-12
          and abs(float(luminance_y(rgb2 * k)) - k * float(luminance_y(rgb2))) < 1e-9)
    verify("#8b 色度守恒提亮：等比 ×1.2 → xy 不变（1e-12）、Y 恰 ×1.2（色相饱和度保持）", ok)
    # #8c 颜色比率 (2.117) 双口径：同加变、等比不变
    ratios0 = rgb[:2] / rgb[1:]                           # R/G, G/B
    ratios_add = (rgb[:2] + d) / (rgb[1:] + d)
    ratios_scale = rgb[:2] * k / (rgb[1:] * k)
    rel_add = float(np.abs(ratios_add - ratios0).max() / np.abs(ratios0).max())
    rel_scale = float(np.abs(ratios_scale - ratios0).max())
    ok = rel_add > 0.05 and rel_scale < 1e-12
    verify("#8c 颜色比率：同加后 R/G、G/B 相对改变 >5%；等比缩放后不变（1e-12）",
           ok, f"同加相对变={rel_add:.3f}")
    # #8d 对角 twist ≡ 逐通道独立缩放（p.113「multiplying each channel with a
    # different scale factor」≡「linear 3 × 3 color twist」的对角特例）
    scales = np.array([1.15, 1.0, 0.9])
    ok = np.allclose(color_twist(rgb2, np.diag(scales)), rgb2 * scales, atol=1e-12)
    verify("#8d 对角 color twist ≡ 逐通道独立缩放（数值全等）", ok)
    # #8e 行和=1 ⇒ 灰不动点（讲者推演：白平衡保灰；矩阵构造约定 C3）
    gray = np.full(3, 0.5)
    ok = np.allclose(color_twist(gray, M_TWIST), gray, atol=1e-12)
    verify("#8e twist 行和=1 ⇒ 中性灰 (0.5,0.5,0.5) 映射后仍为灰（讲者推演）", ok)


# --------------------------------------------------------------- 程序生成图（原书图均未使用）
def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[fig] {name}  {out.stat().st_size}B")


def _gray_strip(ax, g8, title):
    """8 位灰阶条带（横向渐变 0–255 输入）。"""
    xs = np.linspace(0, 255, 512)
    ax.imshow(np.tile(xs, (1, 1)), aspect="auto", cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=12, color=INK)
    ax.set_yticks([])
    ax.set_xticks([0, 128, 511], ["0", "128", "255"])


def fig_gain_bias_gamma():
    """Figure 3.2b–d 参数重演（p.110 图注参数 b=16/a=1.1/γ=1.2；原书图未使用）。"""
    xs = np.linspace(0.0, 1.0, 512)
    outs = [
        ("原图 f", xs, INK),
        (f"亮度 +b={FIG_B}（式 3.3）", np.clip(xs + FIG_B / 255.0, 0, 1), BLUE),
        (f"对比 ×a={FIG_A}（式 3.3）", np.clip(xs * FIG_A, 0, 1), ORANGE),
        (f"伽马 γ={FIG_GAMMA}（式 3.7）", xs ** (1.0 / FIG_GAMMA), GREEN),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(9.2, 6.2), facecolor=PAPER)
    for ax, (title, g, color) in zip(axes, outs):
        ax.imshow(np.tile(g * 255.0, (1, 1)), aspect="auto", cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11.5, color=color, loc="left", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Figure 3.2b–d 参数重演：同一条灰阶带经三种点运算（原书第 110 页图注参数）"
                 "—— 原书图未使用，程序生成",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e021_fig3_2_params.png")


def fig_spatial_gain():
    """式 3.4 空间可变增益两例：渐变滤镜压暗天空 + 渐晕径向衰减（原书图未使用）。"""
    h, w = 120, 180
    sky = np.tile(np.linspace(235, 150, h)[:, None], (1, w))
    ys = np.linspace(0.0, 1.0, h)[:, None]
    a_grad = np.broadcast_to(0.5 + 0.5 * ys, (h, w))
    xs = np.linspace(-1, 1, w)[None, :]
    ys2 = np.linspace(-1, 1, h)[:, None]
    r = np.sqrt((xs * 1.5) ** 2 + ys2 ** 2) / np.sqrt(1.5 ** 2 + 1.0)
    a_vig = 1.0 - 0.45 * r ** 2
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.6), facecolor=PAPER)
    axes[0, 0].imshow(sky, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("(a) 程序生成天空渐变", fontsize=12, color=INK)
    axes[0, 1].imshow(sky * a_grad, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("(b) 渐变滤镜 a(y)=0.5+0.5y（顶部压半，式 3.4）", fontsize=12, color=BLUE)
    im = axes[1, 0].imshow(a_vig, cmap="magma", vmin=0.5, vmax=1.0)
    axes[1, 0].set_title("(c) 渐晕增益场 a(r)=1−0.45r²（讲者推演形式）", fontsize=12, color=ORANGE)
    axes[1, 1].imshow(sky * a_vig, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("(d) 光学渐晕：四周压暗（E019 cos⁴ 直觉）", fontsize=12, color=BROWN)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.03)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("式 3.4 g=a(x)f+b(x) 两例：渐变密度滤镜与渐晕（原书第 112 页）"
                 "—— 原书图未使用，程序生成",
                 fontsize=13, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "e021_spatial_gain.png")


def fig_superposition():
    """式 3.5 叠加原理：线性增益重合 vs 平方分叉（原书图未使用）。"""
    x = np.linspace(0.0, 1.0, 300)
    f1 = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), facecolor=PAPER)
    ax = axes[0]
    ax.plot(x, FIG_A * (x + f1), color=BLUE, lw=2.4, label="$h(f_0+f_1)$")
    ax.plot(x, FIG_A * x + FIG_A * f1, ls="--", color=ORANGE, lw=2.0, label="$h(f_0)+h(f_1)$")
    ax.set_title(f"(a) 纯增益 h(f)={FIG_A}f：两线重合 → 线性（式 3.5）", fontsize=12.5, color=INK)
    ax.set_xlabel("输入 $f_0$", fontsize=11.5, color=INK)
    ax.set_ylabel("输出 g", fontsize=11.5, color=INK)
    ax.legend(fontsize=11, frameon=False)
    ax = axes[1]
    ax.plot(x, (x + f1) ** 2, color=BLUE, lw=2.4, label="$(f_0+f_1)^2$")
    ax.plot(x, x ** 2 + f1 ** 2, ls="--", color=ORANGE, lw=2.0, label="$f_0^2+f_1^2$")
    ax.fill_between(x, x ** 2 + f1 ** 2, (x + f1) ** 2, color=RED, alpha=0.14)
    ax.annotate("差 = $2f_0f_1$（交叉项）", xy=(0.62, (0.92) ** 2), xytext=(0.18, 0.72),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_title("(b) 平方算子：曲线分叉 → 非线性（p.112 反例）", fontsize=12.5, color=INK)
    ax.set_xlabel("输入 $f_0$", fontsize=11.5, color=INK)
    ax.set_ylabel("输出 g", fontsize=11.5, color=INK)
    ax.legend(fontsize=11, frameon=False)
    for ax in axes:
        ax.set_facecolor(PAPER)
        for s in ax.spines.values():
            s.set_color("#c9c2b2")
    fig.suptitle("叠加原理 $h(f_0+f_1)=h(f_0)+h(f_1)$：增益满足、平方违反（原书第 112 页式 3.5）"
                 "—— 原书图未使用，程序生成",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, "e021_superposition.png")


def fig_dissolve():
    """式 3.6 cross-dissolve α 序列五格（原书图未使用）。"""
    n = 200
    yy, xx = np.mgrid[0:n, 0:n]
    f0 = np.full((n, n, 3), 0.0)
    f0[..., 2] = 0.85                                  # 蓝底
    f0[((xx - 60) ** 2 + (yy - 100) ** 2) < 38 ** 2] = 1.0   # 白圆
    f1 = np.full((n, n, 3), 0.0)
    f1[..., 0] = 0.9                                   # 橙底
    f1[(xx > 110) & (xx < 180) & (yy > 60) & (yy < 140)] = 1.0  # 白方
    fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.1), facecolor=PAPER)
    for ax, a in zip(axes, [0.0, 0.25, 0.5, 0.75, 1.0]):
        g = linear_blend(f0, f1, a)
        ax.imshow(g.clip(0, 1))
        ax.set_title(f"α={a}", fontsize=12.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("式 3.6 线性混合：α 0→1 的时间交叉溶解（原书第 112 页）"
                 "—— 原书图未使用，程序生成",
                 fontsize=13, color=INK, y=1.06)
    fig.tight_layout()
    _save(fig, "e021_dissolve.png")


def fig_gamma_curve():
    """式 3.7 伽马曲线族与往返路径（原书图未使用）。"""
    xs = np.linspace(0.0, 1.0, 300)
    fig, ax = plt.subplots(figsize=(7.4, 5.6), facecolor=PAPER)
    ax.plot(xs, xs, color="#9a9384", lw=1.2, ls=":", label="恒等 γ=1")
    ax.plot(xs, gamma_correct(xs, FIG_GAMMA), color=GREEN, lw=2.2, label=f"γ={FIG_GAMMA}（Figure 3.2d）")
    ax.plot(xs, gamma_correct(xs, GAMMA), color=BLUE, lw=2.6, label=f"γ={GAMMA}（式 3.7）")
    v = 0.5
    up = float(gamma_correct(v))
    ax.annotate("", xy=(v, up), xytext=(v, v),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.0))
    ax.annotate("", xy=(up, up), xytext=(v, up),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2, ls="--"))
    ax.plot([v, v, up], [v, up, up], "o", color=ORANGE, ms=5)
    ax.annotate(f"0.5 → {up:.3f}（提亮）", xy=(0.52, 0.66), fontsize=11.5, color=ORANGE)
    ax.set_xlabel("传感器量化值 f", fontsize=12, color=INK)
    ax.set_ylabel("线性化输出 g=[f]^(1/γ)", fontsize=12, color=INK)
    ax.set_title("式 3.7：去掉传感器伽马的非线性（γ≈2.2 拟合多数数码相机，p.112）\n"
                 "—— 原书图未使用，程序生成", fontsize=12.5, color=INK)
    ax.legend(fontsize=11.5, frameon=False, loc="upper left")
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color("#c9c2b2")
    fig.tight_layout()
    _save(fig, "e021_gamma_curve.png")


def fig_chroma_brighten():
    """同加 vs 等比提亮的色度位移对比（xy 演示口径，回指式 2.105；原书图未使用）。"""
    rgb = np.array(SAMPLE_RGB, dtype=float)
    xy0 = chromaticity_xy(rgb)[:2]
    xy_add = chromaticity_xy(rgb + 40.0)[:2]
    xy_k = chromaticity_xy(np.array(BRIGHTEN_RGB, dtype=float) * 1.2)[:2]
    xy_b = chromaticity_xy(np.array(BRIGHTEN_RGB, dtype=float))[:2]
    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=PAPER)
    ax.annotate("", xy=xy_add, xytext=xy0,
                arrowprops=dict(arrowstyle="->", color=RED, lw=2.2))
    ax.plot(*xy0, "o", color=INK, ms=9)
    ax.plot(*xy_add, "o", color=RED, ms=9)
    ax.plot(*xy_k, "o", color=GREEN, ms=9, mfc="none", mew=2.2)
    ax.plot(*xy_b, "x", color=GREEN, ms=9, mew=2.2)
    ax.annotate("同加 40：色度移动 0.051\n（色相/饱和度被改变）", xy=xy_add,
                xytext=(xy_add[0] - 0.14, xy_add[1] + 0.09), fontsize=11.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.annotate("等比 ×1.2：色度不动\n（Y×1.2，色相饱和度保持）", xy=xy_k,
                xytext=(xy_k[0] + 0.05, xy_k[1] - 0.10), fontsize=11.5, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    ax.set_xlim(0.2, 0.85)
    ax.set_ylim(0.1, 0.65)
    ax.set_xlabel("色度坐标 x（RGB 演示口径，式 2.105 同型定义）", fontsize=12, color=INK)
    ax.set_ylabel("色度坐标 y", fontsize=12, color=INK)
    ax.set_title("提亮一张橙色像素：三通道同加 vs 色度守恒等比缩放（p.113 设问）\n"
                 "—— 原书图未使用，程序生成", fontsize=12.5, color=INK)
    ax.grid(True, color="#ddd5c4", lw=0.6, alpha=0.7)
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color("#c9c2b2")
    fig.tight_layout()
    _save(fig, "e021_chroma_brighten.png")


def fig_matting():
    """Figure 3.4 四格重演：源图 / 前景 F / α matte / 合成 C（原书图未使用）。"""
    n = 240
    yy, xx = np.mgrid[0:n, 0:n]
    bg0 = np.tile(np.linspace(0.85, 0.45, n)[:, None], (1, 1, 1)) * np.full((n, n, 3), [1.0, 0.95, 0.85])
    src = bg0.copy()
    mask = ((xx - 120) ** 2 / 70.0 ** 2 + (yy - 120) ** 2 / 95.0 ** 2) < 1.0
    src[mask] = np.array([0.95, 0.45, 0.15])           # 橙色前景椭圆
    F = np.zeros((n, n, 3))
    F[mask] = np.array([0.95, 0.45, 0.15])
    dist = np.sqrt((xx - 120) ** 2 / 70.0 ** 2 + (yy - 120) ** 2 / 95.0 ** 2)
    alpha = np.clip((1.0 - dist) / 0.08 + 0.5, 0.0, 1.0)   # 边界 8% 羽化
    bg1 = np.tile(np.linspace(0.35, 0.75, n)[:, None], (1, 1, 1)) * np.full((n, n, 3), [0.5, 0.9, 0.7])
    C = alpha[..., None] * F + (1.0 - alpha[..., None]) * bg1
    panels = [
        ("(a) 源图", src),
        ("(b) 前景 F", F),
        ("(c) α matte（灰度，边界羽化）", np.repeat(alpha[..., None], 3, axis=2)),
        ("(d) 合成 C（新背景）", C),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), facecolor=PAPER)
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img.clip(0, 1))
        ax.set_title(title, fontsize=11.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Figure 3.4 概念四格重演：抠图与合成（Chuang et al. 2001 四格语义，p.113）"
                 "—— 原书图未使用，程序生成",
                 fontsize=13, color=INK, y=1.04)
    fig.tight_layout()
    _save(fig, "e021_fig3_4_matting.png")


def fig_over():
    """Figure 3.5 四格重演：B ×(1−α) + αF = C（α=0.3，原书图未使用）。"""
    n = 240
    yy, xx = np.mgrid[0:n, 0:n]
    B = np.tile(np.linspace(0.25, 0.6, n)[:, None], (1, 1, 1)) * np.full((n, n, 3), [0.3, 0.45, 0.9])
    for k in range(6):                                  # 背景细线纹理
        B[(yy > 30 + k * 32) & (yy < 36 + k * 32), :] *= 1.5
    alpha = np.clip(1.0 - np.sqrt((xx - 120) ** 2 + (yy - 120) ** 2) / 85.0, 0.0, 1.0)
    F = np.zeros((n, n, 3))
    F[..., 0] = 0.95
    F[..., 1] = 0.75                                    # 黄色前景圆盘
    a = 0.3
    panels = [
        ("B（背景）", B, INK),
        (f"B×(1-α)（α={a}）", B * (1 - a), BLUE),
        ("αF（预乘前景）", alpha[..., None] * F * a, ORANGE),
        ("C = (1-α)B + αF", (1 - a) * B + alpha[..., None] * a * F, GREEN),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.5), facecolor=PAPER)
    for ax, (title, img, color) in zip(axes, panels):
        ax.imshow(img.clip(0, 1))
        ax.set_title(title, fontsize=11.5, color=color)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Figure 3.5 概念四格重演：over 算子 C=(1-α)B+αF（式 3.8，p.114；四格语义同原书）"
                 "—— 原书图未使用，程序生成",
                 fontsize=13, color=INK, y=1.04)
    fig.tight_layout()
    _save(fig, "e021_fig3_5_over.png")


# --------------------------------------------------------------- 同步检查（#S）
def sync_figures():
    """E019/E020 惯例：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步。"""
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


# --------------------------------------------------------------- 区分力自检（#D）
def assert_discrimination():
    """8 种错误实现 monkeypatch 进模块全局，重跑真实 assert_*，每例至少一条断言 FAIL。"""
    print("--- 区分力自检：注入 8 种错误实现（不计数）---")
    cases = [
        ("① gain_bias a/b 交换", "gain_bias", None, [assert_gain_bias]),
        ("② linear_blend α 前后景颠倒", "linear_blend", None, [assert_linear_blend]),
        ("③ over 前景背景颠倒 C=αB+(1−α)F", "over_composite", None, [assert_over]),
        ("④ premult_over 丢背景衰减 C=B+αF", "premult_over", None, [assert_over]),
        ("⑤ gamma 方向反 g=f^γ", "gamma_correct", None, [assert_gamma]),
        ("⑥ 色度除以 R 而非 R+G+B", "chromaticity_xy", None, [assert_color_transforms]),
        ("⑦ twist 矩阵行和 1→1.1", "M_TWIST", None, [assert_color_transforms]),
        ("⑧ 点算子混入 3×3 邻域均值", "gain_bias", None, [assert_pixel_transform]),
    ]
    ok_all = True
    for title, target, value, runners in cases:
        print(f"--- 注入{title} 开始（预期翻转：以下 FAIL 为正确行为，不计入正式断言）---")
        saved = globals()[target]
        n_before = len(RESULTS)          # 每个 case 一次：注入期记录全部不进正式计数
        try:
            if title.startswith("①"):
                globals()[target] = lambda f, a, b: b * np.asarray(f, dtype=float) + a
            elif title.startswith("②"):
                globals()[target] = lambda f0, f1, al: al * np.asarray(f0, dtype=float) + (1.0 - al) * np.asarray(f1, dtype=float)
            elif title.startswith("③"):
                globals()[target] = lambda B, F, al: al * np.asarray(B, dtype=float) + (1.0 - al) * np.asarray(F, dtype=float)
            elif title.startswith("④"):
                globals()[target] = lambda B, Fp, al: np.asarray(B, dtype=float) + np.asarray(Fp, dtype=float)
            elif title.startswith("⑤"):
                globals()[target] = (lambda g: (lambda f, gamma=None: np.asarray(f, dtype=float) ** (g if gamma is None else gamma)))(GAMMA)
            elif title.startswith("⑥"):
                globals()[target] = lambda rgb: np.asarray(rgb, dtype=float) / np.asarray(rgb, dtype=float)[..., 0:1]
            elif title.startswith("⑦"):
                globals()[target] = M_TWIST * 1.1
            elif title.startswith("⑧"):
                def bad_point(f, a, b):
                    f = np.asarray(f, dtype=float)
                    k = np.ones((3, 3)) / 9.0
                    from numpy.lib.stride_tricks import sliding_window_view
                    pad = np.pad(f, 1, mode="edge")
                    win = sliding_window_view(pad, (3, 3)).reshape(f.shape[0], f.shape[1], -1)
                    neigh = (win @ k.ravel())
                    return a * neigh + b
                globals()[target] = bad_point
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
    return ok_all


# --------------------------------------------------------------- 主流程
def main():
    print("=" * 68)
    print("E021 · 图像就是函数：点运算与灰度映射 —— 数值实验与断言")
    print("=" * 68)

    assert_pixel_transform()         # #1 2 条
    assert_gain_bias()               # #2 3 条
    assert_spatial_gain()            # #3 2 条
    assert_superposition()           # #4 4 条
    assert_over()                    # #5 5 条（兑现 f022 refs 承诺）
    assert_linear_blend()            # #6 4 条
    assert_gamma()                   # #7 4 条
    assert_color_transforms()        # #8 5 条

    # 常数核对（不计数）
    verify_const("C1 γ=2.2（p.112「a gamma value of γ ≈ 2.2 is a reasonable fit "
                 "for most digital cameras」源码常数复核）", GAMMA == 2.2)
    verify_const("C2 Figure 3.2 图注参数 b=16、a=1.1、γ=1.2（p.110 图注誊录）",
                 (FIG_B, FIG_A, FIG_GAMMA) == (16, 1.1, 1.2))
    verify_const("C3 twist 矩阵行和=1.000000（讲者构造约定）与 E020 XYZ common "
                 "矩阵亮度行和=1.00000（p.90，回指 E020 #1）",
                 np.allclose(M_TWIST.sum(axis=1), 1.0, atol=1e-12)
                 and abs(float(Y_ROW.sum()) - 1.0) < 1e-6)
    verify_const("C4 演示样例 RGB 常数：(255,128,64)（E020 #5 跨集同例）、"
                 "(200,100,50)×1.2=(240,120,60) 仍在 8 位界内",
                 SAMPLE_RGB == (255, 128, 64)
                 and all(c * 1.2 <= 255.0 + 1e-9 for c in BRIGHTEN_RGB))

    # 图（8 张）
    print("--- 程序生成图（原书图均未使用）---")
    fig_gain_bias_gamma()
    fig_spatial_gain()
    fig_superposition()
    fig_dissolve()
    fig_gamma_curve()
    fig_chroma_brighten()
    fig_matting()
    fig_over()

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
