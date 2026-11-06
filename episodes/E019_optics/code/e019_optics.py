"""
E019 · 从针孔到薄透镜：对焦、曝光、渐晕与像差（§2.2.3 Optics，原书第 74–79 页，
至 §2.3 The digital camera 标题之前——式 2.99/2.100 与 Figure 2.22 图注在 p.78，
式 2.101/2.102、机械渐晕与脚注 12 在 p.79）

数值实验：把薄透镜（式 2.97）、弥散圆与景深（Exercise 2.4）、f 值与 EV（式 2.98、
Exercise 2.5）、双色镜头色差（Figure 2.21 设置）与自然渐晕/基本辐射度关系
（式 2.99–2.102）全部算一遍。

断言（26 条数据断言，分 6 组 + 2 项常数核对（不计数，不调被测函数）+ 1 项同步
+ 1 项区分力自检不计数；错误实现类断言一律由 #D 注入承担，不在正式计数里
复写判据——门A round3 新P1-2）：
  #1  透镜定律（p.75 式 2.97，3 条）：z_o→∞ ⇒ z_i=f（100mm）；z_o=2f ⇒ z_i=2f
      （对称点，放大率 m=1）；Figure 2.19 设置（z_o=5m、f=100mm）⇒ z_i=102.041mm
      与图注 zi=102mm（印刷 p.75）一致（符号写反的错误实现由 #D①注入拦截）
  #2  放大率（讲者按：m=z_i/z_o，薄透镜横向放大率的近轴形式，取大小；3 条）：
      z_o=5m ⇒ m≈0.0204≈1/49；z_o=2f ⇒ m=1（等大）；z_o=1.5f ⇒ m=2（越近放大率
      越大——微距近摄的定量版本）
  #3  弥散圆与景深（p.75 Exercise 2.4 口径；相似三角形；5 条）：c =
      d·|z_fp−z_i′|/z_i′（z_i′ 为物点实际像距、z_fp 为焦平面位置）——对焦 5m
      焦平面挪 +1mm（d=10mm）⇒ c=0.0980mm（≈0.1mm）；同一对焦设置看 3m 物
      c=0.136mm、看 ∞ c=0.204mm（对焦 5m=0.2D，∞ 差 0.2D、3m 差 0.13D——c 与
      屈光度差成正比，谁的屈光度差大谁更糊）；换算全画幅 6.2μm 像素 ≈16px
      （「糊是能数出来的」）；景深边界（c_max=0.03mm、f/10 对焦 5m）近界 4.36m /
      远界 5.86m 且远侧更宽（不对称）；缩光圈 f/20 景深区间严格加宽（包含 f/10
      区间）——「a function of both the focus distance and the aperture」（p.75
      景深定义句）（漏归一化的错误实现由 #D②注入拦截）
  #4  f 值与 EV（p.76 式 2.98；3 条）：标称挡位序列（f_stops()）1.4→22 每挡
      面积比 ∈[1.85,2.15]（√2 进位 + 标称舍入），总进光比 (22/1.4)²=246.9 与
      理论 2⁸=256 差 <5%；式(2.98) 反向经 aperture_diameter(f,N)=f/N：f=100mm
      N=2.8 ⇒ d=35.71mm、N=22 ⇒ d=4.55mm（一挡 EV 快门等效与线性挡反例分别为
      常数核对 / #D③注入）
  #5  色差（p.77；f′=101mm/z_i′=103mm 为 Figure 2.21 图内标签〔原书第 77 页〕、
      f=100mm 为 Figure 2.19 设置〔原书第 75 页〕，z_o=5m；蓝/红配色为讲者按
      正常色散 n_蓝>n_红 ⇒ f_蓝<f_红 推定；7 条）：双色镜头 f_blue=100/f_red=101mm
      ⇒ 像距 102.041/103.082mm、纵向差 1.040mm；放大率相对差 1.02%（横向色差）；
      红焦平面上蓝光弥散斑 c=0.102mm（与 #3 公式同源）；纵向色差量与切线预测
      互证：Δz_i ≈ (z_i/f)²·Δf（dz_i/df=(z_i/f)²，相对偏差 <0.1%）；圆盘核
      MTF=|2J₁(2πfR)/(2πfR)|（几何离焦核，非衍射极限）主瓣形状（尺度不变的数学
      性质）：f∈[0.05,0.95]f₀ 内随频率单调下降、0.95f₀ 处 <0.1（f₀=3.8317/(2πR) 第一
      零点；不外推零点之外——旁瓣回升不在范围）；色差模糊定量耦合（固定
      10 cyc/mm≈全画幅奈奎斯特 1/8）：R=0.051mm（由 Δz_i 算出）使对比度掉到
      0.161（带下界 0.15，拦「漏因子 2」型错误实现）、孔径缩半弥散减半回升
      0.711——衰减幅度由色差数据决定（p.77「strongly attenuated…hard to
      recover」的固定物理频率定量版）；缩半回升比 mtf_small/mtf_big≈4.41
      （宽松区间 [3.5,5.5]，与绝对值无关的稳健断言）
  #6  自然渐晕与基本辐射度关系（式 2.99–2.100 在 p.78、式 2.101–2.102 在 p.79；
      5 条）：cos⁴α 数值点（0°/15°/30°/45° → 1/0.8705/0.5625/0.25；三个来源共
      四个 cos 因子）；全画幅（43.3mm 对角、f=50mm）半对角 23.4° ⇒ cos⁴α≈0.71
      （仅自然项）；进光挡位 E(f/2)/E(f/16)=64=2⁶ 精确；机械渐晕遮挡因子：解析
      圆盘重叠面积 vs 蒙特卡洛采样（20 万点固定种子）一致（±0.01），缩光圈
      f/20 遮挡归零、f/4 有遮挡——p.79「mechanical vignetting can be decreased
      by reducing the camera aperture」的几何版（cos² 错误衰减律由 #D⑤注入拦截）
  #C  常数核对（2 项，不计数、无区分力）：一挡 EV 快门等效（(√2)²=2 vs
      250/125=2，Exercise 2.5 语义）；全幅/1/2.3″ 手机底单像素集光面积比 ≈30.8
      （p.79「smaller sensors … so much noisier」面积项）
  #S  同步检查：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步
      （重跑至一致；stdout 的 md5 供门C 冷启动四方比对）
  #D  区分力自检（不计数）：把 11 种错误实现 monkeypatch 进模块全局（透镜定律
      符号写反、弥散圆漏归一化、f 值线性挡位、无色差镜头、cos² 错误衰减律、
      放大率反向、景深界对称化、MTF 漏因子 2、孔径直径 f·N、辐射度 d/f 未平方、
      缩光圈反向——孔径缩半函数被反向为加倍 d/2→2d），
      重跑真实 assert_* 函数，验证每例至少一条断言实际翻转为 FAIL——区分力由
      「注入 → 真实断言 FAIL」直接证明，而非在函数外复写判据（门A round2 新P1-2、
      round4 残留风险：disc_mtf/fundamental_radiometric/depth_of_field_bounds/
      magnification/aperture_diameter 五函数补齐注入覆盖；round8 重做 ⑪ 为真反向：
      patch half_aperture 使 R_half=2R，耦合断言与缩半回升比断言同时 FAIL——
      round7 版 patch circle_of_confusion×2 不构成反向已废弃）

图（4 张：2 张程序重绘原书图 + 2 张讲者自制图表；原书图均未使用。
原书 Figure 2.20 为镜头刻度照片，本集未重绘）：
  e019_fig2_19.png     Figure 2.19（p.75）重绘：薄透镜光路（f=100mm、孔径 d、
                       zo=5m、zi=102mm、焦平面移动 Δzi 与弥散圆 c，弥散圆张角
                       放大 10 倍示意）
  e019_dof_chart.png   讲者自制图表（非原书图；Exercise 2.4 口径、呼应 p.76
                       Figure 2.20 景深指示窗概念）：景深表（对焦距离 × f 值的
                       近/远界）+ 弥散圆随焦平面位移曲线
  e019_chromatic.png   Figure 2.21（p.77）重绘：双色镜头光路（f′=101mm/z_i′=103mm
                       为 Figure 2.21 图内标签、f=100mm 为 Figure 2.19 设置，
                       蓝/红配色为讲者按正常色散推定）+ 红焦平面上蓝光弥散斑
                       小图（弥散斑放大 20 倍示意、过焦散开线斜率放大约 12 倍）
  e019_vignetting.png  讲者自制图表（非原书图；呼应 p.78 Figure 2.22 渐晕概念）：
                       cos⁴α 衰减曲线 + 全画面自然渐晕热图 + 式 (2.102) 三因子标注

运行：`python3 code/e019_optics.py`
期望输出 `汇总：26/26 数据断言 PASS · 常数核对 2/2 · 区分力自检 11/11 拦截 · 同步检查 PASS —— 全部通过`。
"""
import hashlib
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.special import j1

# 中文字体（同 E014–E018：macOS 优先 Hiragino Sans GB；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
GREEN, PURPLE, RED = "#2a7a4a", "#6b4fa0", "#b03030"
RESULTS = []
CONST_RESULTS = []
EP = pathlib.Path(__file__).resolve().parent.parent          # episodes/E019_optics
FIGDIR = EP / "code" / "outputs" / "figures"


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))


def verify_const(name, ok, detail=""):
    """常数核对（不计数）：纯数值换算/文献常数复核，不调用被测函数、无区分力，
    故不计入数据断言总数（门A round3 新P1-2：避免夸大验证覆盖）。"""
    CONST_RESULTS.append((name, bool(ok)))
    print(f"[const-{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------- 公共函数
def thin_image(z_o, f):
    """式(2.97)（p.75）：薄透镜像距 z_i = 1/(1/f − 1/z_o)（z_o、f 同单位）。"""
    return 1.0 / (1.0 / f - 1.0 / z_o)


def magnification(z_o, f):
    """横向放大率 m = z_i/z_o（讲者按：薄透镜近轴形式，z_i 由式 2.97 给出；本集取大小，实像倒立）。"""
    return thin_image(z_o, f) / z_o


def circle_of_confusion(d, z_fp, z_i_prime):
    """弥散圆（p.75 Exercise 2.4 口径；相似三角形）：孔径直径 d、焦平面位置 z_fp、
    物点实际像距 z_i′ ⇒ 焦平面上的弥散圆 c = d·|z_fp − z_i′|/z_i′。"""
    return d * abs(z_fp - z_i_prime) / z_i_prime


def half_aperture(d):
    """孔径缩半（缩光圈的几何：d → d/2）。独立成函数以便 #D⑪ 注入：
    「缩光圈反向」= 该函数被错误实现为 d → 2d，弥散圆应减半反而翻倍。"""
    return d / 2.0


def depth_of_field_bounds(z_o_focus, f, d, c_max):
    """景深近/远界（p.75 景深定义口径：c ≤ c_max 的场景深度范围；Exercise 2.4）。

    焦平面固定在对焦物距 z_o_focus 的像距 z_fp；物距 z_o 的弥散圆
    c(z_o) = d·|z_fp − z_i′(z_o)|/z_i′。把 z_i′ 反解出阈值得像距界，
    再经式(2.97) 回到物距界。"""
    z_fp = thin_image(z_o_focus, f)
    z_i_near = d * z_fp / (d - c_max)          # 近界像距（> z_fp，物更近）
    z_i_far = d * z_fp / (d + c_max)           # 远界像距（< z_fp，物更远）
    z_near = 1.0 / (1.0 / f - 1.0 / z_i_near)
    z_far = 1.0 / (1.0 / f - 1.0 / z_i_far)
    return z_near, z_far


def disc_mtf(freq, radius):
    """均匀圆盘模糊核的 MTF：|2·J₁(2πfR)/(2πfR)|（几何离焦核，非衍射极限；
    取绝对值——旁瓣为负不代表频率响应反向）。"""
    x = 2 * np.pi * freq * radius
    out = np.where(np.abs(x) < 1e-12, 1.0, 2.0 * j1(np.abs(x)) / np.where(np.abs(x) < 1e-12, 1.0, np.abs(x)))
    return np.abs(out)


def cos4(alpha):
    """自然渐晕因子 cos⁴α（p.78 式 2.99 推得、p.79 式 2.102 末项）。"""
    return np.cos(alpha) ** 4


def f_stops():
    """标称 f 值挡位序列（p.76：f/1.4, f/2, f/2.8, …, f/22，√2 进位）。"""
    return [1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22]


def aperture_diameter(focal_mm, N):
    """式(2.98)（p.76）：f/# = N = f/d ⇒ 孔径直径 d = f/N（同单位，如毫米）。"""
    return focal_mm / N


def fundamental_radiometric(L, f, d, alpha):
    """式(2.102)（p.79）：E = L·(π/4)·(d/f)²·cos⁴α。"""
    return L * (np.pi / 4.0) * (d / f) ** 2 * cos4(alpha)


# ---------------------------------------------------------------- 实验 1
def assert_lens_law():
    """#1 透镜定律（p.75 式 2.97）。"""
    f = 100.0                                  # mm（Figure 2.19 设置）
    z_inf = thin_image(1e12, f)
    verify("式(2.97) 极限：z_o→∞ ⇒ z_i=f（100mm）",
           abs(z_inf - f) < 1e-6, f"z_i={z_inf:.9f} mm")
    z_2f = thin_image(2 * f, f)
    verify("式(2.97) 对称点：z_o=2f ⇒ z_i=2f（物像等距、m=1）",
           abs(z_2f - 2 * f) < 1e-9, f"z_i={z_2f:.6f} mm")
    z_19 = thin_image(5000.0, f)
    verify("Figure 2.19 设置：z_o=5m、f=100mm ⇒ z_i=102.041mm 与图注 zi=102mm 一致",
           abs(z_19 - 102.0) < 0.05, f"z_i={z_19:.4f} mm（图注 102mm，p.75）")


def assert_magnification():
    """#2 放大率（讲者按：m=z_i/z_o）。"""
    m5 = magnification(5000.0, 100.0)
    verify("放大率：z_o=5m ⇒ m≈0.0204≈1/49",
           abs(m5 - 1.0 / 49.0) < 1e-12 and abs(1.0 / m5 - 49.0) < 1e-9,
           f"m={m5:.6f}（1/m={1/m5:.3f}）")
    m_2f = magnification(200.0, 100.0)
    verify("放大率：z_o=2f ⇒ m=1（原像多大、像就多大）",
           abs(m_2f - 1.0) < 1e-12, f"m={m_2f:.12f}")
    m_15f = magnification(150.0, 100.0)
    verify("放大率：z_o=1.5f ⇒ m=2（越近放大率越大——微距近摄的定量版）",
           abs(m_15f - 2.0) < 1e-12, f"m={m_15f:.12f}")


# ---------------------------------------------------------------- 实验 2
def assert_circle_of_confusion():
    """#3 弥散圆与景深（p.75 Exercise 2.4 口径；景深定义句 p.75、式 2.98 p.76）。"""
    f, d, z_fp = 100.0, 10.0, thin_image(5000.0, 100.0)   # 对焦 5m
    c_move = circle_of_confusion(d, z_fp + 1.0, z_fp)      # 焦平面挪 +1mm
    verify("弥散圆：对焦 5m、焦平面挪 +1mm、d=10mm ⇒ c=0.098mm（≈0.1mm）",
           abs(c_move - 10.0 * 1.0 / z_fp) < 1e-12 and abs(c_move - 0.098) < 0.002,
           f"c={c_move:.4f} mm")
    c_near = circle_of_confusion(d, z_fp, thin_image(3000.0, f))   # 对焦 5m 看 3m
    c_inf = circle_of_confusion(d, z_fp, f)                        # 对焦 5m 看 ∞
    verify("同一对焦设置看不同物距：c 与屈光度差成正比——∞（差 0.2D）比 3m（差 0.13D）糊，谁的屈光度差大谁更糊",
           abs(c_inf / c_near - 1.50) < 0.02 and c_inf > c_near,
           f"c(∞)={c_inf:.4f}、c(3m)={c_near:.4f} mm（对焦 5m=0.2D：∞ 差 0.2D、3m 差 0.13D）")
    px = c_move / 0.0062                       # 全画幅常见像素间距 6.2μm（讲者按）
    verify("「糊」能数出来：0.098mm ≈ 16 个全画幅像素",
           15 <= px <= 17, f"{px:.1f} px（6.2μm/px）")
    z_near, z_far = depth_of_field_bounds(5000.0, f, 10.0, 0.03)
    verify("景深界（c_max=0.03mm、f/10、对焦 5m）：近 4.36m / 远 5.86m，且远侧更宽",
           abs(z_near / 1000 - 4.36) < 0.05 and abs(z_far / 1000 - 5.86) < 0.05
           and (z_far - 5000) > (5000 - z_near),
           f"[{z_near/1000:.2f}, {z_far/1000:.2f}] m")
    zn20, zf20 = depth_of_field_bounds(5000.0, f, 5.0, 0.03)
    verify("缩光圈加深景深：f/20 区间严格包含 f/10 区间（p.75「a function of both the focus distance and the aperture」）",
           zn20 < z_near and zf20 > z_far, f"f/20 [{zn20/1000:.2f}, {zf20/1000:.2f}] m ⊃ f/10")


# ---------------------------------------------------------------- 实验 3
def assert_f_number():
    """#4 f 值与 EV（p.76 式 2.98；Exercise 2.5）。"""
    stops = f_stops()
    ratios = [(stops[i + 1] / stops[i]) ** 2 for i in range(len(stops) - 1)]
    verify("挡位序列 1.4→22：每挡面积比 ∈[1.85,2.15]（√2 进位+标称舍入）",
           all(1.85 <= r <= 2.15 for r in ratios),
           f"面积比 {min(ratios):.3f}~{max(ratios):.3f}（√2²=2）")
    total = (stops[-1] / stops[0]) ** 2
    verify("总进光比 (22/1.4)²=246.9 与理论 2⁸=256 差 <5%（标称值舍入）",
           abs(total - 256) / 256 < 0.05, f"{total:.1f} vs 256")
    d28, d22 = aperture_diameter(100.0, stops[2]), aperture_diameter(100.0, stops[-1])
    verify("式(2.98) 反向（d=f/N）：f=100mm 时 f/2.8 ⇒ d=35.71mm、f/22 ⇒ d=4.55mm",
           abs(d28 - 35.714) < 0.001 and abs(d22 - 4.545) < 0.001,
           f"d={d28:.3f} / {d22:.3f} mm")
    verify_const("一挡 EV 等效（Exercise 2.5 语义，常数核对）：面积比 (√2)²=2 与快门比 (1/125)/(1/250)=2 相等",
                 abs((2.0 ** 0.5) ** 2 - 2.0) < 1e-12 and abs(250.0 / 125.0 - 2.0) < 1e-12,
                 "p.76 1/250→1/125（纯常数换算，不计入数据断言）")


# ---------------------------------------------------------------- 实验 4
def assert_chromatic():
    """#5 色差（p.77；f′=101/z_i′=103 为 Figure 2.21 图内标签〔原书第 77 页〕，
    f=100 为 Figure 2.19 设置〔原书第 75 页〕，z_o=5m；蓝/红配色为讲者按正常色散推定）。"""
    f_blue, f_red, z_o = 100.0, 101.0, 5000.0
    z_blue, z_red = thin_image(z_o, f_blue), thin_image(z_o, f_red)
    verify("双色镜头：蓝 f=100（Fig 2.19 设置）⇒ 102.041mm、红 f′=101（Fig 2.21 标签）⇒ 103.082mm，纵向差 1.04mm",
           abs(z_blue - 102.0408) < 1e-3 and abs(z_red - 103.0822) < 1e-3
           and 1.0 < z_red - z_blue < 1.1, f"Δz_i={z_red - z_blue:.4f} mm")
    m_blue, m_red = z_blue / z_o, z_red / z_o
    rel = m_red / m_blue - 1.0
    verify("横向色差：放大率相对差 ≈1.02%",
           0.009 < rel < 0.0115, f"Δm/m={rel * 100:.3f}%")
    c_on_red = circle_of_confusion(10.0, z_red, z_blue)   # 焦平面在红焦点，蓝光弥散
    verify("红焦平面上蓝光弥散斑 c=0.102mm（与 #3 公式同源）",
           abs(c_on_red - 10.0 * (z_red - z_blue) / z_blue) < 1e-12
           and abs(c_on_red - 0.102) < 0.002, f"c={c_on_red:.4f} mm")
    dz_exact = z_red - z_blue
    dz_tangent = (z_blue / f_blue) ** 2 * (f_red - f_blue)   # dz_i/df = (z_i/f)² 的切线预测
    verify("纵向色差量与切线预测互证：Δz_i ≈ (z_i/f)²·Δf（相对偏差 <0.1%）",
           abs(dz_exact - dz_tangent) < 1e-3 * dz_exact,
           f"精确 {dz_exact:.4f} vs 切线 {dz_tangent:.4f} mm")
    R = c_on_red / 2.0                         # 弥散圆半径（由色差 Δz_i 与孔径 d 决定）
    if R > 0.0:                                # 注入「无色差镜头」时 R=0，跳过（前面断言已 FAIL）
        f0 = 3.8317059702075125 / (2 * np.pi * R)  # J₁ 第一正零点（完整精度）
        fs = np.linspace(0.05 * f0, 0.95 * f0, 19)
        mvals = disc_mtf(fs, R)
        verify("圆盘核 MTF 主瓣形状（数学性质，尺度不变）：f∈[0.05,0.95]f₀ 内随频率单调下降、0.95f₀ 处 <0.1",
               bool(np.all(np.diff(mvals) < 0)) and mvals[-1] < 0.1,
               f"主瓣内 {mvals[0]:.2f}→{mvals[-1]:.3f}（f₀={f0:.1f} cyc/mm；形状性质，不外推零点之外——旁瓣回升不在范围）")
        f_test = 10.0                          # 固定物理频率（cyc/mm）：全画幅 6.2μm 像素奈奎斯特 80.6 的 1/8
        mtf_big = float(disc_mtf(f_test, R))
        R_half = circle_of_confusion(half_aperture(10.0), z_red, z_blue) / 2.0   # 孔径缩半 ⇒ 弥散减半
        mtf_small = float(disc_mtf(f_test, R_half))
        verify(f"色差模糊定量耦合（固定 {f_test:.0f} cyc/mm）：弥散 R={R:.4f}mm（由 Δz_i={z_red - z_blue:.2f}mm 算出）"
               f" 使对比度掉到 {mtf_big:.2f}；孔径缩半（d=5）弥散减半回升到 {mtf_small:.2f}",
               0.15 < mtf_big < 0.2 and mtf_big < mtf_small,
               f"0.15 < {mtf_big:.3f} < 0.2 < {mtf_small:.3f}——衰减幅度由色差数据决定"
               "（p.77「strongly attenuated…hard to recover」的固定物理频率定量版；下界拦「漏因子 2」型错误实现）")
        ratio = mtf_small / mtf_big
        verify(f"缩半回升比（与绝对值无关的稳健断言）：mtf_small/mtf_big={ratio:.2f}",
               3.5 < ratio < 5.5,
               f"{mtf_small:.3f}/{mtf_big:.3f}={ratio:.2f}（R→R/2 时圆盘核 MTF 比值由 J₁ 函数行为决定；"
               "宽松区间 [3.5,5.5]，拦 R 计算错置或「缩半」逻辑反向类错误）")


# ---------------------------------------------------------------- 实验 5
def assert_vignetting():
    """#6 自然渐晕与基本辐射度关系（式 2.99–2.100 在 p.78、式 2.101–2.102 在 p.79）。"""
    pts = [(0.0, 1.0), (15.0, 0.8705), (30.0, 0.5625), (45.0, 0.25)]
    ok_pts = all(abs(cos4(np.radians(a)) - v) < 1e-3 for a, v in pts)
    verify("cos⁴α 数值点：0°/15°/30°/45° → 1/0.8705/0.5625/0.25",
           ok_pts, "（讲者按：式 2.102 末项，三个来源共四个 cos 因子合成 cos⁴）")
    half_diag = np.degrees(np.arctan(21.6 / 50.0))
    v_corner = cos4(np.radians(half_diag))
    verify("全画幅（43.3mm 对角）f=50mm 半对角 23.4° ⇒ cos⁴α≈0.71（仅自然项）",
           0.70 <= v_corner <= 0.72, f"α={half_diag:.2f}°、cos⁴α={v_corner:.4f}")
    verify("进光挡位：E(f/2)/E(f/16)=64=2⁶ 精确（f 值平方项）",
           abs(fundamental_radiometric(1, 50, 25, 0) / fundamental_radiometric(1, 50, 50 / 16, 0) - 64.0) < 1e-9,
           "d=25mm(f/2) vs d=3.125mm(f/16)")
    verify_const("传感器面积项（常数核对）：同像素数同 f 值下全画幅/1/2.3″ 手机底单像素集光面积比 ≈30.8",
                 30 <= (36.0 * 24.0) / (6.17 * 4.55) <= 32,
                 f"{(36.0 * 24.0) / (6.17 * 4.55):.1f}×（864 / 28.1 mm²，p.79「smaller sensors … noisier」面积项；"
                 "纯几何常数，不计入数据断言）")
    # 机械渐晕几何模型：镜筒口直径 D_b、孔径止到筒口距离 t、离轴 α ⇒ 斜光束在筒口
    # 平面横移 s=t·tanα；遮挡因子 = 1 − 止盘（横移 s）与筒口盘的重叠面积占比。
    D_b, t_v, alpha_v = 30.0, 10.0, np.radians(30.0)
    s_shift = t_v * np.tan(alpha_v)

    def occlusion_analytic(d_ap):
        """遮挡因子解析式（圆-圆重叠面积标准公式）。"""
        r1, r2 = d_ap / 2.0, D_b / 2.0
        if s_shift >= r1 + r2:
            return 1.0
        if s_shift <= abs(r1 - r2):
            return 0.0
        a1 = r1 ** 2 * np.arccos((s_shift ** 2 + r1 ** 2 - r2 ** 2) / (2 * s_shift * r1))
        a2 = r2 ** 2 * np.arccos((s_shift ** 2 + r2 ** 2 - r1 ** 2) / (2 * s_shift * r2))
        a3 = 0.5 * np.sqrt((-s_shift + r1 + r2) * (s_shift + r1 - r2)
                           * (s_shift - r1 + r2) * (s_shift + r1 + r2))
        return 1.0 - (a1 + a2 - a3) / (np.pi * r1 ** 2)

    rng = np.random.default_rng(2019)
    pts = rng.uniform(-12.5, 12.5, size=(200_000, 2))

    def occlusion_mc(d_ap):
        """同模型的蒙特卡洛：止盘均匀采样、横移 s 后落在筒口外的比例（独立数值路径）。"""
        inside = pts[:, 0] ** 2 + pts[:, 1] ** 2 <= (d_ap / 2.0) ** 2
        px = pts[inside, 0] + s_shift
        blocked = px ** 2 + pts[inside, 1] ** 2 > (D_b / 2.0) ** 2
        return float(blocked.mean())

    occ_big_a, occ_big_mc = occlusion_analytic(25.0), occlusion_mc(25.0)
    occ_small = occlusion_analytic(5.0)
    verify("机械渐晕遮挡因子：解析重叠面积 vs 蒙特卡洛采样（20 万点固定种子）一致（±0.01）",
           abs(occ_big_a - occ_big_mc) < 0.01,
           f"d=25（f/4，f=100mm）：解析 {occ_big_a:.4f} vs MC {occ_big_mc:.4f}")
    verify("机械渐晕方向（几何推导）：缩光圈到 f/20 遮挡归零、f/4 有遮挡——p.79「mechanical vignetting can be decreased by reducing the camera aperture」",
           occ_small == 0.0 and occ_big_a > 0.0,
           f"d=5（f/20，f=100mm）遮挡 {occ_small:.1f} < d=25 遮挡 {occ_big_a:.4f}")


# ---------------------------------------------------------------- 图
def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / name, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"[fig] {name}")


def draw_fig2_19():
    """Figure 2.19（p.75）重绘：薄透镜光路（f=100mm、zo=5m、zi=102mm、Δzi 与 c）。"""
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    f, zi, zo = 100.0, 102.0408, 5000.0
    x_lens = 0.0
    ax.plot([x_lens], [0], marker="o", ms=6, color=INK)
    ax.annotate("P", (x_lens, 0), xytext=(-10, 14), textcoords="offset points", fontsize=11, color=INK)
    # 透镜（双凸示意）与孔径
    ax.plot([x_lens, x_lens], [-1.6, 1.6], color=BLUE, lw=2.5)
    ax.plot([x_lens, x_lens], [-0.45, 0.45], color=INK, lw=6, solid_capstyle="butt")
    ax.annotate("孔径 d", (x_lens, -1.05), xytext=(-64, -4), textcoords="offset points",
                fontsize=10, color=INK)
    # 焦平面（正确位置）与挪动后的焦平面
    GRAY = "#8a8a8a"
    ax.plot([zi, zi], [-1.3, 1.3], color=GRAY, lw=1.6, ls="--")
    ax.annotate("焦平面 z$_i$=102 mm", (zi, 1.3), xytext=(-8, 10), textcoords="offset points",
                fontsize=10, color=INK)
    zi2 = zi + 1.0
    ax.plot([zi2, zi2], [-1.0, 1.0], color=RED, lw=2.0)
    ax.annotate("挪动 Δz$_i$=1mm 后", (zi2, -1.0), xytext=(-30, -26), textcoords="offset points",
                fontsize=10, color=RED)
    # 三条代表光线：从物点（轴上 5m）过孔径边缘到像点再散开
    x_o = -8.0                                  # 物点（示意压缩，非按比例）
    ax.plot([x_o, x_lens], [0, 0.45], color=BROWN, lw=1.2)
    ax.plot([x_lens, zi], [0.45, 0], color=BROWN, lw=1.2)
    ax.plot([x_lens, zi], [-0.45, 0], color=BROWN, lw=1.2)
    ax.plot([zi, zi2], [0, 0.044], color=RED, lw=1.2)   # 过像点后继续传播
    ax.plot([zi, zi2], [0, -0.044], color=RED, lw=1.2)
    ax.plot([zi2, zi2], [-0.044, 0.044], color=INK, lw=4, solid_capstyle="butt")
    ax.annotate("弥散圆 c（张角放大 10 倍示意）", (zi2, 0.07), xytext=(4, 8), textcoords="offset points",
                fontsize=10, color=INK)
    ax.plot([x_o], [0], marker="o", ms=5, color=BROWN)
    ax.annotate("物点 z$_o$=5 m\n（示意压缩，非按比例）", (x_o, 0), xytext=(x_o - 0.2, 0.75),
                ha="center", va="bottom", fontsize=10, color=BROWN)
    # 焦距标注
    ax.annotate("", xy=(f, -1.75), xytext=(x_lens, -1.75),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0))
    ax.text(f / 2, -1.95, "f = 100 mm", ha="center", fontsize=10, color=INK)
    ax.set_xlim(x_o - 0.6, zi2 + 1.6); ax.set_ylim(-2.4, 2.1)
    ax.axis("off")
    ax.set_title("薄透镜光路（式 2.97）：对焦平面挪开 Δz$_i$，像点摊成弥散圆 c\n"
                 "设置同 Figure 2.19（原书第 75 页）：f=100mm、z$_o$=5m、z$_i$=102mm（程序重绘 · 原书图未使用）",
                 fontsize=12.5, color=INK)
    _save(fig, "e019_fig2_19.png")


def draw_dof_chart():
    """讲者自制图表（非原书图；Exercise 2.4 口径、呼应 Figure 2.20 p.76 景深指示窗概念）：景深表 + 弥散圆曲线。"""
    f, c_max = 100.0, 0.03
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    # (a) 景深表：对焦距离 1–10m × f/4–f/22
    focuses = np.linspace(1000, 10000, 9)
    for d, N, c in [(25.0, 4, BLUE), (12.5, 8, GREEN), (8.33, 12, ORANGE), (4.55, 22, RED)]:
        near = [depth_of_field_bounds(zz, f, d, c_max)[0] / 1000 for zz in focuses]
        far = [depth_of_field_bounds(zz, f, d, c_max)[1] / 1000 for zz in focuses]
        ax1.fill_between(focuses / 1000, near, far, alpha=0.18, color=c)
        ax1.plot(focuses / 1000, near, color=c, lw=1.6, label=f"f/{N} 近界")
        ax1.plot(focuses / 1000, far, color=c, lw=1.6, ls="--", label=f"f/{N} 远界")
    ax1.set_xlabel("对焦距离 z$_o$（m）"); ax1.set_ylabel("清晰范围（m，c$_{max}$=0.03mm）")
    ax1.set_title("(a) 景深表（f=100mm）：光圈越小、清晰带越宽\n（自制图表 · 呼应 Figure 2.20 概念 · 原书图未使用）",
                  fontsize=11.5, color=INK)
    ax1.legend(fontsize=8.5, ncol=2, loc="upper left"); ax1.grid(alpha=0.25)
    # (b) 弥散圆 vs 焦平面位移
    z_fp = thin_image(5000.0, f)
    dz = np.linspace(-3, 3, 121)
    cfw = [circle_of_confusion(10.0, z_fp + e, z_fp) for e in dz]
    ax2.plot(dz, cfw, color=INK, lw=2.2)
    ax2.axhline(0.03, color=ORANGE, lw=1.4, ls="--", label="c$_{max}$=0.03mm（可接受模糊）")
    ax2.axhspan(-0.0, 0.03, color=GREEN, alpha=0.12)
    ax2.set_xlabel("焦平面位移 Δz$_i$（mm）"); ax2.set_ylabel("弥散圆 c（mm）")
    ax2.set_title("(b) 弥散圆随焦平面位移线性增长（d=10mm、对焦 5m）\n（自制图表 · 式 2.97+Exercise 2.4 口径）",
                  fontsize=11.5, color=INK)
    ax2.legend(fontsize=9); ax2.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "e019_dof_chart.png")


def draw_chromatic():
    """Figure 2.21（p.77）重绘：双色镜头光路（f′=101mm/z_i′=103mm 为 Figure 2.21
    图内标签、f=100mm 为 Figure 2.19 设置〔原书第 75 页〕；蓝/红配色为讲者按正常
    色散推定）+ 红焦平面上蓝光弥散斑小图（弥散斑放大 20 倍示意）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [2.1, 1]})
    f_blue, f_red, z_o, d = 100.0, 101.0, 5000.0, 10.0
    z_blue, z_red = thin_image(z_o, f_blue), thin_image(z_o, f_red)
    ax1.plot([0, 0], [-1.4, 1.4], color=BLUE, lw=2.5)
    ax1.plot([0, 0], [-0.4, 0.4], color=INK, lw=5, solid_capstyle="butt")
    ax1.plot([z_blue, z_blue], [-1.1, 1.1], color=BLUE, lw=1.8, ls="--")
    ax1.plot([z_red, z_red], [-1.1, 1.1], color=RED, lw=1.8, ls="--")
    ax1.annotate(f"蓝光焦点 z$_i$={z_blue:.1f} mm（f=100）", (z_blue, 1.1),
                 xytext=(-58, 8), textcoords="offset points", fontsize=9.5, color=BLUE)
    ax1.annotate(f"红光焦点 z$_i$′={z_red:.1f} mm（f′=101）", (z_red, -0.6),
                 xytext=(-58, -6), textcoords="offset points", fontsize=9.5, color=RED)
    ax1.plot([-6.5, 0], [0, 0.35], color=BROWN, lw=1.2)
    ax1.plot([0, z_blue], [0.35, 0], color=BLUE, lw=1.2)
    ax1.plot([0, z_red], [0.35, 0], color=RED, lw=1.2)
    ax1.plot([0, z_blue], [-0.35, 0], color=BLUE, lw=1.2)
    ax1.plot([0, z_red], [-0.35, 0], color=RED, lw=1.2)
    ax1.plot([z_blue, z_red], [0, 0.044], color=BLUE, lw=1.2)     # 蓝光过焦点继续发散
    ax1.plot([z_blue, z_red], [0, -0.044], color=BLUE, lw=1.2)
    ax1.annotate("物点 z$_o$=5 m", (-6.5, 0), xytext=(-16, 10), textcoords="offset points",
                 fontsize=10, color=BROWN)
    ax1.annotate(f"Δz$_i$={z_red - z_blue:.2f} mm", ((z_red + z_blue) / 2, -1.15), ha="center",
                 fontsize=10, color=INK)
    ax1.set_xlim(-7.6, z_red + 1.2); ax1.set_ylim(-1.8, 1.9); ax1.axis("off")
    fan_scale = (0.044 / (z_red - z_blue)) / (0.35 / z_blue)   # 散开线斜率相对真实斜率的放大倍数
    ax1.set_title(f"(a) 双色镜头：不同色焦点不同（纵向），放大率略不同（横向）；蓝光过焦散开线斜率放大 {fan_scale:.0f} 倍示意\n"
                  "f′=101mm/z_i′=103mm 为 Figure 2.21 图内标签（原书第 77 页）、f=100mm 为 Figure 2.19 设置（原书第 75 页）\n"
                  "蓝/红配色按正常色散推定（程序重绘 · 原书图未使用）",
                  fontsize=11.5, color=INK)
    # (b) 红焦平面上的蓝光弥散斑
    n = 300
    xy = np.linspace(-1.3, 1.3, n)
    X, Y = np.meshgrid(xy, xy)
    R2 = X * X + Y * Y
    R = circle_of_confusion(d, z_red, z_blue) / 2 * 20   # 盘放大 20 倍便于观察（讲者按）
    spot = np.where(R2 <= (R / max(xy)) ** 2, 1.0, 0.0)
    ax2.imshow(spot, extent=[-1.3, 1.3, -1.3, 1.3], cmap="Blues", origin="lower", vmin=0, vmax=3)
    ax2.plot(0, 0, "+", color=RED, ms=14, mew=2)
    ax2.annotate("红光焦点（点）", (0, 0), xytext=(8, -18), textcoords="offset points",
                 fontsize=9.5, color=RED)
    ax2.annotate(f"蓝光糊成 c={circle_of_confusion(d, z_red, z_blue):.3f}mm 的盘", (0, 0.75),
                 fontsize=9.5, color=INK, ha="center")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("(b) 红焦平面上：蓝光糊成一圈\n（自制图表 · 原书图未使用 · 盘放大 20 倍示意）", fontsize=11.5, color=INK)
    fig.tight_layout()
    _save(fig, "e019_chromatic.png")


def draw_vignetting():
    """讲者自制图表（非原书图；呼应 Figure 2.22 p.78 渐晕概念）：cos⁴α 曲线 + 全画面渐晕热图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    al = np.linspace(0, 50, 241)
    ax1.plot(al, cos4(np.radians(al)), color=INK, lw=2.4, label=r"自然渐晕 cos$^4\alpha$")
    ax1.plot(al, np.cos(np.radians(al)) ** 2, color=RED, lw=1.6, ls="--", label=r"错误律 cos$^2\alpha$（对照）")
    for a, v in [(15, None), (30, None), (45, None)]:
        ax1.plot([a], [cos4(np.radians(a))], "o", color=ORANGE, ms=5)
        ax1.annotate(f"{cos4(np.radians(a)):.2f}", (a, cos4(np.radians(a))),
                     xytext=(-2, 8), textcoords="offset points", fontsize=9, color=ORANGE)
    hd = np.degrees(np.arctan(21.6 / 50.0))
    ax1.axvline(hd, color=GREEN, lw=1.2, ls=":")
    ax1.annotate(f"全画幅半对角 {hd:.1f}°（f=50）\ncos$^4$α={cos4(np.radians(hd)):.2f}",
                 (hd, 0.5), xytext=(10, -52), textcoords="offset points", fontsize=9, color=GREEN)
    ax1.set_xlabel("离轴角 α（度）"); ax1.set_ylabel("相对亮度")
    ax1.set_title(r"(a) 自然渐晕 = cos$^4\alpha$：纯几何衰减、缩光圈不减轻" + "\n（自制图表 · 呼应 Figure 2.22 概念 · 原书图未使用）",
                  fontsize=11.5, color=INK)
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    # (b) 全画面自然渐晕热图（f=50、全画幅 36×24）
    n = 240
    xs = np.linspace(-18, 18, n)
    ys = np.linspace(-12, 12, n)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X * X + Y * Y)
    A = np.arctan(R / 50.0)
    img = cos4(A)
    im = ax2.imshow(img, extent=[-18, 18, -12, 12], origin="lower", cmap="magma")
    ax2.annotate(f"角落亮度 {img[0, 0]:.2f}（仅自然项）", (-17.5, -10.8), fontsize=9.5,
                 color="#f3efe6")
    ax2.set_title("(b) 全画幅画面上的自然渐晕（f=50mm）\n" + r"式 (2.102)：E = L·(π/4)·(d/f)$^2$·cos$^4\alpha$（自制图表）",
                  fontsize=11.5, color=INK)
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    _save(fig, "e019_vignetting.png")


# ---------------------------------------------------------------- 区分力自检
def assert_discrimination():
    """区分力自检（不计数）：把 11 种错误实现 monkeypatch 进模块全局，重跑真实
    assert_* 函数，验证每例至少一条断言实际翻转为 FAIL。

    门A round2 新P1-2：函数外复写判据（same-same 恒假式、常量式、常数求值）
    不算区分力——必须是「替换全局函数 → 真实断言函数 FAIL」。每例：patch →
    跑真实 assert_* → 数 FAIL 条数 → 恢复全局与 RESULTS（不污染正式计数）。
    round4 残留风险整改：disc_mtf/fundamental_radiometric/depth_of_field_bounds/
    magnification/aperture_diameter 五函数补齐注入覆盖（⑥–⑩）。
    round8（r7 P1-1 整改）：⑪ 重做为真「缩光圈反向」——patch half_aperture 为
    d→2d，使 R_half=2R（应减半反而翻倍）→ 耦合断言 mtf_big<mtf_small FAIL 且
    比值 mtf_small/mtf_big≈0.35 掉出 [3.5,5.5] FAIL——「缩半回升比」断言的专属
    拦截例（比值对 ⑧ 漏因子 2 不变）。round7 版 patch circle_of_confusion×2
    不构成反向（R_half/R 仍为 1/2，比值 2.90 的 FAIL 来自绝对 R 翻倍），已废弃。"""
    global RESULTS, CONST_RESULTS
    results_saved = list(RESULTS)
    const_saved = list(CONST_RESULTS)
    real_thin = thin_image
    real_fns = {"thin_image": thin_image, "circle_of_confusion": circle_of_confusion,
                "cos4": cos4, "f_stops": f_stops, "magnification": magnification,
                "depth_of_field_bounds": depth_of_field_bounds, "disc_mtf": disc_mtf,
                "aperture_diameter": aperture_diameter, "half_aperture": half_aperture,
                "fundamental_radiometric": fundamental_radiometric}

    def thin_sign_flipped(z_o, f):
        return 1.0 / (1.0 / f + 1.0 / z_o)            # ① 透镜定律符号写反

    def thin_achromatic(z_o, f):
        return real_thin(z_o, 100.0)                   # ④ 无色差镜头：各色同焦距

    def coc_unnormalized(d, z_fp, z_i_prime):
        return d * abs(z_fp - z_i_prime)               # ② 弥散圆漏 /z_i′ 归一化

    def cos2(alpha):
        return np.cos(alpha) ** 2                      # ⑤ cos² 错误衰减律

    def mag_flipped(z_o, f):
        return z_o / real_thin(z_o, f)                 # ⑥ 放大率反向（m=z_o/z_i）

    def dof_symmetric(z_o_focus, f, d, c_max):
        # ⑦ 景深界物方对称化（忽略像方非线性）
        z_fp = thin_image(z_o_focus, f)
        s = c_max * z_o_focus ** 2 / (d * z_fp)
        return z_o_focus - s, z_o_focus + s

    def mtf_missing_factor2(freq, radius):
        # ⑧ MTF 漏因子 2：J₁(x)/x 而非 2J₁(x)/x（形状相同、数值减半）
        x = 2 * np.pi * freq * radius
        out = np.where(np.abs(x) < 1e-12, 1.0,
                       j1(np.abs(x)) / np.where(np.abs(x) < 1e-12, 1.0, np.abs(x)))
        return np.abs(out)

    def aperture_wrong(focal_mm, N):
        return focal_mm * N                            # ⑨ 孔径直径 f·N（应为 f/N）

    def radiometric_no_square(L, f, d, alpha):
        return L * (np.pi / 4.0) * (d / f) * cos4(alpha)   # ⑩ (d/f) 未平方

    def aperture_doubled(d):
        return 2.0 * d                                     # ⑪ 缩光圈反向：d/2 被实现为 2d

    cases = [
        ("① 透镜定律符号写反", {"thin_image": thin_sign_flipped}, assert_lens_law),
        ("② 弥散圆漏归一化", {"circle_of_confusion": coc_unnormalized}, assert_circle_of_confusion),
        ("③ f 值线性挡位", {"f_stops": lambda: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]},
         assert_f_number),
        ("④ 无色差镜头", {"thin_image": thin_achromatic}, assert_chromatic),
        ("⑤ cos² 错误衰减律", {"cos4": cos2}, assert_vignetting),
        ("⑥ 放大率反向", {"magnification": mag_flipped}, assert_magnification),
        ("⑦ 景深界物方对称化", {"depth_of_field_bounds": dof_symmetric}, assert_circle_of_confusion),
        ("⑧ MTF 漏因子 2", {"disc_mtf": mtf_missing_factor2}, assert_chromatic),
        ("⑨ 孔径直径 f·N", {"aperture_diameter": aperture_wrong}, assert_f_number),
        ("⑩ 辐射度 (d/f) 未平方", {"fundamental_radiometric": radiometric_no_square}, assert_vignetting),
        ("⑪ 缩光圈反向（d→2d）", {"half_aperture": aperture_doubled}, assert_chromatic),
    ]
    print("[区分力] monkeypatch 注入错误实现 → 重跑真实 assert_*，期望至少一条 FAIL：")
    n_caught = 0
    for name, patch, fn in cases:
        globals().update(patch)
        RESULTS = []
        try:
            fn()
        finally:
            n_fail = sum(1 for _, ok in RESULTS if not ok)
            globals().update(real_fns)
            RESULTS = results_saved
            CONST_RESULTS[:] = const_saved
        caught = n_fail > 0
        n_caught += int(caught)
        print(f"[区分力] {name} → 注入后重跑 {fn.__name__}：{n_fail} 条 FAIL"
              f"（{'拦截 ✓' if caught else '未拦截 ✗✗✗'}）")
    print(f"[区分力] {n_caught}/{len(cases)} 错误实现被真实断言拦截")
    return n_caught == len(cases), n_caught, len(cases)


# ---------------------------------------------------------------- 同步
def sync_and_md5():
    """同步检查（先比后写）：outputs/ ↔ slides/figures/。"""
    names = ["e019_fig2_19.png", "e019_dof_chart.png", "e019_chromatic.png", "e019_vignetting.png"]
    figdir = EP / "slides" / "figures"
    print("[sync] 比对在写入前完成，漂移判 FAIL：")
    drift = False
    for name in names:
        new = hashlib.md5((FIGDIR / name).read_bytes()).hexdigest()
        ref = figdir / name
        if not ref.exists():
            print(f"[sync-DRIFT] {name} 在 slides/figures/ 缺失——本轮判 FAIL（将补齐，重跑至一致）")
            drift = True
            continue
        old = hashlib.md5(ref.read_bytes()).hexdigest()
        if old != new:
            print(f"[sync-DRIFT] {name} md5 {old[:8]} ≠ 新图 {new[:8]}——本轮判 FAIL（将覆盖，重跑至一致）")
            drift = True
        else:
            print(f"[sync] {name} md5 {new[:12]} 一致")
    if drift:
        import shutil
        for name in names:
            shutil.copyfile(FIGDIR / name, figdir / name)
            print(f"[sync] 已补齐/覆盖 {figdir / name}")
    return not drift


def main():
    for fn in (assert_lens_law, assert_magnification, assert_circle_of_confusion,
               assert_f_number, assert_chromatic, assert_vignetting):
        fn()
    disc_ok, n_caught, n_cases = assert_discrimination()
    draw_fig2_19()
    draw_dof_chart()
    draw_chromatic()
    draw_vignetting()
    ok = sync_and_md5()
    n_pass = sum(1 for _, p in RESULTS if p)
    n_const_ok = sum(1 for _, p in CONST_RESULTS if p)
    groups = {"#1": 3, "#2": 3, "#3": 5, "#4": 3, "#5": 7, "#6": 5}
    assert sum(groups.values()) == len(RESULTS), \
        f"groups 总数 {sum(groups.values())} ≠ 实际断言数 {len(RESULTS)}"
    head = (f"汇总：{n_pass}/{len(RESULTS)} 数据断言 PASS · 常数核对 {n_const_ok}/{len(CONST_RESULTS)}"
            f" · 区分力自检 {n_caught}/{n_cases} 拦截 · 同步检查 {'PASS' if ok else 'FAIL'}")
    all_ok = n_pass == len(RESULTS) and n_const_ok == len(CONST_RESULTS) and ok and disc_ok
    print(head + (" —— 全部通过" if all_ok else ""))
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
