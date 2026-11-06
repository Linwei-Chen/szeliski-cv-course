"""
E018 · 光如何变成数：光照、反射与 BRDF 入门（§2.2 + §2.2.1 + §2.2.2，原书第 66–74 页）

数值实验：把光照模型（式 2.81）、BRDF（式 2.82–2.83）、反射方程（式 2.84–2.86）、
漫反射/镜面/Phong 着色（式 2.87–2.94）、双色反射模型（式 2.95–2.96）与全局光照
最小系统全部算一遍。

断言（34 条数据断言，分 11 组 + 1 项同步）：
  #1  平方反比定律（p.67）：蒙特卡洛光线追迹——点光源各向同性发射 200 万条光线，
      半径 0.2 的固定接收盘放在 r=1/2/4 处，命中数比 ≈ 16:4:1，log-log 斜率 ≈ -2
      ——「the same light is being spread over a larger (spherical) area」
  #2  cos⁺ 前缩因子（p.69 式 2.85）：θ=0°/30°/60°/90°/120° → 1 / √3/2 / 1/2 / 0 / 0；
      θ>90°（背光）恒为 0——「becoming completely self-shadowed」（p.70）
  #3  BRDF 互易性（p.68 Helmholtz reciprocity）：镜面瓣 BRDF（式 2.91 方向版，代码路径
      不对称：vi 定镜面方向、vr 定视线）在 200 组随机 (v̂i,v̂r) 上 f(vi,vr)=f(vr,vi)——
      对称性是恒等式而非构造；非互易反例（×cosθi，非物理）同一检验立即报不对称
  #4  各向同性（p.68 式 2.83）：入射/出射方向同时绕法线转 α，f 不变（f 只经 θi,θr,|φr-φi|
      计算）；各向同性 ≠ 常数（f 可随相对方位差 |φr-φi| 变化）；各向异性反例（拉丝铝式，
      f 依赖入射光相对划痕的绝对方位）同转 α 后波动显著
  #5  反射方程（p.69 式 2.84→2.86）：把三个离散光源换成球面高斯窄光束做数值积分
      （黎曼和），束宽 σ=6°→3°→1.5° 逐档单调收敛到式(2.86) 求和；Lambertian（fd 常数）
      时积分与式(2.88) 相等；漏掉 cos⁺ 的错误对照立即偏离（检验有区分力）
      ——「replace the integral with a summation」
  #6  镜面反射方向（p.71 式 2.90）：ŝ = (2nnᵀ-I)v 满足 ‖ŝ‖=1、ŝ·n = v·n（法向分量保留）、
      ŝ⊥(n×v)（在入射面内）、切向分量取反 ‖ŝ_t+v_t‖=0——即绕法线转 180°；
      2nnᵀ-I 与 Rodrigues θ=π 旋转矩阵 I+2[n]×² 逐元相等且 det=+1（是旋转不是镜像）
      ——全部检验跑在凑满的 64 组有效样本上
  #7  Phong 剖面（p.71 式 2.91–2.92；参照 p.72 Figure 2.18 设置——光源离法线 30°、
      Exp=10/100/1000——按讲者视线角口径重绘，非原书图形态）：光源离法线 +30°，
      镜面方向由式(2.90) 对光源向量真实计算（ŝ=(2nnᵀ-I)v̂i → 极角 -30°，非手写负号），
      ke=10/100/1000 的剖面峰都在视线对准镜面方向（-30°）处；完整对称剖面的半高宽
      随 ke 单调收窄；高斯式(2.92) 同峰位
  #8  Phong 合成的颜色规律（p.71 式 2.93–2.94；p.72）：非平坦光谱（暖色直射 Li、偏蓝
      环境 La、材质谱 ka）下验证——① ks 为常数（界面反射不染色）→ 镜面方向 ±20° 扫
      21 个视线，高光路光谱/Li 逐波长恒定（高光颜色 = 光源颜色）；② p.72「ka(λ) and
      kd(λ) are the same」→ 环境、漫反射两路乘同一材质谱：各自除以自己的光源谱后
      逐波长相同（kd 真实参与计算），kd≠ka 反例立即失配；且 La≠Li 使两路表观颜色
      仍不同（≠「两路同色」）；③ 有色 ks（金）反例：① 的检验立即失配——检验有区分力
  #9  朗伯球着色（p.70–71 式 2.88）：单位球面法向 n，光从固定方向入射，
      I(n) = fd·[v̂i·n̂]⁺（fd=1）在正对光处最大（1−dot<1e-12）、明暗分界环带两侧翻转
      （漏 max(0,·) 的错误实现同一检验报警）、背光半球恒为零；
      光源方位绕球的 z 轴（经度方向）旋转，球面亮度分布沿经度平移（轴对称）；
      渲染一致性：与 (a) 图共用 lambert_light/lambert_render，读取渲染像素检验——
      ±ux 径向取像素、ρ=sinθ 换算极角，亮侧剖面 = cos⁺(θ−30°)（盘内全程 >0，分界在
      背面 120° 不可见）、背光侧 = cos⁺(θ+30°) 分界在可见区 60°±1°（n·L=0 相对光方向
      恒 90°）；光离视线 45° 的反例同一像素检验立即失配
  #10 双色反射模型（p.73 式 2.95–2.96）：表观颜色 Lr = ci(λ)mi(g) + cb(λ)mb(g)，
      任意几何 g 下光谱向量落在 {ci,cb} 二维子空间内——200 组几何的信号矩阵（未中心化）
      秩 ≤2、子空间投影残差 ≈0；加第三条光谱路的反例秩升至 3，同一检验立即报警
      ——「a simple product between a relative power
      spectrum c(λ)…and a magnitude m(…)」
  #11 全局光照最小系统（p.73–74）：两块正对面片形状因子双尺度独立数值积分满足
      F12·A1 = F21·A2（互易；被积核按构造对称，断言实质检验两套独立求积的收敛一致性）；
      多次反弹稳态的辐射度线性方程组解与不动点迭代解一致；
      能量语义：只发光面片的辐射度高于自发光 E（收到回弹光）——「A large linear system
      can then be set up to solve for the final lightness of each area patch」
  #S  同步检查：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步（重跑至一致）

图（4 张，均为讲者程序重绘/设计，原书图未使用）：
  e018_fig2_14.png        Figure 2.14（p.66）重绘：光度成像简化模型（光源→表面散射→相机）
  e018_phong_profile.png  参照 Figure 2.18（p.72）设置讲者重绘（横轴=视线角）：Phong 三分量剖面 + 极坐标
  e018_lambert_sphere.png 讲者设计：朗伯球渲染（式 2.88）与双经线亮度剖面
  e018_dichromatic.png    讲者设计：双色反射「光谱×几何」分解（秩 2 结构）示意

运行 `python3 code/e018_photometric_image_formation.py`，
期望输出 `34/34 数据断言 PASS · 同步检查 PASS`。
"""
import hashlib
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（同 E014–E017：macOS 优先 Hiragino Sans GB；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
GREEN, PURPLE = "#2a7a4a", "#6b4fa0"
RESULTS = []
EP = pathlib.Path(__file__).resolve().parent.parent          # episodes/E018_photometric_image_formation
FIGDIR = EP / "code" / "outputs" / "figures"


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"——{detail}" if detail else ""))


def cos_plus(theta):
    """式(2.85)：cos⁺θ = max(0, cos θ)（p.69）。"""
    return np.maximum(0.0, np.cos(theta))


def mirror_vec(v, n):
    """式(2.90)：ŝ = (2nnᵀ−I)v，返回镜面方向向量（实验与画图共用，图上的镜面峰不再手写）。"""
    return 2 * float(np.dot(n, v)) * n - v


def mirror_polar(v, n):
    """式(2.90) 镜面方向的极角 atan2(s_x, s_z)。"""
    s = mirror_vec(v, n)
    return float(np.arctan2(s[0], s[2]))


def lambert_light():
    """朗伯球实验与 (a) 图共用的光方向：离视线（z 轴）恰 30°、方位 135°，精确三角构造。"""
    t30, t45 = np.radians(30), np.radians(45)
    return np.array([-np.sin(t30) * np.cos(t45), np.sin(t30) * np.sin(t45), np.cos(t30)])


def lambert_render(L, res=420, half=1.05):
    """式(2.89) 的朗伯球正交投影渲染——draw_lambert_sphere (a) 图与断言 #9 末条共用同一实现。

    盘面网格 x,y = linspace(−half, half, res)，可见半球法向 n=(x, y, √(1−x²−y²))，
    返回亮度图 I = max(0, n·L)（盘外 NaN）。正交投影下盘面半径 ρ = sinθ（θ 为法向
    离视线的极角），断言据此把像素位置换算成 θ。"""
    px = np.linspace(-half, half, res)
    X, Y = np.meshgrid(px, px)
    R2 = X * X + Y * Y
    Z = np.sqrt(np.clip(1.0 - R2, 0.0, 1.0))
    I = np.maximum(0.0, X * L[0] + Y * L[1] + Z * L[2])
    return np.where(R2 <= 1.0, I, np.nan)


# ---------------------------------------------------------------- #1 平方反比
def assert_inverse_square():
    """#1 平方反比定律（p.67）：蒙特卡洛光线追迹——点光源各向同性发射，固定尺寸的
    接收盘放在距离 r 处，被命中数随 1/r² 下降（log-log 斜率 ≈ -2）。"""
    rng = np.random.default_rng(18)
    N = 2_000_000
    d = rng.normal(size=(N, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)   # 高斯归一化 = 球面均匀方向（各向同性发射）
    a = 0.2                                         # 接收盘半径（固定物理尺寸，正对光源）
    counts = []
    for r in (1.0, 2.0, 4.0):
        t = r / d[:, 2]                             # 光线到达平面 z=r 的参数（d_z>0 才会命中正面）
        hit = (d[:, 2] > 0) & ((t * d[:, 0]) ** 2 + (t * d[:, 1]) ** 2 <= a * a)
        counts.append(int(hit.sum()))
    c1, c2, c4 = counts
    slope = np.log(c4 / c1) / np.log(4.0)           # 命中数 ∝ 1/r² → 斜率 ≈ -2
    # 精确期望（代码命中判据是平面接收盘：锥半角 tanα=a/r，命中概率 p=(1−r/√(r²+a²))/2）：
    # N=2e6、a=0.2 → ≈ 19419 : 4963 : 1248；c1/c4 期望 ≈15.56；三组计数为嵌套锥（c4⊂c2⊂c1）
    # 并非独立，σ 按独立泊松近似 ≈0.45（相对 2.9%；按二项条件算约 0.43，结论不受影响）
    # ——断言 |c1/c4−16|<2.0 的下界 14 距期望 ≈3.4σ
    verify(f"蒙特卡洛（N={N:,} 条光线，半径 0.2 的固定接收盘）：命中数比 r=1:2:4 ≈ 16:4:1",
           abs(c1 / c2 - 4) < 0.5 and abs(c1 / c4 - 16) < 2.0,
           f"{c1} : {c2} : {c4}")
    verify("log-log 斜率 ≈ -2（平方反比，p.67「the same light is being spread over a larger (spherical) area」）",
           abs(slope + 2) < 0.1, f"斜率 {slope:+.3f}")


# ---------------------------------------------------------------- #2 cos⁺
def assert_cos_plus():
    """#2 cos⁺ 前缩因子（p.69 式 2.85 + p.70 自遮挡）。"""
    cases = [(0.0, 1.0), (np.radians(30), np.sqrt(3) / 2), (np.radians(60), 0.5),
             (np.pi / 2, 0.0), (np.radians(120), 0.0)]
    vals = [cos_plus(t) for t, _ in cases]
    verify("cos⁺ 剖面：θ=0°/30°/60°/90°/120° → 1 / √3/2 / 1/2 / 0 / 0",
           all(abs(a - b) < 1e-12 for a, (_, b) in zip(vals, cases)),
           "/".join(f"{v:.3f}" for v in vals) + "（90° 以后全 0）")
    verify("cos⁺ 单调不增且背光恒零（θ∈[90°,180°] 全为 0）",
           all(cos_plus(np.radians(a)) < 1e-12 for a in range(90, 181, 5)))


# ---------------------------------------------------------------- #3 互易性
def spec_lobe_brdf(vi, vr, n, ke=20.0):
    """镜面瓣 BRDF（式 2.91 的方向版）：fd + fs·cos^ke∠(vr, ŝ)，ŝ = 2(n·vi)n - vi。

    代码路径不对称——vi 只用来定镜面方向、vr 只用来定视线；互易性 f(vi,vr)=f(vr,vi)
    是要验证的恒等式，不是构造出来的。
    """
    s = 2 * float(np.dot(n, vi)) * n - vi
    return 0.3 + 0.9 * max(0.0, float(np.dot(vr, s))) ** ke


def assert_reciprocity():
    """#3 Helmholtz 互易性（p.68）：非平凡 BRDF 在随机 (v̂i,v̂r) 上 f(vi,vr)=f(vr,vi)；
    非互易反例（×cosθi，非物理）同一检验立即报不对称——检验有区分力。"""
    rng = np.random.default_rng(3)
    n = np.array([0.0, 0.0, 1.0])
    worst = worst_bad = 0.0
    for _ in range(200):
        vi = rng.normal(size=3); vi[2] = abs(vi[2]); vi /= np.linalg.norm(vi)
        vr = rng.normal(size=3); vr[2] = abs(vr[2]); vr /= np.linalg.norm(vr)
        if np.linalg.norm(vi + vr) < 1e-6:
            continue
        worst = max(worst, abs(spec_lobe_brdf(vi, vr, n) - spec_lobe_brdf(vr, vi, n)))
        worst_bad = max(worst_bad, abs(spec_lobe_brdf(vi, vr, n) * float(np.dot(vi, n))
                                       - spec_lobe_brdf(vr, vi, n) * float(np.dot(vr, n))))
    verify("互易性（p.68 Helmholtz reciprocity）：镜面瓣 BRDF 在 200 组随机 (v̂i,v̂r) 上 f(vi,vr)=f(vr,vi)",
           worst < 1e-12, f"最大不对称 {worst:.2e}（交换入射/出射方向答案不变）")
    verify("检验有区分力：非互易反例（×cosθi，非物理 BRDF）同一检验立即报不对称",
           worst_bad > 0.05, f"最大不对称 {worst_bad:.3f}")


# ---------------------------------------------------------------- #4 各向同性
def assert_isotropic():
    """#4 各向同性（p.68 式 2.83）：入射/出射方位同时绕法线转 α，f 不变——f 只依赖相对几何；
    各向同性 ≠ 常数（可随 |φr-φi| 变化）；各向异性反例（拉丝铝式，依赖绝对方位）同转 α 后波动显著。"""
    ti, tr = np.radians(30), np.radians(50)
    dphi = np.radians(70)

    def sph(t, p):
        return np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])

    vi, vr = sph(ti, 0.0), sph(tr, dphi)

    def f_iso(a, b):
        # 各向同性实现：只经 (θi,θr,|φr-φi|) 计算——式(2.83) 的右侧
        ta = np.arccos(np.clip(a[2], -1, 1)); tb = np.arccos(np.clip(b[2], -1, 1))
        dp = abs(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))
        dp = min(dp, 2 * np.pi - dp)
        return 0.3 + 0.9 * np.exp(-((ta - tb) / 0.25) ** 2) + 0.5 * np.cos(dp) ** 8

    def Rz(al):
        c, s = np.cos(al), np.sin(al)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    worst = max(abs(f_iso(Rz(al) @ vi, Rz(al) @ vr) - f_iso(vi, vr))
                for al in np.linspace(0, 2 * np.pi, 25))
    verify("各向同性（式 2.83）：入射、出射方向同时绕法线转 α（扫 25 个 α），f 逐点不变",
           worst < 1e-12, f"最大波动 {worst:.2e}——f 只依赖 (θi,θr,|φr-φi|)")
    # 各向同性 ≠ 常数：f 可以随相对方位差 |φr-φi| 变化（cos⁸(Δφ) 项，式 2.83 保留它为自变量）
    vr2 = sph(tr, np.pi)                             # Δφ=180°（对跖方位，cos⁸ 项取峰）
    verify("各向同性 ≠ 常数：f 随相对方位差 |φr-φi| 变化（Δφ=70° vs 180° 差异显著）",
           abs(f_iso(vi, vr) - f_iso(vi, vr2)) > 0.3,
           f"{f_iso(vi, vr):.3f} vs {f_iso(vi, vr2):.3f}——式(2.83) 保留 |φr-φi| 为自变量")
    # 各向异性反例：拉丝铝——f 额外依赖入射光相对划痕（x 轴）的绝对方位 φi
    def f_aniso(a, b):
        return f_iso(a, b) + 1.4 * (a[0] / np.linalg.norm(a)) ** 2
    spread = max(abs(f_aniso(Rz(al) @ vi, Rz(al) @ vr) - f_aniso(vi, vr))
                 for al in np.linspace(0, 2 * np.pi, 25))
    verify("各向异性反例（划痕沿 x 轴，f 依赖 φi 绝对方位）：同一旋转扫描下波动显著",
           spread > 0.3, f"波动 {spread:.3f}——式(2.83) 的简化不是白给的")


# ---------------------------------------------------------------- #5 反射方程
def assert_reflectance_equation():
    """#5 反射方程（p.69 式 2.84→2.86）：把三个离散光源换成球面高斯窄光束做数值积分
    （黎曼和），束宽 σ=6°→3°→1.5° 逐档收敛到式(2.86) 求和；Lambertian（fd 常数）时
    与式(2.88) 相等；漏掉 cos⁺ 前缩的错误对照同一检验立即偏离（检验有区分力）。
    视线 v_r 对准光源 1（离法线 0.5 rad，其镜面方向在上半球）的镜面方向，使方向依赖项主导。"""
    n_hat = np.array([0.0, 0.0, 1.0])
    lights = [np.array([np.cos(t) * np.sin(el), np.sin(t) * np.sin(el), np.cos(el)])
              for t, el in [(0.3, 0.9), (2.1, 0.5), (4.4, 1.1)]]
    Lk = [0.8, 0.5, 0.3]
    rho = 0.42
    v_r = 2 * float(np.dot(n_hat, lights[1])) * n_hat - lights[1]   # 光源 1 的镜面方向（式 2.90）

    nt, npn = 361, 720
    th = np.linspace(0, np.pi, nt)
    ph = np.linspace(0, 2 * np.pi, npn, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    dirs = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1)
    domega = np.sin(T) * (th[1] - th[0]) * (ph[1] - ph[0])
    cosup = np.maximum(0.0, dirs @ n_hat)

    def beam_field(sigma):
        """三个高斯窄束，各自数值归一到总强度 L_k（∫ Li dω = Σ L_k）。"""
        f = np.zeros(T.shape)
        for L, d in zip(Lk, lights):
            ad = np.arccos(np.clip(dirs @ d, -1, 1))
            g = np.exp(-(ad / sigma) ** 2)
            f += L * g / np.sum(g * domega)
        return f

    def fr_field(vr):
        # 式(2.91) 型非常数 BRDF：fd + 0.6·cos^8∠(vr, ŝ)，ŝ 按式(2.90) 逐方向计算
        s = 2 * (dirs @ n_hat)[:, :, None] * n_hat - dirs
        return rho + 0.6 * np.maximum(0.0, np.einsum("ijk,k->ij", s, vr)) ** 8

    sum86 = sum(L * (rho + 0.6 * max(0.0, float(np.dot(v_r, 2 * float(np.dot(n_hat, d)) * n_hat - d))) ** 8)
                * max(0.0, float(np.dot(d, n_hat))) for L, d in zip(Lk, lights))
    sum88 = rho * sum(L * max(0.0, float(np.dot(d, n_hat))) for L, d in zip(Lk, lights))

    ratios = []
    for sigma in np.radians((6.0, 3.0, 1.5)):
        f = beam_field(sigma)
        ratios.append(float(np.sum(f * fr_field(v_r) * cosup * domega) / sum86))
    verify("式(2.84) 数值积分（窄光束 σ=1.5°，v_r 对准镜面方向）≈ 式(2.86) 求和（δ 光源极限，偏差 <3%）",
           abs(ratios[2] - 1) < 0.03, f"比值 {ratios[2]:.4f}")
    verify("收敛方向：光束 σ=6°→3°→1.5° 逐档收窄，对式(2.86) 的偏差单调减小",
           abs(ratios[2] - 1) < abs(ratios[1] - 1) < abs(ratios[0] - 1),
           " → ".join(f"σ={s}° 偏差 {abs(r - 1):.4f}" for s, r in zip((6, 3, 1.5), ratios)))
    f = beam_field(np.radians(1.5))
    r88 = float(np.sum(f * rho * cosup * domega) / sum88)
    verify("Lambertian（fd=0.42 常数）时积分 = 式(2.88) fd·Σ Li cos⁺θi（偏差 <3%）",
           abs(r88 - 1) < 0.03, f"比值 {r88:.4f}——「replace the integral with a summation」")
    wrong = float(np.sum(f * fr_field(v_r) * domega) / sum86)   # 漏掉 cos⁺ 前缩的错误对照
    verify("检验有区分力：漏掉 cos⁺ 前缩（无权重的全球积分）时同一检验立即偏离 1",
           abs(wrong - 1) > 0.03, f"错误对照比值 {wrong:.4f}（正确实现 {ratios[2]:.4f}）")


# ---------------------------------------------------------------- #6 镜面方向
def assert_specular_direction():
    """#6 镜面反射方向（p.71 式 2.90）：法向分量保留 + 切向分量取反 = 绕法线转 180°；
    2nnᵀ-I 与 Rodrigues θ=π 旋转矩阵逐元相等且 det=+1——全部检验跑在凑满的 64 组有效样本上。"""
    rng = np.random.default_rng(2)
    worst = worst_rodr = 0.0
    got = 0
    while got < 64:
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        n = rng.normal(size=3); n /= np.linalg.norm(n)
        if abs(float(np.dot(v, n))) > 0.95:
            continue
        got += 1
        s = 2 * np.outer(n, n) @ v - v                       # (2nnᵀ-I)v，式(2.90)
        tan_v = v - float(np.dot(v, n)) * n                  # 切向分量（垂直于法线，式 2.29–2.30 记号 v⊥）
        tan_s = s - float(np.dot(s, n)) * n
        worst = max(worst,
                    abs(np.linalg.norm(s) - 1),                            # 单位长度
                    abs(float(np.dot(s, n)) - float(np.dot(v, n))),        # 法向分量保留
                    abs(float(np.dot(s, np.cross(n, v)))),                 # 在入射面内
                    float(np.linalg.norm(tan_s + tan_v)))                  # 切向分量取反
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])   # [n]×
        R = np.eye(3) + 2 * (K @ K)                          # Rodrigues θ=π：I + 2[n]×²
        worst_rodr = max(worst_rodr,
                         float(np.abs(R - 2 * np.outer(n, n) + np.eye(3)).max()),
                         abs(np.linalg.det(R) - 1.0))
    verify("式(2.90)：‖ŝ‖=1、ŝ·n=v·n（法向分量保留）、ŝ⊥(n×v)、切向取反 ‖ŝ_t+v_t‖=0——64 组有效随机 (v,n)",
           worst < 1e-12, f"最坏偏差 {worst:.2e}——法向不变 + 切向翻转 = 绕法线转 180°")
    verify("矩阵 2nnᵀ-I = 绕 n 转 180° 的 Rodrigues 旋转 I+2[n]×²（64 组随机 n 逐元相等且 det=+1，是旋转不是镜像）",
           worst_rodr < 1e-12, f"最坏偏差 {worst_rodr:.2e}")


# ---------------------------------------------------------------- #7 Phong 剖面
def assert_phong_profile():
    """#7 Phong 剖面（p.71 式 2.91–2.92；参照 p.72 Figure 2.18 设置、按讲者视线角口径重绘）：
    光源离法线 +30°，镜面方向由式(2.90) 对光源向量真实计算（ŝ=(2nnᵀ-I)v̂i → 极角 -30°）；
    ke=10/100/1000 的剖面峰都在视线对准镜面方向处；完整对称剖面的半高宽随 ke 单调收窄；
    高斯式(2.92) 同峰位。（断言只验峰位/半高宽等讲者口径下的量，不验原书图形态）"""
    light_off = np.radians(30)                 # 光源方向离法线 +30°（Figure 2.18）
    n = np.array([0.0, 0.0, 1.0])
    v_i = np.array([np.sin(light_off), 0.0, np.cos(light_off)])
    spec_dir = mirror_polar(v_i, n)                      # 式(2.90)：ŝ = (2nnᵀ-I)v̂i 的极角
    verify("式(2.90) 作用于光源 +30°：镜面方向极角 = -30°（法线另一侧，非手写负号）",
           abs(np.degrees(spec_dir) + 30.0) < 1e-9, f"θs = {np.degrees(spec_dir):+.6f}°")
    theta = np.linspace(-np.pi / 2, np.pi / 2, 9001)   # 视线离法线的极角（全范围）
    fwhm = []
    for ke in (10.0, 100.0, 1000.0):
        y = np.where(np.abs(theta - spec_dir) <= np.pi / 2, np.cos(theta - spec_dir) ** ke, 0.0)
        peak = theta[np.argmax(y)]
        half = y >= y.max() / 2
        fwhm.append(np.degrees(theta[half].max() - theta[half].min()))
        verify(f"Phong ke={int(ke)}：峰在视线对准镜面方向 -30°（光源 +30° 的镜像），半高宽 {fwhm[-1]:.2f}°",
               abs(peak - spec_dir) < 1e-3, f"峰位 {np.degrees(peak):+.2f}°")
    verify("半高宽随 ke 单调收窄（高光越指数越锐）",
           fwhm[0] > fwhm[1] > fwhm[2], "/".join(f"{w:.2f}°" for w in fwhm))
    g = np.exp(-9.0 * (theta - spec_dir) ** 2)          # 式(2.92)：cs=3
    gp = theta[np.argmax(g)]
    verify("微面元高斯（式 2.92，cs=3）峰位同样在 -30°",
           abs(gp - spec_dir) < 1e-3, "cos^ke 与高斯都把光聚在镜面方向")


# ---------------------------------------------------------------- #8 Phong 合成
def assert_phong_compose():
    """#8 Phong 合成的颜色规律（p.71 式 2.93–2.94；p.72）：非平坦光谱下验证——
    ① ks 为常数（界面反射不染色）→ 高光路光谱形状 = 光源谱：视线扫过镜面方向 ±20° 的
    21 个位置，spec/Li 逐视线逐波长恒定（高光颜色 = 光源颜色）；
    ② p.72「ka(λ) and kd(λ) are the same」→ 环境、漫反射两路乘同一材质谱：各自除以
    自己的光源谱后（amb/La 与 diff/(Li·cos⁺)）逐波长相同——且因 La≠Li 两路表观颜色
    仍不同（环境偏蓝、漫反射暖色，≠「两路同色」）；kd≠ka 反例同一检验立即失配；
    ③ 有色 ks（金）反例：① 的检验立即失配——检验有区分力。"""
    n = np.array([0.0, 0.0, 1.0])
    v_i = np.array([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))])  # 光源 30°
    s_dir = mirror_vec(v_i, n)                                             # 镜面方向（式 2.90）
    lam = np.linspace(400, 700, 31)
    Li = 0.55 + 0.45 * np.exp(-((lam - 610) / 55) ** 2)   # 暖色直射光谱（非平坦）
    La = 0.40 + 0.30 * np.exp(-((lam - 460) / 60) ** 2)   # 偏蓝环境光谱（p.72：La 常与 Li 色调不同）
    ka = 0.5 + 0.2 * np.sin((lam - 400) / 100)            # 材质谱（与 La/Li 均不同形）
    kd = ka.copy()                                        # p.72：ka 与 kd 相同（典型情形）
    ks_const = 0.9                                        # p.72：界面反射不染色（常数标量）
    ke = 20.0
    dotin = max(0.0, float(np.dot(v_i, n)))               # 光源入射角 30° → cos⁺ = 0.866

    def spec_of(vr, ks):
        return ks * Li * max(0.0, float(np.dot(vr, s_dir))) ** ke

    def amb_of(mat=ka):
        return mat * La                                    # 环境路（ka 真实参与）

    def diff_of(mat=kd):
        return mat * Li * dotin                            # 漫反射路（kd 真实参与）

    # ①③ 高光路：视线沿镜面方向 ±20° 扫 21 个位置（cos^20 从 1 降到 ≈0.29，信号不弱）
    rot_axis = np.array([0.0, 1.0, 0.0])
    def rot_y(v, a):
        c, s = np.cos(a), np.sin(a)
        return c * v + s * np.cross(rot_axis, v) + (1 - c) * float(np.dot(rot_axis, v)) * rot_axis
    view_angles = np.linspace(-np.radians(20), np.radians(20), 21)
    worst_spec = max(float(np.ptp(spec_of(rot_y(s_dir, a), ks_const) / Li)) for a in view_angles)
    verify("演示（式 2.94，ks 常数）：镜面方向 ±20° 扫 21 个视线，高光路光谱/Li 逐波长恒定——高光颜色 = 光源颜色（界面反射不染色，p.72）",
           worst_spec < 1e-12, f"最大形状偏差 {worst_spec:.2e}")
    ks_gold = 0.55 + 0.35 * np.exp(-((lam - 620) / 70) ** 2)   # 有色 ks（铜金类界面反射）
    worst_bad = max(float(np.ptp(spec_of(rot_y(s_dir, a), ks_gold) / Li)) for a in view_angles)
    verify("检验有区分力：把 ks 换成有色谱（金），同一检验（同一批视线）立即失配",
           worst_bad > 0.05, f"最大形状偏差 {worst_bad:.3f}（恒定 ks 下 <1e-12）")

    # ② ka=kd：两路各自除以自己的光源谱后形状相同；kd≠ka 反例失配；La≠Li 下两路表观不同色
    shape_gap = float(np.ptp(amb_of() / La - diff_of() / (Li * dotin)))          # = ptp(ka−kd) = 0
    kb = 0.45 + 0.35 * np.exp(-((lam - 530) / 45) ** 2)
    gap_bad = float(np.ptp(amb_of() / La - diff_of(kb) / (Li * dotin)))          # = ptp(ka−kb)，反例走同一函数
    apparent = float(np.ptp((amb_of() / diff_of())))                             # La≠Li → 比值随波长变
    verify("演示（p.72 ka=kd）：环境路/La 与漫反射路/(Li·cos⁺) 逐波长相同（两路共用同一材质谱）；"
           "kd 换成 kb 后同一检验立即失配；且 La≠Li 使两路表观光谱比值不恒定——两路并不同色（环境偏蓝、漫反射暖色）",
           shape_gap < 1e-12 and gap_bad > 0.05 and apparent > 0.05,
           f"ka=kd 偏差 {shape_gap:.2e}；kb 反例偏差 {gap_bad:.3f}；表观比值峰谷 {apparent:.3f}")


# ---------------------------------------------------------------- #9 朗伯球
def assert_lambert_sphere():
    """#9 朗伯球着色（p.70–71 式 2.88）：亮暗结构四条性质。
    I(n) = fd·[v̂i·n̂]⁺（fd=1）；光源方位绕球的 z 轴（经度方向）旋转 α 后，
    亮度图案沿经度平移 α——式 2.88 只依赖 n·L。末条与 draw_lambert_sphere (a) 图
    共用 lambert_light/lambert_render，直接读取渲染像素做检验（附 45° 反例）。"""
    res = 241
    # 方位角取 res-1=240 个点（不含 2π，endpoint=False）：步长恰 1.5°，36°=24 步循环移位精确对齐
    u, v = np.meshgrid(np.linspace(0, np.pi, res), np.linspace(0, 2 * np.pi, res - 1, endpoint=False))
    n = np.stack([np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u)], axis=-1)
    tilt = np.radians(30)
    L_dir = np.array([0.0, -np.sin(tilt), np.cos(tilt)])  # 光沿经度 270°、极角 30° 方向
    dots = n @ L_dir
    img = np.maximum(0.0, dots)                            # 式(2.89)：[v̂i·n̂]⁺ = max(0, n·L)，fd=1
    k = int(np.argmax(img))
    # 光方向 (u=30°, v=270°) 恰在网格点上（30°=40 个极角步长 0.75°、270°=180 个方位步长 1.5°），
    # 故最亮点 n·L 与 1 只差浮点误差（判 1−dot < 1e-12，比 arccos 阈值稳健）
    verify(f"朗伯球：最亮点 n·L = {dots.flat[k]:.12f}（1−dot < 1e-12；光方向取在网格点上，应重合）",
           1.0 - dots.flat[k] < 1e-12 and abs(img.flat[k] - dots.flat[k]) < 1e-15,
           f"max={img.max():.6f}")
    band = np.abs(dots) < 5e-3
    wrong = dots                                           # 错误实现：漏 max(0,·)，直接拿 n·L 当亮度
    wrong_leak = float(wrong[dots < -1e-9].min())          # 同一检验下错误实现在背光半球泄漏的负亮度（最深 ≈−1）
    verify("明暗分界（环带 |n·L|<5e-3）：受光侧亮度 >0、背光侧恒 0；漏掉 max(0,·) 的错误实现跑同一检验立即报警（背光半球亮度显著为负）",
           img[band & (dots > 0)].min() > 0 and img[band & (dots < 0)].max() == 0.0
           and img[dots < -1e-9].max() == 0.0 and wrong_leak < -1e-3,
           f"环带两侧翻转 ✓；错误实现背光半球泄漏 {wrong_leak:.2e}（判据 <−1e-3，显著报警）")
    # 轴对称：式 2.88 只依赖 n·L → 光源方位绕 z 轴转 α 后，图案沿经度平移 α、逐点等值
    # 方位网格步长 1.5°，取 α=36° 恰为 24 列，可用循环移位精确对齐
    ang = np.radians(36)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    img2 = np.maximum(0.0, n @ (Rz @ L_dir))
    shift = 24   # meshgrid indexing='xy'：方位角沿 axis 0 变化，故沿 axis 0 循环移位
    verify("光源方位绕球的 z 轴（经度方向）旋转 36° 后，亮度图案沿经度循环平移 24 列逐点等值（式 2.88 只看 n·L）",
           np.allclose(img2, np.roll(img, shift, axis=0), atol=1e-12),
           f"最大差 {np.abs(img2 - np.roll(img, shift, axis=0)).max():.2e}")
    # 渲染一致性（与 (a) 图共用 lambert_light/lambert_render——检验读取渲染像素而非解析式）：
    # 沿 ±ux 两条径向线取像素（最近邻），ρ=sinθ 换算极角，与 cos⁺(θ∓30°) 比较（容差按像素
    # 尺寸设定）；背光侧首个零亮度像素即明暗分界，应落在 60°±1°（相对光方向恒 90°，亮侧
    # 分界在背面 120° 盘内不可见）；反例：光改离视线 45°，同一像素检验立即失配
    Lren = lambert_light()
    Iren = lambert_render(Lren)
    ux2 = Lren[:2] / np.linalg.norm(Lren[:2])              # 盘内「亮侧」方位（指向光源横向）

    def radial_profile(Iimg, ux, uy):
        """从盘心沿 ±(ux,uy) 两条半径取渲染像素（最近邻），ρ=sinθ 换算成极角。"""
        res = Iimg.shape[0]
        step = 2 * 1.05 / (res - 1)
        c = (res - 1) / 2
        ts = np.arange(0.0, 0.98, step / 2)
        th = np.degrees(np.arcsin(ts))
        pb = Iimg[np.round(c + ts * uy / step).astype(int), np.round(c + ts * ux / step).astype(int)]
        pk = Iimg[np.round(c - ts * uy / step).astype(int), np.round(c - ts * ux / step).astype(int)]
        return th, pb, pk

    tol = 0.03    # 像素误差界：最近邻位移≈0.0035 × 亮度梯度最大≈3.8（ρ=0.98 处）≈0.013，取约 2 倍余量
    def profile_dev(Iimg):
        th, pb, pk = radial_profile(Iimg, ux2[0], ux2[1])
        eb = float(np.abs(pb - np.maximum(0.0, np.cos(np.radians(th - 30)))).max())
        ek = float(np.abs(pk - np.maximum(0.0, np.cos(np.radians(th + 30)))).max())
        zk = th[pk <= 1e-12]
        return eb, ek, (float(zk[0]) if zk.size else 999.0)

    eb, ek, z0 = profile_dev(Iren)
    L45 = np.array([-np.sin(np.radians(45)) * np.cos(np.radians(45)),
                    np.sin(np.radians(45)) * np.sin(np.radians(45)), np.cos(np.radians(45))])
    eb_bad, ek_bad, z0_bad = profile_dev(lambert_render(L45))   # 反例：光离视线 45°（同方位）
    verify("渲染一致性（与 (a) 图共用 lambert_light/lambert_render，检验读取渲染像素）："
           "两条经线剖面 = cos⁺(θ−30°)/cos⁺(θ+30°)（像素容差内），背光分界像素在 60°±1°"
           "（相对光方向恒 90°，亮侧分界在背面 120° 盘内不可见）；光改离视线 45° 的反例同一检验立即失配",
           eb < tol and ek < tol and abs(z0 - 60.0) < 1.0
           and (eb_bad >= tol or ek_bad >= tol or abs(z0_bad - 60.0) >= 1.0),
           f"剖面最大差 {eb:.4f}/{ek:.4f}（容差 {tol}）；背光分界 {z0:.2f}°；"
           f"45° 反例剖面差 {max(eb_bad, ek_bad):.3f}、分界 {z0_bad:.1f}°（应报警）")


# ---------------------------------------------------------------- #10 双色反射
def assert_dichromatic():
    """#10 双色反射（p.73 式 2.95–2.96）：光谱×几何分离，表观光谱矩阵秩 ≤2；
    再加第三条独立光谱路（有色环境光）秩升到 3——检验有区分力。"""
    lam = np.linspace(400, 700, 41)
    ci = 0.9 * np.ones_like(lam)                         # 界面反射光谱（常白：白光下不染色，p.72）
    cb = 0.8 * np.exp(-((lam - 610) / 50) ** 2)          # 体反射光谱（偏红的体色）
    rng = np.random.default_rng(11)
    rows = []
    for _ in range(200):
        g1, g2 = rng.uniform(0.1, 1.0, size=2)
        mi, mb = g1, 0.6 * g2                             # 任意几何项
        rows.append(ci * mi + cb * mb)
    M = np.stack(rows)                                    # 200 几何 × 41 波长
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    verify("双色模型：200 组几何的表观光谱矩阵（未中心化）秩 ≤2（两条独立光谱路）",
           (s > s.max() * 1e-10).sum() <= 2,
           "奇异值前 4：" + "/".join(f"{x:.2e}" for x in s[:4]) + f"（第 3 个 = {s[2]:.1e}）")
    # 任意几何的表观光谱都落在 ci、cb 张成的二维子空间内（几何只改两路权重）
    basis = np.stack([ci / np.linalg.norm(ci), cb / np.linalg.norm(cb)])
    q, _ = np.linalg.qr(basis.T)
    resid = M - (q @ (q.T @ M.T)).T
    verify("任意几何下表观光谱均在 {ci, cb} 二维子空间内（投影残差 ≈0，颜色只随两路权重变）",
           float(np.abs(resid).max()) < 1e-12, f"最大投影残差 {float(np.abs(resid).max()):.2e}")
    ca = 0.7 * np.exp(-((lam - 460) / 60) ** 2)           # 第三条独立光谱路（偏蓝环境光）
    rows3 = []
    for _ in range(200):
        mi, mb, ma = rng.uniform(0.1, 1.0, size=3)
        rows3.append(ci * mi + cb * mb + ca * ma)
    s3 = np.linalg.svd(np.stack(rows3), full_matrices=False)[1]
    rank3 = int((s3 > s3.max() * 1e-10).sum())
    verify("检验有区分力：再加第三条光谱路（ca·ma）后秩升至 3，同一检验立即报警",
           rank3 >= 3, f"三路模型奇异值秩 = {rank3}")


# ---------------------------------------------------------------- #11 辐射度 toy
def assert_radiosity_toy():
    """#11 全局光照最小系统（p.73–74）：形状因子互易（两块有限面片双尺度独立数值积分）+
    辐射度线性方程组（多次反弹稳态）与不动点迭代解一致。"""
    # 两块平行正对矩形面片：1 号 1×1（z=0，法向 +z）；2 号 0.5×0.5（z=1，法向 -z）
    def grid(n, half, z):
        # 中点采样（每点代表以它为中心的 h² 面元）：O(h²) 求积误差，两组网格差异更小
        h = 2 * half / n
        g = np.linspace(-half + h / 2, half - h / 2, n)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)], axis=-1)
        return pts, h * h

    p1f, dA1f = grid(48, 0.5, 0.0)            # 面片 1 细网格
    p1c, dA1c = grid(13, 0.5, 0.0)            # 面片 1 粗网格
    p2f, dA2f = grid(48, 0.25, 1.0)           # 面片 2 细网格
    p2c, dA2c = grid(13, 0.25, 1.0)           # 面片 2 粗网格
    A1 = len(p1f) * dA1f                      # 离散面积（两组网格一致使用）
    A2 = len(p2f) * dA2f
    A = np.array([A1, A2])

    def kernel(q_src, q_dst):
        """cosθ_src·cosθ_dst/(π r²)：+z 法向的源对 -z 法向的目标（平行正对，全程可见）。"""
        d = q_dst - q_src
        r2 = float(d @ d)
        cz = abs(d[2]) / np.sqrt(r2)          # 两个 cosθ 都等于 |dz|/r
        return cz * cz / (np.pi * r2)

    def double_sum(src, dsrc, dst, ddst):
        acc = 0.0
        for a in src:
            for b in dst:
                acc += kernel(a, b)
        return acc * dsrc * ddst

    # 两个方向各用一套「源粗 × 目标细」的独立二次求积：F12 用粗1×细2，F21 用粗2×细1。
    # 被积核 cos²θ/(πr²) 对 (a,b) 按构造对称——故互易断言实质检验的是「两套独立求积
    # （不同源/目标网格）各自收敛到同一能量交换值」，这正是 radiosity 实现里的真风险点
    F12 = double_sum(p1c, dA1c, p2f, dA2f) / A1
    F21 = double_sum(p2c, dA2c, p1f, dA1f) / A2
    worst_ff = abs(F12 * A[0] - F21 * A[1])
    verify("形状因子互易：双尺度独立求积 F12·A1 = F21·A2（p.74 radiosity 能量交换对称性）",
           0 < F12 < 1 and 0 < F21 < 1 and worst_ff < 5e-4,
           f"F12={F12:.5f}、F21={F21:.5f}、|F12·A1-F21·A2|={worst_ff:.2e}（两次独立二次求积的离散误差量级）")
    rho = np.array([0.7, 0.5])
    E = np.array([1.0, 0.0])                              # 面片 1 是光源
    Fmat = np.array([[0.0, F12], [F21, 0.0]])
    Mrho = rho[:, None] * Fmat                            # (ρF)ᵢⱼ = ρᵢ·Fᵢⱼ（行乘，逐面片反照率）
    # 辐射度方程 B = E + ρ·F·B 的直接解（多次反弹的稳态）
    B_direct = np.linalg.solve(np.eye(2) - Mrho, E)
    # 不动点迭代
    B = E.copy()
    for _ in range(200):
        B = E + rho * (Fmat @ B)
    verify("两块面片辐射度：线性方程组直接解 = 不动点迭代解（多次反弹收敛）",
           np.allclose(B_direct, B, atol=1e-12),
           f"B=({B_direct[0]:.4f},{B_direct[1]:.4f})；面片 2 无直射光也被照亮 {B[1]:.4f}（互反射）")
    verify("能量语义：只发光的面片 1 辐射度高于自发光 E（收到回弹光）",
           B_direct[0] > E[0], f"{B_direct[0]:.4f} > 1.0")


# ---------------------------------------------------------------- 图
def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / name, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"[fig] {name}")


def draw_fig2_14():
    """Figure 2.14（p.66）重绘：光度成像简化模型。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    # 光源
    ax.scatter([1.6], [4.9], s=420, color=ORANGE, zorder=5, marker="*")
    ax.text(1.6, 5.45, "光源", ha="center", fontsize=13, color=INK, fontweight="bold")
    for y in np.linspace(4.1, 3.4, 3):
        ax.annotate("", xy=(4.3, y - 0.55), xytext=(1.9, y),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
    # 表面（物体）
    ax.add_patch(plt.Rectangle((4.3, 1.4), 1.7, 2.4, fc="#e8dcc4", ec=BROWN, lw=1.6))
    ax.text(5.15, 0.95, "物体表面", ha="center", fontsize=12, color=INK)
    # 相机（光学+传感器）
    ax.add_patch(plt.Rectangle((7.6, 2.6), 1.9, 1.3, fc="#e5e9f2", ec=BLUE, lw=1.6))
    ax.text(8.55, 3.9, "相机：光学 + 传感器", ha="center", fontsize=11.5, color=BLUE, fontweight="bold")
    ax.text(8.55, 3.0, "像素 = 数", ha="center", fontsize=12, color=INK)
    ax.annotate("", xy=(7.55, 3.0), xytext=(6.1, 2.7),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8))
    ax.text(7.5, 2.25, "朝向相机的反射光", ha="right", fontsize=11, color=BLUE)
    ax.text(5.0, 0.4, "简化假设：光一次反射即到相机；多次反射不在本图内（p.66 Figure 2.14 简化重绘 · 原书图未使用）",
            ha="center", fontsize=10.5, color=BROWN)
    ax.set_title("光度成像简化模型：光源 → 表面散射 → 朝向相机的反射光（式 2.81–2.96 的舞台）",
                 fontsize=13.5, color=INK, pad=10)
    _save(fig, "e018_fig2_14.png")


def draw_phong_profile():
    """参照 Figure 2.18（p.72）设置、按讲者视线角口径重绘：Phong 分量剖面 + 极坐标。

    讲者口径：横轴为视线离法线的极角，光源固定在离法线 +30°（Figure 2.18 图注），
    镜面方向由式(2.90) 推出在法线另一侧（mirror_polar 同源）——此口径下 Ambient 与
    Diffuse 不随视角变（水平线），Specular 峰在视线对准镜面方向处。原书图注只写
    「离法线的角度」未细说作图口径，其 (a) 中 Diffuse 呈宽钟形——两图形态以各自
    口径为准，不互指（deck 页11 note 已注明）。
    """
    theta = np.deg2rad(np.linspace(-90, 90, 1200))      # 视线离法线的极角
    light_off = np.radians(30)                           # 光源离法线 +30°（Figure 2.18）
    n_z = np.array([0.0, 0.0, 1.0])                      # 与实验 #7 同一法线
    v_light = np.array([np.sin(light_off), 0.0, np.cos(light_off)])
    spec_dir = mirror_polar(v_light, n_z)                # 式(2.90) 真实计算（与 #7 同源，非手写）
    deg_l, deg_s = np.degrees(light_off), np.degrees(spec_dir)   # 图上角度文本全部由此生成
    amb, kdli, ks_amp = 0.25, 0.45, 0.68                 # 三分量幅度：环境 / kd·Li·cos⁺30° / 镜面峰
    diffuse_h = kdli * float(np.cos(light_off))          # 漫反射水平线高度 ≈ 0.39
    fig = plt.figure(figsize=(9.4, 4.3))
    a1 = fig.add_subplot(121)
    a1.plot(np.degrees(theta), np.full_like(theta, amb), color=PURPLE, lw=2.2,
            label="Ambient（常数，式 2.93）")
    a1.plot(np.degrees(theta), np.full_like(theta, diffuse_h), color=GREEN, lw=2.2,
            label="Diffuse（不随视角，式 2.88）")
    for ke, c in [(10, ORANGE), (100, BLUE), (1000, INK)]:
        spec = np.where(np.abs(theta - spec_dir) <= np.pi / 2,
                        np.cos(theta - spec_dir) ** ke, 0.0) * ks_amp
        a1.plot(np.degrees(theta), spec, color=c, lw=1.8, label=f"Specular Exp={ke}")
    a1.axvline(np.degrees(light_off), color=ORANGE, ls=":", lw=1.3)
    a1.axvline(np.degrees(spec_dir), color=BLUE, ls="--", lw=1.3)
    a1.annotate(f"镜面方向 {np.degrees(spec_dir):+.0f}°", xy=(np.degrees(spec_dir) - 1.5, 0.03),
                xytext=(np.degrees(spec_dir) - 1.5, 0.03),
                ha="right", va="center", fontsize=9.5, color=BLUE)
    a1.annotate(f"光源 {np.degrees(light_off):+.0f}°", xy=(np.degrees(light_off) + 1, 0.10),
                xytext=(np.degrees(light_off) + 4, 0.10),
                ha="left", va="center", fontsize=9.5, color=ORANGE)
    a1.set_xlim(-90, 90); a1.set_ylim(-0.03, 0.90)
    a1.set_xlabel("视线离法线的极角（度）"); a1.set_ylabel("相对亮度")
    a1.set_title(f"(a) 分量剖面（横轴=视线离法线的角度）：光源固定在离法线 {deg_l:+.0f}°，\n"
                 f"镜面峰在法线另一侧 {deg_s:+.0f}°（参照 Figure 2.18 设置 · 按视线角口径重绘 · 原书图未使用 · 纵轴归一化）",
                 fontsize=12, color=INK)
    a1.legend(fontsize=9, loc="upper right"); a1.grid(alpha=0.25)

    a2 = fig.add_subplot(122, projection="polar")
    th = np.linspace(-np.pi / 2, np.pi / 2, 721)         # 只画表面法向半球（θ∈[-90°,90°]，与原书 (b) 同口径）
    a2.set_theta_zero_location("N")                      # 法线朝上（与原书 (b) 同口径）；+角度顺时针→与 (a) 横轴同向
    a2.set_theta_direction(-1)
    base = amb + diffuse_h                               # 环境+漫反射基座（都不随视角）≈ 0.64
    for ke, c in [(10, ORANGE), (100, BLUE), (1000, INK)]:
        r = np.where(np.abs(th - spec_dir) <= np.pi / 2,
                     np.cos(th - spec_dir) ** ke, 0.0) * ks_amp + base
        a2.plot(th, r, color=c, lw=1.6, label=f"Exp={ke}")
    a2.plot(th, np.full_like(th, base), color=GREEN, lw=1.7, ls="--",
            label="环境+漫反射（与视角无关）")
    a2.annotate("", xy=(light_off, 1.22), xytext=(light_off, 0.58),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    a2.text(light_off, 1.29, f"光源 {deg_l:+.0f}°", ha="center", fontsize=10, color=ORANGE)
    a2.text(0.14, 1.24, "法线", ha="center", fontsize=10, color=INK)
    a2.set_rlim(0, 1.40)
    a2.set_title(f"(b) 极坐标（只画法向半球）：环境+漫反射是\n常数基座弧，镜面瓣在 {deg_s:+.0f}°（光源 {deg_l:+.0f}° 的镜像）",
                 fontsize=12, color=INK, va="bottom")
    a2.set_rlabel_position(200)
    a2.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0.02, 0.02))   # 下半圆无数据（只画法向半球），避开左上的镜面瓣
    fig.subplots_adjust(wspace=0.28, top=0.84)
    _save(fig, "e018_phong_profile.png")


def draw_lambert_sphere():
    """讲者设计：朗伯球正交投影渲染（式 2.88）与双经线亮度剖面。

    (b) 沿两条经线剖开亮度（横轴：法向离视线的极角 θ）——含光源经线 cos⁺(θ−t) 与
    背光经线 cos⁺(θ+t)，t 为 lambert_light() 的光离视线极角：n·L=0 相对光方向恒在 90°，
    但斜照让它在「离视线」坐标里随方位转动——亮侧经线的分界在 θ=90°+t（已转入背面，
    盘内不可见），背光侧 θ=90°−t 就进入可见区（对应 (a) 盘内黑区）；θ>90° 为背面
    （自遮挡，不可见）。图上角度文本由 t 生成，边界亮度与方位词分别由 tilt/L 推导。断言 #9 末条共用本图的
    lambert_light/lambert_render 做像素级守护。"""
    L = lambert_light()
    I = lambert_render(L)
    fig = plt.figure(figsize=(11.2, 4.6))
    ax1 = fig.add_subplot(121)
    ax1.imshow(I, cmap="magma", origin="lower", extent=[-1.05, 1.05, -1.05, 1.05])
    ax1.axis("off")
    az = float(np.degrees(np.arctan2(L[1], L[0])))         # 光源方位角（x→y），lambert_light() 为 135°
    hword = "左" if np.cos(np.radians(az)) < 0 else "右"    # 方位词由 L 推导，非手写
    vword = "上" if L[1] > 0 else "下"
    ax1.set_title(f"(a) 朗伯球：$I = f_d\\,[\\hat{{v}}_i\\!\\cdot\\!\\hat{{n}}]_+$，取 $f_d{{=}}1$（光从{hword}{vword}）\n"
                  "程序渲染 · 原书图未使用", fontsize=12, color=INK)
    tilt = float(np.arctan2(np.hypot(L[0], L[1]), L[2]))   # 光离视线（z 轴）的极角，由 lambert_light() 推导
    tdeg = int(round(np.degrees(tilt)))                    # ≈30：图上全部角度文本由 tdeg 生成（非独立字面量）
    edge90 = float(np.cos(np.pi / 2 - tilt))               # θ=90° 处亮侧经线亮度 cos⁺(90°−t)，由 tilt 推导
    th = np.linspace(0, np.pi, 300)
    deg = np.degrees(th)
    ax2 = fig.add_subplot(122)
    ax2.axvspan(90, 180, color="#b0a48c", alpha=0.16)
    ax2.text(140, 0.88, "θ>90°：背面（自遮挡，不可见）", fontsize=9.5, color=BROWN,
             ha="center", va="center")
    ax2.plot(deg, np.maximum(0.0, np.cos(th - tilt)), color=GREEN, lw=2.4,
             label=f"亮侧经线 $\\cos^+(\\theta\\!-\\!{tdeg}^\\circ)$")
    ax2.plot(deg, np.maximum(0.0, np.cos(th + tilt)), color=BROWN, lw=2.0, ls=":",
             label=f"背光侧经线 $\\cos^+(\\theta\\!+\\!{tdeg}^\\circ)$")
    ax2.annotate(f"可见边界 90°：亮侧边缘剩 $\\approx\\!{edge90:.2f}$",
                 xy=(90, edge90), xytext=(46, edge90 - 0.06), fontsize=9.5, color=INK, ha="left",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax2.annotate(f"背光侧分界 {90 - tdeg}°（→ (a) 盘内黑区）",
                 xy=(90 - tdeg, 0.02), xytext=(94 - tdeg, 0.12), fontsize=9.5, color=BROWN, ha="left",
                 arrowprops=dict(arrowstyle="->", color=BROWN, lw=0.9))
    ax2.annotate(f"亮侧分界 {90 + tdeg}°（已转背面）",
                 xy=(90 + tdeg, 0.02), xytext=(96 + tdeg, 0.30), fontsize=9.5, color=GREEN, ha="left",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
    ax2.set_xlim(0, 180); ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("法向离视线方向的极角（度）"); ax2.set_ylabel("亮度")
    ax2.set_title(f"(b) 亮度沿两条经线剖开（横轴：法向离视线的极角 θ）\n"
                  f"亮侧分界在离视线 ${90 + tdeg}^\\circ$（离光方向恒 $90^\\circ$）；背光侧 ${90 - tdeg}^\\circ$（本图设计）",
                  fontsize=11.5, color=INK)
    ax2.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.98, 0.74))
    ax2.grid(alpha=0.25)
    _save(fig, "e018_lambert_sphere.png")


def draw_dichromatic():
    """讲者设计：双色反射「光谱×几何」分解。"""
    lam = np.linspace(400, 700, 60)
    ci = 0.9 * np.ones_like(lam)                       # 界面反射常白（白光下不染色，p.72）
    cb = 0.8 * np.exp(-((lam - 610) / 50) ** 2)        # 体反射光谱（偏红体色）
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(lam, ci * 0.5, color=BLUE, lw=2.2, label="界面反射路 ci(λ)·0.5（高光路：白光照明下不染色）")
    ax.plot(lam, cb * 1.8, color=ORANGE, lw=2.2, label="体反射路 cb(λ)·1.8（体色路）")
    ax.plot(lam, ci * 0.5 + cb * 1.8, color=INK, lw=2.4, ls="--", label="表观光谱 = 两条曲线逐波长相加（几何 $g_1$）")
    ax.plot(lam, ci * 0.25 + cb * 0.5, color=BROWN, lw=2.0, ls=":", label="另一几何 $g_2$（ci·0.25+cb·0.5）：只变明暗组合，光谱形状同族")
    ax.set_xlabel("波长 λ（nm）"); ax.set_ylabel("相对功率")
    ax.set_title("双色反射模型（式 2.95–2.96）：表观颜色 = 光谱 c(λ) × 几何 m(g) 两路之和\n"
                 "——秩 2 结构：任意几何只是两条光谱的加权组合（本图设计 · 原书图未使用）",
                 fontsize=12.5, color=INK)
    ax.legend(fontsize=9.5); ax.grid(alpha=0.25)
    _save(fig, "e018_dichromatic.png")


# ---------------------------------------------------------------- 同步
def sync_and_md5():
    """同步检查（先比后写）：outputs/ ↔ slides/figures/。"""
    names = ["e018_fig2_14.png", "e018_phong_profile.png", "e018_lambert_sphere.png", "e018_dichromatic.png"]
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
    for fn in (assert_inverse_square, assert_cos_plus, assert_reciprocity, assert_isotropic,
               assert_reflectance_equation, assert_specular_direction, assert_phong_profile,
               assert_phong_compose, assert_lambert_sphere, assert_dichromatic,
               assert_radiosity_toy):
        fn()
    draw_fig2_14()
    draw_phong_profile()
    draw_lambert_sphere()
    draw_dichromatic()
    ok = sync_and_md5()
    n_pass = sum(1 for _, p in RESULTS if p)
    groups = {"#1": 2, "#2": 2, "#3": 2, "#4": 3, "#5": 4, "#6": 2, "#7": 6,
              "#8": 3, "#9": 4, "#10": 3, "#11": 3}
    assert sum(groups.values()) == len(RESULTS), \
        f"groups 总数 {sum(groups.values())} ≠ 实际断言数 {len(RESULTS)}"
    head = f"汇总：{n_pass}/{len(RESULTS)} 数据断言 PASS · 同步检查 {'PASS' if ok else 'FAIL'}"
    print(head + (" —— 全部通过" if n_pass == len(RESULTS) and ok else ""))
    return n_pass == len(RESULTS) and ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
