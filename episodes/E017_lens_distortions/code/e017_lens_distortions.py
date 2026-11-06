"""
E017 · 镜头的谎言：径向畸变与去畸变（§2.1.5，原书第 63–66 页）

数值实验：把径向畸变模型（式 2.77–2.79）、桶形/枕形方向、κ₁κ₂ 阶次分工、
脚注 7 反向式去畸变、鱼眼等距投影（式 2.80）与「校正后线性归来」全部算一遍。

断言（8 项数据 + 1 项同步）：
  #1  式(2.77) 原式结构（p.64）：xc = (rx·p+tx)/(rz·p+tz)、yc = (ry·p+ty)/(rz·p+tz) 逐元验证；
      (xc,yc) 即归一化像平面坐标 p̄_x/p̄_z、p̄_y/p̄_z（p̄=Rp+t，已含外参，未乘 f、未加 (cx,cy)）；
      R=I、t=0 退化为 (px/pz, py/pz)；R=I、t≠0 时 xc≠px/pz——外参平移进入归一化坐标
  #2  桶形/枕形方向（p.64 式(2.78) + 讲者推演，κ₂=0 时）：观测 x̂c = xc(1+κ₁r²c+κ₂r⁴c)。
      κ₁=−0.2 → r=1 处观测半径 0.8<1（拉向中心=桶形）；κ₁=+0.2 → 1.2>1（推离中心=枕形）
      ——原书「displaced towards (barrel) or away (pincushion)」的符号推演；
      子断言：本集镜头（κ₁=−0.2、κ₂=0.05）在网格范围 r≤√2 内括号恒 <1（P1-3 前提）
  #3  κ₁ 低阶 / κ₂ 高阶分工（讲者归纳）：子实验取 κ₁=κ₂=0.05 归一化设定，κ₂/κ₁ 贡献比 = r²，
      r=0.5→0.25、r=1.0→1.0、r=1.5→2.25——r 较小时 κ₁r² 占主导、r 较大时 κ₂r⁴ 权重上升
      （p.64 式(2.78) 阶次结构）；本集镜头比 = (κ₂/κ₁)·r² = −0.25r²（κ₁ 项在网格内始终主导）
  #4  式(2.79) 收尾与 K 一致（p.64）：xs = f·x̂c+cx 等价于「畸变因子乘入后过式(2.59) 单参数 K」
      ——与 K=(f 0 cx;0 f cy;0 0 1) 直接投影逐点一致
  #5  去畸变与脚注 7 的辨析（p.64 式(2.78)/脚注 7 + 讲者补充）：
      (a) 反例断言：同一组 κ（κ₁=−0.2、κ₂=0.05）直接代入脚注 7 反向式
          x = x̂(1+κ₁r̂²+κ₂r̂⁴) 作校正，直线共线残差比不校正还大（约 2 倍级）
          ——脚注 7 是「另一种参数化约定」，不是式(2.78) 的逆，两套 κ 不能混用（门A P0-1）
      (b) 讲者补充的不动点迭代（x ← x̂/(1+κ₁r²+κ₂r⁴)，r 用当前估计）收敛后恢复误差 <1e-10、
          弯度 <1e-9——κ₂≠0 时反解半径为五次方程、一般无根式解，去畸变走数值求逆，
          网格线在机器精度内站直（迭代次数与终止阈值见实跑输出）
      (c) 两套 κ 换算方向：对同一虚拟镜头的 (r,r̂) 数据，反向约定 x=x̂(1+κ₁′r̂²+κ₂′r̂⁴) 的
          最小二乘 κ₁′ 与 κ₁ 异号（讲者推演：一阶近似下 κ₁′≈−κ₁；拟合范围 r∈[0,√2] 全网格、
          同时拟合 κ₂′，拟合值偏离一阶值源于大半径高阶效应）
  #6  鱼眼等距投影与多项式失配（p.65 式(2.80) + 讲者推演）：r = fθ 精确线性（θ=90° 误差 0）；
      讲者推演的结构失配演示：以针孔像点 r_pin=tanθ 为「理想坐标」、等距投影 r*=θ 为目标，
      用径向多项式 r̂=r_pin(1+κ₁r_pin²+κ₂r_pin⁴) 在 θ∈[0°,60°] 最小二乘拟合 κ——
      区间内尚可，外推到 θ=85° 相对误差发散（两位数百分比以上）
      ——「Fisheye lenses require a model that differs from traditional polynomial models」（p.64）
  #7  K 吸收失效（pp.65–66 + 讲者推演设计）：无畸变时 K[R|t] 与 (KR₁)[R₁ᵀR|R₁ᵀt] 投影逐点相同
      （E016 #5 复现，p.56–57）；带径向畸变时同一重参数化产生不同图像（边缘点差异显著）
      ——「cannot post-multiply K with a rotation … absorb into the global rotation」的数值体现
  #8  校正后线性归来（pp.65–66）：3D 平面网格经「线性投影 + 畸变」成像为弯曲像素；
      去畸变（#5b 数值求逆）后与纯线性投影逐点一致（<1e-9），且网格方向向量一致
      ——「converted the sensor back into a linear imager」的数值体现
  #S  同步检查：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步（重跑至一致）

图（4 张，均为讲者程序重绘/设计，原书图未使用）：
  e017_fig2_13.png        Figure 2.13（p.65）三联重绘：(a) 桶形 (b) 枕形 (c) 鱼眼等距投影
  e017_pipeline.png       讲者设计：投影流水线与畸变插入位置（透视除法→畸变→缩放加主点偏移）
  e017_grid_undistort.png 讲者设计：虚拟镜头 κ₁=−0.2、κ₂=0.05 的网格观测与去畸变前后
  e017_r_curve.png        讲者设计：径向响应曲线 r̂/r（桶形/枕形）与鱼眼 r=fθ 对照

运行 `python3 code/e017_lens_distortions.py`，期望输出 `8/8 数据断言 PASS · 同步检查 PASS`。
"""
import hashlib
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 中文字体（同 E014–E016：macOS 优先 Hiragino Sans GB；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
GREEN = "#2a7a4a"
RESULTS = []
EP = pathlib.Path(__file__).resolve().parent.parent   # episodes/E017_lens_distortions


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[verify] {name}  {detail}  [{'PASS' if ok else 'FAIL'}]")


# ---------------------------------------------------------------- 模型（原书式 2.77–2.79、脚注 7、式 2.80）

def project_xc(pc, R, t):
    """式(2.77)（p.64）：xc = (rx·p+tx)/(rz·p+tz)、yc = (ry·p+ty)/(rz·p+tz)——透视除法后、
    f 缩放与 (cx,cy) 平移前的坐标。pc 形状 (N,3) 或 (3,)，返回 (N,2)。"""
    pc = np.atleast_2d(pc)
    q = R @ pc.T + np.asarray(t).reshape(3, 1)          # (3,N)
    return np.column_stack([q[0] / q[2], q[1] / q[2]])


def distort_factor(r2, k1, k2):
    """式(2.78) 括号：1 + κ₁r² + κ₂r⁴。"""
    return 1.0 + k1 * r2 + k2 * r2 * r2


def distort(x, k1, k2):
    """正向畸变（理想→观测，p.64 式(2.78)）：x̂ = x·(1+κ₁r²+κ₂r⁴)，r 从理想坐标算。"""
    r2 = np.sum(np.asarray(x) ** 2, axis=-1, keepdims=True)
    return x * distort_factor(r2, k1, k2)


def undistort_direct(xh, k1, k2):
    """脚注 7 反向式（p.64）：x = x̂(1+κ₁r̂²+κ₂r̂⁴)，r̂ 从观测坐标算——原书表达的直接使用。"""
    r2h = np.sum(np.asarray(xh) ** 2, axis=-1, keepdims=True)
    return xh * distort_factor(r2h, k1, k2)


def undistort_refine(xh, k1, k2, iters=60, tol=1e-14):
    """讲者补充的不动点精化：x = x̂ / (1+κ₁r²+κ₂r⁴)，r 从当前 x 算，脚注 7 反向式作初值。
    收敛后满足正向式(2.78)：distort(x) = x̂（误差 ~1e-15）。返回 (x, 实际迭代次数)。"""
    x = undistort_direct(xh, k1, k2)
    used = 0
    for i in range(iters):
        used = i + 1
        r2 = np.sum(x ** 2, axis=-1, keepdims=True)
        x_new = xh / distort_factor(r2, k1, k2)
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x, used


def fisheye_xy_grid(extent=3.0, n=23, f=1.0, theta_max_deg=88.0):
    """式(2.80)（p.65）：r = fθ 等距投影。把地面方格（z=1 平面上 extent×extent）
    按入射角 θ=arctan(√(X²+Y²)/z) 映到 (r=fθ, φ)，θ 超出视场的点置 NaN。"""
    g = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(g, g)
    Z = 1.0
    theta = np.arctan(np.hypot(X, Y) / Z)
    phi = np.arctan2(Y, X)
    r = f * theta
    inside = theta <= np.deg2rad(theta_max_deg)
    return np.where(inside, r * np.cos(phi), np.nan), np.where(inside, r * np.sin(phi), np.nan)


def warp_grid(ax, k1, k2, color, n=13, half=1.35, lw=1.1):
    """在 ax 上画 κ 畸变后的方格网（理想方格经正向式(2.78) 变形）。"""
    g = np.linspace(-half, half, n * 4)
    X, Y = np.meshgrid(g, g)
    for j in range(0, n * 4, 4):
        for pts in (np.column_stack([g, np.full_like(g, g[j])]),
                    np.column_stack([np.full_like(g, g[j]), g])):
            w = distort(pts, k1, k2)
            ax.plot(w[:, 0], w[:, 1], color=color, lw=lw, solid_capstyle="round")


# ---------------------------------------------------------------- 断言实验

def assert_eq277():
    """#1 式(2.77) 原式结构与平移项（p.64）。"""
    R = np.eye(3)
    t0 = np.zeros(3)
    pc = np.array([[0.5, 0.3, 2.0], [-0.8, 0.6, 1.5], [1.2, -0.4, 3.0]])
    xy0 = project_xc(pc, R, t0)
    verify("式(2.77) R=I,t=0 → (px/pz, py/pz)",
           np.allclose(xy0, pc[:, :2] / pc[:, 2:3], atol=1e-14),
           f"首点 ({xy0[0,0]:.4f}, {xy0[0,1]:.4f})")
    # 含平移/旋转的非平凡情形：逐元硬算对照
    th = 0.31
    Rr = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    t = np.array([0.2, -0.15, 2.0])
    xy1 = project_xc(pc, Rr, t)
    lhs = (Rr @ pc.T + t.reshape(3, 1))
    ref = np.column_stack([lhs[0] / lhs[2], lhs[1] / lhs[2]])
    verify("式(2.77) 逐元（R,t 非平凡）", np.allclose(xy1, ref, atol=1e-14))
    # (xc,yc) 即归一化像平面坐标（已含外参）：R=I、t≠0 时 xc=(px+tx)/(pz+tz)≠px/pz
    xy_t = project_xc(pc, np.eye(3), np.array([0.2, 0.0, 0.0]))
    ref_t = (pc[:, 0] + 0.2) / pc[:, 2]
    d = xy_t[:, 0] - pc[:, 0] / pc[:, 2]
    verify("(xc,yc) 已含外参（平移进入归一化坐标）",
           np.allclose(xy_t[:, 0], ref_t, atol=1e-14) and np.max(np.abs(d)) > 0.02,
           f"xc=(px+tx)/(pz+tz) 逐元成立；Δx 最大 {np.max(np.abs(d)):.4f}")


def assert_barrel_pincushion():
    """#2 桶形/枕形方向（p.64 式(2.78) + 讲者推演）。"""
    x = np.array([[1.0, 0.0]])
    obs_barrel = distort(x, -0.2, 0.0)
    obs_pin = distort(x, +0.2, 0.0)
    verify("κ₁=−0.2 → 观测半径 0.8<1（拉向中心=桶形）",
           abs(obs_barrel[0, 0] - 0.8) < 1e-14, f"|x̂|={abs(obs_barrel[0,0]):.3f}")
    verify("κ₁=+0.2 → 观测半径 1.2>1（推离中心=枕形）",
           abs(obs_pin[0, 0] - 1.2) < 1e-14, f"|x̂|={abs(obs_pin[0,0]):.3f}")
    # 原书方向句：towards (barrel) or away (pincushion)——符号与方向的一致性（讲者推演的依据）
    r = np.linspace(0.05, 1.5, 300)
    rb = r * distort_factor(r ** 2, -0.2, 0.0)
    rp = r * distort_factor(r ** 2, +0.2, 0.0)
    verify("方向单调性：桶形全程 r̂<r、枕形全程 r̂>r", np.all(rb < r) and np.all(rp > r))
    # P1-3 前提子断言：本集镜头（κ₁=−0.2、κ₂=0.05）在网格范围 r≤√2 内括号不翻过 1
    #（r=0 处 g=1；0<r≤√2 内 g<1——κ₂ 项不够大、不反转为枕形）
    r_grid = np.linspace(1e-6, np.sqrt(2), 500)
    g_grid = distort_factor(r_grid ** 2, -0.2, 0.05)
    g_edge = distort_factor(np.array([2.0]), -0.2, 0.05)[0]
    verify("网格范围 0<r≤√2 内括号恒 <1（桶形前提）", np.max(g_grid) < 1.0,
           f"g(0)=1（中心无畸变）；0<r≤√2 g∈[{g_edge:.3f}, 1)——上确界 1 于 r→0 不可达")


def assert_k_order():
    """#3 κ₁ 低阶 / κ₂ 高阶分工（讲者归纳，p.64 式(2.78) 阶次结构）。
    子实验取 κ₁=κ₂=0.05 的归一化设定，此时 (κ₂r⁴)/(κ₁r²) = r²；
    本集镜头（κ₁=−0.2、κ₂=0.05）的比 = (κ₂/κ₁)·r² = −0.25r²（r 至网格角 √2）。"""
    k1 = k2 = 0.05
    for r, ratio_expect in [(0.5, 0.25), (1.0, 1.0), (np.sqrt(2), 2.0)]:
        term1, term2 = k1 * r ** 2, k2 * r ** 4
        verify(f"归一化设定 κ₁=κ₂ 时 κ₂/κ₁ 贡献比 = r² @ r={r:.3f}",
               abs(term2 / term1 - ratio_expect) < 1e-12, f"{term2/term1:.2f}")
    kA, kB = -0.2, 0.05
    vals = {r: (kB * r ** 4) / (kA * r ** 2) for r in (0.5, 1.0, np.sqrt(2))}
    verify("本集镜头 (κ₂r⁴)/(κ₁r²) = (κ₂/κ₁)·r² = −0.25r²",
           all(abs(v - (-0.25 * r * r)) < 1e-12 for r, v in vals.items()),
           "r=0.5/1.0/√2 → " + "/".join(f"{vals[r]:+.4f}" for r in vals)
           + "（|比|≤0.5<1：网格内 κ₁ 项始终主导，κ₂ 仅边缘加权）")


def assert_eq279_K():
    """#4 式(2.79) 与式(2.59) 单参数 K 一致（p.64）。"""
    f, cx, cy = 800.0, 960.0, 540.0
    k1, k2 = -0.2, 0.05
    xc = np.array([[0.10, -0.06], [0.35, 0.22], [-0.28, 0.05]])
    xh = distort(xc, k1, k2)
    xs1 = f * xh + np.array([cx, cy])
    # 等价路径：畸变因子乘入理想像素后过 K
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    r2 = np.sum(xc ** 2, axis=1)
    xh2 = xc * distort_factor(r2, k1, k2)[:, None]
    xs2 = (K @ np.column_stack([xh2, np.ones(len(xh2))]).T).T
    xs2 = xs2[:, :2] / xs2[:, 2:3]
    verify("式(2.79) = 畸变因子 × 式(2.59) K 投影", np.allclose(xs1, xs2, atol=1e-9),
           f"max|Δ|={np.max(np.abs(xs1-xs2)):.2e}")


def assert_undistort():
    """#5 去畸变与脚注 7 的辨析（p.64 式(2.78)/脚注 7 + 讲者补充；门A P0-1 整改）。"""
    k1, k2 = -0.2, 0.05
    g = np.linspace(-1.0, 1.0, 7)
    X, Y = np.meshgrid(g, g)
    ideal = np.column_stack([X.ravel(), Y.ravel()])
    obs = distort(ideal, k1, k2)                        # 虚拟镜头成像（正向式 2.78）

    def bend(pts_grid):
        """网格弯度（缩放/平移不变）：7 行 7 列各自最小二乘直线拟合后的最大点线距离。"""
        worst = 0.0
        for line in list(pts_grid) + list(pts_grid.transpose(1, 0, 2)):
            c = line.mean(axis=0)
            _, _, vt = np.linalg.svd(line - c)
            worst = max(worst, np.abs((line - c) @ vt[1]).max())
        return worst
    grid_obs = obs.reshape(7, 7, 2)
    res_obs = bend(grid_obs)
    # (a) 反例：同一组 κ 直接代入脚注 7 反向式作校正——弯度不降反升（不是式 2.78 的逆；
    #    一阶近似下净映射 h(r)=g(r)·g(r̂)≈g(r)²，非线性约为原来的两倍——门A P0-1 反例断言）
    wrong = undistort_direct(obs, k1, k2)
    res_wrong = bend(wrong.reshape(7, 7, 2))
    verify("反例：同 κ 代脚注 7 反向式，弯度上升（两套约定不能混用）",
           res_wrong > res_obs, f"弯度 {res_obs:.2e} → {res_wrong:.2e}")
    # (b) 讲者补充的不动点迭代（数值求逆）：严格恢复
    rec, used = undistort_refine(obs, k1, k2)
    err = np.max(np.abs(rec - ideal))
    verify("不动点迭代：理想坐标恢复 <1e-10", err < 1e-10,
           f"max|Δ|={err:.2e}；迭代 {used} 次（终止阈值 1e-14，上限 60）")
    res_refine = bend(rec.reshape(7, 7, 2))
    verify("不动点迭代：弯度 <1e-9（机器精度内站直）", res_refine < 1e-9, f"{res_refine:.2e}")
    verify("往返自洽 distort(undistort(x̂))=x̂",
           np.max(np.abs(distort(rec, k1, k2) - obs)) < 1e-10)
    # (c) 两套 κ 换算方向：对同一镜头数据，反向约定的 κ₁′ 与 κ₁ 异号
    r = np.linalg.norm(ideal, axis=1)
    rh = np.linalg.norm(obs, axis=1)
    A = np.column_stack([rh ** 2, rh ** 4])
    b = r / np.maximum(rh, 1e-15) - 1.0                  # r/r̂ − 1 = κ₁′r̂²+κ₂′r̂⁴ 的最小二乘
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    verify("反向约定的 κ₁′ 与 κ₁ 异号（一阶近似 κ₁′≈−κ₁）", coef[0] * k1 < 0,
           f"κ₁={k1} → κ₁′≈{coef[0]:+.3f}（拟合范围 r∈[0,√2] 全网格 49 点、同时拟合 κ₂′={coef[1]:+.3f}；"
           f"κ₁′ 偏离一阶值（+0.2）约 {coef[0] - 0.2:+.2f}，且级数反演理论值 κ₂′=3κ₁²−κ₂=+0.07 而大范围拟合得 {coef[1]:+.3f}"
           f"——大半径处高阶效应显著）")
    return k1, k2


def assert_fisheye():
    """#6 鱼眼等距投影 vs 多项式失配（p.65 式(2.80) + 讲者推演）。"""
    f = 1.0
    th = np.deg2rad(np.array([0.0, 30.0, 60.0, 90.0]))
    r = f * th
    verify("式(2.80) r=fθ 线性（θ=90° 精确）", abs(r[-1] - f * np.pi / 2) < 1e-14,
           f"r(90°)={r[-1]:.4f}")
    # 讲者推演：以针孔像点 r_pin=tanθ 为「理想坐标」，径向多项式拟合等距目标 r*=θ
    th_fit = np.linspace(1e-4, np.deg2rad(60), 400)
    r_pin = np.tan(th_fit)
    target = f * th_fit
    A = np.column_stack([r_pin ** 3, r_pin ** 5])
    b = target - r_pin                                   # κ₁r³+κ₂r⁵ = r*−r_pin
    (k1f, k2f), *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = np.abs(r_pin * distort_factor(r_pin ** 2, k1f, k2f) - target) / target
    verify("径向多项式拟合 [0°,60°] 区间内尚可（<3%）", resid.max() < 0.03,
           f"κ₁={k1f:.3f} κ₂={k2f:.3f} max={resid.max()*100:.2f}%")
    th85 = np.deg2rad(85.0)
    r85 = np.tan(th85)
    model85 = r85 * distort_factor(r85 ** 2, k1f, k2f)
    err85 = abs(model85 - f * th85) / (f * th85)
    verify("外推 θ=85° 相对误差发散（两位数百分比以上）", err85 > 0.5, f"err={err85*100:.0f}%")


def assert_K_absorption():
    """#7 K 吸收失效（pp.65–66 + 讲者推演设计；无畸变部分复现 E016 #5，p.56–57）。"""
    f, cx, cy = 800.0, 960.0, 540.0
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    th = 0.4
    R1 = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    R = np.eye(3)
    t = np.array([0.1, -0.05, 3.0])
    rng = np.random.default_rng(20261017)
    pts = rng.uniform([-1, -1, 0], [1, 1, 2], size=(60, 3)) + np.array([0, 0, 4])

    def proj(Km, Rm, tm, k1=0.0, k2=0.0):
        xc = project_xc(pts, Rm, tm)
        xh = distort(xc, k1, k2)
        xs = (Km @ np.column_stack([xh, np.ones(len(xh))]).T).T
        return xs[:, :2] / xs[:, 2:3]
    # 无畸变：K[R|t] ≡ (KR₁)[R₁ᵀR | R₁ᵀt]（E016 #5，p.56–57）
    p_a = proj(K, R, t)
    p_b = proj(K @ R1, R1.T @ R, R1.T @ t)
    verify("无畸变：K[R|t] ≡ (KR₁)[R₁ᵀR|R₁ᵀt]（p.56–57 复现）",
           np.allclose(p_a, p_b, atol=1e-9), f"max|Δ|={np.max(np.abs(p_a-p_b)):.2e}")
    # 有畸变：同一重参数化改变图像（畸变作用在不同坐标系上）
    k1, k2 = -0.2, 0.05
    p_c = proj(K, R, t, k1, k2)
    p_d = proj(K @ R1, R1.T @ R, R1.T @ t, k1, k2)
    gap = np.max(np.linalg.norm(p_c - p_d, axis=1))
    verify("带畸变：同一重参数化图像不同（K 吸收失效）", gap > 1.0, f"max 位移 {gap:.1f}px")


def assert_linear_return(k1, k2):
    """#8 校正后线性归来（pp.65–66）。"""
    f, cx, cy = 800.0, 960.0, 540.0
    g = np.linspace(-1.0, 1.0, 9)
    X3, Y3 = np.meshgrid(g, g)
    P3 = np.column_stack([X3.ravel(), Y3.ravel(), np.full(X3.size, 2.0)])
    x_lin = project_xc(P3, np.eye(3), np.zeros(3))      # 线性（针孔）像点（归一化平面）
    x_obs = distort(x_lin, k1, k2)                      # 带畸变观测
    x_rec, _ = undistort_refine(x_obs, k1, k2)          # 校正
    err = np.max(np.abs(x_rec - x_lin))
    verify("去畸变后 ≡ 纯线性投影（<1e-9）", err < 1e-9, f"max|Δ|={err:.2e}")
    # 平行性保持：两组对边向量夹角
    def edge_dirs(pts9):
        G = pts9.reshape(9, 9, 2)
        return np.concatenate([(G[:, -1] - G[:, 0]), (G[-1, :] - G[0, :])])
    v_lin, v_rec = edge_dirs(x_lin), edge_dirs(x_rec)
    cosang = np.abs(np.sum(v_lin * v_rec, axis=1) / (np.linalg.norm(v_lin, axis=1) * np.linalg.norm(v_rec, axis=1)))
    verify("网格方向向量与线性投影一致（cos≈1）", np.min(cosang) > 1 - 1e-9,
           f"min cos={np.min(cosang):.12f}")


# ---------------------------------------------------------------- 图（讲者重绘/设计，原书图未使用）

def _style(ax):
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)


def draw_fig2_13(path):
    """Figure 2.13（p.65）三联重绘：桶形 / 枕形 / 鱼眼等距。
    (a)(b) 采用「铺满画面」画法：解 ideal_half·g(ideal_half)=1.5 取理想网格半宽，
    使同一 ±1.5 视窗内直接对比两种畸变的直线弯向（桶形外鼓、枕形内凹）。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    fig.patch.set_facecolor(PAPER)
    titles = ["(a) 桶形畸变 barrel（$\\kappa_1=-0.06$）",
              "(b) 枕形畸变 pincushion（$\\kappa_1=+0.06$）",
              "(c) 鱼眼 fisheye：r = fθ（式 2.80）"]
    for ax, ttl in zip(axes, titles):
        ax.set_facecolor(PAPER)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(ttl, fontsize=13, color=INK, pad=8)
    # 铺满画面的理想半宽：x·(1±0.06x²) = 1.55 数值求根（角点轻微出窗属传感器裁切，正常）
    def ideal_half(k1, k2=0.0, target=1.55):
        x = target
        for _ in range(60):
            x = target / distort_factor(x * x, k1, k2)
        return x
    warp_grid(axes[0], -0.06, 0.0, BLUE, n=11, half=ideal_half(-0.06), lw=1.0)
    warp_grid(axes[1], +0.06, 0.0, BROWN, n=11, half=ideal_half(+0.06), lw=1.0)
    Xf, Yf = fisheye_xy_grid(extent=3.0, n=25, f=1.0, theta_max_deg=88.0)
    for j in range(0, Xf.shape[0], 2):
        axes[2].plot(Xf[j], Yf[j], color=ORANGE, lw=1.0)
        axes[2].plot(Xf[:, j], Yf[:, j], color=ORANGE, lw=1.0)
    axes[2].add_patch(plt.Circle((0, 0), np.pi / 2 * 0.98, fill=False, color=INK, lw=0.8, ls="--"))
    axes[2].annotate("近 180° 视野\n（side-to-side）", (0.98, 0.55), fontsize=11, color=INK)
    axes[0].annotate("直线向外鼓（观测点拉向中心）", (0.0, -1.83), fontsize=11, color=BLUE, ha="center")
    axes[1].annotate("直线向内凹（观测点推离中心）", (0.0, -1.83), fontsize=11, color=BROWN, ha="center")
    for ax in axes:
        ax.set_xlim(-1.66, 1.66)
        ax.set_ylim(-2.02, 1.62)
    fig.suptitle("径向镜头畸变（据原书 Figure 2.13 重绘 · 原书图未使用）", fontsize=14, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER, dpi=160)
    plt.close(fig)


def draw_pipeline(path):
    """讲者设计：投影流水线与畸变插入位置（式 2.77 → 畸变 → 式 2.79）。"""
    fig, ax = plt.subplots(figsize=(12.6, 3.4))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    boxes = [
        (0.3, "三维点 p", BLUE),
        (2.75, "外参变换\n透视除法 式(2.77)", BLUE),
        (5.2, "径向畸变\n$1+\\kappa_1 r^2+\\kappa_2 r^4$", ORANGE),
        (7.95, "缩放加主点偏移\n式(2.79)", BLUE),
        (10.4, "像素\n(xs, ys)", GREEN),
    ]
    for x0, txt, col in boxes:
        box = FancyBboxPatch((x0, 1.15), 1.9, 1.1, boxstyle="round,pad=0.12",
                             fc="white", ec=col, lw=2.2)
        ax.add_patch(box)
        ax.text(x0 + 0.95, 1.7, txt, ha="center", va="center", fontsize=12.5, color=col)
    for x0 in (2.35, 4.8, 7.55, 10.0):
        ax.add_patch(FancyArrowPatch((x0, 1.7), (x0 + 0.42, 1.7),
                                     arrowstyle="-|>", mutation_scale=22, color=INK, lw=2))
    ax.text(6.15, 2.75, "畸变插在「透视除法之后、f 缩放与主点偏移之前」——原书第 64 页",
            ha="center", fontsize=13.5, color=ORANGE)
    ax.text(6.15, 0.42, "线性投影的流水线被拦腰插进一段非线性（讲者设计示意图 · 原书图未使用）",
            ha="center", fontsize=11, color=BROWN)
    fig.savefig(path, facecolor=PAPER, dpi=160, bbox_inches="tight")
    plt.close(fig)


def draw_grid_undistort(path, k1, k2):
    """讲者设计：理想方格 → 虚拟镜头观测（桶形）→ 去畸变恢复。
    网格 ±1.0（r_max=√2 恰为 g 极小点，桶形视觉干净），与实验 #5 同参数同范围。"""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5))
    fig.patch.set_facecolor(PAPER)
    titles = ["① 理想方格（直线成网）",
              f"② 虚拟镜头观测 $\\kappa_1={k1}$、$\\kappa_2={k2}$",
              "③ 去畸变后（直线重新站直）"]
    g = np.linspace(-1.0, 1.0, 13)
    X, Y = np.meshgrid(g, g)
    for ax, ttl, col in zip(axes, titles, [BLUE, ORANGE, GREEN]):
        ax.set_facecolor(PAPER)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(ttl, fontsize=13, color=INK, pad=8)
        ax.set_xlim(-1.16, 1.16)
        ax.set_ylim(-1.24, 1.18)
    for j in range(13):
        axes[0].plot(g, np.full_like(g, g[j]), color=BLUE, lw=1.0)
        axes[0].plot(np.full_like(g, g[j]), g, color=BLUE, lw=1.0)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    obs = distort(pts, k1, k2)
    rec, _ = undistort_refine(obs, k1, k2)
    for data, col in [(obs, ORANGE), (rec, GREEN)]:
        ax = axes[1] if col == ORANGE else axes[2]
        for j in range(13):
            ax.plot(data[j * 13:(j + 1) * 13, 0], data[j * 13:(j + 1) * 13, 1], color=col, lw=1.0)
            ax.plot(data[j::13, 0], data[j::13, 1], color=col, lw=1.0)
    axes[1].annotate("边缘鼓成桶", (0.0, -1.10), fontsize=11, color=ORANGE, ha="center")
    axes[2].annotate("直线保持", (0.0, -1.10), fontsize=11, color=GREEN, ha="center")
    fig.suptitle("虚拟镜头去畸变实验（讲者设计 · 原书图未使用 · 判据：直线保持，原书第 65–66 页）",
                 fontsize=14, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER, dpi=160)
    plt.close(fig)


def draw_r_curve(path):
    """讲者设计，双面板：左——径向响应因子 g=r̂/r（式 2.78 括号，桶形/枕形）；
    右——鱼眼等距投影 r=fθ（式 2.80）与针孔 r=f·tanθ 对照。
    两面板各自独立坐标轴：g 的自变量是 r，鱼眼的自变量是入射角 θ，不再混在同一根轴上
    （门A R3-P1-2 整改：纵轴标签与绘制量一致，消除 r=0 处 NaN）。"""
    fig, (ax, axf) = plt.subplots(1, 2, figsize=(12.6, 4.6))
    fig.patch.set_facecolor(PAPER)
    # 左：畸变因子 g(r)=1+κ₁r²+κ₂r⁴ = r̂/r（直接画括号本身，r=0 处 g=1，无 NaN）
    r = np.linspace(0, 1.5, 300)
    ax.plot(r, distort_factor(r ** 2, -0.2, 0.05), color=BLUE, lw=2.4,
            label=r"桶形 $\kappa_1=-0.2$、$\kappa_2=0.05$（虚拟镜头）")
    ax.plot(r, distort_factor(r ** 2, 0.2, 0.0), color=BROWN, lw=2.2,
            label=r"枕形 $\kappa_1=+0.2$")
    ax.axhline(1.0, color=INK, lw=1.0, ls="--", alpha=0.6)
    ax.text(1.02, 1.018, r"无畸变 $g=1$（$\hat{r}=r$）", fontsize=10.5, color=INK)
    ax.annotate("四次项在边缘发力", (1.04, 0.868), fontsize=11, color=BLUE)
    ax.set_xlabel("理想径向距离 r", fontsize=12, color=INK)
    ax.set_ylabel(r"$\hat{r}/r$（畸变因子，式 2.78 括号）", fontsize=12, color=INK)
    ax.set_xlim(0, 1.58)
    ax.set_ylim(0.7, 1.6)
    _style(ax)
    ax.legend(fontsize=10.5, loc="upper left", facecolor="white", edgecolor=INK)
    ax.set_title(r"径向响应：桶形 $\hat{r}/r<1$、枕形 $\hat{r}/r>1$", fontsize=13.5, color=INK, pad=10)
    # 右：鱼眼等距投影（θ 为自变量）；针孔 r=f·tanθ 作对照（讲者设计）
    th = np.linspace(0, np.pi / 2, 200)
    axf.plot(np.rad2deg(th), th, color=ORANGE, lw=2.6, label=r"鱼眼等距 $r=f\theta$（式 2.80）")
    axf.plot(np.rad2deg(th), np.tan(th), color=INK, lw=1.6, ls="--",
             label=r"针孔对照 $r=f\tan\theta$")
    axf.annotate("θ=90° 时 r=π/2≈1.57\n（视野对向 180°）", (90, np.pi / 2), xytext=(16, 2.5),
                 fontsize=11, color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE,
                                                            connectionstyle="arc3,rad=-0.2"))
    axf.set_xlabel("入射角 θ（鱼眼的自变量）", fontsize=12, color=INK)
    axf.set_ylabel("像面半径 r（f=1）", fontsize=12, color=INK)
    axf.set_xlim(0, 96)
    axf.set_ylim(0, 4.2)
    _style(axf)
    axf.legend(fontsize=10.5, loc="upper left", facecolor="white", edgecolor=INK)
    axf.set_title("鱼眼换尺子：按角度线性，针孔在边缘发散", fontsize=13.5, color=INK, pad=10)
    fig.suptitle("径向畸变与鱼眼响应（讲者设计 · 原书图未使用）", fontsize=14, color=INK, y=1.0)
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER, dpi=160)
    plt.close(fig)


def sync_and_md5():
    """同步检查（先比后写）：outputs/ ↔ slides/figures/。"""
    outdir = EP / "code" / "outputs"
    figdir = EP / "slides" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    files = sorted(outdir.glob("*.png"))
    drift = []
    for f in files:
        g = figdir / f.name
        if not g.exists():
            drift.append(f.name)
            print(f"[sync-DRIFT] {f.name} 在 slides/figures/ 缺失——本轮判 FAIL（将补齐，重跑至一致）")
        elif hashlib.md5(g.read_bytes()).hexdigest() != hashlib.md5(f.read_bytes()).hexdigest():
            drift.append(f.name)
            print(f"[sync-DRIFT] {f.name} 与 slides/figures/ 不一致——本轮判 FAIL（将覆盖，重跑至一致）")
    ok = len(files) > 0 and not drift
    print(f"[verify] 同步检查（比对在写入前完成，漂移判 FAIL）  共 {len(files)} 张、漂移 {len(drift)} 张  "
          f"[{'PASS' if ok else 'FAIL'}]")
    for f in files:
        (figdir / f.name).write_bytes(f.read_bytes())
        print(f"[demo] 已同步 {figdir / f.name}")
    return ok


def main():
    outdir = EP / "code" / "outputs"
    outdir.mkdir(exist_ok=True)
    draw_fig2_13(outdir / "e017_fig2_13.png")
    draw_pipeline(outdir / "e017_pipeline.png")
    draw_r_curve(outdir / "e017_r_curve.png")
    print("[demo] 前三张重绘图已存 outputs/")
    assert_eq277()
    assert_barrel_pincushion()
    assert_k_order()
    assert_eq279_K()
    k1, k2 = assert_undistort()
    assert_fisheye()
    assert_K_absorption()
    assert_linear_return(k1, k2)
    draw_grid_undistort(outdir / "e017_grid_undistort.png", k1, k2)
    print("[demo] 去畸变实验图已存 outputs/（共 4 张）")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"[FAIL-DETAIL] {name}")
    sync_ok = sync_and_md5()
    print(f"汇总：{n_pass}/{len(RESULTS)} 数据断言 {'PASS' if n_pass == len(RESULTS) else 'FAIL'} · "
          f"同步检查 {'PASS' if sync_ok else 'FAIL'}"
          + (" —— 全部通过" if n_pass == len(RESULTS) and sync_ok else " —— 存在失败"))


if __name__ == "__main__":
    main()
