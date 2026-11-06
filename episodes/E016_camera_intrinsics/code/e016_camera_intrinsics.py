"""
E016 · 内参与外参：一台相机的「身份证」（§2.1.4 后半，印刷 pp.55–63，至 §2.1.5 之前）

数值实验：把传感器单应 Ms、标定矩阵 K、相机矩阵 P=K[R|t]、焦距单位制、
4×4 可逆相机矩阵与 plane plus parallax、双视图映射、物体中心投影全部算一遍。

断言（22 项数据 + 1 项同步）：
  #1  传感器单应结构（式 2.53 原式，p.55）：lhs 按原书 [Rs cs]·S₄₃·x̄s 计算（S₄₃ 为逐元硬编码的
      4×3 矩阵，四行 sx 0 0 / 0 sy 0 / 0 0 0 / 0 0 1），rhs = sx·xs·Rs[:,0]+sy·ys·Rs[:,1]+cs，
      且 [Rs cs]·S₄₃ = Ms
  #2  Ms 列解释（构造性检查，p.55–56 原文）：前两列范数 = sx/sy、第三列 = cs——由 make_Ms 定义保证
  #3  Ms 8 参数化重构与反解（p.56 eight unknowns）：ω=0.42·unit([0.3,−0.5,0.8]) 与 #1 同轴同角，
      8 维向量 (ω,cs,sx,sy) → Rodrigues 重建 Ms′ ≈ Ms；再由 Ms′ 列范数/Gram 反解 8 参数回到原值
  #4  8→7 DoF 耦合（p.56）：make_Ms(Rs,λcs,λsx,λsy)=λMs、K(λMs)=K(Ms)/λ、同一射线像素地址不变
      ——传感器距离与像素间距不可分
  #5  K[R|t] 规范自由：K 右乘 R₁、[R|t] 左乘 R₁ᵀ 后 P 与投影均不变（式 2.55–2.56，p.56）
  #6  RQ 分解：M=K·R → 分解回上三角 K（正对角）+ 正交 R（p.57；术语注：矩阵代数 R=上三角、CV R=旋转）
  #7  三档 K 一致（pp.57–58）：K58 独立于 make_K 按式(2.58) 字面拼装；
      (i) 随机 (fx,fy,s) 取 f=fx、a=fy/fx 时 K57≡K58；(ii) K58(f,1,0)≡式(2.59) 字面矩阵；
      (iii) 同一组 3D 点两对投影一致
  #8  FOV 换算一致性演示：35mm(W=36,f=50) 与像素(W=720,f=1000) 与 unitless(f=50/18) 同一 θH=39.60°
      （式 2.60 p.58、式 2.62 p.59、换算句 p.59）；×18mm 换算链子断言：f_px×18 ≠ 50（原文字面
      对像素 f 不成立）、(f_px/(W_px/2))×18 = 50（讲者推演：×18mm 严格成立于 unitless f）
  #9  归一化设备坐标端点：xs∈{0,W} → x′s=∓W/S、S=max(W,H)（式 2.61，p.59）
  #10 式(2.64) 分块结构：P̃[:3]=P、末行=[0,0,0,1]、det(P̃)≠0 可逆（p.60）
  #11 式(2.65) 投影：P̃·p̄w 归一化 → (xs,ys,1,d)，d = 1/z（p.60；K 末行 [0,0,1] 时）
  #12 式(2.66) plane plus parallax：末行 p₃=s₃[n̂₀|c₀] → d = s₃(n̂₀·pw+c₀)/z（p.60）
  #13 参考平面在无穷：n̂₀=0、c₀=1 → d=1/z 标准 disparity（Okutomi & Kanade 1993，p.61）
  #14 式(2.67) 逆映射：P̃⁻¹·xs ∝ p̄w——像素+视差映回 3D（p.61）
  #15 双视图链：M₁₀ = P̃₁P̃₀⁻¹，M₁₀x̃₀ ∼ x̃₁（式 2.68–2.70，p.61）
  #16 平面场景免深度：d₀=0 → H₁₀ = M₁₀[:3,:3]，H₁₀x̃₀ ∼ x̃₁ 对面上点成立（式 2.71，pp.61–62）
  #17 纯旋转免深度（式 2.72，p.62）：t₀=t₁=0 时 K₁R₁R₀⁻¹K₀⁻¹ 对任意 3D 点成立；
      反例子断言：t₀=t₁≠0（光心不重合）时同一单应误差显著——讲者按：[R|t] 约定下严格条件是
      光心重合（−R₀ᵀt₀=−R₁ᵀt₁），原书「i.e., when t0=t1」在 t=0 处严格成立
  #18 式(2.73) 展开：xs = f(rx·p+tx)/(rz·p+tz)+cx 与 K[R|t] 投影一致（p.63）
  #19 f/tz 歧义：tz≫‖p‖ 时 f、tz 同时翻倍 xs 变化 <1%——变焦 vs 靠近不可分（p.63；Vertigo 彩蛋 p.62）
  #20 重参数化精确恒等：ηz=1/tz、s=ηz·f 时式(2.75) 与式(2.73) 逐点一致（p.63）
  #21 长焦极限：ηz→0 时式(2.75) → 正交投影，相对偏差 ~O(ηz)（p.63）
  #22 解耦对照（p.62–63；讲者设计）：对 (xs,ys) 拼接向量的偏导列做未去均值余弦（JᵀJ 归一化非对角元），
      |cos(J_f,J_tz)|>0.99（p.62 原文 highly correlated）而 |cos(J_s,J_ηz)| 显著更小（p.63 mostly decoupled）
      ——重参数化后 ηz 相对 s 的解耦，是相对 (f,t_z) 而言的
  #S  同步检查：先比对 slides/figures/ 逐文件 md5（漂移判 FAIL），再覆盖同步（重跑至一致）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（同 E014/E015：macOS 优先 Hiragino Sans GB；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import pathlib

rng = np.random.default_rng(20260416)
INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
RESULTS = []


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[verify] {name}  {detail}  [{'PASS' if ok else 'FAIL'}]")


def hat(v):
    """叉积算子 [v]×（E015 式 2.32 同款）。"""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def R3(n, theta):
    """Rodrigues（E015 式 2.38 同款）：R(n̂,θ) = I + sinθ[n̂]× + (1−cosθ)[n̂]²×。"""
    K = hat(n)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ——— 内参构造（与原书式 2.53–2.59 对应）—————————————————————————
def make_Ms(Rs, cs, sx, sy):
    """式(2.53)（p.55）：Ms = [Rs cs]·S₄₃ 的展开，S₄₃ 为 4×3（四行 sx 0 0 / 0 sy 0 / 0 0 0 / 0 0 1）。

    乘积列解释：Ms = [sx·Rs 第一列, sy·Rs 第二列, cs]——S₄₃ 第三行全 0 丢弃 Rs 第三列，
    第四行 (0 0 1) 把齐次 1 送到 cs 列。
    """
    Ms = np.zeros((3, 3))
    Ms[:, 0] = sx * Rs[:, 0]
    Ms[:, 1] = sy * Rs[:, 1]
    Ms[:, 2] = cs
    return Ms


def make_K(fx, fy, s, cx, cy):
    """式(2.57)：上三角标定矩阵 [fx s cx; 0 fy cy; 0 0 1]。"""
    return np.array([[fx, s, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def rq3(M):
    """RQ 分解 M = K·R（K 上三角正对角、R 正交）——p.57 由 P 提取上三角 K 的手段。

    实现：B = J·M·J（J 为行/列反转），B = L·Q'（LQ），LQ 由 QR(Bᵀ) 获得，
    再反转回来。注：矩阵代数把这里的上三角因子叫 R，与 CV 的旋转 R 撞名（p.57 原文注）。
    """
    J = np.eye(3)[::-1]
    B = J @ M @ J
    Q2, R2 = np.linalg.qr(B.T)      # Bᵀ = Q₂R₂，R₂ 上三角 = Lᵀ
    L = R2.T                         # 下三角
    K = J @ L @ J                    # 上三角
    Rc = J @ Q2.T @ J                # 正交
    d = np.sign(np.diag(K))
    d[d == 0] = 1.0
    K = K * d                        # K←DK（正对角）
    Rc = d[:, None] * Rc             # R←D⁻¹R = DR（D 自逆），K·R 不变
    return K, Rc


# ——— 断言 #1–#4：传感器单应 Ms ————————————————————————————————————
def assert_Ms():
    Rs = R3(unit([0.3, -0.5, 0.8]), 0.42)
    cs = np.array([0.02, -0.015, 0.10])
    sx, sy = 6.5e-6, 6.7e-6          # 像素间距（微米级，p.55）
    Ms = make_Ms(Rs, cs, sx, sy)

    # 式(2.53) 原式（p.55）：中间矩阵 4×3、逐元硬编码；[Rs cs] 为 3×4，乘积 3×3
    S43 = np.array([[sx, 0.0, 0.0],
                    [0.0, sy, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]])
    RC = np.hstack([Rs, cs[:, None]])
    pix = np.array([[0.0, 0.0], [1919.0, 1079.0], [640.0, 480.0], [123.0, 456.0]])
    lhs = np.array([RC @ S43 @ np.r_[x, y, 1.0] for x, y in pix])
    rhs = np.array([sx * x * Rs[:, 0] + sy * y * Rs[:, 1] + cs for x, y in pix])
    err = np.abs(lhs - rhs).max()
    verify("#1 传感器单应结构 p=[Rs cs]·S₄₃·x̄s=sx·xs·Rs₀+sy·ys·Rs₁+cs（式2.53 p.55，S₄₃ 逐元硬编码 4×3）",
           err < 1e-15 and np.allclose(RC @ S43, Ms),
           f"4 组像素最大偏差 {err:.2e} · [Rs cs]·S₄₃=Ms 一致")

    n0, n1 = np.linalg.norm(Ms[:, 0]), np.linalg.norm(Ms[:, 1])
    verify("#2 Ms 列解释（构造性检查）：前两列=单位步进向量(‖·‖=sx,sy)、第三列=cs（p.55–56）",
           abs(n0 - sx) < 1e-18 and abs(n1 - sy) < 1e-18 and np.allclose(Ms[:, 2], cs),
           f"构造性检查（由 make_Ms 定义保证）· ‖col0‖={n0:.3e} ‖col1‖={n1:.3e} col2=cs")

    # 8 参数向量 θ₈ = (ω, cs, sx, sy)：ω 与 #1 的 Rs 同轴同角（p.56 eight unknowns）
    omega = 0.42 * unit([0.3, -0.5, 0.8])
    theta8 = np.r_[omega, cs, sx, sy]
    Rs8 = R3(unit(theta8[:3]), np.linalg.norm(theta8[:3]))
    Ms8 = make_Ms(Rs8, theta8[3:6], theta8[6], theta8[7])
    # 由 Ms′ 反解 8 参数：列范数 → (sx,sy)、第三列 → cs、前两列归一化 → Rs → Rodrigues 反解 ω
    sx_r, sy_r = np.linalg.norm(Ms8[:, 0]), np.linalg.norm(Ms8[:, 1])
    r0, r1 = Ms8[:, 0] / sx_r, Ms8[:, 1] / sy_r
    R_rec = np.column_stack([r0, r1, np.cross(r0, r1)])
    ang = np.arccos(np.clip((np.trace(R_rec) - 1.0) / 2.0, -1.0, 1.0))
    axis_rec = np.array([R_rec[2, 1] - R_rec[1, 2],
                         R_rec[0, 2] - R_rec[2, 0],
                         R_rec[1, 0] - R_rec[0, 1]]) / (2.0 * np.sin(ang))
    theta_rec = np.r_[axis_rec * ang, Ms8[:, 2], sx_r, sy_r]
    G = Ms8.T @ Ms8
    verify("#3 Ms 8 参数化重构与反解（ω3+cs3+尺度2，p.56 eight unknowns）",
           np.allclose(Ms8, Ms) and np.allclose(theta_rec, theta8, atol=1e-12)
           and abs(G[0, 0] - sx ** 2) < 1e-18 and abs(G[1, 1] - sy ** 2) < 1e-18,
           "8 参数 → Rodrigues 重建 Ms′ 与 #1 的 Ms 逐元一致 · Gram 对角 = (sx², sy²) · 反解 8 参数回到原值")

    # 8→7 DoF 耦合（p.56）：传感器距离与像素间距同乘 λ → Ms′=λMs、K′=K/λ，
    # 同一条 3D 射线 pc 经 Ms⁻¹ 与 (λMs)⁻¹ 反归一化后像素地址相同——λ 只改尺度、不改地址
    lam = 2.718281828
    Ms_lam = make_Ms(Rs, lam * cs, lam * sx, lam * sy)
    K_of_Ms = np.linalg.inv(Ms)
    K_lam = np.linalg.inv(Ms_lam)
    pc_ray = rng.uniform(-0.8, 0.8, 3); pc_ray[2] = abs(pc_ray[2]) + 1.5   # 固定一条非光轴射线
    xbar_old = np.linalg.inv(Ms) @ pc_ray
    xbar_lam = np.linalg.inv(Ms_lam) @ pc_ray
    addr_old = K_of_Ms @ xbar_old; addr_old = addr_old / addr_old[2]
    addr_lam = K_of_Ms @ xbar_lam; addr_lam = addr_lam / addr_lam[2]
    addr_dev = np.abs(addr_old - addr_lam).max()
    verify("#4 8→7 耦合：(cs,sx,sy)×λ ⇒ Ms′=λMs、K′=K/λ、同一 pc 反归一化地址不变（p.56）",
           np.allclose(Ms_lam, lam * Ms) and np.allclose(K_lam, K_of_Ms / lam)
           and addr_dev < 1e-9,
           f"λ={lam:.6f}：同一 pc 经 Ms⁻¹ 与 (λMs)⁻¹ 归一化地址 max 偏差 {addr_dev:.2e}"
           "（λ 只改尺度——传感器距离与像素间距不可分）")


# ——— 断言 #5–#7：标定矩阵与相机矩阵 ————————————————————————————————
def assert_K_and_P():
    K = make_K(800.0, 820.0, 0.5, 960.0, 540.0)
    Rc = R3(unit([0.2, 0.6, -0.75]), 1.1)
    t = np.array([0.05, -0.03, 4.0])
    P = K @ np.hstack([Rc, t[:, None]])

    R1 = R3(unit([0.4, 0.4, 0.82]), 0.7)
    K2 = K @ R1
    Rt2 = R1.T @ np.hstack([Rc, t[:, None]])
    P2 = K2 @ Rt2
    pts_w = rng.uniform(-1, 1, (6, 3))
    proj1 = np.array([P @ np.r_[p, 1.0] for p in pts_w])
    proj2 = np.array([P2 @ np.r_[p, 1.0] for p in pts_w])
    verify("#5 K[R|t] 规范自由：K→KR₁、[R|t]→R₁ᵀ[R|t] 后 P 与投影不变（式2.55–2.56 p.56）",
           np.allclose(P, P2) and np.allclose(proj1, proj2)
           and not np.allclose(np.tril(K2, -1), 0.0),
           f"P 最大偏差 {np.abs(P - P2).max():.2e} · K·R₁ 非上三角（下三角元和 "
           f"{np.abs(np.tril(K2, -1)).sum():.2f}）——上三角 K 与真实朝向不可分")

    M = K @ Rc                                # 满秩 3×3：K·R_cam
    Khat, Rhat = rq3(M)
    tri = abs(Khat[1, 0]) + abs(Khat[2, 0]) + abs(Khat[2, 1])
    verify("#6 RQ 分解：M=K·R → 上三角 K（正对角）+ 正交 R（p.57）",
           tri < 1e-12 and np.all(np.diag(Khat) > 0) and np.allclose(Khat @ Rhat, M)
           and np.allclose(Rhat.T @ Rhat, np.eye(3))
           and np.allclose(Khat, K) and np.allclose(Rhat, Rc),
           f"下三角元 {tri:.2e} · ‖K̂R̂−M‖={np.abs(Khat @ Rhat - M).max():.2e} · ‖R̂ᵀR̂−I‖={np.abs(Rhat.T @ Rhat - np.eye(3)).max():.2e}"
           f" · 分解唯一：‖K̂−K‖={np.abs(Khat - K).max():.2e} ‖R̂−Rc‖={np.abs(Rhat - Rc).max():.2e}")

    # 式(2.58) 独立实现（不复用 make_K）：f=fx、a=fy/fx 代入式(2.57) 结构后按原书字面拼装
    def K58_of(f, a, s, cx, cy):
        return np.array([[f, s, cx], [0.0, a * f, cy], [0.0, 0.0, 1.0]])

    fx, fy, s0 = (float(rng.uniform(700, 900)), float(rng.uniform(700, 900)), float(rng.uniform(-0.5, 0.5)))
    K57 = make_K(fx, fy, s0, 960.0, 540.0)
    K58 = K58_of(fx, fy / fx, s0, 960.0, 540.0)
    K59_lit = np.array([[fx, 0.0, 960.0], [0.0, fx, 540.0], [0.0, 0.0, 1.0]])   # 式(2.59) 字面硬编码
    K59_via58 = K58_of(fx, 1.0, 0.0, 960.0, 540.0)
    pts7 = rng.uniform(-1, 1, (6, 3)) + np.array([0.0, 0.0, 5.0])

    def _proj(Kk):
        return np.array([(lambda v: v[:2] / v[2])(Kk @ p) for p in pts7])

    verify("#7 三档 K 一致：K57(fx,fy,s)≡K58(f=fx,a=fy/fx,s) · K58(f,1,0)≡式(2.59)字面 · 两对投影一致（pp.57–58）",
           np.allclose(K57, K58) and np.allclose(K59_lit, K59_via58)
           and np.allclose(_proj(K57), _proj(K58)) and np.allclose(_proj(K59_lit), _proj(K59_via58)),
           f"随机 fx={fx:.1f} fy={fy:.1f} s={s0:.2f}：a=fy/fx={fy / fx:.4f} · K₂₅₇≡K₂₅₈ · 式(2.59)≡K₂₅₈(a=1,s=0)")


# ——— 断言 #8–#9：焦距单位制 ————————————————————————————————————————
def assert_focal_units():
    W_mm, f_mm = 36.0, 50.0
    theta_35 = 2 * np.arctan((W_mm / 2) / f_mm)            # 式(2.60) 第一式
    f_u = f_mm / 18.0                                       # 35mm→unitless（式 2.62 的 f）：f_u = f_35/(W/2)，35mm 相机 W/2=18mm
    theta_u = 2 * np.arctan(1.0 / f_u)                      # 式(2.62)：f⁻¹=tan(θH/2)
    W_px, f_px = 720.0, 1000.0                              # 同一相机数码化：f_px = f_35·(W_px/36mm)
    theta_px = 2 * np.arctan((W_px / 2) / f_px)             # 式(2.60) 像素版
    deg = np.degrees(theta_35)
    # ×18mm 换算链（p.59 原文那句只对 unitless f 严格成立；讲者推演见 deck 第 10 页）
    naive = f_px * 18.0                                     # 原文字面直乘像素 f：18000mm ≠ 50mm
    f35_correct = (f_px / (W_px / 2.0)) * 18.0              # 先化 unitless（÷半边宽）再 ×18mm
    verify("#8 FOV 换算一致性演示 + ×18mm 换算链（式2.60/2.62+换算句，pp.58–59；讲者推演）",
           abs(theta_35 - theta_u) < 1e-12 and abs(theta_35 - theta_px) < 1e-12
           and abs(1.0 / f_u - (W_mm / 2) / f_mm) < 1e-15
           and naive > 50.0 + 1.0 and abs(f35_correct - 50.0) < 1e-9,
           f"θH = {deg:.2f}° 三种单位制换算一致 · f_px×18={naive:.0f}mm≠50 · (f_px/(W_px/2))×18={f35_correct:.2f}mm=50"
           "（同一相机的三种单位制换算演示，非独立互证）")

    W, H = 640.0, 480.0
    S = max(W, H)

    def norm_xy(xs, ys):
        return (2 * xs - W) / S, (2 * ys - H) / S

    x0, y0 = norm_xy(0.0, 0.0)
    x1, y1 = norm_xy(W, H)
    verify("#9 归一化设备坐标端点 ±W/S、±H/S，S=max(W,H)（式2.61 p.59）",
           abs(x0 + W / S) < 1e-15 and abs(x1 - W / S) < 1e-15
           and abs(y0 + H / S) < 1e-15 and abs(y1 - H / S) < 1e-15,
           f"xs: 0→{x0:+.4f}, W→{x1:+.4f} · S={S:.0f}（横拍；xs=W 为端点外推——像素有效范围 [0,W)）")


# ——— 断言 #10–#14：4×4 相机矩阵与视差 ————————————————————————————————
def make_Ptilde(K, Rc, t):
    """式(2.64)：P̃ = [K 0; 0ᵀ 1]·[R t; 0ᵀ 1] = K̃·E（4×4 可逆）。"""
    KE = np.eye(4); KE[:3, :3] = K
    E4 = np.eye(4); E4[:3, :3] = Rc; E4[:3, 3] = t
    return KE @ E4


def assert_Ptilde():
    K = make_K(800.0, 800.0, 0.0, 960.0, 540.0)     # 式(2.59) 最简形式
    Rc = R3(unit([0.1, 0.5, 0.86]), 0.6)
    t = np.array([0.1, -0.2, 10.0])
    Pt = make_Ptilde(K, Rc, t)
    P = K @ np.hstack([Rc, t[:, None]])

    verify("#10 式(2.64) 分块结构：P̃[:3]=P、末行=[0,0,0,1]、det≠0 可逆（p.60）",
           np.allclose(Pt[:3], P) and np.allclose(Pt[3], [0.0, 0.0, 0.0, 1.0])
           and abs(np.linalg.det(Pt)) > 1e-6,
           f"det P̃ = {np.linalg.det(Pt):.1f} ≠ 0（末行 [0,0,0,1]——可逆正是式 2.64 的卖点）")

    pw = rng.uniform(-1, 1, (8, 3)) * np.array([1.0, 1.0, 0.3]) + np.array([0.0, 0.0, 5.0])
    xs_h = np.array([Pt @ np.r_[p, 1.0] for p in pw])
    norm = xs_h / xs_h[:, 2:3]                      # 除以第三个分量（p.60 原文）
    # z = (R·pw + t) 的第三分量 = rz·pw + tz = rz·(pw − c)（c = −Rᵀt，式 2.25 回指）
    zcam = np.array([(Rc @ p + t)[2] for p in pw])
    d = norm[:, 3]
    verify("#11 式(2.65) 投影：P̃·p̄w 归一化 → (xs,ys,1,d)，d = 1/z（p.60）",
           np.allclose(norm[:, 2], 1.0)
           and np.allclose(d, 1.0 / zcam),
           f"max|d−1/z| = {np.abs(d - 1.0 / zcam).max():.2e}")

    n0 = unit(np.array([0.2, -0.3, 0.5]))
    c0 = 0.4
    s3 = 2.0
    Pt_pp = Pt.copy()
    Pt_pp[3] = s3 * np.r_[n0, c0]                   # 末行重写 p₃ = s₃[n̂₀|c₀]（式 2.66 前文）
    xsh = np.array([Pt_pp @ np.r_[p, 1.0] for p in pw])
    nrm = xsh / xsh[:, 2:3]
    d_ref = s3 * (pw @ n0 + c0) / zcam
    verify("#12 式(2.66) plane plus parallax：d = s₃(n̂₀·pw+c₀)/z（p.60）",
           np.allclose(nrm[:, 3], d_ref),
           f"max|d−s₃(n̂₀·pw+c₀)/z| = {np.abs(nrm[:, 3] - d_ref).max():.2e}")

    Pt_inf = Pt.copy()
    Pt_inf[3] = np.r_[0.0, 0.0, 0.0, 1.0]           # n̂₀=0、c₀=1：参考平面在无穷
    xsh2 = np.array([Pt_inf @ np.r_[p, 1.0] for p in pw])
    nrm2 = xsh2 / xsh2[:, 2:3]
    verify("#13 参考平面在无穷：n̂₀=0、c₀=1 → d=1/z 标准 disparity（p.61）",
           np.allclose(nrm2[:, 3], 1.0 / zcam),
           "同一 P̃ 只换末行解释（n̂₀=0、c₀=1）——矩阵与 #11 相同，换的是几何解读")

    xs4 = np.column_stack([nrm[:, 0], nrm[:, 1], np.ones(len(pw)), nrm[:, 3]])  # (xs,ys,1,d)
    back = np.array([np.linalg.inv(Pt_pp) @ x for x in xs4])
    back_n = back / back[:, 3:4]
    verify("#14 式(2.67) 逆映射：P̃⁻¹·xs ∝ p̄w——像素+视差映回 3D（p.61）",
           np.allclose(back_n[:, :3], pw),
           f"max‖回投−pw‖ = {np.abs(back_n[:, :3] - pw).max():.2e}")
    return K, Rc, t, Pt, zcam


# ——— 断言 #15–#17：跨相机映射 ————————————————————————————————————————
def assert_two_views(K, Rc, t):
    K1 = make_K(820.0, 820.0, 0.0, 950.0, 530.0)
    R1 = R3(unit([-0.3, 0.7, 0.65]), 0.5)
    t1 = np.array([-0.3, 0.1, 9.5])
    Pt0 = make_Ptilde(K, Rc, t)
    Pt1 = make_Ptilde(K1, R1, t1)

    pw = rng.uniform(-1, 1, (10, 3)) * np.array([1.0, 1.0, 0.4]) + np.array([0.0, 0.0, 5.0])
    x0h = np.array([Pt0 @ np.r_[p, 1.0] for p in pw])
    x1h = np.array([Pt1 @ np.r_[p, 1.0] for p in pw])
    M10 = Pt1 @ np.linalg.inv(Pt0)
    via = np.array([M10 @ x for x in x0h])
    e = []
    for v, w in zip(via, x1h):
        v = v / v[2]; w = w / w[2]
        e.append(np.abs(v / v[3] - w / w[3]).max() if v[3] != 0 else 1.0)
    verify("#15 双视图链：M₁₀ = P̃₁P̃₀⁻¹，M₁₀x̃₀ ∼ x̃₁（式2.68–2.70 p.61）",
           np.max(e) < 1e-10,
           f"10 组点 max 相对偏差 {max(e):.2e}")

    # 平面场景：取 n̂₀·pw + c₀ = 0 的点（把 #12 的参考平面当场景平面）
    n0 = unit(np.array([0.2, -0.3, 0.5])); c0 = 0.4
    pw = rng.uniform(-1, 1, (12, 3)) * np.array([1.0, 1.0, 0.4]) + np.array([0.0, 0.0, 5.0])
    pw = pw - ((pw @ n0 + c0) / np.linalg.norm(n0) ** 2)[:, None] * n0   # 投影到平面上
    Pt0p = Pt0.copy(); Pt0p[3] = np.r_[n0, c0]                          # 末行=平面方程 → d₀=0
    x0h = np.array([Pt0p @ np.r_[p, 1.0] for p in pw])
    x1h = np.array([Pt1 @ np.r_[p, 1.0] for p in pw])
    M10 = Pt1 @ np.linalg.inv(Pt0p)
    H10 = M10[:3, :3]                                                    # d₀=0：忽略末列与末行
    err = 0.0
    for a, b in zip(x0h, x1h):
        lhs = H10 @ np.r_[a[0] / a[2], a[1] / a[2], 1.0]
        rhs = b / b[2]
        lhs = lhs / lhs[2]
        err = max(err, np.abs(lhs - rhs[:3]).max())
    verify("#16 平面免深度：d₀=0 → H₁₀=M₁₀[:3,:3]，H₁₀x̃₀ ∼ x̃₁（式2.71 pp.61–62）",
           err < 1e-9,
           f"12 组面上点 max 偏差 {err:.2e}")

    # 纯旋转（式 2.72，p.62 原文「i.e., when t0 = t1」）：实验取 t₀=t₁=0（光心重合的特例）。
    # 讲者按：[R|t] 约定下严格条件是光心重合（−R₀ᵀt₀ = −R₁ᵀt₁）；t₀=t₁=t≠0 且 R₀≠R₁ 时光心不
    # 重合，式(2.72) 不再成立——下方反例子断言验证。
    K1r = make_K(820.0, 820.0, 0.0, 950.0, 530.0)
    R1r = R1
    Pt0r = make_Ptilde(K, Rc, np.zeros(3))
    Pt1r = make_Ptilde(K1r, R1r, np.zeros(3))
    pw = rng.uniform(-1, 1, (10, 3)) * np.array([1.2, 1.2, 0.5]) + np.array([0.0, 0.0, 6.0])
    Hrot = K1r @ R1r @ np.linalg.inv(Rc) @ np.linalg.inv(K)

    def _rot_err(P0, P1, H, pts):
        e = 0.0
        for p in pts:
            a = P0[:3] @ np.r_[p, 1.0]; a = a / a[2]
            b = P1[:3] @ np.r_[p, 1.0]; b = b / b[2]
            c = H @ np.r_[a[0], a[1], 1.0]; c = c / c[2]
            e = max(e, np.abs(c - b[:3]).max())
        return e

    err0 = _rot_err(Pt0r[:3], Pt1r[:3], Hrot, pw)
    t_bad = np.array([0.6, -0.4, 0.0])                     # t₀=t₁=t≠0：光心不重合
    Pt0n = make_Ptilde(K, Rc, t_bad)
    Pt1n = make_Ptilde(K1r, R1r, t_bad)
    err1 = _rot_err(Pt0n[:3], Pt1n[:3], Hrot, pw)
    verify("#17 纯旋转免深度（t₀=t₁=0）：x̃₁ ∼ K₁R₁R₀⁻¹K₀⁻¹x̃₀ 对任意 3D 点；反例 t₀=t₁≠0 误差显著（式2.72 p.62）",
           err0 < 1e-9 and err1 > 1e-2,
           f"t₀=t₁=0：10 组点 max 偏差 {err0:.2e} · 反例 t₀=t₁≠0：{err1:.1f} px（光心不重合，讲者按）")


# ——— 断言 #18–#22：物体中心投影 ————————————————————————————————————————
def assert_object_centered():
    f, cx, cy = 50.0, 0.0, 0.0
    Rx = np.array([0.9, 0.1, -0.2]); Ry = np.array([-0.1, 0.95, 0.05]); Rz = np.array([0.2, -0.05, 0.98])
    Rc = np.vstack([unit(Rx), unit(Ry), unit(Rz)])
    # 保证正交且行列式 +1：用极分解修正
    U, S, Vt = np.linalg.svd(Rc)
    Rc = U @ Vt
    if np.linalg.det(Rc) < 0:
        Rc = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
    tz = 100.0
    t = np.array([0.5, -0.4, tz])
    K = make_K(f, f, 0.0, cx, cy)

    pw = rng.uniform(-1, 1, (10, 3)) * 0.8            # 物体尺寸 ~1，tz=100 ≫ ‖p‖
    rows = Rc
    direct = []
    for p in pw:
        xs = f * (rows[0] @ p + t[0]) / (rows[2] @ p + t[2]) + cx
        ys = f * (rows[1] @ p + t[1]) / (rows[2] @ p + t[2]) + cy
        direct.append([xs, ys])
    direct = np.array(direct)
    xh = np.array([K @ (Rc @ p + t) for p in pw])
    viaK = xh[:, :2] / xh[:, 2:3]
    verify("#18 式(2.73) 展开：xs=f(rx·p+tx)/(rz·p+tz)+cx 与 K[R|t] 投影一致（p.63）",
           np.allclose(direct, viaK),
           f"max 偏差 {np.abs(direct - viaK).max():.2e}")

    direct2 = []
    for p in pw:
        xs = 2 * f * (rows[0] @ p + t[0]) / (rows[2] @ p + 2 * t[2]) + cx
        ys = 2 * f * (rows[1] @ p + t[1]) / (rows[2] @ p + 2 * t[2]) + cy
        direct2.append([xs, ys])
    rel = np.abs(np.array(direct2) - direct) / (np.abs(direct) + 1e-30)
    verify("#19 f/tz 歧义：f、tz 同时翻倍 → xs 变化 <1%（tz≫‖p‖，p.63）",
           rel.max() < 0.01,
           f"10 组点 max 相对变化 {rel.max() * 100:.3f}%")

    eta = 1.0 / tz
    s = eta * f
    repro = []
    for p in pw:
        xs = s * (rows[0] @ p + t[0]) / (1 + eta * (rows[2] @ p)) + cx
        ys = s * (rows[1] @ p + t[1]) / (1 + eta * (rows[2] @ p)) + cy
        repro.append([xs, ys])
    verify("#20 重参数化恒等：ηz=1/tz、s=ηz·f 时式(2.75) ≡ 式(2.73)（p.63）",
           np.allclose(np.array(repro), direct),
           f"max 偏差 {np.abs(np.array(repro) - direct).max():.2e}")

    eta0 = 1e-6
    f0 = 5e5                                          # s = ηz·f = 0.5 不变
    s0 = eta0 * f0
    t0 = np.array([t[0], t[1], 1.0 / eta0])
    err = 0.0
    for p, d0 in zip(pw, direct):
        xs = s0 * (rows[0] @ p + t0[0]) / (1 + eta0 * (rows[2] @ p)) + cx
        ys = s0 * (rows[1] @ p + t0[1]) / (1 + eta0 * (rows[2] @ p)) + cy
        ortho = np.array([s0 * (rows[0] @ p + t0[0]) + cx, s0 * (rows[1] @ p + t0[1]) + cy])
        err = max(err, np.abs(xs - ortho[0]) / (abs(ortho[0]) + 1e-30),
                  np.abs(ys - ortho[1]) / (abs(ortho[1]) + 1e-30))
    verify("#21 长焦极限：ηz→0 时式(2.75) → 正交投影，偏差 O(ηz)（p.63）",
           err < 1e-5,
           f"ηz=1e−6：与 s(rx·p+tx)+cx 的 max 相对差 {err:.2e}")

    # 解耦对照（讲者设计，对照 p.63 原文）：f 与 tz「highly correlated」，重参数化后 ηz 相对 s
    # 「mostly decoupled」。度量：对 (xs,ys) 拼接向量的偏导列做未去均值余弦（JᵀJ 归一化非对角元）
    pw22 = rng.uniform(-1, 1, (200, 3)) * 0.8              # 加密采样，让余弦稳定
    u = np.array([rows[0] @ p + t[0] for p in pw22])
    v = np.array([rows[1] @ p + t[1] for p in pw22])
    w = np.array([rows[2] @ p + t[2] for p in pw22])
    w0 = np.array([rows[2] @ p for p in pw22])
    Jf = np.concatenate([u / w, v / w])                                     # ∂(xs,ys)/∂f
    Jtz = np.concatenate([-f * u / w ** 2, -f * v / w ** 2])                # ∂(xs,ys)/∂t_z
    Js = np.concatenate([u / (1 + eta * w0), v / (1 + eta * w0)])           # ∂(xs,ys)/∂s
    Jeta = np.concatenate([-s * u * w0 / (1 + eta * w0) ** 2,
                           -s * v * w0 / (1 + eta * w0) ** 2])              # ∂(xs,ys)/∂η_z

    def _cos0(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    c_ftz, c_seta = _cos0(Jf, Jtz), _cos0(Js, Jeta)
    verify("#22 解耦对照：|cos(J_f,J_tz)|>0.99 而 |cos(J_s,J_ηz)| 显著更小（p.62 highly correlated vs p.63 mostly decoupled；讲者设计）",
           abs(c_ftz) > 0.99 and abs(c_seta) < 0.3 * abs(c_ftz),
           f"cos(f,tz)={c_ftz:+.4f} · cos(s,ηz)={c_seta:+.4f}——(s,ηz) 相对 (f,t_z) 显著解耦")


# ——— 五张重绘图（原书 Figure 2.8/2.9/2.10/2.11/2.12，原书图未使用）————————————
def _new_ax(w=9.6, h=5.4):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    return fig, ax


def _tight(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d8cbb4")


def draw_fig2_8(path):
    """Figure 2.8（p.55）：3D 相机中心点 pc 沿射线投到传感器平面 p；cs 在网格角点、sx/sy=一格。"""
    fig = plt.figure(figsize=(9.6, 5.4), dpi=200)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PAPER)
    oc = np.array([0.0, 0.0, 0.0])
    # 传感器平面 ⊥ 光轴（z=1.6）：cs 在网格角点（像素自左上角起算，p.55），ys 向下 = −y 方向
    step = 0.3                                        # 网格间距 = 一格像素间距（sx、sy 与之等长）
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0])
    cs = np.array([-1.2, 1.0, 1.6])
    W_s, H_s = 2.4, 1.8
    for gv in np.arange(0, W_s + 1e-9, step):         # 竖网格线
        ax.plot(*zip(cs + gv * e1, cs + gv * e1 + H_s * e2), color="#c9b992", lw=0.8)
    for hv in np.arange(0, H_s + 1e-9, step):         # 横网格线
        ax.plot(*zip(cs + hv * e2, cs + hv * e2 + W_s * e1), color="#c9b992", lw=0.8)
    # 视锥：光心 → 传感器四角
    for c1 in (0.0, W_s):
        for c2 in (0.0, H_s):
            ax.plot(*zip(oc, cs + c1 * e1 + c2 * e2), color=BLUE, lw=1.0, ls=":")
    # 3D 点与射线：p = (1.6/3.2)·pc，严格落在 Oc→pc 射线与传感器平面的交点（p.56：p = s·pc）
    pc = np.array([0.7, -0.5, 3.2])
    p_img = pc * (1.6 / 3.2)
    ax.plot(*zip(oc, pc), color=ORANGE, lw=2.2)
    ax.plot(*zip(pc, p_img), color=ORANGE, lw=1.2, ls="--")
    ax.scatter(*oc, color=INK, s=55)
    ax.scatter(*cs, color=BLUE, s=45)
    ax.scatter(*pc, color=ORANGE, s=55)
    ax.scatter(*p_img, color=ORANGE, s=45, marker="s")
    ax.text(*oc + np.array([-0.35, -0.55, -0.35]), "光心 Oc", color=INK, fontsize=11)
    ax.text(*(cs + np.array([-1.05, 0.55, -0.1])), "cs（传感器原点·左上角）", color=BLUE, fontsize=10.5)
    ax.text(*(pc + np.array([0.1, 0.05, 0.18])), "pc", color=ORANGE, fontsize=12)
    ax.text(*(p_img + np.array([0.3, -0.36, 0.02])), "p（像点）", color=ORANGE, fontsize=11)
    # 像素间距箭头：长度 = 网格间距（一格），sx 沿 xs 向右、sy 沿 ys 向下；标签沿两边拉开避让
    a1 = cs + 4 * step * e1 + np.array([0.0, 0.16, 0.05])     # 顶边右半段外侧
    q1 = a1 + step * e1
    ax.plot(*zip(a1, q1), color=BROWN, lw=2.6)
    ax.text(*(q1 + np.array([0.05, 0.14, 0.02])), "sx（一格，xs 向右）", color=BROWN, fontsize=10)
    a2 = cs + 3 * step * e2 + np.array([-0.16, 0.0, 0.05])    # 左边下半段外侧
    q2 = a2 + step * e2
    ax.plot(*zip(a2, q2), color=BROWN, lw=2.6)
    ax.text(*(q2 + np.array([-0.62, -0.14, 0.02])), "sy（一格，ys 向下）", color=BROWN, fontsize=10)
    ax.set_title("相机中心点 pc 沿射线投到传感器平面 p（讲者重绘·原书图未使用）", fontsize=12, color=INK)
    ax.view_init(elev=16, azim=-55)
    ax.set_box_aspect((1, 1, 0.8))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def draw_fig2_9(path):
    """Figure 2.9（p.57）：简化内参——f、(cx,cy)、W×H；像平面在节点之前、y 翻转。"""
    fig, ax = _new_ax()
    ax.annotate("", xy=(10.0, 0), xytext=(0.5, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    ax.text(9.9, -0.55, "z（光轴）", color=INK, fontsize=11)
    ax.annotate("", xy=(4.15, -1.2), xytext=(4.15, 3.0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.8))
    ax.text(4.3, 2.6, "y 向下（翻转后，与成像库行坐标一致）", color=ORANGE, fontsize=10.5)
    # 像平面（在节点前）
    ax.fill_betweenx([-1.2, 3.0], 4.4, 4.8, color="#eadfc8", alpha=0.9)
    ax.plot([4.4, 4.8], [3.0, 3.0], color="#c9b992", lw=1)
    ax.plot([4.4, 4.8], [-1.2, -1.2], color="#c9b992", lw=1)
    ax.text(3.75, 3.25, "像平面（W×H 像素）", color=INK, fontsize=11)
    # 节点/光心
    ax.scatter([0.5], [0], color=INK, s=55)
    ax.text(0.15, -0.62, "节点（投影中心）", color=INK, fontsize=11)
    # 主光线与焦距标注
    ax.plot([0.5, 4.6], [0, 0], color=BLUE, lw=1.4, ls="--")
    ax.annotate("", xy=(4.6, 0), xytext=(0.5, 0), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6))
    ax.text(2.2, 0.22, "f（焦距）", color=BLUE, fontsize=12)
    # 主点：光轴与像平面的交点（y=0，落光轴上）；像素点在主点下方（y 向下约定 ys>0）
    ax.scatter([4.6], [0], color=ORANGE, s=55, marker="s", zorder=5)
    ax.text(4.95, -0.45, "主点 (cx, cy)", color=ORANGE, fontsize=11)
    ax.scatter([4.6], [-0.95], color=BLUE, s=40, marker="s")
    ax.text(4.95, -1.06, "像素 (xs, ys)", color=BLUE, fontsize=10.5)
    ax.set_xlim(0, 10.5); ax.set_ylim(-1.9, 3.9)
    ax.set_title("简化成像模型：像平面在节点之前、y 轴翻转（讲者重绘·原书图未使用）", fontsize=12, color=INK)
    _tight(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def draw_fig2_10(path):
    """Figure 2.10（p.58）：中心投影——f、W、水平视场角 θH 的三角关系。"""
    fig, ax = _new_ax()
    zc, W2, f = 0.0, 2.6, 5.0
    theta2 = np.arctan(W2 / f)
    ax.scatter([zc], [0], color=INK, s=55)
    ax.text(-0.35, -0.4, "光心", color=INK, fontsize=11)
    # 像平面
    ax.fill_betweenx([-W2, W2], f, f + 0.22, color="#eadfc8")
    ax.text(f + 0.45, -W2 + 0.25, "像平面宽 W", color=INK, fontsize=11)
    ax.plot([f, f], [-W2, W2], color="#b8a888", lw=1.2)
    # 两条视线（半视场角）
    for s in (+1, -1):
        ax.plot([zc, f], [0, s * W2], color=ORANGE, lw=2.0)
    ax.plot([zc, f], [0, 0], color=BLUE, lw=1.2, ls="--")
    ax.annotate("", xy=(f + 0.25, W2), xytext=(f + 0.25, -W2), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    # 角弧
    tt = np.linspace(0, theta2, 40)
    ax.plot(zc + 2.0 * np.cos(tt), 2.0 * np.sin(tt), color=ORANGE, lw=1.6)
    ax.plot(zc + 2.0 * np.cos(tt), -2.0 * np.sin(tt), color=ORANGE, lw=1.6)
    ax.text(zc + 2.45, 0.55, "θH/2", color=ORANGE, fontsize=12)
    ax.text(zc + 2.45, -0.95, "θH/2", color=ORANGE, fontsize=12)
    ax.annotate("", xy=(f, 0), xytext=(zc, 0), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.6, ls="-"))
    ax.text(zc + f / 2 - 0.35, 0.3, "f", color=BLUE, fontsize=13)
    # 3D 点示意
    ax.scatter([7.6], [1.4], color=BLUE, s=45)
    ax.text(7.75, 1.28, "(X, Y, Z)", color=BLUE, fontsize=10.5)
    ax.scatter([f + 0.11], [0.7], color=BLUE, s=35, marker="s")
    ax.text(f + 0.5, 0.55, "x（像点）", color=BLUE, fontsize=10.5)
    ax.set_xlim(-0.8, 8.6); ax.set_ylim(-3.3, 3.3)
    ax.set_title("中心投影：tan(θH/2) = (W/2)/f（讲者重绘·原书图未使用）", fontsize=12, color=INK)
    _tight(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def draw_fig2_11(path):
    """Figure 2.11（p.60）：左 = regular disparity（inverse depth）；右 = projective depth（参考平面视差）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    for ax, (title, ds, note, plane) in zip(axes, [
            ("左：regular disparity（d = inverse depth 逆深度）", [1.0, 0.67, 0.5], "d=1/z：越远 d 越小", None),
            ("右：projective depth（相对参考平面的视差）", [0.5, 0.0, -0.25], "d=0 落在参考平面上", (5.0, 9.8))]):
        ax.set_facecolor(PAPER)
        ax.scatter([0], [0], color=INK, s=55)
        ax.text(-0.75, -0.28, "光心 C", color=INK, fontsize=10.5)
        ax.annotate("", xy=(9.4, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        ax.text(9.0, -0.35, "Z", color=INK, fontsize=11)
        # 像平面
        ax.fill_betweenx([-1.3, 1.3], 4.0, 4.25, color="#eadfc8")
        ax.text(3.35, 1.5, "像平面", color=INK, fontsize=10.5)
        # 参考平面线（右栏）：过 d=0 的场景点 (7.6, 0.55)；正斜率使 d=0.5 在线上方、d=−0.25 在线下方
        if plane is not None:
            k = 0.375
            y_at = lambda x: 0.55 + k * (x - 7.6)
            ax.plot([plane[0], plane[1]], [y_at(plane[0]), y_at(plane[1])], color=ORANGE, lw=1.6)
            ax.text(plane[0] + 0.05, y_at(plane[0]) + 0.16, "参考平面（d=0）", color=ORANGE, fontsize=10)
        # 三条深度的 3D 点与其投影像点（d 标签错层排布，避免互撞/穿点）
        lab_dy = {6.6: 0.30, 7.6: 0.65, 8.6: 0.30}
        for d, dx in zip(ds, [6.6, 7.6, 8.6]):
            ypt = 0.55 * (1 if d >= 0 else -0.6)
            ax.plot([0, dx], [0, ypt], color=BLUE, lw=1.1, ls=":")
            ax.scatter([dx], [ypt], color=ORANGE, s=45)
            ax.scatter([4.12], [ypt * (4.12 / dx)], color=BLUE, s=32, marker="s")
            ax.text(dx - 0.32, ypt + (lab_dy[dx] if d >= 0 else -0.42),
                    f"d={d:g}", color=ORANGE, fontsize=10.5)
        ax.text(5.2, -1.75, note, color=BROWN, fontsize=10.5)
        ax.set_xlim(-1.2, 9.8); ax.set_ylim(-2.15, 2.0)
        ax.set_title(title, fontsize=11.5, color=INK)
        _tight(ax)
    fig.text(0.5, 0.035, "两种视差：逆深度 vs 参考平面投影视差（讲者重绘·原书图未使用）",
             ha="center", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def draw_fig2_12(path):
    """Figure 2.12（p.62）：(a) 3D 点投进两图（带视差 d）；(b) 共面点诱导平面单应。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), dpi=200)
    fig.patch.set_facecolor(PAPER)
    # (a)
    ax = axes[0]
    ax.set_facecolor(PAPER)
    ax.scatter([5.2], [2.6], color=ORANGE, s=60)
    ax.text(5.4, 2.62, "p = (X, Y, Z, 1)", color=ORANGE, fontsize=11)
    for x0, lab, col in [(1.6, "相机 0", BLUE), (8.8, "相机 1", BROWN)]:
        ax.scatter([x0], [0], color=col, s=55)
        ax.text(x0 - 0.5, -0.5, lab, color=col, fontsize=11)
        ax.plot([x0, 5.2], [0, 2.6], color=col, lw=1.3, ls="--")
        ax.fill_betweenx([0.4, 1.1], x0 - 0.75, x0 + 0.75, color="#eadfc8")
        ax.text(x0 - 0.42, 1.25, "image plane", color=INK, fontsize=9.5)
        ax.scatter([x0 + (5.2 - x0) * (0.75 / 2.6)], [0.75], color=col, s=36, marker="s")
    ax.text(2.55, 1.75, r"$\tilde{x}_0=(x_0,y_0,1,d_0)$", color=BLUE, fontsize=11)
    ax.text(7.0, 1.75, r"$\tilde{x}_1=(x_1,y_1,1,d_1)$", color=BROWN, fontsize=11)
    ax.text(3.4, 0.15, r"$M_{10}$", color=INK, fontsize=13)
    ax.annotate("", xy=(8.0, 0.05), xytext=(2.4, 0.05), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.9, 3.3)
    ax.set_title("(a) 3D 点与两图的视差坐标", fontsize=11.5, color=INK)
    _tight(ax)
    # (b)
    ax = axes[1]
    ax.set_facecolor(PAPER)
    ax.plot([0.6, 9.8], [1.15, 1.15], color=ORANGE, lw=2.0)
    ax.text(6.6, 1.42, "参考平面 " + r"$\hat{n}_0\cdot p + c_0 = 0$", color=ORANGE, fontsize=11)
    ax.scatter([2.2, 5.2, 8.0], [1.15, 1.15, 1.15], color=ORANGE, s=45)
    for x0 in (1.6, 8.8):
        ax.scatter([x0], [0], color=BLUE, s=55)
        ax.plot([x0, x0], [0, 0.42], color=BLUE, lw=1.0)
        ax.fill_betweenx([0.42, 0.92], x0 - 0.75, x0 + 0.75, color="#eadfc8")
    for xp in (2.2, 5.2, 8.0):
        ax.plot([1.6, xp], [0, 1.15], color=BLUE, lw=0.9, ls=":")
        ax.plot([8.8, xp], [0, 1.15], color=BROWN, lw=0.9, ls=":")
        ax.scatter([1.6 + (xp - 1.6) * 0.42 / 1.15], [0.42], color=BLUE, s=26, marker="s")
        ax.scatter([8.8 + (xp - 8.8) * 0.42 / 1.15], [0.42], color=BROWN, s=26, marker="s")
    ax.text(3.4, 0.6, r"$\tilde{x}_1 \sim \tilde{H}_{10}\,\tilde{x}_0$", color=INK, fontsize=12)
    ax.text(6.35, 0.6, "（8 参数单应）", color=INK, fontsize=10.5)
    ax.set_xlim(0, 10.4); ax.set_ylim(-0.9, 3.3)
    ax.set_title("(b) 共面点诱导平面单应", fontsize=11.5, color=INK)
    _tight(ax)
    fig.text(0.5, 0.035, "双视图映射：视差坐标与平面单应（讲者重绘·原书图未使用）",
             ha="center", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, facecolor=PAPER)
    plt.close(fig)


def sync_and_md5():
    """同步检查（先比后写）：先逐文件比对 slides/figures/ 的 md5，漂移判 FAIL；再覆盖同步。

    漂移说明 outputs/ 与 slides/figures/ 不同步（如刚改过绘图代码）——本轮判 FAIL，
    覆盖后重跑至无漂移为止，保证最终汇报的 PASS 状态下两边逐字节一致。
    """
    import hashlib
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    figdir = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures"
    for f in sorted(outdir.glob("*.png")):
        print(f"[md5] {f.name}  {hashlib.md5(f.read_bytes()).hexdigest()}")
    files = sorted(outdir.glob("*.png"))
    drift = []
    for f in files:
        g = figdir / f.name
        if not g.exists():
            drift.append(f.name)          # 目标缺失也算漂移——缺文件必须同步并显式汇报
            print(f"[sync-DRIFT] {f.name} 在 slides/figures/ 缺失——本轮判 FAIL（已补齐，重跑至一致）")
        elif hashlib.md5(g.read_bytes()).hexdigest() != hashlib.md5(f.read_bytes()).hexdigest():
            drift.append(f.name)
            print(f"[sync-DRIFT] {f.name} 与 slides/figures/ 不一致——本轮判 FAIL（已覆盖，重跑至一致）")
    for f in files:
        (figdir / f.name).write_bytes(f.read_bytes())
        print(f"[demo] 已同步 {figdir / f.name}")
    ok = len(files) > 0 and not drift
    print(f"[verify] 同步检查（先比后写，漂移判 FAIL）  共 {len(files)} 张、漂移 {len(drift)} 张  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def main():
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(exist_ok=True)
    draw_fig2_8(outdir / "e016_fig2_8.png")
    draw_fig2_9(outdir / "e016_fig2_9.png")
    draw_fig2_10(outdir / "e016_fig2_10.png")
    draw_fig2_11(outdir / "e016_fig2_11.png")
    draw_fig2_12(outdir / "e016_fig2_12.png")
    print("[demo] 五张重绘图已存 outputs/")
    assert_Ms()
    assert_K_and_P()
    assert_focal_units()
    K, Rc, t, Pt, _ = assert_Ptilde()
    assert_two_views(K, Rc, t)
    assert_object_centered()
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
