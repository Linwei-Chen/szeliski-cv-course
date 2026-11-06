"""
E015 · 3D 到 2D：相机投影的代数（§2.1.2–§2.1.4 投影模型，印刷 pp.43–54）

数值实验：把三维变换族谱（Table 2.2）、三维旋转三表示（轴角/Rodrigues、
指数扭转、单位四元数 + slerp）与四种投影模型（正交、缩放正交、
para-perspective、透视）全部算一遍。

断言（24 项数据 + 1 项同步）：
  #1  Table 2.2 DoF 阶梯 3/6/7/12/15（各家族参数化的雅可比秩 = 表值，p.44）
  #2  平移结构：[I t] 4×4 作用 = x+t（式 2.23，p.44）
  #3  R 正交：RRᵀ=I 且 |R|=+1（式 2.24，p.44）
  #4  刚体保长度：3D 两点距离不变（p.44 Table 2.2 lengths）
  #5  旋转中心形式：R(x−c) = Rx − Rc（式 2.25，p.44）
  #6  相似保夹角：线与线、面与面（法向）夹角在 sR 下不变（式 2.26，p.45）
  #7  仿射保平行：两条不同位置的 3D 平行线段经任意 A 后像方向仍平行（式 2.27，p.45）
  #8  仿射反例：diag(2,1,1) 保平行但线面夹角改变
  #9  射影保直线：3D 共线四点过 H̃ 仍共线（式 2.28，p.45）
  #10 射影尺度等价：H̃ 与 2H̃ 归一化后同一点（式 2.28，p.45）
  #11 Rodrigues 几何构造：v∥/v⊥/v× 分解与 R(n̂,θ)v 逐点一致（式 2.29–2.34，pp.46–47；Fig 2.5 重绘）
  #12 Rodrigues 矩阵正交：RRᵀ=I、|R|=+1
  #13 小旋转线性化：R(ω) ≈ I+[ω]×，误差/θ² → 1/√2（式 2.35，p.47，θ 扫描验证）
  #14 旋转导数：∂Rv/∂ωᵀ = −[v]×（式 2.36，p.47，有限差分验证）
  #15 指数扭转：lim (I+[θn̂]×/k)^k = exp[ω]× = Rodrigues（式 2.37–2.38，pp.47–48）
  #16 四元数轴角来源：q=(sin(θ/2)n̂, cos(θ/2))，R(q) 与 Rodrigues 一致（式 2.39–2.41，p.49）
  #17 四元数乘法复合：R(q₀q₁) = R(q₀)R(q₁)（式 2.42，p.49）
  #18 四元数求逆与对径：R(q⁻¹)=R(q)ᵀ、R(q)=R(−q)（式 2.43 + dual covering，pp.49–50）
  #19 slerp 中点：轴不变、旋转角减半（Algorithm 2.1，p.50）
  #20 正交投影：扔 z 保 x,y，齐次版保 w（式 2.46–2.47，p.51）
  #21 缩放正交与 para-perspective 保平行、透视不保（式 2.48–2.50，p.53–54）
  #22 透视保直线：共线点除以 z 后仍共线（式 2.50–2.51，p.54）
  #23 透视反例 + 消失点兑现：3D 平行线像方向不平行，像延长线交于 (d_x/d_z, d_y/d_z)（式 2.50 推论；Figure 2.7 图注 p.52）
  #24 逆深度：znear=1、zfar→∞、第三行翻号后 归一化第三元 = 1−1/z，d = 1−第三元 = 1/z（式 2.52 特例 + disparity 段，p.54；讲者推演）
  #S  同步检查：输出图与 slides/figures/ 逐文件 md5 比对
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.patches as patches
from matplotlib import font_manager

# 中文字体（同 E014：macOS 优先 Hiragino Sans GB；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import pathlib

rng = np.random.default_rng(20260315)
INK, BLUE, ORANGE, BROWN, PAPER = "#1a2a3a", "#3a5a8c", "#d96830", "#8c6d46", "#faf6ee"
RESULTS = []


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[verify] {name}  {detail}  [{'PASS' if ok else 'FAIL'}]")


# ——— 三维变换构造（与原书式 2.23–2.28 对应）—————————————————————
def T3(t):
    """式(2.23)：3D 平移齐次矩阵 [I t; 0 1]（4×4）。"""
    M = np.eye(4)
    M[:3, 3] = t
    return M


def R3(n, theta):
    """式(2.34)：Rodrigues 公式 R(n̂,θ) = I + sinθ[n̂]× + (1−cosθ)[n̂]²×。"""
    K = hat(n)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def hat(v):
    """式(2.32)：叉积算子矩阵形式 [v]×。"""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_hat(omega):
    """式(2.38)：exp[ω]× = Σ [ω]×ᵐ/m!（Taylor 部分和，[n̂]× 循环恒等式保证收敛）。"""
    E = np.eye(3)
    term = np.eye(3)
    for m in range(1, 40):
        term = term @ hat(omega) / m
        E = E + term
    return E


def S3(s, n, theta, t):
    """式(2.26)：3D 相似变换 [sR t; 0 1]。"""
    M = np.eye(4)
    M[:3, :3] = s * R3(n, theta)
    M[:3, 3] = t
    return M


def A3(A34):
    """式(2.27)：3D 仿射齐次矩阵（A 为 3×4，底行 [0 0 0 1]）。"""
    M = np.vstack([A34, [0.0, 0.0, 0.0, 1.0]])
    return M


def apply_h3(M, pts):
    """4×4 齐次作用 + 归一化：pts (N,3) → (N,3)。"""
    P = np.c_[pts, np.ones(len(pts))]
    out = P @ M.T
    return out[:, :3] / out[:, 3:4]


def quat_R(q):
    """式(2.41)：R(q)，q=(x,y,z,w)。"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_mul(q0, q1):
    """式(2.42)：q₂ = q₀q₁ = (v₀×v₁ + w₀v₁ + w₁v₀, w₀w₁ − v₀·v₁)，q=(v,w)。"""
    v0, w0 = q0[:3], q0[3]
    v1, w1 = q1[:3], q1[3]
    v = np.cross(v0, v1) + w0 * v1 + w1 * v0
    w = w0 * w1 - np.dot(v0, v1)
    return np.r_[v, w]


def slerp(q0, q1, alpha):
    """Algorithm 2.1（p.50）：球面线性插值。"""
    qr = quat_mul(q1, quat_conj(q0))          # qr = q1/q0
    if qr[3] < 0:
        qr = -qr
    theta_r = 2.0 * np.arctan2(np.linalg.norm(qr[:3]), qr[3])
    n_r = qr[:3] / max(np.linalg.norm(qr[:3]), 1e-15)
    theta_a = alpha * theta_r
    qa = np.r_[np.sin(theta_a / 2) * n_r, np.cos(theta_a / 2)]
    return quat_mul(qa, q0)


def quat_conj(q):
    """式(2.43) 伴随段：求逆 = 翻转 w（或 v）的符号。"""
    out = q.copy()
    out[3] = -out[3]
    return out


# —— 1. Table 2.2 DoF 阶梯：3/6/7/12/15（各家族参数化的雅可比秩 = 表值，p.44）—————
def _jac_rank(Mfun, p0):
    """对参数化 M(p) 在 p0 处做中心差分，返回雅可比的秩（= 该家族的自由参数维数）。"""
    p0 = np.asarray(p0, float)
    cols = []
    for i in range(len(p0)):
        e = np.zeros_like(p0)
        e[i] = 1e-7
        cols.append(((Mfun(p0 + e) - Mfun(p0 - e)) / 2e-7).ravel())
    return int(np.linalg.matrix_rank(np.array(cols).T, tol=1e-3))

n_j = unit(rng.uniform(-1, 1, 3))
th_j = rng.uniform(0.5, 1.5)
t_j = rng.uniform(-2, 2, 3)
s_j = rng.uniform(1.1, 2.0)
A_j = rng.uniform(-0.8, 0.8, (3, 4))
H_j = rng.uniform(-0.8, 0.8, 15)
fams = [
    ("平移", lambda p: T3(p), t_j, 3),
    ("刚体", lambda p: T3(p[3:]) @ np.vstack([np.c_[exp_hat(p[:3]), np.zeros(3)], [0, 0, 0, 1]]),
     np.r_[n_j * th_j, t_j], 6),
    ("相似", lambda p: T3(p[4:]) @ np.vstack([np.c_[p[3] * exp_hat(p[:3]), np.zeros(3)], [0, 0, 0, 1]]),
     np.r_[n_j * th_j, s_j, t_j], 7),
    ("仿射", lambda p: np.vstack([p.reshape(3, 4), [0.0, 0.0, 0.0, 1.0]]), A_j.ravel(), 12),
]
ok = True
det1 = []
for name, Mfun, p0, dof in fams:
    r = _jac_rank(Mfun, p0)
    det1.append(f"{name}{r}")
    ok = ok and r == dof
# 射影族不做「参数直接填矩阵元」的平凡检验，而是检验尺度规范自由：
# H 有 16 个矩阵元，但 H 与 cH 归一化后是同一映射——
# 8 个 3D 点的「归一化像点 对 H 全部 16 元」的雅可比，秩应为 16−1 = 15
rs_h = np.random.RandomState(0xE015)          # 独立流，不扰动主 rng 的后续抽样
X_h = np.c_[rs_h.uniform(-1, 1, (8, 3)), np.ones(8)]
p0_h = np.r_[H_j, 1.0]                        # 沿用主流已抽的 H_j，保持后续数值不变
def _proj_img(p):
    q = X_h @ p.reshape(4, 4).T
    return q[:, :3] / q[:, 3:4]          # 归一化 3D 点（三坐标都要：H 第三行也参与）
cols_h = []
for i in range(16):
    e = np.zeros(16)
    e[i] = 1e-7
    cols_h.append(((_proj_img(p0_h + e) - _proj_img(p0_h - e)) / 2e-7).ravel())
r_h = int(np.linalg.matrix_rank(np.array(cols_h).T, tol=1e-3))
det1.append(f"射影{r_h}")
ok = ok and r_h == 15
verify("Table 2.2 DoF 阶梯：3/6/7/12/15（家族参数化的雅可比秩 = 表值）", ok,
       " · ".join(det1)
       + "（平移/仿射为直接参数化的自检；刚体/相似为构造检验；射影为归一化像点雅可比的规范自由检验）")

# —— 2. 平移结构：[I t] 4×4 作用于 x̄ = x+t（式 2.23）———————————————————
x = rng.uniform(-3, 3, 3)
t = rng.uniform(-3, 3, 3)
ok = np.allclose(apply_h3(T3(t), x[None, :])[0], x + t)
verify("平移结构：[I t;0 1] 作用于 x̄ 等于 x+t（式 2.23）", ok,
       f"t=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})")

# —— 3. R 正交：RRᵀ=I 且 |R|=+1（式 2.24 原文断言 RRT = I）————————————————
ns = rng.uniform(-1, 1, (5, 3))
ths = rng.uniform(-np.pi, np.pi, 5)
ok = all(np.allclose(R3(unit(n), th) @ R3(unit(n), th).T, np.eye(3), atol=1e-12)
         and abs(np.linalg.det(R3(unit(n), th)) - 1) < 1e-12 for n, th in zip(ns, ths))
verify("R 正交：RRᵀ=I 且 |R|=+1（式 2.24）", ok,
       f"5 组随机 (n̂,θ)")

# —— 4. 刚体保长度：3D 两点距离在 (R,t) 作用下不变（Table 2.2：rigid 保 lengths）——
p, q = rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)
M = T3(t) @ np.vstack([np.c_[R3(unit(ns[0]), ths[0]), np.zeros(3)], [0, 0, 0, 1]])
p2, q2 = apply_h3(M, np.vstack([p, q]))
ok = abs(np.linalg.norm(p2 - q2) - np.linalg.norm(p - q)) < 1e-9
verify("刚体保长度：3D 两点距离不变（Table 2.2 p.44）", ok,
       f"‖p−q‖={np.linalg.norm(p - q):.4f}")

# —— 5. 旋转中心形式：x′=R(x−c)=Rx−Rc（式 2.25，c 常取相机中心）——————————
c = rng.uniform(-2, 2, 3)
Rm = R3(unit(ns[1]), ths[1])
ok = np.allclose(Rm @ (x - c), Rm @ x - Rm @ c)
verify("旋转中心形式：R(x−c) = Rx − Rc（式 2.25）", ok,
       f"c=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")

# —— 6. 相似保夹角：线与线、面与面（法向）夹角在 sR 下不变（式 2.26，p.45 preserves
#        angles between lines and planes）———————————————————————————————
s = rng.uniform(0.5, 2.5)
M_sim = S3(s, unit(ns[2]), ths[2], rng.uniform(-2, 2, 3))
Rs = M_sim[:3, :3]
u, v = rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)
ang0 = np.arccos(np.clip(unit(u) @ unit(v), -1, 1))
ang1 = np.arccos(np.clip(unit(Rs @ u) @ unit(Rs @ v), -1, 1))
n1, n2 = rng.uniform(-3, 3, 3), rng.uniform(-3, 3, 3)   # 两平面法向
pang0 = np.arccos(np.clip(unit(n1) @ unit(n2), -1, 1))
pang1 = np.arccos(np.clip(unit(Rs @ n1) @ unit(Rs @ n2), -1, 1))
ok = abs(ang0 - ang1) < 1e-9 and abs(pang0 - pang1) < 1e-9
verify("相似保夹角：线夹角与面法向夹角在 sR 下不变（式 2.26）", ok,
       f"线 {np.degrees(ang0):.2f}° · 面 {np.degrees(pang0):.2f}° 均不变")

# —— 7. 仿射保平行：两条不同位置的 3D 平行线段经任意 A 后像方向仍平行（式 2.27）———
A34 = rng.uniform(-2, 2, (3, 4))
while abs(np.linalg.det(A34[:, :3])) < 0.3:
    A34[:, :3] = rng.uniform(-2, 2, (3, 3))
d7 = rng.uniform(-2, 2, 3)                  # 公共方向
p7 = rng.uniform(-2, 2, 3)                  # 线 1 起点
q7 = rng.uniform(-2, 2, 3)                  # 线 2 起点（与线 1 不同处）
mu7 = rng.uniform(0.5, 2.0)                 # 线 2 长度与线 1 不同
seg1 = np.vstack([p7, p7 + d7])
seg2 = np.vstack([q7, q7 + mu7 * d7])
img1 = seg1 @ A34[:, :3].T + A34[:, 3]      # 含平移列：x′ = A x̄ 的完整作用
img2 = seg2 @ A34[:, :3].T + A34[:, 3]
e1_7, e2_7 = img1[1] - img1[0], img2[1] - img2[0]
cr7 = np.cross(e1_7, e2_7)
ok = np.linalg.norm(cr7) < 1e-9 and np.linalg.norm(e1_7) > 1e-3
verify("仿射保平行：两条不同位置的 3D 平行线段经任意 A 后像方向仍平行（式 2.27）", ok,
       f"μ={mu7:.2f} · 像方向叉积范数={np.linalg.norm(cr7):.1e}")

# —— 8. 仿射反例：diag(2,1,1) 保平行但线夹角改变（6→12 DoF 的能力边界）———————
D = np.diag([2.0, 1.0, 1.0])
u2 = np.array([1.0, 1.0, 0.0])
v2 = np.array([1.0, 0.0, 0.0])
a0 = np.degrees(np.arccos(unit(u2) @ unit(v2)))
a1 = np.degrees(np.arccos(unit(D @ u2) @ unit(D @ v2)))
ok = abs(a1 - a0) > 1e-3
verify("仿射反例：diag(2,1,1) 保平行但夹角改变（仿射不保角）", ok,
       f"角 {a0:.1f}°→{a1:.1f}°")

# —— 9. 射影保直线：3D 共线四点过 H̃ 仍共线（式 2.28）———————————————————
l_pts = rng.uniform(-2, 2, (2, 3))
lams = rng.uniform(0.2, 0.8, 2)
extra = np.vstack([(1 - l) * l_pts[0] + l * l_pts[1] for l in lams])
four = np.vstack([l_pts, extra])
H4 = rng.uniform(-2, 2, (4, 4))
H4[3, :] = [0.1, -0.2, 0.3, 1.0]
while abs(np.linalg.det(H4)) < 0.5:
    H4 = rng.uniform(-2, 2, (4, 4)); H4[3, :] = [0.1, -0.2, 0.3, 1.0]
out = apply_h3(H4, four)
d1 = out[1] - out[0]
d_rest = out[2:] - out[0]
ok = all(abs(np.cross(d1, d)) .max() < 1e-9 for d in d_rest)
verify("射影保直线：3D 共线四点过 H̃ 仍共线（式 2.28）", ok,
       "2 个内分点均共线")

# —— 10. 射影尺度等价：H̃ 与 2H̃ 归一化后同一点（p.45 As in 2D…normalized）——————
x1 = apply_h3(H4, four[:2])
x2 = apply_h3(2.0 * H4, four[:2])
ok = np.allclose(x1, x2, atol=1e-12)
verify("射影尺度等价：H̃ 与 2H̃ 归一化后同一点（式 2.28）", ok,
       "2 组随机点逐点一致")

# —— 11. Rodrigues 几何构造：分解式 u = v∥ + cosθ·v⊥ + sinθ·v× 与 R(n̂,θ)v 一致
#         （式 2.29–2.33 推出 2.34；Figure 2.5 流程）——————————————————————
n = unit(rng.uniform(-1, 1, 3))
th = rng.uniform(0.3, 2.5)
v = rng.uniform(-3, 3, 3)
v_par = n * (n @ v)                       # 式(2.29)：v∥ = n̂(n̂·v) = (n̂n̂ᵀ)v
v_perp = v - v_par                        # 式(2.30)：v⊥ = (I − n̂n̂ᵀ)v
v_cross = np.cross(n, v)                  # 式(2.31)：v× = [n̂]×v
u_geo = v_par + np.cos(th) * v_perp + np.sin(th) * v_cross   # 式(2.33) 拼装
u_R = R3(n, th) @ v                       # 式(2.34) 矩阵
ok = np.allclose(u_geo, u_R, atol=1e-12)
verify("Rodrigues 几何构造：v∥+cosθ·v⊥+sinθ·v× 与 R(n̂,θ)v 一致（式 2.33–2.34）", ok,
       f"θ={th:.2f} · 逐点一致")

# —— 12. Rodrigues 矩阵正交：RRᵀ=I、|R|=+1（旋转矩阵本征性质）————————————————
Rm = R3(n, th)
ok = np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-12) and abs(np.linalg.det(Rm) - 1) < 1e-12
verify("Rodrigues 矩阵正交：RRᵀ=I 且 |R|=+1", ok,
       f"det={np.linalg.det(Rm):.6f}")

# —— 13. 小旋转线性化：R(ω) ≈ I+[ω]×，误差/θ² → 1/√2（式 2.35；θ 扫描验证收敛速率）——
# 一阶余项首项 = [ω]²/2，‖[n̂]²×‖_F = √2，故 ‖err‖/θ² → 1/√2 ≈ 0.7071
n_lin = unit(np.array([0.3, -0.5, 0.81]))
ths_lin = (0.1, 0.01, 0.001)
ratios = []
for th_l in ths_lin:
    Rm_l = R3(n_lin, th_l)
    err_l = np.linalg.norm(Rm_l - (np.eye(3) + hat(th_l * n_lin)))
    ratios.append(err_l / th_l ** 2)
ok = all(abs(r - 1 / np.sqrt(2)) < 0.01 for r in ratios) and ratios[0] < ratios[-1]
# 高阶余项交叉项为负：ratio² = 1/2 − θ²/36 + …（‖[n̂]×‖²F=‖[n̂]²×‖²F=2 代入交叉项），故比值从下方单调逼近 1/√2
verify("小旋转线性化：误差/θ² → 1/√2 ≈ 0.707（式 2.35，θ 扫描）", ok,
       " · ".join(f"θ={t}:{r:.4f}" for t, r in zip(ths_lin, ratios)))

# —— 14. 旋转导数：∂Rv/∂ωᵀ = −[v]×（式 2.36；小旋转极限下对精确 R 有限差分）—————
v_t = rng.uniform(-3, 3, 3)
omega_s = 1e-3 * unit(rng.uniform(-1, 1, 3))   # ‖ω‖=10⁻³：式(2.36) 的适用范围
eps = 1e-6
J_fd = np.zeros((3, 3))
for i in range(3):
    dω = np.zeros(3); dω[i] = eps
    R_plus = R3(unit(omega_s + dω), np.linalg.norm(omega_s + dω))
    R_minus = R3(unit(omega_s - dω), np.linalg.norm(omega_s - dω))
    J_fd[:, i] = ((R_plus @ v_t) - (R_minus @ v_t)) / (2 * eps)
J_exact = -hat(v_t)
dev = np.abs(J_fd - J_exact).max()
ok = dev < 5e-3                             # 偏差由 O(‖ω‖) 非线性项主导，量级 10⁻³
verify("旋转导数：小旋转下 ∂Rv/∂ωᵀ = −[v]×（式 2.36，‖ω‖=10⁻³ 差分）", ok,
       f"最大偏差 {dev:.2e}")

# —— 15. 指数扭转：lim (I+[θn̂]×/k)^k = exp[ω]× = R(n̂,θ)（式 2.37–2.38）————————
# 截断误差 ~ θ²/(2k)：k=10⁶（matrix_power 二进制快速幂，仅 ~20 次乘法）时 ≈ 2.9×10⁻⁶（θ=2.39）
omega = th * n
k = 1_000_000
R_pow = np.linalg.matrix_power(np.eye(3) + hat(omega) / k, k)
E = exp_hat(omega)
ok = (np.allclose(R_pow, Rm, atol=1e-5) and np.allclose(E, Rm, atol=1e-10))
verify("指数扭转：(I+[ω]×/k)^k 与 exp[ω]× 均收敛到 Rodrigues（式 2.37–2.38）", ok,
       f"k={k} · 两条路最大偏差 {max(np.abs(R_pow - Rm).max(), np.abs(E - Rm).max()):.2e}")

# —— 16. 四元数轴角来源：q=(sin(θ/2)n̂, cos(θ/2))，R(q) 与 Rodrigues 一致
#         （式 2.39–2.41，p.49）——————————————————————————————————————————————
q = np.r_[np.sin(th / 2) * n, np.cos(th / 2)]
Rq = quat_R(q)
ok = np.allclose(Rq, Rm, atol=1e-12)
verify("四元数轴角来源：R(q) 与 Rodrigues 一致（式 2.39–2.41）", ok,
       f"q=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f},{q[3]:.3f})")

# —— 17. 四元数乘法复合：R(q₀q₁) = R(q₀)R(q₁)（式 2.42，非交换也一并核）——————
q0 = np.r_[np.sin(ths[3] / 2) * unit(ns[3]), np.cos(ths[3] / 2)]
q1 = np.r_[np.sin(ths[4] / 2) * unit(ns[4]), np.cos(ths[4] / 2)]
q2 = quat_mul(q0, q1)
ok = (np.allclose(quat_R(q2), quat_R(q0) @ quat_R(q1), atol=1e-12)
      and not np.allclose(quat_R(q0) @ quat_R(q1), quat_R(q1) @ quat_R(q0)))
verify("四元数乘法复合：R(q₀q₁)=R(q₀)R(q₁) 且不交换（式 2.42）", ok,
       "复合一致 · 顺序交换结果不同")

# —— 18. 四元数求逆与对径：R(q⁻¹)=R(q)ᵀ、R(q)=R(−q)（式 2.43 + dual covering）———
q_inv = quat_conj(q)
ok = (np.allclose(quat_R(q_inv), Rm.T, atol=1e-12)
      and np.allclose(quat_R(-q), Rm, atol=1e-12))
verify("四元数求逆与对径：R(q⁻¹)=Rᵀ、R(−q)=R(q)（式 2.43）", ok,
       "w 翻号即逆 · q 与 −q 同一旋转")

# —— 19. slerp 中点：轴不变、旋转角减半（Algorithm 2.1，p.50）————————————————
qA = np.r_[np.sin(0.0), 0.0, 0.0, 1.0]            # 恒等旋转
alpha = 0.5
qm = slerp(qA, q, alpha)
n_m, th_m = qm[:3] / max(np.linalg.norm(qm[:3]), 1e-15), 2 * np.arccos(np.clip(qm[3], -1, 1))
ok = (np.allclose(unit(n_m), n, atol=1e-9)
      and abs(th_m - alpha * th) < 1e-9
      and np.allclose(quat_R(qm), R3(n, alpha * th), atol=1e-12))
verify("slerp 中点：轴不变、角=α·θ（Algorithm 2.1）", ok,
       f"α={alpha} · θ_m={np.degrees(th_m):.2f}° = θ/2")

# —— 20. 正交投影：扔 z 保 x,y；齐次版保 w（式 2.46–2.47，p.51）——————————————
p3 = rng.uniform(-3, 3, (3, 3))           # 3 个 3D 点（行向量）
x_ortho = p3[:, :2]                        # 式(2.46)：[I2×2|0]p
P_ortho_h = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
P_h = np.c_[p3, np.ones(3)]
x_h = (P_ortho_h @ P_h.T).T
x_h = x_h[:, :2] / x_h[:, 2:3]
ok = np.allclose(x_ortho, x_h, atol=1e-12)
verify("正交投影：扔 z 保 x,y，齐次版保 w（式 2.46–2.47）", ok,
       "3 组点两式一致")

# —— 21. 缩放正交与 para-perspective 保平行、透视不保（式 2.48–2.50，p.53–54；
#         Figure 2.7 图注 p.52：非透视投影平行线保持平行）————————————————————
d_rail = np.array([1.0, 0.0, 0.25])        # 公共方向（d_z ≠ 0：不平行于像平面）
aA = np.array([0.0, 0.3, 1.2])             # 铁轨 A 起点
aB = np.array([0.3, 0.75, 2.6])            # 铁轨 B 起点（深度不同）
rails = np.vstack([aA, aA + d_rail, aB, aB + d_rail])
s_sc = 0.8
x_scaled = s_sc * rails[:, :2]             # 式(2.48)：x = [sI₂ₓ₂|0]p
A49 = np.array([[1.0, 0.1, 0.06, 0.15],    # 式(2.49)：x̃ = A p̃，第三列非零、底行 [0 0 0 1]
                [0.08, 0.9, -0.05, 0.10],
                [0.0, 0.0, 0.0, 1.0]])
xh_pp = np.c_[rails, np.ones(4)] @ A49.T   # 3 维齐次像（w′=1，纯仿射无除法）
x_pp = xh_pp[:, :2] / xh_pp[:, 2:3]

def _rail_cross(x):
    """两条铁轨像方向的 2D 叉积（0 = 平行）。"""
    d1, d2 = x[1] - x[0], x[3] - x[2]
    return d1[0] * d2[1] - d1[1] * d2[0]

x_per = rails[:, :2] / rails[:, 2:3]       # 同样两轨在透视下（式 2.50，对照）
ok = (abs(_rail_cross(x_scaled)) < 1e-12 and abs(_rail_cross(x_pp)) < 1e-12
      and abs(_rail_cross(x_per)) > 1e-3)
verify("缩放正交与 para-perspective 保平行、透视不保（式 2.48–2.50）", ok,
       f"缩放正交/para 像方向叉积=0 · 透视叉积={_rail_cross(x_per):+.4f}")

# —— 22. 透视保直线：3D 共线点除以各自的 z 后仍共线（式 2.50–2.51，p.54）——————
# 直线参数化 p(λ)=a+λd：x(λ),y(λ),z(λ) 均为 λ 的一次式，像 (x/z, y/z) 落在同一直线上
a22 = rng.uniform(-1.0, 1.0, 3)
d22 = rng.uniform(-1.0, 1.0, 3)
a22[2] = rng.uniform(1.5, 4.0)             # z(λ)=z0+λ·dz，全程 ≥ 0.69，远离 0
d22[2] = rng.uniform(-0.3, 0.3)
lams22 = np.array([0.0, 1.0, 2.7])
p22 = a22[None, :] + lams22[:, None] * d22[None, :]
assert np.all(np.abs(p22[:, 2]) > 0.3)
x_per = p22[:, :2] / p22[:, 2:3]           # 式(2.50)：除以各自的 z
dirs = np.diff(x_per, axis=0)
cr = abs(dirs[0, 0] * dirs[1, 1] - dirs[0, 1] * dirs[1, 0])
rel = cr / (np.linalg.norm(dirs[0]) * np.linalg.norm(dirs[1]))   # 相对共线度（sin 夹角）
ok = rel < 1e-9
verify("透视保直线：共线三点除以各自的 z 后仍共线（式 2.50）", ok,
       f"相对叉积={rel:.2e}")

# —— 23. 透视反例 + 消失点兑现：两条 3D 平行线的像不平行，且像点随 λ→∞ 收敛于
#         同一消失点 (d_x/d_z, d_y/d_z)（式 2.50 推论，讲者演示；
#         Figure 2.7 图注 p.52：透视下平行线不再保持平行）——————————————————
p0a = np.array([0.4, 0.3, 2.0])
p0b = np.array([-0.6, -0.2, 2.0])
d = np.array([0.0, 0.0, 1.0])              # 公共方向（纵深）
lams_seq = np.array([0.0, 2.0, 6.0, 30.0, 300.0, 3000.0])
A_line = p0a[None, :] + lams_seq[:, None] * d[None, :]
B_line = p0b[None, :] + lams_seq[:, None] * d[None, :]
xA = A_line[:, :2] / A_line[:, 2:3]        # 式(2.50)
xB = B_line[:, :2] / B_line[:, 2:3]
vp = d[:2] / d[2]                          # 消失点：λ→∞ 的极限
distsA = np.linalg.norm(xA - vp[None, :], axis=1)
distsB = np.linalg.norm(xB - vp[None, :], axis=1)
dirA, dirB = xA[1] - xA[0], xB[1] - xB[0]  # 两条像线的方向
cross_img = abs(dirA[0] * dirB[1] - dirA[1] * dirB[0])
ok = (np.linalg.norm(xA[-1] - vp) < 1e-3 and np.linalg.norm(xB[-1] - vp) < 1e-3
      and np.all(np.diff(distsA) < -1e-9) and np.all(np.diff(distsB) < -1e-9)
      and cross_img > 1e-6)
verify("消失点兑现：两条平行线的像不平行，且都收敛于 (d_x/d_z, d_y/d_z)（式 2.50 推论）", ok,
       f"vp=({vp[0]:.3f},{vp[1]:.3f}) · 像距 {distsA[0]:.3f}→{distsA[-1]:.2e} · 像方向叉积={cross_img:.3f}")

# —— 24. 逆深度：znear=1、zfar→∞ 且第三行翻号后，归一化第三元 = 1 − 1/z，
#         是 1/z 的仿射函数：d = 1 − 第三元 = 1/z（disparity，Okutomi–Kanade，式 2.52 特例 p.54；
#         原书称「becomes the inverse depth」，讲者推演补齐仿射换算）————————————————
z = rng.uniform(1.5, 6.0, 4)
row3 = np.array([0.0, 0.0, 1.0, -1.0])     # [0,0,−zfar/zrange, znear·zfar/zrange] 的极限并翻号
w_row = np.array([0.0, 0.0, 1.0, 0.0])     # 式(2.52) 第四行 → w′=z
pts = np.c_[rng.uniform(-2, 2, (4, 2)), z, np.ones(4)]
third_norm = (pts @ row3) / (pts @ w_row)  # 归一化屏幕向量的第三元 = 1 − 1/z
disp = 1.0 - third_norm                    # d = 1 − 第三元 → disparity = 1/z
ok = np.allclose(disp, 1.0 / z, atol=1e-12)
verify("逆深度：第三元 = 1−1/z，d = 1−第三元 = 1/z（式 2.52 特例，讲者推演）", ok,
       f"z∈[{z.min():.2f},{z.max():.2f}] · d=1−第三元 与 1/z 逐点一致")


# —— 重绘图 1：Figure 2.5（p.46）轴角旋转几何（讲者重绘，原书图未使用）———————
def draw_fig2_5(outpath):
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    # 取 n̂≈z 轴：v∥=(n̂·v)n̂ 与轴共线是几何事实（书里也画在轴上），
    # 轴画细线、v∥ 画粗箭头叠在上面以示区分
    n = np.array([0.0, 0.0, 1.0])
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    base = np.deg2rad(58)                          # n̂ 与 v 的夹角
    v_par = np.cos(base) * n
    v_perp = np.sin(base) * e1
    v = v_par + v_perp
    v_cross = np.cross(n, v)
    th = np.deg2rad(50)
    u = v_par + np.cos(th) * v_perp + np.sin(th) * v_cross
    O = np.zeros(3)
    # 轴线：细灰线（带箭头）+ n̂ 标注
    ax.quiver(*O, *(n * 1.45), color="#9aa7b5", lw=1.0, arrow_length_ratio=0.06)
    ax.text(0, 0, 1.56, r"$\hat{n}$", fontsize=15, color="#5a6a80")
    # v∥：粗青色箭头沿轴线
    ax.quiver(*O, *v_par, color="#1f7a8c", lw=3.2, arrow_length_ratio=0.10)
    ax.text(*(v_par + np.array([0.02, 0.02, 0.14])), r"$v_\parallel$", fontsize=15, color="#1f7a8c")
    # v：蓝
    ax.quiver(*O, *v, color=BLUE, lw=2.2, arrow_length_ratio=0.09)
    ax.text(*(v * 1.12), r"$v$", fontsize=15, color=BLUE)
    # v⊥：绿，从 v∥ 端点到 v 端点
    ax.quiver(*v_par, *v_perp, color="#4a8c62", lw=1.8, arrow_length_ratio=0.09)
    ax.text(*(v_par + 0.50 * v_perp + np.array([0, 0, -0.15])), r"$v_\perp$", fontsize=14, color="#3a7a50")
    # v×：橙，从 v∥ 端点出发（= n̂ × v）
    ax.quiver(*v_par, *v_cross, color=ORANGE, lw=1.8, arrow_length_ratio=0.09)
    ax.text(*(v_par + 0.68 * v_cross + np.array([0, 0.02, 0.10])), r"$v_\times$", fontsize=14, color=ORANGE)
    # u：红，旋转结果
    ax.quiver(*O, *u, color="#a03050", lw=2.4, arrow_length_ratio=0.08)
    ax.text(*(u * 1.10), r"$u$", fontsize=15, color="#a03050")
    # 旋转轨迹弧线（v → u）+ θ 标注
    t_arc = np.linspace(0, th, 40)
    arc = np.array([v_par + np.cos(tt) * v_perp + np.sin(tt) * v_cross for tt in t_arc])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="#a03050", lw=1.4, ls="--")
    arc_mid = v_par + np.cos(np.deg2rad(22)) * v_perp + np.sin(np.deg2rad(22)) * v_cross
    ax.text(*(arc_mid * 1.0 + np.array([0.03, 0.03, 0.10])), r"$\theta$", fontsize=14, color="#a03050")
    # 其余轴线（浅色）
    for s_ in (-1, 1):
        ax.plot([0, 1.6 * s_], [0, 0], [0, 0], color="#dde5ee", lw=0.8, zorder=0)
        ax.plot([0, 0], [0, 1.6 * s_], [0, 0], color="#dde5ee", lw=0.8, zorder=0)
        ax.plot([0, 0], [0, 0], [0, -1.6], color="#dde5ee", lw=0.8, zorder=0)
    ax.text(1.65, 0, 0, "x", fontsize=10, color="#b8c2cd")
    ax.text(0, 1.65, 0, "y", fontsize=10, color="#b8c2cd")
    ax.set_title("E015 · 据原书 Figure 2.5（第 46 页）重绘：绕轴 $\\hat{n}$ 转 $\\theta$ 的几何分解",
                 fontsize=12.5, color=INK, pad=2)
    ax.text2D(0.985, 0.015, "讲者重绘：向量与角度为示例值 · 原书图未使用",
              transform=ax.transAxes, ha="right", fontsize=8.5, color=BROWN)
    ax.view_init(elev=16, azim=-68)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


# —— 重绘图 2：Figure 2.6（p.48）单位四元数球（讲者重绘，原书图未使用）———————
def draw_fig2_6(outpath):
    fig = plt.figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    # 单位球线框
    phi, theta = np.meshgrid(np.linspace(0, np.pi, 24), np.linspace(0, 2 * np.pi, 48))
    xs = np.sin(phi) * np.cos(theta)
    ys = np.sin(phi) * np.sin(theta)
    zs = np.cos(phi)
    ax.plot_wireframe(xs, ys, zs, color="#d8e2ee", lw=0.5, alpha=0.7)
    # 三点轨迹 q0 → q1 → q2（4 维四元数取 (x,y,w) 三轴显示；
    # 旋转轴取在 x-y 平面内（z=0），保证所画点严格落在单位球面上）
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    axis = unit(np.array([0.6, 0.8, 0.0]))
    ang = lambda a: np.r_[np.sin(a / 2) * axis, np.cos(a / 2)]
    q1, q2 = ang(1.9), ang(3.4)
    tt = np.linspace(0, 3.4, 80)
    traj = np.array([ang(a) for a in tt])
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 3], color=BLUE, lw=2.0)
    for qq, name, lab in ((q0, "q0", r"$q_0$"), (q1, "q1", r"$q_1$"), (q2, "q2", r"$q_2$")):
        ax.scatter(*qq[[0, 1, 3]], color=ORANGE, s=42, depthshade=False)
        ax.text(qq[0] * 1.18, qq[1] * 1.18, qq[3] * 1.1 + 0.05, lab, fontsize=14, color=INK)
    # 对径点 −q2
    ax.scatter(*(-q2)[[0, 1, 3]], color="#a03050", s=42, depthshade=False)
    ax.text((-q2)[0] * 1.2 - 0.28, (-q2)[1] * 1.2 + 0.22, (-q2)[3] * 1.1 + 0.06,
            r"$-q_2$", fontsize=14, color="#a03050")
    ax.text(-0.95, 0.55, 1.18, r"$\|q\|=1$", fontsize=13, color="#5a6a80")
    ax.set_title("E015 · 据原书 Figure 2.6（第 48 页）重绘：单位四元数住在单位球上（q 与 $-q$ 同一旋转）",
                 fontsize=12, color=INK, pad=2)
    ax.text2D(0.985, 0.015, "讲者重绘：旋转轴取在 x-y 平面内，点严格落在球面 · 原书图未使用",
              transform=ax.transAxes, ha="right", fontsize=8.5, color=BROWN)
    ax.view_init(elev=16, azim=-55)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.15, 1.15), ax.set_ylim(-1.15, 1.15), ax.set_zlim(-1.15, 1.15)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


# —— 重绘图 3：Figure 2.7（p.52）六种投影模型俯视图（讲者重绘，原书图未使用）—————
def draw_fig2_7(outpath):
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 8.2))
    titles = ["(a) 3D view", "(b) orthography", "(c) scaled orthography",
              "(d) para-perspective", "(e) perspective", "(f) object-centered"]
    # 俯视示意：地面两条平行「铁轨」+ 一个方盒 + 像平面线
    rail_y = (0.30, 0.70)          # 两轨 y 位置
    x0, x1 = 0.10, 1.30            # 铁轨 x 范围
    img_x = 1.72                   # 像平面 x 位置（各模型不同）
    cam = (0.02, 0.38)             # 相机位置（俯视）：低于方盒中心，使物体中心偏离光轴（para 的视线因此倾斜）

    def rails_parallel_to(ax):
        for ry in rail_y:
            xx = np.linspace(x0, x1, 40)
            yy = np.full_like(xx, ry)
            ax.plot(xx, yy, color=BLUE, lw=1.6)

    def draw_box(ax, converging=False):
        bx = (0.62, 0.92)
        by = (0.38, 0.62)
        ax.add_patch(patches.Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                                       fill=False, edgecolor=INK, lw=1.4))
        ax.plot([bx[0]], [by[1]], marker="s", ms=3, color=INK)

    def draw_img_and_projection(ax, mode):
        """像平面竖线 + 铁轨在像上的投影线。"""
        ax.axvline(img_x, color=BROWN, lw=1.6)
        ax.text(img_x + 0.03, 0.94, "像平面", fontsize=9, color=BROWN, rotation=90, va="top")
        for ry in rail_y:
            if mode == "ortho":
                ax.plot([x1, img_x], [ry, ry], color="#a8c0dc", lw=1.0, ls=":")
                ax.plot([img_x - 0.055, img_x], [ry, ry], color=ORANGE, lw=2.6)
            elif mode == "scaled":
                ax.plot([x1, img_x], [ry, ry], color="#a8c0dc", lw=1.0, ls=":")
                yy = 0.5 + (ry - 0.5) * 0.62
                ax.plot([img_x - 0.055, img_x], [yy, yy], color=ORANGE, lw=2.6)
            elif mode == "para":
                # 投影方向平行于「相机→物体中心」视线
                oc = (0.77, 0.50)
                dirv = np.array(oc) - np.array(cam)
                tt = (img_x - x1) / dirv[0]
                y_at_img = ry + dirv[1] * tt
                ax.plot([x1, img_x], [ry, y_at_img], color="#a8c0dc", lw=1.0, ls=":")
                yy = 0.5 + (y_at_img - 0.5) * 0.62
                ax.plot([img_x - 0.055, img_x], [yy, yy], color=ORANGE, lw=2.6)
            elif mode == "persp":
                # 投影线汇于相机；像上结果只由左上小图表达（真透视无「再缩放」，俯视图不画像段）
                tt = (img_x - cam[0]) / (x1 - cam[0])
                y_at_img = cam[1] + (ry - cam[1]) * tt
                ax.plot([cam[0], img_x], [cam[1], y_at_img], color="#a8c0dc", lw=0.9, ls=":")
                ax.plot([img_x], [y_at_img], marker="|", ms=9, mew=2.6, color=ORANGE)

    for ax, title in zip(axes.flat, titles):
        ax.set_facecolor(PAPER)
        for sp in ax.spines.values():
            sp.set_color("#c8bda6")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(title, fontsize=12, color=INK, pad=6)
        ax.set_xlim(-0.05, 2.05), ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        if title == "(a) 3D view":
            for ry in rail_y:
                ax.plot(np.linspace(x0, x1, 40), np.full(40, ry), color=BLUE, lw=1.6)
            draw_box(ax)
            ax.plot([cam[0]], [cam[1]], marker="^", ms=9, color=ORANGE)
            ax.text(0.07, 0.22, "相机", fontsize=9, color=ORANGE, ha="left")
            ax.text(0.77, 0.24, "方盒 + 地面平行线", fontsize=9, color="#5a6a80", ha="center")
        elif title == "(b) orthography":
            rails_parallel_to(ax)
            draw_box(ax)
            draw_img_and_projection(ax, "ortho")
            ax.text(0.6, 0.06, "投影线垂直于像平面 · 像上平行线仍平行", fontsize=9, color="#5a6a80")
        elif title == "(c) scaled orthography":
            rails_parallel_to(ax)
            draw_box(ax)
            draw_img_and_projection(ax, "scaled")
            ax.text(0.6, 0.06, "正交后乘 s · 像上仍平行", fontsize=9, color="#5a6a80")
        elif title == "(d) para-perspective":
            rails_parallel_to(ax)
            draw_box(ax)
            draw_img_and_projection(ax, "para")
            ax.plot([cam[0]], [cam[1]], marker="^", ms=7, color=ORANGE)
            ax.text(0.6, 0.06, "投影方向平行于视线 · 仍仿射、仍平行", fontsize=9, color="#5a6a80")
        elif title == "(e) perspective":
            rails_parallel_to(ax)
            draw_box(ax)
            draw_img_and_projection(ax, "persp")
            ax.plot([cam[0]], [cam[1]], marker="^", ms=7, color=ORANGE)
            ax.text(0.6, 0.06, "投影线汇于相机 · 像上两轨汇向消失点", fontsize=9, color="#5a6a80")
            # 像平面内小图（正视图）：两轨的像汇向消失点
            ins = ax.inset_axes([0.04, 0.74, 0.27, 0.24])
            ins.set_facecolor("#f7efe0")
            for sp in ins.spines.values():
                sp.set_color("#c8bda6")
            ins.set_xticks([]), ins.set_yticks([])
            ins.plot([0.10, 0.5], [0.02, 0.74], color=BLUE, lw=1.6)
            ins.plot([0.90, 0.5], [0.02, 0.74], color=BLUE, lw=1.6)
            ins.plot([0.5], [0.74], marker="o", ms=4, color=ORANGE)
            ins.text(0.5, 0.84, "消失点", fontsize=7.5, color=ORANGE, ha="center")
            ins.set_xlim(0, 1), ins.set_ylim(0, 1)
            ins.set_title("小图：像上的两轨", fontsize=7.5, color="#5a6a80", pad=2)
        else:  # (f) object-centered —— 只保留原书图注名称，不作机制描述（细节不在本集范围）
            for ry in rail_y:
                ax.plot(np.linspace(x0, x1, 40), np.full(40, ry), color=BLUE, lw=1.6)
            draw_box(ax)
            ax.plot([cam[0]], [cam[1]], marker="^", ms=7, color=ORANGE)
            ax.text(0.07, 0.22, "相机", fontsize=9, color=ORANGE, ha="left")
            oc = (0.77, 0.50)
            ax.plot([oc[0]], [oc[1]], marker="o", ms=5, color="#a03050")
            ax.text(oc[0] + 0.04, oc[1] - 0.06, "物体中心", fontsize=9, color="#a03050")
            ax.text(0.6, 0.06, "原书图注名：object-centered（本集不作展开）", fontsize=9, color="#5a6a80")
    fig.suptitle("E015 · 据原书 Figure 2.7（第 52 页）重绘：常用投影模型俯视图（讲者重绘，原书图未使用）",
                 fontsize=13.5, color=INK, y=0.99)
    fig.text(0.985, 0.006, "几何关系示意：线段位置/缩放为示例值，突出「非透视保平行、透视相交」对比",
             ha="right", fontsize=8.5, color=BROWN)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def sync_and_md5():
    import hashlib
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    figdir = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures"
    for f in sorted(outdir.glob("*.png")):
        print(f"[md5] {f.name}  {hashlib.md5(f.read_bytes()).hexdigest()}")
    synced = 0
    total = 0
    for f in sorted(outdir.glob("*.png")):
        total += 1
        dst = figdir / f.name
        if dst.exists() and hashlib.md5(dst.read_bytes()).hexdigest() != hashlib.md5(f.read_bytes()).hexdigest():
            print(f"[sync-注意] {dst.name} 与已有文件漂移，将覆盖")
        dst.write_bytes(f.read_bytes())
        if hashlib.md5(dst.read_bytes()).hexdigest() == hashlib.md5(f.read_bytes()).hexdigest():
            synced += 1
            print(f"[demo] 已同步 {dst}")
    ok = synced == total and total > 0
    print(f"[verify] 输出图已同步 slides/figures/（逐文件 md5 比对）  ({synced}/{total} 一致)  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def main():
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(exist_ok=True)
    draw_fig2_5(outdir / "e015_fig2_5.png")
    draw_fig2_6(outdir / "e015_fig2_6.png")
    draw_fig2_7(outdir / "e015_fig2_7.png")
    print("[demo] 三张重绘图已存 outputs/")
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
