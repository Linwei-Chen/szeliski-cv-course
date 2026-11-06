#!/usr/bin/env python3
"""e032_geometric_transform.py —— E032 §3.6 几何变换 S2 数值实验（权威存档 stdout）。

口径注记七项以 NOTES 常量为单一来源，脚本启动时逐字打印进 stdout 标头；
权威存档=本 stdout（code/outputs/e032_geometric_transform_run1.txt），
其标头即下述注记的逐字文本。行号锚 Lxx 指留档 evidence/e032_book_pp168_179.txt 逐行号。
"""
import shutil

NOTES = [
    "口径注记（七项，与 refs 逐字同步；本 stdout 标头=权威存档标头）：",
    "· 变换族实现：原书 Table 3.3 只给矩阵形式与 DoF/保性质清单，未给具体矩阵数值——"
    "实验用标准构造（R(θ)、sR、一般仿射 A、射影 H̃），性质断言用几何检验"
    "（保平行=两条平行线映后方向叉积为 0、保直线=线上三点映后共线、保长度=两点距离不变），"
    "不作特定矩阵数值断言。",
    "· forward/inverse 口径：最近邻 forward 的「洞」=输出网格中无任何源像素写入的位置；"
    "splatting=双线性权重摊给四邻+逐像素累积+末尾归一（原书 L139–146 语义）；"
    "洞计数只对放大 1.5× 断言相对关系（2× 整数倍缩放下 splatting 的分数权重恒为 0、"
    "退化为最近邻，不具区分力——口径申报）；模糊对照=splatting 与逆向双线性"
    "在共同有效区内对解析连续图的 RMSE 比较（原书「moderate aliasing and blur」语义）。",
    "· 插值质量口径：bicubic 用 Keys 四点立方（a=−0.5，E029 Table 3.2 同族参数化）；"
    "误差对解析连续图（正弦条纹+圆盘）计算，断言三插值器误差排序相对关系。",
    "· MIP-map 层号口径：(3.77) 断言只对 r=2^k 与 r=2^(k+f) 的层号/层间权重复算；"
    "trilinear 权重=(1−frac, frac)——原书 L263–278 文字语义，未给实现细节。",
    "· Beier–Neely 口径：原书只给距离权重语义（u∈[0,1] 内用垂距 v，否则取到 P/Q 的较短距离，"
    "L425–426）与「线段对应=similarity 变换」（L422–424），未给权重公式常数（a、b）——"
    "实验只做单线段距离场语义断言（对称/延长线/垂距最短），不实现完整 morphing 算法。",
    "· morph 口径：对应特征位置由程序形状解析已知（椭圆中心），中间帧把两图 warp 到"
    "同一插值位置 M_t=(1−t)A+tB 后加权——原书 L440–447+L459–463 文字语义；「重影」量化="
    "中间帧灰度介于 0.3–0.7 的像素数（过渡带宽度）。",
    "· 边界处理：变换后超界像素外圈置零——原书未给边界处理（申报在案）。",
]
print("\n".join(NOTES))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
FIGDIR = Path(__file__).resolve().parent.parent / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PASS = []
FAIL = []


def chk(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def readout(label, detail):
    print(f"  [读数] {label}: {detail}")


# ── 测试图：解析连续图（正弦条纹+圆盘），用于插值质量与变换性质检验 ──
def make_test(n=64):
    y, x = np.mgrid[0:n, 0:n].astype(float)
    f = 0.5 + 0.5 * np.sin(2 * np.pi * x / 16) * np.cos(2 * np.pi * y / 16)
    disk = ((x - n / 2) ** 2 + (y - n / 2) ** 2) < (n / 6) ** 2
    f[disk] = 1.0
    return f


def bilinear(f, X, Y):
    n = f.shape[0]
    x0 = np.floor(X).astype(int); y0 = np.floor(Y).astype(int)
    dx = X - x0; dy = Y - y0
    x0c = np.clip(x0, 0, n - 2); y0c = np.clip(y0, 0, n - 2)
    dx = np.where(x0 == x0c, dx, 1.0); dy = np.where(y0 == y0c, dy, 1.0)
    return (f[y0c, x0c] * (1 - dx) * (1 - dy) + f[y0c, x0c + 1] * dx * (1 - dy)
            + f[y0c + 1, x0c] * (1 - dx) * dy + f[y0c + 1, x0c + 1] * dx * dy)


def keys_cubic(f, X, Y, a=-0.5):
    n = f.shape[0]
    def w(t):
        t = np.abs(t)
        return np.where(t <= 1, (a + 2) * t**3 - (a + 3) * t**2 + 1,
                        np.where(t < 2, a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a, 0.0))
    x1 = np.floor(X).astype(int); y1 = np.floor(Y).astype(int)
    dx = X - x1; dy = Y - y1
    acc = np.zeros_like(X, dtype=float)
    for m in range(-1, 3):
        for l in range(-1, 3):
            xi = np.clip(x1 + m, 0, n - 1); yi = np.clip(y1 + l, 0, n - 1)
            acc += w(dx - m) * w(dy - l) * f[yi, xi]
    return acc


def nn_sample(f, X, Y):
    n = f.shape[0]
    xi = np.clip(np.round(X).astype(int), 0, n - 1)
    yi = np.clip(np.round(Y).astype(int), 0, n - 1)
    return f[yi, xi]


def warp_inverse(f, M, shape=None, sampler=bilinear):
    """M：3×3 齐次矩阵（x' = M x，含齐次坐标）。逆映射采样（Algorithm 3.2 语义）。
    对射影矩阵须做齐次除法（src[2] 归一）；仿射时 src[2]=1 无影响。"""
    sh = shape or f.shape
    n = sh[0]
    Y, X = np.mgrid[0:n, 0:n].astype(float)
    pts = np.stack([X, Y, np.ones_like(X)])  # 3×N 目标坐标
    Minv = np.linalg.inv(M)
    src = Minv @ pts.reshape(3, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Xs = (src[0] / src[2]).reshape(n, n)
        Ys = (src[1] / src[2]).reshape(n, n)
    inside = np.isfinite(Xs) & np.isfinite(Ys) \
        & (Xs >= 0) & (Xs <= f.shape[1] - 1) & (Ys >= 0) & (Ys <= f.shape[0] - 1)
    Xc = np.where(np.isfinite(Xs), Xs, 0.0)
    Yc = np.where(np.isfinite(Ys), Ys, 0.0)
    g = sampler(f, np.clip(Xc, 0, f.shape[1] - 1), np.clip(Yc, 0, f.shape[0] - 1))
    g[~inside] = 0.0
    return g


def M_affine(A, t):
    M = np.eye(3); M[:2, :2] = A; M[:2, 2] = t
    return M


def M_projective(H):
    M = np.eye(3); M[:3, :3] = H
    return M


def R_of(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# ══ #1 参数化变换族（Table 3.3/Figure 3.44 语义）══
print("\n#1 参数化变换族（Table 3.3 语义：DoF+保性质几何检验）")
f64 = make_test(64)
theta = np.deg2rad(30)
I2 = np.eye(2); R30 = R_of(theta); s = 1.5
A_gen = np.array([[1.2, 0.4], [0.1, 0.9]])
H_t = np.array([[1.0, 0.15, 0.0], [0.0, 1.0, 0.0], [0.006, 0.003, 1.0]])  # 射影（项取小使画面保留在画布内）
tvec = np.array([6.0, 4.0])
readout("变换参数", f"theta=30°, s=1.5, A=[[1.2,0.4],[0.1,0.9]], H 第三行=[0.006,0.003,1], t={tvec.tolist()}")

# 性质 1：刚体保长度（旋转 30° 后两点距离不变）
p1, p2 = np.array([5.0, 5.0]), np.array([20.0, 9.0])
d0 = np.linalg.norm(p2 - p1)
d_rigid = np.linalg.norm(R30 @ p2 + tvec - (R30 @ p1 + tvec))
readout("rigid 保长度", f"原始 {d0:.6f} vs 旋转后 {d_rigid:.6f}")
chk("1-1 rigid(Euclidean) 保两点距离（3 DoF 语义）", abs(d_rigid - d0) < 1e-12)

# 性质 2：similarity 保角度（两向量夹角不变）
v1, v2 = p2 - p1, np.array([3.0, 10.0])
ang0 = np.degrees(np.arccos(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2)))
w1, w2 = s * R30 @ v1, s * R30 @ v2
ang1 = np.degrees(np.arccos(w1 @ w2 / np.linalg.norm(w1) / np.linalg.norm(w2)))
readout("similarity 保角度", f"原始 {ang0:.6f}° vs 相似变换后 {ang1:.6f}°")
chk("1-2 similarity 保两向量夹角（4 DoF 语义）", abs(ang1 - ang0) < 1e-9)

# 性质 3：affine 保平行（两条平行线映后方向叉积仍为 0，且不退化成一条线）
def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]
par1 = np.array([[0.0, 10.0], [30.0, 10.0]])  # y=10 水平线
par2 = np.array([[0.0, 20.0], [30.0, 20.0]])  # y=20 水平线（与 par1 平行）
cross_before = cross2(par1[1] - par1[0], par2[1] - par2[0])
a_par1 = (A_gen @ par1.T).T + tvec
a_par2 = (A_gen @ par2.T).T + tvec
d1, d2 = a_par1[1] - a_par1[0], a_par2[1] - a_par2[0]
cross_after = cross2(d1, d2)
readout("affine 保平行", f"映前方向叉积 {cross_before:.1f} → 映后方向叉积 {cross_after:.2e}"
        f"（≈0 ⇒ 仍平行）；映后两线错开 {np.linalg.norm(a_par2[0] - a_par1[0]):.4f}（未重合）")
chk("1-3 affine 保平行（6 DoF 语义：映前平行两线映后方向叉积=0 且两线不重合）",
    abs(cross_before) < 1e-9 and abs(cross_after) < 1e-9
    and np.linalg.norm(a_par2[0] - a_par1[0]) > 1e-6,
    f"叉积 {cross_before:.1f}→{cross_after:.2e}")
q_cross1 = abs(cross2(A_gen @ np.array([30.0, 0.0]), A_gen @ np.array([0.0, 10.0])))
chk("1-3b affine 映后两边线性无关（平行四边形未退化成一条线）", q_cross1 > 1e-9, f"叉积 {q_cross1:.4f}")

# 性质 4：projective 保直线但不保平行——平行线映后斜率不等（将相交）
l1 = np.array([[0.0, 10.0], [30.0, 10.0]])   # y=10 水平线
l2 = np.array([[0.0, 20.0], [30.0, 20.0]])   # y=20 水平线
def hmap(pts, H=H_t):
    h = H @ np.vstack([pts.T, np.ones(pts.shape[0])])
    return (h[:2] / h[2]).T
e1, e2 = hmap(l1), hmap(l2)
sl1 = (e1[1, 1] - e1[0, 1]) / (e1[1, 0] - e1[0, 0] + 1e-12)
sl2 = (e2[1, 1] - e2[0, 1]) / (e2[1, 0] - e2[0, 0] + 1e-12)
readout("projective 平行线斜率", f"l1 斜率 {sl1:.4f} vs l2 斜率 {sl2:.4f}（不等⇒将相交）")
# 保直线：l1 上三点映后共线（叉积≈0）
tri = np.array([[0.0, 10.0], [15.0, 10.0], [30.0, 10.0]])
he = hmap(tri)
col_cross = cross2(he[1] - he[0], he[2] - he[0])
readout("projective 保直线", f"l1 上三点映后共线叉积 {col_cross:.3e}（≈0 ⇒ 三点仍共线，仍是直线）")
chk("1-4 projective 保直线（线上三点映后共线，叉积≈0）", abs(col_cross) < 1e-6, f"叉积 {col_cross:.2e}")
chk("1-4b projective 不保平行（两水平线映后斜率不再相等，将相交）",
    np.isfinite(sl1) and np.isfinite(sl2) and abs(sl1 - sl2) > 1e-3,
    f"{sl1:.4f} vs {sl2:.4f}")

# 性质 5：DoF 清单（Table 3.3 语义重绘数据）
dof = {"translation": 2, "rigid": 3, "similarity": 4, "affine": 6, "projective": 8}
readout("Table 3.3 DoF 层级", dof)
chk("1-5 Table 3.3 层级 2/3/4/6/8 DoF（语义重绘依据，原书表未使用）",
    [dof[k] for k in ["translation", "rigid", "similarity", "affine", "projective"]] == [2, 3, 4, 6, 8])

# ══ #2 forward vs inverse warping（Figure 3.45/3.46 语义）══
print("\n#2 forward vs inverse warping（放大 1.5×，无旋转）")
f32 = make_test(32)
K = 1.5
NI = 32
# 目标 47×47：最大格点 46/1.5=30.67 ≤ 31，全部映回源界内（放大语义下 inverse 零洞的干净前提）
NO = 47
M2 = M_affine(K * np.eye(2), np.array([0.0, 0.0]))
out48 = np.zeros((NO, NO))
written = np.zeros((NO, NO), dtype=bool)  # 洞判据=「未被写入」mask（内容 0 值≠洞）
Ys_, Xs_ = np.mgrid[0:NI, 0:NI].astype(float)
Xd = np.round(K * Xs_).astype(int).ravel(); Yd = np.round(K * Ys_).astype(int).ravel()
np.add.at(out48, (Yd, Xd), f32.ravel())
written[Yd, Xd] = True
hole_nn = int((~written).sum())
readout("forward 最近邻洞数", f"{hole_nn} / {NO*NO}（{hole_nn/(NO*NO)*100:.1f}%）")
chk("2-1 forward 最近邻放大产生洞（Figure 3.45 语义：洞>0）", hole_nn > 0, f"{hole_nn}")

# splatting：双线性权重摊给四邻+归一
acc = np.zeros((NO, NO)); wacc = np.zeros((NO, NO))
Xf = (K * Xs_).ravel(); Yf = (K * Ys_).ravel()
for ox in (0, 1):
    for oy in (0, 1):
        xi = np.floor(Xf).astype(int) + ox; yi = np.floor(Yf).astype(int) + oy
        wx = np.where(ox == 0, 1 - (Xf - np.floor(Xf)), Xf - np.floor(Xf))
        wy = np.where(oy == 0, 1 - (Yf - np.floor(Yf)), Yf - np.floor(Yf))
        inside2 = (xi >= 0) & (xi < NO) & (yi >= 0) & (yi < NO)
        np.add.at(acc, (yi[inside2], xi[inside2]), f32.ravel()[inside2] * wx[inside2] * wy[inside2])
        np.add.at(wacc, (yi[inside2], xi[inside2]), wx[inside2] * wy[inside2])
g_splat = np.where(wacc > 1e-12, acc / np.maximum(wacc, 1e-12), 0.0)
hole_splat = int((wacc <= 1e-12).sum())
readout("forward splatting 洞数", f"{hole_splat} / {NO*NO}（权重归一后仍无值的位置）")
chk("2-2 splatting 洞显著少于最近邻（填洞效果；原书「moderate aliasing and blur」语义）",
    hole_splat < hole_nn, f"{hole_splat} < {hole_nn}")

# inverse：双线性逆映射采样；洞判据=逆映射出界（ĥ 无定义），与内容 0 值区分
g_inv = warp_inverse(f32, M2, shape=(NO, NO), sampler=bilinear)
yyi, xxi = np.mgrid[0:NO, 0:NO].astype(float)
inside_inv = (xxi / K >= 0) & (xxi / K <= NI - 1) & (yyi / K >= 0) & (yyi / K <= NI - 1)
hole_inv = int((~inside_inv).sum())
readout("inverse 洞数", f"{hole_inv} / {NO*NO}")
chk("2-3 inverse warping 无洞（Figure 3.46 语义：放大时 ĥ 处处有定义）", hole_inv == 0, f"{hole_inv}")

# inverse 与解析图一致性：连续测试图解析式在 x/1.5 处采样
# （只比正弦条纹连续区：圆盘为二值跳变，插值在间断处必有过渡误差，须排除膨胀带）
yy, xx = np.mgrid[0:NO, 0:NO].astype(float)
srcX, srcY = xx / K, yy / K
analytic_sin = 0.5 + 0.5 * np.sin(2 * np.pi * srcX / 16) * np.cos(2 * np.pi * srcY / 16)
diskm = ((srcX - NI / 2) ** 2 + (srcY - NI / 2) ** 2) < (NI / 6) ** 2
inner = np.zeros_like(diskm, dtype=bool)
inner[2:-2, 2:-2] = True
diskd = diskm.copy()
for _ in range(2):
    diskd = diskd | np.roll(diskd, 1, 0) | np.roll(diskd, -1, 0) | np.roll(diskd, 1, 1) | np.roll(diskd, -1, 1)
region = inner & ~diskd
err_inv = float(np.max(np.abs(g_inv - analytic_sin)[region]))
readout("inverse 双线性 vs 解析式", f"max err（连续条纹内区，排除圆盘膨胀带 {int((inner & ~region).sum())} px）={err_inv:.6f}")
chk("2-4 inverse 双线性与解析连续图一致（连续区，避开边缘 clip 与圆盘间断）", err_inv < 0.05, f"{err_inv:.4f}")

# splatting 模糊语义：共同有效区（splatting 非洞 & inverse 界内 & 连续条纹内区）对解析图 RMSE
valid = (wacc > 1e-12) & inside_inv & region
err_splat = float(np.sqrt(np.mean((g_splat - analytic_sin)[valid] ** 2)))
err_inv_v = float(np.sqrt(np.mean((g_inv - analytic_sin)[valid] ** 2)))
readout("splatting vs 逆向双线性（共同有效区 RMSE，模糊语义）",
        f"splatting {err_splat:.4f} vs 逆向双线性 {err_inv_v:.4f}")
chk("2-5 splatting 比逆向双线性更糊（共同有效区对解析图 RMSE 更大；原书「moderate aliasing and blur」）",
    err_splat > err_inv_v > 0, f"{err_splat:.4f} > {err_inv_v:.4f}")

# ══ #3 旋转 45° 三种插值质量（Figure 3.48/E029 插值核回归）══
print("\n#3 旋转 45°：最近邻/双线性/双三次插值质量")
f96 = make_test(96)
c = 48.0
Rm = R_of(np.deg2rad(45))
tt = np.array([c - (Rm[0, 0] * c + Rm[0, 1] * c), c - (Rm[1, 0] * c + Rm[1, 1] * c)])
M45 = M_affine(Rm, tt)
g_nn = warp_inverse(f96, M45, sampler=nn_sample)
g_bl = warp_inverse(f96, M45, sampler=bilinear)
g_bc = warp_inverse(f96, M45, sampler=keys_cubic)
yy, xx = np.mgrid[0:96, 0:96].astype(float)
Minv = np.linalg.inv(M45)
src = Minv @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
sx, sy = src[0], src[1]
inside = (sx >= 0) & (sx <= 95) & (sy >= 0) & (sy <= 95)
truth = 0.5 + 0.5 * np.sin(2 * np.pi * sx / 16) * np.cos(2 * np.pi * sy / 16)
diskm = ((sx - 48) ** 2 + (sy - 48) ** 2) < 16 ** 2
truth[diskm] = 1.0
truth = truth.reshape(96, 96); inside = inside.reshape(96, 96)
e_nn = float(np.sqrt(np.mean((g_nn - truth)[inside] ** 2)))
e_bl = float(np.sqrt(np.mean((g_bl - truth)[inside] ** 2)))
e_bc = float(np.sqrt(np.mean((g_bc - truth)[inside] ** 2)))
readout("旋转 45° RMSE", f"nearest {e_nn:.4f} / bilinear {e_bl:.4f} / bicubic {e_bc:.4f}")
chk("3-1 插值质量排序：bicubic < bilinear < nearest RMSE（E029 频域结论的空间域复现）",
    e_bc < e_bl < e_nn, f"{e_bc:.4f} < {e_bl:.4f} < {e_nn:.4f}")

# ══ #4 MIP-map 层号 (3.77) 与 trilinear 权重 ══
print("\n#4 MIP-map 层号与 trilinear 权重（(3.77) 复算）")
for r, want_l in [(1.0, 0.0), (2.0, 1.0), (4.0, 2.0), (8.0, 3.0)]:
    l = float(np.log2(r))
    chk(f"4-{int(r)} log2({r:g})={want_l:g}", abs(l - want_l) < 1e-12, f"l={l}")
r = 6.0
l6 = float(np.log2(r))
base, frac = int(np.floor(l6)), l6 - int(np.floor(l6))
readout(f"log2({r:g})", f"l={l6:.6f} → 第 {base} 层权重 {1-frac:.4f} + 第 {base+1} 层权重 {frac:.4f}（trilinear）")
chk("4-5 r=6 → l≈2.585：trilinear 在第 2/3 层间插值（(3.77)+线性混合语义）",
    abs(l6 - 2.5849625) < 1e-6 and base == 2 and 0 < frac < 1, f"l={l6:.4f}")

# ══ #5 Beier–Neely 线段距离场（Figure 3.50a 语义）══
print("\n#5 Beier–Neely 线段距离场（单线段，Figure 3.50a 语义）")
P = np.array([20.0, 48.0]); Q = np.array([76.0, 48.0])  # 线段 PQ（96×96 画布）
readout("线段参数", f"P={P.tolist()}, Q={Q.tolist()}（96×96 画布，水平线段 y=48）")
yy, xx = np.mgrid[0:96, 0:96].astype(float)
d = Q - P; L2 = d @ d
u = (((np.stack([xx.ravel(), yy.ravel()]) - P[:, None]) * d[:, None]).sum(0) / L2).reshape(96, 96)
proj = P[:, None] + np.clip(u, 0, 1).ravel() * d[:, None]  # u∈[0,1] 投影，否则夹到端点
dist = np.linalg.norm(np.stack([xx.ravel(), yy.ravel()]) - proj, axis=0).reshape(96, 96)
readout("线段上方/下方对称性", f"dist(48,24)={dist[24,48]:.4f} vs dist(48,72)={dist[72,48]:.4f}")
chk("5-1 线段距离场沿线上下对称（到线段最小距离语义）",
    abs(dist[24, 48] - dist[72, 48]) < 1e-12)
chk("5-2 延长线上的点取到端点距离而非垂距（u<0 用 P 距离语义：dist(10,48)=10 而非 0）",
    abs(dist[48, 10] - 10.0) < 1e-9, f"dist={dist[48, 10]:.4f}")
dP = np.linalg.norm(np.stack([xx.ravel(), yy.ravel()]) - P[:, None], axis=0).reshape(96, 96)
segline = dist[24, 30:61]  # y=24 行、x=30..60（u∈[0.18,0.71] 全在段内）
chk("5-3 u∈[0,1] 段内距离=垂距、与 x 无关（恒等于 |y−48|=24；若误用到 P 距离会得 36.9@x=48）",
    np.allclose(segline, 24.0, atol=1e-9), f"x=30..60 全段 {segline[0]:.4f}…{segline[-1]:.4f}，对照到 P 距离 {dP[24,48]:.4f}")

# ══ #6 morph：warp+blend vs 纯 dissolve（Figure 3.51 语义）══
print("\n#6 morph vs cross-dissolve（Figure 3.51 语义）")
def shape_a(n=96):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (((xx - 32) / 20) ** 2 + ((yy - 48) / 12) ** 2) < 1.0  # 左侧横椭圆，中心 x=32
def shape_b(n=96):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return (((xx - 64) / 12) ** 2 + ((yy - 48) / 20) ** 2) < 1.0  # 右侧竖椭圆，中心 x=64
Fa = shape_a().astype(float); Fb = shape_b().astype(float)
# 中间帧 t=0.5：两图 warp 到同一插值位置 M_t=(1−t)·A+t·B（A 中心 32→48，B 中心 64→48，完全重合）
t = 0.5
readout("morph 参数", f"t={t}，A 特征平移 +32·t、B 特征平移 −32·t（半程对齐到 x=48）")
Ma = M_affine(np.eye(2), np.array([32 * t, 0.0]))   # A 特征 32 → 48
Mb = M_affine(np.eye(2), np.array([-32 * t, 0.0]))  # B 特征 64 → 48
Wa = warp_inverse(Fa, Ma, sampler=bilinear)
Wb = warp_inverse(Fb, Mb, sampler=bilinear)
morph_mid = (1 - t) * Wa + t * Wb
dissolve_mid = (1 - t) * Fa + t * Fb
# 重影量化：中间帧中灰度介于 0.3–0.7 的像素数（过渡带宽度）
mid_diss = int(((dissolve_mid > 0.3) & (dissolve_mid < 0.7)).sum())
mid_morph = int(((morph_mid > 0.3) & (morph_mid < 0.7)).sum())
readout("t=0.5 中间灰度像素数（重影带宽度）", f"dissolve {mid_diss} vs morph {mid_morph}")
chk("6-1 morph 中间帧重影带显著窄于纯 dissolve（Figure 3.51 顶/底行语义）",
    mid_morph < mid_diss, f"{mid_morph} < {mid_diss}")
inter = float((Wa * Wb).sum())
Fa_sum = float(Fa.sum())
readout("t=0.5 两 warp 形状重叠", f"overlap sum={inter:.0f} vs A 面积 {Fa_sum:.0f}")
chk("6-2 warp 半程后两形状中心重合、重叠超过 A 面积一半（特征对齐→重影消除机制）",
    inter > 0.5 * Fa_sum, f"{inter:.0f} > {0.5*Fa_sum:.0f}")

# ══ 区分力注入（拦截自检）══
print("\n区分力注入")
bad_rigid = np.linalg.norm((2 * R30) @ p2 + tvec - ((2 * R30) @ p1 + tvec))
chk("D1 注入：把 similarity 当 rigid 检验（保长度）应 FAIL（s=2 时距离×2）",
    abs(bad_rigid - d0) > 0.1, f"err={bad_rigid - d0:.4f}")
bad_morph = int(((dissolve_mid > 0.3) & (dissolve_mid < 0.7)).sum())
chk("D2 注入：把 dissolve 当 morph 检验（重影带应收窄）应 FAIL（dissolve 带不窄于 morph 带）",
    bad_morph >= mid_morph, f"{bad_morph} ≥ {mid_morph}")
D3_nn = warp_inverse(f96, M45, sampler=nn_sample)
e_D3 = float(np.sqrt(np.mean((D3_nn - truth)[inside] ** 2)))
chk("D3 注入：把 nearest 当最优插值应 FAIL（RMSE 比 bicubic 高）",
    e_D3 > e_bc, f"{e_D3:.4f} > {e_bc:.4f}")

# ══ 图件 ══
print("\n图件生成")
# 图 A：变换族对照（Table 3.3/Figure 3.44 语义）
fig, axs = plt.subplots(1, 5, figsize=(16, 4.2))
cases = [("translation 2DoF", M_affine(I2, tvec)), ("Euclidean 3DoF", M_affine(R30, tvec)),
         ("similarity 4DoF", M_affine(s * R30, tvec)), ("affine 6DoF", M_affine(A_gen, tvec)),
         ("projective 8DoF", M_projective(H_t))]
for ax, (name, M) in zip(axs, cases):
    g = warp_inverse(f64, M)
    ax.imshow(g, cmap="gray", vmin=0, vmax=1)
    ax.set_title(name, fontsize=18)
    ax.axis("off")
fig.suptitle("参数化变换族：translation/Euclidean/similarity/affine/projective（对应原书表 3.3，程序生成，原书表未使用）", fontsize=18)
fig.tight_layout()
fig.savefig(FIGDIR / "e032_transforms.png", dpi=110); plt.close(fig)
print("  e032_transforms.png")

# 图 B：forward 洞 vs inverse（Figure 3.45/3.46 语义）
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
axs[0].imshow(out48, cmap="gray", vmin=0, vmax=1); axs[0].set_title(f"forward 最近邻：洞 {hole_nn} 个", fontsize=14)
axs[1].imshow(g_splat, cmap="gray", vmin=0, vmax=1); axs[1].set_title(f"forward splatting：洞 {hole_splat} 个", fontsize=14)
axs[2].imshow(g_inv, cmap="gray", vmin=0, vmax=1); axs[2].set_title(f"inverse 双线性：洞 {hole_inv} 个", fontsize=14)
for ax in axs: ax.axis("off")
fig.suptitle("放大 1.5×：forward warping 的洞与模糊 vs inverse warping 无洞（对应原书图 3.45/3.46，程序生成，原书图未使用）", fontsize=15)
fig.tight_layout()
fig.savefig(FIGDIR / "e032_warping.png", dpi=110); plt.close(fig)
print("  e032_warping.png")

# 图 C：旋转 45° 三插值（Figure 3.48 语义）
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (g, nm, e) in zip(axs, [(g_nn, "最近邻", e_nn), (g_bl, "双线性", e_bl), (g_bc, "双三次", e_bc)]):
    ax.imshow(g, cmap="gray", vmin=0, vmax=1); ax.set_title(f"旋转 45° {nm}：RMSE {e:.4f}", fontsize=14)
    ax.axis("off")
fig.suptitle("旋转 45° 插值质量：RMSE 排序 bicubic < bilinear < nearest（对照前几集的插值核结论，程序生成，原书图未使用）", fontsize=15)
fig.tight_layout()
fig.savefig(FIGDIR / "e032_resampling.png", dpi=110); plt.close(fig)
print("  e032_resampling.png")

# 图 D：morph vs dissolve（Figure 3.51 语义）
fig, axs = plt.subplots(1, 3, figsize=(13, 4.4))
axs[0].imshow(Fa, cmap="gray", vmin=0, vmax=1); axs[0].set_title("源 A（t=0）", fontsize=14)
axs[1].imshow(dissolve_mid, cmap="gray", vmin=0, vmax=1); axs[1].set_title(f"纯 dissolve t=0.5：重影带 {mid_diss}", fontsize=14)
axs[2].imshow(morph_mid, cmap="gray", vmin=0, vmax=1); axs[2].set_title(f"warp+dissolve t=0.5：重影带 {mid_morph}", fontsize=14)
for ax in axs: ax.axis("off")
fig.suptitle("morph：先 warp 对齐再 blend 重影窄于直接 cross-dissolve（对应原书图 3.51，程序生成，原书图未使用）", fontsize=15)
fig.tight_layout()
fig.savefig(FIGDIR / "e032_morph.png", dpi=110); plt.close(fig)
print("  e032_morph.png")

for nm in ["e032_transforms.png", "e032_warping.png", "e032_resampling.png", "e032_morph.png"]:
    shutil.copy(FIGDIR / nm, OUT / nm)
print("  4 张图已同步 code/outputs/")

print(f"\n汇总：{len(PASS)} 条 PASS / {len(FAIL)} 条 FAIL")
if FAIL:
    print("FAIL:", FAIL)
    raise SystemExit(1)
print("DONE")
