#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E014 · 二维变换的族谱 —— 把原书 §2.1.1（印刷 pp.40–43）的变换族谱全部数值验证一遍。

依据（印刷页码）：
    式(2.14)(2.15) 平移（p.40）；式(2.16)(2.17) 刚体/欧氏（p.40）；
    式(2.18) 相似（p.41）；式(2.19) 仿射（p.41）；式(2.20)(2.21) 射影/单应（p.41）；
    Table 2.1 层级表（p.42，DoF 2/3/4/6/8）；式(2.22) co-vector l̃′=H̃⁻ᵀl̃（p.42）；
    stretch/squash、planar surface flow、bilinear interpolant（p.43）。

断言（18 项数据 + 1 项同步）：
    平移结构、平移保定向（朝向）与度量、R 正交、刚体保长度、相似矩阵形式、
    相似保角度、相似缩放长度、仿射保平行、仿射保直线、仿射反例、射影尺度等价、
    射影保直线、射影反例（平行边叉积）、群闭合、逆在同族、stretch/squash 非 sR 形式、
    co-vector 恒等、Table 2.1 DoF 表、同步检查。

输出：outputs/e014_fig2_4.png（据原书 Figure 2.4 重绘，五联方格图）+ 断言汇总。
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib import font_manager

# 中文字体（同 E013：macOS 优先 Hiragino Sans GB；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026042)

PAPER = "#faf6ec"
INK = "#1d3557"
RED = "#c1121f"
BROWN = "#8a6d4b"


def unit(v):
    return v / np.linalg.norm(v)


RESULTS = []


def verify(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[verify] {name}  {detail}  [{'PASS' if ok else 'FAIL'}]")


# —— 变换构造（与原书式 2.14–2.21 一一对应）——————————————————————
def T(t):
    """式(2.15)：平移齐次矩阵。"""
    M = np.eye(3)
    M[:2, 2] = t
    return M


def Rmat(theta):
    """式(2.17)：R=[[cosθ,−sinθ],[sinθ,cosθ]]，正交、|R|=1。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def S_sim(s, theta, t):
    """式(2.18)：相似变换齐次矩阵 [sR t; 0 1]。"""
    M = np.eye(3)
    M[:2, :2] = s * Rmat(theta)
    M[:2, 2] = t
    return M


def A_aff(A2, t):
    """式(2.19)：仿射齐次矩阵 [A t; 0 1]，A 任意 2×2。"""
    M = np.eye(3)
    M[:2, :2] = A2
    M[:2, 2] = t
    return M


def H_proj(H33):
    """式(2.20)：射影齐次矩阵，任意 3×3（非奇异）。"""
    return H33


def apply_h(M, xs):
    """齐次作用 + 归一化（式 2.21）：xs (N,2) → (N,2)。"""
    xs_h = np.c_[xs, np.ones(len(xs))]
    out = xs_h @ M.T
    return out[:, :2] / out[:, 2:3]


def angle_between(u, v):
    c = np.dot(unit(u), unit(v))
    return np.arccos(np.clip(c, -1.0, 1.0))


# —— 1. 平移结构：式(2.15) 作用于增广向量 = x + t（式 2.14）———————————
x = rng.uniform(-3, 3, 2)
t = rng.uniform(-3, 3, 2)
ok = np.allclose(apply_h(T(t), x[None, :])[0], x + t)
verify("平移结构：[I t;0 1] 作用于 x̄ 等于 x+t（式 2.14⇄2.15）", ok,
       f"t=({t[0]:.2f},{t[1]:.2f})")

# —— 2. 平移保定向（朝向）与度量：三条边方向与三边长均不变（Table 2.1 p.42：平移保 orientation，且保其下各行全部性质）——
def signed_area(pts):
    xs_, ys_ = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(xs_ * np.roll(ys_, -1) - np.roll(xs_, -1) * ys_))

tri = rng.uniform(-3, 3, (3, 2))
tri_t = apply_h(T(rng.uniform(-2, 2, 2)), tri)
d0 = np.linalg.norm(np.diff(np.vstack([tri, tri[:1]]), axis=0), axis=1)
d1 = np.linalg.norm(np.diff(np.vstack([tri_t, tri_t[:1]]), axis=0), axis=1)
E0 = np.diff(np.vstack([tri, tri[:1]]), axis=0)
E1 = np.diff(np.vstack([tri_t, tri_t[:1]]), axis=0)
U0 = E0 / np.linalg.norm(E0, axis=1, keepdims=True)
U1 = E1 / np.linalg.norm(E1, axis=1, keepdims=True)
cos_t = float(np.min(np.sum(U0 * U1, axis=1)))          # 平移后逐边方向 cos 的最小值
ok = np.allclose(d0, d1, atol=1e-9) and np.allclose(U0, U1, atol=1e-9)
# 对照组：旋转后带符号面积不变、但边方向改变——说明「面积/手性」≠「朝向」（orientation）
th_c = rng.uniform(0.5, 2.0)
tri_r = apply_h(A_aff(Rmat(th_c), np.zeros(2)), tri)
Er = np.diff(np.vstack([tri_r, tri_r[:1]]), axis=0)
Ur = Er / np.linalg.norm(Er, axis=1, keepdims=True)
cos_r = float(np.max(np.sum(U0 * Ur, axis=1)))          # 旋转后逐边方向 cos 的最大值（仍应 <1）
contrast = np.isclose(signed_area(tri), signed_area(tri_r), atol=1e-9) and cos_r < 1 - 1e-3
ok = ok and contrast
verify("平移保定向（朝向）：三边方向与三边长均不变；对照：旋转面积同、方向变（Table 2.1 p.42）", ok,
       f"平移后逐边方向 cos 最小={cos_t:.6f} · 对照旋转逐边 cos 最大={cos_r:.3f} ≠ 1（面积不变）")

# —— 3. R 正交：RRᵀ=I，|R|=+1（式 2.17 原文断言）———————————————————————
ths = rng.uniform(-np.pi, np.pi, 5)
ok = all(np.allclose(Rmat(th) @ Rmat(th).T, np.eye(2), atol=1e-12)
         and abs(np.linalg.det(Rmat(th)) - 1) < 1e-12 for th in ths)
verify("R 正交：RRᵀ=I 且 |R|=+1（式 2.17）", ok,
       f"θ ∈ [{ths.min():.2f}, {ths.max():.2f}] rad · 5 组")

# —— 4. 刚体保长度：‖R(p−q)‖=‖p−q‖（p.40 Euclidean distances are preserved）—————
p, q = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
th = rng.uniform(-np.pi, np.pi)
Rp = Rmat(th) @ p
ok = abs(np.linalg.norm(Rp - Rmat(th) @ q) - np.linalg.norm(p - q)) < 1e-9
verify("刚体保长度：两点距离在 R 作用下不变（式 2.16）", ok,
       f"‖p−q‖={np.linalg.norm(p - q):.4f}")

# —— 5. 相似矩阵形式：sR=[[a,−b],[b,a]]，a=s·cosθ、b=s·sinθ，a²+b²=s²（式 2.18）——
s = rng.uniform(0.5, 2.5)
th = rng.uniform(-np.pi, np.pi)
a, b = s * np.cos(th), s * np.sin(th)
ok = (np.allclose(np.array([[a, -b], [b, a]]), s * Rmat(th), atol=1e-12)
      and abs(a * a + b * b - s * s) < 1e-9)
verify("相似矩阵形式：sR=[[a,−b],[b,a]]，a²+b²=s²（式 2.18）", ok,
       f"s={s:.3f} · a={a:.3f} · b={b:.3f}")

# —— 6. 相似保角度：随机两向量夹角不变（p.41 preserves angles between lines）——————
u, v = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
M = S_sim(s, th, rng.uniform(-2, 2, 2))
ang0 = angle_between(u, v)
ang1 = angle_between(M[:2, :2] @ u, M[:2, :2] @ v)
ok = abs(ang0 - ang1) < 1e-9
verify("相似保角度：夹角在 sR 作用下不变（Table 2.1 相似行）", ok,
       f"角={np.degrees(ang0):.2f}°")

# —— 7. 相似缩放长度：‖sR·v‖=s·‖v‖（s 是任意 scale factor）———————————————————
vv = rng.uniform(-3, 3, 2)
ok = abs(np.linalg.norm(M[:2, :2] @ vv) - s * np.linalg.norm(vv)) < 1e-9
verify("相似缩放长度：‖sR·v‖=s‖v‖（式 2.18 的 s）", ok,
       f"s‖v‖={s * np.linalg.norm(vv):.4f}")

# —— 8. 仿射保平行：平行四边形对边方向向量变换后仍平行（p.41 parallel lines remain）——
A2 = rng.uniform(-2, 2, (2, 2))
while abs(np.linalg.det(A2)) < 0.3:
    A2 = rng.uniform(-2, 2, (2, 2))
edge = rng.uniform(-2, 2, 2)
lam = rng.uniform(0.5, 2.0)
par = lam * edge                      # 与 edge 平行的另一边
e1, e2 = A2 @ edge, A2 @ par
ok = abs(e1[0] * e2[1] - e1[1] * e2[0]) < 1e-9   # 叉积 = 平行
verify("仿射保平行：平行边经任意 A 后仍平行（式 2.19）", ok,
       f"λ={lam:.2f} · 叉积=0")

# —— 9. 仿射保直线：共线三点经任意 A 后仍共线（Table 2.1 p.42：仿射行保平行，且保其下各行性质）——
a_pts = rng.uniform(-2, 2, (2, 2))
lam_c = rng.uniform(0.3, 0.7)
a_third = (1 - lam_c) * a_pts[0] + lam_c * a_pts[1]
a_three = np.vstack([a_pts, a_third])
a_out = apply_h(A_aff(A2, rng.uniform(-2, 2, 2)), a_three)
a_cr = np.cross(np.r_[a_out[1] - a_out[0], 0], np.r_[a_out[2] - a_out[0], 0])[2]
ok = abs(a_cr) < 1e-9
verify("仿射保直线：共线三点变换后仍共线（Table 2.1 仿射行）", ok,
       f"叉积={a_cr:.2e}")

# —— 10. 仿射反例：diag(2,1) 保平行但不保角度与长度（受限于 6 DoF 的能力边界）——————
D = np.diag([2.0, 1.0])
ang_before = angle_between(np.array([1.0, 0.0]), np.array([1.0, 1.0]))
ang_after = angle_between(D @ np.array([1.0, 0.0]), D @ np.array([1.0, 1.0]))
len_before = np.linalg.norm(np.array([1.0, 1.0]))
len_after = np.linalg.norm(D @ np.array([1.0, 1.0]))
ok = abs(ang_after - ang_before) > 1e-3 and abs(len_after - len_before) > 1e-3
verify("仿射反例：diag(2,1) 保平行但角度/长度改变（仿射不保角度）", ok,
       f"角 {np.degrees(ang_before):.1f}°→{np.degrees(ang_after):.1f}° · 长 {len_before:.3f}→{len_after:.3f}")

# —— 11. 射影尺度等价：H 与 2H 归一化后同一结果（p.41 defined up to a scale）———————
H33 = rng.uniform(-2, 2, (3, 3))
H33[2, :] = [0.3, -0.2, 1.0]          # 保证底行非退化
while abs(np.linalg.det(H33)) < 0.5:
    H33 = rng.uniform(-2, 2, (3, 3)); H33[2, :] = [0.3, -0.2, 1.0]
xs = rng.uniform(-2, 2, (4, 2))
x1 = apply_h(H_proj(H33), xs)
x2 = apply_h(H_proj(2.0 * H33), xs)
ok = np.allclose(x1, x2, atol=1e-12)
verify("射影尺度等价：H̃ 与 2H̃ 归一化后同一点（式 2.20–2.21）", ok,
       "4 组随机点逐点一致")

# —— 12. 射影保直线：直线上三点变换后仍共线（p.41 preserve straight lines）—————————
l_pts = rng.uniform(-2, 2, (2, 2))
lam3 = rng.uniform(0.3, 0.7)
third = (1 - lam3) * l_pts[0] + lam3 * l_pts[1]
three = np.vstack([l_pts, third])
out = apply_h(H_proj(H33), three)
cr = np.cross(np.r_[out[1] - out[0], 0], np.r_[out[2] - out[0], 0])[2]
ok = abs(cr) < 1e-9
verify("射影保直线：共线三点变换后仍共线（式 2.20）", ok,
       f"叉积={cr:.2e}")

# —— 13. 射影反例：原本平行的两条边，像的方向不再平行（射影不保平行）—————————————————
e_left = apply_h(H_proj(H33), np.array([[0.0, 0.0], [0.0, 1.0]]))    # 左边缘（原方向竖直）
e_right = apply_h(H_proj(H33), np.array([[1.0, 0.0], [1.0, 1.0]]))   # 右边缘（与左边缘平行）
d1v = unit(e_left[1] - e_left[0])
d2v = unit(e_right[1] - e_right[0])
cr12 = d1v[0] * d2v[1] - d1v[1] * d2v[0]
ok = abs(cr12) > 1e-3      # 两像方向叉积 ≠ 0 → 平行性破坏（纯旋转下叉积恒为 0，此检验有效）
verify("射影反例：两条平行边的像方向不平行（射影不保平行）", ok,
       f"左边方向→({d1v[0]:+.3f},{d1v[1]:+.3f}) · 右边方向→({d2v[0]:+.3f},{d2v[1]:+.3f}) · 叉积={cr12:+.3f}")

# —— 14. 群闭合：相似∘相似=相似、仿射∘仿射=仿射（Table 2.1 嵌套群；p.41 closed under composition）—————
M1 = S_sim(rng.uniform(0.5, 2.0), rng.uniform(-np.pi, np.pi), rng.uniform(-2, 2, 2))
M2 = S_sim(rng.uniform(0.5, 2.0), rng.uniform(-np.pi, np.pi), rng.uniform(-2, 2, 2))
M12 = M2 @ M1
scale_prod = np.sqrt(np.linalg.det(M12[:2, :2]))            # det(s₂R₂·s₁R₁)=s₁²s₂²
rot_prod = M12[:2, :2] / scale_prod
# 相似块的直接结构检验：[[a,−b],[b,a]] 型 ⇔ m00=m11 且 m01=−m10（非同义反复）
ok = (np.isclose(M12[0, 0], M12[1, 1], atol=1e-9)
      and np.isclose(M12[0, 1], -M12[1, 0], atol=1e-9)
      and np.allclose(rot_prod.T @ rot_prod, np.eye(2), atol=1e-9)
      and np.linalg.det(rot_prod) > 0 and scale_prod > 0)
B1 = A_aff(rng.uniform(-2, 2, (2, 2)), rng.uniform(-2, 2, 2))
B2 = A_aff(rng.uniform(-2, 2, (2, 2)), rng.uniform(-2, 2, 2))
B12 = B2 @ B1
B12_expect = np.eye(3); B12_expect[:2, :2] = B2[:2, :2] @ B1[:2, :2]
B12_expect[:2, 2] = B2[:2, :2] @ B1[:2, 2] + B2[:2, 2]
ok = ok and np.allclose(B12, B12_expect, atol=1e-12)
verify("群闭合：相似∘相似=相似、仿射∘仿射=仿射（Table 2.1 嵌套群）", ok,
       f"s∘s 二乘二块为 [[a,−b],[b,a]] 型 · s={scale_prod:.3f} · 仿射块乘一致")

# —— 15. 逆在同族：T⁻¹、S⁻¹、H̃⁻¹ 各自留在族内（p.41 inverse in the same group）—————
t1 = rng.uniform(-2, 2, 2)
Tinv = np.linalg.inv(T(t1))
ok1 = np.allclose(Tinv[:2, 2], -t1) and np.allclose(Tinv[:2, :2], np.eye(2))
Sinv = np.linalg.inv(M1)
s1_scale = np.sqrt(np.linalg.det(M1[:2, :2]))        # M1 = [s₁R t;0 1]，det(s₁R)=s₁²
s_inv = 1.0 / s1_scale
# 验证 S⁻¹ 的旋转块 = (1/s₁)·Rᵀ 型：除以 1/s₁ 后应为正交阵，且与 Rmat(θ) 参数化一致
Rb = Sinv[:2, :2] / s_inv
th_rec = np.arctan2(Rb[1, 0], Rb[0, 0])
ok2 = np.allclose(Rb.T @ Rb, np.eye(2), atol=1e-9) \
    and np.allclose(Rb, Rmat(th_rec), atol=1e-9) \
    and np.allclose(Sinv[:2, :2], (1.0 / s1_scale) * Rmat(-np.arctan2(M1[1, 0], M1[0, 0])), atol=1e-9)
# 射影逆仍是射影：定义性检查——射影群成员资格＝矩阵可逆，往返还原为该性质的数值体现
Hinv = np.linalg.inv(H33)
xs_rt = apply_h(H_proj(Hinv), apply_h(H_proj(H33), xs))
ok3 = np.allclose(xs_rt, xs, atol=1e-9)
ok = ok1 and ok2 and ok3
verify("逆在同族：T⁻¹=[I,−t]、S⁻¹=(1/s)Rᵀ 型、H̃⁻¹ 往返还原（嵌套群）", ok,
       f"s′=1/s₁={s_inv:.3f} · R′=R(−θ₁) · H̃⁻¹ 往返逐点一致（定义性检查：成员资格＝可逆）")

# —— 16. stretch/squash 非 sR 形式：diag(sx,sy)（sx≠sy）不满足 [[a,−b],[b,a]]（p.43 does not nest）——
sx, sy = 2.0, 0.8
Dm = np.diag([sx, sy])
is_sim_form = np.isclose(Dm[0, 0], Dm[1, 1]) and np.isclose(Dm[0, 1], -Dm[1, 0])
ok = (not is_sim_form) and abs(Dm[0, 0] - Dm[1, 1]) > 1e-9
verify("stretch/squash 非 sR 形式：diag(2,0.8) 不满足 [[a,−b],[b,a]]（p.43 does not nest）", ok,
       f"a−d={Dm[0, 0] - Dm[1, 1]:.1f} ≠ 0")

# —— 17. co-vector 恒等：l̃′=H̃⁻ᵀl̃ 时 l̃′·x̃′=l̃·x̃（式 2.22）———————————————————————
l_tilde = rng.uniform(-2, 2, 3)
xs_h = np.c_[xs, np.ones(len(xs))]
lhs = (np.linalg.inv(H33).T @ l_tilde) @ (xs_h @ H33.T).T      # l̃′ᵀH̃x̃
rhs = l_tilde @ xs_h.T
ok = np.allclose(lhs, rhs, atol=1e-9)
verify("co-vector 恒等：l̃′=H̃⁻ᵀl̃ ⇒ l̃′·x̃′=l̃·x̃（式 2.22）", ok,
       "4 组随机点逐点一致")

# —— 18. Table 2.1 DoF 阶梯：2/3/4/6/8 与矩阵形状（p.42 表本体；记账式对照）———————
dofs = {"translation": 2, "rigid (Euclidean)": 3, "similarity": 4,
        "affine": 6, "projective": 8}
# 参数计数：t(2) / t+θ(3) / t+θ+s(4) / 2×3 六元(6) / 3×3 九元−尺度(8)
counted = [2, 1 + 1 + 1, 3 + 1, 6, 9 - 1]
ok = counted == [dofs[k] for k in dofs] and counted == [2, 3, 4, 6, 8]
verify("Table 2.1 DoF 阶梯：2/3/4/6/8（参数计数 = 表值）", ok,
       "平移2 · 刚体3 · 相似4 · 仿射6 · 射影 9−尺度=8")

# —— 重绘图：据原书 Figure 2.4（印刷 p.40）重绘五联方格图（讲者重绘，原书图未使用）—————
def draw_fig2_4(outpath):
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.6))
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    grid = np.linspace(0, 1, 6)
    params = [
        ("translation", r"$\tilde{x}'=[\,I\ \ t\,;\,0\ \ 1\,]\tilde{x}$",
         T(np.array([0.30, 0.18])), "2 DoF"),
        ("rigid (Euclidean)", r"$\tilde{x}'=[\,R\ \ t\,;\,0\ \ 1\,]\tilde{x}$",
         A_aff(Rmat(np.deg2rad(18)), np.array([0.28, 0.10])), "3 DoF"),
        ("similarity", r"$\tilde{x}'=[\,sR\ \ t\,;\,0\ \ 1\,]\tilde{x}$",
         S_sim(1.15, np.deg2rad(20), np.array([0.10, 0.05])), "4 DoF"),
        ("affine", r"$\tilde{x}'=[\,A\ \ t\,;\,0\ \ 1\,]\tilde{x}$",
         A_aff(np.array([[1.05, 0.35], [0.10, 0.90]]), np.array([0.05, 0.02])), "6 DoF"),
        ("projective", r"$\tilde{x}'=\tilde{H}\tilde{x}$",
         H_proj(np.array([[1.0, 0.18, 0.02], [0.10, 0.92, 0.02], [0.18, 0.08, 0.80]])), "8 DoF"),
    ]
    for ax, (name, eq, M, dof) in zip(axes, params):
        ax.set_facecolor(PAPER)
        for sp in ax.spines.values():
            sp.set_color(BROWN)
        ax.set_xticks([]), ax.set_yticks([])
        # 方格线（变换后）
        for gv in grid:
            line = np.c_[np.full(24, gv), np.linspace(-0.08, 1.08, 24)]
            line2 = np.c_[np.linspace(-0.08, 1.08, 24), np.full(24, gv)]
            for ln in (line, line2):
                out = apply_h(M, ln)
                inside = (out[:, 0] > -0.1) & (out[:, 0] < 1.75) & (out[:, 1] > -0.1) & (out[:, 1] < 1.75)
                if inside.any():
                    ax.plot(*out[inside].T, color="#c8d4e3", lw=0.8, zorder=1)
        poly_out = apply_h(M, sq)
        ax.add_patch(patches.Polygon(poly_out, closed=True, facecolor="#a8c0dc",
                                     edgecolor=INK, lw=1.6, alpha=0.85, zorder=2))
        # 原位置虚线框
        ax.add_patch(patches.Polygon(sq, closed=True, fill=False, edgecolor=BROWN,
                                     ls="--", lw=1.0, zorder=3))
        ax.set_xlim(-0.35, 1.75), ax.set_ylim(-0.2, 1.75)
        ax.set_aspect("equal")
        ax.set_title(f"{name} · {dof}", fontsize=11.5, color=INK, pad=8)
        ax.text(0.04, 0.965, eq, ha="left", va="top", fontsize=10, color=INK,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#d8cdb8", alpha=0.92))
    fig.suptitle("E014 · 据原书 Figure 2.4（第 40 页）与 Table 2.1（第 42 页）重绘：同一方格的五级变形",
                 fontsize=13, color=INK, y=0.99)
    fig.text(0.985, 0.008, "讲者设计演示：方格形状/变换参数为示例值；断言见 verify 各项；原书图未使用",
             ha="right", fontsize=8.5, color=BROWN)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def sync_and_md5():
    import hashlib

    def md5(p):
        return hashlib.md5(p.read_bytes()).hexdigest()

    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    figdir = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures"
    files = sorted(outdir.glob("*.png"))
    for f in files:
        print(f"[md5] {f.name}  {md5(f)}")
    synced = 0
    for f in files:
        dst = figdir / f.name
        if dst.exists() and md5(dst) != md5(f):
            print(f"[demo] 注意：幻灯片目录与输出存在漂移，已覆盖同步 {dst.name}")
        dst.write_bytes(f.read_bytes())
        if md5(dst) == md5(f):
            synced += 1
            print(f"[demo] 已同步 {dst}")
        else:
            print(f"[demo] 同步校验失败 {dst}")
    ok = len(files) > 0 and synced == len(files)
    print(f"[verify] 输出图已同步 slides/figures/（逐文件 md5 比对）  ({synced}/{len(files)} 一致)  "
          f"[{'PASS' if ok else 'FAIL'}]")
    return ok


def main():
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(exist_ok=True)
    draw_fig2_4(outdir / "e014_fig2_4.png")
    print(f"[demo] 图已存 {outdir / 'e014_fig2_4.png'}")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, _ in RESULTS:
        if not ok:
            print(f"[FAIL-DETAIL] {name}")
    sync_ok = sync_and_md5()
    print(f"汇总：{n_pass}/{len(RESULTS)} 数据断言 PASS · 同步检查 {'PASS' if sync_ok else 'FAIL'}"
          + (" —— 全部通过" if n_pass == len(RESULTS) and sync_ok else " —— 存在失败"))


if __name__ == "__main__":
    main()
