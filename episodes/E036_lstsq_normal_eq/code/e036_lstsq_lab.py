#!/usr/bin/env python3
"""E036 S2 主实验：最小二乘的直观与正规方程推导——数值验证存档。

原书锚（evidence v1，印刷 p.927–930）：
  A.25  E_VLS = Σ|yᵢ−(m xᵢ+b)|²      —— 直线拟合竖直残差平方（Figure A.2a 概念，图为本集程序重绘）
  A.26  E_PLS = Σ|yᵢ−(Σⱼ aⱼ xᵢʲ)|²   —— 多项式拟合（函数非线性、对参数线性）
  A.28  E_LLS = Σ|aᵢx−bᵢ|² = ‖Ax−b‖² —— 一般线性最小二乘
  A.29/A.30  E_LLS = xᵀ(AᵀA)x − 2xᵀ(Aᵀ)b + ‖b‖²，正规方程 (AᵀA)x = Aᵀb
  A.35 目标 E_TLS = Σ(aᵢx)²=‖Ax‖²；A.36 约束 ‖x‖=1（平凡解 x=0 之故）→解=AᵀA 最小特征向量=最后右奇异向量
  A.40/A.41  E_TLS-2D = Σ(axᵢ+byᵢ+c)²，C=AᵀA（[xᵢ yᵢ 1] 行）
  A.42–A.44  原书处方：1 列无噪声 → 统计最优性破坏 → 先减均值 x̂ᵢ=xᵢ−x̄, ŷᵢ=yᵢ−ȳ，
             拟合 a(x−x̄)+b(y−ȳ)=0（直线过数据中心）——这才是 2D 直线 TLS 的正确解法

实验（断言全过才写 PASS 汇总，任何 FAIL 抛 AssertionError 进 stderr）：
  E1 直线拟合三路解一致：正规方程解 vs np.linalg.lstsq（SVD）vs scipy 数值极小化
  E2 手推解析解（∂E/∂m=∂E/∂b=0 的闭式）与正规方程解一致——「证明」一环
  E3 正规方程解处的数值梯度 ≈ 0（极小点验证）
  E4 多项式拟合（2 次）走同一套正规方程——「对参数线性」演示（A.26）
  E5 复算纯计数：点数/残差数/断言数（讲者复算辖域=纯计数）
  E6 TLS（A.2.1）：A.41 齐次直接解（特征向量=SVD 交叉验证、全向量归一化约束）
     + 原书 A.42–A.44 中心化处方解（过数据中心；未中心化直接解不过质心——这是
     「1 列无噪声、统计不最优」该毛病在图上的表现，原书理由见 p.930，门A round1 P0-1/round3 N-P1-1）
  E7 正弦拟合 A.27：已知频率，A sin(2πfx+φ) 拆 (B,C) 后线性
读数存档：m̂, b̂, E(m̂,b̂), 每点残差, TLS 直接解/中心化解——口播引用一律以本存档为准。
输出：code/outputs/e036_lstsq_lab.txt（stdout/stderr 分离）；figures/ 三张程序生成图。
"""
import json
import pathlib
import sys

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = pathlib.Path(__file__).resolve().parent  # episodes/E036_lstsq_normal_eq/code
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

checks: list[tuple[str, bool, str]] = []


def chk(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        raise AssertionError(f"{name}: {detail}")


# —— 固定种子数据（可复现，口播读数锚）：8 点，真实线 m=1.30, b=0.70，噪声 σ=0.5 ——
rng = np.random.default_rng(2036)
m_true, b_true = 1.30, 0.70
x = np.arange(8, dtype=float)                      # 0..7
y = m_true * x + b_true + rng.normal(0.0, 0.5, 8)  # 竖直噪声（A.25 口径：竖直残差）
n = len(x)

# —— E1 直线拟合：A = [x, 1]，正规方程 (AᵀA)x = Aᵀb ——
A = np.column_stack([x, np.ones(n)])
b_col = y
ATA = A.T @ A
ATb = A.T @ b_col
x_normal = np.linalg.solve(ATA, ATb)
m_hat, b_hat = x_normal
E_hat = float(np.sum((y - (m_hat * x + b_hat)) ** 2))

x_lstsq, *_ = np.linalg.lstsq(A, b_col, rcond=None)
res_min = minimize(lambda p: float(np.sum((y - (p[0] * x + p[1])) ** 2)),
                   x0=[0.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12})
print("== E1 直线拟合三路解一致 ==")
chk("E1a 正规方程 vs lstsq(SVD)",
    np.allclose(x_normal, x_lstsq, atol=1e-10),
    f"max|Δ|={np.max(np.abs(x_normal - x_lstsq)):.2e}")
chk("E1b 正规方程 vs 数值极小化(Nelder-Mead)",
    np.allclose(x_normal, res_min.x, atol=1e-5),
    f"max|Δ|={np.max(np.abs(x_normal - res_min.x)):.2e}")

# —— E2 手推解析解：∂E/∂m=0 与 ∂E/∂b=0 的闭式（E036「证明」一环的讲者复算锚）——
# ∂E/∂m = -2Σ xᵢ(yᵢ−m xᵢ−b) = 0 → m Σx² + b Σx = Σ x y
# ∂E/∂b = -2Σ (yᵢ−m xᵢ−b) = 0   → m Σx  + b n  = Σ y
Sx, Sy, Sxx, Sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
det = n * Sxx - Sx * Sx
m_closed = (n * Sxy - Sx * Sy) / det
b_closed = (Sxx * Sy - Sx * Sxy) / det
print("== E2 手推解析解（Σ 求偏导=0）与正规方程解一致 ==")
chk("E2a m 解析解一致", abs(m_closed - m_hat) < 1e-10,
    f"解析 m*={m_closed:.12f} vs 正规 {m_hat:.12f}")
chk("E2b b 解析解一致", abs(b_closed - b_hat) < 1e-10,
    f"解析 b*={b_closed:.12f} vs 正规 {b_hat:.12f}")

# —— E3 极小点验证：正规方程解处数值梯度 ≈ 0 ——
def grad(p):
    r = y - (p[0] * x + p[1])
    return np.array([-2 * np.sum(x * r), -2 * np.sum(r)])
g = grad([m_hat, b_hat])
print("== E3 极小点数值梯度 ==")
chk("E3 |∇E(m̂,b̂)| < 1e-8", np.max(np.abs(g)) < 1e-8,
    f"|∇|=({g[0]:.2e}, {g[1]:.2e})")

# —— E4 多项式拟合（2 次）：同一套正规方程，函数非线性、对参数线性（A.26）——
A2 = np.column_stack([x ** 2, x, np.ones(n)])
coef2 = np.linalg.solve(A2.T @ A2, A2.T @ y)
E2_val = float(np.sum((y - A2 @ coef2) ** 2))
# 数值极小化交叉验证
res2 = minimize(lambda p: float(np.sum((y - np.polyval(p, x)) ** 2)),
                x0=[0.0, 0.0, 0.0], method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-12})
print("== E4 多项式拟合（2 次）走同一套正规方程 ==")
chk("E4a 二次多项式 vs 数值极小化",
    np.allclose(coef2, res2.x, atol=1e-5),
    f"max|Δ|={np.max(np.abs(coef2 - res2.x)):.2e}")
chk("E4b 二次拟合残差 ≤ 直线拟合残差（嵌套模型）", E2_val <= E_hat + 1e-9,
    f"E2次={E2_val:.4f} ≤ E1次={E_hat:.4f}")

# —— E5 复算纯计数 ——
residuals = y - (m_hat * x + b_hat)
print("== E5 复算纯计数 ==")
chk("E5a 数据点数=8、残差条数=8", n == 8 and len(residuals) == 8, f"n={n}")

# —— E6 Total least squares（A.2.1，印刷 p.929–930）——
# A.35/A.36：齐次平方误差 ‖Ax‖² 约束 ‖x‖=1（平凡解 x=0 之故），解=AᵀA 最小特征向量
# A.40/A.41：2D 直线 E=Σ(axᵢ+byᵢ+c)²，C=AᵀA（每行 [xᵢ, yᵢ, 1]）
M = np.column_stack([x, y, np.ones(n)])
MTM = M.T @ M
w, V = np.linalg.eigh(MTM)
a_tls, b_tls, c_tls = V[:, 0]                       # 最小特征值对应的特征向量（A.41 直接解）
_, _, Vt = np.linalg.svd(M, full_matrices=False)
a_svd, b_svd, c_svd = Vt[-1]                        # 最后右奇异向量（交叉验证）
cx, cy = x.mean(), y.mean()
print("== E6 Total least squares（A.41 直接解 + A.42–A.44 中心化处方）==")
chk("E6a A.41 直接解：特征向量 vs SVD 最后右奇异向量（方向一致）",
    np.allclose(np.abs([a_tls, b_tls, c_tls]), np.abs([a_svd, b_svd, c_svd]), atol=1e-10),
    f"eigh=({a_tls:.6f},{b_tls:.6f},{c_tls:.6f})")
chk("E6b A.36 约束复现：解向量模长=1（全向量归一化）",
    abs(np.hypot(np.hypot(a_tls, b_tls), c_tls) - 1.0) < 1e-10,
    f"‖(a,b,c)‖={np.hypot(np.hypot(a_tls, b_tls), c_tls):.12f}")
m_tls = -a_tls / b_tls
b_tls_line = -c_tls / b_tls
chk("E6c 噪声下 VLS 与 A.41 直接解不同（两条线分得开）",
    abs(m_tls - m_hat) > 1e-3,
    f"VLS m̂={m_hat:.4f} vs 直接解 m*={m_tls:.4f}（Δ={abs(m_tls - m_hat):.4f}）")
# A.42–A.44：1 列无噪声 → 统计最优性破坏 → 原书处方=先减均值再拟合（门A round1 P0-1）
xh, yh = x - cx, y - cy
M2 = np.column_stack([xh, yh])
w2, V2 = np.linalg.eigh(M2.T @ M2)
a_c, b_c = V2[:, 0]                                 # 中心化 2×2 最小特征向量（‖(a,b)‖=1）
m_tls_c = -a_c / b_c
b_tls_c = cy - m_tls_c * cx                         # 直线过数据中心 (x̄, ȳ)
dist_centroid_direct = abs(a_tls * cx + b_tls * cy + c_tls) / np.hypot(a_tls, b_tls)
chk("E6d A.42–A.44 中心化处方解：特征方程成立+直线过数据中心（偏差<1e-12）",
    np.allclose(M2.T @ M2 @ [a_c, b_c], w2[0] * np.array([a_c, b_c]), atol=1e-10)
    and abs(m_tls_c * cx + b_tls_c - cy) < 1e-12,
    f"中心化 TLS m**={m_tls_c:.4f}，过点 ({cx:.4f}, {cy:.4f})")
chk("E6e 未中心化 A.41 直接解不过数据中心（原书理由=1 列无噪声统计不最优；不过数据中心是该毛病在图上的表现）",
    dist_centroid_direct > 1e-3,
    f"质心到直接解直线的正交距离={dist_centroid_direct:.4f}（>1e-3）")
chk("E6f 中心化解 ‖(a,b)‖=1（2×2 特征向量归一化）",
    abs(np.hypot(a_c, b_c) - 1.0) < 1e-10,
    f"‖(a,b)‖={np.hypot(a_c, b_c):.12f}")
chk("E6g 中心化 TLS 与 VLS 斜率分得开（两条线放大看能分开）",
    abs(m_tls_c - m_hat) > 1e-3,
    f"VLS m̂={m_hat:.4f} vs 中心化 TLS m**={m_tls_c:.4f}（Δ={abs(m_tls_c - m_hat):.4f}）")

# —— E7 正弦拟合 A.27：已知频率 f，A sin(2πfx+φ) 对 (B,C)=(A cosφ, A sinφ) 线性 ——
A_true, phi_true, f0 = 2.5, 0.7, 0.5
xs = np.linspace(0, 4, 20)
ys = A_true * np.sin(2 * np.pi * f0 * xs + phi_true) + rng.normal(0, 0.05, 20)
As = np.column_stack([np.sin(2 * np.pi * f0 * xs), np.cos(2 * np.pi * f0 * xs)])
Bc = np.linalg.solve(As.T @ As, As.T @ ys)
A_rec = float(np.hypot(Bc[0], Bc[1]))
phi_rec = float(np.arctan2(Bc[1], Bc[0]))
res_s = minimize(lambda p: float(np.sum((ys - p[0] * np.sin(2 * np.pi * f0 * xs + p[1])) ** 2)),
                 x0=[1.0, 0.0], method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-12})
print("== E7 正弦拟合（A.27）——「对参数线性」第二例 ==")
chk("E7a (B,C) 正规方程 vs 数值极小化(A,φ)",
    abs(A_rec - res_s.x[0]) < 1e-4 and abs(phi_rec - res_s.x[1]) < 1e-4,
    f"恢复 A={A_rec:.4f} φ={phi_rec:.4f} vs 数值 A={res_s.x[0]:.4f} φ={res_s.x[1]:.4f}")
chk("E7b 振幅恢复与真值一致", abs(A_rec - A_true) < 0.05,
    f"A*={A_rec:.4f} vs 真值 {A_true}")

# —— 复算纯计数 ——
chk("E5b 断言总数=18（含本条）", len(checks) + 1 == 18, f"含本条共 {len(checks) + 1} 条")

# —— 读数存档（口播引用唯一来源）——
readings = {
    "n_points": n,
    "m_hat": round(m_hat, 4), "b_hat": round(b_hat, 4),
    "m_true": m_true, "b_true": b_true,
    "E_line": round(E_hat, 4), "E_quad": round(E2_val, 4),
    "residuals": [round(float(r), 4) for r in residuals],
    "sigma_noise": 0.5, "seed": 2036,
    "closed_form": {"m": round(float(m_closed), 6), "b": round(float(b_closed), 6)},
    "e1a_max_delta": float(f"{np.max(np.abs(x_normal - x_lstsq)):.2e}"),
    "e1b_max_delta": float(f"{np.max(np.abs(x_normal - res_min.x)):.2e}"),
    "tls": {"direct_m": round(float(m_tls), 4), "direct_b": round(float(b_tls_line), 4),
            "direct_centroid_offset": round(float(dist_centroid_direct), 4),
            "centered_m": round(float(m_tls_c), 4), "centered_b": round(float(b_tls_c), 4),
            "centered_passes_centroid": True,
            "delta_m_centered_vs_vls": round(float(abs(m_tls_c - m_hat)), 4),
            "delta_m_direct_vs_centered": round(float(abs(m_tls - m_tls_c)), 4)},
    "sine": {"A_true": A_true, "A_rec": round(A_rec, 4),
             "phi_true": phi_true, "phi_rec": round(phi_rec, 4), "f_known": f0,
             "delta_A": round(float(abs(A_rec - A_true)), 4)},
}
(OUT / "e036_lstsq_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))

print("\n== 读数存档 ==")
print(json.dumps(readings, ensure_ascii=False, indent=1))
print("\nLEDGER: 断言 %d 条，全部 PASS" % len(checks))
print("E036-LSTSQ-LAB-DONE")

# —— 图 1：散点+拟合线+竖直残差（Figure A.2a 概念的程序重绘，红线 2）——
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(figsize=(8.6, 5.6), dpi=100)
xs = np.linspace(-0.3, 7.3, 100)
ax.scatter(x, y, marker="x", s=64, color="#c0392b", linewidths=2, zorder=3, label="数据点（含噪声）")
ax.plot(xs, m_hat * xs + b_hat, color="#2f4858", lw=2,
        label=f"拟合线 y = m̂x + b̂（程序生成）")
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [yi, m_hat * xi + b_hat], color="#e67e22", lw=1.2, alpha=0.85, zorder=2)
ax.plot([], [], color="#e67e22", lw=1.2, label="竖直残差（平方后求和最小化）")
ax.set_title("直线拟合：残差平方和最小的那条线（程序生成——原书 Figure A.2a 概念重绘，未用原书原图）",
             fontsize=13)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend(loc="upper left", fontsize=12)
ax.text(0.99, 0.02, "原书第 927–928 页：式 A.25 竖直残差平方和 · 程序重绘，未用原书原图",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color="#5a6472",
        bbox=dict(fc="white", alpha=0.75, ec="none"))
fig.tight_layout()
fig.savefig(FIG / "e036_fit_demo.png")
plt.close(fig)

# —— 图 2：E(m,b) 等高线+极小点（「从猜到证明」的几何直觉）——
# D27 显示尺寸断言依据：deck 页 3 右栏显示盒 765×300 CSS px（.p3fix max-height 300、
# 列宽 765），图按 765×300 @dpi100 出图 → 显示缩放 1:1；dpi100 下 1pt=100/72px，
# 图内最小字 12pt≈16.7 CSS px ≥12 达标（口径更正：12pt≠12px——round3 N-P2-3）
plt.rcParams.update({"font.size": 12})
fig, ax = plt.subplots(figsize=(7.65, 3.0), dpi=100)
ms = np.linspace(m_hat - 1.2, m_hat + 1.2, 240)
bs = np.linspace(b_hat - 1.2, b_hat + 1.2, 240)
MM, BB = np.meshgrid(ms, bs)
E = np.sum((y[None, None, :] - (MM[..., None] * x[None, None, :] + BB[..., None])) ** 2, axis=-1)
cs = ax.contourf(MM, BB, np.log10(E), levels=30, cmap="viridis")
cbar = fig.colorbar(cs, ax=ax)
cbar.set_label("log₁₀ 残差平方和 E(m, b)", fontsize=12)
cbar.ax.tick_params(labelsize=12)
ax.plot(m_hat, b_hat, "*", markersize=16, color="#e67e22",
        label="极小点——正规方程解")
ax.legend(loc="upper left", fontsize=12)
ax.set_title("能量面 E(m, b)：碗底只有一个（程序生成，本节公式 A.29 的可视化）", fontsize=13)
ax.set_xlabel("斜率 m", fontsize=12); ax.set_ylabel("截距 b", fontsize=12)
ax.tick_params(labelsize=12)
ax.text(0.99, 0.02, "原书第 928 页：式 A.29–A.30 正规方程 · 程序生成",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color="#5a6472",
        bbox=dict(fc="white", alpha=0.75, ec="none"))
fig.tight_layout()
fig.savefig(FIG / "e036_energy_contour.png")
plt.close(fig)

# —— 图 3：VLS vs TLS 对比（A.25 竖直残差 vs A.42–A.44 中心化 TLS；Figure A.2a/b 概念重绘）——
# D27：deck 页 5 右栏显示盒 765×560 CSS px，图按 765×500 @dpi100 → 显示缩放 1:1
fig, ax = plt.subplots(figsize=(7.65, 5.0), dpi=100)
ax.scatter(x, y, marker="x", s=64, color="#c0392b", linewidths=2, zorder=3, label="数据点（含噪声）")
ax.plot(xs, m_hat * xs + b_hat, color="#2f4858", lw=2, label=f"竖直量距 VLS（式 A.25）：斜率 {m_hat:.4f}")
ax.plot(xs, m_tls_c * xs + b_tls_c, color="#1f77b4", lw=2, ls="--",
        label=f"垂直量距 TLS（式 A.42–A.44）：斜率 {m_tls_c:.4f}")
ax.plot(xs, m_tls * xs + b_tls_line, color="#6b7280", lw=1.2, ls=":",
        label=f"式 A.41 直接解（未中心化）：斜率 {m_tls:.4f}")
for xi, yi in zip(x, y):
    ax.plot([xi, xi], [yi, m_hat * xi + b_hat], color="#e67e22", lw=1.1, alpha=0.8, zorder=2)
# 中心化 TLS 的正交残差（点到该直线的垂线，‖(a,b)‖=1 → 距离=ax̂+bŷ）
for xi, yi in zip(x, y):
    d = a_c * (xi - cx) + b_c * (yi - cy)
    ax.plot([xi, xi - a_c * d], [yi, yi - b_c * d], color="#6fb3e0", lw=1.1, alpha=0.8, zorder=2)
ax.plot([], [], color="#6fb3e0", lw=1.1, label="TLS 垂线残差（垂直于虚线）")
ax.plot(cx, cy, "+", markersize=12, color="#1f77b4", markeredgewidth=2, zorder=4)
ax.annotate("数据中心（虚线过此点）", (cx, cy), textcoords="offset points",
            xytext=(10, -16), fontsize=12, color="#1f77b4")
ax.set_title("两种残差两种线：竖直量距（VLS）vs 垂直量距（TLS）——同一条数据，答案不同", fontsize=13)
ax.set_xlabel("x", fontsize=12); ax.set_ylabel("y", fontsize=12)
ax.tick_params(labelsize=12)
ax.legend(loc="upper left", fontsize=12)
ax.text(0.99, 0.02, "原书第 927–930 页：式 A.25 / 式 A.40–A.44 · Figure A.2 概念程序重绘，未用原书原图",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12, color="#5a6472",
        bbox=dict(fc="white", alpha=0.75, ec="none"))
fig.tight_layout()
fig.savefig(FIG / "e036_tls_vs_vls.png")
plt.close(fig)
print("FIGURES-DONE 3 张（图 1 留档不上屏；图 2→deck 页 3 765×300；图 3→deck 页 5 765×500，均 1:1 显示，最小字 12pt≈16.7 CSS px）")

# —— P1-5 显示尺寸：PNG 尺寸回归断言 + 渲染阶段显示盒实测分工 ——
# 口径更正（门A round3 N-P2-3）：dpi100 下 1pt=100/72px，12pt≈16.7px（≠12px）；
# 本断言只做 PNG 尺寸回归保护，显示盒/缩放/最小字号的渲染阶段实测由
# code/e036_display_check.py 用 Chrome getBoundingClientRect 完成（非同义反复）。
from PIL import Image as _Img
for _name, _box in {"e036_energy_contour.png": (765, 300),
                    "e036_tls_vs_vls.png": (765, 500)}.items():
    _im = _Img.open(FIG / _name)
    assert _im.size == _box, f"{_name} PNG 尺寸 {_im.size} ≠ deck 显示盒 {_box}（D27 显示字号断言失效）"
print("DISPLAY-SIZE-ASSERT：2 张上屏图 PNG=deck 显示盒（765×300 / 765×500）；1:1 下最小字 12pt≈16.7 CSS px ≥12——渲染阶段实测见 e036_display_check.py")
