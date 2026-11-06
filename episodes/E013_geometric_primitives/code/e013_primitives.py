#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E013 实验：§2.1 几何基元的数值验证 + 原书 Figure 2.2 / 2.3 重绘

讲者设计实验（非原书内容）：
  1) 用随机数值验证 §2.1 开篇（印刷 pp.36–39）的全部代数断言：
     齐次缩放等价、增广向量往返、理想点检测、点在线上、归一化 (n̂,d) 与距离、
     无穷远线、叉积求交点与连线、平行线交于理想点、极坐标/球坐标往返、
     3D 直线两点式与方向式一致性、Plücker 坐标的比例等价与二次约束。
  2) 重绘原书 Figure 2.2(a)(b)（印刷 p.37）与 Figure 2.3（印刷 p.38）；
     另绘一张讲者设计的叉积演示图（相交线得交点 vs 平行线得理想点）。

运行：python3 e013_primitives.py
输出：outputs/e013_fig2_2a.png · e013_fig2_2b.png · e013_fig2_3.png · e013_cross_demo.png + 断言汇总
"""

import hashlib
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Arc, FancyArrowPatch
from mpl_toolkits.mplot3d import art3d

# 中文字体（macOS 优先 Hiragino Sans GB——.ttc 含 W3/W6 字重，normal 回退 300、bold 回退 600，stderr 仅为提示；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

PAPER = "#fbf6ec"
INK = "#1f4e79"
BROWN = "#8a6d3b"
LINE = "#7a5c2e"
RED = "#a63a2e"

rng = np.random.default_rng(13)   # 固定种子：E013，可复现

# ---------------------------------------------------------------- 断言
_results = []


def verify(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"[verify] {name} {'（' + detail + '）' if detail else ''} [{mark}]")


def dehomog(x_t):
    """齐次 → 普通坐标；最后一维 ≈0 判为理想点（原书 p.36：无普通坐标对应）。"""
    x_t = np.asarray(x_t, dtype=float)
    if abs(x_t[-1]) < 1e-12:
        return None   # 理想点
    return x_t[:-1] / x_t[-1]


def cross(l1, l2):
    return np.cross(np.asarray(l1, dtype=float), np.asarray(l2, dtype=float))


def random_line2d():
    """随机 2D 直线 (a,b,c)，保证 (a,b) 非零。"""
    while True:
        l = rng.uniform(-3, 3, 3)
        if np.hypot(l[0], l[1]) > 0.5:
            return l


def random_line3d():
    """随机 3D 平面 (a,b,c,d)，保证 (a,b,c) 非零。"""
    while True:
        m = rng.uniform(-3, 3, 4)
        if np.linalg.norm(m[:3]) > 0.5:
            return m


def unit(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def run_checks():
    # ---- 齐次坐标基本性质（式 2.1–2.2，印刷 p.36）-----------------------
    pts = rng.uniform(-5, 5, (5, 2))
    scales = rng.uniform(0.5, 4.0, 5)
    ok_scale = all(np.allclose(dehomog(np.r_[k * p, k]), p, atol=1e-9) for p, k in zip(pts, scales))
    verify("齐次缩放等价：k·x̄ 去齐次化回到同一普通点（5 组随机数）", ok_scale,
           f"k ∈ [{scales.min():.2f}, {scales.max():.2f}]")

    ws = rng.uniform(-3, 3, 5)
    ok_aug = all(np.allclose(dehomog(w * np.r_[p, 1.0]), p, atol=1e-9) for p, w in zip(pts, ws) if abs(w) > 1e-6)
    verify("增广向量往返：x̃ = w̃·x̄，除以 w̃ 还原 x（5 组随机数）", ok_aug,
           f"w̃ ∈ [{ws.min():.2f}, {ws.max():.2f}]")

    ideal = np.array([2.0, 3.0, 0.0])
    finite = np.array([2.0, 3.0, 1.0])
    verify("理想点检测：w̃=0 判为理想点（不除），w̃≠0 判为普通点",
           dehomog(ideal) is None and np.allclose(dehomog(finite), [2, 3]),
           "ideal=(2,3,0)→None · (2,3,1)→(2,3)")

    # ---- 2D 直线（式 2.3 与归一化，印刷 pp.36–37）-----------------------
    l = random_line2d()
    a, b, c = l
    x0 = rng.uniform(-3, 3)
    p_on = np.array([x0, -(a * x0 + c) / b])
    verify("点在线上：对随机直线构造线上点，x̄·l̃ = 0", abs(a * p_on[0] + b * p_on[1] + c) < 1e-9,
           f"l=({a:.3f},{b:.3f},{c:.3f}) · p=({p_on[0]:.3f},{p_on[1]:.3f})")

    s = np.hypot(a, b)
    n_hat, d = np.array([a, b]) / s, c / s
    foot2 = -d * n_hat          # 原点到直线的垂足（独立构造，非 d 的定义复述）
    verify("归一化 (n̂,d)：‖n̂‖=1；垂足 x₀=−d·n̂ 满足线方程且 ‖x₀‖=|d|（原点距离）",
           abs(np.linalg.norm(n_hat) - 1) < 1e-9
           and abs(n_hat @ foot2 + d) < 1e-9
           and abs(np.linalg.norm(foot2) - abs(d)) < 1e-9,
           f"|d|={abs(d):.4f} · ‖x₀‖={np.linalg.norm(foot2):.4f}")

    p_any = rng.uniform(-3, 3, 2)
    dist_formula = abs(n_hat @ p_any + d)
    dist_direct = abs(a * p_any[0] + b * p_any[1] + c) / s
    verify("点线距离：|n̂·p+d| = p 到直线的欧氏距离", abs(dist_formula - dist_direct) < 1e-9,
           f"p=({p_any[0]:.3f},{p_any[1]:.3f}) · 距离={dist_direct:.4f}")

    l_inf = np.array([0.0, 0.0, 1.0])
    verify("无穷远线 (0,0,1)：理想点在其上（内积 0），有限点不在（内积 1）",
           abs(ideal @ l_inf) < 1e-12 and abs(finite @ l_inf - 1) < 1e-12,
           "ideal·l∞=0 · finite·l∞=1")

    theta = np.arctan2(n_hat[1], n_hat[0])
    verify("极坐标往返：n̂ = (cosθ, sinθ)，θ = atan2(n̂y, n̂x)",
           np.allclose([np.cos(theta), np.sin(theta)], n_hat, atol=1e-9),
           f"θ={theta:.4f} rad")

    # ---- 叉积两连招（式 2.4–2.5，印刷 p.37）-----------------------------
    l1, l2 = random_line2d(), random_line2d()
    while abs(l1[0] * l2[1] - l2[0] * l1[1]) < 0.5:   # 保证不平行
        l2 = random_line2d()
    xsec = cross(l1, l2)
    pt = dehomog(xsec)
    ok_on = abs(l1[0] * pt[0] + l1[1] * pt[1] + l1[2]) < 1e-8 and \
            abs(l2[0] * pt[0] + l2[1] * pt[1] + l2[2]) < 1e-8
    verify("交点 = 线向量叉积：l̃₁×l̃₂ 的交点满足两条直线方程", ok_on and xsec[2] != 0,
           f"交点=({pt[0]:.4f},{pt[1]:.4f}) · w̃={xsec[2]:.3f}≠0")

    p1, p2 = rng.uniform(-3, 3, 2), rng.uniform(-3, 3, 2)
    l_join = cross(np.r_[p1, 1.0], np.r_[p2, 1.0])
    verify("连线 = 点向量叉积：x̃₁×x̃₂ 的连线过两点",
           abs(np.r_[p1, 1.0] @ l_join) < 1e-8 and abs(np.r_[p2, 1.0] @ l_join) < 1e-8,
           f"l̃=({l_join[0]:.3f},{l_join[1]:.3f},{l_join[2]:.3f})")

    par1 = np.array([-0.5, 1.0, 0.8])
    par2 = np.array([-0.5, 1.0, -0.6])
    xpar = cross(par1, par2)
    verify("平行线叉积：第三分量为 0 → 交于理想点（可计算的事实）",
           abs(xpar[2]) < 1e-12 and np.hypot(xpar[0], xpar[1]) > 1e-6,
           f"l̃₁×l̃₂=({xpar[0]:.3f},{xpar[1]:.3f},{xpar[2]:.1f})")

    # ---- 3D 平面（式 2.7–2.8，印刷 p.38）--------------------------------
    m = random_line3d()
    ma, mb, mc, md = m
    s3 = np.linalg.norm(m[:3])
    n3, d3 = m[:3] / s3, md / s3
    foot3 = -d3 * n3            # 原点到平面的垂足（独立构造）
    verify("3D 平面归一化 (n̂,d)：‖n̂‖=1；垂足 x₀=−d·n̂ 满足面方程且 ‖x₀‖=|d|（原点距离）",
           abs(np.linalg.norm(n3) - 1) < 1e-9
           and abs(n3 @ foot3 + d3) < 1e-9
           and abs(np.linalg.norm(foot3) - abs(d3)) < 1e-9,
           f"|d|={abs(d3):.4f} · ‖x₀‖={np.linalg.norm(foot3):.4f}")

    p3 = rng.uniform(-3, 3, 3)
    dist_p = abs(n3 @ p3 + d3)
    dist_dir = abs(m[:3] @ p3 + md) / s3
    verify("点面距离：|n̂·p+d| = p 到平面的欧氏距离", abs(dist_p - dist_dir) < 1e-9,
           f"距离={dist_dir:.4f}")

    phi = np.arcsin(np.clip(n3[2], -1, 1))
    th3 = np.arctan2(n3[1], n3[0])
    sph = np.array([np.cos(th3) * np.cos(phi), np.sin(th3) * np.cos(phi), np.sin(phi)])
    verify("球坐标往返（式 2.8）：n̂ = (cosθcosφ, sinθcosφ, sinφ)",
           np.allclose(sph, n3, atol=1e-9),
           f"θ={th3:.4f} · φ={phi:.4f} rad")

    m_inf = np.array([0.0, 0.0, 0.0, 1.0])
    verify("无穷远平面 (0,0,0,1)：法向为零向量，无法归一化（‖(a,b,c)‖=0）",
           np.allclose(m_inf[:3], 0) and np.linalg.norm(m_inf[:3]) == 0)

    # ---- 3D 直线（式 2.9–2.11，印刷 pp.38–39）---------------------------
    P = rng.uniform(-3, 3, 3)
    Q = rng.uniform(-3, 3, 3)
    lam = rng.uniform(0.05, 0.95)
    r_lam = (1 - lam) * P + lam * Q
    ok_ends = np.allclose((1 - 0.0) * P + 0.0 * Q, P) and np.allclose((1 - 1.0) * P + 1.0 * Q, Q) \
        and np.allclose(0.5 * (P + Q), (1 - 0.5) * P + 0.5 * Q)
    verify("3D 两点式：r=(1−λ)p+λq，λ=0→p、λ=1→q、λ=1/2→中点", ok_ends and
           np.linalg.norm(r_lam - P) <= np.linalg.norm(Q - P) + 1e-9,
           f"λ={lam:.3f} → r 在线段内")

    d_hat = unit(Q - P)
    s_step = 2.7
    Q2 = P + s_step * d_hat          # q 也在直线上（方向式出发点）
    ok_dir = all(np.allclose((1 - t) * P + t * Q2, P + (t * s_step) * d_hat, atol=1e-9)
                 for t in (0.3, 1.0, 2.5))
    verify("方向式一致：q=p+s·d̂ 时 (1−λ)p+λq = p+(λs)d̂（式 2.9 ⇄ 2.11）", ok_dir,
           f"s={s_step} · λ∈{{0.3, 1.0, 2.5}}")

    # ---- Plücker（式 2.12，印刷 p.39）-----------------------------------
    def plucker(ph, qh):
        L = np.outer(ph, qh) - np.outer(qh, ph)
        return L

    ph, qh = np.r_[P, 1.0], np.r_[Q, 1.0]
    L1 = plucker(ph, qh)
    p2h = np.r_[P + 0.35 * (Q - P), 1.0]
    q2h = np.r_[P + 1.60 * (Q - P), 1.0]
    L2 = plucker(p2h, q2h)
    v1, v2 = L1[np.triu_indices(4, 1)], L2[np.triu_indices(4, 1)]
    v1n, v2n = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    if v1n[np.argmax(np.abs(v1n))] * v2n[np.argmax(np.abs(v2n))] < 0:
        v2n = -v2n
    verify("Plücker 比例等价：同一条线、不同点对 → L 比例相同（斜对称矩阵 6 分量）",
           np.allclose(L1, -L1.T, atol=1e-12) and np.allclose(v1n, v2n, atol=1e-9),
           "p̃,q̃ 与 p̃′,q̃′（线上另两点）给同一条 Plücker 线")

    pf = (L1[0, 1] * L1[2, 3] - L1[0, 2] * L1[1, 3] + L1[0, 3] * L1[1, 2])
    verify("Plücker 约束：L 斜对称 + |L|=0 + 二次约束 L01·L23−L02·L13+L03·L12=0",
           np.allclose(L1 + L1.T, 0, atol=1e-12) and abs(np.linalg.det(L1)) < 1e-9 and abs(pf) < 1e-9,
           f"det={np.linalg.det(L1):.2e} · pf={pf:.2e} → 6 数 − 比例 1 − 约束 1 = 4 自由度")

    # ---- 同步检查（字节一致，四张图）-------------------------------------
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    figdir = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures"
    names = ["e013_fig2_2a.png", "e013_fig2_2b.png", "e013_fig2_3.png", "e013_cross_demo.png"]
    same = all((outdir / n).read_bytes() == (figdir / n).read_bytes() for n in names)
    verify("四张输出图已同步 slides/figures/（逐字节比对）", same, f"{len(names)}/{len(names)} 字节一致")


# ---------------------------------------------------------------- 绘图
def _newfig_2d(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    for sp in ax.spines.values():
        sp.set_color(BROWN)
    ax.tick_params(colors=BROWN, labelsize=9)
    return fig, ax


def _axes_arrows(ax, xlen, ylen, fs=12):
    ax.annotate("", xy=(xlen, 0), xytext=(-xlen, 0),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    ax.annotate("", xy=(0, ylen), xytext=(0, -ylen),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.2))
    ax.text(xlen, -0.22, "x", fontsize=fs, color="#666", ha="left", va="top")
    ax.text(0.12, ylen, "y", fontsize=fs, color="#666", ha="left", va="bottom")


def draw_fig2_2a(outpath):
    """重绘原书 Figure 2.2(a)（印刷 p.37）：2D 直线方程，法向量 n̂、距离 d、夹角 θ。"""
    a, b, c = 1.0, 2.0, 2.0                      # x + 2y + 2 = 0
    s = np.hypot(a, b)
    n = np.array([a, b]) / s
    d = c / s                                    # = 2/√5 > 0
    foot = -d * n                                # 原点在直线上的垂足
    fig, ax = _newfig_2d((7.8, 7.2))
    xs = np.array([-3.0, 3.0])
    ax.plot(xs, (-a * xs - c) / b, color=LINE, lw=2.4, zorder=2)
    ax.text(-2.9, 0.62, r"$\bar{x}\cdot\tilde{l}=ax+by+c=0$", fontsize=13, color=LINE)

    ax.annotate("", xy=(0, 0), xytext=tuple(foot),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2,
                                shrinkA=0, shrinkB=0), zorder=3)
    ax.text(-0.14, -0.66, r"$\hat{n}$", fontsize=14, color=INK)
    ax.text(-0.36, -0.24, "$d$", fontsize=14, color=BROWN)
    ax.plot(*foot, "o", ms=5, color=BROWN, zorder=4)

    theta = np.degrees(np.arctan2(n[1], n[0]))
    ax.plot([0, 0.85 * n[0]], [0, 0.85 * n[1]], ls=":", color=BROWN, lw=1.0, zorder=2)
    ax.add_patch(Arc((0, 0), 1.7, 1.7, angle=0, theta1=0, theta2=theta,
                     color=BROWN, lw=1.4, ls="--", zorder=2))
    ax.text(0.92, 0.30, r"$\theta$", fontsize=14, color=BROWN)
    ax.plot([0], [0], "o", ms=4, color="#666")

    _axes_arrows(ax, 3.1, 2.6)
    ax.set_xlim(-3.4, 3.4)
    ax.set_ylim(-2.9, 2.9)
    ax.set_aspect("equal")
    ax.set_title("E013 · 据原书 Figure 2.2(a)（第 37 页）重绘：2D 直线方程",
                 fontsize=13, color=INK, pad=12)
    fig.text(0.985, 0.012, "讲者设计实验：直线取 x+2y+2=0（法向 (1,2)/√5，d=2/√5）；原书图未使用",
             ha="right", fontsize=8.5, color=BROWN)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_fig2_2b(outpath):
    """重绘原书 Figure 2.2(b)（印刷 p.37）：3D 平面方程，法向量 n̂、距离 d。"""
    fig = plt.figure(figsize=(7.8, 7.2))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PAPER)

    tri_x = np.array([1.0, 0.0, 0.0])   # 平面 x+y+z=1 在第一卦限的三角形截面
    tri_y = np.array([0.0, 1.0, 0.0])
    tri_z = np.array([0.0, 0.0, 1.0])
    verts = list(zip(tri_x, tri_y, tri_z))
    poly = art3d.Poly3DCollection([verts], alpha=0.32, facecolors=[INK], linewidths=0)
    ax.add_collection3d(poly)

    n = np.array([-1.0, -1.0, -1.0]) / np.sqrt(3)   # m̃=(-1,-1,-1,1) 归一化，d=1/√3>0
    d = 1 / np.sqrt(3)
    foot = -d * n                                    # = (1/3, 1/3, 1/3)
    ax.quiver(*foot, *(0.5 * n), color=INK, lw=2.2, arrow_length_ratio=0.14)
    ax.text(*(foot + 0.26 * n + np.array([0.02, -0.07, 0.14])), r"$\hat{n}$", fontsize=14, color=INK)
    ax.plot([0, foot[0]], [0, foot[1]], [0, foot[2]], ls="--", color=BROWN, lw=1.6)
    ax.text(0.24, 0.20, 0.05, "$d$", fontsize=14, color=BROWN)
    ax.scatter(*foot, color=BROWN, s=22)
    ax.scatter([0], [0], [0], color="#666", s=22)
    ax.text(0.03, 0.03, 0.06, "$O$", fontsize=11, color="#666")

    for vec, lab in [((1.15, 0, 0), "x"), ((0, 1.15, 0), "y"), ((0, 0, 1.15), "z")]:
        ax.quiver(0, 0, 0, *vec, color="#666", lw=1.1, arrow_length_ratio=0.12)
        ax.text(*(np.array(vec) * 1.10), lab, fontsize=12, color="#666")

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_zlim(-0.05, 1.15)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=32, azim=-33)
    ax.set_axis_off()
    ax.set_title("E013 · 据原书 Figure 2.2(b)（第 37 页）重绘：3D 平面方程",
                 fontsize=13, color=INK, pad=2)
    fig.text(0.5, 0.055, r"$\bar{x}\cdot\tilde{m}=ax+by+cz+d=0$（图取 $x+y+z=1$，$\tilde{m}=(-1,-1,-1,1)$）",
             ha="center", fontsize=11.5, color=LINE)
    fig.text(0.985, 0.012, "讲者设计实验：平面取 x+y+z=1（法向指向原点侧，d=1/√3）；原书图未使用",
             ha="right", fontsize=8.5, color=BROWN)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_fig2_3(outpath):
    """重绘原书 Figure 2.3（印刷 p.38）：3D 直线 r = (1−λ)p + λq。"""
    fig = plt.figure(figsize=(7.8, 7.2))
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PAPER)

    P = np.array([0.05, 0.10, 0.15])
    Q = np.array([0.95, 0.45, 0.85])
    lam = 0.62
    R = (1 - lam) * P + lam * Q

    t_ext = np.linspace(-0.35, 1.35, 2)
    ax.plot(*(P + np.outer(t_ext, Q - P)).T, ls="--", color=BROWN, lw=1.3)
    t_seg = np.linspace(0, 1, 2)
    ax.plot(*(P + np.outer(t_seg, Q - P)).T, color=INK, lw=3.0)

    for pt, lab, col in [(P, "$p$", INK), (Q, "$q$", INK), (R, "$r$", RED)]:
        ax.scatter(*pt, color=col, s=34)
        ax.text(*(pt + np.array([0.04, 0.04, 0.07])), lab, fontsize=14, color=col)

    ax.text(0.55, 0.40, 0.08, "线段（0 ≤ λ ≤ 1）", fontsize=10.5, color=INK)
    ax.text(0.97, 0.30, 0.52, r"$r=(1-\lambda)p+\lambda q$", fontsize=12.5, color=RED)

    for vec, lab in [((1.2, 0, 0), "x"), ((0, 1.2, 0), "y"), ((0, 0, 1.2), "z")]:
        ax.quiver(0, 0, 0, *vec, color="#666", lw=1.1, arrow_length_ratio=0.12)
        ax.text(*(np.array(vec) * 1.10), lab, fontsize=12, color="#666")

    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.15)
    ax.set_zlim(-0.05, 1.15)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-55)
    ax.set_axis_off()
    ax.set_title("E013 · 据原书 Figure 2.3（第 38 页）重绘：3D 直线方程",
                 fontsize=13, color=INK, pad=2)
    fig.text(0.985, 0.012, "讲者设计实验：p、q 与 λ=0.62 为示例值；λ∈[0,1] 为线段，出区间仍在直线上；原书图未使用",
             ha="right", fontsize=8.5, color=BROWN)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_cross_demo(outpath):
    """讲者设计演示图：式(2.4) 数值验证——相交线得普通点，平行线得理想点。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    for ax in axes:
        ax.set_facecolor(PAPER)
        for sp in ax.spines.values():
            sp.set_color(BROWN)
        ax.tick_params(colors=BROWN, labelsize=8)

    # 左：相交两线  l1: x+y=1.3 · l2: x−2y=−0.4
    l1, l2 = np.array([1.0, 1.0, -1.3]), np.array([1.0, -2.0, 0.4])
    xsec = cross(l1, l2)
    pt = dehomog(xsec)
    ax = axes[0]
    xs = np.linspace(-1.5, 2.5, 2)
    for l, lab in [(l1, "$x+y=1.3$"), (l2, "$x-2y=-0.4$")]:
        ax.plot(xs, (-l[0] * xs - l[2]) / l[1], color=LINE, lw=2.0)
        ax.text(2.05, (-l[0] * 2.05 - l[2]) / l[1], lab, fontsize=10.5, color=LINE, va="center")
    ax.plot(*pt, "o", ms=9, color=RED, zorder=4)
    ax.annotate(r"$\tilde{x}=\tilde{l}_1\times\tilde{l}_2$",
                xy=pt, xytext=(0.9, 1.9), fontsize=13, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.9, 1.55, f"交点 = ({pt[0]:.2f}, {pt[1]:.2f})", fontsize=11, color=RED)
    ax.set_title("相交：得普通点（第三分量 ≠ 0）", fontsize=12.5, color=INK)
    ax.set_xlim(-1.6, 3.6)
    ax.set_ylim(-1.6, 2.4)
    ax.set_aspect("equal")

    # 右：平行两线  y=0.5x−0.8 · y=0.5x+0.6
    p1, p2 = np.array([-0.5, 1.0, 0.8]), np.array([-0.5, 1.0, -0.6])
    xpar = cross(p1, p2)
    ax = axes[1]
    xs = np.linspace(-1.5, 2.5, 2)
    for l, lab in [(p1, "$y=0.5x-0.8$"), (p2, "$y=0.5x+0.6$")]:
        ax.plot(xs, (-l[0] * xs - l[2]) / l[1], color=LINE, lw=2.0)
        ax.text(2.05, (-l[0] * 2.05 - l[2]) / l[1], lab, fontsize=10.5, color=LINE, va="center")
    ax.annotate("叉积 = (%.1f, %.1f, %.1f)\n第三分量 = 0 → 理想点（无穷远）" % tuple(xpar),
                xy=(1.6, 0.4), xytext=(-1.25, 1.75), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.15, 0.13, r"$\tilde{l}_1\times\tilde{l}_2$", fontsize=13, color=RED)
    ax.set_title("平行：得理想点（第三分量 = 0）", fontsize=12.5, color=INK)
    ax.set_xlim(-1.6, 3.6)
    ax.set_ylim(-1.6, 2.4)
    ax.set_aspect("equal")

    fig.patch.set_facecolor(PAPER)
    fig.suptitle("E013 实验 · 式(2.4)（第 37 页）数值验证：叉积求交点（讲者设计）",
                 fontsize=14, color=INK)
    fig.text(0.985, 0.012, "代码实跑输出 · 演示线为固定值；交点/连线断言用随机线，平行线断言用固定特例，见 verify 各项",
             ha="right", fontsize=8.5, color=BROWN)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def sync_and_md5():
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    figdir = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures"
    figdir.mkdir(exist_ok=True)
    names = ["e013_fig2_2a.png", "e013_fig2_2b.png", "e013_fig2_3.png", "e013_cross_demo.png"]
    for n in names:
        src = outdir / n
        (figdir / n).write_bytes(src.read_bytes())
        print(f"[md5] {n}  {hashlib.md5(src.read_bytes()).hexdigest()}")
        print(f"[demo] 已同步 {figdir / n}")


def main():
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(exist_ok=True)
    draw_fig2_2a(outdir / "e013_fig2_2a.png")
    print("[demo] 图已存", outdir / "e013_fig2_2a.png")
    draw_fig2_2b(outdir / "e013_fig2_2b.png")
    print("[demo] 图已存", outdir / "e013_fig2_2b.png")
    draw_fig2_3(outdir / "e013_fig2_3.png")
    print("[demo] 图已存", outdir / "e013_fig2_3.png")
    draw_cross_demo(outdir / "e013_cross_demo.png")
    print("[demo] 图已存", outdir / "e013_cross_demo.png")
    sync_and_md5()
    run_checks()
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"汇总：{passed}/{total} PASS —— {'全部通过' if passed == total else '存在失败'}")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
