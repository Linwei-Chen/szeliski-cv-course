#!/usr/bin/env python3
"""E074 实验：五种平面变换与不变量（程序生成，未用原书原图）。

产出三图：
  e074_transform_ladder.png  五种变换作用于同一网格
  e074_alignment_pipeline.png 特征→匹配→几何验证→搬图 的流水线示意
  e074_invariants.png        五种变换的不变量对照表
10 条断言全部数值计算，readings 落盘。
"""
import hashlib
import json
import pathlib
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99"

# ── 五种变换（作用于齐次坐标）──
def rot(d):
    c, s = np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

T = np.array([[1, 0, 0.35], [0, 1, 0.18], [0, 0, 1]], float)
E = np.array([[1, 0, 0.25], [0, 1, 0.10], [0, 0, 1]], float) @ rot(22)
S = np.array([[1, 0, 0.20], [0, 1, 0.05], [0, 0, 1]], float) @ rot(-18) @ np.diag([0.72, 0.72, 1.0])
A = np.array([[1.15, 0.45, 0.10], [0.20, 0.85, 0.06], [0, 0, 1]], float)
H = np.array([[1.00, 0.18, 0.05], [0.10, 0.96, 0.02], [0.30, 0.18, 1.00]], float)

MODELS = [
    ("平移", "translation", T, 2),
    ("欧氏", "euclidean", E, 3),
    ("相似", "similarity", S, 4),
    ("仿射", "affine", A, 6),
    ("投影", "projective", H, 8),
]


def apply(M, pts):
    q = np.c_[pts, np.ones(len(pts))] @ M.T
    return q[:, :2] / q[:, 2:3]


def dist(p, q):
    return float(np.linalg.norm(np.array(p) - np.array(q)))


def angle(a, b, c):
    u, v = np.array(a) - np.array(b), np.array(c) - np.array(b)
    cosv = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.rad2deg(np.arccos(np.clip(cosv, -1, 1))))


def parallel(a, b, c, d):
    u, v = np.array(b) - np.array(a), np.array(d) - np.array(c)
    return abs(u[0] * v[1] - u[1] * v[0]) / (np.linalg.norm(u) * np.linalg.norm(v))


def cross_ratio(a, b, c, d):
    def cr(x, y, z, w):
        return abs((z - x) * (w - y) / ((z - y) * (w - x)))
    return float(cr(a, b, c, d))


# ── 图 1：五种变换作用于同一网格 ──
def grid_segments(n=9, lo=-1, hi=1):
    segs = []
    for t in np.linspace(lo, hi, n):
        segs.append((np.array([[lo, t], [hi, t]]), np.array([[t, lo], [t, hi]])))
    return segs


def draw_warp(ax, M, title, params):
    for h, v in grid_segments():
        for seg in (h, v):
            p = apply(M, seg)
            ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.1, alpha=0.85)
    # 三个彩色标记点，显示方向与形变
    marks = np.array([[-0.6, -0.6], [0.55, -0.4], [0.1, 0.65]])
    pm = apply(M, marks)
    ax.scatter(pm[:, 0], pm[:, 1], s=42, c=[ACCENT, "#2e7d32", "#8e44ad"], zorder=5)
    ax.set_xlim(-1.9, 2.2); ax.set_ylim(-1.7, 2.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}（{params} 参数）", fontsize=15, color=INK, pad=4)


fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2), dpi=150)
fig.patch.set_facecolor(PAPER)
for ax, (cn, en, M, p) in zip(axes.flat, MODELS):
    ax.set_facecolor(PAPER)
    draw_warp(ax, M, cn, p)
axes.flat[-1].axis("off")
axes.flat[-1].text(0.5, 0.58, "参数从 2 到 8", ha="center", va="center", fontsize=18, color=ACCENT, fontweight="bold")
axes.flat[-1].text(0.5, 0.36, "越往下越灵活\n也越难稳", ha="center", va="center", fontsize=16.5, color=INK, linespacing=1.5)
fig.suptitle("同一张网格，五种搬法（程序生成）", fontsize=18, color=INK)
fig.tight_layout()
fig.savefig(FIG / "e074_transform_ladder.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 图 2：对齐流水线 ──
fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fffdf8", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center", fontsize=16.5, color=color, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.045, body, ha="center", va="center", fontsize=15, color=INK, linespacing=1.5)


box(0.04, 0.60, 0.24, 0.26, "两张图", "同一场景\n不同视角", BLUE)
box(0.36, 0.60, 0.24, 0.26, "找特征", "角点与描述子\n（第 7 章）", BLUE)
box(0.68, 0.60, 0.26, 0.26, "配对", "候选匹配\n先都留着", BLUE)
box(0.20, 0.20, 0.26, 0.26, "几何验证", "用变换解释位移\n一致性投票", ACCENT)
box(0.52, 0.20, 0.26, 0.26, "估变换", "解出矩阵\n把图搬过去", ACCENT)
for (x1, y1, x2, y2) in [(0.28, 0.73, 0.36, 0.73), (0.60, 0.73, 0.68, 0.73),
                         (0.80, 0.60, 0.55, 0.46), (0.46, 0.33, 0.52, 0.33)]:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color="#7a5c2e"))
ax.text(0.5, 0.93, "从「像」到「对得上」：对齐的四步", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
ax.text(0.5, 0.075, "红框两步是本集重点：先验几何，再解变换", ha="center", va="center", fontsize=15, color=ACCENT)
fig.savefig(FIG / "e074_alignment_pipeline.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 图 3：不变量对照表 ──
rows = ["平移", "欧氏", "相似", "仿射", "投影"]
cols = ["长度", "角度", "平行", "交比", "参数"]
vals = [
    ["不变", "不变", "不变", "不变", "2"],
    ["不变", "不变", "不变", "不变", "3"],
    ["按比例", "不变", "不变", "不变", "4"],
    ["改变", "改变", "不变", "不变", "6"],
    ["改变", "改变", "可能变", "不变", "8"],
]
fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.5, 0.94, "五种变换：什么不变，什么会变", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
x0, y0, cw, rh = 0.06, 0.79, 0.145, 0.105
x_data = 0.22
for j, c in enumerate(cols):
    ax.text(x_data + j * cw + cw / 2, 0.865, c, ha="center", va="center", fontsize=15, color=ACCENT, fontweight="bold")
for i, r in enumerate(rows):
    yy = y0 - i * rh
    ax.add_patch(FancyBboxPatch((x0, yy - rh * 0.42), 0.89, rh * 0.84,
                                boxstyle="round,pad=0.004,rounding_size=0.008",
                                fc="#fffdf8" if i % 2 == 0 else "#f1e8d8", ec="#d9c9a3", lw=0.8))
    ax.text(x0 + 0.025, yy, r, ha="left", va="center", fontsize=15.5, color=INK, fontweight="bold")
    for j, v in enumerate(vals[i]):
        color = ACCENT if v in ("改变", "可能变") else ("#2e7d32" if v in ("不变", "按比例") else INK)
        ax.text(x_data + j * cw + cw / 2, yy, v, ha="center", va="center", fontsize=15, color=color)
ax.text(0.5, 0.075, "越往下越自由：不变量越少，估参数越难", ha="center", va="center", fontsize=15, color=INK)
fig.savefig(FIG / "e074_invariants.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 数值断言 ──
results = []
pts = np.array([[-0.6, -0.6], [0.55, -0.4], [0.1, 0.65], [-0.8, 0.5], [0.7, 0.7]])
tri = np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.6]])
p0 = pts[0]
d01, d12 = dist(pts[0], pts[1]), dist(pts[1], pts[2])
a0 = angle(pts[0], pts[1], pts[2])
# 平行线：水平线与另一条水平线
par_a = np.array([-0.8, 0.3]); par_b = np.array([0.7, 0.3])
par_c = np.array([-0.7, 0.6]); par_d = np.array([0.8, 0.6])

ET = apply(T, pts); EE = apply(E, pts); ES = apply(S, pts); EA = apply(A, pts); EH = apply(H, pts)
err_T = max(abs(dist(ET[i], ET[j]) - dist(pts[i], pts[j])) for i in range(5) for j in range(5))
err_E = max(abs(dist(EE[i], EE[j]) - dist(pts[i], pts[j])) for i in range(5) for j in range(5))
ratios = [dist(ES[i], ES[j]) / dist(pts[i], pts[j]) for i in range(5) for j in range(i + 1, 5)]
sim_ratio = float(np.mean(ratios)); sim_spread = float(np.max(ratios) - np.min(ratios))
par_before = parallel(par_a, par_b, par_c, par_d)
par_after_A = parallel(apply(A, np.array([par_a, par_b]))[0], apply(A, np.array([par_a, par_b]))[1],
                       apply(A, np.array([par_c, par_d]))[0], apply(A, np.array([par_c, par_d]))[1])
ang_before = angle(par_a, par_b, par_c)
angle_delta = abs(ang_after_A - ang_before) if False else 0.0
ang_after_A = angle(apply(A, np.array([par_a, par_b]))[0], apply(A, np.array([par_a, par_b]))[1],
                    apply(A, np.array([par_c, par_d]))[0])
angle_delta = abs(ang_after_A - ang_before)
# 共线四点交比
line = np.array([[-0.9, 0.2], [-0.3, 0.2], [0.4, 0.2], [0.9, 0.2]])
LH = apply(H, line)
cr_before = cross_ratio(-0.9, -0.3, 0.4, 0.9)
cr_after = cross_ratio(*[float(v) for v in LH[:, 0]])
collinear_err = float(abs(np.cross(LH[1] - LH[0], LH[2] - LH[0])))
collinear = collinear_err < 1e-9
# 投影后平行线不再平行（取水平线 y=.3/.6 在 H 下的方向）
pa = apply(H, np.array([[par_a[0], par_a[1]], [par_b[0], par_b[1]]]))
pc = apply(H, np.array([[par_c[0], par_c[1]], [par_d[0], par_d[1]]]))
par_after_H = parallel(pa[0], pa[1], pc[0], pc[1])

tests = [
    ("平移保持距离", err_T < 1e-9, f"max|Δd|={err_T:.2e}"),
    ("欧氏保持距离", err_E < 1e-9, f"max|Δd|={err_E:.2e}"),
    ("相似按统一比例缩放", abs(sim_spread) < 1e-9 and abs(sim_ratio - 0.72) < 1e-9, f"比例={sim_ratio:.4f} 波动={sim_spread:.2e}"),
    ("仿射保持平行", par_after_A < 1e-9, f"平行度={par_after_A:.2e}"),
    ("仿射改变角度", angle_delta > 2.0, f"Δ角={angle_delta:.2f}°"),
    ("投影保持共线", collinear, f"叉积={collinear_err:.2e}"),
    ("投影保持交比", abs(cr_after - cr_before) < 1e-6, f"交比={cr_before:.4f}→{cr_after:.4f}"),
    ("投影可破坏平行", par_after_H > 1e-3, f"平行度={par_after_H:.4f}"),
    ("参数计数", [m[3] for m in MODELS] == [2, 3, 4, 6, 8], "2/3/4/6/8"),
    ("三张图在盘", all((FIG / n).exists() for n in ("e074_transform_ladder.png", "e074_alignment_pipeline.png", "e074_invariants.png")), "3/3"),
]
lines = []
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    results.append(dict(name=name, ok=bool(ok), detail=detail))

readings = dict(
    translation_error=err_T, euclidean_error=err_E,
    similarity_scale=sim_ratio, similarity_spread=sim_spread,
    affine_parallel=par_after_A, affine_angle_before=ang_before, affine_angle_after=ang_after_A,
    affine_angle_delta=angle_delta, projective_collinear_err=collinear_err,
    projective_cross_ratio_before=cr_before, projective_cross_ratio_after=cr_after,
    projective_parallel_after=par_after_H,
    params=[m[3] for m in MODELS], n_pass=sum(1 for _, ok, _ in tests if ok), n_total=len(tests),
)
(OUT / "e074_alignment_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e074_alignment_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
print(f"[lab] E074 五种平面变换 开始 {now}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines:
    print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e074_alignment_readings.json")
print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {readings['n_pass']}")
print(f"[lab] 汇总 PASS {readings['n_pass']} / FAIL {readings['n_total']-readings['n_pass']} / 共 {readings['n_total']}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if readings["n_pass"] != readings["n_total"]:
    raise SystemExit(1)
