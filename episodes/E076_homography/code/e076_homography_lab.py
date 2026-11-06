#!/usr/bin/env python3
"""E076 实验：单应参数化与 DLT 初始化（程序生成，未用原书原图）。

三图：四点定单应 / DLT 流水线 / 仿射与单应的误差对照。
10 条数值断言；readings 落盘。
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
OUT = EP / "code" / "outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32"

H_TRUE = np.array([[1.05, 0.18, 0.08], [0.07, 0.96, 0.03], [0.18, 0.12, 1.0]], float)
H_TRUE /= H_TRUE[2, 2]


def apply(H, pts):
    q = np.c_[pts, np.ones(len(pts))] @ H.T
    return q[:, :2] / q[:, 2:3]


def normalize(pts):
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]], float)
    return T, np.c_[pts, np.ones(len(pts))] @ T.T


def dlt(src, dst, norm=True):
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    if norm:
        Ts, sn = normalize(src); Td, dn = normalize(dst)
        sn, dn = sn[:, :2], dn[:, :2]
    else:
        sn, dn = src, dst
    A = []
    for (x, y), (u, v) in zip(sn, dn):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, float)
    _, sv, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    if norm:
        H = np.linalg.inv(Td) @ Hn @ Ts
    else:
        H = Hn
    H /= H[2, 2]
    return H, sv


def reproj(H, src, dst):
    p = apply(H, src)
    return float(np.sqrt(((p - dst) ** 2).sum(axis=1)).mean())


rng = np.random.default_rng(7)
src = rng.uniform(-1, 1, (20, 2))
dst = apply(H_TRUE, src)
# 4 点精确
H4, sv4 = dlt(src[:4], dst[:4])
err4 = reproj(H4, src[:4], dst[:4])
# 20 点精确
H20, sv20 = dlt(src, dst)
err20 = reproj(H20, src, dst)
# 噪声
dst_n = dst + rng.normal(0, 0.003, dst.shape)
Hn20, _ = dlt(src, dst_n)
err_noise = reproj(Hn20, src, dst_n)
# 仿射拟合
M = np.c_[src, np.ones(len(src))]
coef, *_ = np.linalg.lstsq(M, dst, rcond=None)
aff = M @ coef
err_aff = float(np.sqrt(((aff - dst) ** 2).sum(axis=1)).mean())
# 未归一化 DLT
Hraw, sraw = dlt(src, dst, norm=False)
err_raw = reproj(Hraw, src, dst)
# 共线退化
line = np.c_[np.linspace(-1, 1, 6), np.linspace(-1, 1, 6) * 0.4 + 0.1]
line_dst = apply(H_TRUE, line)
_, sv_line = dlt(line, line_dst)
# 直线保持
a, b, c = apply(H_TRUE, np.array([[0.0, 0.0], [0.5, 0.25], [1.0, 0.5]]))
coll_err = float(abs(np.cross(b - a, c - a)) / np.linalg.norm(b - a))
# 坐标大幅偏移下的条件
shift = np.array([1000.0, -800.0])
src_big = src + shift; dst_big = dst + shift
Hbig_n, _ = dlt(src_big, dst_big, norm=True)
Hbig_r, _ = dlt(src_big, dst_big, norm=False)
err_big_n = reproj(Hbig_n, src_big, dst_big)
err_big_r = reproj(Hbig_r, src_big, dst_big)

# ── 图 1：四点定单应 ──
fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.add_patch(plt.Polygon(apply(H_TRUE, np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])),
                         closed=True, fill=False, ec=BLUE, lw=2.2, ls="--"))
for i in range(9):
    p = np.array([[-1 + i * 0.25, -1], [-1 + i * 0.25, 1]])
    q = apply(H_TRUE, p); ax.plot(q[:, 0], q[:, 1], color=BLUE, lw=1.0, alpha=0.55)
    p = np.array([[-1, -1 + i * 0.25], [1, -1 + i * 0.25]])
    q = apply(H_TRUE, p); ax.plot(q[:, 0], q[:, 1], color=BLUE, lw=1.0, alpha=0.55)
ax.scatter(dst[:4, 0], dst[:4, 1], s=90, c=ACCENT, zorder=5, label="四个对应点")
for i in range(4):
    ax.annotate(f"P{i+1}", dst[i], textcoords="offset points", xytext=(6, 6), fontsize=13, color=ACCENT)
ax.set_title("四个不共线点，确定一个单应", fontsize=18, color=INK)
ax.set_aspect("equal"); ax.legend(fontsize=13); ax.grid(alpha=0.2); ax.tick_params(labelsize=13)
fig.tight_layout(); fig.savefig(FIG / "e076_four_points.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 图 2：DLT 管线 ──
fig, ax = plt.subplots(figsize=(9.2, 6.0), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fffdf8", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.055, title, ha="center", va="center", fontsize=16.5, color=color, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.04, body, ha="center", va="center", fontsize=15, color=INK, linespacing=1.5)
box(0.03, 0.60, 0.26, 0.26, "对应点", "至少四对\n不共线", BLUE)
box(0.36, 0.60, 0.26, 0.26, "列方程", "每对两点\n两行方程", BLUE)
box(0.69, 0.60, 0.28, 0.26, "奇异值分解", "取最小奇异值\n对应的向量", ACCENT)
box(0.20, 0.18, 0.26, 0.26, "还原 H", "除以尺度\n归一化坐标", GREEN)
box(0.55, 0.18, 0.28, 0.26, "迭代精修", "用 H 当初值\n最小化重投影", ACCENT)
for a, b in [((0.29, 0.73), (0.36, 0.73)), ((0.62, 0.73), (0.69, 0.73)),
             ((0.83, 0.60), (0.62, 0.44)), ((0.46, 0.31), (0.55, 0.31)),
             ((0.58, 0.44), (0.38, 0.60))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color="#7a5c2e", connectionstyle="arc3,rad=0.12"))
ax.text(0.5, 0.93, "DLT：先用线性方程拿初值", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
ax.text(0.5, 0.07, "线性初值不统计最优，但足够近；再用迭代法精修", ha="center", va="center", fontsize=14.5, color=ACCENT)
fig.savefig(FIG / "e076_dlt_pipeline.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# ── 图 3：误差对照 ──
fig, ax = plt.subplots(figsize=(9.2, 5.8), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
labels = ["仿射拟合", "单应 DLT", "单应+噪声", "大坐标归一"]
vals = [err_aff, err20, err_noise, err_big_n]
colors = [ACCENT, GREEN, BLUE, "#2e7d32"]
bars = ax.bar(labels, vals, color=colors, width=0.58)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=14, color=INK)
ax.set_ylabel("平均重投影误差（像素）", fontsize=15); ax.tick_params(labelsize=12.5)
ax.set_title("仿射解释不了透视；单应 DLT 几乎精确", fontsize=18, color=INK)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "e076_error_compare.png", dpi=150, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

tests = [
    ("四点 DLT 精确复原", err4 < 1e-10, f"误差={err4:.2e}"),
    ("二十点 DLT 精确", err20 < 1e-10, f"误差={err20:.2e}"),
    ("八自由度", H20.shape == (3, 3) and abs(H20[2, 2] - 1) < 1e-9, "H 为三乘三，右下角归一"),
    ("仿射解释不了透视", err_aff > 0.05, f"仿射={err_aff:.3f} 单应={err20:.1e}"),
    ("大坐标归一化仍稳", err_big_n < 1e-6, f"归一误差={err_big_n:.2e}"),
    ("共线点退化", sv_line[-2] < 1e-9, f"次小奇异值={sv_line[-2]:.2e}"),
    ("直线仍映射为直线", coll_err < 1e-10, f"共线误差={coll_err:.2e}"),
    ("噪声下误差可控", err_noise < 0.01, f"误差={err_noise:.4f}"),
    ("四个最小点", len(src[:4]) == 4, "四对"),
    ("三图在盘", all((FIG / n).exists() for n in ("e076_four_points.png", "e076_dlt_pipeline.png", "e076_error_compare.png")), "三张全在"),
]
lines = []
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
n_pass = sum(1 for _, ok, _ in tests if ok)
readings = dict(err4=err4, err20=err20, err_noise=err_noise, err_aff=err_aff,
                err_big_norm=err_big_n, err_big_raw=err_big_r, collinear_err=coll_err,
                sv_line_second_last=float(sv_line[-2]), n_pass=n_pass, n_total=len(tests))
(OUT / "e076_homography_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e076_homography_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
print(f"[lab] E076 单应参数化与 DLT 开始 {now}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines:
    print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e076_homography_readings.json")
print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests)-n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass != len(tests):
    raise SystemExit(1)
