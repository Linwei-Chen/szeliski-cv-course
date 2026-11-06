#!/usr/bin/env python3
"""E101 SFM 总览：管线/章节地图/视差-三角化实验（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d", "#6b7280"
RNG = np.random.default_rng(11)


def box(ax, x, y, w, h, fc, ec, title, sub="", fs=15, subfs=14.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(x + w / 2, y + h * 0.60, title, ha="center", va="center", fontsize=fs, color=INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=subfs, color=GRAY)


def arrow(ax, x1, y1, x2, y2, color=GRAY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=1.8, shrinkA=2, shrinkB=2))


# ── 合成几何实验：视差与三角化 ─────────────────────────────
N = 200
X = RNG.uniform(-1.5, 1.5, N)
Z = RNG.uniform(2.0, 8.0, N)
def parallax(b):
    return b / Z
def tri_error(b, noise=0.002):
    x1 = X / Z + RNG.normal(0, noise, N)
    x2 = (X - b) / Z + RNG.normal(0, noise, N)
    d = x1 - x2
    d = np.where(np.abs(d) < 1e-9, np.nan, d)
    Zest = b / d
    rel = np.abs(Zest - Z) / Z
    return float(np.nanmedian(rel))
par_near = float(np.mean(parallax(1.0)[Z < 3.0]))
par_far = float(np.mean(parallax(1.0)[Z > 7.0]))
base_small = 0.5; base_large = 2.0
err_small = tri_error(base_small)
err_large = tri_error(base_large)
err_noisy = tri_error(1.0, noise=0.01)
baselines = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
errs = np.array([tri_error(b) for b in baselines])
zero_ok = np.isnan(tri_error(0.0, noise=0.0))

PIPE = ["输入照片", "特征匹配", "双视图几何", "三角化/位姿", "束调整", "稀疏点云"]
CHAPTER = [
    ("11.1", "几何内参", "p.685–692", "选读：不单独展开"),
    ("11.2", "位姿与三角化", "p.693–702", "第 104 集"),
    ("11.3", "双视图 SFM", "p.703–714", "第 102–103 集"),
    ("11.4", "多帧 SFM / BA", "p.715–733", "第 105–107 集"),
    ("11.5", "SLAM", "p.734–739", "第 108–109 集"),
]

checks = [
    ("视差随深度反比", par_near > par_far * 1.5, f"近 {par_near:.3f} vs 远 {par_far:.3f}"),
    ("基线越大视差越大", np.all(np.diff([float(np.mean(parallax(b))) for b in baselines]) > 0), "b=0.25→4.0 单调增"),
    ("基线越大三角化越准", err_large < err_small * 0.6, f"{err_large:.4f} vs {err_small:.4f}"),
    ("噪声越大误差越大", err_noisy > err_large, f"{err_noisy:.4f} vs {err_large:.4f}"),
    ("零基线不可三角化", bool(zero_ok), "视差=0，深度无定义"),
    ("管线六步齐全", len(PIPE) == 6, "照片→匹配→几何→三角化→BA→点云"),
    ("章节五节齐全", len(CHAPTER) == 5, "11.1–11.5"),
    ("三图在盘", all((FIG / f"e101_{n}.png").exists() for n in ("pipeline", "map", "parallax")), "pipeline/map/parallax"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {"n_pass": n_pass, "n_total": len(checks), "n_points": N,
            "par_near": par_near, "par_near_3": round(par_near, 3),
            "par_far": par_far, "par_far_3": round(par_far, 3),
            "err_small": err_small, "err_small_4": round(err_small, 4),
            "err_large": err_large, "err_large_4": round(err_large, 4),
            "err_noisy": err_noisy, "err_noisy_4": round(err_noisy, 4),
            "base_small": base_small, "base_large": base_large,
            "zero_ok": bool(zero_ok), "zero_parallax": 0, "base_min": 0.25, "base_max": 4.0, "pipeline_steps": len(PIPE), "chapter_sections": len(CHAPTER), "chapter_min": 11.1, "chapter_max": 11.5}

# ── 图 1：SFM 管线 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.2, 4.8))
ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.55, "SFM 管线：从一堆照片到相机位姿 + 稀疏三维点", fontsize=17, color=INK, weight="bold")
for i, name in enumerate(PIPE):
    x = 0.25 + i * 1.78
    col = BLUE if i < 2 else (GREEN if i < 4 else PURPLE)
    box(ax, x, 2.45, 1.55, 0.95, "#eef2f7", col, name, f"{i+1}", fs=14.5, subfs=15)
    if i < len(PIPE) - 1:
        arrow(ax, x + 1.58, 2.92, x + 1.73, 2.92)
arrow(ax, 9.2, 2.40, 5.4, 1.45, ACCENT)
ax.text(7.2, 1.62, "重投影误差回到前面：对齐/位姿/三维点一起调", fontsize=14.5, color=ACCENT)
box(ax, 3.4, 0.55, 4.4, 0.65, "#f7e7e2", ACCENT, "失败信号：视差太小 / 匹配错 / 退化运动", "", fs=14.5)
fig.tight_layout(); fig.savefig(FIG / "e101_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# ── 图 2：章节地图 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.2, 6.2))
ax.set_xlim(0, 11.2); ax.set_ylim(-0.15, 6.2); ax.axis("off")
ax.text(0.2, 5.95, "第 11 章地图：本课程主线与选读边界", fontsize=17, color=INK, weight="bold")
for i, (sec, name, pages, eps) in enumerate(CHAPTER):
    y = 4.75 - i * 1.05
    ax.add_patch(FancyBboxPatch((0.35, y), 1.1, 0.78, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#f3ece0", edgecolor="#c9b89a", lw=1.2))
    ax.text(0.9, y + 0.39, sec, ha="center", va="center", fontsize=15.5, color=ACCENT, weight="bold")
    ax.text(1.7, y + 0.50, name, fontsize=15.5, color=INK, weight="bold")
    ax.text(1.7, y + 0.18, pages, fontsize=14.5, color=GRAY)
    ax.text(7.4, y + 0.39, eps, fontsize=14.5, color=GREEN if "E" in eps else GRAY)
ax.text(0.35, 0.02, "11.1 内参/消失点/单视图度量：本课程只在总览定位，不单独展开；11.6/11.7 阅读与习题选读。",
        fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e101_map.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# ── 图 3：视差与三角化 ────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
ax1.set_xlim(-0.2, 4.2); ax1.set_ylim(-0.3, 3.4); ax1.axis("off")
ax1.plot([0.2, 0.2], [0.7, 0.7], "o", color=BLUE, ms=12)
ax1.plot([2.6, 2.6], [0.7, 0.7], "o", color=BLUE, ms=12)
ax1.plot([0.2, 2.6], [0.7, 0.7], "-", color=INK, lw=2)
ax1.text(1.4, 0.42, "基线 b", ha="center", fontsize=14.5, color=INK)
ax1.plot([1.5], [2.5], "o", color=ACCENT, ms=10)
ax1.plot([0.2, 1.5], [0.7, 2.5], "--", color=GRAY, lw=1.4)
ax1.plot([2.6, 1.5], [0.7, 2.5], "--", color=GRAY, lw=1.4)
ax1.text(1.55, 2.62, "3D 点 P", fontsize=14.5, color=ACCENT)
ax1.text(0.0, 0.2, "左相机", fontsize=14.5, color=BLUE)
ax1.text(2.3, 0.2, "右相机", fontsize=14.5, color=BLUE)
ax1.text(0.35, 3.05, "视差 = 同一点在两幅图里的位置差", fontsize=15, color=INK, weight="bold")
ax2.plot(baselines, errs, "o-", color=ACCENT, lw=2, ms=5)
ax2.set_xlabel("基线 b", fontsize=14.5); ax2.set_ylabel("中位相对误差", fontsize=14.5)
ax2.set_title("基线越大，三角化越稳", fontsize=15, color=INK)
ax2.grid(alpha=0.25); ax2.tick_params(labelsize=13.5)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e101_parallax.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e101_{n}.png").exists() for n in ("pipeline", "map", "parallax")), "pipeline/map/parallax")
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings["n_pass"] = n_pass
print(f"[lab] E101 SFM overview 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e101_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e101_sfm_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e101_sfm_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e101_sfm_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
