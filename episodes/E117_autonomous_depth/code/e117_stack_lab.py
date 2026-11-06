#!/usr/bin/env python3
"""E117 自动驾驶感知栈数值沙盘：深度作为共享中间表示，误差如何传到下游。"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from math import erf, sqrt
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"

# 传感器模型：双目误差 ∝ Z²；LiDAR/雷达近似常数
SIGMA_L, SIGMA_R = 0.05, 0.10
def sigma_stereo(z): return 0.05 * (z / 10.0) ** 2   # 10m 5cm, 20m 20cm, 50m 1.25m
def fused_sigma(sigmas): return 1.0 / sqrt(sum(1.0 / s ** 2 for s in sigmas))
z20, z50 = 20.0, 50.0
s20, s50 = sigma_stereo(z20), sigma_stereo(z50)
fused50 = fused_sigma([s50, SIGMA_L, SIGMA_R])
fused50_camdrop = fused_sigma([SIGMA_L, SIGMA_R])
fused50_lidardrop = fused_sigma([s50, SIGMA_R])

# 自由空间误检：50m 处阈值为 1m（测到 <49m 即误判为障碍）
def phi(x): return 0.5 * (1 + erf(x / sqrt(2)))
fp_stereo = phi((-1.0) / s50)
fp_lidar = phi((-1.0) / SIGMA_L)

# 延迟预算：相机 30 + 双目 20 + 检测 20；LiDAR 50 + 检测 20；融合 10；规划 20
lat_cam, lat_lidar, lat_fuse, lat_plan = 70, 70, 10, 20
lat_total = lat_fuse + lat_plan + 70
speed = 15.0
travel_100ms = speed * 0.1
n_consumers = 3

# 图1 感知栈
fig, ax = plt.subplots(figsize=(11.0, 5.8)); ax.set_xlim(0, 11); ax.set_ylim(0, 5.8); ax.axis("off")
ax.text(0.2, 5.35, "自动驾驶感知栈：深度是共享中间表示，不是一个孤立层", fontsize=19, color=INK, weight="bold")
def box(x, y, w, h, text, fc, ec, fs=15):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)
box(0.2, 3.5, 1.7, 0.85, "相机", "#eef2f7", BLUE)
box(0.2, 2.4, 1.7, 0.85, "LiDAR", "#eef7ee", GREEN)
box(0.2, 1.3, 1.7, 0.85, "雷达", "#f8ece8", ACCENT)
box(2.5, 2.2, 2.1, 1.6, "深度\n（米制/相对）", "#fff3cd", "#b8860b", 16)
box(5.1, 4.0, 2.2, 0.8, "检测/跟踪", "#eef2f7", BLUE, 14)
box(5.1, 2.9, 2.2, 0.8, "自由空间", "#eef7ee", GREEN, 14)
box(5.1, 1.8, 2.2, 0.8, "定位/SLAM", "#f8ece8", ACCENT, 14)
box(8.0, 2.4, 2.5, 1.4, "规划/控制", "#e8eaf6", "#3949ab", 16)
for y in (3.9, 2.8, 1.7):
    ax.add_patch(FancyArrowPatch((1.9, y), (2.5, 3.0 if y == 2.8 else 3.0), arrowstyle="->", color=GRAY, lw=1.5, mutation_scale=12))
for y in (4.4, 3.3, 2.2):
    ax.add_patch(FancyArrowPatch((4.6, 3.0), (5.1, y), arrowstyle="->", color=GRAY, lw=1.5, mutation_scale=12))
for y in (4.4, 3.3, 2.2):
    ax.add_patch(FancyArrowPatch((7.3, y), (8.0, 3.1), arrowstyle="->", color=GRAY, lw=1.5, mutation_scale=12))
ax.text(2.1, 1.05, "深度不是“第几层”，而是多个模块共用的坐标系", fontsize=14, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e117_stack.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 误差随距离
fig, ax = plt.subplots(figsize=(10.2, 5.2))
zz = np.linspace(10, 100, 400)
ax.plot(zz, sigma_stereo(zz), color=BLUE, lw=2.5, label="双目")
ax.axhline(SIGMA_L, color=GREEN, lw=2.0, label="LiDAR")
ax.axhline(SIGMA_R, color=ACCENT, lw=2.0, ls="--", label="雷达")
ax.plot(zz, [fused_sigma([sigma_stereo(z), SIGMA_L, SIGMA_R]) for z in zz], color="#6b3fa0", lw=2.5, label="融合")
ax.axvline(20, color=GRAY, lw=1, ls=":"); ax.axvline(50, color=GRAY, lw=1, ls=":")
ax.text(20.8, 3.0, "20 米", fontsize=13, color=GRAY); ax.text(50.8, 3.0, "50 米", fontsize=13, color=GRAY)
ax.set_xlabel("距离 / 米", fontsize=15); ax.set_ylabel("深度标准差 / 米（对数）", fontsize=15)
ax.set_yscale("log"); ax.set_ylim(0.02, 10); ax.tick_params(labelsize=13)
ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_title("双目远距变差，LiDAR/雷达稳住；融合取长补短", fontsize=17, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e117_error.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 延迟与失效
fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.0))
segs = [("相机", 30, BLUE), ("双目", 20, GREEN), ("检测", 20, ACCENT), ("融合", 10, "#6b3fa0"), ("规划", 20, GRAY)]
left = 0
for name, val, col in segs:
    axes[0].barh(0, val, left=left, color=col, height=0.5, label=name)
    axes[0].text(left + val / 2, 0.0, str(val), ha="center", va="center", fontsize=12, color="white", weight="bold")
    left += val
axes[0].axvline(120, color=ACCENT, lw=2, ls="--")
axes[0].text(121, 0.32, "120ms 预算", fontsize=13, color=ACCENT)
axes[0].set_yticks([]); axes[0].set_xlim(0, 135)
axes[0].set_xlabel("延迟 / 毫秒", fontsize=14); axes[0].tick_params(labelsize=12)
axes[0].set_title("融合链路：30+20+20+10+20 = 100ms", fontsize=15, color=INK)
# 学生版右栏约 803.6px：原 11pt 图例折合约 11.7px，低于 12px 图内字号口径；
# 仅把图例字号提到 12pt（其余字号不变），保证学生版缩放下仍可读。
axes[0].legend(fontsize=12, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.42))
axes[0].grid(axis="x", alpha=0.2)
names = ["相机失效", "LiDAR 失效", "两者都失效"]
vals = [fused50_camdrop, fused50_lidardrop, SIGMA_R]
bars = axes[1].bar(names, vals, color=[BLUE, GREEN, ACCENT], width=0.55)
axes[1].set_ylabel("50 米深度标准差 / 米", fontsize=14); axes[1].tick_params(labelsize=12)
axes[1].set_title("单点失效：还有没有冗余？", fontsize=15, color=INK)
for b, v in zip(bars, vals):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=12, color=INK)
axes[1].set_ylim(0, max(vals) * 1.25); axes[1].grid(axis="y", alpha=0.2)
fig.tight_layout(); fig.savefig(FIG / "e117_latency.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

assertions = [
    ("深度被三个下游模块共享", n_consumers == 3, "检测/自由空间/定位"),
    ("双目 20 米误差小于 0.25 米", s20 < 0.25, f"{s20:.2f} 米"),
    ("双目 50 米误差大于 1 米", s50 > 1.0, f"{s50:.2f} 米"),
    ("LiDAR 50 米误差小于 0.1 米", SIGMA_L < 0.1, f"{SIGMA_L:.2f} 米"),
    ("融合 50 米误差小于 LiDAR", fused50 < SIGMA_L, f"{fused50:.3f} vs {SIGMA_L:.2f} 米"),
    ("双目自由空间误检高于 LiDAR", fp_stereo > 0.1 and fp_lidar < 0.01, f"{fp_stereo*100:.1f}% vs {fp_lidar*100:.1f}%"),
    ("融合链路 120ms 预算内", lat_total <= 120, f"{lat_total} 毫秒"),
    ("相机失效仍有 LiDAR+雷达", fused50_camdrop < 0.1, f"{fused50_camdrop:.3f} 米"),
    ("LiDAR 失效仍有相机+雷达", fused50_lidardrop < 0.2, f"{fused50_lidardrop:.3f} 米"),
    ("100ms 行进 1.5 米", abs(travel_100ms - 1.5) < 1e-9, f"{travel_100ms:.1f} 米"),
]
n_pass = sum(bool(ok) for _, ok, _ in assertions)
readings = {"n_pass": n_pass, "n_total": len(assertions), "n_consumers": n_consumers,
            "sigma_stereo_20": s20, "sigma_stereo_50": s50, "sigma_lidar": SIGMA_L, "sigma_radar": SIGMA_R,
            "fused_50": fused50, "fused_50_3": round(fused50, 3), "fused_camdrop": fused50_camdrop, "fused_lidardrop": fused50_lidardrop,
            "fp_stereo": fp_stereo, "fp_stereo_pct": round(fp_stereo*100, 1), "fp_lidar": fp_lidar,
            "lat_total": lat_total, "travel_100ms": travel_100ms, "speed": speed}
print(f"[lab] E117 感知栈开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in assertions:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] 双目 20m {s20:.2f}m｜50m {s50:.2f}m｜LiDAR {SIGMA_L:.2f}m｜雷达 {SIGMA_R:.2f}m｜融合50m {fused50:.3f}m")
print(f"[lab] 自由空间误检：双目 {fp_stereo*100:.1f}%｜LiDAR {fp_lidar*100:.1f}%｜延迟 {lat_total}ms｜100ms 行进 {travel_100ms:.1f}m")
print(f"[lab] 失效：相机掉线 {fused50_camdrop:.3f}m｜LiDAR 掉线 {fused50_lidardrop:.3f}m｜两者都掉 {SIGMA_R:.2f}m")
for p in sorted(FIG.glob("e117_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e117_stack_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e117_stack_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e117_stack_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(assertions)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(assertions)-n_pass} / 共 {len(assertions)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(assertions): raise SystemExit(1)
