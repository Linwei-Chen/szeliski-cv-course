#!/usr/bin/env python3
"""E116 深度传感器数值沙盘：结构光三角测量、ToF 相位、LiDAR 脉冲。"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
C = 299_792_458.0

# 结构光：Z = f*B/d，深度误差 dZ = Z^2/(f*B) * dd
F, B, DD = 600.0, 0.1, 0.1
Z_sl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
d_sl = F * B / Z_sl
dz_sl = Z_sl ** 2 * DD / (F * B)
sl_d_at_2m = float(F * B / 2.0)
sl_dz_ratio = float(dz_sl[3] / dz_sl[1])

# ToF：单频 30 MHz 的模糊距离；双频 30/40 MHz 的拍频距离
f1, f2 = 30e6, 40e6
R1 = C / (2 * f1)
Rbeat = C / (2 * abs(f2 - f1))
Z_true = 7.0
Z_alias = float(Z_true % R1)
# 双频相位网格搜索
zs = np.linspace(0, Rbeat, 300001)
ph1 = (2 * zs * f1 / C) % 1.0
ph2 = (2 * zs * f2 / C) % 1.0
t1 = (2 * Z_true * f1 / C) % 1.0
t2 = (2 * Z_true * f2 / C) % 1.0
cost = np.abs(np.angle(np.exp(2j * np.pi * (ph1 - t1)))) + np.abs(np.angle(np.exp(2j * np.pi * (ph2 - t2))))
z_dual = float(zs[np.argmin(cost)])
dual_err = abs(z_dual - Z_true)

# LiDAR：往返时间与稀疏采样
t10 = 2 * 10.0 / C * 1e9
t100 = 2 * 100.0 / C * 1e9
n_lidar = 64 * 1024
coverage = n_lidar / (1920 * 1080)

# 镜子：真实镜面 2 米，虚拟像在镜后 2 米，结构光测到 4 米
mirror_true, mirror_est = 2.0, 4.0
mirror_err = abs(mirror_est - mirror_true)

# 图1 三种原理
fig, ax = plt.subplots(figsize=(10.0, 7.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.text(0.2, 9.55, "三种主动测距：三角、相位、计时", fontsize=21, color=INK, weight="bold")
# 行1 结构光
ax.add_patch(Rectangle((0.4, 7.3), 1.2, 0.7, facecolor="#e6ebf2", edgecolor=BLUE, lw=2)); ax.text(1.0, 7.65, "投影器", ha="center", fontsize=18, color=INK)
ax.add_patch(Rectangle((2.6, 7.3), 1.2, 0.7, facecolor="#e6ebf2", edgecolor=BLUE, lw=2)); ax.text(3.2, 7.65, "相机", ha="center", fontsize=18, color=INK)
ax.plot([1.0, 4.1], [8.0, 8.7], color=ACCENT, lw=2, ls="--"); ax.plot([3.2, 4.1], [8.0, 8.2], color=GREEN, lw=2)
ax.annotate("", xy=(2.6, 7.9), xytext=(1.0, 7.9), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.5))
ax.text(1.8, 8.05, "基线 B", fontsize=18, color=GRAY)
ax.text(4.6, 8.3, "结构光：三角测量", fontsize=19, color=BLUE, weight="bold")
ax.text(4.6, 7.75, "测图案的横向位移 d", fontsize=18, color=INK)
ax.text(4.6, 7.25, "Z = f·B / d", fontsize=18, color=INK)
ax.plot([0.3, 9.7], [6.55, 6.55], color=GRAY, lw=1, alpha=0.4)
# 行2 ToF
x = np.linspace(0.4, 4.2, 300)
ax.plot(x, 5.1 + 0.55 * np.sin(2 * np.pi * x / 1.2), color=BLUE, lw=2)
ax.plot(x, 3.9 + 0.55 * np.sin(2 * np.pi * (x - 0.4) / 1.2), color=ACCENT, lw=2)
ax.text(4.6, 5.45, "ToF（相位式）：相位差", fontsize=19, color=GREEN, weight="bold")
ax.text(4.6, 4.9, "测发射光与接收光的相位", fontsize=18, color=INK)
ax.text(4.6, 4.4, "Z = c·Δφ / (4πf)", fontsize=18, color=INK)
ax.plot([0.3, 9.7], [3.35, 3.35], color=GRAY, lw=1, alpha=0.4)
# 行3 LiDAR
ax.add_patch(Rectangle((0.4, 1.2), 1.1, 0.7, facecolor="#eef7ee", edgecolor=GREEN, lw=2)); ax.text(0.95, 1.55, "激光", ha="center", fontsize=18, color=INK)
for yy in np.linspace(0.7, 2.5, 5):
    ax.plot([1.5, 4.0], [1.55, yy], color=GREEN, lw=1.5, alpha=0.8)
ax.text(4.6, 2.15, "LiDAR：计时", fontsize=19, color=ACCENT, weight="bold")
ax.text(4.6, 1.6, "测脉冲往返时间 t", fontsize=18, color=INK)
ax.text(4.6, 1.1, "t = 2Z / c", fontsize=18, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e116_triad.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 误差随距离
fig, ax = plt.subplots(figsize=(10.2, 5.2))
zz = np.linspace(0.5, 10, 400)
ax.plot(zz, zz ** 2 * DD / (F * B), color=BLUE, lw=2.5, label="结构光（∝Z²）")
tof_err = np.where(zz <= R1, 0.01, np.abs(zz - np.mod(zz, R1)))
ax.plot(zz, tof_err, color=GREEN, lw=2.0, label="ToF 单频（每 5 米折回一次）")
ax.axhline(0.005, color=ACCENT, lw=2.0, ls="--", label="LiDAR（计时稳定）")
ax.axvline(R1, color=GRAY, lw=1.2, ls=":")
ax.text(R1 + 0.15, 0.03, f"单频模糊距离 {R1:.1f} 米（示意图呈锯齿）", fontsize=13, color=GRAY)
ax.set_xlabel("真实距离 / 米", fontsize=15); ax.set_ylabel("深度误差 / 米（对数）", fontsize=15)
ax.set_yscale("log"); ax.set_ylim(1e-3, 10); ax.tick_params(labelsize=13)
ax.grid(alpha=0.2, which="both"); ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.set_title("距离越远，三角测量越吃亏；计时/相位不吃这个亏", fontsize=17, color=INK)
fig.tight_layout(); fig.savefig(FIG / "e116_range.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 定性取舍
fig, ax = plt.subplots(figsize=(10.2, 5.0)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
attrs = ["密度", "量程", "精度", "不怕弱纹理", "抗反光"]
scores = {"结构光": [9, 4, 8, 7, 3], "ToF": [8, 7, 6, 9, 5], "LiDAR": [3, 10, 9, 10, 7]}
colors = {"结构光": BLUE, "ToF": GREEN, "LiDAR": ACCENT}
ax.text(0.2, 5.6, "三种深度传感器的定性取舍（定性示意，不是厂商标称）", fontsize=17, color=INK, weight="bold")
for j, a in enumerate(attrs):
    ax.text(0.4, 4.7 - j * 0.85, a, fontsize=14, color=INK)
for i, (name, vals) in enumerate(scores.items()):
    x0 = 2.2 + i * 2.55
    ax.text(x0 + 0.9, 4.95, name, ha="center", fontsize=15, color=colors[name], weight="bold")
    for j, v in enumerate(vals):
        ax.add_patch(Rectangle((x0, 4.45 - j * 0.85), 1.8 * v / 10, 0.5, facecolor=colors[name], alpha=0.75, edgecolor="none"))
ax.text(0.3, 0.25, "结构光：近距密；ToF：中距稳；LiDAR：远距稀而准", fontsize=14, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e116_tradeoff.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

assertions = [
    ("结构光 2 米视差 30 像素", abs(sl_d_at_2m - 30.0) < 1e-9, f"{sl_d_at_2m:.0f} 像素"),
    ("结构光误差随 Z² 增长", abs(sl_dz_ratio - 4.0) < 1e-9, f"{sl_dz_ratio:.1f} 倍"),
    ("结构光 5 米误差超过 0.03 米", dz_sl[4] > 0.03, f"{dz_sl[4]:.3f} 米"),
    ("ToF 单频模糊距离约 5 米", abs(R1 - 5.0) < 0.01, f"{R1:.2f} 米"),
    ("ToF 单频 7 米折回约 2 米", abs(Z_alias - 2.0) < 0.02 and abs(Z_alias - Z_true) > 4.0, f"{Z_alias:.2f} 米"),
    ("ToF 双频解出 7 米", dual_err < 0.05, f"{z_dual:.2f} 米"),
    ("LiDAR 10 米往返 66.7 纳秒", abs(t10 - 66.7) < 0.1, f"{t10:.1f} 纳秒"),
    ("LiDAR 100 米往返 667.3 纳秒", abs(t100 - 667.3) < 0.2, f"{t100:.1f} 纳秒"),
    ("LiDAR 点云覆盖率低于 5%", coverage < 0.05, f"{coverage*100:.1f}%"),
    ("镜子产生 2 米虚像误差", mirror_err > 1.0, f"{mirror_err:.1f} 米"),
]
n_pass = sum(bool(ok) for _, ok, _ in assertions)
readings = {"n_pass": n_pass, "n_total": len(assertions), "F": F, "B": B, "DD": DD,
            "d_2m": sl_d_at_2m, "dz_5m": float(dz_sl[4]), "dz_5m_3": round(float(dz_sl[4]), 3),
            "dz_ratio": sl_dz_ratio, "R1": float(R1), "R1_2": round(float(R1), 2),
            "Rbeat": float(Rbeat), "Z_alias": Z_alias, "Z_alias_2": round(Z_alias, 2),
            "z_dual": z_dual, "z_dual_2": round(z_dual, 2), "dual_err": dual_err,
            "t10": float(t10), "t10_1": round(float(t10), 1), "t100": float(t100), "t100_1": round(float(t100), 1),
            "n_lidar": n_lidar, "coverage": float(coverage), "coverage_pct": round(float(coverage)*100, 1),
            "mirror_true": mirror_true, "mirror_est": mirror_est, "mirror_err": mirror_err}
print(f"[lab] E116 深度传感器开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in assertions:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] 结构光 d(Z)=fB/Z｜2m {sl_d_at_2m:.0f}px｜5m 误差 {dz_sl[4]:.3f}m｜4m/2m 误差比 {sl_dz_ratio:.1f}")
print(f"[lab] ToF 30MHz 模糊 {R1:.2f}m｜7m 折回 {Z_alias:.2f}m｜双频 {z_dual:.2f}m（误差 {dual_err:.3f}m）｜拍频量程 {Rbeat:.2f}m")
print(f"[lab] LiDAR 10m {t10:.1f}ns｜100m {t100:.1f}ns｜{n_lidar} 点｜覆盖率 {coverage*100:.1f}%")
print(f"[lab] 镜子：真实 {mirror_true:.1f}m，虚像 {mirror_est:.1f}m，误差 {mirror_err:.1f}m")
for p in sorted(FIG.glob("e116_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e116_sensors_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e116_sensors_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e116_sensors_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(assertions)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(assertions)-n_pass} / 共 {len(assertions)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(assertions): raise SystemExit(1)
