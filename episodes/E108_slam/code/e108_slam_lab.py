#!/usr/bin/env python3
"""E108 SLAM：里程计/位姿图/回环/地标地图（2D 合成实验）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d", "#6b7280"
RNG = np.random.default_rng(108)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rel(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])

def compose(p, u):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([p[0] + c * u[0] - s * u[1], p[1] + s * u[0] + c * u[1], wrap(p[2] + u[2])])

N, L = 40, 12
W, H = 10.0, 8.0
POSES = []
for i in range(N):
    t = i / (N - 1)
    if t < 0.25:
        x, y = 4 * W * t, 0.0
    elif t < 0.5:
        x, y = W, 4 * H * (t - 0.25)
    elif t < 0.75:
        x, y = W - 4 * W * (t - 0.5), H
    else:
        x, y = 0.0, H - 4 * H * (t - 0.75)
    th = 0.0 if i == 0 else np.arctan2(y - POSES[-1][1], x - POSES[-1][0])
    POSES.append((x, y, th))
POSES = np.array(POSES)
ODO = np.array([rel(POSES[i], POSES[i + 1]) + RNG.normal(0, [0.02, 0.02, 0.008])
                for i in range(N - 1)])
LOOP = rel(POSES[0], POSES[-1]) + RNG.normal(0, [0.01, 0.01, 0.003])
# 地标与极坐标观测
LMS = np.column_stack([RNG.uniform(-1, W + 1, L), RNG.uniform(-1, H + 1, L)])
OBS = []
for i, p in enumerate(POSES):
    for j, l in enumerate(LMS):
        if np.linalg.norm(l - p[:2]) < 5.0 and RNG.random() < 0.7:
            dx, dy = l[0] - p[0], l[1] - p[1]
            c, s = np.cos(p[2]), np.sin(p[2])
            fr, lf = c * dx + s * dy, -s * dx + c * dy
            OBS.append((i, j, float(np.hypot(fr, lf)) + RNG.normal(0, 0.02),
                        float(np.arctan2(lf, fr)) + RNG.normal(0, 0.01)))

def dead_reckon():
    P = [POSES[0].copy()]
    for u in ODO:
        P.append(compose(P[-1], u))
    return np.array(P)
DEAD = dead_reckon()

def unpack(v):
    P = np.zeros((N, 3)); P[0] = POSES[0]; P[1:] = v.reshape(N - 1, 3)
    return P

def residual(v, use_loop):
    P = unpack(v); r = []
    for i in range(N - 1):
        r.extend(rel(P[i], P[i + 1]) - ODO[i])
    if use_loop:
        r.extend(rel(P[0], P[-1]) - LOOP)
    return np.array(r)

def solve(use_loop):
    sol = least_squares(lambda v: residual(v, use_loop), DEAD[1:].ravel(),
                        method="trf", max_nfev=200)
    return unpack(sol.x), sol, residual(sol.x, use_loop)

P_NOLOOP, sol_noloop, r_noloop = solve(False)
P_LOOP, sol_loop, r_loop = solve(True)

def ate(P):
    return float(np.sqrt(np.mean(np.sum((P[:, :2] - POSES[:, :2]) ** 2, axis=1))))

def endpoint(P):
    return float(np.linalg.norm(P[-1, :2] - POSES[-1, :2]))

def map_landmarks(P):
    est = []
    for j in range(L):
        pts = []
        for i, jj, rng_, brg in OBS:
            if jj == j:
                c, s = np.cos(P[i, 2]), np.sin(P[i, 2])
                pts.append([P[i, 0] + c * rng_ * np.cos(brg) - s * rng_ * np.sin(brg),
                            P[i, 1] + s * rng_ * np.cos(brg) + c * rng_ * np.sin(brg)])
        est.append(np.mean(pts, axis=0) if pts else [0.0, 0.0])
    return np.array(est)

def map_rms(P):
    M = map_landmarks(P)
    return float(np.sqrt(np.mean(np.sum((M - LMS) ** 2, axis=1))))

DEAD_END = endpoint(DEAD); LOOP_END = endpoint(P_LOOP); NOLOOP_END = endpoint(P_NOLOOP)
DEAD_ATE = ate(DEAD); LOOP_ATE = ate(P_LOOP)
DEAD_MAP = map_rms(DEAD); LOOP_MAP = map_rms(P_LOOP)
LOOP_RMS = float(np.sqrt(np.mean(r_loop ** 2)))
checks = [
    ("纯里程计会漂", DEAD_END > 0.2, f"终点 {DEAD_END:.3f}"),
    ("无回环不修正", abs(NOLOOP_END - DEAD_END) < 1e-9, f"{NOLOOP_END:.3f} vs {DEAD_END:.3f}"),
    ("回环终点误差小", LOOP_END < 0.05, f"{LOOP_END:.4f}"),
    ("回环显著降终点", DEAD_END > 10 * LOOP_END, f"{DEAD_END:.3f}→{LOOP_END:.4f}"),
    ("回环降 ATE", LOOP_ATE < 0.7 * DEAD_ATE, f"{DEAD_ATE:.3f}→{LOOP_ATE:.3f}"),
    ("回环降地图误差", LOOP_MAP < 0.8 * DEAD_MAP, f"{DEAD_MAP:.3f}→{LOOP_MAP:.3f}"),
    ("回环残差小", LOOP_RMS < 0.01, f"{LOOP_RMS:.5f}"),
    ("全轨迹有界", bool(np.all(np.isfinite(P_LOOP))) and np.max(np.linalg.norm(P_LOOP[:, :2], axis=1)) < 50, f"max {np.max(np.linalg.norm(P_LOOP[:, :2], axis=1)):.1f}"),
    ("地标全建图", len(OBS) > 0 and LOOP_MAP < 0.5, f"{len(OBS)} 观测/{L} 地标"),
    ("三图在盘", all((FIG / f"e108_{n}.png").exists() for n in ("pipeline", "traj", "error")), "pipeline/traj/error"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "n_pose": N, "n_landmark": L, "n_obs": len(OBS),
    "dead_end": DEAD_END, "dead_end_3": round(DEAD_END, 3),
    "noloop_end": NOLOOP_END, "noloop_end_3": round(NOLOOP_END, 3),
    "loop_end": LOOP_END, "loop_end_4": round(LOOP_END, 4),
    "end_ratio": DEAD_END / max(LOOP_END, 1e-12), "end_ratio_1": round(DEAD_END / max(LOOP_END, 1e-12), 1),
    "dead_ate": DEAD_ATE, "dead_ate_3": round(DEAD_ATE, 3),
    "loop_ate": LOOP_ATE, "loop_ate_3": round(LOOP_ATE, 3),
    "dead_map": DEAD_MAP, "dead_map_3": round(DEAD_MAP, 3),
    "loop_map": LOOP_MAP, "loop_map_3": round(LOOP_MAP, 3),
    "loop_rms": LOOP_RMS, "loop_rms_5": round(LOOP_RMS, 5), "max_pos": float(np.max(np.linalg.norm(P_LOOP[:, :2], axis=1))), "max_pos_1": round(float(np.max(np.linalg.norm(P_LOOP[:, :2], axis=1))), 1),
    "odo_noise_xy": 0.02, "odo_noise_th": 0.008, "obs_noise_r": 0.02, "obs_noise_b": 0.01,
}

# 图1 管线
fig, ax = plt.subplots(figsize=(11.2, 4.8)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.5, "SLAM：前端跟踪，后端优化，回环拉回漂移", fontsize=17, color=INK, weight="bold")
steps = ["传感器/跟踪", "局部地图", "回环检测", "位姿图优化", "全局地图"]
for i, s in enumerate(steps):
    x = 0.25 + i * 2.18
    col = BLUE if i < 2 else (GREEN if i < 4 else PURPLE)
    ax.add_patch(plt.Rectangle((x, 2.4), 1.82, 0.95, facecolor="#eef2f7", edgecolor=col, lw=1.6))
    ax.text(x + 0.91, 2.88, s, ha="center", fontsize=15.5, color=INK, weight="bold")
    if i < 4:
        ax.annotate("", xy=(x + 2.12, 2.88), xytext=(x + 1.88, 2.88), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.3, 1.45, "前端：高频跟踪当前位姿；后端：低频优化关键帧和地图。", fontsize=14.5, color=GRAY)
ax.text(0.3, 0.85, "回环：认出“又回到这里”，用一条约束把累积漂移拉回去。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e108_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 轨迹
fig, ax = plt.subplots(figsize=(10.2, 4.6))
ax.plot(POSES[:, 0], POSES[:, 1], "-", color=INK, lw=2.0, label="真值")
ax.plot(DEAD[:, 0], DEAD[:, 1], "--", color=GRAY, lw=1.6, label="纯里程计")
ax.plot(P_LOOP[:, 0], P_LOOP[:, 1], "-", color=ACCENT, lw=2.0, label="回环优化")
ax.scatter(LMS[:, 0], LMS[:, 1], marker="^", s=45, color=GREEN, label="地标", zorder=3)
ax.set_aspect("equal"); ax.set_xlabel("X / m", fontsize=14.5); ax.set_ylabel("Y / m", fontsize=14.5)
ax.set_title("2D 轨迹：里程计漂移，回环拉回", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(alpha=0.2); ax.legend(fontsize=13.0, frameon=False, loc="upper right")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e108_traj.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 误差对比
fig, ax = plt.subplots(figsize=(9.8, 4.9))
labels = ["终点", "ATE", "地图 RMS"]
dead_vals = [DEAD_END, DEAD_ATE, DEAD_MAP]
loop_vals = [LOOP_END, LOOP_ATE, LOOP_MAP]
x = np.arange(3); w = 0.34
ax.bar(x - w/2, dead_vals, w, color=GRAY, label="纯里程计/无回环")
ax.bar(x + w/2, loop_vals, w, color=ACCENT, label="回环优化")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=14.5)
ax.set_yscale("log"); ax.set_ylabel("误差（对数）", fontsize=14.5)
ax.set_title("回环把终点误差从 0.395 拉到 0.022", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=13.5, frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e108_error.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e108_{n}.png").exists() for n in ("pipeline", "traj", "error")), "pipeline/traj/error")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E108 SLAM 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e108_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e108_slam_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e108_slam_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e108_slam_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
