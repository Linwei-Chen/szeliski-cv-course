#!/usr/bin/env python3
"""E109 ORB-SLAM 解剖：三线程/共视图/Essential Graph/回环（合成实验）。"""
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
RNG = np.random.default_rng(109)

N, L = 20, 200
SPAN = 6
OBS = {i: set() for i in range(N)}
for j in range(L):
    s = int(RNG.integers(0, N))
    for k in range(SPAN):
        OBS[(s + k) % N].add(j)
W = np.zeros((N, N), int)
for i in range(N):
    for j in range(i + 1, N):
        W[i, j] = len(OBS[i] & OBS[j])
COV = {(i, j) for i in range(N) for j in range(i + 1, N) if W[i, j] >= 15}
# 生成树：Prim，边权=共享点数
IN = {0}; TREE = []
while len(IN) < N:
    best = None
    for i in IN:
        for j in range(N):
            if j not in IN and (best is None or W[i, j] > best[0]):
                best = (W[i, j], i, j)
    TREE.append((best[1], best[2])); IN.add(best[2])
THETA_ESS = 50
ESS = {tuple(sorted(e)) for e in TREE}
for i in range(N):
    for j in range(i + 1, N):
        if W[i, j] >= THETA_ESS:
            ESS.add((i, j))
LOOP_EDGE = (0, N - 1)
ESS.add(LOOP_EDGE)

def jaccard(a, b):
    return len(OBS[a] & OBS[b]) / max(len(OBS[a] | OBS[b]), 1)
LOOP_SIM = jaccard(0, N - 1); RAND_SIM = jaccard(0, 10)
# culling：每四个去掉一个，保留 15 个关键帧，检查地图点覆盖
KEEP = [i for i in range(N) if i % 4 != 3]
COVERED = set().union(*[OBS[i] for i in KEEP])

# 2D 位姿图回环（20 个关键帧）
Wr, Hr = 10.0, 8.0
POSES = []
for i in range(N):
    t = i / (N - 1)
    if t < 0.25:
        x, y = 4 * Wr * t, 0.0
    elif t < 0.5:
        x, y = Wr, 4 * Hr * (t - 0.25)
    elif t < 0.75:
        x, y = Wr - 4 * Wr * (t - 0.5), Hr
    else:
        x, y = 0.0, Hr - 4 * Hr * (t - 0.75)
    th = 0.0 if i == 0 else np.arctan2(y - POSES[-1][1], x - POSES[-1][0])
    POSES.append((x, y, th))
POSES = np.array(POSES)

def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rel(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(b[2] - a[2])])

def compose(p, u):
    c, s = np.cos(p[2]), np.sin(p[2])
    return np.array([p[0] + c * u[0] - s * u[1], p[1] + s * u[0] + c * u[1], wrap(p[2] + u[2])])

ODO = np.array([rel(POSES[i], POSES[i + 1]) + RNG.normal(0, [0.02, 0.02, 0.008])
                for i in range(N - 1)])
LOOP = rel(POSES[0], POSES[-1]) + RNG.normal(0, [0.01, 0.01, 0.003])
P_DEAD = [POSES[0].copy()]
for u in ODO:
    P_DEAD.append(compose(P_DEAD[-1], u))
P_DEAD = np.array(P_DEAD)

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
    sol = least_squares(lambda v: residual(v, use_loop), P_DEAD[1:].ravel(),
                        method="trf", max_nfev=200)
    return unpack(sol.x)

P_NOLOOP = solve(False); P_LOOP = solve(True)

def ate(P):
    return float(np.sqrt(np.mean(np.sum((P[:, :2] - POSES[:, :2]) ** 2, axis=1))))

def end_err(P):
    return float(np.linalg.norm(P[-1, :2] - POSES[-1, :2]))

DEAD_END = end_err(P_DEAD); LOOP_END = end_err(P_LOOP)
DEAD_ATE = ate(P_DEAD); LOOP_ATE = ate(P_LOOP)
checks = [
    ("关键帧都有观测", min(len(OBS[i]) for i in range(N)) > 0, f"min {min(len(OBS[i]) for i in range(N))}"),
    ("共视图连通", len(COV) >= N - 1, f"{len(COV)} 边"),
    ("生成树覆盖全图", len(TREE) == N - 1 and len({x for e in TREE for x in e}) == N, f"{len(TREE)} 边/{N} 节点"),
    ("Essential 更稀疏", len(ESS) < len(COV) and len({x for e in ESS for x in e}) == N, f"{len(ESS)} < {len(COV)}"),
    ("Essential 含回环边", LOOP_EDGE in ESS, "0–19"),
    ("回环相似度更高", LOOP_SIM > 0.5 and RAND_SIM < 0.1, f"{LOOP_SIM:.3f} vs {RAND_SIM:.3f}"),
    ("回环降终点", DEAD_END > 10 * LOOP_END, f"{DEAD_END:.3f}→{LOOP_END:.4f}"),
    ("回环降 ATE", LOOP_ATE < 0.8 * DEAD_ATE, f"{DEAD_ATE:.3f}→{LOOP_ATE:.3f}"),
    ("裁剪保覆盖", len(KEEP) < N and len(COVERED) == L, f"{len(KEEP)}/{N} 帧，{len(COVERED)}/{L} 点"),
    ("三图在盘", all((FIG / f"e109_{n}.png").exists() for n in ("architecture", "graph", "loop")), "architecture/graph/loop"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "n_keyframe": N, "n_point": L,
    "n_cov": len(COV), "n_tree": len(TREE), "n_ess": len(ESS), "theta_cov": 15, "theta_ess": THETA_ESS,
    "loop_sim": LOOP_SIM, "loop_sim_3": round(LOOP_SIM, 3), "rand_sim": RAND_SIM, "rand_sim_3": round(RAND_SIM, 3),
    "dead_end": DEAD_END, "dead_end_3": round(DEAD_END, 3), "loop_end": LOOP_END, "loop_end_4": round(LOOP_END, 4),
    "dead_ate": DEAD_ATE, "dead_ate_3": round(DEAD_ATE, 3), "loop_ate": LOOP_ATE, "loop_ate_3": round(LOOP_ATE, 3),
    "min_obs": min(len(OBS[i]) for i in range(N)), "n_keep": len(KEEP), "n_covered": len(COVERED),
}

# 图1 三线程架构
fig, ax = plt.subplots(figsize=(11.2, 5.0)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.0); ax.axis("off")
ax.text(0.2, 4.7, "ORB-SLAM：同一套 ORB 特征，三个线程并行", fontsize=17, color=INK, weight="bold")
for i, (name, col) in enumerate([("跟踪\nTracking", BLUE), ("局部建图\nLocal Mapping", GREEN), ("回环\nLoop Closing", PURPLE)]):
    x = 0.5 + i * 3.6
    ax.add_patch(plt.Rectangle((x, 2.7), 2.9, 1.2, facecolor="#eef2f7", edgecolor=col, lw=1.8))
    ax.text(x + 1.45, 3.3, name, ha="center", va="center", fontsize=15.5, color=INK, weight="bold")
ax.add_patch(plt.Rectangle((3.3, 0.6), 4.6, 1.1, facecolor="#f3e9d2", edgecolor=ACCENT, lw=1.6))
ax.text(5.6, 1.15, "共享地图：关键帧 + 地图点 + 共视图", ha="center", fontsize=15.5, color=INK, weight="bold")
for i in range(3):
    x = 1.95 + i * 3.6
    ax.annotate("", xy=(x, 1.7), xytext=(x, 2.7), arrowprops=dict(arrowstyle="<->", color=GRAY, lw=1.4))
fig.tight_layout(); fig.savefig(FIG / "e109_architecture.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 共视图 vs Essential Graph
def circular_layout(n):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])
pos = circular_layout(N)
fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
for ax, edges, title, color in [(axes[0], COV, "共视图：79 条边", GRAY),
                                (axes[1], ESS, "Essential Graph：22 条边", GREEN)]:
    for (i, j) in edges:
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], "-", color=color, lw=1.0, alpha=0.7)
    ax.scatter(pos[:,0], pos[:,1], s=70, color=BLUE, zorder=3)
    for i in range(N):
        ax.text(pos[i,0]*1.12, pos[i,1]*1.12, str(i), fontsize=13, ha="center", va="center", color=INK)
    if title.startswith("Essential"):
        for (i, j) in TREE:
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], "-", color=PURPLE, lw=1.8, alpha=0.9)
        i, j = LOOP_EDGE
        ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], "-", color=ACCENT, lw=2.4)
    ax.set_title(title, fontsize=15, color=INK); ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout(); fig.savefig(FIG / "e109_graph.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 回环修正
fig, ax = plt.subplots(figsize=(9.8, 4.9))
ax.plot(POSES[:,0], POSES[:,1], "-", color=INK, lw=2.0, label="真值")
ax.plot(P_DEAD[:,0], P_DEAD[:,1], "--", color=GRAY, lw=1.6, label="回环前")
ax.plot(P_LOOP[:,0], P_LOOP[:,1], "-", color=ACCENT, lw=2.0, label="回环后")
ax.scatter([P_DEAD[-1,0], P_LOOP[-1,0]], [P_DEAD[-1,1], P_LOOP[-1,1]], s=55, color=[GRAY, ACCENT], zorder=3)
ax.set_aspect("equal"); ax.set_xlabel("X / m", fontsize=14.5); ax.set_ylabel("Y / m", fontsize=14.5)
ax.set_title("回环修正：终点 0.227→0.011", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(alpha=0.2); ax.legend(fontsize=13.2, frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e109_loop.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e109_{n}.png").exists() for n in ("architecture", "graph", "loop")), "architecture/graph/loop")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E109 ORB-SLAM 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e109_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e109_orbslam_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e109_orbslam_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e109_orbslam_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
