#!/usr/bin/env python3
"""E107 全局 SFM：旋转平均/中心求解/共线退化（合成实验）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d", "#6b7280"
RNG = np.random.default_rng(107)

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def expm(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th; Kx = skew(k)
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)

def rerr(Ra, Rb):
    C = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(C) - 1) / 2, -1, 1))))

def avg_rot(Rs, ws=None):
    if ws is None:
        ws = np.ones(len(Rs))
    M = sum(w * R for w, R in zip(ws, Rs))
    U, _, Vt = np.linalg.svd(M); R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    return R

N = 12
RTRUE = []
for j in range(N):
    a = 2 * np.pi * j / N
    RTRUE.append(expm(np.array([0.15 * np.sin(2 * a), a, 0.1 * np.cos(a)])))
R0 = RTRUE[0]; RTRUE = [R0.T @ R for R in RTRUE]
CTRUE = np.array([[3 * np.cos(2 * np.pi * j / N), 0.2 * np.sin(4 * np.pi * j / N),
                   3 * np.sin(2 * np.pi * j / N)] for j in range(N)])
# 边=链+弦；外点只放在弦上；链用于初始化
EDGES = [(i, i + 1) for i in range(N - 1)]
CHORDS = []
for i in range(N):
    for j in range(i + 2, N):
        if RNG.random() < 0.25:
            CHORDS.append((i, j))
EDGES += CHORDS
RREL = {}; OUT_EDGES = set()
for i, j in EDGES:
    R = RTRUE[j] @ RTRUE[i].T
    R = expm(RNG.normal(0, np.deg2rad(0.5), 3)) @ R
    RREL[(i, j)] = R
for k in RNG.choice(len(CHORDS), max(1, int(0.15 * len(CHORDS))), replace=False):
    OUT_EDGES.add(CHORDS[k])
for e in OUT_EDGES:
    RREL[e] = expm(RNG.normal(0, 1.0, 3)) @ RREL[e]
# 链式初始化
RINIT = [None] * N; RINIT[0] = np.eye(3)
for i in range(N - 1):
    RINIT[i + 1] = RREL[(i, i + 1)] @ RINIT[i]
ADJ = {i: [] for i in range(N)}
for i, j in EDGES:
    ADJ[i].append(j); ADJ[j].append(i)

def rotation_average(robust=False, iters=40, thr=1.0):
    R = [r.copy() for r in RINIT]
    for _ in range(iters):
        new = [R[0]]
        for j in range(1, N):
            cand = []; ws = []
            for i in ADJ[j]:
                Rc = RREL[(i, j)] @ R[i] if (i, j) in RREL else RREL[(j, i)].T @ R[i]
                e = rerr(Rc, R[j]); w = 1.0
                if robust and e > thr:
                    w = thr / e
                cand.append(Rc); ws.append(w)
            new.append(avg_rot(cand, ws))
        R = new
    return R

def rot_errors(R):
    return [rerr(R[j], RTRUE[j]) for j in range(N)]

def solve_centers(edges, dirs, scale_fix=1.0):
    idx = {j: 3 * (j - 1) for j in range(1, N)}
    ns = 3 * (N - 1); ne = len(edges); A = []; b = []
    for k, (i, j) in enumerate(edges):
        t = dirs[k]
        for d in range(3):
            row = np.zeros(ns + ne)
            if j > 0: row[idx[j] + d] += 1
            if i > 0: row[idx[i] + d] -= 1
            row[ns + k] = -t[d]
            A.append(row); b.append(0)
    row = np.zeros(ns + ne); row[ns] = 1
    A.append(row); b.append(scale_fix)
    A = np.array(A); b = np.array(b)
    x, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    C = np.zeros((N, 3))
    for j in range(1, N):
        C[j] = x[idx[j]:idx[j] + 3]
    return C, rank, sv

def umeyama(src, dst):
    mu_s = src.mean(0); mu_d = dst.mean(0)
    S = src - mu_s; D = dst - mu_d
    C = D.T @ S / len(src)
    U, s, Vt = np.linalg.svd(C)
    E = np.eye(3); E[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    var = np.mean(np.sum(S ** 2, axis=1))
    sc = np.trace(np.diag(s) @ E) / var
    t = mu_d - sc * R @ mu_s
    return sc, R, t

R_PLAIN = rotation_average(False)
R_ROBUST = rotation_average(True)
E_PLAIN = rot_errors(R_PLAIN); E_ROBUST = rot_errors(R_ROBUST)
plain_med = float(np.median(E_PLAIN)); plain_max = float(np.max(E_PLAIN))
rob_med = float(np.median(E_ROBUST)); rob_max = float(np.max(E_ROBUST))
# 中心：圆环 vs 共线
def center_solution(centers, noise_deg=0.5):
    # 方向来自两两重建，按 0.5° 角噪声模拟；固定第一条边尺度为真值（评估口径）
    dirs = []
    for i, j in EDGES:
        d = centers[j] - centers[i]; d = d / np.linalg.norm(d)
        d = expm(np.deg2rad(RNG.normal(0, noise_deg, 3))) @ d
        dirs.append(d / np.linalg.norm(d))
    scale_fix = float(np.linalg.norm(centers[EDGES[0][1]] - centers[EDGES[0][0]]))
    C, rank, sv = solve_centers(EDGES, dirs, scale_fix)
    true_shift = centers - centers[0]
    errs = np.linalg.norm(C - true_shift, axis=1)
    return C, true_shift, errs, float(sv[0] / sv[-1]), int(rank)
C_CIRC, TRUE_SHIFT, errs_circ, circle_cond, circle_rank = center_solution(CTRUE)
circle_med, circle_max = float(np.median(errs_circ)), float(np.max(errs_circ))
C_COL = np.column_stack([np.linspace(-3, 3, N), np.zeros(N), np.zeros(N)])
_, TRUE_COL_SHIFT, errs_col, col_cond, col_rank = center_solution(C_COL)
col_med, col_max = float(np.median(errs_col)), float(np.max(errs_col))
checks = [
    ("链式初始化覆盖全图", all(r is not None for r in RINIT), f"{sum(r is not None for r in RINIT)}/{N}"),
    ("普通旋转平均怕外点", plain_med > 5.0, f"中位 {plain_med:.2f}°"),
    ("鲁棒旋转平均可用", rob_med < 2.0, f"中位 {rob_med:.2f}°"),
    ("鲁棒最大误差小", rob_max < 3.0, f"最大 {rob_max:.2f}°"),
    ("鲁棒显著优于普通", rob_med < 0.2 * plain_med, f"{plain_med:.2f}°→{rob_med:.2f}°"),
    ("中心求解可用", circle_med < 0.25 and circle_max < 0.4, f"{circle_med:.3f}/{circle_max:.3f}"),
    ("共线退化更差", col_med > 10 * circle_med, f"{col_med:.2f} vs {circle_med:.3f}"),
    ("共线条件数更大", col_cond > 3 * circle_cond, f"{circle_cond:.0f}→{col_cond:.0f}"),
    ("中心求解满秩", circle_rank == 3*(N-1)+len(EDGES), f"rank {circle_rank}"),
    ("三图在盘", all((FIG / f"e107_{n}.png").exists() for n in ("pipeline", "rotation", "centers")), "pipeline/rotation/centers"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "n_cam": N, "n_edges": len(EDGES),
    "n_chords": len(CHORDS), "n_outlier_edges": len(OUT_EDGES),
    "rot_noise_deg": 0.5, "pair_outlier_deg": 57.2957795, "dir_noise_deg": 0.5,
    "plain_med": plain_med, "plain_med_2": round(plain_med, 2),
    "plain_max": plain_max, "plain_max_2": round(plain_max, 2),
    "rob_med": rob_med, "rob_med_2": round(rob_med, 2),
    "rob_max": rob_max, "rob_max_2": round(rob_max, 2),
    "circle_med": circle_med, "circle_med_3": round(circle_med, 3),
    "circle_max": circle_max, "circle_max_3": round(circle_max, 3),
    "circle_cond": circle_cond, "circle_cond_0": int(round(circle_cond)),
    "col_med": col_med, "col_med_2": round(col_med, 2),
    "col_max": col_max, "col_max_2": round(col_max, 2),
    "col_cond": col_cond, "col_cond_0": int(round(col_cond)),
    "circle_rank": circle_rank, "col_rank": col_rank,
}

# 图1 管线
fig, ax = plt.subplots(figsize=(11.2, 4.8)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.5, "全局 SFM：先对齐旋转，再一次性解相机中心", fontsize=17, color=INK, weight="bold")
steps = ["两两重建", "旋转平均", "中心求解", "全局 BA"]
for i, s in enumerate(steps):
    x = 0.6 + i * 2.55
    col = BLUE if i < 2 else (GREEN if i < 3 else PURPLE)
    ax.add_patch(plt.Rectangle((x, 2.4), 2.0, 0.95, facecolor="#eef2f7", edgecolor=col, lw=1.6))
    ax.text(x + 1.0, 2.88, s, ha="center", fontsize=15.5, color=INK, weight="bold")
    if i < 3:
        ax.annotate("", xy=(x + 2.45, 2.88), xytext=(x + 2.15, 2.88), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.4, 1.45, "旋转平均：把带噪声的相对旋转拉到一个全局一致的姿态。", fontsize=14.5, color=GRAY)
ax.text(0.4, 0.85, "中心求解：c_j − c_i = s_ij t_ij，解所有相机中心和边尺度。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e107_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 旋转平均
fig, ax = plt.subplots(figsize=(8.6, 4.9))
x = np.arange(2); w = 0.34
ax.bar(x - w/2, [plain_med, rob_med], w, color=[GRAY, GREEN], label="中位")
ax.bar(x + w/2, [plain_max, rob_max], w, color=[GRAY, GREEN], alpha=0.45, label="最大")
ax.set_xticks(x); ax.set_xticklabels(["普通平均", "鲁棒平均"], fontsize=14.5)
ax.set_ylabel("全局旋转误差 / 度（对数）", fontsize=14.5); ax.set_yscale("log")
ax.set_title("外点：普通平均被拉偏，鲁棒平均拉回来", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(axis="y", alpha=0.25); ax.legend(fontsize=13.5, frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e107_rotation.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 中心几何 + 共线对照
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
axes[0].plot(CTRUE[:, 0], CTRUE[:, 2], "o-", color=BLUE, ms=6, lw=1.5, label="真值")
axes[0].plot(C_CIRC[:, 0], C_CIRC[:, 2], "x--", color=ACCENT, ms=7, lw=1.3, label="估计")
axes[0].set_xlabel("X", fontsize=14.5); axes[0].set_ylabel("Z", fontsize=14.5)
axes[0].set_title("圆环相机：位置可解", fontsize=15, color=INK)
axes[0].legend(fontsize=13.5, frameon=False); axes[0].tick_params(labelsize=14.5); axes[0].grid(alpha=0.2)
axes[1].bar(["圆环", "共线"], [circle_med, col_med], color=[GREEN, ACCENT])
axes[1].set_yscale("log"); axes[1].set_ylabel("相机中心中位误差（对数）", fontsize=14.5)
axes[1].set_title("共线：误差放大", fontsize=15, color=INK)
axes[1].tick_params(labelsize=14.5); axes[1].grid(axis="y", alpha=0.25)
for a in axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e107_centers.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e107_{n}.png").exists() for n in ("pipeline", "rotation", "centers")), "pipeline/rotation/centers")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E107 global SFM 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e107_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e107_global_sfm_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e107_global_sfm_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e107_global_sfm_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
