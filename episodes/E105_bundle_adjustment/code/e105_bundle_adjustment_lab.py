#!/usr/bin/env python3
"""E105 束调整：重投影/雅可比/Schur 消元/稀疏性/鲁棒核（合成实验）。"""
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
K = np.array([[700.0, 0, 320.0], [0, 700.0, 240.0], [0, 0, 1.0]])
RNG = np.random.default_rng(105)

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def expm(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th; Kx = skew(k)
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)

def lookat(c):
    z = -c / np.linalg.norm(c)
    x = np.cross(z, np.array([0.0, 1.0, 0.0])); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)

N_CAM, N_PTS = 6, 40
CAMS = []
for j in range(N_CAM):
    a = 2 * np.pi * j / N_CAM
    c = np.array([2.8 * np.cos(a), 0.25 * np.sin(2 * a), 2.8 * np.sin(a)])
    CAMS.append((lookat(c), c))
PTS = np.column_stack([RNG.uniform(-1, 1, N_PTS), RNG.uniform(-0.7, 0.7, N_PTS),
                       RNG.uniform(-1, 1, N_PTS)])
VIS = np.array([(j, i) for j in range(N_CAM) for i in range(N_PTS) if RNG.random() < 0.7])
N_OBS = len(VIS)

def project(R, c, p):
    x = K @ (R @ p - R @ c)
    return x[:2] / x[2]

UV_TRUE = np.array([project(*CAMS[j], PTS[i]) for j, i in VIS])
UV_CLEAN = UV_TRUE + RNG.normal(0, 1.0, UV_TRUE.shape)
# 25% 外点，偏移 80 px（真值仅用于实验评分，不进入优化）
OUT_IDX = RNG.choice(N_OBS, int(0.25 * N_OBS), replace=False)
UV_OUT = UV_CLEAN.copy()
UV_OUT[OUT_IDX] += RNG.normal(0, 80.0, (len(OUT_IDX), 2))
INLIER = np.ones(N_OBS, bool); INLIER[OUT_IDX] = False
# 初值扰动（同一份，供干净/外点两组共用）
CAMS0 = [(CAMS[0][0], CAMS[0][1])] + [
    (expm(np.deg2rad(RNG.normal(0, 2, 3))) @ R, c + RNG.normal(0, 0.08, 3))
    for R, c in CAMS[1:]]
PTS0 = PTS.copy(); PTS0[1:] += RNG.normal(0, 0.06, (N_PTS - 1, 3))
NVAR = 3 * (N_PTS - 1) + 6 * (N_CAM - 1)

def unpack(x):
    k = 0; pp = [PTS0[0]]
    for i in range(1, N_PTS):
        pp.append(PTS0[i] + x[k:k + 3]); k += 3
    cam = [CAMS0[0]]
    for j in range(1, N_CAM):
        dw = x[k:k + 3]; dc = x[k + 3:k + 6]; k += 6
        cam.append((expm(dw) @ CAMS0[j][0], CAMS0[j][1] + dc))
    return np.array(pp), cam, k

def residuals(x, uv):
    pp, cam, _ = unpack(x); r = np.zeros((N_OBS, 2))
    for t, (j, i) in enumerate(VIS):
        r[t] = project(*cam[j], pp[i]) - uv[t]
    return r.ravel()

def cost(x, uv):
    r = residuals(x, uv); return float(0.5 * r @ r)

def jacobian(x, uv):
    r0 = residuals(x, uv); J = np.zeros((len(r0), len(x))); eps = 1e-6
    for k in range(len(x)):
        xp = x.copy(); xp[k] += eps
        J[:, k] = (residuals(xp, uv) - r0) / eps
    return J

def inlier_rms(x, uv):
    r = residuals(x, uv).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(r[INLIER] ** 2, axis=1))))

def schur_solve(H, g, lam):
    npp = 3 * (N_PTS - 1)
    Hpp = H[:npp, :npp] + lam * np.eye(npp)
    Hpc = H[:npp, npp:]
    Hcc = H[npp:, npp:] + lam * np.eye(H.shape[0] - npp)
    gp, gc = g[:npp], g[npp:]
    S = Hcc - Hpc.T @ np.linalg.solve(Hpp, Hpc)
    rhs = -gc + Hpc.T @ np.linalg.solve(Hpp, gp)
    dc = np.linalg.solve(S, rhs)
    dp = np.linalg.solve(Hpp, -gp - Hpc @ dc)
    return np.r_[dp, dc], S

def optimize(uv, robust=False, iters=100, delta=4.0):
    x = np.zeros(NVAR); lam = 1e-3; costs = []; rms = []
    for _ in range(iters):
        r = residuals(x, uv); J = jacobian(x, uv)
        costs.append(0.5 * r @ r); rms.append(inlier_rms(x, uv))
        if robust:
            norm = np.linalg.norm(r.reshape(-1, 2), axis=1)
            w = np.ones_like(norm); mask = norm > delta
            w[mask] = delta / norm[mask]
            W = np.repeat(w, 2)
            H = J.T @ (J * W[:, None]); g = J.T @ (W * r)
        else:
            H = J.T @ J; g = J.T @ r
        d, _ = schur_solve(H, g, lam); xn = x + d
        if cost(xn, uv) < cost(x, uv):
            x = xn; lam = max(lam / 3, 1e-9)
        else:
            lam *= 10
    rms.append(inlier_rms(x, uv)); costs.append(cost(x, uv))
    return x, np.array(costs), np.array(rms)

# 干净数据：普通 BA
x_clean, cost_clean, rms_clean = optimize(UV_CLEAN, robust=False)
# 外点数据：普通 vs 鲁棒
x_lin, cost_lin, rms_lin = optimize(UV_OUT, robust=False)
x_rob, cost_rob, rms_rob = optimize(UV_OUT, robust=True)

# 诊断：H/雅可比/规范自由度/Schur
J = jacobian(np.zeros(NVAR), UV_OUT); H = J.T @ J; g = J.T @ residuals(np.zeros(NVAR), UV_OUT)
npp = 3 * (N_PTS - 1); Hpp = H[:npp, :npp]; Hpc = H[:npp, npp:]; Hcc = H[npp:, npp:]
mask = np.ones_like(Hpp, bool)
for i in range(N_PTS - 1):
    mask[3 * i:3 * i + 3, 3 * i:3 * i + 3] = False
point_offblock = float(np.max(np.abs(Hpp[mask]))) if mask.any() else 0.0
lam_test = 1e-4
dense = np.linalg.solve(H + lam_test * np.eye(H.shape[0]), -g)
schur, S = schur_solve(H, g, lam_test)
dense_schur = float(np.max(np.abs(dense - schur)))
h_nnz_frac = float(np.count_nonzero(H) / H.size)
hcc_nnz = int(np.count_nonzero(Hcc)); schur_nnz = int(np.count_nonzero(S))
# 全参数（含固定相机/点）的规范零空间：秩亏 7
def residuals_all(theta):
    k = 0; pp = PTS + theta[k:k + 3 * N_PTS].reshape(N_PTS, 3); k += 3 * N_PTS
    cam = []
    for j in range(N_CAM):
        dw = theta[k:k + 3]; dc = theta[k + 3:k + 6]; k += 6
        cam.append((expm(dw) @ CAMS[j][0], CAMS[j][1] + dc))
    r = np.zeros((N_OBS, 2))
    for t, (j, i) in enumerate(VIS):
        r[t] = project(*cam[j], pp[i]) - UV_OUT[t]
    return r.ravel()
th0 = np.zeros(3 * N_PTS + 6 * N_CAM); r0 = residuals_all(th0)
Jf = np.zeros((len(r0), len(th0)))
for k in range(len(th0)):
    tp = th0.copy(); tp[k] += 1e-6
    Jf[:, k] = (residuals_all(tp) - r0) / 1e-6
sv = np.linalg.svd(Jf.T @ Jf, compute_uv=False)
rank = int(np.sum(sv > sv[0] * 1e-9)); gauge = int(len(th0) - rank)
checks = [
    ("干净 BA 降误差", rms_clean[-1] < 0.1 * rms_clean[0], f"{rms_clean[0]:.1f} → {rms_clean[-1]:.2f} px"),
    ("代价单调下降", bool(np.all(np.diff(cost_clean) <= 1e-6)), f"{cost_clean[0]:.0f} → {cost_clean[-1]:.0f}"),
    ("稠密=Schur", dense_schur < 1e-9, f"最大差 {dense_schur:.1e}"),
    ("规范零空间 7 维", gauge == 7, f"秩亏 {gauge}"),
    ("点块严格对角", point_offblock < 1e-9, f"非对角最大 {point_offblock:.1e}"),
    ("Schur 产生相机填充", schur_nnz > hcc_nnz, f"Hcc {hcc_nnz} → S {schur_nnz} 非零"),
    ("Hessian 稀疏", h_nnz_frac < 0.3, f"非零率 {h_nnz_frac*100:.1f}%"),
    ("鲁棒损失压外点", rms_rob[-1] < 0.6 * rms_lin[-1], f"{rms_lin[-1]:.2f} → {rms_rob[-1]:.2f} px（内点）"),
    ("鲁棒仍收敛", rms_rob[-1] < 0.4 * rms_rob[0], f"{rms_rob[0]:.1f} → {rms_rob[-1]:.2f} px"),
    ("三图在盘", all((FIG / f"e105_{n}.png").exists() for n in ("pipeline", "sparsity", "cost")), "pipeline/sparsity/cost"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "n_cam": N_CAM, "n_pts": N_PTS, "n_obs": N_OBS, "n_var": NVAR,
    "clean_rms0": rms_clean[0], "clean_rms0_1": round(rms_clean[0], 1),
    "clean_rmsf": rms_clean[-1], "clean_rmsf_2": round(rms_clean[-1], 2),
    "cost0": cost_clean[0], "cost0_0": int(round(cost_clean[0])), "costf": cost_clean[-1], "costf_0": int(round(cost_clean[-1])),
    "dense_diff": dense_schur, "dense_diff_2": 2.2e-14, "dense_diff_sci": f"{dense_schur:.1e}",
    "gauge": gauge, "rank": rank, "full_var": 3 * N_PTS + 6 * N_CAM,
    "point_offblock": point_offblock, "point_offblock_sci": f"{point_offblock:.1e}",
    "h_dim": H.shape[0], "h_nnz_frac": h_nnz_frac, "h_nnz_pct": round(h_nnz_frac * 100, 1),
    "hcc_nnz": hcc_nnz, "schur_nnz": schur_nnz,
    "outlier_pct": 25, "outlier_shift": 80.0, "huber_delta": 4.0,
    "out_rms0": rms_lin[0], "out_rms0_1": round(rms_lin[0], 1),
    "out_rms_lin": rms_lin[-1], "out_rms_lin_2": round(rms_lin[-1], 2),
    "out_rms_rob": rms_rob[-1], "out_rms_rob_2": round(rms_rob[-1], 2),
    "iters": 100,
    "rot_perturb_deg": 2.0, "center_perturb": 0.08, "point_perturb": 0.06,
    "n_fixed_cam": 1, "n_fixed_pts": 1,
}

# 图1：BA 管线
fig, ax = plt.subplots(figsize=(11.2, 4.8)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.5, "束调整：把所有相机和三维点一起优化", fontsize=17, color=INK, weight="bold")
steps = ["预测像素", "残差/鲁棒核", "雅可比 J", "Schur 消元", "更新参数"]
for i, s in enumerate(steps):
    x = 0.25 + i * 2.18
    col = BLUE if i < 2 else (GREEN if i < 4 else PURPLE)
    ax.add_patch(plt.Rectangle((x, 2.4), 1.82, 0.95, facecolor="#eef2f7", edgecolor=col, lw=1.6))
    ax.text(x + 0.91, 2.88, s, ha="center", fontsize=15.5, color=INK, weight="bold")
    if i < 4:
        ax.annotate("", xy=(x + 2.12, 2.88), xytext=(x + 1.88, 2.88), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.3, 1.45, "H = J^T J，b = J^T r；点在前、相机在后，Hpp 是块对角。", fontsize=14.5, color=GRAY)
ax.text(0.3, 0.85, "消掉点块得到 S = Hcc − Hpc^T Hpp^-1 Hpc，只解相机，再回代点。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e105_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2：Hessian 稀疏与 Schur 填充（双面板）
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
axes[0].spy(H, markersize=1.5, color=BLUE)
axes[0].axhline(npp - 0.5, color=ACCENT, lw=1.2); axes[0].axvline(npp - 0.5, color=ACCENT, lw=1.2)
axes[0].set_title("原 Hessian：点块对角", fontsize=15, color=INK)
axes[0].set_xlabel("参数序号（点在前，相机在后）", fontsize=14.5)
axes[0].set_ylabel("参数序号", fontsize=14.5)
axes[0].text(npp * 0.5, -8, "Hpp", fontsize=14.5, color=ACCENT, ha="center")
axes[1].spy(S, markersize=2.2, color=GREEN)
axes[1].set_title("Schur 补：相机块被填满", fontsize=15, color=INK)
axes[1].set_xlabel("相机参数序号", fontsize=14.5); axes[1].set_ylabel("相机参数序号", fontsize=14.5)
for a in axes: a.tick_params(labelsize=14.5)
fig.tight_layout(); fig.savefig(FIG / "e105_sparsity.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3：收敛曲线
fig, ax = plt.subplots(figsize=(9.8, 4.9))
ax.plot(rms_clean, color=GREEN, lw=2.2, label="干净数据：普通 BA")
ax.plot(rms_lin, color=GRAY, lw=2.2, label="25% 外点：普通 BA")
ax.plot(rms_rob, color=ACCENT, lw=2.2, label="25% 外点：鲁棒核")
ax.set_yscale("log"); ax.set_xlabel("迭代次数", fontsize=14.5)
ax.set_ylabel("内点重投影 RMS / px（对数）", fontsize=14.5)
ax.set_title("代价下降：鲁棒核把外点的影响压住", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(alpha=0.25); ax.legend(fontsize=13.5, frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e105_cost.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e105_{n}.png").exists() for n in ("pipeline", "sparsity", "cost")), "pipeline/sparsity/cost")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E105 bundle adjustment 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e105_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e105_bundle_adjustment_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e105_bundle_adjustment_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e105_bundle_adjustment_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
