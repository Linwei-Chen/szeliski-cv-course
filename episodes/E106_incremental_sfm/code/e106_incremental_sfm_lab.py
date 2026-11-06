#!/usr/bin/env python3
"""E106 增量式 SFM：种子对/视差/PnP 加相机/骨骼集（合成实验）。"""
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
K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
RNG = np.random.default_rng(106)

def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

def expm(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th; Kx = skew(k)
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)

def lookat(c, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])):
    z = target - c; z = z / np.linalg.norm(z)
    x = np.cross(z, up); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)

def project(R, t, X):
    Y = X @ R.T + t
    x = Y @ K.T
    return x[:, :2] / x[:, 2:3], Y[:, 2]

def pnp_dlt(X, uv):
    h = np.column_stack([uv, np.ones(len(uv))])
    b = (np.linalg.inv(K) @ h.T).T; b = b[:, :2] / b[:, 2:3]
    A = []
    for (X0, Y0, Z0), (u, v) in zip(X, b):
        A.append([X0, Y0, Z0, 1, 0, 0, 0, 0, -u * X0, -u * Y0, -u * Z0, -u])
        A.append([0, 0, 0, 0, X0, Y0, Z0, 1, -v * X0, -v * Y0, -v * Z0, -v])
    A = np.array(A); _, _, Vt = np.linalg.svd(A); M = Vt[-1].reshape(3, 4)
    U, S, Vt2 = np.linalg.svd(M[:, :3]); R = U @ Vt2; t = M[:, 3] / np.mean(S)
    if np.linalg.det(R) < 0:
        R = -R; t = -t
    return R, t

def refine(R0, t0, X, uv):
    def fun(p):
        Rr = expm(p[:3]) @ R0
        tt = t0 + p[3:]
        return (project(Rr, tt, X)[0] - uv).ravel()
    sol = least_squares(fun, np.zeros(6), method="trf", max_nfev=200)
    return expm(sol.x[:3]) @ R0, t0 + sol.x[3:]

def umeyama(src, dst):
    mu_s = src.mean(0); mu_d = dst.mean(0)
    S = src - mu_s; D = dst - mu_d
    C = D.T @ S / len(src)
    U, s, Vt = np.linalg.svd(C)
    E = np.eye(3); E[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = U @ E @ Vt
    var = np.mean(np.sum(S ** 2, axis=1))
    scale = np.trace(np.diag(s) @ E) / var
    t = mu_d - scale * R @ mu_s
    return scale, R, t

# 10 个相机绕场景一圈；80 个点；可见性约 65%
N_CAM, N_PTS = 10, 80
CAMS = []
for j in range(N_CAM):
    a = 2 * np.pi * j / N_CAM
    c = np.array([3.0 * np.cos(a), 0.3 * np.sin(2 * a), 3.0 * np.sin(a)])
    CAMS.append((lookat(c), c))
PTS = np.column_stack([RNG.uniform(-1, 1, N_PTS), RNG.uniform(-0.7, 0.7, N_PTS),
                       RNG.uniform(-1, 1, N_PTS)])
VIS = {}
for j, (R, c) in enumerate(CAMS):
    VIS[j] = np.array([i for i in range(N_PTS)
                       if (R @ PTS[i] - R @ c)[2] > 0.5 and RNG.random() < 0.65])
UV = {}
for j, (R, c) in enumerate(CAMS):
    UV[j] = {}
    for i in VIS[j]:
        UV[j][i] = project(R, -R @ c, PTS[[i]])[0][0] + RNG.normal(0, 0.5, 2)

def tri_ray(obs):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for R, c, x in obs:
        v = R.T @ np.linalg.inv(K) @ np.append(x, 1); v = v / np.linalg.norm(v)
        M = np.eye(3) - np.outer(v, v)
        A += M; b += M @ c
    return np.linalg.solve(A, b)

def seed_depth(a, b, idx):
    errs = []
    for i in idx:
        p = tri_ray([(CAMS[a][0], CAMS[a][1], UV[a][i]),
                     (CAMS[b][0], CAMS[b][1], UV[b][i])])
        errs.append(abs(p[2] - PTS[i, 2]))
    return float(np.median(errs)), float(np.percentile(errs, 90))

SEED_IDX = np.intersect1d(VIS[0], VIS[1])
seed_high, seed_high_p90 = seed_depth(0, 1, SEED_IDX)
# 低视差：相机中心只挪 0.08，同看原点；同一批点比较
C_LOW = CAMS[0][1] + np.array([0.08, 0.0, 0.0])
R_LOW = lookat(C_LOW)
CAMS_LOW = (R_LOW, C_LOW)
errs_low = []
for i in SEED_IDX:
    x0 = UV[0][i]
    xl = project(R_LOW, -R_LOW @ C_LOW, PTS[[i]])[0][0] + RNG.normal(0, 0.5, 2)
    p = tri_ray([(CAMS[0][0], CAMS[0][1], x0), (R_LOW, C_LOW, xl)])
    errs_low.append(abs(p[2] - PTS[i, 2]))
seed_low = float(np.median(errs_low)); seed_low_p90 = float(np.percentile(errs_low, 90))

def grow(order):
    cams_est = {order[0]: (CAMS[order[0]][0], CAMS[order[0]][1]),
                order[1]: (CAMS[order[1]][0], CAMS[order[1]][1])}
    pts3 = {}
    for i in np.intersect1d(VIS[order[0]], VIS[order[1]]):
        pts3[i] = tri_ray([(CAMS[j][0], CAMS[j][1], UV[j][i]) for j in order[:2]])
    for j in order:
        if j in cams_est:
            continue
        idx = [i for i in VIS[j] if i in pts3]
        if len(idx) < 8:
            continue
        X = np.array([pts3[i] for i in idx]); UVj = np.array([UV[j][i] for i in idx])
        R, t = pnp_dlt(X, UVj); R, t = refine(R, t, X, UVj)
        rms = float(np.sqrt(np.mean(np.sum((project(R, t, X)[0] - UVj) ** 2, axis=1))))
        if rms > 5.0:
            continue
        cams_est[j] = (R, -R.T @ t)
        for i in set().union(*[set(VIS[k]) for k in cams_est]):
            if i in pts3:
                continue
            seen = [k for k in cams_est if i in VIS[k]]
            if len(seen) >= 2:
                pts3[i] = tri_ray([(cams_est[k][0], cams_est[k][1], UV[k][i]) for k in seen])
    ids = sorted(cams_est)
    src = np.array([cams_est[j][1] for j in ids])
    dst = np.array([CAMS[j][1] for j in ids])
    sc, R, t = umeyama(src, dst)
    cam_err = [float(np.linalg.norm(sc * R @ src[k] + t - dst[k])) for k in range(len(ids))]
    pt_err = [float(np.linalg.norm(sc * R @ p + t - PTS[i])) for i, p in pts3.items()]
    rr = []
    for j in cams_est:
        Rj, cj = cams_est[j]; tj = -Rj @ cj
        for i in VIS[j]:
            if i in pts3:
                rr.append(project(Rj, tj, np.array([pts3[i]]))[0][0] - UV[j][i])
    rr = np.array(rr)
    reproj = float(np.sqrt(np.mean(np.sum(rr ** 2, axis=1))))
    return dict(reg=len(cams_est), cam_med=float(np.median(cam_err)),
                cam_max=float(np.max(cam_err)), pt_med=float(np.median(pt_err)),
                n_pts=len(pts3), reproj=reproj)

full_order = list(range(N_CAM))
skel_order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]
full = grow(full_order); skel = grow(skel_order)
checks = [
    ("高视差种子更准", seed_high < 0.2 * seed_low, f"{seed_high:.4f} vs {seed_low:.4f}"),
    ("顺序生长全注册", full["reg"] == N_CAM, f"{full['reg']}/{N_CAM}"),
    ("顺序相机误差小", full["cam_med"] < 0.01, f"中位 {full['cam_med']:.4f}"),
    ("顺序最大误差小", full["cam_max"] < 0.02, f"最大 {full['cam_max']:.4f}"),
    ("点误差小", full["pt_med"] < 0.01, f"中位 {full['pt_med']:.4f}"),
    ("重投影小", full["reproj"] < 2.0, f"{full['reproj']:.2f} px"),
    ("骨骼集全注册", skel["reg"] == N_CAM, f"{skel['reg']}/{N_CAM}"),
    ("骨骼集误差接近", skel["cam_med"] < 1.5 * full["cam_med"], f"{skel['cam_med']:.4f} vs {full['cam_med']:.4f}"),
    ("骨骼集最大误差小", skel["cam_max"] < 0.02, f"最大 {skel['cam_max']:.4f}"),
    ("三图在盘", all((FIG / f"e106_{n}.png").exists() for n in ("pipeline", "seed", "growth")), "pipeline/seed/growth"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "n_cam": N_CAM, "n_pts": N_PTS,
    "n_seed": int(len(SEED_IDX)), "noise_px": 0.5, "low_baseline": 0.08,
    "seed_high": seed_high, "seed_high_4": round(seed_high, 4),
    "seed_low": seed_low, "seed_low_4": round(seed_low, 4),
    "seed_high_p90": seed_high_p90, "seed_high_p90_4": round(seed_high_p90, 4),
    "seed_low_p90": seed_low_p90, "seed_low_p90_4": round(seed_low_p90, 4),
    "seed_ratio": seed_low / max(seed_high, 1e-12), "seed_ratio_1": round(seed_low / max(seed_high, 1e-12), 1),
    "full_reg": full["reg"], "full_cam_med": full["cam_med"], "full_cam_med_4": round(full["cam_med"], 4),
    "full_cam_max": full["cam_max"], "full_cam_max_4": round(full["cam_max"], 4),
    "full_pt_med": full["pt_med"], "full_pt_med_4": round(full["pt_med"], 4),
    "full_n_pts": full["n_pts"], "full_reproj": full["reproj"], "full_reproj_2": round(full["reproj"], 2),
    "skel_reg": skel["reg"], "skel_cam_med": skel["cam_med"], "skel_cam_med_4": round(skel["cam_med"], 4),
    "skel_cam_max": skel["cam_max"], "skel_cam_max_4": round(skel["cam_max"], 4),
    "skel_pt_med": skel["pt_med"], "skel_pt_med_4": round(skel["pt_med"], 4),
    "skel_n_pts": skel["n_pts"], "skel_reproj": skel["reproj"], "skel_reproj_2": round(skel["reproj"], 2),
}

# 图1：增量管线
fig, ax = plt.subplots(figsize=(11.2, 4.8)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.5, "增量式 SFM：从一对种子图长到完整场景", fontsize=17, color=INK, weight="bold")
steps = ["选种子对", "三角化", "PnP 加相机", "局部 BA", "补新点"]
for i, s in enumerate(steps):
    x = 0.25 + i * 2.18
    col = BLUE if i < 2 else (GREEN if i < 4 else PURPLE)
    ax.add_patch(plt.Rectangle((x, 2.4), 1.82, 0.95, facecolor="#eef2f7", edgecolor=col, lw=1.6))
    ax.text(x + 0.91, 2.88, s, ha="center", fontsize=15.5, color=INK, weight="bold")
    if i < 4:
        ax.annotate("", xy=(x + 2.12, 2.88), xytext=(x + 1.88, 2.88), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.3, 1.45, "种子对要匹配多、视差大；新相机要看到足够多的已三角化点。", fontsize=14.5, color=GRAY)
ax.text(0.3, 0.85, "骨骼集：先重建一小部分，再把剩下的相机用 PnP 折进来，最后 BA。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e106_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2：种子对视差与深度误差
fig, ax = plt.subplots(figsize=(8.4, 4.9))
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [seed_high, seed_low], w, color=[GREEN, ACCENT], label="中位深度误差")
ax.bar(x + w/2, [seed_high_p90, seed_low_p90], w, color=[GREEN, ACCENT], alpha=0.45, label="90 分位")
ax.set_xticks(x); ax.set_xticklabels(["高视差种子", "低视差种子"], fontsize=14.5)
ax.set_yscale("log"); ax.set_ylabel("深度误差 / 世界单位（对数）", fontsize=14.5)
ax.set_title("种子对视差决定深度误差", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(axis="y", alpha=0.25)
ax.legend(fontsize=13.5, frameon=False)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e106_seed.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3：顺序 vs 骨骼集
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
labels = ["顺序", "骨骼集"]; colors = [BLUE, GREEN]
axes[0].bar(labels, [full["cam_med"], skel["cam_med"]], color=colors)
axes[0].set_ylabel("相机中心中位误差", fontsize=14.5)
axes[0].set_title("精度接近", fontsize=15, color=INK)
axes[0].tick_params(labelsize=14.5)
axes[1].bar(labels, [full["reproj"], skel["reproj"]], color=colors)
axes[1].set_ylabel("重投影 RMS / px", fontsize=14.5)
axes[1].set_title("骨骼集仍可用", fontsize=15, color=INK)
axes[1].tick_params(labelsize=14.5)
for a in axes:
    a.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e106_growth.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e106_{n}.png").exists() for n in ("pipeline", "seed", "growth")), "pipeline/seed/growth")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E106 incremental SFM 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e106_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e106_incremental_sfm_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e106_incremental_sfm_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e106_incremental_sfm_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
