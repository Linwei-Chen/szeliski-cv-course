#!/usr/bin/env python3
"""E104 位姿恢复与三角化：DLT/非线性精化/鲁棒损失/射线三角化（合成实验）。"""
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

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project(R, t, X):
    Y = X @ R.T + t
    x = Y @ K.T
    return x[:, :2] / x[:, 2:3], Y[:, 2]

C_TRUE = np.array([1.6, -0.9, 6.0])
R_TRUE = lookat(C_TRUE)
T_TRUE = -R_TRUE @ C_TRUE

def rerr(Ra, Rb):
    C = Ra @ Rb.T
    return float(np.degrees(np.arccos(np.clip((np.trace(C) - 1) / 2, -1, 1))))

def rms(R, t, X, uv):
    return float(np.sqrt(np.mean(np.sum((project(R, t, X)[0] - uv) ** 2, axis=1))))

def dlt_pose(X, uv):
    m = X.mean(axis=0)
    s = np.sqrt(3) / np.mean(np.linalg.norm(X - m, axis=1))
    Xn = (X - m) * s
    T1 = np.eye(4); T1[:3, :3] *= s; T1[:3, 3] = -s * m
    mu = uv.mean(axis=0)
    su = np.sqrt(2) / np.mean(np.linalg.norm(uv - mu, axis=1))
    uvn = (uv - mu) * su
    T2 = np.array([[su, 0, -su * mu[0]], [0, su, -su * mu[1]], [0, 0, 1.0]])
    A = []
    for (X0, Y0, Z0), (u, v) in zip(Xn, uvn):
        A.append([X0, Y0, Z0, 1, 0, 0, 0, 0, -u * X0, -u * Y0, -u * Z0, -u])
        A.append([0, 0, 0, 0, X0, Y0, Z0, 1, -v * X0, -v * Y0, -v * Z0, -v])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(T2) @ Pn @ T1
    J = np.fliplr(np.eye(3))
    Q, Rq = np.linalg.qr((J @ P[:, :3]).T)
    Kq = J @ Rq.T @ J; Rr = J @ Q.T
    S = np.diag(np.sign(np.diag(Kq)))
    Kq = Kq @ S; Rr = S @ Rr
    sc = Kq[2, 2]
    Kq = Kq / sc; P = P / sc
    tq = np.linalg.inv(Kq) @ P[:, 3]
    if np.linalg.det(Rr) < 0:
        Rr = -Rr; tq = -tq
    return Kq, Rr, tq

def refine(R0, t0, X, uv, loss="linear", f_scale=1.0):
    def fun(p):
        Rr = expm(p[:3]) @ R0
        tt = t0 + p[3:]
        return (project(Rr, tt, X)[0] - uv).ravel()
    sol = least_squares(fun, np.zeros(6), method="trf", loss=loss,
                        f_scale=f_scale, max_nfev=500)
    return expm(sol.x[:3]) @ R0, t0 + sol.x[3:]

def scene(n, noise, seed, outlier=0.0):
    rr = np.random.default_rng(seed)
    X = rr.uniform([-1.2, -0.8, 4.0], [1.2, 0.8, 8.0], size=(n, 3))
    uv, _ = project(R_TRUE, T_TRUE, X)
    uv = uv + rr.normal(0, noise, uv.shape)
    idx = np.array([], dtype=int)
    if outlier:
        k = int(round(n * outlier))
        idx = rr.choice(n, k, replace=False)
        uv[idx] += rr.normal(0, 50, (k, 2))
    return X, uv, idx

def pose_once(n=6, noise=1.0, seed=0, robust=False):
    X, uv, idx = scene(n, noise, seed, 0.2 if robust else 0.0)
    Ke, Re, te = dlt_pose(X, uv)
    Rl, tl = refine(Re, te, X, uv, "linear")
    if robust:
        Rr, tr = refine(Re, te, X, uv, "soft_l1", 2.0)
        mask = np.ones(n, bool); mask[idx] = False
        def inrms(R, t):
            return float(np.sqrt(np.mean(np.sum((project(R, t, X)[0][mask] - uv[mask]) ** 2, axis=1))))
        return dict(r_dlt=rerr(Re, R_TRUE), r_ls=rerr(Rl, R_TRUE), r_rob=rerr(Rr, R_TRUE),
                    e_dlt=inrms(Re, te), e_ls=inrms(Rl, tl), e_rob=inrms(Rr, tr))
    Rr, tr = refine(Re, te, X, uv, "linear")
    return dict(r_dlt=rerr(Re, R_TRUE), r_ref=rerr(Rr, R_TRUE),
                e_dlt=rms(Re, te, X, uv), e_ref=rms(Rr, tr, X, uv),
                cerr=float(np.linalg.norm(-Re.T @ te - C_TRUE)))

def median_over(n, noise, key="r_ref", trials=30, robust=False):
    vals = [pose_once(n, noise, s, robust)[key] for s in range(trials)]
    return float(np.median(vals))

# 位姿：干净 DLT、精化、点数、噪声
X0, uv0, _ = scene(12, 0.0, 0)
_, R_clean, t_clean = dlt_pose(X0, uv0)
clean_rot = rerr(R_clean, R_TRUE)
clean_cerr = float(np.linalg.norm(-R_clean.T @ t_clean - C_TRUE))
p6 = pose_once(6, 1.0, 0)
rot6 = median_over(6, 1.0, "r_dlt"); rot6_ref = median_over(6, 1.0, "r_ref")
rms6 = median_over(6, 1.0, "e_dlt"); rms6_ref = median_over(6, 1.0, "e_ref")
rot_pts = {n: median_over(n, 1.0, "r_ref") for n in (6, 12, 50)}
rot_noise = {sg: median_over(20, sg, "r_ref") for sg in (0.5, 1.5, 4.0)}
rob = [pose_once(30, 1.0, s, robust=True) for s in range(20)]
rob_med = {k: float(np.median([d[k] for d in rob])) for k in rob[0]}

# 三角化：2/3 视角、噪声、远点条件数、cheirality
CAMS = [(np.eye(3), np.zeros(3)), (rot_y(np.deg2rad(20)), np.array([1.0, 0.05, 0.1])),
        (rot_y(np.deg2rad(-15)), np.array([-0.8, 0.2, 0.3]))]
def cam_project(R, c, X):
    x = K @ (R @ X - R @ c)
    return x[:2] / x[2]
def cam_ray(R, c, x):
    return R.T @ np.linalg.inv(K) @ np.append(x, 1)
def tri_ray(xs, cams=None):
    cams = CAMS if cams is None else cams
    A = np.zeros((3, 3)); b = np.zeros(3)
    for (R, c), x in zip(cams, xs):
        v = cam_ray(R, c, x); v = v / np.linalg.norm(v)
        M = np.eye(3) - np.outer(v, v)
        A += M; b += M @ c
    return np.linalg.solve(A, b)
def tri_err(nview, noise, seed, X):
    rr = np.random.default_rng(seed)
    xs = [cam_project(R, c, X) for R, c in CAMS[:nview]]
    xs = [x + rr.normal(0, noise, 2) for x in xs]
    return float(np.linalg.norm(tri_ray(xs, CAMS[:nview]) - X))
X_TRUE = np.array([0.2, 0.1, 5.0])
xs_clean = [cam_project(R, c, X_TRUE) for R, c in CAMS]
tri_clean = float(np.linalg.norm(tri_ray(xs_clean) - X_TRUE))
tri2 = float(np.median([tri_err(2, 1.0, s, X_TRUE) for s in range(100)]))
tri3 = float(np.median([tri_err(3, 1.0, s, X_TRUE) for s in range(100)]))
def cond_far(Z):
    Xf = np.array([0.2, 0.1, Z])
    A = []
    for (R, c) in CAMS[:2]:
        P = K @ np.hstack([R, (-R @ c)[:, None]])
        x = cam_project(R, c, Xf)
        A.append(x[0] * P[2] - P[0]); A.append(x[1] * P[2] - P[1])
    A = np.array(A)
    return float(np.linalg.cond(A[:, :3]))
cond5, cond500 = cond_far(5.0), cond_far(500.0)
X_BACK = np.array([0.2, 0.1, -5.0])
xb = [cam_project(R, c, X_BACK) for R, c in CAMS]
p_back = tri_ray(xb)
def depths(p, xs):
    out = []
    for (R, c), x in zip(CAMS, xs):
        v = cam_ray(R, c, x); v = v / np.linalg.norm(v)
        out.append(float(v @ (p - c)))
    return out
depth_true = depths(tri_ray(xs_clean), xs_clean)
depth_back = depths(p_back, xb)

checks = [
    ("干净 DLT 恢复位姿", clean_rot < 1e-5 and clean_cerr < 1e-6,
     f"R 误差 {clean_rot:.2e}°，相机中心 {clean_cerr:.2e}"),
    ("精化降低旋转误差", rot6_ref < rot6, f"{rot6:.3f}° → {rot6_ref:.3f}°"),
    ("精化降低重投影误差", rms6_ref < 0.2 * rms6, f"{rms6:.2f} → {rms6_ref:.2f} px"),
    ("点越多旋转越准", rot_pts[6] > rot_pts[12] > rot_pts[50],
     f"{rot_pts[6]:.3f}/{rot_pts[12]:.3f}/{rot_pts[50]:.3f}°"),
    ("噪声越大旋转越差", rot_noise[0.5] < rot_noise[1.5] < rot_noise[4.0],
     f"{rot_noise[0.5]:.3f}/{rot_noise[1.5]:.3f}/{rot_noise[4.0]:.3f}°"),
    ("鲁棒损失压住外点", rob_med["r_rob"] < 0.5 * rob_med["r_ls"] and rob_med["e_rob"] < rob_med["e_ls"],
     f"R {rob_med['r_ls']:.3f}°→{rob_med['r_rob']:.3f}°，内点 RMS {rob_med['e_ls']:.2f}→{rob_med['e_rob']:.2f} px"),
    ("三角化干净且多视角更准", tri_clean < 1e-9 and tri3 < tri2,
     f"干净 {tri_clean:.2e}；2→3 视角 {tri2:.4f}→{tri3:.4f}"),
    ("远点条件数变大", cond500 > 10 * cond5, f"{cond5:.1f} → {cond500:.1f}"),
    ("cheirality 正负分明", min(depth_true) > 0 and min(depth_back) < 0,
     f"真点 {min(depth_true):+.2f}，镜后点 {min(depth_back):+.2f}"),
    ("三图在盘", all((FIG / f"e104_{n}.png").exists() for n in ("pipeline", "pose", "tri")),
     "pipeline/pose/tri"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks),
    "clean_rot": clean_rot, "clean_rot_sci": f"{clean_rot:.1e}",
    "clean_cerr": clean_cerr, "clean_cerr_sci": f"{clean_cerr:.1e}",
    "rot6": rot6, "rot6_3": round(rot6, 3), "rot6_ref": rot6_ref, "rot6_ref_3": round(rot6_ref, 3),
    "rms6": rms6, "rms6_1": round(rms6, 1), "rms6_ref": rms6_ref, "rms6_ref_2": round(rms6_ref, 2),
    "rot12_3": round(rot_pts[12], 3), "rot50_3": round(rot_pts[50], 3),
    "rot_noise05_3": round(rot_noise[0.5], 3), "rot_noise15_3": round(rot_noise[1.5], 3), "rot_noise4_3": round(rot_noise[4.0], 3),
    "r_dlt": rob_med["r_dlt"], "r_ls": rob_med["r_ls"], "r_ls_3": round(rob_med["r_ls"], 3),
    "r_rob": rob_med["r_rob"], "r_rob_3": round(rob_med["r_rob"], 3),
    "e_ls": rob_med["e_ls"], "e_ls_2": round(rob_med["e_ls"], 2), "e_rob": rob_med["e_rob"], "e_rob_2": round(rob_med["e_rob"], 2),
    "tri_clean": tri_clean, "tri_clean_sci": f"{tri_clean:.1e}",
    "tri2": tri2, "tri2_4": round(tri2, 4), "tri3": tri3, "tri3_4": round(tri3, 4),
    "cond5": cond5, "cond5_1": round(cond5, 1), "cond500": cond500, "cond500_1": round(cond500, 1),
    "depth_true_min": min(depth_true), "depth_true_min_2": round(min(depth_true), 2),
    "depth_back_min": min(depth_back), "depth_back_min_2": round(min(depth_back), 2),
    "n_min": 6, "n_unknown": 12, "n_unknown_free": 11, "p3p_points": 3, "fourth_point": 4,
    "n6": 6, "n12": 12, "n20": 20, "n30": 30, "n50": 50,
    "outlier_ratio": 0.2, "outlier_shift": 50.0,
    "noise_unit": 1.0, "noise05": 0.5, "noise15": 1.5, "noise4": 4.0,
    "n_views2": 2, "n_views3": 3, "w_inf": 0.0, "w_one": 1.0, "mat34": 3, "mat33": 3,
}

# 图1：位姿管线
fig, ax = plt.subplots(figsize=(11.2, 4.8)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.5, "位姿恢复与三角化：从 2D-3D 对应到三维点", fontsize=17, color=INK, weight="bold")
steps = ["归一化", "DLT 解 P", "RQ 分解", "重投影精化", "三角化"]
for i, s in enumerate(steps):
    x = 0.25 + i * 2.18
    col = BLUE if i < 2 else (GREEN if i < 4 else PURPLE)
    ax.add_patch(plt.Rectangle((x, 2.4), 1.82, 0.95, facecolor="#eef2f7", edgecolor=col, lw=1.6))
    ax.text(x + 0.91, 2.88, s, ha="center", fontsize=15.5, color=INK, weight="bold")
    if i < 4:
        ax.annotate("", xy=(x + 2.12, 2.88), xytext=(x + 1.88, 2.88), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.3, 1.45, "P = K[R|t]：DLT 先解 3×4 相机矩阵，再用 RQ 拆出内参和外参。", fontsize=14.5, color=GRAY)
ax.text(0.3, 0.85, "最后最小化重投影误差，把线性初值推到统计最优。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e104_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2：点数/噪声与精化
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
axes[0].plot([6, 12, 50], [rot_pts[6], rot_pts[12], rot_pts[50]], "o-", color=ACCENT, lw=2.2)
axes[0].set_xscale("log"); axes[0].set_xlabel("对应点数", fontsize=14.5)
axes[0].set_ylabel("精化后旋转误差 / 度", fontsize=14.5)
axes[0].set_title("点越多，位姿越稳", fontsize=15, color=INK)
axes[0].tick_params(labelsize=14.5); axes[0].grid(alpha=0.25)
axes[1].plot([0.5, 1.5, 4.0], [rot_noise[0.5], rot_noise[1.5], rot_noise[4.0]], "s-", color=BLUE, lw=2.2)
axes[1].set_xlabel("像素噪声 / px", fontsize=14.5)
axes[1].set_ylabel("精化后旋转误差 / 度", fontsize=14.5)
axes[1].set_title("噪声越大，误差越高", fontsize=15, color=INK)
axes[1].tick_params(labelsize=14.5); axes[1].grid(alpha=0.25)
for a in axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e104_pose.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3：三角化俯视图（x-z 平面）
fig, ax = plt.subplots(figsize=(10.4, 5.0))
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.0, 6.2)
for i, (R, c) in enumerate(CAMS):
    ax.plot(c[0], c[2], "o", ms=9, color=[BLUE, GREEN, PURPLE][i])
    ax.text(c[0] + 0.08, c[2] - 0.35, f"C{i}", fontsize=14.5, color=INK, weight="bold")
ax.plot(X_TRUE[0], X_TRUE[2], "*", ms=18, color=ACCENT)
ax.text(X_TRUE[0] + 0.12, X_TRUE[2] + 0.12, "真实点 P", fontsize=14.5, color=ACCENT, weight="bold")
for (R, c), x in zip(CAMS, xs_clean):
    v = cam_ray(R, c, x); v = v / np.linalg.norm(v)
    end = c + v * 6.4
    ax.plot([c[0], end[0]], [c[2], end[2]], "--", color=GRAY, lw=1.4)
ax.plot([X_BACK[0]], [X_BACK[2]], "x", ms=14, color=PURPLE)
ax.text(X_BACK[0] + 0.12, X_BACK[2] - 0.1, "镜后点：深度为负", fontsize=14.5, color=PURPLE, weight="bold")
ax.axhline(0, color=INK, lw=0.8, alpha=0.35)
ax.set_xlabel("X / 世界单位", fontsize=14.5); ax.set_ylabel("Z / 世界单位", fontsize=14.5)
ax.set_title("三角化：找离所有光线最近的点", fontsize=15, color=INK)
ax.tick_params(labelsize=14.5); ax.grid(alpha=0.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e104_tri.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e104_{n}.png").exists() for n in ("pipeline", "pose", "tri")), "pipeline/pose/tri")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E104 pose/triangulation 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e104_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e104_pose_triangulation_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e104_pose_triangulation_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e104_pose_triangulation_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
