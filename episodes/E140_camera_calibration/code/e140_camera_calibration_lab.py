#!/usr/bin/env python3
"""E140 lab：第11章 §11.1 相机内参标定（张正友平面标定 + 径向畸变）。

边界：合成 7×6 平面靶标、6 个视角；自写 DLT 单应 + Zhang 线性 K，
径向畸变/非线性精调用 OpenCV calibrateCamera 作参照。不是真实相机标定，
分数只用于概念对照。
"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; GOLD = "#C9971C"; PURPLE = "#7A5A8E"
for cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(cand).exists():
        font_manager.fontManager.addfont(cand)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = PAPER
plt.rcParams["axes.facecolor"] = PAPER
plt.rcParams["savefig.facecolor"] = PAPER
plt.rcParams["text.color"] = INK
plt.rcParams["axes.labelcolor"] = INK
plt.rcParams["xtick.color"] = INK
plt.rcParams["ytick.color"] = INK
plt.rcParams["axes.edgecolor"] = INK
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["axes.labelsize"] = 15

print(f"[lab] E140 相机内参标定实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] OpenCV {cv2.__version__} / NumPy {np.__version__}")
print("[lab] 边界：合成 7×6 平面靶标、6 视角；自写 Zhang 线性 K，畸变用 OpenCV 参照；不是真实相机标定")

IMG_W, IMG_H = 640, 480
Ktrue = np.array([[800.0, 0.0, 320.0], [0.0, 780.0, 240.0], [0.0, 0.0, 1.0]])
NX, NY = 7, 6
obj = np.array([[float(x), float(y), 0.0] for y in range(NY) for x in range(NX)], np.float32)

def pose(ax, ay, az, tx, ty, tz):
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, np.array([tx, ty, tz])

POSES = [
    pose(0.20, 0.10, 0.05, -2.0, -1.0, 8.0),
    pose(-0.15, 0.25, -0.10, -1.5, -1.2, 9.0),
    pose(0.10, -0.20, 0.15, -2.5, -0.8, 7.5),
    pose(-0.25, 0.15, 0.20, -1.8, -1.5, 10.0),
    pose(0.15, 0.20, -0.15, -2.2, -1.0, 8.5),
    pose(0.05, -0.10, 0.10, -1.6, -1.1, 9.5),
]

def project(R, t, K=Ktrue):
    P = K @ np.c_[R, t]
    X = np.c_[obj, np.ones(len(obj))]
    x = (P @ X.T).T
    return x[:, :2] / x[:, 2:3]

def distort(xy, K, k1):
    xn = (xy - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    r2 = (xn ** 2).sum(axis=1)
    f = 1.0 + k1 * r2
    return xn * f[:, None] * np.array([K[0, 0], K[1, 1]]) + K[:2, 2]

def homography_dlt(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, float)
    _, _, vh = np.linalg.svd(A)
    H = vh[-1].reshape(3, 3)
    return H / H[2, 2]

def vij(H, i, j):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ])

def zhang_K(homographies):
    A = []
    for H in homographies:
        A.append(vij(H, 0, 1))
        A.append(vij(H, 0, 0) - vij(H, 1, 1))
    _, _, vh = np.linalg.svd(np.asarray(A))
    b = vh[-1]
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    if B[0, 0] < 0:
        B = -B
    v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)
    lam = B[2, 2] - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
    alpha = float(np.sqrt(lam / B[0, 0]))
    beta = float(np.sqrt(lam * B[0, 0] / (B[0, 0] * B[1, 1] - B[0, 1] ** 2)))
    gamma = float(-B[0, 1] * alpha ** 2 * beta / lam)
    u0 = float(gamma * v0 / beta - B[0, 2] * alpha ** 2 / lam)
    return np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])

def reproj_h(H, pts):
    q = (H @ np.c_[pts[:, :2], np.ones(len(pts))].T).T
    q = q[:, :2] / q[:, 2:3]
    return q

def bow(points, nx=NX, ny=NY):
    pts = points.reshape(ny, nx, 2)
    devs = []
    for row in pts:
        A = np.c_[row[:, 0], np.ones(nx)]
        m, b = np.linalg.lstsq(A, row[:, 1], rcond=None)[0]
        devs.append(float(np.abs(row[:, 1] - (m * row[:, 0] + b)).max()))
    return max(devs)

# clean views + our Zhang estimate
rng = np.random.default_rng(140)
views_clean = [project(R, t) for R, t in POSES]
Hs = [homography_dlt(obj[:, :2], v) for v in views_clean]
H_err = float(np.mean([np.sqrt(((reproj_h(H, obj) - v) ** 2).sum(axis=1)).mean() for H, v in zip(Hs, views_clean)]))
K_est = zhang_K(Hs)
fx_err = abs(K_est[0, 0] - Ktrue[0, 0]) / Ktrue[0, 0]
fy_err = abs(K_est[1, 1] - Ktrue[1, 1]) / Ktrue[1, 1]
pp_err = float(np.hypot(K_est[0, 2] - Ktrue[0, 2], K_est[1, 2] - Ktrue[1, 2]))

# noise sensitivity via OpenCV full calibration
noise_runs = {}
for noise in (0.0, 0.5, 1.0):
    views = [v + rng.normal(0, noise, v.shape) for v in views_clean]
    op = [obj.reshape(-1, 1, 3) for _ in POSES]
    ip = [v.reshape(-1, 1, 2).astype(np.float32) for v in views]
    rms, Kc, dist, _, _ = cv2.calibrateCamera(op, ip, (IMG_W, IMG_H), None, None)
    noise_runs[noise] = dict(
        fx=float(Kc[0, 0]), fy=float(Kc[1, 1]), cx=float(Kc[0, 2]), cy=float(Kc[1, 2]),
        fx_rel=float(abs(Kc[0, 0] - Ktrue[0, 0]) / Ktrue[0, 0]),
        cx_abs=float(abs(Kc[0, 2] - Ktrue[0, 2])),
        rms=float(rms),
    )

# radial distortion
K1_TRUE = 0.30
views_dist = [distort(project(R, t), Ktrue, K1_TRUE) for R, t in POSES]
op = [obj.reshape(-1, 1, 3) for _ in POSES]
ip = [v.reshape(-1, 1, 2).astype(np.float32) for v in views_dist]
rms_d, Kd, dist_d, _, _ = cv2.calibrateCamera(op, ip, (IMG_W, IMG_H), None, None)
k1_est = float(dist_d.ravel()[0])
fx_d_err = float(abs(Kd[0, 0] - Ktrue[0, 0]) / Ktrue[0, 0])
bow_dist = bow(views_dist[0])
und = cv2.undistortPoints(views_dist[0].reshape(-1, 1, 2).astype(np.float32), Kd, dist_d, P=Ktrue).reshape(-1, 2)
bow_und = bow(und)

checks = []
def check(name, ok, detail):
    checks.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")

check("干净单应重投影 <0.01px", H_err < 0.01, f"{H_err:.4f}px")
check("Zhang fx 相对误差 <0.1%", fx_err < 0.001, f"{fx_err:.6f}")
check("Zhang fy 相对误差 <0.1%", fy_err < 0.001, f"{fy_err:.6f}")
check("Zhang 主点误差 <0.1px", pp_err < 0.1, f"{pp_err:.4f}px")
check("0.5px 噪声 fx 误差 <2%", noise_runs[0.5]["fx_rel"] < 0.02, f"{noise_runs[0.5]['fx_rel']:.4f}")
check("1px 噪声 fx 误差 <5%", noise_runs[1.0]["fx_rel"] < 0.05, f"{noise_runs[1.0]['fx_rel']:.4f}")
check("1px 噪声主点误差 <6px", noise_runs[1.0]["cx_abs"] < 6.0, f"{noise_runs[1.0]['cx_abs']:.2f}px")
check("畸变下 fx 误差 <0.1%", fx_d_err < 0.001, f"{fx_d_err:.6f}")
check("k1 绝对误差 <0.01", abs(k1_est - K1_TRUE) < 0.01, f"{abs(k1_est - K1_TRUE):.6f}")
check("畸变使直线弯曲 >5px", bow_dist > 5.0, f"{bow_dist:.2f}px")
check("去畸变后弯曲 <0.1px", bow_und < 0.1, f"{bow_und:.4f}px")
check("1px 噪声标定 RMS <2px", noise_runs[1.0]["rms"] < 2.0, f"{noise_runs[1.0]['rms']:.3f}px")

n_pass = int(sum(checks))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：6 个视角与网格
fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.6))
for ax, v, k in zip(axes.ravel(), views_clean, range(1, 7)):
    pts = v.reshape(NY, NX, 2)
    for row in pts:
        ax.plot(row[:, 0], row[:, 1], color=BLUE, lw=1.0, alpha=0.8)
    for col in pts.transpose(1, 0, 2):
        ax.plot(col[:, 0], col[:, 1], color=GREEN, lw=0.8, alpha=0.7)
    ax.scatter(v[:, 0], v[:, 1], s=14, color=ACCENT, zorder=3)
    ax.set_title(f"视角 {k}", fontsize=15)
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"张正友标定：6 个平面靶标视角；单应误差 {H_err:.4f}px", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "e140_views.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：噪声敏感性
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
labels = ["0px", "0.5px", "1px"]
fx_vals = [noise_runs[n]["fx_rel"] * 100 for n in (0.0, 0.5, 1.0)]
cx_vals = [noise_runs[n]["cx_abs"] for n in (0.0, 0.5, 1.0)]
axes[0].bar(labels, fx_vals, color=[GREEN, BLUE, ACCENT])
axes[0].set_title("焦距相对误差 / %", fontsize=15)
axes[0].set_ylabel("误差 / %", fontsize=15)
axes[0].grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
axes[1].bar(labels, cx_vals, color=[GREEN, BLUE, ACCENT])
axes[1].set_title("主点 cx 绝对误差 / px", fontsize=15)
axes[1].set_ylabel("误差 / px", fontsize=15)
axes[1].grid(True, axis="y", color=GRID, alpha=0.5, lw=0.6)
fig.suptitle("噪声越大，内参越飘；多视角和精调是解药", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "e140_noise.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：径向畸变前后
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
for ax, pts, title, color in [
    (axes[0], views_dist[0], f"畸变：直线弯 {bow_dist:.1f}px", ACCENT),
    (axes[1], und, f"去畸变：弯 {bow_und:.3f}px", GREEN)]:
    g = pts.reshape(NY, NX, 2)
    for row in g:
        ax.plot(row[:, 0], row[:, 1], color=color, lw=1.6)
    ax.scatter(pts[:, 0], pts[:, 1], s=12, color=INK, zorder=3)
    ax.set_title(title, fontsize=15)
    ax.set_xlim(0, IMG_W); ax.set_ylim(IMG_H, 0)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"径向畸变 k1={K1_TRUE:.2f} → 估计 {k1_est:.3f}", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "e140_distortion.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

for _name in ("e140_views.png", "e140_noise.png", "e140_distortion.png"):
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] Zhang：fx {K_est[0,0]:.2f}/fy {K_est[1,1]:.2f}/cx {K_est[0,2]:.2f}/cy {K_est[1,2]:.2f}；H 误差 {H_err:.4f}px")
print(f"[lab] 噪声：0/0.5/1px → fx 误差 {fx_vals[0]:.2f}/{fx_vals[1]:.2f}/{fx_vals[2]:.2f}%，cx 误差 {cx_vals[0]:.2f}/{cx_vals[1]:.2f}/{cx_vals[2]:.2f}px")
print(f"[lab] 畸变：k1 {k1_est:.4f}；fx 误差 {fx_d_err:.6f}；弯曲 {bow_dist:.2f}→{bow_und:.4f}px")

readings = dict(
    n_pass=n_pass, n_total=len(checks),
    camera=dict(width=IMG_W, height=IMG_H, K_true=Ktrue.tolist()),
    target=dict(nx=NX, ny=NY, n_points=int(len(obj)), n_views=len(POSES)),
    zhang=dict(K_est=K_est.tolist(), homography_error_px=H_err,
               fx_rel_error=fx_err, fy_rel_error=fy_err, principal_point_error_px=pp_err),
    noise={str(k): v for k, v in noise_runs.items()},
    distortion=dict(k1_true=K1_TRUE, k1_est=k1_est, fx_rel_error=fx_d_err,
                    bow_distorted_px=bow_dist, bow_undistorted_px=bow_und, cv_rms=rms_d),
    protocol=dict(synthetic=True, machine="local", general_benchmark=False,
                  implements="DLT homography + Zhang linear K + OpenCV calibrateCamera reference",
                  not_implemented=["real camera capture", "full bundle adjustment", "self-calibration"]),
    versions=dict(opencv=cv2.__version__, numpy=np.__version__),
)
out = ROOT / "outputs" / "e140_camera_calibration_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
