#!/usr/bin/env python3
"""E110 COLMAP 上机：合成 12 视角 → SIFT → 匹配 → 增量重建（真实 COLMAP 运行）。"""
import hashlib, json, os, pathlib, shutil, sqlite3, sys
from datetime import datetime
os.environ.setdefault("GLOG_minloglevel", "2")
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pycolmap
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
WORK = OUT / "e110_colmap"; IMGS = WORK / "images"; SPARSE = WORK / "sparse"
if WORK.exists(): shutil.rmtree(WORK)
IMGS.mkdir(parents=True); SPARSE.mkdir(parents=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
W, H, F, CX, CY = 800, 600, 700.0, 400.0, 300.0
N_VIEW, HALF_ANGLE = 12, 28.0
TARGET = np.array([0, -0.1, 2.2])
rng = np.random.default_rng(110)

def texture(seed, tint):
    r = np.random.default_rng(seed)
    def octave(n):
        a = r.random((n, n, 3)).astype(np.float32)
        im = Image.fromarray((a * 255).astype(np.uint8)).resize((512, 512), Image.BICUBIC)
        return np.asarray(im).astype(np.float32) / 255
    t = 0.55 * octave(48) + 0.30 * octave(144) + 0.15 * r.random((512, 512, 3)).astype(np.float32)
    t = (t - t.mean()) / (t.std() + 1e-6) * 0.22 + 0.55
    t = np.clip(t, 0, 1) * np.array(tint, dtype=np.float32)
    return np.clip(t, 0, 1)

TEX = {
    "back": texture(1, (0.95, 0.9, 0.82)), "left": texture(2, (0.8, 0.88, 0.95)),
    "floor": texture(3, (0.9, 0.82, 0.7)), "box0": texture(4, (0.95, 0.7, 0.55)),
    "box1": texture(5, (0.65, 0.85, 0.7)), "box2": texture(6, (0.75, 0.75, 0.95)),
}
def sample(tex, u, v):
    h, w, _ = tex.shape
    x = np.clip((u * w).astype(int), 0, w - 1); y = np.clip((v * h).astype(int), 0, h - 1)
    return tex[y, x]
def render(eye):
    fwd = TARGET - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0, 1, 0.])); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    u = (np.arange(W) + 0.5 - CX) / F; v = -(np.arange(H) + 0.5 - CY) / F
    uu, vv = np.meshgrid(u, v)
    d = fwd[None, None, :] + uu[..., None] * right[None, None, :] + vv[..., None] * up[None, None, :]
    d /= np.linalg.norm(d, axis=2, keepdims=True); o = eye[None, None, :]
    best = np.full((H, W), np.inf); img = np.zeros((H, W, 3), np.float32); img[:] = 0.08
    def texuv(name, p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        if name == "back": return (x + 3) / 6, (y + 1.5) / 4
        if name == "floor": return (x + 3) / 6, z / 6
        if name == "left": return z / 6, (y + 1.5) / 4
        return (x + 3) / 6, (y + 1.5) / 4
    def hit_plane(p0, n, cond, name):
        nonlocal best, img
        t = np.sum((p0 - o) * n, axis=2) / (np.sum(d * n, axis=2) + 1e-9)
        p = o + t[..., None] * d; ok = (t > 0) & (t < best) & cond(p)
        if not ok.any(): return
        uu2, vv2 = texuv(name, p); col = sample(TEX[name], uu2, vv2)
        shade = 0.55 + 0.45 * max(float(np.dot(n, np.array([-0.3, 0.9, 0.35]))), 0.0)
        img[ok] = np.clip(col[ok] * shade, 0, 1); best[ok] = t[ok]
    hit_plane(np.array([0, 0, 0.]), np.array([0, 0, 1.]),
              lambda p: (np.abs(p[..., 0]) <= 3) & (p[..., 1] >= -1.5) & (p[..., 1] <= 2.5), "back")
    hit_plane(np.array([0, -1.5, 0.]), np.array([0, 1, 0.]),
              lambda p: (np.abs(p[..., 0]) <= 3) & (p[..., 2] >= 0) & (p[..., 2] <= 6), "floor")
    hit_plane(np.array([-3, 0, 0.]), np.array([1, 0, 0.]),
              lambda p: (p[..., 1] >= -1.5) & (p[..., 1] <= 2.5) & (p[..., 2] >= 0) & (p[..., 2] <= 6), "left")
    boxes = [("box0", np.array([-0.6, -1.5, 1.8]), np.array([0.6, 0.0, 3.0])),
             ("box1", np.array([0.8, -1.5, 3.1]), np.array([1.8, -0.6, 4.1])),
             ("box2", np.array([-1.6, -1.5, 3.6]), np.array([-0.8, -0.5, 4.4]))]
    for name, mn, mx in boxes:
        inv = 1 / (d + 1e-9); t1 = (mn - o) * inv; t2 = (mx - o) * inv
        tmin = np.max(np.minimum(t1, t2), axis=2); tmax = np.min(np.maximum(t1, t2), axis=2)
        ok = (tmax >= tmin) & (tmin > 0) & (tmin < best)
        p = o + tmin[..., None] * d; c = (mn + mx) / 2; sz = (mx - mn); rel = (p - c) / sz + 0.5
        ax = np.argmax(np.stack([np.abs(rel[..., i] - 0.5) for i in range(3)], axis=2), axis=2)
        uu2 = np.where(ax == 0, rel[..., 2], np.where(ax == 1, rel[..., 0], rel[..., 0]))
        vv2 = np.where(ax == 0, rel[..., 1], np.where(ax == 1, rel[..., 2], rel[..., 1]))
        col = sample(TEX[name], uu2, vv2)
        img[ok] = np.clip(col[ok] * 0.85, 0, 1); best[ok] = tmin[ok]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

GT = {}
for i, deg in enumerate(np.linspace(-HALF_ANGLE, HALF_ANGLE, N_VIEW)):
    th = np.deg2rad(deg)
    eye = np.array([7 * np.sin(th), 0.9 + 0.15 * np.sin(th * 2), 7 * np.cos(th)])
    Image.fromarray(render(eye)).save(IMGS / f"view_{i:02d}.jpg", quality=92)
    GT[f"view_{i:02d}.jpg"] = eye

# 真实 COLMAP 运行：SIFT → 全对匹配 → 增量重建
reader = pycolmap.ImageReaderOptions(); reader.camera_model = "PINHOLE"; reader.camera_params = "700,700,400,300"
extract = pycolmap.FeatureExtractionOptions(); extract.use_gpu = False
db = WORK / "database.sqlite"
pycolmap.extract_features(db, IMGS, camera_mode=pycolmap.CameraMode.SINGLE,
                          reader_options=reader, extraction_options=extract, device=pycolmap.Device.cpu)
pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
models = pycolmap.incremental_mapping(db, IMGS, SPARSE)
if not models:
    print("[FAIL] COLMAP 未产生任何模型"); sys.exit(1)
rec = max(models.values(), key=lambda m: m.num_reg_images())
rec.write(SPARSE / "0")

with sqlite3.connect(db) as con:
    n_pairs = con.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
est, gtd = {}, {}
for img in rec.images.values():
    P = img.cam_from_world(); R = P.rotation.matrix(); est[img.name] = -R.T @ np.array(P.translation)
    gtd[img.name] = GT[img.name]
names = sorted(GT); X = np.array([est[n] for n in names]); Y = np.array([gtd[n] for n in names])
mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
C = Yc.T @ Xc / len(X); U, S, Vt = np.linalg.svd(C); dsign = np.sign(np.linalg.det(U @ Vt))
R = U @ np.diag([1, 1, dsign]) @ Vt; var = np.mean(np.sum(Xc ** 2, axis=1))
scale = np.trace(np.diag(S) @ np.diag([1, 1, dsign])) / var
t = my - scale * R @ mx; Xa = (scale * R @ X.T).T + t
err = np.linalg.norm(Xa - Y, axis=1)
cam_rms = float(np.sqrt(np.mean(err ** 2))); cam_max = float(err.max())
focal = np.array(rec.cameras[1].params[:2]); focal_true = np.array([F, F])
focal_err = float(np.max(np.abs(focal - focal_true) / focal_true * 100))
n_obs = sum(img.num_points3D for img in rec.images.values())
feats = [img.num_points2D() for img in rec.images.values()]
tris = [img.num_points3D for img in rec.images.values()]
mean_track = float(rec.compute_mean_track_length()); reproj = float(rec.compute_mean_reprojection_error())
n_reg = rec.num_reg_images(); n_pts = rec.num_points3D()

# 图1 输入接触表
fig, axes = plt.subplots(3, 4, figsize=(14.0, 7.8))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(Image.open(IMGS / f"view_{i:02d}.jpg")); ax.axis("off")
    ax.set_title(f"视角 {i:02d}", fontsize=18, color=INK, pad=3)
fig.suptitle("12 个视角：绕场景一圈，重叠足够才能匹配", fontsize=20, color=INK, weight="bold")
fig.tight_layout(); fig.savefig(FIG / "e110_views.png", dpi=150, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 稀疏重建 + 相机轨迹（对齐到真值坐标）
P3 = np.array([p.xyz for p in rec.points3D.values()])
P3a = (scale * R @ P3.T).T + t
fig = plt.figure(figsize=(12.2, 5.4))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.scatter(P3a[::4, 0], P3a[::4, 1], P3a[::4, 2], s=0.8, color=GRAY, alpha=0.35, label="稀疏点")
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], "-", color=INK, lw=2, label="真值轨迹")
ax.plot(Xa[:, 0], Xa[:, 1], Xa[:, 2], "--", color=ACCENT, lw=2, label="COLMAP 对齐后")
ax.set_xlabel("X", fontsize=14.5); ax.set_ylabel("Y", fontsize=14.5); ax.set_zlabel("Z", fontsize=14.5)
ax.tick_params(labelsize=12.5); ax.set_title("3D 视图：点云 + 12 个相机", fontsize=15, color=INK)
ax.legend(fontsize=12.5, frameon=False, loc="upper right")
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(P3a[::5, 0], P3a[::5, 2], s=1.0, color=GRAY, alpha=0.35)
ax2.plot(Y[:, 0], Y[:, 2], "-o", color=INK, ms=4, lw=1.8, label="真值")
ax2.plot(Xa[:, 0], Xa[:, 2], "--x", color=ACCENT, ms=5, lw=1.5, label="COLMAP")
ax2.set_xlabel("X", fontsize=14.5); ax2.set_ylabel("Z", fontsize=14.5); ax2.tick_params(labelsize=14.5)
ax2.set_title("俯视图：轨迹对齐后重合", fontsize=15, color=INK); ax2.grid(alpha=0.2)
ax2.legend(fontsize=12.8, frameon=False)
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e110_reconstruction.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 每个视角的误差与三角化点数
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
idx = np.arange(N_VIEW)
axes[0].bar(idx, err * 1000, color=[ACCENT if e == err.max() else BLUE for e in err])
axes[0].axhline(cam_rms * 1000, color=GREEN, ls="--", lw=1.5, label=f"RMS {cam_rms*1000:.1f}")
axes[0].set_xlabel("视角编号", fontsize=14.5); axes[0].set_ylabel("相机中心误差 / ×10⁻³ 场景单位", fontsize=14.5)
axes[0].set_title("对齐后相机中心误差", fontsize=15, color=INK); axes[0].tick_params(labelsize=13.5)
axes[0].legend(fontsize=12.5, frameon=False)
axes[1].bar(idx, tris, color=[ACCENT if i == int(np.argmin(tris)) else GREEN for i in idx])
axes[1].set_xlabel("视角编号", fontsize=14.5); axes[1].set_ylabel("三角化点数", fontsize=14.5)
axes[1].set_title("边缘视角点数少，但仍能注册", fontsize=15, color=INK); axes[1].tick_params(labelsize=13.5)
for sp in ("top", "right"):
    axes[0].spines[sp].set_visible(False); axes[1].spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e110_errors.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks = [
    ("12 张输入图在盘", len(list(IMGS.glob("*.jpg"))) == N_VIEW, f"{len(list(IMGS.glob('*.jpg')))} 张"),
    ("COLMAP 全部注册", n_reg == N_VIEW, f"{n_reg}/{N_VIEW}"),
    ("匹配对足够多", n_pairs >= 30, f"{n_pairs} 对"),
    ("稀疏点足够多", n_pts >= 5000, f"{n_pts} 点"),
    ("平均重投影误差小", reproj < 0.5, f"{reproj:.4f} px"),
    ("焦距误差小", focal_err < 2.0, f"{focal_err:.2f}%"),
    ("相机中心 RMS 小", cam_rms < 0.05, f"{cam_rms:.4f}"),
    ("最差相机中心误差小", cam_max < 0.05, f"{cam_max:.4f}"),
    ("平均轨迹长度够", mean_track >= 3.0, f"{mean_track:.2f}"),
    ("三图在盘", all((FIG / f"e110_{n}.png").exists() for n in ("views", "reconstruction", "errors")), "views/reconstruction/errors"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {
    "n_pass": n_pass, "n_total": len(checks), "n_images": N_VIEW, "n_registered": n_reg,
    "n_pairs": n_pairs, "n_points": n_pts, "n_obs": n_obs, "mean_track": mean_track,
    "mean_track_2": round(mean_track, 2), "reproj": reproj, "reproj_4": round(reproj, 4),
    "focal_fx": float(focal[0]), "focal_fy": float(focal[1]), "focal_fx_2": round(float(focal[0]), 2),
    "focal_fy_2": round(float(focal[1]), 2), "focal_err_pct": focal_err, "focal_err_pct_2": round(focal_err, 2),
    "cam_rms": cam_rms, "cam_rms_4": round(cam_rms, 4), "cam_max": cam_max, "cam_max_4": round(cam_max, 4),
    "scale": scale, "scale_4": round(scale, 4), "n_sift_total": int(sum(feats)), "n_sift_mean": int(np.mean(feats)),
    "min_tri": int(min(tris)), "min_tri_view": int(np.argmin(tris)), "view00_tri": int(tris[0]),
    "view00_feat": int(feats[0]), "colmap_version": pycolmap.__version__,
}
print(f"[lab] E110 COLMAP 上机开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] COLMAP/pycolmap 版本 {pycolmap.__version__}｜输入 {N_VIEW} 视角 {W}x{H}｜相机先验 f=700, c=(400,300)")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] 注册 {n_reg}/{N_VIEW}｜匹配 {n_pairs} 对｜稀疏点 {n_pts}｜观测 {n_obs}｜平均轨迹 {mean_track:.2f}")
print(f"[lab] 重投影 {reproj:.4f} px｜焦距 {focal[0]:.2f}/{focal[1]:.2f}（真值 700）｜误差 {focal_err:.2f}%")
print(f"[lab] 相机中心对齐后 RMS {cam_rms:.4f}｜最大 {cam_max:.4f}｜相似尺度 {scale:.4f}")
print(f"[lab] 三角化最少视角 {int(np.argmin(tris))} 号 {min(tris)} 点｜0 号 {tris[0]} 点/{feats[0]} 特征")
for p in sorted(FIG.glob("e110_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e110_colmap_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e110_colmap_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e110_colmap_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
