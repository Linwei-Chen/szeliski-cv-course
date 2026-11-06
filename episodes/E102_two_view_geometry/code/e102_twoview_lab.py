#!/usr/bin/env python3
"""E102 双视图几何：对极约束/E/F/SVD/cheirality（合成实验）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE, GRAY, GOLD = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d", "#6b7280", "#b8860b"
RNG = np.random.default_rng(102)
N = 60
K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
def rot_y(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
R_true = rot_y(np.deg2rad(12)) @ rot_x(np.deg2rad(3))
t_true = np.array([0.7, 0.05, 0.15])
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
E_true = skew(t_true) @ R_true
X = np.stack([RNG.uniform(-1.4, 1.4, N), RNG.uniform(-0.8, 0.8, N), RNG.uniform(3.0, 7.0, N)], axis=1)
P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
P1 = K @ np.hstack([R_true, t_true[:, None]])
def project(P, pts):
    h = np.hstack([pts, np.ones((len(pts), 1))])
    q = (P @ h.T).T
    return q[:, :2] / q[:, 2:3], q[:, 2]
x0, z0 = project(P0, X)
x1, z1 = project(P1, X)
def norm(px):
    h = np.hstack([px, np.ones((len(px), 1))])
    q = (np.linalg.inv(K) @ h.T).T
    return q / q[:, 2:3]
n0, n1 = norm(x0), norm(x1)
res = np.array([n1[i] @ E_true @ n0[i] for i in range(N)])
line = (E_true @ n0.T).T
dist = np.abs(np.sum(n1 * line, axis=1)) / np.linalg.norm(line[:, :2], axis=1)
detE = float(np.linalg.det(E_true))
F_true = np.linalg.inv(K).T @ E_true @ np.linalg.inv(K)
pix_res = np.array([np.hstack([x1[i], 1]) @ F_true @ np.hstack([x0[i], 1]) for i in range(N)])

U, S, Vt = np.linalg.svd(E_true)
if np.linalg.det(U) < 0: U = -U
if np.linalg.det(Vt) < 0: Vt = -Vt
W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
R1 = U @ W @ Vt; R2 = U @ W.T @ Vt; tv = U[:, 2]
cands = [(R1, tv), (R1, -tv), (R2, tv), (R2, -tv)]
def triangulate(P0, P1, p0, p1):
    out = []
    for a, b in zip(p0, p1):
        A = np.vstack([a[0] * P0[2] - P0[0], a[1] * P0[2] - P0[1],
                       b[0] * P1[2] - P1[0], b[1] * P1[2] - P1[1]])
        _, _, vt = np.linalg.svd(A)
        Ph = vt[-1]; out.append(Ph[:3] / Ph[3])
    return np.array(out)
counts = []
selected = None
for Rc, tc in cands:
    Pc = K @ np.hstack([Rc, tc[:, None]])
    Xc = triangulate(P0, Pc, x0, x1)
    front0 = Xc[:, 2] > 0
    front1 = (Xc @ Rc.T + tc)[:, 2] > 0
    cnt = int(np.sum(front0 & front1))
    counts.append(cnt)
    if selected is None or cnt > counts[selected]:
        selected = len(counts) - 1
Rsel, tsel = cands[selected]
ang = np.degrees(np.arccos(np.clip((np.trace(Rsel.T @ R_true) - 1) / 2, -1, 1)))
cos_t = float(abs(np.dot(tsel, t_true) / (np.linalg.norm(tsel) * np.linalg.norm(t_true))))
# 噪声
xn0 = x0 + RNG.normal(0, 1.0, x0.shape); xn1 = x1 + RNG.normal(0, 1.0, x1.shape)
nn0, nn1 = norm(xn0), norm(xn1)
res_noisy = np.array([nn1[i] @ E_true @ nn0[i] for i in range(N)])
noisy_ratio = float(np.mean(np.abs(res_noisy)) / max(np.mean(np.abs(res)), 1e-12))

checks = [
    ("极线约束残差小", float(np.max(np.abs(res))) < 1e-10, f"max {np.max(np.abs(res)):.2e}"),
    ("E 秩为二", abs(detE) < 1e-12, f"det {detE:.2e}"),
    ("极线过对应点", float(np.max(dist)) < 1e-10, f"max {np.max(dist):.2e}"),
    ("SVD 复原 R", ang < 0.01, f"{ang:.4f}°"),
    ("t 方向复原", cos_t > 0.9999, f"cos {cos_t:.6f}"),
    ("cheirality 选对四解", selected == 0 or counts[selected] == N, f"四解前沿点数 {counts}"),
    ("F 像素约束成立", float(np.max(np.abs(pix_res))) < 1e-8, f"max {np.max(np.abs(pix_res)):.2e}"),
    ("噪声使残差变大", noisy_ratio > 100, f"×{noisy_ratio:.0f}"),
    ("三图在盘", all((FIG / f"e102_{n}.png").exists() for n in ("epipolar", "ef", "cheirality")), "epipolar/ef/cheirality"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {"n_pass": n_pass, "n_total": len(checks), "n_points": N,
            "max_epi_res": float(np.max(np.abs(res))), "max_epi_res_sci": f"{np.max(np.abs(res)):.2e}",
            "det_E": detE, "max_line_dist": float(np.max(dist)),
            "R_err_deg": ang, "R_err_deg_4": round(ang, 4), "cos_t": cos_t, "cos_t_6": round(cos_t, 6),
            "counts": counts, "selected": selected, "true_candidate": 0,
            "max_pix_res": float(np.max(np.abs(pix_res))), "noisy_ratio": noisy_ratio, "noisy_ratio_0": round(noisy_ratio),
            "true_front_count": N}


def box(ax, x, y, w, h, fc, ec, title, sub="", fs=15, subfs=14.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=fs, color=INK, weight="bold")
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=subfs, color=GRAY)


# ── 图 1：对极几何示意 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(11.2, 5.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")
ax.text(0.2, 5.3, "对极几何：两台相机、一个三维点、一个平面", fontsize=17, color=INK, weight="bold")
# 相机中心与基线
c0, c1 = (1.2, 1.0), (7.2, 1.0)
ax.plot([c0[0], c1[0]], [c0[1], c1[1]], "-", color=INK, lw=2)
ax.plot(*c0, "o", color=BLUE, ms=12); ax.plot(*c1, "o", color=BLUE, ms=12)
ax.text(c0[0]-0.1, c0[1]-0.35, "c0", fontsize=15, color=BLUE)
ax.text(c1[0]-0.1, c1[1]-0.35, "c1", fontsize=15, color=BLUE)
ax.text(4.2, 0.82, "基线 t", fontsize=14.5, color=INK)
# 图像平面（用斜线表示）
ax.plot([0.5, 1.9], [1.6, 2.9], color=GRAY, lw=2)
ax.plot([6.5, 7.9], [1.6, 2.9], color=GRAY, lw=2)
ax.text(0.35, 2.95, "图像 0", fontsize=14.5, color=GRAY)
ax.text(7.55, 3.0, "图像 1", fontsize=14.5, color=GRAY)
# 三维点与射线
P = (4.3, 3.8)
ax.plot(*P, "o", color=ACCENT, ms=11)
ax.text(P[0]+0.12, P[1]+0.1, "P", fontsize=16, color=ACCENT)
ax.plot([c0[0], P[0]], [c0[1], P[1]], "--", color=BLUE, lw=1.6)
ax.plot([c1[0], P[0]], [c1[1], P[1]], "--", color=BLUE, lw=1.6)
# 极线
ax.plot([1.0, 2.1], [2.7, 1.7], color=PURPLE, lw=2)
ax.plot([6.4, 7.5], [2.7, 1.7], color=PURPLE, lw=2)
ax.text(0.9, 2.35, "l0", fontsize=14.5, color=PURPLE)
ax.text(7.45, 2.35, "l1", fontsize=14.5, color=PURPLE)
ax.text(3.4, 4.35, "对极平面：c0、c1、P 决定的平面", fontsize=14.5, color=GREEN)
box(ax, 2.7, 0.05, 5.6, 0.52, "#e8f0e5", GREEN, "x1^T E x0 = 0 ｜ l1 = E x0 ｜ F = K1^-T E K0^-1", "", fs=14.5)
fig.tight_layout(); fig.savefig(FIG / "e102_epipolar.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# ── 图 2：E 与 F 的关系 ───────────────────────────────────
fig, ax = plt.subplots(figsize=(11.2, 5.2))
ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.2); ax.axis("off")
ax.text(0.2, 4.9, "本质矩阵 E 与基础矩阵 F：同一约束，两套坐标", fontsize=17, color=INK, weight="bold")
box(ax, 0.4, 2.9, 3.0, 1.1, "#e6ebf2", BLUE, "归一化坐标", "x̂ = K⁻¹ x", fs=16)
box(ax, 4.1, 2.9, 3.0, 1.1, "#f7e7e2", ACCENT, "本质矩阵 E", "E = [t]× R", fs=16)
box(ax, 7.8, 2.9, 3.0, 1.1, "#e8f0e5", GREEN, "像素坐标", "x = K x̂", fs=16)
box(ax, 4.1, 1.1, 3.0, 1.0, "#f0e6f5", PURPLE, "基础矩阵 F", "F = inv(K1)^T · E · inv(K0)", fs=16)
for x1, x2 in ((3.4, 4.1), (7.1, 7.8)):
    ax.add_patch(FancyArrowPatch((x1, 3.45), (x2, 3.45), arrowstyle="-|>", mutation_scale=16, color=GRAY, lw=1.8))
ax.add_patch(FancyArrowPatch((5.6, 2.85), (5.6, 2.15), arrowstyle="-|>", mutation_scale=16, color=PURPLE, lw=1.8))
ax.text(0.4, 0.35, "E 要求已知内参 K；F 直接从像素学，内参未知也能用。两者都秩二，极点=零空间向量。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e102_ef.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# ── 图 3：四解与 cheirality ────────────────────────────────
fig, ax = plt.subplots(figsize=(9.6, 5.0))
labels = ["解1", "解2", "解3", "解4"]
colors = [GREEN if i == selected else GRAY for i in range(4)]
bars = ax.bar(labels, counts, color=colors)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, c + 1, str(c), ha="center", fontsize=14.5, color=INK)
ax.set_ylabel("两相机前方点数", fontsize=14.5); ax.set_ylim(0, N + 8)
ax.set_title("SVD 给四组 R/t；用 cheirality 选“点都在相机前方”的那组", fontsize=15, color=INK)
ax.tick_params(labelsize=13.5)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.55, 0.93, f"选中解 {selected+1}：R 误差 {ang:.4f}°｜t 方向 cos={cos_t:.5f}",
        transform=ax.transAxes, fontsize=14.5, color=ACCENT, va="top")
fig.tight_layout(); fig.savefig(FIG / "e102_cheirality.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e102_{n}.png").exists() for n in ("epipolar", "ef", "cheirality")), "epipolar/ef/cheirality")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E102 two-view 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e102_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e102_twoview_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e102_twoview_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e102_twoview_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
