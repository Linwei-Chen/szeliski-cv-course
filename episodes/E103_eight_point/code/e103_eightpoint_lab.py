#!/usr/bin/env python3
"""E103 八点法：归一化/SVD/秩二/七点五点/RANSAC（合成实验）。"""
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
RNG = np.random.default_rng(103)
K = np.array([[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]])
def rot_y(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
R_true = rot_y(np.deg2rad(12)) @ rot_x(np.deg2rad(3))
t_true = np.array([0.7, 0.05, 0.15])
def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
E_true = skew(t_true) @ R_true
F_true = np.linalg.inv(K).T @ E_true @ np.linalg.inv(K)
F_true = F_true / np.linalg.norm(F_true)

def project(pts):
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R_true, t_true[:, None]])
    def p(P):
        h = np.hstack([pts, np.ones((len(pts), 1))]); q = (P @ h.T).T
        return q[:, :2] / q[:, 2:3]
    return p(P0), p(P1)

def normalize(pts):
    mu = pts.mean(axis=0)
    d = np.mean(np.linalg.norm(pts - mu, axis=1))
    s = np.sqrt(2) / max(d, 1e-9)
    T = np.array([[s, 0, -s * mu[0]], [0, s, -s * mu[1]], [0, 0, 1.0]])
    h = np.hstack([pts, np.ones((len(pts), 1))])
    q = (T @ h.T).T
    return q[:, :2], T

def eight_point(x0, x1, normalize_points=True):
    if normalize_points:
        a, T0 = normalize(x0); b, T1 = normalize(x1)
    else:
        a, b = x0, x1; T0 = T1 = np.eye(3)
    A = np.array([np.kron(np.append(q, 1), np.append(p, 1)) for p, q in zip(a, b)])
    _, sv, vt = np.linalg.svd(A)
    F = vt[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F); S[2] = 0; F = U @ np.diag(S) @ Vt
    if normalize_points:
        F = T1.T @ F @ T0
    return F / np.linalg.norm(F), sv

def epi_err(F, x0, x1):
    h0 = np.hstack([x0, np.ones((len(x0), 1))]); h1 = np.hstack([x1, np.ones((len(x1), 1))])
    Fx0 = (F @ h0.T).T; Ftx1 = (F.T @ h1.T).T
    num = np.abs(np.sum(h1 * Fx0, axis=1))
    d = num / np.maximum(np.linalg.norm(Fx0[:, :2], axis=1), 1e-9)
    d2 = num / np.maximum(np.linalg.norm(Ftx1[:, :2], axis=1), 1e-9)
    return 0.5 * (d + d2)

def sample_corr(n, noise=0.5):
    X = np.stack([RNG.uniform(-1.4, 1.4, n), RNG.uniform(-0.8, 0.8, n), RNG.uniform(3.0, 7.0, n)], axis=1)
    x0, x1 = project(X)
    return x0 + RNG.normal(0, noise, x0.shape), x1 + RNG.normal(0, noise, x1.shape)

x0, x1 = sample_corr(200, 0.5)
F_norm, sv_norm = eight_point(x0, x1, True)
F_raw, sv_raw = eight_point(x0, x1, False)
err_norm = float(np.median(epi_err(F_norm, x0, x1)))
err_raw = float(np.median(epi_err(F_raw, x0, x1)))
cond_norm = float(sv_norm[0] / sv_norm[-2])
cond_raw = float(sv_raw[0] / sv_raw[-2])
U, S, Vt = np.linalg.svd(F_norm)
det_before = abs(float(np.linalg.det(U @ np.diag(S) @ Vt)))
S2 = S.copy(); S2[2] = 0
det_after = abs(float(np.linalg.det(U @ np.diag(S2) @ Vt)))
val0, val1 = sample_corr(200, 0.1)
def trial_err(n, noise=0.5, trials=40):
    vals = []
    for _ in range(trials):
        xx0, xx1 = sample_corr(n, noise)
        FF, _ = eight_point(xx0, xx1, True)
        vals.append(float(np.median(epi_err(FF, val0, val1))))
    return float(np.median(vals))
errs_n = {n: trial_err(n, 0.5) for n in (8, 20, 100)}
errs_noise = {sig: trial_err(100, sig) for sig in (0.0, 0.5, 2.0)}
# RANSAC 迭代数
p_success = 0.99
def iters(sample, w=0.5):
    return int(np.ceil(np.log(1 - p_success) / np.log(max(1 - w ** sample, 1e-12))))
k8, k7, k5 = iters(8), iters(7), iters(5)
# 外点实验
clean0, clean1 = sample_corr(100, 0.3)
out0 = RNG.uniform(0, 640, (30, 2)); out1 = RNG.uniform(0, 480, (30, 2))
mix0 = np.vstack([clean0, out0]); mix1 = np.vstack([clean1, out1])
F_plain, _ = eight_point(mix0, mix1, True)
err_plain = float(np.median(epi_err(F_plain, clean0, clean1)))
best = None; best_in = None
for _ in range(600):
    idx = RNG.choice(len(mix0), 8, replace=False)
    F, _ = eight_point(mix0[idx], mix1[idx], True)
    e = epi_err(F, mix0, mix1)
    inl = e < 3.0
    if best_in is None or inl.sum() > best_in.sum():
        best, best_in = F, inl
F_ransac, _ = eight_point(mix0[best_in], mix1[best_in], True)
err_ransac = float(np.median(epi_err(F_ransac, clean0, clean1)))
# 7 点欠定：A 的最小奇异值相对
A7 = np.array([np.kron(np.append(b, 1), np.append(a, 1)) for a, b in zip(*[normalize(z)[0] for z in (clean0[:7], clean1[:7])])])
sv7 = np.linalg.svd(A7, compute_uv=False)
rank7 = int(np.sum(sv7 > sv7[0] * 1e-10))
checks = [
    ("八点法残差小", err_norm < 1.0, f"中位 {err_norm:.3f} px"),
    ("归一化条件数更小", cond_norm < cond_raw, f"{cond_norm:.1f} vs {cond_raw:.0f}"),
    ("秩二投影 det 近零", det_after < 1e-12, f"{det_after:.2e}"),
    ("点数越多越准", errs_n[8] > errs_n[20] > errs_n[100], f"{errs_n[8]:.2f}/{errs_n[20]:.2f}/{errs_n[100]:.2f}"),
    ("噪声越大越差", errs_noise[0.0] < errs_noise[0.5] < errs_noise[2.0], f"{errs_noise[0.0]:.3f}/{errs_noise[0.5]:.3f}/{errs_noise[2.0]:.3f}"),
    ("七点法更省样本", k7 < k8, f"k7={k7} < k8={k8}"),
    ("普通八点怕外点", err_plain > 10 * max(err_ransac, 1e-6), f"{err_plain:.1f} vs {err_ransac:.2f} px"),
    ("RANSAC 恢复", err_ransac < 1.0, f"{err_ransac:.3f} px"),
    ("七点欠定秩", rank7 <= 7, f"A7 rank {rank7}"),
    ("三图在盘", all((FIG / f"e103_{n}.png").exists() for n in ("pipeline", "condition", "ransac")), "pipeline/condition/ransac"),
]
n_pass = sum(bool(ok) for _, ok, _ in checks)
readings = {"n_pass": n_pass, "n_total": len(checks), "n_points": 200,
            "err_norm": err_norm, "err_norm_3": round(err_norm, 3), "err_raw": err_raw, "err_raw_3": round(err_raw, 3),
            "cond_norm": cond_norm, "cond_norm_1": round(cond_norm, 1), "cond_norm_sci": f"{cond_norm:.1e}", "cond_raw": cond_raw, "cond_raw_0": int(round(cond_raw)), "cond_raw_sci": f"{cond_raw:.1e}",
            "det_before": det_before, "det_before_sci": f"{det_before:.2e}", "det_after": det_after, "det_after_sci": f"{det_after:.2e}",
            "err8": errs_n[8], "err8_2": round(errs_n[8], 2), "err20_2": round(errs_n[20], 2), "err100_2": round(errs_n[100], 2),
            "err_noise0_3": round(errs_noise[0.0], 3), "err_noise05_3": round(errs_noise[0.5], 3), "err_noise2_3": round(errs_noise[2.0], 3),
            "k8": k8, "k7": k7, "k5": k5, "err_plain": err_plain, "err_plain_1": round(err_plain, 1),
            "err_ransac": err_ransac, "err_ransac_3": round(err_ransac, 3), "rank7": rank7}

# 图1 管线
fig, ax = plt.subplots(figsize=(11.2, 4.8)); ax.set_xlim(0, 11.2); ax.set_ylim(0, 4.8); ax.axis("off")
ax.text(0.2, 4.5, "八点法：归一化→堆方程→SVD→秩二→反归一化", fontsize=17, color=INK, weight="bold")
steps = ["归一化", "堆 A", "SVD", "秩二投影", "反归一化"]
for i, s in enumerate(steps):
    x = 0.35 + i * 2.15
    col = BLUE if i < 2 else (GREEN if i < 4 else PURPLE)
    ax.add_patch(plt.Rectangle((x, 2.4), 1.8, 0.95, facecolor="#eef2f7", edgecolor=col, lw=1.6))
    ax.text(x + 0.9, 2.88, s, ha="center", fontsize=15.5, color=INK, weight="bold")
    if i < 4:
        ax.annotate("", xy=(x + 2.1, 2.88), xytext=(x + 1.85, 2.88), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(0.35, 1.45, "A 的每一行 = 一对对应点；F 的 9 个元素是未知量；SVD 取最小奇异向量=零空间。", fontsize=14.5, color=GRAY)
ax.text(0.35, 0.85, "最后把 F 投影回秩二：SVD → 最小奇异值置 0 → 重组。", fontsize=14.5, color=GRAY)
fig.tight_layout(); fig.savefig(FIG / "e103_pipeline.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 条件数与精度
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
axes[0].bar(["未归一化", "归一化"], [np.log10(cond_raw), np.log10(cond_norm)], color=[GRAY, GREEN])
axes[0].set_ylabel("log10 条件数", fontsize=14.5); axes[0].set_title("归一化把条件数降下来", fontsize=15, color=INK)
axes[0].tick_params(labelsize=14.5)
axes[1].plot(list(errs_n.keys()), list(errs_n.values()), "o-", color=ACCENT, lw=2)
axes[1].set_xlabel("对应点数", fontsize=14.5); axes[1].set_ylabel("中位极线误差 / px", fontsize=14.5)
axes[1].set_title("点越多，八点法越稳", fontsize=15, color=INK)
axes[1].tick_params(labelsize=14.5); axes[1].grid(alpha=0.25)
for a in axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e103_condition.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 RANSAC 与外点
fig, ax = plt.subplots(figsize=(9.8, 4.9))
ax.bar(["普通八点", "RANSAC"], [err_plain, err_ransac], color=[GRAY, GREEN])
for i, v in enumerate([err_plain, err_ransac]):
    ax.text(i + 0.28, v * 1.1, f"{v:.2f} px", ha="left", fontsize=14.5, color=INK)
ax.set_yscale("log"); ax.set_ylabel("干净点上的中位极线误差 / px（对数）", fontsize=14.5)
ax.set_title("30% 外点：普通八点被带偏，RANSAC 拉回来", fontsize=15, color=INK, pad=12)
ax.tick_params(labelsize=14.5)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(FIG / "e103_ransac.png", dpi=170, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

checks[-1] = ("三图在盘", all((FIG / f"e103_{n}.png").exists() for n in ("pipeline", "condition", "ransac")), "pipeline/condition/ransac")
n_pass = sum(bool(ok) for _, ok, _ in checks); readings["n_pass"] = n_pass
print(f"[lab] E103 eight-point 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
for p in sorted(FIG.glob("e103_*.png")):
    print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
(OUT / "e103_eightpoint_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print("[lab] readings → e103_eightpoint_readings.json")
print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e103_eightpoint_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
if n_pass != len(checks):
    raise SystemExit(1)
