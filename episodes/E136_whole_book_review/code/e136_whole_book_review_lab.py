#!/usr/bin/env python3
"""E136 全书总复盘 lab：同一最小二乘骨架解三个视觉任务 + 一张可机检的“全书地图”。

边界：三个任务都是合成数据（直线拟合、仿射点对齐、球面光度立体）；不是真实系统；
地图节点是本书章节，边是教学依赖，不是书里 Figure 1.12 的复制。
"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["axes.labelsize"] = 14

print(f"[lab] E136 全书复盘实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：三个任务均为合成数据；地图是教学依赖图，不是教材 Figure 1.12 复制")

rng = np.random.default_rng(136)
checks = []
def check(name, ok, detail):
    checks.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")

# ── 任务1：直线拟合（几何） ─────────────────────────────────────────────
N1 = 50
x1 = rng.uniform(-1.0, 1.0, N1)
y1 = 0.6 * x1 + 0.2 + rng.normal(0.0, 0.05, N1)
A1 = np.column_stack([x1, np.ones_like(x1)])
(a_ols, b_ols), *_ = np.linalg.lstsq(A1, y1, rcond=None)
d_ols = np.abs(a_ols * x1 - y1 + b_ols) / np.sqrt(a_ols ** 2 + 1)
mse_ols = float(np.mean(d_ols ** 2))
# TLS：中心化后最小奇异向量
P = np.column_stack([x1, y1]); Pc = P - P.mean(axis=0)
_, _, Vt = np.linalg.svd(Pc, full_matrices=False)
n_tls = Vt[-1]; d_tls = np.abs(Pc @ n_tls); mse_tls = float(np.mean(d_tls ** 2))
# 用于画图
xt = np.linspace(-1, 1, 100)
check("直线拟合斜率误差 <0.05", abs(a_ols - 0.6) < 0.05, f"a={a_ols:.4f}（真值 0.6）")
check("直线拟合截距误差 <0.05", abs(b_ols - 0.2) < 0.05, f"b={b_ols:.4f}（真值 0.2）")
check("TLS 正交误差不高于 OLS", mse_tls <= mse_ols * 1.02, f"TLS {mse_tls:.5f} vs OLS {mse_ols:.5f}")

# ── 任务2：仿射点对齐（运动/配准） ─────────────────────────────────────
N2 = 24
p2 = rng.uniform(0.0, 1.0, (N2, 2))
A_true = np.array([[1.08, 0.18], [-0.12, 0.94]])
t_true = np.array([0.15, -0.08])
q2 = p2 @ A_true.T + t_true + rng.normal(0.0, 0.01, (N2, 2))
# 每个点: qx = a11*x + a12*y + tx; qy = a21*x + a22*y + ty
M = np.zeros((2 * N2, 6)); rhs = np.zeros(2 * N2)
M[0::2, 0] = p2[:, 0]; M[0::2, 1] = p2[:, 1]; M[0::2, 2] = 1.0; rhs[0::2] = q2[:, 0]
M[1::2, 3] = p2[:, 0]; M[1::2, 4] = p2[:, 1]; M[1::2, 5] = 1.0; rhs[1::2] = q2[:, 1]
coef2, *_ = np.linalg.lstsq(M, rhs, rcond=None)
A_est = np.array([[coef2[0], coef2[1]], [coef2[3], coef2[4]]])
t_est = np.array([coef2[2], coef2[5]])
param_err = float(np.max(np.abs(np.concatenate([A_est.ravel() - A_true.ravel(), t_est - t_true]))))
res2 = M @ coef2 - rhs
res2_rms = float(np.sqrt(np.mean(res2 ** 2)))
check("仿射对齐最大参数误差 <0.02", param_err < 0.02, f"{param_err:.4f}")
check("仿射对齐残差 RMS <0.02", res2_rms < 0.02, f"{res2_rms:.4f}")

# ── 任务3：光度立体（三维形状） ───────────────────────────────────────
H3 = W3 = 96
yy, xx = np.mgrid[-1:1:complex(H3), -1:1:complex(W3)]
rr = xx ** 2 + yy ** 2
mask = rr < 0.78 ** 2
nz = np.zeros_like(xx); nz[mask] = np.sqrt(1.0 - rr[mask])
Nmap = np.stack([xx, yy, nz], axis=-1)
Nmap[~mask] = np.array([0.0, 0.0, 1.0])
albedo = 0.55 + 0.30 * (0.5 + 0.5 * np.sin(5 * xx) * np.cos(4 * yy))
albedo = np.clip(albedo, 0.15, 1.0)
L = np.array([[-0.5, -0.5, 0.7071], [0.6, -0.3, 0.7416], [0.0, 0.7, 0.7141]])
L = L / np.linalg.norm(L, axis=1, keepdims=True)
shading = np.maximum(np.einsum("ijk,lk->ijl", Nmap, L), 0.0)
I = albedo[..., None] * shading + rng.normal(0.0, 0.005, (H3, W3, 3))
I = np.clip(I, 0.0, 1.0)
g = np.linalg.solve(L, I.reshape(-1, 3).T).T.reshape(H3, W3, 3)
gnorm = np.linalg.norm(g, axis=-1)
valid = mask & (gnorm > 0.05) & np.all(shading > 0.05, axis=-1)
Nest = g / np.maximum(gnorm[..., None], 1e-8)
cosang = np.clip(np.sum(Nest[valid] * Nmap[valid], axis=1), -1, 1)
ang_err = float(np.degrees(np.mean(np.arccos(cosang))))
albedo_est = gnorm
alb_corr = float(np.corrcoef(albedo[valid], albedo_est[valid])[0, 1])
valid_frac = float(valid.sum() / mask.sum())
check("光度立体平均法向误差 <2°", ang_err < 2.0, f"{ang_err:.3f}°")
check("光度立体反照率相关 >0.99", alb_corr > 0.99, f"{alb_corr:.4f}")
check("有效像素比例 >0.70", valid_frac > 0.70, f"{valid_frac*100:.1f}%")

# ── 任务4：噪声敏感性（三任务共享同一最小二乘骨架） ───────────────────
noise_levels = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
slopes = {}; curves = {}
def trial_line(sigma):
    x = rng.uniform(-1, 1, N1); y = 0.6 * x + 0.2 + rng.normal(0, sigma, N1)
    c, *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)
    return abs(c[0] - 0.6)
def trial_affine(sigma):
    q = p2 @ A_true.T + t_true + rng.normal(0, sigma, (N2, 2))
    rhs = np.zeros(2 * N2); rhs[0::2] = q[:, 0]; rhs[1::2] = q[:, 1]
    c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    Ae = np.array([[c[0], c[1]], [c[3], c[4]]]); te = np.array([c[2], c[5]])
    return float(np.linalg.norm(Ae - A_true) + np.linalg.norm(te - t_true))
def trial_photo(sigma):
    In = np.clip(albedo[..., None] * shading + rng.normal(0, sigma, (H3, W3, 3)), 0, 1)
    gg = np.linalg.solve(L, In.reshape(-1, 3).T).T.reshape(H3, W3, 3)
    nn = gg / np.maximum(np.linalg.norm(gg, axis=-1)[..., None], 1e-8)
    ca = np.clip(np.sum(nn[valid] * Nmap[valid], axis=1), -1, 1)
    return float(np.degrees(np.mean(np.arccos(ca))))
for name, fn in [("直线", trial_line), ("仿射", trial_affine), ("光度立体", trial_photo)]:
    vals = []
    for sig in noise_levels:
        vals.append(float(np.mean([fn(sig) for _ in range(8)])))
    curves[name] = vals
    slopes[name] = float(np.polyfit(np.log(noise_levels), np.log(np.maximum(vals, 1e-12)), 1)[0])
    check(f"噪声敏感性 {name} 斜率 >0.4", slopes[name] > 0.4, f"slope={slopes[name]:.2f}")

# ── 任务5：全书地图（可机检的 DAG） ───────────────────────────────────
layers = {
    "基础": ["1 引言", "2 成像", "3 图像处理"],
    "建模": ["4 拟合/优化", "5 深度学习"],
    "识别": ["6 识别", "7 特征"],
    "运动": ["8 对齐/拼接", "9 运动", "10 计算摄影"],
    "三维": ["11 SfM/SLAM", "12 深度", "13 重建"],
    "渲染": ["14 IBR", "15 结论"],
}
appendices = ["A 线性代数", "B 贝叶斯", "C 数据/软件"]
all_nodes = [n for v in layers.values() for n in v] + appendices
edges = [("1 引言","2 成像"),("1 引言","3 图像处理"),("2 成像","4 拟合/优化"),("3 图像处理","4 拟合/优化"),
         ("4 拟合/优化","5 深度学习"),("5 深度学习","6 识别"),("6 识别","7 特征"),("7 特征","8 对齐/拼接"),
         ("8 对齐/拼接","9 运动"),("9 运动","10 计算摄影"),("8 对齐/拼接","11 SfM/SLAM"),
         ("11 SfM/SLAM","12 深度"),("12 深度","13 重建"),("13 重建","14 IBR"),("14 IBR","15 结论"),
         ("1 引言","A 线性代数"),("A 线性代数","B 贝叶斯"),("B 贝叶斯","4 拟合/优化"),
         ("A 线性代数","C 数据/软件"),("C 数据/软件","15 结论")]
check("地图覆盖 18 个节点且不重", len(all_nodes) == len(set(all_nodes)) == 18, f"{len(all_nodes)} 节点")
check("地图六层齐全", len(layers) == 6, f"{len(layers)} 层")
check("地图边数 ≥16", len(edges) >= 16, f"{len(edges)} 条边")
# 拓扑排序检环
indeg = {n: 0 for n in all_nodes}; adj = {n: [] for n in all_nodes}
for a, b in edges:
    adj[a].append(b); indeg[b] += 1
queue = [n for n in all_nodes if indeg[n] == 0]; seen = 0
while queue:
    n = queue.pop(); seen += 1
    for m in adj[n]:
        indeg[m] -= 1
        if indeg[m] == 0: queue.append(m)
check("地图无环（DAG）", seen == len(all_nodes), f"拓扑访问 {seen}/{len(all_nodes)}")

n_pass = int(sum(checks))

# 图1：三任务（同一套最小二乘）
fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.3))
ax = axes[0]
ax.scatter(x1, y1, s=22, color=BLUE, edgecolor=PAPER, zorder=3, label="带噪样本")
ax.plot(xt, a_ols * xt + b_ols, color=ACCENT, lw=2.2, label="最小二乘直线")
ax.set_title("几何：拟合一条直线", fontsize=15); ax.set_xlabel("x", fontsize=15); ax.set_ylabel("y", fontsize=15)
ax.legend(fontsize=15, frameon=False, loc="upper left"); ax.grid(True, color=GRID, alpha=0.45, lw=0.6)
ax = axes[1]
ax.scatter(p2[:, 0], p2[:, 1], s=22, color=GREEN, edgecolor=PAPER, label="源点", zorder=3)
ax.scatter(q2[:, 0], q2[:, 1], s=22, color=ACCENT, edgecolor=PAPER, label="目标点", zorder=3)
for i in range(0, N2, 3):
    ax.plot([p2[i,0], q2[i,0]], [p2[i,1], q2[i,1]], color=GRID, lw=0.8)
ax.set_title("运动：对应点解仿射", fontsize=15); ax.set_xlabel("x", fontsize=15); ax.set_ylabel("y", fontsize=15)
ax.legend(fontsize=15, frameon=False, loc="upper left"); ax.grid(True, color=GRID, alpha=0.45, lw=0.6)
ax = axes[2]
nx, ny, nz = Nest[..., 0], Nest[..., 1], Nest[..., 2]
rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
rgb[~mask] = 0.95
ax.imshow(np.clip(rgb, 0, 1)); ax.set_title("三维：光度立体反照率/法向", fontsize=15)
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("同一套最小二乘，三种视觉任务", fontsize=18, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "e136_three_tasks.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：噪声敏感性
fig, ax = plt.subplots(figsize=(10.6, 4.6))
colors = {"直线": BLUE, "仿射": GREEN, "光度立体": ACCENT}
for name, vals in curves.items():
    base = vals[0]
    ax.plot(noise_levels, np.array(vals) / base, marker="o", ms=6, lw=2.2, color=colors[name],
            label=f"{name}（斜率 {slopes[name]:.1f}）")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("噪声水平（对数）", fontsize=14); ax.set_ylabel("误差 / 最小噪声误差（对数）", fontsize=14)
ax.set_title("噪声变大，三种任务的误差一起变大", fontsize=16)
ax.legend(fontsize=14, frameon=False); ax.grid(True, which="both", color=GRID, alpha=0.45, lw=0.6)
fig.tight_layout()
fig.savefig(FIG / "e136_noise.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：全书地图
fig, ax = plt.subplots(figsize=(11.2, 5.2))
ax.set_xlim(0, 10.7); ax.set_ylim(0, 6.8); ax.axis("off")
col_x = np.linspace(0.9, 8.6, 6)
colors_layer = ["#DDE6F0", "#E8E1D0", "#E2EBDD", "#F0E1D8", "#E6DDEC", "#DCE8E5"]
# 顶部总箭头
ax.annotate("", xy=(10.2, 6.30), xytext=(0.5, 6.30),
            arrowprops=dict(arrowstyle="-|>", color="#8A7A66", lw=2.0))
for ci, (layer, nodes) in enumerate(layers.items()):
    x = col_x[ci]
    ax.text(x + 0.72, 5.82, layer, fontsize=15, ha="center", va="center", fontweight="bold")
    for ni, node in enumerate(nodes):
        y = 5.15 - ni * 1.02
        box = FancyBboxPatch((x, y - 0.34), 1.45, 0.68, boxstyle="round,pad=0.04,rounding_size=0.08",
                             facecolor=colors_layer[ci], edgecolor=INK, lw=1.0)
        ax.add_patch(box); ax.text(x + 0.725, y, node, fontsize=15, ha="center", va="center")
# 附录
app_y = 1.15
for ai, node in enumerate(appendices):
    x = 1.0 + ai * 2.9
    box = FancyBboxPatch((x, app_y - 0.34), 2.1, 0.68, boxstyle="round,pad=0.04,rounding_size=0.08",
                         facecolor="#F2EBDD", edgecolor=INK, lw=1.0, linestyle="--")
    ax.add_patch(box); ax.text(x + 1.05, app_y, node, fontsize=15, ha="center", va="center")
ax.annotate("", xy=(3.7, 3.65), xytext=(2.3, 1.55),
            arrowprops=dict(arrowstyle="-|>", color="#8A7A66", lw=1.2, linestyle="--",
                            connectionstyle="arc3,rad=-0.15"))
ax.annotate("", xy=(7.8, 1.95), xytext=(5.8, 1.55),
            arrowprops=dict(arrowstyle="-|>", color="#8A7A66", lw=1.2, linestyle="--",
                            connectionstyle="arc3,rad=0.15"))
ax.text(5.0, 0.45, "虚线：附录为主干提供数学、推断与数据资源；箭头=主要依赖，不是全部边",
        fontsize=15, ha="center", va="center", color="#6B5A44")
ax.set_title("全书结构：六层正文 + 三本附录", fontsize=18, pad=12)
fig.tight_layout()
fig.savefig(FIG / "e136_map.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

for _name in ("e136_three_tasks.png", "e136_noise.png", "e136_map.png"):
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] 直线 a={a_ols:.4f} b={b_ols:.4f}；TLS {mse_tls:.5f} vs OLS {mse_ols:.5f}")
print(f"[lab] 仿射参数误差 {param_err:.4f}，残差 RMS {res2_rms:.4f}")
print(f"[lab] 光度立体角误差 {ang_err:.3f}°，反照率相关 {alb_corr:.4f}，有效 {valid_frac*100:.1f}%")
print(f"[lab] 噪声斜率：" + "；".join(f"{k}={v:.2f}" for k, v in slopes.items()))

readings = dict(n_pass=n_pass, n_total=len(checks), image=[W3, H3], n_line=N1, n_affine=N2,
                line=dict(slope=a_ols, intercept=b_ols, mse_tls=mse_tls, mse_ols=mse_ols),
                affine=dict(param_err=param_err, residual_rms=res2_rms),
                photometric=dict(angular_error_deg=ang_err, albedo_corr=alb_corr, valid_frac=valid_frac),
                noise=dict(levels=noise_levels.tolist(), slopes=slopes, curves=curves),
                map=dict(n_nodes=len(all_nodes), n_layers=len(layers), n_edges=len(edges),
                         layers={k: v for k, v in layers.items()}, appendices=appendices),
                protocol=dict(synthetic=True, line_fit=True, affine_alignment=True,
                              photometric_stereo=True, map_is_textbook_figure=False))
out = ROOT / "outputs" / "e136_whole_book_review_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
