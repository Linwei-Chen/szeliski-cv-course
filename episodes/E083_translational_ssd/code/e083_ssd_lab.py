#!/usr/bin/env python3
"""E083 §9.1 平移对齐与 SSD：鲁棒损失、偏置增益、归一化相关（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261103)

N, P0, R = 180, 40, 10
TRUE = (7, -5)
A = 0.05 + 0.90 * (lambda t: (t - t.min()) / (t.max() - t.min()))(gaussian_filter(RNG.normal(0, 1, (N, N)), 1.2))
# B：增益 1.2、偏置 0.1、噪声；再放一块高亮外点（右下）
B0 = np.roll(1.20 * A + 0.10, shift=(TRUE[1], TRUE[0]), axis=(0, 1))
B0 = B0 + RNG.normal(0, 0.006, B0.shape)
patch = A[P0:P0 + 100, P0:P0 + 100]


def search(mode="ssd"):
    best = (1e18, 0, 0)
    for sy in range(-R, R + 1):
        for sx in range(-R, R + 1):
            cand = B0[P0 + TRUE[1] + sy:P0 + TRUE[1] + sy + 100,
                       P0 + TRUE[0] + sx:P0 + TRUE[0] + sx + 100]
            if mode == "ssd":
                v = np.mean((cand - patch) ** 2)
            elif mode == "sad":
                v = np.mean(np.abs(cand - patch))
            elif mode == "ncc":
                a = patch - patch.mean(); b = cand - cand.mean()
                v = -(a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
            elif mode == "corr":
                v = -np.sum(patch * cand)
            elif mode == "bg":
                g, b = np.polyfit(patch.ravel(), cand.ravel(), 1)
                v = np.mean((cand - (g * patch + b)) ** 2)
            if v < best[0]: best = (v, TRUE[0] + sx, TRUE[1] + sy)
    return best


ssd_v, ssd_x, ssd_y = search("ssd")
sad_v, sad_x, sad_y = search("sad")
ncc_v, ncc_x, ncc_y = search("ncc")
bg_v, bg_x, bg_y = search("bg")
# 真值位置的残差
cand_true = B0[P0 + TRUE[1]:P0 + TRUE[1] + 100, P0 + TRUE[0]:P0 + TRUE[0] + 100]
resid = cand_true - patch
ssd_true = float(np.mean(resid ** 2))
sad_true = float(np.mean(np.abs(resid)))
# 鲁棒损失演示：90% 小残差 + 10% 外点，比较最差 5% 像素的贡献
res_syn = np.concatenate([RNG.normal(0, 0.02, 9000), RNG.normal(0, 0.30, 1000)])
mad = float(np.median(np.abs(res_syn - np.median(res_syn))))
a_gm = max(1.4 * mad, 1e-3)
rho_ssd = res_syn ** 2
rho_gm = res_syn ** 2 / (1 + (res_syn / a_gm) ** 2)
k = max(1, int(0.05 * res_syn.size))
idx = np.argsort(np.abs(res_syn))[-k:]
frac_ssd = float(rho_ssd[idx].sum() / rho_ssd.sum())
frac_gm = float(rho_gm[idx].sum() / max(rho_gm.sum(), 1e-12))
ncc_score = float(-ncc_v)

# —— 图 1：三种损失 ——
x = np.linspace(-1.5, 1.5, 1600)
a_plot = 0.10
fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=150)
fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
ax.plot(x, x ** 2, color=BLUE, lw=2.2, label="平方损失 x²")
ax.plot(x, np.abs(x), color=GREEN, lw=2.2, label="绝对值 |x|")
ax.plot(x, x ** 2 / (1 + (x / a_plot) ** 2), color=ACCENT, lw=2.2, label="Geman–McClure")
ax.axvline(0, color=INK, lw=0.8, ls=":")
ax.set_xlabel("残差 e", fontsize=15); ax.set_ylabel("损失 ρ(e)", fontsize=15)
ax.set_yscale("log"); ax.set_ylim(1e-4, 5)
ax.set_title("平方损失在远处最大；鲁棒损失饱和", fontsize=16, color=INK)
ax.legend(fontsize=13, frameon=False)
ax.grid(alpha=0.2)
fig.text(0.5, 0.01, "程序生成｜a=0.10；平方损失无上限，GM 有上限", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 1])
fig.savefig(FIG / "e083_losses.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 2：偏置/增益回归 ——
fig = plt.figure(figsize=(9.6, 4.8), dpi=150)
fig.patch.set_facecolor(PAPER)
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.38)
ax = fig.add_subplot(gs[0]); ax.set_facecolor(PAPER)
ax.scatter(patch.ravel()[::17], cand_true.ravel()[::17], s=8, color=BLUE, alpha=0.45, label="对应像素")
g, b = np.polyfit(patch.ravel(), cand_true.ravel(), 1)
xx = np.linspace(patch.min(), patch.max(), 100)
ax.plot(xx, g * xx + b, color=ACCENT, lw=2, label=f"回归：g={g:.3f}, b={b:.3f}")
ax.set_xlabel("模板 I0", fontsize=15); ax.set_ylabel("目标 I1(x+u)", fontsize=15)
ax.set_title("偏置与增益：先拟合一条线", fontsize=16, color=INK)
ax.legend(fontsize=13, frameon=False)
ax = fig.add_subplot(gs[1]); ax.set_facecolor(PAPER)
vals = [ssd_true, bg_v]
labs = ["原始 SSD", "偏置/增益回归"]
cols = [BLUE, ACCENT]
bars = ax.bar(labs, vals, color=cols, width=0.55)
ax.set_yscale("log")
ax.set_ylabel("残差（对数）", fontsize=15)
ax.set_ylim(1e-6, vals[0] * 5)
ax.set_title("同一对齐，误差差几个量级", fontsize=16, color=INK)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.3, f"{v:.2e}", ha="center", fontsize=13, color=INK)
ax.tick_params(axis="x", labelsize=13)
fig.suptitle("曝光差不是「像不像」能解决的：先建模，再比较", fontsize=17, color=INK, y=0.99)
fig.tight_layout(rect=[0, 0.10, 1, 0.90])
fig.savefig(FIG / "e083_biascorr.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

# —— 图 3：外点与鲁棒损失 ——
fig = plt.figure(figsize=(9.6, 4.6), dpi=150)
fig.patch.set_facecolor(PAPER)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
ax = fig.add_subplot(gs[0]); ax.set_facecolor(PAPER)
ax.hist(res_syn, bins=60, color=BLUE, alpha=0.75)
ax.set_yscale("log"); ax.set_xlabel("残差 e", fontsize=15); ax.set_ylabel("像素数（对数）", fontsize=15)
ax.set_title("残差分布（合成）：多数很小，少数很大", fontsize=16, color=INK)
ax = fig.add_subplot(gs[1]); ax.set_facecolor(PAPER)
fracs = [frac_ssd, frac_gm]
bars = ax.bar(["平方损失", "Geman–McClure"], fracs, color=[BLUE, ACCENT], width=0.5)
ax.set_ylim(0, 1.0); ax.set_ylabel("最差 5% 像素贡献占比", fontsize=15)
ax.set_title("鲁棒损失把外点的影响压下去", fontsize=16, color=INK)
for bar, v in zip(bars, fracs):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=14, color=INK)
ax.tick_params(axis="x", labelsize=14)
fig.suptitle("外点不是靠调阈值，而是靠损失函数的形状", fontsize=17, color=INK, y=0.99)
fig.text(0.5, 0.01, f"程序生成｜MAD={mad:.4f}｜a={a_gm:.4f}", ha="center", fontsize=13, color=ACCENT)
fig.tight_layout(rect=[0, 0.04, 1, 0.94])
fig.savefig(FIG / "e083_outlier.png", facecolor=PAPER, bbox_inches="tight", pad_inches=0.06); plt.close(fig)

tests = [
    ("原始 SSD 命中真值", (ssd_x, ssd_y) == TRUE, f"best=({ssd_x},{ssd_y})"),
    ("原始 SSD 误差非零", ssd_true > 1e-4, f"mse={ssd_true:.2e}"),
    ("偏置/增益回归命中真值", (bg_x, bg_y) == TRUE, f"best=({bg_x},{bg_y})"),
    ("回归误差远小于 SSD", bg_v < 0.2 * ssd_true, f"回归={bg_v:.2e}/SSD={ssd_true:.2e}"),
    ("NCC 命中真值", (ncc_x, ncc_y) == TRUE, f"best=({ncc_x},{ncc_y})"),
    ("NCC 峰值足够高", ncc_score > 0.99, f"NCC={ncc_score:.4f}"),
    ("SAD 命中真值附近", abs(sad_x - TRUE[0]) <= 1 and abs(sad_y - TRUE[1]) <= 1, f"best=({sad_x},{sad_y})"),
    ("回归斜率接近 1.2", abs(g - 1.20) < 0.01, f"g={g:.4f}"),
    ("GM 压外点贡献", frac_gm < 0.5 * frac_ssd, f"GM={frac_gm:.2f}/SSD={frac_ssd:.2f}"),
    ("三图在盘", all((FIG / n).exists() for n in ("e083_losses.png", "e083_biascorr.png", "e083_outlier.png")), "三张全在"),
]
lines = []; n_pass = 0
for name, ok, detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass += bool(ok)
readings = dict(true=list(TRUE), ssd_best=[ssd_x, ssd_y], ssd_true=ssd_true, sad_best=[sad_x, sad_y],
                sad_true=sad_true, gain_fit=float(g), bias_fit=float(b), ncc_best=[ncc_x, ncc_y], ncc_score=ncc_score, bg_best=[bg_x, bg_y], bg_err=bg_v,
                mad=mad, a_gm=a_gm, frac_ssd=frac_ssd, frac_gm=frac_gm,
                n_pass=n_pass, n_total=len(tests))
(OUT / "e083_ssd_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256((OUT / "e083_ssd_readings.json").read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E083 平移对齐与 SSD 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")):
    print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e083_ssd_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests) - n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass != len(tests): raise SystemExit(1)
