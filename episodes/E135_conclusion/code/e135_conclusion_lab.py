#!/usr/bin/env python3
"""E135 第15章结论 lab：同一回归问题上对照四条路线——工程插值、科学模型、统计推断、学习模型。

边界：一维合成信号、已知噪声、已知科学模型基函数；RBF 只是“固定中心+最小二乘输出权”的
小型学习模型，不是神经网络/大模型；所有数字均为合成实验，不外推到真实视觉系统。
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
from matplotlib.gridspec import GridSpec

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; GOLD = "#C9971C"
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

print(f"[lab] E135 结论实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：一维合成信号；已知噪声 sigma=0.08；科学模型已知基函数；RBF 不是神经网络")


def f_true(x):
    return 0.8 * np.sin(2 * np.pi * x) + 0.2 * x


SIGMA = 0.08
SIGMA2 = SIGMA ** 2
N_LIST = [8, 32, 128]
SEEDS = list(range(20))
X_IN = np.linspace(0.0, 1.0, 200)
X_OUT = np.linspace(1.15, 1.60, 100)
X_PLOT = np.linspace(0.0, 1.60, 500)
TRUE_IN = f_true(X_IN); TRUE_OUT = f_true(X_OUT); TRUE_PLOT = f_true(X_PLOT)

PHI_IN = np.column_stack([np.ones_like(X_IN), X_IN, np.sin(2 * np.pi * X_IN)])
PHI_OUT = np.column_stack([np.ones_like(X_OUT), X_OUT, np.sin(2 * np.pi * X_OUT)])
PHI_PLOT = np.column_stack([np.ones_like(X_PLOT), X_PLOT, np.sin(2 * np.pi * X_PLOT)])

N_CENTERS = 16
CENTERS = np.linspace(0.0, 1.0, N_CENTERS)
WIDTH = 0.18
PRIOR_VAR = 1.0


def phi_sci(x):
    return np.column_stack([np.ones_like(x), x, np.sin(2 * np.pi * x)])


def phi_rbf(x):
    return np.exp(-((x[:, None] - CENTERS[None, :]) ** 2) / (2 * WIDTH ** 2))


def fit_sci(x, y):
    A = phi_sci(x)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda z: phi_sci(z) @ coef


def fit_bayes(x, y):
    A = phi_sci(x)
    P = np.eye(A.shape[1]) / PRIOR_VAR
    Sigma = np.linalg.inv(A.T @ A / SIGMA2 + P)
    mu = Sigma @ A.T @ y / SIGMA2
    return (lambda z: phi_sci(z) @ mu), (lambda z: np.sum((phi_sci(z) @ Sigma) * phi_sci(z), axis=1))


def fit_rbf(x, y):
    A = phi_rbf(x)
    lam = 1e-3 * max(1.0, np.trace(A.T @ A) / A.shape[1])
    coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)
    return lambda z: phi_rbf(z) @ coef


def fit_eng(x, y):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    return lambda z: np.interp(z, xs, ys)


METHODS = ["工程插值", "科学模型", "统计推断", "学习模型"]
COLORS = {"工程插值": INK, "科学模型": BLUE, "统计推断": GREEN, "学习模型": ACCENT}
results = {m: {n: {"interp": [], "extrap": []} for n in N_LIST} for m in METHODS}
coverage = {n: [] for n in N_LIST}
widths = {n: [] for n in N_LIST}
example = {}

for seed in SEEDS:
    rng = np.random.default_rng(1000 + seed)
    for n in N_LIST:
        x = np.sort(rng.uniform(0.0, 1.0, n))
        y = f_true(x) + rng.normal(0.0, SIGMA, n)
        preds = {
            "工程插值": fit_eng(x, y),
            "科学模型": fit_sci(x, y),
            "统计推断": fit_bayes(x, y)[0],
            "学习模型": fit_rbf(x, y),
        }
        for m, fn in preds.items():
            results[m][n]["interp"].append(float(np.mean((fn(X_IN) - TRUE_IN) ** 2)))
            results[m][n]["extrap"].append(float(np.mean((fn(X_OUT) - TRUE_OUT) ** 2)))
        mu_fn, var_fn = fit_bayes(x, y)
        mu = mu_fn(X_IN); sd = np.sqrt(var_fn(X_IN))
        coverage[n].append(float(np.mean(np.abs(TRUE_IN - mu) <= 1.96 * sd)))
        widths[n].append(float(np.mean(2 * 1.96 * sd)))
        if seed == 0 and n == 8:
            example[n] = dict(x=x, y=y, preds={m: fn(X_PLOT) for m, fn in preds.items()},
                              mu=mu, sd=sd, x_in=x, y_in=y)

summary = {m: {n: {k: float(np.mean(v)) for k, v in results[m][n].items()} for n in N_LIST} for m in METHODS}
cov = {n: float(np.mean(coverage[n])) for n in N_LIST}
wid = {n: float(np.mean(widths[n])) for n in N_LIST}

checks = []
def check(name, ok, detail):
    checks.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")

check("科学模型 8 样本插值误差接近噪声底", summary["科学模型"][8]["interp"] < 1.5 * SIGMA2,
      f"{summary['科学模型'][8]['interp']:.4f} vs 噪声底 {SIGMA2:.4f}")
check("科学模型 8 样本外推误差低于 0.15", summary["科学模型"][8]["extrap"] < 0.15,
      f"{summary['科学模型'][8]['extrap']:.4f}（工程插值 {summary['工程插值'][8]['extrap']:.4f}）")
check("工程插值 8 样本插值误差高于科学模型", summary["工程插值"][8]["interp"] > summary["科学模型"][8]["interp"],
      f"{summary['工程插值'][8]['interp']:.4f} > {summary['科学模型'][8]['interp']:.4f}")
check("工程插值 8 样本外推明显差于科学模型", summary["工程插值"][8]["extrap"] > 10 * summary["科学模型"][8]["extrap"],
      f"{summary['工程插值'][8]['extrap']:.4f} vs {summary['科学模型'][8]['extrap']:.4f}")
check("学习模型插值误差随样本增加而下降", summary["学习模型"][128]["interp"] < summary["学习模型"][8]["interp"],
      f"{summary['学习模型'][8]['interp']:.4f} → {summary['学习模型'][128]['interp']:.4f}")
check("学习模型 128 样本外推仍差于科学模型", summary["学习模型"][128]["extrap"] > 10 * summary["科学模型"][128]["extrap"],
      f"{summary['学习模型'][128]['extrap']:.4f} vs {summary['科学模型'][128]['extrap']:.4f}")
check("统计推断 8 样本覆盖率接近 95%", 0.90 <= cov[8] <= 0.99, f"覆盖率 {cov[8]*100:.1f}%")
check("统计推断 128 样本覆盖率接近 95%", 0.90 <= cov[128] <= 0.99, f"覆盖率 {cov[128]*100:.1f}%")
check("统计推断区间随样本增加而收窄", wid[128] < 0.6 * wid[8], f"宽度 {wid[8]:.3f} → {wid[128]:.3f}")
check("学习模型参数多于科学模型", N_CENTERS > 3, f"{N_CENTERS} vs 3")
check("科学模型 32 样本外推误差接近噪声底", summary["科学模型"][32]["extrap"] < 1.5 * SIGMA2,
      f"{summary['科学模型'][32]['extrap']:.4f} vs 噪声底 {SIGMA2:.4f}")
check("工程插值外推误差不随样本增加而消失", summary["工程插值"][128]["extrap"] > 10 * summary["科学模型"][128]["extrap"],
      f"{summary['工程插值'][128]['extrap']:.4f} vs {summary['科学模型'][128]['extrap']:.4f}")

n_pass = int(sum(checks))

# 图1：四条路线在 8 个样本上的拟合（阴影=训练域外）
fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.7), sharex=True, sharey=True)
ex = example[8]
for ax, m in zip(axes.ravel(), METHODS):
    ax.axvspan(1.0, 1.6, color=GRID, alpha=0.35, lw=0)
    ax.plot(X_PLOT, TRUE_PLOT, color=GREEN, lw=2.4, label="真实信号")
    ax.plot(X_PLOT, ex["preds"][m], color=COLORS[m], lw=2.0)
    ax.scatter(ex["x"], ex["y"], s=28, color=COLORS[m], edgecolor=PAPER, zorder=3)
    ax.axvline(1.0, color=INK, lw=0.8, ls=":")
    ax.set_title(m, fontsize=15, pad=6)
    ax.grid(True, color=GRID, alpha=0.45, lw=0.6)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(-1.5, 1.5)
fig.suptitle("同一组 8 个带噪样本：四条路线怎么补中间、怎么猜外面", fontsize=18, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "e135_four_views.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：样本数 vs 误差（插值/外推两栏）
fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=False)
for ax, key, title in [(axes[0], "interp", "训练域内：补中间"), (axes[1], "extrap", "训练域外：猜外面")]:
    for m in METHODS:
        vals = [summary[m][n][key] for n in N_LIST]
        ax.plot(N_LIST, vals, marker="o", ms=6, lw=2.0, color=COLORS[m], label=m)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(N_LIST); ax.set_xticklabels([str(n) for n in N_LIST])
    ax.set_xlabel("训练样本数", fontsize=14); ax.set_ylabel("均方误差（对数）", fontsize=14)
    ax.set_title(title, fontsize=15); ax.grid(True, which="both", color=GRID, alpha=0.45, lw=0.6)
axes[0].legend(fontsize=14, frameon=False, loc="best")
fig.suptitle("数据越多，学习模型越准；但外推仍要靠模型假设", fontsize=18, y=0.98)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "e135_data_efficiency.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：贝叶斯区间随样本数收窄
fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2), sharex=True, sharey=True)
rng = np.random.default_rng(1000)
x_full = np.sort(rng.uniform(0.0, 1.0, 128)); y_full = f_true(x_full) + rng.normal(0.0, SIGMA, 128)
for ax, n in zip(axes, N_LIST):
    x = x_full[:n]; y = y_full[:n]
    mu_fn, var_fn = fit_bayes(x, y)
    mu = mu_fn(X_PLOT); sd = np.sqrt(var_fn(X_PLOT))
    ax.fill_between(X_PLOT, mu - 1.96 * sd, mu + 1.96 * sd, color=GREEN, alpha=0.20, lw=0, label="95% 置信区间（均值）")
    ax.plot(X_PLOT, TRUE_PLOT, color=INK, lw=2.0, ls="--", label="真实信号")
    ax.plot(X_PLOT, mu, color=GREEN, lw=2.2, label="后验均值")
    ax.scatter(x, y, s=24, color=GREEN, edgecolor=PAPER, zorder=3)
    ax.set_title(f"{n} 个样本，平均区间宽 {wid[n]:.2f}", fontsize=14)
    ax.grid(True, color=GRID, alpha=0.45, lw=0.6); ax.set_xlim(0, 1.6); ax.set_ylim(-2.0, 2.0)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=14)
fig.suptitle("统计路线不只给答案，还给“这个答案有多不确定”（均值 95% 区间）", fontsize=18, y=0.99)
fig.tight_layout(rect=(0, 0.06, 1, 0.92))
fig.savefig(FIG / "e135_uncertainty.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

print(f"[lab] 四路线插值 MSE（8/32/128）：" + "；".join(
    f"{m}=" + "/".join(f"{summary[m][n]['interp']:.4f}" for n in N_LIST) for m in METHODS))
print(f"[lab] 四路线外推 MSE（8/32/128）：" + "；".join(
    f"{m}=" + "/".join(f"{summary[m][n]['extrap']:.4f}" for n in N_LIST) for m in METHODS))
print(f"[lab] 贝叶斯覆盖率：8={cov[8]*100:.1f}% 32={cov[32]*100:.1f}% 128={cov[128]*100:.1f}%；"
      f"平均区间宽 8={wid[8]:.3f} 32={wid[32]:.3f} 128={wid[128]:.3f}")
for _name in ("e135_four_views.png", "e135_data_efficiency.png", "e135_uncertainty.png"):
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")

readings = dict(n_pass=n_pass, n_total=len(checks), sigma=SIGMA, sigma2=SIGMA2,
                n_list=N_LIST, n_seeds=len(SEEDS), n_centers=N_CENTERS,
                n_params=dict(science=3, statistics=3, learning=N_CENTERS),
                mse_interp=summary, mse_extrap={m: {n: summary[m][n]["extrap"] for n in N_LIST} for m in METHODS},
                coverage=cov, interval_width=wid,
                protocol=dict(synthetic=True, known_noise=True, known_basis=True,
                              rbf="fixed centers + ridge least-squares output weights",
                              neural_network=False, real_world=False))
out = ROOT / "outputs" / "e135_conclusion_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
