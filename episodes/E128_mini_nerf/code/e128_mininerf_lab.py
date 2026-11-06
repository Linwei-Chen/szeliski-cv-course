#!/usr/bin/env python3
"""E128 迷你 NeRF 实验课 lab：二维坐标 MLP + 位置编码，不是完整 3D 体积渲染。"""
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
from scipy.ndimage import laplace

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"
INK = "#2B2B2B"
ACCENT = "#C24E2A"
BLUE = "#3B6EA5"
GREEN = "#4F7A4A"
GRID = "#D8CDBB"
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

N = 40
H = 48
L = 4
STEPS = 500
LR = 0.01
SEED = 1

print(f"[lab] E128 迷你NeRF 开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：二维坐标 MLP 拟合合成汉堡；不是完整 3D NeRF/体积渲染；三图程序生成")

def target(n=N):
    y, x = np.mgrid[0:n, 0:n]
    x = (x + 0.5) / n; y = (y + 0.5) / n
    img = np.ones((n, n, 3)) * np.array([0.97, 0.94, 0.85])
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.52) ** 2)
    img[r < 0.46] = [0.86, 0.90, 0.92]
    img[(r < 0.40) & (y > 0.55)] = [0.80, 0.52, 0.22]
    img[(r < 0.38) & (y > 0.45) & (y < 0.56)] = [0.36, 0.20, 0.12]
    img[(r < 0.40) & (y > 0.42) & (y < 0.46)] = [0.95, 0.72, 0.18]
    img[(r < 0.42) & (y > 0.39) & (y < 0.43)] = [0.35, 0.65, 0.25]
    dome = ((x - 0.5) / 0.40) ** 2 + ((y - 0.34) / 0.22) ** 2
    img[(dome < 1) & (y < 0.40)] = [0.88, 0.60, 0.28]
    for cx, cy in [(0.38, 0.24), (0.50, 0.20), (0.62, 0.24), (0.44, 0.30), (0.57, 0.30)]:
        img[((x - cx) ** 2 + (y - cy) ** 2) < 0.0007] = [0.98, 0.92, 0.72]
    return np.clip(img, 0, 1)

def feats(n, L):
    y, x = np.mgrid[0:n, 0:n]
    x = (x + 0.5) / n * 2 - 1
    y = (y + 0.5) / n * 2 - 1
    base = np.stack([x, y], -1).reshape(-1, 2)
    if L == 0:
        return base
    out = [base]
    for k in range(L):
        f = 2 ** k * np.pi
        out += [np.sin(base * f), np.cos(base * f)]
    return np.concatenate(out, 1)

def psnr(a, b):
    return float(10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12)))

def train(F, T, steps=STEPS, hidden=H, lr=LR, seed=SEED, record=False):
    r = np.random.default_rng(seed); D = F.shape[1]
    W1 = r.normal(0, np.sqrt(2 / D), (D, hidden)); b1 = np.zeros(hidden)
    W2 = r.normal(0, np.sqrt(2 / hidden), (hidden, hidden)); b2 = np.zeros(hidden)
    W3 = r.normal(0, np.sqrt(2 / hidden), (hidden, 3)); b3 = np.zeros(3)
    ps = [W1, b1, W2, b2, W3, b3]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    hist = []; losses = []
    def forward():
        H1 = np.tanh(F @ W1 + b1)
        H2 = np.tanh(H1 @ W2 + b2)
        O = 1 / (1 + np.exp(-(H2 @ W3 + b3)))
        return H1, H2, O
    for t in range(1, steps + 1):
        H1, H2, O = forward()
        loss = float(np.mean((O - T) ** 2)); losses.append(loss)
        if record and (t == 1 or t % 25 == 0 or t == steps):
            hist.append((t, psnr(O.reshape(-1), T.reshape(-1))))
        dO = 2 * (O - T) / len(F) * O * (1 - O)
        dW3 = H2.T @ dO; db3 = dO.sum(0)
        dH2 = (dO @ W3.T) * (1 - H2 ** 2)
        dW2 = H1.T @ dH2; db2 = dH2.sum(0)
        dH1 = (dH2 @ W2.T) * (1 - H1 ** 2)
        dW1 = F.T @ dH1; db1 = dH1.sum(0)
        for i, (p, g) in enumerate(zip(ps, [dW1, db1, dW2, db2, dW3, db3])):
            ms[i] = 0.9 * ms[i] + 0.1 * g
            vs[i] = 0.999 * vs[i] + 0.001 * g * g
            p -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    def fn(Fq):
        H1 = np.tanh(Fq @ W1 + b1)
        H2 = np.tanh(H1 @ W2 + b2)
        return 1 / (1 + np.exp(-(H2 @ W3 + b3)))
    params = int(sum(p.size for p in ps))
    return fn, hist, losses, params

img = target(N); T = img.reshape(-1, 3)
F0 = feats(N, 0); F4 = feats(N, L)
fn0, hist0, loss0, params0 = train(F0, T, record=True)
fn4, hist4, loss4, params4 = train(F4, T, record=True)
out0 = fn0(F0).reshape(N, N, 3); out4 = fn4(F4).reshape(N, N, 3)
p0 = psnr(out0, img); p4 = psnr(out4, img)
gap = p4 - p0
hf0 = float(np.mean((laplace(out0.mean(2)) - laplace(img.mean(2))) ** 2))
hf4 = float(np.mean((laplace(out4.mean(2)) - laplace(img.mean(2))) ** 2))
# 频率扫描
freq = {}
for k in (0, 2, 3, 4, 5):
    if k == 0:
        freq[k] = p0
    elif k == 4:
        freq[k] = p4
    else:
        F = feats(N, k); fn, _, _, _ = train(F, T); freq[k] = psnr(fn(F).reshape(N, N, 3), img)
# 2 倍分辨率连续渲染
big = target(2 * N); Fb = feats(2 * N, L)
out_big = fn4(Fb).reshape(2 * N, 2 * N, 3); p_big = psnr(out_big, big)
protocol = dict(image=N, hidden=H, freqs=len(range(L)), steps=STEPS, seed=SEED,
                lr=LR, representation="2D coordinate MLP", full_nerf=False)

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("汉堡图尺寸/范围正确", img.shape == (N, N, 3) and img.min() >= 0 and img.max() <= 1, f"{img.shape}")
check(f"L={L} 特征维度 = 18", F4.shape[1] == 18, f"{F4.shape[1]}")
check("裸坐标 PSNR ≤ 25", p0 <= 25, f"{p0:.2f}")
check("位置编码 PSNR ≥ 30", p4 >= 30, f"{p4:.2f}")
check("位置编码提升 ≥ 8dB", gap >= 8, f"{gap:.2f}")
check("L=2 不如 L=4", freq[2] < freq[4], f"{freq[2]:.2f} < {freq[4]:.2f}")
check("L=5 相对 L=4 饱和（<+1dB）", freq[5] < freq[4] + 1.0, f"{freq[5]:.2f} vs {freq[4]:.2f}")
check("训练损失下降 ≥ 90%", loss4[-1] <= 0.1 * loss4[0], f"{loss4[0]:.4f}→{loss4[-1]:.4f}")
check("位置编码高频误差更低", hf4 < hf0, f"{hf4:.5f} < {hf0:.5f}")
check("2 倍分辨率渲染 PSNR ≥ 22", p_big >= 22, f"{p_big:.2f}")
check("只增加输入层参数", params4 - params0 == (18 - 2) * H, f"{params4-params0} = 16×{H}")
check("协议字段完整且声明非完整 NeRF", protocol["full_nerf"] is False and protocol["image"] == N, str(protocol))

# 图1 目标/裸坐标/位置编码
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
for ax, im, title, val in [(axes[0], img, "目标（合成汉堡）", None),
                           (axes[1], out0, "裸坐标 MLP", p0),
                           (axes[2], out4, f"位置编码 L={L}", p4)]:
    ax.imshow(np.clip(im, 0, 1), interpolation="nearest")
    ax.set_title(title + (f"\nPSNR {val:.1f} dB" if val else ""), fontsize=19)
    ax.axis("off")
fig.suptitle("同一张 40×40 汉堡：坐标 MLP 能不能画出细节", fontsize=21)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e128_burger.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 图2 训练曲线
fig, ax = plt.subplots(figsize=(11.8, 4.6))
ax.plot([s for s, _ in hist0], [v for _, v in hist0], color=BLUE, lw=2.4, label="裸坐标（L=0）")
ax.plot([s for s, _ in hist4], [v for _, v in hist4], color=ACCENT, lw=2.4, label=f"位置编码（L={L}）")
ax.set_xlabel("训练步", fontsize=18); ax.set_ylabel("训练集 PSNR (dB)", fontsize=18)
ax.set_title("位置编码把细节一点点学回来", fontsize=20)
ax.tick_params(labelsize=17); ax.grid(color=GRID, lw=0.6, alpha=0.6)
ax.legend(fontsize=17, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "e128_curve.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# 图3 频率扫描
fig, ax = plt.subplots(figsize=(11.6, 4.6))
ks = [0, 2, 3, 4, 5]; vals = [freq[k] for k in ks]
bars = ax.bar([str(k) for k in ks], vals, color=[BLUE] + [ACCENT] * (len(ks) - 1), width=0.58)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=18)
ax.set_xlabel("位置编码频率组数 L（L=0 为裸坐标）", fontsize=17)
ax.set_ylabel("PSNR (dB)", fontsize=18)
ax.set_title("频率不是越多越好，L=4 已接近饱和", fontsize=20)
ax.set_ylim(0, max(vals) + 4); ax.tick_params(labelsize=17)
ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.6)
fig.tight_layout()
fig.savefig(FIG / "e128_freq.png", dpi=160, bbox_inches="tight"); plt.close(fig)

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
readings = dict(n_pass=n_pass, n_total=len(checks), p_raw=p0, p_pe=p4, gap=gap,
                hf_raw=hf0, hf_pe=hf4, p_big=p_big,
                freq={str(k): float(v) for k, v in freq.items()},
                loss_start=float(loss4[0]), loss_end=float(loss4[-1]),
                params_raw=params0, params_pe=params4, protocol=protocol,
                fig_sizes={"burger": [12.6, 4.4], "curve": [11.8, 4.6], "freq": [11.6, 4.6]})
(ROOT / "outputs" / "e128_mininerf_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
for f in ("e128_burger.png", "e128_curve.png", "e128_freq.png"):
    print(f"[lab] PNG sha256({f})={hashlib.sha256((FIG/f).read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((ROOT/'outputs'/'e128_mininerf_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
