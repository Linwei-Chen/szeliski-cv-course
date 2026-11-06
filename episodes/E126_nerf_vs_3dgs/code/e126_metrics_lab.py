#!/usr/bin/env python3
"""E126 NeRF vs 3DGS 横向对决：PSNR/SSIM/感知指标与基准读表。

依赖：numpy/scipy/matplotlib。
边界：合成图像上计算 PSNR/SSIM；LPIPS 不本地实现，只读 3DGS 论文 Table 1 的已发表值。
教材 p.100–101 给 MSE/PSNR 定义，p.148–149 给 SSIM/FLIP/神经感知/无参考评估；
论文 p.8 Table 1 给三数据集、三指标、训练时间、FPS、显存。
"""
import hashlib, json, pathlib
from datetime import datetime
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, GRAY = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#6b7280"
N = 256


def make_target():
    y, x = np.mgrid[0:N, 0:N]; x = x / (N - 1); y = y / (N - 1)
    img = 0.45 + 0.20 * np.sin(12 * np.pi * x) * np.cos(8 * np.pi * y) + 0.15 * np.sin(30 * np.pi * (x + y))
    for cx, cy, r, v in [(0.30, 0.35, 0.18, 0.35), (0.70, 0.65, 0.15, -0.30)]:
        m = (x - cx) ** 2 + (y - cy) ** 2 < r * r
        img[m] += v
    return np.clip(img, 0, 1)


def psnr(a, b, L=1.0):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(L * L / max(mse, 1e-12))


def ssim(a, b, L=1.0, w=7):
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    ma, mb = uniform_filter(a, w), uniform_filter(b, w)
    va = uniform_filter(a * a, w) - ma * ma
    vb = uniform_filter(b * b, w) - mb * mb
    cov = uniform_filter(a * b, w) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return float(s.mean())


target = make_target()
blur = gaussian_filter(target, 1.2)
# 二分噪声强度，让噪声图的 PSNR 与模糊图相同
target_psnr = psnr(target, blur)
lo, hi = 0.0, 0.5
for _ in range(40):
    mid = (lo + hi) / 2
    n = np.clip(target + np.random.default_rng(1).normal(0, mid, target.shape), 0, 1)
    if psnr(target, n) > target_psnr:
        lo = mid
    else:
        hi = mid
noise_sigma = (lo + hi) / 2
noise = np.clip(target + np.random.default_rng(1).normal(0, noise_sigma, target.shape), 0, 1)
p_blur, p_noise = psnr(target, blur), psnr(target, noise)
s_blur, s_noise = ssim(target, blur), ssim(target, noise)
p_ident, s_ident = psnr(target, target), ssim(target, target)
psnr_mse01 = psnr(np.zeros((8, 8)), np.full((8, 8), 0.1))
s_sym = abs(ssim(blur, noise) - ssim(noise, blur))

# 论文 p.8 Table 1（Mip-NeRF360 数据集，首列）
bench = {
    "Mip-NeRF360": {
        "Plenoxels": dict(ssim=0.626, psnr=23.08, lpips=0.463, train_min=25.8, fps=6.79, mem_gb=2.1),
        "INGP-Big": dict(ssim=0.699, psnr=25.59, lpips=0.331, train_min=7.5, fps=9.43, mem_gb=0.048),
        "M-NeRF360": dict(ssim=0.792, psnr=27.69, lpips=0.237, train_min=2880.0, fps=0.06, mem_gb=0.0086),
        "Ours-30K": dict(ssim=0.815, psnr=27.21, lpips=0.214, train_min=41.55, fps=134.0, mem_gb=0.734),
    },
    "Tanks&Temples": {
        "M-NeRF360": dict(ssim=0.759, psnr=22.22, lpips=0.257, train_min=2880.0, fps=0.14, mem_gb=0.0086),
        "Ours-30K": dict(ssim=0.841, psnr=23.14, lpips=0.183, train_min=26.9, fps=154.0, mem_gb=0.411),
    },
}
mip = bench["Mip-NeRF360"]; tank = bench["Tanks&Temples"]
protocol = dict(test_every=8, gpu="A6000", metrics=["PSNR", "SSIM", "LPIPS"])

checks = [
 ("PSNR 公式：MSE=0.01 对应 20dB", abs(psnr_mse01 - 20.0) < 1e-9, f"{psnr_mse01:.4f} dB"),
 ("恒等图 PSNR 很高", p_ident > 100, f"{p_ident:.1f} dB"),
 ("恒等图 SSIM 等于 1", abs(s_ident - 1.0) < 1e-12, f"{s_ident:.6f}"),
 ("SSIM 对称", s_sym < 1e-12, f"|Δ|={s_sym:.2e}"),
 ("模糊与噪声 PSNR 对齐", abs(p_blur - p_noise) < 0.01, f"{p_blur:.4f} vs {p_noise:.4f}"),
 ("同样 PSNR 下 SSIM 不同", s_blur - s_noise > 0.005, f"{s_blur:.4f} vs {s_noise:.4f}"),
 ("论文表：Mip 上 PSNR 赢家不是 SSIM 赢家", mip["M-NeRF360"]["psnr"] > mip["Ours-30K"]["psnr"] and mip["Ours-30K"]["ssim"] > mip["M-NeRF360"]["ssim"], f"PSNR {mip['M-NeRF360']['psnr']} > {mip['Ours-30K']['psnr']}; SSIM {mip['Ours-30K']['ssim']} > {mip['M-NeRF360']['ssim']}"),
 ("论文表：LPIPS 偏好 Ours", mip["Ours-30K"]["lpips"] < mip["M-NeRF360"]["lpips"], f"{mip['Ours-30K']['lpips']} < {mip['M-NeRF360']['lpips']}"),
 ("论文表：FPS 偏好 Ours，显存偏好 M-NeRF360", mip["Ours-30K"]["fps"] > mip["M-NeRF360"]["fps"] and mip["Ours-30K"]["mem_gb"] > mip["M-NeRF360"]["mem_gb"], f"FPS {mip['Ours-30K']['fps']} vs {mip['M-NeRF360']['fps']}; Mem {mip['Ours-30K']['mem_gb']}GB vs {mip['M-NeRF360']['mem_gb']}GB"),
 ("数据集换了，PSNR 赢家可能换", tank["Ours-30K"]["psnr"] > tank["M-NeRF360"]["psnr"], f"Tanks: Ours {tank['Ours-30K']['psnr']} > M-NeRF360 {tank['M-NeRF360']['psnr']}"),
 ("基准协议：每 8 张取 1 张测试", protocol["test_every"] == 8 and protocol["gpu"] == "A6000", f"test_every={protocol['test_every']}, GPU={protocol['gpu']}"),
 ("指标清单包含 LPIPS", "LPIPS" in protocol["metrics"] and len(protocol["metrics"]) == 3, "、".join(protocol["metrics"])),
]
n_pass = 0
print(f"[lab] E126 评测 开始 {datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成图计算 PSNR/SSIM；LPIPS 只读论文 Table 1；不是完整基准复现")
for text, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {text} | {detail}")
    n_pass += bool(ok)

# 图1 目标/模糊/噪声
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
for ax, img, title in [(axes[0], target, "目标"), (axes[1], blur, f"模糊：PSNR {p_blur:.2f} / SSIM {s_blur:.3f}"),
                       (axes[2], noise, f"噪声：PSNR {p_noise:.2f} / SSIM {s_noise:.3f}")]:
    ax.imshow(np.clip(img, 0, 1), cmap="gray", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=17, color=INK)
fig.suptitle("同一张图，两种失真：PSNR 相同，结构相似度不同", fontsize=20, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(FIG / "e126_metrics.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图2 PSNR/SSIM 对比
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
for ax, vals, title, ylab, ylim in [(axes[0], [p_blur, p_noise], "PSNR：几乎一样", "PSNR (dB)", (20, 34)),
                                    (axes[1], [s_blur, s_noise], "SSIM：模糊更高", "SSIM", (0.90, 1.0))]:
    bars = ax.bar(["模糊", "噪声"], vals, color=[BLUE, ACCENT], width=0.55)
    ax.set_ylim(*ylim); ax.set_ylabel(ylab, fontsize=18); ax.tick_params(labelsize=17)
    ax.set_title(title, fontsize=18, color=INK)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + (ylim[1]-ylim[0])*0.02, f"{v:.3f}" if ylab == "SSIM" else f"{v:.2f}",
                ha="center", fontsize=18)
fig.suptitle("PSNR 打平，SSIM 不一定打平", fontsize=20, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e126_equal_psnr.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

# 图3 真实基准的三角取舍
fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
for ax, key, xlab, xlog in [(axes[0], "fps", "FPS（对数）", True), (axes[1], "mem_gb", "显存 GB（对数）", True)]:
    for name, d in mip.items():
        ax.scatter(d[key], d["psnr"], s=90, color=ACCENT if name == "Ours-30K" else (GREEN if name == "M-NeRF360" else BLUE))
        ax.annotate(name, (d[key], d["psnr"]), textcoords="offset points", xytext=(-10, 10) if name == "Plenoxels" else (6, -12), ha="right" if name == "Plenoxels" else "left", fontsize=18)
    ax.set_xscale("log"); ax.set_xlim(0.04, 300) if key == "fps" else ax.set_xlim(0.006, 8.0)
    ax.set_ylim(22.4, 28.2)
    ax.set_xlabel(xlab, fontsize=18); ax.set_ylabel("PSNR (dB)", fontsize=18)
    ax.tick_params(labelsize=18); ax.set_title("PSNR vs " + ("速度" if key == "fps" else "显存"), fontsize=19, color=INK)
fig.suptitle("Mip-NeRF360 数据集：没有单一赢家", fontsize=20, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.9])
fig.savefig(FIG / "e126_tradeoff.png", dpi=160, facecolor=PAPER, bbox_inches="tight"); plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), p_blur=p_blur, p_noise=p_noise,
                s_blur=s_blur, s_noise=s_noise, noise_sigma=noise_sigma,
                p_ident=p_ident, s_ident=s_ident, psnr_mse01=psnr_mse01,
                bench=bench, protocol=protocol)
(OUT / "e126_metrics_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for name in ("e126_metrics.png", "e126_equal_psnr.png", "e126_tradeoff.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] readings sha256={hashlib.sha256((OUT/'e126_metrics_readings.json').read_bytes()).hexdigest()}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
raise SystemExit(0 if n_pass == len(checks) else 1)
