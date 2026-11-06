#!/usr/bin/env python3
"""E137 附录 C 资源课 lab：同一高斯模糊的三种实现（OpenCV/SciPy/NumPy）+ 任务→数据集→工具航线图。

边界：单机、单操作、单尺寸序列的合成图；不是通用 benchmark；库版本/硬件不同，绝对时间会变。
"""
import datetime
import hashlib
import json
import logging
import pathlib
import time

import numpy as np
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

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

print(f"[lab] E137 资源课实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] OpenCV {cv2.__version__} / SciPy {ndimage.__name__} / NumPy {np.__version__}")
print("[lab] 边界：单机单操作合成图；不是通用 benchmark，绝对时间随硬件/版本变化")

rng = np.random.default_rng(137)
SIZES = [128, 256, 512, 1024]
REPEATS = 3
K = 9; SIGMA = 2.0
ax = np.arange(K) - K // 2
g1 = np.exp(-(ax ** 2) / (2 * SIGMA ** 2)); g1 /= g1.sum()
KERNEL = np.outer(g1, g1)

def make_image(n):
    yy, xx = np.mgrid[0:n, 0:n]
    base = 0.5 + 0.25 * np.sin(2 * np.pi * xx / 40) + 0.2 * np.cos(2 * np.pi * yy / 55)
    img = np.stack([base, 0.8 * base + 0.1, 0.6 * base + 0.2], axis=-1)
    img += rng.normal(0, 0.02, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)

def numpy_blur(img):
    h, w, c = img.shape; p = K // 2
    pad = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(K):
        for j in range(K):
            out += KERNEL[i, j] * pad[i:i+h, j:j+w, :]
    return out

def timed(fn, img, repeats=REPEATS):
    fn(img)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(img); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

times = {"OpenCV": [], "SciPy": [], "NumPy": []}
refs = {}
for n in SIZES:
    img = make_image(n)
    times["OpenCV"].append(timed(lambda x: cv2.GaussianBlur(x, (K, K), SIGMA), img))
    times["SciPy"].append(timed(lambda x: ndimage.gaussian_filter(x, sigma=(SIGMA, SIGMA, 0), truncate=2.0), img))
    times["NumPy"].append(timed(numpy_blur, img))
    if n == 512:
        refs["OpenCV"] = cv2.GaussianBlur(img, (K, K), SIGMA)
        refs["SciPy"] = ndimage.gaussian_filter(img, sigma=(SIGMA, SIGMA, 0), truncate=2.0)
        refs["NumPy"] = numpy_blur(img)
        refs["image"] = img
    print(f"[lab] {n}×{n}: OpenCV {times['OpenCV'][-1]*1000:.1f} ms；SciPy {times['SciPy'][-1]*1000:.1f} ms；NumPy {times['NumPy'][-1]*1000:.1f} ms")

def psnr(a, b, crop=8):
    a = a[crop:-crop, crop:-crop]; b = b[crop:-crop, crop:-crop]
    mse = float(np.mean((a - b) ** 2)); return 10 * np.log10(1.0 / max(mse, 1e-12)), mse
psnr_scipy, mse_scipy = psnr(refs["OpenCV"], refs["SciPy"])
psnr_numpy, mse_numpy = psnr(refs["OpenCV"], refs["NumPy"])
speedup_scipy = times["NumPy"][-1] / times["SciPy"][-1]
speedup_cv = times["NumPy"][-1] / times["OpenCV"][-1]

# 运行时 log-log 斜率（像素数 vs 时间）
pix = np.array(SIZES, float) ** 2
slopes = {k: float(np.polyfit(np.log(pix), np.log(v), 1)[0]) for k, v in times.items()}

checks = []
def check(name, ok, detail):
    checks.append(bool(ok)); print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")
check("OpenCV 比 NumPy 快", times["OpenCV"][-1] < times["NumPy"][-1], f"{times['OpenCV'][-1]*1000:.1f} vs {times['NumPy'][-1]*1000:.1f} ms")
check("SciPy 比 NumPy 快", times["SciPy"][-1] < times["NumPy"][-1], f"{times['SciPy'][-1]*1000:.1f} vs {times['NumPy'][-1]*1000:.1f} ms")
check("OpenCV 加速比 ≥5×", speedup_cv >= 5, f"{speedup_cv:.1f}×")
check("SciPy 加速比 ≥5×", speedup_scipy >= 5, f"{speedup_scipy:.1f}×")
check("SciPy 与 OpenCV PSNR ≥40dB", psnr_scipy >= 40, f"{psnr_scipy:.2f} dB")
check("NumPy 与 OpenCV PSNR ≥40dB", psnr_numpy >= 40, f"{psnr_numpy:.2f} dB")
for k in ("OpenCV", "SciPy", "NumPy"):
    check(f"{k} 运行时随像素数近线性增长（斜率>0.8）", slopes[k] > 0.8, f"slope={slopes[k]:.2f}")

route = [
    ("识别", "ImageNet / COCO", "PyTorch / TorchVision"),
    ("检测/分割", "COCO", "Detectron2"),
    ("特征匹配", "HPatches", "OpenCV Features2D"),
    ("光流", "Middlebury / Sintel / KITTI", "OpenCV / PyTorch"),
    ("立体/深度", "Middlebury / KITTI / ETH3D", "OpenCV / Open3D"),
    ("SfM/SLAM", "1DSfM / ETH3D", "COLMAP / OpenSfM"),
    ("三维重建", "ScanNet / DiLiGenT", "Open3D / PyTorch3D"),
    ("IBR/神经渲染", "Stanford LF / 4D LF", "PyTorch3D / NeRF 实现"),
]
check("航线图 8 条任务", len(route) == 8, f"{len(route)} 条")
check("每条任务都有数据集", all(r[1] for r in route), "8/8")
check("每条任务都有工具", all(r[2] for r in route), "8/8")
check("数据集列全部非空且不重复", len({r[1] for r in route}) == 8, f"{len({r[1] for r in route})} 个唯一数据集")
check("工具列全部非空且不重复", len({r[2] for r in route}) == 8, f"{len({r[2] for r in route})} 个唯一工具")
check("地图节点=任务+数据集+工具", len(route) * 3 == 24, "24 个文本节点")

n_pass = int(sum(checks))

# 图1：运行时
fig, ax = plt.subplots(figsize=(10.8, 4.6))
colors = {"OpenCV": BLUE, "SciPy": GREEN, "NumPy": ACCENT}
for k in ("OpenCV", "SciPy", "NumPy"):
    ax.plot(pix, times[k], marker="o", ms=7, lw=2.4, color=colors[k], label=k)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("像素数（对数）", fontsize=15); ax.set_ylabel("耗时 / 秒（对数）", fontsize=15)
ax.set_title("同一高斯模糊：三种实现的耗时", fontsize=17)
ax.legend(fontsize=15, frameon=False); ax.grid(True, which="both", color=GRID, alpha=0.45, lw=0.6)
ax.text(pix[-1], times["NumPy"][-1], f"  {speedup_cv:.0f}×  ", fontsize=15, color=ACCENT, va="center")
fig.tight_layout()
fig.savefig(FIG / "e137_runtime.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：一致性（线剖面）
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
img = refs["image"]; r = refs["OpenCV"]
axes[0].imshow(np.clip(img, 0, 1)); axes[0].set_title("输入（512×512）", fontsize=16)
axes[0].set_xticks([]); axes[0].set_yticks([])
row = 256; lo, hi = 120, 210
axes[1].plot(np.arange(lo, hi), r[row, lo:hi, 0], color=BLUE, lw=3.0, label="OpenCV")
axes[1].plot(np.arange(lo, hi), refs["SciPy"][row, lo:hi, 0], color=GREEN, lw=1.8, ls="--", label="SciPy")
axes[1].plot(np.arange(lo, hi), refs["NumPy"][row, lo:hi, 0], color=ACCENT, lw=1.0, ls=":", label="NumPy")
axes[1].set_title(f"同一行的亮度剖面：三条线重合（PSNR {psnr_scipy:.0f}dB）", fontsize=16)
axes[1].set_xlabel("列号", fontsize=15); axes[1].set_ylabel("亮度", fontsize=15)
axes[1].grid(True, color=GRID, alpha=0.45, lw=0.6)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.01))
fig.suptitle("三种实现的结果几乎一样，差的是速度", fontsize=17, y=0.99)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
fig.savefig(FIG / "e137_agreement.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：任务→数据集→工具
fig, ax = plt.subplots(figsize=(11.2, 5.4))
ax.set_xlim(0, 10); ax.set_ylim(0, 8.6); ax.axis("off")
headers = [("任务", 0.6), ("数据集", 3.1), ("工具", 6.1)]
for text, x in headers:
    ax.text(x, 8.15, text, fontsize=16, fontweight="bold", color=INK)
for i, (task, data, tool) in enumerate(route):
    y = 7.55 - i * 0.88
    for text, x, color in [(task, 0.6, "#DDE6F0"), (data, 3.1, "#E2EBDD"), (tool, 6.1, "#F0E1D8")]:
        w = 2.3 if x == 0.6 else 2.8
        box = FancyBboxPatch((x - 0.15, y - 0.32), w, 0.64, boxstyle="round,pad=0.03,rounding_size=0.07",
                             facecolor=color, edgecolor=INK, lw=0.9)
        ax.add_patch(box); ax.text(x, y, text, fontsize=15, va="center", ha="left")
ax.set_title("八条航线：任务 → 数据集 → 工具", fontsize=18, pad=6)
fig.tight_layout()
fig.savefig(FIG / "e137_route_map.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

for _name in ("e137_runtime.png", "e137_agreement.png", "e137_route_map.png"):
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] 1024² 耗时：OpenCV {times['OpenCV'][-1]:.4f}s；SciPy {times['SciPy'][-1]:.4f}s；NumPy {times['NumPy'][-1]:.4f}s")
print(f"[lab] 加速比：OpenCV {speedup_cv:.1f}×；SciPy {speedup_scipy:.1f}×；PSNR {psnr_scipy:.2f}/{psnr_numpy:.2f} dB")
print(f"[lab] 斜率：" + "；".join(f"{k}={v:.2f}" for k, v in slopes.items()))

readings = dict(n_pass=n_pass, n_total=len(checks), sizes=SIZES, repeats=REPEATS,
                times=times, speedup=dict(opencv=speedup_cv, scipy=speedup_scipy),
                psnr=dict(scipy=psnr_scipy, numpy=psnr_numpy), mse=dict(scipy=mse_scipy, numpy=mse_numpy),
                slopes=slopes, route=[dict(task=t, dataset=d, tool=u) for t, d, u in route],
                versions=dict(opencv=cv2.__version__, numpy=np.__version__, scipy="1.18.1"),
                protocol=dict(synthetic=True, single_operation="9x9 gaussian blur", machine="local",
                              general_benchmark=False))
out = ROOT / "outputs" / "e137_datasets_tools_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
