#!/usr/bin/env python3
"""E130 光场 lab：9×9 子孔径图、EPI 斜率测深、重聚焦与合成孔径。"""
import datetime
import hashlib
import json
import logging
import pathlib

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parent
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PAPER = "#FAF6EC"; INK = "#2B2B2B"; ACCENT = "#C24E2A"; BLUE = "#3B6EA5"
GREEN = "#4F7A4A"; GRID = "#D8CDBB"; YELLOW = "#E8B21A"; CYAN = "#1B9AAA"; MAGENTA = "#B23A8A"
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

print(f"[lab] E130 光场 开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：二维图像面+9×9合成相机阵列；EPI 只取中心行，不是真实光场相机；不实现完整 Lumigraph")

# 场景：三块正对相机的平面，深度分别为 1.8 / 3.0 / 6.0。
# 每条光线用 (s, u) 参数化：u = s + (X - s) / Z；视差 d = 1/Z。
W = H = 128
N = 9
S = 0.20
ss = np.linspace(-S, S, N)
Z_FG, Z_MID, Z_BG = 1.8, 3.0, 6.0
depths = [Z_FG, Z_MID, Z_BG]
disparities = [1.0 / z for z in depths]
marker_X = {"fg": 0.0, "mid": 1.0, "bg": -1.80}
marker_R = {"fg": 0.075, "mid": 0.125, "bg": 0.22}
marker_colors = {"fg": np.array([1.0, 0.85, 0.05]),
                 "mid": np.array([0.05, 0.95, 1.0]),
                 "bg": np.array([1.0, 0.05, 0.85])}
marker_names = {"fg": "近处", "mid": "中景", "bg": "远处"}
marker_draw = {"fg": YELLOW, "mid": CYAN, "bg": MAGENTA}

def tex_bg(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.10; base[..., 1] = 0.20; base[..., 2] = 0.38
    st = 0.5 + 0.5 * np.cos(2 * np.pi * (X * 0.7 + Y * 0.4) / 0.18)
    ch = ((np.floor(X / 0.20) + np.floor(Y / 0.20)) % 2)
    base[..., 0] += 0.10 * st + 0.04 * ch
    base[..., 1] += 0.14 * st + 0.04 * ch
    base[..., 2] += 0.18 * st + 0.04 * ch
    return np.clip(base, 0, 1), np.ones(X.shape, bool)

def tex_mid(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.08; base[..., 1] = 0.42; base[..., 2] = 0.22
    st = 0.5 + 0.5 * np.cos(2 * np.pi * (X - Y) / 0.12)
    base[..., 0] += 0.10 * st; base[..., 1] += 0.22 * st; base[..., 2] += 0.08 * st
    mask = (X > 0.4) & (X < 1.35) & (np.abs(Y) < 0.8)
    return np.clip(base, 0, 1), mask

def tex_fg(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.82; base[..., 1] = 0.30; base[..., 2] = 0.10
    ch = ((np.floor(X / 0.09) + np.floor(Y / 0.09)) % 2)
    base[..., 0] += 0.14 * ch; base[..., 1] += 0.10 * ch; base[..., 2] += 0.05 * ch
    return np.clip(base, 0, 1), np.hypot(X, Y) < 0.38

def add_marker(X, Y, img, kind):
    x0 = marker_X[kind]; r = marker_R[kind]
    rr = np.hypot(X - x0, Y)
    core = rr < r * 0.65; ring = (rr >= r * 0.65) & (rr < r)
    img[core] = marker_colors[kind] * 0.9 + 0.1
    img[ring] = 0.95
    return img

u = (np.arange(W) + 0.5) / W - 0.5
v = (np.arange(H) + 0.5) / H - 0.5
U, V = np.meshgrid(u, v)
views = np.zeros((N, N, H, W, 3))
for i, sx in enumerate(ss):
    for j, sy in enumerate(ss):
        img = np.zeros((H, W, 3))
        for Z, kind in sorted([(Z_BG, "bg"), (Z_MID, "mid"), (Z_FG, "fg")], reverse=True):
            X = sx + (U - sx) * Z
            Y = sy + (V - sy) * Z
            if kind == "bg":
                c, a = tex_bg(X, Y)
            elif kind == "mid":
                c, a = tex_mid(X, Y)
            else:
                c, a = tex_fg(X, Y)
            c = add_marker(X, Y, c, kind)
            img = np.where(a[..., None], c, img)
        views[i, j] = img

n_rays = N * N * W * H
n_bytes = n_rays * 3
n_mb = n_bytes / 1e6

def marker_centroid(img, color):
    d = np.linalg.norm(img - color, axis=2)
    ys, xs = np.nonzero(d < 0.22)
    band = np.abs(ys - H // 2) < 10
    xs = xs[band]
    return None if len(xs) == 0 else float(xs.mean())

# 中心行 t=0、v=0 的 EPI；每个 marker 在 9 个相机里找像素质心，再拟合 u = a·s + b。
epi = views[:, N // 2, H // 2, :, :]
slopes, intercepts, visible_counts = {}, {}, {}
for kind in ("fg", "mid", "bg"):
    cols = [marker_centroid(views[i, N // 2], marker_colors[kind]) for i in range(N)]
    vis = [(ss[i], c) for i, c in enumerate(cols) if c is not None and 3 < c < W - 3]
    visible_counts[kind] = len(vis)
    sv = np.array([x for x, _ in vis]); uv = np.array([c for _, c in vis]) / W - 0.5
    a, b = np.polyfit(sv, uv, 1)
    slopes[kind] = float(a); intercepts[kind] = float(b)

slope_true = [1.0 - d for d in disparities]
slope_errors = [abs(slopes[k] - t) for k, t in zip(("fg", "mid", "bg"), slope_true)]
max_slope_error = max(slope_errors)

# 重聚焦：对每个子图按 -(1-1/Z_f)·s 平移，再平均；等价于把某个深度的像素对齐。
def refocus(d_f, aperture=4):
    J = np.zeros((H, W, 3)); n = 0
    k = 1.0 - d_f
    for i, sx in enumerate(ss):
        for j, sy in enumerate(ss):
            if abs(i - N // 2) > aperture or abs(j - N // 2) > aperture:
                continue
            J += ndimage.shift(views[i, j], (-k * sy * H, -k * sx * W, 0),
                               order=1, mode="nearest")
            n += 1
    return J / n, n

def local_sharp(img, cx, cy, half=9):
    y0, x0 = int(round(cy)), int(round(cx))
    crop = img[y0 - half:y0 + half + 1, x0 - half:x0 + half + 1]
    lum = crop @ np.array([0.299, 0.587, 0.114])
    return float(ndimage.laplace(lum).var())

focus_ds = np.arange(0.08, 0.701, 0.005)
focus_curves = {k: [] for k in ("fg", "mid", "bg")}
for d in focus_ds:
    J, _ = refocus(d, 4)
    for kind, Z in zip(("fg", "mid", "bg"), depths):
        cx = (marker_X[kind] / Z + 0.5) * W
        focus_curves[kind].append(local_sharp(J, cx, H / 2))
focus_peaks = {k: float(focus_ds[int(np.argmax(focus_curves[k]))]) for k in focus_curves}
focus_errors = [abs(focus_peaks[k] - d) for k, d in zip(("fg", "mid", "bg"), disparities)]
max_focus_error = max(focus_errors)
focus_contrast = {k: float(max(focus_curves[k]) / max(min(focus_curves[k]), 1e-12)) for k in focus_curves}

# 合成孔径：对焦近处，逐步扩大参与平均的相机方块（1→9→25→49→81 台）。
apertures = [0, 1, 2, 3, 4]
aperture_counts, aperture_fg, aperture_bg = [], [], []
for R in apertures:
    J, n = refocus(1.0 / Z_FG, R)
    aperture_counts.append(n)
    center_fg = (marker_X["fg"] / Z_FG + 0.5) * W
    center_bg = (marker_X["bg"] / Z_BG + 0.5) * W
    aperture_fg.append(local_sharp(J, center_fg, H / 2))
    aperture_bg.append(local_sharp(J, center_bg, H / 2))
fg_frac = aperture_fg[-1] / aperture_fg[0]
bg_frac = aperture_bg[-1] / aperture_bg[0]
fg_bg_ratio_full = aperture_fg[-1] / aperture_bg[-1]
epi_pixels = N * W
epi_ratio = n_rays / epi_pixels

refocus_near, _ = refocus(1.0 / Z_FG, 4)
refocus_mid, _ = refocus(1.0 / Z_MID, 4)
refocus_far, _ = refocus(1.0 / Z_BG, 4)
aperture_one, _ = refocus(1.0 / Z_FG, 1)
aperture_all, _ = refocus(1.0 / Z_FG, 4)

protocol = dict(synthetic=True, grid=[N, N], image=[W, H], baseline=2 * S,
                scene="3 fronto-parallel textured planes + 3 colored markers",
                epi="center row t=0, v=0", refocus="shift-and-add over 9×9 views",
                full_lumigraph=False)

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("相机网格与光线数", views.shape == (N, N, H, W, 3) and n_rays == 1327104,
      f"{N}×{N}×{W}×{H}={n_rays}")
check("真实视差=1/Z", all(abs(d - 1 / Z) < 1e-9 for d, Z in zip(disparities, depths)),
      f"{disparities[0]:.4f}/{disparities[1]:.4f}/{disparities[2]:.4f}")
check("三个marker在9台相机中全可见", all(v == N for v in visible_counts.values()),
      f"{visible_counts['fg']}/{visible_counts['mid']}/{visible_counts['bg']}")
check("EPI斜率接近1−1/Z", max_slope_error < 0.02,
      f"{slopes['fg']:.4f}/{slopes['mid']:.4f}/{slopes['bg']:.4f} 误差≤{max_slope_error:.4f}")
check("EPI斜率随深度增大", slopes["fg"] < slopes["mid"] < slopes["bg"],
      f"{slopes['fg']:.4f}/{slopes['mid']:.4f}/{slopes['bg']:.4f}")
check("重聚焦峰值接近真实视差", max_focus_error < 0.025,
      f"{focus_peaks['fg']:.3f}/{focus_peaks['mid']:.3f}/{focus_peaks['bg']:.3f} 误差≤{max_focus_error:.3f}")
check("重聚焦曲线峰值/谷值>20", min(focus_contrast.values()) > 20,
      f"{focus_contrast['fg']:.1f}/{focus_contrast['mid']:.1f}/{focus_contrast['bg']:.1f}")
check("孔径增大背景局部锐度单调下降", all(aperture_bg[i] > aperture_bg[i+1] for i in range(4)),
      f"{aperture_bg[0]:.4f}/{aperture_bg[1]:.4f}/{aperture_bg[2]:.4f}/{aperture_bg[3]:.4f}/{aperture_bg[4]:.4f}")
check("全孔径前景/背景锐度比>10", fg_bg_ratio_full > 10, f"{fg_bg_ratio_full:.1f}")
check("大孔径前景保留>0.2", fg_frac > 0.2, f"{fg_frac:.4f}")
check("4D数据量与单EPI切片压缩比", n_bytes == 3981312 and int(round(epi_ratio)) == 1152,
      f"{n_bytes}字节/{epi_pixels}像素={epi_ratio:.0f}")
check("lab边界声明为合成光场", protocol["synthetic"] is True and protocol["full_lumigraph"] is False,
      f"{N}×{N}合成相机阵列，非真实光场相机")

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：3×3 子孔径图 + 中心行 EPI
fig = plt.figure(figsize=(13.0, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.22)
left = gs[0].subgridspec(3, 3, wspace=0.03, hspace=0.03)
grid_idx = [(0, 0), (0, 4), (0, 8), (4, 0), (4, 4), (4, 8), (8, 0), (8, 4), (8, 8)]
for a, (i, j) in enumerate(grid_idx):
    ax = fig.add_subplot(left[a // 3, a % 3])
    ax.imshow(views[i, j], interpolation="bilinear")
    ax.axis("off")
fig.text(0.245, 0.035, "相机 s →", ha="center", fontsize=18)
fig.text(0.040, 0.50, "相机 t →", va="center", rotation=90, fontsize=18)

ax = fig.add_subplot(gs[1])
epi_up = np.repeat(epi, 12, axis=0)
ax.imshow(epi_up, extent=[-0.5, 0.5, -S, S], origin="lower", aspect="auto",
          interpolation="nearest")
for kind, color in (("fg", YELLOW), ("mid", CYAN), ("bg", MAGENTA)):
    sfit = np.linspace(-S, S, 60)
    ax.plot(slopes[kind] * sfit + intercepts[kind], sfit, color=color, lw=2.6,
            label=f"{marker_names[kind]} {dict(fg=Z_FG, mid=Z_MID, bg=Z_BG)[kind]:.1f} / {slopes[kind]:.3f}")
ax.set_xlabel("图像坐标 u（归一化）", fontsize=18)
ax.set_ylabel("相机位置 s", fontsize=18, labelpad=8)
ax.yaxis.set_label_position("right")
ax.tick_params(labelsize=18)
ax.legend(fontsize=18, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
          frameon=False, handlelength=1.3)
fig.suptitle("同一场景的 9×9 子孔径图与中心行 EPI：轨迹斜率 = 1 − 1/Z", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e130_epi.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：三个对焦位置的图像 + 三条局部锐度曲线
fig = plt.figure(figsize=(13.2, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.24)
left = gs[0].subgridspec(1, 3, wspace=0.04)
for ax, J, title in [(fig.add_subplot(left[0, 0]), refocus_near, f"对焦近处\n d={disparities[0]:.3f}"),
                     (fig.add_subplot(left[0, 1]), refocus_mid, f"对焦中景\n d={disparities[1]:.3f}"),
                     (fig.add_subplot(left[0, 2]), refocus_far, f"对焦远处\n d={disparities[2]:.3f}")]:
    ax.imshow(J, interpolation="bilinear")
    ax.set_title(title, fontsize=18)
    ax.axis("off")
fig.text(0.32, 0.035, "同一组照片，换一个平移量就换一个焦平面", ha="center", fontsize=18)

ax = fig.add_subplot(gs[1])
for kind, color in (("fg", YELLOW), ("mid", CYAN), ("bg", MAGENTA)):
    arr = np.array(focus_curves[kind]); arr = arr / arr.max()
    ax.plot(focus_ds, arr, color=color, lw=2.3, label=marker_names[kind])
    ax.axvline(dict(fg=disparities[0], mid=disparities[1], bg=disparities[2])[kind],
               color=color, ls=":", lw=1.4)
ax.set_xlim(0.08, 0.70); ax.set_ylim(-0.03, 1.06)
ax.set_xlabel("对焦视差 d_f = 1/Z_f", fontsize=18)
ax.set_ylabel("归一化局部锐度", fontsize=18, labelpad=8)
ax.tick_params(labelsize=18); ax.grid(color=GRID, lw=0.6, alpha=0.6)
ax.legend(fontsize=18, loc="lower center", ncol=3, framealpha=0.92, handlelength=1.2)
fig.suptitle("先拍照，后对焦：把不同深度重新对齐", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e130_refocus.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：小孔径 vs 大孔径 + 背景/前景锐度随孔径变化
fig = plt.figure(figsize=(12.8, 4.7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.30)
left = gs[0].subgridspec(1, 2, wspace=0.05)
for ax, J, title in [(fig.add_subplot(left[0, 0]), aperture_one, "孔径 R=1（9 台）"),
                     (fig.add_subplot(left[0, 1]), aperture_all, "孔径 R=4（81 台）")]:
    ax.imshow(J, interpolation="bilinear")
    ax.set_title(title, fontsize=18)
    ax.axis("off")
fig.text(0.30, 0.035, "对焦都在近处：前景清晰，背景随孔径变大而糊", ha="center", fontsize=18)

ax = fig.add_subplot(gs[1])
ax.plot(apertures, np.array(aperture_fg) / aperture_fg[0], "o-", color=ACCENT, lw=2.5,
        markersize=8, label="前景（对焦处）")
ax.plot(apertures, np.array(aperture_bg) / aperture_bg[0], "o-", color=BLUE, lw=2.5,
        markersize=8, label="背景（失焦处）")
ax.set_yscale("log"); ax.set_ylim(0.001, 1.4)
ax.set_xticks(apertures); ax.set_xticklabels(["1", "9", "25", "49", "81"])
ax.set_xlabel("参与平均的相机数", fontsize=18)
ax.set_ylabel("相对单视角锐度（对数）", fontsize=18, labelpad=8)
ax.tick_params(labelsize=18); ax.grid(color=GRID, lw=0.6, alpha=0.6)
ax.legend(fontsize=18, loc="lower left", handlelength=1.3)
ax.text(0.95, 0.93, f"前景保留 {fg_frac:.2f}\n背景只剩 {bg_frac:.3f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=18, color=INK,
        bbox=dict(facecolor=PAPER, edgecolor=GRID, boxstyle="round,pad=0.35"))
fig.suptitle("合成孔径：对焦近处，背景被大光圈糊掉", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e130_aperture.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), grid=[N, N], image=[W, H],
                baseline=2 * S, n_rays=n_rays, n_bytes=n_bytes, n_mb=n_mb,
                depths=depths, disparities=disparities, marker_X=marker_X,
                slopes=[slopes[k] for k in ("fg", "mid", "bg")],
                intercepts=[intercepts[k] for k in ("fg", "mid", "bg")],
                slope_true=slope_true, slope_errors=slope_errors,
                max_slope_error=max_slope_error,
                visible_counts=[visible_counts[k] for k in ("fg", "mid", "bg")],
                focus_grid=focus_ds.round(4).tolist(),
                focus_curves={k: np.round(focus_curves[k], 6).tolist() for k in focus_curves},
                focus_peaks=[focus_peaks[k] for k in ("fg", "mid", "bg")],
                focus_errors=focus_errors, max_focus_error=max_focus_error,
                focus_contrast=[focus_contrast[k] for k in ("fg", "mid", "bg")],
                apertures=apertures, aperture_counts=aperture_counts,
                aperture_fg=aperture_fg, aperture_bg=aperture_bg,
                aperture_fg_display=[round(x, 4) for x in aperture_fg],
                aperture_bg_display=[round(x, 4) for x in aperture_bg],
                focus_contrast_display=[round(x, 1) for x in focus_contrast.values()],
                fg_frac=fg_frac, bg_frac=bg_frac, fg_bg_ratio_full=fg_bg_ratio_full,
                epi_pixels=epi_pixels, epi_ratio=epi_ratio,
                protocol=protocol,
                fig_sizes={"epi": [13.0, 5.6], "refocus": [13.2, 4.9], "aperture": [12.8, 4.7]})
out = ROOT / "outputs" / "e130_light_field_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
for name in ("e130_epi.png", "e130_refocus.png", "e130_aperture.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] E130 光场 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
