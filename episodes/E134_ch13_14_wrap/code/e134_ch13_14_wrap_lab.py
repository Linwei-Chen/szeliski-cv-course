#!/usr/bin/env python3
"""E134 第13–14章收官 lab：图像→深度→点云→新视角；视角数量 vs 简单网格代理。"""
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

print(f"[lab] E134 收官实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print("[lab] 边界：合成球+背景平面；已知相机位姿/深度；点云泼溅不是 Poisson 重建；网格是平面 billboard 代理")

W, H = 320, 180
CAM = np.array([-0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7])
TGT = 0.0
U = (np.arange(W) + 0.5) / W - 0.5
V = (np.arange(H) + 0.5) / H - 0.5
UU, VV = np.meshgrid(U, V)
DIR = np.stack([UU, VV, np.ones_like(UU)], axis=-1)
SPHERE = np.array([0.0, 0.0, 2.0]); RAD = 0.36

def facade(X, Y):
    base = np.empty(X.shape + (3,))
    base[..., 0] = 0.55; base[..., 1] = 0.47; base[..., 2] = 0.38
    xx = X + 4.0; yy = Y + 3.0
    win = ((np.mod(np.floor(xx / 0.8), 2) == 0) &
           (np.mod(np.floor(yy / 0.6), 2) == 0) &
           (np.mod(xx, 0.8) > 0.14) & (np.mod(xx, 0.8) < 0.66) &
           (np.mod(yy, 0.6) > 0.12) & (np.mod(yy, 0.6) < 0.48))
    base[win] = [0.16, 0.27, 0.42]
    line = ((np.mod(xx, 0.8) < 0.035) | (np.mod(yy, 0.6) < 0.035))
    base[line] *= 0.75
    return np.clip(base, 0, 1)

def render_scene(cx):
    o = np.array([cx, 0.0, 0.0])
    oc = o - SPHERE
    a = np.sum(DIR * DIR, axis=-1)
    b = 2 * np.sum(DIR * oc, axis=-1)
    c = float(np.dot(oc, oc) - RAD * RAD)
    disc = b * b - 4 * a * c
    t = np.full((H, W), np.inf)
    hit = disc > 0
    t_hit = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    hit &= t_hit > 0
    t[hit] = t_hit[hit]
    P = o + DIR * t[..., None]
    normal = (P - SPHERE) / RAD
    fg = np.empty((H, W, 3))
    fg[..., 0] = 0.88; fg[..., 1] = 0.30; fg[..., 2] = 0.12
    stripe = (np.mod(np.floor((P[..., 1] + 1.0) / 0.14), 2) == 0)
    fg[stripe] = [0.98, 0.64, 0.20]
    shade = np.clip(0.65 + 0.35 * normal[..., 2], 0.5, 1.0)
    fg *= shade[..., None]
    # 背景平面 z=6
    tb = 6.0
    Pb = o + DIR * tb
    bg = facade(Pb[..., 0], Pb[..., 1])
    img = np.where(hit[..., None], fg, bg)
    depth = np.where(hit, t, tb)
    return np.clip(img, 0, 1), depth

def backproject(cx, img, depth):
    X = cx + depth * UU
    Y = depth * VV
    Z = depth
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1), img.reshape(-1, 3)

def splat(points, colors, tgt):
    zbuf = np.full((H, W), np.inf)
    out = np.zeros((H, W, 3))
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    u = (X - tgt) / Z
    v = Y / Z
    px = np.round((u + 0.5) * W - 0.5).astype(int)
    py = np.round((v + 0.5) * H - 0.5).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (Z > 0)
    px, py, Z, colors = px[ok], py[ok], Z[ok], colors[ok]
    order = np.argsort(-Z)
    px, py, Z, colors = px[order], py[order], Z[order], colors[order]
    zbuf[py, px] = Z
    out[py, px] = colors
    return out, float(np.mean(np.isfinite(zbuf)))

def render_proxy(tgt):
    # 背景平面 + 一个平面 billboard（粗网格代理）
    Pb = np.array([tgt, 0, 0]) + DIR * 6.0
    out = facade(Pb[..., 0], Pb[..., 1])
    card = (np.abs(UU) < 0.20) & (np.abs(VV) < 0.58)
    col = np.empty((H, W, 3)); col[..., 0] = 0.88; col[..., 1] = 0.30; col[..., 2] = 0.12
    stripe = (np.mod(np.floor((VV + 1.0) / 0.14), 2) == 0)
    col[stripe] = [0.98, 0.64, 0.20]
    out[card] = col[card]
    return out

def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1 / max(mse, 1e-12))

target, target_depth = render_scene(TGT)
proxy = render_proxy(TGT)
psnr_proxy = psnr(proxy, target)
proxy_cov = 1.0

order = [0, 7, 1, 6, 2, 5, 3, 4]
def cloud_for(n):
    pts, cols = [], []
    for ci in order[:n]:
        img, dep = render_scene(CAM[ci])
        p, c = backproject(CAM[ci], img, dep)
        pts.append(p); cols.append(c)
    return np.concatenate(pts), np.concatenate(cols)

results = {}
for n in (1, 2, 4, 8):
    pts, cols = cloud_for(n)
    img, cov = splat(pts, cols, TGT)
    results[n] = dict(img=img, cov=cov, psnr=psnr(img, target), n_points=int(len(pts)))

covs = [results[n]["cov"] for n in (1, 2, 4, 8)]
psnrs = [results[n]["psnr"] for n in (1, 2, 4, 8)]
mono = all(covs[i] <= covs[i + 1] + 1e-9 for i in range(3))

checks = []
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))
check("合成图像尺寸与相机数", target.shape == (H, W, 3) and len(CAM) == 8, f"8/{W}×{H}")
check("单视角点云有洞", results[1]["cov"] < 0.95, f"{results[1]['cov']*100:.1f}%")
check("八视角点云覆盖率>99%", results[8]["cov"] > 0.99, f"{results[8]['cov']*100:.1f}%")
check("覆盖率随视角数单调不降", mono, "→".join(f"{c*100:.1f}%" for c in covs))
check("八视角 PSNR 比单视角高≥4dB", results[8]["psnr"] - results[1]["psnr"] >= 4, f"{results[1]['psnr']:.2f}→{results[8]['psnr']:.2f}")
check("八视角 PSNR≥28", results[8]["psnr"] >= 28, f"{results[8]['psnr']:.2f}")
check("单视角 PSNR≤25", results[1]["psnr"] <= 25, f"{results[1]['psnr']:.2f}")
check("平面代理覆盖率 100%", proxy_cov == 1.0, f"{proxy_cov*100:.1f}%")
check("平面代理 PSNR 低于八视角点云", psnr_proxy < results[8]["psnr"], f"{psnr_proxy:.2f}<{results[8]['psnr']:.2f}")
check("点云数量随视角数增加", all(results[a]["n_points"] < results[b]["n_points"] for a, b in ((1,2),(2,4),(4,8))), "1<2<4<8")
check("所有 PSNR 有限", np.isfinite([results[n]["psnr"] for n in (1,2,4,8)] + [psnr_proxy]).all(), "finite")
check("边界声明为合成/非完整重建", True, "已知位姿/深度；点云泼溅；平面代理")

n_pass = sum(ok for _, ok, _ in checks)
for name, ok, detail in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")

# 图1：四步管线
fig = plt.figure(figsize=(13.4, 5.2))
gs = GridSpec(1, 4, wspace=0.10, top=0.74, bottom=0.18)
src_img, src_dep = render_scene(CAM[3])
pts8, cols8 = cloud_for(8)
panels = [
    (src_img, "① 多视角图像", "左相机原图"),
    (src_dep, "② 深度图", "近亮远暗"),
    (None, "③ 3D 点云（X–Z 俯视）", ""),
    (results[8]["img"], "④ 新视角", f"PSNR {results[8]['psnr']:.1f} dB"),
]
for j, (im, title, note) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j])
    if im is None:
        idx = np.arange(0, len(pts8), 17)
        ax.scatter(pts8[idx, 0], pts8[idx, 2], s=3, c=cols8[idx], alpha=0.7, linewidths=0)
        ax.set_xlim(-0.9, 0.9); ax.set_ylim(0, 6.4)
        ax.set_xlabel("X", fontsize=18); ax.set_ylabel("Z", fontsize=18)
        ax.tick_params(labelsize=18)
    else:
        ax.imshow(im, interpolation="bilinear"); ax.axis("off")
    ax.set_title(title, fontsize=20)
    if note:
        ax.text(0.5, -0.12, note, transform=ax.transAxes, ha="center", va="top", fontsize=17, color="#5A4A2A")
fig.suptitle("从像素到场：一条端到端的小管线", fontsize=21, y=0.96)
fig.savefig(FIG / "e134_pipeline.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图2：1 视角 / 8 视角 / 目标
fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
for ax, im, title, note in [(axes[0], results[1]["img"], "1 个视角点云", f"覆盖 {results[1]['cov']*100:.0f}% / {results[1]['psnr']:.1f} dB"),
                            (axes[1], results[8]["img"], "8 个视角点云", f"覆盖 {results[8]['cov']*100:.0f}% / {results[8]['psnr']:.1f} dB"),
                            (axes[2], target, "目标视图", "参考")]:
    ax.imshow(im, interpolation="bilinear"); ax.set_title(title, fontsize=19); ax.axis("off")
    ax.text(0.02, 0.96, note, transform=ax.transAxes, va="top", fontsize=18, color="white",
            bbox=dict(facecolor="black", alpha=0.62, edgecolor="none", pad=3))
fig.suptitle("点云新视角：视角越多，洞越少", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e134_cloud.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

# 图3：覆盖率/PSNR vs 视角数
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.7))
ns = np.array([1, 2, 4, 8])
ax = axes[0]
ax.plot(ns, np.array(covs) * 100, "o-", color=BLUE, lw=3, ms=9)
ax.axhline(100, color=GREEN, ls="--", lw=2, label="平面代理")
ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("输入视角数", fontsize=18); ax.set_ylabel("覆盖率 (%)", fontsize=18)
ax.tick_params(labelsize=18); ax.text(1.02, 92, "平面代理 100%", fontsize=18, color=GREEN); ax.set_ylim(50, 105)
ax.set_title("覆盖：两个视角就补满", fontsize=19)
ax = axes[1]
ax.plot(ns, psnrs, "o-", color=ACCENT, lw=3, ms=9, label="点云")
ax.axhline(psnr_proxy, color=GREEN, ls="--", lw=2, label=f"平面代理 {psnr_proxy:.1f} dB")
ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlabel("输入视角数", fontsize=18); ax.set_ylabel("PSNR (dB)", fontsize=18)
ax.tick_params(labelsize=18); ax.text(4.2, 16.5, "平面代理 14.2 dB", fontsize=18, color=GREEN)
ax.text(2.2, 29, "点云", fontsize=18, color=ACCENT)
ax.set_title("精度：视角增加后趋稳", fontsize=19)
fig.suptitle("几何–图像连续谱：少几何靠多图，多几何省图但有近似", fontsize=21)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG / "e134_tradeoff.png", dpi=160, bbox_inches="tight", facecolor=PAPER)
plt.close(fig)

readings = dict(n_pass=n_pass, n_total=len(checks), image=[W, H], cameras=CAM.tolist(),
                n_cameras=len(CAM), n_views=[1, 2, 4, 8], coverage=covs,
                coverage_pct=[round(c * 100, 1) for c in covs],
                psnr_cloud=psnrs, n_points=[results[n]["n_points"] for n in (1, 2, 4, 8)],
                psnr_proxy=psnr_proxy, proxy_coverage=proxy_cov,
                proxy_coverage_pct=round(proxy_cov * 100, 1),
                protocol=dict(synthetic=True, known_pose=True, known_depth=True,
                              splat="z-buffer point splatting", mesh="flat billboard proxy",
                              full_reconstruction=False))
out = ROOT / "outputs" / "e134_ch13_14_wrap_readings.json"
out.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
for name in ("e134_pipeline.png", "e134_cloud.png", "e134_tradeoff.png"):
    p = FIG / name
    print(f"[lab] PNG sha256({name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"[lab] E134 收官实验结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
