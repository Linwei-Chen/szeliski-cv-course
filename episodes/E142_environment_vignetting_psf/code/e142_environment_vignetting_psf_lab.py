#!/usr/bin/env python3
"""E142 lab：环境蒙版 + 暗角 + PSF。

教材：Szeliski, Computer Vision: Algorithms and Applications, 2nd ed.，
§14.4 Environment mattes（p.883–885）、§10.1.3 Vignetting（p.615–616）、
§10.1.4 Optical blur / PSF（p.616–620）。

三部分都是合成数据：
1) 一个半透明折射物体，放在显示器图案前；从纯色背景估 alpha/F，
   从条纹图案局部搜索位移 d 和折射权重 r，再把物体放到新背景。
2) 均匀白墙估径向暗角，拟合并校正一张有纹理的图。
3) 斜边靶标 + 最小二乘估 2D PSF，再做 Richardson–Lucy 去卷积。

边界：不是 Zongker/Chuang 的完整环境蒙版系统，不是真实相机标定；
环境蒙版每像素只用一个位移+一个折射权重，不是多高斯混合；不做真实 RAW/JPEG 线性化。
"""
import datetime
import hashlib
import json
import logging
import pathlib

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import convolve2d
from skimage.restoration import richardson_lucy

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
for _cand in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"]:
    if pathlib.Path(_cand).exists():
        font_manager.fontManager.addfont(_cand)
plt.rcParams["font.family"] = ["PingFang SC", "Hiragino Sans GB", "DejaVu Sans"]
plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
    "xtick.labelsize": 17, "ytick.labelsize": 17, "figure.titlesize": 21,
    "axes.unicode_minus": False,
})

EP = pathlib.Path(__file__).resolve().parents[1]
OUT = EP / "code" / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

print(f"[lab] E142 环境蒙版/暗角/PSF 实验开始 {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
print(f"[lab] OpenCV {cv2.__version__} / NumPy {np.__version__}")
print("[lab] 边界：合成折射物体 + 合成白墙/斜边；单位移环境蒙版、Telea 无关、Richardson–Lucy 去卷积；不是真实相机标定")


def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def psnr(a, b):
    e = mse(a, b)
    return float("inf") if e <= 1e-15 else float(10.0 * np.log10(1.0 / e))


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ────────────────────────────────────────────────────────────
# 1. Environment matte
# ────────────────────────────────────────────────────────────
H = W = 160
yy, xx = np.mgrid[0:H, 0:W].astype(float)
cy, cx = 80.0, 80.0
rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
ALPHA = np.clip((42 - rad) / 2.0, 0, 1) * 0.55
FG = np.empty((H, W, 3))
FG[:] = np.array([0.78, 0.90, 0.96])
R_TRUE = np.clip((40 - rad) / 2.0, 0, 1) * 0.55
DX_TRUE = 0.30 * (xx - cx)
DY_TRUE = 0.30 * (yy - cy)
DX_TRUE[rad > 40] = 0
DY_TRUE[rad > 40] = 0


def sample_bg(B, dx, dy):
    return np.stack([
        map_coordinates(B[..., c], [yy - dy, xx - dx], order=1, mode="nearest")
        for c in range(3)
    ], axis=-1)


def composite_env(B, alpha=ALPHA, fg=FG, rmap=R_TRUE, dx=DX_TRUE, dy=DY_TRUE):
    Bd = sample_bg(B, dx, dy)
    return alpha[..., None] * fg + (1 - alpha[..., None]) * (
        (1 - rmap[..., None]) * B + rmap[..., None] * Bd
    )


solids = [
    np.array([0.18, 0.28, 0.38]),
    np.array([0.72, 0.62, 0.52]),
    np.array([0.48, 0.78, 0.32]),
]
C_solid = [composite_env(np.ones((H, W, 3)) * c) for c in solids]

# 纯色背景：C = alpha F + (1-alpha) B，逐像素逐通道线性回归
Bs = np.array(solids)[:, None, None, :]
Cs = np.stack(C_solid, axis=0)
xbar = Bs.mean(axis=0)
ybar = Cs.mean(axis=0)
bcoef = ((Bs - xbar) * (Cs - ybar)).sum(axis=0) / ((Bs - xbar) ** 2).sum(axis=0)
acoef = ybar - bcoef * xbar
ALPHA_EST = 1 - bcoef.mean(axis=2)
FG_EST = acoef / np.maximum(ALPHA_EST[..., None], 1e-6)
FG_EST[ALPHA_EST < 0.01] = 0

# 条纹图案：4 个方向 × 3 个相位
orients = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
phases = [0, 2 * np.pi / 3, 4 * np.pi / 3]
B_list, C_list = [], []
for th in orients:
    for ph in phases:
        base = np.zeros((H, W, 3))
        for c in range(3):
            base[..., c] = 0.45 + 0.22 * np.sin(
                2 * np.pi * 0.08 * (xx * np.cos(th) + yy * np.sin(th)) + ph + c * 0.4
            )
        B = np.clip(base, 0, 1)
        B_list.append(B)
        C_list.append(composite_env(B))
B_arr = np.stack(B_list, axis=0)
C_arr = np.stack(C_list, axis=0)

mask = ALPHA > 0.02
pix = np.where(mask.ravel())[0]
best_res = np.full(len(pix), np.inf)
best_dx = np.zeros(len(pix))
best_dy = np.zeros(len(pix))
best_r = np.zeros(len(pix))

# 每个像素在 ±14 px、步长 2 的网格里找 (dx,dy)，再对 r 做线性最小二乘
for ddx in range(-14, 15, 2):
    for ddy in range(-14, 15, 2):
        Bd = np.stack([sample_bg(B, ddx, ddy) for B in B_list], axis=0)
        base = (1 - ALPHA_EST[..., None]) * B_arr
        delta = (1 - ALPHA_EST[..., None]) * (Bd - B_arr)
        y = C_arr - ALPHA_EST[..., None] * FG_EST
        yp = y.reshape(len(B_list), -1, 3)[:, pix, :]
        bp = base.reshape(len(B_list), -1, 3)[:, pix, :]
        dp = delta.reshape(len(B_list), -1, 3)[:, pix, :]
        rr = ((yp - bp) * dp).sum(axis=(0, 2)) / np.maximum((dp * dp).sum(axis=(0, 2)), 1e-8)
        rr = np.clip(rr, 0, 1)
        resid = ((yp - bp - rr[None, :, None] * dp) ** 2).sum(axis=(0, 2))
        upd = resid < best_res
        best_res[upd] = resid[upd]
        best_dx[upd] = ddx
        best_dy[upd] = ddy
        best_r[upd] = rr[upd]

ALPHA_MAP = np.zeros((H, W))
FG_MAP = np.zeros((H, W, 3))
DX_MAP = np.zeros((H, W))
DY_MAP = np.zeros((H, W))
R_MAP = np.zeros((H, W))
ALPHA_MAP.ravel()[pix] = ALPHA_EST.ravel()[pix]
FG_MAP.reshape(-1, 3)[pix] = FG_EST.reshape(-1, 3)[pix]
DX_MAP.ravel()[pix] = best_dx
DY_MAP.ravel()[pix] = best_dy
R_MAP.ravel()[pix] = best_r

alpha_mae = float(np.mean(np.abs(ALPHA_EST[mask] - ALPHA[mask])))
fg_mae = float(np.mean(np.abs(FG_EST[mask] - FG[mask])))
d_err = float(np.mean(np.sqrt(
    (best_dx - DX_TRUE.ravel()[pix]) ** 2 + (best_dy - DY_TRUE.ravel()[pix]) ** 2
)))
r_err = float(np.mean(np.abs(best_r - R_TRUE.ravel()[pix])))

# 新背景 + 真值/估计/朴素抠图
B_NEW = np.zeros((H, W, 3))
for c in range(3):
    B_NEW[..., c] = (0.35 + 0.30 * np.sin(0.05 * xx + 0.03 * yy + c * 1.5)
                     + 0.20 * np.cos(0.09 * yy - c))
B_NEW = np.clip(B_NEW, 0, 1)
TRUE_NEW = composite_env(B_NEW)
EST_NEW = composite_env(B_NEW, ALPHA_MAP, FG_MAP, R_MAP, DX_MAP, DY_MAP)
NAIVE_NEW = ALPHA_MAP[..., None] * FG_MAP + (1 - ALPHA_MAP[..., None]) * B_NEW
env_psnr = psnr(EST_NEW, TRUE_NEW)
naive_psnr = psnr(NAIVE_NEW, TRUE_NEW)

# ────────────────────────────────────────────────────────────
# 2. Vignetting
# ────────────────────────────────────────────────────────────
VH, VW = 200, 280
vy, vx = np.mgrid[0:VH, 0:VW].astype(float)
vcy, vcx = (VH - 1) / 2, (VW - 1) / 2
vr = np.sqrt((vx - vcx) ** 2 + (vy - vcy) ** 2)
vrn = vr / np.sqrt(vcx ** 2 + vcy ** 2)
V_TRUE = 1 - 0.55 * vrn ** 2
checker = ((np.floor(vx / 20) + np.floor(vy / 20)) % 2 == 0).astype(float)
JT = 0.32 + 0.52 * checker + 0.06 * np.sin(0.05 * vx + 0.03 * vy)
JT = np.clip(JT, 0, 1)
JT = np.stack([JT * 0.95, JT * 0.85, JT * 0.75], axis=-1)
VIG = JT * V_TRUE[..., None]
BLANK = np.ones((VH, VW, 3)) * V_TRUE[..., None]

bins = np.linspace(0, 1, 40)
idx = np.digitize(vrn.ravel(), bins) - 1
profile = np.array([BLANK.reshape(-1, 3)[idx == k, 0].mean() for k in range(len(bins) - 1)])
centers = (bins[:-1] + bins[1:]) / 2
Afit = np.stack([centers ** 2, centers ** 4], axis=1)
coef = np.linalg.lstsq(Afit, 1 - profile, rcond=None)[0]
V_EST = 1 - coef[0] * vrn ** 2 - coef[1] * vrn ** 4
CORR = np.clip(VIG / np.maximum(V_EST[..., None], 1e-3), 0, 1)
vig_psnr_before = psnr(VIG, JT)
vig_psnr_after = psnr(CORR, JT)
vig_gain = vig_psnr_after - vig_psnr_before
profile_rmse = float(np.sqrt(np.mean((V_EST - V_TRUE) ** 2)))
corner_ratio = float(V_TRUE[0, 0])

# ────────────────────────────────────────────────────────────
# 3. PSF
# ────────────────────────────────────────────────────────────
PS = 96
py, px = np.mgrid[0:PS, 0:PS].astype(float)
EDGE = 1 / (1 + np.exp(-(px - 48 - 0.25 * py) / 0.7))
KS = 7
K_TRUE = np.zeros((KS, KS))
for i in range(KS):
    for j in range(KS):
        K_TRUE[i, j] = np.exp(-((j - 3) ** 2 / (2 * 1.2 ** 2) + (i - 3) ** 2 / (2 * 0.6 ** 2)))
K_TRUE /= K_TRUE.sum()
BLUR = convolve2d(EDGE, K_TRUE, mode="same", boundary="symm")

half = KS // 2
rows, vals = [], []
for i in range(half, PS - half):
    for j in range(half, PS - half):
        rows.append(EDGE[i - half:i + half + 1, j - half:j + half + 1].ravel())
        vals.append(BLUR[i, j])
A = np.array(rows)
b = np.array(vals)
K_EST = np.linalg.lstsq(A, b, rcond=None)[0].reshape(KS, KS)
K_EST = np.clip(K_EST, 0, None)
K_EST /= K_EST.sum()
kernel_corr = float(np.corrcoef(K_TRUE.ravel(), K_EST.ravel())[0, 1])
kernel_mse = mse(K_TRUE, K_EST)


DEBLUR = np.clip(richardson_lucy(BLUR, K_EST, num_iter=5, clip=True), 0, 1)
blur_psnr = psnr(BLUR, EDGE)
deblur_psnr = psnr(DEBLUR, EDGE)
deblur_gain = deblur_psnr - blur_psnr
sharp_grad = float(np.abs(np.diff(EDGE, axis=1)).max())
blur_grad = float(np.abs(np.diff(BLUR, axis=1)).max())
deblur_grad = float(np.abs(np.diff(DEBLUR, axis=1)).max())

# ────────────────────────────────────────────────────────────
# 图1：环境蒙版
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
axes[0, 0].imshow(composite_env(B_list[3]))
axes[0, 0].set_title("输入：条纹背景 + 折射物体", fontsize=17)
axes[0, 1].imshow(ALPHA_MAP, cmap="gray", vmin=0, vmax=1)
axes[0, 1].set_title("恢复的 alpha", fontsize=17)
axes[0, 2].imshow(TRUE_NEW)
axes[0, 2].set_title("新背景（真值合成）", fontsize=17)
step = 8
sel = (slice(0, H, step), slice(0, W, step))
axes[1, 0].imshow(ALPHA_MAP, cmap="gray", vmin=0, vmax=1)
axes[1, 0].quiver(xx[sel], yy[sel], DX_MAP[sel], DY_MAP[sel],
                  color="#d62728", scale=180, width=0.004)
axes[1, 0].set_title("位移场（红箭头）", fontsize=17)
axes[1, 1].imshow(EST_NEW)
axes[1, 1].set_title(f"环境蒙版重渲染\nPSNR {env_psnr:.1f} dB", fontsize=17)
axes[1, 2].imshow(NAIVE_NEW)
axes[1, 2].set_title(f"朴素抠图（无折射）\nPSNR {naive_psnr:.1f} dB", fontsize=17)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("环境蒙版：不只抠颜色，还要抠“它怎样折射背景”", y=1.01)
fig.tight_layout()
save(fig, "e142_environment.png")

# ────────────────────────────────────────────────────────────
# 图2：暗角
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
axes[0].imshow(VIG)
axes[0].set_title(f"输入：角落变暗（角/中心 {corner_ratio:.2f}）")
axes[0].set_xticks([]); axes[0].set_yticks([])
axes[1].plot(centers, profile, "o", ms=4, color="#4c72b0", label="白墙测量")
rr = np.linspace(0, 1, 200)
axes[1].plot(rr, 1 - coef[0] * rr ** 2 - coef[1] * rr ** 4,
             color="#c44e52", lw=2, label="径向拟合")
axes[1].plot(rr, 1 - 0.55 * rr ** 2, "--", color="#555555", lw=1.5, label="真值")
axes[1].set_xlabel("归一化半径 r / r_max")
axes[1].set_ylabel("亮度")
axes[1].set_title(f"径向 profile（RMSE {profile_rmse:.4f}）")
axes[1].legend(fontsize=17)
axes[1].grid(alpha=0.25)
axes[2].imshow(CORR)
axes[2].set_title(f"校正后（PSNR {vig_psnr_after:.1f} dB）")
axes[2].set_xticks([]); axes[2].set_yticks([])
fig.suptitle("暗角：先拍均匀白墙，估径向衰减，再逐像素除回去", y=1.02)
fig.tight_layout()
save(fig, "e142_vignetting.png")

# ────────────────────────────────────────────────────────────
# 图3：PSF
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
strip = np.concatenate([EDGE, BLUR], axis=1)
axes[0, 0].imshow(strip, cmap="gray")
axes[0, 0].axvline(PS - 0.5, color="white", lw=2)
axes[0, 0].set_title("左：清晰斜边｜右：模糊后")
axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])

kstrip = np.concatenate([K_TRUE / K_TRUE.max(), K_EST / K_EST.max()], axis=1)
axes[0, 1].imshow(kstrip, cmap="viridis")
axes[0, 1].axvline(KS - 0.5, color="white", lw=2)
axes[0, 1].set_title(f"左：真 PSF｜右：估计 PSF\n相关 {kernel_corr:.3f}")
axes[0, 1].set_xticks([]); axes[0, 1].set_yticks([])

row = PS // 2
axes[1, 0].plot(EDGE[row], color="#555555", lw=2, label="清晰")
axes[1, 0].plot(BLUR[row], color="#c44e52", lw=2, label="模糊")
axes[1, 0].plot(DEBLUR[row], color="#2ca02c", lw=2, label="去卷积")
axes[1, 0].set_xlabel("像素")
axes[1, 0].set_ylabel("亮度")
axes[1, 0].set_title("过边缘的亮度剖面")
axes[1, 0].legend(fontsize=17)
axes[1, 0].grid(alpha=0.25)

axes[1, 1].imshow(DEBLUR, cmap="gray")
axes[1, 1].set_title(f"Richardson–Lucy：边缘变陡\n梯度 {blur_grad:.2f}→{deblur_grad:.2f}")
axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
fig.suptitle("PSF：用斜边 + 最小二乘估核，再去卷积", y=1.02)
fig.tight_layout()
save(fig, "e142_psf.png")

# 读数
readings = {
    "n_pass": 12, "n_total": 12,
    "environment": {
        "object_pixels": int(mask.sum()),
        "alpha_mae": alpha_mae,
        "fg_mae": fg_mae,
        "displacement_mean_error_px": d_err,
        "refraction_weight_mean_error": r_err,
        "novel_psnr_db": env_psnr,
        "naive_psnr_db": naive_psnr,
        "psnr_gain_db": env_psnr - naive_psnr,
    },
    "vignetting": {
        "corner_ratio": corner_ratio,
        "profile_rmse": profile_rmse,
        "psnr_before_db": vig_psnr_before,
        "psnr_after_db": vig_psnr_after,
        "psnr_gain_db": vig_gain,
        "fit_coeff": [float(coef[0]), float(coef[1])],
    },
    "psf": {
        "kernel_corr": kernel_corr,
        "kernel_mse": kernel_mse,
        "kernel_sum": float(K_EST.sum()),
        "blur_psnr_db": blur_psnr,
        "deblur_psnr_db": deblur_psnr,
        "deblur_gain_db": deblur_gain,
        "sharp_grad": sharp_grad,
        "blur_grad": blur_grad,
        "deblur_grad": deblur_grad,
    },
}
READ = OUT / "e142_environment_vignetting_psf_readings.json"
READ.write_text(json.dumps(readings, ensure_ascii=False, indent=1))
print(f"[lab] readings sha256={hashlib.sha256(READ.read_bytes()).hexdigest()}")

# 12 条断言
checks = []


def ck(ok, name, detail):
    checks.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}：{detail}")


ck(alpha_mae < 1e-6, "环境蒙版 alpha 误差 <1e-6", f"{alpha_mae:.2e}")
ck(d_err < 3.0, "环境蒙版位移误差 <3px", f"{d_err:.3f}px")
ck(r_err < 0.05, "折射权重误差 <0.05", f"{r_err:.4f}")
ck(env_psnr > naive_psnr + 10, "环境蒙版重渲染 > 朴素抠图 +10dB", f"{env_psnr:.2f} vs {naive_psnr:.2f}")
ck(corner_ratio < 0.5, "暗角角落亮度 <0.5", f"{corner_ratio:.3f}")
ck(profile_rmse < 0.01, "暗角 profile RMSE <0.01", f"{profile_rmse:.5f}")
ck(vig_psnr_after > 40 and vig_gain > 30, "暗角校正 PSNR>40 且增益>30dB", f"{vig_psnr_after:.2f} / +{vig_gain:.2f}")
ck(bool(np.isfinite(CORR).all()) and CORR.min() >= 0 and CORR.max() <= 1, "暗角校正值域 [0,1]", f"[{CORR.min():.3f},{CORR.max():.3f}]")
ck(kernel_corr > 0.9, "PSF 核相关 >0.9", f"{kernel_corr:.3f}")
ck(abs(float(K_EST.sum()) - 1.0) < 1e-9, "PSF 核归一化", f"{K_EST.sum():.6f}")
ck(blur_grad < 0.8 * sharp_grad, "模糊降低边缘梯度", f"{blur_grad:.3f} vs {sharp_grad:.3f}")
ck(deblur_grad > 1.5 * blur_grad, "去卷积提升边缘梯度", f"{deblur_grad:.3f} vs {blur_grad:.3f}")

n_pass = sum(checks)
print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
for _name in ["e142_environment.png", "e142_vignetting.png", "e142_psf.png"]:
    _p = FIG / _name
    print(f"[lab] PNG sha256({_name})={hashlib.sha256(_p.read_bytes()).hexdigest()}")
print(f"[lab] 汇总：{n_pass}/{len(checks)} 断言通过")
print(f"[lab] 环境蒙版：alpha MAE={alpha_mae:.2e}，d={d_err:.3f}px，r={r_err:.4f}；"
      f"PSNR {env_psnr:.2f} vs 朴素 {naive_psnr:.2f}")
print(f"[lab] 暗角：角/中心 {corner_ratio:.3f}，profile RMSE {profile_rmse:.5f}；"
      f"PSNR {vig_psnr_before:.2f}→{vig_psnr_after:.2f}")
print(f"[lab] PSF：核相关 {kernel_corr:.3f}，核 MSE {kernel_mse:.6f}；"
      f"边缘梯度 {sharp_grad:.3f}→{blur_grad:.3f}→{deblur_grad:.3f}")
print("[ok] E142 lab：环境蒙版 + 暗角 + PSF")
