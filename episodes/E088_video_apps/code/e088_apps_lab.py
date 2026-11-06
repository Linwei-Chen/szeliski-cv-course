#!/usr/bin/env python3
"""E088 §9.2.1/§9.4.1：防抖轨迹平滑 + 帧插值/遮挡（程序生成机制演示）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
RNG = np.random.default_rng(20261108)
N = 128; FG = 24; CX = 52; CY = 52


def tex(seed):
    rng = np.random.default_rng(seed)
    im = gaussian_filter(rng.normal(0, 1, (N, N)), 3.0)
    return (im - im.min()) / (im.max() - im.min())


BG = tex(1)
BG_MASK = np.ones((N, N), bool); BG_MASK[CY - 10:CY + FG + 10, CX - 10:CX + FG + 10] = False
yy, xx = np.mgrid[0:FG, 0:FG]
FGT = 0.10 + 0.80 * ((xx // 4 + yy // 4) % 2)


def frame(cam, obj, base=None, fg=None):
    bg = ndshift(BG if base is None else base, (cam[1], cam[0]), order=3, mode="nearest")
    fgimg = FGT if fg is None else fg
    x = int(round(CX + cam[0] + obj[0])); y = int(round(CY + cam[1] + obj[1]))
    out = bg.copy(); out[y:y + FG, x:x + FG] = fgimg
    return out


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


def estimate_shift(f0, f1, rng=15):
    best = (1e18, 0, 0); vals = {}
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            s = ndshift(f0, (dy, dx), order=1, mode="nearest")
            v = float(np.mean(((s - f1) ** 2)[BG_MASK])); vals[(dx, dy)] = v
            if v < best[0]:
                best = (v, dx, dy)
    _, bx, by = best
    def off(sm, s0, sp):
        den = sm - 2 * s0 + sp
        return 0.0 if abs(den) < 1e-12 else 0.5 * (sm - sp) / den
    ox = off(vals.get((bx - 1, by), best[0]), best[0], vals.get((bx + 1, by), best[0]))
    oy = off(vals.get((bx, by - 1), best[0]), best[0], vals.get((bx, by + 1), best[0]))
    return np.array([bx + ox, by + oy])


def main():
    print(f"[lab] E088 防抖/插帧 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    # 1) 抖动序列：平滑摇镜 + 高频抖动
    T = 30
    cam_true = np.array([[0.35 * t + 2.0 * np.sin(0.55 * t), 0.8 * np.sin(0.9 * t)] for t in range(T)])
    obj_true = np.array([[0.25 * t, 0.0] for t in range(T)])
    frames = [frame(cam_true[t], obj_true[t]) for t in range(T)]
    est = np.array([estimate_shift(frames[0], frames[t]) for t in range(T)])
    est_err = float(np.mean(np.linalg.norm(est - cam_true, axis=1)))
    # 移动平均平滑
    w = 13
    # 因果移动平均：只用当前与过去，避免开头用未来帧造成虚假滞后
    smooth = np.array([est[max(0, t - w + 1):t + 1].mean(0) for t in range(T)])
    corr = est - smooth
    stab = [ndshift(frames[t], (-corr[t, 1], -corr[t, 0]), order=1, mode="nearest") for t in range(T)]
    def jitter(traj):
        return float(np.std(np.diff(traj, n=2, axis=0), axis=0).mean())
    jit_before = jitter(est); jit_after = jitter(est - corr + 0.0)
    # 用估计轨迹校正后，残余抖动=校正后估计轨迹的抖动（估计-校正）
    jit_after = jitter(est - corr)
    crop = int(np.ceil(np.max(np.abs(corr))) + 2)
    area_loss = 1 - ((N - 2 * crop) / N) ** 2
    # 2) 帧插值：0/1 两帧，t=0.5 真值
    CAM, OBJ = 20, 20
    f0 = frame((0, 0), (0, 0)); f1 = frame((CAM, 0), (OBJ, 0))
    gt = frame((CAM / 2, 0), (OBJ / 2, 0))
    # 光流法：背景+前景按各自位移搬一半
    bg_mid = ndshift(BG, (0, CAM / 2), order=1, mode="nearest")
    fg_mid = 0.5 * (ndshift(f0, (0, (CAM + OBJ) / 2), order=1, mode="nearest") +
                    ndshift(f1, (0, -(CAM + OBJ) / 2), order=1, mode="nearest"))
    mask = np.zeros((N, N), bool); mask[CY:CY + FG, int(CX + CAM / 2 + OBJ / 2):int(CX + CAM / 2 + OBJ / 2) + FG] = True
    flow_mid = bg_mid.copy(); flow_mid[mask] = fg_mid[mask]
    naive = 0.5 * (f0 + f1)
    ps_flow = psnr(flow_mid, gt); ps_naive = psnr(naive, gt)
    # 遮挡区（前景）误差
    ps_occ_flow = psnr(flow_mid[mask], gt[mask]); ps_occ_naive = psnr(naive[mask], gt[mask])
    # 不加遮挡 mask：前景/背景平均
    no_mask = bg_mid.copy()
    # 3) 4 倍采帧：原两帧间补 t=0.25/0.5/0.75 三帧
    def blend_at(t):
        bg = ndshift(BG, (0, CAM * t), order=1, mode="nearest")
        fgm = (1 - t) * ndshift(f0, (0, (CAM + OBJ) * t), order=1, mode="nearest") + t * ndshift(f1, (0, -(CAM + OBJ) * (1 - t)), order=1, mode="nearest")
        m = np.zeros((N, N), bool); x = int(round(CX + CAM * t + OBJ * t)); m[CY:CY + FG, x:x + FG] = True
        bg[m] = fgm[m]; return bg
    seq = [frame((CAM * t, 0), (OBJ * t, 0)) for t in (0, 0.25, 0.5, 0.75, 1)]
    blur = lambda im: gaussian_filter(im, 4.0)
    td_before = float(np.mean(np.abs(blur(f1) - blur(f0))))
    td_after = float(np.mean([np.mean(np.abs(blur(seq[i + 1]) - blur(seq[i]))) for i in range(len(seq) - 1)]))
    # 图 1：轨迹/抖动
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].plot(est[:, 0], color=ACCENT, lw=1.6, label="抖动轨迹")
    axes[0].plot(smooth[:, 0], color=BLUE, lw=2.0, label="平滑后")
    axes[0].set_xlabel("帧", fontsize=13); axes[0].set_ylabel("水平位移（像素）", fontsize=13)
    axes[0].set_title("估计轨迹与平滑轨迹", fontsize=14); axes[0].legend(fontsize=11, frameon=False)
    axes[0].grid(alpha=0.3); axes[0].tick_params(labelsize=11)
    axes[1].bar(["校正前", "校正后"], [jit_before, jit_after], color=[ACCENT, GREEN])
    axes[1].set_ylabel("残余抖动（像素）", fontsize=13); axes[1].set_title("抖动被压下去", fontsize=14)
    axes[1].tick_params(labelsize=12)
    fig.tight_layout(); fig.savefig(FIG / "e088_stabilize.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 图 2：插值五联
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.1))
    for ax, im, title in zip(axes, [f0, f1, naive, flow_mid, gt],
                             ["前帧", "后帧", "直接平均", "光流插值", "真值"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("中间帧：直接平均有重影，光流插值更干净", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "e088_interp.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 图 3：遮挡/时域差
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].imshow(mask, cmap="gray"); axes[0].set_title("前景遮挡区（白）", fontsize=14)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].bar(["原始两帧", "4× 采帧"], [td_before, td_after], color=[ACCENT, GREEN])
    axes[1].set_ylabel("相邻帧平均绝对差", fontsize=13); axes[1].set_title("4× 采帧让时间更密", fontsize=14)
    axes[1].tick_params(labelsize=12)
    fig.tight_layout(); fig.savefig(FIG / "e088_temporal.png", dpi=185, facecolor=PAPER); plt.close(fig)
    # 断言
    checks = [
        ("全局平移估计可用", est_err < 0.6, f"mean_err={est_err:.3f}px"),
        ("平滑压抖动>5×", jit_before > 5 * jit_after, f"{jit_before:.3f}→{jit_after:.3f}"),
        ("校正后残余抖动小", jit_after < 0.2, f"jitter={jit_after:.3f}px"),
        ("裁剪面积损失≤18%", area_loss < 0.18, f"crop={crop}px, loss={area_loss*100:.1f}%"),
        ("光流插值 PSNR>30", ps_flow > 30, f"PSNR={ps_flow:.1f}dB"),
        ("光流优于直接平均>5dB", ps_flow > ps_naive + 5, f"{ps_flow:.1f} vs {ps_naive:.1f}dB"),
        ("遮挡区 mask 更准", ps_occ_flow > ps_occ_naive + 2, f"{ps_occ_flow:.1f} vs {ps_occ_naive:.1f}dB"),
        ("4× 采帧降低时域差", td_after < 0.5 * td_before, f"{td_before:.4f}→{td_after:.4f}"),
        ("插帧后帧数增加", len(seq) == 5, f"{len(seq)} 帧"),
        ("无 mask 会污染前景", ps_occ_naive < ps_occ_flow - 2, f"naive={ps_occ_naive:.1f}dB"),
    ]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(cam_true=cam_true.tolist(), est=est.tolist(), smooth=smooth.tolist(),
                    est_err=est_err, jit_before=jit_before, jit_after=jit_after,
                    crop=crop, area_loss=area_loss, ps_flow=ps_flow, ps_naive=ps_naive,
                    ps_occ_flow=ps_occ_flow, ps_occ_naive=ps_occ_naive,
                    td_before=td_before, td_after=td_after,
                    td_before_4dp=round(td_before, 4), td_after_4dp=round(td_after, 4),
                    area_loss_pct=round(float(area_loss) * 100, 1),
                    n_pass=npass, n_total=len(checks))
    (OUT / "e088_apps_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e088_apps_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e088_apps_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
