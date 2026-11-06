#!/usr/bin/env python3
"""E093 HDR 合成：多曝光加权合并 / 帽子权重 / 饱和对照。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 160
EVS = np.arange(-7, -2)
GAMMA = 2.2


def main():
    print(f"[lab] E093 HDR 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    rng = np.random.default_rng(93)
    base = gaussian_filter(rng.normal(size=(N, N)), 2.5)
    base = (base - base.min()) / (base.max() - base.min())
    rad = 10 ** (0.0 + 1.8 * base)
    dr = float(rad.max() / rad.min())
    e = [2.0 ** v for v in EVS]
    obs = np.array([np.round(np.clip((ee * rad) ** (1 / GAMMA), 0, 1) * 255) for ee in e])
    sat = [(o >= 254).mean() for o in obs]
    ests, ws = [], []
    for k in range(len(EVS)):
        o = obs[k] / 255.0
        valid = (o > 0.02) & (o < 0.98)
        ests.append(np.where(valid, (o ** GAMMA) / e[k], np.nan))
        ws.append(np.where(valid, o * (1 - o), 0))
    ests = np.array(ests); ws = np.array(ws)
    wm = np.nansum(ws * ests, axis=0) / np.maximum(np.sum(ws, axis=0), 1e-9)
    naive = np.mean(np.array([(o / 255.0) ** GAMMA / ee for o, ee in zip(obs, e)]), axis=0)
    valid = np.isfinite(wm)
    cover = float(valid.mean())
    rel = lambda a, m: float(np.median(np.abs(a[m] - rad[m]) / rad[m]))
    rel_wm = rel(wm, valid)
    bright = rad > np.percentile(rad, 90)
    rel_bright = rel(wm, valid & bright)
    rel_bright_naive = rel(naive, valid & bright)
    corr = float(np.corrcoef(np.log(wm[valid]), np.log(rad[valid]))[0, 1])
    dr_out = float(np.nanmax(wm[valid]) / np.nanmin(wm[valid]))
    w_hat = np.arange(256) * (255 - np.arange(256))
    hat_ratio = float(w_hat[128] / w_hat[5])

    checks = [
        ("场景动态范围>50", dr > 50, f"{dr:.1f} 倍"),
        ("最长曝光饱和>40%", sat[-1] > 0.40, f"{sat[-1]*100:.1f}%"),
        ("五曝光覆盖全部像素", cover > 0.99, f"{cover*100:.1f}%"),
        ("加权合并误差<1%", rel_wm < 0.01, f"{rel_wm*100:.2f}%"),
        ("合并辐亮度相关>0.999", corr > 0.999, f"r={corr:.5f}"),
        ("合并保留动态范围>50", dr_out > 50, f"{dr_out:.1f} 倍"),
        ("亮区加权误差<1%", rel_bright < 0.01, f"{rel_bright*100:.2f}%"),
        ("亮区朴素平均>10%", rel_bright_naive > 0.10, f"{rel_bright_naive*100:.1f}%"),
        ("帽子权重中间高", hat_ratio > 10, f"中心/边缘={hat_ratio:.1f}×"),
        ("合并饱和像素<1%", 1 - cover < 0.01, f"{(1-cover)*100:.2f}%"),
    ]

    # 图 1：三曝光 + 合并
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.2))
    for ax, k, name in zip(axes[:3], (0, 2, 4), ("短曝光", "中曝光", "长曝光")):
        ax.imshow(obs[k], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name} EV={EVS[k]:+d}\n饱和 {sat[k]*100:.0f}%", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
    axes[3].imshow(np.log10(wm), cmap="inferno"); axes[3].set_title("合并辐亮度 log", fontsize=9.5)
    axes[3].set_xticks([]); axes[3].set_yticks([])
    fig.suptitle("HDR 合成：每张只信自己没饱和的那段", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(FIG / "e093_bracket.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 2：帽子权重 + 亮区误差
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].plot(w_hat / w_hat.max(), color=BLUE, lw=2)
    axes[0].set_xlabel("像素值", fontsize=11); axes[0].set_ylabel("相对权重", fontsize=11)
    axes[0].set_title("帽子权重：两端降到零", fontsize=12); axes[0].grid(alpha=0.3); axes[0].tick_params(labelsize=9)
    axes[1].bar(["加权合并", "朴素平均"], [rel_bright * 100, rel_bright_naive * 100], color=[GREEN, ACCENT])
    axes[1].text(0, rel_bright * 100 + 0.5, f"{rel_bright*100:.2f}%", ha="center", fontsize=10)
    axes[1].text(1, rel_bright_naive * 100 + 0.5, f"{rel_bright_naive*100:.1f}%", ha="center", fontsize=10)
    axes[1].set_title("亮区相对误差（%）", fontsize=12); axes[1].tick_params(labelsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e093_weights.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 3：三阶段管线
    fig, ax = plt.subplots(figsize=(7.6, 2.0)); ax.set_xlim(0, 3); ax.set_ylim(0, 1.3); ax.axis("off")
    stages = [("1 标定", "响应曲线\ng(z)"), ("2 合并", "辐亮度\nlog E"), ("3 色调映射", "显示\nLDR")]
    for j, (name, sub) in enumerate(stages):
        ax.text(0.5 + j, 0.72, f"{name}\n{sub}", ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=BLUE, lw=1.6))
        if j < 2: ax.annotate("", xy=(0.85 + j, 0.72), xytext=(0.65 + j, 0.72),
                              arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.text(1.5, 0.18, "运动分支：对齐/去鬼影/选帧（p.625–626）", ha="center", fontsize=11, color=ACCENT)
    ax.set_title("HDR 三阶段：标定 → 合并 → 色调映射", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "e093_pipeline.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(dr=dr, dr_out=dr_out, sat=[float(x) for x in sat], sat_long=float(sat[-1]),
                    cover=cover, rel_wm=rel_wm, rel_bright=rel_bright,
                    rel_bright_naive=rel_bright_naive, corr=corr, hat_ratio=hat_ratio,
                    rel_pct=round(rel_wm * 100, 2), rel_bright_pct=round(rel_bright * 100, 2),
                    rel_bright_naive_pct=round(rel_bright_naive * 100, 1),
                    sat_long_pct=round(float(sat[-1]) * 100, 1), cover_pct=round(cover * 100, 1), dr_int=round(dr),
                    gamma=GAMMA, ev_min=int(EVS[0]), ev_max=int(EVS[-1]), n_exp=int(len(EVS)),
                    dr_out_int=round(dr_out), n_pass=npass, n_total=len(checks))
    (OUT / "e093_hdr_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print("[lab] readings → e093_hdr_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e093_hdr_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
