#!/usr/bin/env python3
"""E098 图像修补：扩散 vs 示例块（Criminisi 优先级）实验。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve, laplace

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"
N = 64; PATCH = 7


def psnr(a, b):
    mse = float(np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))


def make_scene():
    yy, xx = np.mgrid[0:N, 0:N]
    tex = 0.06 * np.sin(2 * np.pi * xx / 5) * np.sin(2 * np.pi * yy / 7)
    J = np.where(yy > 0.75 * xx + 8, 0.72 + tex, 0.24 + tex)
    J = np.clip(J, 0, 1)
    hole = np.zeros((N, N), bool); hole[22:44, 22:44] = True
    return J, hole


def diffusion_fill(img, hole, iters=400):
    out = img.copy()
    k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float) / 4
    for _ in range(iters):
        avg = convolve(out, k, mode="nearest")
        out[hole] = avg[hole]
    return out


def boundary(mask):
    b = np.zeros_like(mask)
    b[1:-1, 1:-1] = mask[1:-1, 1:-1] & (~mask[:-2, 1:-1] | ~mask[2:, 1:-1] | ~mask[1:-1, :-2] | ~mask[1:-1, 2:])
    return b


def exemplar_fill(img, hole, patch=PATCH, priority_mode="criminisi", step=2):
    out = img.copy(); mask = hole.copy(); conf = (~mask).astype(float)
    r = patch // 2
    known = ~mask
    # avoid borders
    valid_src = []
    for cy in range(r, N - r, step):
        for cx in range(r, N - r, step):
            if known[cy-r:cy+r+1, cx-r:cx+r+1].all():
                valid_src.append((cy, cx))
    src = np.array(valid_src)
    for _ in range(400):
        if not mask.any(): break
        b = boundary(mask)
        ys, xs = np.where(b)
        if len(ys) == 0: break
        best = None
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y-r), min(N, y+r+1); x0, x1 = max(0, x-r), min(N, x+r+1)
            yy0, yy1 = y0-(y-r), patch-(y+r+1-y1); xx0, xx1 = x0-(x-r), patch-(x+r+1-x1)
            tgt = out[y0:y1, x0:x1]; tmask = mask[y0:y1, x0:x1]
            c = conf[y0:y1, x0:x1][~tmask].mean() if (~tmask).any() else 0.0
            if priority_mode == "criminisi":
                gy, gx = np.gradient(out)
                my, mx = np.gradient(mask.astype(float))
                nn = np.hypot(mx, my) + 1e-9
                d = abs(gx[y, x] * mx[y, x] / nn[y, x] + gy[y, x] * my[y, x] / nn[y, x])
                p = c * (0.05 + d)
            else:
                p = c
            if best is None or p > best[0]:
                best = (p, y, x)
        _, y, x = best
        y0, y1 = max(0, y-r), min(N, y+r+1); x0, x1 = max(0, x-r), min(N, x+r+1)
        tgt = out[y0:y1, x0:x1]; tmask = mask[y0:y1, x0:x1]
        # search source patch
        cand = []
        for cy, cx in src:
            sy0, sy1 = cy-r, cy+r+1; sx0, sx1 = cx-r, cx+r+1
            if sy0 < y0 or sx0 < x0 or sy1 > y1 + (N - (y1-y0)) or sx1 > x1 + (N - (x1-x0)):
                pass
            sp = out[sy0:sy1, sx0:sx1]
            if sp.shape != tgt.shape: continue
            err = float(np.mean((sp[~tmask] - tgt[~tmask]) ** 2)) if (~tmask).any() else 0
            cand.append((err, cy, cx))
        if not cand: break
        _, cy, cx = min(cand)
        sp = out[cy-r:cy+r+1, cx-r:cx+r+1]
        out[y0:y1, x0:x1][tmask] = sp[tmask]
        conf[y0:y1, x0:x1][tmask] = conf[y0:y1, x0:x1][~tmask].mean() if (~tmask).any() else c
        mask[y0:y1, x0:x1] = False
    return out


def edge_corr(a, b):
    ga = laplace(a); gb = laplace(b)
    return float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1])


def main():
    print(f"[lab] E098 inpainting 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    J, hole = make_scene()
    I = J.copy(); I[hole] = 0.5
    diff = diffusion_fill(I, hole)
    crim = exemplar_fill(I, hole, PATCH, "criminisi")
    onion = exemplar_fill(I, hole, PATCH, "onion")
    # 大洞
    hole_big = np.zeros((N, N), bool); hole_big[16:50, 16:50] = True
    Jb, _ = make_scene(); Ib = Jb.copy(); Ib[hole_big] = 0.5
    diff_big = diffusion_fill(Ib, hole_big); crim_big = exemplar_fill(Ib, hole_big, PATCH, "criminisi")
    # 小洞
    hole_small = np.zeros((N, N), bool); hole_small[28:38, 28:38] = True
    Is = J.copy(); Is[hole_small] = 0.5
    crim_small = exemplar_fill(Is, hole_small, PATCH, "criminisi")
    # patch 大小
    crim_p11 = exemplar_fill(I, hole, 11, "criminisi")
    # 指标
    fill = hole
    tex_true = float(np.var(laplace(J)[fill])); tex_diff = float(np.var(laplace(diff)[fill])); tex_crim = float(np.var(laplace(crim)[fill]))
    edge_diff = edge_corr(diff, J); edge_crim = edge_corr(crim, J); edge_onion = edge_corr(onion, J)
    p_base = psnr(I, J); p_diff = psnr(diff, J); p_crim = psnr(crim, J); p_onion = psnr(onion, J)
    p_small = psnr(crim_small, J); p_big = psnr(crim_big, Jb); p_p11 = psnr(crim_p11, J)

    checks = [
        ("扩散填满洞", np.isfinite(diff).all() and not np.isclose(diff[fill].mean(), 0.5), "ok"),
        ("扩散优于常数填充", p_diff > p_base + 1, f"{p_diff:.2f} vs {p_base:.2f} dB"),
        ("示例块纹理更接近真值", abs(tex_crim-tex_true) < abs(tex_diff-tex_true), f"{tex_crim:.4f} vs {tex_diff:.4f}"),
        ("PSNR 偏爱扩散", p_diff > p_crim, f"{p_diff:.2f} vs {p_crim:.2f} dB"),
        ("简单边缘扩散相关更高", edge_diff > edge_crim, f"{edge_diff:.3f} vs {edge_crim:.3f}"),
        ("大洞更难", p_big < p_crim - 0.5, f"{p_big:.2f} vs {p_crim:.2f} dB"),
        ("小洞更易", p_small > p_crim, f"{p_small:.2f} vs {p_crim:.2f} dB"),
        ("patch 大小有影响", abs(p_p11 - p_crim) > 0.2, f"{abs(p_p11-p_crim):.2f} dB"),
        ("填充来自已知像素", float(np.mean(np.isin(crim[fill], J[~fill]))) > 0.2, "copy"),
        ("两种顺序都填满", np.isfinite(crim).all() and np.isfinite(onion).all(), "ok"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.2))
    for ax, im, t in zip(axes, (J, I, diff, crim), ("真值", "带洞输入", f"扩散 {p_diff:.1f} dB", f"示例块 {p_crim:.1f} dB")):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1); ax.set_title(t, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("图像修补：扩散填色 vs 示例块搬纹理", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e098_methods.png", dpi=185, facecolor=PAPER); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.25))
    axes[0].imshow(diff, cmap="gray", vmin=0, vmax=1); axes[0].set_title(f"扩散 {p_diff:.1f} dB", fontsize=10.5)
    axes[1].imshow(crim, cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"示例块 {p_crim:.1f} dB", fontsize=10.5)
    axes[2].bar(["真值", "扩散", "示例块"], [tex_true, tex_diff, tex_crim], color=[INK, PURPLE, ACCENT])
    axes[2].set_title("洞内纹理能量", fontsize=10.5); axes[2].tick_params(labelsize=10)
    for ax in axes[:2]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("扩散赢 PSNR，示例块赢纹理", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e098_priority.png", dpi=185, facecolor=PAPER); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.25))
    axes[0].imshow(crim_small, cmap="gray", vmin=0, vmax=1); axes[0].set_title(f"小洞 {p_small:.1f} dB", fontsize=10.5)
    axes[1].imshow(crim_big, cmap="gray", vmin=0, vmax=1); axes[1].set_title(f"大洞 {p_big:.1f} dB", fontsize=10.5)
    axes[2].imshow(crim_p11, cmap="gray", vmin=0, vmax=1); axes[2].set_title(f"patch=11 {p_p11:.1f} dB", fontsize=10.5)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("洞越大、patch 越粗，越容易编错", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(FIG / "e098_params.png", dpi=185, facecolor=PAPER); plt.close(fig)

    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(p_base=p_base, p_diff=p_diff, p_crim=p_crim, p_onion=p_onion,
                    p_small=p_small, p_big=p_big, p_p11=p_p11,
                    p_base_2=round(p_base,2), p_diff_2=round(p_diff,2), p_crim_2=round(p_crim,2),
                    p_onion_2=round(p_onion,2), p_small_2=round(p_small,2), p_big_2=round(p_big,2), p_p11_2=round(p_p11,2),
                    tex_true=tex_true, tex_diff=tex_diff, tex_crim=tex_crim,
                    tex_true_4=round(tex_true,4), tex_diff_4=round(tex_diff,4), tex_crim_4=round(tex_crim,4),
                    edge_diff=edge_diff, edge_crim=edge_crim, edge_onion=edge_onion,
                    edge_diff_3=round(edge_diff,3), edge_crim_3=round(edge_crim,3), edge_onion_3=round(edge_onion,3),
                    patch_diff_2=round(abs(p_p11-p_crim),2), n_pass=npass, n_total=len(checks), patch=PATCH)
    (OUT / "e098_inpaint_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for p in sorted(FIG.glob("e098_*.png")):
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] readings → e098_inpaint_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT/'e098_inpaint_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
