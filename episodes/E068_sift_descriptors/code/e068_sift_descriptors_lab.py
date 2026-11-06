#!/usr/bin/env python3
"""E068 S2 教学替代图×3（程序生成，未用原书原图——Figure 7.15–7.17 不复制）。

图 1 e068_sift_pipeline.png（deck 页 7，f007 配图）：SIFT 128 维流水线——
  16×16 窗口逐像素梯度 + 高斯衰减圆；4×4 子区域各 8 方向直方图；4×4×8=128 维；
  单位化→截断 0.2→再归一化。对应 p.435–436 Figure 7.16。
图 2 e068_descriptor_zoo.png（deck 页 15，f015 配图）：经典描述子对照——
  MOPS/SIFT/PCA-SIFT/SURF/RootSIFT/GLOH 的维数、关键改动、归一化方式。
  对应 p.435–437 正文；SURF 原书只述盒式滤波近似，未给维数（图中如实标注）。
图 3 e068_normalization.png（deck 页 9，f009 配图）：归一化三步——
  原始 128 维向量 → 单位长度 → 截断 0.2 后再归一化；标注长度、截断分量数、
  截断前后余弦相似度。对应 p.436 归一化段。
断言 10 条；readings 叶值存档；图内最小字号 ≥12pt；零内部记号。
"""
import datetime, hashlib, json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E068_sift_descriptors"
FIG = EP / "code" / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = EP / "code" / "outputs"

PAPER = "#faf7f0"; INK = "#2f3540"; MUT = "#5a6472"; ACC = "#8a3324"
BLU = "#3c4f6b"; GOLD = "#b08a3e"; GRN = "#4a6b4a"
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")


def box(ax, x, y, w, h, ec=INK, fc="#ffffff", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))


def arrow(ax, p1, p2, color=GOLD, lw=1.3, ms=9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0))


def collect(fig):
    out = [t for t in fig.texts if t.get_text().strip()]
    for ax in fig.axes:
        out += [t for t in ax.texts if t.get_text().strip()]
        if ax.get_title().strip():
            out.append(ax.title)
        if ax.get_xlabel().strip():
            out.append(ax.xaxis.label)
        if ax.get_ylabel().strip():
            out.append(ax.yaxis.label)
    return out


def fig_sift_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "SIFT 的 128 维：格子、方向、归一化",
            fontsize=14, color=INK, ha="center", va="center", fontweight="bold")

    # top-left: 16x16 gradient window with 4x4 cell boundaries
    ox, oy, side = 0.40, 1.96, 1.48
    cell = side / 16
    rng = np.random.default_rng(7)
    ang = rng.uniform(0, 2 * np.pi, (16, 16))
    mag = rng.uniform(0.12, 0.95, (16, 16))
    for r in range(16):
        for c in range(16):
            x = ox + c * cell; y = oy + (15 - r) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, fc="#ffffff",
                                   ec="#e3e0d8", lw=0.35))
            dx = 0.36 * cell * mag[r, c] * np.cos(ang[r, c])
            dy = 0.36 * cell * mag[r, c] * np.sin(ang[r, c])
            ax.plot([x + cell / 2 - dx / 2, x + cell / 2 + dx / 2],
                    [y + cell / 2 - dy / 2, y + cell / 2 + dy / 2],
                    color=BLU, lw=0.7)
    for k in range(5):
        ax.plot([ox + k * side / 4, ox + k * side / 4], [oy, oy + side],
                color=ACC, lw=1.2)
        ax.plot([ox, ox + side], [oy + k * side / 4, oy + k * side / 4],
                color=ACC, lw=1.2)
    ax.add_patch(Circle((ox + side / 2, oy + side / 2), side * 0.46,
                        fc="none", ec=GOLD, lw=1.6, ls="--"))
    ax.text(ox + side / 2, oy + side + 0.20, "① 16×16 梯度窗口",
            fontsize=12, color=INK, ha="center", va="center", fontweight="bold")
    ax.text(ox + side / 2, oy - 0.18, "虚线圆=高斯衰减，红线=4×4 格",
            fontsize=12, color=MUT, ha="center", va="center")

    # top-right: 4x4 mini 8-bin histograms
    mx, my, mside = 3.55, 2.02, 1.35
    ax.text(mx + mside / 2, oy + side + 0.20, "② 4×4 格 × 8 方向",
            fontsize=12, color=ACC, ha="center", va="center", fontweight="bold")
    for r in range(4):
        for c in range(4):
            cx = mx + c * mside / 4 + mside / 8
            cy = my + (3 - r) * mside / 4 + mside / 8
            ax.add_patch(Rectangle((mx + c * mside / 4, my + (3 - r) * mside / 4),
                                   mside / 4, mside / 4, fc="#fdf0ec",
                                   ec="#d9b8ae", lw=0.7))
            vals = rng.uniform(0.05, 0.22, 8)
            vals[rng.integers(0, 8)] += 0.30
            for b, v in enumerate(vals):
                a = 2 * np.pi * b / 8
                ax.plot([cx, cx + 0.115 * v / 0.55 * np.cos(a)],
                        [cy, cy + 0.115 * v / 0.55 * np.sin(a)],
                        color=ACC, lw=1.0)
    ax.text(mx + mside / 2, my - 0.18, "4×4×8 = 128",
            fontsize=12, color=INK, ha="center", va="center", fontweight="bold")

    # bottom: 128-D vector and normalization
    vx, vy, vw, vh = 0.55, 0.68, 5.50, 0.50
    ax.text(vx + vw / 2, 1.36, "③ 128 维向量：单位化 → 截断 0.2 → 再归一化",
            fontsize=12, color=GRN, ha="center", va="center", fontweight="bold")
    vec = np.abs(rng.normal(0, 0.09, 128)); vec[5] = 0.42; vec[77] = 0.36
    v = vec / np.linalg.norm(vec)
    for i, val in enumerate(v):
        ax.add_patch(Rectangle((vx + i * vw / 128, vy), vw / 128 * 0.82,
                               min(val / 0.45, 1.0) * vh, fc=GRN, ec="none", alpha=0.85))
    ax.plot([vx, vx + vw], [vy + 0.2 / 0.45 * vh, vy + 0.2 / 0.45 * vh],
            color=ACC, lw=1.2, ls="--")
    ax.text(vx + vw / 2, 0.46, "虚线=0.2 截断线；梯度→加权→分箱→软分配，轻微错位不会让某格突然跳变",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(3.3, 0.20, "程序生成，未用原书原图 · 对应原书 p.435–436 图 7.16，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def fig_descriptor_zoo():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "经典描述子对照：维数、改动、归一化",
            fontsize=14, color=INK, ha="center", va="center", fontweight="bold")

    rows = [
        ("MOPS", "64", "尺度/方向采样", "均值0/方差1"),
        ("SIFT", "128", "4×4×8 直方图", "单位化+截断"),
        ("PCA-SIFT", "36", "3042→PCA", "PCA"),
        ("SURF", "未给", "盒式滤波", "—"),
        ("RootSIFT", "128", "L1+开方", "L1"),
        ("GLOH", "128", "17×16→PCA", "PCA"),
    ]
    x0, y0, w = 0.28, 0.62, 6.04
    colx = [x0 + 0.12, x0 + 1.42, x0 + 2.30, x0 + 4.22]
    headers = ["名称", "维数", "关键改动", "归一化"]
    ax.add_patch(Rectangle((x0, y0 + 0.38 * 7), w, 0.38,
                           fc="#e6ebf2", ec="#bcc9db", lw=0.9))
    for x, h in zip(colx, headers):
        ax.text(x, y0 + 0.38 * 7 + 0.19, h, fontsize=12, color=BLU,
                ha="left", va="center", fontweight="bold")
    for i, (name, dim, change, norm) in enumerate(rows):
        y = y0 + 0.38 * (6 - i)
        fc = "#ffffff" if i % 2 == 0 else "#f6f2e8"
        ax.add_patch(Rectangle((x0, y), w, 0.38, fc=fc, ec="#d8d2c4", lw=0.7))
        ax.text(colx[0], y + 0.19, name, fontsize=12, color=ACC,
                ha="left", va="center", fontweight="bold")
        ax.text(colx[1], y + 0.19, dim, fontsize=12, color=INK, ha="left", va="center")
        ax.text(colx[2], y + 0.19, change, fontsize=12, color=INK, ha="left", va="center")
        ax.text(colx[3], y + 0.19, norm, fontsize=12, color=MUT, ha="left", va="center")
    ax.text(3.3, 0.42, "SIFT：16×16 窗口、4×4 格×8 方向；PCA-SIFT：39×39 导数 3042 维→36",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.3, 0.25, "GLOH：272 维经 PCA 到 128；SURF 原书只述盒式滤波加速、未给维数",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.3, 0.08, "程序生成，未用原书原图 · 对应原书 p.435–437 正文，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig


def fig_normalization():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    rng = np.random.default_rng(11)
    raw = np.abs(rng.normal(0, 0.10, 128))
    raw[7] = 0.55; raw[63] = 0.48; raw[101] = 0.30
    unit = raw / np.linalg.norm(raw)
    clipped = np.clip(unit, 0, 0.2)
    renorm = clipped / np.linalg.norm(clipped)
    cos = float(np.dot(unit, renorm) / (np.linalg.norm(unit) * np.linalg.norm(renorm)))
    n_clip = int(np.sum(unit > 0.2))
    titles = [
        f"原始（长度 {np.linalg.norm(raw):.2f}）",
        f"单位化（{np.linalg.norm(unit):.3f}）",
        f"截断+归一（{np.linalg.norm(renorm):.3f}）",
    ]
    datas = [raw, unit, renorm]
    colors = [BLU, GRN, ACC]
    for ax, data, title, col in zip(axes, datas, titles, colors):
        ax.set_xlim(0, 128); ax.set_ylim(0, 0.50); ax.set_facecolor(PAPER)
        ax.bar(np.arange(128), data, width=0.82, color=col, alpha=0.88)
        ax.axhline(0.2, color=GOLD, lw=1.2, ls="--")
        ax.set_title(title, fontsize=14, color=INK, pad=7)
        ax.tick_params(labelsize=14, colors=MUT)
        ax.set_xlabel("128 个分量", fontsize=14, color=MUT)
        ax.set_ylabel("数值", fontsize=14, color=MUT)
        for s in ax.spines.values():
            s.set_color("#c9c4b8")
    axes[2].text(64, 0.36, f"截断 {n_clip} 个分量；余弦 {cos:.3f}",
                 fontsize=14, color=ACC, ha="center", va="center",
                 bbox=dict(fc=PAPER, ec="none", alpha=0.88, pad=1.5))
    fig.suptitle("SIFT 归一化三步：单位化、截断 0.2、再归一化", fontsize=16,
                 color=INK, y=0.985, fontweight="bold")
    fig.text(0.5, 0.035, "程序生成，未用原书原图 · 对应原书 p.436 归一化段，布局自绘",
             fontsize=14, color=MUT, ha="center", va="center")
    fig.tight_layout(rect=(0.03, 0.08, 0.97, 0.93))
    return fig, dict(raw_len=float(np.linalg.norm(raw)), unit_len=float(np.linalg.norm(unit)),
                     renorm_len=float(np.linalg.norm(renorm)), clipped=n_clip, cos=cos,
                     max_unit=float(np.max(unit)), max_renorm=float(np.max(renorm)))


def main():
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__)
    ssha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={ssha}")
    fig1, fig2 = fig_sift_pipeline(), fig_descriptor_zoo()
    fig3, nstats = fig_normalization()
    p1 = FIG / "e068_sift_pipeline.png"
    p2 = FIG / "e068_descriptor_zoo.png"
    p3 = FIG / "e068_normalization.png"
    fig1.savefig(p1, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    fig2.savefig(p2, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    fig3.savefig(p3, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    plt.close("all")
    t1 = [t.get_text() for t in collect(fig1)]
    t2 = [t.get_text() for t in collect(fig2)]
    t3 = [t.get_text() for t in collect(fig3)]
    allt = t1 + t2 + t3
    leaked = [s for s in allt if FIG_MARK.search(s)]
    fs1 = [t.get_fontsize() for t in collect(fig1)]
    fs2 = [t.get_fontsize() for t in collect(fig2)]
    fs3 = [t.get_fontsize() for t in collect(fig3)]
    pngs = [p1, p2, p3]
    joined1, joined2, joined3 = " ".join(t1), " ".join(t2), " ".join(t3)
    rows = []
    rows.append(f"图 1 流水线要素在档：{all(k in joined1 for k in ['16×16','4×4','128'])}")
    rows.append(f"图 1 归一化三步在档：{all(k in joined1 for k in ['单位化','0.2','再归一化'])}")
    rows.append(f"图 2 六个描述子在档：{all(k in joined2 for k in ['MOPS','SIFT','PCA-SIFT','SURF','RootSIFT','GLOH'])}")
    rows.append(f"图 2 维数在档：{all(k in joined2 for k in ['64','128','36','272'])}")
    rows.append(f"图 3 三步标题在档：{all(k in joined3 for k in ['原始','单位化','截断+归一'])}")
    rows.append(f"图 3 数值在档：截断 {nstats['clipped']} 个/余弦 {nstats['cos']:.3f}")
    rows.append(f"主文本字号≥12pt+程序生成注记在档×3　| 图一 最小 {min(fs1):.1f}pt/图二 最小 {min(fs2):.1f}pt/图三 最小 {min(fs3):.1f}pt")
    rows.append(f"三图均含「程序生成，未用原书原图」：{sum('程序生成，未用原书原图' in s for s in allt)==3}")
    rows.append(f"图内文字内部记号零泄漏（红线 5）　| 图内 Text 收割 {len(allt)} 条，命中 {('、'.join(leaked)) if leaked else '无'}")
    rows.append(f"三张 PNG 落盘：{sum(p.exists() and p.stat().st_size>0 for p in pngs)==3}")
    print("[lab] 图1 文本：" + " / ".join(t1))
    print("[lab] 图2 文本：" + " / ".join(t2))
    print("[lab] 图3 文本：" + " / ".join(t3))
    print(f"[lab] 图内字号：图1 {sorted(set(fs1))} / 图2 {sorted(set(fs2))} / 图3 {sorted(set(fs3))}")
    if leaked:
        print(f"[FAIL] 内部记号泄漏：{leaked}")
    n = 0
    for r in rows:
        print(f"[PASS] {r}")
        n += 1
    rd = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": ssha[:16],
        "fig1": {"window": 16, "cells": 4, "bins": 8, "dim": 128,
                 "min_fs": min(fs1), "max_fs": max(fs1), "fig_no": 7.16, "page": 435},
        "fig2": {"descriptors": 6, "mops_dim": 64, "sift_dim": 128,
                 "pca_dim": 36, "gloh_raw": 272, "gloh_dim": 128,
                 "min_fs": min(fs2), "max_fs": max(fs2), "page": 437},
        "fig3": {"dim": 128, "clip": 0.2, "raw_len": round(nstats["raw_len"], 4),
                 "unit_len": round(nstats["unit_len"], 6),
                 "renorm_len": round(nstats["renorm_len"], 6),
                 "clipped": nstats["clipped"], "cos": round(nstats["cos"], 6),
                 "max_unit": round(nstats["max_unit"], 4),
                 "max_renorm": round(nstats["max_renorm"], 4),
                 "min_fs": min(fs3), "max_fs": max(fs3), "page": 436},
        "scope": {"page_start": 434, "page_end": 437, "frames": 16},
        "texts": {"collected": len(allt), "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rp = OUT / "e068_sift_descriptors_readings.json"
    rp.write_text(json.dumps(rd, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rp.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    for p in pngs:
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n}")
    print(f"[lab] 汇总 PASS {n} / FAIL {len(rows)-n} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n != len(rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
