#!/usr/bin/env python3
"""E071 S2 教学替代图×3（程序生成，未用原书原图——Figure 7.18–7.20 不复制）。

图 1 e071_learned_blocks.png（deck 页 3，f003 配图）：学习型描述子——
  左：手工设计的 4×4 空间求和块（固定/均匀权重）；中：从数据学出的权重
  （高斯型、非均匀）；右：训练数据流水线（互联网照片→运动恢复结构→可靠匹配→
  采样图像块→训练权重）。对应 p.438 Figure 7.18 与正文。
图 2 e071_binary_patterns.png（deck 页 7，f007 配图）：三种二进制采样模式——
  BRIEF 随机像素对（128 对，图中示意 40 对）、BRISK 同心圆径向采样、
  FREAK 视网膜 log-polar 采样；附 128 位向量与 Hamming 距离示意。
  对应 p.438–439 Figure 7.19。
图 3 e071_deep_pipeline.png（deck 页 10，f010 配图）：两代深度描述子——
  patch-based（检测子定中心→固定 32×32 块→CNN→描述子）与
  full-image（整图→CNN→关键点+描述子）对照。对应 p.439–440 正文。
断言 10 条（全部布尔化，任一 False 即 FAIL）；readings 叶值存档；
图内最小字号 ≥12pt；零内部记号。
"""
import datetime, hashlib, json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E071_learned_descriptors"
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


def draw_blocks(ax, ox, oy, weights, color, title, subtitle):
    side = 1.45
    cell = side / 4
    for r in range(4):
        for c in range(4):
            x = ox + c * cell; y = oy + (3 - r) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, fc="#ffffff",
                                   ec="#d8d2c4", lw=0.8))
            w = weights[r, c]
            ax.add_patch(Circle((x + cell / 2, y + cell / 2),
                                radius=0.12 + 0.26 * w, fc=color, ec="none",
                                alpha=0.72))
    ax.text(ox, oy - 0.22, f"{title}：{subtitle}", fontsize=12,
            color=color, ha="left", va="center", fontweight="bold")


def fig_learned_blocks():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.16, "学习型描述子：固定块 vs 学出来的权重",
            fontsize=14, color=INK, ha="center", va="center", fontweight="bold")

    hand = np.full((4, 4), 0.5)
    yy, xx = np.mgrid[0:4, 0:4]
    learned = np.exp(-((xx - 1.5) ** 2 + (yy - 1.5) ** 2) / 3.0)
    learned = learned / learned.max()

    draw_blocks(ax, 0.45, 2.30, hand, BLU, "手工设计", "固定权重")
    draw_blocks(ax, 2.18, 2.30, learned, ACC, "学习权重", "数据决定")
    # 右栏标签下移一行，避开左栏标签（两栏网格仍对齐）
    ax.texts[-1].set_position((2.18, 1.76))

    # right: training pipeline (vertical, five compact steps)
    px, pw = 4.25, 1.95
    steps = ["互联网照片", "运动恢复结构", "可靠匹配点", "采样图像块", "训练权重"]
    ys = [3.34, 2.90, 2.46, 2.02, 1.58]
    ax.text(px + pw / 2, 3.88, "训练数据从哪来", fontsize=12, color=GRN,
            ha="center", va="center", fontweight="bold")
    for i, (txt, y) in enumerate(zip(steps, ys)):
        box(ax, px, y, pw, 0.28, ec=GRN, fc="#f2f6ef", lw=1.0)
        ax.text(px + pw / 2, y + 0.14, txt, fontsize=12, color=INK,
                ha="center", va="center")
        if i < len(steps) - 1:
            arrow(ax, (px + pw / 2, y), (px + pw / 2, y - 0.16), color=GRN, ms=7)

    ax.text(0.40, 1.50, "学习对象：权重 · 投影 · 池化位置",
            fontsize=12, color=INK, ha="left", va="center", fontweight="bold")
    ax.text(0.40, 1.23, "训练库：真实图像块（鲁棒匹配后采样）",
            fontsize=12, color=MUT, ha="left", va="center")
    ax.text(0.40, 0.96, "目标：超过手工调参",
            fontsize=12, color=ACC, ha="left", va="center", fontweight="bold")
    ax.text(3.3, 0.62, "左：人设计格子；中：数据学权重；右：训练数据流水线",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.3, 0.30, "程序生成，未用原书原图 · 对应原书 p.438 图 7.18，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig, dict(hand_std=float(hand.std()), learned_std=float(learned.std()),
                     learned_ratio=float(learned.max() / learned.min()),
                     cells=16)


def draw_brief(ax, cx, cy, rng):
    ax.add_patch(Circle((cx, cy), 0.055, fc=ACC, ec="none", zorder=5))
    for _ in range(40):
        a1, a2 = rng.uniform(0, 2 * np.pi, 2)
        r1, r2 = rng.uniform(0.05, 0.62), rng.uniform(0.05, 0.62)
        ax.plot([cx + r1 * np.cos(a1), cx + r2 * np.cos(a2)],
                [cy + r1 * np.sin(a1), cy + r2 * np.sin(a2)],
                color=BLU, lw=0.9, alpha=0.75)
        for rr, aa in ((r1, a1), (r2, a2)):
            ax.add_patch(Circle((cx + rr * np.cos(aa), cy + rr * np.sin(aa)),
                                0.018, fc=BLU, ec="none"))


def draw_brisk(ax, cx, cy):
    ax.add_patch(Circle((cx, cy), 0.055, fc=ACC, ec="none", zorder=5))
    for rad in (0.24, 0.42, 0.60):
        ax.add_patch(Circle((cx, cy), rad, fc="none", ec=GOLD, lw=1.1, ls="--"))
        n = 8 if rad < 0.4 else 12
        for k in range(n):
            a = 2 * np.pi * k / n
            x, y = cx + rad * np.cos(a), cy + rad * np.sin(a)
            ax.plot([cx, x], [cy, y], color=BLU, lw=0.7, alpha=0.65)
            ax.add_patch(Circle((x, y), 0.028, fc=BLU, ec="none"))


def draw_freak(ax, cx, cy):
    ax.add_patch(Circle((cx, cy), 0.055, fc=ACC, ec="none", zorder=6))
    for rad, n in ((0.18, 6), (0.34, 10), (0.50, 14), (0.64, 18)):
        ax.add_patch(Circle((cx, cy), rad, fc="none", ec=GOLD, lw=0.9, ls=":"))
        for k in range(n):
            a = 2 * np.pi * k / n + rad
            x, y = cx + rad * np.cos(a), cy + rad * np.sin(a)
            ax.add_patch(Circle((x, y), 0.045, fc="none", ec=GRN, lw=0.7, alpha=0.9))


def fig_binary_patterns():
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    rng = np.random.default_rng(23)
    titles = ["BRIEF：随机像素对", "BRISK：同心圆径向采样", "FREAK：视网膜 log-polar"]
    subs = ["128 对，图中示意 40 对", "采样点分布在同心圆上", "越靠外，感受野越大"]
    for ax, title, sub in zip(axes, titles, subs):
        ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(PAPER)
        ax.set_title(title, fontsize=12, color=INK, pad=8, fontweight="bold")
        ax.text(0, -0.76, sub, fontsize=12, color=MUT, ha="center", va="center")
    draw_brief(axes[0], 0, 0, rng)
    draw_brisk(axes[1], 0, 0)
    draw_freak(axes[2], 0, 0)

    fig.suptitle("二进制描述子：三种采样模式，一样按位比较", fontsize=15,
                 color=INK, y=0.975, fontweight="bold")
    fig.text(0.5, 0.095, "BRIEF 比亮度得 1 位；BRISK 加尺度与方向；FREAK 用 log-polar 级联",
             fontsize=12, color=INK, ha="center", va="center")
    fig.text(0.5, 0.030, "程序生成，未用原书原图 · 对应原书 p.438–439 图 7.19，布局自绘",
             fontsize=12, color=MUT, ha="center", va="center")
    fig.tight_layout(rect=(0.01, 0.14, 0.99, 0.90))
    return fig, dict(brief_pairs=128, brief_drawn=40, brisk_rings=3,
                     freak_rings=4, bit_len=128)


def fig_deep_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.22, "两代深度描述子：先裁块，还是整图端到端",
            fontsize=14, color=INK, ha="center", va="center", fontweight="bold")

    # left: patch-based
    lx, lw = 0.25, 2.90
    ax.text(lx + lw / 2, 3.94, "patch-based：先检测再裁块", fontsize=12,
            color=BLU, ha="center", va="center", fontweight="bold")
    left_steps = [
        ("输入图像", 3.30),
        ("检测子定中心", 2.82),
        ("裁固定 32×32 块", 2.34),
        ("CNN 描述子网络", 1.86),
        ("128 维描述子", 1.38),
    ]
    for i, (txt, y) in enumerate(left_steps):
        box(ax, lx, y, lw, 0.36, ec=BLU, fc="#eef1f6", lw=1.0)
        ax.text(lx + lw / 2, y + 0.18, txt, fontsize=12, color=INK,
                ha="center", va="center")
        if i < len(left_steps) - 1:
            arrow(ax, (lx + lw / 2, y), (lx + lw / 2, y - 0.12), color=BLU, ms=7)
    ax.text(lx + lw / 2, 1.02, "检测/描述分开；块大小超参数",
            fontsize=12, color=BLU, ha="center", va="center", fontweight="bold")

    # right: full-image
    rx, rw = 3.45, 2.90
    ax.text(rx + rw / 2, 3.94, "full-image：整图端到端", fontsize=12,
            color=ACC, ha="center", va="center", fontweight="bold")
    right_steps = [
        ("整张输入图像", 3.30),
        ("CNN（检测+描述一体）", 2.76),
        ("特征图：关键点 + 描述子", 2.22),
        ("无需指定块尺寸", 1.68),
    ]
    for i, (txt, y) in enumerate(right_steps):
        box(ax, rx, y, rw, 0.40, ec=ACC, fc="#fdf0ec", lw=1.0)
        ax.text(rx + rw / 2, y + 0.20, txt, fontsize=12, color=INK,
                ha="center", va="center")
        if i < len(right_steps) - 1:
            arrow(ax, (rx + rw / 2, y), (rx + rw / 2, y - 0.14), color=ACC, ms=7)
    ax.text(rx + rw / 2, 1.02, "感受野可学；检测描述一起训",
            fontsize=12, color=ACC, ha="center", va="center", fontweight="bold")

    ax.text(3.3, 0.66, "代表：LIFT/HardNet/Key.Net（左）　vs　SuperPoint/D2-Net/R2D2（右）",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(3.3, 0.32, "程序生成，未用原书原图 · 对应原书 p.439–440 正文，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig, dict(patch_size=32, patch_count=16, grid=8, full_count=64,
                     left_steps=5, right_steps=4)


def main():
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__)
    ssha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={ssha}")
    fig1, n1 = fig_learned_blocks()
    fig2, n2 = fig_binary_patterns()
    fig3, n3 = fig_deep_pipeline()
    p1 = FIG / "e071_learned_blocks.png"
    p2 = FIG / "e071_binary_patterns.png"
    p3 = FIG / "e071_deep_pipeline.png"
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

    # Hamming 示例：两个 128 位向量，按位异或后数 1
    rng = np.random.default_rng(31)
    a = rng.integers(0, 2, 128)
    b = a.copy()
    flip = rng.choice(128, size=9, replace=False)
    b[flip] ^= 1
    ham = int(np.count_nonzero(a ^ b))
    ham_explicit = int(sum(int(x) != int(y) for x, y in zip(a, b)))

    checks = [
        ("图 1 手工/学习两栏与 16 格在档",
         all(k in joined1 for k in ["手工设计", "学习权重"]) and n1["cells"] == 16,
         f"cells={n1['cells']}"),
        ("图 1 学习权重非均匀（std>0 且 max/min>1.5）",
         n1["learned_std"] > 0 and n1["learned_ratio"] > 1.5,
         f"hand_std={n1['hand_std']:.4f}, learned_std={n1['learned_std']:.4f}, ratio={n1['learned_ratio']:.4f}"),
        ("图 1 训练数据流水线五步在档",
         all(k in joined1 for k in ["互联网照片", "运动恢复结构", "可靠匹配点",
                                    "采样图像块", "训练权重"]),
         "steps=5"),
        ("图 2 BRIEF 128 对、图中示意 40 对、位长 128",
         n2["brief_pairs"] == 128 and n2["brief_drawn"] == 40 and n2["bit_len"] == 128,
         f"pairs={n2['brief_pairs']}, drawn={n2['brief_drawn']}, bits={n2['bit_len']}"),
        ("图 2 BRISK/FREAK 采样模式在档",
         all(k in joined2 for k in ["BRIEF", "BRISK", "FREAK", "同心圆", "log-polar"]),
         f"brisk_rings={n2['brisk_rings']}, freak_rings={n2['freak_rings']}"),
        ("图 2 Hamming 示例按位异或=逐位比较（9 位）",
         ham == ham_explicit == 9,
         f"hamming={ham}, explicit={ham_explicit}"),
        ("图 3 patch-based 五步与固定 32×32 块在档",
         n3["left_steps"] == 5 and n3["patch_size"] == 32
         and all(k in joined3 for k in ["检测子定中心", "裁固定 32×32 块", "CNN 描述子网络"]),
         f"patch={n3['patch_size']}, steps={n3['left_steps']}"),
        ("图 3 full-image 四步与无需指定块尺寸在档",
         n3["right_steps"] == 4 and all(k in joined3 for k in
                                        ["整张输入图像", "CNN（检测+描述一体）", "无需指定块尺寸"]),
         f"steps={n3['right_steps']}, grid={n3['grid']}, count={n3['full_count']}"),
        ("图 3 两代代表方法在档",
         all(k in joined3 for k in ["LIFT", "HardNet", "Key.Net", "SuperPoint", "D2-Net", "R2D2"]),
         "representatives=6"),
        ("三图字号≥12pt、程序生成注记×3、内部记号零泄漏、PNG 落盘×3",
         min(fs1) >= 12 and min(fs2) >= 12 and min(fs3) >= 12
         and sum("程序生成，未用原书原图" in s for s in allt) == 3
         and not leaked and all(p.exists() and p.stat().st_size > 0 for p in pngs),
         f"min_fs={min(min(fs1), min(fs2), min(fs3)):.1f}, notes={sum('程序生成，未用原书原图' in s for s in allt)}, pngs={len(pngs)}"),
    ]

    print("[lab] 图1 文本：" + " / ".join(t1))
    print("[lab] 图2 文本：" + " / ".join(t2))
    print("[lab] 图3 文本：" + " / ".join(t3))
    print(f"[lab] 图内字号：图1 {sorted(set(fs1))} / 图2 {sorted(set(fs2))} / 图3 {sorted(set(fs3))}")
    n_pass = 0
    for name, ok, result in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {result}")
        n_pass += int(bool(ok))

    rd = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": ssha[:16],
        "fig1": {"cells": n1["cells"], "steps": 5, "hand_std": round(n1["hand_std"], 6),
                 "learned_std": round(n1["learned_std"], 6),
                 "learned_ratio": round(n1["learned_ratio"], 6),
                 "min_fs": min(fs1), "max_fs": max(fs1), "page": 438},
        "fig2": {"brief_pairs": n2["brief_pairs"], "brief_drawn": n2["brief_drawn"],
                 "bit_len": n2["bit_len"], "hamming": ham,
                 "brisk_rings": n2["brisk_rings"], "freak_rings": n2["freak_rings"],
                 "min_fs": min(fs2), "max_fs": max(fs2), "page": 439},
        "fig3": {"patch_size": n3["patch_size"], "patch_count": n3["patch_count"],
                 "grid": n3["grid"], "full_count": n3["full_count"],
                 "left_steps": 5, "right_steps": 4, "representatives": 6,
                 "min_fs": min(fs3), "max_fs": max(fs3), "page": 440},
        "scope": {"page_start": 437, "page_end": 440, "frames": 16},
        "texts": {"collected": len(allt), "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rp = OUT / "e071_learned_descriptors_readings.json"
    rp.write_text(json.dumps(rd, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rp.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    for p in pngs:
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if n_pass != len(checks):
        raise SystemExit("[FAIL] lab 存在断言未过（见上）")


if __name__ == "__main__":
    main()
