#!/usr/bin/env python3
"""E090 复盘课：第 8–9 章 16 站台账 + 三张程序生成图。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP / "code/outputs"; FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER, INK, ACCENT, BLUE, GREEN, PURPLE = "#f6f0e6", "#1c2437", "#a23b2a", "#3d6a99", "#2e7d32", "#7b4f9d"

LEDGER = [
    ("E074", "五变换", "ch8"), ("E075", "LK/最小二乘", "ch8"), ("E076", "单应/DLT", "ch8"),
    ("E077", "全景/捆扎", "ch8"), ("E078", "柱面/球面", "ch8"), ("E079", "混合/曝光", "ch8"),
    ("E080", "手机/360", "ch8"), ("E081", "全景上机", "ch8"),
    ("E082", "运动开篇", "ch9"), ("E083", "平移/SSD", "ch9"), ("E084", "金字塔/FFT", "ch9"),
    ("E085", "参数/样条", "ch9"), ("E086", "KLT", "ch9"), ("E087", "深度光流", "ch9"),
    ("E088", "防抖/插帧", "ch9"), ("E089", "光流/LK 上机", "ch9"),
]
PIPELINE = [
    ("特征", "E065–E072"), ("匹配", "E069/E072"), ("几何校验", "E069/E076"),
    ("估变换", "E074–E078"), ("warp", "E074/E081"), ("混合", "E079/E081"),
]


def main():
    print(f"[lab] E090 复盘 开始 {datetime.now().isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}")
    ids = [r[0] for r in LEDGER]
    nums = [int(i[1:]) for i in ids]
    ch8 = [r for r in LEDGER if r[2] == "ch8"]; ch9 = [r for r in LEDGER if r[2] == "ch9"]

    # 图 1：两行 16 站地图
    fig, ax = plt.subplots(figsize=(7.8, 2.5)); ax.set_xlim(0, 8.5); ax.set_ylim(0, 3); ax.axis("off")
    ax.text(0.05, 2.82, "第 8 章：对齐与拼接", fontsize=13, color=BLUE, fontweight="bold")
    ax.text(0.05, 1.28, "第 9 章：运动估计", fontsize=13, color=GREEN, fontweight="bold")
    for row, (items, y, col) in enumerate([(ch8, 2.35, BLUE), (ch9, 0.8, GREEN)]):
        for j, (eid, name, _) in enumerate(items):
            x = 0.55 + j * 0.98
            ax.text(x, y, f"{eid}\n{name}", ha="center", va="center", fontsize=9.0,
                    bbox=dict(boxstyle="round,pad=0.28", fc="#ffffff", ec=col, lw=1.4))
            if j < 7: ax.annotate("", xy=(x + 0.78, y), xytext=(x + 0.30, y),
                                  arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
    ax.annotate("", xy=(8.05, 2.0), xytext=(8.05, 1.2), arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.6))
    ax.text(8.2, 1.6, "同一问题\n换尺度", rotation=90, va="center", fontsize=10, color=ACCENT)
    ax.text(4.2, 1.62, "特征 → 匹配 → 几何校验 → 变换 → warp → 混合", ha="center", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(FIG / "e090_map.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 2：Ch8 全景六步
    fig, ax = plt.subplots(figsize=(7.6, 1.9)); ax.set_xlim(0, 6); ax.set_ylim(0, 1); ax.axis("off")
    for j, (name, ref) in enumerate(PIPELINE):
        x = 0.45 + j * 0.95
        ax.add_patch(plt.Rectangle((x - 0.36, 0.34), 0.72, 0.44, fc="#ffffff", ec=BLUE, lw=1.5))
        ax.text(x, 0.56, name, ha="center", va="center", fontsize=10.5)
        ax.text(x, 0.16, ref, ha="center", va="center", fontsize=9.0, color="#6b5626")
        if j < 5: ax.annotate("", xy=(x + 0.59, 0.56), xytext=(x + 0.37, 0.56),
                              arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.set_title("第 8 章全景：把两图对齐到同一画布，再决定怎么混", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "e090_pipeline.png", dpi=185, facecolor=PAPER); plt.close(fig)

    # 图 3：Ch9 六级台阶
    labels = ["平移/SSD", "金字塔/FFT/增量", "参数/样条", "KLT", "稠密/深度光流", "应用/上机"]
    refs = ["E083", "E084", "E085", "E086", "E087", "E088/89"]
    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    for i, (x, lab, ref) in enumerate(zip(xs, labels, refs)):
        ax.add_patch(plt.Rectangle((x - 0.38, 0), 0.76, i + 1, fc="#eef2f7", ec=BLUE, lw=1.2))
        ax.text(x, i + 1.18, lab, ha="center", fontsize=9.5)
        ax.text(x, i + 0.45, ref, ha="center", fontsize=9.0, color="#6b5626", rotation=90, va="center")
    ax.set_xlim(-0.6, len(labels) - 0.4); ax.set_ylim(0, 8.0)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("第 9 章：每一步加一种能力，不是换一个名词", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "e090_motion.png", dpi=185, facecolor=PAPER); plt.close(fig)

    checks = [
        ("台账 16 站且 ID 连续", len(LEDGER) == 16 and nums == list(range(74, 90)), f"{ids[0]}–{ids[-1]}, {len(LEDGER)} 站"),
        ("Ch8/Ch9 各 8 站", len(ch8) == 8 and len(ch9) == 8, f"{len(ch8)}+{len(ch9)}"),
        ("全景六步引用齐全", len(PIPELINE) == 6 and all(a and b for a, b in PIPELINE), f"{len(PIPELINE)} 步"),
        ("三图在盘", all((FIG / n).exists() and (FIG / n).stat().st_size > 10000 for n in
                       ("e090_map.png", "e090_pipeline.png", "e090_motion.png")), "map/pipeline/motion"),
    ]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}")
    npass = sum(1 for _, ok, _ in checks if ok)
    readings = dict(ledger=[dict(id=a, name=b, ch=c) for a, b, c in LEDGER],
                    pipeline=[dict(name=a, ref=b) for a, b in PIPELINE],
                    n_stations=len(LEDGER), n_ch8=len(ch8), n_ch9=len(ch9), n_pipeline=len(PIPELINE),
                    n_pass=npass, n_total=len(checks))
    (OUT / "e090_recap_readings.json").write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
    for png in sorted(FIG.glob("*.png")):
        print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
    print("[lab] readings → e090_recap_readings.json")
    print(f"[lab] readings sha256={hashlib.sha256((OUT / 'e090_recap_readings.json').read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {npass}")
    print(f"[lab] 汇总 PASS {npass} / FAIL {len(checks)-npass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.now().isoformat(timespec='seconds')}")
    if npass != len(checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
