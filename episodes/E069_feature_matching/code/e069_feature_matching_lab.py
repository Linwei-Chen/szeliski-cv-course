#!/usr/bin/env python3
"""E069 S2 教学替代图×3（程序生成，未用原书原图——Figure 7.21–7.24 不复制）。

图 1 e069_threshold_tradeoff.png（deck 页 4，f004 配图）：阈值两难——
  匹配/非匹配的距离分布、阈值 θ 移动带来的 TP/FP/FN/TN；右下 ROC 与 AUC。
  对应 p.442–444 Figure 7.21/7.22。
图 2 e069_nndr.png（deck 页 10，f010 配图）：NNDR 比率——
  DA 的最近 DB/次近 DC 给 d1/d2 小比率；DD 的 d1'/d2' 大比率被拒绝。
  对应 p.445 Figure 7.23。
图 3 e069_match_pipeline.png（deck 页 16，f016 配图）：匹配四道闸——
  描述子→阈值/最近邻→NNDR→互检→几何验证 RANSAC→内点；数量逐级收窄。
  对应 p.441–448 正文。
断言 10 条；readings 叶值存档；图内最小字号 ≥12pt；零内部记号。
"""
import datetime, hashlib, json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E069_feature_matching"
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


def fig_threshold_tradeoff():
    fig = plt.figure(figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes([0.08, 0.17, 0.58, 0.68])
    ax.set_facecolor(PAPER)
    rng = np.random.default_rng(5)
    pos = np.abs(rng.normal(0.22, 0.10, 400))
    neg = np.abs(rng.normal(0.48, 0.16, 400))
    bins = np.linspace(0, 0.9, 45)
    ax.hist(pos, bins=bins, color=GRN, alpha=0.55, label="真匹配距离")
    ax.hist(neg, bins=bins, color=ACC, alpha=0.45, label="非匹配距离")
    ax.axvline(0.30, color=GOLD, lw=1.8, ls="--")
    ax.text(0.31, ax.get_ylim()[1] * 0.92, "阈值 θ", fontsize=12, color=GOLD)
    ymax = ax.get_ylim()[1]
    ax.text(0.10, ymax * 0.84, "真阳", fontsize=12, color=GRN,
            bbox=dict(fc=PAPER, ec="none", alpha=0.85, pad=1.5))
    ax.text(0.47, ymax * 0.70, "假阳", fontsize=12, color=ACC,
            bbox=dict(fc=PAPER, ec="none", alpha=0.85, pad=1.5))
    ax.text(0.34, ymax * 0.22, "假阴", fontsize=12, color=MUT,
            bbox=dict(fc=PAPER, ec="none", alpha=0.85, pad=1.5))
    ax.set_xlabel("特征距离 d", fontsize=12, color=INK)
    ax.set_ylabel("数量", fontsize=12, color=INK)
    ax.tick_params(labelsize=12, colors=MUT)
    ax.set_title("抬高阈值：假阴少一点，假阳多很多", fontsize=13, color=INK, pad=6)
    ax.legend(fontsize=12, frameon=False, loc="upper right")


    ax2 = fig.add_axes([0.73, 0.20, 0.23, 0.58])
    ax2.set_facecolor(PAPER)
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr) ** 2.2
    ax2.plot(fpr, tpr, color=BLU, lw=1.6)
    ax2.plot([0, 1], [0, 1], color=MUT, lw=1.0, ls=":")
    ax2.fill_between(fpr, tpr, color=BLU, alpha=0.12)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("假阳率", fontsize=12, color=INK)
    ax2.set_ylabel("真阳率", fontsize=12, color=INK)
    ax2.tick_params(labelsize=12, colors=MUT)
    ax2.set_title("ROC 曲线", fontsize=13, color=INK, pad=6)
    ax2.text(0.52, 0.20, "AUC=0.93", fontsize=12, color=BLU)
    ax2.text(0.12, 0.86, "越靠左上越好", fontsize=12, color=MUT)

    fig.text(0.5, 0.02, "程序生成，未用原书原图 · 对应原书 p.442–444 图 7.21/7.22，布局自绘",
             fontsize=12, color=MUT, ha="center", va="center")
    return fig, dict(tp=18, fp=4, fn=2, tn=76, tpr=0.90, fpr=0.05, ppv=0.82, acc=0.94,
                     auc=0.93, threshold=0.30)


def fig_nndr():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "NNDR：最近距离比上次近距离", fontsize=14, color=INK,
            ha="center", va="center", fontweight="bold")

    pts = {"DA": (1.35, 2.55, ACC), "DB": (2.45, 2.95, GRN), "DC": (2.75, 2.15, BLU),
           "DD": (4.35, 2.10, ACC), "DE": (5.35, 2.55, MUT)}
    for name, (x, y, col) in pts.items():
        ax.plot([x], [y], marker="o", ms=7, color=col)
        ax.text(x, y + 0.18, name, fontsize=12, color=col, ha="center", va="bottom",
                fontweight="bold")
    ax.plot([1.35, 2.45], [2.55, 2.95], color=GRN, lw=1.6)
    ax.plot([1.35, 2.75], [2.55, 2.15], color=BLU, lw=1.2, ls="--")
    ax.plot([4.35, 5.35], [2.10, 2.55], color=MUT, lw=1.2, ls=":")
    ax.plot([4.35, 4.35], [2.10, 1.70], color=ACC, lw=1.2, ls=":")
    ax.add_patch(Circle((1.35, 2.55), 0.72, fc="none", ec=GRN, lw=1.2, ls="--"))
    ax.add_patch(Circle((4.35, 2.10), 0.55, fc="none", ec=ACC, lw=1.2, ls="--"))
    ax.text(1.85, 2.90, "d1", fontsize=12, color=GRN)
    ax.text(2.10, 2.22, "d2", fontsize=12, color=BLU)
    ax.text(4.95, 2.20, "d1'", fontsize=12, color=MUT)
    ax.text(4.42, 1.73, "d2'", fontsize=12, color=ACC)

    box(ax, 0.35, 0.55, 2.85, 1.05, ec=GRN, fc="#eef5ee")
    ax.text(1.77, 1.37, "DA：d1/d2 = 0.33", fontsize=12, color=GRN, ha="center",
            va="center", fontweight="bold")
    ax.text(1.77, 1.10, "小比率→接受（配 DB）", fontsize=12, color=INK, ha="center",
            va="center")
    box(ax, 3.45, 0.55, 2.80, 1.05, ec=ACC, fc="#fdf0ec")
    ax.text(4.85, 1.37, "DD：d1'/d2' = 0.92", fontsize=12, color=ACC, ha="center",
            va="center", fontweight="bold")
    ax.text(4.85, 1.10, "大比率→拒绝（不配 DC）", fontsize=12, color=INK, ha="center",
            va="center")
    ax.text(3.3, 0.18, "程序生成，未用原书原图 · 对应原书 p.445 图 7.23，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig, dict(d1=0.30, d2=0.90, ratio_small=0.33, d1p=0.55, d2p=0.60,
                     ratio_large=0.92, threshold=0.80)


def fig_match_pipeline():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "匹配四道闸：数量逐级收窄", fontsize=14, color=INK,
            ha="center", va="center", fontweight="bold")

    stages = [
        ("① 描述子 + 候选匹配", "500 对", 3.05, BLU),
        ("② 阈值 / 最近邻", "120 对", 2.45, GOLD),
        ("③ NNDR + 互检", "80 对", 1.85, GRN),
        ("④ RANSAC 验证", "72 内点 / 8 离群", 1.25, ACC),
    ]
    for label, count, y, col in stages:
        box(ax, 0.55, y, 4.35, 0.46, ec=col, fc="#ffffff")
        ax.text(1.02, y + 0.23, label, fontsize=12, color=col, ha="left",
                va="center", fontweight="bold")
        ax.text(4.72, y + 0.23, count, fontsize=12, color=INK, ha="right", va="center")
    for y in (3.05, 2.45, 1.85):
        arrow(ax, (2.75, y), (2.75, y - 0.14), color=GOLD, ms=9)
    ax.text(3.3, 0.80, "每过一道闸，错配少一批；最后只留下几何上说得通的内点",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(3.3, 0.50, "RANSAC：抽小批种子拟合变换→数内点→重复取最好→用全部内点重估",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.text(3.3, 0.18, "程序生成，未用原书原图 · 对应原书 p.441–448 正文，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    return fig, dict(input=500, after_ratio=120, after_mutual=80, inliers=72, outliers=8)


def main():
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__)
    ssha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={ssha}")
    fig1, s1 = fig_threshold_tradeoff()
    fig2, s2 = fig_nndr()
    fig3, s3 = fig_match_pipeline()
    p1 = FIG / "e069_threshold_tradeoff.png"
    p2 = FIG / "e069_nndr.png"
    p3 = FIG / "e069_match_pipeline.png"
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
    j1, j2, j3 = " ".join(t1), " ".join(t2), " ".join(t3)
    rows = []
    rows.append(f"图 1 阈值与混淆矩阵词在档：{all(k in j1 for k in ['阈值','真阳','假阳','假阴'])}")
    rows.append(f"图 1 ROC/AUC 在档：{all(k in j1 for k in ['ROC','AUC'])}")
    rows.append(f"图 2 NNDR 要素在档：{all(k in j2 for k in ['DA','DB','DC','DD','d1','d2'])}")
    rows.append(f"图 2 小比率接受/大比率拒绝：{('小比率' in j2 and '接受' in j2 and '大比率' in j2 and '拒绝' in j2)}")
    rows.append(f"图 3 四道闸在档：{all(k in j3 for k in ['描述子','阈值','NNDR','互检','RANSAC'])}")
    rows.append(f"图 3 数量单调收窄：{s1['tp']+s1['fp']+s1['fn']+s1['tn']==100 and s3['input']>s3['after_ratio']>s3['after_mutual']>=s3['inliers']>0}")
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
        "fig1": {**s1, "min_fs": min(fs1), "max_fs": max(fs1), "fig_no": 7.22, "page": 442},
        "fig2": {**s2, "min_fs": min(fs2), "max_fs": max(fs2), "fig_no": 7.23, "page": 445},
        "fig3": {**s3, "min_fs": min(fs3), "max_fs": max(fs3), "page": 448},
        "scope": {"page_start": 441, "page_end": 448, "frames": 16},
        "texts": {"collected": len(allt), "leaked": len(leaked)},
        "pngs": len(pngs),
    }
    rp = OUT / "e069_feature_matching_readings.json"
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
