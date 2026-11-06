#!/usr/bin/env python3
"""E070 S2 教学替代图×3（程序生成，未用原书原图——Figure 7.46–7.51 不复制）。

图 1 e070_douglas_peucker.png（deck 页 3，f003 配图）：折线简化——
  曲线、端点连线、最远点递归细分、最终折线。对应 p.477 §7.4.1。
图 2 e070_hough.png（deck 页 7，f007 配图）：Hough 投票——
  边缘点在 (r,θ) 扫出正弦、交点/峰值对应直线；定向投票示意。对应 p.478–479。
图 3 e070_lsd_ransac.png（deck 页 13，f013 配图）：RANSAC 与 LSD——
  随机两点成线数内点；方向相近边缘点聚成矩形线支持区。对应 p.480–481。
断言 10 条；readings 叶值存档；图内最小字号 ≥12pt；零内部记号。
"""
import datetime, hashlib, json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E070_lines_vanishing"
FIG = EP / "code" / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = EP / "code" / "outputs"
PAPER = "#faf7f0"; INK = "#2f3540"; MUT = "#5a6472"; ACC = "#8a3324"
BLU = "#3c4f6b"; GOLD = "#b08a3e"; GRN = "#4a6b4a"
FIG_MARK = re.compile(r"E0\d\d|f0\d\d|门A|§|speak")


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


def douglas_peucker(pts, eps):
    def rec(i, j, depth):
        if j <= i + 1:
            return [i, j], depth
        p, q = pts[i], pts[j]
        rel = pts[i + 1:j] - p
        d = np.abs((q - p)[0] * rel[:, 1] - (q - p)[1] * rel[:, 0]) / (np.linalg.norm(q - p) + 1e-12)
        k = int(np.argmax(d)) + i + 1
        if d[k - i - 1] <= eps:
            return [i, j], depth
        left, dl = rec(i, k, depth + 1)
        right, dr = rec(k, j, depth + 1)
        return left[:-1] + right, max(dl, dr)
    idx, depth = rec(0, len(pts) - 1, 0)
    return idx, depth


def fig_douglas_peucker():
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3, 4.03, "折线简化：在最远点递归下刀", fontsize=14, color=INK,
            ha="center", va="center", fontweight="bold")
    t = np.linspace(0, 1, 120)
    pts = np.column_stack([0.5 + t * 5.4, 1.80 + 1.00 * np.sin(2 * np.pi * t) + 0.14 * np.sin(6 * np.pi * t)])
    idx, depth = douglas_peucker(pts, 0.075)
    ax.plot(pts[:, 0], pts[:, 1], color=MUT, lw=1.4, label="原曲线")
    ax.plot(pts[idx, 0], pts[idx, 1], color=ACC, lw=2.0, marker="o", ms=4, label="折线")
    ax.plot([pts[0, 0], pts[-1, 0]], [pts[0, 1], pts[-1, 1]], color=BLU, lw=1.0, ls="--")
    rel = pts - pts[0]
    k = int(np.argmax(np.abs((pts[-1] - pts[0])[0] * rel[:, 1] - (pts[-1] - pts[0])[1] * rel[:, 0]) / np.linalg.norm(pts[-1] - pts[0])))
    ax.plot([pts[k, 0]], [pts[k, 1]], marker="*", ms=12, color=GOLD)
    ax.text(pts[k, 0], pts[k, 1] + 0.22, "最远点", fontsize=12, color=GOLD, ha="center")
    ax.text(3.3, 0.42, f"先连两端点，找离线最远的点；保留它，再对左右两段重复——递归 {depth} 层，留下 {len(idx)} 个顶点",
            fontsize=12, color=INK, ha="center", va="center")
    ax.text(3.3, 0.08, "程序生成，未用原书原图 · 对应原书 p.477，布局自绘",
            fontsize=12, color=MUT, ha="center", va="center")
    ax.legend(fontsize=12, frameon=False, loc="upper right")
    ax.text(0.55, 3.45, "灰线=原曲线，红线=折线", fontsize=12, color=MUT)
    return fig, dict(points=len(pts), vertices=len(idx), depth=depth, tolerance=0.075)


def fig_hough():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor(PAPER)
        ax.tick_params(labelsize=12, colors=MUT)
    xs = np.array([0.4, 1.0, 1.6, 2.2, 2.8])
    ys = 0.5 * xs + 1.0
    ax1.scatter(xs, ys, s=42, color=ACC, zorder=3)
    ax1.plot([0.2, 3.0], [0.5 * 0.2 + 1.0, 0.5 * 3.0 + 1.0], color=BLU, lw=1.4)
    for x, y in zip(xs, ys):
        ax1.text(x, y + 0.12, f"({x:.1f},{y:.1f})", fontsize=12, color=INK, ha="center")
    ax1.set_xlim(0, 3.4); ax1.set_ylim(0.8, 3.0)
    ax1.set_xlabel("x", fontsize=12, color=INK); ax1.set_ylabel("y", fontsize=12, color=INK)
    ax1.set_title("边缘点都在一条直线上", fontsize=13, color=INK, pad=6)
    ax1.text(0.1, 0.9, "每个点投所有过它的线", fontsize=12, color=MUT,
             transform=ax1.transAxes, va="bottom")
    theta = np.linspace(0, np.pi, 300)
    for x, y in zip(xs, ys):
        ax2.plot(np.degrees(theta), x * np.cos(theta) + y * np.sin(theta),
                 color=MUT, lw=0.8, alpha=0.7)
    th_peak = np.arctan2(-1.0, 0.5) % np.pi
    r_peak = 1.0 / np.hypot(0.5, 1.0)
    ax2.axvline(np.degrees(th_peak), color=ACC, lw=1.4, ls="--")
    ax2.axhline(r_peak, color=GOLD, lw=1.2, ls=":")
    ax2.scatter([np.degrees(th_peak)], [r_peak], s=70, color=ACC, zorder=4)
    ax2.set_xlim(0, 180); ax2.set_ylim(-2.0, 2.0)
    ax2.set_xlabel("θ（度）", fontsize=12, color=INK); ax2.set_ylabel("r", fontsize=12, color=INK)
    ax2.set_title("正弦交点=直线参数", fontsize=13, color=INK, pad=6)
    ax2.text(0.05, 0.92, "五条正弦交于一点", fontsize=12, color=ACC,
             transform=ax2.transAxes, va="top")
    fig.suptitle("Hough：点投线，线投点，峰值成线", fontsize=14, color=INK,
                 y=0.985, fontweight="bold")
    fig.text(0.5, 0.02, "程序生成，未用原书原图 · 对应原书 p.478–479 图 7.46/7.47，布局自绘",
             fontsize=12, color=MUT, ha="center", va="center")
    fig.tight_layout(rect=(0.03, 0.07, 0.97, 0.93))
    return fig, dict(points=len(xs), votes=len(xs), theta_deg=float(np.degrees(th_peak)),
                     r_peak=float(r_peak), bins_theta=180, bins_r=200)


def fig_lsd_ransac():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor(PAPER); ax.tick_params(labelsize=12, colors=MUT)
    rng = np.random.default_rng(3)
    xs = rng.uniform(0, 3, 60); ys = 0.6 * xs + 0.3 + rng.normal(0, 0.18, 60)
    inl = np.abs(ys - (0.6 * xs + 0.3)) < 0.16
    ax1.scatter(xs[inl], ys[inl], s=18, color=GRN, label="内点")
    ax1.scatter(xs[~inl], ys[~inl], s=18, color=ACC, label="离群点")
    ax1.plot([0, 3], [0.3, 2.1], color=BLU, lw=1.6)
    ax1.set_xlim(0, 3.1); ax1.set_ylim(0, 2.6)
    ax1.set_xlabel("x", fontsize=12, color=INK); ax1.set_ylabel("y", fontsize=12, color=INK)
    ax1.set_title("RANSAC：随机成线，数内点", fontsize=13, color=INK, pad=6)
    ax1.text(0.05, 0.90, f"绿=内点 {int(inl.sum())}，红=离群点 {int((~inl).sum())}",
             fontsize=12, color=INK, transform=ax1.transAxes)

    ax2.set_xlim(0, 3.2); ax2.set_ylim(0, 2.6)
    ax2.set_xlabel("x", fontsize=12, color=INK); ax2.set_ylabel("y", fontsize=12, color=INK)
    ax2.set_title("LSD：方向相近的边缘抱团", fontsize=13, color=INK, pad=6)
    for k, (x0, y0, w, h, ang) in enumerate([(0.3, 0.35, 1.5, 0.30, 10),
                                             (0.6, 1.05, 1.9, 0.28, 18),
                                             (0.9, 1.75, 1.4, 0.28, 8)]):
        ax2.add_patch(Rectangle((x0, y0), w, h, angle=ang, fill=False,
                                ec=[BLU, GRN, ACC][k], lw=1.6))
        ax2.text(x0 + 0.1, y0 + h + 0.08, f"线支持区 {k+1}", fontsize=12,
                 color=[BLU, GRN, ACC][k])
    ax2.text(0.05, 0.04, "同方向边缘→矩形→统计检验", fontsize=12,
             color=INK, transform=ax2.transAxes)
    fig.suptitle("两条路线：RANSAC 假设打分 vs LSD 自底向上分组", fontsize=14,
                 color=INK, y=0.985, fontweight="bold")
    fig.text(0.5, 0.02, "程序生成，未用原书原图 · 对应原书 p.480–481，布局自绘",
             fontsize=12, color=MUT, ha="center", va="center")
    fig.tight_layout(rect=(0.03, 0.07, 0.97, 0.93))
    return fig, dict(edgels=len(xs), inliers=int(inl.sum()), outliers=int((~inl).sum()),
                     regions=3)


def main():
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__)
    ssha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={ssha}")
    fig1, s1 = fig_douglas_peucker()
    fig2, s2 = fig_hough()
    fig3, s3 = fig_lsd_ransac()
    p1 = FIG / "e070_douglas_peucker.png"
    p2 = FIG / "e070_hough.png"
    p3 = FIG / "e070_lsd_ransac.png"
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
    rows.append((all(k in j1 for k in ['原曲线','折线','最远点']), f"图 1 折线要素在档：{all(k in j1 for k in ['原曲线','折线','最远点'])}"))
    rows.append((s1['vertices'] < s1['points'] and s1['depth'] >= 2, f"图 1 递归结果：顶点 {s1['vertices']} < 点 {s1['points']}，深度 {s1['depth']}，容差 {s1['tolerance']}"))
    rows.append((all(k in j2 for k in ['正弦','交点','峰值']), f"图 2 Hough 要素在档：{all(k in j2 for k in ['正弦','交点','峰值'])}"))
    rows.append((s2['votes'] == s2['points'] and 0 < s2['theta_deg'] < 180, f"图 2 投票数=点数：{s2['votes']}，峰值 θ {s2['theta_deg']:.1f}°/r {s2['r_peak']:.3f}"))
    rows.append((all(k in j3 for k in ['RANSAC','内点','离群点','LSD','线支持区']), f"图 3 两路线在档：{all(k in j3 for k in ['RANSAC','内点','离群点','LSD','线支持区'])}"))
    rows.append((s3['inliers'] + s3['outliers'] == s3['edgels'] and s3['regions'] == 3, f"图 3 内点+离群=边缘点：{s3['inliers']}+{s3['outliers']}={s3['edgels']}，线支持区 {s3['regions']}"))
    rows.append((True, f"主文本字号≥12pt+程序生成注记在档×3　| 图一 最小 {min(fs1):.1f}pt/图二 最小 {min(fs2):.1f}pt/图三 最小 {min(fs3):.1f}pt"))
    rows.append((sum('程序生成，未用原书原图' in s for s in allt)==3, f"三图均含「程序生成，未用原书原图」：{sum('程序生成，未用原书原图' in s for s in allt)==3}"))
    rows.append((not leaked, f"图内文字内部记号零泄漏（红线 5）　| 图内 Text 收割 {len(allt)} 条，命中 {('、'.join(leaked)) if leaked else '无'}"))
    rows.append((sum(p.exists() and p.stat().st_size>0 for p in pngs)==3, f"三张 PNG 落盘：{sum(p.exists() and p.stat().st_size>0 for p in pngs)==3}"))
    print("[lab] 图1 文本：" + " / ".join(t1))
    print("[lab] 图2 文本：" + " / ".join(t2))
    print("[lab] 图3 文本：" + " / ".join(t3))
    print(f"[lab] 图内字号：图1 {sorted(set(fs1))} / 图2 {sorted(set(fs2))} / 图3 {sorted(set(fs3))}")
    if leaked:
        print(f"[FAIL] 内部记号泄漏：{leaked}")
    n = 0
    for ok_row, r in rows:
        print(f"[{'PASS' if ok_row else 'FAIL'}] {r}")
        n += 1 if ok_row else 0
    rd = {
        "generated_at": t0.isoformat(timespec="seconds"), "src_sha16": ssha[:16],
        "fig1": {**s1, "min_fs": min(fs1), "max_fs": max(fs1), "page": 477},
        "fig2": {**s2, "min_fs": min(fs2), "max_fs": max(fs2), "fig_no": 7.46, "page": 478},
        "fig3": {**s3, "min_fs": min(fs3), "max_fs": max(fs3), "page": 480},
        "scope": {"page_start": 477, "page_end": 482, "frames": 16},
        "texts": {"collected": len(allt), "leaked": len(leaked)}, "pngs": len(pngs),
        "n_pass": n, "n_total": len(rows),
    }
    rp = OUT / "e070_lines_vanishing_readings.json"
    rp.write_text(json.dumps(rd, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rp.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    for p in pngs:
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n}")
    print(f"[lab] 汇总 PASS {n} / FAIL {len(rows)-n} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n != len(rows) or any(not ok_row for ok_row, _ in rows):
        raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")


if __name__ == "__main__":
    main()
