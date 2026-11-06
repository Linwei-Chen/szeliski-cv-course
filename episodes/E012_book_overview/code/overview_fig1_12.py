#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E012 实验：按原书 Figure 1.12（印刷 p.23）重绘「本书章节依赖图」

讲者设计实验（非原书内容）：
  原书 Figure 1.12 是一张章节依赖图：13 个章节盒子，横轴按 2D(what?) ↔ 3D(where?)
  摆放，箭头表示章节间的（rough）依赖。原书没有提供盒子的坐标表和边表。本实验：
  1) 13 个盒子标签人工誊录自 p.23 原图（图内折行在誊录时合并为空格）；
  2) 盒子横轴位置与依赖箭头为讲者按图读出的近似估计（讲者读图估计，已在图注标注）；
  3) 画成 DAG 布局图：横轴 2D(what?) 在左、3D(where?) 在右，纵轴按读图层次；
  4) 断言检查：盒子数量与编号、标签誊录、无自环、可拓扑排序（无环）、
     唯一入度为零的起点、终点无出边、基座章有出边、横位三段有序（2D 侧<中间带<3D 侧）等；
     数据断言全部通过后才把图同步到 slides/figures/（失败不覆盖幻灯片用图）。

运行：python3 overview_fig1_12.py
输出：outputs/overview_fig1_12.png + 断言汇总
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 中文字体（macOS 优先 Hiragino Sans GB——.ttc 含 W3/W6 字重，normal 回退 300、bold 回退 600，stderr 仅为提示；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 数据
# 13 个章节盒子，标签逐字誊录自原书 Figure 1.12（p.23）；
# 图内折行（如 "4. Model fitting / and optimization"）誊录时合并为单个空格。
# (章节号, 标签, 横位 x[0=2D(what?) 侧, 10=3D(where?) 侧], 层次 y[0=最上])
# x / y 全部为讲者读图估计。
BOXES = [
    (2,  "2. Image formation",                    4.0, 0.0),
    (3,  "3. Image processing",                   1.2, 1.1),
    (4,  "4. Model fitting and optimization",     6.8, 1.1),
    (5,  "5. Deep learning",                      1.8, 2.2),
    (7,  "7. Feature detection and matching",     4.6, 2.2),
    (6,  "6. Recognition",                        1.8, 3.3),
    (8,  "8. Image alignment and stitching",      5.4, 3.3),
    (9,  "9. Motion estimation",                  8.2, 4.4),
    (10, "10. Computational photography",         3.6, 5.5),
    (11, "11. Structure from motion and SLAM",    8.4, 5.5),
    (12, "12. Depth estimation",                  8.4, 6.6),
    (13, "13. 3D reconstruction",                 8.4, 7.7),
    (14, "14. Image-based rendering",             4.6, 8.8),
]

# 依赖箭头（a → b 表示「先读 a 有助于 b」，即 b 依赖 a）——讲者读图估计；
# 原书图注自称 (rough) dependencies，本表只誊出图中可辨认的箭头。
EDGES = [
    (2, 3), (2, 4),
    (3, 4), (3, 5), (3, 7),
    (5, 6), (5, 7),
    (4, 7), (4, 8),
    (7, 6), (7, 8),
    (8, 9), (8, 10),
    (9, 10), (9, 11),
    (11, 12), (12, 13), (12, 14), (13, 14),
    (10, 14),
]

# 基座章节（原书图注：Foundational techniques such as optimization and deep
# learning are widely used in subsequent chapters, p.23）
FOUNDATIONS = {4, 5}
# 终点章节（图中无出边者，读图估计）
SINKS_EXPECTED = {6, 14}
# 横轴三段分组（讲者读图，与口播/幻灯片一致）：
# 偏图像（2D what?）侧：Ch3、5、6；中间带：Ch2、4、7、8、10、14；偏几何（3D where?）侧：Ch9、11、12、13
WHAT_SIDE = [3, 5, 6]
MID_SIDE = [2, 4, 7, 8, 10, 14]
WHERE_SIDE = [9, 11, 12, 13]

# ---------------------------------------------------------------- 断言
_results = []


def verify(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"[verify] {name} {'（' + detail + '）' if detail else ''} [{mark}]")


def topological_sort(nodes, edges):
    """Kahn 拓扑排序；有环返回 None。"""
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    queue = sorted(n for n in nodes if indeg[n] == 0)
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    return order if len(order) == len(nodes) else None


def reachable_from(src, edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, set()).add(b)
    seen, stack = set(), [src]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(adj.get(n, ()))
    return seen


def run_checks():
    ids = [b[0] for b in BOXES]
    labels = [b[1] for b in BOXES]
    xs = {b[0]: b[2] for b in BOXES}
    ys = {b[0]: b[3] for b in BOXES}

    verify("盒子数量与原书 Figure 1.12 一致 = 13", len(BOXES) == 13, f"n={len(BOXES)}")
    verify("章节号 2–14 各出现一次", sorted(ids) == list(range(2, 15)), f"min={min(ids)} max={max(ids)}")
    verify("无重复标签", len(set(labels)) == len(labels), f"去重后 {len(set(labels))}")
    verify("标签誊录格式合法（『N. 标题』且编号一致）",
           all(l.split('. ', 1)[0].isdigit() and int(l.split('. ', 1)[0]) == i for i, l in zip(ids, labels)))
    verify("无自环", all(a != b for a, b in EDGES), f"边数 {len(EDGES)}")
    verify("边端点均为已有盒子", all(a in ids and b in ids for a, b in EDGES))
    order = topological_sort(ids, EDGES)
    verify("依赖图无环（可拓扑排序）", order is not None, f"拓扑序长度 {len(order) if order else 0}")
    srcs = [n for n in ids if all(b != n for _, b in EDGES)]
    verify("唯一入度为零的起点 = 第 2 章（Image formation）", srcs == [2], f"入度为零 {srcs}")
    out_edges = {n: sum(1 for a, _ in EDGES if a == n) for n in ids}
    sinks = {n for n in ids if out_edges[n] == 0}
    verify("终点无出边 = {6 识别, 14 基于图像的渲染}", sinks == SINKS_EXPECTED, f"终点 {sorted(sinks)}")
    verify("基座章各有出边、被后续章直接依赖（读图）",
           all(out_edges[n] >= 1 for n in FOUNDATIONS),
           f"Ch4 出度 {out_edges[4]}（→7,8） · Ch5 出度 {out_edges[5]}（→6,7）")
    reach = reachable_from(2, EDGES)
    verify("全部盒子自第 2 章可达", set(ids) <= reach | {2}, f"可达 {len(reach)}/13（含起点）")
    mid_xs = [xs[m] for m in MID_SIDE]
    verify("横位三段有序：2D 侧 < 中间带 < 3D 侧（读图）",
           max(xs[k] for k in WHAT_SIDE) < min(mid_xs) and max(mid_xs) < min(xs[w] for w in WHERE_SIDE),
           f"2D 侧 x≤{max(xs[k] for k in WHAT_SIDE)} < 中间带 x[{min(mid_xs)}, {max(mid_xs)}] < 3D 侧 x≥{min(xs[w] for w in WHERE_SIDE)}")
    verify("全部横位落在 [0, 10]", all(0 <= x <= 10 for x in xs.values()),
           f"范围 [{min(xs.values())}, {max(xs.values())}]")
    verify("层次自上而下：2 → … → 11 → 12 → 13 → 14",
           ys[2] < ys[11] < ys[12] < ys[13] < ys[14] and ys[2] == min(ys.values()),
           f"y2={ys[2]} y14={ys[14]}")


# ---------------------------------------------------------------- 绘图
BOX_FILL = {True: "#1f4e79", False: "#f5ead8"}   # 基座深蓝，其余纸色


def draw(outpath):
    import textwrap
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    fig.patch.set_facecolor("#fbf6ec")
    ax.set_facecolor("#fbf6ec")

    pos = {b[0]: (b[2], -b[3]) for b in BOXES}   # y 取负使最早在上
    labels = {b[0]: b[1] for b in BOXES}
    W, H = 2.45, 0.74

    for a, b in EDGES:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        # 从盒子边缘出发，避免箭头压在盒子底下
        dx, dy = x2 - x1, y2 - y1
        L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        sx, sy = x1 + dx / L * (W / 2 + 0.08), y1 + dy / L * (H / 2 + 0.05)
        ex, ey = x2 - dx / L * (W / 2 + 0.08), y2 - dy / L * (H / 2 + 0.05)
        arr = FancyArrowPatch((sx, sy), (ex, ey),
                              connectionstyle="arc3,rad=0.10",
                              arrowstyle="-|>", mutation_scale=13,
                              lw=1.3, color="#7a5c2e", zorder=1)
        ax.add_patch(arr)

    for n in pos:
        x, y = pos[n]
        face = BOX_FILL[n in FOUNDATIONS]
        box = FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                             boxstyle="round,pad=0.05,rounding_size=0.10",
                             facecolor=face, edgecolor="#8a6d3b", lw=1.2, zorder=2)
        ax.add_patch(box)
        wrapped = "\n".join(textwrap.wrap(labels[n], 19))
        ax.text(x, y, wrapped, ha="center", va="center",
                fontsize=10.0, linespacing=1.15,
                color=("#fbf6ec" if n in FOUNDATIONS else "#1f4e79"),
                fontweight=("bold" if n in FOUNDATIONS else "normal"), zorder=3)

    # 横轴：2D(what?) ↔ 3D(where?)（照原书 p.23 顶部箭头，画在全部盒子之上）
    yax = -0.78
    ax.annotate("", xy=(11.0, -yax), xytext=(-1.0, -yax),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.4, linestyle="--"))
    ax.text(-1.0, -yax + 0.32, "2D（what?）", fontsize=12.5, color="#444", ha="left")
    ax.text(11.0, -yax + 0.32, "3D（where?）", fontsize=12.5, color="#444", ha="right")
    ax.text(11.0, -yax - 0.42, "横轴为讲者读图估计（原书仅给 rough 摆位）", fontsize=9.0,
            color="#8a6d3b", ha="right")

    # 图例：基座 vs 一般章节
    from matplotlib.patches import Patch
    legend = ax.legend(handles=[Patch(facecolor="#1f4e79", edgecolor="#8a6d3b", label="基座（深色为讲者着色；原书图注：被后续章广泛使用，第 23 页）"),
                                Patch(facecolor="#f5ead8", edgecolor="#8a6d3b", label="章节盒子（标签人工誊录自第 23 页）")],
                       loc="lower left", fontsize=9.5, framealpha=0.95)
    legend.set_zorder(4)

    ax.set_xlim(-1.3, 11.3)
    ax.set_ylim(-9.6, 2.2)
    ax.axis("off")
    ax.set_title("E012 实验 · 讲者据原书 Figure 1.12（第 23 页）重绘：13 章 · 20 条依赖（读图估计）",
                 fontsize=13, color="#1f4e79", pad=12)
    fig.text(0.995, 0.006,
             "讲者设计实验：盒子标签人工誊录自原书图（折行合并为空格）；横位、层次与依赖箭头均为讲者读图估计（原书未提供数据表）",
             ha="right", fontsize=8.5, color="#8a6d3b")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(outpath, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[demo] 图已存 {outpath}")
    # 注意：此处不同步 slides/figures/——同步在 main() 中、数据断言全部通过后才执行


def main():
    outdir = pathlib.Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(exist_ok=True)
    outpath = outdir / "overview_fig1_12.png"
    draw(outpath)
    run_checks()
    data_ok = all(ok for _, ok, _ in _results)
    twin_dir = pathlib.Path(__file__).resolve().parents[1] / "slides" / "figures"
    if data_ok:   # 数据断言全过后才同步，避免失败图覆盖幻灯片用图
        if twin_dir.is_dir():
            (twin_dir / outpath.name).write_bytes(outpath.read_bytes())
            print(f"[demo] 数据断言全过，已同步 {twin_dir / outpath.name}")
    twin_path = twin_dir / outpath.name
    verify("数据断言全过后才同步 slides/figures/（字节一致）",
           data_ok and twin_path.exists() and twin_path.read_bytes() == outpath.read_bytes(),
           "已同步（与 outputs 图逐字节一致）" if data_ok else "存在失败断言，未同步")
    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    print(f"汇总：{passed}/{total} PASS —— {'全部通过' if passed == total else '存在失败'}")
    sys_exit = 0 if passed == total else 1
    raise SystemExit(sys_exit)


if __name__ == "__main__":
    main()
