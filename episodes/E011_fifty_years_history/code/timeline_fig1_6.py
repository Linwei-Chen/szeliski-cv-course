#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E011 实验：按原书 Figure 1.6（印刷 p.11）重绘「计算机视觉五十年时间线」

讲者设计实验（非原书内容）：
  原书 Figure 1.6 是一张 rough timeline，横轴 1970–2020，图中标注了 33 个主题
  标签，但原书没有提供每个条带的起止数据表。本实验：
  1) 从原书 PDF 提取 Figure 1.6 的 33 个标签（人工誊录，见 TOPICS）；
  2) 条带起止年代为讲者按图读出的近似估计（讲者读图估计，已在图注标注）；
  3) 画成水平条带图，按五个十年配色；
  4) 断言检查：标签数量与原书图一致、时间范围合法、五个十年分段齐全等。

运行：python3 timeline_fig1_6.py
输出：outputs/timeline_fig1_6.png + 断言汇总
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体（macOS 优先 Hiragino Sans GB——.ttc 含 W3/W6 字重，normal 回退 300、bold 回退 600，stderr 仅为提示；找不到就退化为系统默认）
for _f in ["Hiragino Sans GB", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]:
    if any(_f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------- 数据
# (主题, 起始年, 结束年) —— 起止为讲者按 Figure 1.6 读图的近似估计
# 标签本身逐字取自原书 Figure 1.6（p.11），共 33 个（含图内连接词 and）。
# 起止年取值规则：仅以下四项锚定正文年份——SfM 1979（Ullman 1979, p.13）、
# Deep learning 2012（AlexNet, p.21）、MRF 1984（Geman and Geman, p.15）、
# Face 1991（Turk and Pentland, p.17）；其余 29 项全部为讲者读图估计。
TOPICS = [
    ("Digital image processing",            1965, 1975),
    ("Blocks world, line labeling",         1965, 1978),
    ("Generalized cylinders",               1970, 1982),
    ("Pattern recognition",                 1970, 1990),
    ("Stereo correspondence",               1975, 1990),
    ("Intrinsic images",                    1975, 1985),
    ("Optical flow",                        1980, 1995),
    ("Shape from shading, texture, and focus", 1980, 1995),
    ("Physically-based modeling",           1985, 1995),
    ("Regularization",                      1985, 1998),
    ("Markov random fields",                1984, 2000),
    ("Kalman filters",                      1988, 2000),
    ("3D range data processing",            1985, 2000),
    ("Structure from motion",               1979, 2000),
    ("Image pyramids",                      1983, 1995),
    ("Projective invariants",               1990, 1998),
    ("Factorization",                       1992, 2000),
    ("Physics-based vision",                1990, 2000),
    ("Image-based modeling and rendering",  1995, 2008),
    ("Particle filtering",                  1998, 2010),
    ("Graph cuts",                          1999, 2012),
    ("Energy-based segmentation",           2000, 2012),
    ("Face recognition and detection",      1991, 2015),
    ("Texture synthesis and inpainting",    1999, 2010),
    ("Feature-based recognition",           2003, 2015),
    ("Category recognition",                2005, 2018),
    ("Machine learning",                    2005, 2020),
    ("Computational photography",           2005, 2018),
    ("Modeling and tracking humans",        2008, 2020),
    ("Semantic segmentation",               2013, 2020),
    ("SLAM and VIO",                        2013, 2020),
    ("Deep learning",                       2012, 2020),
    ("Vision and language",                 2015, 2020),
]

DECADE_COLORS = {          # Okabe-Ito 色盲友好
    "1970s": "#0072B2",
    "1980s": "#56B4E9",
    "1990s": "#009E73",
    "2000s": "#E69F00",
    "2010s": "#D55E00",
}

# ---------------------------------------------------------------- 断言助手
_results = []

def verify(name, cond, detail=""):
    ok = bool(cond)
    _results.append(ok)
    tag = "PASS" if ok else "FAIL"
    line = f"[verify] {name}"
    if detail:
        line += f"（{detail}）"
    print(f"{line} [{tag}]")
    return ok


def decade_of(start):
    """主题按起始年归入哪个十年段（配色分组用）；1970 前起步的归入 1970s 组"""
    return f"{(max(start, 1970) // 10) * 10}s"


# ---------------------------------------------------------------- 校验
def run_checks():
    n = len(TOPICS)
    verify("标签数量与原书 Figure 1.6 一致 = 33", n == 33, f"n={n}")

    labels = [t[0] for t in TOPICS]
    verify("无重复标签", len(set(labels)) == n, f"去重后 {len(set(labels))}")

    bad_order = [t[0] for t in TOPICS if not t[1] < t[2]]
    verify("全部条带 start < end", not bad_order, f"异常 {bad_order}")

    out_of_range = [t[0] for t in TOPICS if t[1] < 1960 or t[2] > 2020]
    verify("时间范围均落在 [1960, 2020]", not out_of_range, f"越界 {out_of_range}")

    too_thin = [t[0] for t in TOPICS if t[2] - t[1] < 3]
    verify("条带宽度 ≥ 3 年（可读性约定）", not too_thin, f"过窄 {too_thin}")

    starts = [t[1] for t in TOPICS]
    ends = [t[2] for t in TOPICS]
    verify("时间线覆盖五十年（起 ≤1970 且 讫 ≥2018）",
           min(starts) <= 1970 and max(ends) >= 2018,
           f"最早 {min(starts)} · 最晚收尾 {max(ends)}")

    for dec, lo, hi in [("1970s", 1970, 1979), ("1980s", 1980, 1989),
                        ("1990s", 1990, 1999), ("2000s", 2000, 2009),
                        ("2010s", 2010, 2019)]:
        cnt = sum(1 for s, _ in [(t[1], t[2]) for t in TOPICS] if lo <= s <= hi)
        n_early = sum(1 for t in TOPICS if t[1] < 1970)
        note = f"（不含 1970 前起步的 {n_early} 个）" if dec == "1970s" else ""
        verify(f"{dec} 起步的主题 ≥ 3 个{note}", cnt >= 3, f"{cnt} 个")

    dl = dict((t[0], (t[1], t[2])) for t in TOPICS)
    verify("Deep learning 起始于 2010s（AlexNet 2012，p.21）",
           dl["Deep learning"][0] >= 2010, f"{dl['Deep learning'][0]}")
    verify("Vision and language 起始 ≥2013（读图估计；§1.2 正文未展开）",
           dl["Vision and language"][0] >= 2013, f"{dl['Vision and language'][0]}")
    for k in ("Semantic segmentation", "SLAM and VIO", "Modeling and tracking humans"):
        verify(f"{k} 起始 ≥2005（读图估计；正文无逐项年代）",
               dl[k][0] >= 2005, f"{dl[k][0]}")

    early = [t[0] for t in TOPICS if t[1] < 1975]
    verify("「方块世界/线段标注」属于最早一批主题（p.12）",
           "Blocks world, line labeling" in early, f"最早批 {len(early)} 个")


# ---------------------------------------------------------------- 绘图
def draw(outpath):
    # 按起始年排序；invert_yaxis 后顶部为最早，与口播「越靠上起步越早」一致
    rows = sorted(TOPICS, key=lambda t: (t[1], t[2]))
    fig, ax = plt.subplots(figsize=(14.2, 9.2), dpi=150)

    for i, (label, s, e) in enumerate(rows):
        c = DECADE_COLORS[decade_of(s)]
        ax.barh(i, e - s, left=s, height=0.62, color=c, alpha=0.88,
                edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([t[0] for t in rows], fontsize=10.5)
    ax.set_xlim(1960, 2022)
    ax.set_xticks(range(1960, 2021, 10))
    ax.set_xlabel("年份（横轴 1960–2022；原书时间轴 1970–2020）", fontsize=12)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d9d4c8", linewidth=0.7, alpha=0.6, zorder=0)
    ax.tick_params(axis="x", labelsize=11)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # 十年分界竖线
    for yr in (1970, 1980, 1990, 2000, 2010, 2020):
        ax.axvline(yr, color="#8a8377", linewidth=1.0, linestyle=(0, (4, 3)),
                   alpha=0.7, zorder=2)

    # 十年段图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88)
               for c in DECADE_COLORS.values()]
    ax.legend(handles, DECADE_COLORS.keys(), loc="upper right", fontsize=11,
              title="按起始年代配色（1970 前起步并入 1970s）", title_fontsize=10.5, frameon=True)

    ax.set_title("E011 实验 · 讲者据原书 Figure 1.6（第 11 页）重绘：\n"
                 "计算机视觉五十年 · 33 个主题（条带起止为讲者读图估计）",
                 fontsize=15, fontweight="bold", color="#1e1e26", pad=14)

    fig.text(0.995, 0.006,
             "讲者设计实验：标签取自原书图；横轴扩至 1960 以容纳两个读图起点早于 1970 的主题；起止年代为读图近似值（原书未提供数据表）",
             ha="right", fontsize=9, color="#8a8378")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, facecolor="white")
    # 与幻灯片目录同步（deck 引用 slides/figures/），无该目录则跳过
    import pathlib
    twin = pathlib.Path(outpath).parents[2] / "slides" / "figures" / pathlib.Path(outpath).name
    if (pathlib.Path(outpath).parents[2] / "slides" / "figures").is_dir():
        twin.write_bytes(pathlib.Path(outpath).read_bytes())
        print(f"[demo] 已同步 {twin}")
    plt.close(fig)
    return outpath


def main():
    run_checks()
    out = draw(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "outputs", "timeline_fig1_6.png"))
    total, passed = len(_results), sum(_results)
    print(f"[demo] 图已存 {out}")
    print(f"汇总：{passed}/{total} PASS —— {'全部通过' if passed == total else '存在失败'}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
