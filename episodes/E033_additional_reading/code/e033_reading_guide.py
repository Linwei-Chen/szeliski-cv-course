#!/usr/bin/env python3
"""e033_reading_guide.py —— E033 §3.7 Additional reading S2 复盘型实证（权威存档 stdout）。

口径注记（七项，与 refs 逐字同步；权威存档=本 stdout 标头）：
· 引文提取口径：正文中括号引文按 regex 提取、分号拆条；内嵌节号（如 1992, Section 5.2）
  计一条；教材书单段的五本教材同按括号年份提取计条；本节引文计数全部由 evidence 文件
  现场解析（不从记忆计数）。
· 主题段划分：以 evidence 头注实测锚为准分八段（教材与顶会刊/合成算子/线性滤波/非线性
  滤波/形态学/金字塔与小波/非线性金字塔/warping 重采样与 morphing），段界=正文自然段。
· 集号映射口径：主题→课程集号映射为制作侧归纳（表随存档打印；映射合法性断言=集号目录
  真实存在）；「本课程未展开」的主题如实标注，不强行归集。
· 年份口径：引文内四位数 19xx/20xx 视为年份；同一引文多个年份取首个；合法性断言
  1982–2017（本节最早 Serra 1982、最晚 Gonzalez and Woods 2017）。
· 图件口径：图 A（§3.7 文献主题图谱）/图 B（第 3 章知识地图）均为程序生成；本节原书
  无图表——红线「原书图未使用」天然满足，图注标「程序生成」。
· 边界口径：p.180 §3.8 Exercises 起的内容不进统计（切点外，仅作切点证据留档）。
· 断言性质：全部为数据溯源/一致性断言（段锚在实测行、计数与文件一致、映射与目录一致），
  无数值算法实验——本节为文献罗列段（复盘型实证）。
"""
import re
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

matplotlib.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent.parent.parent
EP = Path(__file__).resolve().parent.parent
OUT = EP / "code" / "outputs"
OUT.mkdir(exist_ok=True)
FIGDIR = EP / "slides" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PASS = []
FAIL = []
RENDERED = {}  # P1-8：渲染进 PNG 像素的全部文字串（随存档打印+断言）

HEADER = """口径注记（九项，与 refs 逐字同步；权威存档=本 stdout 标头；stderr 已分离到单独文件）：
· 引文提取口径：正文中成对括号、括号内含四位数年份才计一条；分号隔开的引文拆开各计一条；
  内嵌节号（如 1992, Section 5.2）计一条；1983a/b 类字母后缀不拆分（同条）；教材书单段的
  五本教材同按括号年份提取计条；全部由 evidence 文件现场解析（不从记忆计数）。
· 主题段划分：以 evidence 头注实测锚为准分八段（教材与顶会刊/合成算子/线性滤波/非线性
  滤波/形态学/金字塔与小波/非线性金字塔/warping 与 morphing），段界=正文自然段
  （线性滤波=L39–40+L43 跨页，L41 为 p.179 页眉、L42 空行；金字塔与小波=L56–65，
  L66 起归非线性金字塔段）。
· 分项口径：非线性滤波段 L44–52 按类别关键词切片计数——bilateral filtering（双边）/
  related iterative algorithms（迭代）/ variational approaches（变分）/ guided filtering
  （引导）——分项之和=段总数（断言 1-4b）；非线性金字塔段 L66–70 按原书分类切片——
  in the computer vision literature（视觉 2 条）/ in computational photography（计算摄影
  3 条）（断言 1-7b）；warping 与 morphing 段 L71–77 三段切片——image processing
  literature（4 条）/ computer graphics（5 条）/ morphing（2 条）（断言 1-8b）。
· 集号映射口径：主题→课程集号映射为制作侧归纳（MAP 随存档打印；映射合法性断言=集号
  目录真实存在）；「本课程未展开」的主题如实标注，不强行归集。
· 年份口径：引文内四位数 19xx/20xx 视为年份；同一引文多个年份取首个；合法性断言
  1982–2017（本节最早 Serra 1982、最晚 Gonzalez and Woods 2017）。
· 图件口径：图 A（文献主题图谱）/图 B（第 3 章知识地图）均为程序生成，本节原书无图表；
  全部渲染文字串（标题/轴标签/柱标/行标签）随存档打印（P1-8），并断言无 E0XX/§/run/路径；
  图 B 行纵坐标（figure fraction）随存档打印（deck 行高亮遮罩定位依据——门A r2 P2-5）；
  图 A/B PNG 天然尺寸=figsize×dpi 且 savefig 无 tight 裁切（断言 4c——快速复审 N5：
  遮罩按 img 盒百分比定位与 row_fracs 一一对应的前提）。
· 回指与核改证据口径：跨集回指（E021 合成与玻璃反射/E023+E024 可分离/E025 双边滤波展开/
  E030 金字塔/E031 小波+可导向/E032 MIP-map+Beier–Neely）与 MAP 三处核改的证据随存档
  打印（#5 节：GLOSSARY 原行（5-2a steerable/5-2b MIP-map/5-2c bilateral filter/
  5-2d separable filtering，term 精确匹配）/episodes slug 清单/E032 口播原文/E030 f012
  口播原文/E031 f009 口播原文/全课程 E009–E032 五人名 grep 明细+命中上下文+层归属/
  Fattal 2009 p.164 原文与 f009 白话逐义对应证据（5-5e/5-5f/5-5g——快速复审 N1 修正：
  内容层讲过未点名，f010 收窄为「未按篇讲」）/跨页原文附录/status.handoff_out 原文）。
· 去重口径：条数=括号引文出现次数（73）；按作者+年份键去重=72 条不同条目
  （Gomes, Darsa et al. 1999 在 L72 warping 段与 L77 morphing 收尾各出现一次，断言 1-9b）；
  篇数另计=75 篇（72 条目中 1994a,b/1983a,b/2006a,b 三处各含两篇：72−3+2×3=75，断言
  1-9d）；1983a/b 类字母后缀同条不拆（若 a、b 分计则条数 73→76，且合成算子 4→5、
  金字塔与小波 19→20、非线性金字塔 5→6——断言 1-9c）。
· 边界口径：p.180 §3.8 Exercises 起的内容不进统计（切点外，仅作切点证据留档）；
  断言性质：全部为数据溯源/一致性断言，无数值算法实验（复盘型实证）。
"""
print(HEADER)


def chk(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


def readout(label, detail):
    print(f"  [读数] {label}: {detail}")


# ── 载入 evidence（⚠️ 禁用 splitlines——按 \f 保留块结构后统一处理行号=文件行号）──
raw = (EP / "production/evidence/e033_book_pp178_180.txt").read_text()
lines = raw.split("\n")

# ── #1 主题段解析与引文计数 ──
print("\n#1 §3.7 主题段解析与引文计数（数据溯源）")
# 段定义：段名 -> (段首行号(文件行号), 段末行号, 段首关键词)
SECTIONS = [
    ("教材与顶会刊", 30, 34, "some popular"),
    ("合成算子", 35, 38, "compositing operators"),
    ("线性滤波", 39, 43, "linear filtering"),
    ("非线性滤波", 44, 52, "non-linear filtering"),
    ("形态学", 53, 55, "morphology"),
    ("金字塔与小波", 56, 65, "image pyramids and pyramid blending"),
    ("非线性金字塔", 66, 70, "non-linear filters"),
    ("warping 与 morphing", 71, 77, "image warping and resampling"),
]
CITE = re.compile(r"\(([^()]*?\d{4}[a-z]?[^()]*?)\)")
YEAR = re.compile(r"(19|20)\d{2}")


def dehyph(text):
    """合并 pdftotext 换行连字（bi- lateral → bilateral），仅供关键词定位/作者回溯。"""
    return re.sub(r"-\s+", "", text)


def citations_in_text(text):
    """提取文本中的括号引文（分号拆条）。

    返回 (条目文本, 去重键) 列表：作者在括号外的纯年份括号（如 Jähne (1997)）用
    前文作者名+年份作键——否则字符串级去重会把两本 1997 教材误合（P1-9 去重口径）。
    """
    items = []
    for m in CITE.finditer(text):
        pre = dehyph(text[max(0, m.start() - 45):m.start()])
        for part in m.group(1).split(";"):
            part = part.strip()
            if not YEAR.search(part):
                continue
            if re.fullmatch(r"\d{4}[a-z]?(,.*)?", part):
                name = re.search(r"([A-Za-zÀ-ž\-.' ]+?)\s*$", pre)
                key = f"{name.group(1).strip() if name else '?'}|{part}"
            else:
                key = part
            items.append((part, key))
    return items


def citations_in(lo, hi):
    """提取 [lo,hi] 行内的括号引文（分号拆条），返回 (文本, 键) 列表。"""
    return citations_in_text(" ".join(lines[lo - 1:hi]))


counts = {}
years = []
items_all = []
items_keyed = []
for name, lo, hi, kw in SECTIONS:
    assert any(kw in lines[lo - 1 + d] for d in range(3)), f"段锚 {name} 关键词缺失"
    its = citations_in(lo, hi)
    counts[name] = len(its)
    items_all.extend(its)
    items_keyed.extend(k for _, k in its)
    for it, _k in its:
        y = re.search(r"((19|20)\d{2})", it)
        if y:
            years.append(int(y.group(0)))
    # 门A r2 P3-4：线性滤波跨页段打印实际解析行集合，不用跨度写法
    span = "L39–40,L43" if name == "线性滤波" else f"L{lo}–{hi}"
    chk(f"1-{SECTIONS.index((name, lo, hi, kw)) + 1} {name} 段锚在 {span}（关键词「{kw}」）且引文 {len(its)} 条",
        len(its) > 0, span)
# 分项口径（非线性滤波段 L44–52）：按类别关键词切片计数（P1-6）
seg = dehyph(" ".join(lines[43:52]))
keys = ["bilateral filtering", "related iterative algorithms",
        "variational approaches", "guided filtering"]
pos = [seg.find(k) for k in keys] + [len(seg)]
assert all(p >= 0 for p in pos[:4]), f"分项关键词缺失: {pos}"
sub_names = ["双边", "迭代", "变分", "引导"]
sub_counts = [len(citations_in_text(seg[pos[i]:pos[i + 1]])) for i in range(4)]
readout("非线性滤波分项", dict(zip(sub_names, sub_counts)))
chk("1-4b 非线性滤波分项之和=段总数且为 5/8/4/3",
    sum(sub_counts) == counts["非线性滤波"] and sub_counts == [5, 8, 4, 3],
    f"{sub_counts} vs 20")
# 1-7b 非线性金字塔段分项（门A r2 P3-8/P2-7）：视觉 2 条+计算摄影 3 条=5
seg2 = dehyph(" ".join(lines[65:70]))
k2 = ["computer vision literature", "computational photography"]
pos2 = [seg2.find(k) for k in k2] + [len(seg2)]
assert all(p >= 0 for p in pos2[:2]), f"非线性金字塔分项关键词缺失: {pos2}"
sub2 = [len(citations_in_text(seg2[pos2[0]:pos2[1]])),
        len(citations_in_text(seg2[pos2[1]:pos2[2]]))]
readout("非线性金字塔分项", {"视觉": sub2[0], "计算摄影": sub2[1]})
chk("1-7b 非线性金字塔分项 视觉 2+计算摄影 3=段总数 5（Gluckman 2006a,b 计 1 条、"
    "Lyu–Simoncelli 2008 计 1 条；Bae/Farbman/Fattal 各 1 条）",
    sum(sub2) == counts["非线性金字塔"] and sub2 == [2, 3], f"{sub2} vs 5")
# 1-8b warping 与 morphing 段分项（门A r2 P2-7）：图像处理 4+图形学 5+morphing 2=11
seg3 = dehyph(" ".join(lines[70:77]))
k3 = ["in the image processing literature (", ") and in computer graphics (",
      "), where they go under the name", "Combinations of image warping"]
pos3 = [seg3.find(k) for k in k3] + [len(seg3)]
assert all(p >= 0 for p in pos3[:4]), f"warping 分项关键词缺失: {pos3}"
sub3 = [len(citations_in_text(seg3[pos3[0]:pos3[1] + 1])),
        len(citations_in_text(seg3[pos3[1]:pos3[2] + 1])),
        len(citations_in_text(seg3[pos3[3]:]))]
readout("warping 与 morphing 分项", {"图像处理": sub3[0], "图形学": sub3[1], "morphing": sub3[2]})
chk("1-8b warping 与 morphing 分项 4+5+2=段总数 11（图像处理：Wolberg/Dodgson/Gomes 1999/"
    "Szeliski–Winder–Uyttendaele；图形学：Williams/Heckbert/Barkans/Weinhaus–Devarajan/"
    "Akenine-Möller–Haines；morphing：Beier–Neely/Gomes 1999）",
    sum(sub3) == counts["warping 与 morphing"] and sub3 == [4, 5, 2], f"{sub3} vs 11")
readout("各主题引文数", json.dumps(counts, ensure_ascii=False))
total = sum(counts.values())
readout("总引文条数", str(total))
readout("年份范围", f"{min(years)}–{max(years)}（n={len(years)}）")
chk("1-9 年份全部在 1982–2017（Serra 1982–Gonzalez and Woods 2017）",
    1982 <= min(years) and max(years) <= 2017, f"{min(years)}–{max(years)}")
# 去重口径（P1-9+门A r2 P1-5）：条数=出现次数；按作者+年份键去重=条目数；篇数另计
uniq = sorted(set(items_keyed))
dups = [u for u in uniq if items_keyed.count(u) > 1]
readout("去重条目数", f"{len(uniq)} 条不同条目（重复键: {dups}）")
chk("1-9b 去重：73 条=72 条不同条目（Gomes, Darsa et al. 1999 两段各现一次）",
    len(items_keyed) == 73 and len(uniq) == 72 and dups == ["Gomes, Darsa et al. 1999"],
    f"{len(items_keyed)} 条/{len(uniq)} 条目")
# a/b 后缀同条不拆的区分力：Blinn 1994a,b、Burt–Adelson 1983a,b、Gluckman 2006a,b
ab_pairs = sorted(k for k in uniq if re.search(r"\d{4}a,b", k))
readout("a,b 同条键", ab_pairs)
chk("1-9c a,b 字母后缀同条共 3 处未拆（若分计条数 73→76）",
    len(ab_pairs) == 3 and len(items_keyed) == 73)
# 1-9d 篇数口径（门A r2 P1-5）：72 条目中 3 处 a,b 各含两篇 → 75 篇
papers = len(uniq) - len(ab_pairs) + 2 * len(ab_pairs)
chk("1-9d 篇数=75（72 条目−3 处 a,b 单条+3×2 篇）",
    papers == 75 and len(ab_pairs) == 3, f"{len(uniq)}−{len(ab_pairs)}+2×{len(ab_pairs)}={papers}")

# ── #2 主题→课程集号映射（制作侧归纳，映射合法性=集号目录存在）──
print("\n#2 主题→课程集号映射核对")
MAP = {
    "教材与顶会刊": "课程整体（无单集对应——本集即导读）",
    "合成算子": "E021",
    "线性滤波": "E023/E024",
    "非线性滤波": "E025",
    "形态学": "E026",
    "金字塔与小波": "E030/E031",
    "非线性金字塔": "本段 5 条文献未按篇讲（Fattal 2009 思路 E031 f009 白话一句带过；非线性塔构造主题第 22 集沾边——原书第 158–159 页局部拉普拉斯滤波，E030 f012）",
    "warping 与 morphing": "E032",
}
readout("MAP 核改③ v2 记录", "v1 表述「第 22 集金字塔为线性搭法」不实——E030 f012 实讲过"
        "局部拉普拉斯滤波（原书第 158–159 页，非线性塔构造）；修正为「本段 5 条文献未按篇讲、"
        "主题第 22 集沾边」。v1 的 grep 基础（E030/E031 无局部拉普拉斯）为错误结论，"
        "本轮 rglob 全文件 grep 已纠正（5-5b）。门A 快速复审 N1 修正：v2 曾判 Fattal 2009 "
        "与 f009 白话句「无主题交集」——原书 p.164 实证 Fattal 2009 恰挂「lifted weighted "
        "wavelets…adapt…low-level image manipulation」句（5-5e/5-5f/5-5g），即 f009 白话"
        "「加权的提升小波——系数随手头的问题自适应——在低层图像操作里极为有效」的出处——"
        "内容层讲过、未点名，f010 口播收窄为「未按篇讲；Fattal 思路第 23 集一句带过」。")
for name, target in MAP.items():
    eps = re.findall(r"E\d{3}", target)
    ok = all((ROOT / "episodes" / f"{e}_" .rstrip("_")).parent.exists() and
             list((ROOT / "episodes").glob(f"{e}_*")) for e in eps) if eps else True
    chk(f"2-{list(MAP).index(name) + 1} {name} → {target}（集号目录真实存在）", ok)

# ── #3 图 A：§3.7 文献主题图谱（程序生成）──
print("\n#3 图 A 文献主题图谱（程序生成，本节原书无图表）")
names = [s[0] for s in SECTIONS]
vals = [counts[n] for n in names]
fig, ax = plt.subplots(figsize=(13.6, 4.6))


def lesson(e):
    """E0XX → 正课第 N 集（课程 pagefoot 口径：E032=正课第 24 集，差 8）。"""
    return f"正课第 {int(e[1:]) - 8} 集"


SHORT = {"教材与顶会刊": "本集导读", "非线性金字塔": "第 22 集沾边·未按篇讲"}


def short_target(name):
    if name in SHORT:
        return SHORT[name]
    nums = sorted({lesson(e).replace("正课第 ", "").replace(" 集", "")
                   for e in re.findall(r"E\d{3}", MAP[name])})
    return "正课第 " + "、".join(nums) + " 集"


colors = ["#4a6fa5" if "未按篇讲" not in MAP[n] and "课程整体" not in MAP[n] else "#b8b8b8"
          for n in names]
bars = ax.bar(range(len(names)), vals, color=colors)
ax.set_xticks(range(len(names)))
xtlabels = [f"{n}\n（{short_target(n)}）" for n in names]
ax.set_xticklabels(xtlabels, fontsize=9.5)
for i, v in enumerate(vals):
    ax.text(i, v + 0.4, str(v), ha="center", fontsize=11)
ylabel_a = "括号引文条数（原书第 178–179 页实测）"
ax.set_ylabel(ylabel_a)
title_a = ("阅读指南文献主题图谱：各主题段引文数与正课对应"
           "（程序生成；数据=原书第 178–179 页现场解析）")
ax.set_title(title_a, fontsize=12)
ax.set_ylim(0, max(vals) * 1.25)
fig.tight_layout()
fig.savefig(FIGDIR / "e033_litmap.png", dpi=110)
plt.close(fig)
RENDERED["图A"] = [title_a, ylabel_a] + xtlabels + [str(v) for v in vals]
chk("3-1 图 A 生成且总条数与 #1 一致", sum(vals) == total, f"{sum(vals)}={total}")

# ── #4 图 B：第 3 章知识地图（程序生成）──
print("\n#4 图 B 第 3 章知识地图（程序生成）")
CH3 = [
    ("§3.1 点运算", "E021/E022"),
    ("§3.2 线性滤波", "E023/E024"),
    ("§3.3 非线性与二值形态学", "E025/E026"),
    ("§3.4 傅里叶变换", "E027/E028"),
    ("§3.5 插值/金字塔/小波/融合", "E029/E030/E031"),
    ("§3.6 几何变换", "E032"),
    ("§3.7 阅读指南", "E033（本集）"),
]
fig, ax = plt.subplots(figsize=(12.8, 5.6))
ax.axis("off")
ys = np.linspace(0.92, 0.08, len(CH3))
b_labels = []
b_rows = []
for (sec, eps), y in zip(CH3, ys):
    # 渲染层剥掉「§3.x」节号（可见层口径：deck 图件无 § 节号，D20）；CH3 内部数据保留
    disp = sec.split(" ", 1)[1]
    ax.text(0.04, y, disp, fontsize=13, va="center",
            fontweight="bold" if sec == "§3.7 阅读指南" else "normal")
    ax.annotate("", xy=(0.52, y), xytext=(0.30, y),
                arrowprops=dict(arrowstyle="-|>", color="#4a6fa5", lw=1.4))
    nums = sorted({int(e[1:]) - 8 for e in re.findall(r"E\d{3}", eps)})
    label = "正课第 " + "、".join(str(n) for n in nums) + " 集"
    if "E033" in eps:
        label += "（本集）"
    ax.text(0.55, y, label, fontsize=13, va="center", color="#4a6fa5",
            fontweight="normal")
    b_labels.append(disp)
    b_rows.append(label)
    mapping_ok = all(list((ROOT / "episodes").glob(f"{e}_*"))
                     for e in re.findall(r"E\d{3}", eps))
    chk(f"4-{CH3.index((sec, eps)) + 1} 图 B 映射 {sec} → {eps}（集号目录真实存在）",
        mapping_ok)
title_b = "第 3 章知识地图：主题 → 正课对应（程序生成；映射=本课程实际进度）"
ax.text(0.04, 1.0, title_b, fontsize=13, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.06)
fig.tight_layout()
fig.savefig(FIGDIR / "e033_chapter_map.png", dpi=110)
# 门A r2 P2-5（方案二 CSS 遮罩）：行纵坐标随存档打印，deck 遮罩按此定位不手调
fig.canvas.draw()
inv_fig = fig.transFigure.inverted()
row_fracs = []
for (sec, _eps), y in zip(CH3, ys):
    fy = float(inv_fig.transform(ax.transData.transform((0.5, y)))[1])
    row_fracs.append(round(fy, 4))
readout("图B 行纵坐标（figure fraction，自顶向下顺序）", json.dumps(row_fracs))
plt.close(fig)
RENDERED["图B"] = [title_b] + b_labels + b_rows
chk("4b-2 图 B 行坐标 7 行且单调递减（遮罩定位依据）",
    len(row_fracs) == 7 and all(a > b for a, b in zip(row_fracs, row_fracs[1:])),
    f"上四行/下三行分界≈{((row_fracs[3] + row_fracs[4]) / 2):.4f}")

# 快速复审 N5：图 A/B PNG 天然尺寸与裁切实证——savefig 未用 tight 裁切（bbox_inches
#   默认不裁、rcParam 非 tight），PNG 像素尺寸=figsize×dpi，deck 遮罩按 img 盒百分比
#   定位才与 row_fracs（figure fraction）一一对应；渲染管线逐帧以 ?step=N 载入（render_deck）。
print("\n#4c 图件 PNG 天然尺寸（快速复审 N5：遮罩对位实证）")
readout("savefig.bbox rcParam", repr(matplotlib.rcParams["savefig.bbox"]))
for tag, fn, figw, figh in [("图A", "e033_litmap.png", 13.6, 4.6),
                            ("图B", "e033_chapter_map.png", 12.8, 5.6)]:
    w, h = Image.open(FIGDIR / fn).size
    readout(f"{tag} PNG 天然尺寸", f"{w}×{h}（figsize {figw}×{figh} × dpi 110）")
    chk(f"4c {tag} PNG 尺寸=figsize×dpi 且 savefig 无 tight 裁切（遮罩按行坐标对位前提）",
        w == round(figw * 110) and h == round(figh * 110)
        and matplotlib.rcParams["savefig.bbox"] in (None, "none", "standard"),
        f"{w}×{h} vs {round(figw * 110)}×{round(figh * 110)}")

# P1-8：渲染文字串存档+可见层断言（无 E0XX 集号/§ 节号/run/内部路径/「门」）
print("\n#4b 图件渲染文字串（P1-8：全部进入 PNG 像素的文字）")
for figname, strs in RENDERED.items():
    for s in strs:
        readout(figname, s)
BAD_PAT = re.compile(r"E\d{3}|§|run\d|production|evidence|/Users|门")
bad = [s for strs in RENDERED.values() for s in strs if BAD_PAT.search(s)]
chk("4b 图件渲染文字无内部记号（无 E0XX/§/run/路径/「门」）", not bad,
    f"违规: {bad}" if bad else f"共 {sum(len(v) for v in RENDERED.values())} 条已打印")

# ── #5 跨集回指与 MAP 核改证据（P1-7：随审可独立复核）──
print("\n#5 跨集回指与 MAP 核改证据（GLOSSARY 原行/episodes 清单/E032 口播/grep 实测）")
gloss = (ROOT / "GLOSSARY.md").read_text()

# 5-1 GLOSSARY 原行：compositing / glass reflection（首现集证据）
gl_lines = gloss.split("\n")
for term in ["compositing", "glass reflection"]:
    hits = [ln for ln in gl_lines if f"| {term}" in ln.lower()]
    for ln in hits:
        readout(f"GLOSSARY[{term}]", ln.strip())
    chk(f"5-1 GLOSSARY 含 {term} 条目且挂 E021（首现集）",
        any("E021" in ln for ln in hits), f"{len(hits)} 行")

# 5-2 GLOSSARY 原行（门A r2 P0-B②；快速复审 N4：断言编号拆 5-2a/5-2b/5-2c/5-2d，
#   MIP-map 改 term 精确匹配——子串检索曾误命中 image pyramid/trilinear 行，
#   补 separable filtering 原行——标头声明「E024 可分离」此前只有 slug 无原行）
hits_st = [ln for ln in gl_lines if ln.strip().lower().startswith("| steerable pyramid |")]
for ln in hits_st:
    readout("GLOSSARY[steerable pyramid]", ln.strip())
chk("5-2a GLOSSARY steerable pyramid 条目挂 E031（可导向金字塔展开集——f006/f009 回指「可导向」词条）",
    len(hits_st) == 1 and "E031" in hits_st[0], f"命中 {len(hits_st)} 行")
hits_mm = [ln for ln in gl_lines if ln.strip().lower().startswith("| mip-map |")]
for ln in hits_mm:
    readout("GLOSSARY[MIP-map]", ln.strip())
chk("5-2b GLOSSARY MIP-map 条目 term 精确匹配且挂 E030（p.155 开篇提及；E032 展开）",
    len(hits_mm) == 1 and "E030" in hits_mm[0], f"命中 {len(hits_mm)} 行")
hits_bf = [ln for ln in gl_lines if ln.strip().lower().startswith("| bilateral filter |")]
for ln in hits_bf:
    readout("GLOSSARY[bilateral filter]", ln.strip())
chk("5-2c GLOSSARY bilateral filter 条目挂 E024（p.131 首现预告，Figure 3.18 图注）"
    "且行内含 E025（展开承诺兑现：式 3.34–3.37，pp.133–137）",
    len(hits_bf) == 1 and "E024" in hits_bf[0] and "E025" in hits_bf[0],
    f"命中 {len(hits_bf)} 行")
hits_sep = [ln for ln in gl_lines if ln.strip().lower().startswith("| separable filtering |")]
for ln in hits_sep:
    readout("GLOSSARY[separable filtering]", ln.strip())
chk("5-2d GLOSSARY separable filtering 条目挂 E023（f026 预告首现）且行内含 E024"
    "（展开：式 3.20/3.21，p.124——回指证据原行）",
    len(hits_sep) == 1 and "E023" in hits_sep[0] and "E024" in hits_sep[0],
    f"命中 {len(hits_sep)} 行")

# 5-3 episodes 目录清单（映射涉及的集）
mapped_eps = ["E021", "E023", "E024", "E025", "E026", "E030", "E031", "E032", "E033"]
slug_list = {}
for e in mapped_eps:
    g = list((ROOT / "episodes").glob(f"{e}_*"))
    slug_list[e] = g[0].name if g else "（缺）"
readout("episodes slug 清单", json.dumps(slug_list, ensure_ascii=False))
chk("5-3 映射涉及的 9 个集号目录全部真实存在", all("（缺）" not in v for v in slug_list.values()))

# 5-4 E032 f011（MIP-map）/f016（Beier–Neely）口播原文
e032 = json.loads((ROOT / "episodes/E032_geometric_transform/production/shots.json")
                  .read_text())
for fr, kws in [("f011", ["MIP-map", "Williams"]), ("f016", ["Beier", "Neely"])]:
    ch = next(c for c in e032["chunks"] if c["frame"] == fr)
    readout(f"E032 {fr} 口播", ch["narration"])
    chk(f"5-4 E032 {fr} 口播含 {'/'.join(kws)}（跨集回指锚真实）",
        all(k in ch["narration"] for k in kws))

# 5-5 「未讲」证据（MAP 核改③ v2；门A r2 P1-1：grep 扩全部已制作正课集 E009–E032，
#   五人名逐一打印命中行+层归属——层=口播（shots.json narration）/refs 列（shots.json
#   refs）/deck 可见层（deck.html 文本））
print("\n#5a 全课程 grep（E009–E032 正课 1–24 集；shots.json+deck.html）")
NAMES = ["Gluckman", "Lyu", "Farbman", "Bae", "Fattal"]
course_eps = sorted(
    p for p in (ROOT / "episodes").iterdir()
    if p.is_dir() and re.match(r"E(0(0[9]|[12]\d|3[0-2]))_", p.name))
chk("5-5-0 全课程集目录 E009–E032 共 24 集", len(course_eps) == 24,
    f"{[p.name.split('_')[0] for p in course_eps]}")
visible_hits = []   # 口播层或 deck 可见层命中（会推翻「未讲」）
refs_hits = []      # 仅 refs 列命中（引文列出现≠讲过，注明层）
for kw in NAMES:
    for ep in course_eps:
        sj = ep / "production" / "shots.json"
        dh = ep / "slides" / "deck.html"
        if sj.exists():
            d = json.loads(sj.read_text())
            for c in d["chunks"]:
                if kw in c["narration"]:
                    visible_hits.append((kw, ep.name, c["frame"] + " 口播"))
                    readout(f"grep[{kw}] {ep.name} 口播", c["narration"][:150])
                for r_i, r in enumerate(c.get("refs", [])):
                    if kw in r:
                        refs_hits.append((kw, ep.name, c["frame"], f"refs[{r_i}]"))
                        readout(f"grep[{kw}] {ep.name} {c['frame']} refs 列",
                                r[:150])
        if dh.exists():
            for ln_no, ln in enumerate(dh.read_text().split("\n"), 1):
                if kw in ln:
                    visible_hits.append((kw, ep.name, f"deck.html L{ln_no}"))
                    readout(f"grep[{kw}] {ep.name} deck 可见层 L{ln_no}", ln.strip()[:150])
    readout(f"grep[{kw}] 汇总", f"口播/可见层 {sum(1 for h in visible_hits if h[0] == kw)} 处；"
            f"仅 refs 列 {sum(1 for h in refs_hits if h[0] == kw)} 处")
chk("5-5 五人名（Gluckman/Lyu/Farbman/Bae/Fattal）在全部正课集口播与 deck 可见层无命中"
    "（「未讲」口径=按名讲授）",
    not visible_hits, f"可见层命中: {visible_hits}" if visible_hits else
    f"refs 列命中（不属讲授，注明层）: {[(h[0], h[1], h[3] if len(h) > 3 else h[2]) for h in refs_hits]}")

# 5-5b E030 f012 口播原句（局部拉普拉斯=第 22 集沾边证据；门A r2 P1-1：打印原句而非只打印路径）
e030 = json.loads((ROOT / "episodes/E030_pyramid/production/shots.json").read_text())
ch12 = next(c for c in e030["chunks"] if c["frame"] == "f012")
readout("E030 f012 口播", ch12["narration"])
chk("5-5b E030 f012 口播含局部拉普拉斯（第 22 集沾边证据——v1 grep 结论纠正）",
    "局部拉普拉斯" in ch12["narration"])

# 5-5c/5-5d E031 明细（门A r2 P1-1/5.2 节；快速复审 N1：命中须打印关键词前后上下文，
#   只打印 refs 条目前 150 字符时人名不在片段内=证据不充分）
e031 = json.loads((ROOT / "episodes/E031_wavelets_blending/production/shots.json").read_text())
lyu_refs = [(c["frame"], i, r) for c in e031["chunks"] for i, r in enumerate(c["refs"]) if "Lyu" in r]
fattal_refs = [(c["frame"], i, r) for c in e031["chunks"] for i, r in enumerate(c["refs"]) if "Fattal" in r]
fattal_narr = [c["frame"] for c in e031["chunks"] if "Fattal" in c["narration"]]
for kw, hits in [("Lyu", lyu_refs), ("Fattal", fattal_refs)]:
    for fr, i, r in hits:
        pos = r.find(kw)
        readout(f"E031 {fr} refs[{i}] {kw} 命中上下文（±80 字符）",
                f"…{r[max(0, pos - 80):pos + 80]}…")
lyu_narr = [c["frame"] for c in e031["chunks"] if "Lyu" in c["narration"]]
readout("E031 Lyu 命中层", f"refs 列 {[h[0] for h in lyu_refs]}（口播 {lyu_narr or '无'}）")
readout("E031 Fattal 命中层", f"refs 列 {[h[0] for h in fattal_refs]}（口播 {fattal_narr or '无'}）")
ch9 = e031["chunks"][8]
readout("E031 f009 口播原句", ch9["narration"])
chk("5-5c §3.7 的 Lyu and Simoncelli (2008) 未讲（E031 仅 refs 列 2009 另一篇，口播未点名）",
    lyu_refs and not lyu_narr,
    f"refs 命中帧 {[h[0] for h in lyu_refs]}（上下文已打印，命中均在引文列）")
chk("5-5d Fattal (2009) 未讲：E031 仅 refs 列出现该名（口播/deck 可见层未点名）",
    fattal_refs and not fattal_narr
    and "Fattal" not in (ROOT / "episodes/E031_wavelets_blending/slides/deck.html").read_text(),
    f"refs 命中帧 {[h[0] for h in fattal_refs]}（上下文已打印，命中均在引文列）")

# 5-5e/5-5f/5-5g Fattal 2009 思路判定（快速复审 N1：原 5-5e「无主题交集」无依据——
#   p.164 实证 Fattal 2009 挂在 lifted weighted wavelets 句上，即 f009 白话转述的出处。
#   修法=分支一「内容层讲过、未点名」：f010 口播收窄为「未按篇讲；Fattal 思路第 23 集一句带过」）
print("\n#5b-2 Fattal 2009 原文与 f009 白话对应证据（p.164，E031 evidence 实测）")
e031_ev = (ROOT / "episodes/E031_wavelets_blending/production/evidence/e031_book_pp159_169.txt").read_text().split("\n")
fattal_2009_lines = [i + 1 for i, ln in enumerate(e031_ev) if "Fattal 2009" in ln]
f_lo, f_hi = min(fattal_2009_lines), max(fattal_2009_lines)
f_ctx = " ".join(e031_ev[f_lo - 4:f_hi + 2]).strip()
f_ctx1 = re.sub(r"\s+", " ", f_ctx)
readout("E031 evidence Fattal 2009 命中行", f"L{fattal_2009_lines} → {f_ctx1}")
chk("5-5e 原书 p.164 原文：lifted weighted wavelets（系数自适应 underlying problem）挂"
    "（Fattal 2009）且「extremely effective for low-level image manipulation tasks」——"
    "即 E031 f009 白话「加权的提升小波——系数随手头的问题自适应——在低层图像操作里极为有效」的出处",
    "Fattal 2009" in f_ctx1 and "lifted weighted wavelets" in f_ctx1
    and "adapt to the underlying problem" in f_ctx1
    and "low-level image manipulation tasks" in f_ctx1, f"L{fattal_2009_lines}")
f_ref = next(r for fr, _, r in fattal_refs if fr == "f009")
fpos = f_ref.find("Fattal")
readout("E031 f009 refs Fattal 上下文（±80 字符）", f"…{f_ref[max(0, fpos - 80):fpos + 80]}…")
chk("5-5f E031 f009 Fattal 命中所在 refs 引文属 p.164 段（Lifted wavelets/being solved 上下文）",
    "Lifted wavelets" in f_ref and "being solved" in f_ref)
e033_ev = (EP / "production/evidence/e033_book_pp178_180.txt").read_text().split("\n")
l70 = next(ln for ln in e033_ev if "Fattal (2009)" in ln)
chk("5-5g 作者+年份键相同：§3.7 L70「Fattal (2009)」=p.164「Fattal 2009」（同一篇文献），"
    "且 f009 口播白话句含「加权的提升小波/自适应/低层图像操作」三锚（逐义转述 p.164）"
    "——判定：Fattal 2009 思路已在 E031 f009 白话一句带过（未点名），f010 口播收窄为"
    "「未按篇讲」（快速复审 N1 分支一）",
    "Fattal (2009)" in l70 and "Fattal 2009" in f_ctx1
    and all(k in ch9["narration"] for k in ["加权的提升小波", "自适应", "低层图像操作"]),
    f"E033 L{e033_ev.index(l70) + 1}；f009 口播锚齐")

# 5-6 跨页原文附录（门A r2 P2-8：口播报了页码的事实，原书原文必须在本集存档可查）
print("\n#5b 跨页原文附录（p.158–159 / p.168 / p.172——从 E030/E032 evidence 实测）")
e030_ev = (ROOT / "episodes/E030_pyramid/production/evidence/e030_book_pp154_159.txt").read_text().split("\n")
e032_ev = (ROOT / "episodes/E032_geometric_transform/production/evidence/e032_book_pp168_179.txt").read_text().split("\n")
xpage = [
    ("E030 evidence", e030_ev, "local Laplacian filters",
     "local Laplacian filters, which uses local range clipping in the construction of a modified Laplacian pyramid"),
    ("E032 evidence", e032_ev, "first proposed by Williams",
     "MIP-mapping was first proposed by Williams (1983)"),
    ("E032 evidence", e032_ev, "images onto surfaces",
     "texture mapping, which is how the computer graphics community refers to the topic of warping images onto surfaces"),
]
for src, ev, kw, quote in xpage:
    hit_nos = [i + 1 for i, ln in enumerate(ev) if kw in ln]
    ctx = " ".join(ev[max(0, hit_nos[0] - 2):hit_nos[-1] + 1]).strip() if hit_nos else "（无命中）"
    ctx1 = re.sub(r"\s+", " ", ctx)
    readout(f"跨页原文[{kw[:24]}] {src}", f"L{hit_nos} → …{ctx1}…" if hit_nos else ctx)
    joined = re.sub(r"\s+", " ", re.sub(r"-\s+", "", " ".join(ev)))  # 连字合并+空白归一
    qn = re.sub(r"\s+", " ", re.sub(r"-\s+", "", quote))
    chk(f"5-6 {src} 含原书原文（门A r2 P2-8 跨页原文附录）且引句逐字（跨行连读）",
        bool(hit_nos) and qn in joined, f"L{hit_nos}")
# refs 中的英文引句必须落在上述原文内（f011 两句）
e033_shots = json.loads((EP / "production/shots.json").read_text())
refs_blob = " ".join(r for c in e033_shots["chunks"] for r in c["refs"])
joined_e032 = re.sub(r"\s+", " ", re.sub(r"-\s+", "", " ".join(e032_ev)))
for q in ["MIP-mapping was first proposed by Williams (1983)",
          "warping images onto surfaces"]:
    qn = re.sub(r"\s+", " ", re.sub(r"-\s+", "", q))
    chk(f"5-6x refs 英文引句「{q[:36]}…」在跨页原文附录内（红线 6）",
        qn in joined_e032 and q in refs_blob)

# 5-7 status.handoff_out 原文（门A r2 P3-1：移交句「把第 2、3 章的动手题挑几道做透」的依据）
print("\n#5c 移交依据原文（status.handoff_out）")
status = json.loads((EP / "status.json").read_text())
readout("status.handoff_out", status["scope"]["handoff_out"])
chk("5-7 handoff_out 含 E034 习题实验课定位（Ch2–3 综合上机）",
    "E034" in status["scope"]["handoff_out"] and "综合上机" in status["scope"]["handoff_out"])

print(f"\n汇总：{len(PASS)} 条 PASS / {len(FAIL)} 条 FAIL")
if FAIL:
    print("FAIL:", FAIL)
    raise SystemExit(1)
print("DONE")
