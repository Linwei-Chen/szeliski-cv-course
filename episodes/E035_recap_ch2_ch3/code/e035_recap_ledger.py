#!/usr/bin/env python3
"""E035 S2 串联台账实验：读 E013–E034 各集 shots.json（全课程页码引用单源）+run 存档，
产出「概念→定义集→页码锚→回响实验」台账并跑三层断言——回链/存档/章界。
stdout 落 code/outputs/e035_recap_ledger.txt（权威档），stderr 分离。"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]  # code→集→episodes→课程根
EP = ROOT / "episodes" / "E035_recap_ch2_ch3"
OUT = EP / "code" / "outputs" / "e035_recap_ledger.txt"
ANCHORS = (EP / "production" / "evidence" / "e035_book_ch2_ch3_anchors.txt").read_text()

PAGE = re.compile(r"原书第 (\d+) 页")

# 台账行：（概念, 集, 原书节锚（节首页，own 证据证节界）, 回链页（该集实引）, 锚来源, 回响集, 回响说明）
# 锚来源 own=E035 自有证据（E013–E020 老集无页码引用）；link=回链页须在该集 shots.json 引用集合
ROWS = [
    ("几何基元与变换", "E013+E014", "§2.1 Geometric primitives p.36（2.1.2 3D p.43）", 43, "own", "E013+E014", "变换家族一图流（E013 与 E014 同站）"),
    ("相机投影与内参", "E015+E016", "§2.1.4 3D to 2D projections p.51", None, "own", "E015+E016", "投影方程手推"),
    ("镜头畸变", "E017", "§2.1.5 Lens distortions p.63", None, "own", "E017", "畸变模型演示"),
    ("光度成像", "E018", "§2.2 Photometric image formation p.66", None, "own", "E018", "反射模型演示"),
    ("光学", "E019", "§2.2.3 Optics p.74", 74, "link", "E019", "薄透镜与 f 数"),
    ("相机与色彩", "E020", "§2.3 The digital camera p.79（2.3.2 Color p.87）", 87, "link", "E020", "Bayer 与伽马"),
    ("点操作", "E021", "§3.1 Point operators p.109", 109, "link", "E021", "增益偏置≡1×1 卷积"),
    ("直方图均衡", "E022", "§3.1.4 Histogram equalization p.115", 115, "link", "E022", "累计直方图映射"),
    ("线性滤波", "E023", "§3.2 Linear filtering p.119（3.2.3 steerable p.127）", 119, "link", "E023", "锐化核与转向滤波"),
    ("可分离滤波", "E024", "§3.2.1 Separable filtering p.124", 124, "link", "E024", "两趟卷积等价"),
    ("非线性滤波", "E025", "§3.3 More neighborhood operators p.131", 131, "link", "E025", "中值/双边去噪"),
    ("二值与形态", "E026", "§3.3.3 Binary image processing p.138", 138, "link", "E026", "形态学链"),
    ("傅里叶变换", "E027+E028", "§3.4 Fourier transforms p.142", 142, "link", "E027+E028", "频域去噪与 PSNR"),
    ("插值与抽取", "E029", "§3.5 Pyramids and wavelets p.149", 149, "link", "E029", "双线性插值核"),
    ("金字塔", "E030", "§3.5.3 Multi-resolution representations p.154", 155, "link", "E030", "高斯/拉普拉斯金字塔"),
    ("小波", "E031", "§3.5.4 Wavelets p.159", 159, "link", "E031", "小波阈值去噪与融合演示"),
    ("几何变换", "E032", "§3.6 Geometric transformations p.168", 168, "link", "E032", "射影变换演示"),
]

EXC = {"E028": "p.101=PSNR 定义跨页（§2.3 末，章界例外预登记，E035 不引用该条）"}


def cited_pages(slug_dir: pathlib.Path) -> set[int]:
    s = json.loads((slug_dir / "production" / "shots.json").read_text())
    pages = set()
    for c in s["chunks"]:
        for t in (c["narration"], *c.get("refs", [])):
            pages.update(int(m.group(1)) for m in PAGE.finditer(t))
    return pages


def ep_dir(num: str) -> pathlib.Path:
    return ROOT / "episodes" / f"{num}_{SLUG[num]}"


SLUG = {}
for d in (ROOT / "episodes").glob("E0*"):
    m = re.match(r"(E\d+)_(.+)", d.name)
    if m:
        SLUG[m.group(1)] = m.group(2)

lines = []
errors = []
n_link = n_own = n_run = 0


def emit(s=""):
    lines.append(s)


emit("=" * 78)
emit("E035 · 一枚镜头一张图：成像与图像处理全景串联 —— S2 串联台账（程序生成）")
emit("=" * 78)
emit("回链口径：status.scope.citation_policy——页码锚回链各集 shots.json refs/narration")
emit("（E013–E020 老集无页码引用，锚以 E035 自有证据 e035_book_ch2_ch3_anchors.txt 承担）")
emit("注：回链页=该集实引页，可非节首页（金字塔实引 p.155、节首 p.154；锚列以节首页计）")
emit("")

emit("── 一枚镜头（第 2 章 Image formation，印刷 p.33 起）──")
for name, eps, anchor, link_p, src, echo_ep, echo in ROWS[:6]:
    emit(f"  {name}（{eps}）｜ {anchor} ｜ 回链页={link_p or '—（own 锚）'} "
         f"｜ 回响：{echo}（{echo_ep}）")

emit("── 一张图（第 3 章 Image processing，印刷 p.107 起）──")
for name, eps, anchor, link_p, src, echo_ep, echo in ROWS[6:]:
    emit(f"  {name}（{eps}）｜ {anchor} ｜ 回链页={link_p or '—（own 锚）'} "
         f"｜ 回响：{echo}（{echo_ep}）")
emit("")

emit("── 断言 ──")
# A1 页码锚：own 行锚在 E035 自有证据；link 行回链页在该集（任一）引用集合
for name, eps, anchor, link_p, src, echo_ep, echo in ROWS:
    if src == "link":
        assert link_p, f"{name} link 行缺回链页"
        if not any(link_p in cited_pages(ep_dir(one)) for one in eps.split("+")):
            errors.append(f"A1 回链失配: {name} p.{link_p} 不在 {'/'.join(eps.split('+'))} 引用集合")
        n_link += 1
    else:
        m = re.search(r"p\.(\d+)", anchor)
        if f"p.{m.group(1)}" not in ANCHORS:
            errors.append(f"A1 自有锚失配: {name} {anchor} 不在 e035_book_ch2_ch3_anchors.txt")
        n_own += 1
emit(f"  [{'PASS' if not errors else 'FAIL'}] A1 页码锚：回链 {n_link} 行+自有 {n_own} 行"
     f"（E014 行兼有实引 p.43——可见列 14 个数字+3 个横杠「—（own 锚）」）")

# A2 回响实验存档：E021+ 必须有含 PASS 汇总的 run 存档；E013–E020 演示集（老形态，无 run 口径）
run_report = []
SUM_RE = re.compile(r"(汇总：[^\n]+|\d+ 条\s*(?:断言)?\s*PASS[^\n]{0,50})")
for name, eps, anchor, link_p, src, echo_ep, echo in ROWS:
    for one in dict.fromkeys(echo_ep.split("+")):
        d = ep_dir(one)
        n_num = int(one[1:])
        outs = sorted((d / "code" / "outputs").glob("*.txt")) if (d / "code" / "outputs").exists() else []
        best = None
        for r in outs:
            t = r.read_text()
            if "PASS" in t:
                mm = SUM_RE.search(t)
                if mm:
                    best = (r.name, mm.group(1)[:64])
                    break
        if best:
            run_report.append((one, best[0], best[1]))
            n_run += 1
        elif n_num >= 21:
            run_report.append((one, "-", "无含 PASS 汇总的存档"))
            errors.append(f"A2 回响存档缺: {one} 无可用 run 汇总")
        else:
            run_report.append((one, "—", "演示集（老形态，无数值 run 存档口径）"))
emit(f"  [{'PASS' if not any(e.startswith('A2') for e in errors) else 'FAIL'}] A2 回响存档：{n_run} 集含 PASS 汇总行（E021+ 口径）")
for one, fn, summ in run_report:
    emit(f"      {one}: {fn} —— {summ}")

# A3 章界（回链页码落章内区间；E028 p.101 例外预登记）
viol = []
for d in sorted((ROOT / "episodes").glob("E0*")):
    num = d.name.split("_")[0]
    n = int(num[1:])
    if not 13 <= n <= 34:
        continue
    pages = cited_pages(d)
    lo, hi = (33, 106) if n <= 20 else (107, 190)
    out_rng = [p for p in pages if not lo <= p <= hi and p != 101]
    if out_rng:
        viol.append((num, sorted(out_rng)))
emit(f"  [{'PASS' if not viol else 'FAIL'}] A3 章界：Ch2 集⊆[33,106]、Ch3 集⊆[107,190]"
     f"（例外：E028 {EXC['E028']}）")
for num, rng in viol:
    emit(f"      ⚠ {num} 出界页: {rng}")
    errors.append(f"A3 章界: {num} 出界 {rng}")

emit("")
emit(f"台账行数：{len(ROWS)}（成像链 6+处理链 11）；断言错误 {len(errors)} 项")
if errors:
    for e in errors:
        emit(f"  [ERROR] {e}")
    emit("LEDGER-FAIL")
    OUT.write_text("\n".join(lines) + "\n")
    sys.exit(1)
emit("LEDGER-DONE")
OUT.write_text("\n".join(lines) + "\n")
print(f"[ok] {OUT}（{len(lines)} 行）")
