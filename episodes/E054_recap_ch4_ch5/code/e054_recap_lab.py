#!/usr/bin/env python3
"""E054 复盘课回响核对 lab（S2 集 lab-stdout 机检要求 lab.txt+readings.json 齐备——
E043 r4 P1-1 起工具级要求；E035 复盘课当时无机检、本集按现行工具补齐）。
复盘课无新实验——本档定位=「回响账本复核」，四断言全部程序计算、数值叶值全落
readings.json（非空壳申报）：
  R1 台账 A2 主数合计同口径二次实现重算（不 import 台账脚本、重写扫描与正则，
     对照台账输出行——E035 三百五十八条判例同款口径，本集 293）
  R2 台账行数复核 18=6+12（模具链六站+学习链十二站——台账行文直数）
  R3 口播页码 chip⊆台账回链页∪ToC 章界页（口播回链不脱离台账——红线 1 自检）
  R4 口播结构完整性复核 15 帧/9 页/每帧口播与 refs 非空/总字符合理带
（学生版改版 2026-11-05：R3 页码提取兼容「书上第 N 页」——学生版口播措辞，
 chip 集合仍须 ⊆ 台账回链页∪章首页且恰 5 个；R4 总字符带按新口播重设为
 [2250,2500]（改版前 [1800,2100] 对应 7.67 分钟口径），帧/页数仍 15/9。）
红线 5：lab 输出系内部存证非 deck 可见层（不受 deck 记号约束）。
用法：python3 tools/evidence_run.py episodes/E054_recap_ch4_ch5/code/outputs/
e054_recap_lab.txt python3 episodes/E054_recap_ch4_ch5/code/e054_recap_lab.py
（stderr 另行重定向同款口径——evidence_run 原子落盘）。"""

import hashlib
import json
import pathlib
import re
from datetime import datetime

EP = pathlib.Path(__file__).resolve().parent.parent          # episodes/E054_...
ROOT = EP.parent.parent                                       # 课程根
OUT = EP / "code" / "outputs"
SHOTS = json.loads((EP / "production" / "shots.json").read_text())
LEDGER = (OUT / "e054_recap_ledger.txt").read_text()

lines = []
emit = lines.append
results = []            # (名称, ok, 结果段)
now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def leaf_vals(obj):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(leaf_vals(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(leaf_vals(v))
    elif isinstance(obj, bool):
        out.append(obj)
    elif isinstance(obj, (int, float)):
        out.append(obj)
    return out


# ── R1 主数独立重算（同口径二次实现：排除集与求和语义与台账 A2 相同——
#      独立性申报=不 import 台账脚本、重写正则与扫描，对照台账输出行）──
eps_order = re.findall(r"^  .+?（(E\d{3})）｜", LEDGER, re.M)
assert len(eps_order) == 18, f"台账集名提取异常：{len(eps_order)}"
SUM_RE = re.compile(r"(汇总：[^\n]+|\d+\s*条[^\n]{0,10}(?:全部\s*)?PASS[^\n]{0,40}|\d+\s*条，\s*PASS\s*\d+)")
CNT_RE = re.compile(r"(\d+)\s*条")
per_ep = {}
for e in eps_order:
    dirs = list(ROOT.glob(f"episodes/{e}_*"))
    assert len(dirs) == 1, f"{e} 目录不唯一：{dirs}"
    outs = sorted((dirs[0] / "code" / "outputs").glob("*.txt")) if (dirs[0] / "code" / "outputs").exists() else []
    n = 0
    for r in outs:
        if "regression" in r.name or "display" in r.name:
            continue
        t = r.read_text(errors="replace")
        if "PASS" not in t:
            continue
        mm = SUM_RE.search(t)
        if mm:
            c = CNT_RE.search(mm.group(0))
            if c:
                n = int(c.group(1))
                break
    per_ep[e] = n
total_recalc = sum(per_ep.values())
m_total = re.search(r"主数合计=(\d+)", LEDGER)
total_ledger = int(m_total.group(1)) if m_total else -1
r1_ok = total_recalc == total_ledger == 293
results.append(("R1 台账 A2 主数独立重算==台账输出行（独立实现路径逐集扫描重加总）", r1_ok,
                f"复算 {total_recalc}（{len(per_ep)} 集）对照台账 {total_ledger}"))
readings = dict(generated_at=now, src_sha16=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()[:16],
                r1=dict(total=total_recalc, ledger=total_ledger, eps=len(per_ep)),
                per_ep={k: v for k, v in per_ep.items() if v})

# ── R2 台账行数复核 ──
row_lines = re.findall(r"^  .+?（E\d{3}）｜ ", LEDGER, re.M)
mold = len(re.findall(r"^  .+?（E\d{3}）｜ ", LEDGER[:LEDGER.find("── 一整套学习")], re.M))
learn = len(row_lines) - mold
r2_ok = len(row_lines) == 18 and mold == 6 and learn == 12
results.append(("R2 台账行数==18（模具链六站+学习链十二站）", r2_ok,
                f"{len(row_lines)} 行 = 模具链 {mold}+学习链 {learn}"))
readings["r2"] = dict(rows=len(row_lines), mold=mold, learn=learn)

# ── R3 口播页码 chip ⊆ 台账回链页 ∪ ToC 章界页 ──
chips = set()
for c in SHOTS["chunks"]:
    chips |= {int(x) for x in re.findall(r"(?:原书|书上)第 (\d+) 页",
                                   c["narration"])}
link_pages = {int(x) for x in re.findall(r"回链页=(\d+)", LEDGER)}
TOC_PAGES = {191, 235}      # 章头页（e054_book_toc_p4.txt 自有证据——非 ROWS 回链）
in_ok = sorted(p for p in chips if p in link_pages or p in TOC_PAGES)
r3_ok = len(in_ok) == len(chips) and len(chips) == 5
results.append(("R3 口播页码 chip 全数⊆台账回链页∪ToC 章界页（口播回链不脱离台账）", r3_ok,
                f"{len(in_ok)}/{len(chips)} 命中：{'/'.join(str(p) for p in sorted(chips))}"
                f"（191/235 系章头页、其余系 ROWS 回链）"))
readings["r3"] = dict(n_chips=len(chips), n_hit=len(in_ok), chips=sorted(chips))

# ── R4 口播结构完整性复核（15 帧/9 页/每帧口播与 refs 非空/总字符合理带——
#      时长申报=规划 9 分钟随 mech 定案，本档只核结构一致性）──
frames = len(SHOTS["chunks"])
slides = len({c["slide"] for c in SHOTS["chunks"]})
chars = sum(len(c["narration"]) for c in SHOTS["chunks"])
nonempty = all(c["narration"].strip() and c.get("refs") for c in SHOTS["chunks"])
r4_ok = frames == 15 and slides == 9 and nonempty and 2250 <= chars <= 2500
results.append(("R4 口播结构完整性：15 帧 9 页全非空+总字符合理带", r4_ok,
                f"{frames} 帧 {slides} 页 总字符 {chars}（带 [2250,2500]，学生版口径）"))
readings["r4"] = dict(frames=frames, slides=slides, chars=chars, lo=2250, hi=2500)

# ── readings 落盘+输出组装 ──
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
readings["n_pass"] = n_pass
readings["n_total"] = len(results)
rd_path = OUT / "e054_recap_readings.json"
rd_path.write_text(json.dumps(readings, ensure_ascii=False, indent=1) + "\n")
rd_sha = hashlib.sha256(rd_path.read_bytes()).hexdigest()
src_sha = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

emit(f"[lab] E054 复盘课回响核对档 开始 {now}")
emit(f"[lab] 源码 sha256={src_sha}（哨兵版本绑定——E043 r5 P2-5 口径）")
for name, ok, res in results:
    emit(f"[{'PASS' if ok else 'FAIL'}] {name}　| {res}")
fig_dir = OUT / "figures"
for png in sorted(fig_dir.glob("*.png")):
    emit(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
emit("[lab] readings → e054_recap_readings.json")
emit(f"[lab] readings sha256={rd_sha}")
emit(f"[lab] LEDGER：断言 {len(results)} 条，PASS {n_pass}")
emit(f"[lab] 汇总 PASS {n_pass} / FAIL {n_fail} / 共 {len(results)}")
emit(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

print("\n".join(lines))
if n_fail:
    raise SystemExit(1)
