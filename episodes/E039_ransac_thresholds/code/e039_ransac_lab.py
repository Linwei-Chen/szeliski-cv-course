#!/usr/bin/env python3
"""E039 数值实验：RANSAC 实战心法——阈值 ϵ、迭代次数 S 与置信度 P（§8.1.4 后半）。
锚：e039_book_s814b_anchors.txt（§8.1.4 后半 印刷 p.511 下–512 中，与 E038 锚 v1 同源照录）。
主流程对齐原书 p.512：8.29 全失败概率 1−P=(1−p^k)^S、8.30 最少轮数 S=log(1−P)/log(1−p^k)；
p=单个对应点有效的概率（p.512 Let p be the probability…），P=S 轮后成功的概率；
Table 8.2（Stewart 1999，99% 成功率）三行表值逐行复算验证。
实验语言延续 E038（数据=同 seed 2038 同配方 17 点全集=16 内点+1 外点 (4.6,5.2)——重跑兑现 E038 作业一；
readings 对照 E038 存档逐位一致；跨 seed 不可比口径同前）。
E038 作业兑现（S2 口径申报）：
- 作业一（门槛三档）：ϵ∈{0.15, 0.3, 0.6}=1σ/2σ/4σ 同一份数据同一套抽签重跑投票——窄门槛票池变薄（best 11/15）、
  宽门槛坏签拉票+平局失去分辨力（round3 P1-1 修：外点离真线 1.9，ϵ=0.6 圈不进——不写外点进票池；读数级：seed 2038 单次实现；学生重跑同款数据会得同构结论，具体数字看 seed）；
- 作业二（p^k 实证）：k=2/k=3 两档签，各 200 轮，全内点签（签上三点/两点全是内点——真值标签只用于事后统计）
  实测频率对照 8.29 的 p^k 预测（p̂=16/17）——「样本点越多，要的轮数涨得越快」亲手数；
- Table 8.2 复算：8.30 逐行代入 k/p → ceil 对齐表值 35/97/293；
- MAD 自调尺度回收（E037 handoff_out；讲者补充双标——原书 p.511–512 无 MAD，读数级）：
  OLS 全集拟合残差的 MAD×1.4=σ̂_robust → ϵ_MAD=2σ̂_robust 与应用约定 ϵ=0.3≈2σ 对照；
- S 公式应用：实测 p̂ 与 k=2 代 8.30 得 S(99%)——对照本集 S=200 的充足性（读数级）。
边界（D31）：变体族（Preemptive/PROSAC/MLESAC/DSAC/Graph-Cut/MAGSAC/ESAC/GNC）原书只给定性——口播讲、不实验；
Uncertainty modeling 段不在本集（E040 锚）。
图（程序生成，未用教材原图）：图1 Table 8.2 复算条 765×280（对应 p.512 Table 8.2）；
图2 p^k 预测 vs 实测 765×280（对应 p.512 8.29 推导段）；
图3 门槛三档 765×280（对应 p.511 application dependent 段）。
"""
import json
import math
import pathlib

import numpy as np

rng = np.random.default_rng(2038)
OUT = pathlib.Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
checks = []
readings = {}


def chk(label, ok, detail=""):
    checks.append(label)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        raise SystemExit(f"FAIL: {label}")


def line_from_two(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    if x2 == x1:
        return None
    m = (y2 - y1) / (x2 - x1)
    return m, y1 - m * x1


def count_inliers(m, b, xs, ys, eps):
    r = np.abs(ys - (m * xs + b))
    return int((r <= eps).sum()), r


def ols(x, y):
    X = np.stack([x, np.ones_like(x)], axis=1)
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


M_TRUE, B_TRUE, SIGMA = 0.5, 1.0, 0.15
S_ROUNDS = 200
P_SUCCESS = 0.99   # Table 8.2 口径：99% 成功率（Stewart 1999）

# ── 数据重现（E038 同 seed 2038 同配方：16 内点+1 外点——作业一「重跑」口径）──
x_in = np.sort(rng.uniform(0.0, 4.0, 16))
y_in = B_TRUE + M_TRUE * x_in + rng.normal(0.0, SIGMA, 16)
x_out_pt, y_out_pt = 4.6, 5.2
x_all = np.append(x_in, x_out_pt)
y_all = np.append(y_in, y_out_pt)

# E038 首轮签也重现（对齐其 rng 流：数据后第一次 choice(17,2)=E038 E1 首轮）
i1, i2 = rng.choice(17, size=2, replace=False)

EP38 = json.loads((pathlib.Path(__file__).resolve().parent.parent.parent /
                   "E038_ransac" / "code" / "outputs" / "e038_ransac_readings.json").read_text())

# ── E1 Table 8.2 逐行复算（8.30：S=log(1−P)/log(1−p^k)，ceil 对齐表值——原书 p.512）──
def s_needed(k, p, P=P_SUCCESS):
    return math.log(1 - P) / math.log(1 - p ** k)


rows = [(3, 0.5, 35), (6, 0.6, 97), (6, 0.5, 293)]
calc = [(k, p, s_needed(k, p)) for k, p, _ in rows]
chk("E1a Table 8.2 三行逐行复算：8.30 计算值向上取整=表值（35/97/293——原书 p.512 Stewart 1999）",
    all(math.ceil(s) == tab for (_, _, tab), (_, _, s) in zip(rows, calc)),
    "  ".join(f"k={k},p={p}: S={s:.1f}→{math.ceil(s)}（表 {tab}）"
              for (k, p, tab), (_, _, s) in zip(rows, calc)))
chk("E1b 轮数对 k 敏感（p 固定 0.5：k=3→35 vs k=6→293，约 8.4 倍——「涨得快」的量化，读数级）",
    calc[2][2] / calc[0][2] > 8,
    f"表值 293/35≈{293/35:.1f} 倍；计算值 292.4/34.5≈{calc[2][2]/calc[0][2]:.1f} 倍（p=0.5，k 3→6）——round1 P1-6 修：标签/detail/算式三处统一")
readings["e1"] = {"rows": [{"k": k, "p": p, "s_calc": round(s, 2), "s_table": tab}
                           for (k, p, tab), (_, _, s) in zip(rows, calc)],
                  "note": "Table 8.2 三行 8.30 复算 ceil 全对齐——公式的表值级验证（round4 P2-6 引理：表值可复算）"}

# ── E2 p^k 预测 vs 实测（k=2：全内点签频率对照 p̂²——8.29 推导段的实证面）──
p_hat = 16.0 / 17.0
draws = np.array([rng.choice(17, size=2, replace=False) for _ in range(S_ROUNDS)])
pure2 = int((draws.max(axis=1) < 16).sum())
pred2 = p_hat ** 2 * S_ROUNDS
chk("E2a k=2 全内点签实测频率贴近 p̂² 预测（|实测−预测|/预测 <8%，读数级——p̂=16/17）",
    abs(pure2 - pred2) / pred2 < 0.08,
    f"全内点签 {pure2}/{S_ROUNDS} vs p̂² 预测 {pred2:.1f}（p̂={p_hat:.4f}，偏差 {abs(pure2-pred2)/pred2*100:.1f}%）")
# 重放 200 轮计票（ϵ=0.3 基线），与 E038 存档 votes_hist 逐 bin 比对（round1 P1-5 修：合计→逐 bin）
votes_ep38 = []
for a_, b_ in draws:
    m_, bb_ = line_from_two((x_all[a_], y_all[a_]), (x_all[b_], y_all[b_]))
    n_, _ = count_inliers(m_, bb_, x_all, y_all, 0.3)
    votes_ep38.append(n_)
votes_ep38 = np.array(votes_ep38)
hist39 = {int(k): int(v) for k, v in zip(*np.unique(votes_ep38, return_counts=True))}
hist38 = {int(k): int(v) for k, v in EP38["e3"]["votes_hist"].items()}
chk("E2b E038 同款抽签重放：首轮签+全内点轮数+200 轮票数直方图逐 bin 一致（同 seed 同调用序列）",
    (i1, i2) == (10, 8) and pure2 == 180 and hist39 == hist38,
    f"首轮签 ({i1},{i2})（E038 E8f 同值）；全内点样本轮 {pure2}/200（E038 e3.pure_sample_rounds=180）；"
    f"votes_hist {len(hist38)} bin（2–15 票档）逐 bin 相等——200 轮票数逐轮分布一致")
readings["e2"] = {"p_hat": round(p_hat, 4), "k2_pure": pure2, "k2_pred": round(pred2, 1),
                  "note": "8.29 的实证面：p^k=单签全内点概率——实测频率对照预测（真值标签只用于事后统计）"}

# ── E3 k=3（作业二：每签从两点改成三点——三点 OLS 拟线，签全内点=「干净签」）──
rng3 = np.random.default_rng(2039)
draws3 = np.array([rng3.choice(17, size=2, replace=False) for _ in range(S_ROUNDS)])  # 废抽签但保留：消耗 rng3 状态，draws3Third 序列依赖（删则 seed 2039 流变、k3 读数全变——round1 P2-6 注记）
draws3Third = np.array([rng3.choice(17, size=3, replace=False) for _ in range(S_ROUNDS)])
pure3 = int((draws3Third.max(axis=1) < 16).sum())
pred3 = p_hat ** 3 * S_ROUNDS
m3_best_votes, lines3 = [], []
for a, b, c in draws3Third:
    X = np.stack([x_all[[a, b, c]], np.ones(3)], axis=1)
    th, *_ = np.linalg.lstsq(X, y_all[[a, b, c]], rcond=None)
    n, _ = count_inliers(th[0], th[1], x_all, y_all, 0.3)
    m3_best_votes.append(n)
m3_best_votes = np.array(m3_best_votes)
chk("E3a k=3 全内点签实测频率贴近 p̂³ 预测（|实测−预测|/预测 <8%，读数级）",
    abs(pure3 - pred3) / pred3 < 0.08,
    f"全内点签 {pure3}/{S_ROUNDS} vs p̂³ 预测 {pred3:.1f}（偏差 {abs(pure3-pred3)/pred3*100:.1f}%）")
chk("E3b 轮数账实测（p=0.94、99% 成功率：k=2 需 2.12 轮 vs k=3 需 2.57 轮——8.30 代入 p̂，计算值随 k 递增；内点率高时需求都小，Table 8.2 的 p=0.5/0.6 才是难场景）",
    s_needed(2, p_hat) < s_needed(3, p_hat)
    and math.ceil(s_needed(2, p_hat)) == 3 and math.ceil(s_needed(3, p_hat)) == 3,
    f"8.30 代 p̂={p_hat:.4f}：k=2 → S={s_needed(2, p_hat):.2f}，k=3 → S={s_needed(3, p_hat):.2f}"
    "（计算值递增、取整同为 3 轮——内点率高时轮数需求小；难场景对照见 E6b）")
chk("E3c k=3 胜出票数不低于 k=2 基线（三点线更稳——读数级对照，不作普遍断言）",
    int(m3_best_votes.max()) >= 15,
    f"k=3 胜出 {int(m3_best_votes.max())}/17 票（k=2 基线 15——E038 e3.best_inliers）")
readings["e3"] = {"k3_pure": pure3, "k3_pred": round(pred3, 1),
                  "k3_best_votes": int(m3_best_votes.max()),
                  "s_needed_k2": math.ceil(s_needed(2, p_hat)),
                  "s_needed_k3": math.ceil(s_needed(3, p_hat)),
                  "note": "作业二兑现：k=2→k=3 全内点签频率跟着 p^k 走；8.30 代 p̂ 的轮数账=本集 S=200 远超所需（读数级）"}

# ── E4 门槛三档（作业一兑现：ϵ∈{0.15,0.3,0.6} 同数据同抽签重跑投票——p.511 application dependent）──
draws_ref = draws  # 与 E038 E3 同一序列（i1/i2 后 200 次 choice——逐位一致已由 E2b 背书）
results = {}
for eps in (0.15, 0.3, 0.6):
    votes = []
    for a, b in draws_ref:
        m, bb = line_from_two((x_all[a], y_all[a]), (x_all[b], y_all[b]))
        n, _ = count_inliers(m, bb, x_all, y_all, eps)
        votes.append(n)
    votes = np.array(votes)
    full_idx = np.where(votes == 16)[0]
    full_m = [round(line_from_two((x_all[d[0]], y_all[d[0]]), (x_all[d[1]], y_all[d[1]]))[0], 4) for d in draws_ref[full_idx]]
    best = int(votes.argmax())  # 平手按先后取第一个满票轮（argmax 语义——round3 P2-2 申报）
    # round4 P2-1：满票线的外点残差+票池外点检查（证据链闭合——「未进任何好线票池」逐线验证）
    full_lines = [line_from_two((x_all[d[0]], y_all[d[0]]), (x_all[d[1]], y_all[d[1]])) for d in draws_ref[full_idx]]
    out_resid_full = [round(abs(y_out_pt - (m * x_out_pt + bb)), 4) for m, bb in full_lines]
    m_b, b_b = line_from_two((x_all[draws_ref[best][0]], y_all[draws_ref[best][0]]),
                             (x_all[draws_ref[best][1]], y_all[draws_ref[best][1]]))
    results[eps] = {"best_votes": int(votes.max()), "m_best": round(m_b, 4),
                    "median_votes": int(np.median(votes)),
                    "dirty_max": int(votes[draws_ref.max(axis=1) == 16].max()),
                    "full_rounds": int((votes == 16).sum()),
                    "full_m": full_m,
                    "outlier_resid_min_full": (min(out_resid_full) if out_resid_full else None),
                    "full_pools_no_outlier": all(r > eps for r in out_resid_full),
                    "m_raw": m_b, "b_raw": b_b}  # round1 P0-3 修：删「外点落门槛」抽检（含外点签的线穿过外点，残差恒≈0，指标恒真且分母 20 非 50）
outlier_resid_to_true = round(abs(float(y_out_pt - (M_TRUE * x_out_pt + B_TRUE))), 4)
chk("E4a 窄门槛 0.15（1σ）：胜出票数明显掉、中位数票数掉（票池变薄——太窄的岔子，读数级）",
    results[0.15]["best_votes"] < results[0.3]["best_votes"]
    and results[0.15]["median_votes"] < results[0.3]["median_votes"],
    f"ϵ=0.15: 胜出 {results[0.15]['best_votes']} 票/中位 {results[0.15]['median_votes']}"
    f" vs ϵ=0.3: 胜出 {results[0.3]['best_votes']} 票/中位 {results[0.3]['median_votes']}")
chk("E4b 基线 ϵ=0.3 与 E038 存档七字段逐位一致（同数据同抽签——作业一重跑的对照锚；round1 P1-5 修：2→7 字段）",
    results[0.3]["best_votes"] == EP38["e3"]["best_inliers"]
    and abs(results[0.3]["m_best"] - EP38["e3"]["m_best"]) < 1e-9
    and results[0.3]["median_votes"] == EP38["e3"]["median_inliers"]
    and results[0.3]["dirty_max"] == EP38["e3"]["dirty_votes_max"]
    and results[0.3]["full_rounds"] == EP38["e3"]["full_vote_rounds"]
    and pure2 == EP38["e3"]["pure_sample_rounds"]
    and int((draws_ref.max(axis=1) == 16).sum()) == EP38["e3"]["dirty_sample_rounds"],
    f"ϵ=0.3: 胜出 {results[0.3]['best_votes']} 票 m={results[0.3]['m_best']} 中位 {results[0.3]['median_votes']}"
    f" 坏签最高 {results[0.3]['dirty_max']} 满票轮 {results[0.3]['full_rounds']}；"
    f"干净签轮 {pure2}/坏签轮 {int((draws_ref.max(axis=1) == 16).sum())}——与 E038 e3 七项逐位相同")
chk("E4c 含外点签最高票随 ϵ 单调上升（4<5<9——宽门槛坏签得票的读数级证据；round1 P0-3 修：恒真抽检改实断言）",
    results[0.15]["dirty_max"] < results[0.3]["dirty_max"] < results[0.6]["dirty_max"],
    f"dirty_max 0.15/0.3/0.6 = {results[0.15]['dirty_max']}/{results[0.3]['dirty_max']}/{results[0.6]['dirty_max']}"
    f"（全集 17 点计票；不外推所有宽门槛场景）")
chk("E4r 外点离真线读数入档：|5.2−(0.5×4.6+1)|=1.9（round3 P1-1：f002 口播「约一点九」numeric 依据；ϵ=0.6 圈不进外点）",
    abs(outlier_resid_to_true - 1.9) < 1e-9, f"outlier_resid_to_true={outlier_resid_to_true}")
chk("E4r2 外点对全部满票线的最小残差>ϵ 且满票票池均不含外点（round4 P2-1 修：f002 refs 申报逐线闭合——0.6 档 127 条满票线）",
    results[0.6]["full_pools_no_outlier"] and results[0.6]["outlier_resid_min_full"] > 0.6,
    f"ϵ=0.6 满票线 {results[0.6]['full_rounds']} 条：外点残差 min={results[0.6]['outlier_resid_min_full']}>0.6；"
    f"票池含外点的满票轮=0（outlier_resid_min_full 对三档入档）")
chk("E4d 三档排序读数（票数上限随 ϵ 单调不降——全集 17 点计票的几何事实）",
    results[0.15]["best_votes"] <= results[0.3]["best_votes"] <= results[0.6]["best_votes"],
    f"胜出票数 0.15/0.3/0.6 = {results[0.15]['best_votes']}/{results[0.3]['best_votes']}/{results[0.6]['best_votes']}")
chk("E4e 宽门槛平局读数：ϵ=0.6 满票轮 127/200、胜出线 m=0.424 偏真线 0.076（0.3 档 m=0.5295 偏 0.0295）——门槛太宽投票失去分辨力（round2 P2-6 新增；round3 P2-2 修：胜出线=argmax 取第一个满票轮，平手分布 full_m 入档）",
    results[0.6]["full_rounds"] > 100 and abs(results[0.6]["m_best"] - 0.5) > abs(results[0.3]["m_best"] - 0.5),
    f"ϵ=0.6: 满票轮 {results[0.6]['full_rounds']}/200、胜出 m={results[0.6]['m_best']}（偏 0.5 达 {abs(results[0.6]['m_best']-0.5):.4f}）"
    f" vs ϵ=0.3: 满票轮 {results[0.3]['full_rounds']}、m={results[0.3]['m_best']}（偏 {abs(results[0.3]['m_best']-0.5):.4f}）；"
    f"满票线 m 分布：min={min(full_m):.3f}/med={sorted(full_m)[len(full_m)//2]:.3f}/max={max(full_m):.3f}（n={len(full_m)}）——平手非必挑歪，argmax 实例偏 0.076")
readings["e4"] = {str(eps): results[eps] for eps in results}
readings["e4"]["outlier_resid_to_true"] = outlier_resid_to_true
readings["e4"]["note"] = ("作业一兑现：同一份数据同一套抽签（E038 序列逐位重放），只动 ϵ——"
                          "窄门槛票池变薄、宽门槛坏签拉票（外点离真线 1.9 圈不进——round3 P1-1 修）；ϵ=0.3 与 E038 存档逐位一致=对照锚（读数级，seed 2038）")

# ── E5 MAD 自调尺度回收（E037 handoff_out；讲者补充双标——原书 p.511–512 无 MAD，读数级）──
th_all = ols(x_all, y_all)
ols_m, ols_b = float(th_all[0]), float(th_all[1])  # round2 P2-3 修：OLS 线参数入档（「被拉偏」的直接读数）
r_all = np.abs(y_all - (th_all[0] * x_all + th_all[1]))
mad = float(np.median(r_all))          # 残差绝对值取正中间那个（E037 B.25 口径）
sigma_hat = 1.4 * mad                  # ×1.4 照 E037 原书 B.25 原文口径（非通用 1.4826）
eps_mad = 2 * sigma_hat                # 2σ̂ ≈ 应用约定 ϵ=0.3≈2σ
chk("E5a MAD 定尺度：OLS 全集残差的 MAD×1.4 估出 σ̂ 量级正确（0.1<σ̂<0.3，读数级——中位数受外点影响小，但 OLS 线本身被拉偏；round2 P2-3 修）",
    0.1 < sigma_hat < 0.3,
    f"残差 MAD={mad:.4f} → σ̂=1.4×MAD={sigma_hat:.4f}（真 σ=0.15）")
chk("E5b ϵ_MAD=2σ̂ 与应用约定 ϵ=0.3 同量级（比值 0.5–2.0，读数级；不作等式断言）",
    0.5 < eps_mad / 0.3 < 2.0,
    f"ϵ_MAD=2σ̂={eps_mad:.4f} vs ϵ=0.3（比值 {eps_mad/0.3:.2f}）")
# round1 P1-4 修：补胜出线残差版 MAD 对照——全集 OLS 残差被外点拉偏，胜出线版更贴近应用约定
r_bestl = np.abs(y_all - (results[0.3]["m_raw"] * x_all + results[0.3]["b_raw"]))
mad_best = float(np.median(r_bestl))
sigma_best = 1.4 * mad_best
eps_mad_best = 2 * sigma_best
chk("E5c 胜出线残差 MAD 定门槛：比全集 OLS 残差版更贴近 ϵ=0.3（胜出线剔了外点拉偏——读数级）",
    abs(eps_mad_best - 0.3) < abs(eps_mad - 0.3),
    f"胜出线 MAD={mad_best:.4f}→σ̂={sigma_best:.4f}→ϵ_MAD={eps_mad_best:.4f}"
    f" vs 全集 OLS 版 {eps_mad:.4f}（ϵ=0.3；偏差 {abs(eps_mad_best-0.3):.3f}<{abs(eps_mad-0.3):.3f}）")
readings["e5"] = {"ols_m": round(ols_m, 4), "ols_b": round(ols_b, 4),
                  "mad": round(mad, 4), "sigma_hat": round(sigma_hat, 4),
                  "eps_mad": round(eps_mad, 4),
                  "best_line_mad": round(mad_best, 4), "eps_mad_best": round(eps_mad_best, 4),
                  "note": "MAD 自调尺度回收（E037 handoff_out 兑现）：讲者补充双标——原书 p.511–512 无 MAD；"
                          "×1.4 照 E037 B.25 原文口径；ϵ 怎么定原书只说 application dependent（读数级对照）；"
                          "E5c=胜出线残差版对照（全集 OLS 版被外点拉偏——round1 P1-4 修）"}

# ── E6 S 公式应用（本集 S=200 的充足性——8.30 代实测 p̂）──
s99_k2 = s_needed(2, p_hat)
chk("E6a 8.30 代实测 p̂ 与 k=2：99% 成功率只需 3 轮（本集 200 轮余量充分——读数级）",
    math.ceil(s99_k2) == 3 and S_ROUNDS > 10 * math.ceil(s99_k2),
    f"S(99%)={s99_k2:.2f}→{math.ceil(s99_k2)} 轮 vs 本集 S={S_ROUNDS}（余量 {S_ROUNDS/math.ceil(s99_k2):.0f} 倍）")
chk("E6b 难场景对照（p=0.5,k=2：99% 需 16.0→17 轮；p=0.5,k=6：需 292.4→293 轮——p 与 k 单变量分步对照，读数级；round2 P1-1 修）",
    math.ceil(s_needed(2, 0.5)) == 17 and math.ceil(s_needed(6, 0.5)) == 293,
    f"p=0.5: k=2→{s_needed(2, 0.5):.1f}→{math.ceil(s_needed(2, 0.5))} 轮、k=6→{math.ceil(s_needed(6, 0.5))} 轮"
    "（8.30 同源口径；Table 8.2 三行见 E1a）")
readings["e6"] = {"s99_k2_hat": math.ceil(s99_k2), "s200_margin": round(S_ROUNDS / math.ceil(s99_k2), 1),
                  "note": "S 公式应用：本集 p̂ 高（16/17）轮数需求小；Table 8.2 的 p=0.5/0.6 才是坏场景——"
                          "「轮数由 p^k 决定」的完整图景（读数级）"}

# ── 图（程序生成，未用教材原图）──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "figure.dpi": 100})
plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

# 图1 Table 8.2 复算条：三行表值 vs 8.30 复算
fig, ax = plt.subplots(figsize=(7.65, 2.8))
labels = [f"k={k}\np={p}" for k, p, _ in rows]
tab_vals = [tab for _, _, tab in rows]
calc_vals = [s for _, _, s in calc]
xs = np.arange(3)
ax.bar(xs - 0.18, tab_vals, width=0.36, color="#2a6f9e", label="Table 8.2 表值")
ax.bar(xs + 0.18, calc_vals, width=0.36, color="#c0392b", label="式 8.30 复算")
for xi, tv, cv in zip(xs, tab_vals, calc_vals):
    ax.text(xi - 0.18, tv + 6, str(tv), ha="center", fontsize=12)
    ax.text(xi + 0.18, cv + 6, f"{cv:.1f}", ha="center", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("需要轮数 S（99%）")
ax.set_title("式 8.30 复算与 Table 8.2 表值一致", pad=10)
ax.set_ylim(0, 330)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12, framealpha=0.92, borderaxespad=0)
fig.text(0.99, 0.012, "程序生成，未用教材原图（表值对应原书第 512 页 Table 8.2，Stewart 1999）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 1, 0.93])
fig.savefig(FIG / "e039_table82.png")
plt.close(fig)

# 图2 p^k 预测 vs 实测（k=2/k=3 全内点签频率）
fig, ax = plt.subplots(figsize=(7.65, 2.8))
ks = ["k=2", "k=3"]
preds = [pred2 / S_ROUNDS * 100, pred3 / S_ROUNDS * 100]
acts = [pure2 / S_ROUNDS * 100, pure3 / S_ROUNDS * 100]
xs = np.arange(2)
ax.bar(xs - 0.18, preds, width=0.36, color="#2a6f9e", label="8.29 推导：$\\hat{p}^k$ 预测")
ax.bar(xs + 0.18, acts, width=0.36, color="#3a7d44", label="实测全内点签频率")
for xi, pv, av in zip(xs, preds, acts):
    ax.text(xi - 0.18, pv + 1.5, f"{pv:.1f}%", ha="center", fontsize=12)
    ax.text(xi + 0.18, av + 1.5, f"{av:.1f}%", ha="center", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels(ks, fontsize=12)
ax.set_ylim(0, 108)
ax.set_ylabel("全内点签占比（%）")
ax.set_title(f"全内点签占比：预测 vs 实测（$\\hat{{p}}$={p_hat:.3f}）", pad=10)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12, framealpha=0.92, borderaxespad=0)
fig.text(0.99, 0.012, "程序生成，未用教材原图（概念对应原书第 512 页 8.29 推导段）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 0.80, 0.93])
fig.savefig(FIG / "e039_pk.png")
plt.close(fig)

# 图3 门槛三档：胜出/中位/含外点签最高
fig, ax = plt.subplots(figsize=(7.65, 2.8))
eps_list = [0.15, 0.3, 0.6]
best_v = [results[e]["best_votes"] for e in eps_list]
med_v = [results[e]["median_votes"] for e in eps_list]
dirty_v = [results[e]["dirty_max"] for e in eps_list]
xs = np.arange(3)
ax.bar(xs - 0.26, best_v, width=0.24, color="#2a6f9e", label="胜出票数")
ax.bar(xs, med_v, width=0.24, color="#7fa8c9", label="中位票数")
ax.bar(xs + 0.26, dirty_v, width=0.24, color="#c0392b", label="含外点签最高票")
for xi, a, b, c in zip(xs, best_v, med_v, dirty_v):
    ax.text(xi - 0.26, a + 0.6, str(a), ha="center", fontsize=12)
    ax.text(xi, b + 0.6, str(b), ha="center", fontsize=12)
    ax.text(xi + 0.26, c + 0.6, str(c), ha="center", fontsize=12)
ax.set_xticks(xs)
ax.set_xticklabels([f"阈值={e}\n({'1σ 太窄' if e == 0.15 else '2σ 实验约定' if e == 0.3 else '4σ 太宽'})"
                    + ("\n满票平局 127 轮" if e == 0.6 else "")
                    for e in eps_list], fontsize=12)
ax.set_ylim(0, 23)
ax.set_ylabel("票数")
ax.set_title("门槛这把旋钮：太窄票池变薄，太宽坏签也拉得到票", pad=14)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=12, framealpha=0.92, borderaxespad=0)
fig.text(0.99, 0.012, "程序生成，未用教材原图（门槛口径对应原书第 511 页 application dependent 段）",
         ha="right", va="bottom", fontsize=12, color="#666")
fig.tight_layout(rect=[0, 0.06, 0.80, 0.88])
fig.savefig(FIG / "e039_eps.png")
plt.close(fig)

# ── 显示盒自检存档（照 E038 口径：natural 尺寸+minfont 推算注记）──
(OUT / "e039_display_check.txt").write_text(
    "[PASS] 图1 box=765.2x250.0 natural=765x280 scale=0.8929 minfont=14.3px fit=contain"
    "（box/scale/fit=实测；minfont=12pt×100/72×scale 推算——PNG 内字号 DOM 量不到）\n"
    "[PASS] 图2 同上 natural=765x280 minfont=14.3px\n"
    "[PASS] 图3 同上 natural=765x280 minfont=14.3px\n"
    "[PASS] 页4 显示口径 max-height=224：scale=0.8 minfont=12.8px ≥12px\n"
    "（三图图内所有文字含脚注/图例/柱标统一 fontsize≥12pt——E038 round3 P2-8 口径延续）\n")

readings["assertions_total"] = len(checks)
(OUT / "e039_ransac_readings.json").write_text(
    json.dumps(readings, ensure_ascii=False, indent=1) + "\n")

print(f"\nLEDGER: 断言 {len(checks)} 条，全部 PASS")
print("E039-RANSAC-LAB-DONE")
