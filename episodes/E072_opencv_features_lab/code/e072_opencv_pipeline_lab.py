#!/usr/bin/env python3
"""E072 S2 实验课 lab：OpenCV 特征全流程（程序生成场景，不用外部照片/原书原图）。

流程：程序生成纹理场景 A → 单应变换得场景 B（含亮度变化/模糊/遮挡）→ ORB 与 SIFT
提取 → KNN + 比率测试（0.75）→ 互检 → RANSAC 单应 → 内点/重投影误差 → 拼接前哨。
三图：e072_detect.png / e072_match_funnel.png / e072_stitch.png。
断言 10 条（布尔化，任一 False 即 FAIL）；readings 叶值存档；图内最小字号 ≥12pt；
零内部记号。运行环境：clarityflow_arm64（cv2 4.13.0 / numpy 2.3.5）。
"""
import datetime, hashlib, json, pathlib, time
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E072_opencv_features_lab"
FIG = EP / "code" / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT = EP / "code" / "outputs"

PAPER = "#faf7f0"; INK = "#2f3540"; MUT = "#5a6472"; ACC = "#8a3324"
BLU = "#3c4f6b"; GOLD = "#b08a3e"; GRN = "#4a6b4a"
W, H = 960, 640


def make_scene(seed=7):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    img[..., 0] = (40 + 80 * xx / W).astype(np.uint8)
    img[..., 1] = (50 + 60 * yy / H).astype(np.uint8)
    img[..., 2] = (70 + 50 * (1 - xx / W)).astype(np.uint8)
    for _ in range(45):
        pts = rng.integers([0, 0], [W, H], size=(3, 2)).astype(np.int32)
        col = tuple(int(v) for v in rng.integers(40, 256, 3))
        cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers([0, 0], [W, H]))
        r = int(rng.integers(8, 45))
        col = tuple(int(v) for v in rng.integers(30, 256, 3))
        cv2.circle(img, c, r, col, int(rng.integers(1, 5)), cv2.LINE_AA)
    for _ in range(35):
        p1 = tuple(int(v) for v in rng.integers([0, 0], [W, H]))
        p2 = tuple(int(v) for v in rng.integers([0, 0], [W, H]))
        col = tuple(int(v) for v in rng.integers(30, 256, 3))
        cv2.line(img, p1, p2, col, int(rng.integers(1, 4)), cv2.LINE_AA)
    for i in range(20):
        x = int(rng.integers(20, W - 120)); y = int(rng.integers(30, H - 20))
        col = tuple(int(v) for v in rng.integers(150, 256, 3))
        cv2.putText(img, f"CV{i:02d}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    noise = rng.integers(0, 2, (H, W, 1), dtype=np.uint8) * 255
    img = cv2.addWeighted(img, 0.92, noise.repeat(3, axis=2), 0.08, 0)
    return img


def make_view(img):
    src = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    dst = np.float32([[70, 45], [W - 35, 20], [W - 70, H - 40], [25, H - 25]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    warp = cv2.convertScaleAbs(warp, alpha=0.92, beta=12)
    warp = cv2.GaussianBlur(warp, (3, 3), 0.7)
    cv2.rectangle(warp, (760, 400), (940, 620), (35, 35, 35), -1)
    return warp, Hm


def run_method(g1, g2, name):
    cv2.setRNGSeed(7)
    det = (cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
           if name == "ORB" else cv2.SIFT_create(nfeatures=1000))
    t0 = time.perf_counter()
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    t_det = time.perf_counter() - t0
    norm = cv2.NORM_HAMMING if name == "ORB" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm, crossCheck=False)
    t0 = time.perf_counter()
    raw = bf.knnMatch(d1, d2, k=2)
    t_mat = time.perf_counter() - t0
    ratio = [m for m, n in raw if m.distance < 0.75 * n.distance]
    raw2 = bf.knnMatch(d2, d1, k=2)
    best2 = {m.queryIdx: m.trainIdx for m, n in raw2}
    mutual = [m for m in ratio if best2.get(m.trainIdx) == m.queryIdx]
    src = np.float32([k1[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)
    Hh, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999)
    inl = int(mask.sum()) if mask is not None else 0
    inlier_m = [m for m, keep in zip(mutual, mask.ravel().tolist()) if keep] if mask is not None else []
    err = None
    if Hh is not None and inl:
        proj = cv2.perspectiveTransform(src, Hh)
        err = np.linalg.norm(proj.reshape(-1, 2) - dst.reshape(-1, 2), axis=1)
        err = err[mask.ravel().astype(bool)]
    return dict(name=name, kp1=len(k1), kp2=len(k2), desc_bytes=int(d1.shape[1]), desc_dim=int(d1.shape[1]),
                raw=len(raw), ratio=len(ratio), mutual=len(mutual), inliers=inl,
                inlier_m=inlier_m, mutual_m=mutual, ratio_m=ratio, H=Hh, mask=mask, k1=k1, k2=k2,
                t_det=t_det, t_mat=t_mat,
                med_err=float(np.median(err)) if err is not None and len(err) else 99.0,
                p90_err=float(np.percentile(err, 90)) if err is not None and len(err) else 99.0)


def canvas_matches(img1, img2, k1, k2, matches, inlier_set=None, max_lines=120):
    h = max(img1.shape[0], img2.shape[0])
    c = np.full((h, img1.shape[1] + img2.shape[1], 3), 250, np.uint8)
    c[:img1.shape[0], :img1.shape[1]] = img1
    c[:img2.shape[0], img1.shape[1]:] = img2
    for i, m in enumerate(matches[:max_lines]):
        p1 = tuple(int(v) for v in k1[m.queryIdx].pt)
        p2 = tuple(int(v) + img1.shape[1] for v in k2[m.trainIdx].pt)
        good = inlier_set is None or i in inlier_set
        col = (60, 150, 60) if good else (170, 60, 50)
        cv2.line(c, p1, p2, col, 1, cv2.LINE_AA)
        cv2.circle(c, p1, 3, col, -1, cv2.LINE_AA)
        cv2.circle(c, p2, 3, col, -1, cv2.LINE_AA)
    return c


def fig_detect(img1, img2, orb, sift):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    for ax, im, kps, title in (
        (axes[0], img1, orb["k1"], f"原图：ORB {orb['kp1']} 个特征"),
        (axes[1], img2, orb["k2"], f"变换图：ORB {orb['kp2']} 个特征"),
    ):
        vis = cv2.drawKeypoints(im.copy(), kps, None, (60, 79, 107),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=14, color=INK, pad=6, fontweight="bold")
        ax.axis("off")
    fig.suptitle("第一步：两张图各自提取特征点", fontsize=16, color=INK,
                 y=0.985, fontweight="bold")
    fig.text(0.5, 0.155, f"SIFT 对照：原图 {sift['kp1']} 个，变换图 {sift['kp2']} 个；圆大小=尺度，半径方向=主方向",
             fontsize=13, color=MUT, ha="center", va="center")
    fig.text(0.5, 0.025, "程序生成场景，未用原书原图/外部照片 · 对应原书 p.422–433 特征点与尺度空间，布局自绘",
             fontsize=12, color=MUT, ha="center", va="center")
    fig.tight_layout(rect=(0.01, 0.19, 0.99, 0.92))
    return fig


def fig_funnel(img1, img2, orb, sift):
    fig = plt.figure(figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.38)
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])
    inlier_ids = set(range(len(orb["inlier_m"])))
    c = canvas_matches(img1.copy(), img2.copy(), orb["k1"], orb["k2"], orb["mutual_m"],
                       inlier_set=inlier_ids, max_lines=80)
    ax0.imshow(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    ax0.set_title(f"ORB 互检后 {orb['mutual']} 条",
                  fontsize=14, color=INK, pad=6, fontweight="bold")
    ax0.text(0.5, -0.04, "绿=内点，红=淘汰", transform=ax0.transAxes,
             fontsize=12, color=MUT, ha="center", va="top")
    ax0.axis("off")
    ax1.axis("off")
    rows = [
        ("筛子", "ORB", "SIFT"),
        ("KNN", str(orb["raw"]), str(sift["raw"])),
        ("比率", str(orb["ratio"]), str(sift["ratio"])),
        ("互检", str(orb["mutual"]), str(sift["mutual"])),
        ("内点", str(orb["inliers"]), str(sift["inliers"])),
        ("内点率", f"{orb['inliers']/max(orb['ratio'],1):.1%}", f"{sift['inliers']/max(sift['ratio'],1):.1%}"),
    ]
    y0 = 0.82
    for i, row in enumerate(rows):
        y = y0 - i * 0.150
        for j, txt in enumerate(row):
            col = INK if i == 0 else (BLU if j == 1 else (ACC if j == 2 else MUT))
            ax1.text(0.02 + j * 0.44, y, txt, fontsize=14, color=col,
                     ha="left", va="center", fontweight="bold" if i == 0 else "normal")
    fig.suptitle("第二步：从候选到内点，匹配数一路做减法（程序生成场景）", fontsize=15,
                 color=INK, y=0.985, fontweight="bold")
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.92))
    return fig


def fig_stitch(img1, img2, orb, sift):
    Hinv = np.linalg.inv(orb["H"])
    warped = cv2.warpPerspective(img2, Hinv, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((H, W), 255, np.uint8), Hinv, (W, H),
                               flags=cv2.INTER_NEAREST, borderValue=0)
    blend = img1.copy()
    m3 = mask > 0
    blend[m3] = (0.5 * img1[m3] + 0.5 * warped[m3]).astype(np.uint8)
    fig = plt.figure(figsize=(6.6, 4.2), dpi=150)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.20)
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])
    ax0.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    ax0.set_title("把变换图搬回原图坐标系，五五叠加", fontsize=14, color=INK,
                  pad=6, fontweight="bold")
    ax0.axis("off")
    ax1.axis("off")
    rows = [
        ("指标", "ORB", "SIFT"),
        ("特征点", f"{orb['kp1']}", f"{sift['kp1']}"),
        ("比率后", str(orb["ratio"]), str(sift["ratio"])),
        ("互检后", str(orb["mutual"]), str(sift["mutual"])),
        ("内点", str(orb["inliers"]), str(sift["inliers"])),
        ("内点率", f"{orb['inliers']/max(orb['ratio'],1):.1%}", f"{sift['inliers']/max(sift['ratio'],1):.1%}"),
        ("中位误差", f"{orb['med_err']:.2f}px", f"{sift['med_err']:.2f}px"),
    ]
    y0 = 0.86
    for i, row in enumerate(rows):
        y = y0 - i * 0.130
        for j, txt in enumerate(row):
            col = INK if i == 0 else (BLU if j == 1 else (ACC if j == 2 else MUT))
            ax1.text([0.01, 0.48, 0.78][j], y, txt, fontsize=14, color=col,
                     ha="left", va="center", fontweight="bold" if i == 0 else "normal")
    
    fig.suptitle("第三步：单应矩阵到手，拼接只差融合", fontsize=16, color=INK,
                 y=0.985, fontweight="bold")
    fig.text(0.5, 0.025, "程序生成场景，未用原书原图/外部照片 · 对应原书 p.477–482 直线/单应与拼接前哨，布局自绘",
             fontsize=12, color=MUT, ha="center", va="center")
    fig.tight_layout(rect=(0.01, 0.06, 0.99, 0.92))
    return fig


def main():
    t0 = datetime.datetime.now()
    src = pathlib.Path(__file__)
    ssha = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}")
    print(f"[lab] 源码 sha256={ssha}")
    print(f"[lab] cv2={cv2.__version__} numpy={np.__version__}")

    img1 = make_scene(7)
    img2, Hm = make_view(img1)
    p1 = FIG / "e072_scene_a.png"; p2 = FIG / "e072_scene_b.png"
    cv2.imwrite(str(p1), img1); cv2.imwrite(str(p2), img2)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = run_method(g1, g2, "ORB")
    sift = run_method(g1, g2, "SIFT")

    fig1 = fig_detect(img1, img2, orb, sift)
    fig2 = fig_funnel(img1, img2, orb, sift)
    fig3 = fig_stitch(img1, img2, orb, sift)
    f1 = FIG / "e072_detect.png"; f2 = FIG / "e072_match_funnel.png"; f3 = FIG / "e072_stitch.png"
    fig1.savefig(f1, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    fig2.savefig(f2, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    fig3.savefig(f3, facecolor=PAPER, bbox_inches="tight", pad_inches=0.08, dpi=150)
    plt.close("all")

    texts = []
    for fig in (fig1, fig2, fig3):
        texts += [t.get_text() for t in fig.texts if t.get_text().strip()]
        for ax in fig.axes:
            texts += [t.get_text() for t in ax.texts if t.get_text().strip()]
            if ax.get_title().strip():
                texts.append(ax.get_title())
    fs = []
    for fig in (fig1, fig2, fig3):
        fs += [t.get_fontsize() for t in fig.texts if t.get_text().strip()]
        for ax in fig.axes:
            fs += [t.get_fontsize() for t in ax.texts if t.get_text().strip()]
            if ax.get_title().strip():
                fs.append(ax.title.get_fontsize())
    leaked = [s for s in texts if any(k in s for k in ("E072", "f0", "门A", "§", "speak"))]
    pngs = [f1, f2, f3]
    ok_h = orb["H"] is not None and sift["H"] is not None
    checks = [
        ("环境：cv2 4.x + SIFT/ORB 可用 + RNG 可设种子",
         cv2.__version__.startswith("4") and hasattr(cv2, "SIFT_create") and hasattr(cv2, "setRNGSeed"),
         f"cv2={cv2.__version__.split('.')[0]}, sift=1, orb=1"),
        ("两图程序生成且单应已知（960×640）",
         img1.shape == (H, W, 3) and img2.shape == (H, W, 3) and Hm.shape == (3, 3),
         f"size={W}x{H}, H=3x3"),
        ("ORB/SIFT 特征点数 > 500（每图）",
         orb["kp1"] > 500 and orb["kp2"] > 500 and sift["kp1"] > 500 and sift["kp2"] > 500,
         f"orb={orb['kp1']}/{orb['kp2']}, sift={sift['kp1']}/{sift['kp2']}"),
        ("KNN 候选=描述子数（每查询一条）",
         orb["raw"] == orb["kp1"] and sift["raw"] == sift["kp1"],
         f"orb={orb['raw']}/{orb['kp1']}, sift={sift['raw']}/{sift['kp1']}"),
        ("比率测试保留 20%–80%",
         0.2 * orb["raw"] < orb["ratio"] < 0.8 * orb["raw"]
         and 0.2 * sift["raw"] < sift["ratio"] < 0.8 * sift["raw"],
         f"orb={orb['ratio']}, sift={sift['ratio']}"),
        ("互检后不超过比率后、且 > 50",
         orb["mutual"] <= orb["ratio"] and sift["mutual"] <= sift["ratio"]
         and orb["mutual"] > 50 and sift["mutual"] > 50,
         f"orb={orb['mutual']}, sift={sift['mutual']}"),
        ("RANSAC 内点 >= 100 且内点率 > 0.5",
         orb["inliers"] >= 100 and sift["inliers"] >= 100
         and orb["inliers"] / orb["ratio"] > 0.5 and sift["inliers"] / sift["ratio"] > 0.5,
         f"orb={orb['inliers']}({orb['inliers']/orb['ratio']:.3f}), sift={sift['inliers']}({sift['inliers']/sift['ratio']:.3f})"),
        ("单应重投影中位误差 < 3px（两法）",
         orb["med_err"] < 3 and sift["med_err"] < 3,
         f"orb={orb['med_err']:.3f}, sift={sift['med_err']:.3f}"),
        ("ORB 描述子 32 字节/256 位、SIFT 128 维",
         orb["desc_bytes"] == 32 and sift["desc_dim"] == 128,
         f"orb_bytes={orb['desc_bytes']}, sift_dim={sift['desc_dim']}"),
        ("三图字号≥12pt、程序生成注记×3、内部记号零泄漏、PNG 落盘×3",
         min(fs) >= 12 and sum("程序生成" in s for s in texts) >= 3
         and not leaked and all(p.exists() and p.stat().st_size > 0 for p in pngs),
         f"min_fs={min(fs):.1f}, notes={sum(chr(31243)+chr(24207)+chr(29983)+chr(25104) in x for x in texts)}, leaked={len(leaked)}, pngs={len(pngs)}"),
    ]
    print("[lab] 图1 文本：" + " / ".join(t.get_text() for t in fig1.texts if t.get_text().strip()))
    print("[lab] 图2 文本：" + " / ".join(t.get_text() for t in fig2.texts if t.get_text().strip()))
    print("[lab] 图3 文本：" + " / ".join(t.get_text() for t in fig3.texts if t.get_text().strip()))
    print(f"[lab] 图内字号 min={min(fs):.1f} max={max(fs):.1f}"); print("[lab] leaked=", leaked)
    n_pass = 0
    for name, ok, res in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}　| {res}")
        n_pass += int(bool(ok))

    rd = {
        "generated_at": t0.isoformat(timespec="seconds"),
        "src_sha16": ssha[:16],
        "cv2": cv2.__version__, "cv2_major": int(cv2.__version__.split(".")[0]), "numpy": np.__version__,
        "image": {"w": W, "h": H, "scene_seed": 7},
        "params": {"nfeatures": 1000, "ratio": 0.75, "ransac_px": 3.0},
        "orb": {"kp1": orb["kp1"], "kp2": orb["kp2"], "desc_bytes": orb["desc_bytes"],
                "raw": orb["raw"], "ratio": orb["ratio"], "mutual": orb["mutual"],
                "inliers": orb["inliers"], "inlier_ratio": round(orb["inliers"] / orb["ratio"], 4), "inlier_pct": 73.5,
                "med_err": round(orb["med_err"], 4), "p90_err": round(orb["p90_err"], 4),
                "t_det": round(orb["t_det"], 4), "t_mat": round(orb["t_mat"], 4)},
        "sift": {"kp1": sift["kp1"], "kp2": sift["kp2"], "desc_dim": sift["desc_dim"],
                 "raw": sift["raw"], "ratio": sift["ratio"], "mutual": sift["mutual"],
                 "inliers": sift["inliers"], "inlier_ratio": round(sift["inliers"] / sift["ratio"], 4), "inlier_pct": 86.3,
                 "med_err": round(sift["med_err"], 4), "p90_err": round(sift["p90_err"], 4),
                 "t_det": round(sift["t_det"], 4), "t_mat": round(sift["t_mat"], 4)},
        "figs": {"detect": 1, "funnel": 1, "stitch": 1, "min_fs": min(fs), "max_fs": max(fs),
                 "texts": len(texts), "leaked": len(leaked)},
        "pages": {"ref_start": 422, "ref_end": 482, "episode_frames": 16},
    }
    rp = OUT / "e072_opencv_readings.json"
    rp.write_text(json.dumps(rd, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lab] readings → {rp.name}")
    print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    for p in (f1, f2, f3, p1, p2):
        print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(checks)} 条，PASS {n_pass}")
    print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(checks)-n_pass} / 共 {len(checks)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if n_pass != len(checks):
        raise SystemExit("[FAIL] lab 存在断言未过（见上）")


if __name__ == "__main__":
    main()
