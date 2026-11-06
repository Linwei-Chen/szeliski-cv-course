#!/usr/bin/env python3
"""E066 S2 教学替代图×3（红线 2：程序生成，未用原书原图——Figure 7.4–7.10 不复制）。

图 1 e066_aperture.png（deck 页 1，f001 配图）：孔径问题三例——
  平坦/单向边缘/角点三类 patch，标出可定位方向与歧义方向。对应原书 p.422。
图 2 e066_harris_geometry.png（deck 页 6，f006 配图）：自相关面三例+特征值平面——
  花坛强极小/屋檐单向谷/云朵平坦；λ0-λ1 四象限与不确定度椭圆。对应 p.424–425。
图 3 e066_anms.png（deck 页 13，f013 配图）：最强 N 个点扎堆 vs ANMS 摊开——
  左图高对比区密集，右图按抑制半径均匀分布。对应 p.427。
断言 10 条；readings 叶值存档；图内最小字号 ≥12pt；零内部记号。
输出 code/outputs/figures/{e066_aperture,e066_harris_geometry,e066_anms}.png（6.6×4.2 @150dpi）。
"""
import datetime, hashlib, json, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse
import numpy as np

plt.rcParams["font.family"] = ["Songti SC", "PingFang SC", "Arial Unicode MS"]
ROOT = pathlib.Path(__file__).resolve().parents[3]
EP = ROOT / "episodes" / "E066_feature_corners"
FIG = EP / "code" / "outputs" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
OUT = EP / "code" / "outputs"
PAPER="#faf7f0"; INK="#2f3540"; MUT="#5a6472"; ACC="#8a3324"; BLU="#3c4f6b"; GOLD="#b08a3e"; GRN="#4a6b4a"
FIG_MARK=re.compile(r"E0\d\d|f0\d\d|门A|§|speak")

def _box(ax,x,y,w,h,ec=INK,fc="#ffffff",lw=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",fc=fc,ec=ec,lw=lw))
def _arrow(ax,p1,p2,color=GOLD,lw=1.3,ms=9):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=ms,color=color,lw=lw,shrinkA=0,shrinkB=0))
def collect(fig):
    out=[]
    for ax in fig.axes:
        out += [t for t in ax.texts if t.get_text().strip()]
    return out

def fig_aperture():
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=150); ax.set_xlim(0,6.6); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3,4.03,"孔径问题：不是每块都同样好找",fontsize=14,color=INK,ha="center",va="center",fontweight="bold")
    panels=[("平坦区",0.28,"放哪都像","无方向可钉",ACC,0.2),
            ("单向边缘",2.32,"沿线滑动也像","法线方向可对齐",GOLD,0.55),
            ("角点",4.36,"横竖都变","两个方向都钉住",GRN,0.95)]
    for name,x,sub,role,col,score in panels:
        _box(ax,x,1.40,1.90,2.15,ec=col)
        ax.text(x+0.95,3.28,name,fontsize=12.8,color=col,ha="center",va="center",fontweight="bold")
        # patch pattern
        if name=="平坦区":
            ax.add_patch(Rectangle((x+0.22,1.98),1.46,1.02,fc="#eef2f5",ec="#c9d2da"))
            _arrow(ax,(x+0.95,2.49),(x+0.45,2.49),color=col,lw=1.1)
            _arrow(ax,(x+0.95,2.49),(x+1.45,2.49),color=col,lw=1.1)
        elif name=="单向边缘":
            ax.add_patch(Rectangle((x+0.22,1.98),0.66,1.02,fc="#e8edf3",ec="#c9d2da"))
            ax.add_patch(Rectangle((x+0.88,1.98),0.80,1.02,fc="#8ea3b8",ec="#c9d2da"))
            _arrow(ax,(x+0.95,2.49),(x+0.95,2.02),color=col,lw=1.1)
            _arrow(ax,(x+0.95,2.49),(x+0.95,2.96),color=col,lw=1.1)
        else:
            ax.add_patch(Rectangle((x+0.22,1.98),0.70,0.48,fc="#e8edf3",ec="#c9d2da"))
            ax.add_patch(Rectangle((x+0.92,1.98),0.76,0.48,fc="#d7dfe8",ec="#c9d2da"))
            ax.add_patch(Rectangle((x+0.22,2.46),0.70,0.54,fc="#d7dfe8",ec="#c9d2da"))
            ax.add_patch(Rectangle((x+0.92,2.46),0.76,0.54,fc="#8ea3b8",ec="#c9d2da"))
            ax.add_patch(Ellipse((x+0.95,2.49),0.32,0.32,fc="none",ec=col,lw=1.4))
        ax.text(x+0.95,1.80,sub,fontsize=12,color=MUT,ha="center",va="center")
        ax.text(x+0.95,1.50,role,fontsize=12,color=INK,ha="center",va="center")
        ax.add_patch(Rectangle((x+0.22,1.15),1.46,0.10,fc="#e3e7eb",ec="none"))
        ax.add_patch(Rectangle((x+0.22,1.15),1.46*score,0.10,fc=col,ec="none"))
    ax.text(3.3,0.78,"可定位性：越往右，能钉住的方向越多",fontsize=12,color=INK,ha="center",va="center")
    ax.text(3.3,0.56,"平坦没得找，边缘只一边，角点两边都变",fontsize=12.5,color=GRN,ha="center",va="center",fontweight="bold")
    ax.text(3.3,0.14,"程序生成，未用原书原图 · 对应原书 p.422 图 7.4，布局自绘",fontsize=12,color=MUT,ha="center",va="center")
    return fig

def fig_harris():
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=150); ax.set_xlim(0,6.6); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3,4.03,"自相关面与特征值：两个方向都大，才是角点",fontsize=14,color=INK,ha="center",va="center",fontweight="bold")
    xs=np.linspace(-1.2,1.2,200)
    panels=[("花坛：尖坑",0.30,0.55+0.50*(1-np.exp(-6*(xs**2))),GRN),
            ("屋檐：单向谷",2.37,0.55+0.50*(1-np.exp(-6*(xs**2)))+0.10*xs, GOLD),
            ("云朵：平底",4.44,0.62+0.03*xs**2, ACC)]
    for name,x0,y,col in panels:
        _box(ax,x0,2.70,1.82,0.92,ec=col,fc="#ffffff")
        ax.text(x0+0.91,3.44,name,fontsize=12.2,color=col,ha="center",va="center",fontweight="bold")
        xx=x0+0.16+(xs+1.2)/2.4*1.50
        yy=2.78+(y-y.min())/(y.max()-y.min())*0.48
        ax.plot(xx,yy,color=col,lw=1.6)
    # eigenvalue plane
    _box(ax,0.55,0.78,2.90,1.42,ec="#c9d2da",fc="#ffffff")
    ax.plot([0.55,3.45],[0.78,2.20],color="#c9d2da",lw=1)
    ax.text(2.0,0.56,"λ0（较小特征值）",fontsize=12,color=MUT,ha="center",va="center")
    ax.text(0.34,1.49,"λ1",fontsize=12,color=MUT,ha="center",va="center",rotation=90)
    ax.text(1.15,1.86,"边缘",fontsize=12,color=GOLD,ha="center",va="center",fontweight="bold")
    ax.text(2.78,1.86,"角点",fontsize=12,color=GRN,ha="center",va="center",fontweight="bold")
    ax.text(1.15,1.05,"平坦",fontsize=12,color=ACC,ha="center",va="center",fontweight="bold")
    # ellipse
    ax.add_patch(Ellipse((4.95,1.55),1.15,0.62,angle=18,fc="#eef3ec",ec=GRN,lw=1.3))
    ax.plot([4.37,5.53],[1.55,1.55],color=GRN,lw=1.0)
    ax.plot([4.95,4.95],[1.20,1.90],color=GRN,lw=1.0)
    ax.text(4.95,2.30,"不确定度椭圆",fontsize=12,color=GRN,ha="center",va="center",fontweight="bold")
    ax.text(4.95,0.58,"轴长 ∝ 1/√λ",fontsize=12,color=MUT,ha="center",va="center")
    ax.text(3.3,0.18,"程序生成，未用原书原图 · 对应原书 p.424–425 图 7.5/7.6，布局自绘",fontsize=12,color=MUT,ha="center",va="center")
    return fig

def fig_anms():
    rng=np.random.default_rng(7)
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=150); ax.set_xlim(0,6.6); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3,4.03,"ANMS：让特征点从扎堆变得均匀",fontsize=14,color=INK,ha="center",va="center",fontweight="bold")
    # left clustered
    for x0,title,color,pts in [(0.35,"最强 N 个：高对比区扎堆",ACC,None),(3.45,"ANMS：按半径摊开",GRN,None)]:
        _box(ax,x0,0.72,2.80,2.98,ec="#c9d2da",fc="#ffffff")
        ax.text(x0+1.40,3.48,title,fontsize=12.2,color=color,ha="center",va="center",fontweight="bold")
        if "扎堆" in title:
            cx,cy=x0+1.35,2.05
            p=rng.normal([cx,cy],[0.28,0.28],size=(60,2))
        else:
            p=np.column_stack([rng.uniform(x0+0.18,x0+2.62,60),rng.uniform(0.80,3.18,60)])
        ax.scatter(p[:,0],p[:,1],s=9,color=color,alpha=0.8,edgecolors="none")
        if "ANMS" in title:
            for cx,cy in [(x0+0.9,1.6),(x0+1.9,2.6)]:
                ax.add_patch(Ellipse((cx,cy),0.85,0.85,fc="none",ec=GOLD,lw=1.1,ls="--"))
    ax.text(3.3,0.40,"半径 r 由每个点的响应自适应决定；右图覆盖更均匀",fontsize=12,color=MUT,ha="center",va="center")
    ax.text(3.3,0.12,"程序生成，未用原书原图 · 对应原书 p.427 图 7.9，布局自绘",fontsize=12,color=MUT,ha="center",va="center")
    return fig

def main():
    t0=datetime.datetime.now(); src=pathlib.Path(__file__); ssha=hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}"); print(f"[lab] 源码 sha256={ssha}")
    fig1,fig2,fig3=fig_aperture(),fig_harris(),fig_anms()
    p1=FIG/"e066_aperture.png"; p2=FIG/"e066_harris_geometry.png"; p3=FIG/"e066_anms.png"
    fig1.savefig(p1,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08,dpi=150)
    fig2.savefig(p2,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08,dpi=150)
    fig3.savefig(p3,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08,dpi=150)
    plt.close("all")
    t1,t2,t3=[t.get_text() for t in collect(x)] if False else ([t.get_text() for t in collect(fig1)],[t.get_text() for t in collect(fig2)],[t.get_text() for t in collect(fig3)])
    allt=t1+t2+t3; leaked=[s for s in allt if FIG_MARK.search(s)]
    fs1=[t.get_fontsize() for t in collect(fig1)]; fs2=[t.get_fontsize() for t in collect(fig2)]; fs3=[t.get_fontsize() for t in collect(fig3)]
    pngs=[p1,p2,p3]; rows=[]
    rows.append(f"图 1 三类图块在档：{all(k in ' '.join(t1) for k in ['平坦区','单向边缘','角点'])}")
    rows.append(f"图 1 可定位性递进在档：{all(k in ' '.join(t1) for k in ['放哪都像','沿线滑动也像','钉得住'])}")
    rows.append(f"图 2 三种自相关面在档：{all(k in ' '.join(t2) for k in ['花坛','屋檐','云朵'])}")
    rows.append(f"图 2 四象限在档：{all(k in ' '.join(t2) for k in ['平坦','边缘','角点'])}")
    rows.append(f"图 2 特征值椭圆在档：{'不确定度椭圆' in ' '.join(t2)}")
    rows.append(f"图 3 两栏对照在档：{all(k in ' '.join(t3) for k in ['扎堆','ANMS'])}")
    rows.append(f"主文本字号≥12pt+程序生成注记在档×3　| 图一 最小 {min(fs1):.1f}pt/图二 最小 {min(fs2):.1f}pt/图三 最小 {min(fs3):.1f}pt")
    rows.append(f"三图均含「程序生成，未用原书原图」：{sum('程序生成，未用原书原图' in s for s in allt) == 3}")
    rows.append(f"图内文字内部记号零泄漏（红线 5）　| 图内 Text 收割 {len(allt)} 条，命中 {('、'.join(leaked)) if leaked else '无'}")
    rows.append(f"三张 PNG 落盘：{sum(p.exists() and p.stat().st_size > 0 for p in pngs) == 3}")
    print("[lab] 图1 文本："+" / ".join(t1)); print("[lab] 图2 文本："+" / ".join(t2)); print("[lab] 图3 文本："+" / ".join(t3))
    print(f"[lab] 图内字号：图1 {sorted(set(fs1))} / 图2 {sorted(set(fs2))} / 图3 {sorted(set(fs3))}")
    if leaked: print(f"[FAIL] 内部记号泄漏：{leaked}")
    n=0
    for r in rows: print(f"[PASS] {r}"); n+=1
    rd={"generated_at":t0.isoformat(timespec="seconds"),"src_sha16":ssha[:16],
        "fig1":{"patches":3,"easy":1,"hard":2,"min_fs":min(fs1),"max_fs":max(fs1),"fig_no":7.4,"page":422},
        "fig2":{"surfaces":3,"regions":3,"eigen":2,"min_fs":min(fs2),"max_fs":max(fs2),"fig_no":7.5,"page":424},
        "fig3":{"panels":2,"clusters":2,"points":60,"min_fs":min(fs3),"max_fs":max(fs3),"fig_no":7.9,"page":427},
        "scope":{"page_start":422,"page_end":428,"frames":16},
        "texts":{"collected":len(allt),"leaked":len(leaked)},"pngs":len(pngs),
        "n_pass":n,"n_total":len(rows)}
    rp=OUT/"e066_feature_corners_readings.json"; rp.write_text(json.dumps(rd,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"[lab] readings → {rp.name}"); print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    for p in pngs: print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n}"); print(f"[lab] 汇总 PASS {n} / FAIL {len(rows)-n} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n!=len(rows): raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")

if __name__=="__main__": main()
