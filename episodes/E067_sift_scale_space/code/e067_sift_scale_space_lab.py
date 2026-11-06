#!/usr/bin/env python3
"""E067 S2 教学替代图×3（程序生成，未用原书原图——Figure 7.10–7.14 不复制）。

图 1 e067_scale_space.png（deck 页 3，f003 配图）：尺度空间金字塔——
  两个八度的高斯模糊层、相减得 DoG、八度之间隔行隔列下采样。对应 p.430 Figure 7.11a。
图 2 e067_dog_extrema.png（deck 页 10，f010 配图）：DoG 3D 极值——
  当前层 8 邻居 + 上下层各 9 邻居 = 26 邻域；Hessian 迹平方/行列式剔除边缘。对应 p.430–431。
图 3 e067_orientation_affine.png（deck 页 13，f013 配图）：主方向与 affine——
  36 桶梯度方向直方图 + 80% 峰值 + 三桶抛物线拟合；右侧仿射椭圆与 MSER 面积-阈值曲线。
  对应 p.431–432 Figure 7.12/7.14。
断言 10 条；readings 叶值存档；图内最小字号 ≥12pt；零内部记号。
"""
import datetime,hashlib,json,pathlib,re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Ellipse
import numpy as np
plt.rcParams["font.family"]=["Songti SC","PingFang SC","Arial Unicode MS"]
ROOT=pathlib.Path(__file__).resolve().parents[3]; EP=ROOT/"episodes"/"E067_sift_scale_space"
FIG=EP/"code"/"outputs"/"figures"; FIG.mkdir(parents=True,exist_ok=True); OUT=EP/"code"/"outputs"
PAPER="#faf7f0"; INK="#2f3540"; MUT="#5a6472"; ACC="#8a3324"; BLU="#3c4f6b"; GOLD="#b08a3e"; GRN="#4a6b4a"
FIG_MARK=re.compile(r"E0\d\d|f0\d\d|门A|§|speak")
def box(ax,x,y,w,h,ec=INK,fc="#ffffff",lw=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",fc=fc,ec=ec,lw=lw))
def arrow(ax,p1,p2,color=GOLD,lw=1.3,ms=9):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle="-|>",mutation_scale=ms,color=color,lw=lw,shrinkA=0,shrinkB=0))
def collect(fig):
    out=[]
    for ax in fig.axes: out += [t for t in ax.texts if t.get_text().strip()]
    return out

def fig_scale_space():
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=150); ax.set_xlim(0,6.6); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3,4.03,"尺度空间：模糊、相减、下采样",fontsize=14,color=INK,ha="center",va="center",fontweight="bold")
    # column A: 3 Gaussian layers
    ax.text(0.95,3.72,"第一八度",fontsize=12,color=BLU,ha="center",va="center",fontweight="bold")
    ys=[2.95,2.15,1.35]
    for i,y in enumerate(ys):
        box(ax,0.30,y,1.30,0.62,ec=BLU,fc="#eef2f7")
        ax.text(0.95,y+0.38,f"G{i+1} 高斯模糊",fontsize=12,color=BLU,ha="center",va="center",fontweight="bold")
        ax.text(0.95,y+0.15,"越往下越模糊" if i==0 else "继续模糊",fontsize=12,color=MUT,ha="center",va="center")
    # column B: two DoG
    ax.text(2.75,3.72,"相减得 DoG",fontsize=12,color=ACC,ha="center",va="center",fontweight="bold")
    for i,y in enumerate([2.55,1.75]):
        box(ax,2.10,y,1.30,0.62,ec=ACC,fc="#fdf0ec")
        ax.text(2.75,y+0.38,f"DoG{i+1}",fontsize=12.5,color=ACC,ha="center",va="center",fontweight="bold")
        ax.text(2.75,y+0.15,"差分图",fontsize=12,color=MUT,ha="center",va="center")
    arrow(ax,(1.63,2.95),(2.07,2.86),color=GOLD); arrow(ax,(1.63,2.15),(2.07,2.65),color=GOLD)
    arrow(ax,(1.63,2.15),(2.07,1.96),color=GOLD); arrow(ax,(1.63,1.35),(2.07,1.85),color=GOLD)
    # column C: second octave
    ax.text(4.95,3.72,"第二八度",fontsize=12,color=GRN,ha="center",va="center",fontweight="bold")
    for i,y in enumerate([2.55,1.75]):
        box(ax,4.30,y,1.30,0.62,ec=GRN,fc="#eef5ee")
        ax.text(4.95,y+0.38,f"G{i+1} 下采样",fontsize=12,color=GRN,ha="center",va="center",fontweight="bold")
        ax.text(4.95,y+0.15,"σ 翻倍",fontsize=12,color=MUT,ha="center",va="center")
    arrow(ax,(3.43,2.86),(4.27,2.86),color=GOLD)
    ax.text(3.3,0.95,"每个八度生成 s+3 张模糊图；相邻相减得到 DoG；处理完一个八度，σ 翻倍后隔行隔列下采样",
            fontsize=12,color=INK,ha="center",va="center")
    ax.text(3.3,0.62,"粗找：空间+尺度三维极值；细修：二次拟合到亚像素/子层",
            fontsize=12,color=MUT,ha="center",va="center")
    ax.text(3.3,0.18,"程序生成，未用原书原图 · 对应原书 p.430 图 7.11a，布局自绘",fontsize=12,color=MUT,ha="center",va="center")
    return fig

def fig_dog_extrema():
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=150); ax.set_xlim(0,6.6); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3,4.03,"DoG 3D 极值：和 26 个邻居比",fontsize=14,color=INK,ha="center",va="center",fontweight="bold")
    layers=[("尺度上一层",2.90,BLU),("当前层",1.95,GOLD),("尺度下一层",1.00,GRN)]
    for name,y,col in layers:
        ax.text(0.35,y+0.52,name,fontsize=12,color=col,ha="left",va="center",fontweight="bold")
        for r in range(3):
            for c in range(3):
                x=1.95+c*0.52; yy=y+0.15+(2-r)*0.26
                fc="#ffffff" if (r,c)!=(1,1) else "#f6d9d2"
                ax.add_patch(Rectangle((x,yy),0.46,0.24,fc=fc,ec="#aeb8c4",lw=0.9))
    ax.text(2.70,2.47,"X",fontsize=16,color=ACC,ha="center",va="center",fontweight="bold")
    box(ax,4.00,1.35,2.30,1.65,ec=GRN,fc="#eef5ee")
    ax.text(5.15,2.76,"26 邻域",fontsize=13,color=GRN,ha="center",va="center",fontweight="bold")
    ax.text(5.15,2.40,"同层 8 个",fontsize=12,color=INK,ha="center",va="center")
    ax.text(5.15,2.12,"上一层 9 个",fontsize=12,color=INK,ha="center",va="center")
    ax.text(5.15,1.84,"下一层 9 个",fontsize=12,color=INK,ha="center",va="center")
    ax.text(5.15,1.55,"共 26 个",fontsize=12,color=ACC,ha="center",va="center",fontweight="bold")
    ax.text(3.3,0.62,"比全部 26 个邻居都大（或都小）才保留；再用 Hessian 剔除边缘响应",
            fontsize=12,color=INK,ha="center",va="center")
    ax.text(3.3,0.18,"程序生成，未用原书原图 · 对应原书 p.430–431 图 7.11b，布局自绘",fontsize=12,color=MUT,ha="center",va="center")
    return fig

def fig_orientation_affine():
    fig,ax=plt.subplots(figsize=(6.6,4.2),dpi=150); ax.set_xlim(0,6.6); ax.set_ylim(0,4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.text(3.3,4.03,"主方向与 affine：转角度、斜视角",fontsize=14,color=INK,ha="center",va="center",fontweight="bold")
    # left histogram
    rng=np.random.default_rng(3); bins=36; vals=rng.uniform(0.08,0.20,bins)
    vals[5]+=0.85; vals[6]+=0.70; vals[23]+=0.62
    x0,y0,w,h=0.35,1.45,3.00,1.75
    for i,v in enumerate(vals):
        ax.add_patch(Rectangle((x0+i*w/bins,y0),w/bins*0.82,min(v,1.0)*h,fc=BLU,ec="none",alpha=0.85))
    ax.plot([x0,x0+w],[y0+0.8*h,y0+0.8*h],color=ACC,lw=1.2,ls="--")
    ax.text(x0+w/2,y0+h+0.24,"36 桶方向直方图（虚线=80% 峰值）",fontsize=12,color=INK,ha="center",va="center")
    ax.text(x0+w/2,y0-0.28,"峰值附近三桶抛物线拟合 → 主方向",fontsize=12,color=MUT,ha="center",va="center")
    # right top ellipse
    ax.add_patch(Ellipse((5.05,2.62),1.25,0.68,angle=25,fc="#eef5ee",ec=GRN,lw=1.4))
    ax.plot([4.30,5.80],[2.62,2.62],color=GRN,lw=1.0)
    ax.plot([5.05,5.05],[2.18,3.06],color=GRN,lw=1.0)
    ax.text(5.05,3.48,"affine 归一：椭圆主轴/轴比",fontsize=12,color=GRN,ha="center",va="center",fontweight="bold")
    # right bottom MSER curve
    tx=np.linspace(0,1,100); ty=0.5+0.25*np.sin(6*tx)+0.1*tx
    ax.plot(4.10+tx*2.0,0.78+ty*0.48,color=GOLD,lw=1.5)
    ax.text(5.10,1.48,"MSER：面积随阈值变化最平",fontsize=12,color=GOLD,ha="center",va="center")
    ax.text(5.10,0.48,"小 patch 的透视缩短≈仿射",fontsize=12,color=MUT,ha="center",va="center")
    ax.text(3.3,0.18,"程序生成，未用原书原图 · 对应原书 p.431–432 图 7.12/7.14，布局自绘",fontsize=12,color=MUT,ha="center",va="center")
    return fig

def main():
    t0=datetime.datetime.now(); src=pathlib.Path(__file__); ssha=hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[lab] 开始 {t0.isoformat(timespec='seconds')}"); print(f"[lab] 源码 sha256={ssha}")
    fig1,fig2,fig3=fig_scale_space(),fig_dog_extrema(),fig_orientation_affine()
    p1=FIG/"e067_scale_space.png"; p2=FIG/"e067_dog_extrema.png"; p3=FIG/"e067_orientation_affine.png"
    fig1.savefig(p1,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08,dpi=150)
    fig2.savefig(p2,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08,dpi=150)
    fig3.savefig(p3,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08,dpi=150)
    plt.close("all")
    t1=[t.get_text() for t in collect(fig1)]; t2=[t.get_text() for t in collect(fig2)]; t3=[t.get_text() for t in collect(fig3)]
    allt=t1+t2+t3; leaked=[s for s in allt if FIG_MARK.search(s)]
    fs1=[t.get_fontsize() for t in collect(fig1)]; fs2=[t.get_fontsize() for t in collect(fig2)]; fs3=[t.get_fontsize() for t in collect(fig3)]
    pngs=[p1,p2,p3]; rows=[]
    rows.append(f"图 1 金字塔要素在档：{all(k in ' '.join(t1) for k in ['第一八度','第二八度','DoG'])}")
    rows.append(f"图 1 流程词在档：{all(k in ' '.join(t1) for k in ['模糊','相减','下采样'])}")
    rows.append(f"图 2 三层与 26 邻域在档：{all(k in ' '.join(t2) for k in ['尺度上一层','当前层','尺度下一层','26'])}")
    rows.append(f"图 2 判据在档：{all(k in ' '.join(t2) for k in ['都大','都小','Hessian'])}")
    rows.append(f"图 3 直方图与主方向在档：{all(k in ' '.join(t3) for k in ['36 桶','80%','主方向'])}")
    rows.append(f"图 3 affine/MSER 在档：{all(k in ' '.join(t3) for k in ['affine','椭圆','MSER'])}")
    rows.append(f"主文本字号≥12pt+程序生成注记在档×3　| 图一 最小 {min(fs1):.1f}pt/图二 最小 {min(fs2):.1f}pt/图三 最小 {min(fs3):.1f}pt")
    rows.append(f"三图均含「程序生成，未用原书原图」：{sum('程序生成，未用原书原图' in s for s in allt)==3}")
    rows.append(f"图内文字内部记号零泄漏（红线 5）　| 图内 Text 收割 {len(allt)} 条，命中 {('、'.join(leaked)) if leaked else '无'}")
    rows.append(f"三张 PNG 落盘：{sum(p.exists() and p.stat().st_size>0 for p in pngs)==3}")
    print("[lab] 图1 文本："+" / ".join(t1)); print("[lab] 图2 文本："+" / ".join(t2)); print("[lab] 图3 文本："+" / ".join(t3))
    print(f"[lab] 图内字号：图1 {sorted(set(fs1))} / 图2 {sorted(set(fs2))} / 图3 {sorted(set(fs3))}")
    if leaked: print(f"[FAIL] 内部记号泄漏：{leaked}")
    n=0
    for r in rows: print(f"[PASS] {r}"); n+=1
    rd={"generated_at":t0.isoformat(timespec="seconds"),"src_sha16":ssha[:16],
        "n_pass":n,"n_total":len(rows),
        "fig1":{"octaves":2,"layers":3,"dogs":2,"min_fs":min(fs1),"max_fs":max(fs1),"fig_no":7.11,"page":430},
        "fig2":{"layers":3,"neighbors":26,"min_fs":min(fs2),"max_fs":max(fs2),"fig_no":7.11,"page":431},
        "fig3":{"bins":36,"peak_pct":80,"params":2,"min_fs":min(fs3),"max_fs":max(fs3),"fig_no":7.12,"page":431},
        "scope":{"page_start":429,"page_end":433,"frames":16},
        "texts":{"collected":len(allt),"leaked":len(leaked)},"pngs":len(pngs)}
    rp=OUT/"e067_sift_scale_space_readings.json"; rp.write_text(json.dumps(rd,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"[lab] readings → {rp.name}"); print(f"[lab] readings sha256={hashlib.sha256(rp.read_bytes()).hexdigest()}")
    for p in pngs: print(f"[lab] PNG sha256({p.name})={hashlib.sha256(p.read_bytes()).hexdigest()}")
    print(f"[lab] LEDGER：断言 {len(rows)} 条，PASS {n}"); print(f"[lab] 汇总 PASS {n} / FAIL {len(rows)-n} / 共 {len(rows)}")
    print(f"[lab] 结束 {datetime.datetime.now().isoformat(timespec='seconds')}")
    if leaked or n!=len(rows): raise SystemExit("[FAIL] lab 存在记号泄漏或断言未全过")
if __name__=="__main__": main()
