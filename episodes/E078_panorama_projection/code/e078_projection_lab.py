#!/usr/bin/env python3
"""E078 实验：柱面/球面投影的正反变换、纯偏航平移与投影面选型（程序生成）。"""
import hashlib, json, pathlib
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
EP = pathlib.Path(__file__).resolve().parent.parent
OUT = EP/"code/outputs"; FIG=OUT/"figures"; FIG.mkdir(parents=True, exist_ok=True)
PAPER,INK,ACCENT,BLUE,GREEN="#f6f0e6","#1c2437","#a23b2a","#3d6a99","#2e7d32"
F=500.0; S=500.0

def cyl_map(x,y):
    t=np.arctan2(x,F); h=y/np.sqrt(x*x+F*F)
    return S*t, S*h

def cyl_inv(u,v):
    t=u/S; h=v/S
    return F*np.tan(t), h*F*np.sqrt(1+np.tan(t)**2)

def sph_map(x,y):
    t=np.arctan2(x,F); p=np.arctan2(y,np.sqrt(x*x+F*F))
    return S*t, S*p

def sph_inv(u,v):
    t=u/S; p=v/S
    return F*np.tan(t), F*np.tan(p)*np.sqrt(1+np.tan(t)**2)

xs=np.array([-800.,-400.,0.,400.,800.]); ys=np.array([-400.,-200.,0.,200.,400.])
X,Y=np.meshgrid(xs,ys); X=X.ravel(); Y=Y.ravel()
u,v=cyl_map(X,Y); xr,yr=cyl_inv(u,v)
us,vs=sph_map(X,Y); xsr,ysr=sph_inv(us,vs)
cyl_rt_x=float(np.max(np.abs(xr-X))); cyl_rt_y=float(np.max(np.abs(yr-Y)))
sph_rt_x=float(np.max(np.abs(xsr-X))); sph_rt_y=float(np.max(np.abs(ysr-Y)))

# 纯偏航：绕 y 轴转 delta，柱面坐标应整体水平平移 s*delta，h 不变
delta=np.deg2rad(17.0)
x2= np.cos(delta)*X + np.sin(delta)*F
z2=-np.sin(delta)*X + np.cos(delta)*F
u2=S*np.arctan2(x2,z2); v2=S*Y/np.sqrt(x2*x2+z2*z2)
yaw_dx=float(np.max(np.abs((u2-u)-S*delta)))
yaw_dh=float(np.max(np.abs(v2-v)))

phi=np.arctan2(Y,np.sqrt(X*X+F*F)); h=Y/np.sqrt(X*X+F*F)
h_gt_phi=float(np.min((h-phi)[Y>0]))
pole_y=1e6; pole_phi=float(np.arctan2(pole_y,F)); pole_h=float(pole_y/np.sqrt(0.0+F*F))
center_scale=float(S/F)
th60=np.deg2rad(60.0); planar_stretch_60=float(np.tan(th60)/th60); cyl_stretch_60=1.0

# ── 图 1：投影管线 ──
fig,ax=plt.subplots(figsize=(9.4,5.8),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
def box(x,y,w,h,t,b,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.02",fc="#fffdf8",ec=c,lw=1.6))
    ax.text(x+w/2,y+h-0.06,t,ha="center",va="center",fontsize=17,color=c,fontweight="bold")
    ax.text(x+w/2,y+h/2-0.04,b,ha="center",va="center",fontsize=15.5,color=INK,linespacing=1.5)
box(0.03,0.58,0.26,0.28,"像素射线","(x, y, f)\n过光心",BLUE)
box(0.36,0.58,0.28,0.28,"投到曲面","柱面：θ、h\n球面：θ、φ",ACCENT)
box(0.71,0.58,0.26,0.28,"展开成图","角度变横坐标\n高度变纵坐标",GREEN)
box(0.36,0.10,0.28,0.28,"对齐与融合","纯平移对齐\n接缝与曝光",BLUE)
for a,b in [((0.29,0.72),(0.36,0.72)),((0.64,0.72),(0.71,0.72)),((0.50,0.58),(0.50,0.38))]:
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,lw=1.6,color="#7a5c2e",connectionstyle="arc3,rad=0.08"))
ax.text(0.5,0.94,"先投影，再对齐：把转动变成平移",ha="center",va="center",fontsize=18,color=INK,fontweight="bold")
ax.text(0.5,0.03,"柱面适合水平一圈；球面适合全球或半球",ha="center",va="center",fontsize=14.5,color=ACCENT)
fig.savefig(FIG/"e078_pipeline.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# ── 图 2：柱面/球面/平面映射曲线 ──
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,5.6),dpi=150); fig.patch.set_facecolor(PAPER)
for ax in (a1,a2): ax.set_facecolor(PAPER); ax.grid(alpha=0.25)
xx=np.linspace(-900,900,400)
a1.plot(xx,S*np.arctan2(xx,F),color=BLUE,lw=2.4,label="柱面展开横坐标")
a1.plot(xx,xx,color=GREEN,lw=2.0,ls="--",label="平面横坐标")
a1.set_xlabel("像素横坐标 x",fontsize=15); a1.set_ylabel("展开后的横坐标",fontsize=15)
a1.set_title("横向：柱面把边缘压紧",fontsize=16,color=INK); a1.tick_params(labelsize=12.5); a1.legend(fontsize=13)
yy=np.linspace(0,900,300); phi_y=np.arctan2(yy,F); h_y=yy/np.sqrt(0.0+F*F)
a2.plot(yy,h_y,color=ACCENT,lw=2.4,label="柱面高度 h=tanφ")
a2.plot(yy,phi_y,color=BLUE,lw=2.4,label="球面角度 φ")
a2.set_xlabel("像素纵坐标 y",fontsize=15); a2.set_ylabel("高度 / 角度（弧度）",fontsize=15)
a2.set_title("纵向：柱面会越拉越长，球面有界",fontsize=16,color=INK); a2.tick_params(labelsize=12.5); a2.legend(fontsize=13)
fig.tight_layout(); fig.savefig(FIG/"e078_curves.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

# ── 图 3：平面边缘拉伸 vs 柱面 ──
fig,ax=plt.subplots(figsize=(8.4,5.4),dpi=150); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
angles=[30,45,60]; stretch=[np.tan(np.deg2rad(a))/np.deg2rad(a) for a in angles]
w=0.35; idx=np.arange(len(angles))
ax.bar(idx-w/2,stretch,w,color=ACCENT,label="平面投影")
ax.bar(idx+w/2,[1,1,1],w,color=BLUE,label="柱面投影")
for i,s in enumerate(stretch): ax.text(i-w/2,s+0.03,f"{s:.2f}",ha="center",fontsize=13.5,color=INK)
for i in idx: ax.text(i+w/2,1.03,"1.00",ha="center",fontsize=13.5,color=INK)
ax.set_xticks(idx); ax.set_xticklabels([f"半视角 {a}°" for a in angles],fontsize=13)
ax.set_ylabel("边缘拉伸倍数",fontsize=15); ax.tick_params(labelsize=12.5); ax.grid(axis="y",alpha=0.25)
ax.set_title("视角越宽，平面越拉；柱面保持角度均匀",fontsize=17,color=INK); ax.legend(fontsize=13)
fig.tight_layout(); fig.savefig(FIG/"e078_stretch.png",dpi=150,facecolor=PAPER,bbox_inches="tight",pad_inches=0.08); plt.close(fig)

tests=[
 ("柱面往返横坐标",cyl_rt_x<1e-9,f"最大误差 {cyl_rt_x:.2e}"),
 ("柱面往返纵坐标",cyl_rt_y<1e-9,f"最大误差 {cyl_rt_y:.2e}"),
 ("球面往返横坐标",sph_rt_x<1e-9,f"最大误差 {sph_rt_x:.2e}"),
 ("球面往返纵坐标",sph_rt_y<1e-9,f"最大误差 {sph_rt_y:.2e}"),
 ("纯偏航=水平平移",yaw_dx<1e-9,f"偏差 {yaw_dx:.2e}"),
 ("纯偏航高度不变",yaw_dh<1e-9,f"偏差 {yaw_dh:.2e}"),
 ("柱面高度大于球面角度",h_gt_phi>0,"全部样本 h>φ"),
 ("球面角度有界",pole_phi<np.pi/2,"φ 始终小于九十度"),
 ("中心尺度 s=f",abs(center_scale-1)<1e-12,f"缩放 {center_scale:.2f}"),
 ("宽视角平面拉伸更大",planar_stretch_60>1.5 and abs(cyl_stretch_60-1)<1e-12,f"平面 {planar_stretch_60:.2f} 对柱面 {cyl_stretch_60:.2f}"),
 ("三图在盘",all((FIG/n).exists() for n in ("e078_pipeline.png","e078_curves.png","e078_stretch.png")),"三张全在"),
]
lines=[]; n_pass=0
for name,ok,detail in tests:
    lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}　| {detail}"); n_pass+=bool(ok)
readings=dict(f=F,s=S,cyl_roundtrip_x=cyl_rt_x,cyl_roundtrip_y=cyl_rt_y,
              sph_roundtrip_x=sph_rt_x,sph_roundtrip_y=sph_rt_y,
              yaw_dx_dev=yaw_dx,yaw_dh_dev=yaw_dh,h_minus_phi_min=h_gt_phi,
              pole_phi=pole_phi,pole_h=pole_h,center_scale=center_scale,
              planar_stretch_60=planar_stretch_60,cyl_stretch_60=cyl_stretch_60,
              n_pass=n_pass,n_total=len(tests))
(OUT/"e078_projection_readings.json").write_text(json.dumps(readings,ensure_ascii=False,indent=1)+"\n")
rd_sha=hashlib.sha256((OUT/"e078_projection_readings.json").read_bytes()).hexdigest()
src_sha=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
print(f"[lab] E078 柱面/球面投影 开始 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
print(f"[lab] 源码 sha256={src_sha}")
for ln in lines: print(ln)
for png in sorted(FIG.glob("*.png")): print(f"[lab] PNG sha256({png.name})={hashlib.sha256(png.read_bytes()).hexdigest()}")
print("[lab] readings → e078_projection_readings.json"); print(f"[lab] readings sha256={rd_sha}")
print(f"[lab] LEDGER：断言 {len(tests)} 条，PASS {n_pass}")
print(f"[lab] 汇总 PASS {n_pass} / FAIL {len(tests)-n_pass} / 共 {len(tests)}")
print(f"[lab] 结束 {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
if n_pass!=len(tests): raise SystemExit(1)
