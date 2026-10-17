import numpy as np, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Patch
from matplotlib.lines import Line2D
L=16
P=np.array([(q+r/2, r*math.sqrt(3)/2) for q in range(-L,L+1) for r in range(-L,L+1)]); d=np.hypot(*P.T)
ang=np.degrees(np.arctan2(P[:,1],P[:,0]))%60
shell=np.abs(d-9.5394)<1e-3
pick=np.isclose(ang,5.2087,atol=1e-3)|np.isclose(ang,26.9955,atol=1e-3)|np.isclose(ang,33.0045,atol=1e-3)
d331=(d<=9.17)|(shell&pick)
panels=[("331 (partial shell, shown before)",d331,7.83),
        ("313: nearest full disc",d<=9.166,7.68),
        ("367: one more layer (radius 10)",d<=10.001,8.44),
        ("439: two more layers (radius 11)",d<=11.001,9.41)]
IN="#4C78A8"; BUF="#C9D6E8"; RING="#D1495B"; Rplay=2600.0
fig,axs=plt.subplots(1,4,figsize=(19,5.6),facecolor="white")
for axp,(title,G,Rsp) in zip(axs,panels):
    a=Rplay/Rsp; nin=0
    for (x,y),g,dd in zip(P,G,d):
        if not g: continue
        inside=dd*a<=Rplay; nin+=inside
        axp.add_patch(RegularPolygon((x*a,y*a),6,radius=a/math.sqrt(3),orientation=0,
                      facecolor=IN if inside else BUF,edgecolor="white",lw=0.5))
    axp.add_patch(Circle((0,0),Rplay,fill=False,ec=RING,lw=2.2))
    lim=3500; axp.set_xlim(-lim,lim); axp.set_ylim(-lim,lim); axp.set_aspect("equal")
    eta=math.pi*Rsp**2/(G.sum()*math.sqrt(3)/2)
    axp.set_title(f"{title}\n{a:.0f} m cells, {nin} inside, area efficiency {eta:.2f}",fontsize=11)
    axp.set_xticks([-2600,0,2600]); axp.set_yticks([-2600,0,2600]); axp.tick_params(labelsize=8)
    for s in axp.spines.values(): s.set_visible(False)
fig.legend(handles=[Patch(fc=IN,label="cell centre inside playable circle"),Patch(fc=BUF,label="buffer cell"),
    Line2D([0],[0],color=RING,lw=2.2,label="playable boundary")],loc="lower center",ncol=3,frameon=False,fontsize=10)
fig.suptitle("Disc density grids with a 2σ buffer (2.3% worst spill), metres at the 5v5 playable radius 2600 m",fontsize=13)
fig.tight_layout(rect=(0,0.06,1,0.95))
fig.savefig("/home/user/boost-and-broadside/docs/rewrite/figures/density-disc-sizes.png",dpi=120)
