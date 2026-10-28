# Each ship: list of (half_polygon, mirror) in a frame with nose up (y = -50), x >= 0 for halves.
def R(x0,y0,x1,y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
def H(*p): return (list(p),'half')       # right half from centreline, mirrored to a closed shape
def M(p): return (p,'mirror')            # a separate part, drawn and mirrored
import math
def curve(f,n=14): return [f(i/n) for i in range(n+1)]
def bez(p0,p1,p2,n=10): return [((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t*t*p2[0],(1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t*t*p2[1]) for t in [i/n for i in range(n+1)]]
leaf=curve(lambda t:(26*(t**0.85)*((1-t)**0.45)/0.62 if 0<t<1 else 0, -50+90*t))
rdelta=[(0,-50)]+[(40-6+6*math.cos(a),26+4*math.sin(a)) for a in [math.radians(d) for d in range(-60,91,30)]]+[(0,30)]
cchev=bez((0,-48),(14,-2),(40,36))+bez((40,36),(18,26),(0,12))[1:]
ogee=curve(lambda t:(44*(t**1.6)*(1.2-0.2*t), -50+80*t))+[(30,34),(0,34)]
def F(p): return (p,'full')
def HOLE(p): return (p,'hole')
def circ(cx,cy,r,n=28): return [(cx+r*math.sin(2*math.pi*i/n), cy-r*math.cos(2*math.pi*i/n)) for i in range(n)]
comet=[(14*math.sin(a),-30-14*math.cos(a)) for a in [math.radians(d) for d in range(0,101,20)]]+[(12,-20),(6,10),(0,46)]
S = {
 "Squid":          ("wacky", [H((0,-50),(14,-30),(16,0),(12,14),(22,46),(14,46),(6,22),(0,24))]),
 "Fish":           ("wacky", [H((0,-50),(10,-36),(14,-14),(12,8),(6,22),(22,46),(10,46),(0,34))]),
 "Lightning":      ("asymmetric", [F([(10,-50),(-16,4),(-2,4),(-12,44),(12,44),(18,-6),(4,-6)])]),
 "Comet":          ("round nose", [H(*comet)]),
 "Oblique wing":   ("asymmetric (AD-1 style)", [F([(0,-50),(5,-40),(5,36),(12,44),(-12,44),(-5,36),(-5,-40)]), F([(-40,12),(36,-20),(40,-12),(-38,20)])]),
 "Ring wing":      ("annular wing", [F(circ(0,8,30)), HOLE(circ(0,8,17)), H((0,-50),(6,-30),(6,40),(0,44))]),
 "Bat":            ("scalloped", [H((0,-40),(8,-34),(16,-30),(40,-24),(48,-6),(40,4),(36,0),(30,10),(22,6),(16,16),(8,10),(0,22))]),
 "Scarab":         ("beetle", [H((0,-50),(4,-38),(14,-32),(22,-14),(24,10),(18,30),(6,40),(3,12),(0,12))]),
 "Keyhole":        ("orb + spike", [F(circ(0,14,22)), H((0,-50),(14,0),(0,0))]),
 "Quill":          ("asymmetric feather", [F([(0,-50),(14,-20),(16,10),(8,34),(2,46),(-2,46),(-4,20),(-8,-10)])]),
 "Pterodactyl":    ("crest", [H((0,-50),(4,-34),(6,-24),(40,-10),(48,6),(30,0),(10,6),(6,20),(0,28)), F([(-3,-38),(-16,-48),(-5,-30)])]),
 "Origami crane":  ("asymmetric folds", [F([(0,-50),(6,-30),(42,-28),(10,-8),(22,30),(0,12),(-22,30),(-10,-8),(-38,-4),(-6,-30)])]),
 "Saucer beak":    ("disc", [F(circ(0,10,26)), H((0,-50),(11,-12),(0,-12))]),
 "Catamaran":      ("twin hull", [M([(14,-46),(20,-20),(20,40),(8,40),(8,-20)]), H((0,-4),(24,-4),(24,10),(0,16))]),
 "Big swallowtail":("kite", [H((0,-50),(20,-14),(10,0),(26,46),(4,10),(0,12))]),
 "Flame":          ("wavy exhaust", [H((0,-50),(10,-34),(12,10),(20,22),(12,26),(16,40),(6,34),(0,46))]),
}
def polys(parts):
    out=[]
    for p,kind in parts:
        if kind=='half': out.append(p+[(-x,y) for x,y in reversed(p)])
        elif kind in ('full','hole'): out.append(p)
        else: out.append(p); out.append([(-x,y) for x,y in p])
    return out
def path(parts,cx,cy,size):
    ps=polys(parts); xs=[x for p in ps for x,_ in p]; ys=[y for p in ps for _,y in p]
    s=size/max(max(xs)-min(xs),max(ys)-min(ys)); mx=(max(xs)+min(xs))/2; my=(max(ys)+min(ys))/2
    return ps,s,mx,my
def draw(parts,cx,cy,size,fill):
    ps,s,mx,my=path(parts,cx,cy,size)
    kinds=[]
    for p,k in parts: kinds += [k]*(2 if k=='mirror' else 1)
    fills=["#F6F7F9" if k=='hole' else fill for k in kinds]
    return "".join(f'<path d="M '+" L ".join(f"{cx+(x-mx)*s:.2f} {cy+(y-my)*s:.2f}" for x,y in p)+f' Z" fill="{fl}"/>' for p,fl in zip(ps,fills))
BLUE,ORANGE,INK="#2F6FD0","#E07B2E","#2A2F3A"
cols,cw,ch=4,380,250
names=list(S); rows=(len(names)+cols-1)//cols
W,Hh=cols*cw+40,rows*ch+110
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inter,Helvetica,Arial,sans-serif"><rect width="100%" height="100%" fill="#E9ECF0"/>',
 f'<text x="30" y="46" font-size="26" font-weight="700" fill="{INK}">Ship silhouettes, set 6 (wacky)</text>',
 f'<text x="30" y="74" font-size="15" fill="#5A6272">Nose up. Large, plus 24 px and 12 px in both team colours; bottom right: 12 px at other headings.</text>']
for i,n in enumerate(names):
    role,parts=S[n]; x0=20+(i%cols)*cw; y0=95+(i//cols)*ch
    o.append(f'<rect x="{x0+5}" y="{y0}" width="{cw-10}" height="{ch-12}" rx="10" fill="#F6F7F9" stroke="#D5DAE1"/>')
    o.append(draw(parts,x0+100,y0+105,140,INK))
    for j,(sz,c) in enumerate([(24,BLUE),(24,ORANGE),(12,BLUE),(12,ORANGE)]):
        pass
    for j,(sz,c) in enumerate([(24,BLUE),(24,ORANGE),(12,BLUE),(12,ORANGE)]):
        o.append(draw(parts,x0+235+(j%2)*55,y0+70+(j//2)*55,sz,c))
    for j,ang in enumerate([60,150,240]):
        cx,cy=x0+225+j*40,y0+175
        o.append(f'<g transform="rotate({ang} {cx} {cy})">'+draw(parts,cx,cy,12,BLUE)+'</g>')
    o.append(f'<text x="{x0+22}" y="{y0+205}" font-size="17" font-weight="700" fill="{INK}">{i+1}. {n}</text>')
    o.append(f'<text x="{x0+22}" y="{y0+226}" font-size="13" fill="#5A6272">{role}</text>')
o.append('</svg>')
open("ships6.svg","w").write("\n".join(o)); print(W,Hh)
