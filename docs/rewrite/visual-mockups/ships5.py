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
S = {
 "Hawk A (slim)":         ("fighter", [H((0,-48),(4,-40),(5,-24),(22,-18),(44,-4),(42,2),(22,0),(6,6),(12,32),(0,28))]),
 "Hawk B (swept)":        ("fighter", [H((0,-48),(4,-40),(5,-22),(20,-14),(44,6),(41,11),(20,2),(6,8),(12,32),(0,28))]),
 "Hawk C (long nose)":    ("fighter", [H((0,-50),(4,-42),(5,-20),(22,-14),(44,0),(42,6),(20,2),(5,10),(9,34),(0,30))]),
 "Hawk D (gull)":         ("turn fighter", [H((0,-48),(4,-40),(5,-22),(18,-20),(30,-12),(46,4),(43,9),(28,-2),(16,-2),(5,8),(11,32),(0,28))]),
 "Manta A (long nose)":   ("bomber", [H((0,-52),(5,-42),(7,-26),(22,-8),(48,12),(40,20),(20,22),(8,26),(0,24))]),
 "Manta B (concave)":     ("bomber", [H(*([(0,-50)]+bez((6,-34),(14,6),(48,12))+[(40,20),(20,22),(8,26),(0,24)]))]),
 "Manta C (both)":        ("bomber", [H(*([(0,-52),(5,-44)]+bez((6,-28),(16,8),(48,14))+[(40,22),(20,22),(8,28),(0,26)]))]),
 "Manta D (notched)":     ("bomber", [H(*([(0,-50)]+bez((6,-36),(12,4),(46,10))+[(40,20),(22,18),(10,26),(0,18)]))]),
 "Kite A (notch)":        ("scout", [H((0,-50),(22,12),(8,28),(0,22))]),
 "Kite B (cranked)":      ("scout", [H((0,-50),(14,-14),(24,12),(0,32))]),
 "Kite C (cranked+notch)":("scout", [H((0,-50),(12,-16),(24,10),(10,28),(0,22))]),
 "Kite D (tips)":         ("scout", [H((0,-50),(18,0),(28,18),(16,16),(0,32))]),
 "Nib A (waist+notch)":   ("scout", [H((0,-50),(16,-14),(18,0),(9,14),(16,32),(12,42),(0,36))]),
 "Nib B (flared fins)":   ("scout", [H((0,-50),(14,-18),(16,2),(8,16),(18,38),(10,40),(0,34))]),
 "Nib C (shoulders)":     ("scout", [H((0,-50),(12,-22),(20,-6),(14,0),(10,16),(16,34),(8,42),(0,38))]),
 "Nib D (wide flare)":    ("scout", [H((0,-50),(13,-20),(15,0),(7,14),(20,40),(0,32))]),
}
def polys(parts):
    out=[]
    for p,kind in parts:
        if kind=='half': out.append(p+[(-x,y) for x,y in reversed(p)])
        else: out.append(p); out.append([(-x,y) for x,y in p])
    return out
def path(parts,cx,cy,size):
    ps=polys(parts); xs=[x for p in ps for x,_ in p]; ys=[y for p in ps for _,y in p]
    s=size/max(max(xs)-min(xs),max(ys)-min(ys)); mx=(max(xs)+min(xs))/2; my=(max(ys)+min(ys))/2
    return ps,s,mx,my
def draw(parts,cx,cy,size,fill):
    ps,s,mx,my=path(parts,cx,cy,size)
    return "".join(f'<path d="M '+" L ".join(f"{cx+(x-mx)*s:.2f} {cy+(y-my)*s:.2f}" for x,y in p)+f' Z" fill="{fill}"/>' for p in ps)
BLUE,ORANGE,INK="#2F6FD0","#E07B2E","#2A2F3A"
cols,cw,ch=4,380,250
names=list(S); rows=(len(names)+cols-1)//cols
W,Hh=cols*cw+40,rows*ch+110
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inter,Helvetica,Arial,sans-serif"><rect width="100%" height="100%" fill="#E9ECF0"/>',
 f'<text x="30" y="46" font-size="26" font-weight="700" fill="{INK}">Ship silhouettes, set 5 (Hawk, Manta, Kite, Nib variations)</text>',
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
open("ships5.svg","w").write("\n".join(o)); print(W,Hh)
