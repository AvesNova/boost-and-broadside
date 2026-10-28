# Each ship: list of (half_polygon, mirror) in a frame with nose up (y = -50), x >= 0 for halves.
def R(x0,y0,x1,y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
def H(*p): return (list(p),'half')       # right half from centreline, mirrored to a closed shape
def M(p): return (p,'mirror')            # a separate part, drawn and mirrored
S = {
 "Chevron":      ("abstract, any role", [H((0,-45),(36,36),(0,14))]),
 "Dart":         ("scout",   [H((0,-50),(13,42),(0,28))]),
 "Kite":         ("fighter", [H((0,-50),(40,32),(12,28),(0,40))]),
 "Arrowhead":    ("fighter", [H((0,-50),(22,10),(42,38),(14,28),(0,40))]),
 "Mustang":      ("fighter (P-51 style)", [H((0,-48),(5,-44),(7,-30),(7,-8),(46,-4),(47,4),(44,10),(7,12),(5,30),(18,33),(18,40),(4,42),(0,46))]),
 "Elliptic":     ("turn fighter (Spitfire style)", [H((0,-48),(5,-43),(7,-14),(20,-15),(32,-13),(41,-8),(45,0),(41,7),(30,10),(18,11),(7,11),(5,30),(16,33),(19,38),(15,42),(3,42),(0,46))]),
 "Radial":       ("brawler (Fw 190 style)", [H((0,-44),(10,-44),(10,-32),(8,-10),(42,-6),(43,4),(40,9),(8,12),(5,30),(17,33),(17,40),(4,42),(0,46))]),
 "Swept":        ("interceptor (jet style)", [H((0,-48),(5,-40),(6,-12),(44,14),(44,20),(6,12),(5,32),(17,42),(17,46),(3,45),(0,48))]),
 "Komet":        ("interceptor (Me 163 style)", [H((0,-40),(6,-34),(8,-10),(40,18),(38,24),(8,16),(5,30),(0,32))]),
 "Glider":       ("scout (long wing)", [H((0,-44),(4,-40),(4,-6),(50,-1),(50,3),(4,6),(3,34),(14,36),(14,40),(2,41),(0,44))]),
 "Canard":       ("fighter (pusher)", [H((0,-46),(5,-38),(6,40),(0,44)), H((0,-30),(17,-24),(17,-19),(0,-21)), H((0,6),(42,22),(42,28),(0,26))]),
 "Flying wing":  ("bomber / scout (Horten style)", [H((0,-30),(48,12),(46,18),(30,13),(14,18),(0,10))]),
 "Twin boom":    ("heavy fighter (P-38 style)", [H((0,-40),(6,-34),(6,8),(0,12)), H((0,-6),(48,-3),(48,4),(0,6)), M(R(15,-36),(25,40)) if False else M(R(15,-36,25,40)), H((0,35),(25,35),(25,43),(0,43))]),
 "Twin engine":  ("fast bomber (Mosquito style)", [H((0,-44),(5,-40),(6,30),(3,44),(0,44)), H((0,-6),(46,-2),(46,5),(0,8)), M(R(10,-24,18,12)), H((0,34),(17,36),(17,42),(0,43))]),
 "Heavy":        ("bomber (4 engines)", [H((0,-46),(6,-40),(7,30),(4,44),(0,46)), H((0,-10),(50,-3),(50,5),(0,10)), M(R(12,-20,18,0)), M(R(29,-16,35,2)), H((0,32),(21,36),(21,42),(0,44))]),
 "Hammer":       ("bomber (abstract, blunt)", [H((0,-40),(14,-38),(18,-10),(40,10),(40,30),(16,26),(12,40),(0,40))]),
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
 f'<text x="30" y="46" font-size="26" font-weight="700" fill="{INK}">Ship silhouettes</text>',
 f'<text x="30" y="74" font-size="15" fill="#5A6272">Nose up. Large, plus in-game test sizes (24 px and 12 px) in both team colours.</text>']
for i,n in enumerate(names):
    role,parts=S[n]; x0=20+(i%cols)*cw; y0=95+(i//cols)*ch
    o.append(f'<rect x="{x0+5}" y="{y0}" width="{cw-10}" height="{ch-12}" rx="10" fill="#F6F7F9" stroke="#D5DAE1"/>')
    o.append(draw(parts,x0+100,y0+105,140,INK))
    for j,(sz,c) in enumerate([(24,BLUE),(24,ORANGE),(12,BLUE),(12,ORANGE)]):
        o.append(draw(parts,x0+235+(j%2)*55,y0+70+(j//2)*55,sz,c))
    o.append(f'<text x="{x0+22}" y="{y0+205}" font-size="17" font-weight="700" fill="{INK}">{i+1}. {n}</text>')
    o.append(f'<text x="{x0+22}" y="{y0+226}" font-size="13" fill="#5A6272">{role}</text>')
o.append('</svg>')
open("ships.svg","w").write("\n".join(o)); print(W,Hh)
