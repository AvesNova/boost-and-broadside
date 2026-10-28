# Each ship: list of (half_polygon, mirror) in a frame with nose up (y = -50), x >= 0 for halves.
def R(x0,y0,x1,y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
def H(*p): return (list(p),'half')       # right half from centreline, mirrored to a closed shape
def M(p): return (p,'mirror')            # a separate part, drawn and mirrored
S = {
 "Komet":          ("reference (old 9)", [H((0,-40),(6,-34),(8,-10),(40,18),(38,24),(8,16),(5,30),(0,32))]),
 "Long Komet":     ("fighter", [H((0,-50),(5,-42),(6,-6),(34,22),(32,28),(6,22),(4,40),(0,42))]),
 "Wide Komet":     ("bomber", [H((0,-36),(7,-30),(9,-14),(50,14),(48,22),(9,14),(6,26),(0,28))]),
 "Fork tail":      ("fighter", [H((0,-44),(6,-38),(8,-10),(40,18),(38,24),(8,16),(7,34),(3,30),(0,28))]),
 "Canard Komet":   ("fighter", [H((0,-44),(6,-38),(8,-10),(40,18),(38,24),(8,16),(5,30),(0,32)), H((0,-30),(15,-21),(15,-17),(0,-20))]),
 "Crescent":       ("turn fighter", [H((0,-42),(6,-36),(8,-14),(22,-6),(34,6),(42,22),(40,28),(30,18),(18,14),(8,16),(5,30),(0,32))]),
 "Tip fins":       ("fighter", [H((0,-40),(6,-34),(8,-10),(38,14),(42,6),(45,26),(36,22),(8,16),(5,30),(0,32))]),
 "Cranked arrow":  ("interceptor", [H((0,-50),(6,-30),(16,0),(42,30),(40,36),(14,26),(0,34))]),
 "Swallowtail":    ("scout", [H((0,-50),(10,20),(14,44),(4,32),(0,34))]),
 "Needle":         ("scout", [H((0,-50),(3,-44),(4,-4),(22,14),(22,18),(4,14),(4,36),(10,44),(10,46),(0,44))]),
 "Heavy Komet":    ("bomber", [H((0,-36),(10,-34),(14,-20),(14,-6),(46,20),(44,28),(14,22),(12,36),(0,38))]),
 "Spine chevron":  ("fighter", [H((0,-50),(6,-40),(7,-20),(38,30),(34,36),(8,18),(0,22))]),
 "Forward swept":  ("odd one out", [H((0,-46),(6,-40),(7,-6),(36,-22),(40,-18),(10,16),(6,32),(14,40),(14,44),(0,44))]),
 "Tailless delta": ("fighter / bomber", [H((0,-50),(4,-40),(6,-20),(40,30),(8,30),(5,38),(0,38))]),
 "Gull":           ("turn fighter", [H((0,-44),(6,-36),(8,-12),(26,-2),(44,20),(40,26),(26,12),(10,20),(6,32),(0,30))]),
 "Twin prong":     ("bomber", [H((0,-30),(10,-24),(16,-46),(22,-20),(48,24),(44,30),(20,22),(10,34),(0,30))]),
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
 f'<text x="30" y="46" font-size="26" font-weight="700" fill="{INK}">Ship silhouettes, set 2 (Komet family)</text>',
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
open("ships2.svg","w").write("\n".join(o)); print(W,Hh)
