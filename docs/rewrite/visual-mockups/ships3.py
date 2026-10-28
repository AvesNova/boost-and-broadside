# Each ship: list of (half_polygon, mirror) in a frame with nose up (y = -50), x >= 0 for halves.
def R(x0,y0,x1,y1): return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
def H(*p): return (list(p),'half')       # right half from centreline, mirrored to a closed shape
def M(p): return (p,'mirror')            # a separate part, drawn and mirrored
S = {
 "Komet bold":         ("fighter", [H((0,-46),(8,-36),(10,-10),(42,20),(40,28),(10,18),(7,32),(0,34))]),
 "Arrow-Komet":        ("fighter", [H((0,-50),(7,-32),(12,-8),(42,22),(40,30),(10,20),(7,34),(0,36))]),
 "Cranked bold":       ("interceptor", [H((0,-50),(8,-28),(18,0),(44,30),(42,38),(16,28),(0,36))]),
 "Komet long body":    ("fighter", [H((0,-50),(8,-40),(10,-4),(36,22),(34,30),(10,22),(7,38),(0,40))]),
 "Arrow + tail":       ("fighter", [H((0,-50),(7,-30),(16,0),(42,28),(40,34),(12,24),(8,38),(0,40))]),
 "Broad cranked":      ("bomber", [H((0,-46),(12,-30),(22,-4),(48,26),(46,34),(18,26),(10,38),(0,38))]),
 "Komet deep notch":   ("fighter", [H((0,-46),(8,-36),(10,-10),(42,24),(36,28),(10,16),(7,32),(0,34))]),
 "Komet shoulders":    ("heavy fighter", [H((0,-46),(10,-38),(14,-14),(44,20),(42,28),(14,20),(9,34),(0,36))]),
 "Slim arrow":         ("scout", [H((0,-50),(5,-30),(10,-2),(28,28),(26,34),(8,24),(0,30))]),
 "Slim Komet":         ("scout", [H((0,-50),(5,-42),(6,-6),(28,22),(26,28),(6,20),(4,36),(0,38))]),
 "Arrow flat back":    ("fighter / bomber", [H((0,-50),(7,-30),(16,0),(42,32),(10,32),(6,38),(0,38))]),
 "Ogive Komet":        ("fighter", [H((0,-48),(6,-40),(9,-24),(14,-8),(26,6),(42,22),(40,30),(10,20),(7,34),(0,36))]),
 "Heavy Komet 2":      ("bomber", [H((0,-44),(12,-36),(16,-16),(46,18),(44,28),(16,22),(12,36),(0,38))]),
 "Double delta":       ("interceptor", [H((0,-50),(6,-34),(10,-12),(20,6),(46,30),(44,36),(16,30),(8,40),(0,40))]),
 "Komet cropped tips": ("fighter", [H((0,-46),(8,-36),(10,-10),(40,16),(40,28),(10,20),(7,34),(0,36))]),
 "Arrow cropped tips": ("fighter", [H((0,-50),(8,-30),(18,0),(40,24),(40,32),(16,28),(0,34))]),
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
 f'<text x="30" y="46" font-size="26" font-weight="700" fill="{INK}">Ship silhouettes, set 3 (between Komet and Cranked arrow)</text>',
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
open("ships3.svg","w").write("\n".join(o)); print(W,Hh)
