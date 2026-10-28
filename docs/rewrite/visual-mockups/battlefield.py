import math
W,H=1600,980
BLUE,ORANGE="#2F6FD0","#E07B2E"
INK="#2A2F3A"
o=[]
a=o.append
a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Inter,Helvetica,Arial,sans-serif">')
# arena: stadium from x 400..1200 at y 430, r 320
AX0,AX1,AY,AR=400,1200,430,320
def stadium(r):
    return f'M {AX0} {AY-r} L {AX1} {AY-r} A {r} {r} 0 0 1 {AX1} {AY+r} L {AX0} {AY+r} A {r} {r} 0 0 1 {AX0} {AY-r} Z'
a('''<defs>
<pattern id="hatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="12" height="12" fill="#B9BFC9"/><line x1="0" y1="0" x2="0" y2="12" stroke="#A3AAB6" stroke-width="5"/></pattern>
<mask id="fog"><rect width="1600" height="980" fill="white"/>''')
blue_vision=[(700,420,190),(760,510,190),(560,480,170),(900,610,170),(1010,290,300),(180,430,230),(430,300,160),(470,590,160),(330,470,160)]
for x,y,r in blue_vision: a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="black"/>')
a(f'</mask><clipPath id="arena"><path d="{stadium(AR)}"/></clipPath></defs>')
a(f'<rect width="{W}" height="{H}" fill="#E9ECF0"/>')
# outside: damage zone hatch, then thick-air band
a('<g transform="translate(800 440) scale(0.82) translate(-800 -430)">')
a(f'<path d="{stadium(AR+90)}" fill="url(#hatch)"/>')
a(f'<path d="{stadium(AR+45)}" fill="#C7CCD5"/>')
a(f'<path d="{stadium(AR)}" fill="#F6F7F9"/>')
# fields (capsules) inside arena
fields=[  # x0,y0,x1,y1,r,kind
 (560,240,700,320,42,'thick'),(860,170,1010,215,34,'thick'),(905,555,1060,515,52,'thick'),
 (610,660,990,700,28,'thin'),(745,395,830,370,44,'thin'),(800,455,880,480,40,'thick'),
 (1260,640,1330,560,36,'thin'),(300,250,360,220,30,'thin'),
]
a('<g clip-path="url(#arena)">')
for x0,y0,x1,y1,r,k in fields:
    col,op=("#7D869A",.30) if k=='thick' else ("#8CCFE3",.45)
    a(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{col}" stroke-opacity="{op}" stroke-width="{2*r}" stroke-linecap="round"/>')
a('</g>')
# range circles
def ring(x,y,r,c,dash="6 6",op=.35): a(f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="{c}" stroke-opacity="{op}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
# bases
def base(x,y,c):
    ring(x,y,170,c,"10 6",.35)
    a(f'<rect x="{x-26}" y="{y-26}" width="52" height="52" rx="6" fill="{c}"/>')
    a(f'<rect x="{x-12}" y="{y-12}" width="24" height="24" rx="3" fill="#F6F7F9"/>')
def turret(x,y,c,dead=False):
    ring(x,y,150,c)
    pts=" ".join(f"{x+14*math.cos(math.pi/3*i+math.pi/6):.1f},{y+14*math.sin(math.pi/3*i+math.pi/6):.1f}" for i in range(6))
    if dead: a(f'<polygon points="{pts}" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="3 3"/>')
    else: a(f'<polygon points="{pts}" fill="{c}"/><circle cx="{x}" cy="{y}" r="4" fill="#F6F7F9"/>')
base(180,430,BLUE); base(1420,430,ORANGE)
for t in [(430,300),(470,590),(330,470)]: turret(*t,BLUE)
for t in [(1185,255),(1135,565)]: turret(*t,ORANGE)
turret(1300,390,ORANGE,dead=True)
# ships
SH={'fighter':"M 13 0 L -9 -8 L -4 0 L -9 8 Z",
    'bomber':"M 12 0 L 4 -5 L 0 -15 L -6 -15 L -5 -4 L -11 -3 L -11 3 L -5 4 L -6 15 L 0 15 L 4 5 Z",
    'scout':"M 16 0 L -8 -4 L -5 0 L -8 4 Z"}
def trail(pts,c):
    d="M "+" L ".join(f"{x} {y}" for x,y in pts)
    a(f'<path d="{d}" fill="none" stroke="{c}" stroke-opacity=".35" stroke-width="2" stroke-linecap="round"/>')
def ship(x,y,hdg,kind,c,ghost=False):
    if ghost:
        a(f'<path d="{SH[kind]}" transform="translate({x} {y}) rotate({hdg}) scale(1.3)" fill="none" stroke="{c}" stroke-width="1.2" stroke-dasharray="2 2" opacity=".7"/>')
    else:
        a(f'<path d="{SH[kind]}" transform="translate({x} {y}) rotate({hdg}) scale(1.3)" fill="{c}"/>')
# dogfight in the middle
trail([(560,520),(620,470),(665,435),(700,420)],BLUE); ship(700,420,-12,'fighter',BLUE)
trail([(960,330),(910,360),(880,400),(862,428)],ORANGE); ship(860,432,150,'fighter',ORANGE)
trail([(640,600),(700,560),(740,525),(760,510)],BLUE); ship(760,510,-35,'fighter',BLUE)
# fighter tracers with spread
import random; random.seed(3)
for i in range(7):
    t=60+i*22; ang=math.radians(-12+random.uniform(-3,3))
    x=700+t*math.cos(ang); y=420+t*math.sin(ang)
    a(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x+9*math.cos(ang):.1f}" y2="{y+9*math.sin(ang):.1f}" stroke="{BLUE}" stroke-width="2.4" stroke-linecap="round"/>')
# bomber standing off an orange turret, homing shell curving in
trail([(780,700),(840,660),(880,625),(900,610)],BLUE); ship(900,610,-30,'bomber',BLUE)
a(f'<path d="M 930 593 Q 1010 560 1050 575 T 1122 563" fill="none" stroke="{BLUE}" stroke-width="1.5" stroke-dasharray="2 6" stroke-linecap="round"/>')
a(f'<circle cx="1050" cy="575" r="5" fill="{BLUE}"/>')
# second blue bomber heading out
trail([(380,520),(440,500),(500,490),(560,480)],BLUE); ship(560,480,-8,'bomber',BLUE)
# blue scout far forward
trail([(860,380),(920,340),(970,305),(1010,290)],BLUE); ship(1010,290,-20,'scout',BLUE)
# orange ships visible to blue
trail([(1180,180),(1120,215),(1085,240)],ORANGE); ship(1085,240,155,'fighter',ORANGE)
trail([(1200,440),(1150,470),(1120,480)],ORANGE); ship(1110,482,165,'scout',ORANGE)
# flak from orange turret at blue scout? scout is outside its range; draw flak puffs at range edge
# orange ships hidden by fog (drawn as ghosts for the concept)
ship(1290,520,190,'bomber',ORANGE,ghost=True); ship(1240,300,200,'fighter',ORANGE,ghost=True)
ship(620,180,10,'fighter',ORANGE,ghost=True)
# fog (blue team's view)
a(f'<g clip-path="url(#arena)"><rect width="{W}" height="{H}" fill="#2B3240" fill-opacity=".16" mask="url(#fog)"/></g>')
# arena edge line
a(f'<path d="{stadium(AR)}" fill="none" stroke="#9AA2B0" stroke-width="1.5"/>')
a('</g>')
# title
a(f'<text x="40" y="52" font-size="26" font-weight="700" fill="{INK}">Boost and Broadside — battlefield layout (blue team view)</text>')
a(f'<text x="40" y="80" font-size="15" fill="#5A6272">Concept only. Positions, ranges and sizes are illustrative, not to scale.</text>')
# legend
LY=835
a(f'<rect x="40" y="{LY-30}" width="1520" height="150" rx="10" fill="#FFFFFF" stroke="#D5DAE1"/>')
def lt(x,y,s): a(f'<text x="{x}" y="{y+5}" font-size="14" fill="{INK}">{s}</text>')
col=[70,370,680,1000,1290]
# col1 ships
y=LY
a(f'<path d="{SH["fighter"]}" transform="translate({col[0]} {y}) scale(1.3)" fill="{BLUE}"/>'); lt(col[0]+25,y,"Fighter")
a(f'<path d="{SH["bomber"]}" transform="translate({col[0]} {y+35}) scale(1.3)" fill="{BLUE}"/>'); lt(col[0]+25,y+35,"Bomber")
a(f'<path d="{SH["scout"]}" transform="translate({col[0]} {y+70}) scale(1.3)" fill="{BLUE}"/>'); lt(col[0]+25,y+70,"Scout")
a(f'<rect x="{col[0]+120}" y="{y-8}" width="16" height="16" rx="3" fill="{BLUE}"/>'); lt(col[0]+145,y,"Blue team")
a(f'<rect x="{col[0]+120}" y="{y+27}" width="16" height="16" rx="3" fill="{ORANGE}"/>'); lt(col[0]+145,y+35,"Orange team")
# col2 structures
a(f'<rect x="{col[1]-14}" y="{y-14}" width="28" height="28" rx="4" fill="{INK}"/><rect x="{col[1]-6}" y="{y-6}" width="12" height="12" rx="2" fill="#fff"/>'); lt(col[1]+28,y,"Base (win: destroy it)")
pts=" ".join(f"{col[1]+11*math.cos(math.pi/3*i+math.pi/6):.1f},{y+35+11*math.sin(math.pi/3*i+math.pi/6):.1f}" for i in range(6))
a(f'<polygon points="{pts}" fill="{INK}"/>'); lt(col[1]+28,y+35,"Turret (flak)")
pts=" ".join(f"{col[1]+11*math.cos(math.pi/3*i+math.pi/6):.1f},{y+70+11*math.sin(math.pi/3*i+math.pi/6):.1f}" for i in range(6))
a(f'<polygon points="{pts}" fill="none" stroke="{INK}" stroke-width="2" stroke-dasharray="3 3"/>'); lt(col[1]+28,y+70,"Destroyed turret")
# col3 fields
a(f'<line x1="{col[2]-12}" y1="{y}" x2="{col[2]+12}" y2="{y}" stroke="#7D869A" stroke-opacity=".3" stroke-width="20" stroke-linecap="round"/>'); lt(col[2]+35,y,"Thick air (slow, bends paths)")
a(f'<line x1="{col[2]-12}" y1="{y+35}" x2="{col[2]+12}" y2="{y+35}" stroke="#8CCFE3" stroke-opacity=".45" stroke-width="20" stroke-linecap="round"/>'); lt(col[2]+35,y+35,"Thin air (fast)")
a(f'<rect x="{col[2]-22}" y="{y+60}" width="22" height="20" fill="#C7CCD5"/><rect x="{col[2]}" y="{y+60}" width="22" height="20" fill="url(#hatch)"/>'); lt(col[2]+35,y+70,"Edge: thick air, then damage")
# col4 combat
for i in range(3): a(f'<line x1="{col[3]-14+i*11}" y1="{y+(i-1)*2}" x2="{col[3]-7+i*11}" y2="{y+(i-1)*2}" stroke="{BLUE}" stroke-width="2.4" stroke-linecap="round"/>')
lt(col[3]+35,y,"Fighter tracers (random spread)")
a(f'<path d="M {col[3]-16} {y+40} Q {col[3]} {y+25} {col[3]+16} {y+35}" fill="none" stroke="{BLUE}" stroke-width="1.5" stroke-dasharray="2 5"/><circle cx="{col[3]}" cy="{y+31}" r="4.5" fill="{BLUE}"/>'); lt(col[3]+35,y+35,"Bomber shell (locks, homes)")
a(f'<circle cx="{col[3]}" cy="{y+70}" r="13" fill="none" stroke="{INK}" stroke-width="1.5" stroke-dasharray="6 6"/>'); lt(col[3]+35,y+70,"Weapon range")
# col5 fog
a(f'<rect x="{col[4]-14}" y="{y-10}" width="28" height="20" fill="#2B3240" fill-opacity=".16"/>'); lt(col[4]+28,y,"Fog (team-shared vision)")
a(f'<path d="{SH["fighter"]}" transform="translate({col[4]} {y+35}) scale(1.3)" fill="none" stroke="{ORANGE}" stroke-width="1.2" stroke-dasharray="2 2"/>'); lt(col[4]+28,y+35,"Hidden enemy (not shown in game)")
a(f'<path d="M {col[4]-16} {y+70} L {col[4]+16} {y+70}" stroke="{BLUE}" stroke-opacity=".35" stroke-width="2"/>'); lt(col[4]+28,y+70,"Recent path")
a('</svg>')
open("battlefield.svg","w").write("\n".join(o))
