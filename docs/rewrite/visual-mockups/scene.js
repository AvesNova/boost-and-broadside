// Test scene for the perception.py rules, adapted to capsule fields.
const SW=1200, SH=760, R=220, ARENA=[360,390,840,390,300];
// fields: [x0,y0,x1,y1, outer radius, band width, kind]; opaque core = outer - band/2 (perception.py)
const FIELDS=[
 [330,250,420,300,50,24,'thick',2],   // F1 casts the shadow, n=2
 [640,470,780,520,55,24,'thick',1],   // F2 no ally inside, n=1.41
 [560,180,760,140,40,24,'thin',-2],   // F3 ally inside, n=0.5
 [280,560,470,590,38,24,'thin',-1],   // F4 ally inside, n=0.71
 [450,520,500,650,45,24,'thick',2],   // F5 overlaps F4, n=2
 [820,230,930,170,35,24,'thick',1],   // F6, n=1.41
].map(f=>({x0:f[0],y0:f[1],x1:f[2],y1:f[3],r:f[4],w:f[5],core:f[4]-f[5]/2,kind:f[6],lv:f[7]}));
const ALLIES=[ // alive observers
 {id:'A1',x:220,y:330,h:5},{id:'A2',x:640,y:165,h:-15},{id:'A3',x:330,y:570,h:10},{id:'A4',x:230,y:470,h:0}];
const RESPAWN={x:150,y:390,t:'3 s'};
const ENEMIES=[
 {n:1, x:330,y:420,h:200},           // visible, clear line
 {n:2, x:980,y:250,h:190},           // out of range
 {n:3, x:430,y:320,h:170},           // behind F1 (shadow)
 {n:4, x:700,y:495,h:160},           // inside F2, no ally inside
 {n:5, x:720,y:150,h:190},           // inside F3 with A2
 {n:6, x:354,y:313,h:150},           // in F1's see-through band
 {n:7, x:465,y:585,h:200},           // in F4/F5 overlap, ally only in F4
 {n:8, x:420,y:470,h:180},           // seen by A4 only (team-shared)
 {n:9, x:600,y:400,h:180,shoot:1},   // firing: revealed
 {n:10,x:1010,y:390,h:180,spawn:1},  // just spawned: revealed
 {n:11,x:640,y:232,h:200},           // next to A2, who is inside a core
];
const BASES=[{x:150,y:390,team:0},{x:1050,y:390,team:1}];
const TURRETS=[{x:300,y:220,team:0},{x:300,y:650,team:0},{x:920,y:310,team:1},{x:900,y:580,team:1}];
const OWN_BULLETS=[[262,329],[292,328],[322,327],[352,326],[382,325],[412,323],[272,470],[302,470],[332,470],[362,470]];
const ENEMY_BULLETS=[[560,400],[520,400],[480,400],[440,400],[400,400],[360,400]];
// geometry
function segDist(px,py,ax,ay,bx,by){const dx=bx-ax,dy=by-ay,l=dx*dx+dy*dy||1;let t=((px-ax)*dx+(py-ay)*dy)/l;t=Math.max(0,Math.min(1,t));return Math.hypot(px-ax-t*dx,py-ay-t*dy)}
function segSeg(ax,ay,bx,by,cx,cy,dx,dy){const cr=(px,py,qx,qy,rx,ry)=>(qx-px)*(ry-py)-(qy-py)*(rx-px);
 if(cr(ax,ay,bx,by,cx,cy)*cr(ax,ay,bx,by,dx,dy)<0&&cr(cx,cy,dx,dy,ax,ay)*cr(cx,cy,dx,dy,bx,by)<0)return 0;
 return Math.min(segDist(ax,ay,cx,cy,dx,dy),segDist(bx,by,cx,cy,dx,dy),segDist(cx,cy,ax,ay,bx,by),segDist(dx,dy,ax,ay,bx,by))}
const inCore=(x,y,f)=>segDist(x,y,f.x0,f.y0,f.x1,f.y1)<f.core;
const inArena=(x,y)=>segDist(x,y,ARENA[0],ARENA[1],ARENA[2],ARENA[3])<ARENA[4];
function los(o,x,y){for(const f of FIELDS){const a=inCore(o.x,o.y,f),b=inCore(x,y,f);if(a&&b)continue;
 if(segSeg(o.x,o.y,x,y,f.x0,f.y0,f.x1,f.y1)<f.core)return false}return true}
function seenBy(x,y){return ALLIES.filter(a=>Math.hypot(x-a.x,y-a.y)<=R&&los(a,x,y))}
const enemyVisible=e=>e.shoot||e.spawn||seenBy(e.x,e.y).length>0;
const fieldLifted=f=>ALLIES.some(a=>inCore(a.x,a.y,f));
let MASK=null;
function mask(){if(MASK)return MASK;const m=document.createElement('canvas');m.width=SW;m.height=SH;const c=m.getContext('2d');const img=c.createImageData(SW,SH);
 for(let y=0;y<SH;y++)for(let x=0;x<SW;x++){if(!inArena(x,y))continue;let v=false;for(const a of ALLIES){if((x-a.x)**2+(y-a.y)**2<=R*R&&los(a,x,y)){v=true;break}}if(v)img.data[4*(y*SW+x)+3]=255}
 c.putImageData(img,0,0);MASK=m;return m}
// drawing helpers
const SHIP=[[0,-50],[7,-32],[12,-8],[42,22],[40,30],[10,20],[7,34],[0,36]];
function shipPath(c,x,y,deg,s){c.save();c.translate(x,y);c.rotate((deg+90)*Math.PI/180);c.scale(s,s);c.beginPath();
 SHIP.concat(SHIP.slice().reverse().map(([a,b])=>[-a,b])).forEach(([a,b],i)=>i?c.lineTo(a,b):c.moveTo(a,b));c.closePath();c.restore()}
function capPath(c,x0,y0,x1,y1,r){const a=Math.atan2(y1-y0,x1-x0);c.beginPath();c.arc(x1,y1,r,a-Math.PI/2,a+Math.PI/2);c.arc(x0,y0,r,a+Math.PI/2,a+3*Math.PI/2);c.closePath()}
function hexPath(c,x,y,r){c.beginPath();for(let i=0;i<6;i++){const a=Math.PI/3*i+Math.PI/6;c.lineTo(x+r*Math.cos(a),y+r*Math.sin(a))}c.closePath()}
function mkPattern(fn,size=12){const p=document.createElement('canvas');p.width=p.height=size;fn(p.getContext('2d'),size);return p}
// draw terrain (always known) for a style
function drawTerrain(c,S){
 c.fillStyle=S.outside;c.fillRect(0,0,SW,SH);
 const A=ARENA;capPath(c,A[0],A[1],A[2],A[3],A[4]+(S.damageR||60));c.fillStyle=S.damage;c.fill();
 if(S.damageHatch){c.save();c.clip();c.fillStyle=c.createPattern(mkPattern((p,s)=>{p.strokeStyle=S.damageHatch;p.lineWidth=3;p.beginPath();p.moveTo(0,s);p.lineTo(s,0);p.stroke()},12),'repeat');c.fillRect(0,0,SW,SH);c.restore()}
 capPath(c,A[0],A[1],A[2],A[3],A[4]+(S.edgeR||30));c.fillStyle=S.edge;c.fill();
 capPath(c,A[0],A[1],A[2],A[3],A[4]);c.fillStyle=S.arena;c.fill();if(S.arenaLine){c.strokeStyle=S.arenaLine;c.lineWidth=1.5;c.stroke()}
 if(S.grid){c.save();capPath(c,A[0],A[1],A[2],A[3],A[4]);c.clip();c.strokeStyle=S.grid;c.lineWidth=1;for(let x=0;x<SW;x+=40){c.beginPath();c.moveTo(x,0);c.lineTo(x,SH);c.stroke()}for(let y=0;y<SH;y+=40){c.beginPath();c.moveTo(0,y);c.lineTo(SW,y);c.stroke()}c.restore()}
 if(S.indexMap)c.drawImage(indexMap(S),0,0);
 if(S.contourMap)c.drawImage(contourMap(S),0,0);
 for(const f of FIELDS){const col=S.fieldColor?S.fieldColor(f):S[f.kind],lift=fieldLifted(f);
  // see-through band
  capPath(c,f.x0,f.y0,f.x1,f.y1,f.r);c.fillStyle=col;c.globalAlpha=S.bandAlpha;c.fill();c.globalAlpha=1;
  if(S.bandLine){c.setLineDash([4,4]);c.strokeStyle=col;c.lineWidth=1;c.stroke();c.setLineDash([])}
  capPath(c,f.x0,f.y0,f.x1,f.y1,f.core);c.globalAlpha=lift?S.liftAlpha:S.coreAlpha;c.fillStyle=col;c.fill();c.globalAlpha=1;
  const pat=S.pattern&&S.pattern(f.kind,f);if(pat){c.save();c.clip();c.globalAlpha=lift?.45:1;c.fillStyle=c.createPattern(pat,'repeat');c.fillRect(0,0,SW,SH);c.restore()}
  if(lift&&S.liftGlow){c.save();capPath(c,f.x0,f.y0,f.x1,f.y1,f.core);c.shadowColor=S.team[0];c.shadowBlur=S.glowBlur??12;c.strokeStyle=S.team[0];c.lineWidth=2.5;c.stroke();c.restore()}
  else if(S.coreLine){capPath(c,f.x0,f.y0,f.x1,f.y1,f.core);c.strokeStyle=S.coreLine==='kind'?col:S.coreLine;c.lineWidth=S.coreLineW||1.2;c.stroke()}
  if(S.decor)S.decor(c,f,col)}
 // F5 drawn again over lifted F4 so the overlap stays opaque
 const f5=FIELDS[4];capPath(c,f5.x0,f5.y0,f5.x1,f5.y1,f5.core);c.globalAlpha=S.coreAlpha;c.fillStyle=S.fieldColor?S.fieldColor(f5):S.thick;c.fill();c.globalAlpha=1;
 if(S.pattern){const pat=S.pattern('thick',f5);c.save();c.clip();c.fillStyle=c.createPattern(pat,'repeat');c.fillRect(0,0,SW,SH);c.restore()}
 if(S.coreLine){c.strokeStyle=S.coreLine==='kind'?(S.fieldColor?S.fieldColor(f5):S.thick):S.coreLine;c.lineWidth=S.coreLineW||1.2;c.stroke()}
 if(S.decor)S.decor(c,f5,S.fieldColor?S.fieldColor(f5):S.thick);
}
function drawStructures(c,S){
 for(const b of BASES){c.fillStyle=S.team[b.team];c.beginPath();c.roundRect(b.x-24,b.y-24,48,48,6);c.fill();c.fillStyle=S.arena;c.beginPath();c.roundRect(b.x-10,b.y-10,20,20,3);c.fill()}
 for(const t of TURRETS){hexPath(c,t.x,t.y,12);c.fillStyle=S.team[t.team];c.fill();c.beginPath();c.arc(t.x,t.y,4,0,7);c.fillStyle=S.arena;c.fill()}
}
function drawUnits(c,S,opt={}){
 const sc=S.shipScale||.3;
 // own bullets always visible
 c.lineCap='round';
 for(const [x,y] of OWN_BULLETS){c.strokeStyle=S.team[0];c.lineWidth=S.bulletW||2.5;c.beginPath();c.moveTo(x,y);c.lineTo(x+10,y-.5);c.stroke()}
 for(const [x,y] of ENEMY_BULLETS){if(seenBy(x,y).length||opt.truth){c.globalAlpha=seenBy(x,y).length?1:.35;c.strokeStyle=S.team[1];c.lineWidth=S.bulletW||2.5;c.beginPath();c.moveTo(x,y);c.lineTo(x-10,y);c.stroke();c.globalAlpha=1}}
 for(const a of ALLIES){shipPath(c,a.x,a.y,a.h,sc);c.fillStyle=S.team[0];if(S.shipGlow){c.save();c.shadowColor=S.team[0];c.shadowBlur=10;c.fill();c.restore()}else c.fill();if(S.shipLine){c.strokeStyle=S.shipLine;c.lineWidth=1.2;c.stroke()}}
 // respawning ally: shown at base with timer
 c.beginPath();c.arc(RESPAWN.x,RESPAWN.y,34,-Math.PI/2,-Math.PI/2+Math.PI*1.2);c.strokeStyle=S.team[0];c.lineWidth=3;c.stroke();
 for(const e of ENEMIES){const vis=enemyVisible(e);
  if(vis){shipPath(c,e.x,e.y,e.h,sc);c.fillStyle=S.team[1];if(S.shipGlow){c.save();c.shadowColor=S.team[1];c.shadowBlur=10;c.fill();c.restore()}else c.fill();if(S.shipLine){c.strokeStyle=S.shipLine;c.lineWidth=1.2;c.stroke()}
   if(e.shoot||e.spawn){for(const [rr,al] of [[18,1],[28,.45]]){c.beginPath();c.arc(e.x,e.y,rr,0,7);c.strokeStyle=S.team[1];c.globalAlpha=al;c.lineWidth=2;if(e.spawn)c.setLineDash([4,4]);c.stroke();c.setLineDash([]);c.globalAlpha=1}}}
  else if(opt.truth){shipPath(c,e.x,e.y,e.h,sc);c.setLineDash([3,3]);c.strokeStyle=S.team[1];c.lineWidth=1.3;c.stroke();c.setLineDash([])}}
}
function render(canvas,S,opt={}){const c=canvas.getContext('2d');const scale=canvas.width/SW;c.save();c.scale(scale,scale);
 const T=document.createElement('canvas');T.width=SW;T.height=SH;drawTerrain(T.getContext('2d'),S);
 const V=document.createElement('canvas');V.width=SW;V.height=SH;const v=V.getContext('2d');v.drawImage(T,0,0);v.globalCompositeOperation='destination-in';v.drawImage(mask(),0,0);
 // fog = terrain transformed
 c.filter=S.fogFilter||'none';c.drawImage(T,0,0);c.filter='none';
 if(S.fogOverlay){c.save();const A=ARENA;c.fillStyle=S.fogOverlay;c.fillRect(0,0,SW,SH);c.restore()}
 if(S.fogPattern){c.fillStyle=c.createPattern(S.fogPattern(),'repeat');c.fillRect(0,0,SW,SH)}
 // visible region
 if(S.visEdgeBlur){c.filter=`blur(${S.visEdgeBlur}px)`;}
 const VB=document.createElement('canvas');VB.width=SW;VB.height=SH;const vb=VB.getContext('2d');
 if(S.visEdgeBlur){vb.filter=`blur(${S.visEdgeBlur}px)`;vb.drawImage(mask(),0,0);vb.filter='none';vb.globalCompositeOperation='source-in';vb.drawImage(T,0,0)}else vb.drawImage(V,0,0);
 c.filter='none';c.drawImage(VB,0,0);
 if(S.visTint){c.save();c.globalCompositeOperation='source-over';const t=document.createElement('canvas');t.width=SW;t.height=SH;const tc=t.getContext('2d');tc.drawImage(mask(),0,0);tc.globalCompositeOperation='source-in';tc.fillStyle=S.visTint;tc.fillRect(0,0,SW,SH);c.drawImage(t,0,0);c.restore()}
 if(S.visOutline){const o=document.createElement('canvas');o.width=SW;o.height=SH;const oc=o.getContext('2d');oc.filter='blur(1.5px)';oc.drawImage(mask(),0,0);oc.filter='none';
   oc.globalCompositeOperation='source-out';oc.drawImage(mask(),0,0);// thin rim outside mask
   const o2=document.createElement('canvas');o2.width=SW;o2.height=SH;const o2c=o2.getContext('2d');o2c.drawImage(o,0,0);o2c.globalCompositeOperation='source-in';o2c.fillStyle=S.visOutline;o2c.fillRect(0,0,SW,SH);c.drawImage(o2,0,0)}
 if(S.scan){c.save();const t=document.createElement('canvas');t.width=SW;t.height=SH;const tc=t.getContext('2d');tc.drawImage(mask(),0,0);tc.globalCompositeOperation='source-in';tc.fillStyle=tc.createPattern(mkPattern((p,s)=>{p.fillStyle=S.scan;p.fillRect(0,0,s,1)},4),'repeat');tc.fillRect(0,0,SW,SH);c.drawImage(t,0,0);c.restore()}
 drawStructures(c,S);drawUnits(c,S,opt);c.restore()}

// composed refractive index, union rule from field_physics.compose_refractive_index
let LINEAR_BW=0; // 0: smootherstep bands as in the physics; >0: linear ramp, band = LINEAR_BW * |level| outward from the core
function lnIndexAt(x,y){let prod=1,wsum=0,lsum=0;const add=(a,lv)=>{prod*=1-a;wsum+=a;lsum+=a*lv*Math.log(Math.SQRT2)};
 for(const f of FIELDS){let a;if(LINEAR_BW){const d=segDist(x,y,f.x0,f.y0,f.x1,f.y1)-f.core;a=Math.max(0,Math.min(1,1-d/(LINEAR_BW*Math.abs(f.lv))))}
  else{const d=segDist(x,y,f.x0,f.y0,f.x1,f.y1)-(f.r-f.w/2)-f.w/2;const z=Math.max(0,Math.min(1,.5-d/f.w));a=z*z*z*(z*(6*z-15)+10)}add(a,f.lv)}
 if(LINEAR_BW){const de=segDist(x,y,ARENA[0],ARENA[1],ARENA[2],ARENA[3])-ARENA[4];add(Math.max(0,Math.min(1,de/(2*LINEAR_BW))),2)}
 const A=1-prod;return wsum>0?A*lsum/wsum:0}
function mixHex(a,b,t){const p=h=>[1,3,5].map(i=>parseInt(h.slice(i,i+2),16));const A=p(a),B=p(b);return A.map((v,i)=>Math.round(v+(B[i]-v)*t))}
const IMAPS={};
function indexMap(S){const k=S.name;if(IMAPS[k])return IMAPS[k];const m=document.createElement('canvas');m.width=SW;m.height=SH;const c=m.getContext('2d');const img=c.createImageData(SW,SH);
 for(let y=0;y<SH;y++)for(let x=0;x<SW;x++){if(!inArena(x,y))continue;const t=lnIndexAt(x,y)/Math.log(2);if(Math.abs(t)<.003)continue;
  const col=t<0?mixHex(S.arena,S.lowCol,Math.min(1,-t)):mixHex(S.arena,S.highCol,Math.min(1,t));const i=4*(y*SW+x);img.data[i]=col[0];img.data[i+1]=col[1];img.data[i+2]=col[2];img.data[i+3]=255}
 c.putImageData(img,0,0);IMAPS[k]=m;return m}

// isolines of ln n by marching squares; terraces optional
const CMAPS={};
function contourMap(S){const k=S.name;if(CMAPS[k])return CMAPS[k];const m=document.createElement('canvas');m.width=SW;m.height=SH;const c=m.getContext('2d');
 const g=1.5,nx=Math.ceil(SW/g)+1,ny=Math.ceil(SH/g)+1,L=new Float32Array(nx*ny),step=Math.log(Math.SQRT2)/S.perLevel;
 for(let j=0;j<ny;j++)for(let i=0;i<nx;i++)L[j*nx+i]=(LINEAR_BW||inArena(i*g,j*g))?lnIndexAt(i*g,j*g)/step:0;
 if(S.terrace){const img=c.createImageData(SW,SH);for(let y=0;y<SH;y++)for(let x=0;x<SW;x++){if(LINEAR_BW&&segDist(x,y,ARENA[0],ARENA[1],ARENA[2],ARENA[3])>ARENA[4]+2*LINEAR_BW)continue;const v=L[Math.round(y/g)*nx+Math.round(x/g)];const q=S.wholeLevels?Math.sign(v)*Math.floor(Math.abs(v)+.02):Math.trunc(v+Math.sign(v)*.5);if(!q)continue;
   const t=Math.min(1,Math.abs(q)/(2*S.perLevel));const col=mixHex(S.arena,q<0?S.lowCol:S.highCol,t*S.terrace);const p=4*(y*SW+x);img.data[p]=col[0];img.data[p+1]=col[1];img.data[p+2]=col[2];img.data[p+3]=255}c.putImageData(img,0,0)}
 for(let k=-2*S.perLevel;k<=2*S.perLevel;k++){if(!k)continue;const major=k%S.perLevel===0;const lev=k-Math.sign(k)*(S.wholeLevels?.02:.5);
  c.strokeStyle=lev<0?S.lowInk:S.highInk;c.lineWidth=(major&&!S.wholeLevels)?S.majorW:S.minorW;if(lev<0&&S.lowDash)c.setLineDash(S.lowDash);c.beginPath();
  for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){const a=L[j*nx+i],b=L[j*nx+i+1],cc=L[(j+1)*nx+i+1],d=L[(j+1)*nx+i];
   const pts=[];const e=(p,q,x0,y0,x1,y1)=>{if((p<lev)!==(q<lev)){const t=(lev-p)/(q-p);pts.push([(x0+(x1-x0)*t)*g,(y0+(y1-y0)*t)*g])}};
   e(a,b,i,j,i+1,j);e(b,cc,i+1,j,i+1,j+1);e(d,cc,i,j+1,i+1,j+1);e(a,d,i,j,i,j+1);
   for(let q=0;q+1<pts.length;q+=2){c.moveTo(pts[q][0],pts[q][1]);c.lineTo(pts[q+1][0],pts[q+1][1])}}
  c.stroke();c.setLineDash([])}
 CMAPS[k]=m;return m}
