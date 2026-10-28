function pat(kind,col,variant){return mkPattern((p,s)=>{p.strokeStyle=col;p.fillStyle=col;
 if(variant==='hatch'){p.lineWidth=1;p.beginPath();p.moveTo(0,s);p.lineTo(s,0);if(kind==='thick'){p.moveTo(0,0);p.lineTo(s,s)}p.stroke();return}
 if(kind==='thin'){p.lineWidth=1.5;p.beginPath();p.moveTo(0,3);p.lineTo(7,3);p.moveTo(5,9);p.lineTo(12,9);p.stroke()}
 else{[[3,3],[9,9]].forEach(([x,y])=>{p.beginPath();p.arc(x,y,1.4,0,7);p.fill()})}},12)}
const STYLES=[
 {name:'1. Clean flat',desc:'Light, quiet, pattern-typed fields; fog desaturates.',outside:'#DCE1E7',damage:'#C9CED6',damageHatch:'#B4BBC6',edge:'#CDD3DB',arena:'#F6F7F9',arenaLine:'#AEB6C2',
  thin:'#6FBFD9',thick:'#8E82B8',bandAlpha:.12,coreAlpha:.5,liftAlpha:.15,pattern:k=>pat(k,k==='thin'?'rgba(30,110,140,.4)':'rgba(60,45,110,.4)'),liftGlow:1,coreLine:'kind',
  team:['#2F6FD0','#E07B2E'],fogFilter:'grayscale(1) brightness(.92)',fogOverlay:'rgba(40,50,70,.05)',visEdgeBlur:3},
 {name:'2. Night ops',desc:'Dark map, glowing units; fog sinks into darkness.',outside:'#05080F',damage:'#0B1120',damageHatch:'#18213A',edge:'#0E1528',arena:'#121C30',grid:'rgba(255,255,255,.035)',
  thin:'#2EC4E6',thick:'#A27BFF',bandAlpha:.08,coreAlpha:.3,liftAlpha:.08,coreLine:'kind',liftGlow:1,team:['#3D9BFF','#FF8A3D'],shipGlow:1,
  fogFilter:'grayscale(.8) brightness(.45)',visEdgeBlur:6},
 {name:'3. Blueprint',desc:'Technical line art; fog is hatched.',outside:'#0F3460',damage:'#0D2E55',damageHatch:'rgba(255,255,255,.15)',edge:'#123C6E',arena:'#1B4F8C',grid:'rgba(255,255,255,.08)',
  thin:'#9EE7FF',thick:'#FFFFFF',bandAlpha:.05,bandLine:1,coreAlpha:.1,liftAlpha:.03,coreLine:'kind',pattern:k=>pat(k,k==='thin'?'rgba(158,231,255,.6)':'rgba(255,255,255,.45)','hatch'),liftGlow:1,
  team:['#FFFFFF','#FFB347'],fogFilter:'brightness(.62) saturate(.4)',fogPattern:()=>mkPattern((p,s)=>{p.strokeStyle='rgba(255,255,255,.07)';p.lineWidth=1;p.beginPath();p.moveTo(0,s);p.lineTo(s,0);p.stroke()},8)},
 {name:'4. Briefing map',desc:'WW2 paper and ink; fog is aged and cross-hatched.',outside:'#C9BA97',damage:'#BBAA84',damageHatch:'rgba(90,70,40,.35)',edge:'#D3C5A3',arena:'#EDE3CC',arenaLine:'#8C7A5E',
  thin:'#7FA7A0',thick:'#8C7A6B',bandAlpha:.15,coreAlpha:.42,liftAlpha:.14,pattern:k=>pat(k,'rgba(70,50,30,.45)',k==='thick'?'hatch':''),coreLine:'#5B4A3A',
  team:['#2E4A7D','#A63A2B'],liftGlow:1,glowBlur:4,fogFilter:'sepia(.7) brightness(.8) contrast(.9)',fogPattern:()=>mkPattern((p,s)=>{p.strokeStyle='rgba(70,50,30,.10)';p.lineWidth=1;p.beginPath();p.moveTo(0,s);p.lineTo(s,0);p.moveTo(0,0);p.lineTo(s,s);p.stroke()},10),visEdgeBlur:2},
 {name:'5. Radar scope',desc:'Phosphor green; you only "see" what the sweep covers.',outside:'#010603',damage:'#031006',damageHatch:'rgba(60,255,120,.12)',edge:'#04140A',arena:'#06200F',grid:'rgba(60,255,120,.09)',
  thin:'#3CFF78',thick:'#3CFF78',bandAlpha:.03,bandLine:1,coreAlpha:.10,liftAlpha:.03,coreLine:'kind',pattern:k=>pat(k,'rgba(60,255,120,.45)'),liftGlow:1,
  team:['#7CFFD1','#FF5A4E'],shipGlow:1,fogFilter:'brightness(.32)',scan:'rgba(60,255,120,.07)',visOutline:'rgba(60,255,120,.55)'},
 {name:'6. Whiteout',desc:'Pastel sky; fog is white haze, like flying into cloud.',outside:'#DDE3EA',damage:'#CBD3DD',damageHatch:'#BAC4D0',edge:'#D5DCE5',arena:'#EAF3F6',
  thin:'#8FD3C9',thick:'#C3A6D8',bandAlpha:.15,coreAlpha:.6,liftAlpha:.18,liftGlow:1,team:['#3B7DD8','#F08A4B'],
  fogFilter:'blur(1.5px) saturate(.35) brightness(1.06)',fogOverlay:'rgba(255,255,255,.55)',visEdgeBlur:9},
 {name:'7. Poster',desc:'Bold flat colour and thick outlines; fog is grey and striped.',outside:'#1E1E24',damage:'#2A2A33',damageHatch:'#3A3A46',edge:'#2E2E38',arena:'#F2EDE4',
  thin:'#2BB3C0',thick:'#5B4B8A',bandAlpha:.18,coreAlpha:1,liftAlpha:.25,coreLine:'#1E1E24',coreLineW:3,shipLine:'#1E1E24',liftGlow:1,glowBlur:0,
  team:['#1F6FEB','#F2542D'],fogFilter:'grayscale(1) contrast(.6) brightness(.88)',fogPattern:()=>mkPattern((p,s)=>{p.strokeStyle='rgba(0,0,0,.07)';p.lineWidth=6;p.beginPath();p.moveTo(0,s);p.lineTo(s,0);p.stroke()},20),visOutline:'#1E1E24'},
 {name:'8. Monochrome + accent',desc:'Everything grey except the teams; fog is grain.',outside:'#E1E2E4',damage:'#D3D5D8',damageHatch:'#C2C5CA',edge:'#D9DBDE',arena:'#FAFAFA',arenaLine:'#C0C3C8',
  thin:'#A9B0B8',thick:'#5F656D',bandAlpha:.12,coreAlpha:.5,liftAlpha:.12,pattern:k=>pat(k,'rgba(0,0,0,.25)'),liftGlow:1,coreLine:'kind',
  team:['#0A84FF','#FF6B00'],fogFilter:'brightness(.9)',fogPattern:()=>mkPattern((p,s)=>{const r=(()=>{let q=9;return()=>{q=q*16807%2147483647;return q/2147483647}})();for(let i=0;i<30;i++){p.fillStyle=`rgba(0,0,0,${.05+r()*.07})`;p.fillRect(r()*s,r()*s,1.5,1.5)}},24),visEdgeBlur:2},
 {name:'9. Dusk',desc:'Warm evening sky, clouds as dark shapes; fog is blue dusk.',outside:'#251B38',damage:'#33264B',damageHatch:'rgba(255,200,150,.15)',edge:'#3A2C55',arena:'#F6C9A0',
  thin:'#FFE6C2',thick:'#9C6A8E',bandAlpha:.2,coreAlpha:.75,liftAlpha:.25,liftGlow:1,team:['#2347B8','#D7263D'],
  fogFilter:'saturate(.45) brightness(.62) hue-rotate(-12deg)',fogOverlay:'rgba(40,30,90,.25)',visEdgeBlur:10},
];
