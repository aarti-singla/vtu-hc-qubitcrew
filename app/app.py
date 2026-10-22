"""Streamlit dashboard.   streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddpipe import (Compiled, Config, Device, build, evaluate, idle_report,  # noqa: E402
                    qiskit_circuit_from_program, standard_configs)
from ddpipe.plots import timeline_figure  # noqa: E402

RES = ROOT / "results"
st.set_page_config(page_title="QubitCrew · Idle-aware DD + error mitigation", page_icon=None, layout="wide")

INK, MUTED, TEAL, AMBER, GREY, LINE, PANEL = "#14213D", "#5B6678", "#0A7F73", "#D9A441", "#B8B2A4", "#DDD7C8", "#FFFFFF"
CFG_COL = {
    "Baseline": GREY, "baseline": GREY, "No DD": GREY,
    "Qiskit DD": AMBER, "IBM Runtime DD": AMBER, "Runtime DD (XY4)": AMBER, "Qiskit PadDD (X-X)": "#B97A1E",
    "DD uniform": "#6FC3B8", "our DD uniform": "#6FC3B8", "Our DD uniform": "#6FC3B8",
    "DD staggered (ours)": TEAL, "our DD staggered": TEAL, "Our DD staggered": TEAL, "our DD (idle-aware)": TEAL,
    "DD auto (ours)": "#075E55",
    "ZNE only": "#7A86A0", "Runtime ZNE (level 2)": "#7A86A0",
    "Auto DD + ZNE (ours)": "#2B3A67", "our DD + Runtime ZNE": "#2B3A67",
    "Auto DD + ZNE + REM (ours)": "#14213D",
    "baseline + readout mitigation": "#8C8A80", "our DD staggered + readout mitigation": "#0A5A52",
}
pio.templates["qc"] = go.layout.Template(layout=dict(
    font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=14),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[TEAL, AMBER, "#2B3A67", "#6FC3B8", GREY, "#B97A1E", "#7A86A0", "#075E55"],
    xaxis=dict(gridcolor="#E6E1D3", linecolor=GREY, zerolinecolor=GREY, ticks="outside", tickcolor=GREY),
    yaxis=dict(gridcolor="#E6E1D3", linecolor=GREY, zerolinecolor=GREY),
    legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, title=None),
    margin=dict(l=10, r=10, t=36, b=10), hoverlabel=dict(font_family="IBM Plex Sans, sans-serif"),
))
px.defaults.template = "qc"


def show(fig, height=400):
    """Render a plotly figure in the app's style (no boilerplate legend or axis titles)."""
    fig.update_layout(legend_title_text=None, height=height)
    if fig.layout.xaxis.title.text in ("run", "workload", "config"):
        fig.update_xaxes(title_text=None)
    for tr in fig.data:
        if tr.type == "bar":
            tr.hovertemplate = "<b>%{fullData.name}</b><br>%{x}: %{y:.3f}<extra></extra>"
        elif tr.type == "scatter":
            tr.hovertemplate = "<b>%{fullData.name}</b><br>x = %{x}<br>%{y:.3f}<extra></extra>"
    fig.update_layout(hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#D5DBE6", font=dict(color=INK, size=13)),
                      hovermode="closest")
    st.plotly_chart(fig, width="stretch")


def embed(html, height):
    """Embed a self-contained HTML page (works on old and new Streamlit versions)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)

BARS_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:'IBM Plex Sans',system-ui,sans-serif;color:#0D1A30}
.wrap{position:relative;padding:6px 2px 0}
.leg{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 4px 46px}
.lg{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid #D5DBE6;border-radius:999px;padding:5px 12px;font-size:12.5px;cursor:pointer;transition:opacity .2s,border-color .2s;user-select:none}
.lg b{width:10px;height:10px;border-radius:3px;display:inline-block}.lg:hover{border-color:#0A7F73}.lg.off{opacity:.4}
svg{display:block;width:100%;overflow:visible}
.gl{stroke:#DDE2EC;stroke-width:1}.ax{font:11.5px 'IBM Plex Mono',monospace;fill:#51607A}.cat{font:12.5px 'IBM Plex Sans',sans-serif;fill:#0D1A30}
.bar{transition:opacity .18s ease,transform .8s cubic-bezier(.2,.8,.2,1);transform-box:fill-box;transform-origin:50% 100%}
.bar.z{transform:scaleY(0);transition:none}
.dim .bar{opacity:.2}.dim .bar.hot{opacity:1}
.dim .err{opacity:.15}.dim .err.hot{opacity:1}
.tip{position:fixed;pointer-events:none;z-index:9;background:#fff;border:1px solid #D5DBE6;border-radius:10px;padding:9px 12px;font-size:12.5px;line-height:1.5;box-shadow:0 8px 24px rgba(13,26,48,.16);opacity:0;transition:opacity .12s;min-width:150px}
.tip i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px}.tip .v{font:600 15px 'IBM Plex Mono',monospace}.tip small{color:#51607A}
</style></head><body><div class="wrap"><div class="leg" id="leg"></div><div id="c"></div></div><div class="tip" id="tip"></div>
<script>
const D=__DATA__;const $=i=>document.getElementById(i);const on={};D.series.forEach(s=>on[s.name]=true);
const tip=$('tip');
function legend(){const l=$('leg');l.innerHTML='';D.series.forEach(s=>{const b=document.createElement('span');b.className='lg'+(on[s.name]?'':' off');b.innerHTML='<b style="background:'+s.color+'"></b>'+s.name;
 b.onclick=()=>{on[s.name]=!on[s.name];if(!Object.values(on).some(Boolean))on[s.name]=true;legend();draw(false);};
 b.onmouseenter=()=>{const c=$('c').querySelector('svg');if(!c)return;c.classList.add('dim');c.querySelectorAll('[data-s="'+CSS.escape(s.name)+'"]').forEach(e=>e.classList.add('hot'));};
 b.onmouseleave=()=>{const c=$('c').querySelector('svg');if(!c)return;c.classList.remove('dim');c.querySelectorAll('.hot').forEach(e=>e.classList.remove('hot'));};
 l.appendChild(b);});}
function draw(anim){const ss=D.series.filter(s=>on[s.name]),W=Math.max(360,$('c').clientWidth||window.innerWidth-8),H=D.height,L=48,R=W-8,T=28,B=H-34;
 let lo=0,hi=0;ss.forEach(s=>s.v.forEach((v,i)=>{if(v==null)return;const e=(s.e&&s.e[i])||0;lo=Math.min(lo,v-e);hi=Math.max(hi,v+e);}));
 const rng=hi-lo||1;hi+=rng*.06;if(lo<0)lo-=rng*.06;
 const nice=(()=>{const st=Math.pow(10,Math.floor(Math.log10((hi-lo)/4)));const m=[1,2,2.5,5,10].find(k=>k*st>=(hi-lo)/4)*st;return m;})();
 const y0=Math.floor(lo/nice)*nice,y1=Math.ceil(hi/nice)*nice,Y=v=>B-(B-T)*(v-y0)/(y1-y0);
 let g='<svg viewBox="0 0 '+W+' '+H+'" height="'+H+'" role="img" aria-label="'+(D.title||'chart')+'">';
 for(let v=y0;v<=y1+1e-9;v+=nice){g+='<line class="gl" x1="'+L+'" x2="'+R+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text class="ax" x="'+(L-8)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+(+v.toFixed(6))+'</text>';}
 if(y0<0)g+='<line x1="'+L+'" x2="'+R+'" y1="'+Y(0)+'" y2="'+Y(0)+'" stroke="#8A94A8" stroke-width="1.2"/>';
 if(D.ytitle)g+='<text class="ax" x="'+L+'" y="'+(T-3)+'">'+D.ytitle+'</text>';
 const gw=(R-L)/D.cats.length,bw=Math.min(34,(gw-18)/ss.length);
 D.cats.forEach((c,i)=>{const gx=L+i*gw+(gw-bw*ss.length)/2;if(c)g+='<text class="cat" x="'+(L+i*gw+gw/2)+'" y="'+(H-10)+'" text-anchor="middle">'+c+'</text>';
  ss.forEach((s,j)=>{const v=s.v[i];if(v==null)return;const x=gx+j*bw,top=Math.min(Y(v),Y(0)),h=Math.abs(Y(v)-Y(0));
   g+='<rect class="bar'+(anim?' z':'')+'" data-s="'+s.name+'" data-i="'+i+'" data-j="'+j+'" x="'+x+'" y="'+top+'" width="'+(bw-3)+'" height="'+Math.max(1,h)+'" rx="3" fill="'+s.color+'" style="transition-delay:'+(i*60+j*30)+'ms"/>';
   const e=(s.e&&s.e[i])||0;if(e>0){const cx=x+(bw-3)/2;g+='<g class="err" data-s="'+s.name+'" stroke="#0D1A30" stroke-width="1.3"><line x1="'+cx+'" x2="'+cx+'" y1="'+Y(v-e)+'" y2="'+Y(v+e)+'"/><line x1="'+(cx-3)+'" x2="'+(cx+3)+'" y1="'+Y(v+e)+'" y2="'+Y(v+e)+'"/><line x1="'+(cx-3)+'" x2="'+(cx+3)+'" y1="'+Y(v-e)+'" y2="'+Y(v-e)+'"/></g>';}});});
 g+='</svg>';$('c').innerHTML=g;const svg=$('c').querySelector('svg');
 svg.querySelectorAll('.bar').forEach(r=>{const s=ss[+r.dataset.j],i=+r.dataset.i;
  r.onmousemove=ev=>{svg.classList.add('dim');svg.querySelectorAll('.hot').forEach(e=>e.classList.remove('hot'));r.classList.add('hot');svg.querySelectorAll('.err[data-s="'+CSS.escape(s.name)+'"]').forEach(e=>{});
   const v=s.v[i],e=(s.e&&s.e[i])||0;tip.innerHTML='<div><i style="background:'+s.color+'"></i><b>'+s.name+'</b></div><small>'+(D.cats[i]||D.ytitle||'')+'</small><div class="v">'+v.toFixed(D.digits)+(e?' <small>± '+e.toFixed(D.digits)+'</small>':'')+'</div>';
   tip.style.opacity=1;const w=tip.offsetWidth;tip.style.left=Math.min(innerWidth-w-6,ev.clientX+14)+'px';tip.style.top=Math.max(4,ev.clientY-tip.offsetHeight-10)+'px';};
  r.onmouseleave=()=>{svg.classList.remove('dim');r.classList.remove('hot');tip.style.opacity=0;};});
 if(anim)requestAnimationFrame(()=>requestAnimationFrame(()=>svg.querySelectorAll('.z').forEach(r=>r.classList.remove('z'))));}
legend();draw(true);addEventListener('resize',()=>draw(false));
</script></body></html>"""


def bars(df, cat, series, val, err=None, cat_order=None, ser_order=None, ytitle=None, height=380, digits=3):
    """Grouped bars with hover highlight (hovered bar stays dark, the rest fade) and a tidy tooltip."""
    cats = cat_order or list(dict.fromkeys(df[cat])) if cat else [""]
    sers = ser_order or list(dict.fromkeys(df[series]))
    out = []
    for sname in sers:
        d = df[df[series] == sname]
        if d.empty:
            continue
        v, e = [], []
        for c in cats:
            r = d[d[cat] == c] if cat else d
            v.append(None if r.empty else float(r[val].iloc[0]))
            e.append(0.0 if (r.empty or err is None) else float(r[err].iloc[0] or 0))
        out.append({"name": sname, "color": CFG_COL.get(sname, TEAL), "v": v, "e": e})
    payload = {"cats": cats, "series": out, "ytitle": ytitle, "height": height, "digits": digits, "title": ytitle or ""}
    embed(BARS_HTML.replace("__DATA__", json.dumps(payload)), height + 96)


LINES_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:'IBM Plex Sans',system-ui,sans-serif;color:#0D1A30}
.wrap{position:relative;padding:6px 2px 0}
.leg{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 4px 52px}
.lg{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid #D5DBE6;border-radius:999px;padding:5px 12px;font-size:12.5px;cursor:pointer;transition:opacity .2s,border-color .2s;user-select:none}
.lg b{width:14px;height:4px;border-radius:2px;display:inline-block}.lg:hover{border-color:#0A7F73}.lg.off{opacity:.4}
svg{display:block;width:100%;overflow:visible}
.gl{stroke:#DDE2EC;stroke-width:1}.ax{font:11.5px 'IBM Plex Mono',monospace;fill:#51607A}.at{font:11.5px 'IBM Plex Sans',sans-serif;fill:#51607A}
.ln{fill:none;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:0;transition:opacity .18s,stroke-width .18s,stroke-dashoffset 1.5s cubic-bezier(.4,.1,.2,1)}
.ln.z{stroke-dashoffset:1;transition:none}
.pt{transition:opacity .18s,r .18s}
.dim .ln,.dim .pt,.dim .eb{opacity:.14}.dim .hot{opacity:1}.dim .ln.hot{stroke-width:4.2}
.tip{position:fixed;pointer-events:none;z-index:9;background:#fff;border:1px solid #D5DBE6;border-radius:10px;padding:9px 12px;font-size:12.5px;line-height:1.5;box-shadow:0 8px 24px rgba(13,26,48,.16);opacity:0;transition:opacity .12s;min-width:150px}
.tip i{display:inline-block;width:12px;height:4px;border-radius:2px;margin-right:7px;vertical-align:middle}.tip .v{font:600 15px 'IBM Plex Mono',monospace}.tip small{color:#51607A}
</style></head><body><div class="wrap"><div class="leg" id="leg"></div><div id="c"></div></div><div class="tip" id="tip"></div>
<script>
const D=__DATA__;const $=i=>document.getElementById(i),on={},tip=$('tip');D.series.forEach(s=>on[s.name]=true);
function legend(){const l=$('leg');l.innerHTML='';D.series.forEach(s=>{const b=document.createElement('span');b.className='lg'+(on[s.name]?'':' off');b.innerHTML='<b style="background:'+s.color+'"></b>'+s.name;
 b.onclick=()=>{on[s.name]=!on[s.name];if(!Object.values(on).some(Boolean))on[s.name]=true;legend();draw(true);};
 b.onmouseenter=()=>hot(s.name);b.onmouseleave=()=>unhot();l.appendChild(b);});}
function hot(n){const g=$('c').querySelector('svg');if(!g)return;g.classList.add('dim');g.querySelectorAll('.hot').forEach(e=>e.classList.remove('hot'));g.querySelectorAll('[data-s="'+CSS.escape(n)+'"]').forEach(e=>e.classList.add('hot'));}
function unhot(){const g=$('c').querySelector('svg');if(!g)return;g.classList.remove('dim');g.querySelectorAll('.hot').forEach(e=>e.classList.remove('hot'));tip.style.opacity=0;}
let PTS=[];
function draw(anim){const ss=D.series.filter(s=>on[s.name]),W=Math.max(360,$('c').clientWidth||innerWidth-8),H=D.height,L=52,R=W-16,T=30,B=H-44;
 const tx=v=>D.logx?Math.log10(v):v;let xs=[];ss.forEach(s=>s.x.forEach(v=>xs.push(v)));const xmin=Math.min(...xs),xmax=Math.max(...xs);
 let lo=1e9,hi=-1e9;ss.forEach(s=>s.v.forEach((v,i)=>{const e=(s.e&&s.e[i])||0;lo=Math.min(lo,v-e);hi=Math.max(hi,v+e);}));if(D.hline){lo=Math.min(lo,D.hline.y);hi=Math.max(hi,D.hline.y);}
 const pad=(hi-lo||1)*.08;lo-=pad;hi+=pad;const st=Math.pow(10,Math.floor(Math.log10((hi-lo)/4))),nice=[1,2,2.5,5,10].find(k=>k*st>=(hi-lo)/4)*st,y0=Math.floor(lo/nice)*nice,y1=Math.ceil(hi/nice)*nice;
 const X=v=>L+(R-L)*(tx(v)-tx(xmin))/((tx(xmax)-tx(xmin))||1),Y=v=>B-(B-T)*(v-y0)/(y1-y0);
 let g='<svg viewBox="0 0 '+W+' '+H+'" height="'+H+'" role="img" aria-label="'+(D.title||'chart')+'">';
 for(let v=y0;v<=y1+1e-9;v+=nice)g+='<line class="gl" x1="'+L+'" x2="'+R+'" y1="'+Y(v)+'" y2="'+Y(v)+'"/><text class="ax" x="'+(L-8)+'" y="'+(Y(v)+4)+'" text-anchor="end">'+(+v.toFixed(6))+'</text>';
 let ticks=[...new Set(xs)].sort((a,b)=>a-b);if(ticks.length>9){const k=Math.ceil(ticks.length/8);ticks=ticks.filter((_,i)=>i%k===0);}
 ticks.forEach(v=>g+='<text class="ax" x="'+X(v)+'" y="'+(B+18)+'" text-anchor="middle">'+(+v.toPrecision(5))+'</text>');
 g+='<text class="at" x="'+((L+R)/2)+'" y="'+(H-6)+'" text-anchor="middle">'+(D.xlabel||'')+'</text>';
 if(D.ytitle)g+='<text class="at" x="'+L+'" y="'+(T-5)+'">'+D.ytitle+'</text>';
 if(D.hline){g+='<line x1="'+L+'" x2="'+R+'" y1="'+Y(D.hline.y)+'" y2="'+Y(D.hline.y)+'" stroke="#8A94A8" stroke-dasharray="5 5"/><text class="at" x="'+(R-4)+'" y="'+(Y(D.hline.y)-6)+'" text-anchor="end">'+D.hline.label+'</text>';}
 PTS=[];
 ss.forEach((s,j)=>{const d=s.x.map((x,i)=>(i?'L':'M')+X(x)+' '+Y(s.v[i])).join(' ');
  s.v.forEach((v,i)=>{const e=(s.e&&s.e[i])||0;if(e>0)g+='<line class="eb" data-s="'+s.name+'" x1="'+X(s.x[i])+'" x2="'+X(s.x[i])+'" y1="'+Y(v-e)+'" y2="'+Y(v+e)+'" stroke="'+s.color+'" stroke-width="1.6"/>';});
  g+='<path class="ln'+(anim?' z':'')+'" pathLength="1" data-s="'+s.name+'" d="'+d+'" stroke="'+s.color+'" stroke-width="'+(s.w||2.6)+'" '+(s.dash?'stroke-dasharray="6 5" ':'')+'style="transition-delay:'+j*110+'ms"/>';
  s.x.forEach((x,i)=>{g+='<circle class="pt" data-s="'+s.name+'" cx="'+X(x)+'" cy="'+Y(s.v[i])+'" r="4" fill="'+s.color+'"/>';PTS.push({px:X(x),py:Y(s.v[i]),s:s,i:i});});});
 g+='<rect id="ov" x="'+L+'" y="'+T+'" width="'+(R-L)+'" height="'+(B-T)+'" fill="transparent"/></svg>';$('c').innerHTML=g;
 const svg=$('c').querySelector('svg'),ov=svg.querySelector('#ov');
 ov.onmousemove=ev=>{const r=svg.getBoundingClientRect(),mx=(ev.clientX-r.left)*W/r.width,my=(ev.clientY-r.top)*H/r.height;let best=null,bd=1e9;PTS.forEach(p=>{const d=Math.hypot(p.px-mx,(p.py-my)*.6);if(d<bd){bd=d;best=p;}});
  if(!best||bd>70){unhot();return;}hot(best.s.name);const s=best.s,e=(s.e&&s.e[best.i])||0;
  tip.innerHTML='<div><i style="background:'+s.color+'"></i><b>'+s.name+'</b></div><small>'+(D.xlabel||'x')+' = '+(+s.x[best.i].toPrecision(5))+'</small><div class="v">'+s.v[best.i].toFixed(D.digits||3)+(e?' <small>± '+e.toFixed(D.digits||3)+'</small>':'')+'</div>';
  tip.style.opacity=1;const w=tip.offsetWidth;tip.style.left=Math.min(innerWidth-w-6,ev.clientX+14)+'px';tip.style.top=Math.max(4,ev.clientY-tip.offsetHeight-12)+'px';};
 ov.onmouseleave=unhot;
 if(anim)requestAnimationFrame(()=>requestAnimationFrame(()=>svg.querySelectorAll('.z').forEach(e=>e.classList.remove('z'))));}
legend();draw(true);addEventListener('resize',()=>draw(false));
</script></body></html>"""

TIMELINE_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
*{box-sizing:border-box}html,body{margin:0;background:transparent;font-family:'IBM Plex Sans',system-ui,sans-serif;color:#0D1A30}
.wrap{background:#fff;border:1px solid #D5DBE6;border-radius:16px;padding:14px 18px 12px;position:relative}
.top{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:4px}.top b{font-size:15px}
.leg{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:#51607A;margin-left:auto}.leg span{display:inline-flex;align-items:center;gap:7px}.leg i{width:11px;height:11px;border-radius:3px;display:inline-block}
.btn{background:#F3F5F9;border:1px solid #D5DBE6;border-radius:999px;padding:5px 14px;font:500 12.5px 'IBM Plex Sans',system-ui,sans-serif;cursor:pointer;color:#0D1A30}.btn:hover{border-color:#0A7F73;color:#0A7F73}
svg{display:block;width:100%;overflow:visible}.ax{font:11.5px 'IBM Plex Mono',monospace;fill:#51607A}.gl{stroke:#E3E7EF;stroke-width:1}
.blk{transition:opacity .15s}.dim .blk{opacity:.22}.dim .blk.hot{opacity:1}
.tip{position:fixed;pointer-events:none;z-index:9;background:#0D1A30;color:#fff;border-radius:9px;padding:8px 11px;font-size:12.5px;line-height:1.5;opacity:0;transition:opacity .12s}
.tip small{color:#AEB8CC}
</style></head><body><div class="wrap"><div class="top"><b id="ttl"></b><button class="btn" id="rep">Replay</button><div class="leg">
<span><i style="background:#2B3A67"></i>two-qubit gate</span><span><i style="background:#8EA2F0"></i>1q gate</span><span><i style="background:#0A7F73"></i>DD pulse</span><span><i style="background:#E7DEB8"></i>idle window</span><span><i style="background:#F0C9C4"></i>crosstalk-idle window</span></div></div>
<div id="c"></div></div><div class="tip" id="tip"></div>
<script>
const D=__DATA__,$=i=>document.getElementById(i),tip=$('tip');$('ttl').textContent=D.title;
const COL={cx:'#2B3A67',q1:'#8EA2F0',dd:'#0A7F73','idle-idle':'#E7DEB8','crosstalk-idle':'#F0C9C4'};
let prog=1,raf=0;
function draw(){const W=Math.max(420,$('c').clientWidth||innerWidth-40),n=D.rows.length,RH=58,L=46,R=W-14,T=14,H=T+n*RH+34,X=t=>L+(R-L)*t/D.t_end,Yc=r=>T+r*RH+RH/2;
 let g='<svg viewBox="0 0 '+W+' '+H+'" height="'+H+'"><defs><clipPath id="rv"><rect id="rvr" x="0" y="0" width="'+(L+(R-L)*prog)+'" height="'+H+'"/></clipPath></defs>';
 for(let r=0;r<n;r++){g+='<text class="ax" x="'+(L-12)+'" y="'+(Yc(r)+4)+'" text-anchor="end">'+D.rows[r]+'</text>';g+='<line class="gl" x1="'+L+'" x2="'+R+'" y1="'+Yc(r)+'" y2="'+Yc(r)+'"/>';}
 for(let k=0;k<=5;k++){const t=D.t_end*k/5;g+='<text class="ax" x="'+X(t)+'" y="'+(H-10)+'" text-anchor="middle">'+Math.round(t)+'</text>';}
 g+='<text class="ax" x="'+R+'" y="'+(H+4)+'" text-anchor="end">time [ns]</text><g clip-path="url(#rv)">';
 D.win.forEach(w=>{g+='<rect class="blk" data-tip="'+(w.kind==='idle-idle'?'Idle window':'Crosstalk-idle window')+'|q'+D.rows[w.r].slice(1)+' · '+Math.round(w.L)+' ns|from '+Math.round(w.t0)+' ns" x="'+X(w.t0)+'" y="'+(Yc(w.r)-RH*.36)+'" width="'+Math.max(1,X(w.t0+w.L)-X(w.t0))+'" height="'+(RH*.72)+'" rx="6" fill="'+COL[w.kind]+'"/>';});
 D.ev.forEach(e=>{const y0=Yc(Math.min(...e.r))-RH*.3,y1=Yc(Math.max(...e.r))+RH*.3,w=Math.max(e.k==='dd'?4:2,X(e.t+e.dur)-X(e.t));
  g+='<rect class="blk" data-tip="'+e.name+'|'+e.q+' · '+Math.round(e.dur)+' ns|at '+Math.round(e.t)+' ns" x="'+X(e.t)+'" y="'+y0+'" width="'+w+'" height="'+(y1-y0)+'" rx="'+(e.k==='dd'?1.5:5)+'" fill="'+COL[e.k]+'"/>';});
 g+='</g><line id="ph" x1="'+(L+(R-L)*prog)+'" x2="'+(L+(R-L)*prog)+'" y1="'+(T-4)+'" y2="'+(H-26)+'" stroke="#0A7F73" stroke-width="2" opacity="'+(prog<1?1:0)+'"/></svg>';
 $('c').innerHTML=g;const svg=$('c').querySelector('svg');
 svg.querySelectorAll('.blk').forEach(b=>{b.onmousemove=ev=>{svg.classList.add('dim');svg.querySelectorAll('.hot').forEach(x=>x.classList.remove('hot'));b.classList.add('hot');const p=b.dataset.tip.split('|');tip.innerHTML='<b>'+p[0]+'</b><br>'+p[1]+'<br><small>'+p[2]+'</small>';tip.style.opacity=1;const w=tip.offsetWidth;tip.style.left=Math.min(innerWidth-w-6,ev.clientX+14)+'px';tip.style.top=Math.max(4,ev.clientY-tip.offsetHeight-12)+'px';};
  b.onmouseleave=()=>{svg.classList.remove('dim');b.classList.remove('hot');tip.style.opacity=0;};});}
function play(){cancelAnimationFrame(raf);const t0=performance.now();(function f(now){prog=Math.min(1,(now-t0)/2200);prog=prog*prog*(3-2*prog);const W=$('c').clientWidth||600,L=46,R=W-14,x=L+(R-L)*prog;const r=document.getElementById('rvr'),p=document.getElementById('ph');
 if(r)r.setAttribute('width',x);if(p){p.setAttribute('x1',x);p.setAttribute('x2',x);p.setAttribute('opacity',prog<1?1:0);}if(prog<1)raf=requestAnimationFrame(f);})(t0);}
$('rep').onclick=()=>{prog=0;draw();play();};addEventListener('resize',()=>{prog=1;draw();});prog=0;draw();play();
</script></body></html>"""


def lines(series, xlabel="", ytitle=None, height=360, logx=False, hline=None, digits=3, title=""):
    """Line chart with hover highlight (hovered line stays dark, the rest fade) and a tidy tooltip."""
    payload = {"series": series, "xlabel": xlabel, "ytitle": ytitle, "height": height, "logx": logx,
               "hline": hline, "digits": digits, "title": title}
    embed(LINES_HTML.replace("__DATA__", json.dumps(payload)), height + 90)


_PAL = [TEAL, AMBER, "#2B3A67", "#6A7FE0", "#B97A1E", "#7A86A0", "#0B5C54", "#C2576B"]


def line_series(df, x, y, group, err=None, order=None):
    out = []
    for j, g in enumerate(order or list(dict.fromkeys(df[group]))):
        d = df[df[group] == g].sort_values(x)
        if d.empty:
            continue
        out.append({"name": str(g), "color": CFG_COL.get(g, _PAL[j % len(_PAL)]),
                    "x": [float(v) for v in d[x]], "v": [float(v) for v in d[y]],
                    "e": [float(v) for v in d[err]] if err else None,
                    "w": 3.4 if "ours" in str(g) else 2.4})
    return out


def timeline(prog, edges, title):
    """Interactive schedule: gates, DD pulses and idle windows; hover to isolate a block, Replay to animate."""
    from ddpipe.idle import idle_windows
    used = sorted({q for e in prog.events for q in e.qubits})
    row = {q: i for i, q in enumerate(used)}
    ev = []
    for e in prog.events:
        if e.name == "rz":
            continue
        k = "dd" if e.tag == "dd" else ("cx" if e.name == "cx" else "q1")
        nm = "DD pulse (" + e.name + ")" if k == "dd" else ("Two-qubit gate (cx)" if k == "cx" else "1q gate (" + e.name + ")")
        ev.append({"r": [row[q] for q in e.qubits], "t": float(e.t), "dur": float(e.dur), "k": k, "name": nm,
                   "q": ",".join("q%d" % q for q in e.qubits)})
    win = [{"r": row[w.q], "t0": float(w.t0), "L": float(w.L), "kind": w.kind} for w in idle_windows(prog, edges) if w.q in row]
    payload = {"rows": ["q%d" % q for q in used], "t_end": float(prog.t_end), "ev": ev, "win": win, "title": title}
    embed(TIMELINE_HTML.replace("__DATA__", json.dumps(payload)), 58 * len(used) + 150)


def cards(items):
    """Row of summary cards: (label, value, caption, highlighted)."""
    html = "".join(f'<div class="card{" accent" if acc else ""}"><div class="lab">{lab}</div><div class="val">{val}</div>'
                   f'<div class="sub">{sub}</div></div>' for lab, val, sub, acc in items)
    st.markdown(f'<div class="cards">{html}</div>', unsafe_allow_html=True)


def table(df, digits=3):
    st.dataframe(df.style.format(f"{{:.{digits}f}}", na_rep=""), width="stretch")
px.defaults.color_discrete_map = CFG_COL



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
header[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
.block-container { max-width: none; padding: 0 !important; }
.stTabs [role="tablist"] { padding: 0 max(24px, calc((100vw - 1180px) / 2)) !important; background: #F3F5F9; }
[role="tabpanel"] { padding: 28px max(24px, calc((100vw - 1180px) / 2)) 72px !important; }
[role="tabpanel"]:has(.st-key-site) { padding: 0 !important; }
.st-key-site iframe { height: calc(100vh - 52px) !important; display: block; }
h1, h2, h3, h4 { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif !important; font-weight: 500 !important; letter-spacing: 0; color: #14213D; }
h3 { font-size: 1.7rem !important; margin-top: 1.4rem !important; }
h4 { font-size: 1.35rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 1.8rem; border-bottom: 1px solid #DDD7C8; }
.stTabs [data-baseweb="tab"] { padding: .7rem 0; height: auto; background: transparent; font-weight: 500; color: #5B6678; }
.stTabs [aria-selected="true"] { color: #0A7F73; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #0A7F73; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.hero { padding: .6rem 0 1.4rem; }
.hero .kicker { font-size: .78rem; font-weight: 500; letter-spacing: .14em; text-transform: uppercase; color: #0A7F73; }
.hero h1 { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; font-weight: 500; font-size: clamp(2rem, 4.2vw, 3.3rem); line-height: 1.08; margin: .5rem 0 .9rem; max-width: 26ch; }
.hero p { font-size: 1.1rem; line-height: 1.55; color: #3B475C; max-width: 62ch; margin: 0; }
.hero .meta { margin-top: .9rem; font-size: .9rem; color: #5B6678; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: .4rem 0 .2rem; }
.rowlab { font-size: .78rem; font-weight: 500; letter-spacing: .12em; text-transform: uppercase; color: #5B6678; margin: 1.1rem 0 .1rem; }
.card { background: #FFFFFF; border: 1px solid #DDD7C8; border-radius: 14px; padding: 18px 20px; }
.card .lab { font-size: .85rem; color: #5B6678; }
.card .val { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; font-size: 2.2rem; line-height: 1.1; margin: .25rem 0 .1rem; white-space: nowrap; }
.card .sub { font-size: .85rem; color: #5B6678; }
.card.accent { border-color: #0A7F73; }
.card.accent .val { color: #0A7F73; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: .4rem 0; }
.step { background: #FFFFFF; border: 1px solid #DDD7C8; border-radius: 14px; padding: 16px 18px; }
.step .n { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; font-size: 2rem; color: #0A7F73; line-height: 1; }
.step b { display: block; margin: .35rem 0 .2rem; }
.step span { font-size: .92rem; color: #3B475C; line-height: 1.45; }
.card, .step { transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; animation: rise .7s cubic-bezier(.2,.8,.2,1) both; }
.card:hover, .step:hover { transform: translateY(-4px); box-shadow: 0 10px 28px rgba(20,33,61,.10); border-color: #0A7F73; }
.cards .card:nth-child(2), .steps .step:nth-child(2) { animation-delay: .08s; }
.cards .card:nth-child(3), .steps .step:nth-child(3) { animation-delay: .16s; }
.steps .step:nth-child(4) { animation-delay: .24s; } .steps .step:nth-child(5) { animation-delay: .32s; }
@keyframes rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
iframe { border: 0 !important; }
.note { font-size: .88rem; color: #5B6678; line-height: 1.5; margin-top: 1rem; }
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #DDD7C8; border-radius: 14px; padding: 14px 18px; }
[data-testid="stMetricValue"] { font-family: 'Bricolage Grotesque', 'Segoe UI', sans-serif; font-size: 2rem; }
[data-testid="stMetricLabel"] p { color: #5B6678; font-size: .85rem; }
div[data-testid="stDataFrame"] { border: 1px solid #DDD7C8; border-radius: 10px; overflow: hidden; }
.stButton > button[kind="primary"] { background: #0A7F73; border: 0; border-radius: 10px; font-weight: 500; }
[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

ORDER = ["Baseline", "Qiskit DD", "DD uniform", "DD staggered (ours)", "DD auto (ours)", "ZNE only",
         "Auto DD + ZNE (ours)", "Auto DD + ZNE + REM (ours)"]


@st.cache_data
def load(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def compiled(kind, n, topology, zz, sigma, wait):
    dev = Device(n, topology=topology).with_noise(zz_khz=float(zz), sigma_f_khz=float(sigma))
    kw = {"wait_ns": float(wait)} if kind == "ghzwait" else {}
    return Compiled(build(kind, n, **kw), dev)


ab0 = None
_ab_path = RES / "ablation.csv"
if _ab_path.exists():
    ab0 = pd.read_csv(_ab_path)

tabs = st.tabs(["Project site", "Schedule explorer", "Live benchmark", "Results", "Hardware"])

# ------------------------------------------------------------ Project site
with tabs[0]:
    page = ROOT / "site" / "index.html"
    with st.container(key="site"):
        if page.exists():
            html = page.read_text().replace('<html lang="en">', '<html lang="en" data-theme="light">', 1)
            embed(html, 900)
        else:
            st.info("site/index.html is missing. Run `python site/build.py` to generate it.")

# ------------------------------------------------------- Schedule explorer
with tabs[1]:
    c = st.columns(5)
    kind = c[0].selectbox("Circuit", ["ghzwait", "ghz", "qaoa", "vqe", "mirror"], format_func=lambda k: {
        "ghzwait": "GHZ + mid-circuit wait", "ghz": "GHZ", "qaoa": "QAOA ring", "vqe": "VQE (TFIM)", "mirror": "Mirror"}[k])
    n = c[1].slider("Qubits", 3, 6, 4, key="n1")
    strategy = c[2].selectbox("DD strategy", ["none", "uniform", "staggered", "auto"])
    family = c[3].selectbox("Sequence", ["XY4", "XpXm", "CPMG", "XY8"])
    tau = c[4].slider("Pulse spacing τ [ns]", 100, 2400, 600, step=50)
    comp = compiled(kind, n, "ring", 70, 25, 1500)
    cfg = Config("view", dd=strategy, family=family, tau_ns=float(tau))
    prog, _ = comp.program(1.0, cfg)
    rep = idle_report(prog, comp.dev.edges)
    timeline(prog, comp.dev.edges, f"{comp.wl.name} · DD: {strategy}")
    m = st.columns(5)
    m[0].metric("Duration", f"{prog.t_end:.0f} ns")
    m[1].metric("Idle share", f"{rep['idle_pct']:.0f}%")
    m[2].metric("Idle windows", rep["n_windows"])
    m[3].metric("Crosstalk-idle windows", rep["n_crosstalk_idle"])
    m[4].metric("DD pulses", prog.n_dd_pulses)
    st.dataframe(pd.DataFrame(rep["per_qubit"]).round(1), width="stretch", hide_index=True)
    with st.expander("Export this schedule as a Qiskit circuit (for hardware)"):
        qc = qiskit_circuit_from_program(prog)
        st.code(str(qc.draw("text", fold=140)), language=None)

# ------------------------------------------------------------ Live benchmark
with tabs[2]:
    st.write("Runs the full pipeline live on a small circuit (exact density-matrix simulation, a few seconds).")
    c = st.columns(6)
    kind = c[0].selectbox("Circuit ", ["ghzwait", "ghz", "qaoa", "vqe", "mirror"], key="k2")
    n = c[1].slider("Qubits ", 3, 5, 4, key="n2")
    zz = c[2].slider("ZZ crosstalk [kHz]", 0, 150, 70)
    sg = c[3].slider("Slow-drift σ [kHz]", 0, 60, 25)
    wait = c[4].slider("Wait [ns] (GHZ+wait)", 500, 3000, 1500, step=100)
    use_zne = c[5].checkbox("Include ZNE configs", True)
    if st.button("Run benchmark", type="primary"):
        comp = compiled(kind, n, "ring", zz, sg, wait)
        cfgs = [x for x in standard_configs() if use_zne or not x.zne]
        rows, bar = [], st.progress(0.0)
        for i, cf in enumerate(cfgs):
            r = evaluate(comp, cf, K=24)
            rows.append({"config": cf.name, "|1-F|": abs(1 - r.fidelity), "fidelity": r.fidelity,
                         "energy error": r.energy_error, "DD pulses": r.n_dd_pulses})
            bar.progress((i + 1) / len(cfgs))
        df = pd.DataFrame(rows)
        _base = float(df[df.config == "Baseline"]["|1-F|"].iloc[0])
        _dd = df[df.config.str.contains("DD") & ~df.config.str.contains("ZNE") & (df.config != "Qiskit DD")]
        _best = _dd.loc[_dd["|1-F|"].idxmin()]
        cards([("Baseline error |1 − F|", f"{_base:.3f}", "no mitigation", False),
               ("Best DD configuration", f"{_best['|1-F|']:.3f}", _best.config, True),
               ("Reduction", f"{(1 - _best['|1-F|'] / _base) * 100:.0f}%" if _base > 0 else "n/a", "vs baseline, same circuit and noise", True)])
        st.markdown("**State-fidelity error |1 − F|, lower is better**")
        bars(df.assign(c=""), "c", "config", "|1-F|", ser_order=[c for c in ORDER if c in set(df.config)] , ytitle="|1 − F|", height=340)
        st.dataframe(df.round(4), width="stretch", hide_index=True)
        st.caption("ZNE rows are extrapolated estimates and may fall slightly outside [0, 1].")

# ------------------------------------------------------------------ Results
with tabs[3]:
    ab, sc, ta, nz, sh = (load(x) for x in ("ablation", "scaling", "tau_sweep", "noise_sweep", "shots"))
    if ab is None:
        st.warning("Run `python -m ddpipe.experiments` first to generate results/.")
    else:
        d = ab.assign(err=(1 - ab.fidelity).abs())
        _b = d[d.config == "Baseline"].set_index("workload").err
        _a = d[d.config == "DD auto (ours)"].set_index("workload").err
        _q = d[d.config == "Qiskit DD"].set_index("workload").err
        _w = (_b - _a).idxmax()
        cards([("Mean infidelity, no mitigation", f"{_b.mean():.3f}", "six workloads, noise model", False),
               ("Mean infidelity, our auto-DD", f"{_a.mean():.3f}", f"{(1 - _a.mean() / _b.mean()) * 100:.0f}% lower", True),
               ("Largest single gain", _w, f"{_b[_w]:.2f} to {_a[_w]:.2f} infidelity", True),
               ("Better than Qiskit's DD pass", f"{int((_a < _q).sum())} of {len(_a)}", "workloads, by state fidelity", False)])
        st.subheader("1 · Ablation on six workloads")
        bars(d, "workload", "config", "err", ser_order=ORDER, ytitle="|1 − F|, lower is better", height=400)
        piv = d.pivot(index="config", columns="workload", values="fidelity").reindex(ORDER).round(3)
        table(piv)
        st.caption("State fidelity, ideal = 1. ZNE rows above 1 are extrapolation overshoot, not physical: read the DD rows as the trustworthy ones.")
        ee = d.dropna(subset=["energy_error"]).pivot(index="config", columns="workload", values="energy_error")
        if not ee.empty:
            st.markdown("**Energy error |E − E_ideal|** (QAOA / VQE, exact expectation values)")
            table(ee.reindex(ORDER))
        if "auto_choice" in d:
            st.caption("Auto-DD choices: " + ", ".join(
                f"{w}: {c}" for w, c in d[d.config == "DD auto (ours)"][["workload", "auto_choice"]].values))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("2 · Scaling with GHZ size")
            lines(line_series(sc, "n", "fidelity", "config", order=["Baseline", "Qiskit DD", "DD uniform", "DD auto (ours)"]),
                  xlabel="GHZ qubits", ytitle="state fidelity", height=340)
        with c2:
            st.subheader("3 · Pulse spacing & sequence sweep")
            wl = st.selectbox("Workload", ta.workload.unique())
            t = ta[(ta.workload == wl) & (ta.family != "-")]
            _nodd = float(ta[(ta.workload == wl) & (ta.family == "-")].fidelity.iloc[0])
            lines(line_series(t, "tau_ns", "fidelity", "family"), xlabel="pulse spacing τ [ns]", ytitle="state fidelity",
                  logx=True, hline={"y": _nodd, "label": "no DD"}, height=340)
        st.subheader("4 · Robustness to crosstalk and slow drift (GHZ + wait)")
        _c1, _c2 = st.columns(2)
        for _col, _ax in zip((_c1, _c2), list(dict.fromkeys(nz.axis))):
            with _col:
                lines(line_series(nz[nz.axis == _ax], "value", "fidelity", "config",
                                  order=["Baseline", "Qiskit DD", "DD uniform", "DD staggered (ours)", "DD auto (ours)"]),
                      xlabel=_ax, ytitle="state fidelity", height=340)
        if sh is not None:
            st.subheader("5 · ZNE trades bias for variance (finite-shot emulation)")
            s = sh.assign(label=sh.workload + " · " + sh.shots.astype(str) + " shots")
            bars(s, "label", "config", "energy_error", ser_order=ORDER, ytitle="RMSE of energy", height=360)
            st.caption("RMSE includes bias and shot-noise variance. Extrapolation amplifies variance: gains shrink "
                       "(or reverse) at low shot counts.")

# ------------------------------------------------------------------ Hardware
with tabs[4]:
    hw = RES / "hardware.json"
    if hw.exists():
        data = json.loads(hw.read_text())
        st.success(f"Backend: {data.get('backend')} · date: {data.get('date')}")
        st.dataframe(pd.DataFrame(data["rows"]), width="stretch", hide_index=True)
        st.caption("Job IDs: " + ", ".join(data.get("job_ids", [])))
    else:
        st.info("No hardware run recorded yet. Run `python scripts/run_hardware.py` with your IBM Quantum credentials; "
                "it writes `results/hardware.json`, which this tab then displays (with job IDs).")
    # Sweep over idle-wait length: every results/hw_w*us.json, hardware_wait10us.json and hardware_no_wait.json
    sweep_rows = []
    for f in sorted(list(RES.glob("hw_w*us.json")) + list(RES.glob("hardware_wait*.json")) +
                    list(RES.glob("hardware_no_wait.json"))):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in d.get("rows", []):
            if r["config"] in ("baseline", "Runtime DD (XY4)", "our DD (idle-aware)"):
                sweep_rows.append({"wait [us]": d.get("wait_ns", 10000 if "wait10" in f.name else 0) / 1000,
                                   "config": r["config"], "fidelity": r["fidelity"], "stderr": r["stderr"],
                                   "backend": d.get("backend")})
    if sweep_rows:
        sw = pd.DataFrame(sweep_rows).drop_duplicates(["wait [us]", "config"], keep="last")
        st.markdown("#### Hardware sweep: GHZ-4 fidelity vs idle wait")
        lines(line_series(sw, "wait [us]", "fidelity", "config", err="stderr"), xlabel="idle wait [µs]",
              ytitle="GHZ-4 fidelity", height=340)
        st.caption("Real-device runs (error bars are shot noise only; each point is one run, so drift between "
                   "jobs is not included). At 0 us the circuit has no idle window long enough for DD, so our DD "
                   "inserted no pulses there.")
        st.dataframe(sw.pivot(index="wait [us]", columns="config", values="fidelity").round(3), width="stretch")
    # Crosstalk test: four qubits in |+>, idle wait, mean <X> (scripts/run_crosstalk_hw.py -> results/xt_*.json)
    HW_ORDER = ["baseline", "IBM Runtime DD", "Qiskit PadDD (X-X)", "our DD uniform", "our DD staggered",
                "baseline + readout mitigation", "our DD staggered + readout mitigation"]

    def _summaries(files, label, value_key, err_key):
        out = []
        for f in files:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            for s in d.get("summary", []):
                if s.get(value_key) is None:
                    continue
                out.append({"run": label(d), "config": s["config"], "mean <X>": s[value_key],
                            "stderr": s.get(err_key) or 0.0, "runs": s.get("runs", 1)})
        return pd.DataFrame(out)

    xt = _summaries(sorted(list(RES.glob("xt_*.json")) + list(RES.glob("crosstalk_hw*.json"))),
                    lambda d: f"{d.get('backend')} · {d.get('wait_ns', 0) / 1000:g} µs wait", "x_mean", "stderr")
    if not xt.empty:
        st.markdown("#### Crosstalk test: mean ⟨X⟩ of four idle qubits in |+⟩ (ideal 1)")
        bars(xt, "run", "config", "mean <X>", err="stderr", ser_order=HW_ORDER, ytitle="mean ⟨X⟩, ideal 1", height=380)
        st.caption("Staggered DD refocuses ZZ crosstalk that identical (uniform) trains leave in place. Each device was "
                   "tested at one wait, so device and wait change together. Compare setups within one run only: the "
                   "baseline drifts between sessions.")
        pv = xt.pivot(index="run", columns="config", values="mean <X>")
        table(pv[[c for c in HW_ORDER if c in pv.columns]])

    # Spectator test: idle qubits beside a busy CZ pair (scripts/run_spectator_hw.py -> results/spectator_*.json)
    sp = _summaries(sorted(RES.glob("spectator_*.json")), lambda d: str(d.get("backend")),
                    "spectator_x", "spectator_stderr")
    if not sp.empty:
        st.markdown("#### Spectator test: idle qubits beside a busy pair (ideal 1)")
        bars(sp, "run", "config", "mean <X>", err="stderr", ser_order=HW_ORDER, ytitle="mean ⟨X⟩, ideal 1", height=380)
        st.caption("Six qubits in |+>; the middle two run CZ gates that cancel to the identity, so the four outer "
                   "qubits idle next to a busy neighbour. Error bars are shot noise only. All idle windows were "
                   "classified idle-idle, and the busy pair's own value is dominated by CZ errors, so it is not shown.")
        pv = sp.pivot(index="run", columns="config", values="mean <X>")
        table(pv[[c for c in HW_ORDER if c in pv.columns]])
    st.markdown("**Configurations compared on hardware:** no suppression · Runtime DD · our schedule-aware DD "
                "(exported circuit, Runtime DD off) · Runtime ZNE · our DD + Runtime ZNE/TREX.")