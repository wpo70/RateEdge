"""
RateEdge Vol Surface Editor - Full Version
Features: 3D drag with smoothing, change colors (green/red), Vol/Premium toggle, Plasma heatmap
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict
import json

try:
    from streamlit_js_eval import streamlit_js_eval
    HAS_JS_EVAL = True
except ImportError:
    HAS_JS_EVAL = False

HEATMAP_COLORSCALE = [
    [0.00, "#0d0887"], [0.15, "#46039f"], [0.30, "#7201a8"], [0.45, "#9c179e"],
    [0.60, "#bd3786"], [0.75, "#ed7953"], [0.90, "#fca636"], [1.00, "#f0f921"],
]

EXPIRY_YEARS = {
    "1W": 1/52, "2W": 2/52, "1M": 1/12, "2M": 2/12, "3M": 0.25, "6M": 0.5,
    "9M": 0.75, "1Y": 1.0, "18M": 1.5, "2Y": 2.0, "3Y": 3.0, "4Y": 4.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "15Y": 15.0, "20Y": 20.0, "30Y": 30.0,
}

DEFAULT_SMOOTHING = {"enabled": True, "radius": 1, "falloff": 0.5}


def label_to_years(label: str) -> float:
    label = label.upper().strip()
    if label in EXPIRY_YEARS:
        return EXPIRY_YEARS[label]
    if label.endswith("M"):
        return float(label[:-1]) / 12
    if label.endswith("Y"):
        return float(label[:-1])
    return 1.0


def vol_to_premium(vol_bp: float, T: float, tenor_years: float = 10.0) -> float:
    """
    Normal vol (bp) -> ATM straddle FORWARD premium (bp running).
    Formula: Fwd Premium = 2 × 0.3989 × σ(bp) × √T
    """
    if T <= 0:
        return 0.0
    fwd_premium = 2 * 0.3989 * vol_bp * np.sqrt(T)
    return fwd_premium


def premium_to_vol(prem_bp: float, T: float, tenor_years: float = 10.0) -> float:
    """ATM straddle forward premium (bp) -> Normal vol (bp)."""
    if T <= 0:
        return 0.0
    vol_bp = prem_bp / (2 * 0.3989 * np.sqrt(T))
    return vol_bp


def surface_vol_to_premium(df: pd.DataFrame, ccy: str = None) -> pd.DataFrame:
    """Convert vol surface to premium. Uses real prem_matrix from pricer if available,
    otherwise falls back to the simplified formula."""
    import streamlit as st
    if ccy is not None:
        prem_store = st.session_state.get("prem_matrix", {})
        if ccy in prem_store:
            p = prem_store[ccy].copy()
            exp_col = df.columns[0]
            if "Expiry" in p.columns:
                p = p.set_index("Expiry")
            result = df.copy()
            tcols = df.columns[1:].tolist()
            for i, row in df.iterrows():
                exp_lbl = str(row[exp_col])
                for c in tcols:
                    try:
                        result.at[i, c] = round(float(p.loc[exp_lbl, c]), 2)
                    except Exception:
                        T = label_to_years(exp_lbl)
                        result.at[i, c] = round(vol_to_premium(float(row[c]), T), 2)
            return result
    # fallback: simplified formula
    result = df.copy()
    exp_col, tcols = df.columns[0], df.columns[1:].tolist()
    for i, row in df.iterrows():
        T = label_to_years(str(row[exp_col]))
        for c in tcols:
            result.at[i, c] = round(vol_to_premium(float(row[c]), T), 2)
    return result


def surface_premium_to_vol(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    exp_col, tcols = df.columns[0], df.columns[1:].tolist()
    for i, row in df.iterrows():
        T = label_to_years(str(row[exp_col]))
        for c in tcols:
            tenor_years = label_to_years(c)
            result.at[i, c] = round(premium_to_vol(float(row[c]), T, tenor_years), 2)
    return result


def _init_state(ccy: str, surface: pd.DataFrame) -> None:
    if "vol_editor" not in st.session_state:
        st.session_state["vol_editor"] = {}
    ed = st.session_state["vol_editor"]
    for k in ["working", "base", "history", "redo_stack", "view_mode", "smoothing", "paste_data"]:
        if k not in ed:
            ed[k] = {}
    
    # Reset if surface shape changed (new data structure)
    if ccy in ed["working"]:
        if ed["working"][ccy].shape != surface.shape:
            del ed["working"][ccy]
    
    if ccy not in ed["working"]:
        ed["working"][ccy] = surface.copy()
        ed["base"][ccy] = surface.copy()
        ed["history"][ccy] = []
        ed["redo_stack"][ccy] = []
        ed["view_mode"][ccy] = "vol"
        ed["smoothing"][ccy] = DEFAULT_SMOOTHING.copy()
        ed["paste_data"][ccy] = ""


def _push_history(ccy: str) -> None:
    ed = st.session_state["vol_editor"]
    ed["history"][ccy].append(ed["working"][ccy].copy())
    ed["redo_stack"][ccy] = []
    if len(ed["history"][ccy]) > 50:
        ed["history"][ccy] = ed["history"][ccy][-50:]


def _undo(ccy: str) -> bool:
    ed = st.session_state["vol_editor"]
    if ed["history"].get(ccy):
        ed["redo_stack"][ccy].append(ed["working"][ccy].copy())
        ed["working"][ccy] = ed["history"][ccy].pop()
        return True
    return False


def _redo(ccy: str) -> bool:
    ed = st.session_state["vol_editor"]
    if ed["redo_stack"].get(ccy):
        ed["history"][ccy].append(ed["working"][ccy].copy())
        ed["working"][ccy] = ed["redo_stack"][ccy].pop()
        return True
    return False


def _has_changes(ccy: str) -> bool:
    ed = st.session_state["vol_editor"]
    w, b = ed["working"].get(ccy), ed["base"].get(ccy)
    return w is not None and b is not None and not w.equals(b)


def _publish(ccy: str) -> pd.DataFrame:
    ed = st.session_state["vol_editor"]
    ed["base"][ccy] = ed["working"][ccy].copy()
    ed["history"][ccy] = []
    ed["redo_stack"][ccy] = []
    if "vol_data" in st.session_state and ccy in st.session_state["vol_data"]:
        st.session_state["vol_data"][ccy]["atm"] = ed["working"][ccy].copy()
    return ed["working"][ccy].copy()


def _reset(ccy: str) -> None:
    ed = st.session_state["vol_editor"]
    _push_history(ccy)
    ed["working"][ccy] = ed["base"][ccy].copy()


def _create_plotly_surface(df: pd.DataFrame, ccy: str, view_mode: str, changes=None) -> go.Figure:
    exp_col, tcols = df.columns[0], df.columns[1:].tolist()
    expiries = df[exp_col].tolist()
    display_df = surface_vol_to_premium(df, ccy) if view_mode == "fwd_premium" else df
    z_label = "Fwd Premium (bp)" if view_mode == "fwd_premium" else "Vol (bp)"
    z_vals = display_df[tcols].values.astype(float)
    
    # Auto-scale Z axis to data with padding
    z_min_val, z_max_val = z_vals.min(), z_vals.max()
    z_range = z_max_val - z_min_val
    z_min = np.floor((z_min_val - z_range * 0.05) / 5) * 5
    z_max = np.ceil((z_max_val + z_range * 0.05) / 5) * 5
    
    # Don't reverse - keep natural order: X=tenor (1Y to 30Y), Y=expiry (1M to 20Y)
    X, Y = np.meshgrid(np.arange(len(tcols)), np.arange(len(expiries)))
    
    if changes is not None:
        cv = changes[tcols].values.astype(float)
        mc = max(abs(cv.min()), abs(cv.max()), 0.1)
        surfacecolor = cv / mc
        colorscale = [[0,"rgb(180,40,40)"],[0.4,"rgb(100,100,150)"],[0.6,"rgb(100,100,150)"],[1,"rgb(40,160,40)"]]
        cbar_title = "Change"
    else:
        surfacecolor = z_vals
        colorscale = HEATMAP_COLORSCALE
        cbar_title = z_label
    
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=z_vals, surfacecolor=surfacecolor, colorscale=colorscale, opacity=0.92,
        colorbar=dict(title=dict(text=cbar_title, font=dict(color="white")), tickfont=dict(color="white"), len=0.6),
        hovertemplate="<b>%{customdata[0]}</b> × <b>%{customdata[1]}</b><br>"+f"{z_label}: %{{z:.2f}}<extra></extra>",
        customdata=[[[expiries[i], tcols[j]] for j in range(len(tcols))] for i in range(len(expiries))],
    )])
    
    fig.update_layout(
        title=dict(text=f"<b>ATM Vol Surface (Live)</b>", font=dict(color="white"), x=0.5),
        scene=dict(
            # X-axis = Tenor (1Y to 30Y)
            xaxis=dict(
                title=dict(text="Tenor", font=dict(color="white", size=12)),
                ticktext=tcols, tickvals=list(range(len(tcols))),
                backgroundcolor="rgba(20,40,80,0.8)", 
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="#cbd5e1", size=10),
                tickangle=0,
            ),
            # Y-axis = Expiry (1M to 20Y)
            yaxis=dict(
                title=dict(text="Expiry", font=dict(color="white", size=12)),
                ticktext=expiries, tickvals=list(range(len(expiries))),
                backgroundcolor="rgba(20,40,80,0.8)",
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="#cbd5e1", size=10),
                tickangle=0,
            ),
            # Z-axis = Vol/Premium (auto-scaled)
            zaxis=dict(
                title=dict(text=z_label, font=dict(color="white", size=12)),
                range=[z_min, z_max],
                backgroundcolor="rgba(40,60,100,0.8)",
                gridcolor="rgba(255,255,255,0.15)",
                tickfont=dict(color="#cbd5e1", size=10),
            ),
            bgcolor="rgb(15,25,50)",
            # Camera matching 3D editor orientation
            camera=dict(
                eye=dict(x=-1.8, y=-1.8, z=1.0),
                up=dict(x=0, y=0, z=1),
            ),
            aspectratio=dict(x=1.2, y=1.4, z=0.7),
        ),
        paper_bgcolor="rgb(15,23,42)", 
        margin=dict(l=0, r=0, t=50, b=0), 
        height=550,
    )
    return fig


def _render_3d_editor(df, ccy, view_mode, smoothing, base_df, height=580):
    exp_col = df.columns[0]
    expiries, tcols = df[exp_col].tolist(), df.columns[1:].tolist()
    ey = [label_to_years(str(e)) for e in expiries]
    display_df = surface_vol_to_premium(df, ccy) if view_mode == "fwd_premium" else df
    base_display = surface_vol_to_premium(base_df, ccy) if view_mode == "fwd_premium" else base_df
    z_label = "Fwd Premium (bp)" if view_mode == "fwd_premium" else "Vol (bp)"
    z_values = display_df[tcols].values.astype(float).tolist()
    base_vals = base_display[tcols].values.astype(float).tolist()
    zf = display_df[tcols].values.flatten()
    z_min_val, z_max_val = float(zf.min()), float(zf.max())
    z_range = z_max_val - z_min_val
    z_min = float(np.floor((z_min_val - z_range * 0.05) / 10) * 10)
    z_max = float(np.ceil((z_max_val + z_range * 0.05) / 10) * 10)
    
    data = json.dumps({"expiries": expiries, "tenors": tcols, "values": z_values, "baseValues": base_vals, "zMin": z_min, "zMax": z_max, "zLabel": z_label, "viewMode": view_mode, "expiryYears": ey, "ccy": ccy, "smoothing": smoothing})
    
    html = f'''<!DOCTYPE html><html><head><style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#0a1628;font-family:system-ui;overflow:hidden}}
#c{{width:100%;height:{height}px;position:relative}}canvas{{width:100%;height:100%}}
#info{{position:absolute;top:10px;left:10px;color:#94a3b8;font-size:11px;background:rgba(15,23,42,0.9);padding:8px 12px;border-radius:6px;border:1px solid #334155}}
#tip{{position:absolute;display:none;background:rgba(30,41,59,0.95);color:#fff;padding:10px 14px;border-radius:6px;font-size:12px;pointer-events:none;border:1px solid #3b82f6;z-index:100}}
#st{{position:absolute;bottom:10px;left:10px;font-size:11px;background:rgba(15,23,42,0.9);padding:8px 12px;border-radius:6px;border:1px solid #334155}}
.rdy{{color:#22c55e}}.edt{{color:#f59e0b}}.chg{{color:#3b82f6}}.rot{{color:#a855f7}}
#btn{{position:absolute;bottom:10px;right:10px;background:linear-gradient(135deg,#22c55e,#16a34a);color:#fff;border:none;padding:10px 24px;border-radius:6px;font-weight:600;cursor:pointer;display:none;font-size:13px;min-width:100px}}
#btn.show{{display:block}}
#lockBtn{{position:absolute;bottom:10px;right:150px;background:#334155;color:#fff;border:none;padding:10px 16px;border-radius:6px;font-weight:500;cursor:pointer;font-size:12px;min-width:140px}}
#lockBtn.locked{{background:#7c3aed;color:#fff}}
#legend{{position:absolute;top:10px;right:10px;background:rgba(15,23,42,0.9);padding:8px 12px;border-radius:6px;border:1px solid #334155;font-size:10px;color:#94a3b8}}
.li{{display:flex;align-items:center;gap:5px;margin:2px 0}}.lb{{width:12px;height:12px;border-radius:2px}}
.lu{{background:#22c55e}}.ld{{background:#dc2626}}.ln{{background:#3b82f6}}
#title{{position:absolute;top:10px;left:50%;transform:translateX(-50%);color:#fff;font-size:13px;font-weight:600;background:rgba(30,58,95,0.8);padding:5px 14px;border-radius:5px}}
</style></head><body><div id="c"><canvas id="cv"></canvas>
<div id="title">ATM Vol Editor</div>
<div id="info">🖱️ <b>Left-drag</b> points to edit<br>🔄 <b>Right-drag</b> to rotate<br>🔍 <b>Scroll</b> to zoom</div>
<div id="legend"><b>Changes</b><div class="li"><div class="lb lu"></div>Up</div><div class="li"><div class="lb ld"></div>Down</div><div class="li"><div class="lb ln"></div>No change</div></div>
<div id="tip"></div><div id="st" class="rdy">✓ Ready - Rotate to position, then edit</div>
<button id="lockBtn" onclick="toggleLock()">🔓 Rotation ON</button>
<button id="btn" onclick="apply()">✓ Apply</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function(){{
const D={data};

const cn=document.getElementById('c'),cv=document.getElementById('cv'),tip=document.getElementById('tip'),st=document.getElementById('st'),btn=document.getElementById('btn'),lockBtn=document.getElementById('lockBtn');
let changed=false,rotLocked=false;

const scene=new THREE.Scene();
scene.background=new THREE.Color(0x0a1628);
const cam=new THREE.PerspectiveCamera(36,cn.clientWidth/cn.clientHeight,0.1,1000);
const ren=new THREE.WebGLRenderer({{canvas:cv,antialias:true}});
ren.setSize(cn.clientWidth,cn.clientHeight);
ren.setPixelRatio(Math.min(devicePixelRatio,2));

// Lighting
scene.add(new THREE.AmbientLight(0xffffff,0.5));
const dl=new THREE.DirectionalLight(0xffffff,0.6);dl.position.set(-8,15,8);scene.add(dl);
const dl2=new THREE.DirectionalLight(0x6080b0,0.3);dl2.position.set(8,8,-8);scene.add(dl2);

// Grid on floor
const gridH=new THREE.GridHelper(16,16,0x2a3a5a,0x151f30);gridH.position.y=-4;scene.add(gridH);

// Data setup
const vals=JSON.parse(JSON.stringify(D.values)),baseVals=JSON.parse(JSON.stringify(D.baseValues));
const nE=D.expiries.length,nT=D.tenors.length;
const xSp=14/Math.max(nT-1,1),zSp=14/Math.max(nE-1,1);
const ySc=7/(D.zMax-D.zMin),yOf=D.zMin;
const v2y=v=>(v-yOf)*ySc-3.5;

// Axis labels using sprites
function makeLabel(text,pos,color){{
const canvas=document.createElement('canvas');
const ctx=canvas.getContext('2d');
canvas.width=128;canvas.height=32;
ctx.fillStyle=color||'#94a3b8';
ctx.font='bold 20px system-ui';
ctx.textAlign='center';
ctx.fillText(text,64,22);
const tex=new THREE.CanvasTexture(canvas);
const mat=new THREE.SpriteMaterial({{map:tex,transparent:true}});
const sprite=new THREE.Sprite(mat);
sprite.position.set(pos.x,pos.y,pos.z);
sprite.scale.set(2,0.5,1);
return sprite;}}

// Tenor labels (X axis) - left to right
for(let j=0;j<nT;j+=Math.max(1,Math.floor(nT/8))){{
const x=j*xSp-7;
scene.add(makeLabel(D.tenors[j],{{x,y:-4.5,z:-8}},'#60a5fa'));
}}
scene.add(makeLabel('Tenor →',{{x:0,y:-4.5,z:-9}},'#3b82f6'));

// Expiry labels (Z axis) - front to back  
for(let i=0;i<nE;i+=Math.max(1,Math.floor(nE/6))){{
const z=i*zSp-7;
scene.add(makeLabel(D.expiries[i],{{x:-9,y:-4.5,z}},'#f59e0b'));
}}
scene.add(makeLabel('← Expiry',{{x:-10,y:-4.5,z:0}},'#f59e0b'));

// Y-axis (Vol/Premium) labels - at BACK corner (positive z)
const ySteps=5;
for(let k=0;k<=ySteps;k++){{
const yVal=D.zMin+(D.zMax-D.zMin)*k/ySteps;
const y=v2y(yVal);
scene.add(makeLabel(yVal.toFixed(0),{{x:-9,y,z:8}},'#22c55e'));
}}
scene.add(makeLabel(D.zLabel,{{x:-9,y:2,z:9}},'#22c55e'));

// Heatmap color
function getHeatCol(v){{
const t=Math.max(0,Math.min(1,(v-D.zMin)/(D.zMax-D.zMin)));
// Plasma-like: purple -> pink -> orange -> yellow
const h=0.85-t*0.75;
const s=0.6+t*0.3;
const l=0.35+t*0.35;
return new THREE.Color().setHSL(h,s,l);
}}

// Change color (green/red)
function getChgCol(v,b){{
const d=v-b,m=Math.max((D.zMax-D.zMin)*0.05,5);
const t=Math.max(-1,Math.min(1,d/m));
if(Math.abs(t)<0.02)return null;
return t>0?new THREE.Color(0.2,0.5+0.4*t,0.2):new THREE.Color(0.5+0.4*Math.abs(t),0.2,0.2);
}}

const pts=[],sg=new THREE.SphereGeometry(0.15,16,16);
for(let i=0;i<nE;i++)for(let j=0;j<nT;j++){{
const x=j*xSp-7,z=i*zSp-7,y=v2y(vals[i][j]);
const chg=getChgCol(vals[i][j],baseVals[i][j]);
const col=chg||getHeatCol(vals[i][j]);
const mat=new THREE.MeshPhongMaterial({{color:col,emissive:col,emissiveIntensity:0.2,shininess:40}});
const sp=new THREE.Mesh(sg,mat);
sp.position.set(x,y,z);sp.userData={{i,j}};
scene.add(sp);pts.push(sp);
}}

let surf=null;
function updSurf(){{
if(surf){{scene.remove(surf);surf.geometry.dispose();surf.material.dispose();}}
const geo=new THREE.BufferGeometry(),verts=[],cols=[],idx=[];
for(let i=0;i<nE;i++)for(let j=0;j<nT;j++){{
const x=j*xSp-7,z=i*zSp-7,y=v2y(vals[i][j]);
verts.push(x,y,z);
const chg=getChgCol(vals[i][j],baseVals[i][j]);
const c=chg||getHeatCol(vals[i][j]);
cols.push(c.r,c.g,c.b);
}}
for(let i=0;i<nE-1;i++)for(let j=0;j<nT-1;j++){{
const a=i*nT+j;idx.push(a,a+1,a+nT,a+1,a+nT+1,a+nT);
}}
geo.setAttribute('position',new THREE.Float32BufferAttribute(verts,3));
geo.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));
geo.setIndex(idx);geo.computeVertexNormals();
surf=new THREE.Mesh(geo,new THREE.MeshPhongMaterial({{vertexColors:true,side:THREE.DoubleSide,transparent:true,opacity:0.85,shininess:20}}));
scene.add(surf);
}}

function updCols(){{
pts.forEach((sp,idx)=>{{
const i=Math.floor(idx/nT),j=idx%nT;
const chg=getChgCol(vals[i][j],baseVals[i][j]);
const col=chg||getHeatCol(vals[i][j]);
sp.material.color=col;sp.material.emissive=col;
}});
}}
updSurf();

// Camera - view from front-left
let sph={{r:28,p:Math.PI/3.2,t:-Math.PI/4.5}};
function updCam(){{
cam.position.set(sph.r*Math.sin(sph.p)*Math.cos(sph.t),sph.r*Math.cos(sph.p),sph.r*Math.sin(sph.p)*Math.sin(sph.t));
cam.lookAt(0,0,0);
}}
updCam();

// Lock toggle
window.toggleLock=function(){{
rotLocked=!rotLocked;
lockBtn.textContent=rotLocked?'🔒 Rotation OFF':'🔓 Rotation ON';
lockBtn.className=rotLocked?'locked':'';
cv.style.cursor=rotLocked?'crosshair':'grab';
st.textContent=rotLocked?'✓ Locked - Click points to edit':'✓ Unlocked - Rotate view';
st.className=rotLocked?'rdy':'rot';
}};

function smooth(i,j,d){{
if(!D.smoothing.enabled){{vals[i][j]+=d;return;}}
const r=D.smoothing.radius||1,f=D.smoothing.falloff||0.5;
vals[i][j]+=d;
for(let di=-r;di<=r;di++)for(let dj=-r;dj<=r;dj++){{
if(di===0&&dj===0)continue;
const ni=i+di,nj=j+dj;
if(ni>=0&&ni<nE&&nj>=0&&nj<nT){{
const dist=Math.sqrt(di*di+dj*dj);
const w=f*Math.exp(-dist*dist/(2*Math.pow(r/2,2)));
vals[ni][nj]+=d*w;
pts[ni*nT+nj].position.y=v2y(vals[ni][nj]);
}}
}}
updCols();
}}

const rc=new THREE.Raycaster(),ms=new THREE.Vector2();
let sel=null,drag=false,rot=false,ly=0;

cv.onmousedown=e=>{{
const r=cv.getBoundingClientRect();
ms.x=((e.clientX-r.left)/r.width)*2-1;
ms.y=-((e.clientY-r.top)/r.height)*2+1;

// Right click always rotates (unless locked)
if((e.button===2||e.button===1)&&!rotLocked){{rot=true;cv.style.cursor='grabbing';return;}}

rc.setFromCamera(ms,cam);
const h=rc.intersectObjects(pts);
if(h.length){{
sel=h[0].object;
sel.material.emissiveIntensity=0.6;
sel.scale.setScalar(2);
drag=true;ly=e.clientY;
const{{i,j}}=sel.userData,d=vals[i][j]-baseVals[i][j];
st.textContent=`Editing: ${{D.expiries[i]}} × ${{D.tenors[j]}} = ${{vals[i][j].toFixed(1)}} (${{d>=0?'+':''}}${{d.toFixed(1)}})`;
st.className='edt';
}}
}};

cv.onmousemove=e=>{{
const r=cv.getBoundingClientRect();
ms.x=((e.clientX-r.left)/r.width)*2-1;
ms.y=-((e.clientY-r.top)/r.height)*2+1;

if(rot&&!rotLocked){{
sph.t-=e.movementX*0.008;
sph.p=Math.max(0.3,Math.min(Math.PI/2-0.1,sph.p+e.movementY*0.008));
updCam();return;
}}

if(drag&&sel){{
// Scale drag sensitivity based on data range
const range = D.zMax - D.zMin;
// Aim for ~2% of range per 10px drag - more responsive
const dragScale = range / 500;
const dy=(ly-e.clientY)*dragScale;ly=e.clientY;
const{{i,j}}=sel.userData;
smooth(i,j,dy);
sel.position.y=v2y(vals[i][j]);
updSurf();
const d=vals[i][j]-baseVals[i][j];
st.textContent=`${{D.expiries[i]}} × ${{D.tenors[j]}} = ${{vals[i][j].toFixed(1)}} (${{d>=0?'+':''}}${{d.toFixed(1)}})`;
if(!changed){{changed=true;btn.classList.add('show');}}
return;
}}

// Hover
rc.setFromCamera(ms,cam);
const h=rc.intersectObjects(pts);
pts.forEach(p=>{{if(p!==sel){{p.scale.setScalar(1);p.material.emissiveIntensity=0.2;}}}});
if(h.length&&!drag){{
const pt=h[0].object;
pt.scale.setScalar(1.5);
pt.material.emissiveIntensity=0.4;
const{{i,j}}=pt.userData,d=vals[i][j]-baseVals[i][j];
tip.style.display='block';
tip.style.left=(e.clientX-r.left+15)+'px';
tip.style.top=(e.clientY-r.top-10)+'px';
const chgStr=Math.abs(d)>0.1?(d>=0?`<span style="color:#22c55e">+${{d.toFixed(1)}}</span>`:`<span style="color:#dc2626">${{d.toFixed(1)}}</span>`):'<span style="color:#64748b">no change</span>';
tip.innerHTML=`<b>${{D.expiries[i]}} × ${{D.tenors[j]}}</b><br>${{vals[i][j].toFixed(1)}} ${{D.zLabel}}<br>${{chgStr}}`;
}}else tip.style.display='none';
}};

cv.onmouseup=()=>{{
if(sel){{sel.scale.setScalar(1);sel.material.emissiveIntensity=0.2;}}
sel=null;drag=false;rot=false;
cv.style.cursor=rotLocked?'crosshair':'grab';
if(!changed)st.textContent=rotLocked?'✓ Locked - Click points to edit':'✓ Ready - Right-drag to rotate';
else st.textContent='● Changes pending';
st.className=changed?'chg':'rdy';
}};

cv.onmouseleave=()=>{{
if(sel){{sel.scale.setScalar(1);sel.material.emissiveIntensity=0.2;}}
sel=null;drag=false;rot=false;tip.style.display='none';
}};

cv.onwheel=e=>{{e.preventDefault();sph.r=Math.max(15,Math.min(50,sph.r+e.deltaY*0.02));updCam();}};
cv.oncontextmenu=e=>e.preventDefault();

window.apply=function(){{
try{{
const payload = btoa(JSON.stringify({{
  ccy: D.ccy,
  vals: vals,
  mode: D.viewMode,
  ts: Date.now()
}}));
// Copy to clipboard
if(navigator.clipboard){{
  navigator.clipboard.writeText(payload).then(()=>{{
    btn.textContent='✓ Copied! Paste below';
    btn.style.background='#22c55e';
    st.textContent='Paste in box below & click CONFIRM';
    st.style.color='#22c55e';
  }}).catch(()=>{{
    prompt('Copy this (Ctrl+A, Ctrl+C):', payload);
  }});
}}else{{
  prompt('Copy this (Ctrl+A, Ctrl+C):', payload);
}}
}}catch(e){{alert('Error: '+e);}}
}};

(function anim(){{requestAnimationFrame(anim);ren.render(scene,cam);}})();
window.onresize=()=>{{cam.aspect=cn.clientWidth/cn.clientHeight;cam.updateProjectionMatrix();ren.setSize(cn.clientWidth,cn.clientHeight);}};
}})();
</script></body></html>'''
    components.html(html, height=height+100, scrolling=False)


def render_vol_surface_editor(ccy: str, atm_surface: pd.DataFrame, curve: pd.DataFrame = None, ois_curve: pd.DataFrame = None) -> pd.DataFrame:
    # Check for v3d_data in URL params BEFORE init
    force_update = False
    
    try:
        params = st.query_params
        if params.get('v3d_ccy') == ccy and 'v3d_data' in params:
            import base64
            updated = json.loads(base64.b64decode(params['v3d_data']).decode())
            mode = params.get('v3d_mode', 'vol')
            tcols = atm_surface.columns[1:].tolist()
            expiries = atm_surface[atm_surface.columns[0]].tolist()
            
            rebuilt = atm_surface.copy()
            if mode == "fwd_premium":
                for i, row in enumerate(updated):
                    T = label_to_years(str(expiries[i]))
                    for j, v in enumerate(row):
                        tenor_y = label_to_years(tcols[j])
                        rebuilt.iloc[i, j+1] = round(premium_to_vol(v, T, tenor_y), 2)
            else:
                for i, row in enumerate(updated):
                    for j, v in enumerate(row):
                        rebuilt.iloc[i, j+1] = round(v, 2)
            
            # Push history for undo before updating
            _init_state(ccy, atm_surface)  # Ensure state exists
            _push_history(ccy)
            
            atm_surface = rebuilt
            force_update = True
            
            # Clear URL params
            for k in ['v3d_ccy', 'v3d_data', 'v3d_mode', 'v3d_ts']:
                if k in params:
                    del st.query_params[k]
    except Exception as e:
        pass
    
    _init_state(ccy, atm_surface)
    
    # Force update working surface if we rebuilt from v3d_data
    if force_update:
        st.session_state["vol_editor"]["working"][ccy] = atm_surface.copy()
    ed = st.session_state["vol_editor"]
    working, base = ed["working"][ccy], ed["base"][ccy]
    view_mode = ed["view_mode"].get(ccy, "vol")
    smoothing = ed["smoothing"].get(ccy, DEFAULT_SMOOTHING)
    has_changes = _has_changes(ccy)
    
    st.markdown("""<style>
div.stButton>button{font-weight:600!important}
div.stButton>button[kind="primary"]{background:linear-gradient(135deg,#22c55e,#16a34a)!important}
div[data-testid="stRadio"] label span{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
div[data-testid="stRadio"] label{color:#ffffff!important}
input[aria-label="Paste data here:"]{background:#1e293b!important;color:#ffffff!important;border:2px solid #3b82f6!important;padding:14px!important;font-size:16px!important;font-family:monospace!important;letter-spacing:0.5px!important;font-weight:500!important}
input[aria-label="Paste data here:"]::placeholder{color:#64748b!important;font-family:monospace!important}
</style>""", unsafe_allow_html=True)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"### ATM Vol Editor")
    with c2:
        if has_changes:
            st.error("⚠️ Unsaved")
    
    cols = st.columns([2.5, 1, 1, 1, 2, 1.5])
    with cols[0]:
        if st.button("✅ PUBLISH", key=f"pub_{ccy}", type="primary", disabled=not has_changes, use_container_width=True):
            _publish(ccy)
            # Clear paste data after publish
            ed["paste_data"][ccy] = ""
            st.success("Published!")
            st.rerun()
    with cols[1]:
        if st.button("↩️ Undo", key=f"undo_{ccy}", disabled=not ed["history"].get(ccy), use_container_width=True):
            _undo(ccy)
            st.rerun()
    with cols[2]:
        if st.button("↪️ Redo", key=f"redo_{ccy}", disabled=not ed["redo_stack"].get(ccy), use_container_width=True):
            _redo(ccy)
            st.rerun()
    with cols[3]:
        if st.button("🔄 Reset", key=f"reset_{ccy}", disabled=not has_changes, use_container_width=True):
            _reset(ccy)
            st.rerun()
    with cols[4]:
        new_mode = st.radio("View", ["Vol (bp)", "Fwd Premium (bp)"], index=0 if view_mode == "vol" else 1, horizontal=True, key=f"vm_{ccy}", label_visibility="collapsed")
        new_mode = "vol" if "Vol" in new_mode else "fwd_premium"
        if new_mode != view_mode:
            ed["view_mode"][ccy] = new_mode
            st.rerun()
    with cols[5]:
        show_chg = st.checkbox("Show Δ", value=has_changes, key=f"sd_{ccy}")
    
    st.markdown("---")
    
    with st.expander("⚙️ Smoothing", expanded=False):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            smoothing["enabled"] = st.checkbox("Enable", value=smoothing.get("enabled", True), key=f"se_{ccy}")
        with sc2:
            smoothing["radius"] = st.slider("Radius", 1, 3, smoothing.get("radius", 1), key=f"sr_{ccy}")
        with sc3:
            smoothing["falloff"] = st.slider("Falloff", 0.1, 0.9, smoothing.get("falloff", 0.5), key=f"sf_{ccy}")
        ed["smoothing"][ccy] = smoothing
    
    st.markdown("#### 🎯 ATM Vol Editor")
    st.caption("Drag points • Green=up, Red=down • Click Apply, paste below, CONFIRM")
    _render_3d_editor(working, ccy, view_mode, smoothing, base)
    
    # Paste and CONFIRM section
    st.markdown("---")
    st.markdown("#### 📋 Step 2: Paste & Confirm")
    st.caption("After clicking Apply above, paste the data here (Ctrl+V)")
    
    # Initialize paste data if not exists
    if "paste_data" not in ed or ccy not in ed["paste_data"]:
        if "paste_data" not in ed:
            ed["paste_data"] = {}
        ed["paste_data"][ccy] = ""
    
    col1, col2 = st.columns([4, 1])
    with col1:
        paste_data = st.text_input(
            "Paste data here:",
            value=ed["paste_data"][ccy],
            key=f"paste_{ccy}",
            placeholder="Paste here (Ctrl+V)",
            label_visibility="collapsed"
        )
        # Update session state
        ed["paste_data"][ccy] = paste_data
    with col2:
        st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
        confirm_btn = st.button("✅ CONFIRM", key=f"confirm_btn_{ccy}", type="primary", use_container_width=True)
    
    if confirm_btn and paste_data:
        try:
            import base64
            payload = json.loads(base64.b64decode(paste_data).decode())
            if payload.get('ccy') == ccy:
                updated = payload['vals']
                mode = payload.get('mode', 'vol')
                tcols = working.columns[1:].tolist()
                expiries = working[working.columns[0]].tolist()
                
                _push_history(ccy)
                if mode == "fwd_premium":
                    for i, row in enumerate(updated):
                        T = label_to_years(str(expiries[i]))
                        for j, v in enumerate(row):
                            tenor_y = label_to_years(tcols[j])
                            working.iloc[i, j+1] = round(premium_to_vol(v, T, tenor_y), 2)
                else:
                    for i, row in enumerate(updated):
                        for j, v in enumerate(row):
                            working.iloc[i, j+1] = round(v, 2)
                ed["working"][ccy] = working
                # Clear paste data after successful confirmation
                ed["paste_data"][ccy] = ""
                st.success("✅ Changes applied!")
                st.rerun()
            else:
                st.error(f"Currency mismatch")
        except Exception as e:
            st.error(f"Invalid data: {e}")
    elif confirm_btn:
        st.warning("Paste the data first")
    
    with st.expander("📊 ATM Vol Surface (Live)", expanded=False):
        changes = None
        if show_chg and has_changes:
            changes = working.copy()
            for c in working.columns[1:]:
                changes[c] = working[c].astype(float) - base[c].astype(float)
        st.plotly_chart(_create_plotly_surface(working, ccy, view_mode, changes), use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### 📋 Edit Grid")
    
    # Prepare display data
    display = surface_vol_to_premium(working, ccy) if view_mode == "fwd_premium" else working.copy()
    base_display = surface_vol_to_premium(base, ccy) if view_mode == "fwd_premium" else base.copy()
    
    # Calculate changes for styling
    tcols = working.columns[1:].tolist()
    
    # Create styled dataframe showing changes as heatmap
    if has_changes and show_chg:
        st.caption("🟢 Green = increased | 🔴 Red = decreased | Intensity shows magnitude")
        
        # Build change matrix
        change_vals = display[tcols].values.astype(float) - base_display[tcols].values.astype(float)
        max_change = max(abs(change_vals.min()), abs(change_vals.max()), 0.1)
        
        def style_cell(val, row_idx, col_idx):
            """Return CSS style based on change value."""
            try:
                change = change_vals[row_idx, col_idx]
                intensity = min(abs(change) / max_change, 1.0)
                if change > 0.01:
                    # Green gradient
                    g = int(80 + 120 * intensity)
                    return f'background-color: rgba(34, 197, 94, {0.2 + 0.6 * intensity}); color: white; font-weight: 600'
                elif change < -0.01:
                    # Red gradient
                    r = int(80 + 120 * intensity)
                    return f'background-color: rgba(220, 38, 38, {0.2 + 0.6 * intensity}); color: white; font-weight: 600'
                else:
                    return 'background-color: rgba(59, 130, 246, 0.1)'
            except:
                return ''
        
        def apply_heatmap_style(df):
            """Apply heatmap styling to dataframe."""
            styles = pd.DataFrame('', index=df.index, columns=df.columns)
            for i in range(len(df)):
                for j, col in enumerate(tcols):
                    styles.iloc[i, j+1] = style_cell(df.iloc[i, j+1], i, j)
            # Style expiry column
            styles.iloc[:, 0] = 'background-color: #1e293b; color: #94a3b8; font-weight: 600'
            return styles
        
        styled_df = display.style.apply(lambda _: apply_heatmap_style(display), axis=None)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Separate editable grid below
        st.markdown("##### ✏️ Edit Values")
    
    edited = st.data_editor(
        display, 
        use_container_width=True, 
        num_rows="fixed", 
        key=f"grid_{ccy}_{view_mode}",
        column_config={
            working.columns[0]: st.column_config.TextColumn("Expiry", disabled=True),
            **{col: st.column_config.NumberColumn(col, format="%.2f") for col in tcols}
        }
    )
    
    if edited is not None:
        if view_mode == "fwd_premium":
            as_vol = surface_premium_to_vol(edited)
            if not as_vol.equals(working):
                _push_history(ccy)
                ed["working"][ccy] = as_vol
                st.rerun()
        elif not edited.equals(working):
            _push_history(ccy)
            ed["working"][ccy] = edited
            st.rerun()
    
    return ed["working"][ccy].copy()


def render_bulk_adjustment_tools(ccy: str) -> None:
    if "vol_editor" not in st.session_state or ccy not in st.session_state["vol_editor"]["working"]:
        return
    ed = st.session_state["vol_editor"]
    w, tcols = ed["working"][ccy], ed["working"][ccy].columns[1:].tolist()
    
    st.markdown("#### ⚡ Quick Adjustments")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Shift**")
        shift = st.number_input("bp", -50.0, 50.0, 0.0, 0.5, key=f"sh_{ccy}")
        if st.button("Apply", key=f"dosh_{ccy}") and shift != 0:
            _push_history(ccy)
            w[tcols] = w[tcols] + shift
            ed["working"][ccy] = w
            st.rerun()
    with c2:
        st.markdown("**Scale**")
        scale = st.number_input("%", -50.0, 50.0, 0.0, 1.0, key=f"sc_{ccy}")
        if st.button("Apply", key=f"dosc_{ccy}") and scale != 0:
            _push_history(ccy)
            w[tcols] = w[tcols] * (1 + scale/100)
            ed["working"][ccy] = w
            st.rerun()
    with c3:
        st.markdown("**Exp Tilt**")
        tilt = st.number_input("bp/row", -10.0, 10.0, 0.0, 0.25, key=f"ti_{ccy}")
        if st.button("Apply", key=f"doti_{ccy}") and tilt != 0:
            _push_history(ccy)
            for i in range(len(w)):
                w.iloc[i, 1:] = w.iloc[i, 1:] + tilt * i
            ed["working"][ccy] = w
            st.rerun()
    with c4:
        st.markdown("**Tenor Tilt**")
        tt = st.number_input("bp/col", -10.0, 10.0, 0.0, 0.25, key=f"tt_{ccy}")
        if st.button("Apply", key=f"dott_{ccy}") and tt != 0:
            _push_history(ccy)
            for j, col in enumerate(tcols):
                w[col] = w[col] + tt * j
            ed["working"][ccy] = w
            st.rerun()


render_vol_surface_editor_unified = render_vol_surface_editor
render_vol_surface_editor_3d = render_vol_surface_editor


# =============================================================================
# CONVENTIONS TAB
# =============================================================================

AUD_CONVENTIONS = {
    "source": "AFMA Interest Rate Options Conventions - June 2025",
    "currency": "AUD",
    "business_day": "Any day which is not a 'bank close day' under the law of New South Wales",
    "premium_quotation": "Basis points of notional (price only)",
    "premium_example": "If notional = $10m and premium = $10,000, quote = 10bp",
    "atm_reference": "At-the-money rate = swap rate for the underlying structure",
    "option_style": "European (unless American style requested)",
    "day_count": "Actual/365",
    "swaption_freq_short": "Quarterly (maturities ≤ 3Y)",
    "swaption_freq_long": "Semi-annual (maturities ≥ 4Y)",
    "cap_floor_freq": "Quarterly",
    "settlement_index": "BBSW",
    "premium_payment_spot": "T+2 business days (or by agreement)",
    "premium_payment_fwd_cash": "Day after expiry (cash settled)",
    "premium_payment_fwd_phys": "Day of expiry (physically settled)",
    "exercise_time": "10:00am AEST on expiry date",
    "fallback_exercise": "Automatic if ITM by ≥ 10bp (physically settled, on-the-run)",
    "physical_settlement": "Swap commences 1 business day after exercise",
    "cash_settlement_method": "Zero coupon methodology",
    "cash_settlement_payment": "1 business day following exercise date",
}

AUD_MARKET_PARCELS = {
    "expiries": ["1m", "3m", "6m", "9m", "1y", "2y", "3y", "4y", "5y", "7y", "10y", "15y", "20y"],
    "tenors": ["1y", "2y", "3y", "4y", "5y", "7y", "10y", "15y", "20y", "30y"],
    "parcels": [
        [250, 200, 100, 75, 75, 50, 35, 25, 20, 15],
        [200, 200, 100, 75, 75, 50, 35, 25, 15, 10],
        [200, 175, 100, 75, 75, 50, 35, 25, 15, 10],
        [200, 150, 100, 75, 75, 50, 35, 25, 15, 10],
        [200, 150, 100, 75, 75, 50, 35, 25, 15, 10],
        [150, 100, 90, 65, 65, 40, 30, 20, 10, 10],
        [125, 100, 75, 50, 50, 40, 30, 15, 10, 7.5],
        [125, 75, 60, 50, 50, 40, 25, 15, 10, 7.5],
        [100, 75, 60, 50, 50, 40, 25, 15, 10, 7.5],
        [75, 50, 50, 40, 40, 30, 25, 15, 10, 7.5],
        [75, 40, 30, 30, 30, 25, 25, 15, 10, 7.5],
        [50, 25, 20, 20, 20, 20, 15, 10, 10, 7.5],
        [50, 25, 20, 15, 15, 15, 10, 10, 10, 7.5],
    ]
}

ALL_CONVENTIONS = {
    "AUD": AUD_CONVENTIONS,
    "USD": {"source": "To be added", "currency": "USD"},
    "EUR": {"source": "To be added", "currency": "EUR"},
    "GBP": {"source": "To be added", "currency": "GBP"},
    "JPY": {"source": "To be added", "currency": "JPY"},
    "NZD": {"source": "To be added", "currency": "NZD"},
    "CAD": {"source": "To be added", "currency": "CAD"},
}


def render_conventions_tab(selected_ccy: str = "AUD"):
    """Render the conventions tab for the pricer."""
    
    st.markdown("### 📋 Interest Rate Options Conventions")
    
    ccy = st.selectbox(
        "Select Currency",
        options=list(ALL_CONVENTIONS.keys()),
        index=list(ALL_CONVENTIONS.keys()).index(selected_ccy) if selected_ccy in ALL_CONVENTIONS else 0,
        key="conventions_ccy_select"
    )
    
    conv = ALL_CONVENTIONS.get(ccy, {})
    
    if conv.get("source") == "To be added":
        st.warning(f"⚠️ {ccy} conventions not yet available.")
        st.info("Currently available: **AUD** (AFMA June 2025)")
        return
    
    st.caption(f"*Source: {conv.get('source', 'N/A')}*")
    st.divider()
    
    # Quotation & Dealing
    with st.expander("💰 Quotation & Dealing", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Premium Quotation**")
            st.markdown(f"- {conv.get('premium_quotation', 'N/A')}")
            st.markdown(f"- *Example: {conv.get('premium_example', 'N/A')}*")
            st.markdown("**ATM Reference**")
            st.markdown(f"- {conv.get('atm_reference', 'N/A')}")
            st.markdown("**Option Style**")
            st.markdown(f"- {conv.get('option_style', 'N/A')}")
        with col2:
            st.markdown("**Day Count Basis**")
            st.markdown(f"- {conv.get('day_count', 'N/A')}")
            st.markdown("**Swaption Frequency**")
            st.markdown(f"- {conv.get('swaption_freq_short', 'N/A')}")
            st.markdown(f"- {conv.get('swaption_freq_long', 'N/A')}")
            st.markdown("**Cap/Floor Frequency**")
            st.markdown(f"- {conv.get('cap_floor_freq', 'N/A')}")
    
    # Premium Payment
    with st.expander("📅 Premium Payment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spot Premium**")
            st.markdown(f"- {conv.get('premium_payment_spot', 'N/A')}")
        with col2:
            st.markdown("**Forward Premium**")
            st.markdown(f"- Cash settled: {conv.get('premium_payment_fwd_cash', 'N/A')}")
            st.markdown(f"- Physically settled: {conv.get('premium_payment_fwd_phys', 'N/A')}")
    
    # Exercise & Settlement
    with st.expander("⚡ Exercise & Settlement", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Exercise**")
            st.markdown(f"- Expiry time: {conv.get('exercise_time', 'N/A')}")
            st.markdown(f"- Fallback: {conv.get('fallback_exercise', 'N/A')}")
        with col2:
            st.markdown("**Settlement**")
            st.markdown(f"- Index: {conv.get('settlement_index', 'N/A')}")
            st.markdown(f"- Physical: {conv.get('physical_settlement', 'N/A')}")
            st.markdown(f"- Cash method: {conv.get('cash_settlement_method', 'N/A')}")
    
    # Market Parcels (AUD only for now)
    if ccy == "AUD":
        with st.expander("📦 Customary Market Parcels (AUD millions)", expanded=False):
            st.markdown("**Swaption Straddles**")
            parcels_df = pd.DataFrame(
                AUD_MARKET_PARCELS["parcels"],
                index=AUD_MARKET_PARCELS["expiries"],
                columns=AUD_MARKET_PARCELS["tenors"]
            )
            parcels_df.index.name = "Expiry"
            st.dataframe(
                parcels_df.style.background_gradient(cmap="Blues", axis=None),
                use_container_width=True,
                height=500
            )
            st.caption("*Bermuda Swaption customary market parcel: AUD 10 million*")
    
    # Business Day
    with st.expander("🏢 Business Day Definition", expanded=False):
        st.markdown(f"**{ccy} Business Day:** {conv.get('business_day', 'N/A')}")

