"""
PhySpace-Studio: Interactive Web 3D Constitutive X-Ray & Tactile Safety-Envelope Workbench.

Generates a standalone, zero-dependency HTML5 + 3D Interactive Studio (`physpace_studio.html`)
embedded with the real 4-Domain 12-Object telemetry compiled on NVIDIA Grace Blackwell GB10 (`NVIDIA_GB10_128GB` + `Qwen3_8_176B_VLM`).
"""

import json
import os
from typing import Any, Dict


def build_interactive_studio_html(
    safety_bench_data: Dict[str, Any],
    output_html_path: str,
) -> str:
    """
    Generate the self-contained `physpace_studio.html` interactive 3D workbench.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    embedded_json = json.dumps(safety_bench_data, separators=(",", ":"))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhySpace-Studio | 3D Constitutive X-Ray & Tactile Safety Envelope Workbench</title>
<style>
  :root {{
    --bg-main: #0b0f17;
    --bg-panel: #131a26;
    --bg-card: #1b2434;
    --border: #2a364f;
    --text-main: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --safe: #3fb950;
    --warn: #d29922;
    --danger: #f85149;
    --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }}
  body {{ background: var(--bg-main); color: var(--text-main); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header {{
    height: 54px; background: var(--bg-panel); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between; padding: 0 20px;
  }}
  .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 700; font-size: 17px; color: var(--text-main); }}
  .badge-cluster {{
    background: rgba(88, 166, 255, 0.14); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.4);
    font-size: 12px; padding: 4px 10px; border-radius: 99px; font-weight: 600;
  }}
  .main-grid {{
    flex: 1; display: grid; grid-template-columns: 330px 1fr 410px; gap: 1px;
    background: var(--border); overflow: hidden;
  }}
  .panel {{ background: var(--bg-panel); padding: 16px; display: flex; flex-direction: column; gap: 14px; overflow-y: auto; }}
  .section-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.8px; color: var(--accent); font-weight: 700; margin-bottom: 6px; }}
  .domain-group {{ margin-bottom: 8px; }}
  .domain-label {{ font-size: 11px; color: var(--text-muted); margin: 6px 0 4px; font-weight: 600; }}
  .obj-btn {{
    width: 100%; text-align: left; background: var(--bg-card); border: 1px solid var(--border);
    color: var(--text-main); padding: 8px 11px; border-radius: 6px; cursor: pointer;
    margin-bottom: 5px; font-size: 12.5px; display: flex; justify-content: space-between; align-items: center;
    transition: all 0.15s;
  }}
  .obj-btn:hover {{ border-color: var(--accent); }}
  .obj-btn.active {{ background: rgba(88, 166, 255, 0.18); border-color: var(--accent); font-weight: 600; }}
  .obj-tag {{ font-size: 10.5px; color: var(--text-muted); background: #0d1117; padding: 2px 6px; border-radius: 4px; }}
  .card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
  .prop-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 12px; }}
  .prop-item {{ background: #101622; padding: 7px 9px; border-radius: 5px; border: 1px solid #222d42; }}
  .prop-label {{ color: var(--text-muted); font-size: 10.5px; }}
  .prop-val {{ font-size: 13.5px; font-weight: 700; color: var(--text-main); margin-top: 2px; }}
  .center-stage {{ background: #090d14; display: flex; flex-direction: column; position: relative; overflow: hidden; }}
  .viewport-toolbar {{
    position: absolute; top: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between;
    pointer-events: none; z-index: 10;
  }}
  .toolbar-box {{
    pointer-events: auto; background: rgba(19, 26, 38, 0.88); backdrop-filter: blur(6px);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px; display: flex; gap: 12px; align-items: center;
  }}
  .toggle-btn {{
    background: var(--bg-card); color: var(--text-main); border: 1px solid var(--border);
    padding: 5px 11px; border-radius: 5px; cursor: pointer; font-size: 12px; font-weight: 600;
  }}
  .toggle-btn.active {{ background: var(--purple); color: #090d14; border-color: var(--purple); }}
  .status-pill {{
    padding: 6px 14px; border-radius: 6px; font-weight: 700; font-size: 13px; letter-spacing: 0.4px;
  }}
  .status-SAFE {{ background: rgba(63, 185, 80, 0.2); color: var(--safe); border: 1px solid var(--safe); }}
  .status-SLIP {{ background: rgba(210, 153, 34, 0.2); color: var(--warn); border: 1px solid var(--warn); }}
  .status-CRUSH {{ background: rgba(248, 81, 73, 0.25); color: var(--danger); border: 1px solid var(--danger); }}
  #canvas3d {{ flex: 1; width: 100%; height: 100%; cursor: grab; }}
  #canvas3d:active {{ cursor: grabbing; }}
  .scrub-bar {{
    background: var(--bg-panel); border-top: 1px solid var(--border); padding: 14px 22px;
    display: flex; flex-direction: column; gap: 8px;
  }}
  .slider-row {{ display: flex; align-items: center; gap: 14px; }}
  input[type=range] {{ flex: 1; accent-color: var(--accent); height: 6px; cursor: pointer; }}
  .envelope-bar {{
    height: 12px; border-radius: 6px; display: flex; overflow: hidden; border: 1px solid var(--border);
  }}
  canvas.plot-canvas {{ width: 100%; background: #0d121c; border: 1px solid var(--border); border-radius: 6px; }}
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>⚡ PhySpace-Studio</span>
    <span style="color:var(--text-muted);font-weight:400;font-size:13px;">| 3D Shell-Core Constitutive X-Ray & 200Hz Tactile Safety Envelope Workbench</span>
  </div>
  <div class="badge-cluster" id="clusterInfo">Dual-Engine: Qwen3_8_176B_VLM (Qwen3.8-176B) + NVIDIA_GB10_128GB (GB10 RT-Patch-IPC)</div>
</header>

<div class="main-grid">
  <!-- Left Panel: 4-Domain Catalog & Constitutive X-Ray Specs -->
  <div class="panel">
    <div>
      <div class="section-title">1. 4-Domain SCPP Asset Catalog (.physpack)</div>
      <div id="catalogList"></div>
    </div>

    <div class="card">
      <div class="section-title">2. Shell-to-Core Constitutive X-Ray</div>
      <div id="objDesc" style="font-size:11.5px;color:var(--text-muted);margin-bottom:8px;"></div>
      <div class="prop-grid">
        <div class="prop-item"><div class="prop-label">Young's Modulus E</div><div class="prop-val" id="valE">-</div></div>
        <div class="prop-item"><div class="prop-label">Yield / Crush Limit σ_y</div><div class="prop-val" id="valYield" style="color:#f0883e;">-</div></div>
        <div class="prop-item"><div class="prop-label">Poisson's Ratio ν</div><div class="prop-val" id="valNu">-</div></div>
        <div class="prop-item"><div class="prop-label">Friction μ_s / μ_d</div><div class="prop-val" id="valMu">-</div></div>
        <div class="prop-item"><div class="prop-label">Skin Thickness</div><div class="prop-val" id="valSkin">-</div></div>
        <div class="prop-item"><div class="prop-label">Inner 12-DoF Core</div><div class="prop-val" id="valCore">-</div></div>
      </div>
    </div>

    <div class="card">
      <div class="section-title">3. Golden Grasp Safety Envelope</div>
      <div class="prop-grid">
        <div class="prop-item"><div class="prop-label">Anti-Slip Min P_min</div><div class="prop-val" id="valPmin" style="color:var(--warn);">-</div></div>
        <div class="prop-item"><div class="prop-label">Crush Max P_max</div><div class="prop-val" id="valPmax" style="color:var(--danger);">-</div></div>
        <div class="prop-item"><div class="prop-label">Optimal Target P*</div><div class="prop-val" id="valPopt" style="color:var(--safe);">-</div></div>
        <div class="prop-item"><div class="prop-label">Safe Corridor Width</div><div class="prop-val" id="valPwidth">-</div></div>
      </div>
    </div>
  </div>

  <!-- Center Panel: Interactive 3D SCPP Mesh & X-Ray Cutaway Viewport -->
  <div class="center-stage">
    <div class="viewport-toolbar">
      <div class="toolbar-box">
        <button class="toggle-btn" id="btnXray" onclick="toggleXray()">🔬 X-Ray Shell/Core Cutaway: OFF</button>
        <button class="toggle-btn active" id="btnWire" onclick="toggleWire()">🕸️ Tet-Surface Wireframe: ON</button>
        <span style="font-size:11.5px;color:var(--text-muted);">Drag to Rotate 3D | Scroll to Zoom</span>
      </div>
      <div class="toolbar-box">
        <div id="regimePill" class="status-pill status-SAFE">GOLDEN SAFE GRASP WINDOW</div>
      </div>
    </div>

    <canvas id="canvas3d"></canvas>

    <div class="scrub-bar">
      <div style="display:flex;justify-content:space-between;font-size:12.5px;font-weight:600;">
        <span>🤖 Interactive Robotic Finger Pinch Indentation Probe (Keyframe Scrubber)</span>
        <span id="scrubReadout" style="color:var(--accent);">Step 9 | Indent: 1.62 mm | Peak Pressure: 3.10 kPa</span>
      </div>
      <div class="slider-row">
        <span style="font-size:11px;color:var(--warn);">Loose / Slip</span>
        <input type="range" id="pinchSlider" min="0" max="4" step="1" value="2" oninput="onSliderChange(this.value)">
        <span style="font-size:11px;color:var(--danger);">Tight / Crush</span>
      </div>
    </div>
  </div>

  <!-- Right Panel: Live 16x16 Tactile Pascal Pressure Map & Safety Curve -->
  <div class="panel">
    <div class="card">
      <div class="section-title">4. Fingertip 16×16 Pascal Pressure P(u,v) & Shear τ(u,v)</div>
      <canvas id="canvasTactile" class="plot-canvas" width="360" height="290"></canvas>
      <div style="display:flex;justify-content:space-between;font-size:11.5px;color:var(--text-muted);margin-top:6px;">
        <span id="tactileForceReadout">Normal Force F_n: 0.00 N</span>
        <span id="tactileShearReadout">Anti-Gravity Shear F_t: 0.00 N</span>
      </div>
    </div>

    <div class="card">
      <div class="section-title">5. Pinch Stress-Pressure Transfer Curve vs. Crush Limit</div>
      <canvas id="canvasCurve" class="plot-canvas" width="360" height="250"></canvas>
    </div>
  </div>
</div>

<script>
const BENCH_DATA = {embedded_json};
const OBJECTS = BENCH_DATA.objects;
let currentObjId = Object.keys(OBJECTS)[0];
let currentKeyframeIdx = 2;
let xrayMode = false;
let showWire = true;

// 3D Camera state
let camAzim = -0.75;
let camElev = 0.38;
let camZoom = 26.0;
let isDragging = false;
let lastMouseX = 0, lastMouseY = 0;

function initCatalog() {{
  const container = document.getElementById('catalogList');
  container.innerHTML = '';
  const domains = {{}};
  for (const [oid, obj] of Object.entries(OBJECTS)) {{
    if (!domains[obj.domain]) domains[obj.domain] = [];
    domains[obj.domain].push(obj);
  }}
  for (const [domName, list] of Object.entries(domains)) {{
    const grp = document.createElement('div');
    grp.className = 'domain-group';
    const lbl = document.createElement('div');
    lbl.className = 'domain-label';
    lbl.textContent = domName.replace(/_/g, ' ');
    grp.appendChild(lbl);
    for (const item of list) {{
      const btn = document.createElement('button');
      btn.className = 'obj-btn' + (item.object_id === currentObjId ? ' active' : '');
      btn.id = 'btn_' + item.object_id;
      btn.onclick = () => selectObject(item.object_id);
      btn.innerHTML = `<span>${{item.display_name}}</span><span class="obj-tag">σ_y=${{item.constitutive_summary.yield_stress_kpa}}kPa</span>`;
      grp.appendChild(btn);
    }}
    container.appendChild(grp);
  }}
}}

function selectObject(oid) {{
  currentObjId = oid;
  document.querySelectorAll('.obj-btn').forEach(b => b.classList.remove('active'));
  const activeBtn = document.getElementById('btn_' + oid);
  if (activeBtn) activeBtn.classList.add('active');
  updateAllViews();
}}

function toggleXray() {{
  xrayMode = !xrayMode;
  const btn = document.getElementById('btnXray');
  btn.classList.toggle('active', xrayMode);
  btn.textContent = `🔬 X-Ray Shell/Core Cutaway: ${{xrayMode ? 'ON' : 'OFF'}}`;
  render3DViewport();
}}

function toggleWire() {{
  showWire = !showWire;
  const btn = document.getElementById('btnWire');
  btn.classList.toggle('active', showWire);
  btn.textContent = `🕸️ Tet-Surface Wireframe: ${{showWire ? 'ON' : 'OFF'}}`;
  render3DViewport();
}}

function onSliderChange(val) {{
  currentKeyframeIdx = parseInt(val);
  updateAllViews();
}}

function updateAllViews() {{
  const obj = OBJECTS[currentObjId];
  const c = obj.constitutive_summary;
  const env = obj.golden_safety_envelope;
  const kf = obj.interactive_keyframes[currentKeyframeIdx];

  document.getElementById('objDesc').textContent = obj.semantic_desc;
  document.getElementById('valE').textContent = `${{c.youngs_modulus_kpa}} kPa`;
  document.getElementById('valYield').textContent = `${{c.yield_stress_kpa}} kPa`;
  document.getElementById('valNu').textContent = c.poisson_ratio.toFixed(2);
  document.getElementById('valMu').textContent = `${{c.mu_static.toFixed(2)}} / ${{c.mu_dynamic.toFixed(2)}}`;
  document.getElementById('valSkin').textContent = `${{c.shell_thickness_mm}} mm`;
  document.getElementById('valCore').textContent = c.has_rigid_core ? '12-DoF Affine Core' : 'Full Deformable';

  document.getElementById('valPmin').textContent = `${{env.p_slip_min_kpa}} kPa`;
  document.getElementById('valPmax').textContent = `${{env.p_crush_max_kpa}} kPa`;
  document.getElementById('valPopt').textContent = `${{env.optimal_target_pressure_kpa}} kPa`;
  document.getElementById('valPwidth').textContent = `${{env.safety_corridor_width_kpa}} kPa`;

  document.getElementById('scrubReadout').textContent =
    `Step ${{kf.step}} | Indent: ${{kf.indent_mm}} mm | Peak P: ${{kf.peak_pressure_kpa}} kPa | von Mises: ${{kf.von_mises_kpa}} kPa`;
  document.getElementById('tactileForceReadout').textContent = `Normal Force F_n: ${{kf.normal_force_n.toFixed(3)}} N`;
  document.getElementById('tactileShearReadout').textContent = `Anti-Gravity Shear F_t: ${{kf.shear_force_n.toFixed(3)}} N`;

  const pill = document.getElementById('regimePill');
  pill.className = 'status-pill';
  if (kf.regime === 'CRUSH_DAMAGE') {{
    pill.classList.add('status-CRUSH');
    pill.textContent = `⚠️ CRUSH / YIELD EXCEEDED (${{kf.von_mises_kpa}} > ${{c.yield_stress_kpa}} kPa)`;
  }} else if (kf.regime === 'SLIP_DROP_RISK') {{
    pill.classList.add('status-SLIP');
    pill.textContent = `⚠️ INCIPIENT SLIP / DROP RISK (F_n = ${{kf.normal_force_n.toFixed(2)}} N)`;
  }} else {{
    pill.classList.add('status-SAFE');
    pill.textContent = `✅ GOLDEN SAFE GRASP WINDOW (${{kf.von_mises_kpa}} / ${{c.yield_stress_kpa}} kPa)`;
  }}

  render3DViewport();
  renderTactileHeatmap(kf);
  renderSafetyEnvelopeCurve(obj, kf);
}}

function plasmaColor(t, alpha=1.0) {{
  const c = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, 0.15 + 1.15 * c));
  const g = Math.round(255 * Math.max(0, Math.sin(c * Math.PI * 0.85)));
  const b = Math.round(255 * (0.65 - 0.55 * c + 0.25 * Math.cos(c * Math.PI)));
  return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

function render3DViewport() {{
  const canvas = document.getElementById('canvas3d');
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#090d14';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const obj = OBJECTS[currentObjId];
  const kf = obj.interactive_keyframes[currentKeyframeIdx];
  const verts = kf.surface_vertices_cm;
  const press = kf.surface_pressure_kpa;
  const faces = obj.surface_faces_local;

  // Compute Z centroid to center camera
  let zSum = 0;
  for (let i = 0; i < verts.length; i++) zSum += verts[i][2];
  const zCenter = zSum / Math.max(verts.length, 1);

  const cosA = Math.cos(camAzim), sinA = Math.sin(camAzim);
  const cosE = Math.cos(camElev), sinE = Math.sin(camElev);
  const cx = canvas.width * 0.5;
  const cy = canvas.height * 0.54;

  function project(pt) {{
    const x = pt[0], y = pt[1], z = pt[2] - zCenter;
    const rx = x * cosA - y * sinA;
    const ry = x * sinA + y * cosA;
    const rz = z * cosE - ry * sinE;
    const depth = z * sinE + ry * cosE;
    const scale = camZoom * (35.0 / Math.max(35.0 - depth, 8.0));
    return [cx + rx * scale, cy - rz * scale, depth, ry];
  }}

  const projVerts = verts.map(project);
  const maxP = Math.max(...press, 1.0);

  // Build sortable triangles
  const drawList = [];
  for (let f = 0; f < faces.length; f++) {{
    const [i0, i1, i2] = faces[f];
    if (i0 < 0 || i1 < 0 || i2 < 0) continue;
    const p0 = projVerts[i0], p1 = projVerts[i1], p2 = projVerts[i2];
    const avgFront = (p0[3] + p1[3] + p2[3]) / 3.0;
    // In X-Ray mode, slice away front half (avgFront < 0)
    if (xrayMode && avgFront < -0.2) continue;
    const avgDepth = (p0[2] + p1[2] + p2[2]) / 3.0;
    const avgP = (press[i0] + press[i1] + press[i2]) / 3.0;
    drawList.push({{ p0, p1, p2, avgDepth, avgP }});
  }}
  drawList.sort((a, b) => a.avgDepth - b.avgDepth);

  // If X-Ray mode is ON, draw inner 12-DoF Affine Core nodes first
  if (xrayMode && obj.xray_core_vertices_cm) {{
    ctx.fillStyle = '#bc8cff';
    for (const cv of obj.xray_core_vertices_cm) {{
      const pv = project(cv);
      ctx.beginPath();
      ctx.arc(pv[0], pv[1], 3.2, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}

  for (const tri of drawList) {{
    const normP = Math.min(1.0, tri.avgP / maxP);
    ctx.beginPath();
    ctx.moveTo(tri.p0[0], tri.p0[1]);
    ctx.lineTo(tri.p1[0], tri.p1[1]);
    ctx.lineTo(tri.p2[0], tri.p2[1]);
    ctx.closePath();
    ctx.fillStyle = plasmaColor(normP, xrayMode ? 0.62 : 0.85);
    ctx.fill();
    if (showWire) {{
      ctx.strokeStyle = normP > 0.15 ? 'rgba(255,220,100,0.65)' : 'rgba(60,75,105,0.45)';
      ctx.lineWidth = normP > 0.15 ? 1.1 : 0.55;
      ctx.stroke();
    }}
  }}
}}

function renderTactileHeatmap(kf) {{
  const canvas = document.getElementById('canvasTactile');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0d121c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const pGrid = kf.pressure_grid_16x16_kpa;
  const tauGrid = kf.shear_grid_16x16_kpa;
  const H = pGrid.length, W = pGrid[0].length;
  const padL = 38, padT = 18, size = 240;
  const cellW = size / W, cellH = size / H;

  let maxP = 0.5;
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) if (pGrid[r][c] > maxP) maxP = pGrid[r][c];

  for (let r = 0; r < H; r++) {{
    for (let c = 0; c < W; c++) {{
      const val = pGrid[H - 1 - r][c] / maxP;
      ctx.fillStyle = plasmaColor(val, 1.0);
      ctx.fillRect(padL + c * cellW, padT + r * cellH, cellW + 0.6, cellH + 0.6);
    }}
  }}

  // Draw shear traction vectors τ(u, v)
  ctx.strokeStyle = '#58a6ff';
  ctx.fillStyle = '#58a6ff';
  ctx.lineWidth = 1.5;
  for (let r = 0; r < H; r += 2) {{
    for (let c = 0; c < W; c += 2) {{
      const pVal = pGrid[H - 1 - r][c];
      if (pVal < 0.08 * maxP) continue;
      const [tu, tv] = tauGrid[H - 1 - r][c];
      const cx = padL + (c + 0.5) * cellW;
      const cy = padT + (r + 0.5) * cellH;
      const vx = tu * 16.0;
      const vy = -tv * 16.0;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + vx, cy + vy);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx + vx, cy + vy, 2.0, 0, Math.PI * 2);
      ctx.fill();
    }}
  }}

  // Colorbar
  const cbX = padL + size + 18;
  for (let i = 0; i < size; i++) {{
    ctx.fillStyle = plasmaColor(1.0 - i / size, 1.0);
    ctx.fillRect(cbX, padT + i, 14, 2);
  }}
  ctx.fillStyle = '#e6edf3';
  ctx.font = '11px sans-serif';
  ctx.fillText(`${{maxP.toFixed(1)}} kPa`, cbX + 18, padT + 10);
  ctx.fillText('0.0 kPa', cbX + 18, padT + size);
}}

function renderSafetyEnvelopeCurve(obj, kf) {{
  const canvas = document.getElementById('canvasCurve');
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0d121c';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const ind = obj.sweep_curves.indent_mm;
  const pCurve = obj.sweep_curves.pressure_kpa;
  const vmCurve = obj.sweep_curves.von_mises_kpa;
  const yieldLim = obj.constitutive_summary.yield_stress_kpa;

  const padL = 44, padR = 18, padT = 22, padB = 34;
  const W = canvas.width - padL - padR;
  const H = canvas.height - padT - padB;
  const maxY = Math.max(...vmCurve, ...pCurve, yieldLim * 1.25, 2.0);
  const maxX = Math.max(...ind, 1.0);

  function toXY(xVal, yVal) {{
    return [padL + (xVal / maxX) * W, padT + H - Math.min(1.0, yVal / maxY) * H];
  }}

  // Draw Crush Threshold line
  const [, yCrush] = toXY(0, yieldLim);
  ctx.fillStyle = 'rgba(248, 81, 73, 0.12)';
  ctx.fillRect(padL, padT, W, Math.max(0, yCrush - padT));
  ctx.strokeStyle = '#f0883e';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, yCrush);
  ctx.lineTo(padL + W, yCrush);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f0883e';
  ctx.font = '10.5px sans-serif';
  ctx.fillText(`Crush Limit σ_y = ${{yieldLim}} kPa`, padL + 6, Math.max(padT + 12, yCrush - 5));

  // Plot von Mises curve (green) and Contact Pressure curve (blue)
  function drawSeries(xArr, yArr, color) {{
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < xArr.length; i++) {{
      const [px, py] = toXY(xArr[i], yArr[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }}
    ctx.stroke();
  }}
  drawSeries(ind, vmCurve, '#3fb950');
  drawSeries(ind, pCurve, '#58a6ff');

  // Draw current keyframe vertical marker
  const [curX, curY] = toXY(kf.indent_mm, kf.von_mises_kpa);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.moveTo(curX, padT);
  ctx.lineTo(curX, padT + H);
  ctx.stroke();
  ctx.fillStyle = '#3fb950';
  ctx.beginPath();
  ctx.arc(curX, curY, 4.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#8b949e';
  ctx.fillText('Pinch Indentation (mm)', padL + W * 0.32, canvas.height - 8);
}}

// Mouse interaction for 3D canvas
const c3d = document.getElementById('canvas3d');
c3d.addEventListener('mousedown', e => {{ isDragging = true; lastMouseX = e.clientX; lastMouseY = e.clientY; }});
window.addEventListener('mouseup', () => {{ isDragging = false; }});
window.addEventListener('mousemove', e => {{
  if (!isDragging) return;
  const dx = e.clientX - lastMouseX;
  const dy = e.clientY - lastMouseY;
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
  camAzim += dx * 0.012;
  camElev = Math.max(-1.2, Math.min(1.2, camElev + dy * 0.012));
  render3DViewport();
}});
c3d.addEventListener('wheel', e => {{
  e.preventDefault();
  camZoom = Math.max(10.0, Math.min(65.0, camZoom * (e.deltaY < 0 ? 1.08 : 0.92)));
  render3DViewport();
}}, {{ passive: false }});
window.addEventListener('resize', render3DViewport);

initCatalog();
updateAllViews();
</script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_html_path
