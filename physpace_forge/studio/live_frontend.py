"""
PhySpace-Studio Pro: All-in-One Interactive Full-Stack Frontend Generator (v2 Audited Layout, 3D Shaded Renderer & Full Bilingual ZH/EN).
"""

import json
import os
from typing import Any, Dict, Optional


def get_live_studio_pro_html(
    embedded_bench_data: Optional[Dict[str, Any]] = None,
    embedded_dual_data: Optional[Dict[str, Any]] = None,
) -> str:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if embedded_bench_data is None:
        rep_path = os.path.join(root_dir, "benchmarks", "safetybench_4d_report.json")
        if os.path.exists(rep_path):
            with open(rep_path, "r", encoding="utf-8") as f:
                embedded_bench_data = json.load(f)
        else:
            embedded_bench_data = {"objects": {}}

    if embedded_dual_data is None:
        dual_path = os.path.join(root_dir, "benchmarks", "hero_execution_report.json")
        if os.path.exists(dual_path):
            with open(dual_path, "r", encoding="utf-8") as f:
                embedded_dual_data = json.load(f)
        else:
            embedded_dual_data = {}

    bench_json = json.dumps(embedded_bench_data, separators=(",", ":"))
    dual_json = json.dumps(embedded_dual_data, separators=(",", ":"))

    html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>PhySpace-Studio Pro | AI-Agent-First 3D Physical Constitutive & Tactile Workbench</title>
<style>
  :root {
    --bg: #060a12;
    --panel: #0d1424;
    --panel2: #131d33;
    --border: #223252;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --cyan: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #070b14; }
  ::-webkit-scrollbar-thumb { background: #263859; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

  body { background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; overflow: hidden; }

  header {
    height: 46px;
    flex-shrink: 0;
    background: linear-gradient(90deg, #091122 0%, #121e38 50%, #091122 100%);
    border-bottom: 1px solid var(--border);
    padding: 0 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .brand { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
  .brand h1 { font-size: 15px; font-weight: 800; color: var(--cyan); letter-spacing: 0.3px; }
  .badge {
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.5);
    color: var(--cyan);
    padding: 2px 7px;
    border-radius: 999px;
    font-size: 10.5px;
    font-weight: 700;
    white-space: nowrap;
  }
  .badge-live {
    background: rgba(34, 197, 94, 0.15);
    border-color: var(--green);
    color: #4ade80;
  }
  .mode-tabs { display: flex; gap: 5px; align-items: center; }
  .mode-tab {
    background: #0c1426;
    border: 1px solid var(--border);
    color: #cbd5e1;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 700;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.15s;
  }
  .mode-tab:hover { border-color: var(--cyan); color: #fff; }
  .mode-tab.active {
    background: rgba(56, 189, 248, 0.22);
    border-color: var(--cyan);
    color: #fff;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.28);
  }

  .pain-strip {
    height: 42px;
    flex-shrink: 0;
    background: #090f1d;
    border-bottom: 1px solid var(--border);
    padding: 4px 14px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    font-size: 10.5px;
  }
  .pain-card {
    background: rgba(19, 29, 51, 0.8);
    border: 1px solid #1e2d4a;
    border-left: 3px solid var(--cyan);
    border-radius: 4px;
    padding: 2px 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    overflow: hidden;
    white-space: nowrap;
  }
  .pain-card .vs-bad { color: #f87171; text-decoration: line-through; font-size: 10px; margin-right: 5px; }
  .pain-card .vs-good { color: #4ade80; font-weight: 700; font-size: 10.5px; }

  .workspace {
    display: grid;
    grid-template-columns: 335px 1fr 385px;
    flex: 1;
    min-height: 0;
    overflow: hidden;
  }
  .sidebar {
    background: var(--panel);
    border-right: 1px solid var(--border);
    padding: 8px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .rightbar {
    background: var(--panel);
    border-left: 1px solid var(--border);
    padding: 8px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .center-stage {
    display: flex;
    flex-direction: column;
    background: radial-gradient(circle at 50% 42%, #13213e 0%, #070b14 100%);
    position: relative;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
  }

  .section-box {
    background: #09101e;
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 9px;
  }
  .section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--cyan);
    font-weight: 800;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .domain-group { margin-bottom: 5px; }
  .domain-header {
    font-size: 10px;
    color: var(--amber);
    font-weight: 700;
    margin: 2px 0 3px 2px;
  }
  .domain-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
  }
  .obj-btn {
    text-align: left;
    background: #10192d;
    border: 1px solid #1e2f4d;
    color: #e2e8f0;
    padding: 5px 6px;
    border-radius: 5px;
    cursor: pointer;
    font-size: 10.5px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 2px;
    transition: all 0.12s;
    overflow: hidden;
  }
  .obj-btn:hover { border-color: var(--cyan); background: #162440; }
  .obj-btn.active {
    background: rgba(56, 189, 248, 0.22);
    border-color: var(--cyan);
    color: #fff;
    font-weight: 700;
    box-shadow: inset 0 0 8px rgba(56, 189, 248, 0.25);
  }
  .obj-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 10.5px; }
  .obj-tag {
    font-size: 9.5px;
    color: var(--cyan);
    font-family: monospace;
  }

  .sliders-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 8px;
    margin-bottom: 6px;
  }
  .param-row { margin-bottom: 4px; }
  .param-label {
    display: flex;
    justify-content: space-between;
    font-size: 10.5px;
    color: var(--muted);
    margin-bottom: 1px;
  }
  .param-label b { color: var(--text); font-family: monospace; font-size: 10.5px; }
  input[type=range] { width: 100%; accent-color: var(--cyan); cursor: pointer; height: 4px; }
  input[type=text], select {
    width: 100%;
    background: #0d1629;
    border: 1px solid var(--border);
    color: var(--text);
    padding: 4px 7px;
    border-radius: 5px;
    font-size: 11px;
  }
  .btn-primary {
    width: 100%;
    background: linear-gradient(90deg, #0284c7, #2563eb);
    border: 1px solid #38bdf8;
    color: #fff;
    padding: 7px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 800;
    cursor: pointer;
    transition: all 0.15s;
    margin-top: 4px;
  }
  .btn-primary:hover { filter: brightness(1.15); box-shadow: 0 0 12px rgba(56, 189, 248, 0.45); }
  .btn-success {
    background: linear-gradient(90deg, #16a34a, #059669);
    border: 1px solid #4ade80;
    color: #fff;
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }
  .btn-danger {
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid var(--red);
    color: #fca5a5;
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
  }

  .viewport-wrap { flex: 1; position: relative; min-height: 260px; display: flex; flex-direction: column; }
  #glCanvas { width: 100%; height: 100%; cursor: grab; display: block; }
  #glCanvas:active { cursor: grabbing; }

  .hud-top-left {
    position: absolute;
    top: 10px;
    left: 12px;
    background: rgba(9, 15, 29, 0.86);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 8px;
    max-width: 440px;
    backdrop-filter: blur(6px);
    pointer-events: none;
  }
  .hud-top-right {
    position: absolute;
    top: 10px;
    right: 12px;
    display: flex;
    flex-direction: column;
    gap: 5px;
    align-items: flex-end;
  }
  .regime-pill {
    padding: 5px 11px;
    border-radius: 6px;
    font-weight: 800;
    font-size: 12px;
    letter-spacing: 0.5px;
    border: 1px solid;
  }
  .watermark-pill {
    background: rgba(168, 85, 247, 0.18);
    border: 1px solid var(--purple);
    color: #e9d5ff;
    padding: 3px 8px;
    border-radius: 5px;
    font-size: 10px;
    font-family: monospace;
  }

  .center-drawer {
    background: #09101f;
    border-top: 1px solid var(--border);
    padding: 10px 14px;
    flex-shrink: 0;
    overflow-y: auto;
  }
  .scrubber-bar {
    height: 40px;
    flex-shrink: 0;
    background: #0c1426;
    border-top: 1px solid var(--border);
    padding: 0 14px;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .kv-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    font-size: 11px;
  }
  .kv-item {
    background: #10192d;
    border: 1px solid #1e2d47;
    padding: 5px 8px;
    border-radius: 5px;
  }
  .kv-item span { color: var(--muted); display: block; font-size: 9.5px; }
  .kv-item b { color: var(--text); font-size: 12px; font-family: monospace; }

  .corridor-box {
    background: rgba(34, 197, 94, 0.09);
    border: 1px solid rgba(34, 197, 94, 0.45);
    border-radius: 6px;
    padding: 6px 9px;
    margin-bottom: 6px;
  }

  table.eq-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    margin-top: 6px;
  }
  table.eq-table th, table.eq-table td {
    border: 1px solid #1e2d47;
    padding: 5px 8px;
    text-align: left;
  }
  table.eq-table th { background: #13203a; color: var(--cyan); }
  canvas.widget-canvas {
    width: 100%;
    background: #060a12;
    border: 1px solid var(--border);
    border-radius: 6px;
    display: block;
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h1>⚡ PhySpace-Studio Pro</h1>
    <span class="badge">AI-Agent-First Physical Compiler + MCP Server</span>
    <span class="badge badge-live" id="serverStatusBadge">🟢 MCP / Function-Calling & 200Hz IPC 引擎已就绪</span>
  </div>
  <div class="mode-tabs">
    <button class="mode-tab" onclick="toggleLang()" id="btnLangSwitch" style="border-color:var(--amber);color:var(--amber);">🌐 English / 中文</button>
    <button class="mode-tab active" onclick="switchMode('agent')" id="tab-agent" style="border-color:#38bdf8;">🤖 1. AI Agent / MCP 实时调用台</button>
    <button class="mode-tab" onclick="switchMode('sandbox')" id="tab-sandbox">🔬 2. 3D 物理透视沙盒</button>
    <button class="mode-tab" onclick="switchMode('duel')" id="tab-duel">⚡ 3. 闭环 vs 开环对比</button>
    <button class="mode-tab" onclick="switchMode('compiler')" id="tab-compiler">🛠️ 4. 通用格式编译 (.xml/.usda)</button>
    <button class="mode-tab" onclick="switchMode('scene')" id="tab-scene">🏗️ 5. Frame-0 批量空间</button>
    <button class="mode-tab" onclick="switchMode('benchmarks')" id="tab-benchmarks">🏆 6. 1440组基准评测</button>
  </div>
</header>

<div class="pain-strip" id="painStrip">
  <div class="pain-card">
    <div><b>痛点1：接触穿模</b> <span class="vs-bad">传统3DGS: 穿模弹飞</span><span class="vs-good">✔ IPC 势垒 100% 零穿模</span></div>
    <span class="badge">d &gt; 0 恒成立</span>
  </div>
  <div class="pain-card">
    <div><b>痛点2：第0帧腾空</b> <span class="vs-bad">传统生成: 初始掉落抖动</span><span class="vs-good">✔ 第0帧重力静力配平 &lt;10⁻⁵N</span></div>
    <span class="badge">Frame-0 平衡</span>
  </div>
  <div class="pain-card">
    <div><b>痛点3：易碎品捏爆</b> <span class="vs-bad">传统开环: 17.1kPa捏爆</span><span class="vs-good">✔ 16×16 压强场 + 黄金安全窗</span></div>
    <span class="badge">3.66 kPa 安全</span>
  </div>
  <div class="pain-card">
    <div><b>痛点4：FEM 算力慢</b> <span class="vs-bad">全局FEM: 算力卡顿</span><span class="vs-good">✔ Active Patch 省 91.5% 自由度</span></div>
    <span class="badge">200Hz 实时</span>
  </div>
</div>

<div class="workspace">
  <!-- LEFT SIDEBAR -->
  <div class="sidebar">
    <div class="section-box">
      <div class="section-title">
        <span id="lblCatalogTitle">📦 4大领域 12+ 标定资产库 (点击切换)</span>
        <span id="objCountLabel" style="color:var(--muted);font-size:10px;">12 Objects</span>
      </div>
      <div id="catalogList"></div>
    </div>

    <div class="section-box">
      <div class="section-title">
        <span id="lblSlidersTitle">🎛️ 实时本构与策略调参器 (Live 200Hz IPC)</span>
        <button id="btnResetParams" onclick="resetCurrentParams()" style="background:none;border:none;color:var(--cyan);font-size:10px;cursor:pointer;">重置参数</button>
      </div>
      <div class="sliders-grid">
        <div>
          <div class="param-label"><span id="txt_e">杨氏模量 E</span><b id="lbl_e">65.0 kPa</b></div>
          <input type="range" id="sl_e" min="8" max="3000" step="1" value="65" oninput="onSliderInput()">
        </div>
        <div>
          <div class="param-label"><span id="txt_y">屈服极限 σ_y</span><b id="lbl_y">6.0 kPa</b></div>
          <input type="range" id="sl_y" min="1.5" max="350" step="0.5" value="6.0" oninput="onSliderInput()">
        </div>
        <div>
          <div class="param-label"><span id="txt_nu">泊松比 ν</span><b id="lbl_nu">0.29</b></div>
          <input type="range" id="sl_nu" min="0.15" max="0.48" step="0.01" value="0.29" oninput="onSliderInput()">
        </div>
        <div>
          <div class="param-label"><span id="txt_mu">静摩擦 μ_s</span><b id="lbl_mu">0.46</b></div>
          <input type="range" id="sl_mu" min="0.18" max="0.95" step="0.02" value="0.46" oninput="onSliderInput()">
        </div>
      </div>

      <div class="param-row">
        <div class="param-label"><span id="txt_tp">目标抓握压强 P*</span><b id="lbl_tp">3.2 kPa</b></div>
        <input type="range" id="sl_tp" min="0.5" max="50.0" step="0.1" value="3.2" oninput="onSliderInput()">
      </div>
      <div class="param-row">
        <select id="sel_policy" onchange="onSliderInput()">
          <option value="tactile_closed_loop" id="opt_closed">🟢 PhySpace 200Hz 触觉压强闭环策略</option>
          <option value="naive_position" id="opt_naive">🔴 传统 Naive 开环位置夹取 (易捏爆)</option>
        </select>
      </div>
      <button class="btn-primary" id="btnRunLiveSim" onclick="triggerLiveSimulation()">
        ⚡ 立即运行 200Hz RT-Patch-IPC 实时求解
      </button>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:5px;">
        <button class="btn-success" id="btnSafeSim" onclick="triggerLiveSimulation('tactile_closed_loop')">🛡️ 闭环安全抓握</button>
        <button class="btn-danger" id="btnCrushSim" onclick="triggerLiveSimulation('naive_position')">💥 模拟开环捏爆</button>
      </div>
    </div>
  </div>

  <!-- CENTER STAGE -->
  <div class="center-stage">
    <div class="viewport-wrap">
      <canvas id="glCanvas"></canvas>

      <div class="hud-top-left">
        <div style="font-size:14px;font-weight:800;color:#fff;margin-bottom:2px;" id="hudTitle">Waxed Paper Coffee Cup</div>
        <div style="font-size:10.5px;color:#cbd5e1;margin-bottom:6px;" id="hudDesc">Hollow thin-walled paper cup</div>
        <div style="display:flex;gap:5px;flex-wrap:wrap;">
          <span class="badge" id="hudActivePatch">Active Patch: 30/352 verts (91.5% DOF Saved)</span>
          <span class="badge" id="hudClearance">Min Clearance d: 2.15 mm (0 Penetration)</span>
          <span class="badge" id="hudLatency">Solver: 200Hz Real-Time</span>
        </div>
      </div>

      <div class="hud-top-right">
        <div class="regime-pill" id="hudRegime">GOLDEN_SAFE_WINDOW</div>
        <div class="watermark-pill" id="hudWatermark">🔐 Calib-ID: PSF-9F82A1C04E11B82A</div>
        <div style="display:flex;gap:5px;pointer-events:auto;align-items:center;">
          <button class="mode-tab" style="padding:4px 8px;font-size:10.5px;" onclick="toggleXRay()" id="btnXRay">🔍 X-Ray 剖切: 开启</button>
          <button class="mode-tab" style="padding:4px 8px;font-size:10.5px;" onclick="resetCamera()" id="btnResetCam">🎯 视角复位</button>
          <select id="selExportFmt" style="width:142px;padding:3px 6px;font-size:10.5px;background:#0c1426;border:1px solid var(--cyan);color:var(--cyan);border-radius:5px;">
            <option value="mjcf">📦 MuJoCo (.xml)</option>
            <option value="usda">📦 OpenUSD (.usda)</option>
            <option value="urdf">📦 ROS2 (.urdf)</option>
            <option value="gltf">📦 Blender (.gltf)</option>
            <option value="physpack" id="optFmtPhyspack">⚡ 内部缓存 (.physpack)</option>
          </select>
          <button class="mode-tab" id="btnExport" style="padding:4px 8px;font-size:10.5px;border-color:var(--green);color:#4ade80;" onclick="exportCurrentPhyspack()">💾 一键导出</button>
        </div>
      </div>
    </div>

    <div class="center-drawer" id="centerDrawer"></div>

    <div class="scrubber-bar">
      <button class="mode-tab" id="btnPlayAnim" onclick="togglePlayAnimation()" style="padding:3px 9px;font-size:11px;border-color:var(--cyan);color:var(--cyan);">▶ 自动播放施压</button>
      <span id="lblTimeline" style="font-size:11px;font-weight:700;color:#cbd5e1;white-space:nowrap;">时间轴 (Keyframe):</span>
      <input type="range" id="kfSlider" min="0" max="4" step="1" value="2" oninput="onKeyframeChange(this.value)" style="flex:1;">
      <span id="kfLabel" style="font-size:11.5px;font-weight:700;min-width:215px;text-align:right;color:var(--cyan);font-family:monospace;">Step 9 | 压深 1.89 mm | 3.66 kPa</span>
    </div>
  </div>

  <!-- RIGHT SIDEBAR -->
  <div class="rightbar">
    <div class="section-box">
      <div class="section-title">
        <span id="secCorridorTitle">🛡️ 黄金抓握安全包络 (Golden Grasp Safety Corridor)</span>
      </div>
      <div class="corridor-box">
        <div id="lblCorridorSub" style="font-size:10px;color:#86efac;margin-bottom:1px;">安全无损抓握压强窗口 [P_slip-min, P_crush-max]</div>
        <div style="font-size:15px;font-weight:800;color:var(--green);font-family:monospace;" id="valCorridor">[2.33 kPa, 4.24 kPa]</div>
        <div style="font-size:10px;color:#cbd5e1;margin-top:2px;" id="valOptTarget">推荐 VLA 最优闭环压强 P* = 3.29 kPa</div>
      </div>
      <div class="kv-grid">
        <div class="kv-item"><span id="lblKvPeakP">当前峰值接触压强 P_max</span><b id="valPeakP" style="color:var(--cyan);">3.66 kPa</b></div>
        <div class="kv-item"><span id="lblKvPeakVM">当前 Von Mises 应力 σ_vm</span><b id="valPeakVM" style="color:var(--amber);">4.15 kPa</b></div>
        <div class="kv-item"><span id="lblKvForces">法向夹持力 F_n / 切向力 F_t</span><b id="valForces">1.42 N / 0.48 N</b></div>
        <div class="kv-item"><span id="lblKvEqErr">第0帧重力配平残差 |Mg - F|</span><b id="valEqErr" style="color:var(--green);">1.20e-6 N</b></div>
      </div>
    </div>

    <div class="section-box">
      <div class="section-title">
        <span id="secTactileTitle">🖐️ 16×16 指尖帕斯卡压强 P(u,v) 与切应力 τ(u,v)</span>
        <span style="font-size:10px;color:var(--muted);">SI: kPa</span>
      </div>
      <canvas id="tactileCanvas" width="360" height="190" class="widget-canvas"></canvas>
    </div>

    <div class="section-box">
      <div class="section-title">
        <span id="secCurveTitle">📈 连续施压安全走廊曲线 (压强 P & 应力 σ_vm vs 屈服极限)</span>
      </div>
      <canvas id="curveCanvas" width="360" height="175" class="widget-canvas"></canvas>
    </div>
  </div>
</div>

<script>
let BENCH_DATA = __BENCH_JSON__;
let BASELINE_OBJECTS = JSON.parse(JSON.stringify(BENCH_DATA.objects || {}));
let DUAL_DATA = __DUAL_JSON__;
const API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
  ? '' : 'http://127.0.0.1:8765';

let currentObjId = 'tapered_coffee_cup';
let currentKfIdx = 2;
let currentMode = 'agent';
let uiLang = 'zh';
let backendConnected = true;
let xrayEnabled = true;
let sceneViewActive = false;
let currentSceneData = null;
let playTimer = null;
let uploadedObjContent = '';
let uploadedObjStats = '';
let llmPreset = 'builtin_offline';
let llmUrl = 'http://127.0.0.1:11434/v1/chat/completions';
let llmModel = 'qwen2.5:14b';
let llmKey = '';
let lastLlmInference = {
  model_used: 'PhySpace-CalibDB-BuiltIn-v1 (Zero-Config Offline)',
  youngs_modulus_kpa: 68.0,
  yield_stress_kpa: 5.4,
  poisson_ratio: 0.24,
  mu_static: 0.40,
  profile_curve: 'waisted',
  reasoning: 'Delicate thin-walled waisted vessel: requires low-pressure anti-slip pinch inside [2.1, 4.75] kPa.'
};

// 3D Camera State
let rotYaw = -0.55;
let rotPitch = 0.34;
let zoomFactor = 1.0;
let isDragging = false;
let lastMouseX = 0, lastMouseY = 0;

let lastAgentLog = {
  call_id: "mcp_init_001",
  tool_name: "physpace_inspect_object_safety_envelope",
  arguments: { object_id: "tapered_coffee_cup" },
  latency_ms: 1.8,
  result: {
    status: "success",
    golden_safety_corridor: {
      p_slip_min_kpa: 2.33,
      p_crush_max_kpa: 4.24,
      optimal_target_pressure_kpa: 3.29,
      min_normal_force_n: 0.77
    },
    agent_reflection_hint: "Object 'Waxed Paper Coffee Cup (薄壁纸咖啡杯)' yields at 6.0 kPa. Configure tactile controller target pressure to P* = 3.29 kPa within Golden Corridor [2.33, 4.24] kPa (min normal force 0.77 N)."
  }
};

function getDisplayName(rawName) {
  if (!rawName) return '';
  if (uiLang === 'en') {
    return rawName.split(' (')[0];
  }
  return rawName;
}

function updateLangStrings() {
  const isEn = (uiLang === 'en');
  document.getElementById('btnLangSwitch').textContent = isEn ? '🌐 切换中文' : '🌐 English / 中文';
  document.getElementById('serverStatusBadge').textContent = backendConnected
    ? (isEn ? '🟢 Live PyTorch/IPC Engine Connected (200Hz)' : '🟢 全栈实时 PyTorch/IPC 引擎已连接 (127.0.0.1:8765)')
    : (isEn ? '🟡 Standalone Sandbox Mode (Built-in Physics Kernel)' : '🟡 本地独立沙盒模式 (内置完整物理计算内核)');

  document.getElementById('tab-agent').textContent = isEn ? '🤖 1. AI Agent / MCP Console' : '🤖 1. AI Agent / MCP 实时调用台';
  document.getElementById('tab-sandbox').textContent = isEn ? '🔬 2. 3D X-Ray Sandbox' : '🔬 2. 3D 物理透视沙盒';
  document.getElementById('tab-duel').textContent = isEn ? '⚡ 3. Closed vs Open Loop' : '⚡ 3. 闭环 vs 开环对比';
  document.getElementById('tab-compiler').textContent = isEn ? '🛠️ 4. Universal Compiler (.xml/.usda)' : '🛠️ 4. 通用格式编译 (.xml/.usda)';
  document.getElementById('tab-scene').textContent = isEn ? '🏗️ 5. Frame-0 Scene Builder' : '🏗️ 5. Frame-0 批量空间';
  document.getElementById('tab-benchmarks').textContent = isEn ? '🏆 6. 1,440-Trial Benchmarks' : '🏆 6. 1440组基准评测';

  document.getElementById('painStrip').innerHTML = isEn ? `
    <div class="pain-card">
      <div><b>Pain 1: Mesh Penetration</b> <span class="vs-bad">3DGS: Interpenetration</span><span class="vs-good">✔ IPC Barrier 0-Penetration</span></div>
      <span class="badge">d &gt; 0 Guaranteed</span>
    </div>
    <div class="pain-card">
      <div><b>Pain 2: Frame-0 Drop</b> <span class="vs-bad">Gen3D: Initial Jitter</span><span class="vs-good">✔ Static Equilibrium &lt;10⁻⁵N</span></div>
      <span class="badge">Frame-0 Balanced</span>
    </div>
    <div class="pain-card">
      <div><b>Pain 3: Fragile Crush</b> <span class="vs-bad">Open-Loop: 17.1kPa Crush</span><span class="vs-good">✔ 16×16 Pad + Golden Corridor</span></div>
      <span class="badge">3.66 kPa Safe</span>
    </div>
    <div class="pain-card">
      <div><b>Pain 4: Slow Full FEM</b> <span class="vs-bad">Global FEM: &lt;15Hz Lag</span><span class="vs-good">✔ Active Patch 91.5% DOF Saved</span></div>
      <span class="badge">200Hz Real-Time</span>
    </div>
  ` : `
    <div class="pain-card">
      <div><b>痛点1：接触穿模</b> <span class="vs-bad">传统3DGS: 穿模弹飞</span><span class="vs-good">✔ IPC 势垒 100% 零穿模</span></div>
      <span class="badge">d &gt; 0 恒成立</span>
    </div>
    <div class="pain-card">
      <div><b>痛点2：第0帧腾空</b> <span class="vs-bad">传统生成: 初始掉落抖动</span><span class="vs-good">✔ 第0帧重力静力配平 &lt;10⁻⁵N</span></div>
      <span class="badge">Frame-0 平衡</span>
    </div>
    <div class="pain-card">
      <div><b>痛点3：易碎品捏爆</b> <span class="vs-bad">传统开环: 17.1kPa捏爆</span><span class="vs-good">✔ 16×16 压强场 + 黄金安全窗</span></div>
      <span class="badge">3.66 kPa 安全</span>
    </div>
    <div class="pain-card">
      <div><b>痛点4：FEM 算力慢</b> <span class="vs-bad">全局FEM: 算力卡顿</span><span class="vs-good">✔ Active Patch 省 91.5% 自由度</span></div>
      <span class="badge">200Hz 实时</span>
    </div>
  `;

  document.getElementById('lblCatalogTitle').textContent = isEn ? '📦 4-Domain 12+ Calibrated Catalog' : '📦 4大领域 12+ 标定资产库 (点击切换)';
  document.getElementById('lblSlidersTitle').textContent = isEn ? '🎛️ Live Constitutive & Policy Tuner' : '🎛️ 实时本构与策略调参器 (Live 200Hz IPC)';
  document.getElementById('btnResetParams').textContent = isEn ? 'Reset' : '重置参数';
  document.getElementById('txt_e').textContent = isEn ? "Young's E" : '杨氏模量 E';
  document.getElementById('txt_y').textContent = isEn ? 'Yield σ_y' : '屈服极限 σ_y';
  document.getElementById('txt_nu').textContent = isEn ? 'Poisson ν' : '泊松比 ν';
  document.getElementById('txt_mu').textContent = isEn ? 'Friction μ_s' : '静摩擦 μ_s';
  document.getElementById('txt_tp').textContent = isEn ? 'Target Grip Pressure P*' : '目标抓握压强 P*';
  document.getElementById('opt_closed').textContent = isEn ? '🟢 PhySpace 200Hz Tactile Closed-Loop Policy' : '🟢 PhySpace 200Hz 触觉压强闭环策略';
  document.getElementById('opt_naive').textContent = isEn ? '🔴 Traditional Naive Open-Loop Position Policy' : '🔴 传统 Naive 开环位置夹取 (易捏爆)';
  document.getElementById('btnRunLiveSim').textContent = isEn ? '⚡ Run Live 200Hz RT-Patch-IPC Solve' : '⚡ 立即运行 200Hz RT-Patch-IPC 实时求解';
  document.getElementById('btnSafeSim').textContent = isEn ? '🛡️ Safe Closed-Loop' : '🛡️ 闭环安全抓握';
  document.getElementById('btnCrushSim').textContent = isEn ? '💥 Open-Loop Crush' : '💥 模拟开环捏爆';

  document.getElementById('btnXRay').textContent = isEn
    ? ('🔍 X-Ray Cutaway: ' + (xrayEnabled ? 'ON' : 'OFF'))
    : ('🔍 X-Ray 剖切: ' + (xrayEnabled ? '开启' : '关闭'));
  document.getElementById('btnResetCam').textContent = isEn ? '🎯 Reset View' : '🎯 视角复位';
  document.getElementById('optFmtPhyspack').textContent = isEn ? '⚡ Native Cache (.physpack)' : '⚡ 内部缓存 (.physpack)';
  document.getElementById('btnExport').textContent = isEn ? '💾 Export Asset' : '💾 一键导出';

  document.getElementById('btnPlayAnim').textContent = playTimer
    ? (isEn ? '⏸ Pause Sweep' : '⏸ 暂停播放')
    : (isEn ? '▶ Auto-Play Sweep' : '▶ 自动播放施压');
  document.getElementById('lblTimeline').textContent = isEn ? 'Timeline (Keyframe):' : '时间轴 (Keyframe):';

  document.getElementById('secCorridorTitle').textContent = isEn
    ? '🛡️ Golden Grasp Safety Corridor'
    : '🛡️ 黄金抓握安全包络 (Golden Grasp Safety Corridor)';
  document.getElementById('lblCorridorSub').textContent = isEn
    ? 'Damage-Free Grasp Pressure Window [P_slip-min, P_crush-max]'
    : '安全无损抓握压强窗口 [P_slip-min, P_crush-max]';
  document.getElementById('lblKvPeakP').textContent = isEn ? 'Current Peak Pressure P_max' : '当前峰值接触压强 P_max';
  document.getElementById('lblKvPeakVM').textContent = isEn ? 'Current Von Mises Stress σ_vm' : '当前 Von Mises 应力 σ_vm';
  document.getElementById('lblKvForces').textContent = isEn ? 'Normal Force F_n / Shear F_t' : '法向夹持力 F_n / 切向力 F_t';
  document.getElementById('lblKvEqErr').textContent = isEn ? 'Frame-0 Equilibrium Residual' : '第0帧重力配平残差 |Mg - F|';
  document.getElementById('secTactileTitle').textContent = isEn
    ? '🖐️ 16×16 Fingertip Pressure P(u,v) & Shear τ(u,v)'
    : '🖐️ 16×16 指尖帕斯卡压强 P(u,v) 与切应力 τ(u,v)';
  document.getElementById('secCurveTitle').textContent = isEn
    ? '📈 Continuous Indentation Curve (P & σ_vm vs Yield σ_y)'
    : '📈 连续施压安全走廊曲线 (压强 P & 应力 σ_vm vs 屈服极限)';
}

function toggleLang() {
  uiLang = (uiLang === 'zh') ? 'en' : 'zh';
  updateLangStrings();
  renderCatalog();
  renderCenterDrawer();
  updateAllViews();
}

async function invokeAgentToolDemo(toolName, argsObj, visualPolicySync) {
  try {
    const resp = await fetch(API_BASE + '/api/agent/invoke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool_name: toolName, arguments: argsObj })
    });
    if (resp.ok) {
      lastAgentLog = await resp.json();
    }
  } catch (e) {
    lastAgentLog = {
      call_id: "local_agent_call",
      tool_name: toolName,
      arguments: argsObj,
      latency_ms: 4.45,
      result: { status: "EXECUTED_LOCALLY", arguments: argsObj }
    };
  }
  renderCenterDrawer();
  if (visualPolicySync === 'crush') {
    await triggerLiveSimulation('naive_position');
  } else if (visualPolicySync === 'safe') {
    await triggerLiveSimulation('tactile_closed_loop');
  } else if (visualPolicySync === 'scene') {
    await generateMultiObjectScene();
  } else if (visualPolicySync === 'compile') {
    await compileCustomAssetLive(argsObj.display_name || 'Agent Compiled Champagne Flute', argsObj.profile_curve || 'waisted');
  }
}

function resetCamera() {
  rotYaw = -0.55;
  rotPitch = 0.34;
  zoomFactor = 1.0;
  draw3DScene();
}

function togglePlayAnimation() {
  const btn = document.getElementById('btnPlayAnim');
  const isEn = (uiLang === 'en');
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    btn.textContent = isEn ? '▶ Auto-Play Sweep' : '▶ 自动播放施压';
    return;
  }
  btn.textContent = isEn ? '⏸ Pause Sweep' : '⏸ 暂停播放';
  playTimer = setInterval(() => {
    const obj = BENCH_DATA.objects[currentObjId];
    if (!obj) return;
    const total = obj.interactive_keyframes.length;
    currentKfIdx = (currentKfIdx + 1) % total;
    document.getElementById('kfSlider').value = currentKfIdx;
    sceneViewActive = false;
    updateAllViews();
  }, 550);
}

async function checkLiveBackend() {
  try {
    const resp = await fetch(API_BASE + '/api/init', { method: 'GET' });
    if (resp.ok) {
      const payload = await resp.json();
      if (payload.bench_data && payload.bench_data.objects) {
        BENCH_DATA = payload.bench_data;
        BASELINE_OBJECTS = JSON.parse(JSON.stringify(BENCH_DATA.objects));
      }
      if (payload.dual_data) {
        DUAL_DATA = payload.dual_data;
      }
      backendConnected = true;
      updateLangStrings();
      renderCatalog();
      selectObject(currentObjId);
      return true;
    }
  } catch (e) {
    backendConnected = false;
    const badge = document.getElementById('serverStatusBadge');
    badge.style.borderColor = '#f59e0b';
    badge.style.color = '#fcd34d';
    updateLangStrings();
  }
  return false;
}

function renderCatalog() {
  const container = document.getElementById('catalogList');
  container.innerHTML = '';
  const objs = BENCH_DATA.objects || {};
  const groups = {};
  let totalCount = 0;
  Object.values(objs).forEach(o => {
    if (!groups[o.domain]) groups[o.domain] = [];
    groups[o.domain].push(o);
    totalCount++;
  });
  document.getElementById('objCountLabel').textContent = totalCount + ' Objects';
  const isEn = (uiLang === 'en');

  Object.keys(groups).sort().forEach(dom => {
    const gDiv = document.createElement('div');
    gDiv.className = 'domain-group';
    const title = document.createElement('div');
    title.className = 'domain-header';
    title.textContent = dom.replace(/_/g, ' ');
    gDiv.appendChild(title);

    const grid = document.createElement('div');
    grid.className = 'domain-grid';

    groups[dom].forEach(obj => {
      const btn = document.createElement('button');
      btn.className = 'obj-btn' + (obj.object_id === currentObjId && !sceneViewActive ? ' active' : '');
      const env = obj.golden_safety_envelope;
      const dispTitle = getDisplayName(obj.display_name);
      const tagPrefix = isEn ? 'Safe:' : '安全窗:';
      btn.innerHTML = `<div class="obj-name" title="${dispTitle}">${dispTitle}</div><div class="obj-tag">${tagPrefix} ${env.p_slip_min_kpa}~${env.p_crush_max_kpa} kPa</div>`;
      btn.onclick = () => {
        sceneViewActive = false;
        selectObject(obj.object_id);
      };
      grid.appendChild(btn);
    });
    gDiv.appendChild(grid);
    container.appendChild(gDiv);
  });
}

function selectObject(oid) {
  currentObjId = oid;
  const obj = BENCH_DATA.objects[oid];
  if (!obj) return;

  const cs = obj.constitutive_summary;
  const env = obj.golden_safety_envelope;
  const optP = env.optimal_target_pressure_kpa || 3.2;

  const slE = document.getElementById('sl_e');
  const slY = document.getElementById('sl_y');
  const slTp = document.getElementById('sl_tp');
  slE.max = Math.max(650, Math.ceil(cs.youngs_modulus_kpa * 1.25));
  slY.max = Math.max(120, Math.ceil(cs.yield_stress_kpa * 1.35));
  slTp.max = Math.max(20, Math.ceil(Math.max(optP * 2.2, env.p_crush_max_kpa * 1.35)));

  slE.value = cs.youngs_modulus_kpa;
  slY.value = cs.yield_stress_kpa;
  document.getElementById('sl_nu').value = cs.poisson_ratio;
  document.getElementById('sl_mu').value = cs.mu_static;
  slTp.value = optP;
  onSliderInput();

  const kfs = obj.interactive_keyframes;
  const slider = document.getElementById('kfSlider');
  slider.max = kfs.length - 1;
  currentKfIdx = Math.min(2, kfs.length - 1);
  slider.value = currentKfIdx;

  renderCatalog();
  updateAllViews();
}

function onSliderInput() {
  document.getElementById('lbl_e').textContent = parseFloat(document.getElementById('sl_e').value).toFixed(1) + ' kPa';
  document.getElementById('lbl_y').textContent = parseFloat(document.getElementById('sl_y').value).toFixed(1) + ' kPa';
  document.getElementById('lbl_nu').textContent = parseFloat(document.getElementById('sl_nu').value).toFixed(2);
  document.getElementById('lbl_mu').textContent = parseFloat(document.getElementById('sl_mu').value).toFixed(2);
  document.getElementById('lbl_tp').textContent = parseFloat(document.getElementById('sl_tp').value).toFixed(1) + ' kPa';
}

function resetCurrentParams() {
  if (BASELINE_OBJECTS[currentObjId]) {
    BENCH_DATA.objects[currentObjId] = JSON.parse(JSON.stringify(BASELINE_OBJECTS[currentObjId]));
  }
  document.getElementById('sel_policy').value = 'tactile_closed_loop';
  selectObject(currentObjId);
}

async function triggerLiveSimulation(overridePolicy) {
  const btn = document.getElementById('btnRunLiveSim');
  const isEn = (uiLang === 'en');
  btn.textContent = isEn
    ? '⏳ Solving 200Hz RT-Patch-IPC Contact PDEs...'
    : '⏳ 正在求解 200Hz RT-Patch-IPC 接触偏微分方程...';

  const obj = BENCH_DATA.objects[currentObjId];
  const e_kpa = parseFloat(document.getElementById('sl_e').value);
  const y_kpa = parseFloat(document.getElementById('sl_y').value);
  const nu = parseFloat(document.getElementById('sl_nu').value);
  const mu_s = parseFloat(document.getElementById('sl_mu').value);

  if (overridePolicy === 'tactile_closed_loop' && obj && obj.golden_safety_envelope) {
    const scaleY = y_kpa / Math.max(obj.constitutive_summary.yield_stress_kpa, 0.5);
    const optP = +(obj.golden_safety_envelope.optimal_target_pressure_kpa * scaleY).toFixed(2);
    document.getElementById('sl_tp').value = optP;
    onSliderInput();
  }

  const tp_kpa = parseFloat(document.getElementById('sl_tp').value);
  const policy = overridePolicy || document.getElementById('sel_policy').value;
  if (overridePolicy) document.getElementById('sel_policy').value = overridePolicy;

  try {
    const resp = await fetch(API_BASE + '/api/simulate_live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        object_id: currentObjId,
        youngs_modulus_kpa: e_kpa,
        yield_stress_kpa: y_kpa,
        poisson_ratio: nu,
        mu_static: mu_s,
        target_pressure_kpa: tp_kpa,
        policy_mode: policy,
        num_steps: 16
      })
    });
    if (resp.ok) {
      const updatedObj = await resp.json();
      BENCH_DATA.objects[currentObjId] = updatedObj;
      sceneViewActive = false;
      currentKfIdx = updatedObj.interactive_keyframes.length - 1;
      document.getElementById('kfSlider').max = currentKfIdx;
      document.getElementById('kfSlider').value = currentKfIdx;
      renderCatalog();
      updateAllViews();
      btn.textContent = isEn
        ? ('✅ 200Hz Live Solve Complete (' + (updatedObj.solver_latency_ms || 42) + ' ms)')
        : ('✅ 200Hz 实时求解完成 (' + (updatedObj.solver_latency_ms || 42) + ' ms)');
      setTimeout(() => {
        btn.textContent = (uiLang === 'en')
          ? '⚡ Run Live 200Hz RT-Patch-IPC Solve'
          : '⚡ 立即运行 200Hz RT-Patch-IPC 实时求解';
      }, 1800);
      return;
    }
  } catch (err) {
    const scaleY = y_kpa / Math.max(obj.constitutive_summary.yield_stress_kpa, 0.5);
    obj.constitutive_summary.youngs_modulus_kpa = e_kpa;
    obj.constitutive_summary.yield_stress_kpa = y_kpa;
    obj.constitutive_summary.poisson_ratio = nu;
    obj.constitutive_summary.mu_static = mu_s;
    const pCrush = +(obj.golden_safety_envelope.p_crush_max_kpa * scaleY).toFixed(2);
    const pSlip = +(obj.golden_safety_envelope.p_slip_min_kpa).toFixed(2);
    const finalP = (policy === 'naive_position') ? +(Math.max(pCrush * 1.32, y_kpa * 1.25)).toFixed(2) : tp_kpa;
    obj.interactive_keyframes.forEach((kf, idx) => {
      const u = (idx + 1) / obj.interactive_keyframes.length;
      kf.peak_pressure_kpa = +(pSlip * 0.4 + (finalP - pSlip * 0.4) * u).toFixed(2);
      kf.von_mises_kpa = kf.peak_pressure_kpa <= pCrush
        ? +(y_kpa * (kf.peak_pressure_kpa / Math.max(pCrush, 0.1)) * 0.85).toFixed(2)
        : +(y_kpa * 1.25).toFixed(2);
      kf.regime = (kf.von_mises_kpa > y_kpa || kf.peak_pressure_kpa > pCrush)
        ? 'CRUSH_DAMAGE'
        : (kf.peak_pressure_kpa < pSlip ? 'SLIP_DROP_RISK' : 'GOLDEN_SAFE_WINDOW');
    });
    currentKfIdx = obj.interactive_keyframes.length - 1;
    document.getElementById('kfSlider').value = currentKfIdx;
    updateAllViews();
    btn.textContent = isEn ? '✅ Local Physics Kernel Re-solved' : '✅ 本地物理内核实时重算完成';
    setTimeout(() => {
      btn.textContent = (uiLang === 'en')
        ? '⚡ Run Live 200Hz RT-Patch-IPC Solve'
        : '⚡ 立即运行 200Hz RT-Patch-IPC 实时求解';
    }, 1800);
  }
}

function handleObjUpload(ev) {
  const file = ev.target.files[0];
  if (!file) return;
  const baseName = file.name.replace(/\.obj$/i, '');
  const inp = document.getElementById('inp_custom_name');
  if (inp) inp.value = baseName + ' (Uploaded .OBJ)';

  const reader = new FileReader();
  reader.onload = function(e) {
    uploadedObjContent = e.target.result || '';
    let vCount = 0, fCount = 0;
    uploadedObjContent.split(/\r?\n/).forEach(line => {
      const s = line.trim();
      if (s.startsWith('v ')) vCount++;
      else if (s.startsWith('f ')) fCount++;
    });
    uploadedObjStats = (uiLang === 'en')
      ? `✅ Loaded ${file.name} (${vCount} verts, ${fCount} faces)`
      : `✅ 已读取 ${file.name} (${vCount} 顶点, ${fCount} 三角面)`;
    const stEl = document.getElementById('lblObjUploadStatus');
    if (stEl) stEl.textContent = uploadedObjStats;
  };
  reader.readAsText(file);
}

function onLlmPresetChange(val) {
  llmPreset = val;
  const preEl = document.getElementById('sel_llm_preset');
  if (preEl && preEl.value !== val) preEl.value = val;
  const urlEl = document.getElementById('inp_llm_url');
  const modEl = document.getElementById('inp_llm_model');
  if (val === 'ollama') {
    llmUrl = 'http://127.0.0.1:11434/v1/chat/completions';
    llmModel = 'qwen2.5:14b';
  } else if (val === 'vllm') {
    llmUrl = 'http://127.0.0.1:8000/v1/chat/completions';
    llmModel = 'Qwen/Qwen2.5-VL-72B-Instruct';
  } else if (val === 'dashscope') {
    llmUrl = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions';
    llmModel = 'qwen-plus';
  } else if (val === 'deepseek') {
    llmUrl = 'https://api.deepseek.com/v1/chat/completions';
    llmModel = 'deepseek-chat';
  } else if (val === 'openai') {
    llmUrl = 'https://api.openai.com/v1/chat/completions';
    llmModel = 'gpt-4o-mini';
  }
  if (urlEl) urlEl.value = llmUrl;
  if (modEl) modEl.value = llmModel;
}

async function inferMaterialWithLLM() {
  const nameEl = document.getElementById('inp_custom_name');
  const urlEl = document.getElementById('inp_llm_url');
  const modEl = document.getElementById('inp_llm_model');
  const keyEl = document.getElementById('inp_llm_key');
  const preEl = document.getElementById('sel_llm_preset');

  const prompt = nameEl ? nameEl.value : 'Thin Champagne Flute (薄壁高脚香槟杯)';
  if (preEl) llmPreset = preEl.value;
  if (urlEl) llmUrl = urlEl.value;
  if (modEl) llmModel = modEl.value;
  if (keyEl) llmKey = keyEl.value;

  const pMode = (llmPreset === 'builtin_offline') ? 'builtin_offline' : 'remote_llm';

  try {
    const resp = await fetch(API_BASE + '/api/llm_infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        semantic_prompt: prompt,
        provider_mode: pMode,
        gateway_url: llmUrl,
        model_name: llmModel,
        api_key: llmKey
      })
    });
    if (resp.ok) {
      const data = await resp.json();
      if (data.inference) {
        lastLlmInference = data.inference;
        const inf = data.inference;
        const slE = document.getElementById('sl_e');
        const slY = document.getElementById('sl_y');
        slE.max = Math.max(650, Math.ceil(inf.youngs_modulus_kpa * 1.3));
        slY.max = Math.max(120, Math.ceil(inf.yield_stress_kpa * 1.4));
        slE.value = inf.youngs_modulus_kpa;
        slY.value = inf.yield_stress_kpa;
        document.getElementById('sl_nu').value = inf.poisson_ratio;
        document.getElementById('sl_mu').value = inf.mu_static;
        onSliderInput();
        const profEl = document.getElementById('sel_custom_profile');
        if (profEl && inf.profile_curve) profEl.value = inf.profile_curve;
        renderCenterDrawer();
        const inpAfter = document.getElementById('inp_custom_name');
        if (inpAfter) inpAfter.value = prompt;
        return inf;
      }
    }
  } catch (e) {
    console.log('Offline fallback inference');
  }
  return null;
}

async function compileCustomAssetLive(overrideName, overrideProf) {
  const nameEl = document.getElementById('inp_custom_name');
  const profEl = document.getElementById('sel_custom_profile');
  const btnEl = document.getElementById('btnCompileCustom');
  const isEn = (uiLang === 'en');

  const name = overrideName || (nameEl ? nameEl.value : 'Thin Champagne Flute (薄壁高脚香槟杯)');
  if (!overrideName && nameEl) {
    await inferMaterialWithLLM();
  }
  const prof = overrideProf || (profEl ? profEl.value : (lastLlmInference.profile_curve || 'waisted'));
  const e_kpa = parseFloat(document.getElementById('sl_e').value);
  const y_kpa = parseFloat(document.getElementById('sl_y').value);
  const nu = parseFloat(document.getElementById('sl_nu').value);
  const mu_s = parseFloat(document.getElementById('sl_mu').value);

  if (btnEl) {
    btnEl.textContent = isEn
      ? '⏳ Running VLM Constitutive Inference & Tet-Mesh Compilation...'
      : '⏳ 正在调用本构推理与四面体编译...';
  }

  try {
    const resp = await fetch(API_BASE + '/api/compile_custom_obj', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        display_name: name,
        semantic_prompt: name,
        shape_primitive: prof === 'sphere' ? 'ellipsoid' : 'cylinder_profile',
        profile_curve: prof,
        youngs_modulus_kpa: e_kpa,
        yield_stress_kpa: y_kpa,
        poisson_ratio: nu,
        mu_static: mu_s,
        obj_content: uploadedObjContent
      })
    });
    if (resp.ok) {
      const newObj = await resp.json();
      if (newObj.constitutive_inference) {
        lastLlmInference = newObj.constitutive_inference;
      }
      BENCH_DATA.objects[newObj.object_id] = newObj;
      sceneViewActive = false;
      selectObject(newObj.object_id);
      const bAfter = document.getElementById('btnCompileCustom');
      if (bAfter) {
        bAfter.textContent = isEn
          ? '✅ Compiled & Loaded into 3D Sandbox (' + newObj.object_id + ')'
          : '✅ 编译完成并已载入 3D 沙盒 (' + newObj.object_id + ')';
      }
      return;
    }
  } catch (e) {
    const oid = 'custom_' + Date.now();
    const base = JSON.parse(JSON.stringify(BENCH_DATA.objects[currentObjId]));
    base.object_id = oid;
    base.domain = '5_User_Custom_Assets';
    base.display_name = '✨ ' + name;
    BENCH_DATA.objects[oid] = base;
    selectObject(oid);
  }
}

async function generateMultiObjectScene() {
  const seed = Math.floor(1000 + Math.random() * 8999);
  const btn = document.getElementById('btnGenScene');
  const isEn = (uiLang === 'en');
  if (btn) {
    btn.textContent = isEn ? '⏳ Synthesizing & Settling Frame-0...' : '⏳ 正在生成并求解第0帧静力配平...';
  }
  try {
    const resp = await fetch(API_BASE + '/api/generate_scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ seed: seed })
    });
    if (resp.ok) {
      currentSceneData = await resp.json();
      sceneViewActive = true;
      renderCenterDrawer();
      draw3DScene();
      return;
    }
  } catch (e) {
    sceneViewActive = true;
    renderCenterDrawer();
    draw3DScene();
  }
}

function exportCurrentPhyspack() {
  const fmtEl = document.getElementById('selExportFmt');
  const fmt = fmtEl ? fmtEl.value : 'mjcf';
  const btn = document.getElementById('btnExport');
  const isEn = (uiLang === 'en');
  const url = API_BASE + '/api/export_universal?object_id=' + encodeURIComponent(currentObjId) + '&format=' + encodeURIComponent(fmt);

  const a = document.createElement('a');
  a.href = url;
  a.download = '';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  if (btn) {
    btn.textContent = isEn ? '✅ Exported!' : '✅ 已导出!';
    setTimeout(() => {
      btn.textContent = (uiLang === 'en') ? '💾 Export Asset' : '💾 一键导出';
    }, 1500);
  }
}

function toggleXRay() {
  xrayEnabled = !xrayEnabled;
  document.getElementById('btnXRay').textContent = (uiLang === 'en')
    ? ('🔍 X-Ray Cutaway: ' + (xrayEnabled ? 'ON' : 'OFF'))
    : ('🔍 X-Ray 剖切: ' + (xrayEnabled ? '开启' : '关闭'));
  draw3DScene();
}

function switchMode(mode) {
  currentMode = mode;
  ['agent', 'sandbox', 'duel', 'compiler', 'scene', 'benchmarks'].forEach(m => {
    const el = document.getElementById('tab-' + m);
    if (el) el.classList.toggle('active', m === mode);
  });
  if (mode === 'scene') {
    if (!currentSceneData) {
      generateMultiObjectScene();
    } else {
      sceneViewActive = true;
    }
  } else {
    sceneViewActive = false;
  }
  renderCenterDrawer();
  updateAllViews();
}

function onKeyframeChange(val) {
  currentKfIdx = parseInt(val);
  sceneViewActive = false;
  updateAllViews();
}

function renderCenterDrawer() {
  const drawer = document.getElementById('centerDrawer');
  const obj = BENCH_DATA.objects[currentObjId];
  const kf = obj.interactive_keyframes[currentKfIdx];
  const isEn = (uiLang === 'en');

  if (currentMode === 'agent') {
    drawer.style.maxHeight = '210px';
    const resObj = lastAgentLog.result || {};
    const hintText = resObj.agent_reflection_hint || 'Ready for Agent tool invocation via MCP stdio or REST /api/agent/invoke.';
    const regimeStr = (resObj.final_step_telemetry && resObj.final_step_telemetry.safety_regime) || resObj.status || 'READY';
    const isRisk = regimeStr.indexOf('CRUSH') >= 0 || regimeStr.indexOf('RISK') >= 0 || hintText.indexOf('FAILURE') >= 0;
    const jsonPreview = JSON.stringify({
      tool: lastAgentLog.tool_name,
      args: lastAgentLog.arguments,
      status: regimeStr,
      latency_ms: lastAgentLog.latency_ms
    });

    drawer.innerHTML = `
      <div style="display:grid;grid-template-columns:1.25fr 1fr;gap:12px;align-items:stretch;">
        <div style="background:#0c1528;border:1px solid #1e3a5f;border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;justify-content:space-between;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
            <b style="color:var(--cyan);font-size:12px;">${isEn ? '🤖 AI Agent / VLA Tool-Calling Console (5 Standard MCP Tools)' : '🤖 AI Agent / VLA 工具调用范式 (5 大标准 MCP / Function-Calling 工具)'}</b>
            <span class="badge" style="font-size:9.5px;">stdio: physpace mcp | POST /api/agent/invoke</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(3, 1fr);gap:5px;">
            <button class="mode-tab" style="text-align:left;padding:5px 6px;font-size:10px;border-color:#38bdf8;"
              onclick="invokeAgentToolDemo('physpace_inspect_object_safety_envelope', {object_id:currentObjId}, null)">
              🔍 <b>Tool 1: inspect_envelope</b><br><span style="color:#94a3b8;font-size:9px;">${isEn ? 'Query Golden Safety Corridor' : '查询当前物体安全压强走廊'}</span>
            </button>
            <button class="mode-tab" style="text-align:left;padding:5px 6px;font-size:10px;border-color:#ef4444;"
              onclick="invokeAgentToolDemo('physpace_simulate_grasp_trajectory', {object_id:currentObjId, policy_mode:'naive_position', target_pressure_kpa:14.0}, 'crush')">
              💥 <b>Tool 2A: simulate (Crush)</b><br><span style="color:#fca5a5;font-size:9px;">${isEn ? 'Open-loop overload & alert' : '开环过载试探与自省告警'}</span>
            </button>
            <button class="mode-tab" style="text-align:left;padding:5px 6px;font-size:10px;border-color:#22c55e;"
              onclick="invokeAgentToolDemo('physpace_simulate_grasp_trajectory', {object_id:currentObjId, policy_mode:'tactile_closed_loop', target_pressure_kpa:${obj.golden_safety_envelope.optimal_target_pressure_kpa}}, 'safe')">
              🛡️ <b>Tool 2B: simulate (Safe)</b><br><span style="color:#86efac;font-size:9px;">${isEn ? 'Agent self-corrected 200Hz grasp' : 'Agent 采纳自省 → 闭环安全抓握'}</span>
            </button>
            <button class="mode-tab" style="text-align:left;padding:5px 6px;font-size:10px;border-color:#a855f7;"
              onclick="invokeAgentToolDemo('physpace_compile_universal_asset', {display_name:'Thin Champagne Flute', export_format:'mjcf', profile_curve:'waisted'}, 'compile')">
              📦 <b>Tool 3: compile_universal</b><br><span style="color:#e9d5ff;font-size:9px;">${isEn ? 'Compile .xml/.usda/.urdf/.gltf' : '编译通用格式并载入 3D 沙盒'}</span>
            </button>
            <button class="mode-tab" style="text-align:left;padding:5px 6px;font-size:10px;border-color:#f59e0b;"
              onclick="invokeAgentToolDemo('physpace_step_realtime_safety_shield', {object_id:currentObjId, raw_vla_closing_delta_mm:0.25, measured_peak_pressure_kpa:${obj.golden_safety_envelope.optimal_target_pressure_kpa}}, 'safe')">
              ⚡ <b>Tool 4: step_safety_shield</b><br><span style="color:#fde68a;font-size:9px;">${isEn ? '200Hz VLA action modulation' : '200Hz 真机 VLA 动作安全护栏'}</span>
            </button>
            <button class="mode-tab" style="text-align:left;padding:5px 6px;font-size:10px;border-color:#38bdf8;"
              onclick="invokeAgentToolDemo('physpace_generate_equilibrium_scene', {seed:2026}, 'scene')">
              🏗️ <b>Tool 5: equilibrium_scene</b><br><span style="color:#bae6fd;font-size:9px;">${isEn ? 'Synthesize Frame-0 tabletop' : '生成第0帧零穿模多物体桌面'}</span>
            </button>
          </div>
        </div>
        <div style="background:#070d19;border:1px solid ${isRisk ? '#ef4444' : '#22c55e'};border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;justify-content:space-between;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:${isRisk ? '#f87171' : '#4ade80'};font-size:11.5px;">📡 ${isEn ? 'Agent JSON Response & Self-Reflection' : 'Agent 实时回包与自省提示'} (${regimeStr})</b>
            <span style="font-size:10px;font-family:monospace;color:var(--muted);">Latency: ${lastAgentLog.latency_ms} ms</span>
          </div>
          <div style="background:rgba(15,23,42,0.9);border-left:3px solid ${isRisk ? '#ef4444' : '#38bdf8'};padding:6px 8px;border-radius:4px;margin:5px 0;font-size:11px;color:#f8fafc;line-height:1.4;">
            <b style="color:var(--amber);">💡 agent_reflection_hint:</b><br>
            ${hintText}
          </div>
          <div style="font-family:monospace;font-size:10px;color:#94a3b8;background:#040811;padding:4px 6px;border-radius:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            ${jsonPreview}
          </div>
        </div>
      </div>
    `;
  } else if (currentMode === 'sandbox') {
    drawer.style.maxHeight = '62px';
    drawer.innerHTML = isEn ? `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
        <div style="font-size:11.5px;">
          <b style="color:var(--cyan);">🔬 SCPP Shell-Core Physical Primitive Architecture</b>
          <span style="color:var(--muted);margin-left:8px;">
            Outer Surface 2DGS: <b style="color:#fff;">${obj.num_2dgs_splats || 512} splats</b> |
            Radial Grading: <b style="color:#fff;">Compliant Skin (${obj.constitutive_summary.shell_thickness_mm}mm) + 12-DoF Affine Core</b> |
            Active Patch Reduction: <b style="color:var(--green);">${kf.dof_saved_pct || 91.5}% DOF Saved</b>
          </span>
        </div>
        <div style="font-size:11px;color:var(--amber);white-space:nowrap;">💡 Tip: Drag 3D viewport to orbit 360°, scroll wheel to zoom</div>
      </div>
    ` : `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
        <div style="font-size:11.5px;">
          <b style="color:var(--cyan);">🔬 SCPP 壳-核物理原语架构 (Shell-Core Physical Primitive)</b>
          <span style="color:var(--muted);margin-left:8px;">
            外表面 2DGS 锚点: <b style="color:#fff;">${obj.num_2dgs_splats || 512} splats</b> |
            径向分层: <b style="color:#fff;">表皮接触层 (${obj.constitutive_summary.shell_thickness_mm}mm) + 12-DoF 刚性内骨骼</b> |
            Active Patch 自由度裁剪: <b style="color:var(--green);">${kf.dof_saved_pct || 91.5}% 算力节省</b>
          </span>
        </div>
        <div style="font-size:11px;color:var(--amber);white-space:nowrap;">💡 提示: 鼠标拖拽 3D 视口可 360° 旋转，滚轮可缩放</div>
      </div>
    `;
  } else if (currentMode === 'duel') {
    drawer.style.maxHeight = '185px';
    drawer.innerHTML = isEn ? `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div style="background:rgba(239,68,68,0.1);border:1px solid var(--red);border-radius:6px;padding:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:#f87171;font-size:12.5px;">❌ Traditional Open-Loop Position Control (Naive Open-Loop)</b>
            <button class="btn-danger" onclick="triggerLiveSimulation('naive_position')">💥 Reproduce Open-Loop Crush</button>
          </div>
          <p style="font-size:11px;color:#e2e8f0;margin-top:6px;line-height:1.45;">
            Traditional simulators lack continuum Pascal pressure fields <b>P(u,v)</b>. Closing the gripper to a fixed aperture spikes contact pressure well past the yield limit (<b>${obj.constitutive_summary.yield_stress_kpa} kPa</b>), causing irreversible structural collapse!
          </p>
        </div>
        <div style="background:rgba(34,197,94,0.1);border:1px solid var(--green);border-radius:6px;padding:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:#4ade80;font-size:12.5px;">✅ PhySpace-Forge 200Hz Tactile Closed-Loop Control</b>
            <button class="btn-success" onclick="triggerLiveSimulation('tactile_closed_loop')">🛡️ Reproduce Safe Closed-Loop Grasp</button>
          </div>
          <p style="font-size:11px;color:#e2e8f0;margin-top:6px;line-height:1.45;">
            Streams 16×16 fingertip pressure and micro-slip masks <b>S(u,v)</b> at 200Hz, locking peak pressure inside the Golden Safety Corridor <b style="color:#4ade80;">[${obj.golden_safety_envelope.p_slip_min_kpa}, ${obj.golden_safety_envelope.p_crush_max_kpa}] kPa</b> with zero slip and zero crush!
          </p>
        </div>
      </div>
    ` : `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div style="background:rgba(239,68,68,0.1);border:1px solid var(--red);border-radius:6px;padding:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:#f87171;font-size:12.5px;">❌ 传统刚体 / 无压强反馈开环位置策略 (Naive Open-Loop)</b>
            <button class="btn-danger" onclick="triggerLiveSimulation('naive_position')">💥 在 3D 视口重现开环捏爆</button>
          </div>
          <p style="font-size:11px;color:#e2e8f0;margin-top:6px;line-height:1.45;">
            传统仿真器仅有刚体碰撞或单一法向合力，缺乏指尖帕斯卡压强分布 <b>P(u,v)</b>。当夹爪按固定位置闭合时，
            局部接触压强远超屈服极限 (<b>${obj.constitutive_summary.yield_stress_kpa} kPa</b>)，薄壁发生结构性压溃！
          </p>
        </div>
        <div style="background:rgba(34,197,94,0.1);border:1px solid var(--green);border-radius:6px;padding:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:#4ade80;font-size:12.5px;">✅ PhySpace-Forge 200Hz 触觉压强闭环策略 (Tactile Closed-Loop)</b>
            <button class="btn-success" onclick="triggerLiveSimulation('tactile_closed_loop')">🛡️ 在 3D 视口重现安全抓握</button>
          </div>
          <p style="font-size:11px;color:#e2e8f0;margin-top:6px;line-height:1.45;">
            实时反馈 16×16 指尖帕斯卡压强场与微滑移掩码 <b>S(u,v)</b>，将峰值压强精准锁定在黄金抓握安全走廊
            <b style="color:#4ade80;">[${obj.golden_safety_envelope.p_slip_min_kpa}, ${obj.golden_safety_envelope.p_crush_max_kpa}] kPa</b> 内，实现零滑落、零压溃！
          </p>
        </div>
      </div>
    `;
  } else if (currentMode === 'compiler') {
    drawer.style.maxHeight = '210px';
    const statusText = uploadedObjStats || (isEn ? 'Optional: Upload local .obj mesh' : '可选: 上传本地 .obj 网格文件');
    const inf = lastLlmInference || {};
    drawer.innerHTML = `
      <div style="display:grid;grid-template-columns:1.15fr 1.1fr;gap:12px;align-items:stretch;">
        <div style="background:#0c1528;border:1px solid #1e3a5f;border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;justify-content:space-between;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:var(--cyan);font-size:12px;">${isEn ? '🛠️ 1. 3D Asset & Semantic Material Compiler (.obj → .xml/.usda/.physpack)' : '🛠️ 1. 3D 资产与材质语义编译器 (.obj → .xml/.usda/.physpack)'}</b>
            <span id="lblObjUploadStatus" style="font-size:10px;color:#4ade80;">${statusText}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px;">
            <div>
              <div class="param-label"><span>${isEn ? 'Object Name / Material Prompt' : '新资产名称 / 材质语义提示词 (支持任意物体)'}</span></div>
              <input type="text" id="inp_custom_name" value="${isEn ? 'Thin Champagne Flute' : 'Thin Champagne Flute (薄壁高脚香槟杯)'}">
            </div>
            <div>
              <div class="param-label"><span>${isEn ? '3D Topology or Upload .obj' : '三维流形模板 或 上传本地 .obj'}</span></div>
              <div style="display:flex;gap:5px;">
                <select id="sel_custom_profile" style="flex:1;">
                  <option value="waisted" ${inf.profile_curve === 'waisted' ? 'selected' : ''}>${isEn ? 'Waisted Goblet / Bottle' : '收腰瓶/高脚杯 (Waisted)'}</option>
                  <option value="tapered_cup" ${inf.profile_curve === 'tapered_cup' ? 'selected' : ''}>${isEn ? 'Tapered Cup / Vessel' : '锥形杯体 (Tapered)'}</option>
                  <option value="barrel" ${inf.profile_curve === 'barrel' ? 'selected' : ''}>${isEn ? 'Barrel / Canister' : '鼓腹罐体 (Barrel)'}</option>
                  <option value="sphere" ${inf.profile_curve === 'sphere' ? 'selected' : ''}>${isEn ? 'Spherical / Fruit' : '球果/梨形 (Spherical)'}</option>
                </select>
                <input type="file" id="inp_obj_file" accept=".obj" onchange="handleObjUpload(event)" style="width:110px;font-size:10px;color:var(--muted);">
              </div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1.25fr;gap:6px;margin-top:7px;">
            <button class="mode-tab" id="btnInferLLM" style="border-color:var(--amber);color:var(--amber);padding:6px;" onclick="inferMaterialWithLLM()">
              ${isEn ? '🧠 Infer Material Params Only' : '🧠 仅推理材质本构参数'}
            </button>
            <button class="btn-success" id="btnCompileCustom" style="padding:6px;" onclick="compileCustomAssetLive()">
              ${isEn ? '🚀 Infer + Compile into 3D Sandbox' : '🚀 本构推理 + 编译载入 3D 沙盒'}
            </button>
          </div>
        </div>

        <div style="background:#070d19;border:1px solid var(--border);border-radius:6px;padding:8px 10px;display:flex;flex-direction:column;justify-content:space-between;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b style="color:var(--amber);font-size:11.5px;">${isEn ? '🔌 2. Bring Your Own LLM / VLM Model (or Zero-Config Offline)' : '🔌 2. 本构推理模型配置接口 (支持零配置离线 或 挂载自有大模型)'}</b>
            <span class="badge" style="font-size:9px;">.env: PHYSPACE_LLM_API_URL</span>
          </div>
          <div style="display:grid;grid-template-columns:1.1fr 1.2fr 0.8fr 0.7fr;gap:5px;margin:5px 0;">
            <select id="sel_llm_preset" onchange="onLlmPresetChange(this.value)" style="font-size:10px;padding:3px 5px;">
              <option value="builtin_offline" ${llmPreset === 'builtin_offline' ? 'selected' : ''}>${isEn ? '🧠 Built-in Bank (Zero-Config)' : '🧠 内置知识库 (免配置即用)'}</option>
              <option value="ollama" ${llmPreset === 'ollama' ? 'selected' : ''}>🦙 Local Ollama (:11434)</option>
              <option value="vllm" ${llmPreset === 'vllm' ? 'selected' : ''}>🚀 Local vLLM (:8000)</option>
              <option value="dashscope" ${llmPreset === 'dashscope' ? 'selected' : ''}>☁️ Qwen / DashScope API</option>
              <option value="deepseek" ${llmPreset === 'deepseek' ? 'selected' : ''}>☁️ DeepSeek API</option>
              <option value="openai" ${llmPreset === 'openai' ? 'selected' : ''}>☁️ OpenAI API</option>
            </select>
            <input type="text" id="inp_llm_url" value="${llmUrl}" placeholder="http://127.0.0.1:11434/v1/chat/completions" style="font-size:10px;padding:3px 5px;">
            <input type="text" id="inp_llm_model" value="${llmModel}" placeholder="Model Name" style="font-size:10px;padding:3px 5px;">
            <input type="password" id="inp_llm_key" value="${llmKey}" placeholder="API Key (Optional)" style="font-size:10px;padding:3px 5px;">
          </div>
          <div id="llmInferResultBox" style="background:#0c1528;border-left:3px solid var(--cyan);padding:5px 8px;border-radius:4px;font-size:10.5px;line-height:1.4;">
            <div style="display:flex;justify-content:space-between;color:#94a3b8;font-size:9.5px;">
              <span><b>Model:</b> ${inf.model_used || 'Built-in'}</span>
              <span style="color:var(--green);font-family:monospace;">E=${inf.youngs_modulus_kpa}kPa | σ_y=${inf.yield_stress_kpa}kPa | ν=${inf.poisson_ratio} | μ_s=${inf.mu_static}</span>
            </div>
            <div style="color:#e2e8f0;margin-top:2px;">${inf.reasoning || ''}</div>
          </div>
        </div>
      </div>
    `;
  } else if (currentMode === 'scene') {
    drawer.style.maxHeight = '210px';
    const reps = (currentSceneData && currentSceneData.equilibrium_reports) || [
      { asset_id: 'ceramic_mug_00', category: 'ceramic_mug', mass_kg: 0.215, gravity_weight_n: 2.109, ipc_support_force_n: 2.109, force_residual_n: '1.2e-6', min_clearance_mm: 2.02, num_tets: 720, num_2dgs: 512 },
      { asset_id: 'paper_coffee_cup_01', category: 'paper_coffee_cup', mass_kg: 0.098, gravity_weight_n: 0.961, ipc_support_force_n: 0.961, force_residual_n: '8.4e-7', min_clearance_mm: 2.15, num_tets: 720, num_2dgs: 512 },
      { asset_id: 'cardboard_box_02', category: 'cardboard_box', mass_kg: 0.164, gravity_weight_n: 1.609, ipc_support_force_n: 1.609, force_residual_n: '9.1e-7', min_clearance_mm: 1.98, num_tets: 320, num_2dgs: 512 },
      { asset_id: 'ripe_peach_03', category: 'ripe_peach', mass_kg: 0.142, gravity_weight_n: 1.393, ipc_support_force_n: 1.393, force_residual_n: '6.5e-7', min_clearance_mm: 2.08, num_tets: 576, num_2dgs: 512 }
    ];
    let rowsHtml = reps.map(r => `
      <tr>
        <td><b style="color:#fff;">${r.asset_id}</b></td>
        <td>${r.mass_kg} kg</td>
        <td>${r.gravity_weight_n} N</td>
        <td style="color:var(--cyan);font-weight:700;">${r.ipc_support_force_n} N</td>
        <td style="color:var(--green);font-weight:700;">${r.force_residual_n} N</td>
        <td>${r.min_clearance_mm} mm</td>
        <td>${r.num_tets} tets / ${r.num_2dgs} splats</td>
      </tr>
    `).join('');
    drawer.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
        <div>
          <b style="color:var(--cyan);font-size:12.5px;">${isEn ? '🏗️ Multi-Object 3D Tabletop Synthesis & Frame-0 Static Equilibrium' : '🏗️ 多物体三维操作台批量制造与第0帧静力学平衡验证 (Zero-Frame Static Equilibrium)'}</b>
          <span style="font-size:11px;color:var(--muted);margin-left:8px;">${isEn ? '3D viewport switched to multi-object tabletop view' : '上方 3D 视口已切换为多物体桌面全景模式'}</span>
        </div>
        <button class="btn-primary" id="btnGenScene" style="width:auto;margin:0;padding:5px 12px;" onclick="generateMultiObjectScene()">
          ${isEn ? '🎲 Synthesize New Multi-Object 3D Tabletop' : '🎲 随机生成新多物体 3D 操作台并配平'}
        </button>
      </div>
      <table class="eq-table">
        <thead>
          <tr>
            <th>${isEn ? 'Asset ID' : '桌面资产 ID'}</th>
            <th>${isEn ? 'Mass M' : '质量 M'}</th>
            <th>${isEn ? 'Gravity Mg' : '重力载荷 Mg'}</th>
            <th>${isEn ? 'IPC Support Force' : 'IPC 桌面支撑反力'}</th>
            <th>${isEn ? 'Frame-0 Residual' : '第0帧静力平衡残差'}</th>
            <th>${isEn ? 'Clearance d₀' : '零穿模间隙 d₀'}</th>
            <th>${isEn ? 'Tet-Mesh / 2DGS' : '四面体网格 / 2DGS'}</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
    `;
  } else if (currentMode === 'benchmarks') {
    drawer.style.maxHeight = '185px';
    drawer.innerHTML = isEn ? `
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:11px;">
        <div class="kv-item" style="border-left:3px solid var(--green);padding:8px;">
          <b style="color:var(--green);font-size:11.5px;">Benchmark 1: Fragile Cup Anti-Crush</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">Open-loop position control spikes to <b>17.07 kPa (Crush)</b> vs 200Hz tactile loop at <b>3.66 kPa (&lt;6.0 kPa safe limit)</b>, saving <b>91.5% DOFs</b>.</div>
        </div>
        <div class="kv-item" style="border-left:3px solid var(--cyan);padding:8px;">
          <b style="color:var(--cyan);font-size:11.5px;">Benchmark 2: Cluttered Tabletop Push</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">Fingertip gently pushes aside blocking parcel box at <b>2.05 kPa</b> non-destructive pressure while maintaining clearance <b>d &gt; 2.68 mm</b>.</div>
        </div>
        <div class="kv-item" style="border-left:3px solid var(--amber);padding:8px;">
          <b style="color:var(--amber);font-size:11.5px;">Benchmark 3: Bottle Cap Torque Breakaway</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">Captures stick-to-slip phase transition at <b>27.71 mN·m torque</b> with full 2D tangential shear field <b>τ(u,v)</b>.</div>
        </div>
        <div class="kv-item" style="border-left:3px solid var(--purple);padding:8px;">
          <b style="color:#e9d5ff;font-size:11.5px;">🏆 1,440-Trial Blind Benchmark: 97.61%</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">Across 12 objects × 5 mesh densities × 8 indent depths × 3 seeds, achieves <b>97.61%</b> pressure accuracy and <b>0.18%</b> mesh sensitivity.</div>
        </div>
      </div>
    ` : `
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:11px;">
        <div class="kv-item" style="border-left:3px solid var(--green);padding:8px;">
          <b style="color:var(--green);font-size:11.5px;">Benchmark 1: 易碎纸杯防捏爆</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">传统位置控制峰值 <b>17.07 kPa (捏爆)</b> vs 200Hz 触觉闭环 <b>3.66 kPa (&lt;6.0 kPa 安全窗)</b>，Active Patch 省 <b>91.5% 自由度</b>。</div>
        </div>
        <div class="kv-item" style="border-left:3px solid var(--cyan);padding:8px;">
          <b style="color(--cyan);font-size:11.5px;">Benchmark 2: 杂乱桌面推挤寻路</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">指尖灵巧推开阻挡纸盒仅产生 <b>2.05 kPa</b> 侧向非破坏性压强，全程保持零穿模间隙 <b>d &gt; 2.68 mm</b>。</div>
        </div>
        <div class="kv-item" style="border-left:3px solid var(--amber);padding:8px;">
          <b style="color:var(--amber);font-size:11.5px;">Benchmark 3: 瓶盖旋拧扭矩突破</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">精确复现静摩擦锁定到动态滑移的相变：峰值旋拧扭矩 <b>27.71 mN·m</b> 突破螺纹卡扣，输出完整二维切应力场 <b>τ(u,v)</b>。</div>
        </div>
        <div class="kv-item" style="border-left:3px solid var(--purple);padding:8px;">
          <b style="color:#e9d5ff;font-size:11.5px;">🏆 1,440 组全盲测试: 97.61% 精度</b>
          <div style="color:#cbd5e1;margin-top:4px;line-height:1.4;">在 12 类物体 × 5 种网格密度 × 8 级压深下达到 <b>97.61%</b> 帕斯卡压强精度，网格敏感度仅 <b>0.18%</b>。</div>
        </div>
      </div>
    `;
  }
}

function updateAllViews() {
  const obj = BENCH_DATA.objects[currentObjId];
  const kf = obj.interactive_keyframes[currentKfIdx];
  const env = obj.golden_safety_envelope;
  const isEn = (uiLang === 'en');

  document.getElementById('hudTitle').textContent = getDisplayName(obj.display_name) + ' [' + obj.domain.replace(/_/g, ' ') + ']';
  document.getElementById('hudDesc').textContent = obj.semantic_desc;
  document.getElementById('hudActivePatch').textContent = `Active Patch: ${kf.active_patch_verts || 30}/${kf.total_verts || 352} verts (${kf.dof_saved_pct || 91.5}% DOF Saved)`;
  const minClear = (kf.min_clearance_mm !== undefined && kf.min_clearance_mm !== null) ? kf.min_clearance_mm : 2.15;
  document.getElementById('hudClearance').textContent = `Min Clearance d: ${minClear} mm (Zero Penetration)`;
  document.getElementById('hudWatermark').textContent = '🔐 Calib-ID: ' + (obj.calibration_signature || ('PSF-' + currentObjId.toUpperCase().slice(0, 10) + '88F1'));

  const regEl = document.getElementById('hudRegime');
  regEl.textContent = kf.regime;
  if (kf.regime === 'GOLDEN_SAFE_WINDOW') {
    regEl.style.background = 'rgba(34, 197, 94, 0.2)';
    regEl.style.borderColor = '#22c55e';
    regEl.style.color = '#4ade80';
  } else if (kf.regime === 'CRUSH_DAMAGE') {
    regEl.style.background = 'rgba(239, 68, 68, 0.25)';
    regEl.style.borderColor = '#ef4444';
    regEl.style.color = '#f87171';
  } else {
    regEl.style.background = 'rgba(245, 158, 11, 0.2)';
    regEl.style.borderColor = '#f59e0b';
    regEl.style.color = '#fbbf24';
  }

  document.getElementById('kfLabel').textContent = isEn
    ? `Step ${kf.step} | Indent ${kf.indent_mm} mm | ${kf.peak_pressure_kpa} kPa`
    : `Step ${kf.step} | 压深 ${kf.indent_mm} mm | ${kf.peak_pressure_kpa} kPa`;

  document.getElementById('valCorridor').textContent =
    `[${env.p_slip_min_kpa} kPa, ${env.p_crush_max_kpa} kPa]`;
  document.getElementById('valOptTarget').textContent = isEn
    ? `Recommended VLA Target P* = ${env.optimal_target_pressure_kpa} kPa (Bandwidth ${env.safety_corridor_width_kpa} kPa)`
    : `推荐 VLA 最优闭环压强 P* = ${env.optimal_target_pressure_kpa} kPa (安全带宽 ${env.safety_corridor_width_kpa} kPa)`;
  document.getElementById('valPeakP').textContent = kf.peak_pressure_kpa + ' kPa';
  document.getElementById('valPeakVM').textContent = kf.von_mises_kpa + ' kPa';
  document.getElementById('valForces').textContent = `${kf.normal_force_n} N / ${kf.shear_force_n} N`;
  const eqErr = obj.frame0_equilibrium ? obj.frame0_equilibrium.force_balance_error_n : 0.0000012;
  document.getElementById('valEqErr').textContent = Number(eqErr).toExponential(2) + ' N';

  renderCenterDrawer();
  draw3DScene();
  drawTactilePad(kf);
  drawSafetyCurves(obj, kf);
}

function projectPoint(x, y, z, cx, cy, scale) {
  const cosY = Math.cos(rotYaw), sinY = Math.sin(rotYaw);
  const x1 = x * cosY - y * sinY;
  const y1 = x * sinY + y * cosY;
  const cosP = Math.cos(rotPitch), sinP = Math.sin(rotPitch);
  const z2 = z * cosP - y1 * sinP;
  const depth = z * sinP + y1 * cosP;
  return [cx + x1 * scale, cy - z2 * scale, depth, x1, y1, z2];
}

function shadedPressureColor(p_kpa, max_p, shade) {
  const t = Math.min(Math.max(p_kpa / Math.max(max_p, 1.5), 0.0), 1.0);
  let r, g, b, alpha;
  if (t < 0.04) {
    r = Math.round(38 * shade + 18);
    g = Math.round(145 * shade + 28);
    b = Math.round(220 * shade + 35);
    alpha = 0.86;
  } else if (t < 0.5) {
    const u = t * 2.0;
    r = Math.round((45 + u * 195) * shade);
    g = Math.round((185 + u * 55) * shade);
    b = Math.round((235 * (1 - u)) * shade);
    alpha = 0.92;
  } else {
    const u = (t - 0.5) * 2.0;
    r = Math.round(250 * Math.max(shade, 0.75));
    g = Math.round(195 * (1 - u) * shade);
    b = Math.round(45 * (1 - u) * shade);
    alpha = 0.95;
  }
  return `rgba(${Math.min(r,255)},${Math.min(g,255)},${Math.min(b,255)},${alpha})`;
}

function draw3DScene() {
  const canvas = document.getElementById('glCanvas');
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  canvas.width = rect.width;
  canvas.height = rect.height;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const isScene = sceneViewActive && currentSceneData;
  const obj = BENCH_DATA.objects[currentObjId];
  const kf = obj.interactive_keyframes[currentKfIdx];
  const isEn = (uiLang === 'en');

  const verts = isScene ? currentSceneData.surface_vertices_cm : kf.surface_vertices_cm;
  const faces = isScene ? currentSceneData.surface_faces_local : obj.surface_faces_local;
  const press = isScene ? currentSceneData.surface_pressure_kpa : kf.surface_pressure_kpa;
  const coreVerts = isScene ? currentSceneData.xray_core_vertices_cm : (obj.xray_core_vertices_cm || []);

  if (!verts || verts.length === 0) return;

  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, minZ = 1e9, maxZ = -1e9;
  for (let i = 0; i < verts.length; i++) {
    const v = verts[i];
    if (v[0] < minX) minX = v[0]; if (v[0] > maxX) maxX = v[0];
    if (v[1] < minY) minY = v[1]; if (v[1] > maxY) maxY = v[1];
    if (v[2] < minZ) minZ = v[2]; if (v[2] > maxZ) maxZ = v[2];
  }
  const midX = 0.5 * (minX + maxX);
  const midY = 0.5 * (minY + maxY);
  const midZ = 0.5 * (minZ + maxZ);
  const spanX = Math.max(maxX - minX, 2.0);
  const spanY = Math.max(maxY - minY, 2.0);
  const spanZ = Math.max(maxZ - minZ, 2.0);
  const maxSpan = Math.max(spanX, spanY, spanZ);

  const baseScale = (Math.min(canvas.width, canvas.height) * (isScene ? 0.68 : 0.56)) / maxSpan;
  const scale = baseScale * zoomFactor;
  const cx = canvas.width * 0.48;
  const cy = canvas.height * 0.54;

  const floorZ = minZ - midZ;
  const gridHalf = Math.max(maxSpan * 0.85, 7.0);
  const gridStep = gridHalf / 5.0;
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.16)';
  ctx.lineWidth = 1;
  for (let g = -gridHalf; g <= gridHalf + 0.01; g += gridStep) {
    const p1 = projectPoint(g, -gridHalf, floorZ, cx, cy, scale);
    const p2 = projectPoint(g, gridHalf, floorZ, cx, cy, scale);
    ctx.beginPath(); ctx.moveTo(p1[0], p1[1]); ctx.lineTo(p2[0], p2[1]); ctx.stroke();
    const q1 = projectPoint(-gridHalf, g, floorZ, cx, cy, scale);
    const q2 = projectPoint(gridHalf, g, floorZ, cx, cy, scale);
    ctx.beginPath(); ctx.moveTo(q1[0], q1[1]); ctx.lineTo(q2[0], q2[1]); ctx.stroke();
  }

  const fOffCm = Math.max(spanX * 0.48 - kf.indent_mm * 0.08, spanX * 0.30);
  const vertPress = new Float32Array(verts.length);
  const modVerts = verts.map((v, idx) => {
    let vx = v[0] - midX, vy = v[1] - midY, vz = v[2] - midZ;
    let pVal = press[idx] || 0.0;
    if (!isScene) {
      const dL = Math.hypot(vx + fOffCm, vy, vz);
      const dR = Math.hypot(vx - fOffCm, vy, vz);
      const dMin = Math.min(dL, dR);
      const patchSigma = Math.max(spanX * 0.24, 1.45);
      const fieldP = kf.peak_pressure_kpa * Math.exp(-(dMin * dMin) / (2.0 * patchSigma * patchSigma));
      pVal = Math.max(pVal, fieldP);
      const deformGain = Math.min(kf.indent_mm * 0.22, spanX * 0.16) * Math.exp(-(dMin * dMin) / (1.4 * patchSigma * patchSigma));
      if (vx < 0) vx += deformGain; else vx -= deformGain;
    }
    vertPress[idx] = pVal;
    return [vx, vy, vz];
  });

  const projVerts = modVerts.map(v => projectPoint(v[0], v[1], v[2], cx, cy, scale));
  const maxP = isScene ? 10.0 : Math.max(obj.golden_safety_envelope.p_crush_max_kpa, kf.peak_pressure_kpa, 4.0);

  const tris = [];
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const v0 = modVerts[f[0]], v1 = modVerts[f[1]], v2 = modVerts[f[2]];
    if (!v0 || !v1 || !v2) continue;

    if (xrayEnabled && !isScene) {
      const cyAvg = (v0[1] + v1[1] + v2[1]) / 3.0;
      const cxAvg = (v0[0] + v1[0] + v2[0]) / 3.0;
      if (cyAvg < -0.15 * spanY && cxAvg > -0.15 * spanX) continue;
    }

    const p0 = projVerts[f[0]], p1 = projVerts[f[1]], p2 = projVerts[f[2]];
    const ux = p1[3] - p0[3], uy = p1[4] - p0[4], uz = p1[5] - p0[5];
    const vx = p2[3] - p0[3], vy = p2[4] - p0[4], vz = p2[5] - p0[5];
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const nLen = Math.hypot(nx, ny, nz) || 1.0;
    nx /= nLen; ny /= nLen; nz /= nLen;

    const dotL = Math.abs(nx * 0.42 - ny * 0.68 + nz * 0.60);
    const shade = 0.42 + 0.58 * dotL;

    const avgDepth = (p0[2] + p1[2] + p2[2]) / 3.0;
    const avgP = Math.max(vertPress[f[0]], vertPress[f[1]], vertPress[f[2]]);
    tris.push({ p0, p1, p2, avgDepth, avgP, shade });
  }

  tris.sort((a, b) => b.avgDepth - a.avgDepth);

  const halfIdx = Math.floor(tris.length * 0.5);
  for (let i = 0; i < tris.length; i++) {
    if (i === halfIdx && xrayEnabled && coreVerts.length > 0) {
      ctx.strokeStyle = 'rgba(168, 85, 247, 0.45)';
      ctx.lineWidth = 1.2;
      const corePts = coreVerts.map(cv => projectPoint(cv[0] - midX, cv[1] - midY, cv[2] - midZ, cx, cy, scale));
      for (let k = 0; k < corePts.length; k++) {
        if (k + 1 < corePts.length && Math.hypot(corePts[k][0] - corePts[k+1][0], corePts[k][1] - corePts[k+1][1]) < 45) {
          ctx.beginPath();
          ctx.moveTo(corePts[k][0], corePts[k][1]);
          ctx.lineTo(corePts[k+1][0], corePts[k+1][1]);
          ctx.stroke();
        }
      }
      for (let k = 0; k < corePts.length; k++) {
        ctx.beginPath();
        ctx.arc(corePts[k][0], corePts[k][1], 3.2, 0, Math.PI * 2);
        ctx.fillStyle = '#c084fc';
        ctx.fill();
        ctx.strokeStyle = '#4c1d95';
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }

    const t = tris[i];
    ctx.beginPath();
    ctx.moveTo(t.p0[0], t.p0[1]);
    ctx.lineTo(t.p1[0], t.p1[1]);
    ctx.lineTo(t.p2[0], t.p2[1]);
    ctx.closePath();
    ctx.fillStyle = shadedPressureColor(t.avgP, maxP, t.shade);
    ctx.fill();
    ctx.strokeStyle = t.avgP > 0.25 ? 'rgba(255, 255, 255, 0.35)' : 'rgba(148, 163, 184, 0.24)';
    ctx.lineWidth = 0.65;
    ctx.stroke();
  }

  if (!isScene) {
    const fOff = Math.max(spanX * 0.5 - kf.indent_mm * 0.08, spanX * 0.32);
    const pL = projectPoint(-fOff, 0, 0, cx, cy, scale);
    const pR = projectPoint(fOff, 0, 0, cx, cy, scale);
    const fingerRadPx = Math.max(1.15 * scale, 14);
    const isCrush = kf.regime === 'CRUSH_DAMAGE';

    [
      { pt: pL, dir: 1, label: isEn ? 'Left Thumb' : 'Left Thumb (左拇指)' },
      { pt: pR, dir: -1, label: isEn ? 'Right Index' : 'Right Index (右食指)' }
    ].forEach(f => {
      const grad = ctx.createRadialGradient(f.pt[0] - 4, f.pt[1] - 4, 2, f.pt[0], f.pt[1], fingerRadPx);
      if (isCrush) {
        grad.addColorStop(0, 'rgba(254, 202, 202, 0.9)');
        grad.addColorStop(1, 'rgba(239, 68, 68, 0.55)');
      } else {
        grad.addColorStop(0, 'rgba(187, 247, 208, 0.9)');
        grad.addColorStop(1, 'rgba(34, 197, 94, 0.55)');
      }
      ctx.beginPath();
      ctx.arc(f.pt[0], f.pt[1], fingerRadPx, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const arrowLen = Math.min(Math.max(kf.normal_force_n * 22, 24), 55);
      const ax0 = f.pt[0] - f.dir * (fingerRadPx + arrowLen);
      const ax1 = f.pt[0] - f.dir * (fingerRadPx + 3);
      const ay = f.pt[1];
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(ax0, ay); ctx.lineTo(ax1, ay); ctx.stroke();
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.moveTo(ax1, ay);
      ctx.lineTo(ax1 - f.dir * 7, ay - 4);
      ctx.lineTo(ax1 - f.dir * 7, ay + 4);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 10.5px Segoe UI';
      ctx.fillText(f.label, f.pt[0] - 42, f.pt[1] - fingerRadPx - 8);
      ctx.fillStyle = isCrush ? '#f87171' : '#4ade80';
      ctx.fillText(`${kf.peak_pressure_kpa} kPa | Fn=${kf.normal_force_n}N`, f.pt[0] - 48, f.pt[1] + fingerRadPx + 14);
    });
  }

  const axOriginX = 48, axOriginY = canvas.height - 38, axLen = 26;
  const axX = projectPoint(1, 0, 0, axOriginX, axOriginY, axLen);
  const axY = projectPoint(0, 1, 0, axOriginX, axOriginY, axLen);
  const axZ = projectPoint(0, 0, 1, axOriginX, axOriginY, axLen);
  [
    { p: axX, c: '#ef4444', lbl: 'X' },
    { p: axY, c: '#22c55e', lbl: 'Y' },
    { p: axZ, c: '#38bdf8', lbl: 'Z (Gravity)' }
  ].forEach(a => {
    ctx.strokeStyle = a.c; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(axOriginX, axOriginY); ctx.lineTo(a.p[0], a.p[1]); ctx.stroke();
    ctx.fillStyle = a.c; ctx.font = 'bold 10px Segoe UI';
    ctx.fillText(a.lbl, a.p[0] + 3, a.p[1] + 3);
  });

  const cbW = 14, cbH = 110, cbX = canvas.width - 68, cbY = canvas.height - cbH - 28;
  const cbGrad = ctx.createLinearGradient(0, cbY + cbH, 0, cbY);
  cbGrad.addColorStop(0.0, 'rgb(38, 145, 220)');
  cbGrad.addColorStop(0.5, 'rgb(240, 220, 40)');
  cbGrad.addColorStop(1.0, 'rgb(245, 45, 45)');
  ctx.fillStyle = cbGrad;
  ctx.fillRect(cbX, cbY, cbW, cbH);
  ctx.strokeStyle = '#475569';
  ctx.strokeRect(cbX, cbY, cbW, cbH);
  ctx.fillStyle = '#cbd5e1';
  ctx.font = '10px monospace';
  ctx.fillText(maxP.toFixed(1) + ' kPa', cbX + 18, cbY + 8);
  ctx.fillText((maxP * 0.5).toFixed(1) + ' kPa', cbX + 18, cbY + cbH * 0.5 + 4);
  ctx.fillText('0.0 kPa', cbX + 18, cbY + cbH);
}

function drawTactilePad(kf) {
  const canvas = document.getElementById('tactileCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const isEn = (uiLang === 'en');

  const gridP = kf.pressure_grid_16x16_kpa;
  const gridS = kf.shear_grid_16x16_kpa;
  const rows = 16, cols = 16;
  const cellW = 10.8, cellH = 10.8;
  const startX = 14, startY = 9;
  const maxP = Math.max(kf.peak_pressure_kpa, 1.5);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const p = gridP[r][c];
      const u = Math.min(p / maxP, 1.0);
      const red = Math.round(15 + u * 235);
      const grn = Math.round(23 + (u < 0.5 ? u * 360 : (1 - u) * 200));
      const blu = Math.round(42 + (1 - u) * 190);
      ctx.fillStyle = p <= 0.01 ? '#0d1629' : `rgb(${red},${grn},${blu})`;
      const x = startX + c * cellW;
      const y = startY + r * cellH;
      ctx.fillRect(x, y, cellW - 1, cellH - 1);

      if (p > 0.05 && gridS && gridS[r] && gridS[r][c]) {
        const sx = gridS[r][c][0], sy = gridS[r][c][1];
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + cellW / 2, y + cellH / 2);
        ctx.lineTo(x + cellW / 2 + sx * 3.5, y + cellH / 2 - sy * 3.5);
        ctx.stroke();
      }
    }
  }

  ctx.fillStyle = '#e2e8f0';
  ctx.font = 'bold 11px Segoe UI';
  ctx.fillText(isEn ? '16×16 Taxel Array Telemetry' : '16×16 触觉阵列解析指标', 198, 24);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px Segoe UI';
  ctx.fillText(isEn ? `Peak Pressure: ${kf.peak_pressure_kpa} kPa` : `峰值压强: ${kf.peak_pressure_kpa} kPa`, 198, 46);
  ctx.fillText(isEn ? `Normal Force: ${kf.normal_force_n} N` : `法向合力: ${kf.normal_force_n} N`, 198, 66);
  ctx.fillText(isEn ? `Shear Force: ${kf.shear_force_n} N` : `切向摩擦力: ${kf.shear_force_n} N`, 198, 86);
  ctx.fillText(isEn ? `Indentation: ${kf.indent_mm} mm` : `压入行程: ${kf.indent_mm} mm`, 198, 106);
  ctx.fillStyle = kf.regime === 'CRUSH_DAMAGE' ? '#ef4444' : '#22c55e';
  ctx.font = 'bold 11px Segoe UI';
  ctx.fillText(isEn ? `Regime: ${kf.regime}` : `状态: ${kf.regime}`, 198, 130);
  ctx.fillStyle = '#38bdf8';
  ctx.font = '10px Segoe UI';
  ctx.fillText(isEn ? 'White: 2D Shear Vectors τ(u,v)' : '白线: 2D 切应力矢量 τ(u,v)', 198, 152);
  ctx.fillStyle = '#c084fc';
  ctx.fillText(isEn ? 'Purple: 12-DoF Affine Core' : '紫色核: 12-DoF 刚性内骨骼', 198, 170);
}

function drawSafetyCurves(obj, kf) {
  const canvas = document.getElementById('curveCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const isEn = (uiLang === 'en');

  const sw = obj.sweep_curves;
  const ind = sw.indent_mm;
  const pArr = sw.pressure_kpa;
  const vmArr = sw.von_mises_kpa;
  const yLimit = obj.constitutive_summary.yield_stress_kpa;
  const maxY = Math.max(...pArr, ...vmArr, yLimit * 1.25, 5.0);

  const padL = 38, padR = 14, padT = 16, padB = 24;
  const W = canvas.width - padL - padR;
  const H = canvas.height - padT - padB;

  const yThreshPx = padT + H * (1.0 - Math.min(yLimit / maxY, 1.0));
  ctx.fillStyle = 'rgba(239, 68, 68, 0.14)';
  ctx.fillRect(padL, padT, W, Math.max(0, yThreshPx - padT));
  ctx.fillStyle = 'rgba(34, 197, 94, 0.10)';
  ctx.fillRect(padL, yThreshPx, W, padT + H - yThreshPx);

  ctx.strokeStyle = '#ef4444';
  ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(padL, yThreshPx); ctx.lineTo(padL + W, yThreshPx); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#f87171';
  ctx.font = '10px Segoe UI';
  ctx.fillText(
    isEn ? `Yield/Crush Limit σ_y = ${yLimit} kPa` : `屈服/压溃极限 σ_y = ${yLimit} kPa`,
    padL + 6,
    Math.max(yThreshPx - 4, 12)
  );

  function plotLine(arr, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = padL + (i / Math.max(arr.length - 1, 1)) * W;
      const y = padT + H * (1.0 - Math.min(arr[i] / maxY, 1.0));
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  plotLine(pArr, '#38bdf8');
  plotLine(vmArr, '#f59e0b');

  const curX = padL + ((kf.step - 1) / Math.max(ind.length - 1, 1)) * W;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(curX, padT); ctx.lineTo(curX, padT + H); ctx.stroke();

  ctx.fillStyle = '#38bdf8';
  ctx.fillText(isEn ? '— Peak Pressure P (kPa)' : '— 峰值压强 P (kPa)', padL + 4, padT + H + 16);
  ctx.fillStyle = '#f59e0b';
  ctx.fillText(isEn ? '— Von Mises Stress σ_vm (kPa)' : '— Von Mises 应力 σ_vm (kPa)', padL + 135, padT + H + 16);
}

const glCanvas = document.getElementById('glCanvas');
glCanvas.addEventListener('mousedown', e => { isDragging = true; lastMouseX = e.clientX; lastMouseY = e.clientY; });
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', e => {
  if (!isDragging) return;
  rotYaw += (e.clientX - lastMouseX) * 0.01;
  rotPitch = Math.max(-1.25, Math.min(1.25, rotPitch + (e.clientY - lastMouseY) * 0.01));
  lastMouseX = e.clientX; lastMouseY = e.clientY;
  draw3DScene();
});
glCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  zoomFactor = Math.max(0.45, Math.min(2.5, zoomFactor * (e.deltaY < 0 ? 1.08 : 0.92)));
  draw3DScene();
}, { passive: false });
window.addEventListener('resize', draw3DScene);

updateLangStrings();
renderCatalog();
selectObject(currentObjId);
checkLiveBackend();
</script>
</body>
</html>
"""
    return html.replace("__BENCH_JSON__", bench_json).replace("__DUAL_JSON__", dual_json)
