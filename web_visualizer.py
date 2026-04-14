#!/usr/bin/env python3
"""
Lightweight browser visualizer for WMR System Identification.

Trains multiple system ID models, generates trajectories, and serves an
interactive Canvas-based animation at http://localhost:5000

Supported models:
  Kinematic — pure kinematics, no motor dynamics
  FIR       — finite impulse response (feed-forward only)
  ARX-3     — autoregressive, order 3  (underfitting baseline)
  ARX-9     — autoregressive, order 9  (paper's optimal config)
  ARMAX     — ARX + moving-average noise model  (iterative PLR)
  OE        — output error  (simulation-consistent training)

Usage:
    pip install flask
    python web_visualizer.py
    # Open http://localhost:5000
"""

import sys
import numpy as np
from flask import Flask, jsonify

sys.path.insert(0, ".")
from robot_simulator import DifferentialDriveRobot
from data_generator import DataGenerator
from arx_model import MIMOARXModel
from extra_models import MIMOFIRModel, MIMOARMAXModel, MIMOOEModel
from trajectory_generator import TrajectoryGenerator

app = Flask(__name__)
_sim_data: dict = {}

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_DEFS = {
    'fir':    ('FIR',       '#a6e3a1',  'Finite Impulse Response (no AR feedback)'),
    'arx_3':  ('ARX-3',     '#f9e2af',  'ARX order 3 (underfitting baseline)'),
    'arx_9':  ('ARX-9',     '#fab387',  "ARX order 9 (paper's optimal config)"),
    'armax':  ('ARMAX',     '#cba6f7',  'ARX + Moving Average noise model (PLR)'),
    'oe':     ('OE',        '#f38ba8',  'Output Error (simulation-consistent training)'),
}

# ── Embedded HTML ─────────────────────────────────────────────────────────────
def _make_html(model_defs):
    model_btns = '\n'.join(
        f'<button class="mbtn" data-key="{k}" data-color="{v[1]}" title="{v[2]}">'
        f'{v[0]}</button>'
        for k, v in model_defs.items()
    )
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WMR Simulation Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Courier New', monospace; background: #0d0d1a; color: #cdd6f4; overflow: hidden; }
header {
  padding: 8px 18px; background: #181825;
  border-bottom: 1px solid #313244; display: flex; align-items: center; gap: 16px;
}
header h1 { font-size: 0.95rem; color: #f38ba8; }
header p  { font-size: 0.7rem; color: #6c7086; }
.main { display: flex; height: calc(100vh - 40px); }
.sidebar {
  width: 215px; background: #181825; border-right: 1px solid #313244;
  padding: 12px 11px; display: flex; flex-direction: column; gap: 10px; overflow-y: auto;
}
.sidebar h3 { font-size: 0.68rem; color: #f38ba8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
select {
  width: 100%; padding: 5px 7px; border: 1px solid #313244; border-radius: 4px;
  background: #1e1e2e; color: #cdd6f4; font-family: inherit; font-size: 0.78rem; cursor: pointer; outline: none;
}
select:hover { border-color: #f38ba8; }
.ctrl-row { display: flex; gap: 5px; }
.ctrl-row button, .pbtn {
  flex: 1; padding: 5px 6px; border: 1px solid #313244; border-radius: 4px;
  background: #1e1e2e; color: #cdd6f4; font-family: inherit; font-size: 0.78rem; cursor: pointer; outline: none;
}
.ctrl-row button:hover, .pbtn:hover { border-color: #f38ba8; color: #f38ba8; }
.pbtn.active { background: #f38ba8; color: #1e1e2e; border-color: #f38ba8; }
.slider-wrap label { font-size: 0.68rem; color: #6c7086; display: flex; justify-content: space-between; }
.slider-wrap label span { color: #cba6f7; }
input[type=range] { width: 100%; accent-color: #cba6f7; cursor: pointer; margin-top: 3px; }
/* Model buttons */
.model-grid { display: flex; flex-direction: column; gap: 4px; }
.mbtn {
  width: 100%; padding: 5px 8px; border: 1px solid #313244; border-radius: 4px;
  background: #1e1e2e; color: #cdd6f4; font-family: inherit; font-size: 0.78rem;
  cursor: pointer; text-align: left; outline: none; transition: all .15s;
}
.mbtn:hover { border-color: #cdd6f4; }
.mbtn.active { color: #1e1e2e !important; font-weight: bold; }
/* Legend */
.legend { display: flex; flex-direction: column; gap: 5px; }
.leg { display: flex; align-items: center; gap: 7px; font-size: 0.7rem; }
.leg-line { width: 22px; height: 3px; border-radius: 2px; flex-shrink: 0; }
/* Stats */
.stat-block { display: flex; flex-direction: column; gap: 4px; }
.stat { font-size: 0.72rem; display: flex; justify-content: space-between; gap: 6px; }
.stat .val { color: #a6e3a1; font-weight: bold; white-space: nowrap; }
.stat.warn .val { color: #fab387; }
/* Canvas */
canvas { flex: 1; display: block; background: #0d0d1a; }
/* Overlay */
#overlay {
  position: fixed; inset: 0; background: rgba(13,13,26,0.93);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; z-index: 99;
}
#overlay h2 { color: #f38ba8; }
.spinner { width: 36px; height: 36px; border: 3px solid #313244; border-top-color: #f38ba8;
  border-radius: 50%; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
#overlay p { color: #6c7086; font-size: 0.8rem; }
</style>
</head>
<body>

<div id="overlay">
  <div class="spinner"></div>
  <h2>Training models…</h2>
  <p id="loadMsg">This may take 30–60 seconds on first load.</p>
</div>

<header>
  <div>
    <h1>WMR System Identification — Live Simulation</h1>
    <p>Pioneer P3-DX &nbsp;|&nbsp; Ground Truth vs System ID Models</p>
  </div>
</header>

<div class="main">
  <div class="sidebar">

    <div>
      <h3>Trajectory</h3>
      <select id="trajSelect"></select>
    </div>

    <div class="ctrl-row">
      <button id="playBtn">▶ Play</button>
      <button id="resetBtn">↺</button>
    </div>

    <div class="slider-wrap">
      <label>Speed <span id="speedLabel">5×</span></label>
      <input type="range" id="speedSlider" min="1" max="80" value="5">
    </div>

    <div>
      <h3>Model</h3>
      <div class="model-grid" id="modelGrid">
        """ + model_btns + r"""
      </div>
    </div>

    <div>
      <h3>Legend</h3>
      <div class="legend">
        <div class="leg">
          <div class="leg-line" style="background:#89dceb"></div>
          <span>Ground Truth</span>
        </div>
        <div class="leg" id="modelLegend">
          <div class="leg-line" id="modelLegLine" style="background:#fab387"></div>
          <span id="modelLegLabel">ARX-9</span>
        </div>
        <div class="leg">
          <div class="leg-line" style="background:none;border-top:2px dashed #585b70;height:0"></div>
          <span style="color:#585b70">Kinematic Ideal</span>
        </div>
      </div>
    </div>

    <div>
      <h3>Stats</h3>
      <div class="stat-block">
        <div class="stat"><span>Time</span><span class="val" id="sTime">0.00 s</span></div>
        <div class="stat"><span>Step</span><span class="val" id="sStep">0</span></div>
        <div class="stat warn"><span>Pos Error</span><span class="val" id="sPosErr">—</span></div>
        <div class="stat warn"><span>Head Error</span><span class="val" id="sHdErr">—</span></div>
      </div>
    </div>

  </div>
  <canvas id="canvas"></canvas>
</div>

<script>
const DT = 0.01;
let data = {}, currentTraj = null, currentModel = 'arx_9', step = 0, playing = false, rafId = null;

const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');

const MODEL_COLORS = """ + str({k: v[1] for k, v in model_defs.items()}).replace("'", '"') + r""";
const MODEL_LABELS = """ + str({k: v[0] for k, v in model_defs.items()}).replace("'", '"') + r""";

function resize() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  if (currentTraj) render();
}
window.addEventListener('resize', resize);

// ── Load ──────────────────────────────────────────────────────────────────────
fetch('/api/simulation')
  .then(r => r.json())
  .then(d => {
    data = d;
    document.getElementById('overlay').style.display = 'none';
    const sel = document.getElementById('trajSelect');
    const labels = {
      figure8: 'Figure-8', square: 'Square', spiral: 'Spiral',
      slalom: 'Slalom ⚡', zigzag: 'Zigzag ⚡', step_burst: 'Step Burst ⚡',
      fan: 'Fan (multi)',
    };
    Object.keys(data).forEach(k => {
      const o = document.createElement('option');
      o.value = k; o.textContent = labels[k] || k; sel.appendChild(o);
    });
    currentTraj = Object.keys(data)[0];
    sel.value   = currentTraj;
    selectModel('arx_9');
    resize();
  });

// ── Controls ──────────────────────────────────────────────────────────────────
document.getElementById('trajSelect').addEventListener('change', e => {
  currentTraj = e.target.value; step = 0; stopPlay(); render();
});
document.getElementById('playBtn').addEventListener('click', togglePlay);
document.getElementById('resetBtn').addEventListener('click', () => { step = 0; stopPlay(); render(); });
document.getElementById('speedSlider').addEventListener('input', e =>
  document.getElementById('speedLabel').textContent = e.target.value + '×'
);

// Model buttons
document.querySelectorAll('.mbtn').forEach(btn => {
  btn.addEventListener('click', () => selectModel(btn.dataset.key));
});

function selectModel(key) {
  if (!MODEL_COLORS[key]) return;
  currentModel = key;
  const color  = MODEL_COLORS[key];
  const label  = MODEL_LABELS[key];
  // Update button highlight
  document.querySelectorAll('.mbtn').forEach(b => {
    b.classList.toggle('active', b.dataset.key === key);
    if (b.dataset.key === key) {
      b.style.background    = color;
      b.style.borderColor   = color;
    } else {
      b.style.background    = '';
      b.style.borderColor   = '';
      b.style.color         = '';
    }
  });
  // Update legend
  document.getElementById('modelLegLine').style.background  = color;
  document.getElementById('modelLegLabel').textContent       = label;
  step = 0; stopPlay(); render();
}

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) loop();
}
function stopPlay() {
  playing = false;
  document.getElementById('playBtn').textContent = '▶ Play';
  cancelAnimationFrame(rafId);
}
function loop() {
  if (!playing) return;
  const spd  = parseInt(document.getElementById('speedSlider').value);
  const traj = data[currentTraj];
  const maxS = traj.multi ? traj.gt_x[0].length - 1 : traj.gt_x.length - 1;
  step = Math.min(step + spd, maxS);
  render();
  if (step < maxS) rafId = requestAnimationFrame(loop);
  else stopPlay();
}

// ── Coordinate transform ──────────────────────────────────────────────────────
function bbox(traj) {
  const mdata = traj.models[currentModel] || Object.values(traj.models)[0];
  let xs, ys;
  if (traj.multi) {
    xs = traj.gt_x.flat().concat(mdata.x.flat(), traj.kin_x.flat());
    ys = traj.gt_y.flat().concat(mdata.y.flat(), traj.kin_y.flat());
  } else {
    xs = traj.gt_x.concat(mdata.x, traj.kin_x);
    ys = traj.gt_y.concat(mdata.y, traj.kin_y);
  }
  return { minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) };
}
function makeT(traj) {
  const pad = 54, { minX, maxX, minY, maxY } = bbox(traj);
  const rX = maxX - minX || 1, rY = maxY - minY || 1;
  const W = canvas.width - 2*pad, H = canvas.height - 2*pad;
  const scale = Math.min(W/rX, H/rY) * 0.9;
  return { scale, ox: pad + (W - rX*scale)/2 - minX*scale, oy: pad + (H - rY*scale)/2 + maxY*scale };
}
function w2c(wx, wy, T) { return [wx*T.scale + T.ox, -wy*T.scale + T.oy]; }

// ── Drawing ───────────────────────────────────────────────────────────────────
function drawPath(xs, ys, n, color, dash, T, alpha) {
  if (!xs || !xs.length) return;
  ctx.beginPath();
  ctx.strokeStyle = color; ctx.globalAlpha = alpha ?? 1;
  ctx.lineWidth = 1.6; ctx.setLineDash(dash || []);
  const lim = Math.min(n+1, xs.length);
  for (let i = 0; i < lim; i++) {
    const [cx,cy] = w2c(xs[i], ys[i], T);
    i === 0 ? ctx.moveTo(cx,cy) : ctx.lineTo(cx,cy);
  }
  ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}

function drawRobot(wx, wy, theta, color, T, glow) {
  const [cx,cy] = w2c(wx, wy, T);
  const s = Math.max(6, Math.min(15, T.scale * 0.17));
  ctx.save();
  ctx.translate(cx, cy); ctx.rotate(-theta);
  ctx.beginPath();
  ctx.moveTo( s*1.4, 0); ctx.lineTo(-s*0.8, s*0.65);
  ctx.lineTo(-s*0.4, 0); ctx.lineTo(-s*0.8,-s*0.65);
  ctx.closePath();
  if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 12; }
  ctx.fillStyle = color; ctx.fill();
  ctx.restore();
}

function drawGrid(T, traj) {
  const { minX, maxX, minY, maxY } = bbox(traj);
  const span = Math.max(maxX-minX, maxY-minY, 1);
  const gstep = Math.pow(10, Math.floor(Math.log10(span/4)));
  ctx.strokeStyle = '#1e2030'; ctx.lineWidth = 0.6;
  ctx.fillStyle = '#45475a'; ctx.font = '9px monospace';
  for (let v = Math.ceil(minX/gstep)*gstep - gstep; v <= maxX+gstep; v += gstep) {
    const [cx] = w2c(v, 0, T); if (cx < 0 || cx > canvas.width) continue;
    ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx,canvas.height); ctx.stroke();
    ctx.fillText(v.toFixed(1)+'m', cx+2, canvas.height-4);
  }
  for (let v = Math.ceil(minY/gstep)*gstep - gstep; v <= maxY+gstep; v += gstep) {
    const [,cy] = w2c(0, v, T); if (cy < 0 || cy > canvas.height) continue;
    ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(canvas.width,cy); ctx.stroke();
    ctx.fillText(v.toFixed(1), 3, cy-2);
  }
  ctx.strokeStyle = '#313244'; ctx.lineWidth = 1;
  const [ox,oy] = w2c(0,0,T);
  ctx.beginPath(); ctx.moveTo(ox,0); ctx.lineTo(ox,canvas.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,oy); ctx.lineTo(canvas.width,oy); ctx.stroke();
}

function fanColor(i, n) {
  const t = n > 1 ? i/(n-1) : 0.5;
  return `hsl(${Math.round(200 - t*160)},80%,60%)`;
}

// ── Main render ───────────────────────────────────────────────────────────────
function render() {
  if (!currentTraj || !data[currentTraj]) return;
  const traj  = data[currentTraj];
  const mdata = traj.models[currentModel] || Object.values(traj.models)[0];
  const col   = MODEL_COLORS[currentModel] || '#fab387';
  const T     = makeT(traj);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(T, traj);

  if (traj.multi) {
    const n = traj.gt_x.length;
    traj.gt_x.forEach((gx, i) => {
      const fc  = fanColor(i, n);
      const mx  = mdata.x[i], my = mdata.y[i], mt = mdata.theta[i];
      // ghost
      drawPath(gx, traj.gt_y[i], gx.length, fc, [], T, 0.10);
      drawPath(mx, my, mx.length, col, [], T, 0.07);
      drawPath(traj.kin_x[i], traj.kin_y[i], gx.length, '#585b70', [4,4], T, 0.12);
      // traveled
      drawPath(traj.kin_x[i], traj.kin_y[i], step, '#585b70', [4,4], T, 0.40);
      drawPath(gx, traj.gt_y[i], step, fc, [], T);
      drawPath(mx, my, step, col, [], T, 0.65);
      // robots
      const s = Math.min(step, gx.length-1);
      drawRobot(gx[s], traj.gt_y[i][s], traj.gt_theta[i][s], fc, T, true);
      drawRobot(mx[s], my[s], mt[s], col, T, false);
    });
    let avgErr = 0;
    mdata.pos_err.forEach(arr => { avgErr += arr[Math.min(step, arr.length-1)]; });
    updateStats(step, avgErr / mdata.pos_err.length, null);
  } else {
    const N = traj.gt_x.length;
    // ghost full paths (faint)
    drawPath(traj.gt_x,  traj.gt_y,  N, '#89dceb', [], T, 0.10);
    drawPath(mdata.x,    mdata.y,    N, col,       [], T, 0.08);
    drawPath(traj.kin_x, traj.kin_y, N, '#585b70', [4,4], T, 0.07);
    // traveled
    drawPath(traj.kin_x, traj.kin_y, step, '#585b70', [4,4], T, 0.45);
    drawPath(traj.gt_x,  traj.gt_y,  step, '#89dceb', [], T);
    drawPath(mdata.x,    mdata.y,    step, col,       [], T);
    // start marker
    const [sx,sy] = w2c(traj.gt_x[0], traj.gt_y[0], T);
    ctx.beginPath(); ctx.arc(sx,sy,4,0,2*Math.PI);
    ctx.fillStyle = '#a6e3a1'; ctx.fill();
    // robots
    const s = Math.min(step, N-1);
    drawRobot(traj.gt_x[s],  traj.gt_y[s],  traj.gt_theta[s],  '#89dceb', T, true);
    drawRobot(mdata.x[s],    mdata.y[s],    mdata.theta[s],    col,       T, true);
    const pe = s < mdata.pos_err.length  ? mdata.pos_err[s]  : null;
    const he = s < mdata.head_err.length ? mdata.head_err[s] : null;
    updateStats(s, pe, he);
  }
}

function updateStats(s, pe, he) {
  document.getElementById('sTime').textContent   = (s * DT).toFixed(2) + ' s';
  document.getElementById('sStep').textContent   = s;
  document.getElementById('sPosErr').textContent = pe != null ? pe.toFixed(4) + ' m'   : '—';
  document.getElementById('sHdErr').textContent  = he != null ? he.toFixed(4) + ' rad' : '—';
}
</script>
</body>
</html>"""


HTML = _make_html(MODEL_DEFS)


# ── Simulation config ──────────────────────────────────────────────────────────
_CFG = {
    'wheel_radius': 0.0975,
    'wheel_base': 0.381,
    'max_wheel_speed': 1.5,
    'motor_time_constant': 0.1,
    'dt': 0.01,
  'simulator_backend': 'analytic',  # 'analytic' or 'pybullet'
  'pybullet_gui': False,
    'n_samples': 8000,
    'train_split': 0.8,
    'v_range': (-1.0, 1.0),
    'omega_range': (-1.5, 1.5),
    'min_hold_steps': 100,
    'max_hold_steps': 200,
    'rng_seed': 42,
    'ridge_alpha': 1e-8,
    # Paper's optimal ARX orders
    'na_diag': 9, 'na_cross': 3,
    'nb_diag': 9, 'nb_cross': 1,
}


def _to_list(a, stride=2):
    return a[::stride].tolist()


def _build_model_entry(tgen, robot, v_cmds, omega_cmds, gt, kin, stride=2):
    """Run a TrajectoryGenerator and compute comparison metrics vs ground truth."""
    from robot_simulator import _wrap_to_pi as _wtp
    robot.reset()
    arx_traj = tgen.generate(v_cmds, omega_cmds, warmup_robot=robot)
    at = arx_traj
    pos_err  = np.sqrt((gt['x'] - at['x'])**2 + (gt['y'] - at['y'])**2)
    head_err = np.abs(np.array([_wtp(a - b) for a, b in zip(gt['theta'], at['theta'])]))
    s = stride
    return {
        'x':        _to_list(at['x'], s),
        'y':        _to_list(at['y'], s),
        'theta':    _to_list(at['theta'], s),
        'pos_err':  _to_list(pos_err, s),
        'head_err': _to_list(head_err, s),
    }


def run_simulation() -> dict:
    """Train all models and compute trajectories."""
    print("=" * 55)
    print("  [1/4] Creating robot and generating training data…", flush=True)

    if _CFG['simulator_backend'] == 'pybullet':
        from pybullet_simulator import PyBulletDifferentialDriveRobot
        robot = PyBulletDifferentialDriveRobot(
            wheel_radius=_CFG['wheel_radius'],
            wheel_base=_CFG['wheel_base'],
            max_wheel_speed=_CFG['max_wheel_speed'],
            motor_time_constant=_CFG['motor_time_constant'],
            dt=_CFG['dt'],
            use_gui=_CFG['pybullet_gui'],
        )
    else:
        robot = DifferentialDriveRobot(
            wheel_radius=_CFG['wheel_radius'],
            wheel_base=_CFG['wheel_base'],
            max_wheel_speed=_CFG['max_wheel_speed'],
            motor_time_constant=_CFG['motor_time_constant'],
            dt=_CFG['dt'],
        )

    gen = DataGenerator(robot, rng_seed=_CFG['rng_seed'])
    dataset = gen.generate_dataset(
        n_samples=_CFG['n_samples'],
        split=_CFG['train_split'],
        v_range=_CFG['v_range'],
        omega_range=_CFG['omega_range'],
        min_hold_steps=_CFG['min_hold_steps'],
        max_hold_steps=_CFG['max_hold_steps'],
    )
    Y_train, U_train = dataset['train']['y'], dataset['train']['u']

    # ── [2/4] Fit all models ─────────────────────────────────────────────────
    print("  [2/4] Training models…", flush=True)

    models = {}

    print("        FIR …",    flush=True)
    fir = MIMOFIRModel(nb_diag=9, nb_cross=1, ridge_alpha=_CFG['ridge_alpha'])
    fir.fit(Y_train, U_train)
    models['fir'] = fir

    print("        ARX-3 …", flush=True)
    arx3 = MIMOARXModel(na_diag=3, na_cross=1, nb_diag=3, nb_cross=1, ridge_alpha=_CFG['ridge_alpha'])
    arx3.fit(Y_train, U_train)
    models['arx_3'] = arx3

    print("        ARX-9 …", flush=True)
    arx9 = MIMOARXModel(
        na_diag=_CFG['na_diag'], na_cross=_CFG['na_cross'],
        nb_diag=_CFG['nb_diag'], nb_cross=_CFG['nb_cross'],
        ridge_alpha=_CFG['ridge_alpha'],
    )
    arx9.fit(Y_train, U_train)
    models['arx_9'] = arx9

    print("        ARMAX (3 PLR iter) …", flush=True)
    armax = MIMOARMAXModel(
        na_diag=_CFG['na_diag'], na_cross=_CFG['na_cross'],
        nb_diag=_CFG['nb_diag'], nb_cross=_CFG['nb_cross'],
        nc=3, n_iter=3, ridge_alpha=_CFG['ridge_alpha'],
    )
    armax.fit(Y_train, U_train)
    models['armax'] = armax

    print("        OE (5 iter) …", flush=True)
    oe = MIMOOEModel(
        na_diag=_CFG['na_diag'], na_cross=_CFG['na_cross'],
        nb_diag=_CFG['nb_diag'], nb_cross=_CFG['nb_cross'],
        n_iter=5, ridge_alpha=_CFG['ridge_alpha'],
    )
    oe.fit(Y_train, U_train)
    models['oe'] = oe

    # ── [3/4] Named trajectories ──────────────────────────────────────────────
    print("  [3/4] Generating named trajectories…", flush=True)
    result = {}
    stride = 2

    # slalom/zigzag/step_burst use shorter sequences — 1500 steps = 15 s is plenty
    traj_lengths = {
        'figure8': 2000, 'square': 2000, 'spiral': 2000,
        'slalom': 1500, 'zigzag': 1500, 'step_burst': 1000,
    }
    for name in ['figure8', 'square', 'spiral', 'slalom', 'zigzag', 'step_burst']:
        print(f"        {name} …", flush=True)
        v_cmds, omega_cmds = gen.generate_trajectory_commands(
            trajectory_type=name, n_samples=traj_lengths[name]
        )
        # Ground truth (reset robot each time inside simulate)
        robot.reset()
        from trajectory_generator import _kinematic_trajectory
        gt = robot.simulate(v_cmds, omega_cmds)
        kin = _kinematic_trajectory(v_cmds, omega_cmds, _CFG['dt'])
        s = stride

        entry = {
            'gt_x':   _to_list(gt['x'], s),   'gt_y':   _to_list(gt['y'], s),
            'gt_theta': _to_list(gt['theta'], s),
            'kin_x':  _to_list(kin['x'], s),  'kin_y':  _to_list(kin['y'], s),
            'models': {},
            'multi':  False,
        }

        for key, model in models.items():
            tgen = TrajectoryGenerator(model, dt=_CFG['dt'])
            robot.reset()
            entry['models'][key] = _build_model_entry(tgen, robot, v_cmds, omega_cmds, gt, kin, s)

        result[name] = entry

    # ── [4/4] Fan trajectories ────────────────────────────────────────────────
    print("  [4/4] Generating fan trajectories…", flush=True)
    omega_finals = np.linspace(-2.0, 2.0, 9)

    fan_gt_x, fan_gt_y, fan_gt_theta = [], [], []
    fan_kin_x, fan_kin_y = [], []
    fan_models = {k: {'x': [], 'y': [], 'theta': [], 'pos_err': [], 'head_err': []}
                  for k in models}

    for of in omega_finals:
        v_c = np.ones(300) * 0.5
        w_c = np.linspace(0, of, 300)
        robot.reset()
        gt_f  = robot.simulate(v_c, w_c)
        kin_f = _kinematic_trajectory(v_c, w_c, _CFG['dt'])

        fan_gt_x.append(gt_f['x'].tolist()); fan_gt_y.append(gt_f['y'].tolist())
        fan_gt_theta.append(gt_f['theta'].tolist())
        fan_kin_x.append(kin_f['x'].tolist()); fan_kin_y.append(kin_f['y'].tolist())

        for key, model in models.items():
            tgen = TrajectoryGenerator(model, dt=_CFG['dt'])
            robot.reset()
            at = tgen.generate(v_c, w_c, warmup_robot=robot)
            from robot_simulator import _wrap_to_pi as _wtp
            pe = np.sqrt((gt_f['x'] - at['x'])**2 + (gt_f['y'] - at['y'])**2)
            he = np.abs(np.array([_wtp(a - b) for a, b in zip(gt_f['theta'], at['theta'])]))
            fan_models[key]['x'].append(at['x'].tolist())
            fan_models[key]['y'].append(at['y'].tolist())
            fan_models[key]['theta'].append(at['theta'].tolist())
            fan_models[key]['pos_err'].append(pe.tolist())
            fan_models[key]['head_err'].append(he.tolist())

    result['fan'] = {
        'gt_x':   fan_gt_x, 'gt_y':  fan_gt_y, 'gt_theta': fan_gt_theta,
        'kin_x':  fan_kin_x,'kin_y': fan_kin_y,
        'models': fan_models,
        'multi':  True,
    }

    print("  Done — all models computed.", flush=True)
    return result


# ── Flask routes ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return HTML


@app.route('/api/simulation')
def api_simulation():
    return jsonify(_sim_data)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("WMR Simulation Web Visualizer — Multi-Model")
    print("=" * 55)
    _sim_data.update(run_simulation())
    print(f"\nOpen http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
