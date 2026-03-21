#!/usr/bin/env python3
"""
Lightweight browser visualizer for WMR System Identification.

Trains the ARX model, generates trajectories, and serves an interactive
Canvas-based animation at http://localhost:5000

Usage:
    pip install flask
    python web_visualizer.py
"""

import sys
import numpy as np
from flask import Flask, jsonify

sys.path.insert(0, ".")
from robot_simulator import DifferentialDriveRobot
from data_generator import DataGenerator
from arx_model import MIMOARXModel
from trajectory_generator import TrajectoryGenerator

app = Flask(__name__)
_sim_data: dict = {}

# ── Embedded HTML / JS / CSS ──────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>WMR Simulation Viewer</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Courier New', monospace; background: #0d0d1a; color: #cdd6f4; overflow: hidden; }
header {
  padding: 10px 20px; background: #181825;
  border-bottom: 1px solid #313244; display: flex; align-items: center; gap: 20px;
}
header h1 { font-size: 1rem; color: #f38ba8; }
header p  { font-size: 0.72rem; color: #6c7086; }
.main { display: flex; height: calc(100vh - 44px); }
.sidebar {
  width: 210px; background: #181825; border-right: 1px solid #313244;
  padding: 14px 12px; display: flex; flex-direction: column; gap: 12px;
  overflow-y: auto;
}
.sidebar h3 { font-size: 0.72rem; color: #f38ba8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
select, button {
  width: 100%; padding: 6px 8px; border: 1px solid #313244;
  border-radius: 4px; background: #1e1e2e; color: #cdd6f4;
  font-family: inherit; font-size: 0.8rem; cursor: pointer; outline: none;
}
select:hover, button:hover { border-color: #f38ba8; color: #f38ba8; }
button.active { background: #f38ba8; color: #1e1e2e; border-color: #f38ba8; }
.row { display: flex; gap: 6px; }
.row button { flex: 1; }
.slider-wrap label { font-size: 0.72rem; color: #6c7086; display: flex; justify-content: space-between; }
.slider-wrap label span { color: #cba6f7; }
input[type=range] { width: 100%; accent-color: #cba6f7; cursor: pointer; margin-top: 4px; }
.stat-block { display: flex; flex-direction: column; gap: 5px; }
.stat { font-size: 0.75rem; display: flex; justify-content: space-between; }
.stat .val { color: #a6e3a1; font-weight: bold; }
.stat.warn .val { color: #fab387; }
.legend { display: flex; flex-direction: column; gap: 6px; }
.leg { display: flex; align-items: center; gap: 8px; font-size: 0.73rem; }
.leg-line { width: 26px; height: 3px; border-radius: 2px; }
canvas { flex: 1; display: block; cursor: crosshair; }
#overlay {
  position: fixed; inset: 0; background: rgba(13,13,26,0.92);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 16px;
}
#overlay h2 { color: #f38ba8; font-size: 1.1rem; }
.spinner {
  width: 40px; height: 40px; border: 3px solid #313244;
  border-top-color: #f38ba8; border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
#overlay p { color: #6c7086; font-size: 0.8rem; }
</style>
</head>
<body>

<div id="overlay">
  <div class="spinner"></div>
  <h2>Training ARX model…</h2>
  <p>This may take 15–30 seconds on first load.</p>
</div>

<header>
  <div>
    <h1>WMR System Identification — Live Simulation</h1>
    <p>Pioneer P3-DX &nbsp;|&nbsp; Ground Truth vs ARX Model vs Kinematic Ideal</p>
  </div>
</header>

<div class="main">
  <div class="sidebar">

    <div>
      <h3>Trajectory</h3>
      <select id="trajSelect"></select>
    </div>

    <div class="row">
      <button id="playBtn">▶ Play</button>
      <button id="resetBtn">↺</button>
    </div>

    <div class="slider-wrap">
      <label>Speed <span id="speedLabel">5×</span></label>
      <input type="range" id="speedSlider" min="1" max="80" value="5">
    </div>

    <div>
      <h3>Legend</h3>
      <div class="legend">
        <div class="leg">
          <div class="leg-line" style="background:none;border-top:2px dashed #9399b2"></div>
          <span style="color:#9399b2">Kinematic Ideal</span>
        </div>
        <div class="leg">
          <div class="leg-line" style="background:#89dceb"></div>
          <span>Ground Truth</span>
        </div>
        <div class="leg">
          <div class="leg-line" style="background:#fab387"></div>
          <span>ARX Model</span>
        </div>
      </div>
    </div>

    <div>
      <h3>Stats</h3>
      <div class="stat-block">
        <div class="stat"><span>Time</span><span class="val" id="sTime">0.00 s</span></div>
        <div class="stat"><span>Step</span><span class="val" id="sStep">0</span></div>
        <div class="stat warn"><span>Pos. Error</span><span class="val" id="sPosErr">— m</span></div>
        <div class="stat warn"><span>Head. Error</span><span class="val" id="sHdErr">— rad</span></div>
      </div>
    </div>

  </div>
  <canvas id="canvas"></canvas>
</div>

<script>
const DT = 0.01;
let data = {}, currentTraj = null, step = 0, playing = false, rafId = null;

const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

function resize() {
  canvas.width  = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  if (currentTraj) render();
}
window.addEventListener('resize', resize);

// ── Load data ────────────────────────────────────────────────────────────────
fetch('/api/simulation')
  .then(r => r.json())
  .then(d => {
    data = d;
    document.getElementById('overlay').style.display = 'none';
    const sel = document.getElementById('trajSelect');
    const labels = { figure8: 'Figure-8', square: 'Square', spiral: 'Spiral', fan: 'Fan (multi)' };
    Object.keys(data).forEach(k => {
      const o = document.createElement('option');
      o.value = k; o.textContent = labels[k] || k;
      sel.appendChild(o);
    });
    currentTraj = Object.keys(data)[0];
    sel.value = currentTraj;
    resize();
  });

// ── Controls ──────────────────────────────────────────────────────────────────
document.getElementById('trajSelect').addEventListener('change', e => {
  currentTraj = e.target.value; step = 0; stop(); render();
});
document.getElementById('playBtn').addEventListener('click', togglePlay);
document.getElementById('resetBtn').addEventListener('click', () => { step = 0; stop(); render(); });
document.getElementById('speedSlider').addEventListener('input', e => {
  document.getElementById('speedLabel').textContent = e.target.value + '×';
});

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) loop();
}
function stop() {
  playing = false;
  document.getElementById('playBtn').textContent = '▶ Play';
  cancelAnimationFrame(rafId);
}
function loop() {
  if (!playing) return;
  const spd = parseInt(document.getElementById('speedSlider').value);
  const traj = data[currentTraj];
  const maxStep = traj.multi ? traj.gt_x[0].length - 1 : traj.gt_x.length - 1;
  step = Math.min(step + spd, maxStep);
  render();
  if (step < maxStep) { rafId = requestAnimationFrame(loop); }
  else { stop(); }
}

// ── Coordinate transform ──────────────────────────────────────────────────────
function bbox(traj) {
  let xs, ys;
  if (traj.multi) {
    xs = traj.gt_x.flat().concat(traj.arx_x.flat(), traj.kin_x.flat());
    ys = traj.gt_y.flat().concat(traj.arx_y.flat(), traj.kin_y.flat());
  } else {
    xs = traj.gt_x.concat(traj.arx_x, traj.kin_x);
    ys = traj.gt_y.concat(traj.arx_y, traj.kin_y);
  }
  return {
    minX: Math.min(...xs), maxX: Math.max(...xs),
    minY: Math.min(...ys), maxY: Math.max(...ys),
  };
}

function makeT(traj) {
  const pad = 56, { minX, maxX, minY, maxY } = bbox(traj);
  const rX = maxX - minX || 1, rY = maxY - minY || 1;
  const W = canvas.width - 2*pad, H = canvas.height - 2*pad;
  const scale = Math.min(W/rX, H/rY) * 0.9;
  return {
    scale,
    ox: pad + (W - rX*scale)/2 - minX*scale,
    oy: pad + (H - rY*scale)/2 + maxY*scale,
  };
}

function w2c(wx, wy, T) { return [wx*T.scale + T.ox, -wy*T.scale + T.oy]; }

// ── Drawing helpers ────────────────────────────────────────────────────────────
function drawPath(xs, ys, n, color, dash, T, alpha=1) {
  if (!xs || xs.length === 0) return;
  ctx.beginPath();
  ctx.strokeStyle = color; ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.6; ctx.setLineDash(dash || []);
  const lim = Math.min(n+1, xs.length);
  for (let i = 0; i < lim; i++) {
    const [cx,cy] = w2c(xs[i], ys[i], T);
    i === 0 ? ctx.moveTo(cx,cy) : ctx.lineTo(cx,cy);
  }
  ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}

function drawRobot(wx, wy, theta, color, T, glow=true) {
  const [cx,cy] = w2c(wx, wy, T);
  const s = Math.max(7, Math.min(16, T.scale * 0.18));
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-theta);  // canvas Y is flipped
  ctx.beginPath();
  ctx.moveTo( s*1.4,  0);
  ctx.lineTo(-s*0.8,  s*0.65);
  ctx.lineTo(-s*0.4,  0);
  ctx.lineTo(-s*0.8, -s*0.65);
  ctx.closePath();
  if (glow) { ctx.shadowColor = color; ctx.shadowBlur = 14; }
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function drawGrid(T, traj) {
  const { minX, maxX, minY, maxY } = bbox(traj);
  const span = Math.max(maxX-minX, maxY-minY, 1);
  const step = Math.pow(10, Math.floor(Math.log10(span/4)));
  ctx.strokeStyle = '#1e2030'; ctx.lineWidth = 0.7;
  ctx.fillStyle = '#45475a'; ctx.font = '9px monospace';

  for (let v = Math.ceil(minX/step)*step - step; v <= maxX+step; v += step) {
    const [cx] = w2c(v, 0, T);
    if (cx < 0 || cx > canvas.width) continue;
    ctx.beginPath(); ctx.moveTo(cx,0); ctx.lineTo(cx, canvas.height); ctx.stroke();
    ctx.fillText(v.toFixed(1)+'m', cx+2, canvas.height-5);
  }
  for (let v = Math.ceil(minY/step)*step - step; v <= maxY+step; v += step) {
    const [,cy] = w2c(0, v, T);
    if (cy < 0 || cy > canvas.height) continue;
    ctx.beginPath(); ctx.moveTo(0,cy); ctx.lineTo(canvas.width,cy); ctx.stroke();
    ctx.fillText(v.toFixed(1), 3, cy-3);
  }
  // axis lines
  ctx.strokeStyle = '#313244'; ctx.lineWidth = 1;
  const [ox,oy] = w2c(0,0,T);
  ctx.beginPath(); ctx.moveTo(ox,0); ctx.lineTo(ox,canvas.height); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,oy); ctx.lineTo(canvas.width,oy); ctx.stroke();
}

// ── Fan color scale ───────────────────────────────────────────────────────────
function fanColor(i, n) {
  const t = n > 1 ? i/(n-1) : 0.5;
  const r = Math.round(137 + (250-137)*t);
  const g = Math.round(220 - (220-113)*t);
  const b = Math.round(235 - (235-67)*t);
  return `rgb(${r},${g},${b})`;
}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {
  if (!currentTraj || !data[currentTraj]) return;
  const traj = data[currentTraj];
  const T = makeT(traj);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid(T, traj);

  if (traj.multi) {
    // Fan: render N sub-trajectories simultaneously
    const n = traj.gt_x.length;
    traj.gt_x.forEach((gx, i) => {
      const col = fanColor(i, n);
      // ghost full path
      drawPath(gx,            traj.gt_y[i],  gx.length,  col,                  [],     T, 0.12);
      drawPath(traj.arx_x[i], traj.arx_y[i], gx.length, col,                  [],     T, 0.08);
      // traveled
      drawPath(traj.kin_x[i], traj.kin_y[i], step, 'rgba(147,153,178,0.5)', [4,4], T);
      drawPath(gx,            traj.gt_y[i],  step, col,                      [],     T);
      drawPath(traj.arx_x[i], traj.arx_y[i], step, col,                    [],     T, 0.6);
      // robots
      const s = Math.min(step, gx.length-1);
      drawRobot(gx[s],            traj.gt_y[i][s],  traj.gt_theta[i][s],  col,                  T);
      drawRobot(traj.arx_x[i][s], traj.arx_y[i][s], traj.arx_theta[i][s], 'rgba(250,179,135,0.7)', T, false);
    });
    // stats: average pos error
    let totalErr = 0, count = 0;
    traj.pos_err.forEach(arr => { const s = Math.min(step, arr.length-1); totalErr += arr[s]; count++; });
    updateStats(step, totalErr / count, null);
  } else {
    // Single trajectory
    const N = traj.gt_x.length;
    // ghost
    drawPath(traj.gt_x,  traj.gt_y,  N, 'rgba(137,220,235,0.1)', [], T);
    drawPath(traj.arx_x, traj.arx_y, N, 'rgba(250,179,135,0.1)', [], T);
    drawPath(traj.kin_x, traj.kin_y, N, 'rgba(147,153,178,0.07)', [4,4], T);
    // traveled
    drawPath(traj.kin_x, traj.kin_y, step, 'rgba(147,153,178,0.55)', [4,4], T);
    drawPath(traj.gt_x,  traj.gt_y,  step, '#89dceb', [], T);
    drawPath(traj.arx_x, traj.arx_y, step, '#fab387', [], T);
    // start marker
    const [sx,sy] = w2c(traj.gt_x[0], traj.gt_y[0], T);
    ctx.beginPath(); ctx.arc(sx,sy,4,0,2*Math.PI);
    ctx.fillStyle='#a6e3a1'; ctx.fill();
    // robots
    const s = Math.min(step, N-1);
    drawRobot(traj.gt_x[s],  traj.gt_y[s],  traj.gt_theta[s],  '#89dceb', T);
    drawRobot(traj.arx_x[s], traj.arx_y[s], traj.arx_theta[s], '#fab387', T);
    // stats
    const pe = s < traj.pos_err.length ? traj.pos_err[s] : null;
    const he = s < traj.head_err.length ? traj.head_err[s] : null;
    updateStats(s, pe, he);
  }
}

function updateStats(s, posErr, headErr) {
  document.getElementById('sTime').textContent  = (s * DT).toFixed(2) + ' s';
  document.getElementById('sStep').textContent  = s;
  document.getElementById('sPosErr').textContent = posErr  != null ? posErr.toFixed(4)  + ' m'   : '—';
  document.getElementById('sHdErr').textContent  = headErr != null ? headErr.toFixed(4) + ' rad' : '—';
}
</script>
</body>
</html>"""

# ── Simulation config (fast variant, no order study) ─────────────────────────
_CFG = {
    'wheel_radius': 0.0975,
    'wheel_base': 0.381,
    'max_wheel_speed': 1.5,
    'motor_time_constant': 0.1,
    'dt': 0.01,
    'n_samples': 8000,
    'train_split': 0.8,
    'v_range': (-1.0, 1.0),
    'omega_range': (-1.5, 1.5),
    'min_hold_steps': 100,
    'max_hold_steps': 200,
    'rng_seed': 42,
    'na_diag': 9, 'na_cross': 3,
    'nb_diag': 9, 'nb_cross': 1,
    'ridge_alpha': 1e-8,
}


def _to_list(a, stride=2):
    """Downsample numpy array and convert to Python list."""
    return a[::stride].tolist()


def run_simulation() -> dict:
    """Train ARX model and compute all trajectory comparisons."""
    print("  [1/3] Creating robot and training ARX model…", flush=True)
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
    arx = MIMOARXModel(
        na_diag=_CFG['na_diag'], na_cross=_CFG['na_cross'],
        nb_diag=_CFG['nb_diag'], nb_cross=_CFG['nb_cross'],
        ridge_alpha=_CFG['ridge_alpha'],
    )
    arx.fit(dataset['train']['y'], dataset['train']['u'])
    metrics = arx.score(
        dataset['val']['y'],
        arx.predict(
            np.vstack([dataset['train']['y'][-arx.max_lag:], dataset['val']['y']]),
            np.vstack([dataset['train']['u'][-arx.max_lag:], dataset['val']['u']]),
            mode='one_step',
        )[arx.max_lag:],
    )
    print(f"       Fit%: v={metrics['fit_v']:.1f}%  ω={metrics['fit_omega']:.1f}%", flush=True)

    tgen = TrajectoryGenerator(arx, dt=_CFG['dt'])
    result = {}

    print("  [2/3] Generating named trajectories…", flush=True)
    for name in ['figure8', 'square', 'spiral']:
        print(f"        {name}", flush=True)
        v_cmds, omega_cmds = gen.generate_trajectory_commands(
            trajectory_type=name, n_samples=2000
        )
        robot.reset()
        comp = tgen.compare_with_ground_truth(robot, v_cmds, omega_cmds)
        gt, at, kt = comp['ground_truth'], comp['arx_model'], comp['kinematic']
        s = 2  # stride — 1000 points per trajectory is plenty
        result[name] = {
            'gt_x':   _to_list(gt['x'], s),  'gt_y':   _to_list(gt['y'], s),
            'gt_theta': _to_list(gt['theta'], s),
            'arx_x':  _to_list(at['x'], s),  'arx_y':  _to_list(at['y'], s),
            'arx_theta': _to_list(at['theta'], s),
            'kin_x':  _to_list(kt['x'], s),  'kin_y':  _to_list(kt['y'], s),
            'pos_err':  _to_list(comp['position_error'], s),
            'head_err': _to_list(comp['heading_error'],  s),
            'multi': False,
        }

    print("  [3/3] Generating fan trajectories…", flush=True)
    omega_finals = np.linspace(-2.0, 2.0, 9)
    fan: dict = {k: [] for k in
                 ('gt_x','gt_y','gt_theta','arx_x','arx_y','arx_theta',
                  'kin_x','kin_y','pos_err','head_err')}
    for of in omega_finals:
        v_c = np.ones(300) * 0.5
        w_c = np.linspace(0, of, 300)
        robot.reset()
        comp = tgen.compare_with_ground_truth(robot, v_c, w_c)
        gt, at, kt = comp['ground_truth'], comp['arx_model'], comp['kinematic']
        fan['gt_x'].append(gt['x'].tolist());   fan['gt_y'].append(gt['y'].tolist())
        fan['gt_theta'].append(gt['theta'].tolist())
        fan['arx_x'].append(at['x'].tolist());  fan['arx_y'].append(at['y'].tolist())
        fan['arx_theta'].append(at['theta'].tolist())
        fan['kin_x'].append(kt['x'].tolist());  fan['kin_y'].append(kt['y'].tolist())
        fan['pos_err'].append(comp['position_error'].tolist())
        fan['head_err'].append(comp['heading_error'].tolist())
    fan['multi'] = True
    result['fan'] = fan

    print("  Done.", flush=True)
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
    print("WMR Simulation Web Visualizer")
    print("=" * 55)
    _sim_data.update(run_simulation())
    print(f"\nOpen http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
