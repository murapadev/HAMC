'use strict';

const $ = (s) => document.querySelector(s);
const log = $('#log');

// ---------- Remote error logging --------------------------------------------

function reportToServer(level, msg, extras) {
  const body = Object.assign({ level, msg, url: location.href }, extras || {});
  try {
    if (navigator.sendBeacon) {
      const blob = new Blob([JSON.stringify(body)], { type: 'application/json' });
      navigator.sendBeacon('/api/client-log', blob);
      return;
    }
  } catch (_) {}
  fetch('/api/client-log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    keepalive: true,
  }).catch(() => {});
}
window.addEventListener('error', (e) => {
  reportToServer('error', e.message || 'window.onerror', {
    source: e.filename, line: e.lineno, col: e.colno,
    stack: e.error && e.error.stack ? e.error.stack : null,
  });
});
window.addEventListener('unhandledrejection', (e) => {
  const r = e.reason || {};
  reportToServer('error', 'unhandledrejection: ' + (r.message || String(r)), { stack: r.stack || null });
});
(function patchConsoleError() {
  const orig = console.error;
  console.error = function () {
    try {
      const msg = Array.from(arguments).map(a => {
        if (a instanceof Error) return a.message + (a.stack ? '\n' + a.stack : '');
        if (typeof a === 'object') { try { return JSON.stringify(a); } catch (_) { return String(a); } }
        return String(a);
      }).join(' ');
      reportToServer('error', '[console.error] ' + msg);
    } catch (_) {}
    return orig.apply(console, arguments);
  };
})();

// ---------- State -----------------------------------------------------------

const state = {
  source: null,
  palette: { regions: {}, blocks: {}, tiles: {} },
  params: null,
  totals: { global: 0, intermediate: 0, local: 0 },
  metrics: {
    global: { collapsed: 0, backtracks: 0, props: 0 },
    intermediate: { collapsed: 0, backtracks: 0, props: 0 },
    local: { blocks: 0, tiles: 0, backtracks: 0, props: 0 },
    events: 0, rendered: 0,
  },
  // grids replicated client-side (patches applied as events arrive)
  global: { grid: null, w: 0, h: 0 },
  intermediate: { grid: null, w: 0, h: 0 },
  local: { tilemap: null, w: 0, h: 0, blockSize: 0, currentBlock: null, currentTiles: null },
  // visual transient effects
  effects: { global: new Map(), intermediate: new Map(), local: new Map() }, // key -> {kind, t}
  // playback
  queue: [],
  paused: false, stepOnce: false,
  eventsPerSecond: 120,
  lastTick: 0,
  // sparklines
  spark: {
    events: { data: [], lastCount: 0, lastT: performance.now() },
    entropy: { data: [], sum: 0, n: 0 },
  },
  finals: { regions: null, blocks: null, tilemap: null, elapsed: null, timings: null },
};

const canvases = {
  global: $('#canvas-global'),
  intermediate: $('#canvas-intermediate'),
  local: $('#canvas-local'),
};
const tips = {
  global: $('#tip-global'),
  intermediate: $('#tip-intermediate'),
  local: $('#tip-local'),
};

// ---------- Status / banners ------------------------------------------------

function setStatus(text, cls) {
  const el = $('#status');
  el.className = 'status' + (cls ? ' ' + cls : '');
  $('#status-text').textContent = text;
}
function showBanner(msg) {
  const el = $('#banner');
  el.textContent = msg;
  el.hidden = !msg;
}
function clearChildren(el) { while (el.firstChild) el.removeChild(el.firstChild); }

// ---------- Colors ----------------------------------------------------------

function colorFor(level, value) {
  if (!value) return [22, 27, 35];
  const p = state.palette;
  if (level === 'global') return p.regions[value] || [255, 64, 64];
  if (level === 'intermediate') return p.blocks[value] || [255, 64, 64];
  return p.tiles[value] || [255, 64, 64];
}
function shadeForEntropy(n) {
  if (!n || n <= 0) return [16, 22, 30];
  const t = Math.min(1, Math.log2(n + 1) / 4);
  return [Math.round(18 + 28 * t), Math.round(28 + 70 * t), Math.round(46 + 130 * t)];
}

// ---------- Grid drawing ----------------------------------------------------

function cellSize(canvas, w, h) {
  return Math.max(8, Math.floor(Math.min(canvas.width / w, canvas.height / h)));
}

function drawGrid(level) {
  const canvas = canvases[level];
  const cells = state[level].grid;
  if (!cells || !cells.length) return;
  const ctx = canvas.getContext('2d');
  const h = cells.length, w = cells[0].length;
  const cellPx = cellSize(canvas, w, h);
  const drawW = cellPx * w, drawH = cellPx * h;
  if (canvas.width !== drawW || canvas.height !== drawH) {
    canvas.width = drawW; canvas.height = drawH;
  }
  ctx.fillStyle = '#07090d';
  ctx.fillRect(0, 0, drawW, drawH);

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const cell = cells[r][c];
      const collapsed = cell && cell.v;
      const rgb = collapsed ? colorFor(level, cell.v) : shadeForEntropy(cell ? cell.n : 0);
      ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
      ctx.fillRect(c * cellPx, r * cellPx, cellPx - 1, cellPx - 1);
      if (collapsed) {
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.strokeRect(c * cellPx + 0.5, r * cellPx + 0.5, cellPx - 1.5, cellPx - 1.5);
      }
    }
  }

  // overlay effects (ripples, propagation halos)
  drawEffects(ctx, level, cellPx);
}

function drawEffects(ctx, level, cellPx) {
  for (const [key, eff] of state.effects[level]) {
    const [r, c] = key.split(',').map(Number);
    const x = c * cellPx, y = r * cellPx;
    if (eff.kind === 'collapse') {
      // expanding ring
      const radius = cellPx * (0.5 + (1 - eff.t) * 2.0);
      ctx.strokeStyle = 'rgba(103, 211, 255, ' + (eff.t * 0.9) + ')';
      ctx.lineWidth = Math.max(1.5, cellPx * 0.1 * eff.t);
      ctx.beginPath();
      ctx.arc(x + cellPx / 2, y + cellPx / 2, radius, 0, Math.PI * 2);
      ctx.stroke();
    } else if (eff.kind === 'prop') {
      // soft glow over propagated cells
      ctx.fillStyle = 'rgba(199, 148, 255, ' + (eff.t * 0.35) + ')';
      ctx.fillRect(x, y, cellPx - 1, cellPx - 1);
    } else if (eff.kind === 'backtrack') {
      ctx.fillStyle = 'rgba(255, 186, 110, ' + (eff.t * 0.4) + ')';
      ctx.fillRect(x, y, cellPx - 1, cellPx - 1);
    }
  }
}

function drawTilemap() {
  const ctx = canvases.local.getContext('2d');
  const { tilemap, w, h, blockSize, currentBlock, currentTiles } = state.local;
  if (!w || !h) return;
  const cellPx = Math.max(4, Math.floor(Math.min(canvases.local.width / w, canvases.local.height / h)));
  const drawW = cellPx * w, drawH = cellPx * h;
  if (canvases.local.width !== drawW || canvases.local.height !== drawH) {
    canvases.local.width = drawW; canvases.local.height = drawH;
  }
  ctx.fillStyle = '#07090d';
  ctx.fillRect(0, 0, drawW, drawH);

  if (tilemap) {
    for (let r = 0; r < h; r++) {
      for (let c = 0; c < w; c++) {
        const v = tilemap[r][c];
        if (!v) continue;
        const rgb = colorFor('local', v);
        ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
        ctx.fillRect(c * cellPx, r * cellPx, cellPx, cellPx);
      }
    }
  }

  if (currentBlock && currentTiles) {
    const { br, bc } = currentBlock;
    const baseR = br * blockSize;
    const baseC = bc * blockSize;
    for (let r = 0; r < currentTiles.length; r++) {
      for (let c = 0; c < currentTiles[0].length; c++) {
        const cell = currentTiles[r][c];
        const rgb = cell && cell.v ? colorFor('local', cell.v) : shadeForEntropy(cell ? cell.n : 0);
        ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
        ctx.fillRect((baseC + c) * cellPx, (baseR + r) * cellPx, cellPx, cellPx);
      }
    }
    // active block outline
    ctx.strokeStyle = 'rgba(255, 186, 110, 0.9)';
    ctx.lineWidth = 2;
    ctx.strokeRect(
      bc * blockSize * cellPx + 1,
      br * blockSize * cellPx + 1,
      blockSize * cellPx - 2,
      blockSize * cellPx - 2,
    );
  }

  if (blockSize > 0 && w > 0) {
    ctx.strokeStyle = 'rgba(190,210,230,0.10)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= w; x += blockSize) {
      ctx.beginPath(); ctx.moveTo(x * cellPx + 0.5, 0); ctx.lineTo(x * cellPx + 0.5, drawH); ctx.stroke();
    }
    for (let y = 0; y <= h; y += blockSize) {
      ctx.beginPath(); ctx.moveTo(0, y * cellPx + 0.5); ctx.lineTo(drawW, y * cellPx + 0.5); ctx.stroke();
    }
  }

  // local effects (collapse ring)
  drawEffects(ctx, 'local', cellPx);
}

// ---------- Effects ---------------------------------------------------------

function addEffect(level, r, c, kind, life) {
  state.effects[level].set(r + ',' + c, { kind, t: 1, life: life || 1 });
}

function tickEffects(level) {
  const m = state.effects[level];
  let any = false;
  for (const [k, e] of m) {
    e.t -= 0.06 / (e.life || 1);
    if (e.t <= 0) m.delete(k);
    else any = true;
  }
  return any;
}

// ---------- Sparklines ------------------------------------------------------

const SPARK_LEN = 60;
function pushSpark(name, v) {
  const s = state.spark[name];
  s.data.push(v);
  if (s.data.length > SPARK_LEN) s.data.shift();
}
function drawSpark(name, elId, scale) {
  const el = $(elId);
  if (!el) return;
  const poly = el.querySelector('polyline');
  if (!poly) return;
  const data = state.spark[name].data;
  if (!data.length) { poly.setAttribute('points', ''); return; }
  const W = 120, H = 30;
  const maxV = Math.max(0.0001, scale === 'auto' ? Math.max(...data) : scale);
  const step = W / Math.max(1, SPARK_LEN - 1);
  const pts = data.map((v, i) => {
    const x = (i + (SPARK_LEN - data.length)) * step;
    const y = H - (Math.min(1, v / maxV) * (H - 2)) - 1;
    return x.toFixed(1) + ',' + y.toFixed(1);
  }).join(' ');
  poly.setAttribute('points', pts);
}

// ---------- Log -------------------------------------------------------------

function appendLog(event) {
  const li = document.createElement('li');
  li.className = 't-' + event.type;
  const t = event.t != null ? event.t.toFixed(2) + 's' : '-';
  let detail = '';
  if (event.type === 'collapse') {
    detail = '[' + event.pos + '] → ' + event.value
      + (event.propagated && event.propagated.length ? ' · ' + event.propagated.length + ' prop' : '');
  } else if (event.type === 'backtrack')         detail = '↶ at [' + event.pos + ']';
  else if (event.type === 'conflict')            detail = '✗ at [' + event.pos + ']';
  else if (event.type === 'local_block_start')   detail = 'block (' + event.br + ',' + event.bc + ') ' + event.block_type;
  else if (event.type === 'local_block_complete')detail = 'block (' + event.br + ',' + event.bc + ') ✓';
  else if (event.type === 'complete')            detail = 'layer complete';
  else if (event.type === 'run_complete')        detail = 'done in ' + event.elapsed.toFixed(2) + 's';
  else if (event.type === 'run_error')           detail = 'error: ' + event.error;
  else detail = event.type;
  li.textContent = t + '  ' + (event.level || event.type) + '  ' + detail;
  log.prepend(li);
  while (log.childElementCount > 150) log.removeChild(log.lastChild);
}

// ---------- Metrics ---------------------------------------------------------

function updateMetrics() {
  const m = state.metrics, t = state.totals;
  $('#m-global').textContent       = m.global.collapsed       + ' / ' + t.global       + ' · ' + m.global.backtracks       + ' ↶';
  $('#m-intermediate').textContent = m.intermediate.collapsed + ' / ' + t.intermediate + ' · ' + m.intermediate.backtracks + ' ↶';
  $('#m-local').textContent        = m.local.blocks           + ' / ' + t.local        + ' blocks · ' + m.local.tiles + ' tiles';
  $('#m-events').textContent       = m.events.toString();
  $('#rendered').textContent       = m.rendered.toString();
  $('#queue-size').textContent     = state.queue.length.toString();
}

function updateProgress() {
  const m = state.metrics, t = state.totals;
  if (t.global)       $('#prog-global').value       = Math.min(1, m.global.collapsed       / t.global);
  if (t.intermediate) $('#prog-intermediate').value = Math.min(1, m.intermediate.collapsed / t.intermediate);
  if (t.local)        $('#prog-local').value        = Math.min(1, m.local.blocks           / t.local);
}

// ---------- Legends ---------------------------------------------------------

function legendFor(level) {
  const target = $('#legend-' + level);
  clearChildren(target);
  const dict = level === 'global' ? state.palette.regions
            : level === 'intermediate' ? state.palette.blocks
            : state.palette.tiles;
  const used = new Set();
  if (level === 'local') {
    (state.local.tilemap || []).forEach(row => row.forEach(v => v && used.add(v)));
    if (state.local.currentTiles) state.local.currentTiles.forEach(row => row.forEach(c => c && c.v && used.add(c.v)));
  } else {
    const g = state[level].grid;
    if (g) g.forEach(row => row.forEach(c => c && c.v && used.add(c.v)));
  }
  Object.entries(dict).forEach(([name, rgb]) => {
    if (used.size && !used.has(name)) return;
    const span = document.createElement('span');
    const sw = document.createElement('i');
    sw.className = 'sw';
    sw.style.background = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    span.appendChild(sw);
    span.appendChild(document.createTextNode(name));
    target.appendChild(span);
  });
}

// ---------- Event handling --------------------------------------------------

function ensureGrid(level, w, h) {
  const grid = Array.from({ length: h }, () =>
    Array.from({ length: w }, () => ({ v: null, n: 0 })),
  );
  state[level].grid = grid;
  state[level].w = w;
  state[level].h = h;
}

function applyPatch(level, patch) {
  // patch = [r, c, n] or [r, c, 0, value]
  const [r, c] = patch;
  const grid = state[level].grid;
  if (!grid || !grid[r]) return;
  if (patch.length >= 4) {
    grid[r][c] = { v: patch[3], n: 1 };
  } else {
    grid[r][c].n = patch[2];
  }
}

function applyFullGrid(level, full) {
  // Server sends grid as List[List[{v,n}]]
  state[level].grid = full;
  state[level].h = full.length;
  state[level].w = full[0].length;
}

function renderEvent(event) {
  appendLog(event);
  state.metrics.rendered++;

  switch (event.type) {
    case 'run_start': {
      state.palette = event.palette;
      state.params = event.params;
      const p = event.params;
      state.totals = {
        global: p.width * p.height,
        intermediate: p.width * p.subgrid * p.height * p.subgrid,
        local: p.width * p.subgrid * p.height * p.subgrid,
      };
      Object.assign(state.metrics, {
        global: { collapsed: 0, backtracks: 0, props: 0 },
        intermediate: { collapsed: 0, backtracks: 0, props: 0 },
        local: { blocks: 0, tiles: 0, backtracks: 0, props: 0 },
      });
      state.global = { grid: null, w: 0, h: 0 };
      state.intermediate = { grid: null, w: 0, h: 0 };
      state.local = { tilemap: null, w: 0, h: 0, blockSize: 0, currentBlock: null, currentTiles: null };
      for (const k of Object.keys(state.effects)) state.effects[k].clear();
      state.spark.events.data = [];
      state.spark.entropy.data = [];
      state.spark.entropy.sum = 0;
      state.spark.entropy.n = 0;
      [canvases.global, canvases.intermediate, canvases.local].forEach(c => {
        c.getContext('2d').clearRect(0, 0, c.width, c.height);
      });
      ['global', 'intermediate', 'local'].forEach(k => { $('#prog-' + k).value = 0; });
      ['regions', 'blocks', 'tilemap', 'png'].forEach(k => {
        document.querySelector('.downloads [data-kind="' + k + '"]').disabled = true;
      });
      $('#run-summary').textContent =
        p.width + '×' + p.height + ' · subgrid ' + p.subgrid + ' · local ' + p.local;
      setStatus('running', 'running');
      $('#global-tag').textContent = p.width + '×' + p.height;
      $('#intermediate-tag').textContent = (p.width * p.subgrid) + '×' + (p.height * p.subgrid);
      $('#local-tag').textContent = (p.width * p.subgrid * p.local) + '×' + (p.height * p.subgrid * p.local);
      break;
    }

    case 'init':
      if (event.level === 'global' || event.level === 'intermediate') {
        applyFullGrid(event.level, event.grid);
        drawGrid(event.level);
      }
      break;

    case 'collapse':
      if (event.level === 'global' || event.level === 'intermediate') {
        const level = event.level;
        const [r, c] = event.pos;
        // collapse cell
        if (state[level].grid) state[level].grid[r][c] = { v: event.value, n: 1 };
        // apply propagation patches
        if (Array.isArray(event.propagated)) {
          for (const p of event.propagated) applyPatch(level, p);
          state.metrics[level].props += event.propagated.length;
          // halos for propagated cells
          for (const p of event.propagated) addEffect(level, p[0], p[1], 'prop', 1.2);
        }
        state.metrics[level].collapsed++;
        addEffect(level, r, c, 'collapse', 1.0);
        // entropy sparkline
        if (event.entropy != null && isFinite(event.entropy)) {
          state.spark.entropy.sum += event.entropy;
          state.spark.entropy.n++;
          pushSpark('entropy', state.spark.entropy.sum / state.spark.entropy.n);
        }
        drawGrid(level);
        legendFor(level);
      } else if (event.level === 'local') {
        state.metrics.local.tiles++;
        const ct = state.local.currentTiles;
        if (ct) {
          const [r, c] = event.pos;
          if (ct[r] && ct[r][c] !== undefined) ct[r][c] = { v: event.value, n: 1 };
          if (Array.isArray(event.propagated)) {
            for (const p of event.propagated) {
              const [pr, pc] = p;
              if (!ct[pr] || ct[pr][pc] === undefined) continue;
              if (p.length >= 4) ct[pr][pc] = { v: p[3], n: 1 };
              else ct[pr][pc].n = p[2];
            }
          }
          drawTilemap();
        }
      }
      break;

    case 'backtrack':
      if (event.level === 'global' || event.level === 'intermediate') {
        applyFullGrid(event.level, event.grid);
        state.metrics[event.level].backtracks++;
        const [r, c] = event.pos;
        if (r != null && c != null) addEffect(event.level, r, c, 'backtrack', 1.5);
        drawGrid(event.level);
      } else {
        state.metrics.local.backtracks++;
      }
      break;

    case 'complete':
      if (event.level === 'global' || event.level === 'intermediate') {
        applyFullGrid(event.level, event.grid);
        drawGrid(event.level);
        legendFor(event.level);
      }
      break;

    case 'local_init':
      state.local = {
        tilemap: Array.from({ length: event.height }, () => new Array(event.width).fill(null)),
        w: event.width, h: event.height,
        blockSize: event.block_size,
        currentBlock: null, currentTiles: null,
      };
      drawTilemap();
      break;

    case 'local_block_start':
      state.local.currentBlock = { br: event.br, bc: event.bc, type: event.block_type };
      state.local.currentTiles = Array.from({ length: event.size }, () =>
        Array.from({ length: event.size }, () => ({ v: null, n: 0 })),
      );
      drawTilemap();
      break;

    case 'local_block_complete': {
      const { br, bc, tiles } = event;
      const base = br * state.local.blockSize;
      const baseC = bc * state.local.blockSize;
      for (let r = 0; r < tiles.length; r++) {
        for (let c = 0; c < tiles[0].length; c++) {
          state.local.tilemap[base + r][baseC + c] = tiles[r][c];
        }
      }
      state.local.currentBlock = null;
      state.local.currentTiles = null;
      state.metrics.local.blocks++;
      drawTilemap();
      legendFor('local');
      break;
    }

    case 'run_complete':
      state.finals = {
        regions: event.regions, blocks: event.blocks,
        tilemap: event.tilemap, elapsed: event.elapsed, timings: event.timings,
      };
      ['regions', 'blocks', 'tilemap', 'png'].forEach(k => {
        document.querySelector('.downloads [data-kind="' + k + '"]').disabled = false;
      });
      setStatus('done · ' + event.elapsed.toFixed(2) + 's', 'done');
      if (event.timings) {
        $('#run-summary').textContent =
          'G ' + event.timings.global.toFixed(2) + 's · ' +
          'I ' + event.timings.intermediate.toFixed(2) + 's · ' +
          'L ' + event.timings.local.toFixed(2) + 's';
      }
      break;

    case 'run_error':
      setStatus('error', 'error');
      showBanner('Server error: ' + event.error);
      reportToServer('error', 'run_error: ' + event.error);
      break;

    case 'fail':
      reportToServer('warning',
        'generator fail level=' + event.level + ' reason=' + event.reason + ' generator=' + event.generator);
      break;
  }
  if (event.t != null) $('#m-elapsed').innerHTML = event.t.toFixed(2) + '<small>s</small>';
  updateMetrics();
  updateProgress();
}

// ---------- Main loop -------------------------------------------------------

function pump(now) {
  const eps = state.eventsPerSecond;
  const interval = 1000 / Math.max(1, eps);

  if (!state.paused || state.stepOnce) {
    const maxThisFrame = Math.max(1, Math.ceil((now - state.lastTick) / interval));
    let drained = 0;
    while (state.queue.length && drained < maxThisFrame) {
      renderEvent(state.queue.shift());
      drained++;
      if (state.stepOnce) {
        state.stepOnce = false;
        state.paused = true;
        document.getElementById('pause-icon').innerHTML =
          '<polygon points="3,2 14,8 3,14" fill="currentColor"/>';
        break;
      }
    }
    if (drained > 0) state.lastTick = now;
  }

  // Tick visual effects + redraw if needed
  if (tickEffects('global'))       drawGrid('global');
  if (tickEffects('intermediate')) drawGrid('intermediate');
  if (tickEffects('local'))        drawTilemap();

  // Events/sec sparkline (1 Hz sample)
  const evS = state.spark.events;
  if (now - evS.lastT >= 1000) {
    const delta = state.metrics.events - evS.lastCount;
    pushSpark('events', delta);
    evS.lastCount = state.metrics.events;
    evS.lastT = now;
  }
  drawSpark('events',  '#spark-events',  'auto');
  drawSpark('entropy', '#spark-entropy', 'auto');

  updateMetrics();
  requestAnimationFrame(pump);
}
requestAnimationFrame((t) => { state.lastTick = t; state.spark.events.lastT = t; pump(t); });

// ---------- Networking ------------------------------------------------------

function enqueue(event) { state.metrics.events++; state.queue.push(event); }

async function probeServer() {
  try { const r = await fetch('/api/health'); return r.ok; }
  catch (_) { return false; }
}

async function startStream(params) {
  stopStream();
  showBanner('');
  const reachable = await probeServer();
  if (!reachable) {
    showBanner("Can't reach /api/health — is the dashboard server running on this origin?");
    setStatus('error', 'error');
    return;
  }
  const qs = new URLSearchParams(params).toString();
  const url = '/api/stream?' + qs;
  state.source = new EventSource(url);
  state.source.onmessage = (ev) => {
    try { enqueue(JSON.parse(ev.data)); }
    catch (e) { console.error('bad event', e); }
  };
  state.source.onerror = () => {
    if (state.source && state.source.readyState === EventSource.CLOSED) {
      setStatus('disconnected', 'error');
      const m = 'SSE connection closed unexpectedly (' + url + ')';
      showBanner(m); reportToServer('error', m);
      stopStream();
    }
  };
  $('#run').disabled = true;
  $('#stop').disabled = false;
  $('#pause').disabled = false;
  $('#step').disabled = false;
}

function stopStream() {
  if (state.source) { state.source.close(); state.source = null; }
  $('#run').disabled = false;
  $('#stop').disabled = true;
}

// ---------- Downloads -------------------------------------------------------

function downloadJSON(name, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  triggerDownload(name, blob);
}
function downloadCanvasPng(name, canvas) {
  canvas.toBlob((blob) => { if (blob) triggerDownload(name, blob); }, 'image/png');
}
function triggerDownload(name, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 100);
}

// ---------- Tooltip ---------------------------------------------------------

function bindTooltip(level) {
  const canvas = canvases[level];
  const tip = tips[level];
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
    if (level === 'local') {
      const { w, h, tilemap, blockSize, currentBlock, currentTiles } = state.local;
      if (!w || !h) { tip.hidden = true; return; }
      const cellPx = Math.max(4, Math.floor(Math.min(canvas.width / w, canvas.height / h)));
      const c = Math.floor(x / cellPx), r = Math.floor(y / cellPx);
      if (r < 0 || c < 0 || r >= h || c >= w) { tip.hidden = true; return; }
      let v = tilemap && tilemap[r][c];
      if (!v && currentBlock && currentTiles) {
        const lr = r - currentBlock.br * blockSize, lc = c - currentBlock.bc * blockSize;
        if (lr >= 0 && lc >= 0 && lr < currentTiles.length && lc < currentTiles[0].length) {
          v = currentTiles[lr][lc] && currentTiles[lr][lc].v;
        }
      }
      tip.textContent = v ? v + '  (' + r + ',' + c + ')' : '(' + r + ',' + c + ')';
    } else {
      const grid = state[level].grid;
      if (!grid || !grid.length) { tip.hidden = true; return; }
      const w = grid[0].length, h = grid.length;
      const cellPx = cellSize(canvas, w, h);
      const c = Math.floor(x / cellPx), r = Math.floor(y / cellPx);
      if (r < 0 || c < 0 || r >= h || c >= w) { tip.hidden = true; return; }
      const cell = grid[r][c];
      const label = cell && cell.v
        ? cell.v + '  (' + r + ',' + c + ')'
        : 'options: ' + (cell ? cell.n : 0) + '  (' + r + ',' + c + ')';
      tip.textContent = label;
    }
    tip.hidden = false;
    tip.style.left = (e.clientX - rect.left) + 'px';
    tip.style.top  = (e.clientY - rect.top) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tip.hidden = true; });
}
['global', 'intermediate', 'local'].forEach(bindTooltip);

// ---------- Wiring ----------------------------------------------------------

$('#params').addEventListener('submit', (e) => {
  e.preventDefault();
  const form = new FormData(e.target);
  const params = {};
  for (const [k, v] of form.entries()) if (v !== '') params[k] = v;
  state.metrics.events = 0;
  state.metrics.rendered = 0;
  state.queue = [];
  state.paused = false; state.stepOnce = false;
  document.getElementById('pause-icon').innerHTML =
    '<rect x="3" y="2" width="3" height="12" fill="currentColor"/><rect x="10" y="2" width="3" height="12" fill="currentColor"/>';
  clearChildren(log);
  startStream(params);
});

$('#stop').addEventListener('click', () => { stopStream(); setStatus('stopped', 'error'); });

$('#pause').addEventListener('click', () => {
  state.paused = !state.paused;
  const ic = document.getElementById('pause-icon');
  if (state.paused) {
    ic.innerHTML = '<polygon points="3,2 14,8 3,14" fill="currentColor"/>';
  } else {
    ic.innerHTML = '<rect x="3" y="2" width="3" height="12" fill="currentColor"/><rect x="10" y="2" width="3" height="12" fill="currentColor"/>';
  }
});

$('#step').addEventListener('click', () => { state.stepOnce = true; state.paused = false; });

$('#speed').addEventListener('input', (e) => {
  state.eventsPerSecond = parseInt(e.target.value, 10) || 1;
  $('#speed-label').textContent = state.eventsPerSecond + ' ev/s';
});

$('#clear-log').addEventListener('click', () => clearChildren(log));

document.querySelectorAll('.downloads button').forEach(btn => {
  btn.addEventListener('click', () => {
    const k = btn.dataset.kind;
    if (k === 'regions' && state.finals.regions) downloadJSON('regions.json', state.finals.regions);
    if (k === 'blocks'  && state.finals.blocks)  downloadJSON('blocks.json',  state.finals.blocks);
    if (k === 'tilemap' && state.finals.tilemap) downloadJSON('tilemap.json', state.finals.tilemap);
    if (k === 'png')                             downloadCanvasPng('tilemap.png', canvases.local);
  });
});

$('#fullscreen').addEventListener('click', () => {
  document.querySelector('.layer[data-layer="local"]').classList.toggle('fullscreen');
});
