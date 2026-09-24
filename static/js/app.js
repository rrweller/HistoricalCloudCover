/* ═══════════════════════════════════════════════════════════════════════
   Cloud Cover Explorer — front end.

   Two loops:
     • settings  → /api/estimate (debounced) → /api/run → poll /api/job/<id>
     • results   → /api/job/<id>/view whenever the *measure* changes; purely
                   cosmetic changes (colour, banding, opacity) repaint locally
                   from the float grid already in memory.
   ═══════════════════════════════════════════════════════════════════════ */

'use strict';

/* ─────────────────────────── colour ramps ─────────────────────────── */
/* Sequential only — magnitude reads as one hue getting darker. The first is
   the design-system blue; the rest are the perceptually-uniform standards
   people expect from scientific maps. No rainbow ramps. */

const RAMP_HEX = {
  blue: ['#eaf2fd', '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
         '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'],
  viridis: ['#440154', '#482878', '#3e4989', '#31688e', '#26828e', '#1f9e89',
            '#35b779', '#6ece58', '#b5de2b', '#fde725'].slice().reverse(),
  magma: ['#000004', '#180f3d', '#440f76', '#721f81', '#9e2f7f', '#cd4071',
          '#f1605d', '#fd9668', '#feca8d', '#fcfdbf'].slice().reverse(),
  ylgnbu: ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0',
           '#225ea8', '#253494', '#081d58'],
  greys: ['#ffffff', '#f0f0f0', '#d9d9d9', '#bdbdbd', '#969696', '#737373',
          '#525252', '#252525', '#0a0a0a'],
};

const RAMPS = {};
for (const [name, stops] of Object.entries(RAMP_HEX)) {
  RAMPS[name] = stops.map((hex) => [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ]);
}

function sampleRamp(ramp, t) {
  const x = Math.min(1, Math.max(0, t)) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const f = x - i;
  const a = ramp[i], b = ramp[i + 1];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

/* ───────────────── periods the time slider scrubs through ───────────────── */
/* Each stop is a climatology: every matching month pooled across the whole
   year range, not one particular autumn. Slot 0 is always the full year. */

const ALL_MONTHS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const SEASON_SLOTS = [
  { label: 'All year', short: 'Year', months: ALL_MONTHS },
  { label: 'Winter (Dec-Feb)', short: 'Win', months: [12, 1, 2] },
  { label: 'Spring (Mar-May)', short: 'Spr', months: [3, 4, 5] },
  { label: 'Summer (Jun-Aug)', short: 'Sum', months: [6, 7, 8] },
  { label: 'Autumn (Sep-Nov)', short: 'Aut', months: [9, 10, 11] },
];
const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December'];
const MONTH_SLOTS = [SEASON_SLOTS[0]].concat(MONTH_NAMES.map((name, i) => ({
  label: name, short: name.slice(0, 3), months: [i + 1],
})));

const slots = () => (state.granularity === 'month' ? MONTH_SLOTS : SEASON_SLOTS);
const currentSlot = () => slots()[Math.min(state.slot, slots().length - 1)];

const state = {
  boot: null,
  nightMode: 'astronomical',
  granularity: 'season',
  slot: 0,
  jobId: null,
  polling: null,
  pollFailures: 0,
  running: false,
  view: null,        // last /view response
  grid: null,        // { data: Float32Array, width, height, bounds }
  layers: { field: null, points: null },
};

const $ = (id) => document.getElementById(id);
const els = {};
['lat_min', 'lat_max', 'lon_min', 'lon_max', 'grid_resolution', 'res-value',
 'res-hint', 'compass-size', 'start_date', 'end_date', 'workers', 'workers-value', 'method', 'resolution', 'detail-value',
 'est-points', 'est-cached', 'est-fetch', 'est-time', 'est-note', 'estimate',
 'progress', 'progress-phase', 'progress-pct', 'progress-bar', 'progress-count',
 'progress-eta', 'alert', 'btn-run', 'btn-cancel', 'btn-draw', 'map-hint', 'viewbar',
 'legend', 'legend-title', 'legend-canvas', 'legend-min', 'legend-max',
 'night-hint', 'timebar', 'timebar-label', 'timebar-ticks', 'timeslider', 'map-note',
 'colormap', 'bands', 'opacity', 'opacity-value', 'show-scale',
 'show-points', 'btn-png', 'run-meta'].forEach((id) => { els[id] = $(id); });

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const fmtInt = (n) => Number(n).toLocaleString();
const fmtVal = (v, unit) => (v == null || !isFinite(v) ? '—' : `${v.toFixed(1)}${unit || ''}`);

function coords(lat, lon) {
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'}  ${Math.abs(lon).toFixed(2)}°${lon >= 0 ? 'E' : 'W'}`;
}

function fmtDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return '';
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function showAlert(message, kind = 'error') {
  els.alert.hidden = false;
  els.alert.className = `alert alert--${kind}`;
  els.alert.textContent = message;
}
function clearAlert() { els.alert.hidden = true; }

/* ───────────────────────────── the map ────────────────────────────── */

const map = L.map('map', {
  zoomControl: true,
  worldCopyJump: false,
  minZoom: 2,
  attributionControl: true,
}).setView([50, 10], 4);

map.createPane('field').style.zIndex = 350;
const labelPane = map.createPane('labels');
labelPane.style.zIndex = 365;
labelPane.style.pointerEvents = 'none';
map.createPane('dots').style.zIndex = 375;

// Basemap, all keyless Esri services, in four layers.
//
// Hillshade alone is pure terrain — no coastlines, borders, roads or towns —
// so the dark canvas rides on top of it at partial opacity to put those back
// while the relief still shows through. Roads and labels go in the 'labels'
// pane so they stay readable over the cloud-cover overlay rather than being
// buried by it.
const ESRI = 'https://services.arcgisonline.com/ArcGIS/rest/services';
const esriLayer = (path, opts) => L.tileLayer(
  `${ESRI}/${path}/MapServer/tile/{z}/{y}/{x}`, { maxZoom: 14, ...opts },
);

esriLayer('Elevation/World_Hillshade_Dark', {
  attribution: '&copy; Esri, Airbus DS, USGS, NGA, NASA, HERE, Garmin &middot; data: Open-Meteo ERA5',
}).addTo(map);
esriLayer('Canvas/World_Dark_Gray_Base', { opacity: 0.55 }).addTo(map);

esriLayer('Reference/World_Transportation', { pane: 'labels' }).addTo(map);
esriLayer('Reference/World_Boundaries_and_Places', { pane: 'labels' }).addTo(map);

/* ──────────────────── interactive selection box ───────────────────── */

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

class AreaSelector {
  constructor(map, onChange) {
    this.map = map;
    this.onChange = onChange;
    this.bounds = { latMin: 35, latMax: 70, lonMin: -10, lonMax: 40 };
    this.drawing = false;
    this.activeHandle = null;

    this.rect = L.rectangle(this._latLngBounds(), {
      color: '#3987e5', weight: 2, opacity: 0.95,
      fillColor: '#3987e5', fillOpacity: 0.07,
      renderer: L.svg(), interactive: true,
    }).addTo(map);

    this.handles = {};
    for (const key of HANDLES) {
      const marker = L.marker([0, 0], {
        icon: L.divIcon({ className: '', html: `<div class="sel-handle sel-handle--${key}"></div>`, iconSize: [11, 11], iconAnchor: [5.5, 5.5] }),
        draggable: true,
        keyboard: false,
        zIndexOffset: 700,
      }).addTo(map);

      marker.on('dragstart', () => { this.activeHandle = key; });
      marker.on('drag', () => { this._fromHandle(key, marker.getLatLng()); });
      marker.on('dragend', () => { this.activeHandle = null; this._sync(); this._emit(); });
      this.handles[key] = marker;
    }

    this._bindBodyDrag();
    this._bindDrawMode();
    this._sync();
  }

  /* — geometry — */

  _latLngBounds() {
    const b = this.bounds;
    return L.latLngBounds([b.latMin, b.lonMin], [b.latMax, b.lonMax]);
  }

  _normalise(b) {
    let { latMin, latMax, lonMin, lonMax } = b;
    if (latMin > latMax) [latMin, latMax] = [latMax, latMin];
    if (lonMin > lonMax) [lonMin, lonMax] = [lonMax, lonMin];
    latMin = clamp(latMin, -85, 85); latMax = clamp(latMax, -85, 85);
    lonMin = clamp(lonMin, -180, 180); lonMax = clamp(lonMax, -180, 180);
    // Never let the box collapse to nothing — the grid needs some extent.
    if (latMax - latMin < 0.25) latMax = Math.min(85, latMin + 0.25);
    if (lonMax - lonMin < 0.25) lonMax = Math.min(180, lonMin + 0.25);
    return { latMin, latMax, lonMin, lonMax };
  }

  _fromHandle(key, latlng) {
    const b = { ...this.bounds };
    if (key.includes('n')) b.latMax = latlng.lat;
    if (key.includes('s')) b.latMin = latlng.lat;
    if (key.includes('e')) b.lonMax = latlng.lng;
    if (key.includes('w')) b.lonMin = latlng.lng;
    this.bounds = this._normalise(b);
    this._sync();
    this._emit(true);
  }

  _sync() {
    this.rect.setBounds(this._latLngBounds());
    const b = this.bounds;
    const midLat = (b.latMin + b.latMax) / 2;
    const midLon = (b.lonMin + b.lonMax) / 2;
    const at = {
      nw: [b.latMax, b.lonMin], n: [b.latMax, midLon], ne: [b.latMax, b.lonMax],
      e: [midLat, b.lonMax], se: [b.latMin, b.lonMax], s: [b.latMin, midLon],
      sw: [b.latMin, b.lonMin], w: [midLat, b.lonMin],
    };
    for (const key of HANDLES) {
      if (key === this.activeHandle) continue;
      this.handles[key].setLatLng(at[key]);
    }
  }

  _emit(live = false) {
    if (this.onChange) this.onChange(this.bounds, live);
  }

  /* — dragging the whole box — */

  _bindBodyDrag() {
    let origin = null;
    let startBounds = null;

    this.rect.on('mousedown', (e) => {
      if (this.drawing) return;
      origin = e.latlng;
      startBounds = { ...this.bounds };
      this.map.dragging.disable();
      L.DomEvent.stop(e);
    });

    this.map.on('mousemove', (e) => {
      if (!origin) return;
      const dLat = e.latlng.lat - origin.lat;
      const dLon = e.latlng.lng - origin.lng;
      this.bounds = this._normalise({
        latMin: startBounds.latMin + dLat, latMax: startBounds.latMax + dLat,
        lonMin: startBounds.lonMin + dLon, lonMax: startBounds.lonMax + dLon,
      });
      this._sync();
      this._emit(true);
    });

    this.map.on('mouseup', () => {
      if (!origin) return;
      origin = null;
      this.map.dragging.enable();
      this._emit();
    });
  }

  /* — drawing a fresh box — */

  _bindDrawMode() {
    let origin = null;

    this.map.on('mousedown', (e) => {
      if (!this.drawing) return;
      origin = e.latlng;
      L.DomEvent.stop(e);
    });

    this.map.on('mousemove', (e) => {
      if (!this.drawing || !origin) return;
      this.bounds = this._normalise({
        latMin: Math.min(origin.lat, e.latlng.lat), latMax: Math.max(origin.lat, e.latlng.lat),
        lonMin: Math.min(origin.lng, e.latlng.lng), lonMax: Math.max(origin.lng, e.latlng.lng),
      });
      this._sync();
      this._emit(true);
    });

    this.map.on('mouseup', () => {
      if (!this.drawing || !origin) return;
      origin = null;
      this.setDrawing(false);
      this._emit();
    });
  }

  setDrawing(on) {
    this.drawing = on;
    document.querySelector('.stage').classList.toggle('is-drawing', on);
    els['btn-draw'].classList.toggle('is-active', on);
    els['btn-draw'].textContent = on ? 'Cancel drawing' : 'Draw on map';
    for (const key of HANDLES) {
      const el = this.handles[key].getElement();
      if (el) el.style.display = on ? 'none' : '';
    }
    if (on) this.map.dragging.disable(); else this.map.dragging.enable();
  }

  set(bounds, { fit = false, silent = false } = {}) {
    this.bounds = this._normalise(bounds);
    this._sync();
    if (fit) this.map.fitBounds(this._latLngBounds(), { padding: [40, 40], animate: true });
    if (!silent) this._emit();
  }
}

const selector = new AreaSelector(map, (bounds, live) => {
  els.lat_min.value = bounds.latMin.toFixed(2);
  els.lat_max.value = bounds.latMax.toFixed(2);
  els.lon_min.value = bounds.lonMin.toFixed(2);
  els.lon_max.value = bounds.lonMax.toFixed(2);
  updateAreaReadout();
  if (!live) scheduleEstimate();
});

/* ───────────────────────── settings <-> UI ────────────────────────── */

function readSettings() {
  return {
    lat_min: parseFloat(els.lat_min.value),
    lat_max: parseFloat(els.lat_max.value),
    lon_min: parseFloat(els.lon_min.value),
    lon_max: parseFloat(els.lon_max.value),
    grid_resolution: parseInt(els.grid_resolution.value, 10),
    start_date: els.start_date.value,
    end_date: els.end_date.value,
    night_mode: state.nightMode,
    workers: parseInt(els.workers.value, 10),
  };
}

function applySettings(s) {
  els.lat_min.value = Number(s.lat_min).toFixed(2);
  els.lat_max.value = Number(s.lat_max).toFixed(2);
  els.lon_min.value = Number(s.lon_min).toFixed(2);
  els.lon_max.value = Number(s.lon_max).toFixed(2);
  els.grid_resolution.value = s.grid_resolution;
  els.start_date.value = s.start_date;
  els.end_date.value = s.end_date;
  els.workers.value = s.workers;

  setNightMode(s.night_mode || 'astronomical', { silent: true });

  selector.set({ latMin: s.lat_min, latMax: s.lat_max, lonMin: s.lon_min, lonMax: s.lon_max },
    { fit: true, silent: true });
  syncDerivedLabels();
}

function setNightMode(mode, { silent = false } = {}) {
  state.nightMode = mode;
  document.querySelectorAll('#night-mode .seg').forEach((btn) => {
    btn.classList.toggle('is-active', btn.dataset.mode === mode);
  });
  const meta = (state.boot?.night_modes || []).find((m) => m.key === mode);
  els['night-hint'].textContent = meta ? meta.help : '';
  if (!silent) scheduleEstimate();
}

function syncDerivedLabels() {
  const res = parseInt(els.grid_resolution.value, 10);
  els['res-value'].textContent = `${res} × ${res}`;
  els['res-hint'].textContent = `${fmtInt(res * res)} sample points across the area.`;
  els['workers-value'].textContent = els.workers.value;
  els['detail-value'].textContent = `${els.resolution.value} px`;

  updateAreaReadout();
}

function updateAreaReadout() {
  const latSpan = parseFloat(els.lat_max.value) - parseFloat(els.lat_min.value);
  const lonSpan = parseFloat(els.lon_max.value) - parseFloat(els.lon_min.value);
  if (!isFinite(latSpan) || !isFinite(lonSpan)) { els['compass-size'].textContent = '—'; return; }
  els['compass-size'].innerHTML = `${latSpan.toFixed(1)}°<br>×<br>${lonSpan.toFixed(1)}°`;
}

/* ───────────────────────────── estimate ───────────────────────────── */

let estimateTimer = null;
let estimateToken = 0;

function scheduleEstimate() {
  clearTimeout(estimateTimer);
  estimateTimer = setTimeout(runEstimate, 400);
}

async function runEstimate() {
  if (state.running) return;
  const token = ++estimateToken;
  const settings = readSettings();
  try {
    const res = await fetch('/api/estimate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings }),
    });
    const data = await res.json();
    if (token !== estimateToken) return;
    if (!res.ok) { showAlert(data.error || 'Those settings are not valid.'); els['btn-run'].disabled = true; return; }

    clearAlert();
    els['btn-run'].disabled = false;
    els['res-hint'].textContent =
      `${fmtInt(data.points)} sample points, ${fmtInt(data.points_inside)} of them inside your area `
      + `(the rest form the buffer ring).`;
    els['est-points'].textContent = fmtInt(data.points);
    els['est-cached'].textContent = fmtInt(data.ready + data.unverified);
    els['est-fetch'].textContent = data.to_fetch ? `${fmtInt(data.to_fetch)} points` : 'nothing';

    els['est-time'].textContent = data.to_fetch ? `~${fmtDuration(data.eta_seconds)}` : 'seconds';

    const notes = [];
    if (data.to_fetch) {
      notes.push(`Up to ${fmtInt(data.max_requests)} archive requests across ${data.years} year${data.years === 1 ? '' : 's'}.`);
      notes.push('Each point is written to disk as it finishes, so an interrupted run resumes where it stopped.');
    } else {
      notes.push('Everything is on disk — this run will finish in seconds.');
    }
    if (data.unverified) {
      notes.push(`${fmtInt(data.unverified)} points have a cache file that has not been indexed yet; the run will confirm their coverage.`);
    }
    els['est-note'].textContent = notes.join(' ');

    if (data.over_daily_limit) {
      showAlert(
        `This needs up to ${fmtInt(data.max_requests)} requests, over Open-Meteo's free `
        + `allowance of ${fmtInt(data.daily_limit)} per day. It will stall partway. Shrink the grid, `
        + `narrow the date range, or run it across several days — progress is kept between runs.`,
        'warn',
      );
    }
  } catch (err) {
    if (token === estimateToken) showAlert(`Could not reach the server: ${err.message}`);
  }
}

/* ─────────────────────────── running a job ────────────────────────── */

async function startRun() {
  clearAlert();
  const settings = readSettings();
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ settings }),
    });
    const data = await res.json();
    if (!res.ok) { showAlert(data.error || 'Could not start the run.'); return; }

    state.jobId = data.id;
    state.pollFailures = 0;
    setRunning(true);
    pollJob();
  } catch (err) {
    showAlert(`Could not start the run: ${err.message}`);
  }
}

function setRunning(on) {
  state.running = on;
  els['btn-run'].hidden = on;
  els['btn-cancel'].hidden = !on;
  els.progress.hidden = !on;
  els.estimate.hidden = on;
  document.querySelectorAll('#tab-setup input, #tab-setup select, #tab-setup button')
    .forEach((el) => { el.disabled = on; });
  // Re-enabling en masse would undo conditional disables (the hour selects when
  // "whole day" is on), so let the derived state reassert itself.
  if (!on) syncDerivedLabels();
}

function pollJob() {
  clearInterval(state.polling);
  state.polling = setInterval(checkJob, 700);
  checkJob();
}

async function checkJob() {
  if (!state.jobId) return;
  let snap;
  try {
    const res = await fetch(`/api/job/${state.jobId}`);
    snap = await res.json();
    if (!res.ok) throw new Error(snap.error || 'lost the job');
  } catch (err) {
    // A dropped poll usually means the server is busy, not gone. Only give up
    // after several in a row, and say what to do about it when we do.
    state.pollFailures += 1;
    if (state.pollFailures < 5) {
      els['progress-phase'].textContent = `Server not responding, retrying (${state.pollFailures}/5)…`;
      return;
    }
    clearInterval(state.polling);
    setRunning(false);
    showAlert(
      `The server stopped responding, which means the app itself exited — most likely a crash. `
      + `A log for this run was written to .cache/logs/run-${state.jobId}.log, and any crash `
      + `dump to .cache/logs/fault-*.log. Restart with "python app.py" and run it again: every `
      + `point that finished is already cached, so it picks up where it stopped.`,
    );
    return;
  }
  state.pollFailures = 0;

  const pct = snap.total ? Math.round((snap.done / snap.total) * 100) : 0;
  els['progress-phase'].textContent = snap.phase;
  els['progress-pct'].textContent = `${pct}%`;
  els['progress-bar'].style.width = `${pct}%`;
  els['progress-count'].textContent = `${fmtInt(snap.done)} / ${fmtInt(snap.total)} points`;
  els['progress-eta'].textContent = snap.eta != null
    ? `about ${fmtDuration(snap.eta)} left`
    : fmtDuration(snap.elapsed);
  if (snap.restarts) {
    els['progress-count'].textContent +=
      ` · recovered from ${snap.restarts} worker crash${snap.restarts === 1 ? '' : 'es'}`;
  }

  if (snap.status === 'running' || snap.status === 'queued') return;

  clearInterval(state.polling);
  setRunning(false);

  if (snap.status === 'cancelled') { showAlert('Run cancelled.', 'warn'); return; }
  if (snap.status === 'error') { showAlert(snap.error || 'The run failed.'); return; }

  if (snap.failure_count) {
    showAlert(`${snap.failure_count} point${snap.failure_count === 1 ? '' : 's'} could not be fetched and were left out. ${snap.failures[0] || ''}`, 'warn');
  } else {
    clearAlert();
  }

  document.querySelector('.tab[data-tab="results"]').disabled = false;
  switchTab('results');
  await refreshView();
  loadRecent();
}

async function cancelRun() {
  if (!state.jobId) return;
  await fetch(`/api/job/${state.jobId}/cancel`, { method: 'POST' });
}

/* ──────────────────────────── the results ─────────────────────────── */

function viewParams() {
  return {
    metric: 'mean',
    label: currentSlot().label,
    months: currentSlot().months,
    resolution: parseInt(els.resolution.value, 10),
    method: els.method.value,
  };
}

async function refreshView() {
  if (!state.jobId) return;
  try {
    const res = await fetch(`/api/job/${state.jobId}/view`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(viewParams()),
    });
    const data = await res.json();
    if (!res.ok) { showAlert(data.error || 'Could not build the view.'); return; }

    state.view = data;
    state.grid = {
      data: decodeGrid(data.grid.data),
      width: data.grid.width,
      height: data.grid.height,
      bounds: data.grid.bounds,
    };

    els.viewbar.hidden = false;
    els.legend.hidden = false;
    els.timebar.hidden = false;
    renderTimebar();

    if (data.empty) {
      // Legitimate at high latitude: above ~48.5°N the sun never reaches 18°
      // below the horizon around midsummer, so such a period has no data.
      if (state.layers.field) { map.removeLayer(state.layers.field); state.layers.field = null; }
      paintPoints();
      els.legend.hidden = true;
      els['map-note'].hidden = false;
      const period = currentSlot().label.toLowerCase();
      els['map-note'].textContent = state.nightMode === 'astronomical'
        ? `No astronomical night anywhere in this area during ${period} — the sun never`
          + ' drops 18° below the horizon. Try "Sun below the horizon" instead, which'
          + ' still yields data through a high-latitude summer.'
        : `The sun never sets anywhere in this area during ${period}, so there are no`
          + ' night hours to average. Try another period, or the full 24 hours.';
    } else {
      els['map-note'].hidden = true;
      paintOverlay();
      paintPoints();
    }
    renderMeta();
    els['map-hint'].classList.add('is-hidden');
  } catch (err) {
    showAlert(`Could not build the view: ${err.message}`);
  }
}

function decodeGrid(b64) {
  if (!b64) return null;
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function scaleRange() {
  const v = state.view;
  if (els['show-scale'].checked || !v || v.data_min == null) return [0, 100];
  let lo = Math.floor(v.data_min);
  let hi = Math.ceil(v.data_max);
  if (hi - lo < 1) hi = lo + 1;
  return [lo, hi];
}

function paintOverlay() {
  if (!state.grid || !state.grid.data) return;
  const { data, width, height, bounds } = state.grid;
  const ramp = RAMPS[els.colormap.value] || RAMPS.blue;
  const bands = parseInt(els.bands.value, 10);
  const [lo, hi] = scaleRange();
  const span = Math.max(1e-9, hi - lo);

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(width, height);
  const px = img.data;

  for (let i = 0; i < width * height; i += 1) {
    const v = data[i];
    const o = i * 4;
    if (!isFinite(v)) { px[o + 3] = 0; continue; }
    let t = (v - lo) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    if (bands > 0) t = (Math.min(bands - 1, Math.floor(t * bands)) + 0.5) / bands;
    const c = sampleRamp(ramp, t);
    px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  const opacity = parseInt(els.opacity.value, 10) / 100;
  const latLng = [[bounds[0], bounds[2]], [bounds[1], bounds[3]]];

  if (state.layers.field) map.removeLayer(state.layers.field);
  state.layers.field = L.imageOverlay(canvas.toDataURL(), latLng, {
    opacity, pane: 'field', interactive: false,
  }).addTo(map);

  renderLegend();
}

function renderLegend() {
  const ramp = RAMPS[els.colormap.value] || RAMPS.blue;
  const bands = parseInt(els.bands.value, 10);
  const canvas = els['legend-canvas'];
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);

  for (let x = 0; x < width; x += 1) {
    let t = x / (width - 1);
    if (bands > 0) t = (Math.min(bands - 1, Math.floor(t * bands)) + 0.5) / bands;
    const c = sampleRamp(ramp, t);
    ctx.fillStyle = `rgb(${c[0] | 0},${c[1] | 0},${c[2] | 0})`;
    ctx.fillRect(x, 0, 1, height);
  }

  const [lo, hi] = scaleRange();
  const v = state.view;
  const unit = v ? v.metric.unit : '%';
  els['legend-min'].textContent = `${lo}${unit}`;
  els['legend-max'].textContent = `${hi}${unit}`;
  els['legend-title'].textContent = v ? legendTitle(v) : 'Cloud cover';
}

function legendTitle(v) {
  return v.metric.label;
}

function paintPoints() {
  if (state.layers.points) { map.removeLayer(state.layers.points); state.layers.points = null; }
  if (!els['show-points'].checked || !state.view) return;

  // A canvas renderer detaches itself from the map when its last layer goes, so
  // reusing one across toggles silently draws nothing. Build a fresh one each
  // time the layer is shown.
  const renderer = L.canvas({ pane: 'dots' });
  const group = L.layerGroup();
  for (const [lat, lon] of state.view.points) {
    L.circleMarker([lat, lon], {
      radius: 2.6,
      color: '#0b0b0b',        // dark ring keeps them legible on pale fills
      weight: 1,
      opacity: 0.85,
      fillColor: '#ffffff',
      fillOpacity: 1,
      renderer,
      interactive: false,
    }).addTo(group);
  }
  group.addTo(map);
  state.layers.points = group;
}

function renderTimebar() {
  const list = slots();
  state.slot = Math.min(state.slot, list.length - 1);
  els.timeslider.max = String(list.length - 1);
  els.timeslider.value = String(state.slot);
  els['timebar-label'].textContent = list[state.slot].label;
  els['timebar-ticks'].innerHTML = list
    .map((slot, i) => `<span class="${i === state.slot ? 'is-current' : ''}">${slot.short}</span>`)
    .join('');
}

function renderMeta() {
  const v = state.view;
  els['run-meta'].textContent =
    `${v.date_range[0]} to ${v.date_range[1]} · night ${v.night_label} · `
    + `${fmtInt(v.nights_per_point)} nights per point · ${fmtInt(v.points.length)} points sampled.`;
}

/* ───────────────────────────── exports ────────────────────────────── */

async function downloadPng() {
  if (!state.jobId) return;
  const button = els['btn-png'];
  const original = button.textContent;
  button.disabled = true;
  button.textContent = 'Preparing…';
  try {
    const [lo, hi] = scaleRange();
    const res = await fetch(`/api/job/${state.jobId}/export.png`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...viewParams(),
        colormap: els.colormap.value,
        bands: parseInt(els.bands.value, 10) || 10,
        vmin: lo, vmax: hi,
        show_points: els['show-points'].checked,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || res.statusText);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cloudcover_${currentSlot().label.toLowerCase().replace(/[^a-z0-9]+/g, "_")}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (err) {
    showAlert(`Export failed: ${err.message}`);
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

/* ────────────────────────────── wiring ────────────────────────────── */

function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('is-active', t.dataset.tab === name));
  document.querySelectorAll('.tab-panel').forEach((p) => p.classList.toggle('is-active', p.id === `tab-${name}`));
}

function wire() {
  // Tabs and collapsibles
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => { if (!tab.disabled) switchTab(tab.dataset.tab); });
  });
  document.querySelectorAll('[data-toggle]').forEach((head) => {
    head.addEventListener('click', () => $(head.dataset.toggle).classList.toggle('card--collapsed'));
  });

  // Area
  els['btn-draw'].addEventListener('click', () => selector.setDrawing(!selector.drawing));
  ['lat_min', 'lat_max', 'lon_min', 'lon_max'].forEach((id) => {
    els[id].addEventListener('change', () => {
      selector.set({
        latMin: parseFloat(els.lat_min.value), latMax: parseFloat(els.lat_max.value),
        lonMin: parseFloat(els.lon_min.value), lonMax: parseFloat(els.lon_max.value),
      }, { silent: true });
      updateAreaReadout();
      scheduleEstimate();
    });
  });
  els.grid_resolution.addEventListener('input', syncDerivedLabels);
  els.grid_resolution.addEventListener('change', scheduleEstimate);

  // Dates
  els.start_date.addEventListener('change', scheduleEstimate);
  els.end_date.addEventListener('change', scheduleEstimate);
  // Hours counted
  document.querySelectorAll('#night-mode .seg').forEach((btn) => {
    btn.addEventListener('click', () => setNightMode(btn.dataset.mode));
  });

  // Advanced
  els.workers.addEventListener('input', syncDerivedLabels);
  els.method.addEventListener('change', () => { if (state.view) refreshView(); });
  els.resolution.addEventListener('input', syncDerivedLabels);
  els.resolution.addEventListener('change', () => { if (state.view) refreshView(); });

  // Run
  els['btn-run'].addEventListener('click', startRun);
  els['btn-cancel'].addEventListener('click', cancelRun);

  // Time of year — scrubbing is cheap (the server re-aggregates from memory),
  // but debounce so a drag does not queue a request per pixel.
  let slideTimer = null;
  els.timeslider.addEventListener('input', () => {
    state.slot = parseInt(els.timeslider.value, 10);
    renderTimebar();
    clearTimeout(slideTimer);
    slideTimer = setTimeout(refreshView, 180);
  });
  document.querySelectorAll('#granularity .seg').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (state.granularity === btn.dataset.gran) return;
      // Keep the window you were looking at: stepping seasons <-> months maps
      // through the first month of the current selection.
      const month = currentSlot().months[0];
      const wasAllYear = state.slot === 0;
      state.granularity = btn.dataset.gran;
      document.querySelectorAll('#granularity .seg')
        .forEach((b) => b.classList.toggle('is-active', b === btn));
      state.slot = wasAllYear ? 0 : slots().findIndex((sl) => sl.months.includes(month));
      if (state.slot < 0) state.slot = 0;
      renderTimebar();
      refreshView();
    });
  });

  // Presentation — local repaint only
  els.colormap.addEventListener('change', paintOverlay);
  els.bands.addEventListener('change', paintOverlay);
  els.opacity.addEventListener('input', () => {
    els['opacity-value'].textContent = `${els.opacity.value}%`;
    if (state.layers.field) state.layers.field.setOpacity(parseInt(els.opacity.value, 10) / 100);
  });
  els['show-scale'].addEventListener('change', paintOverlay);
  els['show-points'].addEventListener('change', paintPoints);

  els['btn-png'].addEventListener('click', downloadPng);

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && selector.drawing) selector.setDrawing(false);
  });
}

/* ────────────────────────────── startup ───────────────────────────── */

async function boot() {
  const res = await fetch('/api/bootstrap');
  const data = await res.json();
  state.boot = data;

  wire();
  applySettings(data.defaults);
  renderTimebar();
  els['opacity-value'].textContent = `${els.opacity.value}%`;
  runEstimate();
}

boot().catch((err) => showAlert(`Startup failed: ${err.message}`));
