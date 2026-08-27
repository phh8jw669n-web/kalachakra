// main.js — Kalachakra v7 client orchestrator.
//
// Renders the globe by MAPPING a worker-baked regional field texture (globe.js) — a trivial
// texture lookup that holds 60 fps regardless of zoom or window size. The heavy SIREN field
// is computed off the main thread (fieldworker.js), and only when the Julian Date changes.
// The Temporal Helm, fluid scrubber and Observer HUD query the same reused version6 ephemeris
// + SIREN, preserving exact physical/geometric accuracy while staying lag-free.

import { createGlobe } from "./globe.js";
import { makeFieldBaker } from "./field.js";
import { topocentricTensor, BODY_NAMES, N_BODIES } from "../../version6/web/ephemeris6.js";
import { makeSiren, boundLab, labToSrgb, srgbToHex } from "../../version6/web/siren6.js";
import {
  J2000, nowJD, gregorianToJD, jdToGregorian, dtLocalValue, labelFor,
  tzLabel, TZ_OFFSETS,
} from "./timecal.js";

const RES = { "120x60": [120, 60], "180x90": [180, 90], "256x128": [256, 128] };
const DEFAULT_ARCH = { in_features: 33, hidden: 48, hidden_layers: 2, out_features: 3, omega0: 30,
  lab_center: 50, lab_lspan: 50, lab_ab: 90, output_activation: "lab_tanh" };
let JD_MIN = J2000 - 5000 * 365.25, JD_MAX = J2000 + 5000 * 365.25;

const $ = (id) => document.getElementById(id);
const setNotice = (t) => { $("notice").textContent = t || ""; };

const app = {
  mode: "live", jd: nowJD(), playing: false, speed: 1, stepHours: 24,
  opacity: 0.72, tzMode: "local", tzOffsetMin: 0,
  gridW: 180, gridH: 90,
  pin: { lat: 48.8566, lon: 2.3522 },
  weights: null, manifest: null, cities: [], siren: null,
};

const globe = createGlobe($("canvas-wrap"));
globe.setOpacity(app.opacity);

// ---- field worker (with synchronous fallback) ------------------------------
let worker = null, fieldBusy = false, lastBakedJd = NaN, wantJd = null;
let spareBuf = null, syncBaker = null;
let fieldStamps = [];

function startField() {
  try {
    worker = new Worker(new URL("./fieldworker.js", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") { $("backend").textContent = "engine · worker"; requestField(); return; }
      if (m.type === "field") {
        globe.setField(new Uint8Array(m.buf), m.w, m.h);
        spareBuf = m.buf; fieldBusy = false; lastBakedJd = m.jd;
        markFieldRate();
        requestField();
      }
    };
    worker.onerror = () => { worker = null; useSync(); };
    worker.postMessage({ type: "init", weights: app.weights });
  } catch {
    useSync();
  }
}
function useSync() {
  syncBaker = makeFieldBaker(app.weights);
  $("backend").textContent = "engine · main (no worker)";
  requestField();
}
function requestField() {
  if (wantJd === null || wantJd === lastBakedJd) return;
  const jd = wantJd, w = app.gridW, h = app.gridH;
  if (worker) {
    if (fieldBusy) return;
    fieldBusy = true;
    const buf = spareBuf && spareBuf.byteLength === w * h * 4 ? spareBuf : new ArrayBuffer(w * h * 4);
    spareBuf = null;
    worker.postMessage({ type: "bake", jd, w, h, buf }, [buf]);
  } else if (syncBaker) {
    const rgba = syncBaker.bake(jd, w, h);
    globe.setField(rgba, w, h); lastBakedJd = jd; markFieldRate();
  }
}
function markFieldRate() {
  const now = performance.now();
  fieldStamps.push(now);
  while (fieldStamps.length && now - fieldStamps[0] > 1000) fieldStamps.shift();
  $("field-rate").textContent = `field ${fieldStamps.length}/s · ${app.gridW}×${app.gridH}`;
}

// ---- time helpers ----------------------------------------------------------
function tzOffsetMin() { return app.tzMode === "local" ? -new Date().getTimezoneOffset() : app.tzOffsetMin; }
function tzSuffix(off) { return app.tzMode === "local" ? "Local" : tzLabel(off); }
function jumpTo(jd) { app.jd = Math.min(JD_MAX, Math.max(JD_MIN, jd)); wantJd = app.jd; }

// ---- render loop -----------------------------------------------------------
let last = performance.now(), fpsFrames = 0, fpsT = last;
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = (now - last) / 1000; last = now;

  if (app.mode === "live") { app.jd = nowJD(); wantJd = app.jd; }
  else if (app.playing) {
    app.jd = Math.min(JD_MAX, Math.max(JD_MIN, app.jd + app.speed * dt / 86400.0));
    wantJd = app.jd;
  }
  requestField();
  globe.render();
  updateReadouts(now);

  fpsFrames++;
  if (now - fpsT >= 500) { $("fps").textContent = `${Math.round(fpsFrames * 1000 / (now - fpsT))} fps`; fpsFrames = 0; fpsT = now; }
}

// ---- HUD / readouts (throttled) -------------------------------------------
const el = {}; const matRows = [];
let lastHud = 0;
function buildHud() {
  for (const id of ["clock", "date", "matrix", "lab", "hex", "swatch", "pin-lat", "pin-lon", "pin-city", "scrub-label"]) el[id] = $(id);
  el.matrix.textContent = "";
  for (let b = 0; b < N_BODIES; b++) {
    const row = document.createElement("div");
    const name = document.createElement("span"); name.className = "body"; name.textContent = BODY_NAMES[b].padEnd(8);
    const vals = document.createElement("span"); row.append(name, vals); el.matrix.appendChild(row); matRows.push(vals);
  }
}
function fmt(x) { return (x >= 0 ? " " : "-") + Math.abs(x).toFixed(3); }
function updateReadouts(now) {
  const off = tzOffsetMin();
  const lab = labelFor(app.jd, off, tzSuffix(off));
  el.clock.textContent = lab.clock; el.date.textContent = lab.date;
  if (app.mode === "machine") { $("scrub").value = String(app.jd); el["scrub-label"].textContent = `${lab.date} · ${lab.clock}`; }

  if (now - lastHud < 80) return;   // ~12 Hz HUD
  lastHud = now;
  if (!app.siren) return;
  const sky = topocentricTensor(app.pin.lat, app.pin.lon, app.jd);
  for (let b = 0; b < N_BODIES; b++)
    matRows[b].textContent = ` ${fmt(sky[b * 3])} ${fmt(sky[b * 3 + 1])} ${fmt(sky[b * 3 + 2])}`;
  const lab3 = boundLab(app.weights, app.siren(sky));
  const hex = srgbToHex(labToSrgb(lab3[0], lab3[1], lab3[2]));
  el.lab.textContent = `L* ${lab3[0].toFixed(1)}  a* ${lab3[1].toFixed(1)}  b* ${lab3[2].toFixed(1)}`;
  el.hex.textContent = hex; el.swatch.style.background = hex;
  el["pin-lat"].textContent = app.pin.lat.toFixed(4); el["pin-lon"].textContent = app.pin.lon.toFixed(4);
  const c = globe.nearestCity(app.pin.lat, app.pin.lon);
  el["pin-city"].textContent = c ? `${c.name} (${c.region})` : "—";
}

// ---- weights / cities load -------------------------------------------------
async function loadJSON(path) { return (await fetch(path)).json(); }
async function boot() {
  buildHud();
  try {
    app.weights = await loadJSON("weights.json");
  } catch {
    setNotice("no weights.json — showing an UNTRAINED field (train + export to see the learned atlas)");
    app.weights = randomWeights(DEFAULT_ARCH);
  }
  try { app.manifest = await loadJSON("manifest.json"); } catch { app.manifest = null; }
  if (app.manifest && app.manifest.timeline) { JD_MIN = app.manifest.timeline.jd_start; JD_MAX = app.manifest.timeline.jd_end; }
  if (app.manifest && app.manifest.grid) { app.gridW = app.manifest.grid.width; app.gridH = app.manifest.grid.height; syncResSelect(); }
  try { app.cities = await loadJSON("cities.json"); } catch { app.cities = []; }
  globe.setCities(app.cities);
  app.siren = makeSiren(app.weights);

  wantJd = app.jd;
  startField();
  wireControls();
  globe.setReticle(app.pin.lat, app.pin.lon);
  $("scrub").min = String(JD_MIN); $("scrub").max = String(JD_MAX); $("scrub").value = String(app.jd);
  $("dt").value = dtLocalValue(app.jd);
  animate();
  const b = $("boot"); b.style.opacity = "0"; setTimeout(() => b.remove(), 700);
}

function randomWeights(a) {
  const U = (lo, hi) => lo + Math.random() * (hi - lo);
  const layer = (inF, outF, first, act) => {
    const bound = first ? 1 / inF : Math.sqrt(6 / inF) / a.omega0;
    const W = Array.from({ length: outF }, () => Array.from({ length: inF }, () => U(-bound, bound)));
    const b = Array.from({ length: outF }, () => U(-bound, bound));
    return { W, b, activation: act };
  };
  const layers = [layer(a.in_features, a.hidden, true, "sin")];
  for (let k = 1; k < a.hidden_layers; k++) layers.push(layer(a.hidden, a.hidden, false, "sin"));
  layers.push(layer(a.hidden, a.out_features, false, "linear"));
  return { ...a, layers, color_scale: 20 };
}
function syncResSelect() {
  const key = `${app.gridW}x${app.gridH}`;
  const sel = $("res"); if ([...sel.options].some((o) => o.value === key)) sel.value = key;
}

// ---- controls --------------------------------------------------------------
function speedFromSlider(v) { return v >= 0 ? Math.pow(10, v) : -Math.pow(10, -v); }
function fmtSpeed(m) {
  const a = Math.abs(m); const t = a >= 1e6 ? (a / 1e6).toFixed(0) + "M" : a >= 1000 ? (a / 1000).toFixed(0) + "k" : a.toFixed(a < 10 ? 1 : 0);
  return (m < 0 ? "-" : "") + t + "×";
}
function setMode(machine) {
  app.mode = machine ? "machine" : "live";
  $("mode-badge").textContent = machine ? "TIME MACHINE" : "LIVE";
  $("mode-badge").classList.toggle("machine", machine);
  $("btn-mode").textContent = machine ? "Go Live" : "Enter Time Machine";
  $("btn-play").disabled = !machine;
  $("scrub").disabled = !machine;
  if (!machine) { app.playing = false; $("btn-play").classList.remove("on"); $("btn-play").textContent = "⏸ Pause"; }
  else { app.jd = app.jd; $("dt").value = dtLocalValue(app.jd); }
}
function stepOnce(dir) { setMode(true); jumpTo(app.jd + dir * app.stepHours / 24.0); $("dt").value = dtLocalValue(app.jd); }
function zoom(factor) {
  const c = globe.camera, ctl = globe.controls, t = ctl.target;
  const off = c.position.clone().sub(t);
  const dist = Math.max(ctl.minDistance, Math.min(ctl.maxDistance, off.length() * factor));
  c.position.copy(t).add(off.setLength(dist));
}

function wireControls() {
  // timezone select
  const tz = $("tz");
  const opt = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; tz.appendChild(o); };
  opt("local", "Local");
  for (const off of TZ_OFFSETS) opt(String(off), off === 0 ? "UTC" : tzLabel(off));
  tz.onchange = () => { if (tz.value === "local") app.tzMode = "local"; else { app.tzMode = "fixed"; app.tzOffsetMin = parseInt(tz.value, 10); } };

  $("btn-mode").onclick = () => setMode(app.mode !== "machine");
  const play = $("btn-play");
  play.onclick = () => { if (app.mode !== "machine") return; app.playing = !app.playing; play.classList.toggle("on", app.playing); play.textContent = app.playing ? "▶ Playing" : "⏸ Pause"; };
  $("btn-live").onclick = () => { setMode(false); app.jd = nowJD(); wantJd = app.jd; };

  const sp = $("speed");
  sp.oninput = () => { app.speed = speedFromSlider(parseFloat(sp.value)); $("speed-val").textContent = fmtSpeed(app.speed); };

  document.querySelectorAll("#steps button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); app.stepHours = parseFloat(b.dataset.h); $("step-input").value = app.stepHours;
  });
  $("step-input").oninput = (e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) app.stepHours = v; document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel")); };
  $("btn-back").onclick = () => stepOnce(-1);
  $("btn-fwd").onclick = () => stepOnce(1);

  const scrub = $("scrub");
  scrub.oninput = () => { setMode(true); jumpTo(parseFloat(scrub.value)); $("dt").value = dtLocalValue(app.jd); };
  $("go").onclick = () => {
    const v = $("dt").value, m = v && v.match(/^(-?\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/);
    if (!m) { setNotice("enter a date/time"); return; }
    setNotice(""); setMode(true); jumpTo(gregorianToJD(+m[1], +m[2], +m[3], +m[4], +m[5], +(m[6] || 0)));
  };

  $("zoom-in").onclick = () => zoom(0.82);
  $("zoom-out").onclick = () => zoom(1.22);

  const op = $("opacity");
  op.oninput = () => { app.opacity = parseFloat(op.value); globe.setOpacity(app.opacity); $("opacity-val").textContent = app.opacity.toFixed(2); };

  $("res").onchange = (e) => { const [w, h] = RES[e.target.value]; app.gridW = w; app.gridH = h; lastBakedJd = NaN; wantJd = app.jd; requestField(); };
  $("markers").onchange = (e) => globe.setCitiesVisible(e.target.checked);
  $("graticule").onchange = (e) => globe.setGraticule(e.target.checked);

  // panel hide/show + drag
  $("panel-hide").onclick = () => { $("panel").classList.add("hidden"); $("panel-show").style.display = "block"; };
  $("panel-show").onclick = () => { $("panel").classList.remove("hidden"); $("panel-show").style.display = "none"; };
  dragPanel($("panel"), $("panel-header"));
  $("hud-toggle").onclick = () => { const h = $("hud"); h.classList.toggle("collapsed"); $("hud-toggle").textContent = h.classList.contains("collapsed") ? "▸" : "▾"; };

  // globe picking -> pin observer (or nearest city)
  let dragging = false;
  globe.renderer.domElement.addEventListener("pointerdown", (e) => { const p = globe.pick(e); if (p) { dragging = true; setPin(p); } });
  globe.renderer.domElement.addEventListener("pointermove", (e) => { if (dragging) { const p = globe.pick(e); if (p) setPin(p); } });
  window.addEventListener("pointerup", () => { dragging = false; });

  window.addEventListener("resize", () => globe.resize());
  window.addEventListener("keydown", onKey);
}
function setPin(p) { app.pin.lat = p.lat; app.pin.lon = p.lon; globe.setReticle(p.lat, p.lon); }
function onKey(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); if (app.mode !== "machine") setMode(true); $("btn-play").click(); }
  else if (e.key === "l" || e.key === "L") $("btn-live").click();
  else if (e.key === "ArrowRight") stepOnce(1);
  else if (e.key === "ArrowLeft") stepOnce(-1);
  else if (e.key === "ArrowUp") zoom(0.85);
  else if (e.key === "ArrowDown") zoom(1.18);
  else if (e.key === "g" || e.key === "G") { const c = $("graticule"); c.checked = !c.checked; globe.setGraticule(c.checked); }
  else if (e.key === "r" || e.key === "R") { globe.camera.position.set(0, 0.7, 2.8); globe.controls.target.set(0, 0, 0); }
}

function dragPanel(panel, handle) {
  let ox = 0, oy = 0, on = false;
  handle.addEventListener("pointerdown", (e) => {
    if (e.target.tagName === "BUTTON") return;
    on = true; ox = e.clientX - panel.offsetLeft; oy = e.clientY - panel.offsetTop;
    panel.style.right = "auto"; handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener("pointermove", (e) => {
    if (!on) return;
    panel.style.left = Math.max(4, Math.min(window.innerWidth - panel.offsetWidth - 4, e.clientX - ox)) + "px";
    panel.style.top = Math.max(4, Math.min(window.innerHeight - 60, e.clientY - oy)) + "px";
  });
  handle.addEventListener("pointerup", () => { on = false; });
}

boot();
