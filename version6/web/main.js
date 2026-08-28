// main.js — Kalachakra v6 client. A continuous SIREN globe over a world map.
//
// The colour field is still evaluated per pixel in the fragment shader (shader6.js) — the
// exact, infinite-resolution math is UNCHANGED. What changed for performance is *how many*
// pixels we pay for: while you rotate / zoom / scrub / play, the globe renders at a reduced
// buffer scale; the instant motion stops it snaps back to a full-resolution frame. So motion
// is cheap and the still image is pristine.
//
// Around the exact field we add: a world map underneath (geo.js), all planets floating around
// the globe at their sub-points (planets.js), and the full v5 control suite — LIVE / Time
// Machine, timezone, speed, custom steps, scrubber, zoom, field opacity, overlays, HUD.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildVertexShaders, buildPixelShaders, packWeights } from "./shader6.js";
import {
  topocentricTensor, equatorialDirs, gmstRad, BODY_NAMES, N_BODIES,
} from "./ephemeris6.js";
import { makeSiren, boundLab, labToSrgb, srgbToHex } from "./siren6.js";
import {
  J2000, nowJD, gregorianToJD, jdToGregorian, dtLocalValue, labelFor, tzLabel, TZ_OFFSETS,
} from "./timecal.js";
import {
  llToVec, buildOcean, loadCoastlines, buildCoastlines, buildGraticule, tryEarthTexture,
} from "./geo.js";
import { createPlanets } from "./planets.js";

// Dense mesh: the vertex path evaluates the SIREN at every vertex, so a fine tessellation
// keeps the interpolated field faithful while costing ~40x less than per-pixel. ~52k verts.
const SEG_W = 320, SEG_H = 160;
const DEFAULT_ARCH = {
  in_features: 33, hidden: 48, hidden_layers: 2, out_features: 3, omega0: 30,
  lab_center: 50, lab_lspan: 50, lab_ab: 90,
};
const JD_MIN = J2000 - 5000 * 365.25, JD_MAX = J2000 + 5000 * 365.25;
const IDLE_MS = 240;            // after this much stillness -> render one full-res frame
const LIVE_REFRESH_MS = 1000;   // in LIVE mode the sky is re-baked at most this often

const $ = (id) => document.getElementById(id);
const setNotice = (t) => { $("notice").textContent = t || ""; };

// ---- boot overlay (the element lives in index.html) ------------------------
let boot = document.getElementById("boot");
function clearBoot() { if (boot) { boot.remove(); boot = null; } }

// ---- state -----------------------------------------------------------------
const app = {
  mode: "live", jd: nowJD(), playing: false, speed: 10000, stepHours: 24,
  opacity: 0.85, tzMode: "local", tzOffsetMin: 0, quality: "auto",
  pin: { lat: 48.8566, lon: 2.3522 },
  weights: null, siren: null,
};

// ---- three.js scene --------------------------------------------------------
const wrap = $("canvas-wrap");
const renderer = new THREE.WebGLRenderer({ antialias: true });
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04060b);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 100);
camera.position.set(0, 0.6, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.minDistance = 1.06; controls.maxDistance = 8.0; controls.rotateSpeed = 0.6;

let renderDirty = true, prevJD = NaN, motionUntil = 0, lastLiveTick = 0;
controls.addEventListener("change", () => { renderDirty = true; markMotion(); });
function markMotion() { motionUntil = performance.now() + IDLE_MS; }

// ---- adaptive resolution (only bites in the exact per-pixel path) -----------
let cssW = window.innerWidth, cssH = window.innerHeight, appliedScale = 0;
function dprCap(x) { return Math.min(window.devicePixelRatio || 1, x); }
const FULL = () => dprCap(1.75);
function applyScale(s) {
  if (s === appliedScale) return;
  appliedScale = s;
  renderer.setPixelRatio(s);
  renderer.setSize(cssW, cssH, false);                // CSS (100%) handles display size
  renderDirty = true;
}

// ---- field sphere -----------------------------------------------------------
// body directions in the EARTH-FIXED globe frame (sub-point unit vectors), rebuilt on the CPU
// each frame — the GMST spin is folded in here so the shader stays pure vector math.
const bodyEcef = Array.from({ length: N_BODIES }, () => new THREE.Vector3());
const uniforms = {
  u_bodyEcef: { value: bodyEcef },
  u_weights: { value: null }, u_wtexW: { value: 64 }, u_opacity: { value: app.opacity },
};
let globe = null, vertexMat = null, pixelMat = null;

// ---- world map + planets + reticle -----------------------------------------
const ocean = buildOcean(0.997); ocean.renderOrder = 0; scene.add(ocean);
let coastMesh = null, gratMesh = buildGraticule(1.004); gratMesh.renderOrder = 2; gratMesh.visible = false; scene.add(gratMesh);
const planets = createPlanets(scene, BODY_NAMES);
const reticle = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.032, 40),
  new THREE.MeshBasicMaterial({ color: 0x6fe0ff, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthWrite: false }));
reticle.renderOrder = 3; scene.add(reticle);

// ---- weights ---------------------------------------------------------------
async function loadWeights() {
  try { return await (await fetch("weights.json")).json(); }
  catch {
    setNotice("no weights.json — showing an UNTRAINED SIREN (train + export to see the learned field)");
    return randomWeights(DEFAULT_ARCH);
  }
}
function randomWeights(a) {
  const U = (lo, hi) => lo + Math.random() * (hi - lo);
  const layer = (inF, outF, first, act) => {
    const bound = first ? 1 / inF : Math.sqrt(6 / inF) / a.omega0;
    return {
      W: Array.from({ length: outF }, () => Array.from({ length: inF }, () => U(-bound, bound))),
      b: Array.from({ length: outF }, () => U(-bound, bound)), activation: act,
    };
  };
  const layers = [layer(a.in_features, a.hidden, true, "sin")];
  for (let k = 1; k < a.hidden_layers; k++) layers.push(layer(a.hidden, a.hidden, false, "sin"));
  layers.push(layer(a.hidden, a.out_features, false, "linear"));
  return { ...a, layers, output_activation: "lab_tanh", color_scale: 20 };
}
function makeWeightTexture(weights) {
  const flat = packWeights(weights), W = 64, H = Math.ceil(flat.length / W);
  const data = new Float32Array(W * H); data.set(flat);
  const tex = new THREE.DataTexture(data, W, H, THREE.RedFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
  return { tex, W };
}
function buildGlobe(weights) {
  const arch = {
    in_features: weights.in_features ?? 33, hidden: weights.hidden, hidden_layers: weights.hidden_layers,
    out_features: weights.out_features ?? 3, omega0: weights.omega0,
    lab_center: weights.lab_center ?? 50, lab_lspan: weights.lab_lspan ?? 50, lab_ab: weights.lab_ab ?? 90,
  };
  const { tex, W } = makeWeightTexture(weights);
  uniforms.u_weights.value = tex; uniforms.u_wtexW.value = W;
  const mk = (src) => new THREE.ShaderMaterial({
    uniforms, vertexShader: src.vertex, fragmentShader: src.fragment, glslVersion: THREE.GLSL3,
    transparent: true, depthWrite: false,
  });
  vertexMat = mk(buildVertexShaders(arch));      // fast: SIREN per vertex, interpolated
  pixelMat = mk(buildPixelShaders(arch));        // exact: SIREN per pixel
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, SEG_W, SEG_H), vertexMat);
  mesh.renderOrder = 1; scene.add(mesh);
  return mesh;
}

// ---- render loop -----------------------------------------------------------
let last = performance.now(), fpsFrames = 0, fpsT = last;
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = (now - last) / 1000; last = now;

  if (app.mode === "live") {
    if (now - lastLiveTick >= LIVE_REFRESH_MS) { app.jd = nowJD(); lastLiveTick = now; renderDirty = true; }
  } else if (app.playing) {
    app.jd = Math.min(JD_MAX, Math.max(JD_MIN, app.jd + app.speed * dt / 86400.0));
    renderDirty = true; markMotion();
  }
  controls.update();

  // Pick the render path. auto: SIREN-per-vertex while moving (cheap), then a per-pixel exact
  // frame the moment it settles. fast: always per-vertex. exact: always per-pixel.
  const moving = app.playing || now < motionUntil;
  const useVertex = app.quality === "fast" ? true : app.quality === "exact" ? false : moving;
  if (globe) {
    const targetMat = useVertex ? vertexMat : pixelMat;
    if (globe.material !== targetMat) { globe.material = targetMat; renderDirty = true; }
  }
  applyScale(useVertex ? FULL() : (moving ? 0.5 : FULL()));   // downscale only per-pixel-while-moving

  if (renderDirty) {
    if (app.jd !== prevJD) {
      const eq = equatorialDirs(app.jd);                      // (cosδcosα, cosδsinα, sinδ)
      const g = gmstRad(app.jd), cG = Math.cos(g), sG = Math.sin(g);
      for (let b = 0; b < N_BODIES; b++) {
        const x = eq[3 * b], y = eq[3 * b + 1], z = eq[3 * b + 2];
        bodyEcef[b].set(cG * x + sG * y, z, cG * y - sG * x); // equatorial -> earth-fixed sub-point
      }
      planets.update(bodyEcef);
      prevJD = app.jd;
    }
    renderer.render(scene, camera);
    renderDirty = false;
    clearBoot();                                   // first real frame is up — drop the overlay
    fpsFrames++;                                    // count actual draws, not idle rAF ticks
  }

  updateReadouts(now);
  if (now - fpsT >= 500) {
    const fps = Math.round(fpsFrames * 1000 / (now - fpsT));
    $("fps").textContent = fps > 0 ? `${fps} fps` : "idle";
    $("render-info").textContent = `${useVertex ? "vertex" : "pixel"} · ${appliedScale.toFixed(2)}×`;
    fpsFrames = 0; fpsT = now;
  }
}

// ---- HUD --------------------------------------------------------------------
const el = {}; const matRows = [];
let lastHud = 0;
function buildHud() {
  for (const id of ["clock", "date", "matrix", "lab", "hex", "swatch", "pin-lat", "pin-lon", "scrub-label"]) el[id] = $(id);
  el.matrix.textContent = "";
  for (let b = 0; b < N_BODIES; b++) {
    const row = document.createElement("div");
    const name = document.createElement("span"); name.className = "body"; name.textContent = BODY_NAMES[b].padEnd(8);
    const vals = document.createElement("span"); row.append(name, vals); el.matrix.appendChild(row); matRows.push(vals);
  }
}
function tzOffsetMin() { return app.tzMode === "local" ? -new Date().getTimezoneOffset() : app.tzOffsetMin; }
function tzSuffix(off) { return app.tzMode === "local" ? "Local" : tzLabel(off); }
function fmt(x) { return (x >= 0 ? " " : "-") + Math.abs(x).toFixed(3); }
function updateReadouts(now) {
  if (now - lastHud < 100) return;
  lastHud = now;
  const off = tzOffsetMin(), lab = labelFor(app.jd, off, tzSuffix(off));
  el.clock.textContent = lab.clock; el.date.textContent = lab.date;
  if (app.mode === "machine") { $("scrub").value = String(app.jd); el["scrub-label"].textContent = `${lab.date} · ${lab.clock}`; }
  if (!app.siren) return;
  const sky = topocentricTensor(app.pin.lat, app.pin.lon, app.jd);
  for (let b = 0; b < N_BODIES; b++)
    matRows[b].textContent = ` ${fmt(sky[b * 3])} ${fmt(sky[b * 3 + 1])} ${fmt(sky[b * 3 + 2])}`;
  const lab3 = boundLab(app.weights, app.siren(sky));
  const hex = srgbToHex(labToSrgb(lab3[0], lab3[1], lab3[2]));
  el.lab.textContent = `L* ${lab3[0].toFixed(1)}  a* ${lab3[1].toFixed(1)}  b* ${lab3[2].toFixed(1)}`;
  el.hex.textContent = hex; el.swatch.style.background = hex;
  el["pin-lat"].textContent = app.pin.lat.toFixed(4); el["pin-lon"].textContent = app.pin.lon.toFixed(4);
}

// ---- picking ----------------------------------------------------------------
const raycaster = new THREE.Raycaster(); const ndc = new THREE.Vector2();
function pickFromEvent(e) {
  if (!globe) return null;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(globe, false)[0];
  if (!hit) return null;
  const p = hit.point.clone().normalize();
  return { lat: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(p.y, -1, 1))),
           lon: THREE.MathUtils.radToDeg(Math.atan2(-p.z, p.x)), vec: p };   // -z: match un-mirrored map
}
function setPin(pick) {
  app.pin.lat = pick.lat; app.pin.lon = pick.lon;
  reticle.position.copy(pick.vec).multiplyScalar(1.008); reticle.lookAt(0, 0, 0);
  renderDirty = true;
}

// ---- helm helpers ----------------------------------------------------------
function speedFromSlider(v) { return v >= 0 ? Math.pow(10, v) : -Math.pow(10, -v); }
function fmtSpeed(m) {
  const a = Math.abs(m); const t = a >= 1e6 ? (a / 1e6).toFixed(0) + "M" : a >= 1000 ? (a / 1000).toFixed(0) + "k" : a.toFixed(a < 10 ? 1 : 0);
  return (m < 0 ? "-" : "") + t + "×";
}
function jumpTo(jd) { app.jd = Math.min(JD_MAX, Math.max(JD_MIN, jd)); renderDirty = true; markMotion(); }
function setMode(machine) {
  app.mode = machine ? "machine" : "live";
  $("mode-badge").textContent = machine ? "TIME MACHINE" : "LIVE";
  $("mode-badge").classList.toggle("machine", machine);
  $("btn-mode").textContent = machine ? "Go Live" : "Enter Time Machine";
  $("play").disabled = !machine; $("scrub").disabled = !machine;
  if (!machine) { app.playing = false; $("play").classList.remove("on"); $("play").textContent = "⏸ Pause"; }
  else { $("dt").value = dtLocalValue(app.jd); }
  renderDirty = true;
}
function stepOnce(dir) { setMode(true); jumpTo(app.jd + dir * app.stepHours / 24.0); $("dt").value = dtLocalValue(app.jd); }
function zoom(factor) {
  const t = controls.target, off = camera.position.clone().sub(t);
  const dist = Math.max(controls.minDistance, Math.min(controls.maxDistance, off.length() * factor));
  camera.position.copy(t).add(off.setLength(dist)); renderDirty = true; markMotion();
}

function wireControls() {
  const tz = $("tz");
  const opt = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; tz.appendChild(o); };
  opt("local", "Local");
  for (const off of TZ_OFFSETS) opt(String(off), off === 0 ? "UTC" : tzLabel(off));
  tz.onchange = () => { if (tz.value === "local") app.tzMode = "local"; else { app.tzMode = "fixed"; app.tzOffsetMin = parseInt(tz.value, 10); } };

  $("btn-mode").onclick = () => setMode(app.mode !== "machine");
  const play = $("play");
  play.onclick = () => { if (app.mode !== "machine") return; app.playing = !app.playing; play.classList.toggle("on", app.playing); play.textContent = app.playing ? "▶ Playing" : "⏸ Pause"; };
  $("now").onclick = () => { setMode(false); app.jd = nowJD(); renderDirty = true; markMotion(); };

  const sp = $("speed");
  sp.oninput = () => { app.speed = speedFromSlider(parseFloat(sp.value)); $("speed-val").textContent = fmtSpeed(app.speed); };
  app.speed = speedFromSlider(parseFloat(sp.value)); $("speed-val").textContent = fmtSpeed(app.speed);   // sync default

  document.querySelectorAll("#steps button").forEach((b) => b.onclick = () => {
    document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); app.stepHours = parseFloat(b.dataset.h); $("step-input").value = app.stepHours;
  });
  $("step-input").oninput = (e) => { const v = parseFloat(e.target.value); if (!Number.isNaN(v)) app.stepHours = v; document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel")); };
  $("btn-back").onclick = () => stepOnce(-1);
  $("btn-fwd").onclick = () => stepOnce(1);

  const scrub = $("scrub");
  scrub.min = String(JD_MIN); scrub.max = String(JD_MAX);
  scrub.oninput = () => { setMode(true); jumpTo(parseFloat(scrub.value)); $("dt").value = dtLocalValue(app.jd); };
  $("go").onclick = () => {
    const v = $("dt").value, m = v && v.match(/^(-?\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/);
    if (!m) { setNotice("enter a date/time"); return; }
    setNotice(""); setMode(true); jumpTo(gregorianToJD(+m[1], +m[2], +m[3], +m[4], +m[5], +(m[6] || 0)));
  };

  $("zoom-in").onclick = () => zoom(0.82);
  $("zoom-out").onclick = () => zoom(1.22);

  const op = $("opacity");
  op.oninput = () => { app.opacity = parseFloat(op.value); uniforms.u_opacity.value = app.opacity; $("opacity-val").textContent = app.opacity.toFixed(2); renderDirty = true; };
  $("quality").onchange = (e) => { app.quality = e.target.value; renderDirty = true; };

  $("ov-map").onchange = (e) => { ocean.visible = e.target.checked; renderDirty = true; };
  $("ov-coast").onchange = (e) => { if (coastMesh) coastMesh.visible = e.target.checked; renderDirty = true; };
  $("ov-grat").onchange = (e) => { gratMesh.visible = e.target.checked; renderDirty = true; };
  $("ov-planets").onchange = (e) => { planets.setVisible(e.target.checked); renderDirty = true; };
  $("ov-labels").onchange = (e) => { planets.setLabels(e.target.checked); renderDirty = true; };

  $("panel-hide").onclick = () => { $("panel").classList.add("hidden"); $("panel-show").style.display = "block"; };
  $("panel-show").onclick = () => { $("panel").classList.remove("hidden"); $("panel-show").style.display = "none"; };
  dragPanel($("panel"), $("panel-header"));
  $("hud-toggle").onclick = () => { const h = $("hud"); h.classList.toggle("collapsed"); $("hud-toggle").textContent = h.classList.contains("collapsed") ? "▸" : "▾"; };

  let dragging = false;
  renderer.domElement.addEventListener("pointerdown", (e) => { const p = pickFromEvent(e); if (p) { dragging = true; setPin(p); } });
  renderer.domElement.addEventListener("pointermove", (e) => { if (dragging) { const p = pickFromEvent(e); if (p) setPin(p); } });
  window.addEventListener("pointerup", () => { dragging = false; });

  window.addEventListener("resize", () => {
    cssW = window.innerWidth; cssH = window.innerHeight;
    camera.aspect = cssW / cssH; camera.updateProjectionMatrix();
    appliedScale = 0; applyScale(FULL()); renderDirty = true;
  });
  window.addEventListener("keydown", onKey);
}
function onKey(e) {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); if (app.mode !== "machine") setMode(true); $("play").click(); }
  else if (e.key === "l" || e.key === "L") $("now").click();
  else if (e.key === "ArrowRight") stepOnce(1);
  else if (e.key === "ArrowLeft") stepOnce(-1);
  else if (e.key === "ArrowUp") zoom(0.85);
  else if (e.key === "ArrowDown") zoom(1.18);
  else if (e.key === "g" || e.key === "G") { const c = $("ov-grat"); c.checked = !c.checked; gratMesh.visible = c.checked; renderDirty = true; }
  else if (e.key === "r" || e.key === "R") { camera.position.set(0, 0.6, 2.6); controls.target.set(0, 0, 0); renderDirty = true; markMotion(); }
}
function dragPanel(panel, handle) {
  let ox = 0, oy = 0, on = false;
  handle.addEventListener("pointerdown", (e) => {
    if (e.target.tagName === "BUTTON") return;
    on = true; ox = e.clientX - panel.offsetLeft; oy = e.clientY - panel.offsetTop; panel.style.right = "auto"; handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener("pointermove", (e) => {
    if (!on) return;
    panel.style.left = Math.max(4, Math.min(window.innerWidth - panel.offsetWidth - 4, e.clientX - ox)) + "px";
    panel.style.top = Math.max(4, Math.min(window.innerHeight - 60, e.clientY - oy)) + "px";
  });
  handle.addEventListener("pointerup", () => { on = false; });
}

// ---- bootstrap -------------------------------------------------------------
async function main() {
  buildHud(); wireControls();
  cssW = window.innerWidth; cssH = window.innerHeight;
  camera.aspect = cssW / cssH; camera.updateProjectionMatrix();
  applyScale(FULL());                            // start crisp

  app.weights = await loadWeights();
  app.siren = makeSiren(app.weights);
  globe = buildGlobe(app.weights);

  loadCoastlines().then((rings) => {
    if (!rings.length) return;
    coastMesh = buildCoastlines(rings, 1.006); coastMesh.renderOrder = 2;
    coastMesh.visible = $("ov-coast").checked; scene.add(coastMesh); renderDirty = true;
  });
  tryEarthTexture(0.998, (mesh) => {             // optional photographic base if earth.jpg present
    mesh.renderOrder = 0; scene.add(mesh); ocean.visible = false; renderDirty = true;
  });

  setPin({ lat: app.pin.lat, lon: app.pin.lon, vec: llToVec(app.pin.lat, app.pin.lon, 1) });
  $("dt").value = dtLocalValue(app.jd);
  $("scrub").value = String(app.jd);
  animate();
  setTimeout(clearBoot, 4000);                     // safety net if the first frame never renders
}
main().catch((e) => { console.error("v6 main() failed:", e); setNotice("init error: " + e.message); clearBoot(); });
