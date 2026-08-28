// main.js — Kalachakra v9 client. A Topocentric Self-Attention "energy signature" globe.
//
// The whole micro-transformer (11 body tokens -> attention -> pooled read-out -> L*a*b*) runs
// PER VERTEX in shader10.js; the GPU interpolates colour across triangles. The CPU computes the
// 11 Earth-fixed body directions + GMST once per frame and hands them to the shader. Around the
// field: a world map (geo.js), the 11 bodies as 3D spheres (planets10.js), the full v5-style
// Temporal Helm, and an Observer HUD that also streams the per-body attention "energy
// contribution". Self-contained — no cross-version imports.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildShaders, packWeights } from "./shader10.js";
import {
  equatorialDirs, gmstRad, obliquityRad, ascMcEcliptic, eclipticToEquatorial,
  BODY_NAMES, N_BODIES, N_PLANETS,
} from "./ephemeris10.js";
import { localVectors } from "./state10.js";
import { makeModel, oklabToSrgb, srgbToHex } from "./attn10.js";
import {
  J2000, nowJD, gregorianToJD, jdToGregorian, dtLocalValue, labelFor, tzLabel, TZ_OFFSETS,
} from "./timecal.js";
import {
  llToVec, buildOcean, loadCoastlines, buildCoastlines, buildGraticule, tryEarthTexture,
} from "./geo.js";
import { createPlanets, createAnchors } from "./planets10.js";

const DEFAULT_ARCH = {
  arch: "v10_topo_attention", n_bodies: 13, token_dim: 3, d_model: 32, d_ff: 64, d_head: 32,
  n_blocks: 2, vis_bias: 3.0, out_features: 2, okl_l: 0.5, okl_cmax: 0.4,
};
const JD_MIN = J2000 - 5000 * 365.25, JD_MAX = J2000 + 5000 * 365.25;
const LIVE_REFRESH_MS = 1000;

const $ = (id) => document.getElementById(id);
const setNotice = (t) => { $("notice").textContent = t || ""; };
let boot = $("boot");
function clearBoot() { if (boot) { boot.remove(); boot = null; } }

const app = {
  mode: "live", jd: nowJD(), playing: false, speed: 10000, stepHours: 24,
  opacity: 0.85, tzMode: "local", tzOffsetMin: 0, fieldW: 192, fieldH: 96,
  pin: { lat: 48.8566, lon: 2.3522 },
  weights: null, model: null,
};

// ---- WebGL2 capability guard (fail with guidance, not a blank screen) ------
function webgl2Available() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGL2RenderingContext && c.getContext("webgl2"));
  } catch { return false; }
}
const WEBGL2_HELP =
  "WebGL2 is unavailable in this browser, so the field can't render. This is a browser/GPU " +
  "setting, not the app: enable hardware acceleration (chrome://settings → System → “Use " +
  "graphics acceleration”, then relaunch) and check chrome://gpu. If you're on a VM / remote " +
  "desktop, try a local browser, or launch Chrome with --enable-unsafe-swiftshader for a " +
  "software fallback.";
if (!webgl2Available()) {
  setNotice(WEBGL2_HELP);
  if (boot) { boot.textContent = WEBGL2_HELP; boot.style.maxWidth = "560px"; boot.style.whiteSpace = "normal"; boot.style.lineHeight = "1.5"; }
  throw new Error("WebGL2 unavailable");
}

// ---- scene -----------------------------------------------------------------
const wrap = $("canvas-wrap");
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) {
  setNotice(WEBGL2_HELP);
  if (boot) { boot.textContent = WEBGL2_HELP; boot.style.maxWidth = "560px"; boot.style.whiteSpace = "normal"; boot.style.lineHeight = "1.5"; }
  throw e;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
wrap.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04060b);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 100);
camera.position.set(0, 0.6, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.minDistance = 1.06; controls.maxDistance = 8.0; controls.rotateSpeed = 0.6;

let renderDirty = true, prevJD = NaN, lastLiveTick = 0;
controls.addEventListener("change", () => { renderDirty = true; });

// starfield
const starGeo = new THREE.BufferGeometry();
{
  const N = 1500, sp = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, r = 40, s = Math.sqrt(1 - u * u);
    sp[i * 3] = r * s * Math.cos(t); sp[i * 3 + 1] = r * u; sp[i * 3 + 2] = r * s * Math.sin(t);
  }
  starGeo.setAttribute("position", new THREE.BufferAttribute(sp, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x8ea6c8, size: 0.13, transparent: true, opacity: 0.7 })));
}

// ---- field render target + globe -------------------------------------------
// The heavy attention net runs in an OFFSCREEN full-screen pass into an equirectangular
// (lon,lat) texture, recomputed only when the time changes; the globe just samples it. So
// rotating/zooming is a cheap texture lookup and never overloads the GPU.
const SPHERE_SEG = 128;
const bodyEcef = Array.from({ length: N_PLANETS }, () => new THREE.Vector3());
const fieldUniforms = {
  u_bodyEcef: { value: bodyEcef }, u_weights: { value: null }, u_wtexW: { value: 64 },
  u_gmst: { value: 0 }, u_cosEps: { value: 1 }, u_sinEps: { value: 0 },   // for per-texel ASC/MC
};
const globeUniforms = { u_field: { value: null }, u_opacity: { value: app.opacity } };
let globe = null, fieldMat = null, fieldScene = null, fieldCam = null, fieldRT = null;
let fieldDirty = true;

const ocean = buildOcean(0.997); ocean.renderOrder = 0; scene.add(ocean);
let coastMesh = null;
const gratMesh = buildGraticule(1.004); gratMesh.renderOrder = 2; gratMesh.visible = false; scene.add(gratMesh);
const planets = createPlanets(scene, BODY_NAMES.slice(0, N_PLANETS));   // 11 bodies as spheres
const anchors = createAnchors(scene);                                   // ASC + MC as markers
// Earth-fixed sub-point directions of the observer's ASC & MC (for the on-globe markers).
function ascMcMarkers(jd, lat, lon) {
  const { lamAsc, lamMc } = ascMcEcliptic(lat, lon, jd);
  const g = gmstRad(jd), cG = Math.cos(g), sG = Math.sin(g);
  const toEcef = (lam) => { const e = eclipticToEquatorial(lam, jd); return new THREE.Vector3(cG * e[0] + sG * e[1], e[2], cG * e[1] - sG * e[0]); };
  return [toEcef(lamAsc), toEcef(lamMc)];                                // [ASC, MC]
}
const reticle = new THREE.Mesh(new THREE.RingGeometry(0.02, 0.032, 40),
  new THREE.MeshBasicMaterial({ color: 0x6fe0ff, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthWrite: false }));
reticle.renderOrder = 3; scene.add(reticle);

async function loadWeights() {
  try { return await (await fetch("weights.json")).json(); }
  catch {
    setNotice("no weights.json — showing an UNTRAINED attention net (train + export to see the learned field)");
    return randomWeights(DEFAULT_ARCH);
  }
}
function randomWeights(a) {
  const U = (b) => (Math.random() * 2 - 1) * b;
  const mat = (out, inF) => { const b = Math.sqrt(6 / (out + inF)); return Array.from({ length: out }, () => Array.from({ length: inF }, () => U(b))); };
  const vecR = (n, s = 0.02) => Array.from({ length: n }, () => U(s));
  const D = a.d_model, DFF = a.d_ff, DHEAD = a.d_head;
  const blocks = Array.from({ length: a.n_blocks }, () => ({
    Wq: mat(D, D), bq: vecR(D, 0), Wk: mat(D, D), bk: vecR(D, 0), Wv: mat(D, D), bv: vecR(D, 0),
    W1: mat(DFF, D), b1: vecR(DFF, 0), W2: mat(D, DFF), b2: vecR(D, 0), tau: 1.0,
  }));
  return {
    ...a, W_in: mat(D, a.token_dim), b_in: vecR(D, 0), E_body: mat(a.n_bodies, D),
    blocks, q_pool: vecR(D), tau_pool: 1.0,
    Wo1: mat(DHEAD, D), bo1: vecR(DHEAD, 0), Wo2: mat(2, DHEAD), bo2: vecR(2, 0),
    output_activation: "v10_oklch", gamma: 0.35,
  };
}
function makeWeightTexture(weights) {
  const flat = packWeights(weights), W = 64, H = Math.ceil(flat.length / W);
  const data = new Float32Array(W * H); data.set(flat);
  const tex = new THREE.DataTexture(data, W, H, THREE.RedFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter; tex.needsUpdate = true;
  return { tex, W };
}
function archOf(w) {
  return {
    arch: "v10_topo_attention", n_bodies: w.n_bodies ?? 13, n_planets: N_PLANETS, token_dim: w.token_dim ?? 3,
    d_model: w.d_model, d_ff: w.d_ff, d_head: w.d_head, n_blocks: w.n_blocks,
    vis_bias: w.vis_bias ?? 3.0, out_features: w.out_features ?? 2, okl_l: w.okl_l ?? 0.5, okl_cmax: w.okl_cmax ?? 0.4,
  };
}
function makeFieldRT(w, h) {
  if (fieldRT) fieldRT.dispose();
  // The field stores OKLab (a,b) encoded to [0,1]. Prefer a half-float target so the (a,b)
  // carry full precision (no banding when the globe reconstructs colour per pixel); fall back
  // to 8-bit where EXT_color_buffer_float is unavailable (still correct, slightly coarser).
  const halfOK = renderer.extensions.has("EXT_color_buffer_float");
  fieldRT = new THREE.WebGLRenderTarget(w, h, {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
    wrapS: THREE.RepeatWrapping, wrapT: THREE.ClampToEdgeWrapping,
    depthBuffer: false, format: THREE.RGBAFormat,
    type: halfOK ? THREE.HalfFloatType : THREE.UnsignedByteType,
  });
  fieldRT.texture.colorSpace = THREE.NoColorSpace;   // raw (a,b) data, not a colour
  globeUniforms.u_field.value = fieldRT.texture;
  fieldDirty = true;
}
function renderField() {
  const prev = renderer.getRenderTarget();
  renderer.setRenderTarget(fieldRT);
  renderer.render(fieldScene, fieldCam);
  renderer.setRenderTarget(prev);
  fieldDirty = false;
}
function buildGlobe(weights) {
  const sh = buildShaders(archOf(weights));
  const { tex, W } = makeWeightTexture(weights);
  fieldUniforms.u_weights.value = tex; fieldUniforms.u_wtexW.value = W;
  // offscreen full-screen field pass (runs the network once per time change)
  fieldMat = new THREE.ShaderMaterial({
    uniforms: fieldUniforms, vertexShader: sh.field.vertex, fragmentShader: sh.field.fragment,
    glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false,
  });
  fieldScene = new THREE.Scene();
  fieldScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fieldMat));
  fieldCam = new THREE.Camera();
  makeFieldRT(app.fieldW, app.fieldH);
  // globe: cheap per-pixel sample of the field texture
  const globeMat = new THREE.ShaderMaterial({
    uniforms: globeUniforms, vertexShader: sh.globe.vertex, fragmentShader: sh.globe.fragment,
    glslVersion: THREE.GLSL3, transparent: true, depthWrite: false,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, SPHERE_SEG, SPHERE_SEG), globeMat);
  mesh.renderOrder = 1; scene.add(mesh);
  return mesh;
}
function setFieldRes(w, h) {
  app.fieldW = w; app.fieldH = h;
  if (fieldRT) makeFieldRT(w, h);
  renderDirty = true;
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
    renderDirty = true;
  }
  controls.update();

  if (renderDirty) {
    if (app.jd !== prevJD) {
      const eq = equatorialDirs(app.jd);                    // (cosδcosα, cosδsinα, sinδ) x 11
      const g = gmstRad(app.jd), cG = Math.cos(g), sG = Math.sin(g);
      for (let b = 0; b < N_PLANETS; b++) {
        const x = eq[3 * b], y = eq[3 * b + 1], z = eq[3 * b + 2];
        bodyEcef[b].set(cG * x + sG * y, z, cG * y - sG * x);   // equatorial -> earth-fixed sub-point
      }
      const eps = obliquityRad(app.jd);                     // ASC/MC computed per-texel in-shader
      fieldUniforms.u_gmst.value = g;
      fieldUniforms.u_cosEps.value = Math.cos(eps);
      fieldUniforms.u_sinEps.value = Math.sin(eps);
      planets.update(bodyEcef);
      prevJD = app.jd;
      fieldDirty = true;                                    // sky changed -> recompute the field
    }
    // ASC/MC markers depend on the pinned observer AND time -> refresh on any dirty frame
    anchors.update(...ascMcMarkers(app.jd, app.pin.lat, app.pin.lon));
    if (fieldDirty && fieldRT) renderField();               // heavy net: ONCE per time change
    renderer.render(scene, camera);                         // cheap textured globe: every frame
    renderDirty = false;
    clearBoot();
    fpsFrames++;
  }

  updateReadouts(now);
  if (now - fpsT >= 500) {
    const fps = Math.round(fpsFrames * 1000 / (now - fpsT));
    $("fps").textContent = fps > 0 ? `${fps} fps` : "idle";
    $("render-info").textContent = `field ${app.fieldW}×${app.fieldH}`;
    fpsFrames = 0; fpsT = now;
  }
}

// ---- HUD --------------------------------------------------------------------
const el = {}; const matRows = []; const enRows = [];
let lastHud = 0;
function buildHud() {
  for (const id of ["clock", "date", "matrix", "energy", "lab", "hex", "swatch", "pin-lat", "pin-lon", "scrub-label"]) el[id] = $(id);
  el.energy.textContent = "";
  for (let b = 0; b < N_BODIES; b++) {
    const row = document.createElement("div"); row.className = "erow";
    const name = document.createElement("span"); name.className = "ename"; name.textContent = BODY_NAMES[b];
    const bar = document.createElement("span"); bar.className = "ebar";
    const val = document.createElement("span"); val.className = "eval";
    row.append(name, bar, val); el.energy.appendChild(row); enRows.push({ row, bar, val });
  }
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
  if (!app.model) return;
  const local = localVectors(app.pin.lat, app.pin.lon, app.jd);   // Float32Array(33)
  for (let b = 0; b < N_BODIES; b++)
    matRows[b].textContent = ` ${fmt(local[b * 3])} ${fmt(local[b * 3 + 1])} ${fmt(local[b * 3 + 2])}`;
  const r = app.model(local), pool = r.pool;
  let pmax = 1e-6; for (let b = 0; b < N_BODIES; b++) pmax = Math.max(pmax, pool[b]);
  for (let b = 0; b < N_BODIES; b++) {
    enRows[b].bar.style.width = `${Math.round((pool[b] / pmax) * 88)}px`;
    enRows[b].val.textContent = `${(pool[b] * 100).toFixed(0)}%`;
    enRows[b].row.classList.toggle("down", local[b * 3 + 2] < 0);   // below the observer's horizon
  }
  const hex = srgbToHex(oklabToSrgb(r.L, r.ab[0], r.ab[1]));
  let hdeg = r.H * 180 / Math.PI % 360; if (hdeg < 0) hdeg += 360;
  el.lab.textContent = `C ${r.C.toFixed(3)}  H ${hdeg.toFixed(0)}°  ·  L ${r.L.toFixed(2)} fixed`;
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
           lon: THREE.MathUtils.radToDeg(Math.atan2(-p.z, p.x)), vec: p };   // -z: un-mirrored map
}
function setPin(pick) {
  app.pin.lat = pick.lat; app.pin.lon = pick.lon;
  reticle.position.copy(pick.vec).multiplyScalar(1.008); reticle.lookAt(0, 0, 0);
  renderDirty = true;
}

// ---- helm -------------------------------------------------------------------
function speedFromSlider(v) { return v >= 0 ? Math.pow(10, v) : -Math.pow(10, -v); }
function fmtSpeed(m) {
  const a = Math.abs(m); const t = a >= 1e6 ? (a / 1e6).toFixed(0) + "M" : a >= 1000 ? (a / 1000).toFixed(0) + "k" : a.toFixed(a < 10 ? 1 : 0);
  return (m < 0 ? "-" : "") + t + "×";
}
function jumpTo(jd) { app.jd = Math.min(JD_MAX, Math.max(JD_MIN, jd)); renderDirty = true; }
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
  camera.position.copy(t).add(off.setLength(dist)); renderDirty = true;
}

function wireControls() {
  const tz = $("tz");
  const opt = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; tz.appendChild(o); };
  opt("local", "Local"); opt("0", "UTC");
  for (const off of TZ_OFFSETS) if (off !== 0) opt(String(off), tzLabel(off));
  tz.onchange = () => { if (tz.value === "local") app.tzMode = "local"; else { app.tzMode = "fixed"; app.tzOffsetMin = parseInt(tz.value, 10); } };

  $("btn-mode").onclick = () => setMode(app.mode !== "machine");
  const play = $("play");
  play.onclick = () => { if (app.mode !== "machine") return; app.playing = !app.playing; play.classList.toggle("on", app.playing); play.textContent = app.playing ? "▶ Playing" : "⏸ Pause"; };
  $("now").onclick = () => { setMode(false); app.jd = nowJD(); renderDirty = true; };

  const sp = $("speed");
  sp.oninput = () => { app.speed = speedFromSlider(parseFloat(sp.value)); $("speed-val").textContent = fmtSpeed(app.speed); };
  app.speed = speedFromSlider(parseFloat(sp.value)); $("speed-val").textContent = fmtSpeed(app.speed);

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
  op.oninput = () => { app.opacity = parseFloat(op.value); globeUniforms.u_opacity.value = app.opacity; $("opacity-val").textContent = app.opacity.toFixed(2); renderDirty = true; };
  $("quality").onchange = (e) => { const [w, h] = e.target.value.split("x").map(Number); setFieldRes(w, h); };

  $("ov-map").onchange = (e) => { ocean.visible = e.target.checked; renderDirty = true; };
  $("ov-coast").onchange = (e) => { if (coastMesh) coastMesh.visible = e.target.checked; renderDirty = true; };
  $("ov-grat").onchange = (e) => { gratMesh.visible = e.target.checked; renderDirty = true; };
  $("ov-planets").onchange = (e) => { planets.setVisible(e.target.checked); if (e.target.checked) planets.setLabels($("ov-labels").checked); renderDirty = true; };
  $("ov-anchors").onchange = (e) => { anchors.setVisible(e.target.checked); if (e.target.checked) anchors.setLabels($("ov-labels").checked); renderDirty = true; };
  $("ov-labels").onchange = (e) => { planets.setLabels(e.target.checked); anchors.setLabels(e.target.checked); renderDirty = true; };

  $("panel-hide").onclick = () => { $("panel").classList.add("hidden"); $("panel-show").style.display = "block"; };
  $("panel-show").onclick = () => { $("panel").classList.remove("hidden"); $("panel-show").style.display = "none"; };
  dragPanel($("panel"), $("panel-header"));
  $("hud-toggle").onclick = () => { const h = $("hud"); h.classList.toggle("collapsed"); $("hud-toggle").textContent = h.classList.contains("collapsed") ? "▸" : "▾"; };

  let dragging = false;
  renderer.domElement.addEventListener("pointerdown", (e) => { const p = pickFromEvent(e); if (p) { dragging = true; setPin(p); } });
  renderer.domElement.addEventListener("pointermove", (e) => { if (dragging) { const p = pickFromEvent(e); if (p) setPin(p); } });
  window.addEventListener("pointerup", () => { dragging = false; });

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight); renderDirty = true;
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
  else if (e.key === "r" || e.key === "R") { camera.position.set(0, 0.6, 2.6); controls.target.set(0, 0, 0); renderDirty = true; }
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
  app.weights = await loadWeights();
  app.model = makeModel(app.weights);
  globe = buildGlobe(app.weights);
  loadCoastlines().then((rings) => {
    if (!rings.length) return;
    coastMesh = buildCoastlines(rings, 1.006); coastMesh.renderOrder = 2;
    coastMesh.visible = $("ov-coast").checked; scene.add(coastMesh); renderDirty = true;
  });
  tryEarthTexture(0.998, (mesh) => { mesh.renderOrder = 0; scene.add(mesh); ocean.visible = false; renderDirty = true; });
  setPin({ lat: app.pin.lat, lon: app.pin.lon, vec: llToVec(app.pin.lat, app.pin.lon, 1) });
  $("dt").value = dtLocalValue(app.jd);
  $("scrub").value = String(app.jd);
  animate();
  setTimeout(clearBoot, 4000);
}
main().catch((e) => { console.error("v9 main() failed:", e); setNotice("init error: " + e.message); clearBoot(); });
