// main.js — Kalachakra v6 client. A continuous SIREN globe.
//
// Module 2: a Three.js sphere whose ShaderMaterial runs the FULL topocentric ephemeris
//           + SIREN + L*a*b*->sRGB per pixel (shader6.js) — infinite resolution.
// Module 3: a double-precision Julian-Date master clock, split into u_baseDays +
//           u_timeOffset uniforms; play/rewind velocity multiplier; fluid scrubber;
//           exact timestamp injection.
// Module 4: a raycaster picks the floating-point lat/lon under the cursor; a JS port of
//           the same ephemeris + SIREN (ephemeris6.js / siren6.js) fills the HUD matrix.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildShaders, packWeights } from "./shader6.js";
import { topocentricTensor, BODY_NAMES, J2000, N_BODIES } from "./ephemeris6.js";
import { makeSiren, applyOffset, labToSrgb, srgbToHex } from "./siren6.js";

const SEG = 256;                         // sphere segments (silhouette only; colour is per-pixel)
const DEFAULT_ARCH = { in_features: 33, hidden: 48, hidden_layers: 2, out_features: 3, omega0: 30 };
const JD_MIN = J2000 - 5000 * 365.25, JD_MAX = J2000 + 5000 * 365.25;

// ---- boot overlay ----------------------------------------------------------
const boot = document.createElement("div");
boot.id = "boot"; boot.textContent = "compiling SIREN shader…";
document.body.appendChild(boot);
const setNotice = (t) => { document.getElementById("notice").textContent = t || ""; };

// ---- state -----------------------------------------------------------------
const app = {
  jd: nowJD(),                           // 64-bit master clock (Julian Date)
  playing: false,
  speed: 1,                              // real-time multiplier (negative = rewind)
  pin: { lat: 48.8566, lon: 2.3522 },    // pinned observer (Module 4)
  weights: null, siren: null, exposure: 2.5,
};

// ---- three.js scene --------------------------------------------------------
const wrap = document.getElementById("canvas-wrap");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04060b);
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.001, 100);
camera.position.set(0, 0.6, 2.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;           // buttery inertia (Module 2.4)
controls.dampingFactor = 0.06;
controls.minDistance = 1.02;             // down to street-level micro-zoom
controls.maxDistance = 8.0;
controls.rotateSpeed = 0.6;

const uniforms = {
  u_baseDays: { value: 0 },
  u_timeOffset: { value: 0 },
  u_weights: { value: null },
  u_wtexW: { value: 64 },
  u_labOffset: { value: new THREE.Vector3(60, 0, 0) },
  u_exposure: { value: app.exposure },
};
let globe = null;                        // the shader sphere (built once weights are known)

// reticle (Module 4 spatial anchor)
const reticle = new THREE.Mesh(
  new THREE.RingGeometry(0.018, 0.03, 32),
  new THREE.MeshBasicMaterial({ color: 0x6fe0ff, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
scene.add(reticle);

// ---- weights: load trained weights.json, else an untrained random SIREN ----
async function loadWeights() {
  try {
    const w = await (await fetch("weights.json")).json();
    return w;
  } catch {
    setNotice("no weights.json — showing an UNTRAINED SIREN (train + export to see the learned field)");
    return randomWeights(DEFAULT_ARCH);
  }
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
  return { ...a, layers, lab_offset: [60, 0, 0], color_scale: 20 };
}

function makeWeightTexture(weights) {
  const flat = packWeights(weights);
  const W = 64, H = Math.ceil(flat.length / W);
  const data = new Float32Array(W * H);
  data.set(flat);
  const tex = new THREE.DataTexture(data, W, H, THREE.RedFormat, THREE.FloatType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return { tex, W };
}

function buildGlobe(weights) {
  const arch = {
    in_features: weights.in_features ?? 33, hidden: weights.hidden,
    hidden_layers: weights.hidden_layers, out_features: weights.out_features ?? 3,
    omega0: weights.omega0,
  };
  const { vertex, fragment } = buildShaders(arch);
  const { tex, W } = makeWeightTexture(weights);
  uniforms.u_weights.value = tex;
  uniforms.u_wtexW.value = W;
  uniforms.u_labOffset.value.fromArray(weights.lab_offset || [60, 0, 0]);
  const mat = new THREE.ShaderMaterial({ uniforms, vertexShader: vertex, fragmentShader: fragment });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, SEG, SEG), mat);
  scene.add(mesh);
  return mesh;
}

// ---- time helpers (double precision) ---------------------------------------
function nowJD() { return unixToJD(Date.now() / 1000); }
function unixToJD(sec) { return 2440587.5 + sec / 86400.0; }
function gregorianToJD(y, mo, d, h, mi, s) {
  // Meeus (proleptic Gregorian, astronomical year numbering)
  let a, b;
  if (mo <= 2) { y -= 1; mo += 12; }
  a = Math.floor(y / 100); b = 2 - a + Math.floor(a / 4);
  const jd0 = Math.floor(365.25 * (y + 4716)) + Math.floor(30.6001 * (mo + 1)) + d + b - 1524.5;
  return jd0 + (h + mi / 60 + s / 3600) / 24;
}
function jdToGregorian(jd) {
  const z = Math.floor(jd + 0.5), f = jd + 0.5 - z;
  let a = z;
  if (z >= 2299161) { const al = Math.floor((z - 1867216.25) / 36524.25); a = z + 1 + al - Math.floor(al / 4); }
  const b = a + 1524, c = Math.floor((b - 122.1) / 365.25), dd = Math.floor(365.25 * c);
  const e = Math.floor((b - dd) / 30.6001);
  const day = b - dd - Math.floor(30.6001 * e) + f;
  const mo = e < 14 ? e - 1 : e - 13;
  const y = mo > 2 ? c - 4716 : c - 4715;
  const dayInt = Math.floor(day); let frac = (day - dayInt) * 24;
  const h = Math.floor(frac); frac = (frac - h) * 60;
  const mi = Math.floor(frac); const s = Math.round((frac - mi) * 60);
  return { y, mo, d: dayInt, h, mi, s };
}
function jdLabel(jd) {
  const g = jdToGregorian(jd), p = (x) => String(x).padStart(2, "0");
  const era = g.y > 0 ? "CE" : "BCE", yy = g.y > 0 ? g.y : 1 - g.y;
  return { clock: `${p(g.h)}:${p(g.mi)}:${p(Math.min(59, g.s))} UTC`,
           date: `${String(yy)} ${era} · ${p(g.mo)}-${p(g.d)}` };
}

// ---- render loop -----------------------------------------------------------
let last = performance.now();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = (now - last) / 1000; last = now;

  if (app.playing) app.jd += app.speed * dt / 86400.0;   // multiplier x real-time
  app.jd = Math.min(JD_MAX, Math.max(JD_MIN, app.jd));

  // Module 3 precision split: exact integer days + small offset
  const d = app.jd - J2000;
  const base = Math.round(d);
  uniforms.u_baseDays.value = base;
  uniforms.u_timeOffset.value = d - base;
  uniforms.u_exposure.value = app.exposure;

  controls.update();
  renderer.render(scene, camera);

  updateReadouts();
}

// ---- HUD (Module 4) --------------------------------------------------------
let hudTick = 0;
function updateReadouts() {
  const lab = jdLabel(app.jd);
  document.getElementById("clock").textContent = lab.clock;
  document.getElementById("date").textContent = lab.date;
  if (!app.playing) syncScrub();

  if (app.siren && (hudTick++ % 3 === 0)) {              // ~20 Hz HUD refresh
    const sky = topocentricTensor(app.pin.lat, app.pin.lon, app.jd);
    let s = "";
    for (let b = 0; b < N_BODIES; b++) {
      const N = sky[b * 3], E = sky[b * 3 + 1], U = sky[b * 3 + 2];
      s += `<span class="body">${BODY_NAMES[b].padEnd(8)}</span>` +
           `${fmt(N)} ${fmt(E)} ${fmt(U)}\n`;
    }
    document.getElementById("matrix").innerHTML = s;
    const lab3 = applyOffset(app.weights, app.siren(sky));
    const rgb = labToSrgb(lab3[0], lab3[1], lab3[2]);
    document.getElementById("lab").textContent =
      `L* ${lab3[0].toFixed(1)}  a* ${lab3[1].toFixed(1)}  b* ${lab3[2].toFixed(1)}`;
    const hex = srgbToHex(rgb);
    document.getElementById("hex").textContent = hex;
    document.getElementById("swatch").style.background = hex;
    document.getElementById("pin-lat").textContent = app.pin.lat.toFixed(5);
    document.getElementById("pin-lon").textContent = app.pin.lon.toFixed(5);
  }
}
function fmt(x) { const s = x >= 0 ? " " : "-"; return s + Math.abs(x).toFixed(3); }

// ---- raycaster (Module 2.4 / Module 4) -------------------------------------
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function pickFromEvent(e) {
  if (!globe) return null;
  const r = renderer.domElement.getBoundingClientRect();
  ndc.x = ((e.clientX - r.left) / r.width) * 2 - 1;
  ndc.y = -((e.clientY - r.top) / r.height) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  const hit = raycaster.intersectObject(globe, false)[0];
  if (!hit) return null;
  const p = hit.point.clone().normalize();               // globe is unrotated -> world == object
  return { lat: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(p.y, -1, 1))),
           lon: THREE.MathUtils.radToDeg(Math.atan2(p.z, p.x)), vec: hit.point.clone() };
}
function setPin(pick) {
  app.pin.lat = pick.lat; app.pin.lon = pick.lon;
  const v = pick.vec.clone().normalize();
  reticle.position.copy(v.multiplyScalar(1.003));
  reticle.lookAt(0, 0, 0);
}

// ---- Module 3 controls -----------------------------------------------------
function speedFromSlider(v) { return v >= 0 ? Math.pow(10, v) : -Math.pow(10, -v); }
function fmtSpeed(m) {
  const a = Math.abs(m); const t = a >= 1e6 ? (a / 1e6).toFixed(0) + "M" :
    a >= 1000 ? (a / 1000).toFixed(0) + "k" : a.toFixed(a < 10 ? 1 : 0);
  return (m < 0 ? "-" : "") + t + "×";
}
function syncScrub() { document.getElementById("scrub").value = String(app.jd); }

function wireControls() {
  const playBtn = document.getElementById("play");
  playBtn.onclick = () => {
    app.playing = !app.playing;
    playBtn.textContent = app.playing ? "⏸ Pause" : "▶ Play";
    playBtn.classList.toggle("on", app.playing);
    badge();
  };
  const sp = document.getElementById("speed");
  sp.oninput = () => {
    app.speed = speedFromSlider(parseFloat(sp.value));
    document.getElementById("speed-val").textContent = fmtSpeed(app.speed);
    badge();
  };
  document.getElementById("now").onclick = () => { app.jd = nowJD(); syncScrub(); };

  const scrub = document.getElementById("scrub");
  scrub.min = String(JD_MIN); scrub.max = String(JD_MAX);
  scrub.oninput = () => { app.jd = parseFloat(scrub.value); };
  scrub.addEventListener("input", () => {
    const g = jdLabel(app.jd); document.getElementById("scrub-label").textContent = `${g.date} ${g.clock}`;
  });

  document.getElementById("go").onclick = () => {
    const v = document.getElementById("dt").value;    // YYYY-MM-DDTHH:MM:SS (treated as UTC)
    const m = v && v.match(/^(-?\d+)-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/);
    if (!m) { setNotice("enter a date/time"); return; }
    setNotice("");
    app.jd = gregorianToJD(+m[1], +m[2], +m[3], +m[4], +m[5], +(m[6] || 0));
    app.jd = Math.min(JD_MAX, Math.max(JD_MIN, app.jd));
    syncScrub();
  };

  document.getElementById("hud-toggle").onclick = () => {
    const h = document.getElementById("hud");
    h.classList.toggle("collapsed");
    document.getElementById("hud-toggle").textContent = h.classList.contains("collapsed") ? "▸" : "▾";
  };

  let dragging = false;
  renderer.domElement.addEventListener("pointerdown", (e) => {
    const pick = pickFromEvent(e); if (pick) { dragging = true; setPin(pick); }
  });
  renderer.domElement.addEventListener("pointermove", (e) => {
    if (dragging) { const pick = pickFromEvent(e); if (pick) setPin(pick); }
  });
  window.addEventListener("pointerup", () => { dragging = false; });

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}
function badge() {
  document.getElementById("speed-badge").textContent =
    `${fmtSpeed(app.speed)} · ${app.playing ? "playing" : "paused"}`;
}

// ---- bootstrap -------------------------------------------------------------
async function main() {
  wireControls();
  app.weights = await loadWeights();
  app.siren = makeSiren(app.weights);
  globe = buildGlobe(app.weights);
  // place the initial pin
  const lat = app.pin.lat * Math.PI / 180, lon = app.pin.lon * Math.PI / 180;
  setPin({ lat: app.pin.lat, lon: app.pin.lon,
           vec: new THREE.Vector3(Math.cos(lat) * Math.cos(lon), Math.sin(lat), Math.cos(lat) * Math.sin(lon)) });
  // seed the datetime picker with 'now'
  const g = jdToGregorian(app.jd), p = (x) => String(x).padStart(2, "0");
  document.getElementById("dt").value =
    `${p(g.y)}-${p(g.mo)}-${p(g.d)}T${p(g.h)}:${p(g.mi)}:${p(Math.min(59, g.s))}`;
  syncScrub();
  badge();
  animate();
  boot.style.opacity = "0"; setTimeout(() => boot.remove(), 700);
}

main();
