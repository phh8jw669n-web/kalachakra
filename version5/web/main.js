// main.js — version5 client (12-body "True Astrological Shape").
//
// Per update: fetch the ~1.6 KB /telemetry payload (12 bodies + GAST + obliquity) ->
// build the Zero-Redundancy [N,50] physical state — on the GPU via a WebGPU compute
// shader (skycompute.wgsl / gpucompute.js) when available, else the CPU skymath.js loop
// -> run the ONNX metric encoder (onnxruntime-web) which maps the 50-D state directly to
// 3 OKLab colours -> upload the field as a texture. The 3D orbits + sub-planetary glow
// read the 12 bodies' RA/Dec straight from telemetry. The render loop samples
// that neural field per-pixel, converts OKLab->sRGB on the GPU, paints a per-pixel
// 12-body sub-planetary glow, floats the 12 bodies as 3D sprites over their own glow,
// and interpolates everything smoothly between telemetry frames at a locked 60 FPS.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  telemetryToState, mlBodyState, localStateOne, bodyDirection, lerpAngle,
  makeGeoGrid, N_BODIES, STATE_DIM, BODY_NAMES,
} from "./skymath.js";
import { createGpuFeatureEngine } from "./gpucompute.js";

THREE.ColorManagement.enabled = false;

// ---- configuration ---------------------------------------------------------
// Neural inference grid (lon x lat). The shader upsamples with GPU bilinear
// filtering, so the on-screen field is smooth at any grid size — raise on WebGPU.
const GRID_W = 128, GRID_H = 64;
const N = GRID_W * GRID_H;
const LIVE_POLL_MS = 1000;
const CELESTIAL_R = 2.5;                          // radius of the 3D celestial-body shell
const CAM_START = 4.6;                            // frames the R=2.5 sphere at 60deg FOV
const EARTH_SOURCES = [
  "earth.jpg",
  "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg",
];
// distinct colours for the 12 bodies (Sun..Pluto, Mean Node, True Node)
const BODY_COLORS = [
  0xffd44a, 0xdfe6ef, 0xc9a24b, 0xf4c07a, 0xff5a4d, 0xf3b562,
  0xe8d9a0, 0x8fe0e6, 0x6ea8ff, 0xb98cff, 0x9aa7b8, 0x7f8ba0,
];
const BODY_SIZE = [                               // relative sprite scale
  0.16, 0.13, 0.07, 0.09, 0.09, 0.12,
  0.11, 0.09, 0.09, 0.07, 0.06, 0.06,
];

// ---- boot overlay ----------------------------------------------------------
const boot = document.createElement("div");
boot.id = "boot";
boot.textContent = "initialising celestial engine…";
document.body.appendChild(boot);
const setNotice = (t) => { document.getElementById("notice").textContent = t || ""; };

// ---- application state -----------------------------------------------------
const app = {
  mode: "LIVE", playing: true, simTime: new Date(),
  stepHours: 24.0, tickMs: 120, opacity: 0.68, glow: 0.9,
  backend: "…", hasModel: false,
  tzMode: "local", tzOffsetMin: 0,               // clock display timezone (display only)
};

// tiny localStorage wrapper (private mode / blocked storage must not break the UI)
const PREFS_KEY = "kalachakra_v5_ui";
function loadPrefs() { try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch { return {}; } }
function savePrefs(patch) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify({ ...loadPrefs(), ...patch })); } catch { /* ignore */ }
}
let session = null, engineLabel = "analytic fallback";

const grid = makeGeoGrid(GRID_W, GRID_H);
const stateBuf = new Float32Array(N * STATE_DIM);       // the [N,50] ONNX input
const rgbaA = new Uint8Array(N * 4), rgbaB = new Uint8Array(N * 4);

// ---- three.js scene --------------------------------------------------------
const wrap = document.getElementById("canvas-wrap");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x04060b);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight,
                                           0.1, 1000);
camera.position.set(0, 0, CAM_START);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = false;
controls.minDistance = 1.15;
controls.maxDistance = 12.0;

scene.add(makeStarfield());

// Earth (inner, opaque)
const earthMat = new THREE.MeshBasicMaterial({ color: 0x0d1b2a });
const earth = new THREE.Mesh(new THREE.SphereGeometry(0.99, 128, 128), earthMat);
scene.add(earth);
loadEarthTexture(earthMat);

// Energy shell (outer, transparent) — neural field + per-pixel 12-body glow
const texPrev = makeFieldTexture(rgbaA);
const texNext = makeFieldTexture(rgbaB);
let showing = { prev: texPrev, next: texNext };
let blendStart = performance.now(), blendDur = 500;

const fieldUniforms = {
  u_prev: { value: showing.prev },
  u_next: { value: showing.next },
  u_blend: { value: 1.0 },
  u_opacity: { value: app.opacity },
  u_glow: { value: app.glow },
  u_gast: { value: 0.0 },
  u_bodies: { value: Array.from({ length: N_BODIES }, () => new THREE.Vector2()) },
  u_bodyColor: { value: BODY_COLORS.map((c) => new THREE.Color(c)) },
};
const fieldMat = new THREE.ShaderMaterial({
  uniforms: fieldUniforms,
  transparent: true, depthWrite: false, blending: THREE.NormalBlending,
  defines: { NUM_BODIES: N_BODIES },
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D u_prev, u_next;
    uniform float u_blend, u_opacity, u_glow, u_gast;
    uniform vec2 u_bodies[NUM_BODIES];       // (RA, Dec) radians
    uniform vec3 u_bodyColor[NUM_BODIES];

    vec3 oklabToSrgb(vec3 c) {
      float L=c.x,a=c.y,b=c.z;
      float l_=L+0.3963377774*a+0.2158037573*b;
      float m_=L-0.1055613458*a-0.0638541728*b;
      float s_=L-0.0894841775*a-1.2914855480*b;
      float l=l_*l_*l_,m=m_*m_*m_,s=s_*s_*s_;
      vec3 lin=vec3( 4.0767416621*l-3.3077115913*m+0.2309699292*s,
                    -1.2684380046*l+2.6097574011*m-0.3413193965*s,
                    -0.0041960863*l-0.7034186147*m+1.7076147010*s);
      lin=clamp(lin,0.0,1.0);
      return mix(12.92*lin, 1.055*pow(lin,vec3(1.0/2.4))-0.055, step(0.0031308,lin));
    }
    vec3 decode(vec4 t){ return vec3(t.r, t.g*2.0-1.0, t.b*2.0-1.0); }

    void main() {
      // neural OKLab field (blended between two telemetry frames) -> sRGB
      vec3 lab = mix(decode(texture2D(u_prev,vUv)), decode(texture2D(u_next,vUv)), u_blend);
      vec3 col = oklabToSrgb(lab);

      // per-pixel 12-body spherical math: brighten each body's sub-planetary point
      float lat = (vUv.y - 0.5) * 3.141592653589793;
      float lon = (vUv.x - 0.5) * 6.283185307179586;
      float sphi = sin(lat), cphi = cos(lat);
      vec3 glow = vec3(0.0);
      for (int i = 0; i < NUM_BODIES; i++) {
        float H = (u_gast + lon) - u_bodies[i].x;
        float dec = u_bodies[i].y;
        float sinAlt = sphi*sin(dec) + cphi*cos(dec)*cos(H);   // altitude of body i
        glow += u_bodyColor[i] * smoothstep(0.9925, 1.0, sinAlt);  // ~7deg halo
      }
      col += glow * u_glow;

      float a = clamp(u_opacity + (glow.r+glow.g+glow.b)*0.25*u_glow, 0.0, 1.0);
      gl_FragColor = vec4(col, a);
    }`,
});
const shell = new THREE.Mesh(new THREE.SphereGeometry(1.0, 128, 128), fieldMat);
scene.add(shell);

// ---- 3D celestial bodies (Task 5) -----------------------------------------
const bodyGroup = new THREE.Group();
scene.add(bodyGroup);
const glowTex = makeGlowTexture();
const bodySprites = [], labelSprites = [];
for (let i = 0; i < N_BODIES; i++) {
  const mat = new THREE.SpriteMaterial({
    map: glowTex, color: BODY_COLORS[i], transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
  });
  const s = new THREE.Sprite(mat);
  s.scale.setScalar(BODY_SIZE[i] * 2.4);
  bodyGroup.add(s);
  bodySprites.push(s);
  const lab = makeLabelSprite(BODY_NAMES[i]);
  bodyGroup.add(lab);
  labelSprites.push(lab);
}

// ---- interpolated telemetry state -----------------------------------------
let statePrev = null, stateNext = null;

// ---- inference + field upload ---------------------------------------------
let gpu = null;                                   // WebGPU state engine (or null -> CPU)

// Fill stateBuf ([N,50]) for the whole grid: on the GPU when available, else CPU.
async function computeState(tstate) {
  if (gpu && gpu.ready) {
    await gpu.compute(tstate, stateBuf);          // 131k observers in parallel (WGSL)
  } else {
    const body44 = mlBodyState(tstate);           // 44 time-only dims, computed once
    for (let p = 0; p < N; p++) {
      localStateOne(body44, tstate, grid.lat[p], grid.lon[p], stateBuf, p * STATE_DIM);
    }
  }
}

async function inferOklab() {
  if (session) {
    const s = new ort.Tensor("float32", stateBuf, [N, STATE_DIM]);
    const res = await session.run({ state: s });
    return res[session.outputNames ? session.outputNames[0] : "oklab"].data;
  }
  return analyticFallback();
}

// Physics-only pseudo-colour before a model is trained, from the 50-D state: L from
// the Sun's ecliptic Z, hue from the location-dependent Ascendant vector (so it still
// varies across the globe). Bounded to the OKLab box.
function analyticFallback() {
  const out = new Float32Array(N * 3);
  for (let p = 0; p < N; p++) {
    const o = p * STATE_DIM;
    const sunZ = stateBuf[o + 2];                 // Sun ecliptic latitude ~ season proxy
    const ascX = stateBuf[o + 44], ascY = stateBuf[o + 45];
    out[p * 3 + 0] = 0.5 + 0.35 * sunZ;           // L
    out[p * 3 + 1] = 0.45 * ascX;                 // a
    out[p * 3 + 2] = 0.45 * ascY;                 // b
  }
  return out;
}

function encodeField(oklab, rgba) {
  for (let p = 0; p < N; p++) {
    rgba[p * 4 + 0] = clamp255(oklab[p * 3 + 0] * 255);
    rgba[p * 4 + 1] = clamp255((oklab[p * 3 + 1] * 0.5 + 0.5) * 255);
    rgba[p * 4 + 2] = clamp255((oklab[p * 3 + 2] * 0.5 + 0.5) * 255);
    rgba[p * 4 + 3] = 255;
  }
}

let updating = false;
async function updateToTime(date, smoothMs) {
  if (updating) return;
  updating = true;
  try {
    const r = await fetch("/telemetry?time=" + encodeURIComponent(date.toISOString()));
    if (!r.ok) { setNotice(`telemetry ${r.status}: ${await r.text()}`); return; }
    const tstate = telemetryToState(await r.json());
    setNotice("");
    await computeState(tstate);                   // WebGPU compute (or CPU fallback) -> stateBuf
    const oklab = await inferOklab();

    if (stateNext === null) {                     // first frame: prime both buffers
      encodeField(oklab, rgbaA); encodeField(oklab, rgbaB);
      texPrev.needsUpdate = texNext.needsUpdate = true;
      statePrev = stateNext = tstate;
      fieldUniforms.u_blend.value = 1.0;
      return;
    }
    fieldUniforms.u_blend.value = 1.0;            // finish any running blend
    const incoming = showing.prev;
    encodeField(oklab, incoming === texPrev ? rgbaA : rgbaB);
    incoming.needsUpdate = true;
    showing = { prev: showing.next, next: incoming };
    fieldUniforms.u_prev.value = showing.prev;
    fieldUniforms.u_next.value = showing.next;
    statePrev = stateNext; stateNext = tstate;
    blendStart = performance.now();
    blendDur = Math.max(1, smoothMs ?? 500);
  } catch (e) {
    setNotice("update error: " + e.message);
  } finally {
    updating = false;
  }
}

// ---- render loop -----------------------------------------------------------
let lastFrame = performance.now(), fpsAcc = 0, fpsN = 0;
const _dir = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = now - lastFrame; lastFrame = now;

  const t = Math.min(1, (now - blendStart) / blendDur);
  fieldUniforms.u_blend.value = t;

  if (stateNext) {
    const gast = statePrev ? lerpAngle(statePrev.gast, stateNext.gast, t) : stateNext.gast;
    fieldUniforms.u_gast.value = gast;
    for (let i = 0; i < N_BODIES; i++) {
      const ra = statePrev ? lerpAngle(statePrev.ra[i], stateNext.ra[i], t) : stateNext.ra[i];
      const dec = statePrev ? statePrev.dec[i] + (stateNext.dec[i] - statePrev.dec[i]) * t
                            : stateNext.dec[i];
      fieldUniforms.u_bodies.value[i].set(ra, dec);
      const d = bodyDirection(ra, dec, gast, CELESTIAL_R);
      bodySprites[i].position.set(d.x, d.y, d.z);
      _dir.set(d.x, d.y, d.z).multiplyScalar(1.14);
      labelSprites[i].position.copy(_dir);
    }
  }

  controls.update();
  if (app.mode === "LIVE") app.simTime = new Date();
  updateClock();
  renderer.render(scene, camera);

  fpsAcc += dt; fpsN++;
  if (fpsAcc > 500) {
    document.getElementById("fps").textContent = (1000 * fpsN / fpsAcc).toFixed(0) + " fps";
    fpsAcc = 0; fpsN = 0;
  }
}

// ---- time driver -----------------------------------------------------------
// Sequential drivers: the clock only advances once the previous grid inference +
// WebGL upload has FULLY completed (await), so simulation time can never outrun the
// GPU and desync at high resolutions (512x256). Recursive setTimeout, not setInterval.
// `driverToken` is bumped on every stop; a loop that resumes from its await after a
// restart sees a stale token and bows out, so two drivers can never run at once.
let liveTimer = null, playTimer = null, driverToken = 0;

async function playbackLoop(token) {
  if (token !== driverToken || app.mode !== "TIMELINE" || !app.playing) return;
  // advance time, then WAIT for the whole grid update before scheduling the next tick
  app.simTime = new Date(app.simTime.getTime() + app.stepHours * 3600e3);
  await updateToTime(app.simTime, app.tickMs);
  if (token === driverToken && playTimer !== null) {
    playTimer = setTimeout(() => playbackLoop(token), app.tickMs);
  }
}

async function liveLoop(token) {
  if (token !== driverToken || app.mode !== "LIVE") return;
  await updateToTime(new Date(), LIVE_POLL_MS);
  if (token === driverToken && liveTimer !== null) {
    liveTimer = setTimeout(() => liveLoop(token), LIVE_POLL_MS);
  }
}

function startPlayback() {
  stopTimers();
  const token = driverToken;
  playTimer = setTimeout(() => playbackLoop(token), 0);       // start immediately
}

function startLive() {
  stopTimers();
  const token = driverToken;
  // prime immediately, then begin sequential polling once that first frame lands
  updateToTime(new Date(), 600).then(() => {
    if (token === driverToken) liveTimer = setTimeout(() => liveLoop(token), LIVE_POLL_MS);
  });
}

function stopTimers() {
  driverToken++;                                              // invalidate any in-flight loop
  if (liveTimer) clearTimeout(liveTimer), liveTimer = null;
  if (playTimer) clearTimeout(playTimer), playTimer = null;
}

// Play/pause must start & stop the loop itself: unlike setInterval, a recursive
// timeout that returns on !playing does not resume on its own.
function togglePlay() {
  if (app.mode !== "TIMELINE") return;
  setPlay(!app.playing);
  if (app.playing) startPlayback(); else stopTimers();
}

function stepOnce(dir) {
  app.simTime = new Date(app.simTime.getTime() + dir * app.stepHours * 3600e3);
  updateToTime(app.simTime, 350);
  updateClock();
}

// ---- UI --------------------------------------------------------------------
function enterMode(mode) {
  app.mode = mode;
  const isLive = mode === "LIVE";
  document.getElementById("clock").className = isLive ? "live" : "timeline";
  const badge = document.getElementById("mode-badge");
  badge.className = isLive ? "live" : "timeline";
  badge.textContent = isLive ? "LIVE" : "TIME MACHINE";
  document.getElementById("btn-mode").textContent = isLive ? "Enter Time Machine" : "Return to Live";
  document.getElementById("btn-play").disabled = isLive;
  if (isLive) startLive();
  else { app.simTime = new Date(); app.playing = true; setPlay(true); startPlayback(); }
}
function setPlay(p) {
  app.playing = p;
  document.getElementById("btn-play").textContent = p ? "⏸ Pause" : "▶ Play";
}
// display-only timezone offset (minutes). "local" tracks the browser zone for the
// instant being shown; a fixed choice is a constant offset. Never affects the UTC
// timestamps sent to /telemetry — purely how the clock is written.
function tzOffsetMin() {
  return app.tzMode === "local" ? -app.simTime.getTimezoneOffset() : app.tzOffsetMin;
}
function tzSuffix(off) {
  if (app.tzMode === "local") return "Local";
  if (off === 0) return "UTC";
  if (off === 330) return "IST";
  const s = off < 0 ? "-" : "+", a = Math.abs(off);
  return `UTC${s}${String((a / 60) | 0).padStart(2, "0")}:${String(a % 60).padStart(2, "0")}`;
}
function updateClock() {
  const off = tzOffsetMin();
  const d = new Date(app.simTime.getTime() + off * 60000);   // shift so getUTC* = wall clock
  const p = (x) => String(x).padStart(2, "0");
  document.getElementById("clock").textContent =
    `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())} ${tzSuffix(off)}`;
  document.getElementById("date").textContent = fmtDate(d);
}

// ---- item 1: timezone selector --------------------------------------------
// Common world offsets in minutes (incl. the half/quarter-hour zones).
const TZ_OFFSETS = [
  -720, -660, -600, -570, -540, -480, -420, -360, -300, -240, -210, -180, -120, -60,
  0, 60, 120, 180, 210, 240, 270, 300, 330, 345, 360, 390, 420, 480, 525, 540, 570,
  600, 630, 660, 720, 765, 780, 840,
];
const TZ_NAMES = { 0: "UTC", 330: "IST", 345: "NPT", 210: "IRST", 270: "AFT" };
function tzLabel(off) {
  const s = off < 0 ? "-" : "+", a = Math.abs(off);
  const base = `UTC${s}${String((a / 60) | 0).padStart(2, "0")}:${String(a % 60).padStart(2, "0")}`;
  return TZ_NAMES[off] ? `${base} · ${TZ_NAMES[off]}` : base;
}
function buildTzSelect() {
  const sel = document.getElementById("tz");
  const opt = (v, t) => { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); };
  opt("local", "Local (this device)");
  for (const off of TZ_OFFSETS) opt(String(off), off === 0 ? "UTC" : tzLabel(off));
  const pref = loadPrefs().tz;
  sel.value = pref ?? "local";
  applyTz(sel.value);
  sel.onchange = () => { applyTz(sel.value); savePrefs({ tz: sel.value }); };
}
function applyTz(value) {
  if (value === "local") { app.tzMode = "local"; }
  else { app.tzMode = "fixed"; app.tzOffsetMin = parseInt(value, 10); }
  updateClock();
}

// ---- item 2: draggable + hideable panel -----------------------------------
function setupPanel() {
  const panel = document.getElementById("panel");
  const header = document.getElementById("panel-header");
  const hideBtn = document.getElementById("panel-hide");
  const showBtn = document.getElementById("panel-show");
  const prefs = loadPrefs();

  // restore parked position (clamped into view), else default bottom-left
  function place(left, top) {
    const maxL = window.innerWidth - panel.offsetWidth - 8;
    const maxT = window.innerHeight - 44;                     // keep the header grabbable
    panel.style.left = Math.max(8, Math.min(maxL, left)) + "px";
    panel.style.top = Math.max(8, Math.min(maxT, top)) + "px";
    panel.style.bottom = "auto";
  }
  if (prefs.px != null && prefs.py != null) place(prefs.px, prefs.py);
  else place(22, window.innerHeight - panel.offsetHeight - 60);
  if (prefs.hidden) panel.classList.add("hidden");

  // drag by the header (pointer events cover mouse + touch)
  let dragging = false, ox = 0, oy = 0;
  header.addEventListener("pointerdown", (e) => {
    if (e.target === hideBtn) return;
    dragging = true;
    ox = e.clientX - panel.offsetLeft;
    oy = e.clientY - panel.offsetTop;
    header.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  header.addEventListener("pointermove", (e) => { if (dragging) place(e.clientX - ox, e.clientY - oy); });
  header.addEventListener("pointerup", (e) => {
    if (!dragging) return;
    dragging = false;
    header.releasePointerCapture(e.pointerId);
    savePrefs({ px: panel.offsetLeft, py: panel.offsetTop });
  });

  hideBtn.onclick = () => { panel.classList.add("hidden"); savePrefs({ hidden: true }); };
  showBtn.onclick = () => { panel.classList.remove("hidden"); savePrefs({ hidden: false }); };
  window.addEventListener("resize", () => place(panel.offsetLeft, panel.offsetTop));
}

// ---- item 3: zoom (buttons + keys; mouse wheel stays with OrbitControls) ---
function zoom(factor) {
  const t = controls.target;
  const offset = camera.position.clone().sub(t);
  const dist = Math.max(controls.minDistance,
                        Math.min(controls.maxDistance, offset.length() * factor));
  camera.position.copy(t).add(offset.setLength(dist));
  controls.update();
}

function wireUI() {
  buildTzSelect();
  setupPanel();
  document.getElementById("zoom-in").onclick = () => zoom(0.82);
  document.getElementById("zoom-out").onclick = () => zoom(1.22);
  document.getElementById("btn-mode").onclick = () =>
    enterMode(app.mode === "LIVE" ? "TIMELINE" : "LIVE");
  document.getElementById("btn-live").onclick = () => enterMode("LIVE");
  document.getElementById("btn-play").onclick = () => togglePlay();
  document.getElementById("btn-back").onclick = () => stepOnce(-1);
  document.getElementById("btn-fwd").onclick = () => stepOnce(+1);
  document.querySelectorAll("#steps button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel"));
      b.classList.add("sel");
      app.stepHours = parseFloat(b.dataset.h);
      document.getElementById("step-input").value = app.stepHours;
    };
  });
  document.getElementById("step-input").oninput = (e) => {
    const v = parseFloat(e.target.value);
    if (!Number.isNaN(v)) app.stepHours = v;
    document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel"));
  };
  const op = document.getElementById("opacity");
  op.oninput = () => {
    app.opacity = parseFloat(op.value);
    fieldUniforms.u_opacity.value = app.opacity;
    document.getElementById("opacity-val").textContent = app.opacity.toFixed(2);
  };
  const gl = document.getElementById("glow");
  if (gl) gl.onchange = () => { app.glow = gl.checked ? 0.9 : 0.0; fieldUniforms.u_glow.value = app.glow; };
  const lb = document.getElementById("labels");
  if (lb) lb.onchange = () => labelSprites.forEach((s) => (s.visible = lb.checked));

  window.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.key === "l" || e.key === "L") enterMode("LIVE");
    else if (e.key === "r" || e.key === "R") { controls.reset(); camera.position.set(0, 0, CAM_START); }
    else if (e.key === "ArrowLeft" && app.mode === "TIMELINE") stepOnce(-1);
    else if (e.key === "ArrowRight" && app.mode === "TIMELINE") stepOnce(+1);
    else if (e.key === "ArrowUp") { e.preventDefault(); zoom(0.9); }
    else if (e.key === "ArrowDown") { e.preventDefault(); zoom(1.1); }
  });
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

// ---- bootstrap -------------------------------------------------------------
async function main() {
  wireUI();
  animate();
  try {
    const info = await (await fetch("/api/info")).json();
    app.backend = info.backend; app.hasModel = info.has_model;
    document.getElementById("backend").textContent = "backend " + info.backend;
  } catch { setNotice("cannot reach server /api/info"); }

  // Build the 50-D physical state on the GPU (skycompute.wgsl). Its own GPUDevice,
  // separate from the Three.js WebGL context; null -> CPU fallback path.
  try {
    gpu = await createGpuFeatureEngine({
      gridW: GRID_W, gridH: GRID_H, stateDim: STATE_DIM, shaderUrl: "skycompute.wgsl",
    });
  } catch { gpu = null; }
  document.getElementById("fps").title = gpu ? "features: WebGPU compute" : "features: CPU";

  if (app.hasModel && typeof ort !== "undefined") {
    try {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
      session = await ort.InferenceSession.create("model_v5.onnx",
        { executionProviders: ["webgpu", "wasm"], graphOptimizationLevel: "all" });
      engineLabel = "ONNX · webgpu/wasm";
      await verifyGolden();
    } catch (e) {
      setNotice("ONNX load failed, using analytic fallback: " + e.message);
      session = null;
    }
  } else if (!app.hasModel) {
    setNotice("no model_v5.onnx yet — showing analytic fallback (train + export to enable the neural field)");
  }
  const feat = gpu ? " · feat:WebGPU" : " · feat:CPU";
  document.getElementById("engine").textContent =
    (session ? engineLabel : "analytic fallback") + feat;

  enterMode("LIVE");
  boot.style.opacity = "0";
  setTimeout(() => boot.remove(), 700);
}

// Prove in-browser that the JS math matches the server: rebuild golden.json's 50-D
// state from its telemetry and compare.
async function verifyGolden() {
  try {
    const g = await (await fetch("golden.json")).json();
    const ts = telemetryToState(g.telemetry);
    const body44 = mlBodyState(ts);
    const s = new Float32Array(STATE_DIM);
    let maxErr = 0;
    for (const p of g.points) {
      localStateOne(body44, ts, p.lat_deg * Math.PI / 180, p.lon_deg * Math.PI / 180, s, 0);
      for (let k = 0; k < STATE_DIM; k++) maxErr = Math.max(maxErr, Math.abs(s[k] - p.state[k]));
    }
    console.log(`[version5.1] JS<->server state parity: max abs err = ${maxErr.toExponential(2)} ` +
                (maxErr < 1e-4 ? "PASS" : "FAIL"));
  } catch { /* golden.json optional */ }
}

// ---- helpers ---------------------------------------------------------------
function clamp255(x) { return x < 0 ? 0 : x > 255 ? 255 : x | 0; }

function makeFieldTexture(data) {
  const tex = new THREE.DataTexture(data, GRID_W, GRID_H, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.NoColorSpace;
  tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping; tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false; tex.needsUpdate = true;
  return tex;
}

function makeGlowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d");
  const rg = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  rg.addColorStop(0, "rgba(255,255,255,1)");
  rg.addColorStop(0.25, "rgba(255,255,255,0.85)");
  rg.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = rg; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.NoColorSpace; return t;
}

function makeLabelSprite(text) {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.font = "600 30px ui-monospace, monospace";
  g.fillStyle = "rgba(220,235,250,0.92)";
  g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(text, 128, 34);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.NoColorSpace;
  const s = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, transparent: true, depthTest: false, depthWrite: false }));
  s.scale.set(0.42, 0.105, 1);
  return s;
}

function loadEarthTexture(mat, idx = 0) {
  if (idx >= EARTH_SOURCES.length) {
    mat.map = proceduralEarth(); mat.color.set(0xffffff); mat.needsUpdate = true; return;
  }
  new THREE.TextureLoader().setCrossOrigin("anonymous").load(
    EARTH_SOURCES[idx],
    (tex) => { tex.colorSpace = THREE.NoColorSpace; mat.map = tex; mat.color.set(0xffffff); mat.needsUpdate = true; },
    undefined,
    () => loadEarthTexture(mat, idx + 1));
}

function proceduralEarth() {
  const c = document.createElement("canvas"); c.width = 512; c.height = 256;
  const g = c.getContext("2d");
  const grd = g.createLinearGradient(0, 0, 0, 256);
  grd.addColorStop(0, "#0a1420"); grd.addColorStop(0.5, "#12324a"); grd.addColorStop(1, "#0a1420");
  g.fillStyle = grd; g.fillRect(0, 0, 512, 256);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.NoColorSpace; return t;
}

function makeStarfield() {
  const n = 1400, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 60 + Math.random() * 40;
    const th = Math.acos(2 * Math.random() - 1), ph = Math.random() * 2 * Math.PI;
    pos[i * 3] = r * Math.sin(th) * Math.cos(ph);
    pos[i * 3 + 1] = r * Math.sin(th) * Math.sin(ph);
    pos[i * 3 + 2] = r * Math.cos(th);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(geo, new THREE.PointsMaterial({ color: 0x8aa2c0, size: 0.35 }));
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
function fmtDate(d) {
  const y = d.getUTCFullYear(), era = y > 0 ? "CE" : "BCE", yy = y > 0 ? y : 1 - y;
  return `${WEEK[d.getUTCDay()]}, ${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${yy} ${era}`;
}

main();
