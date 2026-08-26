// main.js — version5 client. Decoupled celestial-weather visualiser.
//
// Pipeline per update: fetch the ~2 KB /telemetry payload -> build the [N,10,5]
// local-sky tensor for a lon/lat grid with the SAME maths the server trained on
// (skymath.js) -> run the ONNX encoder on the GPU (onnxruntime-web) -> upload the
// resulting OKLab field as a texture. The render loop then samples that neural field
// per-pixel, converts OKLab->sRGB on the GPU, and composites it over the Earth at a
// locked frame rate — entirely independent of how often telemetry arrives.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  telemetryToEquatorial, localFeaturesOne, oklabToSrgb, makeGeoGrid,
  N_BODIES, RAW_FEATURES,
} from "./skymath.js";

THREE.ColorManagement.enabled = false;           // shaders output display sRGB directly

// ---- configuration ---------------------------------------------------------
// Neural inference grid (lon x lat). The shader upsamples it with GPU bilinear
// filtering, so the on-screen field is smooth (no mesh staircasing) at any grid
// size — raise this for a crisper terminator when running on WebGPU, lower it if
// you are stuck on the single-threaded wasm fallback.
const GRID_W = 128, GRID_H = 64;
const N = GRID_W * GRID_H;
const LIVE_POLL_MS = 1000;                        // telemetry refresh in LIVE mode
const EARTH_SOURCES = [
  "earth.jpg",                                    // optional same-origin drop-in
  "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg",
];

// ---- boot overlay ----------------------------------------------------------
const boot = document.createElement("div");
boot.id = "boot";
boot.textContent = "initialising celestial engine…";
document.body.appendChild(boot);
const setNotice = (t) => { document.getElementById("notice").textContent = t || ""; };

// ---- application state -----------------------------------------------------
const state = {
  mode: "LIVE",                                   // "LIVE" | "TIMELINE"
  playing: true,
  simTime: new Date(),                            // active simulated instant (UTC)
  stepHours: 24.0,
  tickMs: 120,
  opacity: 0.68,
  backend: "…",
  hasModel: false,
};

let session = null;                               // ort.InferenceSession | null
let engineLabel = "analytic fallback";

// grid geometry (equal-angle in lat so it maps 1:1 to the equirectangular sphere UV)
const grid = makeGeoGrid(GRID_W, GRID_H);
const featBuf = new Float32Array(N * N_BODIES * RAW_FEATURES);
const rgbaA = new Uint8Array(N * 4);
const rgbaB = new Uint8Array(N * 4);

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
camera.position.set(0, 0, 2.5);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = false;
controls.minDistance = 1.15;
controls.maxDistance = 6.0;

scene.add(makeStarfield());

// -- Layer 1: physical Earth (inner, opaque) --
const earthMat = new THREE.MeshBasicMaterial({ color: 0x0d1b2a });
const earth = new THREE.Mesh(new THREE.SphereGeometry(0.99, 128, 128), earthMat);
scene.add(earth);
loadEarthTexture(earthMat);

// -- Layer 2: the celestial energy shell (outer, transparent, OKLab->sRGB shader) --
const texPrev = makeFieldTexture(rgbaA);
const texNext = makeFieldTexture(rgbaB);
let showing = { prev: texPrev, next: texNext };   // which texture holds which frame
let blendStart = performance.now();
let blendDur = 500;

const fieldUniforms = {
  u_prev: { value: showing.prev },
  u_next: { value: showing.next },
  u_blend: { value: 1.0 },
  u_opacity: { value: state.opacity },
};
const fieldMat = new THREE.ShaderMaterial({
  uniforms: fieldUniforms,
  transparent: true,
  depthWrite: false,
  blending: THREE.NormalBlending,
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`,
  fragmentShader: /* glsl */`
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D u_prev;
    uniform sampler2D u_next;
    uniform float u_blend;
    uniform float u_opacity;

    // OKLab -> linear sRGB (Bjorn Ottosson) -> gamma. Runs per pixel on the GPU.
    vec3 oklabToSrgb(vec3 c) {
      float L = c.x, a = c.y, b = c.z;
      float l_ = L + 0.3963377774*a + 0.2158037573*b;
      float m_ = L - 0.1055613458*a - 0.0638541728*b;
      float s_ = L - 0.0894841775*a - 1.2914855480*b;
      float l = l_*l_*l_, m = m_*m_*m_, s = s_*s_*s_;
      vec3 lin = vec3(
        4.0767416621*l - 3.3077115913*m + 0.2309699292*s,
       -1.2684380046*l + 2.6097574011*m - 0.3413193965*s,
       -0.0041960863*l - 0.7034186147*m + 1.7076147010*s);
      lin = clamp(lin, 0.0, 1.0);
      return mix(12.92*lin, 1.055*pow(lin, vec3(1.0/2.4)) - 0.055,
                 step(0.0031308, lin));
    }
    // texel (R=L, G=(a+1)/2, B=(b+1)/2) -> OKLab
    vec3 decode(vec4 t) { return vec3(t.r, t.g*2.0-1.0, t.b*2.0-1.0); }

    void main() {
      vec3 lab = mix(decode(texture2D(u_prev, vUv)),
                     decode(texture2D(u_next, vUv)), u_blend);
      gl_FragColor = vec4(oklabToSrgb(lab), u_opacity);
    }`,
});
const shell = new THREE.Mesh(new THREE.SphereGeometry(1.0, 128, 128), fieldMat);
scene.add(shell);

// ---- inference + field upload ---------------------------------------------
function buildFeatures(eq) {
  for (let p = 0; p < N; p++) {
    localFeaturesOne(eq, grid.lat[p], grid.lon[p], featBuf,
                     p * N_BODIES * RAW_FEATURES);
  }
  return featBuf;
}

async function inferOklab(features) {
  if (session) {
    const inName = session.inputNames ? session.inputNames[0] : "features";
    const outName = session.outputNames ? session.outputNames[0] : "oklab";
    const tensor = new ort.Tensor("float32", features, [N, N_BODIES, RAW_FEATURES]);
    const res = await session.run({ [inName]: tensor });
    return res[outName].data;                     // Float32Array [N*3]
  }
  return analyticFallback(features);
}

// A physics-only pseudo-colour so the globe is alive before a model is trained:
// lightness from the Sun's altitude (day/night), hue from the Moon & Jupiter.
function analyticFallback(features) {
  const out = new Float32Array(N * 3);
  const S = N_BODIES * RAW_FEATURES;
  for (let p = 0; p < N; p++) {
    const o = p * S;
    const sunAlt = features[o + 0 * RAW_FEATURES + 0];
    const moonAlt = features[o + 1 * RAW_FEATURES + 0];
    const moonAz = features[o + 1 * RAW_FEATURES + 1];
    const jupAlt = features[o + 5 * RAW_FEATURES + 0];
    const jupAz = features[o + 5 * RAW_FEATURES + 1];
    const day = Math.max(0, Math.min(1, (sunAlt + 0.15) / 0.6));
    out[p * 3 + 0] = 0.12 + 0.78 * day;                        // L
    out[p * 3 + 1] = 0.32 * Math.sin(moonAz) * Math.cos(moonAlt);  // a
    out[p * 3 + 2] = 0.32 * Math.sin(jupAz) * Math.cos(jupAlt);    // b
  }
  return out;
}

function encodeField(oklab, rgba) {
  for (let p = 0; p < N; p++) {
    let L = oklab[p * 3 + 0], a = oklab[p * 3 + 1], b = oklab[p * 3 + 2];
    rgba[p * 4 + 0] = clamp255(L * 255);
    rgba[p * 4 + 1] = clamp255((a * 0.5 + 0.5) * 255);
    rgba[p * 4 + 2] = clamp255((b * 0.5 + 0.5) * 255);
    rgba[p * 4 + 3] = 255;
  }
}

let updating = false;
async function updateToTime(date, smoothMs) {
  if (updating) return;                           // drop overlapping requests
  updating = true;
  try {
    const iso = isoUTC(date);
    const r = await fetch("/telemetry?time=" + encodeURIComponent(iso));
    if (!r.ok) { setNotice(`telemetry ${r.status}: ${await r.text()}`); return; }
    const tel = await r.json();
    setNotice("");
    const eq = telemetryToEquatorial(tel);
    const feats = buildFeatures(eq);
    const oklab = await inferOklab(feats);

    // finish any in-flight blend, then swap prev<-next and write the new frame
    fieldUniforms.u_blend.value = 1.0;
    const incoming = showing.prev;                // safe to overwrite (was old frame)
    const rgba = (incoming === texPrev) ? rgbaA : rgbaB;
    encodeField(oklab, rgba);
    incoming.needsUpdate = true;
    showing = { prev: showing.next, next: incoming };
    fieldUniforms.u_prev.value = showing.prev;
    fieldUniforms.u_next.value = showing.next;
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
function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = now - lastFrame; lastFrame = now;

  fieldUniforms.u_blend.value = Math.min(1, (now - blendStart) / blendDur);
  controls.update();

  // the LIVE clock ticks off the real system clock every frame for a smooth readout
  if (state.mode === "LIVE") state.simTime = new Date();
  updateClock();

  renderer.render(scene, camera);

  fpsAcc += dt; fpsN++;
  if (fpsAcc > 500) {
    document.getElementById("fps").textContent = (1000 * fpsN / fpsAcc).toFixed(0) + " fps";
    fpsAcc = 0; fpsN = 0;
  }
}

// ---- time driver (LIVE poll / TIMELINE playback) ---------------------------
let liveTimer = null, playTimer = null;
function startLive() {
  stopTimers();
  updateToTime(new Date(), 600);
  liveTimer = setInterval(() => { if (state.mode === "LIVE") updateToTime(new Date(), LIVE_POLL_MS); },
                          LIVE_POLL_MS);
}
function startPlayback() {
  stopTimers();
  playTimer = setInterval(() => {
    if (state.mode !== "TIMELINE" || !state.playing) return;
    state.simTime = new Date(state.simTime.getTime() + state.stepHours * 3600e3);
    updateToTime(state.simTime, state.tickMs);
  }, state.tickMs);
}
function stopTimers() {
  if (liveTimer) clearInterval(liveTimer), liveTimer = null;
  if (playTimer) clearInterval(playTimer), playTimer = null;
}
function stepOnce(dir) {
  state.simTime = new Date(state.simTime.getTime() + dir * state.stepHours * 3600e3);
  updateToTime(state.simTime, 350);
  updateClock();
}

// ---- UI wiring -------------------------------------------------------------
function enterMode(mode) {
  state.mode = mode;
  const isLive = mode === "LIVE";
  document.getElementById("clock").className = isLive ? "live" : "timeline";
  const badge = document.getElementById("mode-badge");
  badge.className = isLive ? "live" : "timeline";
  badge.textContent = isLive ? "LIVE" : "TIME MACHINE";
  document.getElementById("btn-mode").textContent =
    isLive ? "Enter Time Machine" : "Return to Live";
  document.getElementById("btn-play").disabled = isLive;
  if (isLive) startLive(); else { state.simTime = new Date(); state.playing = true; setPlay(true); startPlayback(); }
}
function setPlay(p) {
  state.playing = p;
  document.getElementById("btn-play").textContent = p ? "⏸ Pause" : "▶ Play";
}
function updateClock() {
  const d = state.simTime;
  const hh = String(d.getUTCHours()).padStart(2, "0");
  const mm = String(d.getUTCMinutes()).padStart(2, "0");
  const ss = String(d.getUTCSeconds()).padStart(2, "0");
  document.getElementById("clock").textContent = `${hh}:${mm}:${ss} UTC`;
  document.getElementById("date").textContent = fmtDate(d);
}

function wireUI() {
  document.getElementById("btn-mode").onclick = () =>
    enterMode(state.mode === "LIVE" ? "TIMELINE" : "LIVE");
  document.getElementById("btn-live").onclick = () => enterMode("LIVE");
  document.getElementById("btn-play").onclick = () => {
    setPlay(!state.playing);
  };
  document.getElementById("btn-back").onclick = () => stepOnce(-1);
  document.getElementById("btn-fwd").onclick = () => stepOnce(+1);

  document.querySelectorAll("#steps button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel"));
      b.classList.add("sel");
      state.stepHours = parseFloat(b.dataset.h);
      document.getElementById("step-input").value = state.stepHours;
    };
  });
  document.getElementById("step-input").oninput = (e) => {
    const v = parseFloat(e.target.value);
    if (!Number.isNaN(v)) state.stepHours = v;
    document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("sel"));
  };
  const op = document.getElementById("opacity");
  op.oninput = () => {
    state.opacity = parseFloat(op.value);
    fieldUniforms.u_opacity.value = state.opacity;
    document.getElementById("opacity-val").textContent = state.opacity.toFixed(2);
  };

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); if (state.mode === "TIMELINE") setPlay(!state.playing); }
    else if (e.key === "l" || e.key === "L") enterMode("LIVE");
    else if (e.key === "r" || e.key === "R") { controls.reset(); camera.position.set(0, 0, 2.5); }
    else if (e.key === "ArrowLeft") { if (state.mode === "TIMELINE") stepOnce(-1); }
    else if (e.key === "ArrowRight") { if (state.mode === "TIMELINE") stepOnce(+1); }
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

  // server capabilities
  try {
    const info = await (await fetch("/api/info")).json();
    state.backend = info.backend;
    state.hasModel = info.has_model;
    document.getElementById("backend").textContent = "backend " + info.backend;
  } catch { setNotice("cannot reach server /api/info"); }

  // ONNX model (optional — analytic fallback if absent)
  if (state.hasModel && typeof ort !== "undefined") {
    try {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
      session = await ort.InferenceSession.create("model_v5.onnx",
        { executionProviders: ["webgpu", "wasm"], graphOptimizationLevel: "all" });
      engineLabel = "ONNX · " + (session.handler?._ep || "webgpu/wasm");
      await verifyGolden();
    } catch (e) {
      setNotice("ONNX load failed, using analytic fallback: " + e.message);
      session = null;
    }
  } else if (!state.hasModel) {
    setNotice("no model_v5.onnx yet — showing analytic fallback (train + export to enable the neural field)");
  }
  document.getElementById("engine").textContent = session ? engineLabel : "analytic fallback";

  enterMode("LIVE");
  boot.style.opacity = "0";
  setTimeout(() => boot.remove(), 700);
}

// Prove, in-browser, that the JS spherical math matches the server that trained the
// model: rebuild golden.json's features from its telemetry and compare.
async function verifyGolden() {
  try {
    const g = await (await fetch("golden.json")).json();
    const eq = telemetryToEquatorial(g.telemetry);
    let maxErr = 0;
    for (const p of g.points) {
      const out = new Float32Array(N_BODIES * RAW_FEATURES);
      localFeaturesOne(eq, p.lat_deg * Math.PI / 180, p.lon_deg * Math.PI / 180, out);
      for (let i = 0; i < N_BODIES; i++)
        for (let f = 0; f < RAW_FEATURES; f++)
          maxErr = Math.max(maxErr, Math.abs(out[i * RAW_FEATURES + f] - p.features[i][f]));
    }
    console.log(`[version5] JS<->server feature parity: max abs err = ${maxErr.toExponential(2)} ` +
                (maxErr < 1e-4 ? "PASS" : "FAIL"));
  } catch { /* golden.json optional */ }
}

// ---- helpers ---------------------------------------------------------------
function clamp255(x) { return x < 0 ? 0 : x > 255 ? 255 : x | 0; }

function makeFieldTexture(data) {
  const tex = new THREE.DataTexture(data, GRID_W, GRID_H, THREE.RGBAFormat,
                                    THREE.UnsignedByteType);
  tex.colorSpace = THREE.NoColorSpace;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.RepeatWrapping;               // longitude wraps seamlessly
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

function loadEarthTexture(mat, idx = 0) {
  if (idx >= EARTH_SOURCES.length) {
    mat.map = proceduralEarth(); mat.color.set(0xffffff); mat.needsUpdate = true; return;
  }
  new THREE.TextureLoader().setCrossOrigin("anonymous").load(
    EARTH_SOURCES[idx],
    (tex) => { tex.colorSpace = THREE.NoColorSpace; mat.map = tex; mat.color.set(0xffffff); mat.needsUpdate = true; },
    undefined,
    () => loadEarthTexture(mat, idx + 1),
  );
}

function proceduralEarth() {
  const c = document.createElement("canvas");
  c.width = 512; c.height = 256;
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

// UTC ISO for any year (JS toISOString handles the whole BCE..CE range; the server's
// parser accepts both the 4-digit and the expanded ±6-digit forms).
function isoUTC(d) { return d.toISOString(); }

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEK = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
function fmtDate(d) {
  let y = d.getUTCFullYear();
  const era = y > 0 ? "CE" : "BCE";
  const yy = y > 0 ? y : 1 - y;
  return `${WEEK[d.getUTCDay()]}, ${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${yy} ${era}`;
}

main();
