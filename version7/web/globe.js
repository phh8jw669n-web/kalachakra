// globe.js — the v7 render primitive: a texture-mapped globe.
//
// The expensive SIREN field is baked into an equirectangular DataTexture (by the worker);
// this module just maps that texture onto a sphere. The fragment shader is a single texture
// lookup (plus asin/atan for the uv) — no SIREN, no ephemeris, no transcendental sine sweep —
// so it renders at 60 fps at any zoom or window size. Camera moves are free (the texture
// doesn't change); only a new Julian Date triggers a re-bake.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const D2R = Math.PI / 180;

export function llToVec(latDeg, lonDeg, r = 1) {
  const la = latDeg * D2R, lo = lonDeg * D2R;
  return new THREE.Vector3(Math.cos(la) * Math.cos(lo), Math.sin(la), Math.cos(la) * Math.sin(lo)).multiplyScalar(r);
}

export function createGlobe(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x04060b);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.001, 100);
  camera.position.set(0, 0.7, 2.8);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance = 1.05;
  controls.maxDistance = 9.0;
  controls.rotateSpeed = 0.55;
  controls.zoomSpeed = 0.9;

  // --- starfield backdrop --------------------------------------------------
  const starGeo = new THREE.BufferGeometry();
  const N = 1400, sp = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, r = 40;
    const s = Math.sqrt(1 - u * u);
    sp[i * 3] = r * s * Math.cos(t); sp[i * 3 + 1] = r * u; sp[i * 3 + 2] = r * s * Math.sin(t);
  }
  starGeo.setAttribute("position", new THREE.BufferAttribute(sp, 3));
  scene.add(new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x8ea6c8, size: 0.13, sizeAttenuation: true, transparent: true, opacity: 0.7 })));

  // --- dark base sphere (shows through when field opacity < 1) --------------
  const base = new THREE.Mesh(
    new THREE.SphereGeometry(0.998, 96, 64),
    new THREE.MeshBasicMaterial({ color: 0x0a1018 }));
  scene.add(base);

  // --- field texture (equirectangular) + texture-mapped sphere -------------
  const fieldTex = new THREE.DataTexture(new Uint8Array(4).fill(40), 1, 1, THREE.RGBAFormat);
  fieldTex.minFilter = THREE.LinearFilter;
  fieldTex.magFilter = THREE.LinearFilter;
  fieldTex.wrapS = THREE.RepeatWrapping;    // seamless longitude wrap
  fieldTex.wrapT = THREE.ClampToEdgeWrapping;
  fieldTex.needsUpdate = true;

  const uniforms = { u_field: { value: fieldTex }, u_opacity: { value: 0.72 } };
  const fieldMat = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false,
    vertexShader: /* glsl */`
      varying vec3 vObj;
      void main(){ vObj = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
    fragmentShader: /* glsl */`
      precision highp float;
      #define PI 3.141592653589793
      varying vec3 vObj;
      uniform sampler2D u_field;
      uniform float u_opacity;
      void main(){
        vec3 p = normalize(vObj);
        float lat = asin(clamp(p.y,-1.0,1.0));
        float lon = atan(p.z, p.x);
        vec2 uv = vec2(lon/(2.0*PI) + 0.5, 0.5 - lat/PI);
        vec3 c = texture2D(u_field, uv).rgb;
        gl_FragColor = vec4(c, u_opacity);
      }`,
  });
  const globe = new THREE.Mesh(new THREE.SphereGeometry(1.0, 96, 64), fieldMat);
  scene.add(globe);

  // --- graticule (lat/lon grid lines) --------------------------------------
  const grat = buildGraticule(1.001);
  scene.add(grat);

  // --- city markers --------------------------------------------------------
  let cityPoints = null;
  let cityData = [];
  function setCities(cities) {
    cityData = cities;
    if (cityPoints) { scene.remove(cityPoints); cityPoints.geometry.dispose(); }
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(cities.length * 3);
    cities.forEach((c, i) => { const v = llToVec(c.lat, c.lon, 1.012); pos[i * 3] = v.x; pos[i * 3 + 1] = v.y; pos[i * 3 + 2] = v.z; });
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    cityPoints = new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xffffff, size: 0.02, sizeAttenuation: true, transparent: true, opacity: 0.85,
      map: dotSprite(), depthWrite: false, blending: THREE.AdditiveBlending }));
    scene.add(cityPoints);
  }

  // --- observer reticle ----------------------------------------------------
  const reticle = new THREE.Mesh(
    new THREE.RingGeometry(0.02, 0.032, 40),
    new THREE.MeshBasicMaterial({ color: 0x6fe0ff, side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthWrite: false }));
  reticle.visible = false;
  scene.add(reticle);
  function setReticle(latDeg, lonDeg) {
    const v = llToVec(latDeg, lonDeg, 1.006);
    reticle.position.copy(v); reticle.lookAt(0, 0, 0); reticle.visible = true;
  }

  // --- raycasting ----------------------------------------------------------
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  function pick(ev) {
    const r = renderer.domElement.getBoundingClientRect();
    ndc.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
    ndc.y = -((ev.clientY - r.top) / r.height) * 2 + 1;
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObject(globe, false)[0];
    if (!hit) return null;
    const p = hit.point.clone().normalize();
    return {
      lat: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(p.y, -1, 1))),
      lon: THREE.MathUtils.radToDeg(Math.atan2(p.z, p.x)),
    };
  }
  function nearestCity(latDeg, lonDeg, maxDeg = 6) {
    let best = null, bd = maxDeg;
    for (const c of cityData) {
      const d = haversineDeg(latDeg, lonDeg, c.lat, c.lon);
      if (d < bd) { bd = d; best = c; }
    }
    return best;
  }

  function setField(rgba, w, h) {
    fieldTex.image = { data: rgba instanceof Uint8Array ? rgba : new Uint8Array(rgba.buffer || rgba), width: w, height: h };
    fieldTex.needsUpdate = true;
  }
  function setOpacity(o) { uniforms.u_opacity.value = o; }
  function setGraticule(on) { grat.visible = on; }
  function setCitiesVisible(on) { if (cityPoints) cityPoints.visible = on; }

  function resize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  }
  function render() { controls.update(); renderer.render(scene, camera); }

  return {
    renderer, scene, camera, controls,
    setField, setOpacity, setGraticule, setCities, setCitiesVisible,
    setReticle, pick, nearestCity, resize, render,
  };
}

// ---- helpers ---------------------------------------------------------------
function dotSprite() {
  const s = 32, cv = document.createElement("canvas"); cv.width = cv.height = s;
  const ctx = cv.getContext("2d");
  const g = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.4, "rgba(200,230,255,0.75)");
  g.addColorStop(1, "rgba(120,180,255,0)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, s, s);
  const t = new THREE.CanvasTexture(cv); t.needsUpdate = true; return t;
}

function buildGraticule(r) {
  const pts = [];
  const seg = 128;
  for (let lat = -60; lat <= 60; lat += 30) {
    for (let k = 0; k < seg; k++) {
      pts.push(llToVec(lat, -180 + k / seg * 360, r), llToVec(lat, -180 + (k + 1) / seg * 360, r));
    }
  }
  for (let lon = -180; lon < 180; lon += 30) {
    for (let k = 0; k < seg; k++) {
      pts.push(llToVec(-90 + k / seg * 180, lon, r), llToVec(-90 + (k + 1) / seg * 180, lon, r));
    }
  }
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = new THREE.LineBasicMaterial({ color: 0x2a4055, transparent: true, opacity: 0.35 });
  const lines = new THREE.LineSegments(g, m);
  lines.visible = false;
  return lines;
}

function haversineDeg(la1, lo1, la2, lo2) {
  const a = la1 * D2R, b = la2 * D2R, d = (lo2 - lo1) * D2R, e = (la2 - la1) * D2R;
  const h = Math.sin(e / 2) ** 2 + Math.cos(a) * Math.cos(b) * Math.sin(d / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h))) / D2R;
}
