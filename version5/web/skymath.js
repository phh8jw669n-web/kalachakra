// skymath.js — the browser's copy of version5/sky_math.py (v5.1 metric encoder).
//
// Dependency-free ES module: builds the Zero-Redundancy 50-D physical state exactly
// as the server does, so the client's ONNX input is bit-for-bit identical. Also holds
// the telemetry parser (12 bodies, for the 3D orbits + glow) and OKLab->sRGB.
//
// 50-D state layout:
//   [0..43]  11 ML bodies (Sun..Pluto + True Node), each [X, Y, Z, V]:
//            X=cos(β)cos(λ), Y=cos(β)sin(λ), Z=sin(β), V=tanh(v_raw / 15°/day)
//   [44..46] Ascendant  as ecliptic Cartesian unit vector [cos, sin, 0]
//   [47..49] Midheaven  as ecliptic Cartesian unit vector [cos, sin, 0]

export const N_BODIES = 12;                 // telemetry / 3D orbits / glow keep all 12
export const N_ML_BODIES = 11;              // ML state drops the Mean Node
export const STATE_DIM = 50;
// indices into the 12-body telemetry that feed the 50-D state (skip 10 = Mean Node)
export const ML_BODY_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11];
const ANG_VEL_SCALE = 15.0;                 // == kalachakra ...features.ANG_VEL_SCALE
// Canonical body order (== version5.ephemeris.BODY_NAMES): 10 primaries + 2 nodes.
export const BODY_NAMES = [
  "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter",
  "Saturn", "Uranus", "Neptune", "Pluto", "MeanNode", "TrueNode",
];

const DEG2RAD = Math.PI / 180.0;

function wrapPi(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }

// Parse a /telemetry payload into radian arrays for the 12 bodies + scene scalars.
export function telemetryToState(tel) {
  const ra = new Float64Array(N_BODIES), dec = new Float64Array(N_BODIES);
  const lam = new Float64Array(N_BODIES), bet = new Float64Array(N_BODIES);
  const vel = new Float64Array(N_BODIES);          // normalised (deg/day / scale)
  for (let i = 0; i < N_BODIES; i++) {
    const b = tel.bodies[BODY_NAMES[i]];
    ra[i] = b.ra * DEG2RAD;
    dec[i] = b.dec * DEG2RAD;
    lam[i] = b.lon * DEG2RAD;
    bet[i] = b.lat * DEG2RAD;
    vel[i] = b.lon_speed / ANG_VEL_SCALE;
  }
  return {
    ra, dec, lam, bet, vel,
    gast: tel.gast_deg * DEG2RAD,
    eps: tel.obliquity_deg * DEG2RAD,
  };
}

// Ascendant, Midheaven, Vertex (ecliptic longitudes, radians) — mirrors
// sky_math.ascendant_mc_vertex. cos φ / sin φ factored forms stay finite at poles.
export function ascMcVertex(ramc, phi, eps) {
  const st = Math.sin(ramc), ct = Math.cos(ramc);
  const sp = Math.sin(phi), cp = Math.cos(phi);
  const ce = Math.cos(eps), se = Math.sin(eps);
  const mc = Math.atan2(st, ct * ce);
  const asc = Math.atan2(ct * cp, -(st * ce * cp + sp * se));
  const vx = Math.atan2(ct * sp, -(st * ce * sp + cp * se));
  return { asc, mc, vx };
}

// The 44 location-independent body dims [X,Y,Z,V] x 11 for the current timestamp.
// (state.vel is already v_raw/15; tanh here completes V = tanh(v_raw / v_max).)
export function mlBodyState(tstate) {
  const out = new Float32Array(N_ML_BODIES * 4);
  for (let k = 0; k < N_ML_BODIES; k++) {
    const i = ML_BODY_INDICES[k];
    const cb = Math.cos(tstate.bet[i]);
    const o = k * 4;
    out[o + 0] = cb * Math.cos(tstate.lam[i]);
    out[o + 1] = cb * Math.sin(tstate.lam[i]);
    out[o + 2] = Math.sin(tstate.bet[i]);
    out[o + 3] = Math.tanh(tstate.vel[i]);
  }
  return out;
}

// Write the 50-D state for ONE observer into `out` at `off`, given precomputed body44.
export function localStateOne(body44, tstate, latRad, lonRad, out, off) {
  out.set(body44, off);                              // 44 time-only body dims
  const { asc, mc } = ascMcVertex(tstate.gast + lonRad, latRad, tstate.eps);
  out[off + 44] = Math.cos(asc); out[off + 45] = Math.sin(asc); out[off + 46] = 0;
  out[off + 47] = Math.cos(mc); out[off + 48] = Math.sin(mc); out[off + 49] = 0;
}

// Build the [N,50] state tensor (flat Float32Array) for a grid of observers.
export function buildStateBatch(tstate, latRad, lonRad) {
  const n = latRad.length;
  const body44 = mlBodyState(tstate);
  const state = new Float32Array(n * STATE_DIM);
  for (let p = 0; p < n; p++) {
    localStateOne(body44, tstate, latRad[p], lonRad[p], state, p * STATE_DIM);
  }
  return state;
}

// RA/Dec -> Three.js Y-up Cartesian on a celestial sphere of `radius`:
//   x = -R cos(dec) sin(alpha),  y = R sin(dec),  z = R cos(dec) cos(alpha)
// Declination is the Y elevation; the right ascension sweeps the X/Z plane.
export function raDecToCartesian(alpha, dec, radius = 1.0) {
  const cd = Math.cos(dec);
  return {
    x: -radius * cd * Math.sin(alpha),
    y: radius * Math.sin(dec),
    z: radius * cd * Math.cos(alpha),
  };
}

// Place a body over its sub-planetary point on the (Earth-fixed) globe: feed the
// above formula the apparent hour-angle argument (GAST - RA - pi/2). Algebraically
// identical to the globe surface normal at (lat=dec, lon=RA-GAST), so each 3D body
// hovers exactly above its own glow. Pass raw RA instead for an inertial sky.
export function bodyDirection(ra, dec, gast, radius = 1.0) {
  return raDecToCartesian(gast - ra - Math.PI / 2, dec, radius);
}

// shortest-path angular interpolation (for wrap-safe RA/GAST lerp between frames)
export function lerpAngle(a, b, t) { return a + wrapPi(b - a) * t; }

// --- OKLab -> sRGB (Bjorn Ottosson); same matrices as kalachakra ...color.py ---
export function oklabToLinearSrgb(L, a, b) {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
  return [
    +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
}

function gamma(c) {
  c = Math.min(1, Math.max(0, c));
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

export function oklabToSrgb(L, a, b) {
  const lin = oklabToLinearSrgb(L, a, b);
  return [gamma(lin[0]), gamma(lin[1]), gamma(lin[2])];
}

// Equal-angle lon x lat grid (maps 1:1 to the equirectangular sphere UV).
export function makeGeoGrid(gridW, gridH) {
  const n = gridW * gridH;
  const lat = new Float64Array(n), lon = new Float64Array(n);
  for (let j = 0; j < gridH; j++) {
    const la = (-1 + 2 * (j + 0.5) / gridH) * (Math.PI / 2);   // (-pi/2, pi/2)
    for (let i = 0; i < gridW; i++) {
      const lo = (-1 + 2 * (i + 0.5) / gridW) * Math.PI;       // (-pi, pi)
      const k = j * gridW + i;
      lat[k] = la; lon[k] = lo;
    }
  }
  return { lat, lon, gridW, gridH };
}
