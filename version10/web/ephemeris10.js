// ephemeris6.js — the browser's port of version6/ephemeris.py (scalar, one observer).
//
// Dependency-free ES module: transcribes the analytic topocentric ephemeris line for
// line so the HUD's 33-D tensor is bit-for-bit the Python engine's (and the GLSL
// shader's). Used by Module 4 (the Observer's HUD).

export const J2000 = 2451545.0;
const DEG = Math.PI / 180.0;
export const BODY_NAMES = [
  "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter",
  "Saturn", "Uranus", "Neptune", "Pluto", "Node",
  "ASC", "MC",                          // v10: Ascendant + Midheaven (observer-dependent anchors)
];
export const N_PLANETS = 11;            // Sun..Node have geocentric directions (uploaded once/frame)
export const N_BODIES = 13;             // + ASC + MC (computed per observer)
export const STATE_DIM = 39;            // 13 tokens x (N,E,Zenith)
export const LAT_CLAMP = 89.99;

// JPL Keplerian elements (Standish): [a, e, I, L, long.peri, long.node] @ J2000 + rates.
const ELEM = [
  [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
  [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
  [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
  [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
  [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
  [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
  [19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
  [30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
  [39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684],
];
const RATE = [
  [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
  [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
  [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0],
  [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
  [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
  [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
  [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589],
  [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664],
  [-0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482],
];
const EARTH = 2;

function wrap180(deg) { return ((deg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0; }

function keplerE(M, e) {
  let E = M + e * Math.sin(M);
  for (let k = 0; k < 6; k++) E = E - (E - e * Math.sin(E) - M) / (1.0 - e * Math.cos(E));
  return E;
}

// Heliocentric J2000-ecliptic Cartesian (AU) of planet i at T (centuries past J2000).
function heliocentric(T, i) {
  const a = ELEM[i][0] + RATE[i][0] * T;
  const e = ELEM[i][1] + RATE[i][1] * T;
  const inc = (ELEM[i][2] + RATE[i][2] * T) * DEG;
  const L = ELEM[i][3] + RATE[i][3] * T;
  const peri = ELEM[i][4] + RATE[i][4] * T;
  const node = (ELEM[i][5] + RATE[i][5] * T) * DEG;
  const omega = (peri - ELEM[i][5] - RATE[i][5] * T) * DEG;
  const M = wrap180(L - peri) * DEG;
  const E = keplerE(M, e);
  const xp = a * (Math.cos(E) - e);
  const yp = a * Math.sqrt(1.0 - e * e) * Math.sin(E);
  const co = Math.cos(omega), so = Math.sin(omega);
  const ci = Math.cos(inc), si = Math.sin(inc);
  const cn = Math.cos(node), sn = Math.sin(node);
  return [
    (co * cn - so * sn * ci) * xp + (-so * cn - co * sn * ci) * yp,
    (co * sn + so * cn * ci) * xp + (-so * sn + co * cn * ci) * yp,
    (so * si) * xp + (co * si) * yp,
  ];
}

function moonGeocentric(d) {
  const Lp = (218.3164477 + 13.17639648 * d) * DEG;
  const D = (297.8501921 + 12.19074920 * d) * DEG;
  const M = (357.5291092 + 0.98560028 * d) * DEG;
  const Mp = (134.9633964 + 13.06499295 * d) * DEG;
  const F = (93.2720950 + 13.22935024 * d) * DEG;
  const lon = (Lp / DEG
    + 6.289 * Math.sin(Mp) + 1.274 * Math.sin(2 * D - Mp) + 0.658 * Math.sin(2 * D)
    + 0.214 * Math.sin(2 * Mp) - 0.186 * Math.sin(M) - 0.114 * Math.sin(2 * F)
    + 0.059 * Math.sin(2 * D - 2 * Mp) + 0.057 * Math.sin(2 * D - M - Mp)) * DEG;
  const lat = (5.128 * Math.sin(F) + 0.280 * Math.sin(Mp + F) + 0.277 * Math.sin(Mp - F)
    + 0.173 * Math.sin(2 * D - F) + 0.055 * Math.sin(2 * D - Mp + F)
    + 0.046 * Math.sin(2 * D - Mp - F)) * DEG;
  const cb = Math.cos(lat);
  return [cb * Math.cos(lon), cb * Math.sin(lon), Math.sin(lat)];
}

function nodeGeocentric(d) {
  const lon = (125.04452 - 0.05295377 * d) * DEG;
  return [Math.cos(lon), Math.sin(lon), 0.0];
}

// Geocentric ecliptic UNIT directions of the 11 bodies -> array[11][3].
function eclDirs(jd) {
  const T = (jd - J2000) / 36525.0;
  const d = jd - J2000;
  const earth = heliocentric(T, EARTH);
  const out = new Array(N_PLANETS);
  out[0] = [-earth[0], -earth[1], -earth[2]];               // Sun
  out[1] = moonGeocentric(d);                               // Moon
  let bi = 2;
  for (let i = 0; i < 9; i++) {
    if (i === EARTH) continue;
    const h = heliocentric(T, i);
    out[bi++] = [h[0] - earth[0], h[1] - earth[1], h[2] - earth[2]];
  }
  out[10] = nodeGeocentric(d);                              // Node
  for (let k = 0; k < N_PLANETS; k++) {
    const v = out[k];
    const n = Math.max(Math.hypot(v[0], v[1], v[2]), 1e-12);
    out[k] = [v[0] / n, v[1] / n, v[2] / n];
  }
  return out;
}

function obliquity(T) { return (23.439291 - 0.0130042 * T) * DEG; }
export function obliquityRad(jd) { return obliquity((jd - J2000) / 36525.0); }

export function gmstDeg(jd) {
  const T = (jd - J2000) / 36525.0;
  const g = 280.46061837 + 360.98564736629 * (jd - J2000)
    + 0.000387933 * T * T - T * T * T / 38710000.0;
  return ((g % 360.0) + 360.0) % 360.0;
}

export function gmstRad(jd) { return gmstDeg(jd) * DEG; }

// Geocentric EQUATORIAL unit vectors of the 11 bodies -> Float32Array(33) as 11 x (x,y,z),
// where (x,y,z) = (cosδ·cosα, cosδ·sinα, sinδ). These depend only on time (NOT on the
// observer), so the shader computes them ONCE per frame instead of redoing the whole
// Kepler/lunar ephemeris for every pixel. The cheap per-pixel horizontal projection is:
//   sL=sin(lst), cL=cos(lst), sφ=sin(lat), cφ=cos(lat);  cdcosH = cL*x + sL*y;
//   North = z*cφ - sφ*cdcosH;  East = cL*y - sL*x;  Up = z*sφ + cφ*cdcosH;
// which is algebraically identical to topocentricTensor()'s per-body block.
export function equatorialDirs(jd) {
  const T = (jd - J2000) / 36525.0;
  const dirs = eclDirs(jd);
  const eps = obliquity(T);
  const ce = Math.cos(eps), se = Math.sin(eps);
  const out = new Float32Array(N_PLANETS * 3);
  for (let k = 0; k < N_PLANETS; k++) {
    const xe = dirs[k][0], ye = dirs[k][1], ze = dirs[k][2];
    out[k * 3 + 0] = xe;                        // x_eq (obliquity leaves x unchanged)
    out[k * 3 + 1] = ye * ce - ze * se;         // y_eq
    out[k * 3 + 2] = ye * se + ze * ce;         // z_eq = sin(dec)
  }
  return out;
}

// Ascendant & Midheaven ecliptic longitudes (rad) for an observer (verified vs swisseph houses
// to <0.01deg). lat is clamped to +/-89.99deg (finite tan, matches the field shader / Python).
export function ascMcEcliptic(latDeg, lonDeg, jd) {
  const lat = Math.max(-LAT_CLAMP, Math.min(LAT_CLAMP, latDeg)) * DEG;
  const eps = obliquityRad(jd), ce = Math.cos(eps), se = Math.sin(eps);
  const lst = gmstDeg(jd) * DEG + lonDeg * DEG;               // RAMC
  const sR = Math.sin(lst), cR = Math.cos(lst);
  const lamMc = Math.atan2(sR, cR * ce);
  const lamAsc = Math.atan2(cR, -(sR * ce + Math.tan(lat) * se));
  return { lamAsc, lamMc };
}

// Equatorial UNIT direction (cosδcosα, cosδsinα, sinδ) of an ecliptic-plane point at longitude λ.
export function eclipticToEquatorial(lam, jd) {
  const eps = obliquityRad(jd), ce = Math.cos(eps), se = Math.sin(eps);
  const cl = Math.cos(lam), sl = Math.sin(lam);
  return [cl, sl * ce, sl * se];              // ecl (cl,sl,0) -> equ (rotate X by eps)
}

// The 39-D topocentric tensor for one observer -> Float32Array(39) (North,East,Up) x 13
// (11 bodies + ASC + MC). Latitude clamped to +/-89.99deg (matches Python + the field shader).
export function topocentricTensor(latDeg, lonDeg, jd) {
  const lat = Math.max(-LAT_CLAMP, Math.min(LAT_CLAMP, latDeg)) * DEG, lon = lonDeg * DEG;
  const eps = obliquityRad(jd);
  const ce = Math.cos(eps), se = Math.sin(eps);
  const lst = gmstDeg(jd) * DEG + lon;
  const sphi = Math.sin(lat), cphi = Math.cos(lat);
  const dirs = eclDirs(jd).slice();                          // 11 body ecliptic dirs
  const { lamAsc, lamMc } = ascMcEcliptic(latDeg, lonDeg, jd);
  dirs.push([Math.cos(lamAsc), Math.sin(lamAsc), 0.0]);      // token 11 ASC (ecliptic plane)
  dirs.push([Math.cos(lamMc), Math.sin(lamMc), 0.0]);        // token 12 MC
  const out = new Float32Array(STATE_DIM);
  for (let k = 0; k < N_BODIES; k++) {
    const xe = dirs[k][0], ye = dirs[k][1], ze = dirs[k][2];
    const xq = xe, yq = ye * ce - ze * se, zq = ye * se + ze * ce;   // ecl -> equ
    const ra = Math.atan2(yq, xq);
    const dec = Math.asin(Math.max(-1, Math.min(1, zq)));
    const H = lst - ra;
    const sd = Math.sin(dec), cd = Math.cos(dec), sH = Math.sin(H), cH = Math.cos(H);
    out[k * 3 + 0] = sd * cphi - cd * sphi * cH;             // North
    out[k * 3 + 1] = -cd * sH;                               // East
    out[k * 3 + 2] = sd * sphi + cd * cphi * cH;             // Up = sin(alt)
  }
  return out;
}
