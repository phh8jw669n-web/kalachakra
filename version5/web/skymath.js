// skymath.js — the browser's copy of version5/sky_math.py + ephemeris.py maths.
//
// Dependency-free ES module: the same spherical trigonometry the server trains on,
// re-implemented line for line so the client's neural input is bit-for-bit identical
// (PRD "client math must match server math"). Imported by main.js and unit-tested
// against golden.json in Node.
//
// Body feature order per body (== sky_math.COL_*):
//   [ altitude, azimuth, ecl_longitude, ecl_latitude, house_offset, velocity ]
// Observer anchors: [ Ascendant, Midheaven, Vertex ] (radians).

export const N_BODIES = 12;
export const RAW_FEATURES = 6;
export const OBS_FEATURES = 3;
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

// Local sky matrix for ONE observer -> writes 12*6 body features into `feat` at
// `fo` and the 3 observer anchors into `obs` at `oo`. Body-major layout.
export function localFeaturesOne(state, latRad, lonRad, feat, fo, obs, oo) {
  const { ra, dec, lam, bet, vel, gast, eps } = state;
  const sinPhi = Math.sin(latRad), cosPhi = Math.cos(latRad);
  const ramc = gast + lonRad;
  const { asc, mc, vx } = ascMcVertex(ramc, latRad, eps);
  for (let i = 0; i < N_BODIES; i++) {
    const ha = wrapPi(ramc - ra[i]);
    const sinDec = Math.sin(dec[i]), cosDec = Math.cos(dec[i]);
    const sinHa = Math.sin(ha), cosHa = Math.cos(ha);
    let sinAlt = sinPhi * sinDec + cosPhi * cosDec * cosHa;
    if (sinAlt > 1) sinAlt = 1; else if (sinAlt < -1) sinAlt = -1;
    const alt = Math.asin(sinAlt);
    const az = Math.atan2(sinHa * cosDec, cosHa * sinPhi * cosDec - sinDec * cosPhi);
    const o = fo + i * RAW_FEATURES;
    feat[o + 0] = alt;
    feat[o + 1] = az;
    feat[o + 2] = lam[i];
    feat[o + 3] = bet[i];
    feat[o + 4] = wrapPi(lam[i] - asc);           // house offset
    feat[o + 5] = vel[i];
  }
  obs[oo + 0] = asc; obs[oo + 1] = mc; obs[oo + 2] = vx;
}

// Build the [N,12,6] body tensor and [N,3] observer tensor for a grid of observers.
export function buildFeatureBatch(state, latRad, lonRad) {
  const n = latRad.length;
  const features = new Float32Array(n * N_BODIES * RAW_FEATURES);
  const observer = new Float32Array(n * OBS_FEATURES);
  for (let p = 0; p < n; p++) {
    localFeaturesOne(state, latRad[p], lonRad[p],
                     features, p * N_BODIES * RAW_FEATURES, observer, p * OBS_FEATURES);
  }
  return { features, observer };
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
