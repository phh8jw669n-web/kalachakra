// skymath.js — the browser's copy of version5/sky_math.py + the OKLab colour maths.
//
// Dependency-free ES module: the same spherical trigonometry the server trains on,
// re-implemented line for line so the client's neural input is bit-for-bit identical
// (PRD page 10, "client math must match server math"). It is imported by main.js and
// unit-tested against golden.json in Node — see skymath.test.mjs.
//
// Angles are radians. Feature order per body matches sky_math.COL_*:
//   [ altitude, azimuth, RA, declination, hour_angle ]

export const N_BODIES = 10;
export const RAW_FEATURES = 5;
// Canonical body order (== kalachakra.local_autoencoder.features.BODY_NAMES).
export const BODY_NAMES = [
  "Sun", "Moon", "Mercury", "Venus", "Mars",
  "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto",
];

const DEG2RAD = Math.PI / 180.0;

// Wrap an angle to (-pi, pi] exactly as numpy's arctan2(sin, cos) does.
function wrapPi(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }

// Pull the ten bodies' RA/Dec (radians) and GAST (radians) out of a /telemetry
// payload. gast_deg is authoritative; RA/Dec arrive in degrees.
export function telemetryToEquatorial(tel) {
  const ra = new Float64Array(N_BODIES);
  const dec = new Float64Array(N_BODIES);
  for (let i = 0; i < N_BODIES; i++) {
    const b = tel.bodies[BODY_NAMES[i]];
    ra[i] = b.ra * DEG2RAD;
    dec[i] = b.dec * DEG2RAD;
  }
  const gast = tel.gast_deg * DEG2RAD;
  return { ra, dec, gast };
}

// Local sky matrix for ONE observer -> Float32Array of length 10*5, laid out
// body-major ([b0f0..b0f4, b1f0..], matching numpy's [10,5].reshape(-1)).
export function localFeaturesOne(eq, latRad, lonRad, out, offset = 0) {
  const { ra, dec, gast } = eq;
  const sinPhi = Math.sin(latRad), cosPhi = Math.cos(latRad);
  const lst = gast + lonRad;
  for (let i = 0; i < N_BODIES; i++) {
    const ha = wrapPi(lst - ra[i]);
    const sinDec = Math.sin(dec[i]), cosDec = Math.cos(dec[i]);
    const sinHa = Math.sin(ha), cosHa = Math.cos(ha);
    let sinAlt = sinPhi * sinDec + cosPhi * cosDec * cosHa;
    if (sinAlt > 1) sinAlt = 1; else if (sinAlt < -1) sinAlt = -1;
    const alt = Math.asin(sinAlt);
    const az = Math.atan2(sinHa * cosDec,
                          cosHa * sinPhi * cosDec - sinDec * cosPhi);
    const o = offset + i * RAW_FEATURES;
    out[o + 0] = alt;
    out[o + 1] = az;
    out[o + 2] = ra[i];
    out[o + 3] = dec[i];
    out[o + 4] = ha;
  }
  return out;
}

// Build the [N,10,5] input tensor (flat Float32Array) for a grid of observers.
// latRad/lonRad are Float32/64Arrays of length N.
export function buildFeatureBatch(eq, latRad, lonRad) {
  const n = latRad.length;
  const out = new Float32Array(n * N_BODIES * RAW_FEATURES);
  for (let p = 0; p < n; p++) {
    localFeaturesOne(eq, latRad[p], lonRad[p], out, p * N_BODIES * RAW_FEATURES);
  }
  return out;
}

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

// OKLab [L,a,b] -> sRGB [r,g,b] in [0,1].
export function oklabToSrgb(L, a, b) {
  const lin = oklabToLinearSrgb(L, a, b);
  return [gamma(lin[0]), gamma(lin[1]), gamma(lin[2])];
}

// Sphere-uniform observer grid: latitude via arcsin so the poles are not
// over-sampled (matches sky_math.sample_locations). Returns a regular lon x lat
// grid (row-major, lat outer) suitable for a GRID_W x GRID_H DataTexture.
export function makeGeoGrid(gridW, gridH) {
  const n = gridW * gridH;
  const lat = new Float64Array(n), lon = new Float64Array(n);
  for (let j = 0; j < gridH; j++) {
    // sin(lat) uniform in (-1,1): cell centres so no pole singularity.
    const s = -1 + 2 * (j + 0.5) / gridH;
    const la = Math.asin(s);
    for (let i = 0; i < gridW; i++) {
      const lo = (-1 + 2 * (i + 0.5) / gridW) * Math.PI;   // (-pi, pi)
      const k = j * gridW + i;
      lat[k] = la; lon[k] = lo;
    }
  }
  return { lat, lon, gridW, gridH };
}
