// field.js — bake the regional energy field into an equirectangular colour grid.
//
// This is the heart of the v7 performance model: instead of evaluating the ephemeris + SIREN
// per screen pixel every frame (v6), we evaluate it once per grid node into a small
// equirectangular texture, and the globe just samples that texture. The maths is reused
// verbatim from version6 — the factored per-frame ephemeris (equatorialDirs / gmstRad) and
// the SIREN with its bounded, soft-clamped L*a*b* head (makeSiren / boundLab / labToSrgb) —
// so the colours are identical to v6, only the sampling density changes.
//
// No Three.js import here: this module runs in the background worker AND on the main thread.

import { equatorialDirs, gmstRad, N_BODIES } from "../../version6/web/ephemeris6.js";
import { makeSiren, boundLab, labToSrgb } from "../../version6/web/siren6.js";

const D2R = Math.PI / 180.0;

// A reusable field baker bound to one set of weights.
export function makeFieldBaker(weights) {
  const siren = makeSiren(weights);
  const sky = new Float64Array(33);
  let sLc = null, cLc = null, W0 = 0;

  // Compute the [W*H*4] RGBA equirectangular grid for one Julian Date.
  // Row 0 = +90 deg (north), column 0 = -180 deg; matches the globe shader's uv mapping.
  function bake(jd, W, H, out) {
    const rgba = out && out.length === W * H * 4 ? out : new Uint8ClampedArray(W * H * 4);
    const eq = equatorialDirs(jd);              // 11 geocentric equatorial unit vecs (once)
    const g = gmstRad(jd);
    if (W0 !== W) {                              // per-column local-sidereal-time trig
      sLc = new Float64Array(W); cLc = new Float64Array(W); W0 = W;
    }
    for (let i = 0; i < W; i++) {
      const lst = g + (-180.0 + (i + 0.5) / W * 360.0) * D2R;
      sLc[i] = Math.sin(lst); cLc[i] = Math.cos(lst);
    }
    for (let j = 0; j < H; j++) {
      const lat = (90.0 - (j + 0.5) / H * 180.0) * D2R;
      const sphi = Math.sin(lat), cphi = Math.cos(lat);
      for (let i = 0; i < W; i++) {
        const sL = sLc[i], cL = cLc[i];
        for (let b = 0; b < N_BODIES; b++) {
          const x = eq[3 * b], y = eq[3 * b + 1], z = eq[3 * b + 2];
          const cdcosH = cL * x + sL * y;       // cosδ·cosH
          sky[3 * b + 0] = z * cphi - sphi * cdcosH;   // North
          sky[3 * b + 1] = cL * y - sL * x;            // East
          sky[3 * b + 2] = z * sphi + cphi * cdcosH;   // Up
        }
        const lab = boundLab(weights, siren(sky));
        const rgb = labToSrgb(lab[0], lab[1], lab[2]);
        const p = (j * W + i) * 4;
        rgba[p] = rgb[0] * 255; rgba[p + 1] = rgb[1] * 255;
        rgba[p + 2] = rgb[2] * 255; rgba[p + 3] = 255;
      }
    }
    return rgba;
  }

  return { bake };
}
