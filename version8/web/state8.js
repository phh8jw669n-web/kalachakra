// state8.js — build the 88-D state (33 local + 55 chords) in the browser (HUD + parity).
//
// Mirrors version8/state.py exactly: the 33-D topocentric local vectors from ephemeris8.js,
// then the 55 pairwise dot products in the SAME fixed (i<j) order.

import { topocentricTensor, N_BODIES } from "./ephemeris8.js";

export const N_LOCAL = 33;
export const N_CHORD = 55;
export const STATE_DIM = 88;

// the 55 unique (i, j) body pairs, i < j — canonical chord order (matches Python).
export const PAIRS = [];
for (let i = 0; i < N_BODIES; i++) for (let j = i + 1; j < N_BODIES; j++) PAIRS.push([i, j]);

export function chordsFromLocal(local) {           // local: Float32Array(33) -> Float64Array(55)
  const out = new Float64Array(N_CHORD);
  for (let k = 0; k < PAIRS.length; k++) {
    const i = PAIRS[k][0] * 3, j = PAIRS[k][1] * 3;
    out[k] = local[i] * local[j] + local[i + 1] * local[j + 1] + local[i + 2] * local[j + 2];
  }
  return out;
}

// One observer's 88-D state -> Float64Array(88) (local ++ chords).
export function topocentricState(latDeg, lonDeg, jd) {
  const local = topocentricTensor(latDeg, lonDeg, jd);   // Float32Array(33)
  const chords = chordsFromLocal(local);
  const out = new Float64Array(STATE_DIM);
  for (let i = 0; i < N_LOCAL; i++) out[i] = local[i];
  for (let k = 0; k < N_CHORD; k++) out[N_LOCAL + k] = chords[k];
  return out;
}
