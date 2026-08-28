// state9.js — the browser's geometric layer (parity with version9/state.py).
//
// The model input is the 33-D local sky (11 bodies x North/East/Zenith). The horizon-gated
// chords are provided too for the HUD's relational read-out; they are NOT model input (the
// attention network learns the relations itself), only the observer-dependent target the
// field was trained to be isometric to.

import { topocentricTensor, N_BODIES } from "./ephemeris9.js";

export const N_LOCAL = 33;
export const N_CHORD = 55;

// the 55 unique (i, j) body pairs, i < j — canonical chord order (matches Python).
export const PAIRS = [];
for (let i = 0; i < N_BODIES; i++) for (let j = i + 1; j < N_BODIES; j++) PAIRS.push([i, j]);

// One observer's 33-D local sky -> Float32Array(33) (North,East,Zenith)x11 (model input).
export function localVectors(latDeg, lonDeg, jd) {
  return topocentricTensor(latDeg, lonDeg, jd);
}

// Horizon-gated chords R_ij = g_i*g_j*(v_i.v_j), g_b = sigmoid(gate_k*zenith_b) -> Float64Array(55).
export function gatedChords(local, gateK) {
  const g = new Float64Array(N_BODIES);
  for (let b = 0; b < N_BODIES; b++) g[b] = 1.0 / (1.0 + Math.exp(-gateK * local[b * 3 + 2]));
  const out = new Float64Array(PAIRS.length);
  for (let k = 0; k < PAIRS.length; k++) {
    const i = PAIRS[k][0], j = PAIRS[k][1];
    const dot = local[i * 3] * local[j * 3] + local[i * 3 + 1] * local[j * 3 + 1] + local[i * 3 + 2] * local[j * 3 + 2];
    out[k] = g[i] * g[j] * dot;
  }
  return out;
}
