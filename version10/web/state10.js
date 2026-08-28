// state10.js — the browser's geometric layer (parity with version10/state.py).
//
// The model input is the 39-D local sky (13 tokens = 11 bodies + ASC + MC, each North/East/
// Zenith). The horizon-gated chords are provided too for the HUD's relational read-out; they
// are NOT model input (the attention network learns the relations itself), only the
// observer-dependent target the field was trained to be isometric to.

import { topocentricTensor, N_BODIES } from "./ephemeris10.js";

export const N_LOCAL = N_BODIES * 3;                     // 39
export const N_CHORD = (N_BODIES * (N_BODIES - 1)) / 2; // 78

// the 78 unique (i, j) token pairs, i < j — canonical chord order (matches Python).
export const PAIRS = [];
for (let i = 0; i < N_BODIES; i++) for (let j = i + 1; j < N_BODIES; j++) PAIRS.push([i, j]);

// One observer's 39-D local sky -> Float32Array(39) (North,East,Zenith) x 13 (model input).
export function localVectors(latDeg, lonDeg, jd) {
  return topocentricTensor(latDeg, lonDeg, jd);
}

// Horizon-gated chords R_ij = g_i*g_j*(v_i.v_j), g_b = sigmoid(gate_k*zenith_b) -> Float64Array(78).
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
