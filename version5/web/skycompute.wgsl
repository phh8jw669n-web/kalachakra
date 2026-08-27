// skycompute.wgsl — builds the Zero-Redundancy 50-D physical state on the GPU (v5.1).
//
// One thread per grid point (observer). The 44 body dims [X,Y,Z,V] x 11 depend on time
// only, so they are precomputed on the CPU and passed in `body`; each thread copies them
// and then computes its own Ascendant / Midheaven (the only location-dependent part) as
// ecliptic Cartesian unit vectors. Output: out_state[N*50], the exact tensor the ONNX
// metric encoder consumes (verified bit-for-bit against skymath.js / sky_math.py).

const PI      : f32 = 3.141592653589793;
const HALF_PI : f32 = 1.5707963267948966;
const BODY_DIM : u32 = 44u;   // 11 bodies * 4
const STATE   : u32 = 50u;    // + Asc(3) + MC(3)

struct Params {
  gast   : f32,   // Greenwich Apparent Sidereal Time (radians)
  eps    : f32,   // true obliquity of date (radians)
  grid_w : u32,
  grid_h : u32,
  n      : u32,   // = grid_w * grid_h
  _pad0  : u32,
  _pad1  : u32,
  _pad2  : u32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read>       body      : array<f32>;   // 44 time-only dims
@group(0) @binding(2) var<storage, read_write> out_state : array<f32>;   // N * 50

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.n) { return; }                       // guard the ragged last workgroup

  let base = p * STATE;
  for (var b: u32 = 0u; b < BODY_DIM; b = b + 1u) {    // copy the 44 time-only body dims
    out_state[base + b] = body[b];
  }

  // grid on-the-fly (matches skymath.makeGeoGrid) -> observer Ascendant / Midheaven
  let j = p / params.grid_w;
  let i = p % params.grid_w;
  let lat = (-1.0 + 2.0 * (f32(j) + 0.5) / f32(params.grid_h)) * HALF_PI;
  let lon = (-1.0 + 2.0 * (f32(i) + 0.5) / f32(params.grid_w)) * PI;

  let ce = cos(params.eps);
  let se = sin(params.eps);
  let ramc = params.gast + lon;
  let st = sin(ramc);
  let ct = cos(ramc);
  let sphi = sin(lat);
  let cphi = cos(lat);
  let asc = atan2(ct * cphi, -(st * ce * cphi + sphi * se));
  let mc  = atan2(st, ct * ce);

  out_state[base + 44u] = cos(asc);
  out_state[base + 45u] = sin(asc);
  out_state[base + 46u] = 0.0;
  out_state[base + 47u] = cos(mc);
  out_state[base + 48u] = sin(mc);
  out_state[base + 49u] = 0.0;
}
