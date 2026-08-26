// skycompute.wgsl — the geographic feature engine, ported to a WebGPU compute shader.
//
// One thread per grid point (observer location). Each thread derives its own lat/lon
// from global_invocation_id.x, then runs the exact spherical trigonometry of
// version5/sky_math.py / skymath.js (altitude, azimuth, Ascendant, Midheaven, Vertex,
// house offset) and writes the raw feature tensors the ONNX encoder consumes:
//   out_features : [N, 12, 6]  = [alt, az, ecl_lon, ecl_lat, house_offset, velocity]
//   out_observer : [N, 3]      = [Ascendant, Midheaven, Vertex]
// The five cyclic body angles and the three observer angles are left raw here; the
// exported ONNX graph does the sin/cos expansion (and the velocity tanh), exactly as
// in training — so this shader is a drop-in replacement for the CPU buildFeatures().

const PI      : f32 = 3.141592653589793;
const HALF_PI : f32 = 1.5707963267948966;
const NB      : u32 = 12u;   // bodies
const RAW     : u32 = 6u;    // features per body
const OBS     : u32 = 3u;    // observer anchors

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
// body[b*5 + {0:ra, 1:dec, 2:ecl_lon, 3:ecl_lat, 4:velocity}] (radians / scaled), 12 bodies
@group(0) @binding(1) var<storage, read>       body         : array<f32>;
@group(0) @binding(2) var<storage, read_write> out_features : array<f32>;
@group(0) @binding(3) var<storage, read_write> out_observer : array<f32>;

// wrap to (-pi, pi] exactly like numpy/JS atan2(sin, cos)
fn wrap_pi(a: f32) -> f32 { return atan2(sin(a), cos(a)); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.n) { return; }                       // guard the ragged last workgroup

  // grid on-the-fly: p -> (lon column i, lat row j) -> equal-angle lat/lon.
  // Matches skymath.makeGeoGrid so the field texture aligns with the sphere UV.
  let j = p / params.grid_w;
  let i = p % params.grid_w;
  let lat = (-1.0 + 2.0 * (f32(j) + 0.5) / f32(params.grid_h)) * HALF_PI;
  let lon = (-1.0 + 2.0 * (f32(i) + 0.5) / f32(params.grid_w)) * PI;

  let sphi = sin(lat);
  let cphi = cos(lat);
  let ce   = cos(params.eps);
  let se   = sin(params.eps);
  let ramc = params.gast + lon;                         // local sidereal time
  let st   = sin(ramc);
  let ct   = cos(ramc);

  // high-frequency geographic resolvers (pole-safe cos/sin-factored forms)
  let asc = atan2(ct * cphi, -(st * ce * cphi + sphi * se));
  let mc  = atan2(st, ct * ce);
  let vx  = atan2(ct * sphi, -(st * ce * sphi + cphi * se));

  let fo = p * NB * RAW;
  for (var b: u32 = 0u; b < NB; b = b + 1u) {
    let bi  = b * 5u;
    let ra  = body[bi + 0u];
    let dec = body[bi + 1u];
    let lam = body[bi + 2u];
    let bet = body[bi + 3u];
    let vel = body[bi + 4u];

    let ha = wrap_pi(ramc - ra);
    let sd = sin(dec);
    let cd = cos(dec);
    let sh = sin(ha);
    let ch = cos(ha);
    let sinAlt = clamp(sphi * sd + cphi * cd * ch, -1.0, 1.0);
    let alt = asin(sinAlt);
    let az  = atan2(sh * cd, ch * sphi * cd - sd * cphi);

    let o = fo + b * RAW;
    out_features[o + 0u] = alt;
    out_features[o + 1u] = az;
    out_features[o + 2u] = lam;
    out_features[o + 3u] = bet;
    out_features[o + 4u] = wrap_pi(lam - asc);          // house offset
    out_features[o + 5u] = vel;
  }

  let oo = p * OBS;
  out_observer[oo + 0u] = asc;
  out_observer[oo + 1u] = mc;
  out_observer[oo + 2u] = vx;
}
