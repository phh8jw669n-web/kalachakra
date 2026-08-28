// shader8.js — the version8 vertex-shader SIREN (88-D relational field).
//
// The WHOLE network runs PER VERTEX on a SphereGeometry(128,128) (~16k verts) and the GPU
// interpolates colour across triangles for free — 60-120 fps for a 61k-weight net that would
// be unrenderable per pixel. Per vertex:
//   1. singularity-free local basis from the surface normal (blueprint: n̂ = tangent pole,
//      ê = û × n̂ — the sign that matches topocentric_tensor);
//   2. project the 11 Earth-fixed body vectors -> 33 local (North,East,Zenith);
//   3. 55 chord dot products (fixed i<j order, matches state.py);
//   4. the 4x128 SIREN forward + gamut-bounded L*a*b* head + L*a*b*->linear sRGB.
// The field frame negates z (N.x, N.y, -N.z) so the world map reads un-mirrored while the
// field stays physically exact (verified 5.96e-8 vs topocentric_tensor).
//
// Requires WebGL2 (GLSL ES 3.00): texelFetch (in the vertex stage) + sized uniform arrays.

export function packWeights(weights) {
  const flat = [];
  for (const layer of weights.layers) {
    const W = layer.W, b = layer.b;
    for (let o = 0; o < W.length; o++) {
      for (let i = 0; i < W[o].length; i++) flat.push(W[o][i]);
      flat.push(b[o]);
    }
  }
  return new Float32Array(flat);
}

export function buildShaders(arch) {
  const IN = arch.in_features, HID = arch.hidden, HL = arch.hidden_layers, OUT = arch.out_features;
  const NB = 11;

  const vertex = /* glsl */`
    precision highp float;
    #define IN ${IN}
    #define HID ${HID}
    #define HL ${HL}
    #define OUT ${OUT}
    #define NB ${NB}
    #define OMEGA0 ${arch.omega0.toFixed(1)}
    #define LAB_L0 ${(arch.lab_l0 ?? 5).toFixed(1)}
    #define LAB_LSPAN ${(arch.lab_lspan ?? 90).toFixed(1)}
    #define LAB_AB ${(arch.lab_ab ?? 80).toFixed(1)}

    out vec3 vColor;
    uniform vec3 u_bodyEcef[NB];     // Earth-fixed body (sub-point) directions, per frame
    uniform sampler2D u_weights;     // SIREN weights R32F, per-neuron [W..., bias]
    uniform int u_wtexW;

    float wgt(int idx){ return texelFetch(u_weights, ivec2(idx % u_wtexW, idx / u_wtexW), 0).r; }

    vec3 gamutSoft(vec3 c){
      float luma = clamp(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
      vec3 ch = c - luma; float s = 1.0;
      if(ch.r >  1e-5) s = min(s, (1.0 - luma)/ch.r); else if(ch.r < -1e-5) s = min(s, luma/(-ch.r));
      if(ch.g >  1e-5) s = min(s, (1.0 - luma)/ch.g); else if(ch.g < -1e-5) s = min(s, luma/(-ch.g));
      if(ch.b >  1e-5) s = min(s, (1.0 - luma)/ch.b); else if(ch.b < -1e-5) s = min(s, luma/(-ch.b));
      return vec3(luma) + ch * clamp(s, 0.0, 1.0);
    }

    void main(){
      // singularity-free local basis from the (z-negated) field-frame normal
      vec3 N = normalize(position);
      vec3 u = vec3(N.x, N.y, -N.z);
      vec3 nn = vec3(0.0,1.0,0.0) - u*u.y;
      float nl = length(nn);
      vec3 nhat = nl > 1e-6 ? nn/nl : vec3(1.0,0.0,0.0);
      vec3 ehat = cross(u, nhat);

      float inp[IN];
      for(int b=0;b<NB;b++){
        vec3 d = u_bodyEcef[b];
        inp[b*3+0] = dot(d, nhat);     // North
        inp[b*3+1] = dot(d, ehat);     // East
        inp[b*3+2] = dot(d, u);        // Zenith
      }
      int k = 33;                      // 55 chords, fixed i<j order
      for(int i=0;i<NB;i++){
        vec3 vi = vec3(inp[i*3], inp[i*3+1], inp[i*3+2]);
        for(int j=i+1;j<NB;j++){
          vec3 vj = vec3(inp[j*3], inp[j*3+1], inp[j*3+2]);
          inp[k] = dot(vi, vj); k++;
        }
      }

      int wi = 0;
      float cur[HID];
      float nxt[HID];
      for(int o=0;o<HID;o++){ float s=0.0; for(int i=0;i<IN;i++){ s+=wgt(wi)*inp[i]; wi++; } s+=wgt(wi); wi++; cur[o]=sin(OMEGA0*s); }
      for(int l=1;l<HL;l++){
        for(int o=0;o<HID;o++){ float s=0.0; for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; } s+=wgt(wi); wi++; nxt[o]=sin(OMEGA0*s); }
        for(int o=0;o<HID;o++) cur[o]=nxt[o];
      }
      float z[OUT];
      for(int o=0;o<OUT;o++){ float s=0.0; for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; } s+=wgt(wi); wi++; z[o]=s; }

      vec3 Lab = vec3(LAB_L0 + LAB_LSPAN/(1.0+exp(-z[0])), LAB_AB*tanh(z[1]), LAB_AB*tanh(z[2]));
      float fy=(Lab.x+16.0)/116.0, fx=fy+Lab.y/500.0, fz=fy-Lab.z/200.0, dlt=6.0/29.0;
      vec3 xyz;
      xyz.x=(fx>dlt)?fx*fx*fx:3.0*dlt*dlt*(fx-4.0/29.0);
      xyz.y=(fy>dlt)?fy*fy*fy:3.0*dlt*dlt*(fy-4.0/29.0);
      xyz.z=(fz>dlt)?fz*fz*fz:3.0*dlt*dlt*(fz-4.0/29.0);
      xyz*=vec3(0.95047,1.0,1.08883);
      vec3 rgb=vec3(
         3.2404542*xyz.x -1.5371385*xyz.y -0.4985314*xyz.z,
        -0.9692660*xyz.x +1.8760108*xyz.y +0.0415560*xyz.z,
         0.0556434*xyz.x -0.2040259*xyz.y +1.0572252*xyz.z);
      vColor = clamp(gamutSoft(rgb), 0.0, 1.0);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  const fragment = /* glsl */`
    precision highp float;
    in vec3 vColor;
    out vec4 fragColor;
    uniform float u_opacity;
    vec3 toSRGB(vec3 c){ return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4))-0.055, step(0.0031308, c)); }
    void main(){ fragColor = vec4(toSRGB(clamp(vColor, 0.0, 1.0)), u_opacity); }`;

  return { vertex, fragment };
}
