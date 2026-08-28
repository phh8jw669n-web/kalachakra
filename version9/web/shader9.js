// shader9.js — the version9 Topocentric Self-Attention field, rendered OFF the hot path.
//
// The micro-transformer is expensive (~10^5 ops/sample), so running it per vertex on every
// frame overloads weak GPUs / software renderers and hangs the tab during rotation. Instead we
// DECOUPLE compute from framerate with two programs:
//
//   • FIELD  (buildShaders().field): a full-screen pass that, for each texel of an
//     equirectangular (lon,lat) render target, builds the local horizon basis (latitude clamped
//     off the exact poles), runs the WHOLE network from the weight texture (identical maths to
//     attn9.js / attention.py) and writes the OKLab chroma (a,b), encoded to [0,1]. Rendered
//     ONCE per time change (not per frame).
//   • GLOBE  (buildShaders().globe): a trivial pass on the sphere that samples that field
//     texture by (lon,lat) from the surface point and reconstructs OKLab -> sRGB PER PIXEL.
//     Interpolating the Cartesian (a,b) (not a hue angle) avoids the branch-cut "rainbow bead"
//     artifact; this runs every frame, so rotating/zooming is nearly free.
//
// The globe reads lon = atan2(-z, x) so the world map stays un-mirrored while the field is
// physically exact. Requires WebGL2 (GLSL ES 3.00): texelFetch.
//
// Weights are packed by packWeights() in the EXACT order the shader reads them; buildShaders()
// injects the matching offsets as #defines so the shader indexes the texture directly.

export function packWeights(w) {
  const out = [];
  const mat = (M) => { for (let o = 0; o < M.length; o++) for (let i = 0; i < M[o].length; i++) out.push(M[o][i]); };
  const vec = (v) => { for (let i = 0; i < v.length; i++) out.push(v[i]); };
  mat(w.W_in); vec(w.b_in); mat(w.E_body);
  for (const blk of w.blocks) {
    mat(blk.Wq); vec(blk.bq); mat(blk.Wk); vec(blk.bk); mat(blk.Wv); vec(blk.bv);
    mat(blk.W1); vec(blk.b1); mat(blk.W2); vec(blk.b2); out.push(blk.tau);
  }
  vec(w.q_pool); out.push(w.tau_pool);
  mat(w.Wo1); vec(w.bo1); mat(w.Wo2); vec(w.bo2);
  return new Float32Array(out);
}

export function buildShaders(arch) {
  const D = arch.d_model, DFF = arch.d_ff, DHEAD = arch.d_head;
  const NB = arch.n_bodies, NBL = arch.n_blocks, TOK = arch.token_dim;
  const OUTC = arch.out_features ?? 2;      // pure chroma: a*, b*
  // ---- weight-texture offsets (must mirror packWeights order exactly) ----
  let o = 0;
  const OFF_WIN = o; o += D * TOK;
  const OFF_BIN = o; o += D;
  const OFF_EBODY = o; o += NB * D;
  const OFF_BLOCKS = o;
  const SB = (D * D + D) * 3 + (DFF * D + DFF) + (D * DFF + D) + 1;
  o += SB * NBL;
  const OFF_QPOOL = o; o += D;
  const OFF_TAUPOOL = o; o += 1;
  const OFF_WO1 = o; o += DHEAD * D;
  const OFF_BO1 = o; o += DHEAD;
  const OFF_WO2 = o; o += OUTC * DHEAD;
  const OFF_BO2 = o; o += OUTC;
  // within-block sub-offsets (relative to a block base)
  const BWQ = 0, BBQ = D * D, BWK = D * D + D, BBK = 2 * D * D + D, BWV = 2 * D * D + 2 * D,
    BBV = 3 * D * D + 2 * D, BW1 = 3 * D * D + 3 * D, BB1 = 3 * D * D + 3 * D + DFF * D,
    BW2 = 3 * D * D + 3 * D + DFF * D + DFF, BB2 = 3 * D * D + 3 * D + DFF * D + DFF + D * DFF,
    BTAU = SB - 1;

  const defs = `
    #define D ${D}
    #define DFF ${DFF}
    #define DHEAD ${DHEAD}
    #define NB ${NB}
    #define NBL ${NBL}
    #define TOK ${TOK}
    #define INV_SQRTD ${(1.0 / Math.sqrt(D)).toFixed(8)}
    #define VISB ${(arch.vis_bias ?? 3.0).toFixed(4)}
    #define OKL_L ${(arch.okl_l ?? 0.5).toFixed(5)}
    #define OKL_CMAX ${(arch.okl_cmax ?? 0.4).toFixed(5)}
    #define OFF_WIN ${OFF_WIN}
    #define OFF_BIN ${OFF_BIN}
    #define OFF_EBODY ${OFF_EBODY}
    #define OFF_BLOCKS ${OFF_BLOCKS}
    #define SB ${SB}
    #define OFF_QPOOL ${OFF_QPOOL}
    #define OFF_TAUPOOL ${OFF_TAUPOOL}
    #define OFF_WO1 ${OFF_WO1}
    #define OFF_BO1 ${OFF_BO1}
    #define OFF_WO2 ${OFF_WO2}
    #define OFF_BO2 ${OFF_BO2}
    #define BWQ ${BWQ}
    #define BBQ ${BBQ}
    #define BWK ${BWK}
    #define BBK ${BBK}
    #define BWV ${BWV}
    #define BBV ${BBV}
    #define BW1 ${BW1}
    #define BB1 ${BB1}
    #define BW2 ${BW2}
    #define BB2 ${BB2}
    #define BTAU ${BTAU}`;

  // The WHOLE network as a GLSL function returning OKLab chroma (a,b) from the local horizon
  // basis. No colour conversion here — the field stores (a,b) and the globe converts per pixel.
  const net = /* glsl */`
    ${defs}
    uniform vec3 u_bodyEcef[NB];     // Earth-fixed body (sub-point) directions, per time step
    uniform sampler2D u_weights;     // packed attention weights (R32F)
    uniform int u_wtexW;

    float W(int idx){ return texelFetch(u_weights, ivec2(idx % u_wtexW, idx / u_wtexW), 0).r; }
    // Whole topocentric attention net -> OKLab chroma (a,b) for horizon frame (nhat,ehat,up).
    vec2 netAB(vec3 nhat, vec3 ehat, vec3 up){
      float inp[NB*TOK]; float vis[NB];
      for(int b=0;b<NB;b++){
        vec3 d = u_bodyEcef[b];
        inp[b*TOK+0]=dot(d,nhat); inp[b*TOK+1]=dot(d,ehat); inp[b*TOK+2]=dot(d,up);
        vis[b] = VISB * inp[b*TOK+2];
      }
      float tok[NB*D];
      for(int b=0;b<NB;b++)
        for(int oo=0;oo<D;oo++){
          float s = W(OFF_BIN+oo) + W(OFF_EBODY + b*D + oo);
          for(int i=0;i<TOK;i++) s += W(OFF_WIN + oo*TOK + i) * inp[b*TOK+i];
          tok[b*D+oo] = s;
        }
      float kk[NB*D]; float vv[NB*D];
      for(int bl=0; bl<NBL; bl++){
        int bb = OFF_BLOCKS + bl*SB;
        float sc = INV_SQRTD * W(bb+BTAU);
        for(int b=0;b<NB;b++)
          for(int oo=0;oo<D;oo++){
            float sk = W(bb+BBK+oo), sv = W(bb+BBV+oo);
            for(int i=0;i<D;i++){ float ti = tok[b*D+i]; sk += W(bb+BWK+oo*D+i)*ti; sv += W(bb+BWV+oo*D+i)*ti; }
            kk[b*D+oo]=sk; vv[b*D+oo]=sv;
          }
        for(int qi=0; qi<NB; qi++){
          float q[D];
          for(int oo=0;oo<D;oo++){ float sq = W(bb+BBQ+oo); for(int i=0;i<D;i++) sq += W(bb+BWQ+oo*D+i)*tok[qi*D+i]; q[oo]=sq; }
          float sco[NB]; float smax=-1e30;
          for(int j=0;j<NB;j++){ float s=0.0; for(int d=0;d<D;d++) s+=q[d]*kk[j*D+d]; s=s*sc+vis[j]; sco[j]=s; if(s>smax) smax=s; }
          float Z=0.0; for(int j=0;j<NB;j++){ sco[j]=exp(sco[j]-smax); Z+=sco[j]; }
          for(int d=0;d<D;d++){ float acc=0.0; for(int j=0;j<NB;j++) acc+=sco[j]*vv[j*D+d]; tok[qi*D+d]+=acc/Z; }
        }
        for(int b=0;b<NB;b++){
          float h[DFF];
          for(int oo=0;oo<DFF;oo++){ float s=W(bb+BB1+oo); for(int d=0;d<D;d++) s+=W(bb+BW1+oo*D+d)*tok[b*D+d]; h[oo]=tanh(s); }
          for(int oo=0;oo<D;oo++){ float s=W(bb+BB2+oo); for(int d=0;d<DFF;d++) s+=W(bb+BW2+oo*DFF+d)*h[d]; tok[b*D+oo]+=s; }
        }
      }
      float psc = INV_SQRTD * W(OFF_TAUPOOL);
      float pw[NB]; float smax=-1e30;
      for(int b=0;b<NB;b++){ float s=0.0; for(int d=0;d<D;d++) s+=tok[b*D+d]*W(OFF_QPOOL+d); s=s*psc+vis[b]; pw[b]=s; if(s>smax) smax=s; }
      float Z=0.0; for(int b=0;b<NB;b++){ pw[b]=exp(pw[b]-smax); Z+=pw[b]; }
      float pooled[D];
      for(int d=0;d<D;d++){ float acc=0.0; for(int b=0;b<NB;b++) acc+=pw[b]*tok[b*D+d]; pooled[d]=acc/Z; }
      float hh[DHEAD];
      for(int oo=0;oo<DHEAD;oo++){ float s=W(OFF_BO1+oo); for(int d=0;d<D;d++) s+=W(OFF_WO1+oo*D+d)*pooled[d]; hh[oo]=tanh(s); }
      // OKLCH polar head: C = cmax*sigmoid(z0), H = z1 (raw radians) -> OKLab (a,b).
      // Return (a,b) directly (continuous, no branch cut) — the field stores THIS, so all
      // interpolation is Cartesian & perceptually uniform (no rainbow-bead hue interpolation).
      float z0=W(OFF_BO2+0), z1=W(OFF_BO2+1);
      for(int d=0;d<DHEAD;d++){ z0+=W(OFF_WO2+0*DHEAD+d)*hh[d]; z1+=W(OFF_WO2+1*DHEAD+d)*hh[d]; }
      float C = OKL_CMAX / (1.0 + exp(-z0));
      return vec2(C*cos(z1), C*sin(z1));
    }`;

  // OKLab (a,b) at the fixed neutral L -> gamma sRGB, with a HUE- and LIGHTNESS-preserving
  // chroma clip to the sRGB gamut boundary (bisection; no hue/luminance shift, no hard clip).
  // Runs PER PIXEL on the globe, on the interpolated (a,b).
  const conv = /* glsl */`
    #define OKL_L ${(arch.okl_l ?? 0.5).toFixed(5)}
    #define OKL_CMAX ${(arch.okl_cmax ?? 0.4).toFixed(5)}
    vec3 toSRGB(vec3 c){ return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4))-0.055, step(0.0031308, c)); }
    vec3 oklab2lin(float L, float a, float b){
      float L_=L+0.3963377774*a+0.2158037573*b;
      float M_=L-0.1055613458*a-0.0638541728*b;
      float S_=L-0.0894841775*a-1.2914855480*b;
      float l=L_*L_*L_, m=M_*M_*M_, s=S_*S_*S_;
      return vec3( 4.0767416621*l -3.3077115913*m +0.2309699292*s,
                  -1.2684380046*l +2.6097574011*m -0.3413193965*s,
                  -0.0041960863*l -0.7034186147*m +1.7076147010*s);
    }
    bool inGamut(vec3 c){ return all(greaterThanEqual(c, vec3(-0.001))) && all(lessThanEqual(c, vec3(1.001))); }
    vec3 abToSRGB(vec2 ab){
      float C = length(ab);
      vec2 dir = C > 1e-9 ? ab / C : vec2(1.0, 0.0);       // hue direction (no atan needed)
      if(!inGamut(oklab2lin(OKL_L, C*dir.x, C*dir.y))){
        float lo=0.0, hi=C;
        for(int i=0;i<14;i++){ float mid=0.5*(lo+hi); if(inGamut(oklab2lin(OKL_L, mid*dir.x, mid*dir.y))) lo=mid; else hi=mid; }
        C=lo;
      }
      return toSRGB(clamp(oklab2lin(OKL_L, C*dir.x, C*dir.y), 0.0, 1.0));
    }`;

  // FIELD pass — full-screen quad over an equirectangular (lon,lat) target; one texel per sky.
  const fieldVertex = /* glsl */`
    precision highp float;
    out vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
  const fieldFragment = /* glsl */`
    precision highp float;
    ${net}
    #define PI 3.14159265358979
    #define LAT_MAX 1.5706256            // 89.99 deg: keep the topocentric frame off the exact pole
    in vec2 vUv;
    out vec4 fragColor;
    void main(){
      float lon = (vUv.x*2.0 - 1.0) * PI;
      float lat = clamp((vUv.y - 0.5) * PI, -LAT_MAX, LAT_MAX);   // stabilise the poles (no gimbal)
      float cl = cos(lat), sl = sin(lat), co = cos(lon), so = sin(lon);
      vec3 up = vec3(cl*co, sl, cl*so);                 // observer zenith (ECEF); |up.y| < 1 now
      vec3 nn = vec3(0.0,1.0,0.0) - up*up.y; float nl = length(nn);
      vec3 nhat = nl > 1e-6 ? nn/nl : vec3(1.0,0.0,0.0);
      vec3 ehat = cross(up, nhat);
      vec2 ab = netAB(nhat, ehat, up);                  // OKLab chroma (a,b)
      fragColor = vec4(ab / (2.0*OKL_CMAX) + 0.5, 0.0, 1.0);      // encode (a,b) in [-cmax,cmax] -> [0,1]
    }`;

  // GLOBE pass — sample the field's interpolated (a,b) by (lon,lat) of the surface point
  // (un-mirrored), then reconstruct OKLab -> sRGB PER PIXEL. Interpolation happens on the
  // Cartesian (a,b), so gradients are smooth & bead-free; gamut clipping is per-pixel exact.
  const globeVertex = /* glsl */`
    precision highp float;
    out vec3 vPos;
    void main(){ vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
  const globeFragment = /* glsl */`
    precision highp float;
    ${conv}
    #define PI 3.14159265358979
    in vec3 vPos;
    out vec4 fragColor;
    uniform sampler2D u_field;
    uniform float u_opacity;
    void main(){
      vec3 p = normalize(vPos);
      float lon = atan(-p.z, p.x);                      // un-mirrored (matches raycaster + map)
      float lat = asin(clamp(p.y, -1.0, 1.0));
      vec2 uv = vec2(lon/(2.0*PI) + 0.5, lat/PI + 0.5);
      vec2 ab = (texture(u_field, uv).rg - 0.5) * (2.0*OKL_CMAX);  // decode interpolated (a,b)
      fragColor = vec4(abToSRGB(ab), u_opacity);
    }`;

  return {
    field: { vertex: fieldVertex, fragment: fieldFragment },
    globe: { vertex: globeVertex, fragment: globeFragment },
  };
}
