// shader6.js — the SIREN globe shader (Module 2/3). Runs the FULL topocentric
// ephemeris + SIREN + L*a*b*->sRGB per pixel in GLSL, so the globe is infinite-
// resolution. Weights arrive in a float data texture; the architecture is templated
// via #defines. The ephemeris is a line-for-line transcription of ephemeris6.js and
// the SIREN walk matches packWeights() below (both mirror-tested in Node).
//
// Time precision (Module 3): the JD is split into u_baseDays (exact integer days from
// J2000, constant during animation) + u_timeOffset (small, animated), so d = base +
// offset keeps the animated part precise and scrubbing stays smooth.

// Flatten weights.json into the per-neuron [W_row..., bias] order the shader walks.
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

  const vertex = /* glsl */`
    varying vec3 vObj;
    void main() {
      vObj = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  const fragment = /* glsl */`
    precision highp float;
    #define IN ${IN}
    #define HID ${HID}
    #define HL ${HL}
    #define OUT ${OUT}
    #define OMEGA0 ${arch.omega0.toFixed(1)}
    #define PI 3.141592653589793
    #define DEG 0.017453292519943295

    varying vec3 vObj;
    uniform float u_baseDays;      // exact integer days since J2000 (constant during anim)
    uniform float u_timeOffset;    // small animated delta (days)
    uniform sampler2D u_weights;   // SIREN weights, R32F, row = per-neuron [W..., bias]
    uniform int u_wtexW;           // weight texture width
    uniform vec3 u_labOffset;      // display gauge shift
    uniform float u_exposure;      // colour gain for vividness

    // JPL Keplerian elements (Standish): a,e,I,L,peri,node  @J2000 + rates/century.
    const float ELEM[54] = float[54](
      0.38709927,0.20563593,7.00497902,252.25032350,77.45779628,48.33076593,
      0.72333566,0.00677672,3.39467605,181.97909950,131.60246718,76.67984255,
      1.00000261,0.01671123,-0.00001531,100.46457166,102.93768193,0.0,
      1.52371034,0.09339410,1.84969142,-4.55343205,-23.94362959,49.55953891,
      5.20288700,0.04838624,1.30439695,34.39644051,14.72847983,100.47390909,
      9.53667594,0.05386179,2.48599187,49.95424423,92.59887831,113.66242448,
      19.18916464,0.04725744,0.77263783,313.23810451,170.95427630,74.01692503,
      30.06992276,0.00859048,1.77004347,-55.12002969,44.96476227,131.78422574,
      39.48211675,0.24882730,17.14001206,238.92903833,224.06891629,110.30393684);
    const float RATE[54] = float[54](
      0.00000037,0.00001906,-0.00594749,149472.67411175,0.16047689,-0.12534081,
      0.00000390,-0.00004107,-0.00078890,58517.81538729,0.00268329,-0.27769418,
      0.00000562,-0.00004392,-0.01294668,35999.37244981,0.32327364,0.0,
      0.00001847,0.00007882,-0.00813131,19140.30268499,0.44441088,-0.29257343,
      -0.00011607,-0.00013253,-0.00183714,3034.74612775,0.21252668,0.20469106,
      -0.00125060,-0.00050991,0.00193609,1222.49362201,-0.41897216,-0.28867794,
      -0.00196176,-0.00004397,-0.00242939,428.48202785,0.40805281,0.04240589,
      0.00026291,0.00005105,0.00035372,218.45945325,-0.32241464,-0.00508664,
      -0.00031596,0.00005170,0.00004818,145.20780515,-0.04062942,-0.01183482);

    float wrap180(float d){ return mod(d + 180.0, 360.0) - 180.0; }

    float keplerE(float M, float e){
      float E = M + e*sin(M);
      for(int k=0;k<6;k++) E = E - (E - e*sin(E) - M)/(1.0 - e*cos(E));
      return E;
    }

    vec3 heliocentric(float T, int i){
      int o = i*6;
      float a=ELEM[o]+RATE[o]*T;
      float e=ELEM[o+1]+RATE[o+1]*T;
      float inc=(ELEM[o+2]+RATE[o+2]*T)*DEG;
      float L=ELEM[o+3]+RATE[o+3]*T;
      float peri=ELEM[o+4]+RATE[o+4]*T;
      float node=(ELEM[o+5]+RATE[o+5]*T)*DEG;
      float omega=(peri-ELEM[o+5]-RATE[o+5]*T)*DEG;
      float M=wrap180(L-peri)*DEG;
      float E=keplerE(M,e);
      float xp=a*(cos(E)-e);
      float yp=a*sqrt(1.0-e*e)*sin(E);
      float co=cos(omega),so=sin(omega),ci=cos(inc),si=sin(inc),cn=cos(node),sn=sin(node);
      return vec3(
        (co*cn-so*sn*ci)*xp + (-so*cn-co*sn*ci)*yp,
        (co*sn+so*cn*ci)*xp + (-so*sn+co*cn*ci)*yp,
        (so*si)*xp + (co*si)*yp);
    }

    vec3 moonDir(float d){
      float Lp=(218.3164477+13.17639648*d)*DEG;
      float D=(297.8501921+12.19074920*d)*DEG;
      float M=(357.5291092+0.98560028*d)*DEG;
      float Mp=(134.9633964+13.06499295*d)*DEG;
      float F=(93.2720950+13.22935024*d)*DEG;
      float lon=(Lp/DEG + 6.289*sin(Mp)+1.274*sin(2.0*D-Mp)+0.658*sin(2.0*D)
        +0.214*sin(2.0*Mp)-0.186*sin(M)-0.114*sin(2.0*F)
        +0.059*sin(2.0*D-2.0*Mp)+0.057*sin(2.0*D-M-Mp))*DEG;
      float lat=(5.128*sin(F)+0.280*sin(Mp+F)+0.277*sin(Mp-F)+0.173*sin(2.0*D-F)
        +0.055*sin(2.0*D-Mp+F)+0.046*sin(2.0*D-Mp-F))*DEG;
      float cb=cos(lat);
      return vec3(cb*cos(lon), cb*sin(lon), sin(lat));
    }

    float gmstDeg(float d){
      float T=d/36525.0;
      float g=280.46061837+360.98564736629*d+0.000387933*T*T - T*T*T/38710000.0;
      return mod(g,360.0);
    }

    float wgt(int idx){ return texelFetch(u_weights, ivec2(idx % u_wtexW, idx / u_wtexW), 0).r; }

    void main(){
      vec3 p = normalize(vObj);
      float lat = asin(clamp(p.y,-1.0,1.0));
      float lon = atan(p.z, p.x);

      float d = u_baseDays + u_timeOffset;
      float T = d/36525.0;
      vec3 earth = heliocentric(T, 2);

      float eps=(23.439291-0.0130042*T)*DEG;
      float ce=cos(eps), se=sin(eps);
      float lst=gmstDeg(d)*DEG + lon;
      float sphi=sin(lat), cphi=cos(lat);

      float sky[IN];
      // fill the 33-D tensor: 11 bodies x (North,East,Up)
      for(int bcount=0; bcount<11; bcount++){
        vec3 g;
        if(bcount==0){ g=-earth; }
        else if(bcount==1){ g=moonDir(d); }
        else if(bcount==10){ float nl=(125.04452-0.05295377*d)*DEG; g=vec3(cos(nl),sin(nl),0.0); }
        else {
          // bodies 2..9 = Mercury,Venus,Mars,..,Pluto -> element rows skipping Earth(2):
          // 2->0, 3->1, 4->3, 5->4, ... 9->8
          int erow = (bcount<=3) ? (bcount-2) : (bcount-1);
          g = heliocentric(T, erow) - earth;
        }
        g = normalize(g);
        float xq=g.x, yq=g.y*ce-g.z*se, zq=g.y*se+g.z*ce;
        float ra=atan(yq,xq);
        float dec=asin(clamp(zq,-1.0,1.0));
        float H=lst-ra;
        float sd=sin(dec),cd=cos(dec),sH=sin(H),cH=cos(H);
        sky[bcount*3+0]=sd*cphi - cd*sphi*cH;     // North
        sky[bcount*3+1]=-cd*sH;                    // East
        sky[bcount*3+2]=sd*sphi + cd*cphi*cH;      // Up
      }

      // SIREN forward (weights walked per-neuron: [W_row..., bias])
      int wi=0;
      float cur[HID];
      // layer 0: HID x IN, sin
      for(int o=0;o<HID;o++){
        float s=0.0;
        for(int i=0;i<IN;i++){ s+=wgt(wi)*sky[i]; wi++; }
        s+=wgt(wi); wi++;
        cur[o]=sin(OMEGA0*s);
      }
      // hidden sin layers 1..HL-1: HID x HID
      for(int l=1;l<HL;l++){
        float nxt[HID];
        for(int o=0;o<HID;o++){
          float s=0.0;
          for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; }
          s+=wgt(wi); wi++;
          nxt[o]=sin(OMEGA0*s);
        }
        for(int o=0;o<HID;o++) cur[o]=nxt[o];
      }
      // output linear: OUT x HID
      float lab[OUT];
      for(int o=0;o<OUT;o++){
        float s=0.0;
        for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; }
        s+=wgt(wi); wi++;
        lab[o]=s;
      }

      vec3 Lab = vec3(lab[0], lab[1], lab[2]) + u_labOffset;
      Lab.yz *= u_exposure;

      // L*a*b* (D65) -> linear sRGB
      float fy=(Lab.x+16.0)/116.0, fx=fy+Lab.y/500.0, fz=fy-Lab.z/200.0;
      float dlt=6.0/29.0;
      vec3 xyz;
      xyz.x = (fx>dlt)?fx*fx*fx:3.0*dlt*dlt*(fx-4.0/29.0);
      xyz.y = (fy>dlt)?fy*fy*fy:3.0*dlt*dlt*(fy-4.0/29.0);
      xyz.z = (fz>dlt)?fz*fz*fz:3.0*dlt*dlt*(fz-4.0/29.0);
      xyz *= vec3(0.95047,1.0,1.08883);
      vec3 rgb=vec3(
         3.2404542*xyz.x -1.5371385*xyz.y -0.4985314*xyz.z,
        -0.9692660*xyz.x +1.8760108*xyz.y +0.0415560*xyz.z,
         0.0556434*xyz.x -0.2040259*xyz.y +1.0572252*xyz.z);
      // smooth gamut clamp (soft, avoids harsh clipping)
      rgb = rgb/(1.0+max(vec3(0.0), rgb-1.0));
      rgb = clamp(rgb, 0.0, 1.0);
      rgb = mix(12.92*rgb, 1.055*pow(rgb, vec3(1.0/2.4))-0.055, step(0.0031308, rgb));
      gl_FragColor = vec4(rgb, 1.0);
    }`;

  return { vertex, fragment };
}
