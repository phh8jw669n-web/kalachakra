// geo.js — the world map that sits UNDER the energy field.
//
// A dark ocean sphere + real Natural-Earth coastlines (coastlines.json) + a lat/lon
// graticule, so the field reads as a geographic overlay. If an equirectangular `earth.jpg`
// (or earth.png) is dropped next to index.html it is used as a photographic base instead —
// same convention as version5.

import * as THREE from "three";

const D2R = Math.PI / 180;

export function llToVec(latDeg, lonDeg, r = 1) {
  const la = latDeg * D2R, lo = lonDeg * D2R, cl = Math.cos(la);
  return new THREE.Vector3(cl * Math.cos(lo), Math.sin(la), cl * Math.sin(lo)).multiplyScalar(r);
}

export function buildOcean(r, color = 0x0b1a2b) {
  return new THREE.Mesh(new THREE.SphereGeometry(r, 96, 64),
    new THREE.MeshBasicMaterial({ color }));
}

export async function loadCoastlines(url = "coastlines.json") {
  try {
    const d = await (await fetch(url)).json();
    return d.lines || [];
  } catch { return []; }
}

export function buildCoastlines(rings, r = 1.004, color = 0x5c7fa3, opacity = 0.9) {
  const pts = [];
  for (const ring of rings) {
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      pts.push(llToVec(a[1], a[0], r), llToVec(b[1], b[0], r));
    }
  }
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
}

export function buildGraticule(r = 1.003, stepDeg = 30, color = 0x2a4055, opacity = 0.4) {
  const pts = [], seg = 96;
  for (let lat = -60; lat <= 60; lat += stepDeg)
    for (let k = 0; k < seg; k++)
      pts.push(llToVec(lat, -180 + k / seg * 360, r), llToVec(lat, -180 + (k + 1) / seg * 360, r));
  for (let lon = -180; lon < 180; lon += stepDeg)
    for (let k = 0; k < seg; k++)
      pts.push(llToVec(-90 + k / seg * 180, lon, r), llToVec(-90 + (k + 1) / seg * 180, lon, r));
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
}

// Optional photographic base: resolves to a textured mesh if earth.jpg/png loads, else null.
export function tryEarthTexture(r, onLoad) {
  const loader = new THREE.TextureLoader();
  for (const url of ["earth.jpg", "earth.png"]) {
    loader.load(url, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 96, 64),
        new THREE.MeshBasicMaterial({ map: tex }));
      // Three's sphere UV: u=0 at +X seam; our lon uses atan2(z,x), so rotate to align.
      mesh.rotation.y = -Math.PI / 2;
      onLoad(mesh);
    }, undefined, () => { /* not present — silent */ });
  }
}
