// fieldworker.js — bakes the regional field grid off the main thread, so the globe render
// loop never blocks. It owns the (reused) SIREN baker and answers "bake this Julian Date"
// requests, transferring the finished RGBA buffer back to the main thread.
//
// Protocol:
//   main -> worker : {type:"init", weights}
//                    {type:"bake", jd, w, h, buf}   (buf: optional ArrayBuffer to reuse)
//   worker -> main : {type:"ready"}
//                    {type:"field", jd, w, h, buf}   (buf transferred back)

import { makeFieldBaker } from "./field.js";

let baker = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    baker = makeFieldBaker(msg.weights);
    self.postMessage({ type: "ready" });
    return;
  }
  if (msg.type === "bake" && baker) {
    const { jd, w, h } = msg;
    const reuse = msg.buf && msg.buf.byteLength === w * h * 4
      ? new Uint8ClampedArray(msg.buf) : null;
    const rgba = baker.bake(jd, w, h, reuse);
    self.postMessage({ type: "field", jd, w, h, buf: rgba.buffer }, [rgba.buffer]);
  }
};
