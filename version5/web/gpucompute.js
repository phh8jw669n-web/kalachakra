// gpucompute.js — WebGPU compute pipeline for the 50-D state engine (v5.1).
//
// Builds the Zero-Redundancy [N,50] physical state on the GPU (skycompute.wgsl): one
// thread per grid point, all N observers in parallel. Runs on its OWN GPUDevice,
// separate from the Three.js WebGL context (they coexist). The 44 time-only body dims
// are precomputed on the CPU (11 bodies, negligible) and passed in; the shader copies
// them and computes each observer's Ascendant/Midheaven. Returns null when WebGPU is
// unavailable, so the caller falls back to the CPU path.
//
// Data transfer uses Option B (staging buffer + mapAsync readback), then hands the
// [N,50] Float32Array to onnxruntime-web. NOTE on Option A (zero-copy) at the bottom.

import { mlBodyState, N_ML_BODIES } from "./skymath.js";

export async function createGpuFeatureEngine({
  gridW, gridH, stateDim, shaderUrl = "skycompute.wgsl",
}) {
  if (typeof navigator === "undefined" || !navigator.gpu) return null;

  let device;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return null;
    device = await adapter.requestDevice();
  } catch {
    return null;
  }
  if (!device) return null;

  let code;
  try {
    code = await (await fetch(shaderUrl)).text();
  } catch {
    device.destroy?.();
    return null;
  }

  const N = gridW * gridH;
  const stateBytes = N * stateDim * 4;
  const bodyBytes = N_ML_BODIES * 4 * 4;                  // 44 floats
  const U = GPUBufferUsage;

  let pipeline;
  try {
    const module = device.createShaderModule({ code });
    pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "main" },
    });
  } catch {
    device.destroy?.();
    return null;
  }

  const paramsBuf = device.createBuffer({ size: 32, usage: U.UNIFORM | U.COPY_DST });
  const bodyBuf = device.createBuffer({ size: bodyBytes, usage: U.STORAGE | U.COPY_DST });
  const outState = device.createBuffer({ size: stateBytes, usage: U.STORAGE | U.COPY_SRC });
  const stage = device.createBuffer({ size: stateBytes, usage: U.COPY_DST | U.MAP_READ });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bodyBuf } },
      { binding: 2, resource: { buffer: outState } },
    ],
  });

  // Params: [gast:f32, eps:f32, grid_w:u32, grid_h:u32, n:u32, pad*3].
  const paramsAB = new ArrayBuffer(32);
  const paramsF = new Float32Array(paramsAB), paramsU = new Uint32Array(paramsAB);
  paramsU[2] = gridW; paramsU[3] = gridH; paramsU[4] = N;
  const workgroups = Math.ceil(N / 64);

  // Compute one frame; writes the [N,50] state into stateOut (a reused Float32Array).
  async function compute(tstate, stateOut) {
    paramsF[0] = tstate.gast; paramsF[1] = tstate.eps;
    device.queue.writeBuffer(paramsBuf, 0, paramsAB);
    device.queue.writeBuffer(bodyBuf, 0, mlBodyState(tstate));   // 44 time-only dims

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);                 // ceil(N / 64) workgroups of 64
    pass.end();
    enc.copyBufferToBuffer(outState, 0, stage, 0, stateBytes);
    device.queue.submit([enc.finish()]);

    await stage.mapAsync(GPUMapMode.READ);
    stateOut.set(new Float32Array(stage.getMappedRange()));
    stage.unmap();
  }

  // NOTE (Option A, zero-copy to ORT): to skip the readback, set ort.env.webgpu.device
  // = device BEFORE creating the InferenceSession, then wrap `outState` with
  // ort.Tensor.fromGpuBuffer(..., {dataType:'float32', dims:[N, stateDim]}) and run with
  // IO binding — keeping the data on the GPU. Requires this device to be ORT's device
  // and a WebGPU-provider build that accepts GPU input tensors; validate live before
  // enabling. Until then Option B (above) is the robust, portable path.

  return { device, compute, N, workgroups, ready: true, backend: "webgpu" };
}
