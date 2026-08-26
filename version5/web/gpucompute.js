// gpucompute.js — WebGPU compute pipeline for the geographic feature engine.
//
// Migrates the ~1.5M-trig-op buildFeatures() loop off the JS main thread onto the GPU
// via skycompute.wgsl: one thread per grid point, all N observers in parallel. Runs on
// its OWN GPUDevice, entirely separate from Three.js's WebGLRenderer context (they
// coexist without conflict). Returns null if WebGPU is unavailable, so the caller falls
// back to the CPU path.
//
// Data transfer uses Option B (staging buffer + mapAsync readback): the [N,12,6] and
// [N,3] float arrays are pulled back to JS and handed to onnxruntime-web. Even with the
// copy this is far faster than the sequential JS trig it replaces, and it needs no
// GPUDevice sharing with ORT (which Option A / zero-copy would require — see NOTE).

export async function createGpuFeatureEngine({
  gridW, gridH, nBodies, rawFeatures, obsFeatures, shaderUrl = "skycompute.wgsl",
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
  const featCount = N * nBodies * rawFeatures;           // [N,12,6]
  const obsCount = N * obsFeatures;                       // [N,3]
  const featBytes = featCount * 4, obsBytes = obsCount * 4;
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
  const bodyBuf = device.createBuffer({ size: nBodies * 5 * 4, usage: U.STORAGE | U.COPY_DST });
  const outFeat = device.createBuffer({ size: featBytes, usage: U.STORAGE | U.COPY_SRC });
  const outObs = device.createBuffer({ size: obsBytes, usage: U.STORAGE | U.COPY_SRC });
  const stageFeat = device.createBuffer({ size: featBytes, usage: U.COPY_DST | U.MAP_READ });
  const stageObs = device.createBuffer({ size: obsBytes, usage: U.COPY_DST | U.MAP_READ });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: bodyBuf } },
      { binding: 2, resource: { buffer: outFeat } },
      { binding: 3, resource: { buffer: outObs } },
    ],
  });

  // Params: [gast:f32, eps:f32, grid_w:u32, grid_h:u32, n:u32, pad*3]. grid/n are
  // static; gast/eps are rewritten each frame.
  const paramsAB = new ArrayBuffer(32);
  const paramsF = new Float32Array(paramsAB), paramsU = new Uint32Array(paramsAB);
  paramsU[2] = gridW; paramsU[3] = gridH; paramsU[4] = N;
  const bodyArr = new Float32Array(nBodies * 5);         // [ra,dec,lam,bet,vel] x bodies
  const workgroups = Math.ceil(N / 64);

  // Compute one frame; writes the raw features into featOut/obsOut (reused buffers).
  async function compute(state, featOut, obsOut) {
    paramsF[0] = state.gast; paramsF[1] = state.eps;
    device.queue.writeBuffer(paramsBuf, 0, paramsAB);
    for (let b = 0; b < nBodies; b++) {
      const k = b * 5;
      bodyArr[k] = state.ra[b]; bodyArr[k + 1] = state.dec[b];
      bodyArr[k + 2] = state.lam[b]; bodyArr[k + 3] = state.bet[b]; bodyArr[k + 4] = state.vel[b];
    }
    device.queue.writeBuffer(bodyBuf, 0, bodyArr);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups);               // ceil(N / 64) workgroups of 64
    pass.end();
    enc.copyBufferToBuffer(outFeat, 0, stageFeat, 0, featBytes);
    enc.copyBufferToBuffer(outObs, 0, stageObs, 0, obsBytes);
    device.queue.submit([enc.finish()]);

    await Promise.all([
      stageFeat.mapAsync(GPUMapMode.READ),
      stageObs.mapAsync(GPUMapMode.READ),
    ]);
    featOut.set(new Float32Array(stageFeat.getMappedRange()));
    obsOut.set(new Float32Array(stageObs.getMappedRange()));
    stageFeat.unmap();
    stageObs.unmap();
  }

  // NOTE (Option A, zero-copy to ORT): to skip the readback entirely, set
  // ort.env.webgpu.device = device BEFORE creating the InferenceSession, then wrap
  // outFeat/outObs with ort.Tensor.fromGpuBuffer(...) and run with IO binding. That
  // keeps the data on the GPU but requires this device to be ORT's device and a
  // WebGPU-provider build that supports GPU input tensors; validate live before
  // enabling. Until then Option B (above) is the robust, portable path.

  return { device, compute, N, workgroups, ready: true, backend: "webgpu" };
}
