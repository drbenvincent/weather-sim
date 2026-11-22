export async function requestDevice(): Promise<GPUDevice> {
  if (!navigator.gpu) {
    throw new Error("This browser has no WebGPU. Chrome, Edge, or another current WebGPU browser is required.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU is present, but no adapter could be requested.");
  }
  return adapter.requestDevice();
}

export async function shaderError(device: GPUDevice, module: GPUShaderModule, label: string): Promise<void> {
  device.pushErrorScope("validation");
  const info = await module.getCompilationInfo();
  const failure = info.messages.find((message) => message.type === "error");
  const scopeError = await device.popErrorScope();
  if (failure) {
    throw new Error(`${label}: ${failure.lineNum}:${failure.linePos} ${failure.message}`);
  }
  if (scopeError) {
    throw new Error(`${label}: ${scopeError.message}`);
  }
}
