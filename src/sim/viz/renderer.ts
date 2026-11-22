import type { FieldView } from "../types.ts";
import { PARTICLE_COUNT } from "../engine/solver.ts";
import { shaderError } from "../engine/gpu.ts";
import fieldWgsl from "../engine/wgsl/render.wgsl?raw";
import prelude from "../engine/wgsl/prelude.wgsl?raw";
import tracerAdvectWgsl from "../engine/wgsl/tracer-advect.wgsl?raw";
import tracerDrawWgsl from "../engine/wgsl/tracer-draw.wgsl?raw";

export type FieldMode = 0 | 1 | 2 | 3 | 4;

export function exaggeration(canvas: HTMLCanvasElement, extentX: number, extentZ: number): number {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width === 0 || height === 0) return 1;
  return (height / width) / (extentZ / extentX);
}

export class Renderer {
  private readonly fieldPipeline: GPURenderPipeline;
  private readonly tracerPipeline: GPURenderPipeline;
  private readonly tracerCompute: GPUComputePipeline;
  private readonly fieldUniform: GPUBuffer;
  private fieldGroup: GPUBindGroup | null = null;
  private tracerDraw: GPUBindGroup | null = null;
  private tracerStep: GPUBindGroup | null = null;
  private scalar: GPUBuffer | null = null;
  private velocity: GPUBuffer | null = null;
  private divergence: GPUBuffer | null = null;

  private constructor(
    private readonly device: GPUDevice,
    private readonly context: GPUCanvasContext,
    fieldPipeline: GPURenderPipeline,
    tracerPipeline: GPURenderPipeline,
    tracerCompute: GPUComputePipeline,
    fieldUniform: GPUBuffer,
  ) {
    this.fieldPipeline = fieldPipeline;
    this.tracerPipeline = tracerPipeline;
    this.tracerCompute = tracerCompute;
    this.fieldUniform = fieldUniform;
  }

  static async create(device: GPUDevice, canvas: HTMLCanvasElement): Promise<Renderer> {
    const format = navigator.gpu.getPreferredCanvasFormat();
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("The canvas could not create a WebGPU context.");
    context.configure({ device, format, alphaMode: "opaque" });
    const fieldModule = device.createShaderModule({ code: fieldWgsl, label: "field" });
    const tracerModule = device.createShaderModule({ code: prelude + tracerAdvectWgsl, label: "tracers" });
    const tracerDrawModule = device.createShaderModule({ code: prelude + tracerDrawWgsl, label: "tracer-draw" });
    await Promise.all([
      shaderError(device, fieldModule, "field"),
      shaderError(device, tracerModule, "tracers"),
      shaderError(device, tracerDrawModule, "tracer-draw"),
    ]);
    const fieldPipeline = device.createRenderPipeline({
      label: "field",
      layout: "auto",
      vertex: { module: fieldModule, entryPoint: "vs" },
      fragment: { module: fieldModule, entryPoint: "fs", targets: [{ format }] },
      primitive: { topology: "triangle-list" },
    });
    const tracerPipeline = device.createRenderPipeline({
      label: "tracer-draw",
      layout: "auto",
      vertex: { module: tracerDrawModule, entryPoint: "vs" },
      fragment: {
        module: tracerDrawModule,
        entryPoint: "fs",
        targets: [{
          format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
    const tracerCompute = device.createComputePipeline({
      label: "tracer-advect",
      layout: "auto",
      compute: { module: tracerModule, entryPoint: "advect" },
    });
    const fieldUniform = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: "render-uniform",
    });
    return new Renderer(device, context, fieldPipeline, tracerPipeline, tracerCompute, fieldUniform);
  }

  resize(canvas: HTMLCanvasElement): void {
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.min(1920, Math.round(canvas.clientWidth * ratio)));
    const height = Math.max(1, Math.round(width * canvas.clientHeight / Math.max(canvas.clientWidth, 1)));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  draw(
    canvas: HTMLCanvasElement,
    view: FieldView,
    tracer: { params: GPUBuffer; display: GPUBuffer; particles: GPUBuffer; velocity: GPUBuffer; scalar: GPUBuffer; refs: GPUBuffer },
    mode: FieldMode,
    time: number,
    divScale: number,
    showTracers: boolean,
  ): void {
    this.writeUniform(canvas, view, mode, time, divScale);
    this.bind(view, tracer);
    const encoder = this.device.createCommandEncoder({ label: "draw" });
    if (showTracers) {
      const compute = encoder.beginComputePass({ label: "tracer-step" });
      compute.setPipeline(this.tracerCompute);
      compute.setBindGroup(0, this.tracerStep!);
      compute.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 64));
      compute.end();
    }
    const texture = this.context.getCurrentTexture();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: texture.createView(),
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0.043, g: 0.055, b: 0.075, a: 1 },
      }],
    });
    pass.setPipeline(this.fieldPipeline);
    pass.setBindGroup(0, this.fieldGroup!);
    pass.draw(3);
    if (showTracers) {
      pass.setPipeline(this.tracerPipeline);
      pass.setBindGroup(0, this.tracerDraw!);
      pass.draw(6, PARTICLE_COUNT);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroy(): void {
    this.fieldUniform.destroy();
  }

  private writeUniform(canvas: HTMLCanvasElement, view: FieldView, mode: FieldMode, time: number, divScale: number): void {
    const data = new ArrayBuffer(48);
    const packed = new DataView(data);
    packed.setUint32(0, mode, true);
    packed.setFloat32(4, canvas.width, true);
    packed.setFloat32(8, canvas.height, true);
    packed.setFloat32(12, time, true);
    packed.setFloat32(16, view.extentX, true);
    packed.setFloat32(20, view.extentZ, true);
    packed.setFloat32(24, view.dx, true);
    packed.setFloat32(28, view.dz, true);
    packed.setUint32(32, view.nx, true);
    packed.setUint32(36, view.nz, true);
    packed.setFloat32(40, Math.max(divScale, 1e-6), true);
    this.device.queue.writeBuffer(this.fieldUniform, 0, data);
  }

  private bind(
    view: FieldView,
    tracer: { params: GPUBuffer; display: GPUBuffer; particles: GPUBuffer; velocity: GPUBuffer; scalar: GPUBuffer; refs: GPUBuffer },
  ): void {
    const scalarChanged = this.scalar !== view.scalar || this.divergence !== view.divergence;
    const velocityChanged = this.velocity !== tracer.velocity;
    if (scalarChanged) {
      this.scalar = view.scalar;
      this.divergence = view.divergence;
      this.fieldGroup = this.device.createBindGroup({
        layout: this.fieldPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.fieldUniform } },
          { binding: 1, resource: { buffer: view.scalar } },
          { binding: 2, resource: { buffer: view.refs } },
          { binding: 3, resource: { buffer: view.divergence } },
        ],
      });
    }
    if (velocityChanged || scalarChanged || this.tracerDraw === null) {
      this.tracerDraw = this.device.createBindGroup({
        layout: this.tracerPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tracer.params } },
          { binding: 1, resource: { buffer: tracer.particles } },
          { binding: 2, resource: { buffer: tracer.velocity } },
          { binding: 3, resource: { buffer: view.scalar } },
          { binding: 4, resource: { buffer: view.refs } },
        ],
      });
    }
    if (velocityChanged || this.tracerStep === null) {
      this.velocity = tracer.velocity;
      this.tracerStep = this.device.createBindGroup({
        layout: this.tracerCompute.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: tracer.params } },
          { binding: 1, resource: { buffer: tracer.display } },
          { binding: 2, resource: { buffer: tracer.particles } },
          { binding: 3, resource: { buffer: tracer.velocity } },
        ],
      });
    }
  }
}
