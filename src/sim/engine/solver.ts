import { cosineBasis, cosineScale, neumannEigenvalues } from "./poisson.ts";
import { emptyDiagnostics, type Diagnostics, type Domain, type FieldView, type InitialFields } from "../types.ts";
import prelude from "./wgsl/prelude.wgsl?raw";
import advectWgsl from "./wgsl/advect.wgsl?raw";
import forcesWgsl from "./wgsl/forces.wgsl?raw";
import microWgsl from "./wgsl/micro.wgsl?raw";
import projectWgsl from "./wgsl/project.wgsl?raw";
import reduceWgsl from "./wgsl/reduce.wgsl?raw";
import { shaderError } from "./gpu.ts";

const DT = 4;
const NU = 40;
const SPONGE = 0.04;
const SPONGE_TOP = 7500;
const SPONGE_DEPTH = 2500;
// 0.5 s^-1 removes a falling drop in tens of seconds. 0.005 lets rain cross the dry cold pool.
const K_EVAP = 0.005;
export const PARTICLE_COUNT = 3000;
const PARTICLE_FLOATS = 12;

type Slot =
  | "uniform"
  | "scalSrc"
  | "scalDst"
  | "velSrc"
  | "velDst"
  | "refs"
  | "basisX"
  | "basisZ"
  | "tables"
  | "div"
  | "spec"
  | "pressure"
  | "columns"
  | "stats"
  | "peak";

interface Pipe {
  pipeline: GPUComputePipeline;
  bindings: number[];
  slots: Slot[];
}

function storage(device: GPUDevice, size: number, label: string, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ size, usage, label });
}

export class Solver {
  readonly domain: Domain;
  simTime = 0;
  stepSeconds(): number {
    return this.dt;
  }
  divFirst = 0;
  private scalarIndex = 0;
  private velocityIndex = 0;
  private dt = DT;
  private cache = emptyDiagnostics();
  private stagingSlot = 0;
  private readonly stagingMapped = [false, false];
  private stepsSinceRead = 0;
  private generation = 0;
  private readonly groups = new Map<string, GPUBindGroup>();
  private readonly firstPeak: GPUBuffer;

  private constructor(
    private readonly device: GPUDevice,
    domain: Domain,
    private readonly uniform: GPUBuffer,
    private readonly scalar: [GPUBuffer, GPUBuffer],
    private readonly velocity: [GPUBuffer, GPUBuffer],
    private readonly divergence: GPUBuffer,
    private readonly spectral: GPUBuffer,
    private readonly pressure: GPUBuffer,
    private readonly refs: GPUBuffer,
    private readonly basisX: GPUBuffer,
    private readonly basisZ: GPUBuffer,
    private readonly tables: GPUBuffer,
    private readonly columns: GPUBuffer,
    private readonly stats: GPUBuffer,
    private readonly staging: [GPUBuffer, GPUBuffer],
    private readonly particles: GPUBuffer,
    private readonly display: GPUBuffer,
    private readonly pipes: Record<string, Pipe>,
  ) {
    this.domain = domain;
    this.firstPeak = storage(
      device, 4, "first-peak",
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    );
  }

  static async create(device: GPUDevice, domain: Domain, initial: InitialFields): Promise<Solver> {
    const { nx, nz } = domain;
    const cells = nx * nz;
    const faces = (nx + 1) * nz + nx * (nz + 1);
    const cellBytes = cells * 4;
    const scalarBytes = cellBytes * 4;
    const velocityBytes = faces * 4;
    const gpu = GPUBufferUsage;
    const uniform = storage(device, 80, "uniform", gpu.UNIFORM | gpu.COPY_DST);
    const scalar: [GPUBuffer, GPUBuffer] = [
      storage(device, scalarBytes, "scalar-0", gpu.STORAGE | gpu.COPY_DST | gpu.COPY_SRC),
      storage(device, scalarBytes, "scalar-1", gpu.STORAGE | gpu.COPY_DST),
    ];
    const velocity: [GPUBuffer, GPUBuffer] = [
      storage(device, velocityBytes, "velocity-0", gpu.STORAGE | gpu.COPY_DST),
      storage(device, velocityBytes, "velocity-1", gpu.STORAGE | gpu.COPY_DST),
    ];
    const divergence = storage(device, cellBytes, "divergence", gpu.STORAGE | gpu.COPY_DST);
    const spectral = storage(device, cellBytes, "spectral", gpu.STORAGE);
    const pressure = storage(device, cellBytes, "pressure", gpu.STORAGE);
    const refs = storage(device, nz * 16, "refs", gpu.STORAGE | gpu.COPY_DST);
    const basisX = storage(device, nx * nx * 4, "basis-x", gpu.STORAGE | gpu.COPY_DST);
    const basisZ = storage(device, nz * nz * 4, "basis-z", gpu.STORAGE | gpu.COPY_DST);
    const tables = storage(device, 2 * (nx + nz) * 4, "tables", gpu.STORAGE | gpu.COPY_DST);
    const columns = storage(device, nx * 32, "columns", gpu.STORAGE);
    const stats = storage(device, 256, "stats", gpu.STORAGE | gpu.COPY_SRC | gpu.COPY_DST);
    const staging: [GPUBuffer, GPUBuffer] = [
      storage(device, 256, "staging-0", gpu.MAP_READ | gpu.COPY_DST),
      storage(device, 256, "staging-1", gpu.MAP_READ | gpu.COPY_DST),
    ];
    const particles = storage(device, PARTICLE_COUNT * PARTICLE_FLOATS * 4, "particles", gpu.STORAGE | gpu.COPY_DST);
    const display = storage(device, 16, "display", gpu.UNIFORM | gpu.COPY_DST);
    const sources: Record<string, string> = {
      advect: prelude + advectWgsl,
      forces: prelude + forcesWgsl,
      micro: prelude + microWgsl,
      project: prelude + projectWgsl,
      reduce: prelude + reduceWgsl,
    };
    const modules = Object.fromEntries(
      Object.entries(sources).map(([name, code]) => [name, device.createShaderModule({ code, label: name })]),
    );
    await Promise.all(Object.entries(modules).map(([name, module]) => shaderError(device, module, name)));
    const make = (module: string, entry: string, bindings: number[], slots: Slot[]): Pipe => ({
      pipeline: device.createComputePipeline({
        label: entry,
        layout: "auto",
        compute: { module: modules[module], entryPoint: entry },
      }),
      bindings,
      slots,
    });
    const pipes: Record<string, Pipe> = {
      advectScalar: make("advect", "advect_scalar", [0, 1, 2, 3], ["uniform", "scalSrc", "scalDst", "velSrc"]),
      advectU: make("advect", "advect_u", [0, 3, 4], ["uniform", "velSrc", "velDst"]),
      advectV: make("advect", "advect_v", [0, 3, 4], ["uniform", "velSrc", "velDst"]),
      forceU: make("forces", "force_u", [0, 2, 3], ["uniform", "velSrc", "velDst"]),
      forceV: make("forces", "force_v", [0, 1, 2, 3, 4], ["uniform", "scalSrc", "velSrc", "velDst", "refs"]),
      micro: make("micro", "micro", [0, 1, 2, 3], ["uniform", "scalSrc", "scalDst", "refs"]),
      divergence: make("project", "divergence", [0, 1, 4], ["uniform", "velSrc", "div"]),
      prepare: make("project", "prepare_rhs", [0, 3, 4], ["uniform", "div", "spec"]),
      dctRow: make("project", "dct_row", [0, 3, 4, 5, 7], ["uniform", "spec", "pressure", "basisX", "tables"]),
      dctColumn: make("project", "dct_column", [0, 3, 4, 6, 7], ["uniform", "pressure", "spec", "basisZ", "tables"]),
      lambda: make("project", "apply_lambda", [0, 3, 4, 7], ["uniform", "spec", "pressure", "tables"]),
      idctColumn: make("project", "idct_column", [0, 3, 4, 6, 7], ["uniform", "pressure", "spec", "basisZ", "tables"]),
      idctRow: make("project", "idct_row", [0, 3, 4, 5, 7], ["uniform", "spec", "pressure", "basisX", "tables"]),
      gradientU: make("project", "gradient_u", [0, 1, 2, 3], ["uniform", "velSrc", "velDst", "pressure"]),
      gradientV: make("project", "gradient_v", [0, 1, 2, 3], ["uniform", "velSrc", "velDst", "pressure"]),
      divBefore: make("reduce", "div_before", [0, 3, 6], ["uniform", "div", "peak"]),
      divAfter: make("reduce", "div_after", [0, 3, 5], ["uniform", "div", "stats"]),
      columns: make("reduce", "columns", [0, 1, 2, 4], ["uniform", "velSrc", "scalSrc", "columns"]),
      finalize: make("reduce", "finalize", [0, 4, 5], ["uniform", "columns", "stats"]),
    };
    const solver = new Solver(
      device, domain, uniform, scalar, velocity, divergence, spectral, pressure,
      refs, basisX, basisZ, tables, columns, stats, staging, particles, display, pipes,
    );
    solver.writeTables();
    solver.writeUniform(DT);
    solver.writeInitial(initial);
    return solver;
  }

  private get dx(): number {
    return this.domain.extentX / this.domain.nx;
  }

  private get dz(): number {
    return this.domain.extentZ / this.domain.nz;
  }

  private bufferFor(slot: Slot): GPUBuffer {
    if (slot === "uniform") return this.uniform;
    if (slot === "scalSrc") return this.scalar[this.scalarIndex];
    if (slot === "scalDst") return this.scalar[1 - this.scalarIndex];
    if (slot === "velSrc") return this.velocity[this.velocityIndex];
    if (slot === "velDst") return this.velocity[1 - this.velocityIndex];
    if (slot === "refs") return this.refs;
    if (slot === "basisX") return this.basisX;
    if (slot === "basisZ") return this.basisZ;
    if (slot === "tables") return this.tables;
    if (slot === "div") return this.divergence;
    if (slot === "spec") return this.spectral;
    if (slot === "pressure") return this.pressure;
    if (slot === "columns") return this.columns;
    if (slot === "stats") return this.stats;
    return this.firstPeak;
  }

  private group(name: string): GPUBindGroup {
    const key = `${name}:${this.scalarIndex}:${this.velocityIndex}`;
    const cached = this.groups.get(key);
    if (cached) return cached;
    const pipe = this.pipes[name];
    const entries = pipe.bindings.map((binding, index) => ({
      binding,
      resource: { buffer: this.bufferFor(pipe.slots[index]) },
    }));
    const group = this.device.createBindGroup({
      layout: pipe.pipeline.getBindGroupLayout(0),
      entries,
      label: name,
    });
    this.groups.set(key, group);
    return group;
  }

  private run(pass: GPUComputePassEncoder, name: string, x: number, y = 1): void {
    pass.setPipeline(this.pipes[name].pipeline);
    pass.setBindGroup(0, this.group(name));
    pass.dispatchWorkgroups(x, y);
  }

  /** One pass per dependent stage. Writes are not visible to later dispatches in the same pass. */
  private pass(encoder: GPUCommandEncoder, label: string, work: Array<[string, number, number]>): void {
    const compute = encoder.beginComputePass({ label });
    for (const [name, x, y] of work) this.run(compute, name, x, y);
    compute.end();
  }

  step(count: number): void {
    if (count <= 0) return;
    const { nx, nz } = this.domain;
    const ux = Math.ceil((nx + 1) / 64);
    const vx = Math.ceil(nx / 64);
    const row = Math.ceil(nz / 64);
    const encoder = this.device.createCommandEncoder({ label: "step" });
    for (let n = 0; n < count; n += 1) {
      this.pass(encoder, "advect", [
        ["advectScalar", vx, nz],
        ["advectU", ux, nz],
        ["advectV", vx, nz + 1],
      ]);
      this.scalarIndex = 1 - this.scalarIndex;
      this.velocityIndex = 1 - this.velocityIndex;
      this.pass(encoder, "forces", [
        ["forceU", ux, nz],
        ["forceV", vx, nz + 1],
      ]);
      this.velocityIndex = 1 - this.velocityIndex;
      this.pass(encoder, "micro", [["micro", vx, nz]]);
      this.scalarIndex = 1 - this.scalarIndex;
      encoder.clearBuffer(this.stats);
      this.pass(encoder, "div-before-field", [["divergence", vx, nz]]);
      this.pass(encoder, "div-before", [["divBefore", 1, 1]]);
      this.pass(encoder, "prepare", [["prepare", 1, 1]]);
      this.pass(encoder, "dct-row", [["dctRow", row, 1]]);
      this.pass(encoder, "dct-column", [["dctColumn", vx, 1]]);
      this.pass(encoder, "lambda", [["lambda", vx, nz]]);
      this.pass(encoder, "idct-column", [["idctColumn", vx, 1]]);
      this.pass(encoder, "idct-row", [["idctRow", row, 1]]);
      this.pass(encoder, "gradient", [
        ["gradientU", ux, nz],
        ["gradientV", vx, nz + 1],
      ]);
      this.velocityIndex = 1 - this.velocityIndex;
      this.pass(encoder, "residual", [
        ["divergence", vx, nz],
        ["columns", vx, 1],
      ]);
      this.pass(encoder, "div-after", [["divAfter", 1, 1]]);
      this.pass(encoder, "finalize", [["finalize", 1, 1]]);
      this.simTime += this.dt;
    }
    this.stepsSinceRead += count;
    const readSlot = this.stagingSlot;
    const issued = !this.stagingMapped[readSlot] && (this.stepsSinceRead >= 6 || !this.cache.ready);
    if (issued) {
      encoder.copyBufferToBuffer(this.stats, 0, this.staging[readSlot], 0, 256);
      encoder.copyBufferToBuffer(this.firstPeak, 0, this.staging[readSlot], 44, 4);
      this.stagingMapped[readSlot] = true;
      this.stagingSlot = 1 - readSlot;
      this.stepsSinceRead = 0;
    }
    this.device.queue.submit([encoder.finish()]);
    if (issued) void this.readback(readSlot, this.generation);
    this.guardCfl();
  }

  private guardCfl(): void {
    const speed = this.cache.maxSpeed;
    if (!Number.isFinite(speed) || speed <= 0 || this.dt <= 0.5) return;
    const spacing = Math.min(this.dx, this.dz);
    if (speed * this.dt / spacing > 0.8) {
      this.dt = Math.max(0.5, this.dt * 0.5);
      this.writeUniform(this.dt);
    }
  }


  private async readback(slot: number, generation: number): Promise<void> {
    const buffer = this.staging[slot];
    try {
      await buffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(buffer.getMappedRange()).slice();
      buffer.unmap();
      if (generation !== this.generation) return;
      this.divFirst = data[11];
      this.cache = {
        ready: true,
        time: this.simTime,
        maxUpdraft: data[0],
        maxQc: data[1],
        maxQr: data[2],
        cloudTop: data[3],
        cloudX: data[4],
        rainRate: data[5],
        frontX: data[6],
        frontGrad: data[7],
        divAfter: data[9],
        divFirst: data[11],
        maxSpeed: data[10],
        sfcTheta: data.slice(12, 28),
        midTheta: data.slice(28, 44),
      };
    } catch {
      // The page left, or the device was lost, while this buffer was mapped.
    } finally {
      this.stagingMapped[slot] = false;
    }
  }

  diagnostics(): Diagnostics {
    return { ...this.cache, time: this.simTime, divFirst: this.divFirst };
  }

  view(): FieldView {
    return {
      scalar: this.scalar[this.scalarIndex],
      velocity: this.velocity[this.velocityIndex],
      divergence: this.divergence,
      refs: this.refs,
      particles: this.particles,
      nx: this.domain.nx,
      nz: this.domain.nz,
      extentX: this.domain.extentX,
      extentZ: this.domain.extentZ,
      dx: this.dx,
      dz: this.dz,
    };
  }

  tracerState(): {
    params: GPUBuffer;
    display: GPUBuffer;
    particles: GPUBuffer;
    velocity: GPUBuffer;
    scalar: GPUBuffer;
    refs: GPUBuffer;
  } {
    return {
      params: this.uniform,
      display: this.display,
      particles: this.particles,
      velocity: this.velocity[this.velocityIndex],
      scalar: this.scalar[this.scalarIndex],
      refs: this.refs,
    };
  }

  writeTracerDelta(dt: number): void {
    const data = new ArrayBuffer(16);
    const view = new DataView(data);
    view.setFloat32(0, dt, true);
    view.setUint32(4, PARTICLE_COUNT, true);
    this.device.queue.writeBuffer(this.display, 0, data);
  }

  reset(initial: InitialFields): void {
    this.generation += 1;
    this.scalarIndex = 0;
    this.velocityIndex = 0;
    this.simTime = 0;
    this.divFirst = 0;
    this.dt = DT;
    this.cache = emptyDiagnostics();
    this.stepsSinceRead = 0;
    this.groups.clear();
    this.writeUniform(DT);
    this.writeInitial(initial);
    this.device.queue.writeBuffer(this.stats, 0, new Float32Array(64));
    this.device.queue.writeBuffer(this.firstPeak, 0, new Float32Array(1));
    this.device.queue.writeBuffer(this.divergence, 0, new Float32Array(this.domain.nx * this.domain.nz));
  }

  destroy(): void {
    this.generation += 1;
    for (const buffer of [
      this.uniform, ...this.scalar, ...this.velocity, this.divergence, this.spectral, this.pressure,
      this.refs, this.basisX, this.basisZ, this.tables, this.columns, this.stats, ...this.staging,
      this.particles, this.display, this.firstPeak,
    ]) {
      buffer.destroy();
    }
  }

  private writeUniform(dt: number): void {
    const { nx, nz, extentX, extentZ } = this.domain;
    const dx = this.dx;
    const dz = this.dz;
    const data = new ArrayBuffer(80);
    const view = new DataView(data);
    view.setUint32(0, nx, true);
    view.setUint32(4, nz, true);
    const floats = [
      dx, dz, dt, NU, 9.81, 2500, SPONGE, 2.2, K_EVAP, 0.0005, 5, 0.002, 1.2, 1000,
      extentX, extentZ, SPONGE_TOP, SPONGE_DEPTH,
    ];
    floats.forEach((value, index) => view.setFloat32(8 + index * 4, value, true));
    this.device.queue.writeBuffer(this.uniform, 0, data);
  }

  private writeTables(): void {
    const { nx, nz } = this.domain;
    const dx = this.dx;
    const dz = this.dz;
    const tables = new Float32Array(2 * (nx + nz));
    tables.set(cosineScale(nx), 0);
    tables.set(cosineScale(nz), nx);
    tables.set(neumannEigenvalues(nx, dx), nx + nz);
    tables.set(neumannEigenvalues(nz, dz), 2 * nx + nz);
    this.device.queue.writeBuffer(this.tables, 0, tables);
    this.device.queue.writeBuffer(this.basisX, 0, cosineBasis(nx));
    this.device.queue.writeBuffer(this.basisZ, 0, cosineBasis(nz));
  }

  private writeInitial(initial: InitialFields): void {
    const { nx, nz } = this.domain;
    const cells = nx * nz;
    const scalars = new Float32Array(cells * 4);
    scalars.set(initial.theta, 0);
    scalars.set(initial.qv, cells);
    scalars.set(initial.qc, cells * 2);
    scalars.set(initial.qr, cells * 3);
    this.device.queue.writeBuffer(this.scalar[0], 0, scalars);
    const faces = (nx + 1) * nz + nx * (nz + 1);
    const velocity = new Float32Array(faces);
    velocity.set(initial.u, 0);
    velocity.set(initial.v, (nx + 1) * nz);
    this.device.queue.writeBuffer(this.velocity[0], 0, velocity);
    const refs = new Float32Array(nz * 4);
    refs.set(initial.thetaRef, 0);
    refs.set(initial.qvRef, nz);
    refs.set(initial.pressure, nz * 2);
    refs.set(initial.exner, nz * 3);
    this.device.queue.writeBuffer(this.refs, 0, refs);
    this.writeParticles();
  }

  private writeParticles(): void {
    const { extentX, extentZ } = this.domain;
    const data = new Float32Array(PARTICLE_COUNT * PARTICLE_FLOATS);
    const hash = (n: number) => {
      const x = Math.sin(n * 127.1) * 43758.5453;
      return x - Math.floor(x);
    };
    for (let i = 0; i < PARTICLE_COUNT; i += 1) {
      const x = hash(i + 1.3) * extentX;
      const z = hash(i * 1.7 + 9.2) * extentZ;
      const base = i * PARTICLE_FLOATS;
      for (let t = 0; t < 5; t += 1) {
        data[base + t * 2] = x;
        data[base + t * 2 + 1] = z;
      }
      data[base + 10] = hash(i + 4.4) * 200.0;
      data[base + 11] = hash(i + 8.1);
    }
    this.device.queue.writeBuffer(this.particles, 0, data);
  }
}
