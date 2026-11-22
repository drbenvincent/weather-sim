export const BINS = 16;

export interface Domain {
  nx: number;
  nz: number;
  extentX: number;
  extentZ: number;
}

export interface ParamDef {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  unit: string;
  note: string;
  restarts: boolean;
  options?: number[];
}

export type ParamValues = Record<string, number>;

export interface InitialFields {
  theta: Float32Array;
  qv: Float32Array;
  qc: Float32Array;
  qr: Float32Array;
  u: Float32Array;
  v: Float32Array;
  thetaRef: Float32Array;
  qvRef: Float32Array;
  pressure: Float32Array;
  exner: Float32Array;
}

export interface Diagnostics {
  ready: boolean;
  time: number;
  maxUpdraft: number;
  maxQc: number;
  maxQr: number;
  cloudTop: number;
  cloudX: number;
  rainRate: number;
  frontX: number;
  frontGrad: number;
  divAfter: number;
  divFirst: number;
  maxSpeed: number;
  sfcTheta: Float32Array;
  midTheta: Float32Array;
}

export interface LessonStage {
  id: string;
  title: string;
  text: string;
  reached: boolean;
}

export interface Lesson {
  stage: number;
  stages: LessonStage[];
}

export interface Scene {
  id: string;
  title: string;
  kicker: string;
  summary: string;
  domain: Domain;
  params: ParamDef[];
  modelNote: string;
  modelNoteShort: string;
  createInitial(values: ParamValues): InitialFields;
  interpret(diag: Diagnostics, values: ParamValues): Lesson;
}

export interface FieldView {
  scalar: GPUBuffer;
  velocity: GPUBuffer;
  divergence: GPUBuffer;
  refs: GPUBuffer;
  particles: GPUBuffer;
  nx: number;
  nz: number;
  extentX: number;
  extentZ: number;
  dx: number;
  dz: number;
}

export function emptyDiagnostics(): Diagnostics {
  return {
    ready: false,
    time: 0,
    maxUpdraft: 0,
    maxQc: 0,
    maxQr: 0,
    cloudTop: 0,
    cloudX: 0,
    rainRate: 0,
    frontX: 0,
    frontGrad: 0,
    divAfter: 0,
    divFirst: 0,
    maxSpeed: 0,
    sfcTheta: new Float32Array(BINS),
    midTheta: new Float32Array(BINS),
  };
}

export function paramValues(params: ParamDef[]): ParamValues {
  const values: ParamValues = {};
  for (const param of params) values[param.id] = param.value;
  return values;
}
