import { buoyancy, referenceColumn, saturationMixingRatio } from "../engine/thermo.ts";
import {
  BINS,
  emptyDiagnostics,
  type Diagnostics,
  type InitialFields,
  type Lesson,
  type LessonStage,
  type ParamDef,
  type ParamValues,
  type Scene,
} from "../types.ts";

export const DOMAIN = { nx: 320, nz: 128, extentX: 160_000, extentZ: 10_000 };
export const COLD_EDGE = 45_000;
export const WEDGE_EDGE = 85_000;
const WARM_TOP = 4_500;
const COLD_TOP = 2_500;
const RH_CAP = 0.97;
const RH_ALOFT = 0.3;

const MODEL_NOTE =
  "Two-dimensional Boussinesq solver on a grid of a few hundred metres. Water vapor, cloud, and rain. Condensation is a saturation adjustment (Clausius–Clapeyron) and releases latent heat. Rain forms by Kessler autoconversion and falls at a terminal velocity. Small eddies are only an eddy viscosity. No Coriolis force, no radiation, no ice. The warm-front slope begins as a wedge of cooler air; the clouds and the cold surge are computed.";

export const params: ParamDef[] = [
  {
    id: "contrast",
    label: "Temperature contrast",
    min: 0.5,
    max: 1.5,
    step: 0.05,
    value: 1,
    unit: "×",
    note: "Scales the cold-pool and warm-air anomalies. Restarts the run.",
    restarts: true,
  },
  {
    id: "humidity",
    label: "Humidity",
    min: 0.15,
    max: 0.95,
    step: 0.01,
    value: 0.82,
    unit: "",
    note: "Warm-sector relative humidity. Near 20% the lift may stay clear; near 90% cloud and rain are abundant. Restarts the run.",
    restarts: true,
  },
  {
    id: "speed",
    label: "Speed",
    min: 1,
    max: 8,
    step: 1,
    value: 2,
    unit: "×",
    note: "Substeps per physics tick. 2× is the calibrated rate. Does not restart.",
    restarts: false,
    options: [1, 2, 4, 8],
  },
];

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function below(z: number, depth: number, blend: number): number {
  if (depth <= 0) return 0;
  return 1 - smoothstep(Math.max(0, depth - blend), depth + blend, z);
}

function weights(x: number, z: number): { cold: number; warm: number; wedge: number } {
  const cold = (1 - smoothstep(COLD_EDGE - 4_000, COLD_EDGE + 4_000, x)) * below(z, COLD_TOP, 400);
  const depth = Math.max(0, (x - WEDGE_EDGE) / (DOMAIN.extentX - WEDGE_EDGE)) * 2_000;
  const wedge = smoothstep(WEDGE_EDGE - 4_000, WEDGE_EDGE + 4_000, x) * below(z, depth, 350);
  const warm = below(z, WARM_TOP, 600) * (1 - cold) * (1 - wedge);
  return { cold, warm, wedge };
}

function humidityScale(humidity: number): { cold: number; warm: number; wedge: number; aloft: number } {
  const scale = humidity / 0.82;
  return {
    cold: Math.min(RH_CAP, 0.35 * scale),
    warm: Math.min(RH_CAP, humidity),
    wedge: Math.min(RH_CAP, 0.7 * scale),
    aloft: RH_ALOFT,
  };
}

export function createInitial(values: ParamValues): InitialFields {
  const { nx, nz, extentX, extentZ } = DOMAIN;
  const dx = extentX / nx;
  const dz = extentZ / nz;
  const contrast = values.contrast ?? 1;
  const rh = humidityScale(values.humidity ?? 0.82);
  const theta = new Float32Array(nx * nz);
  const qv = new Float32Array(nx * nz);
  const thetaRef = new Float32Array(nz);
  const qvRef = new Float32Array(nz);
  const pressure = new Float32Array(nz);
  const exner = new Float32Array(nz);
  for (let k = 0; k < nz; k++) {
    const column = referenceColumn((k + 0.5) * dz);
    thetaRef[k] = column.theta;
    pressure[k] = column.pressure;
    exner[k] = column.exner;
    qvRef[k] = RH_ALOFT * saturationMixingRatio(column.temperature - 273.15, column.pressure);
  }
  for (let k = 0; k < nz; k++) {
    const z = (k + 0.5) * dz;
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * dx;
      const w = weights(x, z);
      const anomaly = contrast * (-10 * w.cold + 4 * w.warm - 4 * w.wedge);
      const cover = w.cold + w.warm + w.wedge;
      const relative = rh.cold * w.cold + rh.warm * w.warm + rh.wedge * w.wedge + rh.aloft * (1 - cover);
      const index = k * nx + i;
      theta[index] = thetaRef[k] + anomaly;
      const temperature = theta[index] * exner[k] - 273.15;
      qv[index] = Math.min(relative, RH_CAP) * saturationMixingRatio(temperature, pressure[k]);
    }
  }
  return {
    theta,
    qv,
    qc: new Float32Array(nx * nz),
    qr: new Float32Array(nx * nz),
    u: new Float32Array((nx + 1) * nz),
    v: new Float32Array(nx * (nz + 1)),
    thetaRef,
    qvRef,
    pressure,
    exner,
  };
}

function meanBins(values: Float32Array, x0: number, x1: number): number {
  const width = DOMAIN.extentX / BINS;
  let sum = 0;
  let n = 0;
  for (let b = 0; b < values.length; b++) {
    const left = b * width;
    if (left < x1 && left + width > x0) {
      sum += values[b];
      n++;
    }
  }
  return n ? sum / n : 0;
}
function layerReference(): { sfc: number; mid: number } {
  const dz = DOMAIN.extentZ / DOMAIN.nz;
  let sfc = 0;
  let sfcCount = 0;
  let mid = 0;
  let midCount = 0;
  for (let k = 0; k < DOMAIN.nz; k++) {
    const z = (k + 0.5) * dz;
    const theta = referenceColumn(z).theta;
    if (z < 4 * dz) {
      sfc += theta;
      sfcCount += 1;
    }
    if (z >= 800 && z <= 2000) {
      mid += theta;
      midCount += 1;
    }
  }
  return { sfc: sfc / sfcCount, mid: mid / midCount };
}

const LAYER_REF = layerReference();

function warmAloft(diag: Diagnostics): boolean {
  if (!diag.ready) return false;
  const sfc = meanBins(diag.sfcTheta, 50_000, WEDGE_EDGE) - LAYER_REF.sfc;
  const mid = meanBins(diag.midTheta, 50_000, WEDGE_EDGE) - LAYER_REF.mid;
  return mid > sfc + 2;
}

function stageText(id: string, diag: Diagnostics, values: ParamValues): string {
  const contrast = values.contrast ?? 1;
  const cold0 = 290 - 10 * contrast - 273.15;
  const warm0 = 290 + 4 * contrast - 273.15;
  const cold = diag.ready ? meanBins(diag.sfcTheta, 0, 20_000) - 273.15 : cold0;
  const warm = diag.ready ? meanBins(diag.sfcTheta, 55_000, 80_000) - 273.15 : warm0;
  if (id === "masses") {
    return `A cold pool near ${cold.toFixed(0)} °C sits against warmer air near ${warm.toFixed(0)} °C, a contrast of about ${(warm - cold).toFixed(0)} °C. The cold air is denser. The run starts with no wind, so only that density difference can drive the surge.`;
  }
  if (id === "undercut") {
    const moved = (diag.frontX - COLD_EDGE) / 1000;
    return `The cold pool has spread ${moved.toFixed(1)} km from its starting edge near 45 km. Denser air runs along the ground and the warmer air stays aloft above it. The leading edge, where the surface temperature rises sharply, is the cold front.`;
  }
  if (id === "lift") {
    return `Air at the nose is rising at ${diag.maxUpdraft.toFixed(1)} m/s. It expands as it rises and keeps its potential temperature, so its temperature falls because the pressure, through the Exner function, drops with height. Cloud forms only if that rising air reaches saturation; the humidity control sets how moist it started. The warm-front wedge on the right is steeper than a synoptic front, so the meeting can fit this domain.`;
  }
  if (id === "cloud") {
    return `The rising air has reached its dew point. Cloud water is present, with a top near ${(diag.cloudTop / 1000).toFixed(1)} km and a peak of ${(diag.maxQc * 1000).toFixed(2)} g/kg. Latent heat released as that vapor condenses warms the parcel by about 2.5 °C for each gram per kilogram.`;
  }
  if (id === "rain") {
    const aloft = diag.maxQr * 1000;
    const rate = diag.rainRate * 3.6e6;
    const ground =
      diag.rainRate > 1e-6
        ? `The surface rate is about ${rate.toFixed(1)} mm/h.`
        : "It has not yet reached the ground.";
    return `Cloud water collected and fell out. Rain is falling through the slice, up to ${aloft.toFixed(2)} g/kg. ${ground} The white on the canvas is cloud water, not a painted cloud, and the streaks are rain.`;
  }
  const aloft = warmAloft(diag);
  const km = (diag.frontX / 1000).toFixed(0);
  if (aloft && diag.frontX >= WEDGE_EDGE) {
    return `The cold surge has reached the warm-front wedge near ${km} km. Between 800 m and 2 km the air in the former warm sector is warmer than the surface, so that warm air has been lifted off the ground. The meeting is an occlusion: the cold pool undercuts, and the wedge holds the warm air up.`;
  }
  if (aloft) {
    return `The cold surge is near ${km} km, still short of the warm-front wedge. In the former warm sector the air between 800 m and 2 km is warmer than the surface, so that warm air has been lifted off the ground. The meeting is an occlusion read from that stack.`;
  }
  return `The surface temperature gradient has entered the warm-front wedge near ${km} km, so this meeting is the occlusion of the two fronts. The cold pool is undercutting that wedge. The air above the surface there is not yet warmer than the ground, so the warm-over-cold stack is not in the fields yet.`;
}

const STAGE_DEFS = [
  { id: "masses", title: "Two air masses" },
  { id: "undercut", title: "Undercutting" },
  { id: "lift", title: "Lift" },
  { id: "cloud", title: "Cloud" },
  { id: "rain", title: "Rain" },
  { id: "meeting", title: "Meeting" },
];

export function interpret(diag: Diagnostics, values: ParamValues): Lesson {
  const flags = [
    true,
    diag.ready && diag.frontX - COLD_EDGE > 3_000,
    diag.ready && diag.maxUpdraft > 1,
    diag.ready && diag.maxQc > 1e-5,
    diag.ready && (diag.rainRate > 1e-6 || diag.maxQr > 2e-4),
    diag.ready && (diag.frontX >= WEDGE_EDGE || warmAloft(diag)),
  ];
  let stage = 0;
  for (let i = 1; i < flags.length; i++) {
    if (!flags[i]) break;
    stage = i;
  }
  const stages: LessonStage[] = STAGE_DEFS.map((def, index) => ({
    id: def.id,
    title: def.title,
    text: stageText(def.id, diag.ready ? diag : emptyDiagnostics(), values),
    reached: index <= stage,
  }));
  return { stage, stages };
}

export function parcelBuoyancy(fields: InitialFields, i: number, k: number): number {
  const index = k * DOMAIN.nx + i;
  return buoyancy(fields.theta[index], fields.qv[index], 0, 0, fields.thetaRef[k], fields.qvRef[k]);
}

export const frontal: Scene = {
  id: "frontal",
  title: "Cold front meets a warm front",
  kicker: "A density current",
  summary: "Cold air undercuts warm, moist air. The lift, the cloud, and the rain are computed rather than drawn.",
  domain: DOMAIN,
  params,
  modelNote: MODEL_NOTE,
  modelNoteShort:
    "A two-dimensional slice. Vapor condenses when the air is cold enough; rain follows a simple warm-rain scheme. The warm-front slope is a wedge, steeper than a synoptic front.",
  createInitial,
  interpret,
};
