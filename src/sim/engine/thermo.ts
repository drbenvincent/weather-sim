export const RD = 287;
export const CP = 1004;
export const P0 = 1e5;
export const THETA0 = 290;
export const LAPSE = 0.004;
export const LATENT = 2.5e6;
export const LCP = LATENT / CP;
export const EPS = 0.622;
export const G = 9.81;

export interface Column {
  theta: number;
  pressure: number;
  exner: number;
  temperature: number;
}

/** Hydrostatic reference. dΠ/dz = −g / (cp θ), θ = θ0 + Γz. */
export function referenceColumn(z: number): Column {
  const theta = THETA0 + LAPSE * z;
  const exner = 1 - (G / (CP * LAPSE)) * Math.log(theta / THETA0);
  return {
    theta,
    exner,
    pressure: P0 * Math.pow(Math.max(exner, 1e-4), CP / RD),
    temperature: theta * exner,
  };
}

export function saturationVaporPressure(celsius: number): number {
  return 611.2 * Math.exp((17.67 * celsius) / (celsius + 243.5));
}

export function saturationMixingRatio(celsius: number, pressure: number): number {
  const es = saturationVaporPressure(celsius);
  return (EPS * es) / Math.max(pressure - es, 1);
}

export function saturationSlope(celsius: number, pressure: number): number {
  const es = saturationVaporPressure(celsius);
  const des = (es * 17.67 * 243.5) / (celsius + 243.5) ** 2;
  return (EPS * pressure * des) / Math.max(pressure - es, 1) ** 2;
}

export function buoyancy(
  theta: number,
  qv: number,
  qc: number,
  qr: number,
  thetaRef: number,
  qvRef: number,
): number {
  return G * ((theta - thetaRef) / thetaRef + 0.608 * (qv - qvRef) - qc - qr);
}

export interface Parcel {
  theta: number;
  qv: number;
  qc: number;
  qr: number;
}

/**
 * Closed-form saturation adjustment.
 * dq = excess / (1 + (L/cp) dqs/dT), dT = (L/cp) dq.
 * Condensation cannot exceed available vapor; evaporation cannot exceed cloud water.
 */
export function adjustSaturation(parcel: Parcel, exner: number, pressure: number): Parcel {
  let { theta, qv, qc, qr } = parcel;
  for (let i = 0; i < 5; i++) {
    const celsius = theta * exner - 273.15;
    const qs = saturationMixingRatio(celsius, pressure);
    const slope = saturationSlope(celsius, pressure);
    const excess = qv - qs;
    const dq = Math.min(qv, Math.max(-qc, excess / (1 + LCP * slope)));
    theta += (LCP * dq) / exner;
    qv -= dq;
    qc += dq;
  }
  return { theta, qv: Math.max(qv, 0), qc: Math.max(qc, 0), qr };
}
