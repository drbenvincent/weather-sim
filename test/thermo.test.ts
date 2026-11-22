import assert from "node:assert/strict";
import test from "node:test";
import {
  CP,
  G,
  LCP,
  adjustSaturation,
  buoyancy,
  referenceColumn,
  saturationMixingRatio,
  saturationSlope,
  saturationVaporPressure,
} from "../src/sim/engine/thermo.ts";

test("Bolton saturation at 20 °C and 1000 hPa is about 14.9 g/kg", () => {
  const qs = saturationMixingRatio(20, 1e5);
  assert.ok(Math.abs(qs * 1000 - 14.9) < 0.2, `got ${qs * 1000}`);
  assert.ok(saturationVaporPressure(20) > 2000);
});

test("saturation slope matches a finite difference", () => {
  const slope = saturationSlope(10, 9e4);
  const h = 1e-3;
  const num =
    (saturationMixingRatio(10 + h, 9e4) - saturationMixingRatio(10 - h, 9e4)) / (2 * h);
  assert.ok(Math.abs(slope - num) / num < 1e-6);
});

test("hydrostatic reference recovers dΠ/dz = −g/(cp θ)", () => {
  const z = 2500;
  const h = 1;
  const mid = referenceColumn(z);
  const num = (referenceColumn(z + h).exner - referenceColumn(z - h).exner) / (2 * h);
  assert.ok(Math.abs(num + G / (CP * mid.theta)) < 1e-8);
  assert.ok(mid.exner < 1);
  assert.ok(referenceColumn(0).exner === 1);
  assert.ok(referenceColumn(10000).pressure < 40000);
});

test("saturation adjustment conserves water and warms on condensation", () => {
  const column = referenceColumn(1500);
  const qs = saturationMixingRatio(column.temperature - 273.15, column.pressure);
  const before = { theta: column.theta, qv: qs + 0.002, qc: 0, qr: 0.001 };
  const after = adjustSaturation(before, column.exner, column.pressure);
  assert.ok(Math.abs(after.qv + after.qc + after.qr - (before.qv + before.qc + before.qr)) < 1e-8);
  assert.ok(after.theta > before.theta);
  assert.ok(after.qc > 0);
  assert.ok(after.qc <= before.qv + before.qc + 1e-9);
  const celsius = after.theta * column.exner - 273.15;
  const residual = after.qv - saturationMixingRatio(celsius, column.pressure);
  assert.ok(Math.abs(residual) < 1e-6, `residual ${residual}`);
});

test("evaporation cools and cannot exceed cloud water", () => {
  const column = referenceColumn(1500);
  const qs = saturationMixingRatio(column.temperature - 273.15, column.pressure);
  const before = { theta: column.theta, qv: qs - 0.004, qc: 0.001, qr: 0 };
  const after = adjustSaturation(before, column.exner, column.pressure);
  assert.ok(after.qc === 0);
  assert.ok(after.theta < before.theta);
  assert.ok(after.qv + after.qc <= before.qv + before.qc + 1e-9);
  assert.ok(Math.abs(after.theta - before.theta + (LCP * (before.qc - after.qc)) / column.exner) < 1e-6);
});

test("a warm moist parcel is buoyant and a cold dry parcel sinks", () => {
  const ref = referenceColumn(500);
  const warm = buoyancy(ref.theta + 4, 0.012, 0, 0, ref.theta, 0.004);
  const cold = buoyancy(ref.theta - 10, 0.003, 0, 0, ref.theta, 0.004);
  assert.ok(warm > 0.1);
  assert.ok(cold < -0.2);
});
