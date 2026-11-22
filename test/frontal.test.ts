import assert from "node:assert/strict";
import test from "node:test";
import { saturationMixingRatio } from "../src/sim/engine/thermo.ts";
import { createInitial, interpret, parcelBuoyancy, params, COLD_EDGE, DOMAIN, WEDGE_EDGE } from "../src/sim/scenes/frontal.ts";
import { emptyDiagnostics, paramValues, type Diagnostics } from "../src/sim/types.ts";

function ready(patch: Partial<Diagnostics>): Diagnostics {
  return { ...emptyDiagnostics(), ready: true, frontX: COLD_EDGE, ...patch };
}

test("the lock-exchange starts dry, colder on the left, and below saturation", () => {
  const fields = createInitial(paramValues(params));
  assert.equal(fields.qc.reduce((sum, value) => sum + value, 0), 0);
  assert.equal(fields.qr.reduce((sum, value) => sum + value, 0), 0);
  assert.ok(parcelBuoyancy(fields, 8, 4) < -0.05);
  assert.ok(parcelBuoyancy(fields, 140, 6) > 0.02);
  const { nx, nz } = DOMAIN;
  for (let k = 0; k < nz; k += 8) {
    for (let i = 0; i < nx; i += 16) {
      const index = k * nx + i;
      const temperature = fields.theta[index] * fields.exner[k] - 273.15;
      const qs = saturationMixingRatio(temperature, fields.pressure[k]);
      assert.ok(fields.qv[index] <= qs * 0.971, `supersaturated at ${i},${k}`);
    }
  }
});

test("meeting waits until the surge reaches the warm-front wedge", () => {
  const values = paramValues(params);
  const raining = {
    maxUpdraft: 2,
    maxQc: 1e-3,
    maxQr: 1e-3,
    rainRate: 1e-5,
    frontX: 60_000,
  };
  assert.equal(interpret(ready(raining), values).stage, 4);
  assert.equal(interpret(ready({ ...raining, frontX: WEDGE_EDGE }), values).stage, 5);
  assert.equal(interpret(ready({ frontX: WEDGE_EDGE, maxUpdraft: 2 }), values).stage, 2);
});

test("the opening stage does not claim the current air is at rest", () => {
  const lesson = interpret(ready({ frontX: COLD_EDGE + 500, maxSpeed: 8 }), paramValues(params));
  assert.equal(lesson.stage, 0);
  assert.equal(lesson.stages[0].text.includes("at rest"), false);
  assert.equal(lesson.stages[0].text.includes("no wind"), true);
});

test("lift explains cooling by potential temperature, not the reference lapse", () => {
  const lesson = interpret(ready({ frontX: COLD_EDGE + 4_000, maxUpdraft: 2 }), paramValues(params));
  assert.equal(lesson.stage, 2);
  const text = lesson.stages[2].text;
  assert.equal(text.includes("reference lapse"), false);
  assert.equal(text.includes("potential temperature"), true);
  assert.equal(text.includes("Exner"), true);
});

test("meeting names lift-off only when mid-level air is warmer than the surface", () => {
  const values = paramValues(params);
  const raining = { maxUpdraft: 2, maxQc: 1e-3, maxQr: 1e-3, rainRate: 1e-5, frontX: 60_000 };
  const stacked = ready({
    ...raining,
    sfcTheta: new Float32Array(16).fill(280),
    midTheta: new Float32Array(16).fill(294),
  });
  const lesson = interpret(stacked, values);
  assert.equal(lesson.stage, 5);
  assert.equal(lesson.stages[5].text.includes("lifted off the ground"), true);
  assert.equal(lesson.stages[5].text.includes("still short of the warm-front wedge"), true);
  const arrived = interpret(ready({ ...raining, frontX: WEDGE_EDGE }), values);
  assert.equal(arrived.stages[5].text.includes("lifted off the ground"), false);
});

function columnBins(theta: Float32Array): { sfcTheta: Float32Array; midTheta: Float32Array } {
  const { nx, nz, extentX, extentZ } = DOMAIN;
  const dx = extentX / nx;
  const dz = extentZ / nz;
  const width = extentX / 16;
  const sfcTheta = new Float32Array(16);
  const midTheta = new Float32Array(16);
  for (let bin = 0; bin < 16; bin++) {
    const x0 = bin * width;
    const x1 = x0 + width;
    let sfc = 0;
    let mid = 0;
    let count = 0;
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * dx;
      if (x < x0 || x >= x1) continue;
      let sfcSum = 0;
      let sfcCount = 0;
      let midSum = 0;
      let midCount = 0;
      for (let k = 0; k < nz; k++) {
        const z = (k + 0.5) * dz;
        const value = theta[k * nx + i];
        if (z < 4 * dz) {
          sfcSum += value;
          sfcCount += 1;
        }
        if (z >= 800 && z <= 2000) {
          midSum += value;
          midCount += 1;
        }
      }
      sfc += sfcSum / Math.max(sfcCount, 1);
      mid += midSum / Math.max(midCount, 1);
      count += 1;
    }
    sfcTheta[bin] = sfc / Math.max(count, 1);
    midTheta[bin] = mid / Math.max(count, 1);
  }
  return { sfcTheta, midTheta };
}

test("the resting lapse is not an occlusion", () => {
  const values = paramValues(params);
  const bins = columnBins(createInitial(values).theta);
  const lesson = interpret(
    ready({ maxUpdraft: 2, maxQc: 1e-3, maxQr: 1e-3, rainRate: 1e-5, frontX: 60_000, ...bins }),
    values,
  );
  assert.equal(lesson.stage, 4);
  assert.equal(lesson.stages[5].text.includes("lifted off the ground"), false);
});
