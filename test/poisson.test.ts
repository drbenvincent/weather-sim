import assert from "node:assert/strict";
import test from "node:test";
import {
  cosineBasis,
  cosineScale,
  neumannEigenvalues,
  projectNeumann,
} from "../src/sim/engine/poisson.ts";

test("DCT-II round-trips a random field", () => {
  const n = 8;
  const basis = cosineBasis(n);
  const scale = cosineScale(n);
  const values = [0.2, -1.1, 0.4, 3, -0.7, 0.1, 2.2, -0.3];
  const hat = new Float32Array(n);
  for (let m = 0; m < n; m++) {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += values[i] * basis[m * n + i];
    hat[m] = scale[m] * sum;
  }
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let m = 0; m < n; m++) sum += scale[m] * hat[m] * basis[m * n + i];
    assert.ok(Math.abs(sum - values[i]) < 1e-5, `mode ${i}: ${sum}`);
  }
});

test("the zero eigenvalue is the only null mode", () => {
  const lambda = neumannEigenvalues(8, 500);
  assert.ok(lambda[0] === 0);
  for (let m = 1; m < lambda.length; m++) assert.ok(lambda[m] < 0);
});

test("an 8×6 projection drops divergence by 1e6", () => {
  const nx = 8;
  const nz = 6;
  const u = new Float32Array((nx + 1) * nz);
  const v = new Float32Array(nx * (nz + 1));
  let seed = 7;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647 - 0.5;
  };
  for (let k = 0; k < nz; k++) {
    for (let i = 1; i < nx; i++) u[k * (nx + 1) + i] = rand();
  }
  for (let k = 1; k < nz; k++) {
    for (let i = 0; i < nx; i++) v[k * nx + i] = rand();
  }
  const result = projectNeumann(u, v, nx, nz, 500, 80, 4);
  assert.ok(result.before > 1e-4, `before ${result.before}`);
  assert.ok(result.after < result.before / 1e6, `before ${result.before} after ${result.after}`);
  for (let k = 0; k < nz; k++) {
    assert.equal(result.u[k * (nx + 1)], 0);
    assert.equal(result.u[k * (nx + 1) + nx], 0);
  }
  for (let i = 0; i < nx; i++) {
    assert.equal(result.v[i], 0);
    assert.equal(result.v[nz * nx + i], 0);
  }
});
