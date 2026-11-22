/** DCT-II basis for a Neumann Laplacian on cell centers. basis[m * n + i]. */
export function cosineBasis(n: number): Float32Array {
  const basis = new Float32Array(n * n);
  for (let m = 0; m < n; m++) {
    for (let i = 0; i < n; i++) {
      basis[m * n + i] = Math.cos((Math.PI * m * (i + 0.5)) / n);
    }
  }
  return basis;
}

export function cosineScale(n: number): Float32Array {
  const scale = new Float32Array(n);
  for (let m = 0; m < n; m++) scale[m] = Math.sqrt((m === 0 ? 1 : 2) / n);
  return scale;
}

/** Eigenvalues of the second difference with a zero normal derivative. Negative semi-definite. */
export function neumannEigenvalues(n: number, spacing: number): Float32Array {
  const lambda = new Float32Array(n);
  const factor = -4 / (spacing * spacing);
  for (let m = 0; m < n; m++) {
    const s = Math.sin((Math.PI * m) / (2 * n));
    lambda[m] = factor * s * s;
  }
  return lambda;
}

function transformRows(
  src: Float32Array,
  nx: number,
  nz: number,
  basis: Float32Array,
  scale: Float32Array,
  inverse: boolean,
): Float32Array {
  const dst = new Float32Array(nx * nz);
  for (let k = 0; k < nz; k++) {
    for (let out = 0; out < nx; out++) {
      let sum = 0;
      for (let inn = 0; inn < nx; inn++) {
        if (inverse) sum += scale[inn] * src[k * nx + inn] * basis[inn * nx + out];
        else sum += src[k * nx + inn] * basis[out * nx + inn];
      }
      dst[k * nx + out] = inverse ? sum : scale[out] * sum;
    }
  }
  return dst;
}

function transformColumns(
  src: Float32Array,
  nx: number,
  nz: number,
  basis: Float32Array,
  scale: Float32Array,
  inverse: boolean,
): Float32Array {
  const dst = new Float32Array(nx * nz);
  for (let i = 0; i < nx; i++) {
    for (let out = 0; out < nz; out++) {
      let sum = 0;
      for (let inn = 0; inn < nz; inn++) {
        if (inverse) sum += scale[inn] * src[inn * nx + i] * basis[inn * nz + out];
        else sum += src[inn * nx + i] * basis[out * nz + inn];
      }
      dst[out * nx + i] = inverse ? sum : scale[out] * sum;
    }
  }
  return dst;
}

export function maxCellDivergence(
  u: Float32Array,
  v: Float32Array,
  nx: number,
  nz: number,
  dx: number,
  dz: number,
): number {
  let max = 0;
  for (let k = 0; k < nz; k++) {
    for (let i = 0; i < nx; i++) {
      const du = (u[k * (nx + 1) + i + 1] - u[k * (nx + 1) + i]) / dx;
      const dv = (v[(k + 1) * nx + i] - v[k * nx + i]) / dz;
      max = Math.max(max, Math.abs(du + dv));
    }
  }
  return max;
}

/**
 * Exact Neumann projection on a staggered grid.
 * The GPU solver uploads these tables and performs the same sum.
 */
export function projectNeumann(
  uIn: Float32Array,
  vIn: Float32Array,
  nx: number,
  nz: number,
  dx: number,
  dz: number,
  dt: number,
): { u: Float32Array; v: Float32Array; before: number; after: number } {
  const before = maxCellDivergence(uIn, vIn, nx, nz, dx, dz);
  const rhs = new Float32Array(nx * nz);
  let mean = 0;
  for (let k = 0; k < nz; k++) {
    for (let i = 0; i < nx; i++) {
      const du = (uIn[k * (nx + 1) + i + 1] - uIn[k * (nx + 1) + i]) / dx;
      const dv = (vIn[(k + 1) * nx + i] - vIn[k * nx + i]) / dz;
      const value = (du + dv) / dt;
      rhs[k * nx + i] = value;
      mean += value;
    }
  }
  mean /= nx * nz;
  for (let n = 0; n < rhs.length; n++) rhs[n] -= mean;

  const basisX = cosineBasis(nx);
  const basisZ = cosineBasis(nz);
  const scaleX = cosineScale(nx);
  const scaleZ = cosineScale(nz);
  const lambdaX = neumannEigenvalues(nx, dx);
  const lambdaZ = neumannEigenvalues(nz, dz);

  const row = transformRows(rhs, nx, nz, basisX, scaleX, false);
  const column = transformColumns(row, nx, nz, basisZ, scaleZ, false);
  for (let n = 0; n < nz; n++) {
    for (let m = 0; m < nx; m++) {
      const lambda = lambdaX[m] + lambdaZ[n];
      column[n * nx + m] = m === 0 && n === 0 ? 0 : column[n * nx + m] / lambda;
    }
  }
  const backZ = transformColumns(column, nx, nz, basisZ, scaleZ, true);
  const pressure = transformRows(backZ, nx, nz, basisX, scaleX, true);

  const u = uIn.slice();
  const v = vIn.slice();
  for (let k = 0; k < nz; k++) {
    for (let i = 1; i < nx; i++) {
      const grad = (pressure[k * nx + i] - pressure[k * nx + i - 1]) / dx;
      u[k * (nx + 1) + i] -= dt * grad;
    }
    u[k * (nx + 1)] = 0;
    u[k * (nx + 1) + nx] = 0;
  }
  for (let i = 0; i < nx; i++) {
    for (let k = 1; k < nz; k++) {
      const grad = (pressure[k * nx + i] - pressure[(k - 1) * nx + i]) / dz;
      v[k * nx + i] -= dt * grad;
    }
    v[i] = 0;
    v[nz * nx + i] = 0;
  }
  return { u, v, before, after: maxCellDivergence(u, v, nx, nz, dx, dz) };
}
