@group(0) @binding(1) var<storage, read> vel_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> vel_dst: array<f32>;
@group(0) @binding(3) var<storage, read> field_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> field_out: array<f32>;
@group(0) @binding(5) var<storage, read> basis_x: array<f32>;
@group(0) @binding(6) var<storage, read> basis_z: array<f32>;
@group(0) @binding(7) var<storage, read> tables: array<f32>;

fn scale_x(m: u32) -> f32 { return tables[m]; }
fn scale_z(n: u32) -> f32 { return tables[P.nx + n]; }
fn lambda_x(m: u32) -> f32 { return tables[P.nx + P.nz + m]; }
fn lambda_z(n: u32) -> f32 { return tables[2u * P.nx + P.nz + n]; }

@compute @workgroup_size(64, 1)
fn divergence(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i >= P.nx || k >= P.nz) {
    return;
  }
  let du = (vel_src[u_at(i + 1u, k)] - vel_src[u_at(i, k)]) / P.dx;
  let dv = (vel_src[v_at(i, k + 1u)] - vel_src[v_at(i, k)]) / P.dz;
  field_out[cell(i, k)] = du + dv;
}

@compute @workgroup_size(1)
fn prepare_rhs() {
  let n = P.nx * P.nz;
  var mean = 0.0;
  for (var t = 0u; t < n; t = t + 1u) {
    mean = mean + field_in[t];
  }
  mean = mean / f32(n);
  for (var t = 0u; t < n; t = t + 1u) {
    field_out[t] = (field_in[t] - mean) / P.dt;
  }
}

@compute @workgroup_size(64)
fn dct_row(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= P.nz) {
    return;
  }
  for (var m = 0u; m < P.nx; m = m + 1u) {
    var sum = 0.0;
    for (var i = 0u; i < P.nx; i = i + 1u) {
      sum = sum + field_in[k * P.nx + i] * basis_x[m * P.nx + i];
    }
    field_out[k * P.nx + m] = scale_x(m) * sum;
  }
}

@compute @workgroup_size(64)
fn dct_column(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.nx) {
    return;
  }
  for (var n = 0u; n < P.nz; n = n + 1u) {
    var sum = 0.0;
    for (var k = 0u; k < P.nz; k = k + 1u) {
      sum = sum + field_in[k * P.nx + i] * basis_z[n * P.nz + k];
    }
    field_out[n * P.nx + i] = scale_z(n) * sum;
  }
}
@compute @workgroup_size(64, 1)
fn apply_lambda(@builtin(global_invocation_id) id: vec3u) {
  let m = id.x;
  let n = id.y;
  if (m >= P.nx || n >= P.nz) {
    return;
  }
  if (m == 0u && n == 0u) {
    field_out[0] = 0.0;
    return;
  }
  let lambda = lambda_x(m) + lambda_z(n);
  field_out[n * P.nx + m] = field_in[n * P.nx + m] / lambda;
}

@compute @workgroup_size(64)
fn idct_column(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.nx) {
    return;
  }
  for (var k = 0u; k < P.nz; k = k + 1u) {
    var sum = 0.0;
    for (var n = 0u; n < P.nz; n = n + 1u) {
      sum = sum + scale_z(n) * field_in[n * P.nx + i] * basis_z[n * P.nz + k];
    }
    field_out[k * P.nx + i] = sum;
  }
}

@compute @workgroup_size(64)
fn idct_row(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= P.nz) {
    return;
  }
  for (var i = 0u; i < P.nx; i = i + 1u) {
    var sum = 0.0;
    for (var m = 0u; m < P.nx; m = m + 1u) {
      sum = sum + scale_x(m) * field_in[k * P.nx + m] * basis_x[m * P.nx + i];
    }
    field_out[k * P.nx + i] = sum;
  }
}

@compute @workgroup_size(64, 1)
fn gradient_u(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i > P.nx || k >= P.nz) {
    return;
  }
  if (i == 0u || i == P.nx) {
    vel_dst[u_at(i, k)] = 0.0;
    return;
  }
  let grad = (field_in[cell(i, k)] - field_in[cell(i - 1u, k)]) / P.dx;
  vel_dst[u_at(i, k)] = vel_src[u_at(i, k)] - P.dt * grad;
}

@compute @workgroup_size(64, 1)
fn gradient_v(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i >= P.nx || k > P.nz) {
    return;
  }
  if (k == 0u || k == P.nz) {
    vel_dst[v_at(i, k)] = 0.0;
    return;
  }
  let grad = (field_in[cell(i, k)] - field_in[cell(i, k - 1u)]) / P.dz;
  vel_dst[v_at(i, k)] = vel_src[v_at(i, k)] - P.dt * grad;
}
