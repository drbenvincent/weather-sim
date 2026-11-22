@group(0) @binding(1) var<storage, read> scal_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> scal_dst: array<f32>;
@group(0) @binding(3) var<storage, read> vel_src: array<f32>;
@group(0) @binding(4) var<storage, read_write> vel_dst: array<f32>;

fn clamp_cell(x: f32, z: f32) -> vec2f {
  return vec2f(
    clamp(x, 0.5 * P.dx, P.extent_x - 0.5 * P.dx),
    clamp(z, 0.5 * P.dz, P.extent_z - 0.5 * P.dz),
  );
}

fn backtrace(x: f32, z: f32) -> vec2f {
  let v1 = vec2f(sample_u(&vel_src, x, z), sample_v(&vel_src, x, z));
  let mid = clamp_cell(x - 0.5 * P.dt * v1.x, z - 0.5 * P.dt * v1.y);
  let v2 = vec2f(sample_u(&vel_src, mid.x, mid.y), sample_v(&vel_src, mid.x, mid.y));
  return clamp_cell(x - P.dt * v2.x, z - P.dt * v2.y);
}

@compute @workgroup_size(64, 1)
fn advect_scalar(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i >= P.nx || k >= P.nz) {
    return;
  }
  let x = (f32(i) + 0.5) * P.dx;
  let z = (f32(k) + 0.5) * P.dz;
  let back = backtrace(x, z);
  let p = cell(i, k);
  let n = P.nx * P.nz;
  for (var f = 0u; f < 4u; f = f + 1u) {
    scal_dst[f * n + p] = sample_scalar(&scal_src, f, back.x, back.y);
  }
}

@compute @workgroup_size(64, 1)
fn advect_u(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i > P.nx || k >= P.nz) {
    return;
  }
  if (i == 0u || i == P.nx) {
    vel_dst[u_at(i, k)] = 0.0;
    return;
  }
  let x = f32(i) * P.dx;
  let z = (f32(k) + 0.5) * P.dz;
  let back = backtrace(x, z);
  vel_dst[u_at(i, k)] = sample_u(&vel_src, back.x, back.y);
}

@compute @workgroup_size(64, 1)
fn advect_v(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i >= P.nx || k > P.nz) {
    return;
  }
  if (k == 0u || k == P.nz) {
    vel_dst[v_at(i, k)] = 0.0;
    return;
  }
  let x = (f32(i) + 0.5) * P.dx;
  let z = f32(k) * P.dz;
  let back = backtrace(x, z);
  vel_dst[v_at(i, k)] = sample_v(&vel_src, back.x, back.y);
}
