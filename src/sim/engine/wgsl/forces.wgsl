@group(0) @binding(1) var<storage, read> scal: array<f32>;
@group(0) @binding(2) var<storage, read> vel_src: array<f32>;
@group(0) @binding(3) var<storage, read_write> vel_dst: array<f32>;
@group(0) @binding(4) var<storage, read> refs: array<f32>;

fn buoyancy_at(i: u32, k: u32) -> f32 {
  let p = cell(i, k);
  let n = P.nx * P.nz;
  let theta = scal[p];
  let qv = scal[n + p];
  let qc = scal[2u * n + p];
  let qr = scal[3u * n + p];
  let theta_ref = refs[k];
  let qv_ref = refs[P.nz + k];
  return P.g * ((theta - theta_ref) / theta_ref + 0.608 * (qv - qv_ref) - qc - qr);
}

@compute @workgroup_size(64, 1)
fn force_u(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i > P.nx || k >= P.nz) {
    return;
  }
  if (i == 0u || i == P.nx) {
    vel_dst[u_at(i, k)] = 0.0;
    return;
  }
  let uc = vel_src[u_at(i, k)];
  let ul = vel_src[u_at(i - 1u, k)];
  let ur = vel_src[u_at(i + 1u, k)];
  var uz_m = uc;
  var uz_p = uc;
  if (k > 0u) {
    uz_m = vel_src[u_at(i, k - 1u)];
  }
  if (k + 1u < P.nz) {
    uz_p = vel_src[u_at(i, k + 1u)];
  }
  let lap = (ul - 2.0 * uc + ur) / (P.dx * P.dx) + (uz_m - 2.0 * uc + uz_p) / (P.dz * P.dz);
  let z = (f32(k) + 0.5) * P.dz;
  vel_dst[u_at(i, k)] = (uc + P.dt * P.nu * lap) * sponge_factor(z);
}

@compute @workgroup_size(64, 1)
fn force_v(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i >= P.nx || k > P.nz) {
    return;
  }
  if (k == 0u || k == P.nz) {
    vel_dst[v_at(i, k)] = 0.0;
    return;
  }
  let vc = vel_src[v_at(i, k)];
  var vl = vc;
  var vr = vc;
  if (i > 0u) {
    vl = vel_src[v_at(i - 1u, k)];
  }
  if (i + 1u < P.nx) {
    vr = vel_src[v_at(i + 1u, k)];
  }
  let vz_m = vel_src[v_at(i, k - 1u)];
  let vz_p = vel_src[v_at(i, k + 1u)];
  let lap = (vl - 2.0 * vc + vr) / (P.dx * P.dx) + (vz_m - 2.0 * vc + vz_p) / (P.dz * P.dz);
  let buoy = 0.5 * (buoyancy_at(i, k - 1u) + buoyancy_at(i, k));
  vel_dst[v_at(i, k)] = (vc + P.dt * P.nu * lap + P.dt * buoy) * sponge_factor(f32(k) * P.dz);
}
