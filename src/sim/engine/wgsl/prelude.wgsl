struct Params {
  nx: u32,
  nz: u32,
  dx: f32,
  dz: f32,
  dt: f32,
  nu: f32,
  g: f32,
  lcp: f32,
  sponge: f32,
  k_accr: f32,
  k_evap: f32,
  qc_thresh: f32,
  vt: f32,
  k_auto: f32,
  rho_air: f32,
  rho_water: f32,
  extent_x: f32,
  extent_z: f32,
  sponge_z: f32,
  sponge_depth: f32,
}

@group(0) @binding(0) var<uniform> P: Params;

fn cell(i: u32, k: u32) -> u32 {
  return k * P.nx + i;
}

fn field_at(field: u32, i: u32, k: u32) -> u32 {
  return field * P.nx * P.nz + cell(i, k);
}

fn u_at(i: u32, k: u32) -> u32 {
  return k * (P.nx + 1u) + i;
}

fn v_at(i: u32, k: u32) -> u32 {
  return (P.nx + 1u) * P.nz + k * P.nx + i;
}

fn sat_vapor(tc: f32) -> f32 {
  return 611.2 * exp(17.67 * tc / (tc + 243.5));
}

fn sat_mixing(tc: f32, pressure: f32) -> f32 {
  let e = sat_vapor(tc);
  return 0.622 * e / max(pressure - e, 1.0);
}

fn dqs_dt(tc: f32, pressure: f32) -> f32 {
  let e = sat_vapor(tc);
  let des = e * 17.67 * 243.5 / pow(tc + 243.5, 2.0);
  return 0.622 * pressure * des / pow(max(pressure - e, 1.0), 2.0);
}

fn sponge_factor(z: f32) -> f32 {
  if (z <= P.sponge_z) {
    return 1.0;
  }
  let s = (z - P.sponge_z) / P.sponge_depth;
  return 1.0 / (1.0 + P.dt * P.sponge * s * s);
}

fn sample_scalar(s: ptr<storage, array<f32>, read>, field: u32, x: f32, z: f32) -> f32 {
  let gx = clamp(x / P.dx - 0.5, 0.0, f32(P.nx - 1u));
  let gz = clamp(z / P.dz - 0.5, 0.0, f32(P.nz - 1u));
  let i0 = u32(gx);
  let k0 = u32(gz);
  let i1 = min(i0 + 1u, P.nx - 1u);
  let k1 = min(k0 + 1u, P.nz - 1u);
  let tx = gx - f32(i0);
  let tz = gz - f32(k0);
  let a = (*s)[field_at(field, i0, k0)];
  let b = (*s)[field_at(field, i1, k0)];
  let c = (*s)[field_at(field, i0, k1)];
  let d = (*s)[field_at(field, i1, k1)];
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

fn sample_u(s: ptr<storage, array<f32>, read>, x: f32, z: f32) -> f32 {
  let gx = clamp(x / P.dx, 0.0, f32(P.nx));
  let gz = clamp(z / P.dz - 0.5, 0.0, f32(P.nz - 1u));
  let i0 = min(u32(gx), P.nx);
  let k0 = u32(gz);
  let i1 = min(i0 + 1u, P.nx);
  let k1 = min(k0 + 1u, P.nz - 1u);
  let tx = gx - f32(i0);
  let tz = gz - f32(k0);
  let a = (*s)[u_at(i0, k0)];
  let b = (*s)[u_at(i1, k0)];
  let c = (*s)[u_at(i0, k1)];
  let d = (*s)[u_at(i1, k1)];
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

fn sample_v(s: ptr<storage, array<f32>, read>, x: f32, z: f32) -> f32 {
  let gx = clamp(x / P.dx - 0.5, 0.0, f32(P.nx - 1u));
  let gz = clamp(z / P.dz, 0.0, f32(P.nz));
  let i0 = u32(gx);
  let k0 = min(u32(gz), P.nz);
  let i1 = min(i0 + 1u, P.nx - 1u);
  let k1 = min(k0 + 1u, P.nz);
  let tx = gx - f32(i0);
  let tz = gz - f32(k0);
  let a = (*s)[v_at(i0, k0)];
  let b = (*s)[v_at(i1, k0)];
  let c = (*s)[v_at(i0, k1)];
  let d = (*s)[v_at(i1, k1)];
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

fn sample_ref(r: ptr<storage, array<f32>, read>, which: u32, z: f32) -> f32 {
  let gz = clamp(z / P.dz - 0.5, 0.0, f32(P.nz - 1u));
  let k0 = u32(gz);
  let k1 = min(k0 + 1u, P.nz - 1u);
  let t = gz - f32(k0);
  return mix((*r)[which * P.nz + k0], (*r)[which * P.nz + k1], t);
}
