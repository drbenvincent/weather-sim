@group(0) @binding(1) var<storage, read> scal_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> scal_dst: array<f32>;
@group(0) @binding(3) var<storage, read> refs: array<f32>;

@compute @workgroup_size(64, 1)
fn micro(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let k = id.y;
  if (i >= P.nx || k >= P.nz) {
    return;
  }
  let p = cell(i, k);
  let n = P.nx * P.nz;
  var theta = scal_src[p];
  var qv = scal_src[n + p];
  var qc = scal_src[2u * n + p];
  var qr = scal_src[3u * n + p];
  let qr_old = qr;
  let pressure = refs[2u * P.nz + k];
  let exner = refs[3u * P.nz + k];

  for (var iter = 0u; iter < 5u; iter = iter + 1u) {
    let tc = theta * exner - 273.15;
    let qs = sat_mixing(tc, pressure);
    let slope = dqs_dt(tc, pressure);
    let excess = qv - qs;
    var dq = excess / (1.0 + P.lcp * slope);
    dq = clamp(dq, -qc, qv);
    theta = theta + (P.lcp * dq) / exner;
    qv = qv - dq;
    qc = qc + dq;
  }
  qv = max(qv, 0.0);
  qc = max(qc, 0.0);

  if (qc > P.qc_thresh) {
    let converted = min(P.k_auto * (qc - P.qc_thresh) * P.dt, qc);
    qc = qc - converted;
    qr = qr + converted;
  }
  if (qc > 0.0 && qr > 0.0) {
    let accr = min(P.k_accr * qc * pow(qr, 0.875) * P.dt, qc);
    qc = qc - accr;
    qr = qr + accr;
  }
  let tc = theta * exner - 273.15;
  let qs = sat_mixing(tc, pressure);
  if (qv < qs && qr > 0.0) {
    let deficit = qs - qv;
    let evap = min(P.k_evap * deficit * sqrt(qr) * P.dt, min(qr, deficit));
    qr = qr - evap;
    qv = qv + evap;
    theta = theta - (P.lcp * evap) / exner;
  }
  var qr_above = 0.0;
  if (k + 1u < P.nz) {
    qr_above = scal_src[3u * n + cell(i, k + 1u)];
  }
  let settled = qr_old + (qr_above - qr_old) * P.vt * P.dt / P.dz;
  qr = max(qr + settled - qr_old, 0.0);

  scal_dst[p] = theta;
  scal_dst[n + p] = qv;
  scal_dst[2u * n + p] = qc;
  scal_dst[3u * n + p] = qr;
}
