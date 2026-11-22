const BINS: u32 = 16u;
const STRIDE: u32 = 8u;

@group(0) @binding(1) var<storage, read> vel: array<f32>;
@group(0) @binding(2) var<storage, read> scal: array<f32>;
@group(0) @binding(3) var<storage, read> div_field: array<f32>;
@group(0) @binding(4) var<storage, read_write> cols: array<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<f32>;
@group(0) @binding(6) var<storage, read_write> first_peak: array<f32>;

@compute @workgroup_size(1)
fn div_before() {
  var peak = 0.0;
  let n = P.nx * P.nz;
  for (var t = 0u; t < n; t = t + 1u) {
    peak = max(peak, abs(div_field[t]));
  }
  if (first_peak[0] == 0.0 && peak > 0.0) {
    first_peak[0] = peak;
  }
}

@compute @workgroup_size(1)
fn div_after() {
  var peak = 0.0;
  let n = P.nx * P.nz;
  for (var t = 0u; t < n; t = t + 1u) {
    peak = max(peak, abs(div_field[t]));
  }
  stats[9] = peak;
}

@compute @workgroup_size(64)
fn columns(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= P.nx) {
    return;
  }
  var max_w = 0.0;
  var max_qc = 0.0;
  var max_qr = 0.0;
  var cloud_top = 0.0;
  var sfc = 0.0;
  var sfc_n = 0.0;
  var mid = 0.0;
  var mid_n = 0.0;
  var rain = 0.0;
  var speed = 0.0;
  let n = P.nx * P.nz;
  for (var k = 0u; k < P.nz; k = k + 1u) {
    let z = (f32(k) + 0.5) * P.dz;
    let p = cell(i, k);
    let qc = scal[2u * n + p];
    let qr = scal[3u * n + p];
    let theta = scal[p];
    max_qc = max(max_qc, qc);
    max_qr = max(max_qr, qr);
    if (qc > 1e-5 && z > cloud_top) {
      cloud_top = z;
    }
    if (z < 4.0 * P.dz) {
      sfc = sfc + theta;
      sfc_n = sfc_n + 1.0;
    }
    if (z >= 800.0 && z <= 2000.0) {
      mid = mid + theta;
      mid_n = mid_n + 1.0;
    }
    if (k == 0u) {
      rain = qr * P.vt * P.rho_air / P.rho_water;
    }
    let uc = vel[u_at(i, k)];
    let ur = vel[u_at(i + 1u, k)];
    let vv = vel[v_at(i, k)];
    speed = max(speed, max(abs(uc), max(abs(ur), abs(vv))));
    max_w = max(max_w, vv);
  }
  let base = i * STRIDE;
  cols[base] = max_w;
  cols[base + 1u] = max_qc;
  cols[base + 2u] = max_qr;
  cols[base + 3u] = cloud_top;
  cols[base + 4u] = sfc / max(sfc_n, 1.0);
  cols[base + 5u] = mid / max(mid_n, 1.0);
  cols[base + 6u] = rain;
  cols[base + 7u] = speed;
}

@compute @workgroup_size(1)
fn finalize() {
  var max_w = 0.0;
  var max_qc = 0.0;
  var max_qr = 0.0;
  var cloud_top = 0.0;
  var cloud_x = 0.0;
  var rain = 0.0;
  var speed = 0.0;
  var best_grad = -1e9;
  var front_x = 0.0;
  for (var i = 0u; i < P.nx; i = i + 1u) {
    let base = i * STRIDE;
    let x = (f32(i) + 0.5) * P.dx;
    max_w = max(max_w, cols[base]);
    max_qr = max(max_qr, cols[base + 2u]);
    rain = max(rain, cols[base + 6u]);
    speed = max(speed, cols[base + 7u]);
    if (cols[base + 1u] > max_qc) {
      max_qc = cols[base + 1u];
    }
    if (cols[base + 3u] > cloud_top) {
      cloud_top = cols[base + 3u];
      cloud_x = x;
    }
    let left = select(i - 1u, i, i == 0u);
    let right = select(i + 1u, i, i + 1u == P.nx);
    let grad = (cols[right * STRIDE + 4u] - cols[left * STRIDE + 4u]) / (f32(right - left) * P.dx);
    if (grad > best_grad) {
      best_grad = grad;
      front_x = x;
    }
  }
  stats[0] = max_w;
  stats[1] = max_qc;
  stats[2] = max_qr;
  stats[3] = cloud_top;
  stats[4] = cloud_x;
  stats[5] = rain;
  stats[6] = front_x;
  stats[7] = best_grad;
  stats[10] = speed;

  let width = P.extent_x / f32(BINS);
  for (var b = 0u; b < BINS; b = b + 1u) {
    let x0 = f32(b) * width;
    let x1 = x0 + width;
    var sfc = 0.0;
    var mid = 0.0;
    var count = 0.0;
    for (var i = 0u; i < P.nx; i = i + 1u) {
      let x = (f32(i) + 0.5) * P.dx;
      if (x >= x0 && x < x1) {
        sfc = sfc + cols[i * STRIDE + 4u];
        mid = mid + cols[i * STRIDE + 5u];
        count = count + 1.0;
      }
    }
    stats[12u + b] = sfc / max(count, 1.0);
    stats[28u + b] = mid / max(count, 1.0);
  }
}
