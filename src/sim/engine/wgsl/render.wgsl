struct RenderParams {
  mode: u32,
  width: f32,
  height: f32,
  time: f32,
  extent_x: f32,
  extent_z: f32,
  dx: f32,
  dz: f32,
  nx: u32,
  nz: u32,
  div_scale: f32,
  pad: f32,
}

@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> scal: array<f32>;
@group(0) @binding(2) var<storage, read> refs: array<f32>;
@group(0) @binding(3) var<storage, read> div_field: array<f32>;

fn at(field: u32, i: u32, k: u32) -> u32 {
  return field * R.nx * R.nz + k * R.nx + i;
}

fn sample_field(field: u32, x: f32, z: f32) -> f32 {
  let gx = clamp(x / R.dx - 0.5, 0.0, f32(R.nx - 1u));
  let gz = clamp(z / R.dz - 0.5, 0.0, f32(R.nz - 1u));
  let i0 = u32(gx);
  let k0 = u32(gz);
  let i1 = min(i0 + 1u, R.nx - 1u);
  let k1 = min(k0 + 1u, R.nz - 1u);
  let tx = gx - f32(i0);
  let tz = gz - f32(k0);
  let a = scal[at(field, i0, k0)];
  let b = scal[at(field, i1, k0)];
  let c = scal[at(field, i0, k1)];
  let d = scal[at(field, i1, k1)];
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

fn sample_div(x: f32, z: f32) -> f32 {
  let gx = clamp(x / R.dx - 0.5, 0.0, f32(R.nx - 1u));
  let gz = clamp(z / R.dz - 0.5, 0.0, f32(R.nz - 1u));
  let i0 = u32(gx);
  let k0 = u32(gz);
  let i1 = min(i0 + 1u, R.nx - 1u);
  let k1 = min(k0 + 1u, R.nz - 1u);
  let tx = gx - f32(i0);
  let tz = gz - f32(k0);
  let a = div_field[k0 * R.nx + i0];
  let b = div_field[k0 * R.nx + i1];
  let c = div_field[k1 * R.nx + i0];
  let d = div_field[k1 * R.nx + i1];
  return mix(mix(a, b, tx), mix(c, d, tx), tz);
}

fn sample_theta_ref(z: f32) -> f32 {
  let gz = clamp(z / R.dz - 0.5, 0.0, f32(R.nz - 1u));
  let k0 = u32(gz);
  let k1 = min(k0 + 1u, R.nz - 1u);
  return mix(refs[k0], refs[k1], gz - f32(k0));
}

fn hash(n: f32) -> f32 {
  return fract(sin(n * 127.1) * 43758.5453);
}

fn temp_color(anomaly: f32) -> vec3f {
  let bg = vec3f(0.043, 0.055, 0.075);
  let steel = vec3f(0.10, 0.24, 0.34);
  let steel_deep = vec3f(0.05, 0.13, 0.20);
  let copper = vec3f(0.72, 0.45, 0.26);
  let copper_hot = vec3f(0.89, 0.66, 0.42);
  if (anomaly < 0.0) {
    let t = clamp(-anomaly / 12.0, 0.0, 1.0);
    return mix(bg, mix(steel, steel_deep, t), smoothstep(0.0, 0.28, t));
  }
  let t = clamp(anomaly / 8.0, 0.0, 1.0);
  return mix(bg, mix(copper, copper_hot, t), smoothstep(0.0, 0.22, t));
}

fn cloud_sample(x: f32, z: f32) -> f32 {
  let c0 = sample_field(2u, x, z);
  let c1 = sample_field(2u, x + 900.0, z);
  let c2 = sample_field(2u, x - 900.0, z);
  let c3 = sample_field(2u, x, z + 180.0);
  let c4 = sample_field(2u, x, z - 180.0);
  return c0 * 0.42 + (c1 + c2 + c3 + c4) * 0.145;
}

fn rain_alpha(uv: vec2f, qr: f32, time: f32) -> f32 {
  if (qr < 1e-6) {
    return 0.0;
  }
  let column = floor(uv.x * 74.0);
  let h = hash(column + 2.7);
  let y = fract(uv.y * 7.0 + time * 0.11 + h);
  let streak = smoothstep(0.0, 0.04, y) * smoothstep(0.38, 0.12, y);
  let across = smoothstep(0.62, 0.08, abs(fract(uv.x * 74.0 + h * 0.3) - 0.5));
  return streak * across * clamp(qr * 1600.0, 0.0, 0.9);
}

struct VsOut {
  @builtin(position) pos: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var x = -1.0;
  var y = -1.0;
  if (vi == 1u) {
    x = 3.0;
  }
  if (vi == 2u) {
    y = 3.0;
  }
  return VsOut(vec4f(x, y, 0.0, 1.0));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4f {
  let uv = in.pos.xy / vec2f(R.width, R.height);
  let x = uv.x * R.extent_x;
  let z = (1.0 - uv.y) * R.extent_z;
  let anomaly = sample_field(0u, x, z) - sample_theta_ref(z);
  let qc = cloud_sample(x, z);
  let qr = sample_field(3u, x, z);
  var rgb = temp_color(anomaly);
  if (R.mode == 1u) {
    rgb = temp_color(anomaly);
  } else if (R.mode == 2u) {
    let a = clamp(qc * 700.0, 0.0, 1.0);
    rgb = mix(vec3f(0.043, 0.055, 0.075), vec3f(0.82, 0.84, 0.86), a);
  } else if (R.mode == 3u) {
    rgb = vec3f(0.043, 0.055, 0.075);
  } else if (R.mode == 4u) {
    let d = clamp(sample_div(x, z) / max(R.div_scale, 1e-6), -1.0, 1.0);
    let bg = vec3f(0.043, 0.055, 0.075);
    if (d >= 0.0) {
      rgb = mix(bg, vec3f(0.78, 0.48, 0.28), d);
    } else {
      rgb = mix(bg, vec3f(0.28, 0.48, 0.58), -d);
    }
  }
  if (R.mode == 0u || R.mode == 2u) {
    let above = sample_field(2u, x, z + 250.0);
    let top = clamp((qc - above) * 600.0, 0.0, 1.0);
    let bright = mix(0.62, 1.0, top);
    let alpha = clamp(qc * 650.0, 0.0, 0.82);
    rgb = mix(rgb, vec3f(bright), alpha);
  }
  if (R.mode == 0u || R.mode == 3u) {
    let streak = rain_alpha(uv, qr, R.time);
    rgb = mix(rgb, vec3f(0.90, 0.93, 0.95), streak);
    rgb = min(rgb + vec3f(0.22) * streak, vec3f(1.0));
  }
  if (uv.y > 0.992) {
    rgb = vec3f(0.604, 0.580, 0.533);
  }
  let height = z / R.extent_z;
  rgb = rgb * mix(1.0, 0.86, height);
  return vec4f(rgb, 1.0);
}
