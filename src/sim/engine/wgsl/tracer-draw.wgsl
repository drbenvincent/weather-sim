struct Particle {
  pos: vec2f,
  t0: vec2f,
  t1: vec2f,
  t2: vec2f,
  t3: vec2f,
  life: f32,
  seed: f32,
}

@group(0) @binding(1) var<storage, read> particles: array<Particle>;
@group(0) @binding(2) var<storage, read> vel: array<f32>;
@group(0) @binding(3) var<storage, read> scal: array<f32>;
@group(0) @binding(4) var<storage, read> refs: array<f32>;

struct TrailOut {
  @builtin(position) clip: vec4f,
  @location(0) color: vec4f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TrailOut {
  let p = particles[ii];
  let v = vec2f(sample_u(&vel, p.pos.x, p.pos.y), sample_v(&vel, p.pos.x, p.pos.y));
  let speed = length(v);
  let clip_v = vec2f(v.x / P.extent_x, v.y / P.extent_z) * 2.0;
  let clip_len = length(clip_v);
  let dir = select(vec2f(1.0, 0.0), clip_v / max(clip_len, 1e-6), clip_len > 1e-6);
  let moving = speed > 1.2;
  let motion = select(0.0, clamp((speed - 1.2) / 10.0, 0.0, 1.0), moving);
  let trail = dir * 0.07 * motion;
  let nrm = vec2f(-dir.y, dir.x) * select(0.0028, 0.0032, moving);
  let head = vec2f(p.pos.x / P.extent_x * 2.0 - 1.0, p.pos.y / P.extent_z * 2.0 - 1.0);
  let tail = head - trail;
  let along = select(0.0, 1.0, vi == 2u || vi == 4u || vi == 5u);
  let side = select(-1.0, 1.0, vi == 1u || vi == 2u || vi == 4u);
  let q = mix(head, tail, along) + nrm * side;
  let anomaly = sample_scalar(&scal, 0u, p.pos.x, p.pos.y) - sample_ref(&refs, 0u, p.pos.y);
  let warm = clamp(anomaly / 8.0, -1.0, 1.0) * 0.5 + 0.5;
  let rgb = mix(vec3f(0.55, 0.72, 0.82), vec3f(0.95, 0.78, 0.55), warm);
  let fade = (1.0 - along * motion) * select(0.5, 0.82, moving);
  return TrailOut(vec4f(q, 0.0, 1.0), vec4f(rgb, fade));
}

@fragment
fn fs(in: TrailOut) -> @location(0) vec4f {
  return in.color;
}
