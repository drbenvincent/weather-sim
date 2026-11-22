struct Display {
  dt: f32,
  count: u32,
}

struct Particle {
  pos: vec2f,
  t0: vec2f,
  t1: vec2f,
  t2: vec2f,
  t3: vec2f,
  life: f32,
  seed: f32,
}

@group(0) @binding(1) var<uniform> D: Display;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<storage, read> vel: array<f32>;

fn respawn(seed: f32) -> vec2f {
  let x = fract(sin(seed * 12.9898) * 43758.5) * P.extent_x;
  let z = fract(sin(seed * 78.233) * 24634.6) * P.extent_z;
  return vec2f(x, z);
}

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= D.count || D.dt == 0.0) {
    return;
  }
  var p = particles[id.x];
  let v = vec2f(sample_u(&vel, p.pos.x, p.pos.y), sample_v(&vel, p.pos.x, p.pos.y));
  p.t3 = p.t2;
  p.t2 = p.t1;
  p.t1 = p.t0;
  p.t0 = p.pos;
  p.pos = p.pos + v * D.dt;
  p.life = p.life + D.dt;
  let outside = p.pos.x < 0.0 || p.pos.x > P.extent_x || p.pos.y < 0.0 || p.pos.y > P.extent_z;
  if (outside || p.life > 900.0) {
    p.seed = fract(p.seed * 1.37 + 0.13);
    p.pos = respawn(p.seed);
    p.t0 = p.pos;
    p.t1 = p.pos;
    p.t2 = p.pos;
    p.t3 = p.pos;
    p.life = 0.0;
  }
  particles[id.x] = p;
}
