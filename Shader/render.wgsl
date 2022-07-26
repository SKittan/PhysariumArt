struct Uniforms {
    nAgents: u32,
    sizeX: u32,
    sizeY: u32,
    deposit: f32,
    decay: f32,
    v: f32,
    phi_sens: f32,
    sens_range_min: f32,
    sens_range_max: f32,
    sense_steps: f32,
    seed: u32,
};
@group(0) @binding(0) var<storage, read> slime: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@fragment
fn main(@location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let x = u32(tex_coords.x * f32(uniforms.sizeX));
    let y = u32(tex_coords.y * f32(uniforms.sizeY));
    let index = x + y*uniforms.sizeX;

    return vec4<f32>(vec3<f32>(slime[index]), 1.);
}