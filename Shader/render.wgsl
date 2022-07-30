struct Uniforms {
    nAgents: u32,
    sizeX: u32,
    sizeY: u32,
    deposit: f32,
    decay: f32,
    v: f32,
    phi_sens: f32,
    turn_speed: f32,
    sens_range_min: f32,
    sens_range_max: f32,
    sense_steps: f32,
    w_nutriment: f32,
    seed: u32,
};

struct Color {
    r: f32,
    g: f32,
    b: f32
}

@group(0) @binding(0) var<storage, read> slime: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> slime_color: array<Color>;

@fragment
fn main(@location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let x = u32(tex_coords.x * f32(uniforms.sizeX));
    let y = u32(tex_coords.y * f32(uniforms.sizeY));
    let index = x + y*uniforms.sizeX;

    return vec4<f32>(slime_color[index].r * slime[index],
                     slime_color[index].g * slime[index],
                     slime_color[index].b * slime[index],
                     1.
                     );
}