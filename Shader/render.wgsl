[[block]]
struct Slime {
    c: [[stride(4)]] array<f32>; // concentration
};
[[group(0), binding(0)]]
var<storage, read> slime: Slime;

[[block]]
struct Uniforms {
    nAgents: u32;
    sizeX: u32;
    sizeY: u32;
};
[[group(0), binding(1)]]
var<uniform> uniforms: Uniforms;


[[stage(fragment)]]
fn main([[location(0)]] tex_coords: vec2<f32>) -> [[location(0)]] vec4<f32> {
    let x: u32 = u32(tex_coords.x*f32(uniforms.sizeX));
    let y: u32 = u32(tex_coords.y*f32(uniforms.sizeY));
    let index: u32 = x + y*uniforms.sizeX;

    // Debug: Visualize render area
    let c: f32 = f32(x+y) / f32(uniforms.sizeX*uniforms.sizeY);
    return vec4<f32>(vec3<f32>(slime.c[index] + c), 1.);

}