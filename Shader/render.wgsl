[[location(0)]]
var<storage, read> tex_coords: vec2<f32>;
[[location(0)]]
var<storage, read_write> f_color: vec4<f32>;

[[group(0), binding(0)]]
var<storage, read> slime: array<f32>;

[[block]]
struct Uniforms {
    nAgents: u32;
    sizeX: u32;
    sizeY: u32;
};
[[group(0), binding(1)]]
var<uniform> uniforms: Uniforms;


struct FragmentOutput {
    [[location(0)]] f_color: vec4<f32>;
};

[[stage(fragment)]]
fn main() -> FragmentOutput {
   let x: u32 = u32(tex_coords.x*f32(uniforms.sizeX));
   let y: u32 = u32(tex_coords.y*f32(uniforms.sizeY));
   let index: u32 = x + y*uniforms.sizeX;

    return FragmentOutput(vec4<f32>(vec3<f32>(slime[index]), 1.));
}