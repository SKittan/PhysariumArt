[[group(0), binding(0)]]
var<storage, read_write> agents: array<f32>;

[[group(0), binding(1)]]
var<storage, read_write> slime: array<f32>;

[[block]]
struct Uniforms {
    nAgents: u32;
    sizeX: u32;
    sizeY: u32;
};
[[group(0), binding(2)]]
var<uniform> uniforms: Uniforms;

[[stage(compute), workgroup_size(256)]]
fn main([[builtin(local_invocation_index)]] liIdx: u32) {
    let len = u32(f32(uniforms.sizeX * uniforms.sizeY) / 256.);
    let i0 = liIdx * len;

    for(var i: u32 = i0; i<(i0+len); i=i+u32(1)){
        slime[i] = f32(i)/f32(uniforms.sizeX * uniforms.sizeY);
    }
}