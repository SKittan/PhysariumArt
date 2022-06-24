[[block]]
struct Agents {
    x: f32;
    y: f32;
    phi: f32;
    sens: u32;  // sensor values: 0, 1, 2, 4, 1+2, 2+4
};
[[group(0), binding(0)]]
var<storage, read_write> agents: Agents;

[[block]]
struct Slime {
    c: [[stride(4)]] array<f32>; // concentration
};
[[group(0), binding(1)]]
var<storage, read_write> slime: Slime;

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
        slime.c[i] = f32(i)/f32(uniforms.sizeX * uniforms.sizeY);
    }
}