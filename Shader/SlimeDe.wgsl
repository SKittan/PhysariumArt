[[block]]
struct Slime {
    c: [[stride(4)]] array<f32>; // concentration
};
[[group(0), binding(0)]]
var<storage, read> slime_in: Slime;
[[group(0), binding(1)]]
var<storage, read_write> slime_out: Slime;


[[block]]
struct Uniforms {
    nAgents: u32;
    sizeX: u32;
    sizeY: u32;
    decay: f32;
};
[[group(0), binding(2)]]
var<uniform> uniforms: Uniforms;


[[stage(compute), workgroup_size(10)]]
fn main([[builtin(local_invocation_index)]] liIdx: u32)
{
    let len = u32(f32(uniforms.sizeX * uniforms.sizeY) / 10.);
    let i0 = liIdx * len;

    for (var i=i0; i<i0+len; i=i+u32(1)){
        // copy and decay
        slime_out.c[i] =  slime_in.c[i]*uniforms.decay;

    }
}