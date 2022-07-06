struct Agent {
    x: f32;
    y: f32;
    phi: f32;
    sens: u32;  // sensor values: 0, 1, 2, 4, 1+2, 2+4
};
[[block]]
struct Agents {
    agents: array<Agent>;
};
[[group(0), binding(0)]]
var<storage, read_write> in: Agents;

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
    decay: f32;
};
[[group(0), binding(2)]]
var<uniform> uniforms: Uniforms;


[[stage(compute), workgroup_size(10)]]
fn main([[builtin(local_invocation_index)]] liIdx: u32)
{
    let pi = 3.14159;

    in.agents[liIdx].phi= cos(sin(0.1 * f32(liIdx) * 95123878123.3214) *
                            2318746123.897631) * 2. * pi;
    in.agents[liIdx].x= in.agents[liIdx].x+ cos(in.agents[liIdx].phi);
    in.agents[liIdx].y= in.agents[liIdx].y+ sin(in.agents[liIdx].phi);

    let len = u32(f32(uniforms.sizeX * uniforms.sizeY) / 256.);
    let i0 = liIdx * len;

    let index: u32 = u32(round(in.agents[liIdx].x+
                               in.agents[liIdx].y* f32(uniforms.sizeX)));
    slime.c[index] = slime.c[index] + 0.1;
}