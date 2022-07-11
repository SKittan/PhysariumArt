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


[[stage(compute), workgroup_size(256)]]
fn main([[builtin(local_invocation_index)]] liIdx: u32)
{
    let pi = 3.14159;

    let len = u32(f32(uniforms.nAgents) / 256.);
    let i0 = liIdx * len;

    for (var i=i0; i<i0+len; i=i+u32(1)){
        if(i > uniforms.nAgents) {
            break;
        }

        in.agents[i].phi = cos(sin(f32(i) * 95123878123.3214) *
                                2318746123.897631) * pi;
        in.agents[i].x = in.agents[i].x + cos(in.agents[i].phi);
        in.agents[i].y = in.agents[i].y + sin(in.agents[i].phi);

        // Wrap environment for agents
        let max_x = f32(uniforms.sizeX);
        let max_y = f32(uniforms.sizeY);
        if (in.agents[i].x < 0.){
            in.agents[i].x = in.agents[i].x + max_x;
        } else {if (in.agents[i].x > max_x) {
            in.agents[i].x = max_x - in.agents[i].x;
        }}
        if (in.agents[i].y < 0.){
            in.agents[i].y = in.agents[i].y + max_y;
        } else {if (in.agents[i].y > max_y) {
            in.agents[i].y = max_y - in.agents[i].y;
        }}

        let index: u32 = u32(round(in.agents[i].x +
                                in.agents[i].y * max_x));
        slime.c[index] = slime.c[index] + 0.1;
    }
}