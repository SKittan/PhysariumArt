struct Agent {
    x: f32,
    y: f32,
    phi: f32,
    sens: u32  // sensor values: 0, 1, 2, 4, 1+2, 2+4
};

struct Uniforms {
    nAgents: u32,
    sizeX: u32,
    sizeY: u32,
    decay: f32
};

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read_write> slime: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute
@workgroup_size(32)
fn main(@builtin(local_invocation_index) liIdx: u32)
{

    let len = u32(f32(uniforms.nAgents) / 32.);
    let i0 = liIdx * len;

    for (var i=i0; i<i0+len; i=i+u32(1)){
        if(i > uniforms.nAgents) {
            break;
        }

        agents[i].x = agents[i].x + cos(agents[i].phi);
        agents[i].y = agents[i].y + sin(agents[i].phi);

        // Wrap environment for agents
        let max_x = f32(uniforms.sizeX);
        let max_y = f32(uniforms.sizeY);
        if (agents[i].x < 0.){
            agents[i].x = agents[i].x + max_x;
        } else {if (agents[i].x > max_x) {
            agents[i].x = max_x - agents[i].x;
        }}
        if (agents[i].y < 0.){
            agents[i].y = agents[i].y + max_y;
        } else {if (agents[i].y > max_y) {
            agents[i].y = max_y - agents[i].y;
        }}

        let index: u32 = u32(round(agents[i].x + agents[i].y * max_x));
        slime[index] = slime[index] + 0.01;
    }
}