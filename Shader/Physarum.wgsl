struct Agent {
    x: f32,
    y: f32,
    phi: f32,
};

struct Uniforms {
    nAgents: u32,
    sizeX: u32,
    sizeY: u32,
    deposit: f32,
    decay: f32,
    v: f32,
    d_phi_sens: f32,
    phi_sens_0: f32,
    phi_sens_1: f32,
    sens_range: f32
};

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> slime_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> slime_out: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_index) liIdx: u32)
{
    let max_x = f32(uniforms.sizeX);
    let max_y = f32(uniforms.sizeY);
    let len = u32(f32(uniforms.nAgents) / 256.);
    let i0 = liIdx * len;

    var phi_max: f32;  // orientation at max. concentration
    var phi_sens: f32;  // Current sensor orientation
    var c: f32;
    var c_max: f32;  // max. concentration

    for (var i=i0; i<i0+len; i=i+u32(1)){
        if(i > uniforms.nAgents) {
            break;
        }

        // Detect max. slime concentration
        // It's unlikely that there are often multiple cells
        // with exact the same concentration -> first detection is selected
        phi_max = 0.;  // init phi_max for a no slime case
        c_max = 0.;
        for (var d_phi_sens = uniforms.phi_sens_0;
             d_phi_sens<=uniforms.phi_sens_1;
             d_phi_sens = d_phi_sens + uniforms.d_phi_sens) {
            phi_sens = agents[i].phi - d_phi_sens;
            c = slime_in[u32(round((cos(phi_sens)*uniforms.sens_range +
                                    agents[i].x) +
                                    (sin(phi_sens)*uniforms.sens_range +
                                    agents[i].y) * max_x))];
            if (c > c_max) {
                c_max = c;
                phi_max = phi_sens;
            }
        }

        agents[i].phi = phi_max;
        agents[i].x = agents[i].x + cos(agents[i].phi) * uniforms.v;
        agents[i].y = agents[i].y + sin(agents[i].phi) * uniforms.v;

        // Wrap environment for agents
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
        slime_out[index] = slime_out[index] + uniforms.deposit;
    }
}