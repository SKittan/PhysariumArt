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
    sens_range_min: f32,
    sens_range_max: f32,
    sense_steps: f32,
    f_explore:f32,
    seed_1: u32,
    seed_2: u32
};

@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var<storage, read> slime_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> slime_out: array<f32>;
@group(0) @binding(3) var<storage, read> nutriment: array<f32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

// Hash function www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
fn hash(state: u32) -> u32
{
    var out = state ^ 2747636419u;
    out = out * 2654435769u;
    out ^= (out >> u32(16));
    out = out * 2654435769u;
    out ^= (out >> u32(16));
    out = out * 2654435769u;
    return out;
}

fn scaleToRange01(state: u32) -> f32
{
    return f32(state) / 4294967295.0;
}

fn rng(seed: u32) -> f32
{
    return scaleToRange01(hash(seed));
}

fn sense(phi: f32, a_x: f32, a_y: f32, max_x: f32, max_y: f32, n_agent: u32)
-> f32
{
    var c = 0.;

    for(var r = uniforms.sens_range_min; r <= uniforms.sens_range_max; r=r+1.)
    {
        let s_x = floor(cos(phi)*r + a_x);
        let s_y = floor(sin(phi)*r + a_y);

        if (s_x < 0. || s_x >= max_x ||s_y < 0. || s_y >= max_y) {break;}

        let s_i = u32(s_x) + u32(max_x*s_y);
        c = c + slime_in[s_i] + nutriment[s_i];
    }

    return c / uniforms.sense_steps +
           rng(uniforms.seed_2 + n_agent) * uniforms.f_explore;
}

@compute
@workgroup_size(256)
fn main(@builtin(global_invocation_id) gId: vec3<u32>,
        @builtin(local_invocation_index) lIdx: u32)
{
    let pi2 = 3.14159*2.;
    let max_x = f32(uniforms.sizeX);
    let max_y = f32(uniforms.sizeY);
    let len = u32(f32(uniforms.nAgents) / 256.);
    let i0 = gId.x * len;

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
            phi_sens = agents[i].phi + d_phi_sens;

            c = sense(phi_sens, agents[i].x, agents[i].y, max_x, max_y, i);
            if (c > c_max) {
                c_max = c;
                phi_max = phi_sens;
            } else {if ((c == c_max) &&
                        (rng(uniforms.seed_1 + lIdx)) > 0.5)
            {
                // randomly take new phi
                phi_max = phi_sens;
            }}
        }

        agents[i].phi = phi_max;
        // limit phi between -360 and 360
        if (agents[i].phi < -pi2) {
            agents[i].phi = agents[i].phi + pi2;
        } else { if(agents[i].phi > pi2) {
            agents[i].phi = agents[i].phi - pi2;
        }}

        agents[i].x = min(max(agents[i].x +
                              cos(agents[i].phi) * uniforms.v, 0.), max_x);
        agents[i].y = min(max(agents[i].y +
                              sin(agents[i].phi) * uniforms.v, 0.), max_y);

        let index: u32 = u32(floor(agents[i].x)) +
                         u32(floor(agents[i].y)) * uniforms.sizeX;
        slime_out[index] = slime_out[index] + uniforms.deposit;
    }
}