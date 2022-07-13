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

@group(0) @binding(0) var<storage, read> slime_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> slime_out: array<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@compute
@workgroup_size(256)
fn main(@builtin(local_invocation_index) liIdx: u32)
{
    let len = u32(f32(uniforms.sizeX * uniforms.sizeY) / 256.);
    let i0 = liIdx * len;

    for (var i=i0; i<i0+len; i=i+u32(1)){
        // get slime pixel coordinates
        var y0: u32 = u32(floor(f32(i) / f32(uniforms.sizeX)));
        var x0: u32 = i - y0*uniforms.sizeX;
        // reset slime out value
        slime_out[i] = 0.;
        // sum up 3x3 neighbours
        for (var dx=-1; dx<2; dx=dx+1){
            for(var dy=-1; dy<2; dy=dy+1){
                var x = i32(x0) + dx;
                var y = i32(y0) + dy;
                if (x < 0) {
                    x = i32(uniforms.sizeX) - 1;
                } else {if (u32(x) == uniforms.sizeX) {
                    x = 0;
                }}
                if (y < 0) {
                    y = i32(uniforms.sizeY) - 1;
                } else {if (u32(y) == uniforms.sizeY) {
                    y = 0;
                }}
                let idx = u32(x) + u32(y)*uniforms.sizeX;
                slime_out[i] = slime_out[i] + slime_in[idx];
        }}
        // calculate mean and decay
        slime_out[i] = min(slime_out[i] / 9. * 0.9, 1.);
    }
}