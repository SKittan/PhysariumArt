
struct VertexOutput {
    [[location(0)]] tex_coords: vec2<f32>;
    [[builtin(position)]] clip_position: vec4<f32>;
};

[[stage(vertex)]]
fn main(
    [[builtin(vertex_index)]] in_vertex_index: u32,
) -> VertexOutput {
    let x = f32(1 - i32(in_vertex_index)) * 0.5;
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
    let tex_coords: vec2<f32> = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    let clip_position = vec4<f32>(x, y, 0.0, 1.0);

    return VertexOutput(tex_coords, clip_position);
}