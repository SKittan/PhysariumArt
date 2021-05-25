
#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) readonly buffer Buffer {
    float[] slime;
};
layout(set = 0, binding = 1) uniform Uniforms {
    uint nAgents;
    uint sizeX;
    uint sizeY;
};

void main() {
    uint x = uint(tex_coords.x*sizeX);
    uint y = uint(tex_coords.y*sizeY);
    uint index = x*y + y;
    f_color = vec4(vec3(slime[index]), 1.);
}