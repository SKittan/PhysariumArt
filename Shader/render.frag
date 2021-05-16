
#version 450

layout(location = 0) in vec2 tex_coords;
layout(location = 0) out vec4 f_color;


float rnd(vec2 xy) {
    return fract(sin(dot(xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    f_color = vec4(vec3(rnd(tex_coords)), 1.);
}