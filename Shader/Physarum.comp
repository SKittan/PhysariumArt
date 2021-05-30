#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) buffer Buffer {
    float[] slime;
};
layout(set = 0, binding = 1) uniform Uniforms {
    uint nAgents;
    uint sizeX;
    uint sizeY;
};


void main() {
    uint len = sizeX * sizeY / 256;
    uint i0 = gl_LocalInvocationIndex * len;

    for(uint i=i0; i<(i0+len); i++){
        slime[i] = float(i)/float(sizeX * sizeY);
    }
}