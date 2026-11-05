#version 330
#include "common.glsl"
// Filled triangles in world coordinates, a colour per vertex.
in vec2 position;
in vec4 tint;

out vec4 v_colour;

void main() {
    v_colour = tint;
    gl_Position = clip_of(position);
}
