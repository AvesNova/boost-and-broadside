#version 330
#include "common.glsl"
// Lines of constant on-screen width: each vertex moves off its point along the normal.
in vec2 position;
in vec2 normal;
in float side;  // -1 or +1
in float width;  // pixels
in vec4 tint;

out vec4 v_colour;

void main() {
    v_colour = tint;
    gl_Position = clip_of(position + normal * side * 0.5 * width * pixel);
}
