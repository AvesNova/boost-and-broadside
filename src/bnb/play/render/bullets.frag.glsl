#version 330
#include "common.glsl"
uniform float length_pixels;
uniform float width_pixels;

in vec2 along_across;
flat in vec4 v_colour;
out vec4 colour;

void main() {
    float along = clamp(along_across.x, 0.0, length_pixels);
    float distance = length(vec2(along_across.x - along, along_across.y));
    float alpha = v_colour.a * line_coverage(distance, width_pixels);
    if (alpha <= 0.0) discard;
    colour = vec4(v_colour.rgb, alpha);
}
