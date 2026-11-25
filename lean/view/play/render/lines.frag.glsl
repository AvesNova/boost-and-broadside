#version 330
#include "common.glsl"
// A line with round ends.
in vec2 along_across;
flat in float v_length;
flat in float v_width;
flat in vec4 v_colour;
out vec4 colour;

void main() {
    float along = clamp(along_across.x, 0.0, v_length);
    float distance = length(vec2(along_across.x - along, along_across.y));
    float alpha = v_colour.a * line_coverage(distance, v_width);
    if (alpha <= 0.0) discard;
    colour = vec4(v_colour.rgb, alpha);
}
