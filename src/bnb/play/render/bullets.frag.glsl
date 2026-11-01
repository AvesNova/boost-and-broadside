#version 330
#include "common.glsl"
uniform float length_pixels;
uniform float width_pixels;
uniform sampler2D fog_mask;  // 1 where the view's team sees, in screen space
uniform bool fogged;

in vec2 along_across;
flat in vec4 v_colour;
flat in float v_masked;
out vec4 colour;

void main() {
    if (fogged && v_masked > 0.5 && texture(fog_mask, gl_FragCoord.xy / viewport).r < 0.5)
        discard;
    float along = clamp(along_across.x, 0.0, length_pixels);
    float distance = length(vec2(along_across.x - along, along_across.y));
    float alpha = v_colour.a * line_coverage(distance, width_pixels);
    if (alpha <= 0.0) discard;
    colour = vec4(v_colour.rgb, alpha);
}
