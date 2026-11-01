#version 330
#include "common.glsl"
// A short streak behind each bullet along its direction, a fixed size on screen.
in vec2 corner;  // x in [0, 1] from the tail to the tip, y in [-1, 1] across
in vec2 tip;
in vec2 direction;
in vec4 tint;

uniform float length_pixels;
uniform float width_pixels;

out vec2 along_across;  // pixels from the tail, pixels from the axis
flat out vec4 v_colour;

void main() {
    float pad = width_pixels;
    float along = mix(-pad, length_pixels + pad, corner.x);
    float across = corner.y * (0.5 * width_pixels + 1.0);
    along_across = vec2(along, across);
    vec2 side = vec2(-direction.y, direction.x);
    vec2 world = tip + (direction * (along - length_pixels) + side * across) * pixel;
    v_colour = tint;
    gl_Position = clip_of(world);
}
