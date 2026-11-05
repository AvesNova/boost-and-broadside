#version 330
#include "common.glsl"
// One instance per line segment: a quad around it, a fixed width on screen.
in vec2 corner;  // x in [0, 1] from the start to the end, y in [-1, 1] across
in vec2 start;
in vec2 end;
in float width_pixels;
in vec4 tint;

out vec2 along_across;  // pixels from the start along the segment, pixels from its axis
flat out float v_length;  // the segment's length in pixels
flat out float v_width;
flat out vec4 v_colour;

void main() {
    vec2 span = end - start;
    float metres = length(span);
    vec2 unit = metres > 0.0 ? span / metres : vec2(1.0, 0.0);
    vec2 side = vec2(-unit.y, unit.x);
    float pad = 0.5 * width_pixels + 1.0;
    v_length = metres / pixel;
    float along = mix(-pad, v_length + pad, corner.x);
    float across = corner.y * pad;
    along_across = vec2(along, across);
    v_width = width_pixels;
    v_colour = tint;
    gl_Position = clip_of(start + (unit * along + side * across) * pixel);
}
