#version 330
#include "common.glsl"
// A tracer: the quad around a bullet's round head and its trail behind it along its
// direction of flight; the head and the widths are fixed sizes on screen.
in vec2 corner;  // x in [0, 1] from the trail's end to past the head, y in [-1, 1] across
in vec2 head;
in vec2 direction;
in float trail_metres;
in vec4 head_tint;
in vec4 trail_tint;

uniform float head_pixels;  // the head's radius
uniform vec2 trail_pixels;  // the trail's width at its end and at the head

out vec2 along_across;  // pixels ahead of the head (negative behind), pixels from the axis
flat out float trail_length;  // pixels
flat out vec4 v_head;
flat out vec4 v_trail;

void main() {
    trail_length = trail_metres / pixel;
    float reach = max(head_pixels, 0.5 * trail_pixels.y) + 1.0;
    float along = mix(-trail_length - 1.0, reach, corner.x);
    float across = corner.y * reach;
    along_across = vec2(along, across);
    vec2 side = vec2(-direction.y, direction.x);
    v_head = head_tint;
    v_trail = trail_tint;
    gl_Position = clip_of(head + (direction * along + side * across) * pixel);
}
