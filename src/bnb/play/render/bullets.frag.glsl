#version 330
#include "common.glsl"
uniform float head_pixels;
uniform vec2 trail_pixels;

in vec2 along_across;
flat in float trail_length;
flat in vec4 v_head;
flat in vec4 v_trail;
out vec4 colour;

void main() {
    float along = along_across.x, across = along_across.y;
    float head = clamp(head_pixels + 0.5 - length(along_across), 0.0, 1.0);
    // The trail: 0 at the head, 1 at its end, tapering and fading as a ship's trail does.
    float back = trail_length > 0.0 ? -along / trail_length : 2.0;
    float fresh = 1.0 - back;
    float width = mix(trail_pixels.x, trail_pixels.y, fresh);
    float inside = back >= 0.0 && back <= 1.0 ? line_coverage(abs(across), width) : 0.0;
    float trail = v_trail.a * fresh * inside;
    float alpha = max(v_head.a * head, trail);
    if (alpha <= 0.0) discard;
    colour = vec4(mix(v_trail.rgb, v_head.rgb, head), alpha);
}
