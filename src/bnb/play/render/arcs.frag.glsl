#version 330
#include "common.glsl"
// A disc filled up to the middle of its line, and a line along an arc with round ends.
in vec2 local;
flat in float v_radius;
flat in float v_width;
flat in vec4 v_fill;
flat in vec4 v_line;
flat in vec2 v_arc;

const float PI = 3.14159265;

out vec4 colour;

void main() {
    float from_centre = length(local);
    float offset = mod(atan(local.y, local.x) - v_arc.x + PI, 2.0 * PI) - PI;
    float distance;
    if (abs(offset) <= v_arc.y) {
        distance = abs(from_centre - v_radius);
    } else {
        float end = v_arc.x + sign(offset) * v_arc.y;
        distance = length(local - v_radius * vec2(cos(end), sin(end)));
    }
    float line_alpha = v_line.a * line_coverage(distance / pixel, v_width / pixel);
    float fill_alpha = v_fill.a * clamp((v_radius - from_centre) / pixel + 0.5, 0.0, 1.0);
    float alpha = line_alpha + fill_alpha * (1.0 - line_alpha);
    if (alpha <= 0.0) discard;
    vec3 rgb = (v_line.rgb * line_alpha + v_fill.rgb * fill_alpha * (1.0 - line_alpha)) / alpha;
    colour = vec4(rgb, alpha);
}
