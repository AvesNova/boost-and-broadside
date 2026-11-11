#version 330
#include "common.glsl"
// The distance contours (`contours.py`): the signed distance (m) to the shape set, the
// smooth union of circles first_circle..last_circle - 1 of the table and, with `edge`, the
// region beyond the arena; a line LINE_WIDTH pixels wide at every multiple of the fine and
// the coarse spacing, the fine level's faded by `fade`, and every fifth of each level (the
// index contours) stronger.
uniform sampler2D circles;  // (N, 1): centre x, y and radius (m); the fields, then the zones
uniform int first_circle;
uniform int last_circle;
uniform bool edge;
uniform float arena_radius;
uniform float smoothing;  // the smooth union's radius (m)
uniform vec4 ink;
uniform float spacing;  // the fine level's (m)
uniform float factor;
uniform float fade;

const float LINE_WIDTH = 1.0;  // pixels
const float INDEX_EVERY = 5.0;  // lines per index contour
const float INDEX_STRENGTH = 1.8;  // an index contour's opacity, times a plain line's
const float NONE = 1e9;  // the distance to an empty set (m)

out vec4 colour;

// The quadratic smooth minimum: min(a, b), less h²·k/4 where they are within k of each
// other (h = 1 - |a - b|/k), so the union's crease is rounded over a band k wide.
float smooth_minimum(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}

float signed_distance(vec2 world) {
    float distance = edge ? arena_radius - length(world) : NONE;
    for (int i = first_circle; i < last_circle; ++i) {
        vec3 circle = texelFetch(circles, ivec2(i, 0), 0).xyz;
        distance = smooth_minimum(distance, length(world - circle.xy) - circle.z, smoothing);
    }
    return distance;
}

// Coverage of the lines at every multiple of `step` (m); per_pixel: metres of distance
// per pixel here.
float lines(float distance, float step, float per_pixel) {
    float off = abs(distance / step - floor(distance / step + 0.5)) * step;
    return line_coverage(off / per_pixel, LINE_WIDTH);
}

void main() {
    float distance = signed_distance(world_of(gl_FragCoord.xy));
    float per_pixel = max(length(vec2(dFdx(distance), dFdy(distance))), 1e-3 * pixel);
    float coarse_step = spacing * factor;
    float plain = max(lines(distance, spacing, per_pixel) * (1.0 - fade),
                      lines(distance, coarse_step, per_pixel));
    float index = max(lines(distance, INDEX_EVERY * spacing, per_pixel) * (1.0 - fade),
                      lines(distance, INDEX_EVERY * coarse_step, per_pixel));
    float alpha = ink.a * max(plain, INDEX_STRENGTH * index);
    colour = vec4(ink.rgb, min(alpha, 1.0));
}
