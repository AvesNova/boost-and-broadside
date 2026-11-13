#version 330
#include "common.glsl"
// The distance contours (`contours.py`): the distance (m) to the nearest edge of the shape
// set, every circle first_circle..last_circle - 1 of the table and, with `edge`, the map
// edge, from either side, so a shape inside another keeps its own rings; a line LINE_WIDTH pixels wide at every multiple of the fine and
// the coarse spacing, the fine level's faded by `fade`, and every fifth of each level (the
// index contours) stronger.
uniform sampler2D circles;  // (N, 1): centre x, y and radius (m); the fields, then the zones
uniform int first_circle;
uniform int last_circle;
uniform bool edge;
uniform float arena_radius;
uniform vec4 ink;
uniform float spacing;  // the fine level's (m)
uniform float factor;
uniform float fade;

const float LINE_WIDTH = 1.0;  // pixels
const float INDEX_EVERY = 5.0;  // lines per index contour
const float INDEX_STRENGTH = 1.8;  // an index contour's opacity, times a plain line's
const float SOFTNESS = 6.0;  // p of the soft minimum: where two edges are equally far it
                             // is 2^(-1/p) of their distance, rounding the crease
const float MIN_SLOPE = 0.7;  // the least distance change per pixel counted, in pixels' metres
const float NEAR = 1e-3;  // m: the closest an edge counts (a soft minimum term stays finite)

out vec4 colour;

// The soft minimum of the edges' distances, (sum d_i^-p)^(-1/p): 0 on every edge, at most
// the nearest edge's distance, and rounded where two edges are about equally far (the
// crease between two shapes' rings), relative to the distance, so far creases round more.
float edge_distance(vec2 world) {
    float sum = 0.0;
    if (edge) {
        sum += pow(max(abs(arena_radius - length(world)), NEAR), -SOFTNESS);
    }
    for (int i = first_circle; i < last_circle; ++i) {
        vec3 circle = texelFetch(circles, ivec2(i, 0), 0).xyz;
        sum += pow(max(abs(length(world - circle.xy) - circle.z), NEAR), -SOFTNESS);
    }
    return pow(sum, -1.0 / SOFTNESS);
}

// Coverage of the lines at every multiple of `step` (m); per_pixel: metres of distance
// per pixel here.
float lines(float distance, float step, float per_pixel) {
    float off = abs(distance / step - floor(distance / step + 0.5)) * step;
    return line_coverage(off / per_pixel, LINE_WIDTH);
}

void main() {
    float distance = edge_distance(world_of(gl_FragCoord.xy));
    // A distance to an edge changes by about a pixel's metres per pixel, less only at the
    // creases; on an edge itself (where |d| folds) the screen derivative can vanish, so it
    // is floored there, keeping the edges' own lines whole.
    float per_pixel = max(length(vec2(dFdx(distance), dFdy(distance))), MIN_SLOPE * pixel);
    float coarse_step = spacing * factor;
    float plain = max(lines(distance, spacing, per_pixel) * (1.0 - fade),
                      lines(distance, coarse_step, per_pixel));
    float index = max(lines(distance, INDEX_EVERY * spacing, per_pixel) * (1.0 - fade),
                      lines(distance, INDEX_EVERY * coarse_step, per_pixel));
    float alpha = ink.a * max(plain, INDEX_STRENGTH * index);
    colour = vec4(ink.rgb, min(alpha, 1.0));
}
