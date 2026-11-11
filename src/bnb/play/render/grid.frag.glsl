#version 330
#include "common.glsl"
// The dot grid: each fragment's distance to the nearest dot of the fine level and of the
// coarse level (`factor` times the spacing), as a dot DOT_RADIUS pixels round, the fine
// level's own dots faded by `fade`; inside the arena only. pattern: 0 square, 1 triangle,
// 2 hex (`grid.py`).
uniform float arena_radius;
uniform vec4 dot;
uniform int pattern;
uniform float spacing;  // the fine level's (m)
uniform float factor;
uniform float fade;

const float DOT_RADIUS = 1.5;  // pixels
const float EDGE_GAP = 3.0;  // pixels: no dot this close to the arena's edge
const float ROOT3 = 1.7320508;

out vec4 colour;

// The distance (m) from p to the nearest point of the lattice shift + cell * (i, j).
float rectangle_distance(vec2 p, vec2 cell, vec2 shift) {
    vec2 q = (p - shift) / cell;
    return length((q - floor(q + 0.5)) * cell);
}

// The triangular lattice of side `side`: two rectangular lattices (side, side·√3), the
// second shifted by half of each.
float triangle_distance(vec2 p, float side, vec2 shift) {
    vec2 cell = vec2(side, side * ROOT3);
    return min(rectangle_distance(p, cell, shift), rectangle_distance(p, cell, shift + 0.5 * cell));
}

// The distance (m) from p to the nearest dot of the pattern at this spacing. The honeycomb
// of side s is the triangular lattice of side s·√3 and its copy shifted by (0, s).
float lattice_distance(vec2 p, float s) {
    if (pattern == 0) {
        return rectangle_distance(p, vec2(s), vec2(0.0));
    }
    if (pattern == 1) {
        return triangle_distance(p, s, vec2(0.0));
    }
    return min(triangle_distance(p, s * ROOT3, vec2(0.0)), triangle_distance(p, s * ROOT3, vec2(0.0, s)));
}

float coverage(float distance) {
    return clamp(DOT_RADIUS + 0.5 - distance / pixel, 0.0, 1.0);
}

void main() {
    vec2 world = world_of(gl_FragCoord.xy);
    float inside = clamp((arena_radius - length(world)) / pixel - EDGE_GAP, 0.0, 1.0);
    float fine = coverage(lattice_distance(world, spacing)) * (1.0 - fade);
    float coarse = coverage(lattice_distance(world, spacing * factor));
    colour = vec4(dot.rgb, dot.a * max(fine, coarse) * inside);
}
