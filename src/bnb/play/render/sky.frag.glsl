#version 330
#include "common.glsl"
// The sky inside the arena disc; beyond its edge the boundary damage zone, hatched in the
// boundary colour over the outside colour, and a line on the edge.
uniform float arena_radius;
uniform vec4 sky;
uniform vec4 outside;
uniform vec4 boundary;
uniform vec4 edge;

const float HATCH_SPACING = 12.0;  // pixels between hatch lines
const float HATCH_WIDTH = 3.0;  // pixels
const float HATCH_OPACITY = 0.35;
const float EDGE_WIDTH = 1.5;  // pixels

out vec4 colour;

void main() {
    vec2 world = world_of(gl_FragCoord.xy);
    float beyond = (length(world) - arena_radius) / pixel;  // pixels past the edge
    // Hatching anchored to the world, so it pans with the map.
    float stripe = (world.x + world.y) * 0.70710678 / (HATCH_SPACING * pixel);
    float hatch = line_coverage(abs(fract(stripe) - 0.5) * HATCH_SPACING, HATCH_WIDTH);
    vec3 zone = mix(outside.rgb, boundary.rgb, HATCH_OPACITY * hatch);
    vec3 ground = mix(sky.rgb, zone, clamp(beyond + 0.5, 0.0, 1.0));
    colour = vec4(mix(ground, edge.rgb, line_coverage(abs(beyond), EDGE_WIDTH)), 1.0);
}
