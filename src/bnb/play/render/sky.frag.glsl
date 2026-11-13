#version 330
#include "common.glsl"
// The arena's ground and its edge, in two layers (`sky.py`). Ground (`layer` 0, first): the
// sky inside the arena disc, the outside colour beyond it. Edge (`layer` 1, over the
// fields, contours and zones): beyond the edge a veil of the outside colour hatched in the
// boundary colour, so fields that reach past the edge show through dimmed, and a line on
// the edge. The edge layer follows the boundary (`boundary_radius`: elimination's shrinks
// inside the arena); a faint ring stays on the arena's edge once it has.
uniform float arena_radius;
uniform float boundary_radius;
uniform vec4 sky;
uniform vec4 outside;
uniform vec4 boundary;
uniform vec4 edge;
uniform int layer;

const float HATCH_SPACING = 12.0;  // pixels between hatch lines
const float HATCH_WIDTH = 3.0;  // pixels
const float HATCH_OPACITY = 0.35;
const float VEIL_OPACITY = 0.6;  // the outside colour's over what lies beyond the edge
const float EDGE_WIDTH = 1.5;  // pixels
const float RING_OPACITY = 0.35;  // the arena's edge, once the boundary lies inside it

out vec4 colour;

void main() {
    vec2 world = world_of(gl_FragCoord.xy);
    float beyond = (length(world) - arena_radius) / pixel;  // pixels past the edge
    float past = clamp(beyond + 0.5, 0.0, 1.0);
    if (layer == 0) {
        colour = vec4(mix(sky.rgb, outside.rgb, past), 1.0);
        return;
    }
    beyond = (length(world) - boundary_radius) / pixel;  // pixels past the boundary
    past = clamp(beyond + 0.5, 0.0, 1.0);
    // Hatching anchored to the world, so it pans with the map: the veil's colour is the
    // outside colour with the hatch's share of boundary colour, its opacity the veil's
    // raised toward 1 on the hatch lines.
    float stripe = (world.x + world.y) * 0.70710678 / (HATCH_SPACING * pixel);
    float hatch = HATCH_OPACITY
        * line_coverage(abs(fract(stripe) - 0.5) * HATCH_SPACING, HATCH_WIDTH);
    vec3 zone = mix(outside.rgb, boundary.rgb, hatch);
    float veil = past * (VEIL_OPACITY + (1.0 - VEIL_OPACITY) * hatch);
    float line = line_coverage(abs(beyond), EDGE_WIDTH);
    // the arena's edge, faint, once the boundary has shrunk a pixel inside it
    float inside = clamp((arena_radius - boundary_radius) / pixel, 0.0, 1.0);
    float ring = RING_OPACITY * inside
        * line_coverage(abs(length(world) - arena_radius) / pixel, EDGE_WIDTH);
    line = max(line, ring);
    colour = vec4(mix(zone, edge.rgb, line), max(veil, line));
}
