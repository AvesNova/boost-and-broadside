#version 330
#include "common.glsl"
// The fog mask: 1 where some living observer of the team sees this point, by the rule of
// bnb.game.perception.sight.sighted (a test holds the two together away from the mask's
// edges). A core blocks a sight line when exactly one end is inside it, or neither is and
// the line passes within its radius; an observer sees as far as its range.
uniform sampler2D observers;  // (O, 1): x, y, vision range
uniform int observer_count;
uniform sampler2D cores;  // (M, 1): x, y, core radius squared
uniform int core_count;

out float seen;

bool clear_line(vec2 observer, vec2 target) {
    vec2 line = target - observer;
    float length_squared = dot(line, line);
    for (int i = 0; i < core_count; ++i) {
        vec3 core = texelFetch(cores, ivec2(i, 0), 0).xyz;
        vec2 from_target = target - core.xy;
        vec2 to_centre = core.xy - observer;
        bool target_in = dot(from_target, from_target) < core.z;
        bool observer_in = dot(to_centre, to_centre) < core.z;
        if (target_in != observer_in) return false;
        if (!observer_in) {
            float along = dot(to_centre, line);
            float across = to_centre.x * line.y - to_centre.y * line.x;
            if (across * across < core.z * length_squared && along > 0.0
                && along < length_squared) return false;
        }
    }
    return true;
}

void main() {
    vec2 target = world_of(gl_FragCoord.xy);
    seen = 0.0;
    for (int o = 0; o < observer_count; ++o) {
        vec3 observer = texelFetch(observers, ivec2(o, 0), 0).xyz;
        vec2 line = target - observer.xy;
        if (dot(line, line) > observer.z * observer.z) continue;
        if (clear_line(observer.xy, target)) {
            seen = 1.0;
            break;
        }
    }
}
