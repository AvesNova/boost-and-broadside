#version 330
#include "common.glsl"
// The fog mask: 1 where some living observer of the team sees this point, by the rule of
// bnb.game.perception.sight.sighted (a test holds the two together away from the mask's
// edges). Each field has a core, a band (core edge to outer edge) and outside. A field
// blocks a sight line when exactly one end is inside its core and the other is outside its
// outer disc, or neither end is inside its core and the line passes within the core
// radius; an observer sees as far as its range.
uniform sampler2D observers;  // (O, 1): x, y, vision range
uniform int observer_count;
uniform sampler2D cores;  // (M, 1): x, y, core radius squared, outer radius squared
uniform int core_count;

out float seen;

bool clear_line(vec2 observer, vec2 target) {
    vec2 line = target - observer;
    float length_squared = dot(line, line);
    for (int i = 0; i < core_count; ++i) {
        vec4 core = texelFetch(cores, ivec2(i, 0), 0);
        vec2 from_target = target - core.xy;
        vec2 to_centre = core.xy - observer;
        float target_squared = dot(from_target, from_target);
        float observer_squared = dot(to_centre, to_centre);
        bool target_in = target_squared < core.z;
        bool observer_in = observer_squared < core.z;
        bool both_in_field = target_squared < core.w && observer_squared < core.w;
        if (target_in != observer_in && !both_in_field) return false;
        if (!target_in && !observer_in) {
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
    for (int index = 0; index < observer_count; ++index) {
        vec3 observer = texelFetch(observers, ivec2(index, 0), 0).xyz;
        vec2 line = target - observer.xy;
        if (dot(line, line) > observer.z * observer.z) continue;
        if (clear_line(observer.xy, target)) {
            seen = 1.0;
            break;
        }
    }
}
