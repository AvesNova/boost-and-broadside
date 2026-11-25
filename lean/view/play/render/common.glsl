// Shared by every pass. camera: the view's centre (m) and the clip units per metre;
// pixel: metres per pixel; viewport: the target's size in pixels.
uniform vec4 camera;
uniform float pixel;
uniform vec2 viewport;

vec4 clip_of(vec2 world) {
    return vec4((world - camera.xy) * camera.zw, 0.0, 1.0);
}

vec2 world_of(vec2 fragment) {
    vec2 clip = fragment / viewport * 2.0 - 1.0;
    return camera.xy + clip / camera.zw;
}

// Coverage of a line `width` pixels wide at a distance of `distance` pixels from its middle.
float line_coverage(float distance, float width) {
    return clamp(0.5 * width + 0.5 - distance, 0.0, 1.0);
}
