// WebFGSR - debug visualisations of SR inputs/outputs, written to a
// presentation-size rgba16float texture (nearest upscale from the source).
//
// mode 0: motion vectors   (src.rg, uv units)  -> hue = direction, saturation = magnitude
// mode 1: depth            (src.b, linear depth / far) -> grey ramp
// mode 2: SR blend weight  (src.a of the SR output) -> heat ramp
// mode 3: copy             (src.rgb), e.g. the jittered low-res input
// mode 4: raw G-buffer for validation: rg = 0.5 + 25 * motion (uv), b = 2 * depth

struct VizParams {
    // x: mode   y: motion-vector scale (pixels of src that saturate)   z: depth range   w: unused
    modeAndScale : vec4f,
};

@group(0) @binding(0) var<uniform> params : VizParams;
@group(0) @binding(1) var src : texture_2d<f32>;
@group(0) @binding(2) var dst : texture_storage_2d<rgba16float, write>;

fn heat(t : f32) -> vec3f {
    let x = clamp(t, 0.0, 1.0);
    return clamp(vec3f(1.5 - abs(4.0 * x - 3.0), 1.5 - abs(4.0 * x - 2.0), 1.5 - abs(4.0 * x - 1.0)), vec3f(0.0), vec3f(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let dsize = textureDimensions(dst);
    if (id.x >= dsize.x || id.y >= dsize.y) {
        return;
    }
    let ssize = textureDimensions(src);
    let sp = min(vec2i((vec2f(id.xy) + vec2f(0.5)) * vec2f(ssize) / vec2f(dsize)), vec2i(ssize) - vec2i(1));
    let v = textureLoad(src, sp, 0);
    let mode = i32(params.modeAndScale.x + 0.5);

    var c = v.rgb;
    if (mode == 0) {
        // direction -> hue, magnitude (saturating at modeAndScale.y source pixels) -> saturation
        let mvPix = v.xy * vec2f(ssize);
        let mag = clamp(length(mvPix) / max(params.modeAndScale.y, 1e-6), 0.0, 1.0);
        let angle = atan2(mvPix.y, mvPix.x);
        let hue = vec3f(angle, angle + 2.0943951, angle + 4.1887902);
        let wheel = 0.5 + 0.5 * cos(hue);
        c = mix(vec3f(0.92), wheel, mag);
    } else if (mode == 1) {
        let d = clamp(v.z / max(params.modeAndScale.z, 1e-6), 0.0, 1.0);
        c = vec3f(pow(1.0 - d, 2.0));
    } else if (mode == 2) {
        // weight of the current frame: ~0.05..0.25 when history is kept, 1 when rejected
        c = heat(v.a);
    } else if (mode == 4) {
        c = vec3f(0.5 + 25.0 * v.x, 0.5 + 25.0 * v.y, 2.0 * v.z);
    }
    textureStore(dst, vec2i(id.xy), vec4f(c, 1.0));
}
