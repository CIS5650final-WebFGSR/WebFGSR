// WebFGSR - supersampled reference: accumulate one jittered full-res sample.
//
// accOut = sample * weight            (first sample, params.x = weight)
// accOut = accIn + sample * weight    (following samples)
// With weight = 1/N after N samples accOut is the box-filtered average of N
// sub-pixel offsets, i.e. an N-sample supersampled ground truth.
// Whether this is the first sample is decided by which pipeline runs it
// (FIRST_SAMPLE is prepended as a const by the host).

struct AccParams {
    // x: weight applied to the sample (1/N)   yzw: unused
    weight : vec4f,
};

@group(0) @binding(0) var<uniform> params : AccParams;
@group(0) @binding(1) var sampleTex : texture_2d<f32>;
@group(0) @binding(2) var accIn : texture_2d<f32>;
@group(0) @binding(3) var accOut : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let size = textureDimensions(accOut);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let p = vec2i(id.xy);
    let s = textureLoad(sampleTex, p, 0) * params.weight.x;
    // accIn is always read so the binding stays live in the pipeline layout
    let previous = textureLoad(accIn, p, 0);
    var acc = s + previous;
    if (FIRST_SAMPLE) {
        acc = s;
    }
    textureStore(accOut, p, vec4f(acc.rgb, 1.0));
}
