// WebFGSR - first-frame upsample (no history yet).
//
// WGSL port of Mob-FGSR resources/SuperResolution/UpsampleFirstFrame.comp
// (https://github.com/Mob-FGSR/MobFGSR, CC BY-NC 4.0).
//
// Bilinear upsample of the jittered low-res colour, with the jitter undone so
// the result is registered to the un-jittered high-res pixel grid.

#include "common.wgsl"

@group(0) @binding(0) var<uniform> params : SRParams;
@group(0) @binding(1) var lrColor : texture_2d<f32>;
@group(0) @binding(2) var lrColorSampler : sampler; // linear, clamp-to-edge
@group(0) @binding(3) var historyOut : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let sizeHR = vec2i(params.presentationSize.xy);
    let posHR = vec2i(id.xy);
    if (posHR.x >= sizeHR.x || posHR.y >= sizeHR.y) {
        return;
    }

    let uv = (vec2f(posHR) + vec2f(0.5)) * params.presentationSize.zw;
    let jitteredUV = uv + params.jitterAndThreshold.xy * params.renderSize.zw;
    let color = textureSampleLevel(lrColor, lrColorSampler, jitteredUV, 0.0);

    // alpha carries the blend weight for debug views: 1 = "current frame only"
    textureStore(historyOut, posHR, vec4f(color.rgb, 1.0));
}
