// WebFGSR - temporal super resolution: reproject, rectify and blend history.
//
// WGSL port of Mob-FGSR resources/SuperResolution/BlendHistory.comp
// (https://github.com/Mob-FGSR/MobFGSR, CC BY-NC 4.0).
//
// Per high-res pixel:
//   1. take the dilated motion vector of the low-res texel under the pixel and
//      reproject into last frame's high-res output (4x4 Catmull-Rom filter),
//   2. clamp that history sample to the colour AABB of the 3x3 low-res
//      neighbourhood (rectification, kills ghosting),
//   3. compare this frame's depth with last frame's depth at the reprojected
//      position; a mismatch means a disocclusion -> drop history,
//   4. otherwise blend history with the bilinearly upsampled current frame.
//      The current frame gets more weight the closer the high-res pixel centre
//      is to this frame's jittered low-res sample.
//
// Differences from the reference (also listed in README.md):
//   - the 128x128 weight LUT (resources/lut.exr) is replaced by the analytic
//     Catmull-Rom kernel it tabulates (verified to 1e-4),
//   - history is rgba16float instead of rgba8 (no 8-bit quantisation stall),
//   - the blend weight is written to alpha for debug views,
//   - neighbour coordinates are clamped to size-1.

#include "common.wgsl"

@group(0) @binding(0) var<uniform> params : SRParams;
// jittered low-res colour (display-referred)
@group(0) @binding(1) var lrColor : texture_2d<f32>;
@group(0) @binding(2) var lrColorSampler : sampler; // linear, clamp-to-edge
// rg = dilated motion vector, b = dilated depth (this frame / previous frame)
@group(0) @binding(3) var dilatedCur : texture_2d<f32>;
@group(0) @binding(4) var dilatedPrev : texture_2d<f32>;
// last frame's high-res output
@group(0) @binding(5) var historyPrev : texture_2d<f32>;
@group(0) @binding(6) var historyOut : texture_storage_2d<rgba16float, write>;

// The reference writes these matrices in row-major *source order* but GLSL
// (like WGSL) fills matrices column by column, so what actually runs is the
// transpose of the textbook YCoCg pair. The two are still exact inverses of
// each other, so the round trip is lossless; only the axes of the clamping box
// differ from true YCoCg. Transcribed literally to stay faithful.
fn rgbToYCoCg(color : vec3f) -> vec3f {
    let m = mat3x3f(
        0.25, 0.5, 0.25,
        0.5, 0.0, -0.5,
        -0.25, 0.5, -0.25);
    return m * color;
}

fn yCoCgToRgb(color : vec3f) -> vec3f {
    let m = mat3x3f(
        1.0, 1.0, -1.0,
        1.0, 0.0, 1.0,
        1.0, -1.0, -1.0);
    return m * color;
}

fn catmullRomWeights(t : f32) -> vec4f {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4f(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2);
}

// Equivalent of sampleWithLut() in the reference (resources/Sample.glsl):
// 4x4 taps around uv, weights = separable Catmull-Rom, edges clamped.
fn sampleHistory(uv : vec2f) -> vec3f {
    let sizeHR = vec2i(params.presentationSize.xy);
    let fPos = uv * params.presentationSize.xy - vec2f(0.5);
    let base = floor(fPos);
    let t = fPos - base;
    let ibase = vec2i(base);
    let wx = catmullRomWeights(t.x);
    let wy = catmullRomWeights(t.y);

    var result = vec3f(0.0);
    var weightSum = 0.0;
    for (var j = 0; j < 4; j++) {
        for (var i = 0; i < 4; i++) {
            let p = clamp(ibase + vec2i(i - 1, j - 1), vec2i(0), sizeHR - vec2i(1));
            let w = wx[i] * wy[j];
            result += textureLoad(historyPrev, p, 0).rgb * w;
            weightSum += w;
        }
    }
    return result / weightSum;
}

fn clampHistoryWithAABB(history : vec3f, posLR : vec2i) -> vec3f {
    let sizeLR = vec2i(params.renderSize.xy);
    let center = rgbToYCoCg(textureLoad(lrColor, posLR, 0).rgb);
    var colorMin = center;
    var colorMax = center;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            if (i == 0 && j == 0) {
                continue;
            }
            let p = clamp(posLR + vec2i(i, j), vec2i(0), sizeLR - vec2i(1));
            let c = rgbToYCoCg(textureLoad(lrColor, p, 0).rgb);
            colorMin = min(colorMin, c);
            colorMax = max(colorMax, c);
        }
    }
    return yCoCgToRgb(clamp(rgbToYCoCg(history), colorMin, colorMax));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let sizeHR = vec2i(params.presentationSize.xy);
    let sizeLR = vec2i(params.renderSize.xy);
    let posHR = vec2i(id.xy);
    if (posHR.x >= sizeHR.x || posHR.y >= sizeHR.y) {
        return;
    }

    let uv = (vec2f(posHR) + vec2f(0.5)) * params.presentationSize.zw;
    let posLR = min(vec2i(uv * params.renderSize.xy), sizeLR - vec2i(1));

    // The reference samples the dilated buffers with GL_NEAREST.
    let dilated = textureLoad(dilatedCur, posLR, 0);
    let mv = dilated.xy;
    let currentDepth = dilated.z;
    let prevUV = uv - mv;

    var history = sampleHistory(prevUV);
    if (params.options.x > 0.5) {
        history = clampHistoryWithAABB(history, posLR);
    }

    let prevPosLR = clamp(vec2i(floor(prevUV * params.renderSize.xy)), vec2i(0), sizeLR - vec2i(1));
    let historyDepth = textureLoad(dilatedPrev, prevPosLR, 0).z;

    let jitteredUV = uv + params.jitterAndThreshold.xy * params.renderSize.zw;
    let current = textureSampleLevel(lrColor, lrColorSampler, jitteredUV, 0.0).rgb;

    let depthDiff = abs(currentDepth - historyDepth);
    let onScreen = all(prevUV >= vec2f(0.0)) && all(prevUV <= vec2f(1.0));
    let depthOk = (params.options.y < 0.5) || (depthDiff < params.jitterAndThreshold.z);

    var result = current;
    var weight = 1.0;
    if (onScreen && depthOk) {
        // distance from this high-res pixel centre to the nearest jittered
        // low-res sample, in low-res texels
        let dist = fract(jitteredUV * params.renderSize.xy) - vec2f(0.5);
        weight = clamp((0.25 - 0.4 * dot(dist, dist)) * params.options.z, 0.0, 1.0);
        result = mix(history, current, weight);
    }

    textureStore(historyOut, posHR, vec4f(result, weight));
}
