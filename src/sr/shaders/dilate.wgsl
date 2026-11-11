// WebFGSR - depth / motion-vector dilation (low resolution).
//
// WGSL port of Mob-FGSR resources/Preprocessing/Dilate.comp
// (https://github.com/Mob-FGSR/MobFGSR, CC BY-NC 4.0).
//
// For every low-res texel, find the nearest-to-camera depth in the 3x3
// neighbourhood and take the motion vector of that texel. This makes the
// foreground "win" along silhouettes so that edges reproject with the object
// in front of them.
//
// Differences from the reference:
//   - depth and motion vectors arrive packed in one G-buffer texture
//     (rg = motion vector, b = depth) and leave packed the same way,
//   - neighbour coordinates are clamped to size-1 (the reference clamps to
//     size, which is out of range by one).

#include "common.wgsl"

@group(0) @binding(0) var<uniform> params : SRParams;
// rg = motion vector (uv_cur - uv_prev), b = normalised linear depth
@group(0) @binding(1) var gbuffer : texture_2d<f32>;
// rg = dilated motion vector, b = dilated (nearest) depth
@group(0) @binding(2) var dilatedOut : texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let size = vec2i(params.renderSize.xy);
    let pos = vec2i(id.xy);
    if (pos.x >= size.x || pos.y >= size.y) {
        return;
    }

    var nearestPos = pos;
    var nearestDepth = textureLoad(gbuffer, pos, 0).z;

    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            if (i == 0 && j == 0) {
                continue;
            }
            let samplePos = clamp(pos + vec2i(i, j), vec2i(0), size - vec2i(1));
            let d = textureLoad(gbuffer, samplePos, 0).z;
            if (d < nearestDepth) {
                nearestPos = samplePos;
                nearestDepth = d;
            }
        }
    }

    let mv = textureLoad(gbuffer, nearestPos, 0).xy;
    textureStore(dilatedOut, pos, vec4f(mv, nearestDepth, 0.0));
}
