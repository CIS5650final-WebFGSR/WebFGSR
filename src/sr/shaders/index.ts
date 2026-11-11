// WGSL sources for the compute passes.
//
// WGSL has no preprocessor, so `#include "common.wgsl"` is resolved here at
// module-load time (one level, one shared file).

import commonSource from "./common.wgsl?raw";
import dilateRaw from "./dilate.wgsl?raw";
import upsampleFirstFrameRaw from "./upsampleFirstFrame.wgsl?raw";
import blendHistoryRaw from "./blendHistory.wgsl?raw";
import accumulateRaw from "./accumulate.wgsl?raw";
import encodeRaw from "./encode.wgsl?raw";
import debugVizRaw from "./debugViz.wgsl?raw";

const INCLUDE_COMMON = '#include "common.wgsl"';

function resolveIncludes(source: string, name: string): string {
    if (!source.includes(INCLUDE_COMMON)) {
        throw new Error(`SR shader ${name} does not include common.wgsl`);
    }
    return source.replace(INCLUDE_COMMON, commonSource);
}

/** Low-res depth / motion-vector dilation. Bindings: 0 params, 1 gbuffer, 2 dilatedOut. */
export const dilateWGSL = resolveIncludes(dilateRaw, "dilate");

/** First-frame bilinear upsample. Bindings: 0 params, 1 lrColor, 2 lrColorSampler, 3 historyOut. */
export const upsampleFirstFrameWGSL = resolveIncludes(upsampleFirstFrameRaw, "upsampleFirstFrame");

/**
 * History reprojection + rectification + blend. Bindings: 0 params, 1 lrColor,
 * 2 lrColorSampler, 3 dilatedCur, 4 dilatedPrev, 5 historyPrev, 6 historyOut.
 */
export const blendHistoryWGSL = resolveIncludes(blendHistoryRaw, "blendHistory");

/** Reference accumulation. Bindings: 0 params, 1 sampleTex, 2 accIn, 3 accOut. */
export function accumulateWGSL(firstSample: boolean): string {
    return `const FIRST_SAMPLE : bool = ${firstSample ? "true" : "false"};\n${accumulateRaw}`;
}

/** Float -> rgba8unorm with a vertical flip, for readback. Bindings: 0 src, 1 dst. */
export const encodeWGSL = encodeRaw;

/** Debug visualisation. Bindings: 0 params, 1 src, 2 dst. */
export const debugVizWGSL = debugVizRaw;
