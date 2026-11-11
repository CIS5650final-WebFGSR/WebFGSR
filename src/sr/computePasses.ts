// Babylon ComputeShader wrappers for every compute pass in the pipeline.
//
// Babylon facts this file relies on (read from the installed 9.35 source):
//  - `engine.computeShader` must be imported for side effects (webgpuEngine
//    does not register it).
//  - ComputeShader needs an explicit bindingsMapping (no WGSL reflection).
//  - setTexture(name, tex, false) + setTextureSampler: bindSampler=true would
//    bind the sampler at binding-1 and rebuild the bind group every dispatch.
//  - Re-setting a different texture each frame (ping-pong) only rebuilds the
//    bind group; the pipeline is created once.
//  - Dispatches are recorded on the engine's single render encoder in call
//    order, so calling them between render-target renders keeps ordering.
//  - gpuTimeInFrame exists only if engine.enableGPUTimingMeasurements was true
//    when the ComputeShader was constructed.

import "@babylonjs/core/Engines/WebGPU/Extensions/engine.computeShader";
import { ComputeShader } from "@babylonjs/core/Compute/computeShader";
import { UniformBuffer } from "@babylonjs/core/Materials/uniformBuffer";
import { RawTexture } from "@babylonjs/core/Materials/Textures/rawTexture";
import { TextureSampler } from "@babylonjs/core/Materials/Textures/textureSampler";
import { Constants } from "@babylonjs/core/Engines/constants";
import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import type { BaseTexture } from "@babylonjs/core/Materials/Textures/baseTexture";

import { accumulateWGSL, blendHistoryWGSL, debugVizWGSL, dilateWGSL, encodeWGSL, upsampleFirstFrameWGSL } from "./shaders";

const WG = 8;

export type StorageKind = "rgba16f" | "rgba32f" | "rgba8";

export function createStorageTexture(engine: AbstractEngine, width: number, height: number, kind: StorageKind, name: string): RawTexture {
    const type =
        kind === "rgba16f" ? Constants.TEXTURETYPE_HALF_FLOAT : kind === "rgba32f" ? Constants.TEXTURETYPE_FLOAT : Constants.TEXTURETYPE_UNSIGNED_BYTE;
    const sampling = kind === "rgba32f" ? Constants.TEXTURE_NEAREST_SAMPLINGMODE : Constants.TEXTURE_BILINEAR_SAMPLINGMODE;
    const tex = new RawTexture(null, width, height, Constants.TEXTUREFORMAT_RGBA, engine, false, false, sampling, type, Constants.TEXTURE_CREATIONFLAG_STORAGE);
    tex.name = name;
    tex.wrapU = Constants.TEXTURE_CLAMP_ADDRESSMODE;
    tex.wrapV = Constants.TEXTURE_CLAMP_ADDRESSMODE;
    return tex;
}

function linearClampSampler(): TextureSampler {
    return new TextureSampler().setParameters(
        Constants.TEXTURE_CLAMP_ADDRESSMODE,
        Constants.TEXTURE_CLAMP_ADDRESSMODE,
        Constants.TEXTURE_CLAMP_ADDRESSMODE,
        1,
        Constants.TEXTURE_BILINEAR_SAMPLINGMODE,
    );
}

function makeCS(engine: AbstractEngine, name: string, source: string, bindings: string[]): ComputeShader {
    const bindingsMapping: Record<string, { group: number; binding: number }> = {};
    bindings.forEach((b, i) => (bindingsMapping[b] = { group: 0, binding: i }));
    const cs = new ComputeShader(name, engine, { computeSource: source }, { bindingsMapping });
    cs.onError = (_effect, errors) => console.error(`[compute:${name}]`, errors);
    return cs;
}

const groups = (n: number) => Math.ceil(n / WG);

/** Parameters of one SR frame. */
export interface SRFrameInput {
    lrColor: BaseTexture;
    gbuffer: BaseTexture;
    /** jitter of the low-res colour, in low-res texels */
    jitter: readonly [number, number];
}

export interface SRSettings {
    /** depth_diff_threshold_sr of the reference (on linear depth / far) */
    depthThreshold: number;
    /** clamp history to the 3x3 YCoCg AABB of the current frame */
    clampHistory: boolean;
    /** reject history when depths disagree */
    depthRejection: boolean;
    /** multiplies the current-frame blend weight (1 = reference) */
    weightScale: number;
}

export const DEFAULT_SR_SETTINGS: SRSettings = {
    depthThreshold: 0.01,
    clampHistory: true,
    depthRejection: true,
    weightScale: 1,
};

/**
 * Mob-FGSR temporal super resolution (Dilate -> UpsampleFirstFrame |
 * BlendHistory) with ping-ponged dilated buffers and history.
 */
export class SuperResolution {
    readonly history: [RawTexture, RawTexture];
    readonly dilated: [RawTexture, RawTexture];
    readonly dilateCS: ComputeShader;
    readonly firstFrameCS: ComputeShader;
    readonly blendCS: ComputeShader;
    private readonly ubo: UniformBuffer;
    private cur = 0;
    private historyValid = false;
    /** true when the last executed frame used the first-frame path */
    lastWasFirstFrame = true;

    constructor(
        private readonly engine: AbstractEngine,
        readonly renderWidth: number,
        readonly renderHeight: number,
        readonly outputWidth: number,
        readonly outputHeight: number,
    ) {
        this.history = [
            createStorageTexture(engine, outputWidth, outputHeight, "rgba16f", "srHistoryA"),
            createStorageTexture(engine, outputWidth, outputHeight, "rgba16f", "srHistoryB"),
        ];
        this.dilated = [
            createStorageTexture(engine, renderWidth, renderHeight, "rgba32f", "srDilatedA"),
            createStorageTexture(engine, renderWidth, renderHeight, "rgba32f", "srDilatedB"),
        ];

        this.ubo = new UniformBuffer(engine, undefined, true, "srParams");
        this.ubo.addUniform("renderSize", 4);
        this.ubo.addUniform("presentationSize", 4);
        this.ubo.addUniform("jitterAndThreshold", 4);
        this.ubo.addUniform("options", 4);
        this.ubo.create();

        const sampler = linearClampSampler();
        this.dilateCS = makeCS(engine, "srDilate", dilateWGSL, ["params", "gbuffer", "dilatedOut"]);
        this.firstFrameCS = makeCS(engine, "srUpsampleFirstFrame", upsampleFirstFrameWGSL, ["params", "lrColor", "lrColorSampler", "historyOut"]);
        this.blendCS = makeCS(engine, "srBlendHistory", blendHistoryWGSL, [
            "params",
            "lrColor",
            "lrColorSampler",
            "dilatedCur",
            "dilatedPrev",
            "historyPrev",
            "historyOut",
        ]);
        for (const cs of [this.dilateCS, this.firstFrameCS, this.blendCS]) {
            cs.setUniformBuffer("params", this.ubo);
        }
        this.firstFrameCS.setTextureSampler("lrColorSampler", sampler);
        this.blendCS.setTextureSampler("lrColorSampler", sampler);
    }

    /** Current SR output (rgb = colour, a = weight given to the current frame). */
    get output(): RawTexture {
        return this.history[this.cur];
    }

    /** Dilated motion/depth of the current frame (rg = mv, b = depth). */
    get currentDilated(): RawTexture {
        return this.dilated[this.cur];
    }

    get hasHistory(): boolean {
        return this.historyValid;
    }

    isReady(): boolean {
        return this.dilateCS.isReady() && this.firstFrameCS.isReady() && this.blendCS.isReady();
    }

    /** Drop history: the next frame takes the first-frame path. */
    reset(): void {
        this.historyValid = false;
    }

    /** Records the SR passes for one rendered frame. Returns false if shaders are not ready yet. */
    execute(input: SRFrameInput, settings: SRSettings): boolean {
        if (!this.isReady()) {
            return false;
        }
        const next = this.cur ^ 1;
        const prev = this.cur;

        this.ubo.updateFloat4("renderSize", this.renderWidth, this.renderHeight, 1 / this.renderWidth, 1 / this.renderHeight);
        this.ubo.updateFloat4("presentationSize", this.outputWidth, this.outputHeight, 1 / this.outputWidth, 1 / this.outputHeight);
        this.ubo.updateFloat4("jitterAndThreshold", input.jitter[0], input.jitter[1], settings.depthThreshold, 0);
        this.ubo.updateFloat4("options", settings.clampHistory ? 1 : 0, settings.depthRejection ? 1 : 0, settings.weightScale, 0);
        this.ubo.update();

        this.dilateCS.setTexture("gbuffer", input.gbuffer, false);
        this.dilateCS.setStorageTexture("dilatedOut", this.dilated[next]);
        if (!this.dilateCS.dispatch(groups(this.renderWidth), groups(this.renderHeight), 1)) {
            return false;
        }

        let ok: boolean;
        if (!this.historyValid) {
            this.firstFrameCS.setTexture("lrColor", input.lrColor, false);
            this.firstFrameCS.setStorageTexture("historyOut", this.history[next]);
            ok = this.firstFrameCS.dispatch(groups(this.outputWidth), groups(this.outputHeight), 1);
            this.lastWasFirstFrame = true;
        } else {
            this.blendCS.setTexture("lrColor", input.lrColor, false);
            this.blendCS.setTexture("dilatedCur", this.dilated[next], false);
            this.blendCS.setTexture("dilatedPrev", this.dilated[prev], false);
            this.blendCS.setTexture("historyPrev", this.history[prev], false);
            this.blendCS.setStorageTexture("historyOut", this.history[next]);
            ok = this.blendCS.dispatch(groups(this.outputWidth), groups(this.outputHeight), 1);
            this.lastWasFirstFrame = false;
        }
        this.cur = next;
        this.historyValid = ok;
        return ok;
    }
}

/** Spatial-only baseline: bilinear upsample of an un-jittered low-res render. */
export class BilinearUpscale {
    readonly output: RawTexture;
    readonly cs: ComputeShader;
    private readonly ubo: UniformBuffer;

    constructor(engine: AbstractEngine, renderWidth: number, renderHeight: number, outputWidth: number, outputHeight: number) {
        this.output = createStorageTexture(engine, outputWidth, outputHeight, "rgba16f", "bilinearOut");
        this.ubo = new UniformBuffer(engine, undefined, false, "bilinearParams");
        this.ubo.addUniform("renderSize", 4);
        this.ubo.addUniform("presentationSize", 4);
        this.ubo.addUniform("jitterAndThreshold", 4);
        this.ubo.addUniform("options", 4);
        this.ubo.create();
        this.ubo.updateFloat4("renderSize", renderWidth, renderHeight, 1 / renderWidth, 1 / renderHeight);
        this.ubo.updateFloat4("presentationSize", outputWidth, outputHeight, 1 / outputWidth, 1 / outputHeight);
        this.ubo.updateFloat4("jitterAndThreshold", 0, 0, 0, 0);
        this.ubo.updateFloat4("options", 0, 0, 1, 0);
        this.ubo.update();
        this.cs = makeCS(engine, "bilinearUpscale", upsampleFirstFrameWGSL, ["params", "lrColor", "lrColorSampler", "historyOut"]);
        this.cs.setUniformBuffer("params", this.ubo);
        this.cs.setTextureSampler("lrColorSampler", linearClampSampler());
        this.cs.setStorageTexture("historyOut", this.output);
    }

    isReady(): boolean {
        return this.cs.isReady();
    }

    execute(lrPlain: BaseTexture): boolean {
        this.cs.setTexture("lrColor", lrPlain, false);
        const { width, height } = this.output.getSize();
        return this.cs.dispatch(groups(width), groups(height), 1);
    }
}

/** N-sample supersampled reference, accumulated one jittered render at a time. */
export class ReferenceAccumulator {
    private readonly acc: [RawTexture, RawTexture];
    private readonly firstCS: ComputeShader;
    private readonly restCS: ComputeShader;
    private readonly ubo: UniformBuffer;
    private cur = 0;
    private count = 0;

    constructor(
        engine: AbstractEngine,
        readonly width: number,
        readonly height: number,
        readonly sampleCount: number,
    ) {
        this.acc = [createStorageTexture(engine, width, height, "rgba32f", "refAccA"), createStorageTexture(engine, width, height, "rgba32f", "refAccB")];
        this.ubo = new UniformBuffer(engine, undefined, false, "accParams");
        this.ubo.addUniform("weight", 4);
        this.ubo.create();
        this.ubo.updateFloat4("weight", 1 / sampleCount, 0, 0, 0);
        this.ubo.update();
        const bindings = ["params", "sampleTex", "accIn", "accOut"];
        this.firstCS = makeCS(engine, "refAccumulateFirst", accumulateWGSL(true), bindings);
        this.restCS = makeCS(engine, "refAccumulate", accumulateWGSL(false), bindings);
        this.firstCS.setUniformBuffer("params", this.ubo);
        this.restCS.setUniformBuffer("params", this.ubo);
    }

    isReady(): boolean {
        return this.firstCS.isReady() && this.restCS.isReady();
    }

    /** The accumulated reference (valid once sampleCount samples were added). */
    get output(): RawTexture {
        return this.acc[this.cur];
    }

    get samplesAdded(): number {
        return this.count;
    }

    begin(): void {
        this.count = 0;
    }

    add(sample: BaseTexture): boolean {
        const next = this.cur ^ 1;
        const cs = this.count === 0 ? this.firstCS : this.restCS;
        cs.setTexture("sampleTex", sample, false);
        cs.setTexture("accIn", this.acc[this.cur], false);
        cs.setStorageTexture("accOut", this.acc[next]);
        const ok = cs.dispatch(groups(this.width), groups(this.height), 1);
        if (ok) {
            this.cur = next;
            this.count++;
        }
        return ok;
    }
}

/** Converts float colour textures to top-row-first rgba8 for PNG readback. */
export class Encoder {
    readonly cs: ComputeShader;
    private readonly targets = new Map<string, RawTexture>();

    constructor(private readonly engine: AbstractEngine) {
        this.cs = makeCS(engine, "encodeRGBA8", encodeWGSL, ["src", "dst"]);
    }

    isReady(): boolean {
        return this.cs.isReady();
    }

    /** Records the conversion of `src` into the rgba8 target registered under `key`. */
    encode(key: string, src: BaseTexture): RawTexture {
        const { width, height } = src.getSize();
        let dst = this.targets.get(key);
        if (!dst || dst.getSize().width !== width || dst.getSize().height !== height) {
            dst?.dispose();
            dst = createStorageTexture(this.engine, width, height, "rgba8", `encoded_${key}`);
            this.targets.set(key, dst);
        }
        this.cs.setTexture("src", src, false);
        this.cs.setStorageTexture("dst", dst);
        this.cs.dispatch(groups(width), groups(height), 1);
        return dst;
    }
}

export const enum VizMode {
    MotionVectors = 0,
    Depth = 1,
    BlendWeight = 2,
    Copy = 3,
    /** quantitative export of the G-buffer (see debugViz.wgsl mode 4) */
    RawGBuffer = 4,
}

/**
 * Debug visualisations rendered into a presentation-size texture.
 * Each parameter set gets its own constant uniform buffer: a dynamic buffer
 * written twice in one frame would hand the last value to every dispatch
 * (writeBuffer runs before the frame's command buffer).
 */
export class DebugViz {
    readonly output: RawTexture;
    readonly cs: ComputeShader;
    private readonly ubos = new Map<string, UniformBuffer>();

    constructor(
        private readonly engine: AbstractEngine,
        width: number,
        height: number,
    ) {
        this.output = createStorageTexture(engine, width, height, "rgba16f", "debugViz");
        this.cs = makeCS(engine, "debugViz", debugVizWGSL, ["params", "src", "dst"]);
        this.cs.setUniformBuffer("params", this.uboFor(VizMode.Copy, 8, 0.35));
    }

    private uboFor(mode: VizMode, mvSaturationPixels: number, depthRange: number): UniformBuffer {
        const key = `${mode}/${mvSaturationPixels}/${depthRange}`;
        let ubo = this.ubos.get(key);
        if (!ubo) {
            ubo = new UniformBuffer(this.engine, undefined, false, `vizParams_${key}`);
            ubo.addUniform("modeAndScale", 4);
            ubo.create();
            ubo.updateFloat4("modeAndScale", mode, mvSaturationPixels, depthRange, 0);
            ubo.update();
            this.ubos.set(key, ubo);
        }
        return ubo;
    }

    isReady(): boolean {
        return this.cs.isReady();
    }

    /** Records one visualisation into `target` (defaults to this.output). */
    execute(src: BaseTexture, mode: VizMode, mvSaturationPixels = 8, depthRange = 0.35, target: RawTexture = this.output): boolean {
        this.cs.setUniformBuffer("params", this.uboFor(mode, mvSaturationPixels, depthRange));
        this.cs.setTexture("src", src, false);
        this.cs.setStorageTexture("dst", target);
        const { width, height } = target.getSize();
        return this.cs.dispatch(groups(width), groups(height), 1);
    }
}
