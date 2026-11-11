// WebFGSR frame pipeline.
//
// Per frame (inside scene.render()):
//   onBeforeRender            sync camera poses, build un-jittered and jittered
//                             projections, set G-buffer matrices, choose the
//                             render targets of this frame
//   customRenderTargets       low-res colour (jittered), low-res G-buffer
//                             (jittered raster, un-jittered motion), optional
//                             un-jittered low-res colour (bilinear baseline)
//                             and full-res native colour
//   onAfterRenderTargets      compute: SR (dilate + first-frame | blend),
//                             bilinear baseline, N-sample reference (manual
//                             re-renders of a full-res target), debug views,
//                             rgba8 encoding for readback
//   main pass                 renders no meshes (display camera layerMask 0)
//   onAfterRender             present to the canvas, roll previous matrices
//
// Camera/matrix facts this relies on (Babylon 9.35 source):
//  - ObjectRenderer.initRender calls camera.getProjectionMatrix(true) for
//    every render target; only a frozen projection survives that, so each
//    camera is frozen to an app-owned Matrix that is rewritten per frame
//    (markAsUpdated() so the scene UBO notices).
//  - Babylon matrices are row-vector: proj * Translation(jx, jy, 0) shifts NDC
//    by (jx, jy) for both handednesses.
//  - Render targets are drawn Y-flipped on WebGPU (texel row 0 = bottom), so
//    NDC y and texel y point the same way: no sign flips anywhere.

import { FreeCamera } from "@babylonjs/core/Cameras/freeCamera";
import { TargetCamera } from "@babylonjs/core/Cameras/targetCamera";
import { RenderTargetTexture } from "@babylonjs/core/Materials/Textures/renderTargetTexture";
import { Constants } from "@babylonjs/core/Engines/constants";
import { Color4 } from "@babylonjs/core/Maths/math.color";
import { Matrix, Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";
import type { Scene } from "@babylonjs/core/scene";
import type { BaseTexture } from "@babylonjs/core/Materials/Textures/baseTexture";
import type { RawTexture } from "@babylonjs/core/Materials/Textures/rawTexture";
import type { Camera } from "@babylonjs/core/Cameras/camera";

import { GBufferMaterials } from "./gbuffer";
import { MipBiasPlugin, installMipBias } from "./mipBias";
import { Presenter } from "./presenter";
import { getJitter, supersampleOffset, type JitterSequenceName } from "./jitter";
import {
    BilinearUpscale,
    DebugViz,
    DEFAULT_SR_SETTINGS,
    Encoder,
    ReferenceAccumulator,
    SuperResolution,
    VizMode,
    createStorageTexture,
    type SRSettings,
} from "../sr/computePasses";
import { SKY_COLOR, type SponzaContent } from "../scene/sponzaScene";

export interface PipelineOptions {
    outputWidth: number;
    outputHeight: number;
    /** output / render size per axis (Mob-FGSR: 2) */
    renderScale: number;
    /** vertical field of view, radians */
    fov: number;
    near: number;
    far: number;
    /** reference uses referenceGrid x referenceGrid sub-pixel samples */
    referenceGrid: number;
}

export const DEFAULT_PIPELINE_OPTIONS: PipelineOptions = {
    outputWidth: 1920,
    outputHeight: 1080,
    renderScale: 2,
    fov: 1.0,
    near: 0.05,
    far: 100,
    referenceGrid: 4,
};

export interface PipelineSettings extends SRSettings {
    jitterSequence: JitterSequenceName;
    /** -1 deliberately flips the jitter handed to the SR shaders (validation experiment) */
    jitterSign: number;
    /** texture LOD bias of the jittered low-res pass (log2(1/renderScale) = -1 at 2x; 0 = off) */
    mipBias: number;
}

export const DEFAULT_PIPELINE_SETTINGS: PipelineSettings = {
    ...DEFAULT_SR_SETTINGS,
    jitterSequence: "mobfgsr",
    jitterSign: 1,
    mipBias: -1,
};

/** Images the pipeline can produce. */
export type SeriesKey =
    | "sr" // Mob-FGSR output (full res)
    | "bilinear" // un-jittered low res, bilinear upsample (full res)
    | "native" // full-res render, 1 sample per pixel
    | "reference" // full-res, referenceGrid^2 samples per pixel
    | "lr" // jittered low-res input (low res; nearest-upscaled when presented)
    | "motion" // motion-vector visualisation (full res)
    | "depth" // depth visualisation (full res)
    | "weight" // SR current-frame weight heat map (full res)
    | "mvraw"; // low-res G-buffer as RGBA8: rg = 0.5 + 25 * motion, b = 2 * depth (validation)

export interface PresentRequest {
    /** main image (right side in split mode) */
    a: SeriesKey;
    /** comparison image (left side) */
    b?: SeriesKey;
    split?: number;
}

export interface FrameRequest {
    sr: boolean;
    bilinear: boolean;
    native: boolean;
    reference: boolean;
    present: PresentRequest | null;
    encode: SeriesKey[];
}

export function seriesNeeds(keys: SeriesKey[]): Pick<FrameRequest, "sr" | "bilinear" | "native" | "reference"> {
    const s = new Set(keys);
    return {
        sr: s.has("sr") || s.has("lr") || s.has("motion") || s.has("depth") || s.has("weight") || s.has("mvraw"),
        bilinear: s.has("bilinear"),
        native: s.has("native"),
        reference: s.has("reference"),
    };
}

type PerfCounterLike = { counter: { lastSecAverage: number; current: number; total: number; count: number } } | undefined;

export class Pipeline {
    readonly renderWidth: number;
    readonly renderHeight: number;
    readonly displayCamera: FreeCamera;
    settings: PipelineSettings = { ...DEFAULT_PIPELINE_SETTINGS };
    request: FrameRequest = { sr: true, bilinear: false, native: false, reference: false, present: { a: "sr" }, encode: [] };

    // cameras: identical pose, different frozen projections
    private readonly camLR: TargetCamera; // low res, jittered
    private readonly camPlain: TargetCamera; // un-jittered (low-res baseline and native)
    private readonly camRef: TargetCamera; // reference sub-pixel offsets

    // matrices (app-owned; cameras hold references to the frozen ones)
    private readonly proj = new Matrix();
    private readonly projLR = new Matrix();
    private readonly projPlain = new Matrix();
    private readonly projRef = new Matrix();
    private readonly vpCurrent = new Matrix();
    private readonly vpPrevious = new Matrix();
    private readonly tmp = new Matrix();
    private hasPrevious = false;

    // render targets
    readonly lrColor: RenderTargetTexture;
    readonly gbuffer: RenderTargetTexture;
    readonly lrPlain: RenderTargetTexture;
    readonly hrNative: RenderTargetTexture;
    readonly hrRef: RenderTargetTexture;

    // compute
    readonly sr: SuperResolution;
    readonly bilinear: BilinearUpscale;
    readonly reference: ReferenceAccumulator;
    readonly encoder: Encoder;
    readonly viz: DebugViz;
    private readonly vizTargets = new Map<SeriesKey, RawTexture>();
    readonly presenter: Presenter;
    readonly gbufferMaterials: GBufferMaterials;
    /** number of PBR materials patched with the LOD-bias plugin */
    readonly mipBiasedMaterials: number;

    /** jitter (low-res texels) used by the last rendered low-res frame */
    currentJitter: readonly [number, number] = [0, 0];
    /** frames rendered through the SR path since start (drives the jitter sequence) */
    private jitterFrame = 0;
    private lastComputeFrameId = -1;
    /** rgba8 readback targets recorded this frame */
    private encoded = new Map<SeriesKey, RawTexture>();
    frameCounter = 0;

    constructor(
        readonly engine: WebGPUEngine,
        readonly content: SponzaContent,
        readonly options: PipelineOptions = DEFAULT_PIPELINE_OPTIONS,
    ) {
        const scene = content.scene;
        const { outputWidth: W, outputHeight: H, renderScale } = options;
        this.renderWidth = Math.round(W / renderScale);
        this.renderHeight = Math.round(H / renderScale);
        const w = this.renderWidth;
        const h = this.renderHeight;

        // Display camera: drives inputs/pose, draws nothing itself.
        this.displayCamera = new FreeCamera("display", new Vector3(0, 1.7, 0), scene);
        this.displayCamera.layerMask = 0;
        this.displayCamera.minZ = options.near;
        this.displayCamera.maxZ = options.far;
        this.displayCamera.fov = options.fov;
        this.displayCamera.speed = 0.08;
        this.displayCamera.angularSensibility = 3000;
        scene.activeCamera = this.displayCamera;

        const makeCam = (name: string, frozen: Matrix) => {
            const cam = new TargetCamera(name, new Vector3(0, 1.7, 0), scene, false);
            cam.minZ = options.near;
            cam.maxZ = options.far;
            cam.fov = options.fov;
            cam.freezeProjectionMatrix(frozen);
            return cam;
        };
        this.buildProjection(this.proj);
        this.projLR.copyFrom(this.proj);
        this.projPlain.copyFrom(this.proj);
        this.projRef.copyFrom(this.proj);
        this.camLR = makeCam("camLR", this.projLR);
        this.camPlain = makeCam("camPlain", this.projPlain);
        this.camRef = makeCam("camRef", this.projRef);

        const meshes = content.meshes;
        const makeRTT = (name: string, width: number, height: number, type: number, camera: Camera, clear: Color4) => {
            const rtt = new RenderTargetTexture(name, { width, height }, scene, {
                generateMipMaps: false,
                type,
                format: Constants.TEXTUREFORMAT_RGBA,
                samplingMode: type === Constants.TEXTURETYPE_FLOAT ? Constants.TEXTURE_NEAREST_SAMPLINGMODE : Constants.TEXTURE_BILINEAR_SAMPLINGMODE,
                generateDepthBuffer: true,
                generateStencilBuffer: false,
                samples: 1,
                // colour targets only; the G-buffer material ignores lights
                enableClusteredLights: content.clusteredLights > 0 && type !== Constants.TEXTURETYPE_FLOAT,
            });
            rtt.activeCamera = camera;
            rtt.renderList = [...meshes];
            rtt.clearColor = clear;
            rtt.renderParticles = false;
            rtt.renderSprites = false;
            rtt.ignoreCameraViewport = true;
            rtt.wrapU = Constants.TEXTURE_CLAMP_ADDRESSMODE;
            rtt.wrapV = Constants.TEXTURE_CLAMP_ADDRESSMODE;
            return rtt;
        };
        const HALF = Constants.TEXTURETYPE_HALF_FLOAT;
        this.lrColor = makeRTT("lrColor", w, h, HALF, this.camLR, SKY_COLOR);
        this.gbuffer = makeRTT("lrGBuffer", w, h, Constants.TEXTURETYPE_FLOAT, this.camLR, new Color4(0, 0, 1, 1));
        this.lrPlain = makeRTT("lrPlain", w, h, HALF, this.camPlain, SKY_COLOR);
        this.hrNative = makeRTT("hrNative", W, H, HALF, this.camPlain, SKY_COLOR);
        this.hrRef = makeRTT("hrRef", W, H, HALF, this.camRef, SKY_COLOR);

        // Per-pass texture LOD bias: only the jittered low-res pass (SR input) is biased.
        this.mipBiasedMaterials = installMipBias(scene.materials);
        const setBias = (rtt: RenderTargetTexture, bias: () => number) =>
            rtt.onBeforeRenderObservable.add(() => {
                MipBiasPlugin.currentBias = bias();
                scene.resetCachedMaterial();
            });
        setBias(this.lrColor, () => this.settings.mipBias);
        setBias(this.lrPlain, () => 0);
        setBias(this.hrNative, () => 0);
        setBias(this.hrRef, () => 0);

        this.gbufferMaterials = new GBufferMaterials(scene, options.far);
        for (const mesh of meshes) {
            this.gbuffer.setMaterialForRendering(mesh, this.gbufferMaterials.forMesh(mesh, mesh === content.mover));
        }

        this.sr = new SuperResolution(engine, w, h, W, H);
        this.bilinear = new BilinearUpscale(engine, w, h, W, H);
        this.reference = new ReferenceAccumulator(engine, W, H, options.referenceGrid * options.referenceGrid);
        this.encoder = new Encoder(engine);
        this.viz = new DebugViz(engine, W, H);
        this.presenter = new Presenter(engine);

        scene.onBeforeRenderObservable.add(() => this.beforeRender());
        scene.onAfterRenderTargetsRenderObservable.add(() => this.afterRenderTargets());
        scene.onAfterRenderObservable.add(() => this.afterRender());
    }

    private buildProjection(out: Matrix): void {
        const o = this.options;
        Matrix.PerspectiveFovLHToRef(
            o.fov,
            o.outputWidth / o.outputHeight,
            o.near,
            o.far,
            out,
            true,
            this.engine.isNDCHalfZRange,
            0,
            this.engine.useReverseDepthBuffer,
        );
        out.markAsUpdated();
    }

    /** dst = proj shifted by (dx, dy) texels of a width x height target */
    private jitteredProjection(dst: Matrix, dxTexels: number, dyTexels: number, width: number, height: number): void {
        Matrix.TranslationToRef((2 * dxTexels) / width, (2 * dyTexels) / height, 0, this.tmp);
        this.proj.multiplyToRef(this.tmp, dst);
        dst.markAsUpdated();
    }

    /** Places the display camera (and therefore every render camera). */
    setPose(eye: Vector3, target: Vector3): void {
        this.displayCamera.position.copyFrom(eye);
        this.displayCamera.setTarget(target);
    }

    /** Forget temporal state (camera cut, settings change). */
    resetHistory(): void {
        this.sr.reset();
        this.hasPrevious = false;
    }

    isReady(): boolean {
        return (
            this.sr.isReady() &&
            this.bilinear.isReady() &&
            this.reference.isReady() &&
            this.encoder.isReady() &&
            this.viz.isReady() &&
            this.presenter.isReady() &&
            this.lrColor.isReadyForRendering() &&
            this.gbuffer.isReadyForRendering() &&
            this.lrPlain.isReadyForRendering() &&
            this.hrNative.isReadyForRendering() &&
            this.hrRef.isReadyForRendering()
        );
    }

    private syncCameras(): void {
        const d = this.displayCamera;
        for (const cam of [this.camLR, this.camPlain, this.camRef]) {
            cam.position.copyFrom(d.position);
            if (d.rotationQuaternion) {
                cam.rotationQuaternion = d.rotationQuaternion.clone();
            } else {
                cam.rotation.copyFrom(d.rotation);
            }
            cam.getViewMatrix(true);
        }
    }

    private beforeRender(): void {
        const req = this.request;
        this.syncCameras();

        // projections: un-jittered for motion vectors / baselines, jittered for SR input
        const jitter = req.sr ? getJitter(this.settings.jitterSequence, this.jitterFrame) : ([0, 0] as const);
        this.currentJitter = jitter;
        this.jitteredProjection(this.projLR, jitter[0], jitter[1], this.renderWidth, this.renderHeight);
        this.projPlain.copyFrom(this.proj);
        this.projPlain.markAsUpdated();

        const view = this.camLR.getViewMatrix();
        view.multiplyToRef(this.proj, this.vpCurrent);
        this.vpCurrent.markAsUpdated();
        if (!this.hasPrevious) {
            this.vpPrevious.copyFrom(this.vpCurrent);
            this.vpPrevious.markAsUpdated();
        }
        this.gbufferMaterials.setViewProjections(this.vpCurrent, this.vpPrevious);
        this.gbufferMaterials.setMipBias(this.settings.mipBias);

        const targets = this.content.scene.customRenderTargets;
        targets.length = 0;
        if (req.sr) {
            targets.push(this.lrColor, this.gbuffer);
        }
        if (req.bilinear) {
            targets.push(this.lrPlain);
        }
        if (req.native) {
            targets.push(this.hrNative);
        }
    }

    /** Renders `rtt` right now (mirrors Scene._renderRenderTarget). */
    private renderTargetNow(rtt: RenderTargetTexture, camera: Camera): void {
        const scene = this.content.scene as Scene & { _intermediateRendering: boolean };
        const previousCamera = scene.activeCamera;
        scene._intermediateRendering = true;
        scene.incrementRenderId();
        scene.activeCamera = camera;
        this.engine.setViewport(camera.viewport);
        scene.updateTransformMatrix();
        rtt.render(false, false);
        scene._intermediateRendering = false;
        scene.activeCamera = previousCamera;
    }

    private vizTarget(key: SeriesKey): RawTexture {
        let t = this.vizTargets.get(key);
        if (!t) {
            const lowRes = key === "mvraw";
            const w = lowRes ? this.renderWidth : this.options.outputWidth;
            const h = lowRes ? this.renderHeight : this.options.outputHeight;
            t = createStorageTexture(this.engine, w, h, "rgba16f", `viz_${key}`);
            this.vizTargets.set(key, t);
        }
        return t;
    }

    /** Texture holding `key` this frame (records a visualisation pass if needed). */
    private textureFor(key: SeriesKey, forPresent: boolean): BaseTexture {
        switch (key) {
            case "sr":
                return this.sr.output;
            case "bilinear":
                return this.bilinear.output;
            case "native":
                return this.hrNative;
            case "reference":
                return this.reference.output;
            case "lr":
                if (!forPresent) {
                    return this.lrColor;
                }
                this.viz.execute(this.lrColor, VizMode.Copy, 8, 0.35, this.vizTarget("lr"));
                return this.vizTarget("lr");
            case "motion":
                this.viz.execute(this.gbuffer, VizMode.MotionVectors, 3, 0.35, this.vizTarget("motion"));
                return this.vizTarget("motion");
            case "depth":
                this.viz.execute(this.gbuffer, VizMode.Depth, 3, 0.3, this.vizTarget("depth"));
                return this.vizTarget("depth");
            case "weight":
                this.viz.execute(this.sr.output, VizMode.BlendWeight, 8, 0.35, this.vizTarget("weight"));
                return this.vizTarget("weight");
            case "mvraw":
                this.viz.execute(this.gbuffer, VizMode.RawGBuffer, 8, 0.35, this.vizTarget("mvraw"));
                return this.vizTarget("mvraw");
        }
    }

    private afterRenderTargets(): void {
        // fires twice per frame (also inside the camera render): run once
        const frameId = this.content.scene.getFrameId();
        if (frameId === this.lastComputeFrameId) {
            return;
        }
        this.lastComputeFrameId = frameId;
        const req = this.request;

        if (req.sr) {
            const sign = this.settings.jitterSign;
            const ok = this.sr.execute(
                { lrColor: this.lrColor, gbuffer: this.gbuffer, jitter: [sign * this.currentJitter[0], sign * this.currentJitter[1]] },
                this.settings,
            );
            if (ok) {
                this.jitterFrame++;
            }
        } else {
            this.sr.reset();
        }
        if (req.bilinear) {
            this.bilinear.execute(this.lrPlain);
        }
        if (req.reference) {
            const grid = this.options.referenceGrid;
            const { outputWidth: W, outputHeight: H } = this.options;
            this.reference.begin();
            for (let s = 0; s < grid * grid; s++) {
                const [ox, oy] = supersampleOffset(s, grid);
                this.jitteredProjection(this.projRef, ox, oy, W, H);
                this.renderTargetNow(this.hrRef, this.camRef);
                this.reference.add(this.hrRef);
            }
        }

        this.encoded.clear();
        for (const key of req.encode) {
            this.encoded.set(key, this.encoder.encode(key, this.textureFor(key, false)));
        }

        const p = req.present;
        if (p) {
            this.presenter.texA = this.textureFor(p.a, true);
            this.presenter.texB = p.b ? this.textureFor(p.b, true) : null;
            this.presenter.splitEnabled = !!p.b;
            this.presenter.splitPosition = p.split ?? 0.5;
        }
    }

    private afterRender(): void {
        if (this.request.present) {
            this.presenter.render();
        }
        this.vpPrevious.copyFrom(this.vpCurrent);
        this.vpPrevious.markAsUpdated();
        this.hasPrevious = true;
        this.frameCounter++;
    }

    /** Reads back an image recorded via request.encode in the frame that just ended (top row first, RGBA8). */
    async readEncoded(key: SeriesKey): Promise<{ width: number; height: number; data: Uint8Array }> {
        const tex = this.encoded.get(key);
        if (!tex) {
            throw new Error(`series ${key} was not encoded this frame`);
        }
        const { width, height } = tex.getSize();
        const view = await tex.readPixels(0, 0, null, true, false, 0, 0, width, height);
        if (!view) {
            throw new Error(`readPixels failed for ${key}`);
        }
        const bytes = new Uint8Array(view.buffer, view.byteOffset, width * height * 4);
        return { width, height, data: new Uint8Array(bytes) };
    }

    /** GPU time per pass in ms (Babylon timestamp queries; last-second averages). */
    gpuTimesMs(): Record<string, number> {
        const ms = (c: PerfCounterLike) => (c ? c.counter.lastSecAverage / 1e6 : NaN);
        const rt = (t: RenderTargetTexture) => (t.renderTarget as unknown as { gpuTimeInFrame?: PerfCounterLike })?.gpuTimeInFrame;
        return {
            lrColor: ms(rt(this.lrColor)),
            gbuffer: ms(rt(this.gbuffer)),
            dilate: ms(this.sr.dilateCS.gpuTimeInFrame as PerfCounterLike),
            blend: ms(this.sr.blendCS.gpuTimeInFrame as PerfCounterLike),
            lrPlain: ms(rt(this.lrPlain)),
            bilinearUp: ms(this.bilinear.cs.gpuTimeInFrame as PerfCounterLike),
            native: ms(rt(this.hrNative)),
            mainPass: ms(this.engine.gpuTimeInFrameForMainPass as PerfCounterLike),
        };
    }

    /** Raw counters (total ns, frame count) for windowed averages in benchmarks. */
    gpuCounters(): Record<string, { total: number; count: number }> {
        const raw = (c: PerfCounterLike) => (c ? { total: c.counter.total, count: c.counter.count } : { total: 0, count: 0 });
        const rt = (t: RenderTargetTexture) => (t.renderTarget as unknown as { gpuTimeInFrame?: PerfCounterLike })?.gpuTimeInFrame;
        return {
            lrColor: raw(rt(this.lrColor)),
            gbuffer: raw(rt(this.gbuffer)),
            dilate: raw(this.sr.dilateCS.gpuTimeInFrame as PerfCounterLike),
            blend: raw(this.sr.blendCS.gpuTimeInFrame as PerfCounterLike),
            lrPlain: raw(rt(this.lrPlain)),
            bilinearUp: raw(this.bilinear.cs.gpuTimeInFrame as PerfCounterLike),
            native: raw(rt(this.hrNative)),
            mainPass: raw(this.engine.gpuTimeInFrameForMainPass as PerfCounterLike),
        };
    }
}
