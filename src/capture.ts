// Deterministic offline capture and GPU benchmarking, driven from Playwright
// through window.webfgsr (see tools/capture.mjs).
//
// Frames are stepped manually (beginFrame / scene.render / endFrame) at fixed
// simulation times, read back as RGBA8 and POSTed as PNG to the dev server
// (vite.config.ts writes them under results/).

import type { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";
import type { Pipeline, SeriesKey } from "./render/pipeline";
import { seriesNeeds } from "./render/pipeline";
import type { CameraPath } from "./scene/cameraPath";

export interface CaptureOptions {
    /** results/<run>/... */
    run: string;
    series: SeriesKey[];
    frames: number;
    fps: number;
    /** simulation time of captured frame 0 */
    t0: number;
    /** SR frames rendered before frame 0 so the history is converged */
    warmupFrames: number;
    /** only save every n-th frame (all frames are still rendered) */
    saveEvery?: number;
    /** folder name prefix inside frames/ (default: none) */
    prefix?: string;
}

export interface BenchmarkOptions {
    label: string;
    series: SeriesKey[];
    frames: number;
    warmupFrames: number;
    fps: number;
    t0: number;
}

async function postFile(path: string, body: Blob | string): Promise<void> {
    const res = await fetch(`/__save?path=${encodeURIComponent(path)}`, { method: "POST", body });
    if (!res.ok) {
        throw new Error(`save ${path} failed: ${res.status} ${await res.text()}`);
    }
}

async function encodePng(width: number, height: number, rgba: Uint8Array): Promise<Blob> {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
        throw new Error("2d context unavailable");
    }
    const pixels = new Uint8ClampedArray(rgba.buffer as ArrayBuffer, rgba.byteOffset, width * height * 4);
    ctx.putImageData(new ImageData(pixels, width, height), 0, 0);
    return canvas.convertToBlob({ type: "image/png" });
}

export class CaptureController {
    constructor(
        private readonly engine: WebGPUEngine,
        private readonly pipeline: Pipeline,
        private readonly path: CameraPath,
        private readonly setSimulationTime: (t: number) => void,
    ) {}

    /** One deterministic frame at simulation time t. */
    stepFrame(t: number): void {
        this.setSimulationTime(t);
        this.engine.beginFrame();
        this.pipeline.content.scene.render();
        this.engine.endFrame();
    }

    private async gpuIdle(): Promise<void> {
        const device = (this.engine as unknown as { _device: GPUDevice })._device;
        await device.queue.onSubmittedWorkDone();
    }

    /** Renders frames until every shader/material/target is ready. */
    async warmUp(maxFrames = 600): Promise<number> {
        const p = this.pipeline;
        p.request = { sr: true, bilinear: true, native: true, reference: false, present: null, encode: [] };
        for (let i = 0; i < maxFrames; i++) {
            this.stepFrame(0);
            await this.gpuIdle();
            if (i >= 3 && p.isReady()) {
                // a few extra frames so the RENDER_ONCE shadow map and async pipelines settle
                for (let k = 0; k < 4; k++) {
                    this.stepFrame(0);
                }
                await this.gpuIdle();
                p.resetHistory();
                return i;
            }
            await new Promise((r) => setTimeout(r, 15));
        }
        throw new Error("pipeline never became ready");
    }

    async capture(o: CaptureOptions): Promise<Record<string, unknown>> {
        const p = this.pipeline;
        const needs = seriesNeeds(o.series);
        p.resetHistory();
        // warm-up: SR only, camera held at the path start (path clamps t < 0), object keeps moving
        p.request = { sr: true, bilinear: false, native: false, reference: false, present: null, encode: [] };
        for (let i = o.warmupFrames; i > 0; i--) {
            this.stepFrame(o.t0 - i / o.fps);
        }
        await this.gpuIdle();

        const saveEvery = Math.max(1, o.saveEvery ?? 1);
        const pending: Promise<void>[] = [];
        const started = performance.now();
        let saved = 0;
        for (let f = 0; f < o.frames; f++) {
            const t = o.t0 + f / o.fps;
            const save = f % saveEvery === 0;
            p.request = { ...needs, sr: true, present: null, encode: save ? o.series : [] };
            this.stepFrame(t);
            if (!save) {
                continue;
            }
            const images = await Promise.all(o.series.map((k) => p.readEncoded(k)));
            const index = String(f).padStart(4, "0");
            images.forEach((img, i) => {
                const folder = `${o.prefix ?? ""}${o.series[i]}`;
                pending.push(encodePng(img.width, img.height, img.data).then((blob) => postFile(`${o.run}/frames/${folder}/${index}.png`, blob)));
            });
            saved++;
            if (pending.length > 24) {
                await Promise.all(pending.splice(0, pending.length));
            }
            if (f % 30 === 0) {
                console.log(`[capture ${o.run}] frame ${f + 1}/${o.frames}`);
            }
        }
        await Promise.all(pending);
        const meta = {
            run: o.run,
            series: o.series,
            frames: o.frames,
            savedFrames: saved,
            saveEvery,
            fps: o.fps,
            t0: o.t0,
            warmupFrames: o.warmupFrames,
            outputSize: [p.options.outputWidth, p.options.outputHeight],
            renderSize: [p.renderWidth, p.renderHeight],
            referenceSamples: p.options.referenceGrid ** 2,
            settings: p.settings,
            pathDuration: this.path.duration,
            seconds: (performance.now() - started) / 1000,
        };
        await postFile(`${o.run}/capture_${o.prefix ?? ""}meta.json`, JSON.stringify(meta, null, 2));
        return meta;
    }

    /**
     * Per-pass GPU time (ms) averaged over `frames` frames, from Babylon's
     * timestamp-query counters (windowed: delta total / delta count).
     */
    async benchmark(o: BenchmarkOptions): Promise<Record<string, unknown>> {
        const p = this.pipeline;
        const needs = seriesNeeds(o.series);
        p.resetHistory();
        p.request = { ...needs, sr: needs.sr, reference: false, present: null, encode: [] };
        for (let i = 0; i < o.warmupFrames; i++) {
            this.stepFrame(o.t0 + i / o.fps);
            await this.gpuIdle();
        }
        // timestamp results arrive a few frames late: let them drain, then open the window
        for (let i = 0; i < 4; i++) {
            await this.gpuIdle();
            await new Promise((r) => setTimeout(r, 20));
        }
        const before = p.gpuCounters();
        const cpuStart = performance.now();
        for (let i = 0; i < o.frames; i++) {
            this.stepFrame(o.t0 + (o.warmupFrames + i) / o.fps);
            await this.gpuIdle();
        }
        const cpuMs = (performance.now() - cpuStart) / o.frames;
        for (let i = 0; i < 6; i++) {
            this.stepFrame(o.t0 + (o.warmupFrames + o.frames) / o.fps);
            await this.gpuIdle();
            await new Promise((r) => setTimeout(r, 20));
        }
        const after = p.gpuCounters();
        const passes: Record<string, { ms: number; frames: number }> = {};
        for (const key of Object.keys(after)) {
            const dc = after[key].count - before[key].count;
            const dt = after[key].total - before[key].total;
            if (dc > 0) {
                passes[key] = { ms: dt / dc / 1e6, frames: dc };
            }
        }
        return {
            label: o.label,
            series: o.series,
            outputSize: [p.options.outputWidth, p.options.outputHeight],
            renderSize: [p.renderWidth, p.renderHeight],
            frames: o.frames,
            wallMsPerFrame: cpuMs,
            passes,
        };
    }

    async saveJson(path: string, data: unknown): Promise<void> {
        await postFile(path, JSON.stringify(data, null, 2));
    }
}
