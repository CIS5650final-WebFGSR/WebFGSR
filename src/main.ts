// WebFGSR entry point: Mob-FGSR temporal super resolution in Babylon.js/WebGPU.
//
// URL parameters:
//   ?out=1920x1080   presentation (output) size; the scene renders at out/scale
//   ?scale=2         render scale per axis
//   ?refgrid=4       reference supersampling grid (4 -> 16 samples per pixel)
//   ?view=split      initial view (see VIEWS)
//   ?mover=0         disable the moving object
//   ?lights=1024     add N clustered point lights (shading-heavy variant)
//   ?capture=1       offline mode: no render loop, window.webfgsr API only

import { WebGPUEngine } from "@babylonjs/core/Engines/webgpuEngine";
import GUI from "lil-gui";

import { createSponzaScene } from "./scene/sponzaScene";
import { SPONZA_WALK } from "./scene/cameraPath";
import { DEFAULT_PIPELINE_OPTIONS, Pipeline, type FrameRequest, type PipelineOptions, type PresentRequest } from "./render/pipeline";
import { CaptureController } from "./capture";
import type { JitterSequenceName } from "./render/jitter";

const params = new URLSearchParams(location.search);
const captureMode = params.get("capture") === "1";

function parseSize(value: string | null, fallback: [number, number]): [number, number] {
    const m = value?.match(/^(\d+)x(\d+)$/);
    return m ? [Number(m[1]), Number(m[2])] : fallback;
}

const VIEWS: Record<string, { label: string; present: PresentRequest; needs: Partial<FrameRequest> }> = {
    split: { label: "Split: bilinear | Mob-FGSR", present: { a: "sr", b: "bilinear" }, needs: { bilinear: true } },
    sr: { label: "Mob-FGSR SR (2x)", present: { a: "sr" }, needs: {} },
    bilinear: { label: "Bilinear upscale (2x)", present: { a: "bilinear" }, needs: { bilinear: true } },
    native: { label: "Native full res", present: { a: "native" }, needs: { native: true } },
    splitNative: { label: "Split: native | Mob-FGSR", present: { a: "sr", b: "native" }, needs: { native: true } },
    lr: { label: "Input: jittered low-res colour", present: { a: "lr" }, needs: {} },
    motion: { label: "Input: motion vectors", present: { a: "motion" }, needs: {} },
    depth: { label: "Input: linear depth", present: { a: "depth" }, needs: {} },
    weight: { label: "SR: current-frame weight", present: { a: "weight" }, needs: {} },
};

function showFatal(message: string): void {
    const el = document.getElementById("fatal")!;
    el.textContent = message;
    el.style.display = "flex";
}

async function main(): Promise<void> {
    if (!navigator.gpu) {
        showFatal("WebGPU is not available in this browser.\nUse a recent desktop Chrome.");
        return;
    }
    const [outW, outH] = parseSize(params.get("out"), [DEFAULT_PIPELINE_OPTIONS.outputWidth, DEFAULT_PIPELINE_OPTIONS.outputHeight]);
    const options: PipelineOptions = {
        ...DEFAULT_PIPELINE_OPTIONS,
        outputWidth: outW,
        outputHeight: outH,
        renderScale: Number(params.get("scale") ?? DEFAULT_PIPELINE_OPTIONS.renderScale),
        referenceGrid: Number(params.get("refgrid") ?? DEFAULT_PIPELINE_OPTIONS.referenceGrid),
    };

    const canvas = document.getElementById("renderCanvas") as HTMLCanvasElement;
    canvas.width = outW;
    canvas.height = outH;
    const engine = new WebGPUEngine(canvas, {
        antialias: false,
        adaptToDeviceRatio: false,
        stencil: false,
        powerPreference: "high-performance",
        deviceDescriptor: { requiredFeatures: ["timestamp-query", "float32-filterable"] as GPUFeatureName[] },
    });
    await engine.initAsync();
    // must be on before render targets / compute shaders are created
    engine.enableGPUTimingMeasurements = true;
    engine.setSize(outW, outH);

    const content = await createSponzaScene(engine, Number(params.get("lights") ?? 0));
    if (params.get("mover") === "0") {
        content.setMoverEnabled(false);
    }
    const pipeline = new Pipeline(engine, content, options);
    const path = SPONZA_WALK;

    let pathMode = true;
    const pose = path.evaluate(0);
    const setSimulationTime = (t: number) => {
        pipeline.gbufferMaterials.capturePreviousWorlds();
        content.setMoverTime(t);
        pipeline.gbufferMaterials.bindPreviousWorlds();
        if (pathMode) {
            path.evaluate(t, pose);
            pipeline.setPose(pose.eye, pose.target);
        }
    };
    setSimulationTime(0);

    const capture = new CaptureController(engine, pipeline, path, setSimulationTime);
    const api = {
        ready: false,
        options,
        renderSize: [pipeline.renderWidth, pipeline.renderHeight],
        pathDuration: path.duration,
        settings: pipeline.settings,
        warmUp: (maxFrames?: number) => capture.warmUp(maxFrames),
        capture: (o: Parameters<CaptureController["capture"]>[0]) => capture.capture(o),
        benchmark: (o: Parameters<CaptureController["benchmark"]>[0]) => capture.benchmark(o),
        saveJson: (p: string, d: unknown) => capture.saveJson(p, d),
        setSettings: (s: Partial<typeof pipeline.settings>) => Object.assign(pipeline.settings, s),
        setMoverEnabled: (e: boolean) => content.setMoverEnabled(e),
        clusteredLights: content.clusteredLights,
        adapterInfo: () => {
            const info = (engine as unknown as { _adapterInfo?: GPUAdapterInfo })._adapterInfo;
            return { vendor: info?.vendor, architecture: info?.architecture, description: info?.description, ua: navigator.userAgent };
        },
    };
    (window as unknown as { webfgsr: typeof api }).webfgsr = api;

    if (captureMode) {
        document.body.classList.add("capture");
        api.ready = true;
        console.log("[webfgsr] capture mode ready");
        return;
    }

    // ---------------------------------------------------------------- live demo
    const state = {
        view: params.get("view") && VIEWS[params.get("view")!] ? params.get("view")! : "split",
        playing: true,
        speed: 1.0,
        time: 0,
        split: 0.5,
        camera: "path" as "path" | "free",
        jitterSequence: pipeline.settings.jitterSequence as JitterSequenceName,
        clampHistory: pipeline.settings.clampHistory,
        depthRejection: pipeline.settings.depthRejection,
        depthThreshold: pipeline.settings.depthThreshold,
        mover: params.get("mover") !== "0",
        resetHistory: () => pipeline.resetHistory(),
    };

    const gui = new GUI({ title: "WebFGSR · Milestone 1" });
    gui.add(state, "view", Object.fromEntries(Object.entries(VIEWS).map(([k, v]) => [v.label, k]))).name("View");
    gui.add(state, "split", 0.05, 0.95, 0.01).name("Split position");
    const camFolder = gui.addFolder("Camera");
    camFolder
        .add(state, "camera", { "Scripted walk": "path", "Free fly (WASD + drag)": "free" })
        .name("Mode")
        .onChange((v: string) => {
            pathMode = v === "path";
            if (pathMode) {
                pipeline.displayCamera.detachControl();
            } else {
                pipeline.displayCamera.attachControl(true);
                pipeline.displayCamera.keysUp.push(87);
                pipeline.displayCamera.keysDown.push(83);
                pipeline.displayCamera.keysLeft.push(65);
                pipeline.displayCamera.keysRight.push(68);
            }
            pipeline.resetHistory();
        });
    camFolder.add(state, "playing").name("Play");
    camFolder.add(state, "speed", 0.1, 2, 0.05).name("Speed");
    camFolder
        .add(state, "time", 0, path.duration, 0.01)
        .name("Time (s)")
        .listen()
        .onChange(() => pipeline.resetHistory());
    const srFolder = gui.addFolder("Super resolution");
    srFolder
        .add(state, "jitterSequence", ["mobfgsr", "quad", "halton", "none"])
        .name("Jitter")
        .onChange((v: JitterSequenceName) => (pipeline.settings.jitterSequence = v));
    srFolder.add(state, "clampHistory").name("History clamp").onChange((v: boolean) => (pipeline.settings.clampHistory = v));
    srFolder.add(state, "depthRejection").name("Depth rejection").onChange((v: boolean) => (pipeline.settings.depthRejection = v));
    srFolder
        .add(state, "depthThreshold", 0.0005, 0.05, 0.0005)
        .name("Depth threshold")
        .onChange((v: number) => (pipeline.settings.depthThreshold = v));
    srFolder.add(state, "resetHistory").name("Reset history");
    gui.add(state, "mover").name("Moving object").onChange((v: boolean) => content.setMoverEnabled(v));

    const hud = document.getElementById("hud")!;
    const labelLeft = document.getElementById("labelLeft")!;
    const labelRight = document.getElementById("labelRight")!;
    const names: Record<string, string> = {
        sr: "Mob-FGSR 2x",
        bilinear: "Bilinear 2x",
        native: "Native",
        lr: "Low-res input",
        motion: "Motion vectors",
        depth: "Depth",
        weight: "Current-frame weight",
        reference: "Reference",
    };
    let lastHud = 0;
    let frames = 0;
    let fpsWindowStart = performance.now();
    let fps = 0;
    let last = performance.now();

    engine.runRenderLoop(() => {
        const now = performance.now();
        const dt = Math.min((now - last) / 1000, 0.1);
        last = now;
        if (state.playing && pathMode) {
            state.time += dt * state.speed;
            if (state.time > path.duration) {
                state.time = 0;
                pipeline.resetHistory();
            }
        }
        setSimulationTime(state.time);

        const view = VIEWS[state.view];
        pipeline.request = {
            sr: true,
            bilinear: !!view.needs.bilinear,
            native: !!view.needs.native,
            reference: false,
            present: { ...view.present, split: state.split },
            encode: [],
        };
        content.scene.render();

        frames++;
        if (now - fpsWindowStart > 1000) {
            fps = (frames * 1000) / (now - fpsWindowStart);
            frames = 0;
            fpsWindowStart = now;
        }
        if (now - lastHud > 250) {
            lastHud = now;
            const g = pipeline.gpuTimesMs();
            const f = (v: number) => (Number.isFinite(v) && v > 0 ? `${v.toFixed(3)} ms` : "  -");
            const srTotal = g.lrColor + g.gbuffer + g.dilate + (Number.isFinite(g.blend) ? g.blend : 0);
            hud.textContent = [
                `WebFGSR  ${pipeline.renderWidth}x${pipeline.renderHeight} -> ${outW}x${outH}` +
                    (content.clusteredLights ? `   ${content.clusteredLights} clustered lights` : ""),
                `frame ${fps.toFixed(0)} fps   jitter ${pipeline.currentJitter.map((v) => v.toFixed(2)).join(", ")}`,
                ``,
                `GPU (timestamp queries)`,
                `  low-res colour   ${f(g.lrColor)}`,
                `  low-res G-buffer ${f(g.gbuffer)}`,
                `  dilate           ${f(g.dilate)}`,
                `  blend history    ${f(g.blend)}`,
                `  = SR path        ${f(srTotal)}`,
                view.needs.native ? `  native full res  ${f(g.native)}` : "",
                view.needs.bilinear ? `  low-res plain    ${f(g.lrPlain)}` : "",
                view.needs.bilinear ? `  bilinear up      ${f(g.bilinearUp)}` : "",
                `  present          ${f(g.mainPass)}`,
            ]
                .filter((l) => l !== "")
                .join("\n");
            const p = view.present;
            labelRight.textContent = names[p.a];
            labelLeft.textContent = p.b ? names[p.b] : "";
            labelLeft.style.visibility = p.b ? "visible" : "hidden";
        }
    });
    api.ready = true;
}

main().catch((e) => {
    console.error(e);
    showFatal(`WebFGSR failed to start:\n${e instanceof Error ? e.message : String(e)}`);
});
