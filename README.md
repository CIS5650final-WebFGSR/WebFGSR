# WebFGSR

Super resolution and frame interpolation for the web: a WebGPU / Babylon.js port of
[Mob-FGSR](https://github.com/Mob-FGSR/MobFGSR) (SIGGRAPH 2024).
CIS 5650 GPU Programming and Architecture, Fall 2026 - Hangyu Zhang, Weike Qian, Yikai Li.

**Milestone 1 (Nov 11): temporal super resolution, 2x per axis.** Babylon.js renders Sponza at
half resolution with a sub-pixel jittered camera plus a depth / motion-vector G-buffer; the
Mob-FGSR SR passes run as hand-written WGSL compute shaders and reconstruct the full-resolution
frame from the current frame and reprojected history.

## Run it

Requires Node 20+ and desktop Chrome with WebGPU.

```sh
npm install
npm run dev            # open the printed URL in Chrome
```

Controls (top-right panel): view (split bilinear | SR, SR, bilinear, native, inputs, SR weight),
scripted walk or free fly (WASD + mouse drag), SR switches (jitter sequence, history clamp, depth
rejection, depth threshold). The HUD shows per-pass GPU time from timestamp queries (Chrome rounds
these to 100 us unless it runs with `--enable-webgpu-developer-features`).

URL parameters: `?out=1920x1080` output size, `?scale=2` render scale, `?view=split|sr|bilinear|native|splitNative|lr|motion|depth|weight`,
`?mover=0` no moving object, `?moverSpin=0` moving object translates without rotating,
`?lights=1024` shading-heavy variant with clustered point lights, `?shadowEveryFrame=1` re-render
the shadow map every frame, `?refgrid=8` reference supersampling grid (offline only).

## Pipeline

```
Babylon.js (per frame)                          WGSL compute (src/sr/shaders)
  960x540 colour, jittered +-1/4 px  ---------\
  960x540 G-buffer: motion + depth  -> Dilate -> Blend history -> 1920x1080 output
                                                 ^  (reproject, YCoCg clamp,     |
                                                 |   depth test, jitter-aware    |
                                                 |   weight)                     |
                                                 +------- history (ping-pong) <--+
```

| File | What it does |
|---|---|
| `src/render/pipeline.ts` | Cameras, render targets, per-frame orchestration, readback |
| `src/render/gbuffer.ts` | WGSL ShaderMaterial: motion vectors from un-jittered matrices, linear depth, alpha test |
| `src/render/mipBias.ts` | Material plugin: texture LOD bias for the low-res pass |
| `src/render/jitter.ts` | Jitter sequences (Mob-FGSR's 12-entry table, quad, Halton) |
| `src/sr/shaders/dilate.wgsl` | Port of `Preprocessing/Dilate.comp` |
| `src/sr/shaders/upsampleFirstFrame.wgsl` | Port of `SuperResolution/UpsampleFirstFrame.comp` |
| `src/sr/shaders/blendHistory.wgsl` | Port of `SuperResolution/BlendHistory.comp` |
| `src/sr/computePasses.ts` | Babylon `ComputeShader` wrappers, bilinear baseline, reference accumulation |
| `src/scene/*` | Sponza scene, deterministic camera path, moving object, clustered lights |
| `tools/capture.mjs` | Headless Chrome driver: smoke screenshots, frame capture, GPU benchmark |
| `tools/metrics.mjs` | PSNR / SSIM against the reference (matches scikit-image to 1e-14) |
| `tools/media.mjs` | Split-screen / side-by-side videos and stills (ffmpeg-static) |

Conventions: Babylon draws render targets Y-flipped on WebGPU, so texel row 0 is the bottom of the
image and NDC y, texel y and motion y all point the same way. Motion vectors are
`uv_current - uv_previous` (Mob-FGSR's sign), computed from un-jittered matrices; jitter is in
low-res texels. Depth is linear view depth / far (far = 100 m).

### Differences from the reference implementation

* Inputs are rendered live instead of loaded from PNGs. The reference builds its "low-res" colour
  by point-sampling full-res images (`IO/LoadLRColor.comp`), which gives it full-res texture detail;
  we render at half resolution and compensate with a texture LOD bias of -1 (`mipBias.ts`).
* The 128x128 weight LUT (`resources/lut.exr`) is replaced by the analytic Catmull-Rom kernel it
  tabulates (checked to 1e-4).
* History is stored as rgba16float instead of rgba8.
* Neighbour clamps use `size - 1` (the reference clamps to `size`, one texel out of range).
* Motion vectors and depth come from one rgba32float G-buffer pass instead of separate textures.

## Evaluation

```sh
node tools/capture.mjs capture --run sponza_walk --series sr,bilinear,native,reference \
     --frames 480 --warmup 32 --query refgrid=8          # 8 s walk at 60 fps, 64-spp reference
node tools/metrics.mjs --run results/sponza_walk          # PSNR / SSIM vs reference
node tools/capture.mjs bench --size 1920x1080             # per-pass GPU time (also 3840x2160, --query lights=1024)
node tools/media.mjs split --run results/sponza_walk --left bilinear --right sr --out split.mp4
```

Results on an Apple M4 Pro, Chrome 157, 1920x1080 output (`results/summary.json`):

| 480 frames vs 64-spp reference | PSNR (dB) | SSIM |
|---|---|---|
| Bilinear 2x (540p rendered without jitter) | 32.92 | 0.882 |
| **Mob-FGSR 2x (this port)** | **34.17** | **0.918** |
| Mob-FGSR 2x without LOD bias | 33.38 | 0.896 |
| Mob-FGSR 2x, jitter sign flipped (convention check) | 31.11 | 0.826 |
| Native 1080p, 1 sample per pixel | 38.66 | 0.981 |

| GPU time (ms) | native | SR total (render + SR passes) |
|---|---|---|
| 1080p output, sun + sky light | 0.77 | 1.14 (0.52 + 0.62) |
| 4K output, sun + sky light | 1.86 | 2.60 (0.87 + 1.73) |
| 1080p output, 1024 clustered lights | 8.68 | 3.73 (3.22 + 0.51) |
| 4K output, 1024 clustered lights | 29.95 | 10.78 (9.05 + 1.74) |

Input validation:

* Jitter / motion conventions: flipping the jitter sign on purpose costs 3.1 dB (table above).
* Motion vectors: warping the previous native frame with the captured G-buffer motion
  (`--series mvraw`) leaves 1.9 mean abs error (8-bit) vs 9.5 without motion compensation; on the
  moving object 0.67 vs 6.9 (15.4 with the vectors negated).
* Reference: the 64-spp reference is within 61 dB of a 144-spp one (16 spp: 55 dB); reruns are
  bit-identical.
* Known artifact: on the tumbling object, accumulated history lags its changing shading; the YCoCg
  clamp hides most of it (object-interior error 8.3 without clamp when spinning, 2.1 when only
  translating). This is what the reliability cues of Milestone 2/3 target.

## Third-party code and assets

* **Mob-FGSR** (C++/OpenGL reference, CC BY-NC 4.0): the WGSL shaders in `src/sr/shaders/` are
  ports of its compute shaders and keep its attribution; the jitter table is copied from
  `src/offscreen_renderer.h`.
* **Babylon.js** (Apache-2.0): rendering, glTF loading, compute shader plumbing.
* **Sponza** (Crytek, glTF version from Khronos glTF-Sample-Assets, Cryengine Limited License
  Agreement): `public/assets/sponza/`, see its `LICENSE.md`.
* lil-gui (MIT), playwright-core (Apache-2.0), ffmpeg-static (GPL ffmpeg binary, used offline only).
