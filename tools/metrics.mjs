// tools/metrics.mjs - image quality metrics for a captured run.
//
// Every frame series under <runDir>/frames/ is compared, frame by frame, with a reference
// series (the supersampled ground truth):
//
//   node tools/metrics.mjs --run results/<run> [--reference reference] [--series a,b,c]
//                          [--start N] [--end N] [--jobs N] [--out metrics.json]
//
// Metrics are computed on the 8-bit RGB values (alpha ignored):
//   MSE / PSNR  mean squared error over all pixels and channels, PSNR = 10*log10(255^2 / MSE)
//   SSIM        the same number scikit-image returns for
//                 structural_similarity(a, b, channel_axis=2, gaussian_weights=True, sigma=1.5,
//                                       use_sample_covariance=False, data_range=255)
//
// Library use:  import { computeMetrics, readPngRGB } from './metrics.mjs'
// (importing this file never starts the CLI; run `node tools/metrics.mjs --help` for the CLI).

import { realpathSync } from 'node:fs';
import { mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import { availableParallelism } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { readPngRGB } from './lib/png.mjs';

export { readPngRGB };

// ---------------------------------------------------------------------------------------------
// Metric kernels (pure functions on packed RGB Uint8Arrays)
// ---------------------------------------------------------------------------------------------

const CHANNELS = 3;
const DATA_RANGE = 255;

// SSIM configuration: Wang et al. 2004, as scikit-image sets it up for gaussian_weights=True.
const SIGMA = 1.5;
// Window radius = int(truncate * sigma + 0.5) with truncate = 3.5, i.e. an 11x11 window. It is
// also the width of the border cropped from the SSIM map. blurRow() is written out for exactly
// this radius, so it is a literal rather than derived from SIGMA.
const RADIUS = 5;
const PAD = RADIUS * CHANNELS; // horizontal padding of a row buffer, in samples
const C1 = (0.01 * DATA_RANGE) ** 2;
const C2 = (0.03 * DATA_RANGE) ** 2;

// One half of the normalised Gaussian window: KERNEL[k] is the weight of the taps at offset +-k
// (built like scipy.ndimage's _gaussian_kernel1d).
const KERNEL = (() => {
  const taps = [];
  for (let x = -RADIUS; x <= RADIUS; x++) taps.push(Math.exp((-0.5 / (SIGMA * SIGMA)) * (x * x)));
  const sum = taps.reduce((s, v) => s + v, 0);
  return Float64Array.from({ length: RADIUS + 1 }, (_, k) => taps[RADIUS + k] / sum);
})();

// scipy.ndimage 'reflect' boundary (half-sample symmetric, d c b a | a b c d | d c b a):
// index -1 maps to 0, -2 to 1, n to n-1, n+1 to n-2. Valid while the overshoot is <= n.
const reflect = (i, n) => (i < 0 ? -i - 1 : i >= n ? 2 * n - 1 - i : i);

const psnrFromMse = (mse) => 10 * Math.log10((DATA_RANGE * DATA_RANGE) / mse);

/** Mean squared error over all samples (pixels x 3 channels) of two packed RGB buffers. */
export function mseRGB(a, b) {
  let sse = 0; // sum of integer squares: exact in a double for any realistic frame size
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sse += d * d;
  }
  return sse / a.length;
}

/**
 * Horizontal 11-tap Gaussian over one row of interleaved RGB samples. `src` carries PAD samples
 * of padding on each side; neighbouring pixels of a channel are CHANNELS (3) samples apart.
 * The taps are written out (rather than looped over) because this is the hottest loop of the
 * tool. Pairs are added outermost first, the summation order of scipy's correlate1d.
 */
function blurRow(src, dst) {
  const [w0, w1, w2, w3, w4, w5] = KERNEL;
  for (let i = 0, j = PAD; i < dst.length; i++, j++) {
    dst[i] =
      w0 * src[j] +
      w5 * (src[j - 15] + src[j + 15]) +
      w4 * (src[j - 12] + src[j + 12]) +
      w3 * (src[j - 9] + src[j + 9]) +
      w2 * (src[j - 6] + src[j + 6]) +
      w1 * (src[j - 3] + src[j + 3]);
  }
}

/**
 * Mean SSIM of two packed RGB images, matching scikit-image (see the file header).
 *
 * Per channel: Gaussian-weighted local means, variances and covariance (population, not sample),
 * the SSIM map from those, a RADIUS-wide border cropped, then the mean; finally the mean of the
 * three channel values. The Gaussian is separable and applied rows-first then columns, with
 * 'reflect' boundaries, exactly like scipy.ndimage.gaussian_filter, so the whole SSIM map (not
 * only its cropped mean) agrees with scikit-image's `full=True` output.
 *
 * The image is streamed one output row at a time, all three channels interleaved, so the working
 * set is a handful of row buffers instead of full-frame float planes.
 *
 * @param {Uint8Array} a packed RGB, width*height*3
 * @param {Uint8Array} b packed RGB, same size
 * @param {Float64Array|Float32Array} [map] optional width*height*3 buffer that receives the
 *        uncropped per-sample SSIM map (same layout as the input)
 * @returns {number}
 */
export function ssimRGB(a, b, width, height, map = null) {
  const win = 2 * RADIUS + 1;
  if (width < win || height < win) {
    throw new Error(`image ${width}x${height} is smaller than the ${win}x${win} SSIM window`);
  }
  const rowLen = width * CHANNELS; // samples per row
  const pad = PAD;
  const w0 = KERNEL[0];

  // Vertically filtered rows of a, b, a*a, b*b, a*b (padded for the horizontal pass) ...
  const vA = new Float64Array(rowLen + 2 * pad);
  const vB = new Float64Array(rowLen + 2 * pad);
  const vAA = new Float64Array(rowLen + 2 * pad);
  const vBB = new Float64Array(rowLen + 2 * pad);
  const vAB = new Float64Array(rowLen + 2 * pad);
  const padded = [vA, vB, vAA, vBB, vAB];
  // ... and the same five after the horizontal pass, i.e. the fully filtered local moments.
  const mA = new Float64Array(rowLen);
  const mB = new Float64Array(rowLen);
  const mAA = new Float64Array(rowLen);
  const mBB = new Float64Array(rowLen);
  const mAB = new Float64Array(rowLen);
  const ssimRow = new Float64Array(rowLen);

  const channelSum = [0, 0, 0];

  for (let y = 0; y < height; y++) {
    // Vertical pass. Inputs are small integers, so the products and pair sums below are exact.
    const o = y * rowLen;
    for (let i = 0; i < rowLen; i++) {
      const p = a[o + i];
      const q = b[o + i];
      const j = i + pad;
      vA[j] = w0 * p;
      vB[j] = w0 * q;
      vAA[j] = w0 * (p * p);
      vBB[j] = w0 * (q * q);
      vAB[j] = w0 * (p * q);
    }
    for (let k = RADIUS; k >= 1; k--) {
      const o1 = reflect(y - k, height) * rowLen;
      const o2 = reflect(y + k, height) * rowLen;
      const wk = KERNEL[k];
      for (let i = 0; i < rowLen; i++) {
        const p1 = a[o1 + i];
        const p2 = a[o2 + i];
        const q1 = b[o1 + i];
        const q2 = b[o2 + i];
        const j = i + pad;
        vA[j] += wk * (p1 + p2);
        vB[j] += wk * (q1 + q2);
        vAA[j] += wk * (p1 * p1 + p2 * p2);
        vBB[j] += wk * (q1 * q1 + q2 * q2);
        vAB[j] += wk * (p1 * q1 + p2 * q2);
      }
    }

    // Reflect-pad left and right (pixel -1-p mirrors pixel p, pixel width+p mirrors width-1-p).
    for (const row of padded) {
      for (let p = 0; p < RADIUS; p++) {
        for (let c = 0; c < CHANNELS; c++) {
          row[pad - (p + 1) * CHANNELS + c] = row[pad + p * CHANNELS + c];
          row[pad + rowLen + p * CHANNELS + c] = row[pad + rowLen - (p + 1) * CHANNELS + c];
        }
      }
    }

    // Horizontal pass.
    blurRow(vA, mA);
    blurRow(vB, mB);
    blurRow(vAA, mAA);
    blurRow(vBB, mBB);
    blurRow(vAB, mAB);

    // SSIM for this row.
    for (let i = 0; i < rowLen; i++) {
      const ua = mA[i];
      const ub = mB[i];
      const va = mAA[i] - ua * ua;
      const vb = mBB[i] - ub * ub;
      const vab = mAB[i] - ua * ub;
      ssimRow[i] = ((2 * ua * ub + C1) * (2 * vab + C2)) / ((ua * ua + ub * ub + C1) * (va + vb + C2));
    }
    if (map) map.set(ssimRow, o);

    // Only the interior (border of RADIUS pixels cropped) counts towards the mean.
    if (y >= RADIUS && y < height - RADIUS) {
      let r = 0;
      let g = 0;
      let bl = 0;
      for (let i = pad; i < rowLen - pad; i += CHANNELS) {
        r += ssimRow[i];
        g += ssimRow[i + 1];
        bl += ssimRow[i + 2];
      }
      channelSum[0] += r;
      channelSum[1] += g;
      channelSum[2] += bl;
    }
  }

  const count = (width - 2 * RADIUS) * (height - 2 * RADIUS);
  return (channelSum[0] / count + channelSum[1] / count + channelSum[2] / count) / CHANNELS;
}

/**
 * Compare two decoded images ({width, height, data: packed RGB}).
 * Identical images have no finite PSNR: psnr is null and identical is true (ssim is exactly 1).
 * @returns {{mse: number, psnr: number|null, ssim: number, identical: boolean}}
 */
export function compareRGB(imgA, imgB) {
  if (imgA.width !== imgB.width || imgA.height !== imgB.height) {
    throw new Error(`size mismatch: ${imgA.width}x${imgA.height} vs ${imgB.width}x${imgB.height}`);
  }
  const mse = mseRGB(imgA.data, imgB.data);
  if (mse === 0) return { mse: 0, psnr: null, ssim: 1, identical: true };
  return {
    mse,
    psnr: psnrFromMse(mse),
    ssim: ssimRGB(imgA.data, imgB.data, imgA.width, imgA.height),
    identical: false,
  };
}

/**
 * MSE, PSNR and SSIM between two PNG files of the same size.
 * @param {string} pathA
 * @param {string} pathB
 * @returns {Promise<{mse: number, psnr: number|null, ssim: number, identical: boolean, width: number, height: number}>}
 */
export async function computeMetrics(pathA, pathB) {
  const [imgA, imgB] = await Promise.all([readPngRGB(pathA), readPngRGB(pathB)]);
  return { ...compareRGB(imgA, imgB), width: imgA.width, height: imgA.height };
}

// ---------------------------------------------------------------------------------------------
// Worker side: one task = one frame index, compared against the reference for every series
// (so each reference frame is decoded only once).
// ---------------------------------------------------------------------------------------------

const WORKER_TAG = 'webfgsrMetricsWorker';

async function compareFrame({ frame, refPath, items }) {
  const out = { frame, width: 0, height: 0, refError: null, results: [], skipped: [] };
  let ref;
  try {
    ref = await readPngRGB(refPath);
  } catch (err) {
    out.refError = err.message;
    return out;
  }
  out.width = ref.width;
  out.height = ref.height;
  for (const { series, file } of items) {
    try {
      const img = await readPngRGB(file);
      if (img.width !== ref.width || img.height !== ref.height) {
        out.skipped.push({
          series,
          reason: `size ${img.width}x${img.height} differs from the reference ${ref.width}x${ref.height}`,
        });
        continue;
      }
      out.results.push({ series, ...compareRGB(ref, img) });
    } catch (err) {
      out.skipped.push({ series, reason: err.message });
    }
  }
  return out;
}

function startWorker() {
  parentPort.on('message', async (task) => {
    try {
      parentPort.postMessage({ result: await compareFrame(task) });
    } catch (err) {
      parentPort.postMessage({ error: err?.stack ?? String(err) });
    }
  });
}

/** Run `tasks` on `jobs` worker threads; each idle worker pulls the next task. */
function runPool(tasks, jobs, onResult) {
  return new Promise((resolve, reject) => {
    const workers = [];
    let next = 0;
    let done = 0;
    let settled = false;
    const settle = (err) => {
      if (settled) return;
      settled = true;
      Promise.all(workers.map((w) => w.terminate())).then(() => (err ? reject(err) : resolve()));
    };
    for (let n = Math.min(jobs, tasks.length); n > 0; n--) {
      // This same file is the worker entry point; WORKER_TAG tells it to serve tasks.
      const worker = new Worker(new URL(import.meta.url), { workerData: { [WORKER_TAG]: true } });
      workers.push(worker);
      const feed = () => {
        if (next < tasks.length) worker.postMessage(tasks[next++]);
      };
      worker.on('message', (msg) => {
        if (msg.error) return settle(new Error(`metrics worker failed: ${msg.error}`));
        onResult(msg.result);
        if (++done === tasks.length) settle();
        else feed();
      });
      worker.on('error', settle);
      worker.on('exit', (code) => settle(new Error(`metrics worker exited unexpectedly (code ${code})`)));
      feed();
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Run level: discover series and frames, fan out, aggregate
// ---------------------------------------------------------------------------------------------

export const defaultJobs = () => Math.max(1, availableParallelism() - 2);

const pad4 = (frame) => String(frame).padStart(4, '0');

/** [3, 4, 5, 9] -> "0003-0005, 0009"; very fragmented lists are cut short. */
function describeFrames(frames, maxRanges = 8) {
  const sorted = [...frames].sort((x, y) => x - y);
  const ranges = [];
  for (let i = 0; i < sorted.length; ) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    ranges.push(i === j ? pad4(sorted[i]) : `${pad4(sorted[i])}-${pad4(sorted[j])}`);
    i = j + 1;
  }
  if (ranges.length <= maxRanges) return ranges.join(', ');
  return `${ranges.slice(0, maxRanges).join(', ')}, ... (${ranges.length - maxRanges} more ranges)`;
}

/** Names of the series folders under frames/, sorted. */
async function listSeries(framesDir) {
  const names = [];
  for (const name of await readdir(framesDir)) {
    if (name.startsWith('.')) continue;
    if ((await stat(path.join(framesDir, name))).isDirectory()) names.push(name);
  }
  return names.sort();
}

/** Map of frame index -> file path for the NNNN.png files of one series, limited to [start, end). */
async function listFrames(dir, start, end) {
  const frames = new Map();
  for (const name of (await readdir(dir)).sort()) {
    const m = /^(\d+)\.png$/i.exec(name);
    if (!m) continue;
    const frame = Number(m[1]);
    if (frame >= start && frame < end && !frames.has(frame)) frames.set(frame, path.join(dir, name));
  }
  return frames;
}

function summarize(perFrame) {
  let mseSum = 0;
  let ssimSum = 0;
  let psnrSum = 0;
  let psnrCount = 0;
  let psnrMin = null;
  let psnrMinFrame = null;
  let ssimMin = Infinity;
  let ssimMinFrame = null;
  for (const f of perFrame) {
    mseSum += f.mse;
    ssimSum += f.ssim;
    if (f.psnr !== null) {
      psnrSum += f.psnr;
      psnrCount++;
      if (psnrMin === null || f.psnr < psnrMin) [psnrMin, psnrMinFrame] = [f.psnr, f.frame];
    }
    if (f.ssim < ssimMin) [ssimMin, ssimMinFrame] = [f.ssim, f.frame];
  }
  const frameCount = perFrame.length;
  const mseMean = mseSum / frameCount;
  return {
    frameCount,
    // Frames identical to the reference have infinite PSNR; they are counted here and left out
    // of psnrMean / psnrMin (which are null when every frame is identical).
    identicalFrames: frameCount - psnrCount,
    mseMean,
    psnrMean: psnrCount ? psnrSum / psnrCount : null,
    psnrOfMeanMse: mseMean > 0 ? psnrFromMse(mseMean) : null,
    psnrMin,
    psnrMinFrame,
    ssimMean: ssimSum / frameCount,
    ssimMin,
    ssimMinFrame,
    perFrame,
  };
}

/**
 * Evaluate a whole run folder and return the report object that the CLI writes to metrics.json.
 *
 * @param {object} options
 * @param {string} options.run run folder containing frames/<series>/NNNN.png
 * @param {string} [options.reference] name of the ground-truth series (default "reference")
 * @param {string[]|null} [options.series] series to evaluate (default: all but the reference)
 * @param {number} [options.start] first frame index (default 0)
 * @param {number} [options.end] frame index to stop before (default: no limit)
 * @param {number} [options.jobs] worker threads (default: cores - 2)
 * @param {(message: string) => void} [options.onWarning] called for every warning as it arises
 * @param {(done: number, total: number) => void} [options.onProgress] called after each frame
 */
export async function evaluateRun({
  run,
  reference = 'reference',
  series = null,
  start = 0,
  end = Infinity,
  jobs = defaultJobs(),
  onWarning = () => {},
  onProgress = () => {},
}) {
  const warnings = [];
  const warn = (message) => {
    warnings.push(message);
    onWarning(message);
  };

  const runDir = path.resolve(run);
  const framesDir = path.join(runDir, 'frames');
  let available;
  try {
    available = await listSeries(framesDir);
  } catch {
    throw new Error(`${framesDir} is not a readable folder (expected <runDir>/frames/<series>/NNNN.png)`);
  }
  if (!available.includes(reference)) {
    throw new Error(
      `reference series "${reference}" not found under ${framesDir} (found: ${available.join(', ') || 'no series'})`,
    );
  }

  const range = start > 0 || end !== Infinity ? ` in frame range [${start}, ${end === Infinity ? 'end' : end})` : '';
  const refFrames = await listFrames(path.join(framesDir, reference), start, end);
  if (refFrames.size === 0) throw new Error(`reference series "${reference}" has no NNNN.png frames${range}`);

  // Pick the series and pair their frames with the reference.
  const selected = [];
  const tasksByFrame = new Map(); // frame -> { frame, refPath, items: [{series, file}] }
  for (const name of new Set(series ?? available.filter((s) => s !== reference))) {
    if (name === reference) {
      warn(`series "${name}" is the reference itself; not compared`);
      continue;
    }
    if (!available.includes(name)) {
      warn(`series "${name}": no such folder under ${framesDir}; series skipped`);
      continue;
    }
    const frames = await listFrames(path.join(framesDir, name), start, end);
    const common = [...frames.keys()].filter((f) => refFrames.has(f));
    const missing = [...refFrames.keys()].filter((f) => !frames.has(f));
    const extra = [...frames.keys()].filter((f) => !refFrames.has(f));
    if (extra.length) {
      warn(`series "${name}": ${extra.length} frame(s) without a reference frame, skipped: ${describeFrames(extra)}`);
    }
    if (common.length === 0) {
      warn(`series "${name}": none of the ${refFrames.size} reference frame(s)${range} is present; series skipped`);
      continue;
    }
    if (missing.length) {
      warn(
        `series "${name}": ${missing.length} of ${refFrames.size} reference frame(s) missing, skipped: ${describeFrames(missing)}`,
      );
    }
    selected.push(name);
    for (const frame of common) {
      if (!tasksByFrame.has(frame)) tasksByFrame.set(frame, { frame, refPath: refFrames.get(frame), items: [] });
      tasksByFrame.get(frame).items.push({ series: name, file: frames.get(frame) });
    }
  }
  if (selected.length === 0) throw new Error('no series to compare with the reference');

  // Fan out over worker threads.
  const tasks = [...tasksByFrame.values()].sort((x, y) => x.frame - y.frame);
  const frameResults = [];
  await runPool(tasks, Math.max(1, Math.floor(jobs)), (result) => {
    frameResults.push(result);
    onProgress(frameResults.length, tasks.length);
  });
  frameResults.sort((x, y) => x.frame - y.frame); // workers finish in any order

  // Collect per-series results and group the skipped frames by reason, so a whole series at the
  // wrong size gives one warning rather than one per frame.
  const perSeries = new Map(selected.map((name) => [name, []]));
  const skips = new Map(selected.map((name) => [name, new Map()])); // series -> reason -> frames
  const refErrors = new Map(); // reason -> frames
  const sizes = new Map(); // "WxH" -> frames, for the reference
  for (const res of frameResults) {
    if (res.refError) {
      if (!refErrors.has(res.refError)) refErrors.set(res.refError, []);
      refErrors.get(res.refError).push(res.frame);
      continue;
    }
    const size = `${res.width}x${res.height}`;
    if (!sizes.has(size)) sizes.set(size, []);
    sizes.get(size).push(res.frame);
    for (const { series: name, mse, psnr, ssim, identical } of res.results) {
      perSeries.get(name).push({ frame: res.frame, mse, psnr, ssim, ...(identical ? { identical: true } : {}) });
    }
    for (const { series: name, reason } of res.skipped) {
      const byReason = skips.get(name);
      if (!byReason.has(reason)) byReason.set(reason, []);
      byReason.get(reason).push(res.frame);
    }
  }
  for (const [reason, frames] of refErrors) {
    warn(`reference "${reference}": ${frames.length} frame(s) skipped for every series (${reason}): ${describeFrames(frames)}`);
  }
  if (sizes.size > 1) {
    const list = [...sizes].map(([size, frames]) => `${size} (${frames.length} frames)`).join(', ');
    warn(`reference "${reference}": frames are not all the same size: ${list}`);
  }

  const report = { run: path.basename(runDir), reference, width: 0, height: 0, frames: [0, 0], series: {}, warnings };
  let first = Infinity;
  let last = -Infinity;
  for (const name of selected) {
    for (const [reason, frames] of skips.get(name)) {
      warn(`series "${name}": ${frames.length} frame(s) skipped (${reason}): ${describeFrames(frames)}`);
    }
    const perFrame = perSeries.get(name);
    if (perFrame.length === 0) {
      warn(`series "${name}": no frame could be compared; series left out of the report`);
      continue;
    }
    report.series[name] = summarize(perFrame);
    first = Math.min(first, perFrame[0].frame);
    last = Math.max(last, perFrame[perFrame.length - 1].frame);
  }
  if (first === Infinity) throw new Error('no frame could be compared with the reference (see the warnings above)');

  // Size of the first compared reference frame (they are all the same unless warned about above).
  const firstRef = frameResults.find((res) => res.frame === first);
  report.width = firstRef.width;
  report.height = firstRef.height;
  report.frames = [first, last];
  return report;
}

// ---------------------------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------------------------

/** Pretty JSON, with every perFrame entry folded onto a single line so long runs stay readable. */
export function formatJson(report) {
  const pretty = JSON.stringify(report, null, 2);
  return `${pretty.replace(/\{\s+"frame":[^{}]*\}/g, (entry) => entry.replace(/\s*\n\s*/g, ' '))}\n`;
}

/** series,frame,mse,psnr,ssim - one row per compared frame; psnr is empty for identical frames. */
export function formatCsv(report) {
  const quote = (text) => (/[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text);
  const lines = ['series,frame,mse,psnr,ssim'];
  for (const [name, stats] of Object.entries(report.series)) {
    for (const f of stats.perFrame) {
      const psnr = f.psnr === null ? '' : f.psnr.toFixed(4);
      lines.push(`${quote(name)},${f.frame},${f.mse.toFixed(6)},${psnr},${f.ssim.toFixed(6)}`);
    }
  }
  return `${lines.join('\n')}\n`;
}

/** Compact fixed-width table of the per-series summaries. */
export function formatTable(report) {
  const dB = (value) => (value === null ? 'inf' : value.toFixed(3));
  const at = (frame) => (frame === null ? '-' : pad4(frame));
  const rows = [['series', 'frames', 'PSNR mean', 'PSNR(mean MSE)', 'PSNR min', '@frame', 'SSIM mean', 'SSIM min', '@frame']];
  for (const [name, s] of Object.entries(report.series)) {
    rows.push([
      name,
      String(s.frameCount),
      dB(s.psnrMean),
      dB(s.psnrOfMeanMse),
      dB(s.psnrMin),
      at(s.psnrMinFrame),
      s.ssimMean.toFixed(5),
      s.ssimMin.toFixed(5),
      at(s.ssimMinFrame),
    ]);
  }
  const widths = rows[0].map((_, col) => Math.max(...rows.map((row) => row[col].length)));
  const lines = rows.map((row) =>
    row.map((cell, col) => (col === 0 ? cell.padEnd(widths[col]) : cell.padStart(widths[col]))).join('  '),
  );
  lines.splice(1, 0, widths.map((w) => '-'.repeat(w)).join('  '));
  return lines.join('\n');
}

// ---------------------------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------------------------

const HELP = `Usage: node tools/metrics.mjs --run <runDir> [options]

Compares every frame series of a captured run with a reference series and reports
PSNR and SSIM per frame and per series.

Input layout:
  <runDir>/frames/<series>/NNNN.png    NNNN = zero-padded frame index; 8-bit RGB or RGBA PNG,
                                       all series the same size (e.g. reference, native,
                                       bilinear, sr)

Options:
  --run <dir>          run folder, must contain frames/ (required)
  --reference <name>   ground-truth series                       [default: reference]
  --series <a,b,c>     series to evaluate, comma separated       [default: every other folder
                       under frames/]
  --start <N>          first frame index to evaluate             [default: 0]
  --end <N>            frame index to stop before (exclusive)    [default: no limit]
  --jobs <N>           worker threads                            [default: CPU cores - 2, here ${defaultJobs()}]
  --out <file>         where to write the JSON report            [default: <runDir>/metrics.json]
                       (a relative path is relative to the current directory)
  -h, --help           show this help

Output:
  <runDir>/metrics.json  { run, reference, width, height, frames: [first, last], warnings,
                           series: { <name>: { frameCount, identicalFrames, mseMean, psnrMean,
                           psnrOfMeanMse, psnrMin, psnrMinFrame, ssimMean, ssimMin, ssimMinFrame,
                           perFrame: [{ frame, mse, psnr, ssim }] } } }
  <runDir>/metrics.csv   series,frame,mse,psnr,ssim  (one row per compared frame)
  stdout                 one summary row per series
  With --out, the CSV is written next to the JSON: same name, .csv extension.

Metrics, on the 8-bit RGB values (alpha is ignored):
  MSE    mean squared error over all pixels and the 3 channels
  PSNR   10*log10(255^2 / MSE) in dB. psnrMean is the mean of the per-frame PSNRs,
         psnrOfMeanMse the PSNR of the mean MSE. A frame identical to its reference has no
         finite PSNR: it is written as "psnr": null, "identical": true (empty cell in the
         CSV), counted in identicalFrames and left out of psnrMean / psnrMin.
  SSIM   same value as scikit-image's structural_similarity(a, b, channel_axis=2,
         gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255):
         11x11 Gaussian window, K1=0.01, K2=0.03, 5 px border cropped, mean over R, G, B.

Only frame indices present in both a series and the reference are compared. Missing frames,
frames whose size differs from the reference and unreadable PNGs are skipped with a warning
(on stderr and in the JSON "warnings" list).

Exit code: 0 on success (warnings included), 1 on a runtime error, 2 on bad arguments.`;

class UsageError extends Error {}

function parseCli(argv) {
  let values;
  try {
    ({ values } = parseArgs({
      args: argv,
      strict: true,
      allowPositionals: false,
      options: {
        run: { type: 'string' },
        reference: { type: 'string' },
        series: { type: 'string' },
        start: { type: 'string' },
        end: { type: 'string' },
        jobs: { type: 'string' },
        out: { type: 'string' },
        help: { type: 'boolean', short: 'h' },
      },
    }));
  } catch (err) {
    throw new UsageError(err.message);
  }
  if (values.help) return { help: true };

  const integer = (name, min) => {
    if (values[name] === undefined) return undefined;
    if (!/^\d+$/.test(values[name]) || Number(values[name]) < min) {
      throw new UsageError(`--${name} expects an integer >= ${min}, got "${values[name]}"`);
    }
    return Number(values[name]);
  };
  const options = {
    run: values.run,
    reference: values.reference ?? 'reference',
    series: undefined,
    start: integer('start', 0) ?? 0,
    end: integer('end', 0) ?? Infinity,
    jobs: integer('jobs', 1) ?? defaultJobs(),
    out: values.out,
  };
  if (!options.run) throw new UsageError('missing required option --run <runDir>');
  if (options.end <= options.start) {
    throw new UsageError(`--end (${options.end}, exclusive) must be greater than --start (${options.start})`);
  }
  if (values.series !== undefined) {
    options.series = values.series.split(',').map((name) => name.trim()).filter(Boolean);
    if (options.series.length === 0) throw new UsageError('--series expects a comma separated list of series names');
  }
  return options;
}

/** Progress on stderr: a rewritten line on a terminal, an occasional line when piped. */
function progressReporter() {
  const tty = process.stderr.isTTY;
  let lastPrint = 0;
  return (done, total) => {
    const finished = done === total;
    const now = Date.now();
    if (!finished && now - lastPrint < (tty ? 100 : 5000)) return;
    lastPrint = now;
    if (tty) process.stderr.write(`\rmetrics: ${done}/${total} frames${finished ? '\n' : ''}`);
    else if (!finished) process.stderr.write(`metrics: ${done}/${total} frames\n`);
  };
}

async function main(argv) {
  let options;
  try {
    options = parseCli(argv);
  } catch (err) {
    if (!(err instanceof UsageError)) throw err;
    console.error(`metrics: ${err.message}\nTry: node tools/metrics.mjs --help`);
    return 2;
  }
  if (options.help) {
    console.log(HELP);
    return 0;
  }

  const started = performance.now();
  const report = await evaluateRun({
    ...options,
    onWarning: (message) => console.error(`metrics: warning: ${message}`),
    onProgress: progressReporter(),
  });

  const jsonPath = path.resolve(options.out ?? path.join(options.run, 'metrics.json'));
  const csvPath = path.join(path.dirname(jsonPath), `${path.basename(jsonPath, path.extname(jsonPath))}.csv`);
  await mkdir(path.dirname(jsonPath), { recursive: true });
  await writeFile(jsonPath, formatJson(report));
  await writeFile(csvPath, formatCsv(report));

  const names = Object.keys(report.series);
  const pairs = names.reduce((n, name) => n + report.series[name].frameCount, 0);
  const seconds = (performance.now() - started) / 1000;
  console.log(
    `run ${report.run}: ${names.length} series vs "${report.reference}", ${report.width}x${report.height}, ` +
      `frames ${pad4(report.frames[0])}-${pad4(report.frames[1])}\n`,
  );
  console.log(formatTable(report));
  console.log(
    `\n${pairs} frame pairs in ${seconds.toFixed(1)} s (${options.jobs} jobs)` +
      (report.warnings.length ? `, ${report.warnings.length} warning(s)` : ''),
  );
  console.log(`wrote ${jsonPath}`);
  console.log(`wrote ${csvPath}`);
  return 0;
}

/** True when this file is what `node` was asked to run (as opposed to being imported). */
function isCliEntry() {
  if (!isMainThread || !process.argv[1]) return false;
  try {
    return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (!isMainThread && workerData?.[WORKER_TAG]) {
  startWorker();
} else if (isCliEntry()) {
  try {
    process.exitCode = await main(process.argv.slice(2));
  } catch (err) {
    console.error(`metrics: error: ${err.message}`);
    process.exitCode = 1;
  }
}
