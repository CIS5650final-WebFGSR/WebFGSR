#!/usr/bin/env node
// tools/media.mjs - turn captured frame sequences into comparison videos and stills for slides.
//
// Input : <run>/frames/<series>/NNNN.png  (8-bit PNG, every series the same size)
// Output: <run>/media/<name>.mp4|.png
//
// How it works (all image work is done by the ffmpeg binary from `ffmpeg-static`):
//  * Frames stay in planar RGB (gbrp) through crop / nearest-neighbour scale / compositing and
//    are converted exactly once, at the end, to BT.709 limited-range yuv420p. That keeps
//    colours from drifting and keeps upscaled pixels crisp until the very last step.
//  * Labels are small RGBA PNGs: ffmpeg's drawtext renders the glyphs, this script measures
//    them and draws the rounded box (so labels can be shrunk to fit and all share a baseline).
//  * The wipe is a `maskedmerge` of the two series with a mask that slides sideways, so both
//    series stay at the same image coordinates on either side of the divider.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { PNG } from 'pngjs';
import {
  BT709_TAGS,
  RGB_TO_YUV420_BT709,
  ToolError,
  countFrames,
  grabRgbFrame,
  mp4HasFaststart,
  probeVideo,
  readPngSize,
  requireCapabilities,
  runFfmpeg,
  yuvToRgbFilter,
} from './lib/ffmpeg.mjs';

const HELP = `media.mjs - comparison videos and stills from captured frame sequences

Frames are read from <run>/frames/<series>/NNNN.png (4-digit zero-padded frame index, 8-bit
PNG, every series the same size). Outputs go to <run>/media/ unless --out contains a directory.

Usage:
  node tools/media.mjs split  --run DIR --left A --right B         --out NAME.mp4 [options]
  node tools/media.mjs sbs    --run DIR --series A,B,C             --out NAME.mp4 [options]
  node tools/media.mjs still  --run DIR --series A,B,C --frame N   --out NAME.png [options]
  node tools/media.mjs poster --video FILE.mp4 [--time S] [--run DIR] --out NAME.png

Subcommands:
  split    Wipe comparison video. Left of a thin light divider is series A, right of it is
           series B, both at the SAME image coordinates. The divider sits at the centre, or
           sweeps left -> right -> left over the clip with --wipe.
  sbs      Side-by-side video: one panel per series, every panel showing the same region.
  still    The sbs composition for a single frame, written as PNG.
  poster   One frame of a video as PNG (the still image shown behind a video on a slide).

Options for split, sbs and still:
  --run DIR             Run folder containing frames/<series>/.
  --out NAME            Output file name (.mp4 for split/sbs, .png for still/poster). A bare
                        name goes to <run>/media/; a path with a directory is used as given.
  --labels "L1,L2"      One label per series, in order (default: the series names).
                        Write \\, for a comma inside a label; an empty entry hides that label.
  --no-labels           Do not draw labels.
  --label-pos POS       top (default) or bottom.
  --crop x,y,w,h        Region of the source frames to show, in source pixels (default: all).
  --scale K             Enlarge the region K times with NEAREST-NEIGHBOUR so source pixels stay
                        crisp (default 1). Use an integer. If the result does not fit --max-size,
                        K is lowered to the largest integer that fits; if even K=1 does not
                        fit, the picture is shrunk with an area filter (a note is printed).
  --max-size WxH        Largest allowed output (default 1920x1080, even numbers). The output
                        is never upscaled to it; odd sizes get 1 px of black padding.
  --font FILE           TrueType/OpenType font for labels (default: Arial Bold or similar).
  --font-size PX        Label text height (default: 40 px for a picture that fills 1920x1080,
                        proportionally less for smaller pictures; shrunk if a label is too wide).

Options for split and sbs (video):
  --fps N               Playback frame rate (default 60).
  --start N, --end N    First / last frame index to use, inclusive (default: all frames that
                        every series has).
  --crf N               Starting x264 quality (default 18; lower = better/larger).
  --preset NAME         x264 preset (default slow).
  --max-mb N            File size limit in MB, 1 MB = 1,000,000 bytes (default 20). If the encode
                        is larger it is automatically re-encoded with a higher CRF until it fits.

Options for split only:
  --left A, --right B   Series shown left / right of the divider.
  --wipe                Sweep the divider smoothly left -> right -> left instead of fixing it.
  --wipe-cycles N       Number of left-right-left sweeps over the clip (default 1).

Options for sbs and still:
  --series A,B,C        Series to show, left to right.
  --frame N             (still) frame index to use.

Options for poster:
  --video FILE          Video to read (also looked up in <run>/media/ when --run is given).
  --time S              Time of the frame in seconds (default 0 = first frame).
  --out NAME            PNG name; a bare name goes to <run>/media/, or next to the video
                        when --run is not given.

Videos are MP4 / H.264 High / yuv420p (BT.709, limited range) / even size / +faststart / no
audio. After every encode the first frame is decoded again and compared with the source
composite to confirm that colours did not shift.

Examples:
  node tools/media.mjs split --run results/sponza --left bilinear --right sr \\
       --labels "Bilinear upscale,Mob-FGSR" --wipe --out wipe.mp4
  node tools/media.mjs sbs --run results/sponza --series native,bilinear,sr,reference \\
       --crop 880,420,160,180 --scale 3 --out crop.mp4
  node tools/media.mjs still --run results/sponza --series bilinear,sr,reference --frame 120 \\
       --crop 880,420,160,180 --scale 4 --out crop.png
  node tools/media.mjs poster --run results/sponza --video wipe.mp4 --time 2.5 --out wipe.png

Exit code: 0 on success, 1 on a processing error, 2 on a usage error.`;

// ---------------------------------------------------------------------------------------
// Look and feel (sizes are for a picture that fills a 1920x1080 slide; they scale with it)
// ---------------------------------------------------------------------------------------
const LABEL_FONT_PX = 40; // readable from the back of a classroom at 1080p
const MIN_FONT_PX = 12;
const LABEL_BOX_RGB = [12, 14, 18];
const LABEL_BOX_ALPHA = 0.66;
const DIVIDER_RGB = [245, 245, 245];
const WIPE_RANGE = [0.05, 0.95]; // divider travel, as fractions of the picture width
const MAX_CRF = 40; // beyond this the picture is too degraded to be worth showing

// Label fonts, first existing one wins. The second list is used when a label contains
// characters (e.g. CJK) that the Latin fonts lack.
const FONTS_LATIN = [
  '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
  '/System/Library/Fonts/Supplemental/Arial.ttf',
  '/System/Library/Fonts/Helvetica.ttc',
  '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
  '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
  'C:\\Windows\\Fonts\\arialbd.ttf',
];
const FONTS_WIDE = [
  '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
  '/System/Library/Fonts/STHeiti Medium.ttc',
  '/System/Library/Fonts/Hiragino Sans GB.ttc',
  '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
  'C:\\Windows\\Fonts\\msyh.ttc',
];

// ---------------------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------------------
class UsageError extends ToolError {}

const frameName = (index) => `${String(index).padStart(4, '0')}.png`;
const megabytes = (bytes) => (bytes / 1e6).toFixed(2);
const even = (n) => n + (n % 2);
const clamp01 = (v) => Math.min(1, Math.max(0, v));

function parseNumber(flag, value, { integer = false, min = -Infinity, max = Infinity } = {}) {
  const n = Number(value);
  if (String(value).trim() === '' || !Number.isFinite(n) || (integer && !Number.isInteger(n)) || n < min || n > max) {
    const kind = integer ? 'an integer' : 'a number';
    const range = `${min > -Infinity ? ` >= ${min}` : ''}${max < Infinity ? ` <= ${max}` : ''}`;
    throw new UsageError(`${flag} must be ${kind}${range}, got "${value}"`);
  }
  return n;
}

function parseList(flag, value) {
  const items = String(value ?? '').split(',').map((s) => s.trim()).filter(Boolean);
  if (!items.length) throw new UsageError(`${flag} is required (comma-separated series names)`);
  return items;
}

function requireOption(o, name) {
  if (o[name] == null || o[name] === '') throw new UsageError(`--${name} is required`);
  return o[name];
}

function resolveRunDir(o) {
  const dir = path.resolve(requireOption(o, 'run'));
  if (!fs.existsSync(path.join(dir, 'frames'))) {
    throw new ToolError(`no frames/ folder in run directory ${dir}`);
  }
  return dir;
}

/** Bare names go to defaultDir; anything with a directory part is used as given. */
function resolveOut(out, ext, defaultDir) {
  if (!out) throw new UsageError('--out is required');
  if (path.extname(out).toLowerCase() !== ext) throw new UsageError(`--out must end in ${ext}, got "${out}"`);
  const file = path.isAbsolute(out) || /[\\/]/.test(out) ? path.resolve(out) : path.join(defaultDir, out);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  return file;
}

/** Write to a sibling temp file and rename, so a failed run never leaves a broken output. */
function writeAtomically(outFile, write) {
  const ext = path.extname(outFile);
  const partial = `${outFile.slice(0, -ext.length)}.partial-${process.pid}${ext}`;
  try {
    const result = write(partial);
    if (!fs.existsSync(partial) || fs.statSync(partial).size === 0) {
      throw new ToolError(`ffmpeg produced no output for ${outFile}`);
    }
    fs.renameSync(partial, outFile);
    return result;
  } finally {
    fs.rmSync(partial, { force: true });
  }
}

function withTempDir(fallbackParent, fn) {
  let dir;
  try {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'webfgsr-media-'));
  } catch {
    dir = fs.mkdtempSync(path.join(fallbackParent, '.media-tmp-')); // e.g. read-only system temp
  }
  try {
    return fn(dir);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

// ---------------------------------------------------------------------------------------
// Frame series
// ---------------------------------------------------------------------------------------
function loadSeries(runDir, name) {
  const root = path.join(runDir, 'frames');
  const dir = path.join(root, name);
  const isDir = (p) => fs.existsSync(p) && fs.statSync(p).isDirectory();
  if (!isDir(dir)) {
    const available = fs.readdirSync(root).filter((d) => isDir(path.join(root, d)));
    throw new ToolError(`series "${name}" not found in ${root} (available: ${available.join(', ') || 'none'})`);
  }
  const frames = new Set();
  for (const file of fs.readdirSync(dir)) {
    const m = file.match(/^(\d{4,})\.png$/);
    if (m && frameName(Number(m[1])) === file) frames.add(Number(m[1]));
  }
  if (!frames.size) throw new ToolError(`series "${name}" has no NNNN.png frames in ${dir}`);
  const sorted = [...frames].sort((a, b) => a - b);
  const { width, height } = readPngSize(path.join(dir, frameName(sorted[0])));
  return { name, dir, frames, first: sorted[0], last: sorted[sorted.length - 1], width, height };
}

function loadSeriesList(runDir, names) {
  const list = names.map((name) => loadSeries(runDir, name));
  for (const s of list) {
    if (s.width !== list[0].width || s.height !== list[0].height) {
      throw new ToolError(
        `all series must have the same size: "${list[0].name}" is ${list[0].width}x${list[0].height} but "${s.name}" is ${s.width}x${s.height}`,
      );
    }
  }
  return list;
}

/** Frame range (inclusive) that every series has without gaps. */
function resolveRange(seriesList, o) {
  const start = o.start != null ? parseNumber('--start', o.start, { integer: true, min: 0 }) : Math.max(...seriesList.map((s) => s.first));
  const end = o.end != null ? parseNumber('--end', o.end, { integer: true, min: 0 }) : Math.min(...seriesList.map((s) => s.last));
  if (end < start) throw new ToolError(`empty frame range ${start}..${end}`);
  for (const s of seriesList) {
    for (let i = start; i <= end; i++) {
      if (!s.frames.has(i)) {
        throw new ToolError(`series "${s.name}" is missing ${frameName(i)} (need frames ${start}..${end}; it has ${s.first}..${s.last})`);
      }
    }
  }
  return { start, end, count: end - start + 1 };
}

/** ffmpeg input arguments for a numbered PNG sequence starting at `start`. */
function sequenceInput(series, start, fps) {
  // "sequence" patterns only treat %0Nd specially; a literal % in the path must be doubled.
  const pattern = path.join(series.dir.replaceAll('%', '%%'), '%04d.png');
  return ['-framerate', String(fps), '-pattern_type', 'sequence', '-start_number', String(start), '-i', pattern];
}

// ---------------------------------------------------------------------------------------
// Layout: crop region -> panel size -> canvas size
// ---------------------------------------------------------------------------------------
function parseCrop(value, srcW, srcH) {
  if (value == null) return { x: 0, y: 0, w: srcW, h: srcH, full: true };
  const parts = String(value).split(',');
  if (parts.length !== 4) throw new UsageError(`--crop must be x,y,w,h, got "${value}"`);
  const [x, y, w, h] = parts.map((p, i) => parseNumber('--crop', p.trim(), { integer: true, min: i < 2 ? 0 : 1 }));
  if (x + w > srcW || y + h > srcH) {
    throw new ToolError(`--crop ${x},${y},${w},${h} reaches outside the ${srcW}x${srcH} source frames`);
  }
  return { x, y, w, h, full: x === 0 && y === 0 && w === srcW && h === srcH };
}

function parseMaxSize(value) {
  const m = String(value ?? '1920x1080').match(/^(\d+)x(\d+)$/i);
  if (!m || Number(m[1]) < 2 || Number(m[2]) < 2 || Number(m[1]) % 2 || Number(m[2]) % 2) {
    throw new UsageError(`--max-size must be WxH with even numbers, got "${value}"`);
  }
  return { maxW: Number(m[1]), maxH: Number(m[2]) };
}

/**
 * Decide the per-panel scale so that `panels` panels side by side fit the size limit.
 * Returns the panel size, the scaler to use and the (even) canvas size.
 */
function planLayout(o, srcW, srcH, panels) {
  const crop = parseCrop(o.crop, srcW, srcH);
  const { maxW, maxH } = parseMaxSize(o['max-size']);
  const requested = o.scale != null ? parseNumber('--scale', o.scale, { min: 0.01, max: 64 }) : 1;
  const notes = [];
  let factor = requested;
  const fit = Math.min(maxW / (panels * crop.w), maxH / crop.h); // largest factor that fits
  const what = `${panels > 1 ? `${panels} panels of ` : ''}${crop.w}x${crop.h}`;
  if (factor > fit) {
    if (fit >= 1) {
      // Keep source pixels crisp: fall back to the largest whole-number enlargement that fits.
      factor = Math.floor(fit + 1e-9);
      notes.push(`--scale ${requested} does not fit ${maxW}x${maxH} (${what}); using --scale ${factor} instead`);
    } else {
      factor = fit;
      notes.push(
        `${what}${requested !== 1 ? ` at --scale ${requested}` : ''} do not fit ${maxW}x${maxH}; shrinking to ${(fit * 100).toFixed(1)}% with an area filter ` +
          '(fine detail is averaged - use --crop for pixel-level comparisons)',
      );
    }
  } else if (factor > 1 && !Number.isInteger(factor)) {
    notes.push(`--scale ${factor} is not an integer: enlarged source pixels will not all have the same size`);
  }
  const panelW = Math.max(1, Math.floor(crop.w * factor + 1e-6));
  const panelH = Math.max(1, Math.floor(crop.h * factor + 1e-6));
  const contentW = panels * panelW;
  return {
    crop,
    factor,
    panels,
    panelW,
    panelH,
    scaler: panelW === crop.w && panelH === crop.h ? null : factor >= 1 ? 'neighbor' : 'area',
    contentW,
    contentH: panelH,
    canvasW: even(contentW), // H.264 yuv420p needs even dimensions
    canvasH: even(panelH),
    notes,
  };
}

/** Filter chain that turns one source stream into a panel (planar RGB, cropped, scaled). */
function panelFilters(layout) {
  const { crop, panelW, panelH, scaler } = layout;
  const chain = ['format=gbrp'];
  if (!crop.full) chain.push(`crop=${crop.w}:${crop.h}:${crop.x}:${crop.y}:exact=1`);
  if (scaler) chain.push(`scale=${panelW}:${panelH}:flags=${scaler}`);
  chain.push('setsar=1');
  return chain.join(',');
}

/** Append padding to even size plus the final pixel-format conversion; output pad is [out]. */
function finishGraph(parts, layout, target) {
  const chain = [];
  if (layout.canvasW !== layout.contentW || layout.canvasH !== layout.contentH) {
    chain.push(`pad=${layout.canvasW}:${layout.canvasH}:0:0:black`);
  }
  chain.push(target === 'video' ? RGB_TO_YUV420_BT709 : 'format=rgb24');
  return [...parts, `[rgb]${chain.join(',')}[out]`].join(';');
}

// ---------------------------------------------------------------------------------------
// Labels and divider strips (small RGBA PNGs overlaid by ffmpeg)
// ---------------------------------------------------------------------------------------
function parseLabels(o, seriesNames) {
  if (o['no-labels']) return seriesNames.map(() => '');
  if (o.labels == null) return [...seriesNames];
  // split on commas that are not escaped as "\,"
  const labels = o.labels.split(/(?<!\\),/).map((s) => s.replace(/\\,/g, ',').replace(/\s+/g, ' ').trim());
  if (labels.length !== seriesNames.length) {
    throw new UsageError(`--labels has ${labels.length} entries but there are ${seriesNames.length} series (${seriesNames.join(', ')})`);
  }
  return labels;
}

function pickFont(o, labels) {
  if (o.font) {
    const file = path.resolve(o.font);
    if (!fs.existsSync(file)) throw new ToolError(`--font file not found: ${file}`);
    return file;
  }
  // Anything beyond Latin/Greek/Cyrillic and common symbols needs a font with wider coverage.
  const wide = labels.some((l) => /[^\u0000-\u052F\u2000-\u22FF]/.test(l));
  const candidates = wide ? [...FONTS_WIDE, ...FONTS_LATIN] : FONTS_LATIN;
  const found = candidates.find((f) => fs.existsSync(f));
  if (!found) throw new ToolError('no label font found on this machine; pass --font /path/to/font.ttf (or --no-labels)');
  return found;
}

/**
 * Render `text` in white on black with ffmpeg's drawtext and return the green channel as a
 * glyph coverage map plus its ink bounding box. The baseline is pinned to row `base`
 * (y = base - max_glyph_a) so labels with and without descenders line up.
 */
function renderInk(text, fontName, fontPx, tmp, index) {
  const margin = Math.ceil(fontPx);
  const width = Math.min(16000, Math.ceil(fontPx * (1.15 * [...text].length + 1)) + 2 * margin);
  const height = Math.ceil(fontPx * 2.6);
  const base = Math.round(fontPx * 1.7);
  const textFile = `label${index}.txt`;
  fs.writeFileSync(path.join(tmp, textFile), text); // a text file sidesteps all filter escaping
  const draw =
    `drawtext=fontfile=${fontName}:textfile=${textFile}:expansion=none:fontcolor=white:` +
    `fontsize=${fontPx}:x=${margin}:y=${base}-max_glyph_a`;
  const { stdout } = runFfmpeg(
    ['-loglevel', 'error', '-f', 'lavfi', '-i', `color=c=black:s=${width}x${height}:d=1,format=gbrp`,
      '-vf', draw, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
    { cwd: tmp }, // relative font/text names: no path escaping needed inside the filter
  );
  if (stdout.length !== width * height * 3) throw new ToolError(`label rendering failed for "${text}"`);
  const coverage = new Uint8Array(width * height);
  let x0 = width, x1 = -1, y0 = height, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = stdout[(y * width + x) * 3 + 1];
      coverage[y * width + x] = v;
      if (v > 8) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { coverage, width, height, base, x0, x1, y0, y1 };
}

/** Compose one label: rounded semi-opaque dark box with the white glyphs on top. */
function buildLabelPng(ink, geom, file) {
  const { boxTop, boxH, padX, radius } = geom;
  const boxW = ink.x1 - ink.x0 + 1 + 2 * padX;
  const png = new PNG({ width: boxW, height: boxH });
  for (let y = 0; y < boxH; y++) {
    for (let x = 0; x < boxW; x++) {
      // Anti-aliased rounded rectangle: distance of the pixel centre to the corner circles.
      const dx = Math.max(Math.abs(x + 0.5 - boxW / 2) - (boxW / 2 - radius), 0);
      const dy = Math.max(Math.abs(y + 0.5 - boxH / 2) - (boxH / 2 - radius), 0);
      const boxA = LABEL_BOX_ALPHA * clamp01(radius - Math.hypot(dx, dy) + 0.5);
      const sx = x - padX + ink.x0;
      const sy = y + boxTop;
      const t = sx >= 0 && sx < ink.width && sy >= 0 && sy < ink.height ? ink.coverage[sy * ink.width + sx] / 255 : 0;
      // white text "over" the box, stored with straight (non-premultiplied) alpha
      const a = t + boxA * (1 - t);
      const o = (y * boxW + x) * 4;
      for (let c = 0; c < 3; c++) {
        png.data[o + c] = a > 0 ? Math.round((255 * t + LABEL_BOX_RGB[c] * boxA * (1 - t)) / a) : 0;
      }
      png.data[o + 3] = Math.round(a * 255);
    }
  }
  fs.writeFileSync(file, PNG.sync.write(png));
  return { file, width: boxW, height: boxH };
}

/**
 * Render all labels of one composition at a common font size, shrinking it until the widest
 * label fits `maxWidth`. Returns { fontPx, images: [{file,width,height} | null, ...] }.
 */
function makeLabels(texts, { fontPx, maxWidth, fontFile, tmp }) {
  if (texts.every((t) => !t)) return { fontPx, images: texts.map(() => null) };
  // Give the font a plain local name so its real path never has to be escaped in a filter.
  const fontName = `font${path.extname(fontFile) || '.ttf'}`;
  const fontLink = path.join(tmp, fontName);
  if (!fs.existsSync(fontLink)) {
    try {
      fs.symlinkSync(fontFile, fontLink);
    } catch {
      fs.copyFileSync(fontFile, fontLink);
    }
  }
  for (let attempt = 0; ; attempt++) {
    const inks = texts.map((t, i) => (t ? renderInk(t, fontName, fontPx, tmp, i) : null));
    const drawn = inks.filter(Boolean);
    texts.forEach((t, i) => {
      if (t && !inks[i]) console.warn(`warning: label "${t}" produced no visible glyphs with this font; skipped`);
    });
    if (!drawn.length) return { fontPx, images: texts.map(() => null) };
    // One box geometry for all labels: cap height centred vertically, grown if some glyph
    // (accent, descender) would come too close to the edge.
    const padX = Math.round(fontPx * 0.5);
    const padY = fontPx * 0.42;
    const guard = fontPx * 0.16;
    const base = drawn[0].base;
    const boxTop = Math.max(0, Math.floor(Math.min(base - fontPx * 0.72 - padY, ...drawn.map((k) => k.y0 - guard))));
    const boxBottom = Math.min(drawn[0].height, Math.ceil(Math.max(base + padY, ...drawn.map((k) => k.y1 + 1 + guard))));
    const geom = { boxTop, boxH: boxBottom - boxTop, padX, radius: Math.round(fontPx * 0.3) };
    const widest = Math.max(...drawn.map((k) => k.x1 - k.x0 + 1 + 2 * padX));
    if (widest > maxWidth && fontPx > MIN_FONT_PX && attempt < 6) {
      fontPx = Math.max(MIN_FONT_PX, Math.min(fontPx - 1, Math.floor((fontPx * maxWidth) / widest)));
      continue;
    }
    if (widest > maxWidth) console.warn(`warning: labels do not fit even at ${fontPx}px and will be clipped; use shorter labels`);
    const images = inks.map((k, i) => (k ? buildLabelPng(k, geom, path.join(tmp, `label${i}.png`)) : null));
    return { fontPx, images };
  }
}

/** Vertical strip: light line with a 1 px dark edge on both sides so it shows on any content. */
function makeDividerPng(height, lineW, tmp) {
  const width = lineW + 2;
  const png = new PNG({ width, height });
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const o = (y * width + x) * 4;
      const edge = x === 0 || x === width - 1;
      png.data[o] = edge ? 0 : DIVIDER_RGB[0];
      png.data[o + 1] = edge ? 0 : DIVIDER_RGB[1];
      png.data[o + 2] = edge ? 0 : DIVIDER_RGB[2];
      png.data[o + 3] = edge ? 115 : 255;
    }
  }
  const file = path.join(tmp, 'divider.png');
  fs.writeFileSync(file, PNG.sync.write(png));
  return { file, width };
}

/** Sizes of labels, margins and divider for a canvas, relative to a 1920x1080 slide. */
function styleFor(o, layout) {
  const fill = Math.max(layout.canvasW / 1920, layout.canvasH / 1080); // share of a 1080p slide it fills
  const fontPx = o['font-size'] != null
    ? parseNumber('--font-size', o['font-size'], { integer: true, min: 6, max: 400 })
    : Math.max(16, Math.round(LABEL_FONT_PX * fill));
  const labelPos = o['label-pos'] ?? 'top';
  if (labelPos !== 'top' && labelPos !== 'bottom') throw new UsageError(`--label-pos must be top or bottom, got "${labelPos}"`);
  return {
    fontPx,
    labelPos,
    margin: Math.round(fontPx * 0.6),
    lineW: Math.max(2, 2 * Math.round(layout.canvasW / 960)), // 4 px at 1920 wide, always even
  };
}

// ---------------------------------------------------------------------------------------
// Filter graphs. Inputs: 0..n-1 = frame streams, then label PNGs, then the divider PNG.
// Every graph ends in a planar-RGB stream labelled [rgb].
// ---------------------------------------------------------------------------------------

/** Tracks ffmpeg input arguments and hands out input indices. */
class Inputs {
  args = [];
  count = 0;
  add(...inputArgs) {
    this.args.push(...inputArgs);
    return this.count++;
  }
}

/** Panels side by side, a label on each, light seams between them. */
function sbsGraph({ layout, style, frameInputs, labels, labelInputs, dividerInput, divider }) {
  const { panelW, panelH } = layout;
  const parts = [];
  const panels = frameInputs.map((input, i) => {
    parts.push(`[${input}:v]${panelFilters(layout)}[p${i}]`);
    const label = labels[i];
    if (!label) return `[p${i}]`;
    const x = Math.round((panelW - label.width) / 2);
    const y = style.labelPos === 'top' ? style.margin : panelH - label.height - style.margin;
    parts.push(`[p${i}][${labelInputs[i]}:v]overlay=x=${x}:y=${y}:format=gbrp[q${i}]`);
    return `[q${i}]`;
  });
  if (panels.length === 1) {
    parts.push(`${panels[0]}null[rgb]`);
    return parts;
  }
  parts.push(`${panels.join('')}hstack=inputs=${panels.length}[row0]`);
  // The single divider image feeds one overlay per seam, so it has to be split first.
  const seams = panels.length - 1;
  if (seams > 1) parts.push(`[${dividerInput}:v]split=${seams}${Array.from({ length: seams }, (_, i) => `[d${i}]`).join('')}`);
  for (let i = 0; i < seams; i++) {
    const strip = seams > 1 ? `[d${i}]` : `[${dividerInput}:v]`;
    const x = (i + 1) * panelW - divider.width / 2;
    parts.push(`[row${i}]${strip}overlay=x=${x}:y=0:format=gbrp${i === seams - 1 ? '[rgb]' : `[row${i + 1}]`}`);
  }
  return parts;
}

/**
 * Wipe: series A left of the divider, series B right of it, same coordinates.
 * The mask is a white|black image twice the picture width, cropped at a sliding offset so
 * the white (= show A) part ends exactly at the divider position.
 */
function splitGraph({ layout, style, fps, frames, wipeCycles, labels, labelInputs, dividerInput, divider }) {
  const W = layout.panelW;
  const H = layout.panelH;
  const y = (label) => (style.labelPos === 'top' ? style.margin : H - label.height - style.margin);
  const parts = [`[0:v]${panelFilters(layout)}[a0]`, `[1:v]${panelFilters(layout)}[b0]`];
  // Labels are baked into their own series before merging, so the wipe hides a label
  // together with its side and a label can never sit on the other series' pixels.
  let a = '[a0]';
  let b = '[b0]';
  if (labels[0]) {
    parts.push(`[a0][${labelInputs[0]}:v]overlay=x=${style.margin}:y=${y(labels[0])}:format=gbrp[a1]`);
    a = '[a1]';
  }
  if (labels[1]) {
    parts.push(`[b0][${labelInputs[1]}:v]overlay=x=${W - labels[1].width - style.margin}:y=${y(labels[1])}:format=gbrp[b1]`);
    b = '[b1]';
  }
  // Divider x position per output frame n (0-based). Cosine easing: starts at the left
  // end, reaches the right end mid-cycle and returns, so the clip loops seamlessly.
  let pos = String(Math.round(W / 2));
  if (wipeCycles > 0) {
    const [lo, hi] = WIPE_RANGE;
    const start = (lo * W).toFixed(3);
    const halfTravel = (((hi - lo) * W) / 2).toFixed(3);
    const omega = ((2 * Math.PI * wipeCycles) / frames).toFixed(8);
    pos = `round(${start}+${halfTravel}*(1-cos(${omega}*n)))`;
  }
  parts.push(`color=c=white:s=${W}x${H}:r=${fps},format=gbrp,pad=${2 * W}:${H}:0:0:black,crop=${W}:${H}:x='${W}-${pos}':y=0:exact=1[mask]`);
  parts.push(`${b}format=gbrp[base]`, `${a}format=gbrp[top]`);
  parts.push('[base][top][mask]maskedmerge[merged]');
  parts.push(`[merged][${dividerInput}:v]overlay=x='${pos}-${divider.width / 2}':y=0:format=gbrp[rgb]`);
  return parts;
}

// ---------------------------------------------------------------------------------------
// Encoding, verification, reporting
// ---------------------------------------------------------------------------------------

/** Encode to H.264; raise CRF and re-encode until the file is within maxBytes. Returns the CRF used. */
function encodeVideo({ inputArgs, graph, frames, outFile, crf, preset, maxBytes }) {
  return writeAtomically(outFile, (partial) => {
    for (;;) {
      const t0 = Date.now();
      runFfmpeg([
        '-loglevel', 'error', '-y', ...inputArgs,
        '-filter_complex', graph, '-map', '[out]', '-frames:v', String(frames), '-an',
        '-c:v', 'libx264', '-profile:v', 'high', '-preset', preset, '-crf', String(crf),
        '-pix_fmt', 'yuv420p', ...BT709_TAGS, '-movflags', '+faststart', partial,
      ]);
      const size = fs.statSync(partial).size;
      console.log(`  encoded ${frames} frames at CRF ${crf} in ${((Date.now() - t0) / 1000).toFixed(1)} s -> ${megabytes(size)} MB`);
      if (size <= maxBytes) return crf;
      if (crf >= MAX_CRF) {
        throw new ToolError(
          `still ${megabytes(size)} MB at CRF ${crf} (limit ${megabytes(maxBytes)} MB). Shorten the clip (--start/--end), ` +
            'lower --fps or reduce the picture size.',
        );
      }
      // x264 output roughly halves for every +6 CRF; overshoot by one step to converge quickly.
      const next = Math.min(MAX_CRF, crf + Math.max(2, Math.ceil(6 * Math.log2(size / maxBytes)) + 1));
      console.log(`  ${megabytes(size)} MB is over the ${megabytes(maxBytes)} MB limit -> re-encoding at CRF ${next}`);
      crf = next;
    }
  });
}

/** Mean absolute error and per-channel mean signed error between two RGB24 buffers. */
function compareRgb(source, decoded) {
  const abs = [0, 0, 0];
  const signed = [0, 0, 0];
  for (let i = 0; i < source.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      const d = decoded[i + c] - source[i + c];
      abs[c] += Math.abs(d);
      signed[c] += d;
    }
  }
  const pixels = source.length / 3;
  return { mae: (abs[0] + abs[1] + abs[2]) / (3 * pixels), bias: signed.map((s) => s / pixels) };
}

/**
 * Check the finished MP4 against what the slide tool accepts, compare its first frame with
 * the source composite (colour shift check) and print the summary.
 */
function verifyAndReportVideo({ command, outFile, layout, inputArgs, rgbGraph, expectedFrames, fps, crf, preset, maxBytes }) {
  const info = probeVideo(outFile);
  const frames = countFrames(outFile);
  const size = fs.statSync(outFile).size;
  const problems = [];
  if (info.codec !== 'h264' || info.profile !== 'High') problems.push(`codec is ${info.codec} (${info.profile}), expected h264 (High)`);
  if (info.pixFmt !== 'yuv420p') problems.push(`pixel format is ${info.pixFmt}, expected yuv420p`);
  if (info.width % 2 || info.height % 2) problems.push(`odd size ${info.width}x${info.height}`);
  if (info.width !== layout.canvasW || info.height !== layout.canvasH) problems.push(`size ${info.width}x${info.height} differs from planned ${layout.canvasW}x${layout.canvasH}`);
  if (info.hasAudio) problems.push('file has an audio stream');
  if (!mp4HasFaststart(outFile)) problems.push('moov atom is not at the start (faststart missing)');
  if (frames !== expectedFrames) problems.push(`${frames} frames in file, expected ${expectedFrames}`);
  if (size > maxBytes) problems.push(`size ${size} bytes exceeds the limit`);

  // Colour check: frame 0 of the MP4, decoded back to RGB, against the composite before encoding.
  const source = grabRgbFrame([...inputArgs, '-filter_complex', rgbGraph, '-map', '[out]'], info.width, info.height);
  const decoded = grabRgbFrame(['-i', outFile, '-vf', yuvToRgbFilter(info)], info.width, info.height);
  const { mae, bias } = compareRgb(source, decoded);
  const signed = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;
  // A real colour shift (wrong matrix/range) shows as a large bias; MAE alone also grows with CRF.
  const colourOk = mae <= 4 && bias.every((v) => Math.abs(v) <= 1.5);

  console.log(`${command}: ${outFile}`);
  console.log(`  resolution : ${info.width}x${info.height}`);
  console.log(`  duration   : ${(frames / fps).toFixed(3)} s (${frames} frames)`);
  console.log(`  fps        : ${info.fps}`);
  console.log(`  size       : ${megabytes(size)} MB (${size} bytes, limit ${megabytes(maxBytes)} MB)`);
  console.log(`  video      : H.264 ${info.profile}, ${info.pixFmt} (${info.colorTags}), CRF ${crf}, preset ${preset}, faststart, no audio`);
  console.log(
    `  colour     : frame 0 decoded vs source composite: MAE ${mae.toFixed(2)}, bias R ${signed(bias[0])} G ${signed(bias[1])} B ${signed(bias[2])} ` +
      `(8-bit levels) -> ${colourOk ? 'ok' : 'CHECK'}`,
  );
  if (!colourOk) {
    console.warn('warning: decoded colours differ from the source more than expected (heavy compression, or a colour-conversion problem if the bias is large)');
  }
  if (problems.length) throw new ToolError(`output does not meet the video requirements:\n  - ${problems.join('\n  - ')}`);
}

function reportImage(command, outFile, extra) {
  const { width, height } = readPngSize(outFile);
  const size = fs.statSync(outFile).size;
  console.log(`${command}: ${outFile}`);
  console.log(`  resolution : ${width}x${height}`);
  console.log(`  duration   : still image (1 frame)${extra.duration ? `, ${extra.duration}` : ''}`);
  console.log(`  fps        : n/a${extra.fps ? ` (${extra.fps})` : ''}`);
  console.log(`  size       : ${megabytes(size)} MB (${size} bytes)`);
}

function videoOptions(o) {
  return {
    fps: o.fps != null ? parseNumber('--fps', o.fps, { min: 1, max: 240 }) : 60,
    crf: o.crf != null ? parseNumber('--crf', o.crf, { integer: true, min: 0, max: MAX_CRF }) : 18,
    preset: o.preset ?? 'slow',
    maxBytes: Math.floor((o['max-mb'] != null ? parseNumber('--max-mb', o['max-mb'], { min: 0.01 }) : 20) * 1e6),
  };
}

// ---------------------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------------------
function cmdSplit(o) {
  const runDir = resolveRunDir(o);
  const names = [requireOption(o, 'left'), requireOption(o, 'right')];
  const outFile = resolveOut(o.out, '.mp4', path.join(runDir, 'media'));
  const video = videoOptions(o);
  const wipeCycles = o.wipe || o['wipe-cycles'] != null
    ? (o['wipe-cycles'] != null ? parseNumber('--wipe-cycles', o['wipe-cycles'], { integer: true, min: 1 }) : 1)
    : 0;
  requireCapabilities({ filters: ['drawtext', 'overlay', 'maskedmerge', 'crop', 'scale', 'pad'], encoders: ['libx264'] });

  const seriesList = loadSeriesList(runDir, names);
  const range = resolveRange(seriesList, o);
  const layout = planLayout(o, seriesList[0].width, seriesList[0].height, 1);
  const style = styleFor(o, layout);
  const texts = parseLabels(o, names);
  layout.notes.forEach((n) => console.log(`note: ${n}`));

  withTempDir(path.dirname(outFile), (tmp) => {
    const inputs = new Inputs();
    seriesList.forEach((s) => inputs.add(...sequenceInput(s, range.start, video.fps)));
    // Each label must fit its half of the picture at the centre position.
    const { images: labels } = makeLabels(texts, {
      fontPx: style.fontPx,
      maxWidth: Math.floor(layout.panelW / 2) - 2 * style.margin - style.lineW,
      fontFile: pickFont(o, texts),
      tmp,
    });
    const labelInputs = labels.map((l) => (l ? inputs.add('-i', l.file) : null));
    const divider = makeDividerPng(layout.panelH, style.lineW, tmp);
    const dividerInput = inputs.add('-i', divider.file);
    const parts = splitGraph({ layout, style, fps: video.fps, frames: range.count, wipeCycles, labels, labelInputs, dividerInput, divider });

    console.log(`split: ${names[0]} | ${names[1]}, frames ${range.start}..${range.end} (${range.count}) at ${video.fps} fps, ${wipeCycles ? `wipe x${wipeCycles}` : 'divider at centre'}`);
    const crf = encodeVideo({ inputArgs: inputs.args, graph: finishGraph(parts, layout, 'video'), frames: range.count, outFile, ...video });
    verifyAndReportVideo({
      command: 'split', outFile, layout, inputArgs: inputs.args, rgbGraph: finishGraph(parts, layout, 'rgb'),
      expectedFrames: range.count, fps: video.fps, crf, preset: video.preset, maxBytes: video.maxBytes,
    });
  });
}

/** Shared by sbs (video) and still (one PNG): builds inputs and the side-by-side graph. */
function prepareSbs(o, seriesList, layout, tmp, frameInputArgs) {
  const names = seriesList.map((s) => s.name);
  const style = styleFor(o, layout);
  const texts = parseLabels(o, names);
  const inputs = new Inputs();
  const frameInputs = frameInputArgs.map((args) => inputs.add(...args));
  const { images: labels } = makeLabels(texts, {
    fontPx: style.fontPx,
    maxWidth: layout.panelW - 2 * style.margin,
    fontFile: pickFont(o, texts),
    tmp,
  });
  const labelInputs = labels.map((l) => (l ? inputs.add('-i', l.file) : null));
  const divider = makeDividerPng(layout.panelH, style.lineW, tmp);
  const dividerInput = seriesList.length > 1 ? inputs.add('-i', divider.file) : null;
  const parts = sbsGraph({ layout, style, frameInputs, labels, labelInputs, dividerInput, divider });
  return { inputs, parts };
}

function cmdSbs(o) {
  const runDir = resolveRunDir(o);
  const names = parseList('--series', o.series);
  const outFile = resolveOut(o.out, '.mp4', path.join(runDir, 'media'));
  const video = videoOptions(o);
  requireCapabilities({ filters: ['drawtext', 'overlay', 'hstack', 'crop', 'scale', 'pad'], encoders: ['libx264'] });

  const seriesList = loadSeriesList(runDir, names);
  const range = resolveRange(seriesList, o);
  const layout = planLayout(o, seriesList[0].width, seriesList[0].height, names.length);
  layout.notes.forEach((n) => console.log(`note: ${n}`));

  withTempDir(path.dirname(outFile), (tmp) => {
    const { inputs, parts } = prepareSbs(o, seriesList, layout, tmp, seriesList.map((s) => sequenceInput(s, range.start, video.fps)));
    console.log(`sbs: ${names.join(' | ')}, frames ${range.start}..${range.end} (${range.count}) at ${video.fps} fps, panels ${layout.panelW}x${layout.panelH}`);
    const crf = encodeVideo({ inputArgs: inputs.args, graph: finishGraph(parts, layout, 'video'), frames: range.count, outFile, ...video });
    verifyAndReportVideo({
      command: 'sbs', outFile, layout, inputArgs: inputs.args, rgbGraph: finishGraph(parts, layout, 'rgb'),
      expectedFrames: range.count, fps: video.fps, crf, preset: video.preset, maxBytes: video.maxBytes,
    });
  });
}

function cmdStill(o) {
  const runDir = resolveRunDir(o);
  const names = parseList('--series', o.series);
  const frame = parseNumber('--frame', requireOption(o, 'frame'), { integer: true, min: 0 });
  const outFile = resolveOut(o.out, '.png', path.join(runDir, 'media'));
  requireCapabilities({ filters: ['drawtext', 'overlay', 'hstack', 'crop', 'scale', 'pad'] });

  const seriesList = loadSeriesList(runDir, names);
  for (const s of seriesList) {
    if (!s.frames.has(frame)) throw new ToolError(`series "${s.name}" has no frame ${frameName(frame)} (it has ${s.first}..${s.last})`);
  }
  const layout = planLayout(o, seriesList[0].width, seriesList[0].height, names.length);
  layout.notes.forEach((n) => console.log(`note: ${n}`));

  withTempDir(path.dirname(outFile), (tmp) => {
    const { inputs, parts } = prepareSbs(o, seriesList, layout, tmp, seriesList.map((s) => ['-i', path.join(s.dir, frameName(frame))]));
    writeAtomically(outFile, (partial) => {
      runFfmpeg(['-loglevel', 'error', '-y', ...inputs.args, '-filter_complex', finishGraph(parts, layout, 'rgb'),
        '-map', '[out]', '-frames:v', '1', '-update', '1', '-c:v', 'png', partial]);
    });
  });
  reportImage('still', outFile, { duration: `frame ${frameName(frame).slice(0, -4)} of ${names.join(' | ')}` });
}

function cmdPoster(o) {
  const runDir = o.run ? path.resolve(o.run) : null;
  const given = requireOption(o, 'video');
  let video = path.resolve(given);
  if (!fs.existsSync(video) && runDir && !path.isAbsolute(given)) video = path.join(runDir, 'media', given);
  if (!fs.existsSync(video)) throw new ToolError(`video not found: ${given}`);
  const time = o.time != null ? parseNumber('--time', o.time, { min: 0 }) : 0;
  const outFile = resolveOut(o.out, '.png', runDir ? path.join(runDir, 'media') : path.dirname(video));

  const info = probeVideo(video);
  if (Number.isFinite(info.duration) && time >= info.duration) {
    throw new ToolError(`--time ${time} is past the end of the video (${info.duration.toFixed(3)} s)`);
  }
  writeAtomically(outFile, (partial) => {
    // -ss before -i seeks accurately to the first frame at or after `time`.
    runFfmpeg(['-loglevel', 'error', '-y', '-ss', String(time), '-i', video, '-map', '0:v:0',
      '-vf', yuvToRgbFilter(info), '-frames:v', '1', '-update', '1', '-c:v', 'png', partial]);
  });
  reportImage('poster', outFile, {
    duration: `taken at ${time.toFixed(3)} s of ${path.basename(video)} (${info.duration.toFixed(3)} s long)`,
    fps: `source video ${info.fps} fps`,
  });
}

// ---------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------
const str = { type: 'string' };
const flag = { type: 'boolean' };
const COMMON = { run: str, out: str, help: { type: 'boolean', short: 'h' } };
const COMPOSE = { labels: str, 'no-labels': flag, 'label-pos': str, crop: str, scale: str, 'max-size': str, font: str, 'font-size': str };
const VIDEO = { fps: str, start: str, end: str, crf: str, preset: str, 'max-mb': str };
const COMMANDS = {
  split: { run: cmdSplit, options: { ...COMMON, ...COMPOSE, ...VIDEO, left: str, right: str, wipe: flag, 'wipe-cycles': str } },
  sbs: { run: cmdSbs, options: { ...COMMON, ...COMPOSE, ...VIDEO, series: str } },
  still: { run: cmdStill, options: { ...COMMON, ...COMPOSE, series: str, frame: str } },
  poster: { run: cmdPoster, options: { ...COMMON, video: str, time: str } },
};

function main(argv) {
  const [name, ...rest] = argv;
  if (!name) {
    console.error(HELP);
    return 2;
  }
  if (name === '--help' || name === '-h' || name === 'help') {
    console.log(HELP);
    return 0;
  }
  try {
    const command = COMMANDS[name];
    if (!command) throw new UsageError(`unknown subcommand "${name}" (expected ${Object.keys(COMMANDS).join(', ')})`);
    let values;
    try {
      ({ values } = parseArgs({ args: rest, options: command.options, strict: true, allowPositionals: false }));
    } catch (err) {
      throw new UsageError(err.message);
    }
    if (values.help) {
      console.log(HELP);
      return 0;
    }
    command.run(values);
    return 0;
  } catch (err) {
    if (err instanceof UsageError) {
      console.error(`error: ${err.message}\nRun "node tools/media.mjs --help" for usage.`);
      return 2;
    }
    if (err instanceof ToolError) {
      console.error(`error: ${err.message}`);
      return 1;
    }
    console.error(err); // unexpected: keep the stack trace
    return 1;
  }
}

process.exitCode = main(process.argv.slice(2));
