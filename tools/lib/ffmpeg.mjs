// Thin helpers around the ffmpeg binary shipped by the `ffmpeg-static` npm package.
// (ffmpeg-static ships no ffprobe, so probing is done by parsing `ffmpeg -i` output.)
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import ffmpegStatic from 'ffmpeg-static';

/** Error whose message is meant for the user (printed without a stack trace). */
export class ToolError extends Error {}

const MAX_BUFFER = 512 * 1024 * 1024; // room for a few raw RGB frames on stdout

let cachedBinary = null;
export function ffmpegBinary() {
  if (cachedBinary) return cachedBinary;
  if (!ffmpegStatic || !fs.existsSync(ffmpegStatic)) {
    throw new ToolError(
      `ffmpeg binary from the "ffmpeg-static" package not found (${ffmpegStatic ?? 'no path'}). ` +
        'Run `npm install` in the project root.',
    );
  }
  cachedBinary = ffmpegStatic;
  return cachedBinary;
}

/**
 * Run ffmpeg synchronously. Returns { stdout: Buffer, stderr: string, status }.
 * Throws ToolError (with the tail of stderr) on a non-zero exit unless allowFail is set.
 */
export function runFfmpeg(args, { cwd, allowFail = false } = {}) {
  const res = spawnSync(ffmpegBinary(), ['-hide_banner', '-nostdin', ...args], {
    cwd,
    maxBuffer: MAX_BUFFER,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (res.error) throw new ToolError(`could not start ffmpeg: ${res.error.message}`);
  const stderr = res.stderr ? res.stderr.toString('utf8') : '';
  if (res.status !== 0 && !allowFail) {
    const tail = stderr.trim().split('\n').slice(-12).join('\n');
    throw new ToolError(`ffmpeg failed (exit ${res.status ?? res.signal}):\n${tail}`);
  }
  return { stdout: res.stdout ?? Buffer.alloc(0), stderr, status: res.status };
}

/** Make sure this ffmpeg build has the filters/encoders the tool is designed around. */
export function requireCapabilities({ filters = [], encoders = [] }) {
  const have = (listing, name) => new RegExp(`^\\s*\\S+\\s+${name}\\s`, 'm').test(listing);
  const missing = [];
  if (filters.length) {
    const out = runFfmpeg(['-filters']).stdout.toString('utf8');
    for (const f of filters) if (!have(out, f)) missing.push(`filter ${f}`);
  }
  if (encoders.length) {
    const out = runFfmpeg(['-encoders']).stdout.toString('utf8');
    for (const e of encoders) if (!have(out, e)) missing.push(`encoder ${e}`);
  }
  if (missing.length) {
    throw new ToolError(`this ffmpeg build (${ffmpegBinary()}) lacks: ${missing.join(', ')}`);
  }
}

/** Width/height of a PNG, read straight from its IHDR chunk (no full decode). */
export function readPngSize(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const buf = Buffer.alloc(24);
    const got = fs.readSync(fd, buf, 0, 24, 0);
    if (got < 24 || buf.readUInt32BE(0) !== 0x89504e47 || buf.toString('latin1', 12, 16) !== 'IHDR') {
      throw new ToolError(`not a PNG file: ${file}`);
    }
    return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) };
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Probe the first video stream of a file by parsing `ffmpeg -i` output, e.g.
 *   Duration: 00:00:01.00, start: 0.000000, bitrate: 3393 kb/s
 *   Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 ..., 60 fps, ...
 */
export function probeVideo(file) {
  if (!fs.existsSync(file)) throw new ToolError(`file not found: ${file}`);
  // `ffmpeg -i X` with no output exits non-zero by design; the stream info is on stderr.
  const { stderr } = runFfmpeg(['-i', file], { allowFail: true });
  const line = stderr.split('\n').find((l) => /Stream #\d+:\d+.*: Video:/.test(l));
  if (!line) {
    const tail = stderr.trim().split('\n').slice(-3).join('\n');
    throw new ToolError(`no video stream found in ${file}:\n${tail}`);
  }
  const m = line.match(/Video: (\w+)(?: \(([^)]*)\))?.*?, (\w+)(?:\(([^)]*)\))?, (\d+)x(\d+)/);
  if (!m) throw new ToolError(`could not parse ffmpeg stream info: ${line.trim()}`);
  const fps = line.match(/, ([\d.]+) fps/) ?? line.match(/, ([\d.]+) tbr/);
  const dur = stderr.match(/Duration: (\d+):(\d+):([\d.]+)/);
  return {
    codec: m[1],
    profile: m[2] ?? '',
    pixFmt: m[3],
    colorTags: m[4] ?? '', // e.g. "tv, bt709, progressive"
    width: Number(m[5]),
    height: Number(m[6]),
    fps: fps ? Number(fps[1]) : NaN,
    duration: dur ? Number(dur[1]) * 3600 + Number(dur[2]) * 60 + Number(dur[3]) : NaN,
    hasAudio: /Stream #\d+:\d+.*: Audio:/.test(stderr),
  };
}

/** Exact number of video frames (demux only, no decode). */
export function countFrames(file) {
  // An explicit -stats makes ffmpeg print the final "frame=" line even at loglevel error.
  const { stderr } = runFfmpeg(['-loglevel', 'error', '-stats', '-i', file, '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-']);
  const all = [...stderr.matchAll(/frame=\s*(\d+)/g)];
  return all.length ? Number(all[all.length - 1][1]) : NaN;
}

/** True when the MP4 `moov` box precedes `mdat` (i.e. -movflags +faststart took effect). */
export function mp4HasFaststart(file) {
  const fd = fs.openSync(file, 'r');
  try {
    const size = fs.fstatSync(fd).size;
    const head = Buffer.alloc(16);
    let pos = 0;
    while (pos + 8 <= size) {
      fs.readSync(fd, head, 0, 16, pos);
      let boxSize = head.readUInt32BE(0);
      const type = head.toString('latin1', 4, 8);
      if (boxSize === 1) boxSize = Number(head.readBigUInt64BE(8)); // 64-bit box size
      else if (boxSize === 0) boxSize = size - pos; // box runs to end of file
      if (type === 'moov') return true;
      if (type === 'mdat') return false;
      if (boxSize < 8) break; // corrupt
      pos += boxSize;
    }
    return false;
  } finally {
    fs.closeSync(fd);
  }
}

// swscale flags used for every colour conversion: accurate rounding and full chroma
// resolution so RGB <-> YUV stays within about one 8-bit level apart from 4:2:0 loss.
const SWS_FLAGS = 'bicubic+accurate_rnd+full_chroma_int+full_chroma_inp';

/**
 * Filter that turns full-range sRGB (PNG) into BT.709 limited-range yuv420p. Without the
 * explicit matrix/range swscale would use BT.601, which shifts colours once a player
 * decodes the (HD) stream as BT.709. The stream is tagged to match (see RGB_TAGS).
 */
export const RGB_TO_YUV420_BT709 = `scale=out_color_matrix=bt709:out_range=tv:flags=${SWS_FLAGS},format=yuv420p`;

/** Output options that tag the encoded stream as BT.709 limited range. */
export const BT709_TAGS = ['-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709'];

/** Filter that converts a probed video back to full-range RGB, honouring its colour tags. */
export function yuvToRgbFilter(info) {
  const tags = info.colorTags.split(',').map((s) => s.trim()).filter(Boolean);
  // ffmpeg prints either "bt709" (matrix = primaries = transfer) or "matrix/primaries/transfer".
  const cs = tags.map((t) => t.split('/')[0]).find((t) => /^(bt709|bt470bg|smpte170m|bt2020nc|bt2020c)$/.test(t));
  let matrix;
  if (cs === 'bt709') matrix = 'bt709';
  else if (cs === 'bt470bg' || cs === 'smpte170m') matrix = 'bt601';
  else if (cs) matrix = 'bt2020';
  else matrix = info.height >= 720 ? 'bt709' : 'bt601'; // untagged: usual HD/SD convention
  const range = tags.includes('pc') || /^yuvj/.test(info.pixFmt) ? 'pc' : 'tv';
  return `scale=in_color_matrix=${matrix}:in_range=${range}:out_range=pc:flags=${SWS_FLAGS},format=rgb24`;
}

/** Run ffmpeg and return one frame as raw RGB24 bytes (width * height * 3). */
export function grabRgbFrame(args, width, height) {
  const { stdout } = runFfmpeg(['-loglevel', 'error', ...args, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']);
  if (stdout.length !== width * height * 3) {
    throw new ToolError(`expected a ${width}x${height} RGB frame (${width * height * 3} bytes), got ${stdout.length} bytes`);
  }
  return stdout;
}
