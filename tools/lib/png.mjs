// PNG decoding helpers for the offline tools.
//
// Frames are handed around as { width, height, data } where data is a Uint8Array of
// tightly packed 8-bit RGB (3 bytes per pixel, rows top to bottom). Alpha is dropped as-is:
// no blending and no un-premultiplying, the colour bytes stored in the file are what you get.

import { readFile } from 'node:fs/promises';
import pngjs from 'pngjs'; // CommonJS package, so PNG comes off the default export

const { PNG } = pngjs;

/**
 * Decode an in-memory PNG file to packed 8-bit RGB.
 * pngjs normalises every PNG flavour (grey, palette, 16-bit, interlaced) to 8-bit RGBA first.
 * @param {Uint8Array} fileBytes contents of a .png file
 * @returns {{width: number, height: number, data: Uint8Array}}
 */
export function decodePngRGB(fileBytes) {
  const { width, height, data: rgba } = PNG.sync.read(fileBytes);
  const pixels = width * height;
  const data = new Uint8Array(pixels * 3);
  for (let s = 0, d = 0; d < data.length; s += 4, d += 3) {
    data[d] = rgba[s];
    data[d + 1] = rgba[s + 1];
    data[d + 2] = rgba[s + 2];
  }
  return { width, height, data };
}

/**
 * Read a PNG file and decode it to packed 8-bit RGB.
 * @param {string} path
 * @returns {Promise<{width: number, height: number, data: Uint8Array}>}
 */
export async function readPngRGB(path) {
  const fileBytes = await readFile(path);
  try {
    return decodePngRGB(fileBytes);
  } catch (err) {
    // pngjs messages are cryptic on their own ("read requests waitng on finished stream").
    throw new Error(`not a readable PNG (truncated or corrupt?): ${err.message}`, { cause: err });
  }
}
