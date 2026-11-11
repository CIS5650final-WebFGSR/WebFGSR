// Sub-pixel jitter sequences.
//
// A jitter is expressed in *texels of the target being rendered* and means
// "the rendered image is shifted by this many texels" (see the convention
// block in src/sr/shaders/common.wgsl). For the SR input that target is the
// low-res colour buffer; for the supersampled reference it is the full-res one.

export type JitterSequenceName = "mobfgsr" | "quad" | "halton" | "none";

export type Jitter = readonly [number, number];

/**
 * The 12-entry table shipped in Mob-FGSR (src/offscreen_renderer.h,
 * jitterSequence). All offsets are +-0.25 low-res texels, i.e. the centres of
 * the four high-res pixels covered by one low-res texel at 2x upscaling. Note
 * that it is not uniform: per 6 frames it visits (+,-) and (-,-) twice and
 * (-,+), (+,+) once. Kept as shipped for a faithful port.
 */
const MOBFGSR_SEQUENCE: readonly Jitter[] = [
    [-0.25, -0.25],
    [+0.25, -0.25],
    [-0.25, +0.25],
    [+0.25, -0.25],
    [-0.25, -0.25],
    [+0.25, +0.25],
    [-0.25, -0.25],
    [+0.25, -0.25],
    [-0.25, +0.25],
    [+0.25, -0.25],
    [-0.25, -0.25],
    [+0.25, +0.25],
];

/** Uniform 4-phase version of the same 2x2 pattern (each sub-pixel once per 4 frames). */
const QUAD_SEQUENCE: readonly Jitter[] = [
    [-0.25, -0.25],
    [+0.25, +0.25],
    [+0.25, -0.25],
    [-0.25, +0.25],
];

function halton(index: number, base: number): number {
    let f = 1;
    let r = 0;
    let i = index;
    while (i > 0) {
        f /= base;
        r += f * (i % base);
        i = Math.floor(i / base);
    }
    return r;
}

/** Halton(2,3), 16 phases, in [-0.5, 0.5) texels. */
const HALTON_SEQUENCE: readonly Jitter[] = Array.from({ length: 16 }, (_, i) => [halton(i + 1, 2) - 0.5, halton(i + 1, 3) - 0.5] as const);

const ZERO: Jitter = [0, 0];

export function jitterSequenceLength(name: JitterSequenceName): number {
    switch (name) {
        case "mobfgsr":
            return MOBFGSR_SEQUENCE.length;
        case "quad":
            return QUAD_SEQUENCE.length;
        case "halton":
            return HALTON_SEQUENCE.length;
        case "none":
            return 1;
    }
}

/** Jitter for the given frame counter (wraps around the sequence). */
export function getJitter(name: JitterSequenceName, frameCounter: number): Jitter {
    const wrap = (seq: readonly Jitter[]) => seq[((frameCounter % seq.length) + seq.length) % seq.length];
    switch (name) {
        case "mobfgsr":
            return wrap(MOBFGSR_SEQUENCE);
        case "quad":
            return wrap(QUAD_SEQUENCE);
        case "halton":
            return wrap(HALTON_SEQUENCE);
        case "none":
            return ZERO;
    }
}

/**
 * Regular n x n sub-pixel grid for the supersampled reference: sample s of
 * n*n, offsets in full-res texels within (-0.5, 0.5).
 */
export function supersampleOffset(sampleIndex: number, gridSize: number): Jitter {
    const i = sampleIndex % gridSize;
    const j = Math.floor(sampleIndex / gridSize) % gridSize;
    return [(i + 0.5) / gridSize - 0.5, (j + 0.5) / gridSize - 0.5];
}
