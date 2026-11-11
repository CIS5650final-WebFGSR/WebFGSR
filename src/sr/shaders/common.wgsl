// WebFGSR - shared declarations for the super-resolution compute passes.
//
// Port of the uniform block in Mob-FGSR (resources/Uniforms.glsl,
// https://github.com/Mob-FGSR/MobFGSR, CC BY-NC 4.0). Only the fields the SR
// path needs are kept, and everything is packed into vec4s so the CPU-side
// layout is unambiguous.
//
// Conventions (all in *texel space of the render targets*, origin at texel
// row/column 0, whatever screen orientation that happens to be):
//   uv            = (texel + 0.5) / size
//   motion vector = uv_current - uv_previous           (so prevUV = uv - mv)
//   jitter        = shift of the low-res image, in low-res texels: the scene
//                   point that an un-jittered render puts at uv is found in
//                   the jittered low-res colour at uv + jitter / renderSize
//   depth         = linear view depth normalised to [0, 1] (0 = camera)

struct SRParams {
    // low-res (render) size: w, h, 1/w, 1/h
    renderSize : vec4f,
    // high-res (presentation) size: W, H, 1/W, 1/H
    presentationSize : vec4f,
    // x, y: jitter in low-res texels   z: depth_diff_threshold_sr   w: unused
    jitterAndThreshold : vec4f,
    // x: 1 = clamp history to the neighbourhood AABB (reference behaviour)
    // y: 1 = depth-based history rejection enabled (reference behaviour)
    // z: blend weight scale (1 = reference)       w: unused
    options : vec4f,
};
