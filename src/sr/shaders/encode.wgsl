// WebFGSR - quantise a float colour texture to rgba8unorm for readback.
//
// Babylon render targets on WebGPU are stored bottom row first (texel row 0 =
// bottom of the image). PNG wants top row first, so rows are flipped here and
// the CPU can hand the bytes straight to ImageData.

@group(0) @binding(0) var src : texture_2d<f32>;
@group(0) @binding(1) var dst : texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let size = textureDimensions(dst);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let p = vec2i(id.xy);
    let c = textureLoad(src, p, 0);
    let flipped = vec2i(p.x, i32(size.y) - 1 - p.y);
    textureStore(dst, flipped, vec4f(clamp(c.rgb, vec3f(0.0), vec3f(1.0)), 1.0));
}
