// Texture LOD bias for the low-resolution passes.
//
// The low-res colour pass picks texture mip levels for its own (half)
// resolution, so the reconstructed full-res image can never contain texture
// detail finer than that - temporal upsamplers therefore render their input
// with a negative mip bias (log2(render / output) = -1 at 2x). WebGPU samplers
// have no LOD bias, so this material plugin redirects the PBR shader's material
// texture helper (TEXRD) to `textureSampleBias(..., uniforms.lrMipBias)`.
//
// Mob-FGSR's own evaluation sidesteps the issue: its low-res colour is point
// sampled from full-res renders (resources/IO/LoadLRColor.comp), i.e. it
// carries full-res texture detail.
//
// The bias is a global switched per render target (see Pipeline): the
// jittered low-res pass gets the bias, everything else gets 0.

import { MaterialPluginBase } from "@babylonjs/core/Materials/materialPluginBase";
import { ShaderLanguage } from "@babylonjs/core/Materials/shaderLanguage";
import type { Material } from "@babylonjs/core/Materials/material";
import type { UniformBuffer } from "@babylonjs/core/Materials/uniformBuffer";
import type { MaterialDefines } from "@babylonjs/core/Materials/materialDefines";

export class MipBiasPlugin extends MaterialPluginBase {
    /** bias applied by every patched material right now (set per render pass) */
    static currentBias = 0;

    constructor(material: Material) {
        super(material, "LowResMipBias", 300, { LR_MIPBIAS: false }, true, false);
        this._enable(true);
    }

    override getClassName(): string {
        return "MipBiasPlugin";
    }

    override isCompatible(shaderLanguage: ShaderLanguage): boolean {
        return shaderLanguage === ShaderLanguage.WGSL;
    }

    override prepareDefines(defines: MaterialDefines): void {
        defines["LR_MIPBIAS"] = true;
    }

    override getUniforms(): { ubo: Array<{ name: string; size: number; type: string }> } {
        return { ubo: [{ name: "lrMipBias", size: 1, type: "float" }] };
    }

    override bindForSubMesh(uniformBuffer: UniformBuffer): void {
        uniformBuffer.updateFloat("lrMipBias", MipBiasPlugin.currentBias);
    }

    override getCustomCode(shaderType: string, shaderLanguage?: ShaderLanguage): { [pointName: string]: string } | null {
        if (shaderType !== "fragment" || shaderLanguage !== ShaderLanguage.WGSL) {
            return null;
        }
        return {
            CUSTOM_FRAGMENT_DEFINITIONS:
                "fn lrBiasSample(t : texture_2d<f32>, s : sampler, uv : vec2f) -> vec4f { return textureSampleBias(t, s, uv, uniforms.lrMipBias); }",
            // Babylon's PBR shader reads every material texture (albedo, bump,
            // metal/rough, ...) through `fn TEXRD(t, ts, uv) { return textureSample(t, ts, uv); }`;
            // only that body is redirected. Shadow maps, BRDF LUT and cubes are untouched.
            "!(fn\\s+TEXRD\\s*\\([^)]*\\)\\s*->\\s*vec4f\\s*\\{\\s*return\\s+)textureSample\\(": "$1lrBiasSample(",
        };
    }
}

/** Attaches the plugin to every material that has a WGSL PBR shader. */
export function installMipBias(materials: Material[]): number {
    let n = 0;
    for (const m of materials) {
        if (m.getClassName() === "PBRMaterial") {
            new MipBiasPlugin(m);
            n++;
        }
    }
    return n;
}
