// Low-resolution G-buffer pass: motion vectors + linear depth.
//
// Output (rgba32float render target, Babylon texel convention: row 0 = bottom,
// uv = (texel + 0.5) / size, ndc = uv * 2 - 1):
//   rg = motion vector in uv units, uv_current - uv_previous (Mob-FGSR sign:
//        prevUV = uv - mv), computed from UN-jittered current/previous
//        view-projections so the projection jitter never leaks into it
//   b  = linear view depth / far   (clip.w / far)
//   a  = 1
// Rasterisation itself uses the jittered `viewProjection` (auto-bound by
// ShaderMaterial from the frozen jittered camera), so every G-buffer sample
// describes exactly the surface point the colour pass shaded.
//
// One G-buffer material per source material keeps alpha-tested geometry
// (Sponza's leaves/chains/plants) and back-face culling identical to the
// colour pass; dynamic meshes get their own instance with a previous-world
// matrix.

import { ShaderMaterial } from "@babylonjs/core/Materials/shaderMaterial";
import { ShaderLanguage } from "@babylonjs/core/Materials/shaderLanguage";
import { Material } from "@babylonjs/core/Materials/material";
import { Matrix } from "@babylonjs/core/Maths/math.vector";
import type { Scene } from "@babylonjs/core/scene";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { BaseTexture } from "@babylonjs/core/Materials/Textures/baseTexture";

const VERTEX = /* wgsl */ `
attribute position : vec3f;
#ifdef ALPHATEST
attribute uv : vec2f;
varying vUV : vec2f;
#endif
uniform world : mat4x4f;
uniform viewProjection : mat4x4f;
uniform vpCurrent : mat4x4f;
uniform vpPrevious : mat4x4f;
#ifdef DYNAMIC
uniform previousWorld : mat4x4f;
#endif
varying vCurClip : vec4f;
varying vPrevClip : vec4f;

@vertex
fn main(input : VertexInputs) -> FragmentInputs {
    let p = vec4f(vertexInputs.position, 1.0);
    let worldPos = uniforms.world * p;
    vertexOutputs.position = uniforms.viewProjection * worldPos;
    vertexOutputs.vCurClip = uniforms.vpCurrent * worldPos;
#ifdef DYNAMIC
    vertexOutputs.vPrevClip = uniforms.vpPrevious * (uniforms.previousWorld * p);
#else
    vertexOutputs.vPrevClip = uniforms.vpPrevious * worldPos;
#endif
#ifdef ALPHATEST
    vertexOutputs.vUV = vertexInputs.uv;
#endif
}
`;

const FRAGMENT = /* wgsl */ `
varying vCurClip : vec4f;
varying vPrevClip : vec4f;
#ifdef ALPHATEST
varying vUV : vec2f;
var albedoSamplerSampler : sampler;
var albedoSampler : texture_2d<f32>;
uniform alphaCutOff : f32;
uniform mipBias : f32;
#endif
uniform invFar : f32;

@fragment
fn main(input : FragmentInputs) -> FragmentOutputs {
#ifdef ALPHATEST
    // same LOD bias as the colour pass, so alpha-tested coverage matches it
    let alpha = textureSampleBias(albedoSampler, albedoSamplerSampler, input.vUV, uniforms.mipBias).a;
    if (alpha < uniforms.alphaCutOff) {
        discard;
    }
#endif
    let curNdc = input.vCurClip.xy / input.vCurClip.w;
    let prevNdc = input.vPrevClip.xy / input.vPrevClip.w;
    let mv = 0.5 * (curNdc - prevNdc);
    fragmentOutputs.color = vec4f(mv, input.vCurClip.w * uniforms.invFar, 1.0);
}
`;

interface AlphaTestInfo {
    texture: BaseTexture;
    cutoff: number;
}

/** Reads alpha-test parameters from a glTF-loaded PBRMaterial (or anything shaped like one). */
function alphaTestInfo(material: Material | null): AlphaTestInfo | null {
    if (!material || material.transparencyMode !== Material.MATERIAL_ALPHATEST) {
        return null;
    }
    const m = material as unknown as { albedoTexture?: BaseTexture | null; alphaCutOff?: number };
    if (!m.albedoTexture) {
        return null;
    }
    return { texture: m.albedoTexture, cutoff: m.alphaCutOff ?? 0.5 };
}

export class GBufferMaterials {
    private readonly bySource = new Map<Material | null, ShaderMaterial>();
    private readonly dynamic = new Map<AbstractMesh, { material: ShaderMaterial; previousWorld: Matrix; hasPrevious: boolean }>();
    readonly all: ShaderMaterial[] = [];
    private readonly alphaTested: ShaderMaterial[] = [];

    constructor(
        private readonly scene: Scene,
        private readonly far: number,
    ) {}

    private create(name: string, alpha: AlphaTestInfo | null, dynamic: boolean, backFaceCulling: boolean): ShaderMaterial {
        const defines: string[] = [];
        const uniforms = ["world", "viewProjection", "vpCurrent", "vpPrevious", "invFar"];
        const samplers: string[] = [];
        if (alpha) {
            defines.push("#define ALPHATEST");
            uniforms.push("alphaCutOff", "mipBias");
            samplers.push("albedoSampler");
        }
        if (dynamic) {
            defines.push("#define DYNAMIC");
            uniforms.push("previousWorld");
        }
        const material = new ShaderMaterial(
            name,
            this.scene,
            { vertexSource: VERTEX, fragmentSource: FRAGMENT },
            {
                attributes: alpha ? ["position", "uv"] : ["position"],
                uniforms,
                samplers,
                defines,
                needAlphaTesting: !!alpha,
                shaderLanguage: ShaderLanguage.WGSL,
            },
        );
        material.backFaceCulling = backFaceCulling;
        material.setFloat("invFar", 1 / this.far);
        if (alpha) {
            material.setTexture("albedoSampler", alpha.texture);
            material.setFloat("alphaCutOff", alpha.cutoff);
            material.setFloat("mipBias", 0);
            this.alphaTested.push(material);
        }
        this.all.push(material);
        return material;
    }

    /** G-buffer material to use for `mesh` (created on first use). */
    forMesh(mesh: AbstractMesh, isDynamic: boolean): ShaderMaterial {
        const source = mesh.material ?? null;
        const alpha = alphaTestInfo(source);
        const culling = source ? source.backFaceCulling : true;
        if (isDynamic) {
            let entry = this.dynamic.get(mesh);
            if (!entry) {
                entry = {
                    material: this.create(`gbuffer_dynamic_${mesh.name}`, alpha, true, culling),
                    previousWorld: Matrix.Identity(),
                    hasPrevious: false,
                };
                this.dynamic.set(mesh, entry);
            }
            return entry.material;
        }
        let material = this.bySource.get(source);
        if (!material) {
            material = this.create(`gbuffer_${source?.name ?? "default"}`, alpha, false, culling);
            this.bySource.set(source, material);
        }
        return material;
    }

    /** Texture LOD bias of the colour pass the G-buffer must match (alpha test). */
    setMipBias(bias: number): void {
        for (const m of this.alphaTested) {
            m.setFloat("mipBias", bias);
        }
    }

    /** Per-frame camera matrices (both un-jittered). */
    setViewProjections(current: Matrix, previous: Matrix): void {
        for (const m of this.all) {
            m.setMatrix("vpCurrent", current);
            m.setMatrix("vpPrevious", previous);
        }
    }

    /**
     * Call once per frame BEFORE dynamic meshes move: remembers their current
     * world matrix as next frame's "previous".
     */
    capturePreviousWorlds(): void {
        for (const [mesh, entry] of this.dynamic) {
            entry.previousWorld.copyFrom(mesh.computeWorldMatrix(true));
            entry.previousWorld.markAsUpdated();
            entry.hasPrevious = true;
        }
    }

    /** Call once per frame AFTER dynamic meshes moved (before rendering). */
    bindPreviousWorlds(): void {
        for (const [mesh, entry] of this.dynamic) {
            if (!entry.hasPrevious) {
                entry.previousWorld.copyFrom(mesh.computeWorldMatrix(true));
                entry.previousWorld.markAsUpdated();
                entry.hasPrevious = true;
            }
            entry.material.setMatrix("previousWorld", entry.previousWorld);
        }
    }
}
