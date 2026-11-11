// Fullscreen WGSL pass that puts a presentation-size texture (or a split
// comparison of two) on the canvas.
//
// Texture orientation: Babylon render targets and our compute outputs store
// row 0 = bottom of the image; the post-process vertex shader's vUV is y-up,
// so sampling at vUV shows them upright on the canvas.

import "@babylonjs/core/ShadersWGSL/postprocess.vertex";
import { EffectRenderer, EffectWrapper } from "@babylonjs/core/Materials/effectRenderer";
import { ShaderLanguage } from "@babylonjs/core/Materials/shaderLanguage";
import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import type { BaseTexture } from "@babylonjs/core/Materials/Textures/baseTexture";

const PRESENT_WGSL = /* wgsl */ `
varying vUV : vec2f;
var texASampler : sampler;
var texA : texture_2d<f32>;
var texBSampler : sampler;
var texB : texture_2d<f32>;
// x: split on/off   y: split position (uv.x)   z: divider half width (uv)   w: unused
uniform split : vec4f;

@fragment
fn main(input : FragmentInputs) -> FragmentOutputs {
    let a = textureSample(texA, texASampler, input.vUV).rgb;
    let b = textureSample(texB, texBSampler, input.vUV).rgb;
    var c = a;
    if (uniforms.split.x > 0.5) {
        if (input.vUV.x < uniforms.split.y) {
            c = b;
        }
        if (abs(input.vUV.x - uniforms.split.y) < uniforms.split.z) {
            c = vec3f(0.92, 0.92, 0.9);
        }
    }
    fragmentOutputs.color = vec4f(c, 1.0);
}
`;

export class Presenter {
    private readonly renderer: EffectRenderer;
    private readonly wrapper: EffectWrapper;
    /** main (right-hand) image */
    texA: BaseTexture | null = null;
    /** comparison (left-hand) image, used when splitEnabled */
    texB: BaseTexture | null = null;
    splitEnabled = false;
    splitPosition = 0.5;

    constructor(private readonly engine: AbstractEngine) {
        this.renderer = new EffectRenderer(engine);
        this.wrapper = new EffectWrapper({
            engine,
            name: "present",
            fragmentShader: PRESENT_WGSL,
            uniformNames: ["split"],
            samplerNames: ["texA", "texB"],
            shaderLanguage: ShaderLanguage.WGSL,
        });
        this.wrapper.onApplyObservable.add(() => {
            const effect = this.wrapper.effect;
            const a = this.texA!;
            effect.setTexture("texA", a);
            effect.setTexture("texB", this.splitEnabled && this.texB ? this.texB : a);
            const halfWidth = 1.5 / this.engine.getRenderWidth();
            effect.setFloat4("split", this.splitEnabled ? 1 : 0, this.splitPosition, halfWidth, 0);
        });
    }

    isReady(): boolean {
        return this.wrapper.isReady();
    }

    /** Draws into whatever framebuffer is bound (the canvas when called after the main pass). */
    render(): void {
        if (!this.texA || !this.wrapper.isReady()) {
            return;
        }
        this.renderer.render(this.wrapper);
    }
}
