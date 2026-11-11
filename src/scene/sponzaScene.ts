// Sponza test scene: Khronos glTF Sponza (Crytek), a sun with a static shadow
// map, a sky-coloured fill light, and one moving object for rigid-object
// motion vectors.

import "@babylonjs/core/Lights/Shadows/shadowGeneratorSceneComponent";
import "@babylonjs/core/Lights/Clustered/clusteredLightingSceneComponent";
import { ClusteredLightContainer } from "@babylonjs/core/Lights/Clustered/clusteredLightContainer";
import { PointLight } from "@babylonjs/core/Lights/pointLight";
import "@babylonjs/loaders/glTF";
import { Scene } from "@babylonjs/core/scene";
import { ImportMeshAsync } from "@babylonjs/core/Loading/sceneLoader";
import { DirectionalLight } from "@babylonjs/core/Lights/directionalLight";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { ShadowGenerator } from "@babylonjs/core/Lights/Shadows/shadowGenerator";
import { RenderTargetTexture } from "@babylonjs/core/Materials/Textures/renderTargetTexture";
import { ImageProcessingConfiguration } from "@babylonjs/core/Materials/imageProcessingConfiguration";
import { PBRMaterial } from "@babylonjs/core/Materials/PBR/pbrMaterial";
import { CreateTorusKnot } from "@babylonjs/core/Meshes/Builders/torusKnotBuilder";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { Quaternion, Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { AbstractEngine } from "@babylonjs/core/Engines/abstractEngine";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";

/** Display-referred sky colour (the render targets hold tone-mapped sRGB values). */
export const SKY_COLOR = new Color4(0.62, 0.74, 0.88, 1);

export interface SponzaContent {
    scene: Scene;
    /** every mesh with geometry that the render targets draw */
    meshes: AbstractMesh[];
    /** the moving object (also contained in `meshes`) */
    mover: Mesh;
    sun: DirectionalLight;
    shadowGenerator: ShadowGenerator;
    /** places the moving object for simulation time t (seconds) */
    setMoverTime(t: number): void;
    setMoverEnabled(enabled: boolean): void;
    /** number of clustered point lights (0 = the default two-light setup) */
    clusteredLights: number;
}

/** Small deterministic PRNG (mulberry32) so every run places the same lights. */
function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Optional shading-heavy variant: N coloured point lights spread through the
 * atrium and arcades, rendered with Babylon's clustered (forward+) lighting.
 * Mirrors the many-light Sponza used to motivate the project.
 */
function addClusteredLights(scene: Scene, count: number): number {
    const rand = mulberry32(5650);
    const lights: PointLight[] = [];
    for (let i = 0; i < count; i++) {
        const p = new Vector3(-13.5 + 27 * rand(), 0.3 + 8.5 * rand(), -5.6 + 11.2 * rand());
        const light = new PointLight(`clustered${i}`, p, scene);
        const hue = rand();
        light.diffuse = Color3.FromHSV(hue * 360, 0.55 + 0.35 * rand(), 1.0);
        light.specular = light.diffuse.scale(0.5);
        light.intensity = 0.25 + 0.45 * rand();
        light.range = 2.0 + 1.5 * rand();
        light.shadowEnabled = false;
        lights.push(light);
    }
    const container = new ClusteredLightContainer("clusteredLights", lights, scene);
    container.maxRange = 3.5;
    return container.lights.length;
}

export async function createSponzaScene(engine: AbstractEngine, clusteredLightCount = 0, assetUrl = "/assets/sponza/Sponza.gltf"): Promise<SponzaContent> {
    const scene = new Scene(engine);
    scene.clearColor = new Color4(0.06, 0.07, 0.08, 1);
    scene.skipPointerMovePicking = true;
    scene.skipFrustumClipping = false;

    // The render targets store display-referred colour: PBR applies tone
    // mapping + gamma in the material (applyByPostProcess stays false), which
    // matches Mob-FGSR's 8-bit sRGB inputs.
    const ip = scene.imageProcessingConfiguration;
    ip.toneMappingEnabled = true;
    ip.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES;
    ip.exposure = 1.3;
    ip.contrast = 1.1;

    const result = await ImportMeshAsync(assetUrl, scene);
    const sponzaMeshes = result.meshes.filter((m) => m.getTotalVertices() > 0);

    // Sun through the open roof, plus a sky/bounce fill so the arcades are not black.
    const sun = new DirectionalLight("sun", new Vector3(-0.2, -1, 0.12).normalize(), scene);
    sun.position = new Vector3(8, 30, -6);
    sun.intensity = 5.0;
    sun.diffuse = new Color3(1.0, 0.93, 0.82);
    sun.shadowMinZ = 1;
    sun.shadowMaxZ = 60;

    const fill = new HemisphericLight("fill", new Vector3(0, 1, 0), scene);
    fill.intensity = 1.1;
    fill.diffuse = new Color3(0.75, 0.82, 0.95);
    fill.groundColor = new Color3(0.32, 0.27, 0.22);
    fill.specular = new Color3(0.05, 0.05, 0.05);

    const shadowGenerator = new ShadowGenerator(4096, sun);
    shadowGenerator.usePercentageCloserFiltering = true;
    shadowGenerator.filteringQuality = ShadowGenerator.QUALITY_HIGH;
    shadowGenerator.bias = 0.0008;
    shadowGenerator.normalBias = 0.015;
    for (const m of sponzaMeshes) {
        shadowGenerator.addShadowCaster(m, false);
        m.receiveShadows = true;
    }
    // Sponza is static: render the shadow map once (it is rendered during the
    // first frame's main pass, so the first frame's custom targets see an
    // empty map - capture always starts after warm-up frames).
    const shadowMap = shadowGenerator.getShadowMap();
    if (shadowMap) {
        shadowMap.refreshRate = new URLSearchParams(location.search).get("shadowEveryFrame") === "1" ? 1 : RenderTargetTexture.REFRESHRATE_RENDER_ONCE;
    }

    // Moving object: rigid motion through the courtyard (does not cast shadows,
    // so the static shadow map stays valid).
    const mover = CreateTorusKnot("mover", { radius: 0.45, tube: 0.13, radialSegments: 192, tubularSegments: 24, p: 2, q: 3 }, scene);
    const moverMat = new PBRMaterial("moverMat", scene);
    moverMat.albedoColor = new Color3(0.95, 0.45, 0.12);
    moverMat.metallic = 0.0;
    moverMat.roughness = 0.35;
    mover.material = moverMat;
    mover.receiveShadows = true;
    mover.rotationQuaternion = Quaternion.Identity();

    const center = new Vector3(0.0, 1.35, 0.0);
    // ?moverSpin=0 keeps the object's orientation fixed (pure translation, for motion-vector tests)
    const moverSpin = new URLSearchParams(location.search).get("moverSpin") !== "0";
    const setMoverTime = (t: number) => {
        // slow lap around the courtyard centre, ~1.2 m/s, tumbling
        const w = 0.55;
        mover.position.set(center.x + 3.2 * Math.cos(w * t), center.y + 0.25 * Math.sin(1.7 * t), center.z + 1.3 * Math.sin(w * t));
        const spin = moverSpin ? t : 0;
        Quaternion.FromEulerAnglesToRef(0.9 * spin, 1.3 * spin, 0.4 * spin, mover.rotationQuaternion!);
        mover.computeWorldMatrix(true);
    };
    setMoverTime(0);

    const setMoverEnabled = (enabled: boolean) => mover.setEnabled(enabled);
    const clusteredLights = clusteredLightCount > 0 ? addClusteredLights(scene, clusteredLightCount) : 0;

    return {
        scene,
        meshes: [...sponzaMeshes, mover],
        mover,
        sun,
        shadowGenerator,
        setMoverTime,
        setMoverEnabled,
        clusteredLights,
    };
}
