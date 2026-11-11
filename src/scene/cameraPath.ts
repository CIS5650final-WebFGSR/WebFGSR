// Deterministic camera path: centripetal-free uniform Catmull-Rom spline over
// (time, eye, target) keyframes. Every capture of the same time t yields the
// same pose, so reference and reconstructed frames match exactly.

import { Vector3 } from "@babylonjs/core/Maths/math.vector";

export interface CameraKey {
    t: number;
    eye: [number, number, number];
    target: [number, number, number];
}

export interface CameraPose {
    eye: Vector3;
    target: Vector3;
}

function catmullRom(p0: number, p1: number, p2: number, p3: number, s: number): number {
    const s2 = s * s;
    const s3 = s2 * s;
    return 0.5 * (2 * p1 + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s2 + (-p0 + 3 * p1 - 3 * p2 + p3) * s3);
}

export class CameraPath {
    constructor(readonly keys: CameraKey[]) {
        if (keys.length < 2) {
            throw new Error("camera path needs at least two keys");
        }
    }

    get duration(): number {
        return this.keys[this.keys.length - 1].t;
    }

    /** Pose at time t (clamped to the path). */
    evaluate(t: number, out?: CameraPose): CameraPose {
        const keys = this.keys;
        const tt = Math.min(Math.max(t, keys[0].t), this.duration);
        let i = 0;
        while (i < keys.length - 2 && tt > keys[i + 1].t) {
            i++;
        }
        const k0 = keys[Math.max(i - 1, 0)];
        const k1 = keys[i];
        const k2 = keys[i + 1];
        const k3 = keys[Math.min(i + 2, keys.length - 1)];
        const s = (tt - k1.t) / Math.max(k2.t - k1.t, 1e-9);
        const pose = out ?? { eye: new Vector3(), target: new Vector3() };
        pose.eye.set(
            catmullRom(k0.eye[0], k1.eye[0], k2.eye[0], k3.eye[0], s),
            catmullRom(k0.eye[1], k1.eye[1], k2.eye[1], k3.eye[1], s),
            catmullRom(k0.eye[2], k1.eye[2], k2.eye[2], k3.eye[2], s),
        );
        pose.target.set(
            catmullRom(k0.target[0], k1.target[0], k2.target[0], k3.target[0], s),
            catmullRom(k0.target[1], k1.target[1], k2.target[1], k3.target[1], s),
            catmullRom(k0.target[2], k1.target[2], k2.target[2], k3.target[2], s),
        );
        return pose;
    }
}

/**
 * Default walk through the Sponza atrium (Babylon world units = metres after
 * the glTF node's 0.008 scale): starts at one end of the courtyard, walks down
 * the middle while panning across the arcades, curtains and plants.
 */
export const SPONZA_WALK = new CameraPath([
    { t: 0.0, eye: [10.5, 1.7, -0.6], target: [0.0, 2.6, 0.6] },
    { t: 2.0, eye: [7.0, 1.75, -0.2], target: [-2.0, 3.4, 3.0] },
    { t: 4.0, eye: [3.5, 1.8, 0.4], target: [-6.0, 2.6, -1.5] },
    { t: 6.0, eye: [0.0, 1.9, 0.3], target: [-9.0, 3.6, 1.5] },
    { t: 8.0, eye: [-3.5, 2.0, -0.2], target: [-12.0, 2.4, -2.5] },
]);
