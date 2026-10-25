"""Sequential structure-from-motion over keyframes, built on OpenCV primitives.

SIFT features -> two-view initialization (essential matrix) -> each next keyframe registered by PnP
against already-triangulated points -> new points triangulated against its neighbours.
The result is aligned so the detected floor is y = 0 with +Y up, and scaled so the median camera height
is a handheld 1.5 m (an assumption the CAD editor's measure tool replaces with a real calibration).

No bundle adjustment: drift grows on long paths, which is what the verify loop is there to expose.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .scene import write_atomic

MAX_DIM = 1280
RATIO = 0.75
REPROJ_PX = 4.0
MIN_PARALLAX_DEG = 1.0
MIN_INIT_SHIFT = 0.02  # median feature shift, as a fraction of image width, required to initialize
CAMS_ABOVE_FLOOR = 0.8  # share of cameras that must sit above a plane for it to count as the floor
CAMERA_HEIGHT_M = 1.5


def intrinsics(w, h, hfov_deg):
    f = 0.5 * w / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


class _Frame:
    def __init__(self, path, sift):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cannot read {path}")
        h, w = img.shape[:2]
        s = min(1.0, MAX_DIM / max(h, w))
        self.img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
        self.h, self.w = self.img.shape[:2]
        kp, self.des = sift.detectAndCompute(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), None)
        self.pts = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)
        self.kp2pt = {}
        self.R = self.t = None


def _match(a, b, matcher):
    if a.des is None or b.des is None or len(a.des) < 2 or len(b.des) < 2:
        return np.empty((0, 2), int)
    good = []
    for m in matcher.knnMatch(a.des, b.des, k=2):
        if len(m) == 2 and m[0].distance < RATIO * m[1].distance:
            good.append((m[0].queryIdx, m[0].trainIdx))
    return np.array(good, int).reshape(-1, 2)


def _triangulate(K, a, b, ia, ib):
    P1, P2 = K @ np.hstack([a.R, a.t]), K @ np.hstack([b.R, b.t])
    X = cv2.triangulatePoints(P1, P2, a.pts[ia].T, b.pts[ib].T)
    X = (X[:3] / X[3]).T
    keep = np.ones(len(X), bool)
    rays = []
    for f, idx in ((a, ia), (b, ib)):
        Xc = (f.R @ X.T + f.t).T
        keep &= Xc[:, 2] > 0
        proj = (K @ Xc.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        keep &= np.linalg.norm(proj - f.pts[idx], axis=1) < REPROJ_PX
        rays.append(X - (-f.R.T @ f.t).ravel())
    cos = np.sum(rays[0] * rays[1], 1) / (np.linalg.norm(rays[0], axis=1) * np.linalg.norm(rays[1], axis=1) + 1e-12)
    keep &= np.degrees(np.arccos(np.clip(cos, -1, 1))) > MIN_PARALLAX_DEG
    return X, keep


def reconstruct(images: list[Path], hfov_deg: float = 65.0, log=print) -> dict:
    sift, matcher = cv2.SIFT_create(nfeatures=4000), cv2.BFMatcher(cv2.NORM_L2)
    frames = [_Frame(p, sift) for p in images]
    if len(frames) < 2:
        raise ValueError("need at least two keyframes")
    K = intrinsics(frames[0].w, frames[0].h, hfov_deg)
    xyz, rgb = [], []

    def add_points(a, b, ia, ib):
        X, keep = _triangulate(K, a, b, ia, ib)
        n = 0
        for x, qa, qb, ok in zip(X, ia, ib, keep):
            if ok and qa not in a.kp2pt and qb not in b.kp2pt:
                pid = len(xyz)
                xyz.append(x)
                u, v = a.pts[qa].astype(int)
                rgb.append(a.img[min(v, a.h - 1), min(u, a.w - 1)][::-1])
                a.kp2pt[qa] = b.kp2pt[qb] = pid
                n += 1
        return n

    # 1. Initialize from the first pair with enough baseline.
    init = None
    for j in range(1, min(len(frames), 12)):
        m = _match(frames[0], frames[j], matcher)
        if len(m) < 60:
            continue
        p0, pj = frames[0].pts[m[:, 0]], frames[j].pts[m[:, 1]]
        if np.median(np.linalg.norm(p0 - pj, axis=1)) < MIN_INIT_SHIFT * frames[0].w:
            continue  # too little camera motion for a reliable baseline
        E, mask = cv2.findEssentialMat(p0, pj, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            continue
        n_in, R, t, mask = cv2.recoverPose(E, p0, pj, K, mask=mask)
        if n_in >= 40:
            init = (j, R, t, m[mask.ravel() > 0])
            break
    if init is None:
        raise RuntimeError("could not initialize: the first 12 keyframes lack overlap or camera motion; move the camera more slowly and sideways")
    j, R, t, m = init
    frames[0].R, frames[0].t = np.eye(3), np.zeros((3, 1))
    frames[j].R, frames[j].t = R, t
    add_points(frames[0], frames[j], m[:, 0], m[:, 1])
    log(f"sfm: initialized on keyframes 0-{j} with {len(xyz)} points")

    # 2. Register the remaining keyframes in order by PnP against their registered neighbours.
    order = list(range(1, j)) + list(range(j + 1, len(frames)))
    registered = [0, j]
    for i in order:
        f = frames[i]
        refs = sorted(registered, key=lambda r: abs(r - i))[:3]
        obj, img, qis, seen = [], [], [], set()
        matches = {}
        for r in refs:
            mm = _match(frames[r], f, matcher)
            matches[r] = mm
            for qa, qi in mm:
                pid = frames[r].kp2pt.get(qa)
                if pid is not None and qi not in seen:
                    seen.add(qi)
                    obj.append(xyz[pid]); img.append(f.pts[qi]); qis.append((qi, pid))
        if len(obj) < 25:
            log(f"sfm: keyframe {i} skipped ({len(obj)} 2D-3D matches)")
            continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(np.array(obj), np.array(img), K, None, iterationsCount=300,
                                                 reprojectionError=REPROJ_PX, confidence=0.999)
        if not ok or inl is None or len(inl) < 20:
            log(f"sfm: keyframe {i} skipped (PnP failed)")
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(np.array(obj)[inl], np.array(img)[inl], K, None, rvec, tvec)
        f.R, f.t = cv2.Rodrigues(rvec)[0], tvec.reshape(3, 1)
        for k in inl:
            f.kp2pt[qis[k][0]] = qis[k][1]
        r = refs[0]
        mm = matches[r]
        fresh = np.array([(qa, qi) for qa, qi in mm if qa not in frames[r].kp2pt and qi not in f.kp2pt], int).reshape(-1, 2)
        added = add_points(frames[r], f, fresh[:, 0], fresh[:, 1]) if len(fresh) else 0
        registered.append(i)
        log(f"sfm: keyframe {i} registered ({len(inl)} inliers, +{added} points)")

    xyz_a, rgb_a = np.array(xyz), np.array(rgb, dtype=np.uint8)
    # Drop far outliers (bad triangulations at tiny parallax).
    d = np.linalg.norm(xyz_a - np.median(xyz_a, 0), axis=1)
    keep = d < np.percentile(d, 98) * 1.5
    xyz_a, rgb_a = xyz_a[keep], rgb_a[keep]
    cams = {i: (frames[i].R, frames[i].t) for i in sorted(registered)}
    return {"K": K, "size": (frames[0].w, frames[0].h), "cams": cams, "xyz": xyz_a, "rgb": rgb_a,
            "registered": len(registered), "total": len(frames)}


def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def fit_floor(xyz, up_guess, centers=None, iters=600, seed=0):
    """RANSAC floor: a plane roughly perpendicular to camera-up with (nearly) every camera above it.

    The camera-above test is what separates a floor from a ceiling or a tabletop seen from below.
    Returns (normal, d, inlier_ratio); normal is None when no plane qualifies.
    """
    rng = np.random.default_rng(seed)
    extent = np.linalg.norm(np.percentile(xyz, 95, 0) - np.percentile(xyz, 5, 0)) or 1.0
    thresh, best = 0.01 * extent, (None, None, 0)
    for _ in range(iters):
        p = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up_guess) < 0.8:
            continue
        n = n if n @ up_guess > 0 else -n
        d = -n @ p[0]
        if centers is not None and np.mean(centers @ n + d > thresh) < CAMS_ABOVE_FLOOR:
            continue
        inl = int((np.abs(xyz @ n + d) < thresh).sum())
        if inl > best[2]:
            best = (n, d, inl)
    return best[0], best[1], best[2] / len(xyz)


def align(rec: dict) -> dict:
    """Floor at y=0, +Y up, first camera at x=z=0 looking toward -Z, scaled to a 1.5 m camera height."""
    cams, xyz = rec["cams"], rec["xyz"]
    centers = {i: (-R.T @ t).ravel() for i, (R, t) in cams.items()}
    up = np.mean([-R[1] for R, _ in cams.values()], 0)  # camera -y (image up) in world
    up /= np.linalg.norm(up)
    cpos = np.array(list(centers.values()))
    n, d, ratio = fit_floor(xyz, up, cpos) if len(xyz) >= 50 else (None, None, 0)
    detected = n is not None and ratio >= 0.05
    if not detected:
        # No floor found: put the ground just below the lowest 3% of points that are under the cameras.
        below = (xyz @ up)[(xyz @ up) < np.min(cpos @ up)]
        n, d = up, -(np.percentile(below, 3) if len(below) else np.min(cpos @ up) - CAMERA_HEIGHT_M)
    A = _rot_between(n, np.array([0.0, 1.0, 0.0]))
    first = min(cams)
    fwd = A @ cams[first][0][2]  # camera +z (forward) in world, after leveling
    yaw = math.atan2(fwd[0], -fwd[2])
    c, s = math.cos(yaw), math.sin(yaw)
    A = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ A  # rotate so the first view looks down -Z
    offset = d * (A @ n)  # plane n.x + d = 0 sits at y = -d after A; shift it to y = 0
    c0 = A @ centers[first] + offset
    offset = offset - np.array([c0[0], 0, c0[2]])
    heights = np.array([(A @ c + offset)[1] for c in centers.values()])
    h = float(np.median(heights)) if np.median(heights) > 1e-6 else 1.0
    s = CAMERA_HEIGHT_M / h
    T = lambda p: s * (A @ p + offset)
    out_cams = []
    for i, (R, t) in cams.items():
        Rw = A @ R.T @ np.diag([1.0, -1.0, -1.0])  # OpenCV camera -> three.js camera axes, in world
        out_cams.append((i, T(centers[i]), Rw))
    return {"cams": out_cams, "xyz": np.array([T(p) for p in xyz]), "rgb": rec["rgb"],
            "floor_detected": bool(detected), "floor_ratio": float(ratio), "scale": s}


def quat_from_matrix(R):
    """Rotation matrix -> (x, y, z, w)."""
    q = np.empty(4)
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q[:] = [(R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S, 0.25 * S]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        S = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[i] = 0.25 * S
        q[j] = (R[j, i] + R[i, j]) / S
        q[k] = (R[k, i] + R[i, k]) / S
        q[3] = (R[k, j] - R[j, k]) / S
    return tuple(float(v) for v in q / np.linalg.norm(q))


def write_points(path: Path, xyz, rgb, limit=80000, seed=0):
    if len(xyz) > limit:
        idx = np.random.default_rng(seed).choice(len(xyz), limit, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    write_atomic(path, json.dumps({"count": len(xyz), "positions": np.round(xyz, 4).ravel().tolist(),
                                   "colors": rgb.astype(int).ravel().tolist()}))
