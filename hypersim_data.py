"""Hypersim loading for the VGGT head study.

Only "strict-standard" scenes are used (selected earlier): 1024x768, fx = fy = 886.81 px,
principal point at the image center. This makes the ray-distance -> planar-depth
conversion a fixed formula.

Image preprocessing must match vggt.utils.load_fn.load_and_preprocess_images(mode="crop"):
width 1024 -> 518, height 768 -> round(768 * 518/1024 / 14) * 14 = 392 (no crop needed).
The two axes are scaled by slightly different factors, so fx and fy are scaled separately.
"""
import json
import os

import h5py
import numpy as np
from PIL import Image

H0, W0 = 768, 1024          # Hypersim native resolution
F0 = 886.81                 # focal length (px) of strict-standard scenes
W1 = 518                    # VGGT input width
H1 = round(H0 * (W1 / W0) / 14) * 14   # 392, VGGT input height
PATCH = 14                  # VGGT patch size -> 28 x 37 tokens


def frame_paths(root, scene, frame):
    base = os.path.join(root, scene, "images")
    img = os.path.join(base, "scene_cam_00_final_preview", f"frame.{frame:04d}.tonemap.jpg")
    dep = os.path.join(base, "scene_cam_00_geometry_hdf5", f"frame.{frame:04d}.depth_meters.hdf5")
    return img, dep


def ray_to_planar(dist):
    """Hypersim depth_meters is distance to the camera center; convert to planar z (pixel centers)."""
    u = np.arange(W0) + 0.5 - W0 / 2
    v = np.arange(H0) + 0.5 - H0 / 2
    x, y = np.meshgrid(u, v)
    return dist * F0 / np.sqrt(x ** 2 + y ** 2 + F0 ** 2)


def load_gt_depth(path):
    """Planar GT depth at VGGT input resolution (H1, W1), float32. 0 marks invalid pixels.
    Conversion happens at native resolution; resizing uses NEAREST only (never blend depths)."""
    with h5py.File(path, "r") as f:
        dist = f["dataset"][:].astype(np.float32)
    z = ray_to_planar(dist).astype(np.float32)
    z[~np.isfinite(z) | (z <= 0)] = 0.0
    z_small = Image.fromarray(z).resize((W1, H1), Image.Resampling.NEAREST)
    return np.asarray(z_small, dtype=np.float32).copy()


def intrinsics():
    """Pinhole intrinsics at (H1, W1). Pixel (u, v) has its center at (u + 0.5, v + 0.5)."""
    return {"fx": F0 * W1 / W0, "fy": F0 * H1 / H0, "cx": W1 / 2, "cy": H1 / 2}


def unproject(z, K):
    """(H, W) planar depth -> (H, W, 3) camera-frame points."""
    h, w = z.shape
    u = (np.arange(w) + 0.5 - K["cx"]) / K["fx"]
    v = (np.arange(h) + 0.5 - K["cy"]) / K["fy"]
    x, y = np.meshgrid(u, v)
    return np.stack([x * z, y * z, z], axis=-1)


def available_frames(root, scene):
    frames = []
    for f in range(1000):
        img, dep = frame_paths(root, scene, f)
        if os.path.isfile(img) and os.path.isfile(dep):
            frames.append(f)
    return frames


def make_groups(root, scenes, n_groups=5, k=4, stride=5, seed=0):
    """Non-overlapping groups of k frames with a fixed stride, one group per contiguous block
    of the trajectory, random offset inside each block (fixed seed)."""
    rng = np.random.default_rng(seed)
    groups = []
    span = stride * (k - 1) + 1
    for scene in scenes:
        frames = available_frames(root, scene)
        block = len(frames) // n_groups
        if block < span:
            raise ValueError(f"{scene}: {len(frames)} frames is too few for {n_groups} groups of span {span}")
        for g in range(n_groups):
            start = g * block + int(rng.integers(0, block - span + 1))
            groups.append({"scene": scene, "group": g,
                           "frames": [frames[start + stride * i] for i in range(k)]})
    return groups


def load_or_make_groups(path, root, scenes, **kw):
    if os.path.isfile(path):
        with open(path) as f:
            groups = json.load(f)
        print(f"loaded {len(groups)} groups from {path}")
        return groups
    groups = make_groups(root, scenes, **kw)
    with open(path, "w") as f:
        json.dump(groups, f, indent=1)
    print(f"created {len(groups)} groups -> {path}")
    return groups
