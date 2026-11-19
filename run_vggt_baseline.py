#!/usr/bin/env python3
"""Zero-training baseline: official VGGT depth head on the Hypersim test scenes (checkpoint G1).

Outputs in --out:
  groups.json            fixed frame groups (reuse for token caching and all heads)
  per_frame.csv          one row per evaluated frame
  summary.txt            frame-mean and scene-mean of every metric
  pred/<scene>_g<k>.npz  predicted depth / confidence (float16), for later reuse
  vis/*.png              RGB | GT | prediction | relative error, boundary band outlined
  vis/topdown_*.png      top-down view of the GT point cloud (walls must look straight)

Example (on a GPU node):
  python run_vggt_baseline.py --hypersim_root /scratch/$USER/hypersim \
      --scenes scenes_test.txt --vggt_repo /scratch/$USER/vggt \
      --weights /scratch/$USER/VGGT-1B/model.pt --out /scratch/$USER/results/vggt_official
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

import hypersim_data as hd
from metrics import frame_metrics, regions


def save_vis(path, rgb, z, p, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = (z > 0) & (p > 0)
    s = np.median(z[valid] / p[valid])
    ps = p * s
    rel = np.where(valid, np.abs(ps - z) / np.where(valid, z, 1), np.nan)
    band = regions(z, valid)["band"]
    vmax = np.quantile(z[valid], 0.98)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
    ax[0].imshow(rgb); ax[0].set_title(title)
    ax[1].imshow(np.where(z > 0, z, np.nan), vmin=0, vmax=vmax, cmap="turbo"); ax[1].set_title("GT planar depth")
    ax[2].imshow(ps, vmin=0, vmax=vmax, cmap="turbo"); ax[2].set_title("VGGT depth (median-aligned)")
    ax[3].imshow(rel, vmin=0, vmax=0.2, cmap="magma"); ax[3].set_title("abs rel error (0-0.2), band outlined")
    ax[3].contour(band, levels=[0.5], colors="cyan", linewidths=0.4)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_topdown(path, z, K, title):
    """Top-down (x-z) scatter of GT points. Correct planar conversion -> walls are straight lines;
    a missing ray->planar conversion bends them into arcs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = hd.unproject(z, K)[z > 0]
    P = P[np.random.default_rng(0).choice(len(P), min(len(P), 60000), replace=False)]
    fig, a = plt.subplots(figsize=(6, 6))
    a.scatter(P[:, 0], P[:, 2], s=0.2, c=P[:, 1], cmap="viridis")
    a.set_aspect("equal"); a.set_xlabel("x (m)"); a.set_ylabel("z (m)"); a.set_title(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypersim_root", required=True)
    ap.add_argument("--scenes", required=True, help="scenes_test.txt")
    ap.add_argument("--vggt_repo", required=True, help="path to the cloned facebookresearch/vggt repo")
    ap.add_argument("--weights", required=True, help="VGGT-1B model.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=None, help="groups.json (default: <out>/groups.json; created if missing)")
    ap.add_argument("--vis_groups", type=int, default=3, help="save figures for the first N groups")
    ap.add_argument("--limit_groups", type=int, default=0, help="debug: only run the first N groups")
    ap.add_argument("--device", default="cuda", help="cuda (default); cpu only for debugging")
    args = ap.parse_args()

    sys.path.insert(0, args.vggt_repo)
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    os.makedirs(os.path.join(args.out, "pred"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    scenes = [l.strip() for l in open(args.scenes) if l.strip()]
    groups = hd.load_or_make_groups(args.groups or os.path.join(args.out, "groups.json"),
                                    args.hypersim_root, scenes)
    if args.limit_groups:
        groups = groups[:args.limit_groups]

    dev = args.device
    if dev == "cuda":
        assert torch.cuda.is_available(), "run this on a GPU node"
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.bfloat16
    model = VGGT()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval().to(dev)
    K = hd.intrinsics()
    print(f"input {hd.W1}x{hd.H1}, K={K}, dtype={dtype}", flush=True)

    rows = []
    t0 = time.time()
    for gi, g in enumerate(groups):
        paths = [hd.frame_paths(args.hypersim_root, g["scene"], f) for f in g["frames"]]
        images = load_and_preprocess_images([p[0] for p in paths], mode="crop")
        assert tuple(images.shape) == (len(paths), 3, hd.H1, hd.W1), images.shape
        with torch.no_grad(), torch.autocast(device_type=dev, dtype=dtype):
            out = model(images.to(dev))
        depth = out["depth"][0, ..., 0].float().cpu().numpy()       # (S, H, W)
        conf = out["depth_conf"][0].float().cpu().numpy()          # (S, H, W)
        assert depth.shape == (len(paths), hd.H1, hd.W1), depth.shape
        np.savez_compressed(os.path.join(args.out, "pred", f"{g['scene']}_g{g['group']}.npz"),
                            depth=depth.astype(np.float16), conf=conf.astype(np.float16),
                            frames=np.array(g["frames"]))

        for i, (f, (_, dep_path)) in enumerate(zip(g["frames"], paths)):
            z = hd.load_gt_depth(dep_path)
            m = frame_metrics(depth[i], z, K)
            if m is None:
                print(f"skip {g['scene']} frame {f}: too few valid pixels", flush=True)
                continue
            rows.append({"scene": g["scene"], "group": g["group"], "frame": f, **m})
            if gi < args.vis_groups:
                rgb = images[i].permute(1, 2, 0).numpy()
                tag = f"{g['scene']}_f{f:04d}"
                save_vis(os.path.join(args.out, "vis", f"{tag}.png"), rgb, z, depth[i], tag)
                if i == 0:
                    save_topdown(os.path.join(args.out, "vis", f"topdown_{tag}.png"), z, K, tag)
        print(f"[{gi + 1}/{len(groups)}] {g['scene']} g{g['group']} frames {g['frames']} "
              f"absrel={np.mean([r['absrel_all'] for r in rows[-len(paths):]]):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    metric_keys = [k for k in keys if k not in ("scene", "group", "frame", "n_valid")]
    lines = [f"frames={len(rows)} scenes={len(set(r['scene'] for r in rows))}",
             f"{'metric':22s} {'frame_mean':>11s} {'scene_mean':>11s}"]
    for k in metric_keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        per_scene = [np.nanmean([r[k] for r in rows if r["scene"] == s]) for s in sorted(set(r["scene"] for r in rows))]
        lines.append(f"{k:22s} {np.nanmean(vals):11.5f} {np.nanmean(per_scene):11.5f}")
    text = "\n".join(lines)
    open(os.path.join(args.out, "summary.txt"), "w").write(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
