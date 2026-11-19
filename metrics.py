"""Per-frame metrics for the head study. All heads must be evaluated with these functions.

Definitions (fixed in advance):
- Scale: per-frame median alignment, s = median(gt / pred) over valid pixels.
- Boundary band: GT depth discontinuities dilated by a disk of radius r (default 3 px).
  A discontinuity between neighbours i, i+1 requires |dlog z| > thr (default 0.05) AND
  |dlog z| > 3 x the larger neighbouring step along the same axis, so smooth slanted
  surfaces (constant log-depth slope) are not counted as edges.
- Interior: valid & not boundary.  Far: valid & z >= 80th percentile of the frame.
- Flying point: boundary-band pixel whose 3D error exceeds 5% of its GT depth.
- Coarse/fine split of the relative error r = (pred_s - z) / z over valid pixels:
  r_low = mean of r over the valid pixels of each 14x14 patch, r_high = r - r_low.
  Within each patch r_high sums to zero, so mean(r^2) = mean(r_low^2) + mean(r_high^2) exactly.
"""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from hypersim_data import PATCH, unproject


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x ** 2 + y ** 2 <= r ** 2


def _edges_along_axis1(lz, valid, thr):
    d = np.abs(np.diff(lz, axis=1))                      # (H, W-1)
    ok = valid[:, 1:] & valid[:, :-1]
    d = np.where(ok, d, 0.0)
    left = np.pad(d, ((0, 0), (1, 0)))[:, :-1]           # previous step
    right = np.pad(d, ((0, 0), (0, 1)))[:, 1:]          # next step
    jump = ok & (d > thr) & (d > 3 * np.maximum(left, right))
    edge = np.zeros_like(valid)
    edge[:, 1:] |= jump
    edge[:, :-1] |= jump
    return edge


def depth_edges(z, valid, thr=0.05):
    lz = np.log(np.where(valid, z, 1.0))
    return _edges_along_axis1(lz, valid, thr) | _edges_along_axis1(lz.T, valid.T, thr).T


def regions(z, valid, thr=0.05, r=3):
    band = binary_dilation(depth_edges(z, valid, thr), structure=disk(r)) & valid
    far = valid & (z >= np.quantile(z[valid], 0.8)) if valid.any() else valid
    return {"all": valid, "band": band, "interior": valid & ~band, "far": far}


def coarse_fine(r, valid, patch=PATCH):
    """Exact orthogonal split of mean(r^2) over valid pixels into patch-mean and within-patch parts."""
    h, w = r.shape
    assert h % patch == 0 and w % patch == 0, (h, w)
    rv = np.where(valid, r, 0.0).reshape(h // patch, patch, w // patch, patch)
    cnt = valid.reshape(h // patch, patch, w // patch, patch).sum(axis=(1, 3), keepdims=True)
    mean = rv.sum(axis=(1, 3), keepdims=True) / np.maximum(cnt, 1)
    r_low = np.broadcast_to(mean, rv.shape).reshape(h, w)
    r_high = r - r_low
    n = valid.sum()
    total = float((r[valid] ** 2).sum() / n)
    low = float((r_low[valid] ** 2).sum() / n)
    high = float((r_high[valid] ** 2).sum() / n)
    assert abs(total - low - high) <= 1e-6 * max(total, 1e-12) + 1e-12, (total, low, high)
    return total, low, high


def frame_metrics(pred, z, K, chamfer=True):
    """pred, z: (H, W) depth at the same resolution; z == 0 marks invalid GT."""
    valid = (z > 0) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 100:
        return None
    s = float(np.median(z[valid] / pred[valid]))
    p = pred * s
    reg = regions(z, valid)
    rel = np.abs(p - z) / np.where(valid, z, 1.0)
    P, G = unproject(p, K), unproject(z, K)
    err3 = np.linalg.norm(P - G, axis=-1)
    rel3 = err3 / np.where(valid, z, 1.0)

    out = {"scale": s, "n_valid": int(valid.sum())}
    for name, m in reg.items():
        out[f"frac_{name}"] = float(m.sum() / valid.sum())
        out[f"absrel_{name}"] = float(rel[m].mean()) if m.any() else np.nan
        out[f"pt_relerr_{name}"] = float(rel3[m].mean()) if m.any() else np.nan
    ratio = np.maximum(p[valid] / z[valid], z[valid] / p[valid])
    out["delta1"] = float((ratio < 1.25).mean())
    out["flying_band"] = float((rel3[reg["band"]] > 0.05).mean()) if reg["band"].any() else np.nan

    r = np.where(valid, (p - z) / np.where(valid, z, 1.0), 0.0)
    out["mse_rel"], out["mse_rel_coarse"], out["mse_rel_fine"] = coarse_fine(r, valid)

    if chamfer:
        a, b = P[valid], G[valid]
        med = float(np.median(z[valid]))
        acc = cKDTree(b).query(a, workers=-1)[0]
        comp = cKDTree(a).query(b, workers=-1)[0]
        out["acc_rel"] = float(acc.mean() / med)    # Chamfer terms in units of the frame's median depth
        out["comp_rel"] = float(comp.mean() / med)
    return out
