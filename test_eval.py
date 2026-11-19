"""Fast checks for hypersim_data.py and metrics.py (no GPU, no real data). Run: python test_eval.py"""
import os
import tempfile

import h5py
import numpy as np
from PIL import Image

import hypersim_data as hd
from metrics import coarse_fine, depth_edges, frame_metrics, regions

K = hd.intrinsics()


def test_sizes():
    assert (hd.H1, hd.W1) == (392, 518) and hd.H1 % 14 == 0 and hd.W1 % 14 == 0


def test_ray_to_planar_fronto_parallel_plane():
    u = np.arange(hd.W0) + 0.5 - hd.W0 / 2
    v = np.arange(hd.H0) + 0.5 - hd.H0 / 2
    x, y = np.meshgrid(u, v)
    dist = 3.0 * np.sqrt(x ** 2 + y ** 2 + hd.F0 ** 2) / hd.F0   # plane z = 3 m
    assert np.allclose(hd.ray_to_planar(dist), 3.0, atol=1e-5)


def test_load_gt_depth_nan_and_nearest():
    with tempfile.TemporaryDirectory() as d:
        dist = np.full((hd.H0, hd.W0), 2.0, np.float32)
        dist[:, : hd.W0 // 2] = 5.0                 # sharp step
        dist[:10, :10] = np.nan                     # invalid
        p = os.path.join(d, "x.hdf5")
        with h5py.File(p, "w") as f:
            f["dataset"] = dist
        z = hd.load_gt_depth(p)
        assert z.shape == (hd.H1, hd.W1)
        assert z[0, 0] == 0                          # NaN -> invalid
        zz = z[z > 0]
        planar5 = hd.ray_to_planar(np.full((hd.H0, hd.W0), 5.0))
        planar2 = hd.ray_to_planar(np.full((hd.H0, hd.W0), 2.0))
        # nearest resize: every value must come from the native map (no blended depths)
        native = np.concatenate([planar5[:, : hd.W0 // 2].ravel(), planar2[:, hd.W0 // 2:].ravel()])
        assert zz.min() >= native.min() - 1e-4 and zz.max() <= native.max() + 1e-4
        assert not np.any((zz > planar2.max() + 1e-3) & (zz < planar5.min() - 1e-3))


def test_edges_flat_ramp_step():
    valid = np.ones((28, 37), bool)
    assert not depth_edges(np.full((28, 37), 2.0), valid).any()             # flat: none
    ramp = np.exp(np.linspace(0, 0.08 * 36, 37))[None].repeat(28, 0)          # 8% per px slope
    assert not depth_edges(ramp, valid).any()                                 # smooth slope: none
    step = np.full((28, 37), 2.0); step[:, 20:] = 4.0
    e = depth_edges(step, valid)
    assert e[:, 19].all() and e[:, 20].all() and e.sum() == 2 * 28          # exactly the two sides
    invalid = valid.copy(); invalid[:, 20:] = False
    assert not depth_edges(step, invalid).any()                               # no edge into invalid


def test_coarse_fine_identity():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(28, 42)); valid = rng.random((28, 42)) > 0.3
    t, lo, hi = coarse_fine(r, valid)
    assert abs(t - lo - hi) < 1e-9
    const = np.full((28, 42), 0.1)                       # patch-constant error -> all coarse
    t, lo, hi = coarse_fine(const, np.ones_like(valid))
    assert hi < 1e-12 and abs(lo - 0.01) < 1e-12


def test_perfect_and_scaled_prediction():
    z = np.full((hd.H1, hd.W1), 3.0, np.float32); z[:, 300:] = 6.0; z[:5] = 0
    for pred in (z.copy(), z * 0.37):                    # perfect, and perfect up to scale
        pred = np.where(z > 0, pred, 1.0)
        m = frame_metrics(pred, z, K)
        assert m["absrel_all"] < 1e-6 and m["flying_band"] == 0 and m["delta1"] == 1
        assert m["mse_rel"] < 1e-10 and m["acc_rel"] < 1e-6
    blurred = z.copy(); blurred[:, 298:302] = 4.5        # 4 px of in-between depth at the edge
    m = frame_metrics(np.where(z > 0, blurred, 1.0), z, K)
    assert m["flying_band"] > 0 and m["absrel_interior"] < 1e-6 and m["absrel_band"] > 0


def test_groups():
    with tempfile.TemporaryDirectory() as d:
        for f in range(100):
            img, dep = hd.frame_paths(d, "sc", f)
            os.makedirs(os.path.dirname(img), exist_ok=True); os.makedirs(os.path.dirname(dep), exist_ok=True)
            open(img, "w").close(); open(dep, "w").close()
        g = hd.make_groups(d, ["sc"])
        assert len(g) == 5
        used = [f for x in g for f in x["frames"]]
        assert len(used) == len(set(used)) == 20                                # non-overlapping
        assert all(np.all(np.diff(x["frames"]) == 5) for x in g)
        assert g == hd.make_groups(d, ["sc"])                                   # deterministic


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("all tests passed")
