# What does VGGT's dense head actually compute?

## 1. Core question

3D foundation models are often described as a backbone that holds the geometry plus heads that "simply act as format converters". VGGT's DPT head reads only backbone tokens (layers 4, 11, 17, 23), never the image, so it cannot add new information. The open question is **how heavy that conversion is**:

> How much of VGGT's point-cloud quality comes from the simplest possible readout (a linear map of last-layer tokens)? Which extra computation does the rest need: nonlinear decoding, intermediate-layer fusion, spatial context, or high-resolution upsampling? Where in the point cloud does each one matter?

## 2. Data

**Hypersim** only (synthetic indoor scenes with exact dense depth).

| | |
|---|---|
| Scene filter | "Strict-standard" camera: 1024×768, f = 886.81 px, square pixels, centered principal point (284 of 482 scenes) |
| Camera / frames | `cam_00` only; scenes with all 100 frames publicly released |
| Split | Official scene split (`metadata_images_split_scene_v1.csv`), sampled with seed 0: **25 train scenes** (from `train`) for training heads, **15 test scenes** (from `test`) for evaluation only |
| Files per frame | `tonemap.jpg` (RGB), `depth_meters.hdf5` (depth) |
| Preprocessing | Ray distance → planar depth at native resolution; resize to 518×392 (VGGT `crop` mode) with **nearest neighbour** for depth; intrinsics scaled per axis |
| Eval groups | Per test scene, 5 non-overlapping groups of 4 frames (stride 5) → **75 groups, 300 frames**, fixed in `groups.json` |

Hypersim is part of VGGT's training data, so absolute numbers are optimistic. All heads share the same frozen backbone, so comparisons between heads are unaffected.

## 3. Experiment design

The **VGGT-1B backbone is frozen**, and tokens from layers 4/11/17/23 are cached. Five depth heads are trained on the cached tokens with identical loss (VGGT's confidence-weighted depth loss with gradient term), optimizer, steps, and 2 seeds each. Each step up the ladder adds exactly one component.

| Head | Input | Architecture | Adds |
|---|---|---|---|
| H1 | layer 23 | per-token linear → 14×14 pixel shuffle | baseline readout |
| C | layer 23 | per-token 2-layer MLP (hidden 2048) → pixel shuffle | nonlinear decoding only (capacity control) |
| H2 | layers 4/11/17/23 | per-token linear → pixel shuffle | intermediate layers |
| H3 | layers 4/11/17/23 | 2 × 3×3 residual conv blocks at token resolution → linear → pixel shuffle | spatial context |
| H4 | layers 4/11/17/23 | official DPT architecture, re-initialized | progressive high-res upsampling |
| Ref | — | official pretrained DPT head (no training) | protocol check |

**DPT head anatomy** (VGGT depth head, 518×392 input = 28×37 tokens; measured on one frame, fp32):

| Stage | Operation | Resolution | Params |
|---|---|---|---|
| Read-out | LayerNorm + 1×1 conv per tapped layer (4/11/17/23) | 28×37 | 5.8M |
| Pyramid | layer 4 ×4 (transposed conv, k=s=4), layer 11 ×2, layer 17 ×1, layer 23 ↓2 (3×3, stride 2) | 112×148 … 14×19 | 11.5M |
| Fusion | 3×3 conv to 256 ch, then 4 RefineNet blocks top-down (residual 3×3 convs, bilinear ×2) | 14×19 → 224×296 | 15.0M |
| Output | 3×3 conv (128 ch) → bilinear to full res → 3×3 conv (32 ch) → 1×1 → depth, conf | 224×296 → 392×518 | 0.3M |

Total 32.7M params, ≈460 MB of training activations per frame. **76% of the activations sit at ≥112×148**, in stages that hold only ~16% of the params.

Optional variant **H4-lite**: DPT truncated after the 112×148 fusion stage, bilinear to full res, same output convs. It splits the H4 − H3 step into multi-scale fusion (H4-lite − H3) and high-res convs (H4 − H4-lite, the part VGGT-Ω removed).

| Comparison | What it shows |
|---|---|
| C − H1 | gain from nonlinear decoding of the last layer |
| H2 − H1 | information in intermediate layers that the last layer lacks |
| H3 − H2 | value of neighbouring tokens (spatial context) |
| H4 − H3 | value of high-res convolutions (what VGGT-Ω removed) |
| H4 vs Ref | sanity gate: H4 must be within 10% AbsRel of Ref, otherwise training is insufficient |

**Metrics** (`metrics.py`, per frame, median scale alignment, GT intrinsics):

- Region-wise AbsRel and 3D point error. Regions: **boundary band** (GT depth discontinuities, dilated 3 px), **interior**, **far** (top 20% depth).
- **Flying-point rate** in the band: share of points with 3D error > 5% of depth.
- **Coarse/fine split** of relative MSE: 14×14 patch means vs. within-patch residual. The split is exact: total = coarse + fine.
- Chamfer accuracy and completeness (normalized by median depth); δ1.

**Statistics.** 95% CIs from scene-level bootstrap (15 scenes). An effect counts only if its CI excludes 0 and it exceeds the seed-to-seed spread.

## 4. Results (TBD)

**Table 1. Global metrics and cost**

| Head | AbsRel ↓ | δ1 ↑ | Acc ↓ | Comp ↓ | Params | Train mem (GB) |
|---|---|---|---|---|---|---|
| Ref | 0.0171 | 0.9838 | 0.0067 | 0.0108 | 32.65M | — |
| H1 | | | | | | |
| C | | | | | | |
| H2 | | | | | | |
| H3 | | | | | | |
| H4 | | | | | | |

**Table 2. Where the error is (AbsRel by region, flying-point rate)**

| Head | All | Band | Interior | Far | Flying (band) ↓ |
|---|---|---|---|---|---|
| Ref | 0.0171 | 0.0588 | 0.0080 | 0.0285 | 0.191 |
| H1 | | | | | |
| C | | | | | |
| H2 | | | | | |
| H3 | | | | | |
| H4 | | | | | |

**Table 3. Coarse vs. fine error (relative MSE, ×10⁻³)**

| Head | Total | Coarse (patch-level) | Fine (within-patch) |
|---|---|---|---|
| Ref | 11.90 | 2.12 (18%) | 9.78 (82%) |
| H1 | | | |
| C | | | |
| H2 | | | |
| H3 | | | |
| H4 | | | |

Ref: official VGGT-1B depth head, 15 test scenes, 300 frames. Band = 16.7% of valid pixels.

**Table 4. Attribution: error reduction per added component (mean [95% CI]); needs trained heads H1–H4**

| Step | Band AbsRel | Interior AbsRel | Coarse MSE | Fine MSE | Flying | Δ Train mem |
|---|---|---|---|---|---|---|
| C − H1 (nonlinearity) | | | | | | |
| H2 − H1 (layers) | | | | | | |
| H3 − H2 (context) | | | | | | |
| H4 − H3 (high-res) | | | | | | |