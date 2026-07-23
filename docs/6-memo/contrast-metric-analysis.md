# Contrast Metric & Crop↔Mosaic Correlation Analysis

Date: 2026-07-23. Data: 64 auto-sampled flake/substrate colour pairs from the
July-7 (41) and July-13 (23) per-flake crops, joined to Notion AFM thicknesses.
All colours BGR. This memo answers two questions posed before wiring the mosaic
pipeline: (1) can beige zoomed-in crops be correlated to their zoomed-out
mosaic counterparts, and (2) what contrast formulation is most foolproof under
nonuniform substrate and varying exposure.

## Q2 — Best contrast metric (the actionable result)

### Exposure / white-balance invariance (synthetic perturbation test)
A per-channel exposure/white-balance change is *multiplicative* (F→kF, S→kS).

| metric | gain ×1.25 | WB (G×1.15) | black-level +12 |
|---|---|---|---|
| `S − F` (plain diff) | Δ=17.1 | Δ=10.2 | Δ=0 |
| `(S−F)/S` (current) | **0** | **0** | Δ=0.056 |
| `(S−F)/(S+F)` (ndi) | **0** | **0** | **Δ=0.015** |
| `ln(S/F)` | **0** | **0** | Δ=0.032 |

→ Any ratio metric is perfectly invariant to exposure & white balance; plain
difference is not. Use a ratio. Among ratios, **normalized difference
`(S−F)/(S+F)` is the most robust to additive black-level offset** and is
bounded in [−1,1] (numerically stable when the local substrate is dark/small,
where `(S−F)/S` blows up).

### Thickness discrimination (d′ on the G channel, ndi)
`(S−F)/(S+F)` also has the best separation of the 35–45nm target band from
off-band: d′(target vs thin)=0.87, d′(target vs thick)=0.50 — beating
`(S−F)/S` (0.80/0.33) and plain diff (0.60/0.23).

### Channel roles (important)
Per-channel d′ separating target(35–45) from thick(≥100), and correlation of
ndi with thickness:

| channel | d′ target-vs-thick | corr(ndi, thickness) | role |
|---|---|---|---|
| G | 0.50 | −0.33 | **hBN presence** — large signal, weak thickness info |
| B | 1.72 | — | intermediate |
| **R** | **3.53** | **−0.74** | **thickness** — the channel that rejects thick flakes |

The R channel encodes thickness (this generalizes the graphite "contrast_R
tracks thickness" finding to hBN). G tells you *it's hBN*; R tells you *how
thick*.

### Classifier: use Mahalanobis, NOT SNR weighting
A 3D **Mahalanobis distance** from the target-band centroid (using the target
band covariance) separates cleanly: target 1.15±0.69, thin 1.80, thick 3.14 —
a threshold ≈1.8–2.0 isolates the target band.

⚠️ This overturns the plan's §8a weighting. The SNR weight
`mean²/(std²+floor)` **down-weights R** (its mean contrast is small) — but R is
the very channel that discriminates thickness. SNR weighting optimizes for
"matches the mean signature," not "is this the right thickness." **Mahalanobis
distance uses all three channels correctly and needs no hand-tuned weights.**

### Recommendation for the mosaic detector
1. Contrast = **normalized difference `(S−F)/(S+F)`** per channel.
2. Substrate = **local** (coarse grid, interpolated) — within-session substrate
   std is 25–34 counts (illumination falloff), so a single global substrate is
   wrong; ndi's boundedness makes local-and-sometimes-dark substrate safe.
3. Classify with **Mahalanobis distance from the target-band centroid** using
   the band covariance; tune the threshold on the July-7/C validation.
4. Optional: subtract an estimated black level (mosaic dark border) before ndi
   to remove the one residual sensitivity (additive offset).
5. Robustify calibration stats (median / robust covariance) — see caveat below.

### Caveat
Even for the invariant metrics, the same 35–45nm band differed ~1.4σ between
July-7 and July-13 crops, driven mostly by July-13's high variance (rel_G std
0.37 vs 0.07) — likely noisy crop border-substrate estimation and a few
outliers. Moving to in-domain mosaic calibration + local substrate + robust
statistics should tighten this. Watch for outlier calibration samples.

## Q1 — Correlating crops to mosaics

- **Magnification**: crop ≈ 0.066 µm/px (measured from the 20 µm scale bar);
  mosaic ≈ 1.1 µm/px (8529 px across a ~1 cm chip) → **~16× ratio**. A 36 µm
  flake is ~540 px in a crop but only ~33 px on the mosaic.
- **Automatic correlation is hard / unreliable**: at ~33 px the flake is a tiny
  blob among hundreds of similar blobs; colours differ across domains (beige vs
  purple substrate, and absolute flake colour differs), so intensity template
  matching fails and only shape matching could work — degraded further by
  rotation and JPEG artifacts. → **Manual coordinate marking (what the user is
  doing) is the right call**, not a workaround to be automated away.
- **What crops ARE good for**: same-domain color-signature validation
  (milestone 1 hit 100% recall). Once the user marks mosaic coordinates, we get
  crop↔mosaic correspondences and can *measure* the crop→mosaic colour transform
  directly (and potentially learn a mapping so historical crop-only flakes can
  seed mosaic calibration later).
- **Feasible semi-automation** (future): given an approximate region (a minimap
  / rough stage position), snap to the nearest flake blob — reliable, unlike
  blind full-mosaic search.

## Net effect on the plan
`F_0.2.0` §8a (weighted Euclidean) should be revised to: ndi contrast + local
substrate + Mahalanobis classification. To apply when the clean rescans + marked
coordinates land.
