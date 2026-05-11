# P5 - DOMDSM-aware PiLoT refinement adaptation

## Purpose

This experiment moves away from using external grid search or line search as the main method. P3/P4 showed that PiLoT's refined update contains a useful direction signal but the raw update overshoots and altitude is unstable. P5 adds DOM/DSM-specific inputs around PiLoT's internal refinement path: CRS-aware pose offsets, structure-aware 3D point sampling, an opt-in DOM/DSM back-project wrapper, an interpretable candidate pose generator, and single/batch diagnostic scripts.

The goal is not to declare PiLoT unsuitable for DOM/DSM. The goal is to make its inputs and constraints match DOM/DSM geometry and then inspect whether the learned optimizer output becomes more reasonable.

## PiLoT optimization principle

In this DOM/DSM path, `DOMDSMRenderer` renders an RGB/depth reference view from the current pose. The depth image is back-projected into 3D points. The rendered RGB and query image are passed through PiLoT's feature extractor, and the learned optimizer updates the query pose from feature residuals/Jacobians over the sampled 3D points. Therefore DOM/DSM adaptation should focus on feature-domain compatibility, 3D point quality, local pose parameterization, and degree-of-freedom constraints.

## Implemented Changes

- `pixloc/utils/dom_dsm/pose_adapter.py`: DOM/DSM CRS transformers, WGS84 <-> DOM XY conversion, ENU meter offsets, euler normalization entrypoints.
- `pixloc/utils/dom_dsm/point_sampling.py`: uniform, rendered DOM gradient, depth gradient, and combined structure-aware point sampling with debug visualization.
- `pixloc/utils/dom_dsm/domdsm_refine.py`: opt-in DOM/DSM back-project wrapper that samples structured 2D points and then calls existing `sample_3d_points()`.
- `pixloc/utils/dom_dsm/pose_candidates.py`: DOM/DSM east/north/alt/yaw candidate list generation. PixLoc `Pose` batch conversion is intentionally left as a TODO pending axis-sign validation.
- `tools/run_domdsm_aware_refinement_single.py`: single-query comparison of baseline PiLoT and DOM/DSM-aware sampling modes with optional freeze-alt/freeze-pitch-roll evaluation.
- `tools/run_domdsm_aware_refinement_batch.py`: small-batch wrapper and aggregate metrics; it records when the pose file is too small for real batch evaluation.

## Single-query Results

Command:

```bash
./.conda/pilot22/bin/python tools/run_domdsm_aware_refinement_single.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_single \
  --width 512 \
  --sampling-modes uniform dom_gradient depth_gradient combined \
  --freeze-alt \
  --freeze-pitch-roll
```

| Method | Sampling | Freeze alt | Freeze pitch/roll | East | North | Alt | Overlap | Chamfer |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| initial | none | false | false | 0.000000 | 0.000000 | 0.000000 | 0.629278 | 5.493694 |
| baseline refined | legacy_random | false | false | -13.207826 | -2.105169 | 4.097170 | 0.306725 | 9.169733 |
| uniform | uniform | true | true | -13.207822 | -2.105164 | 0.000000 | 0.395933 | 6.163984 |
| dom_gradient | dom_gradient | true | true | -13.207822 | -2.105172 | 0.000000 | 0.395933 | 6.163984 |
| depth_gradient | depth_gradient | true | true | -13.207831 | -2.105160 | 0.000000 | 0.395933 | 6.163984 |
| combined | combined | true | true | -13.207825 | -2.105166 | 0.000000 | 0.395933 | 6.163984 |

Runtime notes:

- GPU: `NVIDIA GeForce GTX 1080`, compute capability `[6, 1]`.
- CUDA/PyTorch: `torch 2.4.1+cu124`, CUDA `12.4`.
- Runtime still printed `Kernel launch failed: no kernel image is available for execution on the device`.
- `overall_loss_all_zero = true` for baseline and all sampling modes, so optimizer loss is not treated as trustworthy.

## Batch Results

Command:

```bash
./.conda/pilot22/bin/python tools/run_domdsm_aware_refinement_batch.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --image-dir data_caiwangcun/query/images/exif_test_16x9 \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_batch \
  --max-images 10 \
  --sampling-mode combined \
  --width 512 \
  --freeze-alt \
  --freeze-pitch-roll
```

Current pose file has only one image, so real batch evaluation is not yet available.

| Metric | Value |
| --- | ---: |
| num_images | 1 |
| initial_mean_chamfer | 5.493694 |
| refined_mean_chamfer | 6.163984 |
| improve_rate | 0.000000 |
| median_chamfer_delta | 0.670290 |
| median_overlap_delta | -0.233345 |
| mean_translation_update_m | 13.988042 |
| alt_drift_mean_m | 4.097192 |

## Analysis

Structured sampling did not improve PiLoT's raw output in this run. Uniform, DOM-gradient, depth-gradient, and combined sampling all returned essentially the same refined translation: about east `-13.2078m`, north `-2.1052m`, raw alt drift `+4.097m`. After applying the requested DOM/DSM safety constraints for evaluation (freeze alt and freeze pitch/roll), all four modes produce overlap `0.395933` and chamfer `6.163984`, which is better than raw full baseline but still worse than yawfix initial.

Freeze-alt reduces the severity of the baseline failure. Baseline raw refined full has chamfer `9.169733`; evaluating the same large translation with initial altitude and initial downward rotation gives chamfer `6.163984`. This is consistent with P4: altitude update is harmful, but freezing altitude alone does not fix the overshoot in east/north.

The update magnitude is not more reasonable yet. The DOM/DSM-aware sampling wrapper changes the selected 2D points and provides sampling debug images, but the optimizer still outputs the same large update magnitude of about `13.99m`. That means this first adaptation layer is not enough while the optimizer loss path is compromised.

The current primary blocker is CUDA/loss reliability, followed by pose parameterization. Since `direct_abs_cost_cuda` still reports a GTX 1080 kernel image error and `overall_loss` is all zeros, the learned optimizer's residual/loss/update path is not trustworthy in this environment. The candidate generation module now provides DOM/DSM-local east/north/yaw candidates, but it is not yet connected to PixLoc `Pose` batch construction; that is the next internal adaptation step after fixing or bypassing the CUDA loss issue.

## Conclusion

Baseline PiLoT assumes rendered reference and query feature maps are sufficiently aligned in appearance, and that randomly sampled depth points provide stable geometry. DOM/DSM violates part of these assumptions. The P5 code adds the DOM/DSM adaptation hooks needed to address CRS-aware pose updates, structured sampling, and constrained DOF evaluation, but this single-query run does not yet improve stability over yawfix initial. The main reason is that PiLoT's optimizer still returns the same overshooting translation under every sampling mode, while CUDA loss is all zeros. The PiLoT optimizer should not be discarded; the next useful step is to make the optimizer's DOM/DSM candidate pose batch and loss path reliable, then test trust-region/freezing policies inside the refinement acceptance logic rather than as an external line search.

## Artifacts

- single summary: `docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_single/summary_metrics.json`
- single visuals: `docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_single/`
- batch summary: `docs/experiments/dom_dsm_prepare/domdsm_aware_refinement_results_batch/batch_summary.json`
- batch note: `pose file has only one image; batch evaluation not yet available`
