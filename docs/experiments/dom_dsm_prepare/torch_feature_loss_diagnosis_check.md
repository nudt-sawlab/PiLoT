# P6 - Torch feature-loss diagnosis without direct_abs_cost_cuda

## Purpose

The local GTX 1080 cannot run the current `direct_abs_cost_cuda` optimizer/loss path, so P4/P5 `overall_loss_all_zero=true` cannot be trusted. This experiment bypasses the CUDA extension and does not run optimizer updates. It uses PyTorch-only feature residuals to compare fixed candidate poses in PiLoT feature space and DOM/DSM visual metrics.

## Why Not Use direct_abs_cost_cuda

- Current GPU: `NVIDIA GeForce GTX 1080`, compute capability `sm_61`.
- Loaded `direct_abs_cost_cuda` binary marker: `sm_86` only from the binary check.
- Runtime full refinement prints `Kernel launch failed: no kernel image is available for execution on the device`.
- P4/P5 reported `overall_loss` as all zeros.
- Therefore this experiment does not call `residual_jacobian_batch_quat_cuda`, does not call `optimizer_step_cuda`, and does not use learned optimizer pose updates.

## Candidate Poses

| Candidate | East | North | Alt | Euler |
| --- | ---: | ---: | ---: | --- |
| initial | 0.000000 | 0.000000 | 0.000000 | `[0.0, 180.0, 29.2]` |
| p4_scale_025_fixed_alt | -3.301955 | -0.526292 | 0.000000 | `[0.0, 180.0, 29.2]` |
| p3_best_chamfer | -5.000000 | 0.000000 | 0.000000 | `[0.0, 180.0, 29.2]` |
| p3_best_overlap | -5.000000 | 5.000000 | 0.000000 | `[0.0, 180.0, 29.2]` |
| raw_refined_translation_fixed_alt | -13.207800 | -2.105200 | 0.000000 | `[0.0, 180.0, 29.2]` |
| raw_refined_translation_full_alt | -13.207800 | -2.105200 | 4.097000 | `[0.0, 180.0, 29.2]` |
| raw_refined_full | -13.207800 | -2.105200 | 4.097000 | `[180.0, 0.0, -144.8]` |

All east/north/alt offsets were applied through `pose_adapter.apply_enu_offset()` in the DOM/DSM projected CRS. No lon/lat scale approximation was used.

## Pose Roundtrip Check

- roundtrip passed: `True`
- raster CRS: `EPSG:32650`
- east/north errors were below `1e-3m`; altitude errors were below `1e-6m`.
- `make_domdsm_downward_euler(29.2)` returned `[0.0, 180.0, 29.2]`.
- refined Euler `[180, 0, -144.8]` was recorded only. Euler convention still needs separate handling, so all candidates except `raw_refined_full` use fixed downward euler `[0, 180, 29.2]`.

## Feature-loss vs Visual Metric

| Candidate | East | North | Alt | Visual overlap | Visual chamfer | Torch feature loss | Rank by visual | Rank by feature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 0.000000 | 0.000000 | 0.000000 | 0.629278 | 5.493694 | 0.151881 | 4 | 2 |
| p4_scale_025_fixed_alt | -3.301955 | -0.526292 | 0.000000 | 0.801166 | 4.506433 | 0.160338 | 3 | 4 |
| p3_best_chamfer | -5.000000 | 0.000000 | 0.000000 | 0.886565 | 3.934921 | 0.157648 | 2 | 3 |
| p3_best_overlap | -5.000000 | 5.000000 | 0.000000 | 1.077362 | 3.148854 | 0.146716 | 1 | 1 |
| raw_refined_translation_fixed_alt | -13.207800 | -2.105200 | 0.000000 | 0.395933 | 6.164812 | 0.238604 | 5 | 6 |
| raw_refined_translation_full_alt | -13.207800 | -2.105200 | 4.097000 | 0.331206 | 6.853325 | 0.236013 | 6 | 5 |
| raw_refined_full | -13.207800 | -2.105200 | 4.097000 | 0.307403 | 9.151419 | 0.322350 | 7 | 7 |

Spearman feature loss vs chamfer: `0.857143`.
Spearman feature loss vs overlap: `-0.857143`.

Feature loss ranks `p3_best_overlap` first and rejects both raw refined translations and raw refined full. Visual metrics rank `p3_best_overlap`, `p3_best_chamfer`, and `p4_scale_025_fixed_alt` above initial, while raw refined candidates are worse. This matches situation A: PiLoT feature signal is useful, but the CUDA optimizer/update path is invalid on the current GPU.

## Residual Visualization Analysis

Residual overlays and histograms were saved per candidate under `docs/experiments/dom_dsm_prepare/torch_feature_loss_diagnosis/<candidate>/`. The raw refined full candidate has the largest torch feature loss (`0.322350`) and worst visual chamfer (`9.151419`), indicating that residuals grow when pose convention and translation are both applied. The small/west-north candidate `p3_best_overlap` has the lowest feature loss (`0.146716`) and best visual metrics, so the feature space contains an alignment signal independent of the broken CUDA extension.

## Local Finite-difference Diagnosis

| Perturbation | Feature loss | Visual chamfer | Direction |
| --- | ---: | ---: | --- |
| initial | 0.151881 | 5.493694 | baseline |
| east -1m | 0.164833 | 5.420573 |  |
| east +1m | 0.168955 | 5.080698 |  |
| north -1m | 0.168750 | 5.654038 |  |
| north +1m | 0.154562 | 4.645459 |  |
| yaw -1deg | 0.156184 | 5.044136 |  |
| yaw +1deg | 0.188670 | 5.567031 |  |
| alt -1m | 0.164894 | 5.245727 |  |
| alt +1m | 0.152001 | 6.065792 |  |

Estimated local direction: `{'alt': 'flat', 'east': 'flat', 'north': 'flat', 'yaw': 'flat'}`.

At the ±1m/±1deg scale, feature loss is lowest at initial, so the local finite difference is effectively flat. Visual chamfer improves for some one-step perturbations, especially north +1m and yaw -1deg, but feature loss does not produce a strong local descent direction from initial at this small step size. This means the feature signal is visible across fixed candidate poses but weak or non-smooth locally around initial with this sampling setup.

## Interpretation

- PiLoT feature loss also considers raw refined worse: yes. `raw_refined_full` has the worst feature loss, and raw refined translations are much worse than initial/small-offset candidates.
- PiLoT feature loss prefers an initial-neighborhood candidate: yes. It ranks `p3_best_overlap` best and keeps `p3_best_chamfer` / `p4_scale_025_fixed_alt` ahead of raw refined candidates.
- Feature loss and visual metric are aligned: yes for fixed candidates. Spearman correlation is strong positive versus chamfer and strong negative versus overlap.
- Pose/CRS conversion is not the blocker: roundtrip checks passed.
- Refined degradation is most consistent with CUDA optimizer/update path failure on the current GPU. Feature-domain mismatch is not the primary explanation for the fixed-candidate ranking, though local gradients are weak and should not be overinterpreted as an optimizer-ready signal.

## Conclusion

Current CUDA optimizer output is not trustworthy on GTX 1080. The PyTorch feature-loss diagnosis shows that PiLoT's feature space does contain a useful alignment signal independent of the failed CUDA extension: it rejects raw refined poses and ranks the visually better small-offset candidate best. The immediate priority is to fix the CUDA loss/update path or implement a reliable PyTorch optimizer/debug fallback. After that, DOM/DSM-specific constraints such as freezing altitude/pitch/roll and a trust-region update should be evaluated inside the refinement loop rather than as an external search algorithm.

## Artifacts

- summary: `docs/experiments/dom_dsm_prepare/torch_feature_loss_diagnosis/summary_metrics.json`
- pose roundtrip: `docs/experiments/dom_dsm_prepare/torch_feature_loss_diagnosis/pose_roundtrip.json`
- local gradient: `docs/experiments/dom_dsm_prepare/torch_feature_loss_local_gradient/local_gradient_metrics.json`
