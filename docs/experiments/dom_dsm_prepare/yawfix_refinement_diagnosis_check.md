# P4 - Diagnose PiLoT refinement update around yawfix initial

## Purpose

This experiment runs the normal PiLoT `RenderLocalizer.run_query()` path, then diagnoses why the returned refined pose is worse than the yawfix initial pose under DOM/DSM visual validation. It does not replace PiLoT with grid search; the line search only scales PiLoT's own refined translation update to inspect direction, step length, altitude update, pose conversion, and loss/metric consistency.

## Inputs

- config: `configs/caiwangcun_domdsm_16x9.yaml`
- query image: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- pose file: `data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt`
- base translation: `[114.4368608916, 30.3913609745, 391.462]`
- base euler pitch/roll/yaw: `[0.0, 180.0, 29.2]`
- renderer width: `512`
- line search scales: `[0.0, 0.25, 0.5, 0.75, 1.0]`
- alt modes: `['fixed_initial', 'scaled_refined', 'full_refined']`
- GPU: `NVIDIA GeForce GTX 1080`, capability `[6, 1]`

Command:

```bash
./.conda/pilot22/bin/python tools/diagnose_yawfix_refinement_update.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results \
  --width 512 \
  --line-scales 0 0.25 0.5 0.75 1.0 \
  --alt-modes fixed_initial scaled_refined full_refined
```

## Initial vs Raw Refined

| Candidate | East | North | Alt | Euler | Overlap | Chamfer |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| initial | 0.000000 | 0.000000 | 0.000000 | `[0.000, 180.000, 29.200]` | 0.629278 | 5.493694 |
| raw refined full | -13.207818 | -2.105168 | 4.097183 | `[180.000, -0.000, -144.800]` | 0.306725 | 9.169733 |
| refined translation + initial rotation | -13.207818 | -2.105168 | 4.097183 | `[0.000, 180.000, 29.200]` | 0.331206 | 6.853325 |

The direct raw refined pose is genuinely worse than the yawfix initial pose. Keeping the initial rotation while using the full refined translation is also worse, so the full translation step itself is damaging. The raw refined Euler expression further worsens chamfer, so pose convention/rotation representation remains suspicious but is not the only issue.

## PiLoT Refined Update

- refined delta east/north/alt: `[-13.207818, -2.105168, 4.097183]` m
- refined delta lon/lat/alt: `[-0.0001368516142150611, -2.1674723278408692e-05, 4.0971829335317125]`
- refined euler pitch/roll/yaw: `[179.99995341022662, -0.00011105648747170372, -144.79992977992384]`
- diff_R: `1.999977707862854`
- diff_t: `0.028450489044189453`
- selected candidate index: `None`
- optimizer overall loss: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- fail_list: `[False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]`

The run still printed `Kernel launch failed: no kernel image is available for execution on the device` on the GTX 1080. The returned `overall_loss` is all zeros, so optimizer loss cannot be trusted here as evidence of alignment improvement. Candidate-level debug was not added in this step; candidate conversion/loss ranking therefore remains not checked.

## Translation Line Search

| Scale | Alt mode | East | North | Alt | Overlap | Chamfer |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | fixed_initial | 0.000000 | 0.000000 | 0.000000 | 0.629278 | 5.493694 |
| 0.00 | scaled_refined | 0.000000 | 0.000000 | 0.000000 | 0.629278 | 5.493694 |
| 0.00 | full_refined | 0.000000 | 0.000000 | 4.097183 | 0.607249 | 5.710145 |
| 0.25 | fixed_initial | -3.301955 | -0.526292 | 0.000000 | 0.801166 | 4.506433 |
| 0.25 | scaled_refined | -3.301955 | -0.526292 | 1.024296 | 0.808729 | 5.328036 |
| 0.25 | full_refined | -3.301955 | -0.526292 | 4.097183 | 0.636875 | 5.408424 |
| 0.50 | fixed_initial | -6.603909 | -1.052584 | 0.000000 | 0.671865 | 4.511799 |
| 0.50 | scaled_refined | -6.603909 | -1.052584 | 2.048591 | 0.674286 | 5.267746 |
| 0.50 | full_refined | -6.603909 | -1.052584 | 4.097183 | 0.589348 | 5.577503 |
| 0.75 | fixed_initial | -9.905864 | -1.578876 | 0.000000 | 0.485207 | 5.149661 |
| 0.75 | scaled_refined | -9.905864 | -1.578876 | 3.072887 | 0.407530 | 5.265011 |
| 0.75 | full_refined | -9.905864 | -1.578876 | 4.097183 | 0.414493 | 6.534786 |
| 1.00 | fixed_initial | -13.207818 | -2.105168 | 0.000000 | 0.395933 | 6.163984 |
| 1.00 | scaled_refined | -13.207818 | -2.105168 | 4.097183 | 0.331206 | 6.853325 |
| 1.00 | full_refined | -13.207818 | -2.105168 | 4.097183 | 0.331206 | 6.853325 |

Best by overlap:
- `line_search/scale_0.25_alt_scaled_refined`: overlap `0.808729`, chamfer `5.328036`, offset E/N/A `[-3.301955, -0.526292, 1.024296]` m

Best by chamfer:
- `line_search/scale_0.25_alt_fixed_initial`: overlap `0.801166`, chamfer `4.506433`, offset E/N/A `[-3.301955, -0.526292, 0.000000]` m

## Interpretation

- scale=1.0 is worse than initial: `True`. Full refined translation drops overlap from `0.629278` to `0.331206` and worsens chamfer from `5.493694` to `6.853325` when rotation is held fixed.
- scale=0.25 or 0.5 improves initial: `True`. The best useful step is scale `0.25`; overlap improves to `0.808729` and best chamfer improves to `4.506433`.
- direction useful: `True`. Small positive scaling along PiLoT's update improves DOM/DSM visual metrics.
- overshoot: `True`. A small fraction of the update helps, but the full update is harmful.
- alt harmful: `True`. The best chamfer uses `fixed_initial` altitude; full refined altitude at scale 0 already worsens chamfer from `5.493694` to `5.710145`.
- pose conversion suspicious: `partially suspicious`. Raw refined Euler makes the full refined pose worse than refined translation with initial rotation, but the translation-only full update is already bad. Rotation convention should still be handled separately from translation step acceptance.
- loss-metric mismatch: `not reliably checked`. The CUDA extension compatibility warning remains and `overall_loss` is all zeros, so the feature loss signal is not usable for this run. If the extension is rebuilt for sm_61 and loss improves while visual metrics degrade, that would confirm a loss/DOM-DSM visual metric mismatch.

## Conclusion

The PiLoT optimizer should not be discarded. The refined update contains a useful translation direction signal: a 0.25 scale step along PiLoT's own update improves both visual overlap and chamfer over yawfix initial. However, the raw update is not directly acceptable because it overshoots the useful step length, moves north in an unhelpful direction for this sample, and the altitude update degrades the visual metric. A trust region or acceptance line search should be tested before accepting translation, and altitude should be frozen or gated until it improves a validation metric. The refined Euler output remains convention-sensitive, while the current optimizer loss is not trustworthy because the CUDA extension still reports GTX 1080 kernel compatibility errors.

## Artifacts

- summary: `docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results/summary_metrics.json`
- initial visuals: `docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results/initial/`
- raw refinement log: `docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results/refinement_raw/run_log.json`
- line search visuals: `docs/experiments/dom_dsm_prepare/yawfix_refinement_diagnosis_results/line_search/`
