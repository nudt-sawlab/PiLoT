# Yawfix Clamped Refinement Check

## Purpose

This experiment tests whether the yawfix full refinement degrades visual
alignment because the optimizer applies a pose update that is too large.

The previous yawfix run showed good initial alignment:

```text
initial_edge_overlap_ratio = 0.629278
initial_edge_chamfer = 5.493694
```

But after full refinement, visual alignment became worse:

```text
refined_edge_overlap_ratio = 0.306533
refined_edge_chamfer = 9.167938
```

This diagnostic keeps the same yawfix input, runs one full single-image
refinement, then re-renders several post-processing strategies without changing
`main.py`, `DOMDSMRenderer`, or `tools/run_dom_dsm_single_full.py`.

## Input Pose

- Query: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- Config: `configs/caiwangcun_domdsm_16x9.yaml`
- Pose file: `data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt`
- Pose row:

```text
0000.jpg 114.4368608916 30.3913609745 391.462 180.0 0.0 29.2
```

Pose file format:

```text
image_name lon lat alt roll pitch yaw
```

The script internally uses euler order:

```text
pitch roll yaw
```

## Command

```bash
./.conda/pilot22/bin/python tools/run_dom_dsm_single_full_clamped.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir outputs/exif_test_16x9_yawfix_clamped
```

Runtime note: the command was run in `Ubuntu-22.04` WSL with `PYTHONPATH`
including `DirectAbsoluteCostCuda` and `LD_LIBRARY_PATH` including the `pilot22`
torch library directory. The GTX 1080 still printed `Kernel launch failed: no
kernel image is available for execution on the device`, but `run_query`
returned success.

## Strategies

- `keep_initial`: do not accept any refinement update; render the yawfix initial pose.
- `raw_refined`: render the raw pose returned by `localizer.run_query()`.
- `clamp_translation_only`: accept translation update only; keep initial euler.
- `clamp_rotation_only`: accept euler update only; keep initial translation.
- `clamp_small_delta`: clamp updates around the initial pose.

`clamp_small_delta` thresholds:

```json
{
  "max_trans_delta_lon_lat_alt": [0.00002, 0.00002, 1.0],
  "max_euler_delta_pitch_roll_yaw": [5.0, 5.0, 5.0]
}
```

Raw refinement delta:

```json
{
  "delta_lon_lat_alt_roll_pitch_yaw": [
    -0.00013685170053179263,
    -0.00002167469277836176,
    4.097174058027576,
    -180.00011105653073,
    179.99995341015875,
    -173.99992977988015
  ]
}
```

The raw update is very large in both translation and rotation.

## Lightweight Results

Artifacts were copied to:

```text
docs/experiments/dom_dsm_prepare/yawfix_clamped_results/
|-- keep_initial/
|-- raw_refined/
|-- clamp_translation_only/
|-- clamp_rotation_only/
|-- clamp_small_delta/
`-- summary_metrics.json
```

Each strategy directory contains:

```text
rendered_rgb.png
overlay.png
edge_overlay.png
checkerboard.png
metrics.json
```

## Metrics

Lower chamfer is better. Higher overlap is better.

| Strategy | Edge overlap ratio | Edge chamfer | Valid depth ratio | Render yaw | Translation summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `keep_initial` | 0.634347 | 6.274406 | 0.999993 | 29.2 | initial |
| `raw_refined` | 0.318668 | 9.999294 | 0.998806 | -144.799930 | raw refined translation |
| `clamp_translation_only` | 0.326950 | 7.723006 | 0.998169 | 29.2 | raw refined translation |
| `clamp_rotation_only` | 0.215586 | 11.153697 | 1.000000 | -144.799930 | initial translation |
| `clamp_small_delta` | 0.124765 | 22.261497 | 1.000000 | 24.2 | clamped small delta |

Best by edge overlap ratio:

```text
keep_initial
```

Best by edge chamfer:

```text
keep_initial
```

## Comparison

| Experiment | Pose/strategy | Edge overlap ratio | Edge chamfer |
| --- | --- | ---: | ---: |
| yawfix visual check | initial | 0.629278 | 5.493694 |
| yawfix visual check | refined | 0.306533 | 9.167938 |
| clamped run | `keep_initial` | 0.634347 | 6.274406 |
| clamped run | `raw_refined` | 0.318668 | 9.999294 |
| clamped run | `clamp_translation_only` | 0.326950 | 7.723006 |
| clamped run | `clamp_rotation_only` | 0.215586 | 11.153697 |
| clamped run | `clamp_small_delta` | 0.124765 | 22.261497 |

The clamped run reproduces the same pattern as the previous yawfix visual
check: initial yawfix alignment is strong, while the raw refined pose is worse.

## Conclusion

The best strategy is `keep_initial`. It is best by both edge overlap ratio and
edge chamfer.

The raw refinement update is very large:

- lon changes by about `0.00013685` degrees
- lat changes by about `0.00002167` degrees
- alt changes by about `4.10 m`
- roll/pitch/yaw output-order deltas are near `-180`, `+180`, and `-174` degrees

This supports the hypothesis that refinement is degrading the yawfix result
because the accepted update is too large or otherwise inconsistent with the
already-good initial alignment.

However, the tested clamps do not produce a better pose than simply keeping the
initial pose. `clamp_translation_only` is slightly better than raw refined in
these metrics, while `clamp_rotation_only` and `clamp_small_delta` are worse.
That suggests the issue is not solved by a naive fixed clamp alone. The next
step should inspect the refinement update formulation, angle wrapping, optimizer
confidence/acceptance criteria, and GTX 1080 CUDA extension behavior before
trusting the refined pose.
