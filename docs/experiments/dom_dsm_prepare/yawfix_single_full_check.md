# Yawfix Single Full Check

## Purpose

This experiment verifies whether the orientation convention result
`render_yaw = -dji_yaw` improves DOM/DSM renderer-only overlay and full
single-image refinement visual alignment.

The prior orientation convention check found that `dji_yaw=-29.2` is best
rendered as `29.2`, with edge overlap ratio `0.672436` and chamfer `4.495986`.

## Inputs

- Query: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- Config: `configs/caiwangcun_domdsm_16x9.yaml`
- Yawfix pose: `data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt`
- Pose row:

```text
0000.jpg 114.4368608916 30.3913609745 391.462 180.0 0.0 29.2
```

Only yaw changed from `-29.2` to `29.2`. Lon, lat, alt, roll, and pitch were
kept unchanged.

## Commands

Renderer-only yawfix:

```bash
./.conda/pilot22/bin/python tools/render_dom_dsm_single.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir outputs/exif_test_16x9_yawfix_single_512 \
  --width 512
```

Full single-image refinement:

```bash
./.conda/pilot22/bin/python tools/run_dom_dsm_single_full.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir outputs/exif_test_16x9_yawfix_single_full
```

Visual validation:

```bash
./.conda/pilot22/bin/python tools/validate_refined_pose_visual.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --run-log outputs/exif_test_16x9_yawfix_single_full/run_log.json \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --output-dir outputs/exif_test_16x9_yawfix_single_full/visual_check
```

Runtime note: the full refinement command was run in the `Ubuntu-22.04` WSL
distribution with `PYTHONPATH` including `DirectAbsoluteCostCuda` and
`LD_LIBRARY_PATH` including the `pilot22` torch library directory. The GTX 1080
still printed repeated `Kernel launch failed: no kernel image is available for
execution on the device`, but `run_query` returned success.

## Lightweight Artifacts

Artifacts copied from `outputs/`:

```text
docs/experiments/dom_dsm_prepare/yawfix_results/
|-- renderer_only_512/
|   |-- rendered_rgb.png
|   |-- rendered_depth.png
|   |-- query_render_overlay.png
|   |-- edge_overlay.png
|   |-- checkerboard_overlay.png
|   `-- render_stats_512.json
|-- full_single/
|   |-- rendered_rgb.png
|   |-- rendered_depth.png
|   |-- query_render_overlay.png
|   |-- result_pose.txt
|   `-- run_log.json
`-- full_visual_check/
    |-- initial_rendered_rgb.png
    |-- refined_rendered_rgb.png
    |-- initial_overlay.png
    |-- refined_overlay.png
    |-- initial_edge_overlay.png
    |-- refined_edge_overlay.png
    |-- initial_checkerboard.png
    |-- refined_checkerboard.png
    `-- visual_compare_metrics.json
```

## Results

Renderer-only stats from `render_stats_512.json`:

```json
{
  "valid_depth_ratio": 1.0,
  "depth_min": 358.0,
  "depth_max": 380.0,
  "render_time_sec": 711.122470161994
}
```

Full single-image refinement from `run_log.json`:

```json
{
  "run_query_success": true,
  "valid_depth_ratio": 0.9999932183159722,
  "depth_min": 360.0,
  "depth_max": 380.0,
  "render_time_sec": 128.84976935386658,
  "run_query_time_sec": 3.0331273078918457,
  "total_time_sec": 254.59769225120544
}
```

Refined pose:

```text
0000.jpg 114.43672403988498 30.391339299808664 395.5591768939048 -0.0001110565401035226 179.99995341015037 -144.79992977987283
```

Visual metrics from `visual_compare_metrics.json`:

```json
{
  "initial_edge_overlap_ratio": 0.629277566539924,
  "refined_edge_overlap_ratio": 0.3065326633165829,
  "initial_edge_chamfer": 5.493693828582764,
  "refined_edge_chamfer": 9.167937755584717
}
```

## Comparison

Lower chamfer is better. Higher overlap is better.

| Experiment | Pose | Edge overlap ratio | Edge chamfer |
| --- | --- | ---: | ---: |
| old 16x9 | initial | 0.076046 | 34.550188 |
| old 16x9 | refined | 0.070817 | 33.844048 |
| orientation best `02_neg_dji_yaw` | initial only | 0.672436 | 4.495986 |
| yawfix full visual | initial | 0.629278 | 5.493694 |
| yawfix full visual | refined | 0.306533 | 9.167938 |

## Conclusion

Yawfix initial alignment is much better than the old 16:9 initial overlay:
edge overlap improves from `0.076046` to `0.629278`, and chamfer improves from
`34.550188` to `5.493694`.

This strongly supports the orientation convention diagnosis: the large initial
overlay offset is mainly caused by the yaw sign mismatch between DJI yaw and
the yaw expected by `DOMDSMRenderer`.

The refined yawfix output is worse than yawfix initial, with overlap dropping
from `0.629278` to `0.306533` and chamfer increasing from `5.493694` to
`9.167938`. It is still better than the old 16:9 refined result, but the
degradation after refinement suggests the refinement path, CUDA extension
behavior on GTX 1080, or optimizer interface should be investigated separately.
