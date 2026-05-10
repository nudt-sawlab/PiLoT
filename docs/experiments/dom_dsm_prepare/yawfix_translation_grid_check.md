# Yawfix Translation Grid Check

## Purpose

This experiment searches east/north/alt translation offsets around the yawfix
initial pose using only `DOMDSMRenderer`. It does not run `RenderLocalizer`.

The goal is to determine whether the refined translation moves in a reasonable
direction or whether it moves too far from the visually aligned yawfix initial
pose.

## Inputs

- Config: `configs/caiwangcun_domdsm_16x9.yaml`
- Query: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- Base pose: `data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt`
- Base euler: `[0.0, 180.0, 29.2]`
- Base translation: `[114.4368608916, 30.3913609745, 391.462]`
- DOM/DSM CRS: `EPSG:32650`

The script uses `pyproj.Transformer` to convert WGS84 lon/lat to the DOM/DSM
CRS, apply east/north offsets in meters, and transform back to WGS84. It does
not use approximate lon/lat scale factors.

## Commands

The original requested 512-wide `-30..30 step 5` grid was started, but it was
too slow for the prototype renderer. A 512-wide yawfix render can take several
minutes, and the full grid requires 169+ renders. The long run was stopped
before it produced results.

First coarse 256-wide search:

```bash
./.conda/pilot22/bin/python tools/search_yawfix_translation_grid.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir docs/experiments/dom_dsm_prepare/yawfix_translation_grid_results_coarse_256 \
  --east-range -30 30 15 \
  --north-range -30 30 15 \
  --alt-offsets 0 \
  --width 256 \
  --summary-every 2
```

Local 256-wide search around the coarse/refined region:

```bash
./.conda/pilot22/bin/python tools/search_yawfix_translation_grid.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test_16x9_yawfix.txt \
  --output-dir docs/experiments/dom_dsm_prepare/yawfix_translation_grid_results \
  --east-range -25 0 5 \
  --north-range -10 10 5 \
  --alt-offsets -5 0 5 \
  --width 256 \
  --summary-every 5
```

The final checked result directory is:

```text
docs/experiments/dom_dsm_prepare/yawfix_translation_grid_results/
```

It contains `partial_metrics.jsonl`, `summary_metrics.json`, and top10
candidate visual folders with:

```text
rendered_rgb.png
overlay.png
edge_overlay.png
checkerboard.png
metrics.json
```

## Metric Note

The script uses the same edge metric pattern as the other visual diagnostics.
Because overlap is computed from dilated query/render edge neighborhoods, the
reported `edge_overlap_ratio` can exceed `1.0` at 256-wide resolution.

For this reason, `edge_chamfer` is treated as the more stable ranking signal,
while overlap is still reported for continuity with previous experiments.

## Base Initial Metrics

At 256-wide resolution:

```json
{
  "east_offset_m": 0.0,
  "north_offset_m": 0.0,
  "alt_offset_m": 0.0,
  "edge_overlap_ratio": 1.0348162475822051,
  "edge_chamfer": 3.9567079544067383
}
```

## Refined Translation Metrics

The yawfix full refined translation maps to:

```json
{
  "east_offset_m": -13.207827881269623,
  "north_offset_m": -2.10516388528049,
  "alt_offset_m": 4.097176893904816
}
```

Using this refined translation with the yawfix initial rotation gives:

```json
{
  "edge_overlap_ratio": 0.7015873015873015,
  "edge_chamfer": 4.707603931427002
}
```

This is worse than the base initial pose and much worse than the best local
grid candidates.

## Grid Search Top10

Top10 by edge overlap ratio:

| Rank | East m | North m | Alt m | Edge overlap ratio | Edge chamfer |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -5 | 0 | 0 | 1.433333 | 2.410713 |
| 2 | -5 | 5 | 5 | 1.380822 | 2.222156 |
| 3 | -5 | 5 | 0 | 1.271930 | 2.007303 |
| 4 | -5 | 0 | -5 | 1.260736 | 4.664337 |
| 5 | -5 | 0 | 5 | 1.212034 | 2.721199 |
| 6 | -10 | 5 | 0 | 1.194384 | 2.130534 |
| 7 | -5 | 5 | -5 | 1.149560 | 3.272926 |
| 8 | -5 | -5 | 0 | 1.140590 | 3.893077 |
| 9 | 0 | 5 | 5 | 1.125356 | 3.450986 |
| 10 | -10 | 5 | -5 | 1.101190 | 2.877854 |

Best by edge overlap ratio:

```json
{
  "east_offset_m": -5.0,
  "north_offset_m": 0.0,
  "alt_offset_m": 0.0,
  "edge_overlap_ratio": 1.4333333333333333,
  "edge_chamfer": 2.410712718963623
}
```

Best by edge chamfer:

```json
{
  "east_offset_m": -5.0,
  "north_offset_m": 5.0,
  "alt_offset_m": 0.0,
  "edge_overlap_ratio": 1.2719298245614035,
  "edge_chamfer": 2.007302761077881
}
```

## Comparison

| Pose / candidate | East m | North m | Alt m | Edge overlap ratio | Edge chamfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| base initial | 0.0 | 0.0 | 0.0 | 1.034816 | 3.956708 |
| refined translation | -13.21 | -2.11 | 4.10 | 0.701587 | 4.707604 |
| best overlap grid | -5.0 | 0.0 | 0.0 | 1.433333 | 2.410713 |
| best chamfer grid | -5.0 | 5.0 | 0.0 | 1.271930 | 2.007303 |

The best local offsets improve over the base initial pose at 256-wide
resolution. Both best candidates stay close to the original pose: around 5 m
west, 0-5 m north, and no altitude change.

## Refined Translation Direction

The refined translation is:

```text
east = -13.21 m
north = -2.11 m
alt = +4.10 m
```

The grid optimum is near:

```text
east = -5 m
north = 0..5 m
alt = 0 m
```

So the refined translation is not a completely random direction: it moves west,
which is broadly consistent with the grid preference for a small westward
shift. But it goes too far west, moves slightly south instead of north/neutral,
and increases altitude when the best local grid candidates use `alt=0`.

## Conclusion

The refined translation is not the best local translation. It appears to move
in a partially reasonable direction in east/west, but it overshoots and applies
an altitude increase that hurts the visual metrics.

At 256-wide resolution, the best translation is a small offset near the initial
pose, approximately:

```text
east = -5 m
north = 0..5 m
alt = 0 m
```

This supports the interpretation that RenderLocalizer's refined translation is
too large and partly misdirected, rather than a useful improvement over the
yawfix initial pose.

Before changing pipeline behavior, confirm the best local candidates with a
small 512-wide local search around `east=-5`, `north=0..5`, `alt=0`, because
the exhaustive 512-wide grid is too slow for the current prototype renderer.
