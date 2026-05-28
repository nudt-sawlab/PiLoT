# DOM/DSM Orientation Convention Check

## Purpose

This experiment tests whether the overlay rotation error can be explained by an
incorrect conversion from DJI yaw to the yaw expected by `DOMDSMRenderer`.

The test renders eight yaw candidates from the same camera position, fixed
`pitch=0`, fixed `roll=180`, and the 16:9 query image. It does not modify
`DOMDSMRenderer`, does not modify `main.py`, does not run `RenderLocalizer`, and
does not import or run `direct_abs_cost_cuda`.

## Inputs

- Config: `configs/caiwangcun_domdsm_16x9.yaml`
- Query image: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- Translation: `[114.4368608916, 30.3913609745, 391.462]`
- DJI yaw: `-29.2`
- Fixed pitch/roll: `[0.0, 180.0]`
- Render size: `512x288`

## Command

```bash
./.conda/pilot22/bin/python tools/analyze_dom_dsm_orientation_conventions.py
```

Outputs are written under:

```text
docs/experiments/dom_dsm_prepare/orientation_convention_results/
```

Each candidate directory contains:

```text
rendered_rgb.png
overlay.png
edge_overlay.png
checkerboard.png
metrics.json
```

The root output directory also contains `summary_metrics.json`.

## Results

Lower edge chamfer is better. Higher edge overlap ratio is better.

| Candidate | Expression | Render yaw | Edge overlap ratio | Edge chamfer | Edge overlap count | Valid depth ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `01_dji_yaw` | `dji_yaw` | -29.2 | 0.080323 | 33.572376 | 169 | 0.999939 |
| `02_neg_dji_yaw` | `-dji_yaw` | 29.2 | 0.672436 | 4.495986 | 1049 | 0.999953 |
| `03_90_minus_dji_yaw` | `90 - dji_yaw` | 119.2 | 0.042776 | 64.568729 | 90 | 1.000000 |
| `04_dji_yaw_minus_90` | `dji_yaw - 90` | -119.2 | 0.000000 | 103.617004 | 0 | 1.000000 |
| `05_dji_yaw_plus_90` | `dji_yaw + 90` | 60.8 | 0.099555 | 30.046200 | 179 | 1.000000 |
| `06_dji_yaw_plus_180` | `dji_yaw + 180` | 150.8 | 0.144962 | 53.181820 | 305 | 0.999946 |
| `07_180_minus_dji_yaw` | `180 - dji_yaw` | 209.2 | 0.047124 | 76.396076 | 77 | 0.999973 |
| `08_minus_90_minus_dji_yaw` | `-90 - dji_yaw` | -60.8 | 0.083650 | 41.376118 | 176 | 1.000000 |

Best by edge overlap ratio:

```text
02_neg_dji_yaw
```

Best by edge chamfer:

```text
02_neg_dji_yaw
```

## Interpretation

The `-dji_yaw` candidate is strongly better than every other tested convention.
It has an edge overlap ratio of `0.672436`, while the next-best overlap ratio is
`0.144962` from `dji_yaw + 180`. It also has the lowest chamfer distance,
`4.495986`, while the next-lowest chamfer distance is `30.046200` from
`dji_yaw + 90`.

For this query and pose, the evidence points to a yaw-sign convention mismatch:
the DOM/DSM renderer alignment is best when rendering with:

```text
render_yaw = -dji_yaw
```

Given `dji_yaw = -29.2`, the best render yaw is:

```text
29.2
```

This result is only an orientation convention diagnostic. It does not change the
pipeline and should be validated on additional images before changing pose
conversion logic.
