# Refined Pose Convention Check

## Purpose

This experiment checks whether `RenderLocalizer`'s `refined_pose` can be passed
directly to `DOMDSMRenderer`, or whether it needs a pose convention conversion
before re-rendering.

The input yawfix run had strong initial alignment, but the direct refined render
was worse in the previous visual check:

```text
initial_edge_overlap_ratio = 0.629278
refined_edge_overlap_ratio = 0.306533
initial_edge_chamfer = 5.493694
refined_edge_chamfer = 9.167938
```

## Inputs

- Config: `configs/caiwangcun_domdsm_16x9.yaml`
- Run log: `outputs/exif_test_16x9_yawfix_single_full/run_log.json`
- Query image: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- Output directory: `docs/experiments/dom_dsm_prepare/refined_pose_convention_results/`

Initial pose from run log:

```json
{
  "translation_lon_lat_alt": [114.4368608916, 30.3913609745, 391.462],
  "euler_pitch_roll_yaw": [0.0, 180.0, 29.2]
}
```

Refined pose from run log:

```json
{
  "translation_lon_lat_alt": [114.43672403988498, 30.391339299808664, 395.5591768939048],
  "euler_pitch_roll_yaw": [179.99995341015037, -0.0001110565401035226, -144.79992977987283]
}
```

## Command

```bash
./.conda/pilot22/bin/python tools/analyze_refined_pose_conventions.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --run-log outputs/exif_test_16x9_yawfix_single_full/run_log.json \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --output-dir docs/experiments/dom_dsm_prepare/refined_pose_convention_results
```

The script only calls `DOMDSMRenderer.render()`. It does not run
`RenderLocalizer` and does not import `direct_abs_cost_cuda`.

Each candidate directory contains:

```text
rendered_rgb.png
overlay.png
edge_overlay.png
checkerboard.png
metrics.json
```

The output root contains `summary_metrics.json`, including candidates sorted by
edge overlap ratio and by edge chamfer.

## Candidate Metrics

Lower chamfer is better. Higher overlap is better.

| Candidate | Description | Edge overlap ratio | Edge chamfer |
| --- | --- | ---: | ---: |
| `01_initial_baseline` | yawfix initial pose | 0.672436 | 4.495986 |
| `02_direct_refined` | refined pose as logged | 0.359841 | 9.933161 |
| `03_neg_refined_yaw` | refined yaw sign flipped | 0.123574 | 34.513943 |
| `04_refined_yaw_plus_180` | refined yaw + 180 | 0.033835 | 75.824127 |
| `05_refined_yaw_minus_180` | refined yaw - 180 | 0.033835 | 75.824127 |
| `06_swap_roll_pitch` | swap refined pitch/roll | 0.033561 | 75.929642 |
| `07_equivalent_downward_form` | refined translation, euler near `[0,180,29.2]` | 0.297127 | 7.968849 |
| `08_keep_initial_rotation_refined_translation` | refined translation, initial euler | 0.298738 | 7.957027 |
| `09_keep_initial_translation_refined_rotation` | initial translation, direct refined euler | 0.190583 | 10.749441 |
| `10_equiv_initial_translation` | initial translation, equivalent downward refined euler | 0.685658 | 4.662417 |
| `11_equiv_refined_yaw_negated` | equivalent downward euler with yaw sign flipped | 0.123574 | 31.917183 |
| `12_refined_translation_initial_yaw_only` | refined translation, initial downward yaw | 0.298738 | 7.957027 |

Best by edge overlap ratio:

```text
10_equiv_initial_translation
```

Best by edge chamfer:

```text
01_initial_baseline
```

## Answers

### 1. Is direct refined really worse?

Yes. In this re-render test, `02_direct_refined` has overlap `0.359841` and
chamfer `9.933161`, while the yawfix initial baseline has overlap `0.672436`
and chamfer `4.495986`.

This confirms that passing the logged refined pose directly to
`DOMDSMRenderer` is worse than keeping the yawfix initial pose.

### 2. Is there an equivalent Euler angle expression issue?

Partly. The refined euler is near:

```text
[180, 0, -144.8]
```

The `07_equivalent_downward_form` candidate rewrites it to near:

```text
[0, 180, 29.2]
```

That fixes the rotation convention shape, but with refined translation it still
only reaches overlap `0.297127` and chamfer `7.968849`. With initial
translation, the equivalent downward expression becomes
`10_equiv_initial_translation`, which reaches overlap `0.685658` and chamfer
`4.662417`.

So the refined rotation has an Euler-equivalent representation close to the
initial yawfix rotation, but converting rotation alone does not rescue the full
refined pose because the refined translation remains harmful.

### 3. Does refined translation or refined rotation cause the drop?

Both can hurt, but refined translation is the bigger issue in this test.

Using refined translation with initial rotation:

```text
08_keep_initial_rotation_refined_translation
overlap = 0.298738
chamfer = 7.957027
```

Using initial translation with direct refined rotation:

```text
09_keep_initial_translation_refined_rotation
overlap = 0.190583
chamfer = 10.749441
```

Using initial translation with the equivalent downward refined rotation:

```text
10_equiv_initial_translation
overlap = 0.685658
chamfer = 4.662417
```

This indicates the direct refined rotation representation is not suitable for
DOMDSMRenderer, but once expressed in the downward form, rotation is essentially
fine. The refined translation is what prevents the converted refined pose from
matching or beating the initial pose.

### 4. Is any refined pose reinterpretation better than yawfix initial?

By overlap ratio, yes:

```text
10_equiv_initial_translation overlap = 0.685658
01_initial_baseline overlap = 0.672436
```

By chamfer, no:

```text
01_initial_baseline chamfer = 4.495986
10_equiv_initial_translation chamfer = 4.662417
```

The only reinterpretation that slightly beats initial overlap keeps the initial
translation and only uses an equivalent downward version of the refined
rotation. It does not validate the full refined pose because it discards the
refined translation.

### 5. Is the current refined pose trustworthy?

No, not as a full pose. The direct refined pose is worse than initial, and all
candidates using refined translation perform much worse than initial:

```text
07_equivalent_downward_form overlap = 0.297127
08_keep_initial_rotation_refined_translation overlap = 0.298738
12_refined_translation_initial_yaw_only overlap = 0.298738
```

The results show two separate issues:

- Refined euler has an equivalent downward representation that should be used
  for DOMDSMRenderer-style visualization.
- Refined translation degrades alignment and appears unreliable for this
  single-image yawfix test.

## Conclusion

The direct refined pose should not be passed to `DOMDSMRenderer` as-is.

There is an Euler representation issue: `[179.99995, -0.00011, -144.79993]`
can be re-expressed as approximately `[0, 180, 29.20007]`, matching the yawfix
downward convention. But this only helps when the initial translation is kept.

No tested full refined-pose reinterpretation that keeps refined translation
beats the yawfix initial pose. Therefore the current refined pose is not
trustworthy as a complete render pose; the refined translation and pose update
logic need separate investigation.
