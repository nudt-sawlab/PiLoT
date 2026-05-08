# Query 16x9 Crop Visual Check

## Purpose

This experiment checks whether removing the original query image aspect-ratio
mismatch improves DOM/DSM overlay alignment.

The localization flow is unchanged. This step only prepares one cropped query
image, uses a derived config, and runs single-image renderer/visual checks.

## Inputs

- Source query: `data_caiwangcun/query/images/exif_test/0000.jpg`
- Source size: `5280x3956`
- Cropped query: `data_caiwangcun/query/images/exif_test_16x9/0000.jpg`
- Cropped size: `5280x2970`
- Crop box: `left=0, top=493, right=5280, bottom=3463`
- Config: `configs/caiwangcun_domdsm_16x9.yaml`

`cam_query` remains `3840x2160` with the original intrinsics:

```yaml
cam_query:
  max_size: 512
  model: "PINHOLE"
  width: 3840
  height: 2160
  params: [2700.0, 2700.0, 1915.7, 1075.1]
  distortion: [0, 0, 0, 0, 0]
```

The derived config only switches the dataset/output name to
`exif_test_16x9`.

## Commands

Prepare the crop:

```bash
./.conda/pilot22/bin/python tools/prepare_query_16x9_crop.py
```

Check geometry:

```bash
./.conda/pilot22/bin/python tools/check_query_camera_geometry.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg
```

Run renderer-only overlay:

```bash
./.conda/pilot22/bin/python tools/render_dom_dsm_single.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --pose-file data_caiwangcun/query/poses/exif_test.txt \
  --output-dir outputs/exif_test_16x9_single_512 \
  --width 512
```

Run initial/refined visual validation:

```bash
./.conda/pilot22/bin/python tools/validate_refined_pose_visual.py \
  --config configs/caiwangcun_domdsm_16x9.yaml \
  --run-log docs/experiments/dom_dsm_prepare/single_full/run_log.json \
  --query-image data_caiwangcun/query/images/exif_test_16x9/0000.jpg \
  --output-dir outputs/exif_test_16x9_single_full/visual_check
```

## Geometry Result

```text
original_image: width=5280, height=2970, aspect=1.777778
cam_query: width=3840, height=2160, aspect=1.777778
original_vs_cam_query_aspect_delta=0.000000%
query_resize_ratio=7.5
query_image_shape_after_read_image=[396, 704, 3]
read_image_output: width=704, height=396, aspect=1.777778
render_camera_gs: width=512, height=288, aspect=1.777778
render_camera_gs=[512.0, 288.0, 256.0, 144.0, 360.0, 360.0]
read_image_vs_render_camera_gs_aspect_delta=0.000000%
original_vs_render_camera_gs_aspect_delta=0.000000%
non_uniform_stretch=NO: read_image output and render_camera_gs aspect ratios are within the configured threshold.
```

The crop removes the aspect-ratio mismatch between the raw image,
`cam_query`, `read_image()` output, and `render_camera_gs`.

## Renderer-only Output

Artifacts:

```text
outputs/exif_test_16x9_single_512/rendered_rgb.png
outputs/exif_test_16x9_single_512/rendered_depth.png
outputs/exif_test_16x9_single_512/query_render_overlay.png
outputs/exif_test_16x9_single_512/edge_overlay.png
outputs/exif_test_16x9_single_512/checkerboard_overlay.png
outputs/exif_test_16x9_single_512/render_stats_512.json
```

Renderer-only stats:

```json
{
  "valid_depth_ratio": 1.0,
  "depth_min": 356.0,
  "depth_max": 386.0,
  "render_time_sec": 337.22960296399833,
  "image_size": {
    "width": 512,
    "height": 288
  }
}
```

## Initial/Refined Visual Output

Artifacts:

```text
outputs/exif_test_16x9_single_full/visual_check/initial_overlay.png
outputs/exif_test_16x9_single_full/visual_check/initial_edge_overlay.png
outputs/exif_test_16x9_single_full/visual_check/initial_checkerboard.png
outputs/exif_test_16x9_single_full/visual_check/refined_overlay.png
outputs/exif_test_16x9_single_full/visual_check/refined_edge_overlay.png
outputs/exif_test_16x9_single_full/visual_check/refined_checkerboard.png
outputs/exif_test_16x9_single_full/visual_check/visual_compare_metrics.json
```

## Edge Metrics Comparison

Lower chamfer is better. Higher overlap ratio is better.

| Image | Pose | Edge overlap ratio | Edge chamfer | Query edges | Render edges | Overlap edges |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| original | initial | 0.116162 | 31.349481 | 2663 | 2574 | 299 |
| original | refined | 0.095238 | 30.231738 | 2663 | 2562 | 244 |
| 16x9 crop | initial | 0.076046 | 34.550188 | 2104 | 2574 | 160 |
| 16x9 crop | refined | 0.070817 | 33.844048 | 2104 | 2562 | 149 |

## Interpretation

The 16:9 crop fixes the geometry-size mismatch and removes the non-uniform
resize between query and render dimensions.

For this pose/log pair, the edge metrics do not improve after cropping:
initial overlap drops from `0.116162` to `0.076046`, and refined overlap drops
from `0.095238` to `0.070817`. Chamfer distance also increases for both
initial and refined overlays.

This suggests the previous overlay issue was not explained only by the
non-uniform resize. The crop is still useful as a controlled input because it
keeps query, camera, and render aspect ratios consistent for the next
single-image experiments.
