# Query Camera Geometry Check

## Purpose

This check verifies the geometry consistency between the query image,
`default_confs.cam_query`, `read_image()`, and the DOM/DSM render camera used by
the single-image experiment.

It is diagnostic only and does not modify the localization flow.

## Script

```bash
python tools/check_query_camera_geometry.py
```

The script:

- Reads the raw query image size from `data_caiwangcun/query/images/exif_test/0000.jpg`.
- Reads `default_confs.cam_query` from `configs/caiwangcun_domdsm.yaml`.
- Computes `render_camera_gs` with the same size arithmetic used by
  `tools/run_dom_dsm_single_full.py`.
- Calls the existing `read_image()` function and prints the resulting
  `query_image_shape`.
- Reports whether resizing from the read query image to the render camera would
  require non-uniform x/y scaling.

## Command

Run in the PiLoT conda environment:

```bash
./.conda/pilot22/bin/python tools/check_query_camera_geometry.py
```

On Windows from this workspace, the same run was executed through WSL:

```powershell
wsl.exe bash -lc 'cd /mnt/d/aiproject/PiLoT_work && ./.conda/pilot22/bin/python tools/check_query_camera_geometry.py'
```

## Result

```text
original_image: width=5280, height=3956, aspect=1.334681
cam_query: width=3840, height=2160, aspect=1.777778
original_vs_cam_query_aspect_delta=33.198653%
WARNING: original image aspect ratio differs from cam_query aspect ratio by 33.20%, above 1.00%.
query_resize_ratio=7.5
query_image_shape_after_read_image=[527, 704, 3]
read_image_output: width=704, height=527, aspect=1.335863
render_camera_gs: width=512, height=288, aspect=1.777778
render_camera_gs=[512.0, 288.0, 256.0, 144.0, 360.0, 360.0]
read_image_vs_render_camera_gs_aspect_delta=33.080808%
original_vs_render_camera_gs_aspect_delta=33.198653%
non_uniform_stretch=YES: read_image output and render_camera_gs have different aspect ratios, so resizing between them would stretch x/y by different factors.
```

## Interpretation

The raw query image is `5280x3956`, with aspect ratio `1.334681`.

`cam_query` is configured as `3840x2160`, with aspect ratio `1.777778`.
This exceeds the 1% warning threshold, so the script emits a warning.

`read_image()` preserves the raw image aspect ratio apart from integer rounding:
the resized query shape is `[527, 704, 3]`, with aspect ratio `1.335863`.

`render_camera_gs` is `512x288`, with aspect ratio `1.777778`.
Therefore, resizing the `read_image()` output to the render camera dimensions
would be non-uniform and would stretch x/y by different factors.
