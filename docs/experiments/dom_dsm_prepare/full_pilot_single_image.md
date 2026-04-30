# Full PiLoT Single Image Test

## Environment

- Conda env path: `/mnt/d/aiproject/PiLoT_work/.conda/pilot`
- Torch version: `2.7.1+cu118`
- CUDA available: `True`
- CUDA version: `11.8`
- GPU: `NVIDIA GeForce GTX 1080`
- GPU memory during checks: `1313 MiB / 8192 MiB`
- `direct_abs_cost_cuda` import status: failed, `ModuleNotFoundError("No module named 'direct_abs_cost_cuda'")`

## Input

- Query image path: `data_caiwangcun/query/images/exif_test/0000.jpg`
- Pose file path: `data_caiwangcun/query/poses/exif_test.txt`
- Pose row: `0000.jpg 114.4368608916 30.3913609745 391.4620 180.000000 0.000000 -29.200000`
- Pose format: `image_name lon lat alt roll pitch yaw`
- DOM path: `data_caiwangcun/reference/caiwangcun_dom.tif`
- DSM path: `data_caiwangcun/reference/caiwangcun_dsm.tif`
- Config path: `configs/caiwangcun_domdsm.yaml`

## Renderer-only Baseline

- Output directory: `outputs/exif_test_single_512`
- Image size: `512x288`
- `valid_depth_ratio`: `0.99993896484375`
- `depth_min/depth_max`: `360.0 / 390.0`
- Render time: `77.93734488100006 sec`
- Render params: `near_m=330.0`, `far_m=430.0`, `ray_step_m=10.0`
- Overlay conclusion: the 512-wide overlay and checkerboard show broadly consistent viewpoint, direction and region. The left water/shoreline area, central vegetation/bare soil, and right-side road network are in the expected locations.

## Full PiLoT Run

- Command: `python main.py --config configs/caiwangcun_domdsm.yaml --name exif_test --viz`
- Success/failure: failed; no refined pose was produced.
- Failure stage: checkpoint loading.
- Main-process observation: the full command did not produce `outputs/exif_test.txt` before the 900 second timeout. `outputs/exif_test/` existed but remained empty.
- Isolated localization check: direct `RenderLocalizer(conf)` initialization failed while loading `data_demo/pretrained_model/model@mapscape@512@Fourier.ckpt`.
- Error: `FileNotFoundError: [Errno 2] No such file or directory: 'data_demo/pretrained_model/model@mapscape@512@Fourier.ckpt'`
- Output files:
  - `outputs/exif_test_single_512/rendered_rgb.png`
  - `outputs/exif_test_single_512/rendered_depth.png`
  - `outputs/exif_test_single_512/query_render_overlay.png`
  - `outputs/exif_test_single_512/edge_overlay.png`
  - `outputs/exif_test_single_512/checkerboard_overlay.png`
  - `outputs/exif_test_single_512/render_stats_512.json`
- Initial pose: `lon=114.4368608916`, `lat=30.3913609745`, `alt=391.4620`, `roll=180.0`, `pitch=0.0`, `yaw=-29.2`
- Refined pose: not available.
- Pose delta: not available.
- Runtime: full command stopped after `900 sec` timeout; isolated checkpoint failure reproduced in about `36 sec`.
- Visualization video: not generated.
- Full-run render image: not generated in `outputs/exif_test/`.

## Conclusion

- The test has not entered PiLoT refinement successfully.
- A refined pose was not produced.
- The current blocker is the missing PiLoT checkpoint at `data_demo/pretrained_model/model@mapscape@512@Fourier.ckpt`.
- `direct_abs_cost_cuda` is also not importable from the environment, but the run did not reach the learned optimizer stage because checkpoint loading failed first.
- Continue to full single-image refinement only after restoring the checkpoint path. After that, re-check whether `direct_abs_cost_cuda` is required by the optimizer path before testing orthorectification or target localization.
