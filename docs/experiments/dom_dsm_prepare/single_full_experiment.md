# DOM DSM Single-process Full PiLoT Experiment

## Purpose

This experiment verifies whether a one-image DOM+DSM PiLoT refinement can run without the `main.py` multiprocessing CUDA tensor sharing failure.

The previous full `main.py` run entered the single-image flow but failed in a spawned process with:

```text
RuntimeError: CUDA error: invalid resource handle
```

The new test keeps rendering, back-projection, model initialization, and `RenderLocalizer.run_query()` in one Python process.

## Script

New script:

```text
tools/run_dom_dsm_single_full.py
```

Important constraints:

- Does not import `main.py`.
- Does not import `DualProcessTask`.
- Does not use `torch.multiprocessing`, `Queue`, or child processes.
- Does not modify `DOMDSMRenderer` geometry.
- Only runs `data_caiwangcun/query/images/exif_test/0000.jpg`.

The script copies the relevant logic from `main.py`:

- Camera setup from `default_confs.cam_query`.
- Pose loading from `data_caiwangcun/query/poses/exif_test.txt`.
- `DOMDSMRenderer.render(trans, euler)`.
- Back-project via `sample_3d_points()`.
- `RenderLocalizer(conf)` where `conf = yaml["default_confs"]["from_render_test"]`.
- `localizer.run_query(...)` in the current process.

One standalone-script-only input adaptation was needed: the raw query image is `5280x3956`, while the config camera assumes `3840x2160` and `max_size=512`. The first resize therefore produced `704x527`, but `BaseRefiner.zero_pad(512)` requires input dimensions no larger than `512x512`. The script now resizes the query image passed to `run_query()` to the render camera size `512x288`. This does not change `main.py` or renderer geometry.

## Environment

Environment path:

```text
/mnt/d/aiproject/PiLoT_work/.conda/pilot22
```

Runtime:

```text
Python 3.8.20
torch 2.4.1+cu124
CUDA available: true
CUDA runtime: 12.4
GPU: NVIDIA GeForce GTX 1080
direct_abs_cost_cuda import: ok
Ubuntu 22.04 glibc: 2.35
```

Known environment note:

- `triton==3.0.0` is still not installed because repeated downloads failed hash validation through the proxy.
- The single-process test completed despite this.
- During refinement, the console printed repeated `Kernel launch failed: no kernel image is available for execution on the device`, but `run_query` returned success.

## Inputs

Config:

```text
configs/caiwangcun_domdsm.yaml
```

Query image:

```text
data_caiwangcun/query/images/exif_test/0000.jpg
```

Pose file:

```text
data_caiwangcun/query/poses/exif_test.txt
```

Initial pose:

```text
0000.jpg 114.4368608916 30.3913609745 391.462 180.0 0.0 -29.2
```

The pose file format is:

```text
image_name lon lat alt roll pitch yaw
```

Internally the script uses:

```text
euler = [pitch, roll, yaw]
trans = [lon, lat, alt]
```

## Steps

Run static checks:

```bash
grep -n "import main\|DualProcessTask\|torch.multiprocessing\|Queue" tools/run_dom_dsm_single_full.py || true
python -m py_compile tools/run_dom_dsm_single_full.py
```

Run environment check:

```bash
python -c "import torch, direct_abs_cost_cuda; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Run the single-image experiment:

```bash
python tools/run_dom_dsm_single_full.py
```

## Results

The script successfully passed the previous multiprocessing failure point and completed `RenderLocalizer.run_query()`.

Summary from `run_log.json`:

```json
{
  "run_query_success": true,
  "render_time_sec": 53.03266477584839,
  "back_project_time_sec": 0.40281200408935547,
  "localizer_init_time_sec": 27.21785068511963,
  "run_query_time_sec": 1.8203535079956055,
  "total_time_sec": 84.12126398086548,
  "points_3d_count": 500,
  "valid_depth_ratio": 0.9999593098958334,
  "depth_min": 360.0,
  "depth_max": 390.0
}
```

Refined pose output order follows `main.py`:

```text
image_name lon lat alt roll pitch yaw
```

Result:

```text
0000.jpg 114.43672403992267 30.391339299846756 395.55915982834995 -9.94896617078202e-05 -179.9999344297537 156.80006429664994
```

Delta:

```text
-0.00013685167733967774 -2.167465324376394e-05 4.097159828349959 -180.00009948966172 -179.9999344297537 186.00006429664992
```

## Artifacts

Lightweight artifacts copied from `outputs/exif_test_single_full/` into Git-trackable docs:

```text
docs/experiments/dom_dsm_prepare/single_full/run_log.json
docs/experiments/dom_dsm_prepare/single_full/result_pose.txt
docs/experiments/dom_dsm_prepare/single_full/rendered_rgb.png
docs/experiments/dom_dsm_prepare/single_full/rendered_depth.png
docs/experiments/dom_dsm_prepare/single_full/query_render_overlay.png
```

## Conclusion

The single-process experiment confirms that the DOM+DSM full PiLoT chain can get past the previous multiprocessing CUDA tensor sharing failure and can return a refined pose for `0000.jpg`.

The next technical risk is result credibility on GTX 1080 because `direct_abs_cost_cuda` prints `no kernel image is available for execution on the device`. Before using the refined pose for target localization or evaluation, verify whether the CUDA extension binaries include support for the GTX 1080 compute capability, or run the same script on a newer supported GPU.
