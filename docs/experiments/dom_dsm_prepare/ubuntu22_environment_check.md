# Ubuntu 22.04 Environment Check

## Current Default Execution Environment

The current default local execution environment for this repository is:

- GitHub repository: `jhvbgg5558/PiLoT`
- local path: `/mnt/d/aiproject/PiLoT_work`
- working branch: `feature/dom-dsm-renderer`
- system: `Ubuntu-22.04 WSL`
- Python: `./.conda/pilot22/bin/python`
- Python version: `3.8.20`
- PyTorch: `2.4.1+cu124`
- CUDA runtime: `12.4`
- GPU: `NVIDIA GeForce GTX 1080`
- GPU compute capability: `sm_61`

Unless a later experiment note explicitly says otherwise, environment-specific conclusions in the DOM/DSM experiment documents should be interpreted against this baseline.


## Authoritative Current Runtime Path

Use this section first when opening a new terminal/window. The DOM/DSM experiments in this repo were run from the `Ubuntu-22.04` WSL distribution, not from the default `Ubuntu-20.04` distribution.

Open the correct distro from Windows PowerShell:

```powershell
wsl -d Ubuntu-22.04
```

Then use the project and Python exactly as follows:

```bash
cd /mnt/d/aiproject/PiLoT_work
./.conda/pilot22/bin/python --version
```

Path mapping:

```text
Windows project path: D:\aiproject\PiLoT_work
WSL project path:     /mnt/d/aiproject/PiLoT_work
Do not use as repo:   D:\aiproject\Pilot or /mnt/d/aiproject/Pilot
```

The `pilot22` entry under the repo is intentionally a Linux symlink into the Ubuntu-22.04 ext4 filesystem:

```text
/mnt/d/aiproject/PiLoT_work/.conda/pilot22 -> /home/farsee2/pilot22
/mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python -> python3.8
real Python: /home/farsee2/pilot22/bin/python3.8
```

If `/home/farsee2/pilot22/bin/python3.8` is missing, you are almost certainly in the wrong WSL distro. Check with:

```bash
cat /etc/os-release
wsl.exe -l -v   # from PowerShell, not inside WSL
```

Expected runtime facts:

```text
Python: 3.8.20
PyTorch: 2.4.1+cu124
CUDA runtime reported by torch: 12.4
GPU: NVIDIA GeForce GTX 1080
Compute capability: sm_61
```

Recommended sanity check:

```bash
cd /mnt/d/aiproject/PiLoT_work
readlink .conda/pilot22
readlink -f .conda/pilot22/bin/python
./.conda/pilot22/bin/python - <<'PY'
import os, sys, torch
print("sys.executable:", sys.executable)
print("real executable:", os.path.realpath(sys.executable))
print("sys.prefix:", sys.prefix)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

## Repository Requirements Check

README installation notes only say that `DirectAbsoluteCostCuda` provides pre-built binaries for Python 3.8 / 3.9 / 3.10 on Linux x86_64:

```bash
cd DirectAbsoluteCostCuda
pip install .
cd ..
```

The README does not explicitly require Ubuntu 22.04 or any specific glibc version.

`environment.yaml` specifies:

- `python=3.8.20`
- `torch==2.4.1+cu124`
- `torchvision==0.19.1+cu124`
- `torchaudio==2.4.1+cu124`
- `segmentation-models-pytorch==0.3.3`

`environment.yaml` does not explicitly require Ubuntu 22.04, glibc 2.32, or any specific Linux distribution version.

## Current Ubuntu 20.04 Blocker

Current WSL distribution list only shows `Ubuntu-20.04` plus `docker-desktop`.

On Ubuntu 20.04:

- system glibc: `ldd (Ubuntu GLIBC 2.31-0ubuntu9.18) 2.31`
- `DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-310-x86_64-linux-gnu.so` requires `GLIBC_2.32`
- the same binary also requires newer `libstdc++` symbols such as `GLIBCXX_3.4.29` and `CXXABI_1.3.13`

Do not manually upgrade glibc inside Ubuntu 20.04. Use a newer WSL distribution or rebuild the CUDA extension for the current OS/toolchain.

## Ubuntu 22.04 WSL Plan

Install Ubuntu 22.04 from Windows PowerShell if it is not already present:

```powershell
wsl --install -d Ubuntu-22.04
wsl -l -v
```

Then open the new distribution once and complete the Linux username/password setup:

```powershell
wsl -d Ubuntu-22.04
```

From inside Ubuntu 22.04:

```bash
sudo apt update
sudo apt install -y git build-essential curl wget ca-certificates
```

Install or expose conda/mamba in Ubuntu 22.04. If Miniconda is not installed in that distro:

```bash
mkdir -p ~/miniconda3
wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -u -p ~/miniconda3
~/miniconda3/bin/conda init bash
exec bash
```

## Minimal Verification Chain

Use the existing repository and data under the mounted D drive:

```bash
cd /mnt/d/aiproject/PiLoT_work
git status --short
```

Create a separate environment so the Ubuntu 20.04 `.conda/pilot` environment remains untouched:

```bash
conda create -y -p /mnt/d/aiproject/PiLoT_work/.conda/pilot22 python=3.8.20 pip=24.2 setuptools=75.1.0 wheel=0.44.0
conda activate /mnt/d/aiproject/PiLoT_work/.conda/pilot22
```

Install dependencies as close to `environment.yaml` as possible:

```bash
python -m pip install --upgrade pip
python -m pip install \
  torch==2.4.1+cu124 \
  torchvision==0.19.1+cu124 \
  torchaudio==2.4.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  numpy==1.24.4 scipy==1.10.1 opencv-python==4.13.0.92 \
  matplotlib==3.7.5 pillow==10.4.0 pyyaml==6.0.3 tqdm==4.67.3 \
  imageio==2.35.1 pyproj==3.5.0 pycolmap==3.12.5 \
  pytorch-lightning==2.4.0 torchmetrics==1.5.2 timm==0.9.2 \
  segmentation-models-pytorch==0.3.3 efficientnet-pytorch==0.7.1 \
  scikit-learn==1.3.2 scikit-image==0.21.0 ftfy==6.2.3 \
  regex==2024.11.6 safetensors==0.5.3 pykalman==0.9.7 \
  omegaconf==2.3.0 pandas==2.0.3 plyfile==1.0.3 \
  pycocotools==2.0.7 setproctitle==1.3.7 coloredlogs==15.0.1 \
  rich==13.9.4 packaging==24.2 tensorboard==2.14.0 \
  tensorboardx==2.6.2.2 wandb==0.24.2 rasterio shapely
```

If `torch==2.4.1+cu124` is unstable on the GTX 1080, record the failure and switch only the PyTorch stack to CUDA 11.8:

```bash
python -m pip install --force-reinstall \
  torch==2.4.1+cu118 \
  torchvision==0.19.1+cu118 \
  torchaudio==2.4.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Install the prebuilt CUDA extension:

```bash
cd /mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda
python -m pip install .
cd /mnt/d/aiproject/PiLoT_work
```

Run the minimum environment checks:

```bash
python -V
ldd --version | head -1

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
import direct_abs_cost_cuda
print("direct_abs_cost_cuda import ok")
PY
```

Only after `direct_abs_cost_cuda` imports successfully, test `RenderLocalizer` initialization:

```bash
python - <<'PY'
import os, sys, time, yaml
os.chdir("/mnt/d/aiproject/PiLoT_work")
sys.path.insert(0, os.getcwd())
from pixloc.localization.localizer import RenderLocalizer

conf = yaml.safe_load(open("configs/caiwangcun_domdsm.yaml"))["default_confs"]["from_render_test"]
t0 = time.time()
localizer = RenderLocalizer(conf)
print("RenderLocalizer ready", time.time() - t0)
PY
```

Only after `RenderLocalizer` initializes successfully, run the full single-image test:

```bash
cd /mnt/d/aiproject/PiLoT_work
rm -rf outputs/exif_test outputs/exif_test.txt outputs/exif_test_xyz.txt
/usr/bin/time -p python main.py --config configs/caiwangcun_domdsm.yaml --name exif_test --viz
```

Collect outputs:

```bash
ls -R outputs | head -100
cat outputs/exif_test.txt || true
find outputs -iname "*.png" -o -iname "*.mp4" -o -iname "*.txt" | sort
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true
```

## Execution Status

Executed on `Ubuntu-22.04` WSL.

### WSL and glibc

```text
wsl -l -v
Ubuntu-22.04  Running  2

python: 3.8.20
ldd (Ubuntu GLIBC 2.35-0ubuntu3.13) 2.35
```

### Environment

The requested project path is kept as:

```text
/mnt/d/aiproject/PiLoT_work/.conda/pilot22
```

Because package installation directly under `/mnt/d` hit DrvFS `Operation not permitted` errors on binary package files, `.conda/pilot22` is a symlink to an ext4 environment:

```text
/mnt/d/aiproject/PiLoT_work/.conda/pilot22 -> /home/farsee2/pilot22
```

This keeps the activation path stable while avoiding Windows-mounted filesystem binary install failures.

### PyTorch and CUDA

`torch==2.4.1+cu124` was installed successfully. Several large CUDA wheels had to be downloaded with resumable `wget` because pip downloads through the proxy repeatedly timed out or produced hash mismatches.

Final check:

```text
torch: 2.4.1+cu124
cuda available: True
cuda: 12.4
gpu: NVIDIA GeForce GTX 1080
```

Known deviation:

- `triton==3.0.0` is still not installed. Repeated downloads from PyPI / PyTorch were corrupted by network/proxy interruption and failed hash validation.
- `torch` imports and CUDA is available without `triton`; this is enough for the current PiLoT single-image gate unless code paths invoke torch compilation/triton kernels.
- `opencv-python==4.13.0.92` from `environment.yaml` was not used; `opencv-python==4.10.0.84` was installed as a Python 3.8 compatible wheel.

Current `pip check` only reports:

```text
torch 2.4.1+cu124 requires triton, which is not installed.
```

### DirectAbsoluteCostCuda

Direct installation from `/mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda` failed with DrvFS metadata write errors:

```text
Operation not permitted: '/mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda/direct_abs_cost_cuda.egg-info/...'
```

The same source directory was copied to `/tmp/DirectAbsoluteCostCuda_pilot22` and installed from ext4 successfully.

Import check:

```text
direct_abs_cost_cuda import ok
```

This confirms the Ubuntu 20.04 glibc blocker is resolved on Ubuntu 22.04.

### Input Files

```text
configs/caiwangcun_domdsm.yaml
data_caiwangcun/query/images/exif_test/0000.jpg
data_caiwangcun/query/poses/exif_test.txt
data_caiwangcun/reference/caiwangcun_dom.tif
data_caiwangcun/reference/caiwangcun_dsm.tif
data_demo/pretrained_model/model@mapscape@512@Fourier.ckpt
```

Pose line:

```text
0000.jpg 114.4368608916 30.3913609745 391.4620 180.000000 0.000000 -29.200000
```

### Renderer-only Baseline

The existing 512-wide renderer-only check remains valid:

```json
{
  "valid_depth_ratio": 0.99993896484375,
  "depth_min": 360.0,
  "depth_max": 390.0,
  "render_time_sec": 77.93734488100006,
  "image_size": {
    "width": 512,
    "height": 288
  }
}
```

The overlay conclusion from the 512 renderer-only gate was that road, shoreline, and vegetation boundaries are broadly consistent enough to attempt the full single-image PiLoT run.

### RenderLocalizer Initialization

Command:

```bash
python - <<'PY'
import time, yaml, torch
from pixloc.localization.localizer import RenderLocalizer
conf = yaml.safe_load(open("configs/caiwangcun_domdsm.yaml"))["default_confs"]["from_render_test"]
t0 = time.time()
localizer = RenderLocalizer(conf)
print("RenderLocalizer ready", round(time.time() - t0, 3))
PY
```

Result:

```text
RenderLocalizer ready 31.274
```

The checkpoint loaded successfully from:

```text
data_demo/pretrained_model/model@mapscape@512@Fourier.ckpt
```

### Full PiLoT Single-image Run

Command:

```bash
/usr/bin/timeout 240s python -u main.py --config configs/caiwangcun_domdsm.yaml --name exif_test --viz
```

Result:

```text
STATUS:124
```

Failure stage:

- Configuration read: passed.
- Query image list loading: passed; progress showed `1/1`.
- `RenderLocalizer` standalone initialization: passed before the full run.
- Full multiprocessing run: failed/hung during spawned process CUDA tensor reconstruction.

Key traceback:

```text
RuntimeError: CUDA error: invalid resource handle
  File ".../torch/multiprocessing/reductions.py", line 149, in rebuild_cuda_tensor
    storage = storage_cls._new_shared_cuda(
```

No refined pose was produced:

```text
outputs/exif_test/     exists but contains no files
outputs/exif_test.txt  not generated
```

Diagnostic log:

```text
outputs/exif_test_ubuntu22_run.log
```

### Conclusion

Ubuntu 22.04 resolves the `direct_abs_cost_cuda` glibc import blocker. PyTorch 2.4.1+cu124 imports successfully and sees the GTX 1080. `RenderLocalizer` initializes successfully with the local checkpoint.

The current blocker is now inside the full PiLoT multiprocessing path, specifically CUDA tensor sharing/reconstruction under `spawn`, before a refined pose is written. Per the current restriction, `main.py` and `DOMDSMRenderer` were not modified. Next work should focus on a minimal single-image execution path or multiprocessing CUDA handoff fix, not renderer geometry or long-sequence testing.
