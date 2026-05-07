# Ubuntu 22.04 Environment Check

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

This document prepares the Ubuntu 22.04 reproduction chain. It has not yet run inside Ubuntu 22.04 on this machine because the currently installed WSL distribution is `Ubuntu-20.04`.

Next gate: install/start `Ubuntu-22.04`, create `.conda/pilot22`, and stop immediately if `direct_abs_cost_cuda` still fails to import.
