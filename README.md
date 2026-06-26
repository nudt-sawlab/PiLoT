# PiLoT-Train

Training code for PiLoT.

## Requirements

- Linux with NVIDIA GPU (CUDA)
- Conda 

## Installation (conda, env name: `pilot-train`)

```bash
cd PiLoT-Train
conda env create -f environment.yml
conda activate pilot-train
```

## Dataset layout

Point `dataset_dir` in the train config to your dataset root. Expected structure:

```
<dataset_dir>/
  Train/
  Validation/
  ...
```

Download from [Hugging Face](https://huggingface.co/datasets/choyaa/PiLoT-data) (`choyaa/PiLoT-data`):

```bash
# install CLI if needed: pip install huggingface_hub
hf download choyaa/PiLoT-data --repo-type dataset --local-dir /path/to/your/dataset
```

This creates `/path/to/your/dataset/Train/` and `/path/to/your/dataset/Validation/`. To download only one split:

```bash
hf download choyaa/PiLoT-data Train --repo-type dataset --local-dir /path/to/your/dataset
hf download choyaa/PiLoT-data Validation --repo-type dataset --local-dir /path/to/your/dataset
```

Training configs (under `configs/train/`):

| File | Purpose |
|------|---------|
| `train_Aero_seq_pilot_fusion_crop_512.yaml` | **Template** with `${work_dir}` and `/path/to/your/dataset` |
| `train_Aero_seq_pilot_fusion_crop_512.local.yaml` | **Local** config with machine-specific paths (gitignored) |

Copy the template for a new machine:

```bash
cp configs/train/train_Aero_seq_pilot_fusion_crop_512.yaml \
   configs/train/train_Aero_seq_pilot_fusion_crop_512.local.yaml
```

Then edit `save_dir`, `dataset_dir`, and `device.gpu_ids` in the `.local.yaml` file.

## Configure training 

```yaml
save_dir: ${work_dir}/workspace/train_Aero_seq_pilot_fusion
data:
  dataset_dir: /path/to/your/dataset
device:
  gpu_ids: [0， 1, 2, 3, 4, 5, 6, 7]
```

## Run training

```bash
cd PiLoT-Train
python run.py python run.py +train=train_Aero_seq_pilot_fusion_crop_512
```

Logs and checkpoints are written under `save_dir` (default: `workspace/train_Aero_seq_pilot_fusion/` relative to `work_dir`).

## TensorBoard

```bash
tensorboard --logdir=workspace/train_Aero_seq_pilot_fusion --port 8123
```

