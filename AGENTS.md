# AGENTS.md — PiLoT agent guide

Instructions for AI coding assistants (Cursor, Claude Code, Codex, Windsurf, …)
working in this repository.

## What this repo does

**PiLoT** localizes a UAV camera from a query image + a geo-referenced 3D map.
Two processes run in parallel (`main.py`):

1. **Render worker** — renders color + depth from the current pose estimate.
2. **Localization worker** — refines pose via dense feature alignment (PixLoc).

Closed loop: localization output → next render pose → next frame.

## Repo layout

```
main.py                     # entry point, dual-process orchestration
configs/demos/              # official YAML configs (start here)
pixloc/localization/        # refiner, feature extraction
pixloc/pixlib/models/       # learned optimizer (CUDA)
pixloc/utils/citygs/        # CityGaussian renderer
pixloc/utils/gs3d/          # vanilla 3DGS PLY renderer
pixloc/utils/osg/           # legacy OSG / 3D Tiles renderer
scripts/                    # run & download helpers
data_demo/                  # demo data (not in git; download from HF)
DirectAbsoluteCostCuda/     # build pose optimizer CUDA ext
```

## Install (conda env: `pilot`)

Prerequisites: Linux, NVIDIA GPU, CUDA, conda.

```bash
git clone https://github.com/Choyaa/PiLoT.git && cd PiLoT

# CityGaussian (SMBU renderer) — also accepts ../CityGaussian if already cloned
git clone https://github.com/DekuLiuTesla/CityGaussian.git third_party/CityGaussian

cd third_party/CityGaussian
conda env create -f environment.yml -n pilot
conda activate pilot
cd ../..

pip install -r requirements.txt

cd DirectAbsoluteCostCuda && CUDA_HOME=/usr/local/cuda python setup_build.py install && cd ..

# Feicuiwan PLY renderer only:
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/graphdeco-inria/gaussian-splatting.git#subdirectory=submodules/simple-knn
```

Verify:

```bash
python -c "import direct_abs_cost_cuda, torch; print('ok', torch.__version__)"
```

Full details: [docs/install.md](docs/install.md)

## Download demo data

```bash
pip install huggingface_hub
./scripts/download_data_demo.sh
```

Source: [Hugging Face — PiLoT-data](https://huggingface.co/datasets/choyaa/PiLoT-data)  
Layout: [data_demo/README.md](data_demo/README.md)

## Run demos

```bash
conda activate pilot

# SMBU seq2 (CityGaussian map, normalized coords)
./scripts/run_smbu_seq2.sh
./scripts/run_smbu_seq2.sh --viz

# Feicuiwan (vanilla 3DGS PLY, ECEF coords)
./scripts/run_feicuiwan.sh
./scripts/run_feicuiwan.sh 3dgs_test --viz
```

Outputs: `outputs/<output_name>.txt` (+ `outputs/<name>/` images and `visualization.mp4` with `--viz`).

Direct Python:

```bash
python main.py -c configs/demos/smbu_seq2.yaml
python main.py -c configs/demos/feicuiwan.yaml --name 3dgs_test
```

## Config cheat sheet

| Demo | Config | `render_config.type` | Coordinates | Pose columns |
|------|--------|----------------------|-------------|--------------|
| SMBU seq2 | `configs/demos/smbu_seq2.yaml` | `citygs` | normalized `x y z` | `name x y z pitch roll yaw` |
| Feicuiwan | `configs/demos/feicuiwan.yaml` | `3dgs` | ECEF `lon lat alt` | `name lon lat alt roll pitch yaw` |

Key env vars (optional overrides):

- `CITYGAUSSIAN_ROOT` — CityGaussian clone (auto-detects `../CityGaussian`)
- `SMBU_MODEL_DIR` — default `data_demo/smbu_model`

## Common agent tasks

| Task | What to read / edit |
|------|---------------------|
| Install from scratch | This file → `docs/install.md` |
| Run a demo | `configs/demos/*.yaml`, `scripts/run_*.sh` |
| Prepare own sequence | [docs/custom_data.md](docs/custom_data.md) |
| Swap / add map renderer | [docs/render_backends.md](docs/render_backends.md) |
| Coordinate conventions | [docs/coordinate_systems.md](docs/coordinate_systems.md) |
| Tune localization | `configs/demos/*.yaml` → `refinement`, `refine` sections; `pixloc/localization/base_refiner.py` |

## Do not

- Commit `data_demo/` contents (large; use Hugging Face).
- Hard-code machine-specific absolute paths in `configs/demos/` — use `${ENV:-default}`.
- Change `DirectAbsoluteCostCuda/` iteration counts (`num_iters==4` / `==2`) without reading `learned_optimizer.py` — they trigger special modes.

## Citation

```bibtex
@inproceedings{cheng2026pilot,
  title={PiLoT: Neural Pixel-to-3D Registration for UAV-based Ego and Target Geo-localization},
  author={Cheng, Xiaoya and Wang, Long and Liu, Yan and Liu, Xinyi and Tan, Hanlin and Liu, Yu and Zhang, Maojun and Yan, Shen},
  booktitle={CVPR}, year={2026}
}
```
