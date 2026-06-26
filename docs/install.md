# Installation (`pilot` environment)

Both demo cases (SMBU + Feicuiwan) run in the **same** conda environment named
`pilot`. We recommend building it on top of the [CityGaussian](https://github.com/DekuLiuTesla/CityGaussian) `gspl` stack so that CityGaussian and vanilla 3DGS rasterizers coexist.

## 1. Clone repositories

```bash
git clone https://github.com/Choyaa/PiLoT.git
cd PiLoT

git clone https://github.com/DekuLiuTesla/CityGaussian.git third_party/CityGaussian
```

## 2. Create conda env `pilot`

From the CityGaussian repo (provides PyTorch + gspl-compatible CUDA extensions):

```bash
cd third_party/CityGaussian
conda env create -f environment.yml -n pilot   # use CityGaussian's env file
conda activate pilot
cd ../..   # back to PiLoT root
```

If you already have a `gspl` env, you can reuse it and rename, or install PiLoT
deps into `gspl` and `conda activate gspl` — functionally equivalent.

> **Alternative:** `conda env create -f environment.yaml -n pilot` in the PiLoT
> repo works for Feicuiwan-only; for SMBU you still need CityGaussian installed.

## 3. Install PiLoT dependencies

```bash
conda activate pilot
pip install -r requirements.txt
```

## 4. Build the pose optimizer CUDA extension

```bash
cd DirectAbsoluteCostCuda
CUDA_HOME=/usr/local/cuda python setup_build.py install
cd ..
```

## 5. Feicuiwan renderer (vanilla 3DGS PLY)

Required for Case 3 (`feicuiwan.yaml`). SMBU-only users can skip if they never run Feicuiwan.

```bash
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/graphdeco-inria/gaussian-splatting.git#subdirectory=submodules/simple-knn
```

## 6. Download data

From the PiLoT repo root:

```bash
pip install huggingface_hub
./scripts/download_data_demo.sh
```

Layout: [data_demo/README.md](../data_demo/README.md).

## Verify

```bash
conda activate pilot
python -c "import direct_abs_cost_cuda; import torch; print('ok', torch.__version__)"
```
