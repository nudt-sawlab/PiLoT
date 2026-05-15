# DirectAbsoluteCostCuda sm_61 Rebuild Check

## Purpose

The current machine is an NVIDIA GeForce GTX 1080 with compute capability `sm_61`. The prebuilt `direct_abs_cost_cuda` binary loaded by Python exposes only an `sm_86` marker, which explains the runtime `no kernel image is available for execution on the device` warning and the all-zero P4/P5 `overall_loss`. This experiment attempted to prepare a rebuild from the early PiLoT CUDA source for `sm_61` and then validate the minimal P4 loss path.

## Source

- repo: `nudt-sawlab/PiLoT`
- commit: `98d5d4eab838e2b813bcb003f14abbb8afa6f7d4`
- files:
  - `DirectAbsoluteCostCuda/DirectAbsoluteCost_cuda.cu`
  - `DirectAbsoluteCostCuda/setup.py`

The local `DirectAbsoluteCostCuda/DirectAbsoluteCost_cuda.cu` and `setup.py` were compared against the early commit and already matched it. The current `setup.py` uses `CUDAExtension(name='direct_abs_cost_cuda', sources=['DirectAbsoluteCost_cuda.cu'])`.

## Interface Check

The early CUDA source exports the symbols required by current `pixloc/pixlib/models/learned_optimizer.py`:

| Symbol/check | Result | Source lines |
| --- | --- | --- |
| `residual_jacobian_batch_quat_cuda` | exists | line 474, pybind line 781 |
| `optimizer_step_cuda` | exists | line 555, pybind line 782 |
| `optimizer_step_cuda_v2/v3` | exists | lines 637/691, pybind lines 783/784 |
| `PYBIND11_MODULE` | exists | line 779 |
| `m.def` exports | exists | lines 781-784 |

Therefore the early source is not blocked by missing Python-facing symbols.

## Build Environment

| Field | Value |
| --- | --- |
| Python | `3.8.20` |
| Python executable | `/mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python` |
| torch | `2.4.1+cu124` |
| torch CUDA | `12.4` |
| CUDA available | `True` |
| GPU | `NVIDIA GeForce GTX 1080` |
| compute capability | `[6, 1]` |
| nvcc | `missing: nvcc: command not found` |
| requested arch | `TORCH_CUDA_ARCH_LIST="6.1"` |

Torch can see the GTX 1080, but the CUDA toolkit compiler is not installed or not on PATH. The environment has PyTorch CUDA runtime support, but not the `nvcc` compiler needed to build a CUDA extension.

## Build Result

Build status: **not attempted after environment gate failed**.

Reason: `nvcc --version` failed with `nvcc: command not found`. Per the task constraints, rebuild must stop here and must not be faked. No `build_ext --inplace` command was run after this failure, and no sm_61 `.so` was produced.

Current prebuilt binaries remain backed up under:

```text
DirectAbsoluteCostCuda/prebuilt_backup/
```

No generated `.so` is intended for git staging.

## Binary Check Result

JSON report: `docs/experiments/dom_dsm_prepare/direct_abs_cost_cuda_check/binary_check_rebuilt_sm61.json`

This report reflects the current loaded binary after the failed environment gate, not a successful rebuild.

| Field | Value |
| --- | --- |
| import success | `True` |
| module path | `/mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-38-x86_64-linux-gnu.so` |
| has residual | `True` |
| has step | `True` |
| cuobjdump available | `False` |
| architecture source | `strings fallback` |
| detected markers | `['sm_86']` |
| current GPU supported | `False` |
| support message | `Current direct_abs_cost_cuda binary does not support GTX 1080 sm_61.` |

The currently loaded binary still exposes only `sm_86` through the fallback marker check and does not support GTX 1080 `sm_61`.

## Minimal P4 Validation

Minimal P4 validation was **not run** because no sm_61 rebuild was possible. Running P4 against the unchanged sm_86 prebuilt binary would only reproduce the known failure and would not validate the requested rebuild.

| Check | Result |
| --- | --- |
| no kernel image warning | not retested after rebuild; no rebuild available |
| run_query_success | not run |
| overall_loss_all_zero | not run |
| refined east/north/alt | not run |
| initial overlap/chamfer | not run |
| refined overlap/chamfer | not run |

## Interpretation

This is case C: rebuild could not proceed because the CUDA compiler is unavailable. The early source has the required pybind interface, so the blocker is not missing symbols. The practical route on this machine remains the PyTorch feature-loss diagnostic/fallback path unless the CUDA toolkit is installed or a compatible sm_61 binary is built elsewhere.

## Next Step

Install a CUDA toolkit matching the PyTorch CUDA line closely enough for extension builds, or use a machine/container with `nvcc` available. Then rerun:

```bash
cd /mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda
rm -rf build
rm -f direct_abs_cost_cuda*.so
export TORCH_CUDA_ARCH_LIST="6.1"
../.conda/pilot22/bin/python setup.py build_ext --inplace
```

After a successful build, rerun the binary check and the minimal P4 validation before doing any larger P4/P5 reruns.
