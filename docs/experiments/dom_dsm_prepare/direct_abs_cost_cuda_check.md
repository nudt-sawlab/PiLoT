# DirectAbsoluteCostCuda Binary Compatibility Check

## Purpose

P4/P5 showed `overall_loss` is all zeros while full refinement prints `Kernel launch failed: no kernel image is available for execution on the device` on the local GTX 1080. This check diagnoses whether the loaded `direct_abs_cost_cuda` binary supports the current GPU architecture and records fallback options for debugging the loss path.

## Command

```bash
cd /mnt/d/aiproject/PiLoT_work
./.conda/pilot22/bin/python tools/check_direct_abs_cost_cuda.py
```

Output JSON:

```text
docs/experiments/dom_dsm_prepare/direct_abs_cost_cuda_check/binary_check.json
```

## Current Environment

| Field | Value |
| --- | --- |
| Python | `3.8.20` |
| sys.executable | `/mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python` |
| torch | `2.4.1+cu124` |
| torch CUDA | `12.4` |
| CUDA available | `True` |
| GPU | `NVIDIA GeForce GTX 1080` |
| GPU compute capability | `[6, 1]` |

The active GPU is `NVIDIA GeForce GTX 1080`, compute capability `sm_61`.

## Extension Module

| Field | Value |
| --- | --- |
| import success | `True` |
| .so path | `/mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-38-x86_64-linux-gnu.so` |
| has residual_jacobian_batch_quat_cuda | `True` |
| has optimizer_step_cuda | `True` |

Exported symbols:

```text
optimizer_step_cuda
optimizer_step_cuda_v2
optimizer_step_cuda_v3
residual_jacobian_batch_quat_cuda
```

## Binary Architecture

- `cuobjdump` available: `False`
- architecture source for judgment: `strings fallback`
- cuobjdump arches: `[]`
- strings arch markers: `['sm_86']`
- support judgment: `Current direct_abs_cost_cuda binary does not support GTX 1080 sm_61.`

`cuobjdump --list-elf` was not available in this WSL environment, so the script used `strings` as a fallback. The loaded Python 3.8 binary only exposes `sm_86` markers. It does not expose `sm_61` or `compute_61`.

Therefore:

> Current direct_abs_cost_cuda binary does not support GTX 1080 sm_61.

This directly explains the runtime `no kernel image is available for execution on the device` warning when the optimizer kernels are called.

## Why overall_loss Is All Zero

The module imports successfully, so import-only checks are insufficient. The failure appears when runtime kernels such as residual/Jacobian or optimizer-step CUDA code are launched. Because the loaded `.so` lacks the current GPU architecture, those kernels cannot run on GTX 1080. In P4/P5, `overall_loss` being all zeros should be treated as a CUDA loss-path failure symptom, not as successful convergence or a valid zero residual.

## Source and Rebuild Status

- extension dir exists: `True`
- source available: `False`
- files present:

```text
DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-310-x86_64-linux-gnu.so
DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-38-x86_64-linux-gnu.so
DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-39-x86_64-linux-gnu.so
DirectAbsoluteCostCuda/setup.py
```

The local `setup.py` explicitly installs prebuilt `.so` files only. No `.cu`, `.cpp`, `.cc`, `.cuh`, `.h`, or `.hpp` source files are present, so a real local rebuild is not possible from this checkout. I did not attempt to fake a rebuild.

## Recommended Fixes

A. Run on a supported GPU

The current binary appears built for `sm_86`. Re-run refinement on a GPU with compatible architecture, for example an Ampere `sm_86` GPU, to verify whether nonzero optimizer loss and stable updates return.

B. Rebuild from source for GTX 1080

If the internal CUDA source package is available, rebuild with GTX 1080 support:

```bash
cd /mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda
rm -rf build *.egg-info
TORCH_CUDA_ARCH_LIST="6.1" /mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python -m pip install . --force-reinstall -v
```

This command requires actual CUDA source. With the current prebuilt-only `setup.py`, it would only reinstall the existing incompatible binary.

C. Add a PyTorch debug fallback

If source is unavailable, add a slow but inspectable PyTorch implementation or debug fallback for the residual/Jacobian/loss path. That fallback should be used only for diagnostics, not as a claimed production replacement, and should log residual statistics, loss values, update magnitude, and whether gradients/Jacobians are finite.

## Conclusion

The current `direct_abs_cost_cuda` loss path is not usable on the local GTX 1080 because the loaded `.so` does not support `sm_61`. This is the most likely cause of both the runtime kernel warning and the all-zero `overall_loss` observed in P4/P5. Refined poses from this environment should not be treated as trustworthy until the extension is rebuilt for `sm_61`, run on a supported GPU, or replaced by a diagnostic PyTorch fallback for loss/update inspection.
