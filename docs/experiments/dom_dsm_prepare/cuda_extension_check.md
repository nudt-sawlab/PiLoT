# CUDA Extension Runtime Check

## Purpose

This check verifies whether `direct_abs_cost_cuda` is compatible with the
current GTX 1080 runtime. The yawfix full refinement returns
`run_query_success=true`, but it still prints:

```text
Kernel launch failed: no kernel image is available for execution on the device
```

If the CUDA extension does not support `sm_61`, refined poses should not be
treated as trustworthy.

## Command

Run in Ubuntu-22.04 WSL:

```bash
cd /mnt/d/aiproject/PiLoT_work
./.conda/pilot22/bin/python tools/check_direct_abs_cost_cuda_runtime.py
```

Outputs:

```text
outputs/direct_abs_cost_cuda_check/runtime_check.json
docs/experiments/dom_dsm_prepare/cuda_extension_check/runtime_check.json
```

## Environment

From `runtime_check.json`:

```json
{
  "sys_executable": "/mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python",
  "python_version": "3.8.20",
  "torch_version": "2.4.1+cu124",
  "torch_cuda_version": "12.4",
  "cuda_available": true,
  "device_name_0": "NVIDIA GeForce GTX 1080",
  "device_capability_0": [6, 1]
}
```

The active GPU is therefore:

```text
NVIDIA GeForce GTX 1080, compute capability sm_61
```

## Extension Import

The extension imports successfully:

```json
{
  "direct_abs_cost_cuda_import_success": true,
  "direct_abs_cost_cuda_file": "/mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-38-x86_64-linux-gnu.so"
}
```

Exported callable symbols:

```text
optimizer_step_cuda
optimizer_step_cuda_v2
optimizer_step_cuda_v3
residual_jacobian_batch_quat_cuda
```

Python signatures were unavailable for all exported builtins, so the diagnostic
script did not guess input tensors or call the kernels.

## Binary Architecture Markers

`strings` found these architecture markers in the loaded `.so`:

```text
sm_35
sm_37
sm_50
sm_52
sm_53
sm_60
sm_61
sm_62
sm_70
sm_72
sm_75
sm_80
sm_86
sm_87
sm_89
sm_90
```

The binary contains an `sm_61` marker:

```json
{
  "contains_sm_61_marker": true
}
```

`cuobjdump` was not available in the environment, so the check could not list
the embedded cubins independently.

## Error Pattern Check

The import-only diagnostic did not reproduce any of these patterns:

```json
{
  "no kernel image is available for execution on the device": false,
  "invalid device function": false,
  "CUDA error": false,
  "invalid resource handle": false
}
```

This does not clear the extension at runtime. The warning appears during actual
full refinement, when the optimizer kernels are called. The diagnostic skipped
minimal kernel invocation because the extension exposes no Python signatures
that make a safe minimal test input clear.

## Rebuild Status

The `DirectAbsoluteCostCuda` directory exists, but this checkout contains only
prebuilt `.so` files and `setup.py`:

```text
DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-38-x86_64-linux-gnu.so
DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-39-x86_64-linux-gnu.so
DirectAbsoluteCostCuda/direct_abs_cost_cuda.cpython-310-x86_64-linux-gnu.so
DirectAbsoluteCostCuda/setup.py
```

No `.cu`, `.cpp`, `.cc`, `.cuh`, `.h`, or `.hpp` source files were found, so a
real local rebuild was not attempted.

If the internal source package is restored, rebuild for GTX 1080 with:

```bash
cd /mnt/d/aiproject/PiLoT_work/DirectAbsoluteCostCuda
rm -rf build *.egg-info
TORCH_CUDA_ARCH_LIST="6.1" /mnt/d/aiproject/PiLoT_work/.conda/pilot22/bin/python -m pip install . --force-reinstall -v
```

Note: with the current prebuilt-only `setup.py`, this command would only
reinstall the matching prebuilt `.so`; it would not compile CUDA source.

## Judgment

The binary appears to include `sm_61` markers and imports successfully on the
GTX 1080 environment. That means the failure is not proven to be a simple
"missing sm_61 binary" issue from this diagnostic alone.

However, the actual full refinement still prints `Kernel launch failed: no
kernel image is available for execution on the device` while optimizer kernels
are being used. Because no safe minimal kernel invocation is available and
`cuobjdump` is absent, the runtime compatibility of the called kernels remains
unverified.

Therefore, refined poses should still be treated as potentially affected by the
CUDA extension/runtime path until one of these is done:

- rebuild `DirectAbsoluteCostCuda` from source for `TORCH_CUDA_ARCH_LIST=6.1`,
- run the same refinement on a newer GPU supported by the binary,
- or add a known-safe minimal kernel test using the real extension function
  signatures.

No cudafix rerun was performed because no source rebuild was possible in this
checkout.
