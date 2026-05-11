#!/usr/bin/env python3
"""Inspect direct_abs_cost_cuda binary architecture support."""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
OUT_DIR = REPO_ROOT / "docs/experiments/dom_dsm_prepare/direct_abs_cost_cuda_check"
OUT_JSON = OUT_DIR / "binary_check.json"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _run(cmd: List[str], timeout: int = 60) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.fspath(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "command": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _find_tool(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    for candidate in [
        Path("/usr/local/cuda/bin") / name,
        Path("/usr/local/cuda-12/bin") / name,
        Path("/usr/local/cuda-11/bin") / name,
    ]:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return os.fspath(candidate)
    return None


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "sys_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "exception": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_compute_capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        info["exception"] = f"{type(exc).__name__}: {exc}"
    return info


def _import_extension() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "import_success": False,
        "direct_abs_cost_cuda_file": None,
        "symbols": [],
        "has_residual_jacobian_batch_quat_cuda": False,
        "has_optimizer_step_cuda": False,
        "exception": None,
    }
    if EXT_DIR.exists() and os.fspath(EXT_DIR) not in sys.path:
        sys.path.insert(0, os.fspath(EXT_DIR))
    try:
        import torch  # noqa: F401
        import direct_abs_cost_cuda

        result["import_success"] = True
        result["direct_abs_cost_cuda_file"] = getattr(direct_abs_cost_cuda, "__file__", None)
        symbols = sorted(name for name in dir(direct_abs_cost_cuda) if not name.startswith("__"))
        result["symbols"] = symbols
        result["has_residual_jacobian_batch_quat_cuda"] = hasattr(
            direct_abs_cost_cuda, "residual_jacobian_batch_quat_cuda"
        )
        result["has_optimizer_step_cuda"] = hasattr(direct_abs_cost_cuda, "optimizer_step_cuda")
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    return result


def _source_status() -> Dict[str, Any]:
    source_exts = {".cu", ".cpp", ".cc", ".cxx", ".cuh", ".h", ".hpp"}
    files = []
    if EXT_DIR.exists():
        files = [p.relative_to(REPO_ROOT).as_posix() for p in EXT_DIR.rglob("*") if p.is_file()]
    source_files = [f for f in files if Path(f).suffix in source_exts]
    return {
        "extension_dir_exists": EXT_DIR.exists(),
        "files": files,
        "source_files": source_files,
        "source_available": bool(source_files),
    }


def _parse_arches(text: str) -> List[str]:
    arches = set(re.findall(r"\bsm_[0-9]+\b", text))
    arches.update(re.findall(r"\bcompute_[0-9]+\b", text))
    # cuobjdump can also print arch = sm_XX.
    arches.update(re.findall(r"arch\s*=\s*(sm_[0-9]+)", text))
    return sorted(arches, key=lambda item: (item.split("_")[0], int(item.split("_")[1])))


def _binary_arch_info(so_path: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "so_path": so_path,
        "cuobjdump_available": False,
        "cuobjdump_path": None,
        "cuobjdump_list_elf": None,
        "cuobjdump_arches": [],
        "strings_available": False,
        "strings_arch_markers": [],
        "architecture_source_for_judgment": None,
    }
    if not so_path or not Path(so_path).exists():
        return info
    cuobjdump = _find_tool("cuobjdump")
    if cuobjdump:
        info["cuobjdump_available"] = True
        info["cuobjdump_path"] = cuobjdump
        result = _run([cuobjdump, "--list-elf", so_path], timeout=120)
        info["cuobjdump_list_elf"] = result
        combined = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
        info["cuobjdump_arches"] = _parse_arches(combined)
        if info["cuobjdump_arches"]:
            info["architecture_source_for_judgment"] = "cuobjdump --list-elf"
    strings_tool = _find_tool("strings") or shutil.which("strings")
    if strings_tool:
        result = _run([strings_tool, so_path], timeout=120)
        info["strings_available"] = result["returncode"] == 0
        info["strings_arch_markers"] = _parse_arches(result.get("stdout") or "")
        if not info["architecture_source_for_judgment"] and info["strings_arch_markers"]:
            info["architecture_source_for_judgment"] = "strings fallback"
    return info


def _supports_current_gpu(torch_info: Dict[str, Any], arch_info: Dict[str, Any]) -> Tuple[Optional[bool], str, List[str]]:
    capability = torch_info.get("gpu_compute_capability")
    if not capability:
        return None, "GPU compute capability unavailable", []
    sm = f"sm_{int(capability[0])}{int(capability[1])}"
    compute = f"compute_{int(capability[0])}{int(capability[1])}"
    arches = arch_info.get("cuobjdump_arches") or arch_info.get("strings_arch_markers") or []
    if not arches:
        return None, f"No architecture markers found for current GPU {sm}", arches
    supported = sm in arches or compute in arches
    if supported:
        return True, f"Current GPU capability {sm} appears present in direct_abs_cost_cuda binary.", arches
    return False, f"Current direct_abs_cost_cuda binary does not support GTX 1080 {sm}.", arches


def main() -> int:
    torch_info = _torch_info()
    ext_info = _import_extension()
    arch_info = _binary_arch_info(ext_info.get("direct_abs_cost_cuda_file"))
    supported, support_message, judged_arches = _supports_current_gpu(torch_info, arch_info)
    source_status = _source_status()
    result = {
        "torch": torch_info,
        "direct_abs_cost_cuda": ext_info,
        "binary_architecture": arch_info,
        "current_gpu_supported_by_binary": supported,
        "support_message": support_message,
        "judged_arches": judged_arches,
        "source_status": source_status,
        "overall_loss_zero_interpretation": (
            "If optimizer kernels fail at runtime, residual/loss computation can be invalid or skipped; "
            "P4/P5 overall_loss all-zero should therefore be treated as CUDA loss-path failure evidence, "
            "not as successful convergence."
        ),
        "recommendations": [
            "Run on a GPU whose SM architecture is verified inside the loaded .so.",
            "If CUDA source is available, rebuild with TORCH_CUDA_ARCH_LIST=\"6.1\" for GTX 1080.",
            "If source is unavailable, add a PyTorch debug fallback for residual/Jacobian/loss inspection instead of relying on direct_abs_cost_cuda.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if supported is False:
        print("Current direct_abs_cost_cuda binary does not support GTX 1080 sm_61.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
