#!/usr/bin/env python3
"""Runtime diagnostics for the direct_abs_cost_cuda extension."""

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = REPO_ROOT / "outputs/direct_abs_cost_cuda_check/runtime_check.json"
DOC_JSON = (
    REPO_ROOT
    / "docs/experiments/dom_dsm_prepare/cuda_extension_check/runtime_check.json"
)
EXT_DIR = REPO_ROOT / "DirectAbsoluteCostCuda"
ERROR_PATTERNS = [
    "no kernel image is available for execution on the device",
    "invalid device function",
    "CUDA error",
    "invalid resource handle",
]


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _run_child_import(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    code = r"""
import inspect
import json
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo / "DirectAbsoluteCostCuda"))

result = {
    "import_success": False,
    "direct_abs_cost_cuda_file": None,
    "module_symbols": [],
    "callable_symbols": [],
    "signatures": {},
    "exception_type": None,
    "exception": None,
}

try:
    import torch  # noqa: F401
    import direct_abs_cost_cuda
    result["import_success"] = True
    result["direct_abs_cost_cuda_file"] = getattr(direct_abs_cost_cuda, "__file__", None)
    symbols = [name for name in dir(direct_abs_cost_cuda) if not name.startswith("__")]
    result["module_symbols"] = symbols
    for name in symbols:
        value = getattr(direct_abs_cost_cuda, name)
        if callable(value):
            result["callable_symbols"].append(name)
            try:
                result["signatures"][name] = str(inspect.signature(value))
            except Exception as exc:
                result["signatures"][name] = f"unavailable: {type(exc).__name__}: {exc}"
except Exception as exc:
    result["exception_type"] = type(exc).__name__
    result["exception"] = str(exc)

print(json.dumps(result, sort_keys=True))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(EXT_DIR)
        + os.pathsep
        + str(REPO_ROOT)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.fspath(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    parsed: Dict[str, Any]
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        parsed = {
            "import_success": False,
            "direct_abs_cost_cuda_file": None,
            "module_symbols": [],
            "callable_symbols": [],
            "signatures": {},
            "exception_type": type(exc).__name__,
            "exception": f"Could not parse child JSON: {exc}",
        }
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "parsed": parsed,
    }


def _torch_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_import_success": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": None,
        "device_name_0": None,
        "device_capability_0": None,
        "torch_lib_dir": None,
        "exception": None,
    }
    try:
        import torch

        info["torch_import_success"] = True
        info["torch_version"] = torch.__version__
        info["torch_cuda_version"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_lib_dir"] = str(Path(torch.__file__).resolve().parent / "lib")
        if torch.cuda.is_available():
            info["device_name_0"] = torch.cuda.get_device_name(0)
            info["device_capability_0"] = list(torch.cuda.get_device_capability(0))
    except Exception as exc:
        info["exception"] = f"{type(exc).__name__}: {exc}"
    return info


def _candidate_so_for_python() -> Optional[Path]:
    ver = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    candidates = sorted(EXT_DIR.glob(f"direct_abs_cost_cuda.{ver}*.so"))
    if candidates:
        return candidates[0]
    candidates = sorted(EXT_DIR.glob("direct_abs_cost_cuda.*.so"))
    return candidates[0] if candidates else None


def _run_text_command(command: List[str], timeout: int = 30) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=os.fspath(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def _binary_arch_info(module_path: Optional[str]) -> Dict[str, Any]:
    so_path = Path(module_path) if module_path else _candidate_so_for_python()
    info: Dict[str, Any] = {
        "so_path": str(so_path) if so_path else None,
        "strings_arch_markers": [],
        "contains_sm_61_marker": False,
        "cuobjdump_available": False,
        "cuobjdump_result": None,
    }
    if not so_path or not so_path.exists():
        return info

    strings_result = _run_text_command(["strings", os.fspath(so_path)], timeout=60)
    markers = sorted(set(re.findall(r"(?:sm|compute)_[0-9]+", strings_result["stdout"])))
    info["strings_arch_markers"] = markers
    info["contains_sm_61_marker"] = "sm_61" in markers or "compute_61" in markers

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump:
        info["cuobjdump_available"] = True
        info["cuobjdump_result"] = _run_text_command([cuobjdump, "--list-elf", os.fspath(so_path)], timeout=60)
    return info


def _detect_patterns(*texts: str) -> Dict[str, bool]:
    combined = "\n".join(texts)
    return {pattern: pattern in combined for pattern in ERROR_PATTERNS}


def main() -> int:
    torch_info = _torch_info()
    torch_lib_dir = torch_info.get("torch_lib_dir")

    plain_attempt = _run_child_import()
    env_attempt = None
    if torch_lib_dir:
        env_attempt = _run_child_import(
            {
                "LD_LIBRARY_PATH": str(torch_lib_dir)
                + os.pathsep
                + os.environ.get("LD_LIBRARY_PATH", "")
            }
        )

    selected_attempt = plain_attempt
    if not selected_attempt["parsed"].get("import_success") and env_attempt:
        selected_attempt = env_attempt

    import_result = selected_attempt["parsed"]
    module_path = import_result.get("direct_abs_cost_cuda_file")
    arch_info = _binary_arch_info(module_path)

    captured_texts = [
        plain_attempt.get("stdout", ""),
        plain_attempt.get("stderr", ""),
    ]
    if env_attempt:
        captured_texts.extend([env_attempt.get("stdout", ""), env_attempt.get("stderr", "")])

    result: Dict[str, Any] = {
        "sys_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        **torch_info,
        "direct_abs_cost_cuda_import_success": bool(import_result.get("import_success")),
        "direct_abs_cost_cuda_file": module_path,
        "plain_import_attempt": plain_attempt,
        "torch_lib_env_import_attempt": env_attempt,
        "selected_import_result": import_result,
        "module_symbols": import_result.get("module_symbols", []),
        "callable_symbols": import_result.get("callable_symbols", []),
        "callable_signatures": import_result.get("signatures", {}),
        "minimal_function_test": {
            "attempted": False,
            "reason": (
                "No function was called because the extension exposes no Python "
                "signature information sufficient to construct a safe minimal input."
            ),
        },
        "captured_error_patterns": _detect_patterns(*captured_texts),
        "binary_arch_info": arch_info,
        "direct_absolute_cost_cuda_source_files": sorted(
            str(path.relative_to(REPO_ROOT))
            for path in EXT_DIR.glob("*")
            if path.suffix in {".cu", ".cpp", ".cc", ".cuh", ".h", ".hpp"}
        ),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOC_JSON.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_jsonable(result), indent=2, sort_keys=True)
    OUTPUT_JSON.write_text(text + "\n", encoding="utf-8")
    DOC_JSON.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["direct_abs_cost_cuda_import_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
