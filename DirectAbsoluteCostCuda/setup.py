"""Install pre-built direct_abs_cost_cuda extension.

This package distributes only pre-compiled .so binaries.
To rebuild from source (requires CUDA toolkit), use the internal build system.
"""

import glob
import platform
import shutil
import sys
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.install import install


class _BinaryDistribution(Distribution):
    """Force platform-specific wheel."""

    def has_ext_modules(self):
        return True


class _InstallPrebuilt(install):
    """Copy the matching pre-built .so into site-packages."""

    def run(self):
        super().run()
        src_dir = Path(__file__).parent
        ver = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        pattern = f"direct_abs_cost_cuda.{ver}*.so"
        candidates = sorted(src_dir.glob(pattern))
        if not candidates:
            all_so = sorted(src_dir.glob("direct_abs_cost_cuda.*.so"))
            raise FileNotFoundError(
                f"No pre-built .so found for Python {sys.version_info.major}."
                f"{sys.version_info.minor} ({pattern}).\n"
                f"Available: {[s.name for s in all_so]}"
            )
        dst = Path(self.install_lib)
        dst.mkdir(parents=True, exist_ok=True)
        for so in candidates:
            shutil.copy2(so, dst / so.name)
            print(f"Installed {so.name} -> {dst}")


setup(
    name="direct_abs_cost_cuda",
    version="0.1.0",
    description="Pre-built CUDA extension for DirectAbsoluteCost optimization",
    packages=[],
    ext_modules=[],
    cmdclass={"install": _InstallPrebuilt},
    distclass=_BinaryDistribution,
    zip_safe=False,
)
