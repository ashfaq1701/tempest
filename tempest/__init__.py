"""tempest — temporal random walk sampling with optional GPU acceleration."""
from __future__ import annotations

import ctypes
import glob
import os
import site


def _preload_nvidia_cuda_libs() -> None:
    # RTLD_GLOBAL so the extension's dlopen resolves libcudart/libcurand
    # to the pip-provided nvidia-*-cu12 copies under site-packages.
    site_roots = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        site_roots.append(user_site)

    seen: set[str] = set()
    unique_roots = [r for r in site_roots if not (r in seen or seen.add(r))]

    # add a pattern here if we start linking another CUDA library
    patterns = ("libcudart.so.12*", "libcurand.so.10*")

    for root in unique_roots:
        for pattern in patterns:
            for so_path in glob.glob(
                os.path.join(root, "nvidia", "*", "lib", pattern)
            ):
                try:
                    ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


_preload_nvidia_cuda_libs()

from _tempest import *  # noqa: E402, F401, F403
