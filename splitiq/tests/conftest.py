"""Pytest configuration: point juliacall at the local dev Julia project.

If `PYTHON_JULIACALL_PROJECT`/`PYTHON_JULIACALL_EXE` are already set (e.g. by
`make test`), they are left untouched; otherwise they default to the
`.julia_dev/` project built by `scripts/setup_julia_dev.sh` and whichever
`julia` is first on `PATH`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault('PYTHON_JULIACALL_PROJECT', str(_PKG_ROOT / '.julia_dev'))

_julia_on_path = shutil.which('julia')
if _julia_on_path is not None:
    os.environ.setdefault('PYTHON_JULIACALL_EXE', _julia_on_path)
