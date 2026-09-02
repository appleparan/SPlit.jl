"""The Python package shares its version with SPlit.jl."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from splitiq import __version__

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent


def _julia_version() -> str:
    text = (_REPO_ROOT / 'Project.toml').read_text()
    match = re.search(r'^version = "([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((_PKG_ROOT / 'pyproject.toml').read_text())
    assert __version__ == pyproject['project']['version']


def test_version_matches_julia_package() -> None:
    assert __version__ == _julia_version()


def test_juliapkg_pins_the_matching_julia_tag() -> None:
    deps = json.loads((_PKG_ROOT / 'src' / 'splitiq' / 'juliapkg.json').read_text())
    assert deps['packages']['SPlit']['rev'] == f'v{__version__}'
