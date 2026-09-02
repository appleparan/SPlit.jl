#!/bin/sh
# Build a local Julia project at .julia_dev/ that develops SPlit.jl from
# this checkout instead of the git-pinned rev in src/splitiq/juliapkg.json,
# and pins PythonCall to the version juliacall itself requires.
#
# Usage:
#   scripts/setup_julia_dev.sh
#   JULIA=/path/to/julia scripts/setup_julia_dev.sh
#
# Prints the PYTHON_JULIACALL_PROJECT / PYTHON_JULIACALL_EXE exports to use
# for running tests against this dev project, e.g.:
#   eval "$(scripts/setup_julia_dev.sh)"
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pkg_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
repo_root=$(CDPATH= cd -- "$pkg_dir/.." && pwd)
dev_project="$pkg_dir/.julia_dev"

if [ -n "${JULIA:-}" ]; then
  julia_bin="$JULIA"
else
  julia_bin=$(command -v julia) || {
    echo 'error: julia not found on PATH; set JULIA=/path/to/julia' >&2
    exit 1
  }
fi

# Locate juliacall's own juliapkg.json via find_spec, without importing the
# package itself — importing juliacall eagerly resolves/downloads its own
# Julia project and prints [juliapkg] progress lines to stdout, which would
# otherwise contaminate the version string captured below.
pythoncall_version=$(
  cd "$pkg_dir" && uv run python3 -c '
import importlib.util
import json
import pathlib

spec = importlib.util.find_spec("juliacall")
juliacall_dir = pathlib.Path(spec.submodule_search_locations[0])
juliapkg_path = juliacall_dir / "juliapkg.json"
data = json.loads(juliapkg_path.read_text())
print(data["packages"]["PythonCall"]["version"])
'
)
# Strip a leading compat operator (e.g. "=0.9.35" -> "0.9.35"); Pkg.add's
# version kwarg wants a bare version number, not a compat-entry string.
pythoncall_version=$(printf '%s' "$pythoncall_version" | sed 's/^[=^~]*//')

mkdir -p "$dev_project"

SPLITIQ_PYTHONCALL_VERSION="$pythoncall_version" SPLITIQ_REPO_ROOT="$repo_root" \
  "$julia_bin" --startup-file=no --project="$dev_project" -e '
import Pkg

pythoncall_version = ENV["SPLITIQ_PYTHONCALL_VERSION"]
repo_root = ENV["SPLITIQ_REPO_ROOT"]

Pkg.add(Pkg.PackageSpec(name = "PythonCall", version = pythoncall_version))
Pkg.develop(Pkg.PackageSpec(path = repo_root))
Pkg.instantiate()
Pkg.precompile()
'

echo "export PYTHON_JULIACALL_PROJECT=$dev_project"
echo "export PYTHON_JULIACALL_EXE=$julia_bin"
