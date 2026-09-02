# Convenience wrappers around the project's uv-based toolchain.
# Run `make <target>` from the project root.

.PHONY: setup julia-dev format lint typecheck test docs build clean

setup:
	uv sync --group dev --group docs

# Build .julia_dev/, a local Julia project that develops SPlit.jl from this
# checkout instead of the git-pinned rev in src/splitiq/juliapkg.json.
julia-dev:
	./scripts/setup_julia_dev.sh

format:
	uv run ruff format .

lint:
	uv run ruff check --fix .

typecheck:
	uv run ty check

# Runs against .julia_dev/ (see the julia-dev target); run that first.
test:
	PYTHON_JULIACALL_PROJECT=$(CURDIR)/.julia_dev PYTHON_JULIACALL_EXE=$$(command -v julia) uv run pytest

docs:
	uv run mkdocs build --strict

build:
	uv build

clean:
	rm -rf build dist htmlcov site .coverage* .pytest_cache .ruff_cache
