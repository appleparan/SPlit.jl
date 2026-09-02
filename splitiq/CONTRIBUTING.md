# Contributing to splitiq

Contributions are welcome, and they are greatly appreciated!

## How to contribute

1. Fork [SPlit.jl](https://github.com/appleparan/SPlit.jl) and clone your fork:

    ```bash
    git clone https://github.com/<your-username>/SPlit.jl
    cd SPlit.jl/splitiq
    ```

2. Create a new branch for your changes:

    ```bash
    git checkout -b feat/short-description
    ```

3. Install dependencies and build the local Julia dev project (needed before running tests):

    ```bash
    uv sync --group dev --group docs
    make julia-dev
    ```

4. Make your changes, then run the checks below.

5. Commit with a conventional-commit message and push your branch:

    ```bash
    git commit -m "feat: add a clear and concise commit message"
    git push origin feat/short-description
    ```

6. Open a Pull Request against `appleparan/SPlit.jl` from your branch.

## Reporting issues

Use the repository's [issue tracker](https://github.com/appleparan/SPlit.jl/issues) for bugs
and feature requests. For bugs, include steps to reproduce; for features, explain the problem
it would solve.

## Checks before opening a PR

All commands below run from the `splitiq/` directory unless noted:

```bash
make test        # pytest, against .julia_dev/ built by `make julia-dev`
make lint         # ruff check --fix
make format       # ruff format
make typecheck    # ty check
make docs         # properdocs build --strict
```

Pre-commit hooks are configured at the repository root and must be run from there, not from
`splitiq/`:

```bash
cd .. && uvx pre-commit run -a
```

## Coding style

Follow the conventions already in `src/splitiq/`: type hints and Google-style docstrings on
public functions, single quotes, 100-character lines (enforced by `ruff format`/`ruff check`).

Thank you for contributing.
