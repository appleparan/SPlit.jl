"""Lazy access to the Julia runtime backing splitiq.

Nothing here starts Julia at import time: the interpreter boots on the
first call to :func:`julia`, so importing :mod:`splitiq` stays cheap.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# An opaque Julia value: a `juliacall.Main`/`juliacall.ModuleValue` handle, or
# any Julia object it returns. juliacall ships no type stubs, so this stays
# `Any` deliberately rather than accumulating per-call type-checker ignores.
type JuliaValue = Any

# Defines helpers Julia-side that are awkward to express as one-off `seval`
# calls from Python: DataFrames and CategoricalArrays are not direct
# dependencies of the Julia project juliacall resolves, but are reachable
# through the bindings SPlit itself imports them under.
_HELPER_MODULE_SOURCE = """
module SplitiqHelpers

import SPlit
import PythonCall: pyconvert

function build_dataframe(names, cols)
    names_v = pyconvert(Vector{String}, names)
    cols_v = Vector{Any}(pyconvert(Vector{Any}, cols))
    return SPlit.DataFrames.DataFrame(cols_v, names_v; copycols = false)
end

function categorical_column(values, levels)
    values_v = pyconvert(Vector{String}, values)
    levels_v = pyconvert(Vector{String}, levels)
    return SPlit.CategoricalArrays.CategoricalVector(values_v; levels = levels_v)
end

end # module SplitiqHelpers
"""


@lru_cache(maxsize=1)
def julia() -> JuliaValue:
    """Return the lazily initialized Julia ``Main`` handle with SPlit loaded.

    Julia starts and ``SPlit``/``Random`` are brought into scope on the
    first call; later calls return the cached handle.

    Returns:
        The ``juliacall.Main`` handle, with ``SPlit``, ``Random``, and the
        private ``SplitiqHelpers`` module evaluated into it.
    """
    # `Main` is injected into `juliacall` at runtime by PythonCall, so static
    # analysis of the package cannot see it.
    from juliacall import Main as jl  # noqa: N813  # ty: ignore[unresolved-import]

    jl.seval('using SPlit, Random')
    jl.seval(_HELPER_MODULE_SOURCE)
    return jl


@contextmanager
def _translate_error() -> Iterator[None]:
    """Translate a Julia ``ArgumentError`` into a Python ``ValueError``.

    Every other Julia error propagates unchanged as ``juliacall.JuliaError``,
    so callers that need to translate a different Julia exception type (for
    example ``ErrorException``) should catch ``JuliaError`` around this
    context manager rather than inside it.

    Yields:
        Nothing; the context is used only for its exception translation.

    Raises:
        ValueError: When the wrapped Julia exception is an ``ArgumentError``.
    """
    from juliacall import JuliaError

    try:
        yield
    except JuliaError as exc:
        jl = julia()
        if jl.isa(exc.exception, jl.ArgumentError):
            message = str(jl.sprint(jl.showerror, exc.exception))
            raise ValueError(message) from exc
        raise
