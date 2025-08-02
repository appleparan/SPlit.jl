"""
Julia interface for SPlit.jl
"""

import os
import sys
from pathlib import Path
from typing import Optional


class JuliaInterface:
    """Interface to Julia SPlit.jl package"""

    def __init__(self):
        self._julia = None
        self._split_jl = None
        self._initialized = False

    def _get_julia_project_path(self) -> str:
        """Get path to Julia project directory"""
        # Assume splitiq package is in SPlit.jl/splitiq/
        current_dir = Path(__file__).parent.parent.parent.parent
        return str(current_dir)

    def initialize(self) -> None:
        """Initialize Julia and load SPlit.jl package"""
        if self._initialized:
            return

        try:
            import juliacall

            # Set Julia project to SPlit.jl directory
            julia_project = self._get_julia_project_path()
            os.environ["JULIA_PROJECT"] = julia_project

            # Import Julia
            from juliacall import Main as jl
            self._julia = jl

            # Load SPlit.jl package
            jl.seval("using Pkg; Pkg.activate(@__DIR__)")
            jl.seval("using SPlit")

            self._split_jl = jl.SPlit
            self._initialized = True

        except ImportError:
            from .exceptions import JuliaInitializationError
            raise JuliaInitializationError(
                "juliacall is required but not installed. "
                "Install with: pip install juliacall"
            )
        except Exception as e:
            from .exceptions import JuliaInitializationError
            raise JuliaInitializationError(f"Failed to initialize Julia: {e}")

    @property
    def julia(self):
        """Get Julia Main module"""
        if not self._initialized:
            self.initialize()
        return self._julia

    @property
    def split_jl(self):
        """Get SPlit.jl module"""
        if not self._initialized:
            self.initialize()
        return self._split_jl
