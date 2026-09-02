"""splitiq package.

Optimal train/test splitting via support points, backed by SPlit.jl
"""

from splitiq._version import __version__
from splitiq.estimators import Exact, RandomFeatures, RandomSlices, Subsample
from splitiq.quality import energydistance, mmd, splitquality
from splitiq.ratio import optimal_split_ratio
from splitiq.split import SplitResult, datasplit

__author__ = """Jongsu Liam Kim"""
__email__ = 'jongsukim8@gmail.com'

__all__ = [
    'Exact',
    'RandomFeatures',
    'RandomSlices',
    'SplitResult',
    'Subsample',
    '__version__',
    'datasplit',
    'energydistance',
    'mmd',
    'optimal_split_ratio',
    'splitquality',
]
