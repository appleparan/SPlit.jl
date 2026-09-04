"""splitiq package.

Distribution-preserving subset selection for tabular data and embeddings:
train/test splits, k-fold multiplets, and training-data selection, backed by
SPlit.jl
"""

from splitiq._version import __version__
from splitiq.comparison import SplitComparison, compare
from splitiq.estimators import Exact, RandomFeatures, RandomSlices, Subsample
from splitiq.multiplet import multiplet
from splitiq.quality import energydistance, mmd, splitquality
from splitiq.ratio import optimal_split_ratio
from splitiq.split import SplitResult, datasplit, select_rows

__author__ = """Jongsu Liam Kim"""
__email__ = 'jongsukim8@gmail.com'

__all__ = [
    'Exact',
    'RandomFeatures',
    'RandomSlices',
    'SplitComparison',
    'SplitResult',
    'Subsample',
    '__version__',
    'compare',
    'datasplit',
    'energydistance',
    'mmd',
    'multiplet',
    'optimal_split_ratio',
    'select_rows',
    'splitquality',
]
