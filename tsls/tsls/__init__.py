"""
The Li and Stephens HMM on a tree sequence.
"""

from _tsls import LibraryError, MatchImpossible  # noqa: F401

from .matching import (  # noqa: F401
    AncestorMatcher,
    CompressedMatrix,
    MatcherIndexes,
    encode_alleles,
    load_compressed_matrix,
)
