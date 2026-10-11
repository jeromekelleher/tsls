#
# MIT License
#
# Copyright (C) 2018-2026 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Wrappers around the low-level tsinfer HMM matching engine.
"""

import _tsinfer_hmm

from . import arg_ops


class MatcherIndexes(_tsinfer_hmm.MatcherIndexes):
    """Wrapper around the C MatcherIndexes, built from a tree sequence."""

    def __init__(self, ts, *, vestigial_root=True, num_alleles=None):
        if vestigial_root:
            ts = arg_ops.add_vestigial_root(ts)
        tables = ts.dump_tables()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        self.sequence_length = ts.sequence_length
        self.num_sites = ts.num_sites
        super().__init__(ll_tables, num_alleles=num_alleles)


class AncestorMatcher(_tsinfer_hmm.AncestorMatcher):
    """Wrapper with the same ``find_path`` API as ``AncestorMatcher``."""

    def __init__(
        self,
        matcher_indexes,
        recombination,
        mismatch,
        likelihood_threshold=None,
        weight_by_n=None,
    ):
        kwargs = {}
        if likelihood_threshold is not None:
            kwargs["likelihood_threshold"] = likelihood_threshold
        if weight_by_n is not None:
            kwargs["weight_by_n"] = weight_by_n
        super().__init__(
            matcher_indexes,
            recombination=recombination,
            mismatch=mismatch,
            **kwargs,
        )

    def find_path(self, h, start, end, match_out):
        path_len, left, right, parent, matched = super().find_path(h, start, end)
        match_out[:] = matched
        return left[:path_len], right[:path_len], parent[:path_len]
