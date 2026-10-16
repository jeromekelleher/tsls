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

import numpy as np

import _tsinfer_hmm

from . import arg_ops


def encode_alleles(ts, alleles):
    """
    Return the (ancestral_state, derived_state) arrays for the specified tree
    sequence, encoding each state as its index in the specified list of
    alleles. Raise a ValueError if any state is not in the list.
    """
    alleles = list(alleles)
    if len(alleles) > np.iinfo(np.int8).max:
        raise ValueError("Too many alleles")
    allele_index = {allele: j for j, allele in enumerate(alleles)}
    if len(allele_index) != len(alleles):
        raise ValueError("Alleles must be unique")

    def encode(states, name):
        unique_states, inverse = np.unique(np.asarray(states), return_inverse=True)
        code = np.zeros(len(unique_states), dtype=np.int8)
        for j, state in enumerate(unique_states):
            if state not in allele_index:
                raise ValueError(f"Unknown {name}: {state!r}")
            code[j] = allele_index[state]
        return code[inverse]

    ancestral_state = encode(ts.sites_ancestral_state, "ancestral state")
    derived_state = encode(ts.mutations_derived_state, "derived state")
    return ancestral_state, derived_state


class MatcherIndexes(_tsinfer_hmm.MatcherIndexes):
    """
    Wrapper around the C MatcherIndexes, built from a tree sequence.

    By default, the ancestral state of every site is 0 and every mutation has
    derived state 1. If ``alleles`` is specified, states are instead encoded
    as their index in this list, and ``num_alleles`` defaults to its length.
    """

    def __init__(self, ts, *, vestigial_root=True, alleles=None, num_alleles=None):
        if vestigial_root:
            ts = arg_ops.add_vestigial_root(ts)
        ancestral_state = None
        derived_state = None
        if alleles is not None:
            ancestral_state, derived_state = encode_alleles(ts, alleles)
            if num_alleles is None:
                num_alleles = np.full(ts.num_sites, len(alleles), dtype=np.uint64)
        tables = ts.dump_tables()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        self.sequence_length = ts.sequence_length
        self.num_sites = ts.num_sites
        super().__init__(
            ll_tables,
            num_alleles=num_alleles,
            ancestral_state=ancestral_state,
            derived_state=derived_state,
        )


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
