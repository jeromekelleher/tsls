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
Integrity tests for the low-level module.
"""

import inspect
import sys

import numpy as np
import pytest
import tskit

import _tsinfer_hmm
from tsinfer_hmm import arg_ops

IS_WINDOWS = sys.platform == "win32"


class TestUninitialised:
    def test_uninitialised(self):
        for _, cls in inspect.getmembers(_tsinfer_hmm):
            if (
                isinstance(cls, type)
                and not issubclass(cls, Exception)
                and not issubclass(cls, tuple)
            ):
                methods = []
                attributes = []
                for name, value in inspect.getmembers(cls):
                    if not name.startswith("__"):
                        if inspect.isdatadescriptor(value):
                            attributes.append(name)
                        else:
                            methods.append(name)
                uninitialised = cls.__new__(cls)
                for attr in attributes:
                    with pytest.raises((SystemError, ValueError)):
                        getattr(uninitialised, attr)
                    with pytest.raises((SystemError, ValueError, AttributeError)):
                        setattr(uninitialised, attr, None)
                for method_name in methods:
                    method = getattr(uninitialised, method_name)
                    with pytest.raises((SystemError, ValueError)):
                        method()


def make_matcher_indexes_and_matcher(num_samples=4):
    """Build a MatcherIndexes and AncestorMatcher from a small tree sequence."""
    tables = tskit.Tree.generate_balanced(num_samples).tree_sequence.dump_tables()
    tables.sequence_length = 2
    tables.edges.right = np.full(len(tables.edges), 2, dtype=np.float64)
    tables.sites.add_row(position=1, ancestral_state="A")
    tables.mutations.add_row(site=0, node=1, derived_state="T")
    ts = tables.tree_sequence()
    ts = arg_ops.add_vestigial_root(ts)
    ll_tables = _tsinfer_hmm.LightweightTableCollection(ts.sequence_length)
    ll_tables.fromdict(ts.dump_tables().asdict())
    mi = _tsinfer_hmm.MatcherIndexes(ll_tables)
    recombination = np.array([1e-9])
    mismatch = np.array([0.0])
    am = _tsinfer_hmm.AncestorMatcher(mi, recombination=recombination, mismatch=mismatch)
    return ts, mi, am


class TestAncestorMatcher:
    def test_total_memory_before_find_path(self):
        _, _, am = make_matcher_indexes_and_matcher()
        mem = am.total_memory
        if IS_WINDOWS:
            assert mem >= 2**31
        else:
            assert isinstance(mem, int)
            # Before find_path, only the initial block is allocated
            assert mem > 0

    def test_total_memory_after_find_path(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites)
        mem = am.total_memory
        if IS_WINDOWS:
            assert mem >= 2**31
        else:
            assert isinstance(mem, int)
            assert mem > 0

    def test_mean_traceback_size(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites)
        tb = am.mean_traceback_size
        assert isinstance(tb, float)
        assert tb >= 0

    def test_init_bad_recombination(self):
        ts, mi, _ = make_matcher_indexes_and_matcher()
        mismatch = np.array([0.0])
        with pytest.raises(ValueError):
            _tsinfer_hmm.AncestorMatcher(mi, recombination="bad", mismatch=mismatch)
        with pytest.raises(ValueError, match="Size of recombination"):
            _tsinfer_hmm.AncestorMatcher(
                mi, recombination=np.array([1e-9, 1e-9]), mismatch=mismatch
            )

    def test_init_bad_mismatch(self):
        ts, mi, _ = make_matcher_indexes_and_matcher()
        recombination = np.array([1e-9])
        with pytest.raises(ValueError):
            _tsinfer_hmm.AncestorMatcher(mi, recombination=recombination, mismatch="bad")
        with pytest.raises(ValueError, match="Size of mismatch"):
            _tsinfer_hmm.AncestorMatcher(
                mi, recombination=recombination, mismatch=np.array([0.0, 0.0])
            )

    def test_init_extended_checks(self):
        ts, mi, _ = make_matcher_indexes_and_matcher()
        am = _tsinfer_hmm.AncestorMatcher(
            mi,
            recombination=np.array([1e-9]),
            mismatch=np.array([0.0]),
            extended_checks=True,
        )
        assert am is not None

    def test_init_weight_by_n_false(self):
        ts, mi, _ = make_matcher_indexes_and_matcher()
        am = _tsinfer_hmm.AncestorMatcher(
            mi,
            recombination=np.array([1e-9]),
            mismatch=np.array([0.0]),
            weight_by_n=False,
        )
        assert am is not None

    def test_find_path_bad_haplotype(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        with pytest.raises(ValueError):
            am.find_path("not an array", 0, ts.num_sites)
        with pytest.raises(ValueError, match="Dim != 1"):
            am.find_path(np.zeros((1, 1), dtype=np.int8), 0, ts.num_sites)
        with pytest.raises(ValueError, match="Incorrect size"):
            am.find_path(np.zeros(100, dtype=np.int8), 0, ts.num_sites)

    def test_find_path_returns_values(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        result = am.find_path(h, 0, ts.num_sites)
        assert len(result) == 5
        num_edges, left, right, parent, match = result
        assert isinstance(num_edges, int)
        assert left.dtype == np.uint32
        assert right.dtype == np.uint32
        assert parent.dtype == np.int32
        assert match.dtype == np.int8

    def test_find_path_match_impossible(self):
        # mu=0 with a mismatching haplotype makes p_e=0 for all nodes,
        # triggering TSI_ERR_MATCH_IMPOSSIBLE_EXTREME_MUTATION_PROBA.
        # Build a tree sequence where all nodes carry allele 0 at a site
        # (no mutations), then query h=[1] so nothing matches.
        tables = tskit.Tree.generate_balanced(4).tree_sequence.dump_tables()
        tables.sequence_length = 2
        tables.edges.right = np.full(len(tables.edges), 2, dtype=np.float64)
        tables.sites.add_row(position=1, ancestral_state="A")
        # No mutations: all nodes carry allele 0
        ts = tables.tree_sequence()
        ts = arg_ops.add_vestigial_root(ts)
        ll_tables = _tsinfer_hmm.LightweightTableCollection(ts.sequence_length)
        ll_tables.fromdict(ts.dump_tables().asdict())
        mi = _tsinfer_hmm.MatcherIndexes(ll_tables)
        am = _tsinfer_hmm.AncestorMatcher(
            mi,
            recombination=np.array([1e-9]),
            mismatch=np.array([0.0]),
        )
        h = np.array([1], dtype=np.int8)
        with pytest.raises(_tsinfer_hmm.MatchImpossible):
            am.find_path(h, 0, ts.num_sites)

    def test_get_traceback(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites)
        for site in range(ts.num_sites):
            tb = am.get_traceback(site)
            assert isinstance(tb, dict)

    def test_get_traceback_bad_site(self):
        ts, _, am = make_matcher_indexes_and_matcher()
        h = np.zeros(ts.num_sites, dtype=np.int8)
        am.find_path(h, 0, ts.num_sites)
        with pytest.raises(ValueError, match="site out of range"):
            am.get_traceback(ts.num_sites)


class TestMatcherIndexes:
    def test_single_tree(self):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        ts = arg_ops.add_vestigial_root(ts)
        tables = ts.dump_tables()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        _ = _tsinfer_hmm.MatcherIndexes(ll_tables)

    def test_node_0_not_root(self):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        tables = ts.dump_tables()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        with pytest.raises(_tsinfer_hmm.LibraryError, match="Node 0 must be the root"):
            _tsinfer_hmm.MatcherIndexes(ll_tables)

    def test_num_alleles(self):
        ts, mi, _ = make_matcher_indexes_and_matcher()
        num_alleles = np.array([2] * ts.num_sites, dtype=np.uint64)
        ll_tables = _tsinfer_hmm.LightweightTableCollection(ts.sequence_length)
        ll_tables.fromdict(ts.dump_tables().asdict())
        mi = _tsinfer_hmm.MatcherIndexes(ll_tables, num_alleles=num_alleles)
        assert mi is not None

    def test_allele_encoding(self):
        ts, _, _ = make_matcher_indexes_and_matcher()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(ts.sequence_length)
        ll_tables.fromdict(ts.dump_tables().asdict())
        mi = _tsinfer_hmm.MatcherIndexes(
            ll_tables,
            num_alleles=np.full(ts.num_sites, 4, dtype=np.uint64),
            ancestral_state=np.zeros(ts.num_sites, dtype=np.int8),
            derived_state=np.full(ts.num_mutations, 3, dtype=np.int8),
        )
        assert mi is not None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_alleles": np.full(3, 2, dtype=np.uint64)},
            {"ancestral_state": np.zeros(3, dtype=np.int8)},
            {"derived_state": np.zeros(3, dtype=np.int8)},
        ],
    )
    def test_allele_encoding_bad_length(self, kwargs):
        ts, _, _ = make_matcher_indexes_and_matcher()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(ts.sequence_length)
        ll_tables.fromdict(ts.dump_tables().asdict())
        with pytest.raises(ValueError):
            _tsinfer_hmm.MatcherIndexes(ll_tables, **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"ancestral_state": [2]}, "Ancestral state"),
            ({"ancestral_state": [-1]}, "Ancestral state"),
            ({"derived_state": [2]}, "derived state"),
            ({"derived_state": [-1]}, "derived state"),
        ],
    )
    def test_allele_encoding_bad_state(self, kwargs, match):
        ts, _, _ = make_matcher_indexes_and_matcher()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(ts.sequence_length)
        ll_tables.fromdict(ts.dump_tables().asdict())
        with pytest.raises(_tsinfer_hmm.LibraryError, match=match):
            _tsinfer_hmm.MatcherIndexes(ll_tables, **kwargs)

    def test_print_state(self, tmpdir):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        ts = arg_ops.add_vestigial_root(ts)
        tables = ts.dump_tables()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        mi = _tsinfer_hmm.MatcherIndexes(ll_tables)
        with pytest.raises(TypeError):
            mi.print_state()

        path = tmpdir / "output.txt"
        with open(path, "w") as f:
            mi.print_state(f)
        with open(path) as f:
            output = f.read()
        assert len(output) > 0
        assert "indexes" in output

    def test_print_state_bad_file(self):
        ts = tskit.Tree.generate_balanced(4).tree_sequence
        ts = arg_ops.add_vestigial_root(ts)
        tables = ts.dump_tables()
        ll_tables = _tsinfer_hmm.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        mi = _tsinfer_hmm.MatcherIndexes(ll_tables)
        with pytest.raises(TypeError):
            mi.print_state("not a file")
