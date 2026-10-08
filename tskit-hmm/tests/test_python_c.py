# MIT License
#
# Copyright (c) 2018-2025 Tskit Developers
# Copyright (c) 2015-2018 University of Oxford
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
"""
Test cases for the low level C interface to the tskit HMM.
"""

import msprime
import numpy as np
import pytest

import _tskit_hmm
import tskit_hmm


class LowLevelTestCase:
    def get_example_tree_sequence(
        self, sample_size=10, length=1, mutation_rate=1, random_seed=1
    ):
        ts = msprime.simulate(
            sample_size,
            recombination_rate=0.1,
            mutation_rate=mutation_rate,
            random_seed=random_seed,
            length=length,
        )
        return tskit_hmm.ll_tree_sequence(ts)


class TestLsHmm(LowLevelTestCase):
    """
    Tests for the LsHmm class.
    """

    def test_uninitialised_tree_sequence(self):
        ts = _tskit_hmm.TreeSequence()
        with pytest.raises(ValueError):
            _tskit_hmm.LsHmm(ts, None, None)

    def test_constructor(self):
        ts = self.get_example_tree_sequence()
        with pytest.raises(TypeError):
            _tskit_hmm.LsHmm()
        with pytest.raises(TypeError):
            _tskit_hmm.LsHmm(None)
        values = np.zeros(ts.get_num_sites())
        for bad_array in ["asdf", [[], []], None]:
            with pytest.raises(ValueError):
                _tskit_hmm.LsHmm(ts, bad_array, values)
            with pytest.raises(ValueError):
                _tskit_hmm.LsHmm(ts, values, bad_array)

    def test_bad_rate_arrays(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        assert m > 0
        values = np.zeros(m)
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size)
            with pytest.raises(ValueError):
                _tskit_hmm.LsHmm(ts, bad_array, values)
            with pytest.raises(ValueError):
                _tskit_hmm.LsHmm(ts, values, bad_array)

    def test_haplotype_input(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        fm = _tskit_hmm.CompressedMatrix(ts)
        vm = _tskit_hmm.ViterbiMatrix(ts)
        norm = np.ones(m)
        ls_hmm = _tskit_hmm.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size, dtype=np.int8)
            with pytest.raises(ValueError, match="haplotype array"):
                ls_hmm.forward_matrix(bad_array, fm)
            with pytest.raises(ValueError, match="haplotype array"):
                ls_hmm.backward_matrix(bad_array, norm, fm)
            with pytest.raises(ValueError, match="haplotype array"):
                ls_hmm.viterbi_matrix(bad_array, vm)
        for bad_array in [[0.002], [[], []], None]:
            with pytest.raises(ValueError):
                ls_hmm.forward_matrix(bad_array, fm)
            with pytest.raises(ValueError):
                ls_hmm.viterbi_matrix(bad_array, vm)
            with pytest.raises(ValueError):
                ls_hmm.backward_matrix(bad_array, norm, fm)

    def test_norm_input(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        cm = _tskit_hmm.CompressedMatrix(ts)
        h = np.zeros(m, dtype=np.int32)
        ls_hmm = _tskit_hmm.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_size in [0, m - 1, m + 1, m + 2]:
            bad_array = np.zeros(bad_size)
            with pytest.raises(ValueError, match="forward_norm array"):
                ls_hmm.backward_matrix(h, bad_array, cm)

        for bad_array in [[0.002], [[], []], None]:
            with pytest.raises(ValueError):
                ls_hmm.backward_matrix(h, bad_array, cm)

    def test_output_type_errors(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        h = np.zeros(m, dtype=np.int8)
        norm = np.ones(m)
        ls_hmm = _tskit_hmm.LsHmm(ts, np.zeros(m), np.zeros(m))
        for bad_type in [ls_hmm, None, m, []]:
            with pytest.raises(TypeError):
                ls_hmm.forward_matrix(h, bad_type)
            with pytest.raises(TypeError):
                ls_hmm.viterbi_matrix(h, bad_type)
            with pytest.raises(TypeError):
                ls_hmm.backward_matrix(h, norm, bad_type)

        other_ts = self.get_example_tree_sequence()
        output = _tskit_hmm.CompressedMatrix(other_ts)
        with pytest.raises(_tskit_hmm.LibraryError):
            ls_hmm.forward_matrix(h, output)
        with pytest.raises(_tskit_hmm.LibraryError):
            ls_hmm.backward_matrix(h, norm, output)
        output = _tskit_hmm.ViterbiMatrix(other_ts)
        with pytest.raises(_tskit_hmm.LibraryError):
            ls_hmm.viterbi_matrix(h, output)

    def test_empty_forward_matrix(self):
        for mu in [0, 1]:
            ts = self.get_example_tree_sequence(mutation_rate=mu)
            m = ts.get_num_sites()
            fm = _tskit_hmm.CompressedMatrix(ts)
            assert fm.num_sites == m
            assert np.array_equal(np.zeros(m), fm.normalisation_factor)
            assert np.array_equal(np.zeros(m, dtype=np.uint32), fm.num_transitions)
            F = fm.decode()
            assert np.all(F >= 0)
            for j in range(m):
                assert fm.get_site(j) == []

    def test_empty_viterbi_matrix(self):
        for mu in [0, 1]:
            ts = self.get_example_tree_sequence(mutation_rate=mu)
            m = ts.get_num_sites()
            vm = _tskit_hmm.ViterbiMatrix(ts)
            assert vm.num_sites == m
            # TODO we should have the same semantics for 0 sites
            if m == 0:
                h = vm.traceback()
                assert len(h) == 0
            else:
                with pytest.raises(_tskit_hmm.LibraryError):
                    vm.traceback()

    def verify_compressed_matrix(self, ts, output):
        S = output.normalisation_factor
        N = output.num_transitions
        assert np.all(0 < S)
        assert np.all(S < 1)
        assert np.all(N > 0)
        F = output.decode()
        assert F.shape == (ts.get_num_sites(), ts.get_num_samples())
        assert np.all(F >= 0)
        m = ts.get_num_sites()
        for j in range(m):
            site_list = output.get_site(j)
            assert len(site_list) == N[j]
            for item in site_list:
                assert len(item) == 2
                node, value = item
                assert 0 <= node < ts.get_num_nodes()
                assert value >= 0
        for site in [m, m + 1, 2 * m]:
            with pytest.raises(ValueError):
                output.get_site(site)

    def test_forward_matrix(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        output = _tskit_hmm.CompressedMatrix(ts)
        ls_hmm = _tskit_hmm.LsHmm(ts, np.zeros(m) + 0.1, np.zeros(m) + 0.1)
        rv = ls_hmm.forward_matrix([0 for _ in range(m)], output)
        assert rv is None
        self.verify_compressed_matrix(ts, output)

    def test_backward_matrix(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        fm = _tskit_hmm.CompressedMatrix(ts)
        bm = _tskit_hmm.CompressedMatrix(ts)
        h = np.zeros(m, dtype=np.int32)
        ls_hmm = _tskit_hmm.LsHmm(ts, np.zeros(m) + 0.1, np.zeros(m) + 0.1)
        ls_hmm.forward_matrix(h, fm)
        ls_hmm.backward_matrix(h, fm.normalisation_factor, bm)
        self.verify_compressed_matrix(ts, bm)

    def test_viterbi_matrix(self):
        ts = self.get_example_tree_sequence()
        m = ts.get_num_sites()
        output = _tskit_hmm.ViterbiMatrix(ts)
        ls_hmm = _tskit_hmm.LsHmm(ts, np.zeros(m) + 0.1, np.zeros(m) + 0.1)
        rv = ls_hmm.viterbi_matrix([0 for _ in range(m)], output)
        assert rv is None
        self.verify_compressed_matrix(ts, output)
        h = output.traceback()
        assert isinstance(h, np.ndarray)
