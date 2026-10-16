"""
The Li and Stephens HMM on a tree sequence.
"""

import numpy as np

import _tsls


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


class MatcherIndexes(_tsls.MatcherIndexes):
    """
    The information about a tree sequence needed for matching, which can be
    shared between matchers (including in different threads).

    The states of the HMM at a given position are the nodes in the tree at
    that position: nodes with a parent or at least one child. Isolated
    nodes cannot be copied. Coordinates must be integers.

    By default, the ancestral state of every site is 0 and every mutation has
    derived state 1. If ``alleles`` is specified, states are instead encoded
    as their index in this list, and ``num_alleles`` defaults to its length.
    """

    def __init__(self, ts, *, alleles=None, num_alleles=None):
        ancestral_state = None
        derived_state = None
        if alleles is not None:
            ancestral_state, derived_state = encode_alleles(ts, alleles)
            if num_alleles is None:
                num_alleles = np.full(ts.num_sites, len(alleles), dtype=np.uint64)
        tables = ts.dump_tables()
        ll_tables = _tsls.LightweightTableCollection(tables.sequence_length)
        ll_tables.fromdict(tables.asdict())
        self.ts = ts
        self.sequence_length = ts.sequence_length
        self.num_sites = ts.num_sites
        super().__init__(
            ll_tables,
            num_alleles=num_alleles,
            ancestral_state=ancestral_state,
            derived_state=derived_state,
        )


class CompressedMatrix:
    """
    The output of the forward or backward algorithm. At each site we store a
    list of (node, value) pairs, and the value for a node in the tree is the
    value of its nearest ancestor (including itself) in this list.
    """

    def __init__(self, ts, normalisation_factor, offsets, nodes, values):
        self.ts = ts
        self.normalisation_factor = normalisation_factor
        self.offsets = offsets
        self.nodes = nodes
        self.values = values

    @property
    def num_sites(self):
        return len(self.normalisation_factor)

    @property
    def log_likelihood(self):
        """
        The log likelihood of the haplotype (for the forward matrix).
        """
        return float(np.sum(np.log(self.normalisation_factor)))

    def get_site(self, site):
        """
        Return the (nodes, values) arrays stored for the specified site.
        """
        start = self.offsets[site]
        stop = self.offsets[site + 1]
        return self.nodes[start:stop], self.values[start:stop]

    def decode(self, nodes=None):
        """
        Return the dense (num_sites, len(nodes)) array of values for the
        specified nodes (default: all nodes). Nodes that are not in the tree
        at a site have value 0.
        """
        N = self.ts.num_nodes
        if nodes is None:
            nodes = np.arange(N)
        nodes = np.asarray(nodes, dtype=np.int32)
        D = np.zeros((self.num_sites, len(nodes)))
        for tree in self.ts.trees():
            # The parent array has an extra entry for the virtual root, which
            # is the parent of the roots (index -1) and never has a value.
            parent = tree.parent_array
            in_tree = (parent[:N] != -1) | (tree.num_children_array[:N] > 0)
            for site in tree.sites():
                site_nodes, site_values = self.get_site(site.id)
                value = np.full(N + 1, np.nan)
                value[site_nodes] = site_values
                # Each node takes the value of its nearest ancestor with a
                # value: at each step, copy values from the ancestor and move
                # up a level, so this takes as many steps as the tree is deep.
                ancestor = parent.copy()
                unset = np.where(in_tree & np.isnan(value[:N]))[0]
                while len(unset) > 0:
                    if np.any(ancestor[unset] == -1):
                        raise ValueError("Node in tree with no stored ancestor")
                    value[unset] = value[ancestor[unset]]
                    ancestor[unset] = parent[ancestor[unset]]
                    unset = unset[np.isnan(value[unset])]
                D[site.id] = np.where(in_tree[nodes], value[nodes], 0)
        return D


class AncestorMatcher(_tsls.AncestorMatcher):
    """
    Runs the Viterbi, forward and backward algorithms for haplotypes against
    the tree sequence in the specified MatcherIndexes, using the specified
    per-site recombination and mismatch probabilities.
    """

    def __init__(
        self,
        matcher_indexes,
        recombination,
        mismatch,
        *,
        likelihood_threshold=None,
        weight_by_n=None,
        extended_checks=False,
    ):
        kwargs = {"extended_checks": extended_checks}
        if likelihood_threshold is not None:
            kwargs["likelihood_threshold"] = likelihood_threshold
        if weight_by_n is not None:
            kwargs["weight_by_n"] = weight_by_n
        self.ts = matcher_indexes.ts
        super().__init__(
            matcher_indexes,
            recombination=recombination,
            mismatch=mismatch,
            **kwargs,
        )

    def find_path(self, h, start, end, match_out):
        """
        Run the Viterbi algorithm on the haplotype h over sites [start, end),
        writing the matched haplotype into match_out, and return the
        (left, right, parent) arrays of the copying path, in right-to-left
        order.
        """
        path_len, left, right, parent, matched = super().find_path(h, start, end)
        match_out[:] = matched
        return left[:path_len], right[:path_len], parent[:path_len]

    def forward_matrix(self, h):
        """
        Run the forward algorithm on the haplotype h and return the
        CompressedMatrix of normalised forward probabilities.
        """
        return CompressedMatrix(self.ts, *super().forward_matrix(h))

    def backward_matrix(self, h, normalisation_factor):
        """
        Run the backward algorithm on the haplotype h, scaled by the
        normalisation factors of the forward matrix, and return the
        CompressedMatrix of backward probabilities.
        """
        return CompressedMatrix(
            self.ts, *super().backward_matrix(h, normalisation_factor)
        )
