"""
The tskit Li and Stephens HMM.
"""

import _tskit_hmm


def ll_tree_sequence(ts):
    """
    Return a low-level _tskit_hmm.TreeSequence for the specified
    tskit.TreeSequence, for use with the LsHmm, CompressedMatrix and
    ViterbiMatrix classes.
    """
    tables = ts.dump_tables()
    ll_tables = _tskit_hmm.LightweightTableCollection(tables.sequence_length)
    ll_tables.fromdict(tables.asdict())
    return _tskit_hmm.TreeSequence(ll_tables)
