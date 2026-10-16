"""
Capture the exact per-site Viterbi and forward values (as tsls
CompressedMatrix files) for queries on sc2ts sub-ARGs and simulated ARGs, so
that compression can be analysed and prototyped offline.

Writes results/<name>.trees (the tree sequence the matrices refer to),
results/<name>_<query>_<algorithm>.npz and results/manifest.csv.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import tskit

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "thresholding"))
import thresholding as th
import tsls

v = th.v
RESULTS = pathlib.Path(__file__).resolve().parent / "results"
SC2TS_DATA = ROOT / "benchmarking" / "data"


def sc2ts_queries():
    """
    Return the list of (query_id, label) for the queries we capture: two each
    of copies with k = 0, 2 and 10 mutations, and one- and two-breakpoint
    recombinants with k = 0 and 5.
    """
    df = pd.read_csv(SC2TS_DATA / "queries_scaling" / "queries.csv")
    num_breakpoints = df.breakpoints.fillna("").astype(str).str.split().str.len()
    chosen = []
    for kind, k, nb in [
        ("copy", 0, 0),
        ("copy", 2, 0),
        ("copy", 10, 0),
        ("recombinant", 0, 1),
        ("recombinant", 5, 1),
        ("recombinant", 0, 2),
    ]:
        ids = df[(df.kind == kind) & (df.k == k) & (num_breakpoints == nb)].query_id
        for query_id in ids[:2]:
            label = f"{kind}{nb if kind == 'recombinant' else ''}_k{k}"
            chosen.append((int(query_id), label))
    return chosen


def capture(name, ts, queries, mu, rho, weight_by_n, rows):
    ts_path = RESULTS / f"{name}.trees"
    ts.dump(ts_path)
    mi = tsls.MatcherIndexes(ts, alleles=v.ALLELES)
    for query_id, label, h in queries:
        am = v.make_matcher(
            ts, mu, rho, weight_by_n=weight_by_n, count_distinct=True, matcher_indexes=mi
        )
        (_, _, parent), matched, vm = am.viterbi_matrix(h)
        fm = am.forward_matrix(h)
        for algorithm, cm in [("viterbi", vm), ("forward", fm)]:
            path = RESULTS / f"{name}_{query_id}_{algorithm}.npz"
            cm.dump(path)
            rows.append(
                {
                    "arg": name,
                    "trees": ts_path.name,
                    "query": query_id,
                    "label": label,
                    "algorithm": algorithm,
                    "matrix": path.name,
                    "num_nodes": ts.num_nodes,
                    "num_sites": ts.num_sites,
                    "num_samples": ts.num_samples,
                    "mu": mu,
                    "rho": rho,
                    "weight_by_n": weight_by_n,
                    "num_segments": len(parent),
                    "num_mismatches": int(np.sum((matched != h) & (h != -1))),
                    "mean_traceback_size": am.mean_traceback_size,
                    "mean_num_distinct_values": am.mean_num_distinct_values,
                }
            )
        np.save(RESULTS / f"{name}_{query_id}_query.npy", h)
        print(f"Captured {name} query {query_id} ({label})", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sizes", default="10000,30000", help="sc2ts sub-ARG sizes (comma separated)"
    )
    parser.add_argument("--quick", action="store_true", help="One query of each kind")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    mu, rho = v.hmm_params(3)

    # Simulated ARGs: held-out samples, as in the verification suite
    for num_samples in [50, 1000]:
        for seed in [1] if args.quick else [1, 2]:
            ts, h, _, _ = th.threading_case(num_samples, seed)
            capture(
                f"msprime_n{num_samples}_s{seed}",
                ts,
                [(0, "held_out", h)],
                mu,
                rho,
                False,
                rows,
            )

    # sc2ts sub-ARGs
    if (SC2TS_DATA / "subargs").exists():
        H = np.load(SC2TS_DATA / "queries_scaling" / "queries.npz")["H"]
        queries = sc2ts_queries()
        if args.quick:
            seen = set()
            queries = [q for q in queries if not (q[1] in seen or seen.add(q[1]))]
        for size in [int(x) for x in args.sizes.split(",")]:
            ts = tskit.load(SC2TS_DATA / "subargs" / f"sc2ts_bench_n{size}.trees")
            tables = ts.dump_tables()
            tables.edges.keep_rows(tables.edges.parent != 0)
            ts = tables.tree_sequence()
            capture(
                f"sc2ts_n{size}",
                ts,
                [(query_id, label, H[query_id]) for query_id, label in queries],
                mu,
                rho,
                False,
                rows,
            )
    else:
        print(f"No sc2ts data in {SC2TS_DATA}: skipping")
    pd.DataFrame(rows).to_csv(RESULTS / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
