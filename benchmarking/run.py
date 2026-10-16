"""
Run the HMM engines over a set of queries and record the results.

Writes one CSV row per (query, engine). Rows are flushed as they are
written, and an existing output file is resumed by skipping the
(query, engine) pairs it already contains.
"""

import csv
import pathlib
import time

import click
import numpy as np
import pandas as pd
import tskit
import tszip
from tqdm import tqdm

import engines
import queries
import subargs

RESULT_COLUMNS = [
    "query_id",
    "engine",
    "num_mismatches_param",
    "mismatch_threshold",
    "mu",
    "rho",
    "status",
    "setup_time",
    "wall_time",
    "cpu_time",
    "num_segments",
    "parents",
    "breakpoints",
    "num_mismatches",
    "log_likelihood",
    "stats",
]


def load_ts(path):
    if str(path).endswith(".tsz"):
        return tszip.load(path)
    return tskit.load(path)


def completed_pairs(out_path):
    """
    Return the set of (query_id, engine) pairs in an existing results file.
    """
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(out_path, usecols=["query_id", "engine"])
    return set(zip(df.query_id, df.engine))


def summarise_match(ts, node_ids, h, result, mu, rho):
    """
    Return the result row fields describing the specified match. The log
    likelihood is computed under the unweighted model with the specified
    parameters, so that it is comparable between engines. Parents are
    reported as original node IDs, so that they are comparable between
    sub-ARGs.
    """
    left, right, parent, matched, stats = result
    if matched is None:
        matched = engines.path_haplotype(ts, left, right, parent)
    non_missing = h != queries.MISSING
    log_likelihood = engines.path_log_likelihood(h, len(parent), matched, mu, rho)
    return {
        "num_segments": len(parent),
        "parents": queries.format_list(node_ids[parent]),
        "breakpoints": queries.format_list(left[1:]),
        "num_mismatches": int(np.sum(h[non_missing] != matched[non_missing])),
        "log_likelihood": log_likelihood,
        "stats": " ".join(f"{key}={value}" for key, value in stats.items()),
    }


@click.command()
@click.argument("ts_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("query_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "-e",
    "--engine",
    "engine_names",
    type=click.Choice(list(engines.ENGINES)),
    multiple=True,
    default=list(engines.ENGINES),
    show_default=True,
)
@click.option("-o", "--output", type=click.Path(dir_okay=False), required=True)
@click.option(
    "--num-mismatches",
    type=int,
    default=3,
    show_default=True,
    help="Number of mismatches equivalent to one recombination",
)
@click.option(
    "--mismatch-threshold",
    type=int,
    default=None,
    help="Likelihood threshold for tsinfer, in numbers of mismatches (as sc2ts)",
)
@click.option("--kind", type=click.Choice(["copy", "recombinant"]), default=None)
@click.option("--limit", type=int, default=None, help="Run at most this many queries")
def main(
    ts_path,
    query_dir,
    engine_names,
    output,
    num_mismatches,
    mismatch_threshold,
    kind,
    limit,
):
    """
    Run the HMM engines on the queries in QUERY_DIR against the benchmark ARG
    at TS_PATH.
    """
    ts = load_ts(ts_path)
    node_ids = subargs.load_node_ids(ts_path, ts)
    H, positions, df = queries.load_queries(query_dir)
    if not np.array_equal(positions, ts.sites_position):
        raise ValueError("Query sites do not match the tree sequence")
    if kind is not None:
        df = df[df.kind == kind]
    if limit is not None:
        df = df.head(limit)

    out_path = pathlib.Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_pairs(out_path)
    write_header = len(done) == 0

    mu, rho = engines.solve_num_mismatches(num_mismatches)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if write_header:
            writer.writeheader()
        for name in engine_names:
            todo = [qid for qid in df.query_id if (qid, name) not in done]
            if len(todo) == 0:
                continue
            before = time.perf_counter()
            options = {}
            if name == "tsinfer":
                options["mismatch_threshold"] = mismatch_threshold
            engine = engines.ENGINES[name](ts, mu=mu, rho=rho, **options)
            setup_time = time.perf_counter() - before
            click.echo(f"Set up {name} in {setup_time:.2f}s")
            for query_id in tqdm(todo, desc=name):
                h = H[query_id]
                row = {
                    "query_id": query_id,
                    "engine": name,
                    "num_mismatches_param": num_mismatches,
                    "mismatch_threshold": options.get("mismatch_threshold"),
                    "mu": engine.mu,
                    "rho": engine.rho,
                    "setup_time": setup_time,
                }
                wall_before = time.perf_counter()
                cpu_before = time.thread_time()
                try:
                    result = engine.match(h)
                except Exception as e:  # noqa: BLE001 (record any failure)
                    row["status"] = f"{type(e).__name__}: {e}"
                else:
                    row["status"] = "ok"
                row["wall_time"] = time.perf_counter() - wall_before
                row["cpu_time"] = time.thread_time() - cpu_before
                if row["status"] == "ok":
                    row.update(summarise_match(ts, node_ids, h, result, mu, rho))
                writer.writerow(row)
                f.flush()


if __name__ == "__main__":
    main()
