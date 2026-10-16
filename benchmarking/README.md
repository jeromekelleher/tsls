# HMM benchmarks

Benchmarks for the Li and Stephens HMM engines in tsls:

- `tsinfer`: tsinfer-hmm's `AncestorMatcher`
- `tskit`: tskit-hmm's `LsHmm` Viterbi

Each engine runs Viterbi matching over a set of query haplotypes with
known ground truth, and the results are recorded so the engines can be
characterised across a range of queries. The first target is the sc2ts
SARS-CoV-2 ARG. Benchmarks for ARGs inferred by tsinfer will follow.

## Setup

```
uv sync --group test
uv run pytest
```

Input data goes in `data/`, which is not tracked by git. The sc2ts ARG
is `data/sc2ts_viridian_v2.2.trees.tsz`.

## Workflow

### 1. Prepare the ARG

```
uv run python prepare.py data/sc2ts_viridian_v2.2.trees.tsz data/sc2ts_bench.trees
```

This step makes the ARG usable by both engines:

- It deletes the 75 sites with non-ACGT states (deletions coded as `-`),
  because tskit-hmm only supports ACGT.
- It adds the vestigial root edge 0 → 1, as sc2ts does before matching.
  tsinfer-hmm then uses the ARG as is, and node IDs are the same for
  both engines.

### 1b. Make sub-ARGs (optional)

```
uv run python subargs.py data/sc2ts_bench.trees data/subargs
```

This writes a series of nested sub-ARGs for measuring how the engines
scale with ARG size. The default sizes are 10k, 30k, 100k, 300k and 1M
samples. Each sub-ARG is the ARG "as of" an earlier date: all nodes at
least as old as a time cutoff.

- Parts of the ARG that are no longer ancestral to any remaining sample
  are pruned. These are ancestors that sc2ts only added later, along
  with their descendants. tskit-hmm also requires every node in a tree
  to have a sample below it.
- All sites are kept, so any query set can be run against any sub-ARG.
- Node IDs change, so each sub-ARG has a `.node_ids.npy` file giving
  each node's ID in the full ARG. `queries.py` and `run.py` report node
  IDs (`nodes`, `parents`) as full-ARG IDs, so they are comparable
  across sub-ARGs.

To measure scaling with a fixed set of queries, generate the queries
from the smallest sub-ARG. Its samples are in every larger sub-ARG, so
every query is valid for all of them:

```
uv run python queries.py data/subargs/sc2ts_bench_n10000.trees data/queries_scaling \
    --num-nodes 10 --divergence 0,1,2,5,10 --num-recombinants 10
for n in 10000 30000 100000 300000 1000000; do
    uv run python run.py data/subargs/sc2ts_bench_n$n.trees data/queries_scaling \
        -o results/scaling_n$n.csv
done
```

### 2. Generate queries

```
uv run python queries.py data/sc2ts_bench.trees data/queries
```

Queries are written to `queries.npz`, which holds the haplotype matrix
`H` (int8, A=0 C=1 G=2 T=3, -1 missing) and the site positions.
`queries.csv` holds one row of metadata per query. There are two kinds
of query:

- `copy`: the haplotype of a random sample node with `k` random
  mutations added. By default there are 50 nodes and k = 0..10. The
  source node stays in the ARG, so a k=0 query has an exact match.
- `recombinant`: a mosaic of two random sample nodes, A|B (one
  breakpoint) or A|B|A (two breakpoints), with k = 0, 2 or 5 mutations
  added. Each segment has at least 2 sites at which the parents differ.

Metadata columns: `nodes` (the source or parents), `k`, `mutated_sites`,
`breakpoints` (site indexes), `breakpoint_intervals`, `parent_diff` and
`segment_diffs`. A breakpoint `b` in the interval `(lo, hi]` gives the
same haplotype as any other breakpoint in that interval. Lists are
space-separated.

### 3. Run

```
uv run python run.py data/sc2ts_bench.trees data/queries -o results/sc2ts.csv
```

Options:

- `-e/--engine` (repeatable) runs a subset of the engines.
- `--kind` and `--limit` select a subset of the queries.
- `--num-mismatches` sets the model parameters.
- `--mismatch-threshold` sets tsinfer's likelihood threshold, as sc2ts does.

Rows are written and flushed one at a time. Rerunning the same command
resumes, skipping any (query, engine) pairs already in the output, so
use a separate output file for each parameter set.

Each row of the results has these columns:

- `status`: `ok` or the error message
- `setup_time`, `wall_time`, `cpu_time`: setup and per-query times in seconds
- `num_segments`, `parents`, `breakpoints`: the path (breakpoints are
  genome coordinates)
- `num_mismatches`
- `log_likelihood`: the path log likelihood under the shared model (see below)
- `stats`: engine-specific statistics
  - tsinfer: `total_memory`, `mean_traceback_size`
  - tskit: `total_transitions`, `max_transitions`, `optimum_log_likelihood`

## Model parameters

Both engines use the same Viterbi recursion. At each site, staying on a
node has likelihood `p_last * (1 - rho + rho / n)` and switching has
likelihood `rho / n`. The engines differ in what `n` is:

- tsinfer-hmm, run with `weight_by_n=False` as in sc2ts, uses `n = 1`.
- tskit-hmm uses `n = num_samples`.

Likelihoods are renormalised at every site, so only the ratio of
switching to staying matters. A tsinfer `rho` of `r` is therefore
equivalent to a tskit `rho` of `r / (r + (1 - r) / n)`. With this
conversion, both engines optimise the same objective.

`mu` and `r` are chosen as sc2ts does (`solve_num_mismatches`), so that
`--num-mismatches` mismatches (default 3) cost about the same as one
recombination.

`log_likelihood` is computed in the same units for both engines:
`(segments - 1) log r + mismatches log mu + matches log(1 - 3 mu)`.
For tskit, `optimum_log_likelihood` is the optimum found by the forward
pass, derived from the normalisation factors. If it differs from
`log_likelihood`, the traceback returned a suboptimal path.

## Caveats

- tsinfer-hmm can copy from any node; tskit-hmm can copy only from
  samples. Their paths are not directly comparable, but `parents` uses
  the same node IDs for both.
- tskit-hmm gives suboptimal results when samples are internal nodes,
  which they often are in sc2ts. Both its forward pass optimum and its
  traceback can be wrong. See the xfail test
  `test_agree_internal_samples`.
- Both packages build with `NDEBUG` undefined, so C assertions are
  active during timing.
- Matching against the full sc2ts ARG is slow; see the findings below.
  Use `--limit` and the sub-ARGs to try things out.

## Findings so far (October 2026)

These are partial, exploratory results. The runs shared a 4-core machine
with other jobs, so treat absolute times as rough. The scaling queries
are the 110 queries from the sub-ARG section (10 nodes, k = 0, 1, 2, 5,
10, and 10 recombinants of each type with k = 0, 2, 5), run with
`--num-mismatches 3`.

| engine | samples | queries run | mean time | max time | mean memory | mean traceback size |
|---|---|---|---|---|---|---|
| tsinfer | 10k | 110 | 0.56s | 1.1s | 64 MB | 908 |
| tsinfer | 30k | 110 | 2.8s | 5.1s | 204 MB | 2,813 |
| tsinfer | 100k | 52 | 7.1s | 12.6s | 560 MB | 5,106 |
| tsinfer | 3.1M (full) | 4 | 130–157s | 201s | 3.2–6.3 GB | 22k–45k |
| tskit | 10k | 26 | 46s | 69s | | |
| tskit | 3.1M (full) | 0 | none finished in 15 minutes | | | |

### tsinfer-hmm

- Time grows faster than linearly with ARG size: about 5× from 10k to
  30k samples, and 2.5× from 30k to 100k. Memory and traceback size grow
  roughly in step with time.
- Time hardly depends on the query. Divergence (k = 0–10) makes no
  difference, and recombinants take only about 10% longer than copies.
  The cost is set by the ARG, not by the query.
- On the full ARG, tsinfer 0.5.1 (as used by sc2ts) chooses the same
  parents, with the same traceback size and memory, and takes 175–180s
  per query. The cost is therefore inherent to the algorithm, not
  something introduced by the multiallelic extension.
- An sc2ts-style likelihood threshold of 5 mismatches reduced time only
  modestly (to about 125–140s per query) and did not reduce memory.
- The paths are as expected. Copy queries give one segment with exactly
  k mismatches. Recombinants average 1.0–1.5 more mismatches than k and
  mostly a single segment: with `--num-mismatches 3`, a segment with
  only 2–3 informative sites is cheaper to explain with mismatches than
  with a recombination.

### tskit-hmm

- At 10k samples it is about 80× slower than tsinfer, and on the full
  ARG it did not finish a single query.
- It gives suboptimal results on sc2ts ARGs, where about 11% of samples
  are internal nodes. At 10k samples it was worse than tsinfer on 10 of
  the 26 queries compared, always by exactly one mismatch, including
  exact copies (k = 0). It was never better. There are two separate
  failures:
  - Forward pass. When the source is an internal sample (for example
    query 0), the optimum implied by the normalisation factors is
    already one mismatch worse than an exact copy that is available.
  - Traceback. When the source is a leaf (for example query 20), the
    forward pass optimum is correct, but the traceback returns a path
    one mismatch worse. This happened on 5 of the 26 queries.

  When all samples are leaves, the two engines agree exactly. The
  failures are reproduced by `test_agree_internal_samples`.
- It aborts with a C assertion (`lib/haplotype_matching.c:349`) if a tree
  contains a node with no samples below it. `subargs.py` prunes such
  ancestry.

### Implications

- tskit-hmm is not usable as a reference on sc2ts-style ARGs until the
  internal-sample bugs are fixed.
- For tsinfer-hmm, the main targets are memory and traceback size on
  large ARGs.
- For clean timings, run sub-ARGs up to about 300k samples one at a time.
  Restrict tskit to the smallest sub-ARGs with `--limit`.
