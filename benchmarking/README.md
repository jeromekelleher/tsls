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
- Matching against the full sc2ts ARG is slow. tsinfer takes about
  2–3 minutes and 3–7 GB per query, which matches tsinfer 0.5.1 as used
  by sc2ts. tskit is slower still. Use `--limit` to try things out.
