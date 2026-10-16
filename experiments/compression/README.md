# Likelihood compression

This directory captures the exact per-site Viterbi and forward likelihoods
computed by tsls and analyses how they are compressed. The aim is to have
fixed, real data for developing better representations and algorithms,
without changing the engine.

```
uv sync
uv run python capture.py --sizes 10000,30000,100000   # about 35 minutes
uv run python report.py                               # results/report.md
```

`capture.py` uses `AncestorMatcher.viterbi_matrix` and `forward_matrix`. It
saves each result as a `CompressedMatrix` with `CompressedMatrix.dump`, next
to the tree sequence it refers to. Reload a matrix with
`tsls.load_compressed_matrix(path, ts)`.

The Viterbi matrix records, at every site, the list of (node, value) pairs
stored in the traceback, with each node's `recombination_required` flag.
The values are normalised by the maximum, and the maximum is stored as the
site's normalisation factor. Results are written to `results/`, which is not
tracked by git. The sc2ts part needs the sub-ARGs and queries in
`benchmarking/data`.

The cases captured are:
- **sc2ts sub-ARGs** with 10k, 30k and 100k samples (11k, 33k and 111k
  nodes), using the benchmark settings (k = 3, `weight_by_n=False`). The
  queries are two each of copies with k = 0, 2 and 10 extra mutations, and
  recombinants with one or two breakpoints.
- **msprime ARGs**: held-out samples matched against 50 or 1000 other
  samples, with the same HMM settings.

`analysis.py` computes the measures:
- From the stored lists, at every site: the list size, distinct values,
  overlap with the previous site's list, the entries added and removed,
  recombination flags, and whether the engine stores a new traceback list.
- By decoding every node's value, at 200 sampled sites:
  - the minimum number of stored nodes ("change points");
  - redundant stored nodes;
  - clade sizes;
  - change points after rounding the values;
  - an approximate minimum if only samples had to be represented, as in
    tskit (10 sites).

## The representation

At each site the engine stores a list of (node, value) pairs, and every
node in the tree takes the value of its nearest ancestor in the list.

In tsls every node in the tree is a state with its own value. So the
smallest list that represents the values exactly is the set of **change
points**: nodes whose value differs from their parent's, plus the roots.

Merging siblings with equal values isn't possible, because their parent has
its own, different value. This corrects an earlier suggestion in
`experiments/thresholding`.

## Findings

Ranges are over the captured queries. "Change points" and samples-only
figures come from 200 and 10 sampled sites respectively.

| | sc2ts 10k | sc2ts 30k | sc2ts 100k | msprime 50 / 1000 |
|---|---|---|---|---|
| Viterbi stored / site | 312–1,242 | 515–4,088 | 666–7,598 | 9–19 |
| Viterbi change points / site | 314–1,279 | 519–4,181 | 676–7,582 | 6–16 |
| Viterbi distinct values / site | 4.1–6.3 | 4.3–6.8 | 4.5–7.1 | 3–5 |
| Viterbi samples-only / site | 299–1,302 | 458–4,395 | 465–7,285 | 4–11 |
| Viterbi median change-point clade size (samples) | 1.0 | 1.0 | 1.0–1.1 | 3–105 |
| Viterbi list overlap with previous site (Jaccard) | ≥ 0.999 | ≥ 0.997 | ≥ 0.991 | 0.45–0.64 |
| Viterbi entries added + removed / site | 1.0–1.3 | 3.3–4.5 | 10–13 | 8–12 |
| Sites where the engine stores a new list | 21–24% | 44–49% | 74–77% | 98–100% |
| Traceback entries: engine ÷ delta encoding | 74–230× | 77–429× | 50–428× | 1.1–1.6× |
| Forward stored / site | 255–257 | 320–321 | 355–356 | 35–163 |
| Forward distinct values / site | 180–182 | 148 | 91 | 30–122 |
| Forward change points, log10 values rounded to integers | 4.2 | 2.4–2.6 | 5.7 | 10–24 |

### 1. The stored lists are already minimal

For both Viterbi and forward, the stored list equals the minimum number of
change points, give or take sampling noise, with almost no redundant
entries. On sc2ts there are under 6 redundant entries per site. On the
small simulated ARGs there are 2–5: these are mutation nodes inserted at
the site that end up equal to their parent.

Representing only the samples, as tskit does, would save at most 5–30% on
sc2ts. So the all-nodes model is not what makes the lists large.

### 2. Why Viterbi stores so many nodes with so few values on sc2ts

The Viterbi values are almost entirely four exact classes:
- 1: matches the query since the last switch;
- μ/(1 − 3μ): one mismatch;
- (μ/(1 − 3μ))²: two mismatches;
- ρ: recombined.

With `weight_by_n=False`, the probability of staying on a node is 1, and
normalisation divides out the match probability. So a node's value is
constant while it matches the query. A third mismatch takes it below ρ,
where it is replaced by the recombination value.

The stored nodes are therefore the boundary of the query's "at most two
mismatches" neighbourhood in the ARG. On sc2ts, nearly all of them are
single samples (median clade size 1): samples with private mutations under
the very large polytomies of the sc2ts trees.

The list size depends on where the query sits in the ARG, not on how
divergent it is. Copies of the same source with k = 0, 2 or 10 extra
mutations give the same counts. Different sources differ by up to 11×.

### 3. The Viterbi traceback is dominated by repeats

The lists hardly change along the sequence. On sc2ts, about 1–13 entries
are added or removed per site, against hundreds or thousands stored. But
the engine reuses the previous list only if it is identical, so it stores
a full copy at 21–77% of sites.

A **delta encoding** stores the first list, then only the entries added and
removed. It would reduce traceback memory by about 50–430× on sc2ts. On the
full ARG, traceback memory reached 3–7 GB per query. On simulated ARGs the
lists change almost completely at every site, so the gain there is small
(1.1–1.6×).

### 4. Forward has many distinct values but very little structure

The forward lists are also minimal (about 255–355 values per site on sc2ts),
but every value changes at every site. That's because the recombination
term ρ/n·S is added to every node, so a delta encoding of values doesn't
help.

The values cluster tightly, though. Rounding log10 values to integers
leaves only about 2–6 change points per site on sc2ts, which is 50–140×
fewer. Approximate compression, such as quantising values or a floor, is
very effective for forward/backward. That fits the thresholding results,
where `floor = 1e-4` cut forward storage by about 200×.

## Implications

- **Viterbi memory:** delta-encode the traceback. Store each node's value
  and flag only when it changes, and walk the changes backwards in the
  traceback. This is exact.
- **Viterbi time:** delta encoding alone doesn't reduce the number of
  tracked nodes, which is already minimal, or the per-site loop over them.
  But only about 1–13 values change per site on sc2ts, because a uniform
  match factor cancels in normalisation. An engine that keeps log-values
  with a global offset could update only the nodes under the site's
  mutations, plus those crossing the recombination level. The per-site work
  would then scale with the number of changes rather than with the list
  size. This needs prototyping and profiling.
- **Forward/backward time:** apart from nodes under disagreeing mutations,
  each site's update is the same scale-and-shift for every node. A global
  transform plus explicit updates for those nodes would give forward the
  same structure. Separately, quantising values is a large and effective
  approximation.
- The captured matrices (`results/*.npz`) can be used to size and check
  any new encoding offline before it is implemented in C.
