"""
Analyse the matrices written by capture.py and write results/report.md with
tables and plots.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import tskit
from matplotlib import pyplot as plt

import analysis
import tsls

RESULTS = pathlib.Path(__file__).resolve().parent / "results"


def summarise_matrix(row, num_sites, parsimony_sites):
    """
    Return the (summary dict, list measures, decoded measures, value histogram)
    tuple for one captured matrix.
    """
    ts = tskit.load(RESULTS / row.trees)
    cm = tsls.load_compressed_matrix(RESULTS / row.matrix, ts)
    lists = analysis.list_measures(cm)
    decoded = analysis.decoded_measures(
        cm, num_sites=num_sites, parsimony_sites=parsimony_sites
    )
    nonempty = lists[lists.stored > 0]
    summary = {
        "arg": row.arg,
        "label": row.label,
        "query": row.query,
        "algorithm": row.algorithm,
        "num_nodes": row.num_nodes,
        "num_samples": row.num_samples,
        "num_segments": row.num_segments,
        "num_mismatches": row.num_mismatches,
        "stored": nonempty.stored.mean(),
        "change_points": decoded.change_points.mean(),
        "redundant": decoded.redundant.mean(),
        "distinct": nonempty.distinct.mean(),
        "median_clade_size": decoded.median_clade_size.mean(),
        "mean_clade_size": decoded.mean_clade_size.mean(),
        "samples_only": decoded.samples_only.mean(),
        "jaccard": nonempty.jaccard.mean(),
        "delta_per_site": (nonempty.added + nonempty.removed).mean(),
        "mutation_nodes": nonempty.mutation_nodes.mean(),
    }
    for digits in [1, 2, 4]:
        summary[f"quantised_{digits}"] = decoded[f"quantised_{digits}"].mean()
    if row.algorithm == "viterbi":
        summary["recombination_flags"] = nonempty.recombination_flags.mean()
        summary["new_list_fraction"] = nonempty.new_list.mean()
        # Entries stored in the traceback by the engine (lists are reused only
        # if identical) against a delta encoding of the changes
        summary["traceback_entries"] = nonempty.stored[nonempty.new_list].sum()
        summary["delta_entries"] = (nonempty.added + nonempty.removed).sum() + (
            nonempty.stored.iloc[0]
        )
    return summary, lists, decoded, analysis.value_histogram(cm, top=6)


def plot_along_genome(lists, decoded, title, path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(lists.site, lists.stored, label="stored", linewidth=0.8)
    ax.plot(
        decoded.site,
        decoded.change_points,
        "o",
        markersize=3,
        label="minimum (change points)",
    )
    if "quantised_1" in decoded:
        ax.plot(
            decoded.site,
            decoded.quantised_1,
            "x",
            markersize=3,
            label="change points, log10 values rounded",
        )
    ax.set_yscale("symlog")
    ax.set_xlabel("Site")
    ax.set_ylabel("Nodes")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_encodings(summary, path):
    viterbi = summary[summary.algorithm == "viterbi"]
    labels = [f"{r.arg}\n{r.label}" for r in viterbi.itertuples()]
    x = np.arange(len(viterbi))
    fig, ax = plt.subplots(figsize=(max(8, len(viterbi) * 0.8), 4))
    width = 0.35
    ax.bar(x - width / 2, viterbi.traceback_entries, width, label="engine traceback")
    ax.bar(x + width / 2, viterbi.delta_entries, width, label="delta encoding")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=90, fontsize=7)
    ax.set_ylabel("Stored (node, value) entries over all sites")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--num-sites", type=int, default=200, help="Sites to decode per matrix"
    )
    parser.add_argument(
        "--parsimony-sites",
        type=int,
        default=10,
        help="Sites for the samples-only parsimony",
    )
    args = parser.parse_args()
    manifest = pd.read_csv(RESULTS / "manifest.csv")
    rows = []
    plots = RESULTS / "plots"
    plots.mkdir(exist_ok=True)
    histograms = []
    for row in manifest.itertuples():
        summary, lists, decoded, histogram = summarise_matrix(
            row, args.num_sites, args.parsimony_sites
        )
        rows.append(summary)
        name = f"{row.arg}_{row.query}_{row.algorithm}"
        lists.to_csv(RESULTS / f"{name}_lists.csv", index=False)
        decoded.to_csv(RESULTS / f"{name}_decoded.csv", index=False)
        plot_along_genome(
            lists,
            decoded,
            f"{row.arg} {row.label} {row.algorithm}",
            plots / f"{name}.png",
        )
        if row.algorithm == "viterbi":
            histogram.insert(0, "case", f"{row.arg} {row.label}")
            histograms.append(histogram)
        print(f"Analysed {name}", flush=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    plot_encodings(summary, plots / "encodings.png")

    viterbi = summary[summary.algorithm == "viterbi"].copy()
    viterbi["delta_saving"] = viterbi.traceback_entries / viterbi.delta_entries
    forward = summary[summary.algorithm == "forward"]
    lines = [
        "# Likelihood compression report",
        "",
        "Per-site means. `stored` is the engine's list; `change_points` is the",
        "minimum for the nearest-ancestor encoding; `samples_only` is the",
        "(approximate) minimum if only samples had to be represented.",
        "",
        "## Viterbi",
        "",
        viterbi[
            [
                "arg",
                "label",
                "num_segments",
                "num_mismatches",
                "stored",
                "change_points",
                "redundant",
                "distinct",
                "samples_only",
                "median_clade_size",
                "jaccard",
                "delta_per_site",
                "recombination_flags",
                "new_list_fraction",
            ]
        ].to_markdown(index=False, floatfmt=".3g"),
        "",
        "## Viterbi traceback memory: engine against a delta encoding",
        "",
        "Entries stored over all sites by the engine (which reuses a list only",
        "if it is identical to the previous site's), against a delta encoding",
        "that stores the first list and then only the entries added and removed.",
        "",
        viterbi[
            ["arg", "label", "traceback_entries", "delta_entries", "delta_saving"]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Most common Viterbi values (fraction of stored entries)",
        "",
        pd.concat(histograms).to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Forward",
        "",
        forward[
            [
                "arg",
                "label",
                "stored",
                "change_points",
                "distinct",
                "samples_only",
                "quantised_1",
                "quantised_2",
                "quantised_4",
                "jaccard",
                "delta_per_site",
            ]
        ].to_markdown(index=False, floatfmt=".3g"),
        "",
        "`quantised_d`: change points after rounding log10 values to 1/d.",
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
