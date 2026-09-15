"""Cross-dataset figure: how memorisation shifts the receiver operating point.

Reads the per-disease sensitivity/specificity CSVs that memorisation_analysis.py writes
under figs/<dataset>/files/ (its --rop_comparison output) and draws them as one forest
plot across datasets: for each disease, the change in sensitivity and specificity on
future records between models that saw the patient's earlier data and models that did
not, with confidence intervals and significance markers.

Multiplicity is corrected *here*, not in the analysis: the family is this figure, so it
spans every dataset drawn and is only known at this point. The CSVs carry raw p-values,
which are Bonferroni-corrected over m = 2 x the number of classes shown (each contributes a
sensitivity and a specificity comparison). The intervals come from inverting each comparison's
own run-label permutation test at the matching per-comparison level alpha/m, using the null
tails in the companion *_rop_bootstrap*.npz files, so a bar clear of zero *is* a corrected
p below alpha.

Requires memorisation_analysis.py to have been run for every dataset in LOGDIRS below.

Usage:
    python rop_comparison_plot.py              # -> figs/rop_comparison.pdf
    python rop_comparison_plot.py --denovo     # de novo disease cases only
    python rop_comparison_plot.py --heedb_only # HEEDB with all of its classes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from pathlib import Path
import os, math
import logging
from src.plotting import bonferroni_figure_correction, get_significance_str
from src.utils import get_dataset_str, get_dataset_color_hex
from absl import app, flags  # type: ignore

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force jax to use CPU only
FONT_SIZE = 6
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (3.4, 2.0),
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "axes.grid": True,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": False,
        "pdf.fonttype": 42,
    }
)
# silence fontTools' INFO chatter when embedding Type 42 fonts into PDFs
logging.getLogger("fontTools").setLevel(logging.ERROR)
FLAGS = flags.FLAGS
flags.DEFINE_boolean("denovo", False, "Whether to visualise results for de-novo cases of diseases.")
flags.DEFINE_float(
    "alpha",
    0.05,
    "Family-wise error rate for the Bonferroni correction applied across this figure.",
)
flags.DEFINE_boolean(
    "heedb_only",
    False,
    "Plot HEEDB only, with all of its classes instead of the top-10 subset.",
)

# how many of HEEDB's classes the multi-dataset figure shows, ranked by de novo case count
HEEDB_TOP_N = 10

LOGDIRS = {
    "heedb": "./figs/heedb/files",
    "mimic-iv-ed_rf": "./figs/mimic-iv-ed_rf/files",
    "mimic-cxr": "./figs/mimic-cxr/files",
    "mimic-ecg": "./figs/mimic-ecg/files",
}


def denovo_case_counts(logdir: Path, dataset_name: str) -> dict:
    """Per-class count of de novo positive cases, keyed by the raw class label.

    Read from the `(denovo)` sensitivity CSV regardless of which variant of the figure is
    being drawn, so the class ordering is the same in both. The de novo and 'seen' frames
    carry the same classes, so every class of either variant is covered.
    """
    path = logdir / f"{dataset_name}_sensitivity_comparison(denovo).csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is needed to order {dataset_name}'s classes by de novo case count; "
            f"run memorisation_analysis.py for this dataset"
        )
    frame = pd.read_csv(path)
    return dict(zip(frame["Class"], frame["n_pos_cases"]))


def main(argv):
    from matplotlib.gridspec import GridSpec  # type: ignore
    fig = plt.figure(figsize=(7.2, 6.69))
    gs = GridSpec(1, 7, figure=fig, width_ratios=[6, 10, 0.5, 10, 0.5, 0.5, 0.5],
                  wspace=0.5)
    axs = [fig.add_subplot(gs[0, i]) for i in range(7)]
    axs = np.array(axs)
    class_names = []
    sensitivity_effects = []
    specificity_effects = []
    sensitivity_nulls = []
    specificity_nulls = []
    sensitivity_nperm = []
    specificity_nperm = []
    dataset_names = []
    pos_case_numbers = []
    neg_case_numbers = []
    tp_effects = []
    sensitivity_pvals = []
    specificity_pvals = []

    logdirs = {"heedb": LOGDIRS["heedb"]} if FLAGS.heedb_only else LOGDIRS
    for i, (dataset_name, logdir) in enumerate(logdirs.items()):
        ldir = Path(logdir)
        suffix = "(denovo)" if FLAGS.denovo else ""
        sensitivity_data = pd.read_csv(ldir / f"{dataset_name}_sensitivity_comparison{suffix}.csv")
        print(f"\n{dataset_name} (Sensitivity): \n {sensitivity_data}")
        specificity_data = pd.read_csv(ldir / f"{dataset_name}_specificity_comparison{suffix}.csv")
        print(f"\n{dataset_name} (Specificity): \n {specificity_data}")
        # permutation-null tails, row-aligned with the frames above
        with np.load(ldir / f"{dataset_name}_rop_bootstrap{suffix}.npz") as bundle:
            if "sensitivity_null_lower" not in bundle:
                raise KeyError(
                    f"{ldir / f'{dataset_name}_rop_bootstrap{suffix}.npz'} carries no "
                    f"permutation null tails; re-run memorisation_analysis.py for this dataset"
                )
            # (n_comparisons, n_tail) each; stacked as (lower, upper) so the row-alignment
            # bookkeeping below moves them with everything else
            sens_null = np.stack(
                [bundle["sensitivity_null_lower"], bundle["sensitivity_null_upper"]], axis=1
            )
            spec_null = np.stack(
                [bundle["specificity_null_lower"], bundle["specificity_null_upper"]], axis=1
            )
            sens_nperm = bundle["sensitivity_n_permutations"]
            spec_nperm = bundle["specificity_n_permutations"]
        # specificity can be missing a class that had no negative cases; align it to the
        # sensitivity rows so effects, p-values and null tails stay on the same class
        spec_pos = {cls: i for i, cls in enumerate(specificity_data["Class"])}
        keep = [spec_pos.get(cls) for cls in sensitivity_data["Class"]]
        if any(k is None for k in keep):
            missing = [c for c, k in zip(sensitivity_data["Class"], keep) if k is None]
            raise ValueError(
                f"{dataset_name}: no specificity comparison for {missing}; the figure pairs "
                f"the two per class, so this needs handling before plotting"
            )
        specificity_data = specificity_data.iloc[keep].reset_index(drop=True)
        spec_null, spec_nperm = spec_null[keep], spec_nperm[keep]
        if dataset_name == "heedb":
            counts = denovo_case_counts(ldir, dataset_name)
            if not FLAGS.heedb_only:
                # visualise a subset of HEEDB's classes: the ones with the most de novo cases.
                # Derived from the data rather than pinned, so it cannot drift away from the
                # counts as the analysis is re-run.
                ranked = sorted(
                    sensitivity_data["Class"], key=lambda cls: counts[cls], reverse=True
                )
                top_n_classes = ranked[:HEEDB_TOP_N]
                print(
                    f"... HEEDB top {HEEDB_TOP_N} classes by de novo case count: "
                    + ", ".join(f"{cls} ({counts[cls]:,})" for cls in top_n_classes)
                )
                mask = sensitivity_data["Class"].isin(top_n_classes).to_numpy()
                sensitivity_data = sensitivity_data[mask].reset_index(drop=True)
                specificity_data = specificity_data[mask].reset_index(drop=True)
                sens_null, spec_null = sens_null[mask], spec_null[mask]
                sens_nperm, spec_nperm = sens_nperm[mask], spec_nperm[mask]
            # order HEEDB's classes by de novo case count, commonest at the top of the figure.
            # Row 0 is drawn at the bottom, so sorting ascending puts the largest count last,
            # i.e. topmost. Both variants of the figure use the de novo counts, so the class
            # order is comparable between them.
            order = np.argsort(
                [counts[cls] for cls in sensitivity_data["Class"]], kind="stable"
            )
            sensitivity_data = sensitivity_data.iloc[order].reset_index(drop=True)
            specificity_data = specificity_data.iloc[order].reset_index(drop=True)
            sens_null, spec_null = sens_null[order], spec_null[order]
            sens_nperm, spec_nperm = sens_nperm[order], spec_nperm[order]
            # replace ICD codes with disease names for better readability
            from src.data.constants import HEEDB_CODE_DICT
            heedb_class_names = [HEEDB_CODE_DICT[int(code_str.split("_")[1])] for code_str in sensitivity_data["Class"].tolist()]
            sensitivity_data["Class"] = heedb_class_names    
        class_names.extend(sensitivity_data["Class"].tolist())
        sensitivity_effects.extend(sensitivity_data["sample_mean_diff"].tolist())
        specificity_effects.extend(specificity_data["sample_mean_diff"].tolist())
        sensitivity_nulls.append(sens_null)
        specificity_nulls.append(spec_null)
        sensitivity_nperm.append(sens_nperm)
        specificity_nperm.append(spec_nperm)
        dataset_names.extend([dataset_name] * len(sensitivity_data))
        pos_case_numbers.extend(sensitivity_data["n_pos_cases"].tolist())
        neg_case_numbers.extend(sensitivity_data["n_neg_cases"].tolist())
        sensitivity_pvals.extend(sensitivity_data["pval_raw"].tolist())
        specificity_pvals.extend(specificity_data["pval_raw"].tolist())
    # Bonferroni over this figure's family: every class shown contributes a sensitivity and a
    # specificity comparison, so m = 2 x the number of classes drawn. This is the only place the
    # family size is known -- the analysis writes raw p-values because one dataset's call cannot
    # see the others that share the figure.
    n_classes = len(class_names)
    m_comparisons = 2 * n_classes
    alpha = FLAGS.alpha
    sensitivity_nulls = np.concatenate(sensitivity_nulls, axis=0)
    specificity_nulls = np.concatenate(specificity_nulls, axis=0)
    sensitivity_nperm = np.concatenate(sensitivity_nperm, axis=0)
    specificity_nperm = np.concatenate(specificity_nperm, axis=0)
    sensitivity_raw_pvals = np.asarray(sensitivity_pvals, dtype=float)
    specificity_raw_pvals = np.asarray(specificity_pvals, dtype=float)
    sensitivity_pvals, sens_lo, sens_hi, ci_level = bonferroni_figure_correction(
        sensitivity_raw_pvals, sensitivity_effects,
        sensitivity_nulls[:, 0], sensitivity_nulls[:, 1], sensitivity_nperm,
        n_classes, alpha=alpha,
    )
    specificity_pvals, spec_lo, spec_hi, _ = bonferroni_figure_correction(
        specificity_raw_pvals, specificity_effects,
        specificity_nulls[:, 0], specificity_nulls[:, 1], specificity_nperm,
        n_classes, alpha=alpha,
    )
    print(
        f"... Bonferroni correction over the {m_comparisons} comparisons in this figure "
        f"({n_classes} classes x sensitivity/specificity, alpha={alpha}): "
        f"{int((sensitivity_pvals < alpha).sum())} sensitivity and "
        f"{int((specificity_pvals < alpha).sum())} specificity comparisons significant. "
        f"Intervals invert the permutation test at the matching {ci_level*100:.4f}% level"
    )
    for label, lo, hi, nperm in (
        ("sensitivity", sens_lo, sens_hi, sensitivity_nperm),
        ("specificity", spec_lo, spec_hi, specificity_nperm),
    ):
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError(
                f"{label}: the permutation null has too few draws to be inverted at "
                f"alpha/m = {alpha/m_comparisons:.3g} (smallest attainable p-value is "
                f"{2/(int(nperm.min())+1):.3g}); re-run the analysis with a larger "
                f"--n_permutations"
            )
    # convert to arrays and scale to percentage
    sensitivity_effects_arr = np.array(sensitivity_effects) * 100
    specificity_effects_arr = np.array(specificity_effects) * 100
    sensitivity_CIs_lower_arr = sens_lo * 100
    sensitivity_CIs_upper_arr = sens_hi * 100
    specificity_CIs_lower_arr = spec_lo * 100
    specificity_CIs_upper_arr = spec_hi * 100
    colors = [get_dataset_color_hex(ds_name) for ds_name in dataset_names]
    print(f"class_names: {len(class_names)}")
    print(f"sensitivity_effects_arr: {sensitivity_effects_arr.shape}")
    print(f"sensitivity_CIs_lower_arr: {sensitivity_CIs_lower_arr.shape}")
    print(f"sensitivity_CIs_upper_arr: {sensitivity_CIs_upper_arr.shape}")
    print(f"specificity_effects_arr: {specificity_effects_arr.shape}")
    print(f"specificity_CIs_lower_arr: {specificity_CIs_lower_arr.shape}")
    print(f"specificity_CIs_upper_arr: {specificity_CIs_upper_arr.shape}")

    # plot results
    sens_ax = axs[1]
    spec_ax = axs[3]
    sens_pval_ax = axs[2]
    spec_pval_ax = axs[4]
    pos_cases_ax = axs[5]
    neg_cases_ax = axs[6]

    sens_effects = sens_ax.scatter(
        sensitivity_effects_arr,
        class_names,
        color=colors,
        s=10,
        alpha=1.0,
        zorder=0,
    )
    spec_effects = spec_ax.scatter(
        specificity_effects_arr,
        class_names,
        color=colors,
        s=10,
        alpha=1.0,
        zorder=0,
    )
    sens_cis = sens_ax.hlines(
        y=class_names,
        xmin=sensitivity_CIs_lower_arr,
        xmax=sensitivity_CIs_upper_arr,
        color="gray",
        alpha=0.3,
        lw=1.5,
        zorder=-1,
    )
    spec_cis = spec_ax.hlines(
        y=class_names,
        xmin=specificity_CIs_lower_arr,
        xmax=specificity_CIs_upper_arr,
        color="gray",
        alpha=0.3,
        lw=1.5,
        zorder=-1,
    )
    sens_ax.spines[["left", "right", "top"]].set_visible(False)
    sens_ax.axvline(x=0, color="black", lw=1, ls="--", zorder=-2, alpha=0.1)
    spec_ax.spines[["left", "right", "top"]].set_visible(False)
    spec_ax.axvline(x=0, color="black", lw=1, ls="--", zorder=-2, alpha=0.1)
    sens_ax.set_xlabel("Sensitivity (%)")
    spec_ax.set_xlabel("Specificity (%)")
    class_name_table = axs[0].table(
        cellText=[[class_name] for class_name in reversed(class_names)], # tables starts from the top so we reverse the order
        cellLoc="left",
        colLoc="left",
        loc="center left",
        bbox=[0.0, 0, 1, 1],
        edges='open',
    )
    class_name_table.auto_set_font_size(False)
    class_name_table.set_fontsize(FONT_SIZE)
    sens_pval_table = sens_pval_ax.table(
        cellText=[
            [f"{sensitivity_pvals[i]:.1e}" if sensitivity_pvals[i] != 1.0 else sensitivity_pvals[i]]
            for i in reversed(range(len(class_names)))
        ],
        cellLoc="center",
        colLoc="left",
        loc="upper right",
        bbox=[0.0, 0, 1, 1],
        edges='open',
    )
    sens_pval_table.auto_set_font_size(False)
    sens_pval_table.set_fontsize(FONT_SIZE)
    spec_pval_table = spec_pval_ax.table(
        cellText=[
            [f"{specificity_pvals[i]:.1e}" if specificity_pvals[i] != 1.0 else specificity_pvals[i]]
            for i in reversed(range(len(class_names)))
        ],
        cellLoc="center",
        colLoc="left",
        loc="upper right",
        bbox=[0.0, 0, 1, 1],
        edges='open',
    )
    spec_pval_table.auto_set_font_size(False)
    spec_pval_table.set_fontsize(FONT_SIZE)
    pos_cases_table = pos_cases_ax.table(
        cellText=[
            [f"{pos_case_numbers[i]:,}"]
            for i in reversed(range(len(class_names)))
        ],
        cellLoc="center",
        colLoc="center",
        loc="upper right",
        bbox=[0.0, 0, 1, 1],
        edges='open',
    )
    pos_cases_table.auto_set_font_size(False)
    pos_cases_table.set_fontsize(FONT_SIZE)
    neg_cases_table = neg_cases_ax.table(
        cellText=[
            [f"{neg_case_numbers[i]:,}"]
            for i in reversed(range(len(class_names)))
        ],
        cellLoc="center",
        colLoc="center",
        loc="upper right",
        bbox=[0.0, 0, 1, 1],
        edges='open',
    )
    neg_cases_table.auto_set_font_size(False)
    neg_cases_table.set_fontsize(FONT_SIZE)

    # Add significance stars on the axes for significant results only
    def pval_to_stars(p):
        if p <= 0.001:
            return "***"
        elif p <= 0.01:
            return "**"
        elif p <= 0.05:
            return "*"
        return None

    for i in range(len(class_names)):
        sens_stars = pval_to_stars(sensitivity_pvals[i])
        spec_stars = pval_to_stars(specificity_pvals[i])
        if sens_stars:
            sens_ax.text(
                1.025, i, sens_stars,
                transform=sens_ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=FONT_SIZE,
            )
        if spec_stars:
            spec_ax.text(
                1.025, i, spec_stars,
                transform=spec_ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=FONT_SIZE,
            )

    axs[0].axis("off")
    sens_pval_ax.axis("off")
    spec_pval_ax.axis("off")
    pos_cases_ax.axis("off")
    neg_cases_ax.axis("off")
    sens_ax.set_yticks([])
    spec_ax.set_yticks([])
    for a in axs.flat:
        a.set_ylim(-0.5, len(class_names)-0.5)

    # # Create legend handles with circular markers for each dataset
    # from matplotlib.lines import Line2D # type: ignore
    # legend_handles = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=get_dataset_color_hex(ds_name),
    #         markersize=6, label=get_dataset_str(ds_name))
    #     for ds_name in LOGDIRS.keys()
    # ]            
    # sens_ax.legend(handles=legend_handles, loc="lower center", fontsize=FONT_SIZE, bbox_to_anchor=(0.5, -0.125), ncol=len(LOGDIRS), frameon=False)
    fig.subplots_adjust(left=0.021, right=0.979, bottom=0.05, top=0.95)
    stem = "rop_comparison_heedb" if FLAGS.heedb_only else "rop_comparison"
    filename = f"{stem}(denovo).pdf" if FLAGS.denovo else f"{stem}.pdf"
    fig.savefig(
        f"./figs/{filename}", dpi=300)

    print("saved pdf to", filename)

if __name__ == "__main__":
    app.run(main)