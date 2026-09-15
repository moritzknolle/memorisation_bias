"""Memorisation analysis across privacy budgets (MIMIC-ECG and HEEDB).

Runs the analysis of memorisation_analysis.py for every run set under --log_root

    <log_root>/dp/eps{1,10,100,1000}[_p]/   DP-SGD, record-level [patient-level]
    <log_root>/nonprivate[_p]/              eps=inf

and plots memorised-record counts, empirical survival functions of the effect size and the test
statistic, and test-set AUROC against epsilon.

* Results are cached per run set; --recompute forces re-analysis, --plot_only only redraws
  the figures from the cached CSVs.
* Run directories trained on the same patient subset are deduplicated.
* A run set is only tested if every record has at least MIN_GROUP_SIZE predictions in both
  the IN and the OUT group.

Reference curves for random partitioning are read from the output of memorisation_analysis.py.

Usage:
    python dp_memorisation_analysis.py --dataset=mimic-ecg --log_root=<log_root>
"""

import hashlib
import json
import logging
import os
from functools import partial
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from absl import app, flags
from tqdm import tqdm

from src.data.dataset_factory import get_dataset
from src.memorisation.core import (
    convert_patientmasks_to_recordmask,
    get_test_object,
    load_from_dir,
    load_predictions,
    test_significantly_different,
    RandomSubsetPredictionPartitioner,
)
from src.utils import (
    DP_COLOR_PATIENT,
    DP_COLOR_RECORD,
    dp_esf_styles,
    fig_dir_exists,
    get_act_fn,
    get_data_root,
    get_patient_col,
)

tqdm.pandas(leave=False)

FONT_SIZE = 7
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (2.2, 2.0),
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
        "axes.grid": False,
        "grid.alpha": 0.1,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
    }
)
# silence fontTools' INFO chatter when embedding Type 42 fonts into PDFs
logging.getLogger("fontTools").setLevel(logging.ERROR)

flags.DEFINE_string("dataset", "mimic-ecg", "Dataset to analyse: 'mimic-ecg' or 'heedb'.")
flags.DEFINE_string(
    "log_root", "", "Directory with the run sets of the dataset. Default: ./logs/<dataset>."
)
flags.DEFINE_string(
    "dp_logdir", "", "DP subdirectories (eps1, eps10, ...). Default: <log_root>/dp."
)
flags.DEFINE_string(
    "nonprivate_logdir",
    "",
    "Record-level non-private (eps=inf) runs. Default: <log_root>/nonprivate.",
)
flags.DEFINE_string(
    "nonprivate_logdir_p",
    "",
    "Patient-level non-private (eps=inf) runs. Default: <log_root>/nonprivate_p.",
)
flags.DEFINE_string(
    "nonprivate_figs_dir",
    "",
    "Output directory of memorisation_analysis.py for this dataset (random-partitioning "
    "reference CSVs). Default: ./figs/<dataset>.",
)
flags.DEFINE_string(
    "out_dir",
    "",
    "Directory for figures and result CSVs. Default: ./figs/<dataset>_dp.",
)
flags.DEFINE_string(
    "auroc_ylim",
    "",
    "y-limits of the AUROC panel as 'lo,hi'; unset uses the dataset default.",
)
flags.DEFINE_string(
    "teststat_xlim",
    "",
    "x-limits of the test-statistic eSF panels as 'lo,hi'; unset autoscales.",
)
flags.DEFINE_string("csv_root", "./data/csv/", "Directory with CSV files.")
flags.DEFINE_string("save_root", "./data/npy/", "Directory with npy/memmap files.")
flags.DEFINE_string(
    "test", "energy", "Statistical test type: 'energy' or 'hotelling'."
)
flags.DEFINE_boolean("multiprocessing", True, "Use multiprocessing for loading.")
flags.DEFINE_boolean("plot_legend", True, "Whether to plot a legend for each sub panel")
flags.DEFINE_boolean(
    "recompute", False, "Recompute all configs, ignoring cached per-config CSVs."
)
flags.DEFINE_boolean(
    "plot_only",
    False,
    "Only regenerate plots from cached CSVs, even if the runs are newer than them.",
)
flags.DEFINE_integer("threads", 8, "Number of threads for multiprocessing.")
flags.DEFINE_integer("seed", 42, "Random seed.")
FLAGS = flags.FLAGS

# Dataset-factory arguments. The record- and patient-level ("one ECG per patient") variants
# are separate datasets.
DATASET_SPECS = {
    "mimic-ecg": {
        "record_dataset": "mimic-ecg",
        "patient_dataset": "mimic-ecg_p",
        "resolution": 250,
        "auroc_ylim": (82.5, 95.0),
    },
    "heedb": {
        "record_dataset": "heedb",
        "patient_dataset": "heedb_p",
        "resolution": None,
        "auroc_ylim": None,  # autoscale
    },
}

# Run sets: (dir_name, epsilon, dp_level)
DP_CONFIGS = [
    ("eps1", 1, "record"),
    ("eps10", 10, "record"),
    ("eps100", 100, "record"),
    ("eps1000", 1000, "record"),
    ("eps1_p", 1, "patient"),
    ("eps10_p", 10, "patient"),
    ("eps100_p", 100, "patient"),
    ("eps1000_p", 1000, "patient"),
]

# Non-private baselines (eps=inf), analysed like the DP run sets
NONPRIVATE_CONFIGS = [
    ("nonprivate", np.inf, "record"),
    ("nonprivate_p", np.inf, "patient"),
]

ALL_CONFIGS = DP_CONFIGS + NONPRIVATE_CONFIGS

# The energy test needs more than 3 predictions on each side of a record's IN/OUT split
MIN_GROUP_SIZE = 4

COLOR_RECORD = DP_COLOR_RECORD
COLOR_PATIENT = DP_COLOR_PATIENT
# dashes of the eps=infinity reference lines, shared by the count and AUROC panels
BASELINE_STYLE = (0, (4, 2))
BASELINE_LW = 1.0

# DP levels in panel order (record-level left, patient-level right), with their labels
DP_LEVEL_LABELS = {"record": "Record-level DP", "patient": "Patient-level DP"}

# The two per-record quantities the test yields, each drawn as its own set of eSF
# figures: the effect size (how much the prediction moved) and the test statistic the
# p-values came from. The key is the figure-name slot, so "pdelta" keeps the effect-size
# figures at the file names they have always had.
ESF_METRICS = {
    "pdelta": {
        "column": "effect_size",
        "name": "effect size",
        "scale": 100.0,  # probability -> percent
        "xlabel": "Change in predicted probability (%)",
        "xlim": (0.0, 100.0),
    },
    "teststat": {
        "column": "test_statistic",
        "name": "test statistic",
        "scale": 1.0,
        "xlabel": None,  # named after the selected test, see _metric_xlabel
        "xlim": None,  # autoscaled unless --teststat_xlim is given
    },
}

# x-axis label of the test-statistic panels, per --test
TEST_STATISTIC_LABELS = {
    "energy": "Energy test statistic",
    "hotelling": "Hotelling test statistic",
}

# Splits in panel order (future left, historical right), with their panel titles
SPLIT_TITLES = {"future": "Future records", "historical": "Historical records"}

# Shared y-axis label of the two-panel count figure
COUNT_YLABEL = "Records with significant\nchange in prediction (count)"

# Per-config result files, keyed by split
RESULT_FILES = {
    "historical": "mem_data_historical.csv",
    "future": "mem_data_future.csv",
}
# Records how many model runs the cached CSVs were computed from
MANIFEST_FILE = "run_manifest.json"


def dataset_spec() -> dict:
    """Log locations and dataset-factory arguments for the selected dataset."""
    if FLAGS.dataset not in DATASET_SPECS:
        raise ValueError(
            f"Unknown dataset {FLAGS.dataset!r}, expected one of {list(DATASET_SPECS)}"
        )
    return DATASET_SPECS[FLAGS.dataset]


def _flag_path(flag_value: str, default) -> Path:
    """Path from a flag, falling back to the dataset's default when unset."""
    return Path(flag_value) if flag_value else Path(default)


def log_root() -> Path:
    """Directory holding all run sets of the selected dataset."""
    return _flag_path(FLAGS.log_root, f"./logs/{FLAGS.dataset}")


def dp_log_root() -> Path:
    """Directory holding the DP run sets (eps1, eps10, ...)."""
    return _flag_path(FLAGS.dp_logdir, log_root() / "dp")


def config_log_dir(dp_root: Path, dir_name: str) -> Path:
    """Directory holding the runs of a config.

    The DP run sets are subdirectories of `--dp_logdir`; the non-private run sets are
    siblings of `dp/` and have their own flags.
    """
    if dir_name == "nonprivate":
        return _flag_path(FLAGS.nonprivate_logdir, log_root() / "nonprivate")
    if dir_name == "nonprivate_p":
        return _flag_path(FLAGS.nonprivate_logdir_p, log_root() / "nonprivate_p")
    return dp_root / dir_name


def min_partition_size(runs: list[Path]) -> int:
    """Smallest IN or OUT group the training-subset masks give any patient.

    Records inherit their patient's IN/OUT status, so this bounds the group sizes
    the statistical test will see.
    """
    masks = np.stack([np.load(r / "patient_subset_mask.npy").astype(bool) for r in runs])
    in_counts = masks.sum(axis=0)
    return int(min(in_counts.min(), len(runs) - in_counts.max()))


def run_dirs(config_dir: Path) -> list[Path]:
    """Run directories of a config, one per unique training subset.

    Indices of failed runs are handed out again, which can leave two run directories
    trained on the same subset. Runs are deduplicated by their training-subset mask
    (keeping the first directory by name) so that no subset is counted twice.
    """
    if not config_dir.is_dir():
        return []
    unique: dict[str, Path] = {}
    for run in sorted(d for d in config_dir.iterdir() if d.is_dir()):
        mask_path = run / "patient_subset_mask.npy"
        if not mask_path.exists():  # unfinished run, no predictions to load either
            continue
        unique.setdefault(hashlib.md5(mask_path.read_bytes()).hexdigest(), run)
    return list(unique.values())


def available_configs(dp_root: Path) -> tuple[list[tuple], list[tuple]]:
    """Configs to analyse, as (testable, has runs at all).

    The memorisation test needs a minimum group size per record. The AUROC panel only reads
    info.json, so configs with too few runs to test are still included there.
    """
    testable, with_runs, missing, too_thin = [], [], [], []
    for cfg in ALL_CONFIGS:
        config_dir = config_log_dir(dp_root, cfg[0])
        runs = run_dirs(config_dir)
        if not runs:
            missing.append(cfg[0])
            continue
        with_runs.append(cfg)
        n_dirs = sum(1 for d in config_dir.iterdir() if d.is_dir())
        if n_dirs > len(runs):
            print(
                f"  {cfg[0]}: {n_dirs} run directories cover {len(runs)} unique "
                f"training subsets, using one directory per subset"
            )
        smallest = min_partition_size(runs)
        if smallest < MIN_GROUP_SIZE:
            too_thin.append(f"{cfg[0]} ({len(runs)} runs, thinnest split {smallest})")
            continue
        testable.append(cfg)
    if missing:
        print(f"No runs found for {', '.join(missing)} — these configs are skipped.")
    if too_thin:
        print(
            f"Too few runs to test {', '.join(too_thin)} — the test needs at least "
            f"{MIN_GROUP_SIZE} predictions on each side of every record's IN/OUT split. "
            "These arms still appear in the performance panel."
        )
    return testable, with_runs


def read_manifest_runs(config_out_dir: Path) -> int | None:
    """Number of model runs the cached results were computed from, if recorded."""
    path = config_out_dir / MANIFEST_FILE
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("n_runs")


def write_manifest(config_out_dir: Path, n_runs: int):
    """Record how many model runs the results just written were computed from."""
    with open(config_out_dir / MANIFEST_FILE, "w") as f:
        json.dump({"n_runs": n_runs}, f)


def cache_is_current(config_out_dir: Path, config_dir: Path) -> bool:
    """True if cached per-config CSVs cover exactly the model runs on disk.

    Runs that appear or finish after the CSVs were written invalidate the cache.
    """
    paths = [config_out_dir / fname for fname in RESULT_FILES.values()]
    if not all(p.exists() for p in paths):
        return False
    runs = run_dirs(config_dir)
    recorded_runs = read_manifest_runs(config_out_dir)
    if recorded_runs is not None and recorded_runs != len(runs):
        return False
    return min(p.stat().st_mtime for p in paths) > max(d.stat().st_mtime for d in runs)


def summarise_results(
    results: dict, epsilon: float, dp_level: str, dir_name: str, n_runs: int
) -> dict:
    """Build a summary row (rejection counts/rates per split) from per-record results.

    `n_runs` is included because detection power depends on the number of runs.
    """
    row = {
        "epsilon": epsilon,
        "dp_level": dp_level,
        "dir_name": dir_name,
        "n_runs": n_runs,
    }
    for split in RESULT_FILES:
        rejected = results[split]["df"]["rejected"].astype(bool)
        row[f"n_rejected_{split}"] = int(rejected.sum())
        row[f"n_total_{split}"] = int(len(rejected))
        row[f"rate_{split}"] = float(rejected.mean() * 100)
    return row


def analyse_dp_config(
    config_dir: Path,
    train_historical_dataset,
    train_future_dataset,
    transform_fn,
    patient_col: str,
    test_type: str,
):
    """Run memorisation analysis for a single DP configuration directory."""
    logdirs = run_dirs(config_dir)
    print(f"  Found {len(logdirs)} model runs in {config_dir.name}")

    results = {}

    # --- Future (long_eval) records ---
    load_fn_future = partial(
        load_from_dir,
        prefix_str="long_eval",
        load_as_memmap=True if FLAGS.dataset == "heedb" else False,  # use memmap for HEEDB since the predictions are too large to fit into memory
    )
    preds_list_future, patient_ids_future, masks_future = load_predictions(
        log_dirs=logdirs,
        load_fn=load_fn_future,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    masks_future = convert_patientmasks_to_recordmask(
        patient_masks=masks_future,
        patient_ids=patient_ids_future,
        dataset=train_future_dataset,
        patient_id_col=patient_col,
    )
    future_partitioner = RandomSubsetPredictionPartitioner(
        preds_list=preds_list_future,
        inclusion_masks=masks_future,
        dtype=preds_list_future[0].dtype,
        transform_fn=transform_fn,
    )
    stats_future, rejected_future, pvals_future, corrected_pvals_future = test_significantly_different(
        prediction_partioner=future_partitioner,
        test_fn_object=get_test_object(test_type),
        alpha=0.05,
    )
    # compute effect_size for future records
    mean_in_future, mean_out_future = [], []
    for i in tqdm(range(len(future_partitioner)), desc="  mean IN/OUT (future)", leave=False):
        r = future_partitioner(i, reduce="mean")
        mean_in_future.append(r[0])
        mean_out_future.append(r[1])
    mean_in_future = np.array(mean_in_future, dtype=mean_in_future[0].dtype)
    mean_out_future = np.array(mean_out_future, dtype=mean_out_future[0].dtype)
    effect_size_future = np.max(np.abs(mean_in_future - mean_out_future), axis=-1).squeeze()

    results["future"] = {
        "n_rejected": int(sum(rejected_future)),
        "n_total": len(rejected_future),
        "rate": sum(rejected_future) / len(rejected_future) * 100,
        "df": pd.DataFrame({
            "id": train_future_dataset.dataframe.index,
            "test_statistic": stats_future,
            "pvals(raw)": pvals_future,
            "pvals(corrected)": corrected_pvals_future,
            "effect_size": effect_size_future,
            "rejected": rejected_future,
            "pid": train_future_dataset.dataframe[patient_col],
        }),
    }

    # --- Historical (train) records ---
    load_fn_historical = partial(
        load_from_dir,
        prefix_str="train",
        load_as_memmap=True if FLAGS.dataset == "heedb" else False,  # use memmap for HEEDB since the predictions are too large to fit into memory
    )
    preds_list_historical, patient_ids_historical, masks_historical = load_predictions(
        log_dirs=logdirs,
        load_fn=load_fn_historical,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    masks_historical = convert_patientmasks_to_recordmask(
        patient_masks=masks_historical,
        patient_ids=patient_ids_historical,
        dataset=train_historical_dataset,
        patient_id_col=patient_col,
    )
    historical_partitioner = RandomSubsetPredictionPartitioner(
        preds_list=preds_list_historical,
        inclusion_masks=masks_historical,
        dtype=preds_list_historical[0].dtype,
        transform_fn=transform_fn,
    )
    stats_historical, rejected_historical, pvals_historical, corrected_pvals_historical = test_significantly_different(
        prediction_partioner=historical_partitioner,
        test_fn_object=get_test_object(test_type),
        alpha=0.05,
    )
    # compute effect_size for historical records
    mean_in_historical, mean_out_historical = [], []
    for i in tqdm(range(len(historical_partitioner)), desc="  mean IN/OUT (historical)", leave=False):
        r = historical_partitioner(i, reduce="mean")
        mean_in_historical.append(r[0])
        mean_out_historical.append(r[1])
    mean_in_historical = np.array(mean_in_historical, dtype=mean_in_historical[0].dtype)
    mean_out_historical = np.array(mean_out_historical, dtype=mean_out_historical[0].dtype)
    effect_size_historical = np.max(np.abs(mean_in_historical - mean_out_historical), axis=-1).squeeze()

    results["historical"] = {
        "n_rejected": int(sum(rejected_historical)),
        "n_total": len(rejected_historical),
        "rate": sum(rejected_historical) / len(rejected_historical) * 100,
        "df": pd.DataFrame({
            "id": train_historical_dataset.dataframe.index,
            "test_statistic": stats_historical,
            "pvals(raw)": pvals_historical,
            "pvals(corrected)": corrected_pvals_historical,
            "effect_size": effect_size_historical,
            "rejected": rejected_historical,
            "pid": train_historical_dataset.dataframe[patient_col],
        }),
    }

    return results


def _epsilon_label(eps: float) -> str:
    """Format epsilon as a power-of-10 LaTeX label (1 -> '$1$', 10 -> '$10^1$', inf -> '$\\infty$')."""
    if not np.isfinite(eps):
        return r"$\infty$"
    if eps == 1:
        return r"$1$"
    exp = int(round(np.log10(eps)))
    return rf"$10^{exp}$"


def _curve_label(eps: float) -> str:
    """Legend label for one epsilon's curve, e.g. '$\\varepsilon=10^2$'."""
    if not np.isfinite(eps):
        return r"$\varepsilon=\infty$ (non-private)"
    return rf"$\varepsilon={_epsilon_label(eps).strip('$')}$"


def _aligned_values(summary_df: pd.DataFrame, dp_level: str, epsilons: list, col: str):
    """Values of `col` for `dp_level`, aligned to `epsilons` (NaN where a config is missing).

    NaN entries are drawn as absent bars.
    """
    rows = summary_df[summary_df["dp_level"] == dp_level].set_index("epsilon")
    return np.array([rows[col].get(e, np.nan) for e in epsilons], dtype=float)


def _level_label(name: str, totals: np.ndarray) -> str:
    """Legend label carrying the denominator, e.g. 'Record-level DP (n=480,200)'.

    Levels without any finished config are kept out of the legend.
    """
    finite = totals[np.isfinite(totals)]
    if len(finite) == 0:
        return "_nolegend_"
    return f"{name} (n={int(finite.max()):,})"


def _count_ylim(counts) -> tuple[float, float]:
    """(floor, ceiling) of the log y-axis the count panels share.

    A log axis cannot show a zero-height bar, so the floor sits a decade below the
    smallest non-zero count of *any* panel -- derived per panel it would differ
    between them, and on a shared axis the last one drawn would win and clip the
    others. NaN counts are configs that have not been analysed.
    """
    positive = [c for c in counts if np.isfinite(c) and c > 0]
    if not positive:
        return 1e-3, 1.0
    return min(positive) / 10, max(positive) * 2


def _label_bars(
    ax,
    bars,
    labels,
    ylim: tuple[float, float],
    inside_threshold: float | str | None = "auto",
):
    """Annotate bars with their counts on the y-range `ylim` and return the texts.

    Labels sit inside bars taller than `inside_threshold` ("auto" = a tenth of the
    tallest bar), which keeps the y-range from having to grow for every label.
    Exact zeros cannot be drawn on a log axis, so they are labelled explicitly just
    above the floor.
    """
    ax.set_ylim(*ylim)
    bottom = ylim[0]
    heights = [b.get_height() for b in bars]
    if inside_threshold == "auto":
        positive = [h for h in heights if np.isfinite(h) and h > 0]
        inside_threshold = max(positive) / 10 if positive else None

    texts = []
    for bar, label in zip(bars, labels):
        if label is None:  # config not available
            continue
        height = bar.get_height()
        if height == 0:
            y_pos, va, label = bottom * 1.3, "bottom", "0"
        elif inside_threshold is not None and height >= inside_threshold:
            y_pos, va = height / 1.15, "top"
        else:
            y_pos, va = height * 1.15, "bottom"
        texts.append(
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                label,
                ha="center",
                va=va,
                fontsize=FONT_SIZE - 1,
                rotation=90,
                color="black",
            )
        )
    return texts


def _fit_ylim_to_labels(fig, panels, bottom: float, max_frac: float = 0.97):
    """Grow the shared log-scale y-range until every panel's bar labels fit inside it.

    `panels` is a list of (ax, texts, legend); the range has to satisfy the worst
    panel, since they share the axis. The labels (and the legend boxes) have a fixed
    height in points, so their height as a fraction of the axes shrinks as the range
    grows; a few iterations converge on a tight fit that clears the legends.
    """
    panels = [(ax, texts, legend) for ax, texts, legend in panels if texts]
    if not panels:
        return
    for _ in range(6):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        needed = max(
            t.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted()).y1
            for ax, texts, _ in panels
            for t in texts
        )
        limit = max_frac
        for ax, _, legend in panels:
            if legend is None:
                continue
            legend_bottom = (
                legend.get_window_extent(renderer).transformed(ax.transAxes.inverted()).y0
            )
            limit = min(limit, legend_bottom - 0.03)
        limit = max(limit, 0.35)
        if needed <= limit:
            break
        top = max(ax.get_ylim()[1] for ax, _, _ in panels)
        decades = np.log10(top) - np.log10(bottom)
        for ax, _, _ in panels:
            ax.set_ylim(bottom, bottom * 10 ** (decades * needed / limit))


def _draw_nonprivate_baseline(ax, value, color: str) -> bool:
    """Draw one level's eps=infinity baseline as a dashed line across `ax`.

    The non-private arms are not a privacy budget, so they read better as a reference
    level than as another group on the epsilon axis. The line carries no value label
    -- the panels are too dense for one to sit anywhere without landing on a bar
    label or the legend. Returns whether a line was drawn (a level without finished
    non-private runs has none).
    """
    if value is None or not np.isfinite(value):
        return False
    # zorder below the bars and scatter dots (1), so the data sits on top of the line
    ax.axhline(
        value, color=color, linestyle=BASELINE_STYLE, linewidth=BASELINE_LW,
        alpha=0.9, zorder=0.5,
    )
    return True


def _baseline_handle():
    """Legend proxy for the dashed non-private reference lines (colour-neutral).

    The level a line belongs to is already in the legend via its colour, so one
    grey entry carries the meaning of the dashes without doubling the legend.
    """
    return plt.Line2D(
        [], [], color="0.35", linestyle=BASELINE_STYLE, linewidth=BASELINE_LW,
        label=r"Non-private ($\varepsilon=\infty$)",
    )


def _hidden_by(box, obstacle, shift: float = 0.0) -> bool:
    """Would `box`, moved up by `shift` pixels, land on `obstacle` (a display bbox)?"""
    if obstacle is None:
        return False
    return (
        box.x0 < obstacle.x1
        and box.x1 > obstacle.x0
        and box.y0 + shift < obstacle.y1
        and box.y1 + shift > obstacle.y0
    )


def _nudge_labels_off_baselines(fig, panels, pad_px: float = 1.5):
    """Shift bar labels vertically until no non-private baseline crosses one.

    `panels` is a list of (ax, texts, baseline values, legend). A crossed label moves
    up until it clears the topmost line it touches or down until it clears the bottom
    one, whichever keeps it inside the axes and out from behind the legend; where both
    directions do, the shorter move wins, so the label hugs the line instead of
    drifting away from the bar it belongs to. The work happens in display space because
    a label's height in data coordinates depends on the (log) y-range, which is only
    final once every panel is drawn; moving one label never drags another, so a couple
    of passes settle labels crossing more than one line.
    """
    panels = [
        (ax, texts, [v for v in values if np.isfinite(v)], legend)
        for ax, texts, values, legend in panels
        if texts
    ]
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        moved = False
        for ax, texts, values, legend in panels:
            lines_px = [ax.transData.transform((0, v))[1] for v in values]
            floor_px, ceiling_px = (
                ax.transData.transform((0, bound))[1] for bound in ax.get_ylim()
            )
            legend_box = legend.get_window_extent(renderer) if legend else None
            for text in texts:
                box = text.get_window_extent(renderer=renderer)
                crossing = [y for y in lines_px if box.y0 - pad_px <= y <= box.y1 + pad_px]
                if not crossing:
                    continue
                up = max(crossing) + pad_px - box.y0
                down = min(crossing) - pad_px - box.y1
                # (lands behind the legend, distance) -- least bad move first
                candidates = [
                    (_hidden_by(box, legend_box, shift), abs(shift), shift)
                    for shift, fits in ((up, box.y1 + up <= ceiling_px),
                                        (down, box.y0 + down >= floor_px))
                    if fits
                ]
                if not candidates:
                    continue
                shift = min(candidates)[2]
                x, y = text.get_position()
                y_px = ax.transData.transform((0, y))[1] + shift
                text.set_position((x, ax.transData.inverted().transform((0, y_px))[1]))
                moved = True
        if not moved:
            break


def plot_memorisation_counts(summary_df: pd.DataFrame, out_dir: Path):
    """Plot memorised-record counts vs epsilon as two panels: future left, historical right.

    Bars carry the count of records the test rejects, not the share of the cohort;
    the denominators are in each panel's legend, which is why the legends stay per
    panel -- the two splits have different cohorts. The non-private baseline of each
    level is a dashed line in that level's colour rather than an eps = infinity group
    on the axis. Both panels are drawn on one shared log y-range wide enough for
    either split, so their counts are directly comparable even though they differ by
    orders of magnitude.
    """
    epsilons = [e for e in sorted(summary_df["epsilon"].unique()) if np.isfinite(e)]
    x = np.arange(len(epsilons))
    width = 0.35

    counts, n_total, baselines = {}, {}, {}
    for split in SPLIT_TITLES:
        for lvl in ("patient", "record"):
            counts[split, lvl] = _aligned_values(
                summary_df, lvl, epsilons, f"n_rejected_{split}"
            )
            n_total[split, lvl] = _aligned_values(
                summary_df, lvl, epsilons, f"n_total_{split}"
            )
            baselines[split, lvl] = _aligned_values(
                summary_df, lvl, [np.inf], f"n_rejected_{split}"
            )[0]
    ylim = _count_ylim(
        np.concatenate([*counts.values(), np.array(list(baselines.values()))])
    )

    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.0))
    panels, label_panels = [], []
    for ax, (split, split_title) in zip(axes, SPLIT_TITLES.items()):
        bars_p = ax.bar(x - width / 2, counts[split, "patient"], width, color=COLOR_PATIENT,
                        alpha=0.6,
                        label=_level_label("Patient-level", n_total[split, "patient"]))
        bars_r = ax.bar(x + width / 2, counts[split, "record"], width, color=COLOR_RECORD,
                        alpha=0.6,
                        label=_level_label("Record-level", n_total[split, "record"]))

        bar_labels = [
            f"N={int(n):,}" if np.isfinite(n) else None
            for lvl in ("patient", "record")
            for n in counts[split, lvl]
        ]

        #ax.set_title(split_title, fontsize=FONT_SIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([_epsilon_label(e) for e in epsilons])
        ax.set_yscale("log")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # add value labels: inside bar for tall bars, above for short ones. One
        # threshold for both panels, since "tall" is now the same scale in each
        all_bars = list(bars_p) + list(bars_r)
        texts = _label_bars(ax, all_bars, bar_labels, ylim, inside_threshold=1e4)

        # non-private (eps = infinity) baselines of both levels
        drew_baseline = False
        for lvl, color in (("patient", COLOR_PATIENT), ("record", COLOR_RECORD)):
            drew_baseline |= _draw_nonprivate_baseline(
                ax, baselines[split, lvl], color
            )

        legend = None
        if FLAGS.plot_legend:
            handles, labels = ax.get_legend_handles_labels()
            if drew_baseline:
                handles.append(_baseline_handle())
                labels.append(handles[-1].get_label())
            legend = ax.legend(
                handles,
                labels,
                fontsize=FONT_SIZE - 1,
                loc="upper center",
                handlelength=1.1,
                handletextpad=0.5,
                borderpad=0.3,
                labelspacing=0.3,
            )
        if legend is not None:
            # keep constrained_layout from shrinking the axes around the legend box
            legend.set_in_layout(False)
        panels.append((ax, texts, legend))
        label_panels.append(
            (ax, texts, [baselines[split, lvl] for lvl in ("patient", "record")], legend)
        )

    #_fit_ylim_to_labels(fig, panels, ylim[0])
    axes[0].set_ylim((ylim[0], 1e6))
    axes[1].set_ylim((ylim[0], 1e6))
    # only now is the y-range final, so bar labels can be measured against the lines
    _nudge_labels_off_baselines(fig, label_panels)
    fig.supxlabel(r"Privacy Budget ($\varepsilon$)", fontsize=FONT_SIZE)
    axes[0].set_ylabel("Future records with significant\n change in prediction (count)", fontsize=FONT_SIZE)
    axes[1].set_ylabel("Historical records with significant\n change in prediction (count)", fontsize=FONT_SIZE)

    save_path = out_dir / "dp_memorisation_count.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved count plot: {save_path}")


def _metric_xlabel(metric_key: str) -> str:
    """x-axis label of a metric's eSF panels.

    Resolved at plot time rather than in ESF_METRICS, because the test statistic is
    named after whichever test --test selected.
    """
    metric = ESF_METRICS[metric_key]
    if metric["xlabel"] is not None:
        return metric["xlabel"]
    return TEST_STATISTIC_LABELS.get(FLAGS.test, f"{FLAGS.test.capitalize()} test statistic")


def _metric_xlim(metric_key: str) -> tuple[float, float | None]:
    """x-limits of a metric's eSF panels; a None upper bound autoscales.

    Effect sizes are bounded percentages and so get fixed limits; the test statistic
    has no natural range, so it autoscales unless --teststat_xlim pins it.
    """
    if metric_key == "teststat" and FLAGS.teststat_xlim:
        lo, hi = (float(v) for v in FLAGS.teststat_xlim.split(","))
        return lo, hi
    return ESF_METRICS[metric_key]["xlim"] or (0.0, None)


def _esf_entries(values: dict, dp_level: str, split: str) -> list[tuple]:
    """(epsilon, per-record values) pairs of one setting, ascending in epsilon.

    The non-private arm carries eps=infinity, so it sorts last and takes the
    darkest step of the ramp.
    """
    return sorted(
        [(eps, arr) for (eps, lvl, s), arr in values.items()
         if lvl == dp_level and s == split],
        key=lambda x: x[0],
    )


def _draw_esf_panel(
    ax,
    entries: list[tuple],
    dp_level: str,
    metric_key: str,
    ref_arr=None,
    legend_title: str | None = None,
):
    """Draw the eSF curves of a single setting -- one DP level, one split -- onto `ax`.

    Curves are coloured along that level's epsilon ramp, so a budget keeps its
    colour wherever it is drawn. The dashed reference is the non-private model's
    random partition, i.e. what the metric looks like in the absence of memorisation.
    The axis is left unlabelled: the figure carries one shared x-label for both panels.
    `legend_title` heads the legend, which is where the panel names its DP level.
    """
    scale = ESF_METRICS[metric_key]["scale"]
    styles = dp_esf_styles(dp_level, [eps for eps, _ in entries])
    for eps, arr in entries:
        color, alpha = styles[eps]
        res = scipy.stats.ecdf(arr * scale)
        res.sf.plot(ax=ax, color=color, linestyle="-", label=_curve_label(eps), alpha=alpha)

    if ref_arr is not None:
        res_ref = scipy.stats.ecdf(ref_arr * scale)
        res_ref.sf.plot(
            ax=ax, color="#888888", linestyle="--",
            label="Random partitioning", alpha=0.8,
        )

    ax.set_xlim(*_metric_xlim(metric_key))
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if FLAGS.plot_legend:
        legend = ax.legend(
            fontsize=FONT_SIZE - 1,
            loc="upper right",
            title=legend_title,
            title_fontsize=FONT_SIZE - 1,
        )
        legend.set_alignment("left")


def plot_survival_function_panels(
    values: dict,
    out_dir: Path,
    metric_key: str,
    ref_values: dict | None = None,
):
    """Plot a two-panel eSF figure per split: record-level DP left, patient-level right.

    Each panel holds exactly one setting, and the two share both axes so the levels
    can be read against each other; the shared x-axis carries one figure-level label
    rather than the same label twice. A panel names its DP level in its legend title,
    falling back to a panel title when --noplot_legend drops the legend. A split whose
    levels have not both finished gets no figure -- the layout is the comparison.
    """
    for split in ["historical", "future"]:
        entries = {lvl: _esf_entries(values, lvl, split) for lvl in DP_LEVEL_LABELS}
        if not all(entries.values()):
            print(
                f"  Skipping the {metric_key} eSF plot ({split}): "
                "only one DP level available"
            )
            continue

        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.65), sharex=True, sharey=True)
        ref_arr = ref_values.get(split) if ref_values else None
        for ax, (dp_level, level_label) in zip(axes, DP_LEVEL_LABELS.items()):
            _draw_esf_panel(
                ax, entries[dp_level], dp_level, metric_key,
                ref_arr=ref_arr, legend_title=level_label,
            )
            if not FLAGS.plot_legend:
                ax.set_title(level_label, fontsize=FONT_SIZE)
        axes[0].set_ylabel("1 - cumulative probability")
        fig.supxlabel(_metric_xlabel(metric_key), fontsize=FONT_SIZE)

        save_path = out_dir / f"dp_esf_{metric_key}_{split}.pdf"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved eSF plot: {save_path}")


def collect_auroc_scores(dp_root: Path, configs: list[tuple] | None = None) -> pd.DataFrame:
    """Collect test-set macro AUROC scores from info.json files for each DP config."""
    rows = []
    for dir_name, epsilon, dp_level in configs if configs is not None else ALL_CONFIGS:
        config_dir = config_log_dir(dp_root, dir_name)
        if not config_dir.exists():
            continue
        for run_dir in run_dirs(config_dir):
            info_path = run_dir / "info.json"
            if not info_path.exists():
                continue
            with open(info_path) as f:
                info = json.load(f)
            auroc = info.get("test_metrics", {}).get("_test_macro_auroc")
            if auroc is not None:
                rows.append({
                    "dir_name": dir_name,
                    "epsilon": epsilon,
                    "dp_level": dp_level,
                    "run": run_dir.name,
                    "macro_auroc": float(auroc),
                })
    return pd.DataFrame(rows)


def _dot_handle(color: str, label: str):
    """Legend proxy matching the coloured scatter clouds of the AUROC plot."""
    return plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=4, alpha=0.6, label=label)


def _auroc_marker(ax, x_pos, vals, color):
    """Draw individual run AUROCs plus a mean +/- std marker at `x_pos`."""
    if len(vals) == 0:
        return False
    jitter = np.random.uniform(-0.025, 0.025, size=len(vals))
    ax.scatter(np.full(len(vals), x_pos) + jitter, vals, color=color, s=4, alpha=0.3, zorder=1)
    ax.errorbar(
        x_pos,
        vals.mean(),
        yerr=vals.std(),
        fmt="o",
        color="white",
        markeredgecolor="black",
        markeredgewidth=0.7,
        markersize=4,
        ecolor="black",
        capsize=4,
        elinewidth=0.7,
        zorder=3,
    )
    return True


def _auroc_ylim(values: np.ndarray) -> tuple[float, float]:
    """y-limits of the AUROC panel.

    The --auroc_ylim flag wins; otherwise the dataset's fixed limits are used, and
    datasets without them get limits rounded outwards from the data.
    """
    if FLAGS.auroc_ylim:
        lo, hi = (float(v) for v in FLAGS.auroc_ylim.split(","))
        return lo, hi
    fixed = dataset_spec().get("auroc_ylim")
    if fixed is not None:
        return fixed
    lo = np.floor((values.min() - 1) / 2.5) * 2.5
    hi = np.ceil((values.max() + 1) / 2.5) * 2.5
    return float(lo), float(hi)


def plot_auroc_bars(auroc_df: pd.DataFrame, out_dir: Path):
    """Plot scatter + error-bar chart of test-set macro AUROC per epsilon.

    The non-private runs of each level are drawn as a dashed line at their mean
    AUROC, in that level's colour, instead of as an eps = infinity group.
    """
    epsilons = [e for e in sorted(auroc_df["epsilon"].unique()) if np.isfinite(e)]
    offset = 0.15  # half-spacing between record and patient dots

    fig, ax = plt.subplots()
    handles = []
    drew_baseline = False

    for dp_level, color, x_off, label in [
        ("patient", COLOR_PATIENT, -offset, "Patient-level"),
        ("record", COLOR_RECORD, offset, "Record-level"),
    ]:
        sub = auroc_df[auroc_df["dp_level"] == dp_level]
        if sub.empty:
            continue
        drawn = False
        for idx, eps in enumerate(epsilons):
            vals = sub[sub["epsilon"] == eps]["macro_auroc"].values * 100
            drawn |= _auroc_marker(ax, idx + x_off, vals, color)
        if drawn:
            handles.append(_dot_handle(color, label))
        nonprivate = sub[~np.isfinite(sub["epsilon"])]["macro_auroc"].values * 100
        if len(nonprivate):
            drew_baseline |= _draw_nonprivate_baseline(
                ax, float(nonprivate.mean()), color
            )

    ax.set_xlabel(r"Privacy Budget ($\varepsilon$)")
    ax.set_ylabel("Diagnostic Performance (%)")
    lo, hi = _auroc_ylim(auroc_df["macro_auroc"].values * 100)
    ax.set_ylim(lo, hi)
    ax.set_yticks(np.arange(lo, hi + 0.1, 2.5))
    ax.set_xticks(np.arange(len(epsilons)))
    ax.set_xticklabels([_epsilon_label(e) for e in epsilons])
    if drew_baseline:
        handles.append(_baseline_handle())
    if FLAGS.plot_legend:
        ax.legend(handles=handles, fontsize=FONT_SIZE, loc="lower right").set_in_layout(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = out_dir / "dp_test_macro_auroc.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved AUROC plot: {save_path}")


def _cached_results(config_out_dir: Path) -> dict | None:
    """Load per-record results for one config from its cached CSVs, if present."""
    paths = {split: config_out_dir / fname for split, fname in RESULT_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    return {split: {"df": pd.read_csv(p)} for split, p in paths.items()}


def _read_column(path: Path, column: str) -> np.ndarray | None:
    """One column of a CSV, or None if the file or the column is absent."""
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda c: c == column)
    return df[column].values if column in df.columns else None


def _load_reference_values(out_dir: Path, base_figs_dir: Path) -> tuple[dict, dict]:
    """Load the random-partitioning results of memorisation_analysis.py, per metric and split.

    Read from <base_figs_dir>/files/, written by memorisation_analysis.py with --sanity_checks.

    Returns (values, sources), both keyed by metric: {split: array} and the names of
    the analyses those arrays came from.
    """
    sources = {
        str(base_figs_dir): {
            "historical": base_figs_dir / "files" / f"{FLAGS.dataset}_mem_data_train_sanity.csv",
            "future": base_figs_dir / "files" / f"{FLAGS.dataset}_mem_data_le_sanity.csv",
        },
    }
    ref, ref_sources = {}, {}
    for key, metric in ESF_METRICS.items():
        ref[key], ref_sources[key] = {}, []
        for split in RESULT_FILES:
            for source_name, paths in sources.items():
                values = _read_column(paths[split], metric["column"])
                if values is not None:
                    ref[key][split] = values
                    if source_name not in ref_sources[key]:
                        ref_sources[key].append(source_name)
                    break
    return ref, ref_sources


def main(argv):
    np.random.seed(FLAGS.seed)
    spec = dataset_spec()
    dp_root = dp_log_root()
    out_dir = _flag_path(FLAGS.out_dir, f"./figs/{FLAGS.dataset}_dp")
    fig_dir_exists(out_dir)
    print(f"Analysing {FLAGS.dataset} DP runs from {dp_root}, writing to {out_dir}")

    # only configs with finished runs are analysed
    configs, configs_with_runs = available_configs(dp_root)
    if not configs_with_runs:
        print(f"No model runs found under {dp_root}, nothing to do.")
        return
    if configs:
        print(
            "Analysing "
            + ", ".join(
                f"{c[0]} ({len(run_dirs(config_log_dir(dp_root, c[0])))} runs)"
                for c in configs
            )
        )
    else:
        print("No arm has enough runs for the memorisation test yet.")
    orphaned = [
        c[0] for c in ALL_CONFIGS
        if c not in configs and (out_dir / c[0] / RESULT_FILES["historical"]).exists()
    ]
    if orphaned:
        print(
            f"NOTE: {', '.join(orphaned)} have cached CSVs but no model runs; "
            "they are left out of the figures."
        )

    # Collect test-set AUROC scores from info.json files
    auroc_df = collect_auroc_scores(dp_root, configs_with_runs)
    if not auroc_df.empty:
        auroc_path = out_dir / "dp_test_macro_auroc.csv"
        auroc_df.to_csv(auroc_path, index=False)
        print(f"Collected AUROC scores for {len(auroc_df)} runs, saved to {auroc_path}")
        plot_auroc_bars(auroc_df, out_dir)
    else:
        print("WARNING: No AUROC scores found in info.json files")

    # Random-partition null distributions drawn as references in the eSF plots
    ref_values, ref_sources = _load_reference_values(
        out_dir, _flag_path(FLAGS.nonprivate_figs_dir, f"./figs/{FLAGS.dataset}")
    )
    for key, metric in ESF_METRICS.items():
        if ref_values[key]:
            print(
                f"Loaded random-partition {metric['name']} reference from "
                f"{', '.join(ref_sources[key])} ({', '.join(ref_values[key])})"
            )
        else:
            print(f"WARNING: No random-partition {metric['name']} reference CSVs found")

    patient_col = get_patient_col(FLAGS.dataset)
    transform_fn = get_act_fn(FLAGS.dataset)
    datasets: dict[str, tuple] = {}

    def get_datasets(dp_level: str):
        """Load (and cache) the dataset variant matching a DP level."""
        if dp_level not in datasets:
            name = spec["patient_dataset" if dp_level == "patient" else "record_dataset"]
            print(f"Loading {name} dataset ({dp_level}-level)...")
            train_hist, _, train_future, _ = get_dataset(
                dataset_name=name,
                csv_root=Path(FLAGS.csv_root),
                data_root=get_data_root(FLAGS.dataset),
                save_root=Path(FLAGS.save_root),
                resolution=spec["resolution"],
                use_cached=True,
                load_from_disk=True,
                write_to_disk=False,
            )
            datasets[dp_level] = (train_hist, train_future)
        return datasets[dp_level]

    summary_rows = []
    # per-record values behind the eSF figures: metric -> (epsilon, dp_level, split)
    esf_values = {key: {} for key in ESF_METRICS}
    # arms whose cached CSVs predate a metric's column, reported once at the end
    stale_cache = {key: [] for key in ESF_METRICS}

    for dir_name, epsilon, dp_level in configs:
        config_dir = config_log_dir(dp_root, dir_name)
        config_out_dir = out_dir / dir_name

        print(f"\n{'='*60}")
        arm = (
            f"eps={epsilon:g}, {dp_level}-level DP"
            if np.isfinite(epsilon)
            else f"non-private, {dp_level}-level"
        )
        print(f"Analysing {dir_name} ({arm})")
        print(f"{'='*60}")

        use_cache = FLAGS.plot_only or (
            not FLAGS.recompute and cache_is_current(config_out_dir, config_dir)
        )
        results = _cached_results(config_out_dir) if use_cache else None

        if results is not None:
            print(f"  Using cached results from {config_out_dir}/")
        elif FLAGS.plot_only:
            print(f"  No cached results in {config_out_dir}/, skipping (--plot_only).")
            continue
        else:
            train_hist, train_future = get_datasets(dp_level)
            results = analyse_dp_config(
                config_dir=config_dir,
                train_historical_dataset=train_hist,
                train_future_dataset=train_future,
                transform_fn=transform_fn,
                patient_col=patient_col,
                test_type=FLAGS.test,
            )
            # Save per-record results to subdirectory
            config_out_dir.mkdir(parents=True, exist_ok=True)
            for split, fname in RESULT_FILES.items():
                results[split]["df"].to_csv(config_out_dir / fname, index=False)
            write_manifest(config_out_dir, len(run_dirs(config_dir)))
            print(f"  Saved per-record CSVs to {config_out_dir}/")

        row = summarise_results(
            results, epsilon, dp_level, dir_name, len(run_dirs(config_dir))
        )
        summary_rows.append(row)
        for split in RESULT_FILES:
            df = results[split]["df"]
            for key, metric in ESF_METRICS.items():
                if metric["column"] in df.columns:
                    esf_values[key][(epsilon, dp_level, split)] = df[metric["column"]].values
                elif dir_name not in stale_cache[key]:
                    stale_cache[key].append(dir_name)
            print(
                f"  {split.capitalize():<11}: {row[f'n_rejected_{split}']}/"
                f"{row[f'n_total_{split}']} ({row[f'rate_{split}']:.2f}%)"
            )

    if not summary_rows:
        print("\nNo results available to plot.")
        return

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "dp_memorisation_rates.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")
    print(summary_df.to_string(index=False))

    # Plot bar charts
    plot_memorisation_counts(summary_df, out_dir)

    # Plot empirical survival functions, one set of figures per metric
    for key, metric in ESF_METRICS.items():
        if stale_cache[key]:
            print(
                f"NOTE: the cached CSVs of {', '.join(stale_cache[key])} have no "
                f"'{metric['column']}' column; re-run with --recompute to include "
                f"these arms in the {metric['name']} eSF plots."
            )
        if not esf_values[key]:
            print(f"Skipping the {metric['name']} eSF plots: no arm provides it.")
            continue
        plot_survival_function_panels(
            esf_values[key], out_dir, key, ref_values=ref_values[key]
        )
    print("\nDone.")


if __name__ == "__main__":
    app.run(main)
