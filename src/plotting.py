"""Plotting helpers and the receiver-operating-point comparison.

* Plots used by the analysis scripts: empirical survival functions (`esf_plot`), memorisation
  rate by follow-up interval (`plot_binned_memorisation_rate`), bar charts (`bar_plot`), and
  the records/patients with the largest IN/OUT prediction change
  (`plot_images_by_change_in_prediction`, `plot_patient_trajectories`).
* `get_time_deltas`: months between a future record and the same patient's most recent
  historical record.
* Receiver-operating-point comparison (`compute_decision_thresholds`,
  `rop_comparison_by_disease`): per-disease sensitivity and specificity of IN vs OUT models at
  fixed decision thresholds. p-values come from a run-label permutation test
  (`run_label_permutation_test`), confidence intervals from inverting the same test
  (`permutation_interval`, `bonferroni_figure_correction`).

Figure style is set by the calling scripts.
"""

import math
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import matplotlib.patches as mpatches  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics as skm  # type: ignore
from tqdm import tqdm
from functools import partial

from .data.datasets import BaseDataset
from .memorisation.core import RandomSubsetPredictionPartitioner
from .utils import get_label_list

my_blue = "#03012d"


def plot_images_by_change_in_prediction(
    in_preds: np.ndarray,
    out_preds: np.ndarray,
    dataset: BaseDataset,
    dataset_name: str,
    label_name_list: List[str],
    patient_id_col: str,
    pvals: Optional[np.ndarray] = None,
    plot_highest: bool = True,
    out_dir: Optional[Path] = None,
):
    """Plot a grid of images alongside their IN vs OUT predicted probabilities.

    A qualitative companion to the statistical test: it shows *what* the records with the
    largest memorisation-induced prediction change actually look like. Image datasets
    only (MIMIC-CXR).

    Args:
        in_preds: Mean IN-model probabilities, shape (n_records, n_classes).
        out_preds: Mean OUT-model probabilities, same shape.
        dataset: Dataset the images are read from, indexed like `in_preds`.
        dataset_name: Dataset name, used in the output file name.
        label_name_list: Short class names for the per-class bar labels.
        patient_id_col: Column of `dataset.dataframe` holding the patient id.
        pvals: Optional per-record p-values, rendered as significance stars.
        plot_highest: If True, show the records with the largest change; if False, a
            random sample of records, which is the visual control for that selection.
        out_dir: Directory to write the figure to. Not saved when None.
    """
    scores = np.max(
        np.abs(in_preds - out_preds), axis=-1
    ).squeeze()  # maximum change in probability across all classes
    height_ratios = [2 if i % 2 == 0 else 1 for i in range(2 * 4)]
    fig, axes = plt.subplots(8, 5, figsize=(8, 12), height_ratios=height_ratios)
    sorted_idcs = (
        np.argsort(-scores) if plot_highest else np.random.permutation(len(scores))
    )
    i = 0
    for r in range(4):
        r = r * 2
        for c in range(5):
            idx = int(sorted_idcs[i])
            img, target = dataset.__getitem__(idx)
            img = np.array(img)
            pid = dataset.dataframe.iloc[idx][patient_id_col]
            assert (
                len(img.shape) == 3
            ), f"Expected color or gray-scale image, found: {img.shape}"
            if img.shape[-1] not in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            if img.min() < 0:
                # re-scale image to [0, 1]
                img = (img + 1) / 2
            label_str = str(target)
            sig_str = get_significance_str(pvals[idx]) if pvals is not None else ""
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].axis("off")
            axes[r, c].set_title(
                r"pid={}, $\max (\Delta)$={:.2f}, {}".format(pid, scores[idx], sig_str),
                fontsize=6,
            )
            axes[r, c].text(
                0.5,
                0.025,
                label_str,
                fontsize=5,
                transform=axes[r, c].transAxes,
                ha="center",
                bbox=dict(
                    facecolor="white", alpha=0.5, boxstyle="square", edgecolor="black"
                ),
            )
            axes[r + 1, c].bar(
                np.arange(len(out_preds[idx])) - 0.05,
                out_preds[idx],
                color="red",
                alpha=0.3,
            )
            axes[r + 1, c].bar(
                np.arange(len(out_preds[idx])) + 0.05,
                in_preds[idx],
                color="mediumblue",
                alpha=0.3,
            )
            axes[r + 1, c].legend(
                ["IN", "OUT"], loc="upper right", framealpha=0.5, fontsize=4
            )
            axes[r + 1, c].set_ylim((0, 1))
            assert len(out_preds[idx]) == len(
                label_name_list
            ), f"Expected {len(label_name_list)} labels, found {len(out_preds[idx])}"
            axes[r + 1, c].set_xticks(
                np.arange(len(out_preds[idx])),
                label_name_list,
                rotation=90,
                fontsize=5,
            )
            axes[r + 1, c].set_yticks(
                [0, 0.25, 0.5, 0.75, 1.0], [0, 0.25, 0.5, 0.75, 1.0], fontsize=5
            )
            axes[r + 1, c].spines[["right", "top"]].set_visible(False)
            i += 1

    save_str = "highest" if plot_highest else "random"
    if out_dir is not None:
        plt.savefig(
            out_dir / f"{dataset_name}_imgs_{save_str}.pdf",
            bbox_inches="tight",
            dpi=600,
        )
    plt.show()


def get_significance_str(p_val: float) -> str:
    """Star notation for a p-value: '*' <= 0.05, '**' <= 0.01, '***' <= 0.001, else 'n.s'."""
    if p_val <= 0.05:
        sig_str = "*"
        if p_val <= 0.01:
            sig_str = "**"
            if p_val <= 0.001:
                sig_str = "***"
    else:
        sig_str = "n.s"
    return sig_str


def plot_patient_trajectories(
    train_historical_dataset: BaseDataset,
    train_future_dataset: BaseDataset,
    patient_ids: pd.Series,
    patient_id_col: str,
    study_order_col: str,
    in_preds: np.ndarray,
    out_preds: np.ndarray,
    dataset_name: str,
    out_dir: Optional[Path] = None,
    pvals: Optional[List] = None,
):
    """Plot each patient's imaging history as one row: historical records, then future ones.

    Puts a memorised record back in its clinical context -- the training records that
    preceded it are shown next to it, with the IN/OUT prediction difference underneath,
    so a large change can be read against what the model had already seen of that
    patient. Image datasets only (MIMIC-CXR).

    Args:
        train_historical_dataset: Split holding the records the models trained on.
        train_future_dataset: Split holding the later records, indexed like `in_preds`.
        patient_ids: Patients to plot, one row (image strip + prediction strip) each.
        patient_id_col: Column of both dataframes holding the patient id.
        study_order_col: Column to sort a patient's records chronologically by.
        in_preds: Mean IN-model probabilities over the future records.
        out_preds: Mean OUT-model probabilities over the future records.
        dataset_name: Dataset name, used in the output file name.
        out_dir: Directory to write the figure to. Not saved when None.
        pvals: Optional per-record p-values, rendered as significance stars.
    """
    if pvals is not None:
        assert len(pvals) == len(
            train_future_dataset
        ), f"Expected {len(train_future_dataset)} pvals, found {len(pvals)}"
    fontsize = 7
    plt.rcParams.update({"font.size": fontsize})
    in_out_pred_diff = in_preds - out_preds
    assert len(in_out_pred_diff) == len(
        train_future_dataset
    ), f"Expected {len(train_future_dataset)} in_out_pred_diff, found {len(in_out_pred_diff)}"
    n_patients = len(patient_ids)
    n_images = 25
    # every second row needs to be 1.5 times higher
    height_ratios = [2 if i % 2 == 0 else 1 for i in range(2 * n_patients)]
    fig, axes = plt.subplots(
        2 * n_patients,
        n_images,
        figsize=(2 * n_images, 5 * n_patients),
        height_ratios=height_ratios,
    )
    for i, p_id in enumerate(patient_ids):
        i = i * 2
        # training image idcs
        patient_df = train_historical_dataset.dataframe[
            train_historical_dataset.dataframe[patient_id_col] == p_id
        ].sort_values([study_order_col])
        train_idcs = patient_df.index
        n_train_images = len(train_idcs)
        n_too_many = n_train_images - (n_images // 2)
        print(f"found {len(train_idcs)} training images for patient {p_id}")
        # only show the last (in chronoligcal order) training images
        if n_too_many > 0:
            train_idcs = train_idcs[n_too_many:]
        j = 0
        axes[i, 0].text(
            -0.2,
            0.3,
            f"{p_id}",
            fontsize=fontsize,
            rotation=90,
            transform=axes[i, 0].transAxes,
        )
        axes[i + 1, 0].text(
            -0.2,
            0.3,
            f"{p_id}",
            fontsize=fontsize,
            rotation=90,
            transform=axes[i + 1, 0].transAxes,
        )
        for idx in train_idcs:
            if j >= n_images // 2:
                break
            img, target = train_historical_dataset.__getitem__(idx)
            axes[i, j].imshow(img, cmap="gray")
            title_str = "Train"
            title_str += f" {train_historical_dataset.dataframe.iloc[idx][study_order_col]}"
            axes[i, j].set_title(title_str, fontsize=fontsize)
            label_str = str(target)
            axes[i, j].text(
                0.5,
                0.025,
                label_str,
                fontsize=5,
                transform=axes[i, j].transAxes,
                ha="center",
                bbox=dict(
                    facecolor="white", alpha=0.5, boxstyle="square", edgecolor="black"
                ),
            )
            axes[i + 1, j].axis("off")
            axes[i, j].axis("off")
            j += 1
        # longitudinal follow-up images
        patient_df = train_future_dataset.dataframe[
            train_future_dataset.dataframe[patient_id_col] == p_id
        ].sort_values([study_order_col])
        eval_idcs = patient_df.index
        print(f"found {len(eval_idcs)} follow-up images for patient {p_id}")
        for idx in eval_idcs:
            if j >= n_images:
                break
            img, target = train_future_dataset.__getitem__(idx)
            axes[i, j].imshow(img, cmap="gray")
            max_pred_change = np.max(in_out_pred_diff[idx])
            max_pred_change_idc = np.argmax(in_out_pred_diff[idx])
            p_val_str = get_significance_str(pvals[idx]) if pvals is not None else ""
            title_str = f"Followup {train_future_dataset.dataframe.iloc[idx][study_order_col]}:\n class idx={max_pred_change_idc}, change={max_pred_change:.3f}, {p_val_str}d"
            axes[i, j].set_title(title_str, fontsize=fontsize)
            label_str = str(target)
            axes[i, j].text(
                0.5,
                0.025,
                label_str,
                fontsize=5,
                transform=axes[i, j].transAxes,
                ha="center",
                bbox=dict(
                    facecolor="white", alpha=0.5, boxstyle="square", edgecolor="black"
                ),
            )
            axes[i + 1, j].bar(
                np.arange(len(in_out_pred_diff[idx])) - 0.05,
                out_preds[idx],
                color="red",
                alpha=0.3,
            )
            axes[i + 1, j].bar(
                np.arange(len(in_out_pred_diff[idx])) + 0.05,
                in_preds[idx],
                color="mediumblue",
                alpha=0.3,
            )
            axes[i + 1, j].legend(
                ["IN", "OUT"], loc="upper right", fontsize=fontsize, framealpha=0.6
            )
            assert len(in_out_pred_diff[idx]) == len(
                get_label_list(dataset_name)
            ), f"Expected {len(get_label_list(dataset_name))} labels, found {len(in_out_pred_diff[idx])}"
            axes[i + 1, j].set_xticks(
                np.arange(len(in_out_pred_diff[idx])),
                get_label_list(dataset_name),
                rotation=90,
                fontsize=5,
            )
            axes[i + 1, j].set_ylim((0, 1.05))
            axes[i + 1, j].spines[["right", "top"]].set_visible(False)
            axes[i, j].axis("off")
            j += 1
    # turn off axes for all empty subplots
    for ax in axes.flat:
        if not ax.has_data():
            ax.axis("off")

    if out_dir is not None:
        plt.savefig(
            out_dir / f"{dataset_name}_patient_trajectories.pdf",
            bbox_inches="tight",
            dpi=300,
        )


def aggregate_by_patient(data: np.ndarray, patient_ids: pd.Series):
    """
    Aggregates data by patient ID (mean, max, std) to obtain a single data point per patient.
    Args:
        data: numpy array of shape (n_samples,) containing the record-level data.
        patient_ids: numpy array of shape (n_samples,) containing the patient ID corresponding to each image.

    Returns:
        aggregated_data: dictionary containing the aggregated patient-level data accessible by patient ID.
    """
    assert data.shape[0] == len(
        patient_ids
    ), f"Shapes do not match: {data.shape} vs {len(patient_ids)}"
    assert isinstance(
        patient_ids, pd.Series
    ), f"Expected patient_ids to be a pandas Series, but got {type(patient_ids)}"
    n_patients = patient_ids.nunique()
    data_df = pd.DataFrame.from_dict({"data": data, "patient_id": patient_ids.values})
    aggregated = (
        data_df.groupby("patient_id")["data"]
        .agg(mean="mean", std="std", max="max", count="count")
        .reset_index()
    )
    assert (
        len(aggregated) == n_patients
    ), f"Number of patients do not match dictionary size: {len(aggregated)} vs {n_patients}"
    return aggregated


def esf_plot(
    data_arr: np.ndarray,
    xlabel: str,
    patient_ids: pd.Series = None,
    conf_level: float = 0.95,
    draw_conf: bool = False,
    aggregation_mode: str = "max",
    line_style: str = "-",
    save_path: Optional[str] = None,
    color: str = my_blue,
    alpha: float = 0.9,
    figsize: Tuple[int, int] = (2, 2),
    log_scale: bool = True,
    xticks: Optional[List[float]] = None,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """
    Plot the empirical survival function of MIA data.
    If patient_ids is provided, aggregate record-level MIA data by patient ID.
        Args:
            data_arr: np.ndarray data
            xlabel: str, x-axis label
            patient_ids: pd.Series, patient IDs
            aggregation_mode: str, how to obtain patient-level scores from record-level ones. One of either max or mean.
            line_style: str, line style for the plot
            conf_level: float, confidence level
            draw_conf: bool, whether to draw confidence intervals
            save_path: str, path to save the plot
            color: str, color of the plot
            alpha: float, alpha value for curves
            log_scale: bool, whether to use log scale for y-axis
            label: str, optional legend label
            ax: plt.Axes, optional axes to plot on
        Returns:
            None
    """
    if patient_ids is not None:
        if not aggregation_mode in ["max", "mean"]:
            raise ValueError(
                "Please provide a valid aggregation model ('max' or 'mean')."
            )
        data_df = aggregate_by_patient(data_arr, patient_ids)
        data = data_df[aggregation_mode]
        print(
            f"... esf probability change plot aggregated by patient ID: n_patients={len(data)}, n_records={len(patient_ids)}"
        )
    else:
        data = data_arr
        print(f"... esf probability change plot at record-level: n_records={len(data)}")
    ylim = 1 / len(data)
    set_plot_style = False
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        set_plot_style = True
    res = scipy.stats.ecdf(data)
    res.sf.plot(ax=ax, color=color, label=label, alpha=alpha, linestyle=line_style)
    if draw_conf:
        conf = res.sf.confidence_interval(confidence_level=conf_level)
        conf.low.plot(ax=ax, color=color, linestyle="--", lw=1, alpha=0.8)
        conf.high.plot(ax=ax, color=color, linestyle="--", lw=1, alpha=0.8)
    if set_plot_style:
        plt.ylim((ylim, 1))
        plt.xlabel(xlabel)
        plt.ylabel("1 - Cumulative Probability")
        if log_scale:
            plt.semilogy()
    ax.spines[["right", "top"]].set_visible(False)
    if xticks is not None:
        ax.set_xticks(xticks)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


def calculate_binned_memorisation_rate(
    dataframe: pd.DataFrame,
    bins: List[Tuple[int, int]],
    patient_id_col: str,
):
    """
    Calculate memorisation rate for specific time period bins.

    Args:
        dataframe: DataFrame with 'time_delta(months)' and 'rejected' columns
        bins: List of tuples (start, end) representing time bins in months
        patient_id_col: Name of the column containing patient IDs

    Returns:
        bin_labels: List of bin label strings
        mem_rates: List of memorisation rates for each bin
        counts: List of total samples in each bin
        rejected_counts: List of rejected samples in each bin
    """
    bin_labels = []
    mem_rates = []
    counts = []
    rejected_counts = []

    for start, end in bins:
        # Filter data within the bin range
        mask = (dataframe["time_delta(months)"] >= start) & (
            dataframe["time_delta(months)"] < end
        )
        bin_df = dataframe[mask]

        if len(bin_df) > 0:
            mem_rate = bin_df["rejected"].mean()
            count = len(bin_df)
            rejected_count = bin_df["rejected"].sum()
        else:
            mem_rate = 0
            count = 0
            rejected_count = 0

        # Create label
        label = f"{start}-{end}"

        bin_labels.append(label)
        mem_rates.append(mem_rate)
        counts.append(count)
        rejected_counts.append(rejected_count)

    return bin_labels, mem_rates, counts, rejected_counts


def plot_binned_memorisation_rate(
    dataframe: pd.DataFrame, patient_id_col: str, save_path: Optional[str] = None
):
    """Bar chart of memorisation rate against how long after training the record was taken.

    Bins run 0-6, 6-12, 12-18, 18-24, 24-48, 48-60, 60-120 months, plus one open bin when
    the follow-up reaches further. The question it answers is whether memorisation decays
    with the follow-up interval, so the bin count is annotated on each bar: the long bins
    hold far fewer records and their rates are correspondingly noisy.

    Args:
        dataframe: Future-record dataframe carrying 'time_delta(months)' (see
            `get_time_deltas`) and the boolean 'rejected' column from the analysis.
        patient_id_col: Column holding the patient id.
        save_path: Path to write the figure to. Not saved when None.
    """
    assert (
        "time_delta(months)" in dataframe.columns
    ), f"DataFrame must contain 'time_delta(months)' column."
    assert "rejected" in dataframe.columns, f"DataFrame must contain 'rejected' column."
    time_bins = [(0, 6), (6, 12), (12, 18), (18, 24), (24, 48), (48, 60), (60, 120)]
    max_time_delta = dataframe["time_delta(months)"].max()
    if max_time_delta > 120:
        time_bins = time_bins + [(120, max_time_delta)]
    fig, ax = plt.subplots(figsize=(3, 1.5))
    bin_labels, mem_rates, counts, rejected_counts = calculate_binned_memorisation_rate(
        dataframe=dataframe,
        bins=time_bins,
        patient_id_col=patient_id_col,
    )
    # Create bars
    bars = ax.bar(
        range(len(bin_labels)), mem_rates, color="darkblue", alpha=0.7, width=0.7
    )
    ax.set_xlabel("Follow-up Period (Months)")
    ax.set_ylabel("Memorisation Rate")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45)
    ax.spines[["right", "top"]].set_visible(False)
    # Add value labels on top of bars
    for i, (bar, rate, count, rej_count) in enumerate(
        zip(bars, mem_rates, counts, rejected_counts)
    ):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"n={rej_count}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0, max(mem_rates) * 1.15 if max(mem_rates) > 0 else 1.0)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    # Print summary statistics
    print("\n... Memorisation Rate by Follow-up Period:")
    print("-" * 60)
    for label, rate, count, rej_count in zip(
        bin_labels, mem_rates, counts, rejected_counts
    ):
        print(f"{label:>10} months: {rate*100:>6.2f}% ({rej_count}/{count} samples)")
    print("-" * 60)


def get_time_deltas(
    future_data: pd.DataFrame,
    historic_data: pd.DataFrame,
    patient_id_col: str,
    study_date_col: str,
) -> np.ndarray:
    """Follow-up interval in months for every future record.

    The interval is the number of calendar month-ends in [last historical record, future record],
    computed on dates so that it does not depend on the time of day of acquisition.
    `get_time_delta` implements the same definition for a single row.

    Args:
        future_data: Future-record frame; one interval is returned per row, in row order.
        historic_data: Historic-record frame the most recent record per patient comes from.
        patient_id_col: Column holding the patient id in both frames.
        study_date_col: Column holding the acquisition timestamp in both frames. Must already
            be datetime in both -- the caller converts.

    Returns:
        (len(future_data),) int array of months since the patient's last historic record.
    """
    for label, frame in (("future", future_data), ("historic", historic_data)):
        if not pd.api.types.is_datetime64_any_dtype(frame[study_date_col]):
            raise TypeError(
                f"{label} '{study_date_col}' must be datetime, got "
                f"{frame[study_date_col].dtype}; convert before calling"
            )
    latest_historic = historic_data.groupby(patient_id_col)[study_date_col].max()
    start = future_data[patient_id_col].map(latest_historic)
    end = future_data[study_date_col]
    missing = start.isna()
    if missing.any():
        # the row-wise version raised IndexError from `.iloc[0]` here, which said nothing
        # about which patient was at fault
        raise ValueError(
            f"{int(missing.sum())} future records belong to patients with no historic record "
            f"(e.g. {future_data.loc[missing, patient_id_col].iloc[0]!r}); the follow-up "
            f"interval is undefined for them"
        )
    before = end < start
    if before.any():
        raise AssertionError(
            f"Future data must be later than historic data: {int(before.sum())} records "
            f"predate their patient's most recent historic record, first at "
            f"historic date {start[before].iloc[0]}, followup date {end[before].iloc[0]}"
        )
    s = start.dt.normalize().dt
    e = end.dt.normalize().dt
    # whole months between the two dates, plus the end month itself when the follow-up runs
    # to that month's last day -- i.e. the count of month-ends in the closed interval
    whole_months = (e.year * 12 + e.month) - (s.year * 12 + s.month)
    ends_on_a_month_end = e.day == e.days_in_month
    return (whole_months + ends_on_a_month_end.astype(int)).to_numpy()


def get_time_delta(
    future_data_row: pd.Series,
    historic_data: pd.DataFrame,
    patient_id_col: str,
    study_date_col: str,
    verbose: bool = False,
):
    """Months between a future record and that patient's most recent historic record.

    Single-row version of `get_time_deltas`, which is the one used by the analysis. Both
    timestamps are reduced to their date before counting month-ends.

    Args:
        future_data_row (pd.Series): A row from the future data DataFrame.
        historic_data (pd.DataFrame): The historic data DataFrame for the patient.
        patient_id_col (str): The name of the column containing patient IDs.
        study_date_col (str): The name of the column containing study dates.
        verbose (bool): Whether to print verbose output.
    Returns:
        int: Calendar month-ends between the most recent historic record and this record.
    """
    p_id = future_data_row[patient_id_col]
    followup_date = future_data_row[study_date_col]
    historic_patient_data = historic_data[
        historic_data[patient_id_col] == p_id
    ].sort_values(
        by=study_date_col, ascending=False
    )  # most recent historic record first
    if verbose:
        print(
            f"calculating time delta between historic date: {historic_patient_data.iloc[0][study_date_col]} and future date: {future_data_row[study_date_col]}"
        )

    assert isinstance(followup_date, pd.Timestamp) and isinstance(
        historic_patient_data.iloc[0][study_date_col], pd.Timestamp
    ), f"date columns must be of type pd.Timestamp, found {type(followup_date)} and {type(historic_patient_data.iloc[0][study_date_col])}"
    # check followup date is after historic date
    assert (
        followup_date >= historic_patient_data.iloc[0][study_date_col]
    ), f"Future data must be later than historic data: historic date: {historic_patient_data.iloc[0][study_date_col]}, followup date: {followup_date}"
    time_delta_months = len(
        pd.date_range(
            start=historic_patient_data.iloc[0][study_date_col].normalize(),
            end=followup_date.normalize(),
            freq="ME",
        )
    )
    if verbose:
        print(f"Time delta: {time_delta_months} months")
    return time_delta_months


def bar_plot(
    labels: List[str],
    values: List[float],
    ylabel: str = "Value",
    xlabel: str = "",
    save_path: Optional[Path] = None,
    bar_width: float = 0.15,
    inner_padding: float = 0.75,
    color_positive: str = "red",
    color_negative: str = "green",
    color_zero: str = "gray",
    alpha: float = 0.4,
    add_value_labels: bool = True,
    add_zero_line: bool = True,
    fontsize: int = 6,
    error_bars: Optional[List[float]] = None,
):
    """
    Create a bar plot with fixed bar widths and proper spacing, inspired by composition_comparison_plot.

    Args:
        labels: List of category labels for x-axis
        values: List of values to plot (can be positive or negative)
        ylabel: Label for y-axis
        xlabel: Label for x-axis
        save_path: Optional path to save the plot
        bar_width: Width of each bar in inches
        inner_padding: Padding on left and right of the plot
        color_positive: Color for positive values
        color_negative: Color for negative values
        color_zero: Color for zero values
        alpha: Transparency of bars
        add_value_labels: Whether to add value labels on top of bars
        add_zero_line: Whether to add a horizontal line at y=0
        fontsize: Font size
        error_bars: Optional list of error bar values (e.g., 95% CI half-widths)

    Returns:
        None
    """
    from mpl_toolkits.axes_grid1 import Divider, Size  # type: ignore

    assert len(labels) == len(
        values
    ), f"Length mismatch: {len(labels)} labels vs {len(values)} values"

    if error_bars is not None:
        assert len(error_bars) == len(
            values
        ), f"Length mismatch: {len(error_bars)} error bars vs {len(values)} values"

    # Calculate colors based on values
    colors = [
        color_positive if v > 0 else color_negative if v < 0 else color_zero
        for v in values
    ]

    # Calculate figure dimensions
    inner_width = bar_width * len(labels)
    inner_height = 1.0
    left_margin = 0.4
    right_margin = 0.15
    bottom_margin = 0.6
    top_margin = 0.25

    total_width = inner_width + left_margin + right_margin
    total_height = inner_height + bottom_margin + top_margin

    # Create figure with fixed dimensions
    fig = plt.figure(figsize=(total_width, total_height), layout=None)

    # Create divider for precise axes placement
    h = [Size.Fixed(left_margin), Size.Fixed(inner_width), Size.Fixed(right_margin)]
    v = [Size.Fixed(bottom_margin), Size.Fixed(inner_height), Size.Fixed(top_margin)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1)
    )

    # Add horizontal line at y=0 if requested
    if add_zero_line:
        ax.hlines(
            y=0, xmin=-100, xmax=100, ls="--", color="black", alpha=0.5, zorder=-1
        )

    # Create bars with optional error bars
    bars = ax.bar(
        labels,
        values,
        yerr=error_bars,
        color=colors,
        linewidth=1,
        edgecolor="black",
        width=4 * bar_width,
        align="center",
        alpha=alpha,
        capsize=2 if error_bars is not None else 0,
        error_kw={"elinewidth": 1, "capthick": 1} if error_bars is not None else None,
    )

    # Get bar positions for setting x-limits
    bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]

    # Add value labels on bars if requested
    if add_value_labels:
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            offset = 1
            # Adjust offset if error bars are present
            if error_bars is not None:
                o = error_bars[i] + 2 if value >= 0 else -(error_bars[i] + 2)
                offset = int(o)
            else:
                offset = offset if value >= 0 else -offset
            if value <= 5:
                # put text label above bar
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + offset,
                    f"{int(value)}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=fontsize - 2,
                )
            else:
                # put text label just above zero
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    0.5,
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize - 2,
                )

    # Set labels and styling
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(
        (
            bar_positions[0] - (bar_width / 2) - inner_padding,
            bar_positions[-1] + (bar_width / 2) + inner_padding,
        )
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=90)
    ax.spines[["right", "top"]].set_visible(False)

    # Save if path provided
    if save_path:
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=".*layoutgrids.*"
            )
            fig.savefig(save_path, dpi=300)
    plt.close()


def compute_decision_thresholds(
    preds: np.ndarray, targets: np.ndarray, verbose: bool = False
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute optimal thresholds for each class using Youden's J statistic.
    Args:
        preds (np.ndarray): model predictions of shape (n_records, n_classes)
        targets (np.ndarray): ground truth targets of shape (n_records, n_classes)
    Returns:
        List[float]: optimal thresholds per class
    """
    assert (
        preds.shape == targets.shape
    ), f"Shapes of preds and targets must match, got {preds.shape} and {targets.shape}"
    assert (
        preds.shape == targets.shape
    ), f"Shapes of preds and targets must match, got {preds.shape} and {targets.shape}"
    thresholds, fprs, tprs, aucs = [], [], [], []
    n_classes = targets.shape[1]
    for i in tqdm(range(n_classes), desc="Computing optimal thresholds", leave=False):
        fpr, tpr, thr = skm.roc_curve(targets[:, i], preds[:, i])
        auc = skm.auc(fpr, tpr)
        youden_j = tpr - fpr
        threshold_idx = np.argmax(youden_j)
        threshold = thr[threshold_idx]
        (
            print(
                f"Class {i}: optimal threshold={threshold:.2f} (FPR={fpr[threshold_idx]:.2f}, TPR={tpr[threshold_idx]:.2f}, AUC={auc:.2f})"
            )
            if verbose
            else None
        )
        thresholds.append(threshold)
        fprs.append(fpr[threshold_idx])
        tprs.append(tpr[threshold_idx])
        aucs.append(auc)
    assert (
        len(thresholds) == n_classes
    ), f"Number of thresholds ({len(thresholds)}) must match number of classes ({n_classes})"
    return thresholds, fprs, tprs, aucs


def auc_comparison_unseen_disease(
    prediction_partioner: RandomSubsetPredictionPartitioner,
    targets: np.ndarray,
    class_names: List[str],
    dataframe: pd.DataFrame,
    verbose: bool = True,
    max_cases: int = 100_000,
) -> Tuple[
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, int],
]:
    """Per-disease AUROC of IN models against OUT models, on de novo cases only.

    Restricted to positive cases whose disease was *not* present in that patient's
    historical (training) records, so the models cannot have seen the label for that
    patient before, matched against an equal number of randomly drawn negatives. Every
    model contributes one AUROC, so the two lists are a paired sample over models rather
    than a single number.

    `rop_comparison_by_disease` is the threshold-level version of the same comparison and
    is what the published figures use; this one is threshold-free.

    Args:
        prediction_partioner: Partitioner over the future-record predictions.
        targets: Ground-truth labels of the future records, (n_records, n_classes).
        class_names: Class names, aligned with `targets` columns and matched against the
            dataframe's 'unseen_diseases' column.
        dataframe: Future-record dataframe, must carry 'unseen_diseases'.
        verbose: Print mean +/- sd per class.
        max_cases: Cap on cases per class (positives are truncated to half of it), which
            bounds runtime on the large datasets.

    Returns:
        (auc_in, auc_out, n_cases): three dicts keyed by class name, holding the per-model
        AUROCs of the IN models, of the OUT models, and the number of cases used.
    """
    assert (
        "unseen_diseases" in dataframe.columns
    ), f"le_dataframe must contain 'unseen_diseases' column"
    n_classes = targets.shape[1]
    assert n_classes == len(
        class_names
    ), f"Number of classes {n_classes} must match number of class names {len(class_names)}"
    auc_in, auc_out, n_cases_dict = {}, {}, {}
    for class_idx in range(n_classes):
        # for every disease class, we pick cases of unseen disease (i.e. positive cases of disease where this disease was never present in the patient's historical data used for model training)
        class_name = class_names[class_idx]
        disease_i_unseen = dataframe.unseen_diseases.str.contains(
            class_name, na=False
        ).values
        pos_cases = np.nonzero(disease_i_unseen)[0]
        if len(pos_cases) > max_cases // 2:
            print(
                f"WARNING: class {class_name} has {len(pos_cases)} positive cases with unseen disease, which exceeds the maximum of {max_cases}. Randomly sampling {max_cases} cases for AUC comparison."
            )
            pos_cases = pos_cases[: max_cases // 2]
        n_pos = len(pos_cases)
        if n_pos > 0:
            # pick a matching number of negative cases at random from the remaining data
            neg_cases = np.random.choice(
                np.nonzero(~disease_i_unseen)[0], size=n_pos, replace=False
            )
            cases_considered = np.concatenate([pos_cases, neg_cases])
            print(
                f"... performing AUC comparison with N_pos={n_pos} and N_neg={len(neg_cases)} of {class_name}. Total cases considered: {len(cases_considered)}"
            )
            p_in, p_out = prediction_partioner.get_many(
                cases_considered, reduce=partial(np.take, indices=class_idx, axis=-1)
            )
            assert (
                p_in.shape == p_out.shape
            ), f"Shape mismatch between p(IN) {p_in.shape} and p(OUT) {p_out.shape}"
            aucs_in = [
                skm.roc_auc_score(
                    y_true=targets[cases_considered, class_idx],
                    y_score=p_in[i],
                )
                for i in range(p_in.shape[0])
            ]
            aucs_out = [
                skm.roc_auc_score(
                    y_true=targets[cases_considered, class_idx],
                    y_score=p_out[i],
                )
                for i in range(p_out.shape[0])
            ]
            if verbose:
                print(
                    f"\tAUC_in: {np.mean(aucs_in):.4f} ± {np.std(aucs_in):.4f}, AUC_out: {np.mean(aucs_out):.4f} ± {np.std(aucs_out):.4f}"
                )
            auc_in[class_name] = aucs_in
            auc_out[class_name] = aucs_out
            n_cases_dict[class_name] = n_pos + len(neg_cases)
        else:
            print(f"WARNING: no cases with unseen disease for class {class_name}")
            auc_in[class_name] = [np.nan]
            auc_out[class_name] = [np.nan]
            n_cases_dict[class_name] = 0
    return auc_in, auc_out, n_cases_dict


class PermutationTestResult(NamedTuple):
    """Outcome of `run_label_permutation_test`."""

    p_value: float
    statistic: float
    per_record_delta: np.ndarray
    valid: np.ndarray
    rate_in: float
    rate_out: float
    null_lower: np.ndarray
    null_upper: np.ndarray
    n_permutations: int


def run_label_permutation_test(
    correct: np.ndarray,
    inclusion: np.ndarray,
    n_permutations: int = 100_000,
    rng: Optional[np.random.Generator] = None,
    n_null_tail: int = 2_000,
) -> "PermutationTestResult":
    """Permutation test for an IN/OUT gap, reassigning run labels to inclusion masks.

    Tests the sharp null that including a case's patient in a run's training subset does not
    change that run's predictions on the case:

        H0:  W  is independent of  M

    with `W[r, j]` the correctness of run r on case j and `M[r, j]` whether run r trained on
    case j's patient. Under H0 the test below is exact, because `generate_masks`
    (src/training/training.py) assigns each patient a uniformly random subset of exactly
    `subset_ratio * n_runs` of the runs, independently across patients, so

    * the mask block over any set of patients is row-exchangeable -- permuting run labels
      leaves its distribution unchanged, and
    * that block is independent of the block over all other patients, which is what makes the
      runs' predictions (a function of the rest of their training data under H0) independent
      of the inclusion pattern being permuted.

    Conditional on `W`, every reassignment of run labels to mask rows is therefore equally
    likely under H0. Patient clustering needs no separate handling: a run's whole mask row
    moves as a unit, so all of a patient's records flip together, which is exactly the
    dependence a per-case label swap would have to model explicitly.

    The statistic is the per-record IN-minus-OUT difference in correctness rate, averaged over
    records -- the gap between the average single model that trained on a patient and the
    average one that did not:

        delta_j = (1/n_in[j])  * sum_r  M[r,j]  * W[r,j]
                - (1/n_out[j]) * sum_r (1-M[r,j]) * W[r,j]
        T       = mean_j delta_j

    Every run is thresholded on its own and contributes, so with ~100 runs on each side
    `delta_j` lands on a fine grid instead of collapsing to {-1, 0, +1} the way a difference
    between two ensemble decisions does.

    Recomputing T for each permutation would cost `n_permutations * n_runs * n_cases`. It
    reduces to one matrix product instead. `S_j = sum_r W[r,j]` is permutation-invariant, so
    with `A_j = 1/n_in[j] + 1/n_out[j]`:

        T(pi) = sum_r G[pi(r), r] - C
        G     = M @ V.T          V[r,j] = A_j * W[r,j] / n_cases
        C     = mean_j( S_j / n_out[j] )

    G is (n_runs, n_runs), built once; each permutation is then `n_runs` gathers into it, so
    the number of permutations is essentially free. Column sums of M are preserved by a row
    permutation, hence `n_in[j]` is invariant and the usable records are the same under every
    permutation -- they can be dropped once, up front.

    Args:
        correct: Per-run correctness, (n_runs, n_cases). Boolean or 0/1.
        inclusion: Per-run inclusion mask, (n_runs, n_cases). True where the run trained on
            that case's patient.
        n_permutations: Number of random run-label permutations drawn for the null.
        rng: Generator for the permutation draws.
        n_null_tail: How many of the most extreme null draws to keep from each tail. The
            confidence interval is the inversion of this test (`permutation_interval`), and
            inverting it at level 1 - a only ever reads the `ceil(a/2 * (n_permutations+1)) - 1`
            -th order statistic from each end, so keeping the tails is enough and avoids
            carrying all `n_permutations` draws through to the figure. The default covers any
            level down to a = 2 * (n_null_tail + 1) / (n_permutations + 1); `permutation_interval`
            raises rather than silently truncating if a caller asks for more.

    Returns:
        PermutationTestResult with the two-sided p-value, the observed statistic T, the
        per-record differences behind it, the mask of
        which input columns were usable, the mean IN and OUT rates the difference is between,
        and the two tails of the null distribution that `permutation_interval` inverts.
        Records where every run is IN, or every run is OUT, carry no information and
        are dropped -- impossible under the real design, but the null-control partitioner can
        produce them.
    """
    correct = np.asarray(correct)
    inclusion = np.asarray(inclusion)
    if correct.ndim != 2:
        raise ValueError(f"correct must be 2D (n_runs, n_cases), got {correct.shape}")
    if correct.shape != inclusion.shape:
        raise ValueError(
            f"correct has shape {correct.shape} but inclusion has {inclusion.shape}"
        )
    n_runs, n_cases = correct.shape
    empty = np.array([], dtype=float)
    if n_runs == 0 or n_cases == 0:
        return PermutationTestResult(
            1.0, 0.0, empty, np.zeros(n_cases, dtype=bool), float("nan"), float("nan"),
            empty, empty, n_permutations,
        )
    w_all = correct.astype(np.float64)
    m_all = inclusion.astype(np.float64)
    n_in_all = m_all.sum(axis=0)
    # a record with no IN run (or no OUT run) has no difference to contribute; `n_in` is
    # invariant under a row permutation, so this selection holds for the whole null
    valid = (n_in_all > 0) & (n_in_all < n_runs)
    w = w_all[:, valid]
    m = m_all[:, valid]
    n_valid = int(w.shape[1])
    if n_valid == 0:
        return PermutationTestResult(
            1.0, 0.0, empty, valid, float("nan"), float("nan"), empty, empty, n_permutations
        )
    n_in = n_in_all[valid]
    n_out = n_runs - n_in
    totals = w.sum(axis=0)
    in_rate = (m * w).sum(axis=0) / n_in
    out_rate = (totals - (m * w).sum(axis=0)) / n_out
    delta = in_rate - out_rate
    # the (n_runs, n_runs) reduction described above
    weights = (1.0 / n_in + 1.0 / n_out) / n_valid
    g = m @ (w * weights).T
    offset = float((totals / n_out).sum() / n_valid)
    # taken from the same product as the null so that algebraically equal permutations compare
    # exactly equal -- the statistic is discrete and ties decide the p-value
    t_obs = float(np.trace(g) - offset)
    assert np.isclose(
        t_obs, float(delta.mean()), rtol=1e-9, atol=1e-9
    ), f"permutation reduction disagrees with the direct statistic: {t_obs} vs {delta.mean()}"
    rng = np.random.default_rng() if rng is None else rng
    # a tolerance is needed for the same reason: without it, floating-point noise splits ties
    # the wrong way and the p-value comes out anti-conservative
    atol = 1e-12 * max(1.0, abs(t_obs))
    runs = np.arange(n_runs)
    n_ge = n_le = 0
    # the extreme draws of the null, kept so the interval can invert this test; only the
    # `n_null_tail` smallest and largest can ever be the endpoint, so the middle is discarded
    # chunk by chunk instead of being accumulated
    n_tail = max(1, min(int(n_null_tail), int(n_permutations)))
    kept = np.empty(0, dtype=np.float64)
    # drawn in chunks: the (n_permutations, n_runs) index matrix is large on its own
    chunk = max(1, int(2_000_000 // n_runs))
    drawn = 0
    while drawn < n_permutations:
        size = min(chunk, n_permutations - drawn)
        perms = rng.permuted(np.broadcast_to(runs, (size, n_runs)).copy(), axis=1)
        null = g[perms, runs].sum(axis=1) - offset
        n_ge += int(np.count_nonzero(null >= t_obs - atol))
        n_le += int(np.count_nonzero(null <= t_obs + atol))
        # a draw among the global `n_tail` smallest is among the running `n_tail` smallest at
        # every step, so dropping the middle here cannot lose an endpoint
        kept = np.concatenate([kept, null])
        if kept.size > 2 * n_tail:
            kept = np.concatenate(
                [
                    np.partition(kept, n_tail - 1)[:n_tail],
                    np.partition(kept, kept.size - n_tail)[kept.size - n_tail :],
                ]
            )
        drawn += size
    kept.sort()
    lower_tail = kept[:n_tail].copy()
    upper_tail = kept[-n_tail:].copy()
    # +1 in numerator and denominator: never report p = 0 off a finite number of draws
    p = min(1.0, 2.0 * min(n_ge + 1, n_le + 1) / (n_permutations + 1))
    return PermutationTestResult(
        p, t_obs, delta, valid, float(in_rate.mean()), float(out_rate.mean()),
        lower_tail, upper_tail, int(n_permutations),
    )


def permutation_interval(
    statistic: float,
    null_lower: np.ndarray,
    null_upper: np.ndarray,
    n_permutations: int,
    alpha: float,
) -> Tuple[float, float]:
    """Confidence interval for one comparison by inverting its run-label permutation test.

    The interval is the set of shifts the test does not reject: under the additive model in
    which the true effect is `d`, `T_obs - d` is a draw from the permutation null, so

        CI = { d : the two-sided permutation p-value of  T_obs - d  is >= alpha }

    `run_label_permutation_test` forms that p-value as `2 * min(p_plus, p_minus)` with
    `p_plus = (#{null >= T_obs - d} + 1) / (B + 1)` and `p_minus` its mirror, so the interval
    is accepted exactly when both tails are >= alpha/2. Writing
    `c = ceil(alpha/2 * (B + 1)) - 1` for the number of null draws that have to sit at or
    beyond `T_obs - d`, that condition is `null_(c) <= T_obs - d <= null_(B-c+1)` and

        CI = [ T_obs - (c-th largest null),  T_obs - (c-th smallest null) ]

    By construction, zero falls outside this interval if and only if the p-value is below
    alpha. It is a randomisation interval, conditional on the observed cases; it does not
    include case-sampling uncertainty.

    Args:
        statistic: `T_obs` for this comparison.
        null_lower: The smallest null draws, ascending (`PermutationTestResult.null_lower`).
        null_upper: The largest null draws, ascending (`PermutationTestResult.null_upper`).
        n_permutations: `B`, the number of draws the null was built from.
        alpha: Per-comparison error rate, i.e. already divided by the family size.

    Returns:
        (lower, upper). Both are +/-inf when `alpha <= 2 / (B + 1)`, the level at which the
        test cannot reject whatever the data say and the interval is therefore unbounded.
    """
    null_lower = np.asarray(null_lower, dtype=float)
    null_upper = np.asarray(null_upper, dtype=float)
    if null_lower.size == 0 or null_upper.size == 0:
        return (-math.inf, math.inf)
    c = math.ceil(alpha / 2 * (n_permutations + 1)) - 1
    if c < 1:
        # the smallest attainable p-value, 2/(B+1), is already >= alpha: nothing is rejected
        return (-math.inf, math.inf)
    if c > null_lower.size or c > null_upper.size:
        raise ValueError(
            f"inverting the test at alpha={alpha:g} needs the {c}-th order statistic of the "
            f"null, but only {min(null_lower.size, null_upper.size)} draws were kept from each "
            f"tail; re-run the analysis with a larger n_null_tail"
        )
    return (float(statistic - null_upper[-c]), float(statistic - null_lower[c - 1]))


def bonferroni_figure_correction(
    raw_pvals: np.ndarray,
    statistics: np.ndarray,
    null_lower: np.ndarray,
    null_upper: np.ndarray,
    n_permutations: np.ndarray,
    n_classes: int,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Bonferroni-correct one metric's p-values over a whole figure, with matching intervals.

    The family is the published figure, not one dataset: every class it draws contributes a
    sensitivity *and* a specificity comparison, so m = 2 * n_classes. That size is unknown to
    `rop_comparison_by_disease`, which sees one dataset at a time and therefore reports raw
    p-values and raw null tails; the correction belongs wherever the figure is assembled.

    Intervals come from inverting each comparison's own permutation test at the per-comparison
    level alpha/m (`permutation_interval`), the same level the corrected p-values are read
    against. Deriving both from one m and one null makes them agree exactly rather than
    approximately: an interval clear of zero *is* a corrected p below alpha, because the
    interval is defined as the shifts that p-value does not reject.

    Args:
        raw_pvals: Uncorrected p-values for one metric, (n_classes,).
        statistics: Observed `T_obs` per comparison, (n_classes,), row-aligned with `raw_pvals`.
        null_lower: Smallest null draws per comparison, ascending, (n_classes, n_tail).
        null_upper: Largest null draws per comparison, ascending, (n_classes, n_tail).
        n_permutations: Draws behind each comparison's null, (n_classes,).
        n_classes: Classes drawn in the figure; the family is twice this.
        alpha: Family-wise error rate.

    Returns:
        (adjusted_pvals, ci_lower, ci_upper, ci_level).
    """
    raw_pvals = np.asarray(raw_pvals, dtype=float)
    statistics = np.asarray(statistics, dtype=float)
    null_lower = np.asarray(null_lower, dtype=float)
    null_upper = np.asarray(null_upper, dtype=float)
    n_permutations = np.asarray(n_permutations)
    for name, arr in (
        ("p-values", raw_pvals),
        ("statistics", statistics),
        ("lower null tails", null_lower),
        ("upper null tails", null_upper),
        ("permutation counts", n_permutations),
    ):
        if arr.shape[0] != n_classes:
            raise ValueError(
                f"{name} have {arr.shape[0]} rows but {n_classes} classes are drawn"
            )
    m_comparisons = 2 * n_classes
    adjusted = np.minimum(raw_pvals * m_comparisons, 1.0)
    per_comparison_alpha = alpha / m_comparisons
    ci_level = 1.0 - per_comparison_alpha
    bounds = [
        permutation_interval(
            statistics[i], null_lower[i], null_upper[i], int(n_permutations[i]),
            per_comparison_alpha,
        )
        for i in range(n_classes)
    ]
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])
    return adjusted, lower, upper, ci_level


def rop_comparison_by_disease(
    prediction_partioner: RandomSubsetPredictionPartitioner,
    targets: np.ndarray,
    thresholds: List[float],
    class_names: List[str],
    dataframe: pd.DataFrame,
    memorisation_status:np.ndarray,
    patient_id_col: str,
    verbose: bool = True,
    n_permutations: int = 100_000,
    denovo_disease: Optional[str] = None,
    seed: int = 0,
) -> Tuple[Dict, Dict, Dict[str, np.ndarray]]:
    """Per-disease sensitivity and specificity of IN models against OUT models.

    Translates the abstract IN/OUT probability gap into the clinical quantities it moves.
    For each class, every model is binarised on its own at that class's fixed operating point
    (`thresholds`, from `compute_decision_thresholds` on the held-out test set), giving a
    per-model correctness matrix over the cases. The reported effect is the per-record
    difference between the correctness rate among the models that trained on that record's
    patient and the rate among those that did not, averaged over records -- the gap between
    the average single model that saw a patient and the average one that did not, which is
    what deployment looks like.

    The p-value comes from `run_label_permutation_test`, which permutes run labels against the
    inclusion mask. That is the experiment's own randomisation, so the test is exact, it uses
    all `n_runs` models rather than collapsing them into two ensembles, and it accounts for
    patient clustering automatically -- a run's whole mask row moves as a unit, so a patient's
    records flip together. Sensitivity is the comparison over the positive cases, specificity
    over the negative ones. The tails of each permutation null are returned alongside, for the
    plot to invert into the interval it draws.

    Only records the analysis flagged as memorised enter the comparison
    (`memorisation_status`) -- the question is how much the operating point moves *where*
    memorisation was detected, not on average.

    Two case definitions, selected by `denovo_disease`:

    * True: de novo disease -- the class was absent from that patient's historical
      records ('unseen_diseases'). Positive cases only, each matched to an age- and
      sex-matched negative control drawn from the same split.
    * False: 'seen' disease -- the class was already present historically
      ('seen_diseases'), taking that mask's positives and negatives as they fall.

    **No multiplicity correction is applied here** -- only raw p-values are returned. The
    family is the published figure, which pools every class of every dataset it displays, so
    its size is not knowable from a single call. `rop_comparison_plot.py` applies a Bonferroni
    correction over the whole figure (m = 2 x the number of classes shown, each contributing a
    sensitivity and a specificity comparison) and inverts each null at the matching
    per-comparison level alpha/m, so the intervals and the corrected p-values agree exactly.

    Args:
        prediction_partioner: Partitioner over the future-record predictions.
        targets: Ground-truth labels of the future records, (n_records, n_classes).
        thresholds: Decision threshold per class.
        class_names: Class names, aligned with `targets` columns; also matched against
            the dataframe's 'unseen_diseases' / 'seen_diseases' strings.
        dataframe: Future-record dataframe; needs 'unseen_diseases'/'seen_diseases' plus
            'age' and 'sex' for the matched-control sampling.
        memorisation_status: Boolean mask of records the test rejected.
        patient_id_col: Column of `dataframe` holding the patient ID, used as the cluster
            label for the permutation and to count the patients each comparison spans.
        verbose: Print mean +/- sd per class.
        n_permutations: Run-label permutations drawn per comparison, which floors the
            attainable raw p-value at 2/(n_permutations + 1).
        denovo_disease: Select the de novo case definition (see above).
        seed: Seed for the permutation draws.

    Returns:
        (sensitivity_results, specificity_results, null_tails): the first two are dicts of
        column -> list, ready for `pd.DataFrame`, holding per class the difference, the two
        rates it is between, the case counts, the size of the test behind it and the raw
        p-value. `null_tails` holds, for each of 'sensitivity' and 'specificity' and row-aligned
        with the corresponding frame, the arrays the plot needs to form intervals at the
        corrected level: '<metric>_null_lower' / '<metric>_null_upper', the
        (n_comparisons, n_null_tail) float64 tails of each permutation null, which
        `permutation_interval` inverts into the interval actually drawn, alongside
        '<metric>_n_permutations'.
        Classes without positive cases are skipped with a warning, and specificity is
        skipped for classes without negative cases.
    """
    assert (
        "unseen_diseases" in dataframe.columns
    ), f"dataframe must contain 'unseen_diseases' column"
    n_classes = targets.shape[1]
    assert (
        n_classes == len(thresholds) == len(class_names)
    ), f"Number of classes {n_classes} must match number of thresholds {len(thresholds)} and class names {len(class_names)}"
    sensitivity_results: Dict[str, List[Union[str, float, int]]] = {
        "Class": [],
        "sample_mean_diff": [],
        "rate_in": [],
        "rate_out": [],
        "n_pos_cases": [],
        "n_neg_cases": [],
        "n_records_tested": [],
        "n_patients": [],
        "n_runs": [],
        "n_permutations": [],
        "pval_raw": [],
    }
    specificity_results: Dict[str, List[Union[str, float, int]]] = {
        "Class": [],
        "sample_mean_diff": [],
        "rate_in": [],
        "rate_out": [],
        "n_pos_cases": [],
        "n_neg_cases": [],
        "n_records_tested": [],
        "n_patients": [],
        "n_runs": [],
        "n_permutations": [],
        "pval_raw": [],
    }
    dataframe['age_binned'] = pd.cut(dataframe['age'], bins=range(0, 105, 5))
    # the tails of each comparison's permutation null, which the figure inverts into the
    # interval it draws; kept row-aligned with the frames above
    sens_nulls: List[PermutationTestResult] = []
    spec_nulls: List[PermutationTestResult] = []
    rng = np.random.default_rng(seed)
    for class_idx in range(n_classes):
        t = float(thresholds[class_idx])
        class_name = class_names[class_idx]
        # consider only denovo disease cases where memorisation was detected (positive cases only)
        if denovo_disease:
            disease_i_unseen = dataframe.unseen_diseases.str.contains(
            class_name, na=False
            ).values
            mask = disease_i_unseen & memorisation_status
            pos_cases = np.nonzero(mask)[0]
            # sample age- and sex-matched control cases
            neg_cases = set(np.nonzero(targets[:, class_idx] == 0)[0])
            matched_controls = []
            for pcase in pos_cases:
                case_age, case_sex = dataframe.iloc[pcase][["age_binned", "sex"]]
                pool = dataframe[
                    (dataframe.age_binned == case_age) & (dataframe.sex == case_sex)
                ]
                pool = pool[pool.index.isin(neg_cases)]
                if not pool.empty:
                    matched_cases = np.random.choice(pool.index.values)
                    matched_controls.append(matched_cases)
            cases_considered = np.concatenate([pos_cases, np.array(matched_controls)])
            n_pos, n_neg = len(pos_cases), len(matched_controls)
        # consider only cases of 'seen' disease where memorisation was detected (positive and negative cases)
        else:
            disease_i_seen = dataframe.seen_diseases.str.contains(
            class_name, na=False
            ).values
            mask = disease_i_seen & memorisation_status
            cases_considered = np.nonzero(mask)[0]
            # int(): targets are float16, whose integer spacing is 2 above 2048, so summing
            # them silently rounds case counts of a few thousand to the nearest even number
            n_pos = int(np.sum(targets[cases_considered, class_idx]))
            n_neg = int(cases_considered.shape[0] - n_pos)
            # for the outcome_critical class in MIMIC-IV-ED there are no positive cases where the patient already had a positive case in their historical data
            # this is because outcome_critical is "compositely defined as either inpatient mortality or transfer to an ICU within 12 hours."
            # for this specific setting, we sample age- and sex- matched positive cases to enable an ROP comparison
            if n_pos == 0 and class_name == "outcome_critical":
                # pool of positive cases to sample age- and sex-matched positives from
                pos_pool = set(np.nonzero(targets[:, class_idx] == 1)[0])
                matched_positives = []
                for ncase in cases_considered:
                    case_age, case_sex = dataframe.iloc[ncase][["age_binned", "sex"]]
                    pool = dataframe[
                        (dataframe.age_binned == case_age) & (dataframe.sex == case_sex)
                    ]
                    pool = pool[pool.index.isin(pos_pool)]
                    if not pool.empty:
                        matched_case = np.random.choice(pool.index.values)
                        matched_positives.append(matched_case)
                cases_considered = np.concatenate(
                    [cases_considered, np.array(matched_positives)]
                )
                n_pos = int(np.sum(targets[cases_considered, class_idx]))
                n_neg = int(cases_considered.shape[0] - n_pos)
        if n_pos > 0:
            print(
                f"... performing sensitivity comparison with N_pos={n_pos} and N_neg={n_neg} cases of {class_name}"
            )
            # every model's prediction on every case, unpartitioned: the permutation test
            # permutes the inclusion mask against this matrix, so the two are fetched apart
            # and no model may be dropped
            preds = prediction_partioner.get_many_unpartioned(
                cases_considered, reduce=partial(np.take, indices=class_idx, axis=-1)
            )
            inclusion = prediction_partioner.get_inclusion(cases_considered, rng=rng)
            assert (
                preds.shape == inclusion.shape
            ), f"Shape mismatch between predictions {preds.shape} and inclusion mask {inclusion.shape}"
            n_runs = preds.shape[0]
            y_true = targets[cases_considered, class_idx].astype(bool)
            # each model is thresholded on its own at the class's fixed operating point
            flagged = preds >= t
            # patient is the cluster: inclusion was randomised per patient and expanded to
            # records, so a patient's records share one IN/OUT draw
            clusters = dataframe.iloc[cases_considered][patient_id_col].to_numpy()
            # Sensitivity over the positive cases: there "correct" means the case was caught,
            # so a group of models' rate of correct decisions is exactly its sensitivity.
            sens = run_label_permutation_test(
                flagged[:, y_true],
                inclusion[:, y_true],
                n_permutations=n_permutations,
                rng=rng,
            )
            sens_clusters = clusters[y_true][sens.valid]
            sens_nulls.append(sens)
            n_sens_patients = int(pd.unique(sens_clusters).size)
            print(
                f"\tSensitivity (per model): IN={sens.rate_in:.4f}, OUT={sens.rate_out:.4f}, "
                f"diff={sens.statistic:+.4f} over {sens.valid.sum()} records "
                f"({n_sens_patients} patients) and {n_runs} models, p(raw)={sens.p_value:.3g}"
            )
            sensitivity_results["Class"].append(class_name)
            sensitivity_results["sample_mean_diff"].append(sens.statistic)
            sensitivity_results["rate_in"].append(sens.rate_in)
            sensitivity_results["rate_out"].append(sens.rate_out)
            sensitivity_results["n_pos_cases"].append(n_pos)
            sensitivity_results["n_neg_cases"].append(n_neg)
            sensitivity_results["n_records_tested"].append(int(sens.valid.sum()))
            sensitivity_results["n_patients"].append(n_sens_patients)
            sensitivity_results["n_runs"].append(n_runs)
            sensitivity_results["n_permutations"].append(n_permutations)
            sensitivity_results["pval_raw"].append(sens.p_value)
            # Specificity is only defined if the case set contains negative cases: with none,
            # every model would score a constant 0 and the comparison would report a spurious
            # difference of 0.
            if n_neg > 0:
                neg = ~y_true
                # over the negative cases "correct" means the case was *not* flagged
                spec = run_label_permutation_test(
                    ~flagged[:, neg],
                    inclusion[:, neg],
                    n_permutations=n_permutations,
                    rng=rng,
                )
                spec_clusters = clusters[neg][spec.valid]
                spec_nulls.append(spec)
                n_spec_patients = int(pd.unique(spec_clusters).size)
                print(
                    f"\tSpecificity (per model): IN={spec.rate_in:.4f}, OUT={spec.rate_out:.4f}, "
                    f"diff={spec.statistic:+.4f} over {spec.valid.sum()} records "
                    f"({n_spec_patients} patients) and {n_runs} models, p(raw)={spec.p_value:.3g}"
                )
                specificity_results["Class"].append(class_name)
                specificity_results["sample_mean_diff"].append(spec.statistic)
                specificity_results["rate_in"].append(spec.rate_in)
                specificity_results["rate_out"].append(spec.rate_out)
                specificity_results["n_pos_cases"].append(n_pos)
                specificity_results["n_neg_cases"].append(n_neg)
                specificity_results["n_records_tested"].append(int(spec.valid.sum()))
                specificity_results["n_patients"].append(n_spec_patients)
                specificity_results["n_runs"].append(n_runs)
                specificity_results["n_permutations"].append(n_permutations)
                specificity_results["pval_raw"].append(spec.p_value)
            else:
                print(
                    f"WARNING: no negative cases for class {class_name}, skipping the specificity comparison (n_pos={n_pos}, n_neg={n_neg})"
                )
        else:
            print(f"WARNING: insufficient case numbers to perform sensitivity/specificity comparison for class {class_name} (n_pos={n_pos}, n_neg={n_neg})")
    # No multiplicity correction here. The figure pools every class of every dataset it shows,
    # so the family size is not knowable from one call; `rop_comparison_plot.py` applies the
    # Bonferroni correction over the whole figure and inverts each permutation null at the
    # matching per-comparison level to get the interval it draws.
    if not sensitivity_results["pval_raw"] and not specificity_results["pval_raw"]:
        print("WARNING: no class yielded a comparison")
        return sensitivity_results, specificity_results, {}
    null_tails = {}
    for label, nulls in (("sensitivity", sens_nulls), ("specificity", spec_nulls)):
        # float64 throughout: the interval endpoint is a difference of two nearly equal
        # numbers, and rounding one of them can move an endpoint across zero
        null_tails[f"{label}_null_lower"] = (
            np.stack([r.null_lower for r in nulls], axis=0)
            if nulls else np.zeros((0, 0), dtype=np.float64)
        )
        null_tails[f"{label}_null_upper"] = (
            np.stack([r.null_upper for r in nulls], axis=0)
            if nulls else np.zeros((0, 0), dtype=np.float64)
        )
        null_tails[f"{label}_n_permutations"] = np.array(
            [r.n_permutations for r in nulls], dtype=np.int64
        )
    print(
        f"... {len(sensitivity_results['pval_raw'])} sensitivity and "
        f"{len(specificity_results['pval_raw'])} specificity comparisons; raw p-values only, "
        f"multiplicity is corrected across the figure at plot time"
    )
    return sensitivity_results, specificity_results, null_tails
