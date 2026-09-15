"""Memorisation analysis of one set of leave-many-out runs.

Loads the runs in --logdir and tests for every record whether the predictions of models trained
on the record's patient (IN) differ from those of models that were not (OUT):

1. Partition each record's predictions into IN/OUT using the runs' patient subset masks.
2. Two-sample test (--test=energy or hotelling) on the predicted probabilities, followed by
   Benjamini-Hochberg correction at alpha=0.05 over all records.
3. Effect size = max over classes of |mean(p_IN) - mean(p_OUT)|.

The analysis is run on the future records and, with --test_mia_signal, on the historical
training records. --sanity_checks repeats both with IN/OUT assigned at random. --rop_comparison
compares per-disease sensitivity and specificity at Youden thresholds computed on the test set.

Outputs are written to ./figs/<--name or dataset>/, result CSVs to its files/ subdirectory.

Usage:
    python memorisation_analysis.py --dataset=mimic-ecg --logdir=<logdir>
    python memorisation_analysis.py --dataset=mimic-iv-ed --logdir=<logdir> --name=mimic-iv-ed_rf
"""

import json
import logging
import os
from functools import partial
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # to avoid annoying warning messages

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
from absl import app, flags  # type: ignore
from tqdm import tqdm  # type: ignore
import scipy

from src.data.dataset_factory import get_dataset
from src.memorisation.core import (
    convert_patientmasks_to_recordmask,
    get_test_object,
    load_from_dir,
    load_predictions,
    test_significantly_different,
    RandomSubsetPredictionPartitioner,
)
from src.metrics import nan_macro_auroc
from src.plotting import (
    esf_plot,
    get_time_deltas,
    plot_binned_memorisation_rate,
    plot_images_by_change_in_prediction,
    plot_patient_trajectories,
    compute_decision_thresholds,
    rop_comparison_by_disease,
)
from src.utils import (
    fig_dir_exists,
    get_act_fn,
    get_data_root,
    get_label_list,
    get_label_list_short,
    get_patient_col,
    get_study_date_col,
    get_study_order_col,
)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force jax to use CPU only
FONT_SIZE = 7
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "figure.figsize": (3.4, 2.0),
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
color_a = "#7c8483"
color_b = "#982649"

flags.DEFINE_string("logdir", None, "Directory with the runs to analyse.")
flags.mark_flag_as_required("logdir")
flags.DEFINE_string(
    "name",
    None,
    "Optional override for the name used for the output directory (./figs/{name}) and the csv file prefixes. "
    "Defaults to the dataset name (suffixed with '_sk' for sklearn models). Useful to keep results of several "
    "model architectures trained on the same dataset apart.",
)
flags.DEFINE_boolean(
    "is_sklearn",
    None,
    "Whether the logged predictions come from an sklearn model (which stores probabilities, so no activation "
    "function is applied) rather than a keras/jax model (which stores logits). Defaults to auto-detection from "
    "the run config recorded in the log directory (see detect_sklearn_model).",
)
flags.DEFINE_string(
    "dataset",
    "mimic-iv-ed",
    "The dataset to analyse. This script will try to look for log directories at FLAGS.logdir",
)
flags.DEFINE_string(
    "csv_root", "./data/csv/", "The directory where the csv files can be found."
)
flags.DEFINE_string(
    "save_root",
    "./data/npy/",
    "Path to root folder where the npy/memmap files are stored.",
)
flags.DEFINE_integer("n_logs", None, "Number of log directories to use.")
flags.DEFINE_enum(
    "test",
    "energy",
    ["hotelling", "energy"],
    "The type of statistical test to use. Hotelling is faster than energy-based tests but assumes normality.",
)
flags.DEFINE_boolean(
    "test_mia_signal",
    True,
    "Whether to perform statistical tests between IN/OUT model predictions also on the training dataset. This is the signal that a membership inference attack exploits.",
)
flags.DEFINE_list("img_size", [512, 512], "Image size (for plotting only).")
flags.DEFINE_bool(
    "plot_images",
    False,
    "If set to True, plot some sample images most vulnerable records and randomly sample records.",
)
flags.DEFINE_bool(
    "sanity_checks",
    True,
    "If set to True, performs a series of sanity checks.",
)
flags.DEFINE_bool(
    "rop_comparison",
    True,
    "If set to True, performs a receiver operating point comparison of memorisation induced probability changes.",
)
flags.DEFINE_boolean(
    "multiprocessing", True, "Whether to use multiprocessing for parallel computation."
)
flags.DEFINE_integer("threads", 8, "Number of threads to use for multiprocessing.")
flags.DEFINE_integer("seed", 21, "Random seed.")
FLAGS = flags.FLAGS


def detect_sklearn_model(base_dir: Path) -> bool:
    """Infer whether a log directory holds predictions from an sklearn model.

    sklearn models store class probabilities (no activation function must be applied),
    keras/jax models store logits (which need a sigmoid/softmax). The sklearn training
    scripts record a "model_class" entry in the run config, which the keras/jax ones do
    not, so the first readable info.json settles it. This is deliberately not based on
    the directory name, which does not survive renaming the log directories.
    """
    for run_dir in sorted(base_dir.iterdir()):
        info_path = run_dir / "info.json"
        if not run_dir.is_dir() or not info_path.exists():
            continue
        with open(info_path) as f:
            return "model_class" in json.load(f).get("wandb_config", {})
    print(
        f"... found no info.json in {base_dir}, falling back to the directory name to detect sklearn models"
    )
    return "sklearn" in str(base_dir).lower()


def main(argv):
    np.random.seed(FLAGS.seed)
    assert len(FLAGS.img_size) == 2, "img_size must be a list of two integers"
    print(
        "... analysing log files for dataset",
        FLAGS.dataset,
        "from",
        FLAGS.logdir,
    )
    train_historical_dataset, val_dataset, train_future_dataset, test_dataset = get_dataset(
        dataset_name=FLAGS.dataset,
        resolution=64 if FLAGS.dataset == "mimic-cxr" else None,
        csv_root=Path(FLAGS.csv_root),
        save_root=Path(FLAGS.save_root),
        data_root=get_data_root(FLAGS.dataset),
        write_to_disk=False,
    )
    base_dir = Path(FLAGS.logdir)
    is_sklearn_model = (
        FLAGS.is_sklearn
        if FLAGS.is_sklearn is not None
        else detect_sklearn_model(base_dir)
    )
    print(f"... treating predictions as {'sklearn probabilities' if is_sklearn_model else 'keras/jax logits'}")
    DATASET_NAME = FLAGS.name or (
        f"{FLAGS.dataset}_sk" if is_sklearn_model else FLAGS.dataset
    )
    out_dir = Path(f"./figs/{DATASET_NAME}")
    fig_dir_exists(out_dir)
    logdirs = [d for d in base_dir.iterdir()]
    if FLAGS.n_logs is not None:
        logdirs = logdirs[: FLAGS.n_logs]
    transform_fn = get_act_fn(FLAGS.dataset, is_sklearn_model=is_sklearn_model)
    print(
        f"... automatically retrieved transform (activation) function: {transform_fn}"
    )
    load_fn_le = partial(
        load_from_dir,
        prefix_str="long_eval",
        load_as_memmap=True if FLAGS.dataset=="heedb" else False,  # use memmap for HEEDB since the predictions are too large to fit into memory
    )
    # load model predictions and subset masks for longitudinal evaluation dataset
    preds_list_le, patient_ids_le, masks_le = load_predictions(
        log_dirs=logdirs,
        load_fn=load_fn_le,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    labels_le = train_future_dataset.__get_all_targets__()
    # load model predictions for test dataset
    load_fn_test = partial(
        load_from_dir,
        prefix_str="test",
        load_as_memmap=True if FLAGS.dataset=="heedb" else False,  # use memmap for HEEDB since the predictions are too large to fit into memory
    )
    # test dataset patients are disjoint from training dataset patients, so we do not retrieve inclusion masks
    preds_list_test, *_ = load_predictions(
        log_dirs=logdirs,
        load_fn=load_fn_test,
        multi_processing=FLAGS.multiprocessing,
        threads=FLAGS.threads,
    )
    labels_test = test_dataset.__get_all_targets__()
    # convert patient-level inclusion masks to record-level inclusion masks by looking up corresponding patient ids
    masks_le = convert_patientmasks_to_recordmask(
        patient_masks=masks_le,
        patient_ids=patient_ids_le,
        dataset=train_future_dataset,
        patient_id_col=get_patient_col(FLAGS.dataset),
    )
    # split predictions into two groups:
    # IN model predictions: predictions from models trained on the respective patient's data
    # OUT model predictions: predictions from models not trained on respective patient's data
    le_partioner = RandomSubsetPredictionPartitioner(
        preds_list=preds_list_le,
        inclusion_masks=masks_le,
        dtype=preds_list_le[0].dtype,
        transform_fn=transform_fn,
    )
    test_partioner = RandomSubsetPredictionPartitioner(
        preds_list=preds_list_test,
        dtype=preds_list_test[0].dtype,
        transform_fn=transform_fn,
    )
    # statisictal testing to determine whether predictions provided by IN models differ significantly different from those of OUT models
    test_statistics, rejected_le, pvals_le, corrected_pvals_le = test_significantly_different(
        prediction_partioner=le_partioner,
        test_fn_object=get_test_object(FLAGS.test),
        alpha=0.05,
    )
    print("\n\n", "=" * 50, "RESULTS", "=" * 50)
    assert (
        len(rejected_le) == len(corrected_pvals_le) == len(train_future_dataset)
    ), f"lengths do not match: {len(rejected_le)}, {len(corrected_pvals_le)}, {len(train_future_dataset)}"
    print(
        f"found n_signficicantly different predictions in long eval dataset: {sum(rejected_le)}/{len(corrected_pvals_le)} ({sum(rejected_le)/len(corrected_pvals_le)*100:.2f}%)"
    )
    # compute average IN and OUT predictions (needed for the effect size below, independently of the ROP comparison)
    mean_in_pred_le, mean_out_pred_le = [], []
    for i in tqdm(range(len(le_partioner)), desc="extracting average IN/OUT LE predictions", leave=False):
        r = le_partioner(i, reduce="mean")
        mean_in_pred_le.append(r[0])
        mean_out_pred_le.append(r[1])
    mean_in_pred_le = np.array(mean_in_pred_le, dtype=mean_in_pred_le[0].dtype)
    mean_out_pred_le = np.array(mean_out_pred_le, dtype=mean_out_pred_le[0].dtype)
    print("... retrieved average IN/OUT LE predictions. shape:", mean_in_pred_le.shape, mean_out_pred_le.shape)
    if FLAGS.rop_comparison:
        mean_test_pred = []
        for i in tqdm(range(len(test_partioner)), desc="extracting average test predictions", leave=False):
            mean_test_pred.append(test_partioner.get_unpartioned(i, reduce='mean'))  # all test patients are OUT
        mean_test_pred = np.array(mean_test_pred, dtype=mean_test_pred[0].dtype)
        print("... retrieved average test prediction. shape:", mean_test_pred.shape)
        label_names = get_label_list(FLAGS.dataset)
        label_names_short = get_label_list_short(FLAGS.dataset)
        n_classes = len(label_names)
        # basic macro AUROC comparison
        macro_auc_in = nan_macro_auroc(y_true=labels_le, y_score=mean_in_pred_le)
        macro_auc_out = nan_macro_auroc(y_true=labels_le, y_score=mean_out_pred_le)
        print(
            f"macro AUROC: IN models={(macro_auc_in)*100:.2f}, OUT models={macro_auc_out*100:.2f}%  both over all N={len(train_future_dataset)} longitudinal eval cases)"
        )
        print(
            "... computing optimal decision thresholds (Youden's J statistic) on test set using the average predictions across all models"
        )
        # compute Youden thresholds
        (
            youden_thresholds,
            fprs,
            tprs,
            aucs,
        ) = compute_decision_thresholds(
            mean_test_pred, targets=labels_test, verbose=False
        )
        for i, class_name in enumerate(label_names_short):
            print(
                f"\t{class_name}: theta={youden_thresholds[i]:.3f}, FPR={fprs[i]:.3f}, TPR={tprs[i]:.3f}, AUROC={aucs[i]:.3f}"
            )
        print("... performing receiver operating point comparison for cases of de novo disease")
        # Sensitivity + Specificity comparison for cases of de novo disease
        (
            sensitivity_results_denovo, specificity_results_denovo, rop_null_tails_denovo
        ) = rop_comparison_by_disease(
            prediction_partioner=le_partioner,
            targets=labels_le,
            thresholds=youden_thresholds,
            class_names=label_names,
            dataframe=train_future_dataset.dataframe,
            memorisation_status=rejected_le,
            patient_id_col=get_patient_col(FLAGS.dataset),
            verbose=True,
            denovo_disease=True,
        )
        sensitivity_results_denovo_df = pd.DataFrame(sensitivity_results_denovo)
        specificity_results_denovo_df = pd.DataFrame(specificity_results_denovo)
        print(f"Sensitivity results (de novo disease): \n{sensitivity_results_denovo_df}")
        print(f"Specificity results (de novo disease): \n{specificity_results_denovo_df}")
        sensitivity_results_denovo_df.to_csv(
            out_dir / f"files/{DATASET_NAME}_sensitivity_comparison(denovo).csv",
            index=False,
        )
        specificity_results_denovo_df.to_csv(
            out_dir / f"files/{DATASET_NAME}_specificity_comparison(denovo).csv",
            index=False,
        )
        # the permutation-null tails the figure inverts into its intervals: the level they
        # are read at depends on the size of the figure's comparison family, which is only
        # known once the figure is drawn
        np.savez_compressed(
            out_dir / f"files/{DATASET_NAME}_rop_bootstrap(denovo).npz", **rop_null_tails_denovo
        )
        print("... performing receiver operating point comparison for case of 'seen' disease")
        # Sensitivity + Specifitiy comparison for cases of 'seen' disease
        (
            sensitivity_results, specificity_results, rop_null_tails
        ) = rop_comparison_by_disease(
            prediction_partioner=le_partioner,
            targets=labels_le,
            thresholds=youden_thresholds,
            class_names=label_names,
            dataframe=train_future_dataset.dataframe,
            memorisation_status=rejected_le,
            patient_id_col=get_patient_col(FLAGS.dataset),
            verbose=True,
            denovo_disease=False,
        )
        sensitivity_results_df = pd.DataFrame(sensitivity_results)
        specificity_results_df = pd.DataFrame(specificity_results)
        print(f"Sensitivity results : \n{sensitivity_results_df}")
        print(f"Specificity results : \n{specificity_results_df}")
        sensitivity_results_df.to_csv(
            out_dir / f"files/{DATASET_NAME}_sensitivity_comparison.csv",
            index=False,
        )
        specificity_results_df.to_csv(
            out_dir / f"files/{DATASET_NAME}_specificity_comparison.csv",
            index=False,
        )
        # the permutation-null tails the figure inverts into its intervals: the level they
        # are read at depends on the size of the figure's comparison family, which is only
        # known once the figure is drawn
        np.savez_compressed(
            out_dir / f"files/{DATASET_NAME}_rop_bootstrap.npz", **rop_null_tails
        )
    # perform statistical testing also for IN/OUT predictions on training dataset
    if FLAGS.test_mia_signal:
        print("... loading training dataset predictions")
        train_load_fn = partial(
            load_from_dir,
            prefix_str="train",
            load_as_memmap=True if FLAGS.dataset=="heedb" else False,  # use memmap for HEEDB since the predictions are too large to fit into memory
        )
        # load predictions and subset masks from log directories
        preds_list_train, patient_ids_train, masks_train = load_predictions(
            log_dirs=logdirs,
            load_fn=train_load_fn,
            multi_processing=FLAGS.multiprocessing,
            threads=FLAGS.threads,
        )
        # convert patient-level masks to record-level masks
        masks_train = convert_patientmasks_to_recordmask(
            patient_masks=masks_train,
            patient_ids=patient_ids_train,
            dataset=train_historical_dataset,
            patient_id_col=get_patient_col(FLAGS.dataset),
        )
        train_partioner = RandomSubsetPredictionPartitioner(
            preds_list=preds_list_train,
            inclusion_masks=masks_train,
            dtype=preds_list_train[0].dtype,
            transform_fn=transform_fn,
        )
        mean_in_pred_train, mean_out_pred_train = [], []
        for i in tqdm(range(len(train_partioner)), desc="extracting average IN/OUT TRAIN predictions", leave=False):
            r = train_partioner(i, reduce="mean")
            mean_in_pred_train.append(r[0])
            mean_out_pred_train.append(r[1])
        mean_in_pred_train = np.array(mean_in_pred_train, dtype=mean_in_pred_train[0].dtype)
        mean_out_pred_train = np.array(mean_out_pred_train, dtype=mean_out_pred_train[0].dtype)
        effect_size_train = np.max(np.abs(mean_in_pred_train - mean_out_pred_train), axis=-1).squeeze()
        # statistical test for significant differences
        train_test_statistics, train_rejected, train_pvals, train_corrected_pvals = (
            test_significantly_different(
                prediction_partioner=train_partioner,
                test_fn_object=get_test_object(FLAGS.test),
                alpha=0.05,
            )
        )
        print(
            f"found n_signficicantly different predictions in TRAIN dataset: {sum(train_rejected)}/{len(train_corrected_pvals)} ({sum(train_rejected)/len(train_corrected_pvals)*100:.2f}%)"
        )
        train_mem_results = {"id":train_historical_dataset.dataframe.index, "test_statistic": train_test_statistics, "pvals(raw)": train_pvals, "pvals(corrected)": train_corrected_pvals, "effect_size": effect_size_train, "rejected": train_rejected, "pid": train_historical_dataset.dataframe[get_patient_col(FLAGS.dataset)]}
        train_mem_df = pd.DataFrame(train_mem_results)
        train_mem_df.to_csv(
            out_dir / f"files/{DATASET_NAME}_mem_data_train.csv",
            index=False,
        )
    print("=" * 50, "RESULTS", "=" * 50)
    print("\n")
    # maximum change in predicted probability (across all classes) between average IN and average OUT model predictions
    effect_size_le = np.max(np.abs(mean_in_pred_le- mean_out_pred_le), axis=-1).squeeze()
    if FLAGS.plot_images:
        plot_images_by_change_in_prediction(
            in_preds=mean_in_pred_le,
            out_preds=mean_out_pred_le,
            pvals=corrected_pvals_le,
            dataset=train_future_dataset,
            dataset_name=FLAGS.dataset,
            label_name_list=get_label_list_short(FLAGS.dataset),
            patient_id_col=get_patient_col(FLAGS.dataset),
            plot_highest=True,
            out_dir=out_dir,
        )
        plot_images_by_change_in_prediction(
            in_preds=mean_in_pred_le,
            out_preds=mean_out_pred_le,
            pvals=corrected_pvals_le,
            dataset=train_future_dataset,
            dataset_name=FLAGS.dataset,
            label_name_list=get_label_list_short(FLAGS.dataset),
            patient_id_col=get_patient_col(FLAGS.dataset),
            plot_highest=False,
            out_dir=out_dir,
        )
    # eSF plot of maximum change in average predicted probabilities between IN and OUT models
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    esf_plot(
        data_arr=effect_size_le,
        ax=ax,
        xlabel=r"$\mathrm{max}(|p_{{IN}} - p_{{OUT}})|$",
        color='black',
    )
    ax.set_xlabel(r"$\mathrm{max}(|p_{{IN}} - p_{{OUT}})|$")
    ax.set_ylabel("1-Cumulative Probability")
    ax.legend(fontsize=FONT_SIZE, loc="upper right")
    ax.set_yscale("log")
    fig.savefig(
        out_dir / "esf_pdelta.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    # time-delta analysis between most recent historic record and followup records
    study_date_col = get_study_date_col(FLAGS.dataset)
    if not isinstance(train_historical_dataset.dataframe[study_date_col].dtype, pd.Timestamp):
        print("converting study date to datetime (train dataset)")
        train_historical_dataset.dataframe[study_date_col] = pd.to_datetime(
            train_historical_dataset.dataframe[study_date_col]
        )
    if not isinstance(
        train_future_dataset.dataframe[study_date_col].dtype, pd.Timestamp
    ):
        print("converting study date to datetime (long eval dataset)")
        train_future_dataset.dataframe[study_date_col] = pd.to_datetime(
            train_future_dataset.dataframe[study_date_col]
        )
    print("... computing time difference to most recent (historic) training record")
    train_future_dataset.dataframe["time_delta(months)"] = get_time_deltas(
        future_data=train_future_dataset.dataframe,
        historic_data=train_historical_dataset.dataframe,
        patient_id_col=get_patient_col(FLAGS.dataset),
        study_date_col=study_date_col,
    )
    max_followup_months = train_future_dataset.dataframe["time_delta(months)"].max()
    ticks = [i * 12 for i in range((max_followup_months // 12))] + [
        max_followup_months
    ]
    ticks[0] += 1  # avoid zero tick
    # eSF plot of time deltas (all available follow-up)
    fig, ax = plt.subplots(figsize=(4, 2))
    esf_plot(
        data_arr=train_future_dataset.dataframe["time_delta(months)"].values,
        ax=ax,
        xlabel="Followup period (months)",
    )
    # eSF plot of time deltas (subset of records where we detected memorisation)
    memorised_time_deltas = train_future_dataset.dataframe.loc[
        rejected_le, "time_delta(months)"
    ].values
    esf_legend_labels = ["All follow-up records"]
    # models that memorise nothing leave this subset empty, which has no eSF to plot
    if len(memorised_time_deltas) > 0:
        esf_plot(
            data_arr=memorised_time_deltas,
            ax=ax,
            color="red",
            xlabel="Followup period (months)",
        )
        esf_legend_labels.append("Memorised records")
    else:
        print(
            "... no records with a significant change detected, skipping the memorised-records eSF curve"
        )
    ax.set_xticks(ticks)
    ax.set_xlabel("Followup period (months)")
    ax.set_ylabel("1-Cumulative Probability")
    ax.legend(
        esf_legend_labels,
        fontsize=FONT_SIZE,
        loc="lower left",
    )
    ax.set_yscale("log")
    fig.savefig(
        out_dir / "esf_plot_time_deltas_memorisation.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    long_eval_df = train_future_dataset.dataframe.copy()
    long_eval_df["rejected"] = rejected_le
    plot_binned_memorisation_rate(
        dataframe=long_eval_df,
        patient_id_col=get_patient_col(FLAGS.dataset),
        save_path=out_dir / "followup_period_vs_memorisation_rate.pdf",
    )
    # manhattan plot of time deltas vs. -log10(pvals)
    fig, ax = plt.subplots(figsize=(3, 1.7))
    ax.axhline(
        y=-np.log10(0.05),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Significance Threshold)",
    )
    ax.scatter(
        train_future_dataset.dataframe["time_delta(months)"].values,
        -np.log10(corrected_pvals_le),
        color=color_a,
        s=2,
        alpha=0.7,
    )
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_xlabel("Followup period (months)")
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(
        out_dir / "manhattan_plot_time_delta_vs_pvals.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    # save p values, rejection status, time deltas, and effect_size for all records to a csv file
    out = {"id": train_future_dataset.dataframe.index, "test_statistic": test_statistics, "pvals(corrected)": corrected_pvals_le, "pvals(raw)":pvals_le, "time_deltas": train_future_dataset.dataframe["time_delta(months)"].values, "effect_size": effect_size_le, "rejected": rejected_le, "pid": train_future_dataset.dataframe[get_patient_col(FLAGS.dataset)]}
    out_df = pd.DataFrame.from_dict(out)
    out_df.to_csv(
        out_dir / f"files/{DATASET_NAME}_mem_data.csv",
        index=False,
    )
    print(f"... saved p values, rejection status, time deltas, and effect_size for all records to csv at: {out_dir}/files/{DATASET_NAME}_mem_data.csv")
    # extract the 10 patients with the highest effect_size
    effect_size_idcs = np.argsort(effect_size_le)[
        -50:
    ]  # many of the records with the highest differences are the same patients so we pick a few more than needed here
    effect_size_pids = train_future_dataset.dataframe.iloc[effect_size_idcs][
        get_patient_col(FLAGS.dataset)
    ].unique()[:15]
    if FLAGS.plot_images:
        plot_patient_trajectories(
            train_historical_dataset=train_historical_dataset,
            train_future_dataset=train_future_dataset,
            patient_ids=effect_size_pids,
            patient_id_col=get_patient_col(FLAGS.dataset),
            study_order_col=get_study_order_col(FLAGS.dataset),
            in_preds=mean_in_pred_le,
            out_preds=mean_out_pred_le,
            pvals=corrected_pvals_le,
            dataset_name=FLAGS.dataset,
            out_dir=out_dir,
        )
    print("=" * 50, "RESULTS", "=" * 50)

    if FLAGS.sanity_checks and FLAGS.test_mia_signal:
        print("\n\n", "=" * 50, "SANITY CHECKS", "=" * 50)
        le_partioner = RandomSubsetPredictionPartitioner(
            preds_list=preds_list_le,
            inclusion_masks=None, # we pass None, to partition model predictions randomly
            dtype=preds_list_le[0].dtype,
            transform_fn=transform_fn,
        )
        train_partioner = RandomSubsetPredictionPartitioner(
            preds_list=preds_list_train,
            inclusion_masks=None,  # we pass None, to partition model predictions randomly
            dtype=preds_list_train[0].dtype,
            transform_fn=transform_fn,
        )
        test_statistics_le_sanity, rejected_le_sanity, pvals_le_sanity, corrected_pvals_le_sanity = test_significantly_different(
            prediction_partioner=le_partioner,
            test_fn_object=get_test_object(FLAGS.test),
            alpha=0.05,
        )
        print(
            f".... Sanity check: found n_signficicantly different predictions in LONG EVAL dataset with random partitioning: {sum(rejected_le_sanity)}/{len(train_future_dataset.dataframe)} ({sum(rejected_le_sanity)/len(train_future_dataset.dataframe)*100:.2f}%)"
        )
        test_statistics_train_sanity, rejected_train_sanity, pvals_train_sanity, corrected_pvals_train_sanity = test_significantly_different(
            prediction_partioner=train_partioner,
            test_fn_object=get_test_object(FLAGS.test),
            alpha=0.05,
        )
        print(
            f".... Sanity check: found n_signficicantly different predictions in TRAIN dataset with random partitioning: {sum(rejected_train_sanity)}/{len(train_historical_dataset.dataframe)} ({sum(rejected_train_sanity)/len(train_historical_dataset.dataframe)*100:.2f}%)"
        )
        mean_in_pred_le_sanity, mean_out_pred_le_sanity = [], []
        for i in tqdm(range(len(le_partioner)), desc="extracting average IN/OUT LE predictions for sanity check", leave=False):
            r = le_partioner(i, reduce="mean")
            mean_in_pred_le_sanity.append(r[0])
            mean_out_pred_le_sanity.append(r[1])
        mean_in_pred_le_sanity = np.array(mean_in_pred_le_sanity, dtype=mean_in_pred_le_sanity[0].dtype)
        mean_out_pred_le_sanity = np.array(mean_out_pred_le_sanity, dtype=mean_out_pred_le_sanity[0].dtype)
        mean_in_pred_train_sanity, mean_out_pred_train_sanity = [], []
        for i in tqdm(range(len(train_partioner)), desc="extracting average IN/OUT TRAIN predictions for sanity check", leave=False):
            r = train_partioner(i, reduce="mean")
            mean_in_pred_train_sanity.append(r[0])
            mean_out_pred_train_sanity.append(r[1])
        mean_in_pred_train_sanity = np.array(mean_in_pred_train_sanity, dtype=mean_in_pred_train_sanity[0].dtype)
        mean_out_pred_train_sanity = np.array(mean_out_pred_train_sanity, dtype=mean_out_pred_train_sanity[0].dtype)
        effect_size_le_sanity = np.max(np.abs(mean_in_pred_le_sanity - mean_out_pred_le_sanity), axis=-1).squeeze()
        effect_size_train_sanity = np.max(np.abs(mean_in_pred_train_sanity - mean_out_pred_train_sanity), axis=-1).squeeze()
        sanity_results_dict_train = {
            "id": train_historical_dataset.dataframe.index,
            "test_statistic": test_statistics_train_sanity,
            "pvals(corrected)": corrected_pvals_train_sanity,
            "pvals(raw)": pvals_train_sanity,
            "effect_size": effect_size_train_sanity,
            "rejected": rejected_train_sanity,
            "pid": train_historical_dataset.dataframe[get_patient_col(FLAGS.dataset)],
        }
        sanity_results_dict_le = {
            "id": train_future_dataset.dataframe.index,
            "test_statistic": test_statistics_le_sanity,
            "pvals(corrected)": corrected_pvals_le_sanity,
            "pvals(raw)": pvals_le_sanity,
            "effect_size": effect_size_le_sanity,
            "rejected": rejected_le_sanity,
            "pid": train_future_dataset.dataframe[get_patient_col(FLAGS.dataset)],
        }
        sanity_results_df_train = pd.DataFrame(sanity_results_dict_train)
        sanity_results_df_le = pd.DataFrame(sanity_results_dict_le)
        sanity_results_df_train.to_csv(
            out_dir / f"files/{DATASET_NAME}_mem_data_train_sanity.csv",
            index=False,
        )
        sanity_results_df_le.to_csv(
            out_dir / f"files/{DATASET_NAME}_mem_data_le_sanity.csv",
            index=False,
        )





if __name__ == "__main__":
    app.run(main)
