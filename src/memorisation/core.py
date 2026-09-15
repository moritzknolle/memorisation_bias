"""Turning a set of leave-many-out training runs into a per-record memorisation verdict.

The pipeline the analysis entry points drive:

1. `load_predictions` / `load_from_dir` read each run directory: the logits over one
   split, the patient ids, and the boolean mask of which patients that run trained on.
2. `convert_patientmasks_to_recordmask` expands the patient-level masks to record level,
   since subsetting happens per patient but the test is per record.
3. `RandomSubsetPredictionPartitioner` splits a record's predictions across runs into
   IN (the patient was in that run's training subset) and OUT, applying the activation
   function on the way out. It indexes lazily so the logits can stay memory-mapped.
4. `test_significantly_different` runs a two-sample test per record and applies
   Benjamini-Hochberg FDR correction across records.

Passing `inclusion_masks=None` to the partitioner assigns IN/OUT at random instead, which
is the null control the sanity-check outputs and reference curves are built from.
"""

from typing import Callable
import hyppo.ksample
import numpy as np
from pathlib import Path
import keras  # type: ignore
import scipy
from typing import List
import multiprocessing
from tqdm import tqdm
import joblib
import pandas as pd
import hyppo
from typing import Optional, Tuple, Union
from operator import itemgetter
from statsmodels.stats.multitest import multipletests  # type: ignore

from ..data.datasets import BaseDataset


def load_from_dir(
    l_dir: Path,
    prefix_str: str,
    verbose: bool = False,
    preds_dtype: np.dtype = np.dtype(np.float16),
    load_as_memmap: bool = False,
):
    """
    Loads one run's raw model outputs for a single split, along with the patient ids and
    the training-subset mask needed to partition them into IN/OUT groups later.

    Args:
        l_dir (Path): Run directory, holding {prefix_str}_logits.npy, patient_ids.pkl and
            patient_subset_mask.npy (as written by `RetrainLogger`).
        prefix_str (str): Which split to read: "train", "long_eval" (= train_future) or "test".
        verbose (bool): Whether to print loaded file shapes and additional logs.
        preds_dtype (np.dtype): The data type to cast the predictions to. Default is np.float16.
            Ignored when loading as a memmap, which keeps the on-disk dtype.
        load_as_memmap (bool): Whether to load the logits as a memory-mapped array to save
            memory. Default is False. Needed for the large datasets, where 200 runs' logits
            do not fit in RAM.

    Returns:
        tuple, or None if the directory could not be read:
            preds (np.ndarray): Raw model outputs (logits), shape (n_records, n_classes).
                No activation function is applied here -- the partitioner does that.
            patient_ids (pd.Series): Patient ids, in the order patient_subset_mask indexes them.
            patient_subset_mask (np.ndarray): Boolean mask over patients, True where the
                patient was in this run's training subset.

    Raises:
        Exception: If any NaN values are found in the logits array.
    """
    try:
        preds = np.load(
            l_dir / f"{prefix_str}_logits.npy",
            mmap_mode="r" if load_as_memmap else None,
        )
        patient_ids = joblib.load(l_dir / "patient_ids.pkl")
        patient_subset_mask = np.array(
            np.load(l_dir / "patient_subset_mask.npy"), dtype=bool
        )
        if verbose:
            print(
                f"Loaded logits of shape {preds.shape} and subset mask of shape {patient_subset_mask.shape} from {l_dir}"
            )
        if not np.count_nonzero(np.isnan(preds)) == 0:
            raise Exception(f"Found NaNs in logits from {l_dir}")
    except Exception as e:
        print(f"Could not load logits from {l_dir}, error message:\n {e}")
        return None
    if not load_as_memmap:
        if preds.dtype != preds_dtype:
            preds = preds.astype(preds_dtype)
        if not np.count_nonzero(np.isnan(preds)) == 0:
            raise Exception(f"Found NaNs in predictions from {l_dir}")
    if preds is None or patient_subset_mask is None or patient_ids is None:
        print(
            f"WARNING: Could not load predictions, subset mask or patient ids from {l_dir}"
        )
        return None
    return preds, patient_ids, patient_subset_mask


def load_predictions(
    log_dirs: List[Path],
    load_fn: Callable,
    multi_processing: bool = True,
    threads: int = 16,
):
    """
    Load the predictions and training-subset masks of a set of runs.

    Directories that cannot be read are skipped.

    Args:
        log_dirs:  A list of paths to the log directories, one per run.
        load_fn: A callable returning (preds, patient_ids, patient_subset_mask) for one
            directory, i.e. `load_from_dir` bound to a split via functools.partial.
        multi_processing: Whether to load predictions in parralel using multiprocessing.
            Turn this off for memmapped datasets, where the workers' copies are the
            problem rather than the load time.
        threads: The number of threads to use for multiprocessing.

    Returns:
        preds_list: A list of n_runs arrays (or memmaps) of shape (n_records, n_classes).
            Kept as a list, not stacked, so memmapped runs are never materialised at once.
        patient_ids: The patient ids shared by all runs. Asserted identical across runs,
            since the masks are meaningless if the ordering differs.
        masks_arr: A boolean array of shape (n_runs, n_patients) of training-subset masks.
    """
    # check all directories are valid
    log_dirs = [l_dir for l_dir in log_dirs if l_dir.is_dir()]
    if multi_processing:
        with multiprocessing.Pool(processes=threads) as pool:
            results = list(
                tqdm(
                    pool.imap(load_fn, log_dirs),
                    desc="Loading predictions",
                    total=len(log_dirs),
                    leave=False,
                )
            )
    else:
        results = [
            load_fn(l_dir)
            for l_dir in tqdm(
                log_dirs, desc="Loading predictions", total=len(log_dirs), leave=False
            )
        ]
    preds_list, patient_subset_mask_list = [], []
    prev_patient_ids = None
    for r in tqdm(results, desc="Processing predictions", leave=False):
        if r is not None:
            assert np.count_nonzero(np.isnan(r[0])) == 0, f"Found NaN in logits: {r[0]}"
            preds, curr_patient_ids, patient_subset_mask = r
            preds_list.append(preds)
            patient_subset_mask_list.append(patient_subset_mask)
            # check for patient id mismatch
            if prev_patient_ids is not None:
                assert prev_patient_ids.equals(
                    curr_patient_ids
                ), "Found patient id mismatch. Patient ids are assumed to stay constant between runs. Check for Errors ..."
            prev_patient_ids = curr_patient_ids

    masks_arr = np.stack(patient_subset_mask_list, axis=0)
    return preds_list, prev_patient_ids, masks_arr


def convert_patientmasks_to_recordmask(
    patient_masks: np.ndarray,
    patient_ids: pd.Series,
    dataset: BaseDataset,
    patient_id_col: str,
):
    """Expand per-patient training-subset masks into per-record masks.

    Subsets are drawn over patients, so a run's mask says which *patients* it trained on;
    the statistical test works per record. A record is marked IN when its patient is, and
    the expansion is applied to whichever split is being analysed -- for the future split
    this is what encodes "the model saw earlier data from this patient", even though it
    never saw the record itself.

    Args:
        patient_masks: Boolean array (n_runs, n_patients), one row per run.
        patient_ids: Patient ids in the order `patient_masks` indexes them (as saved
            alongside each run's logits).
        dataset: Dataset whose `dataframe` gives the record order of the predictions.
        patient_id_col: Column of that dataframe holding the patient id.

    Returns:
        np.ndarray: Boolean array (n_runs, len(dataset)).
    """
    assert patient_masks.shape[1] == len(
        patient_ids
    ), f"Patient masks shape {patient_masks.shape} does not match patient ids shape {len(patient_ids)}"
    # convert patient-level masks to record-level masks
    n_runs, n_patients = patient_masks.shape
    record_patient_ids = dataset.dataframe[patient_id_col]
    record_masks = np.zeros((n_runs, len(dataset)), dtype=bool)
    for i in tqdm(
        range(n_runs), desc="Converting patient masks to record masks", leave=False
    ):
        # selected patients
        selected_patients = patient_ids[patient_masks[i]]
        # selected records
        selected_records_mask = record_patient_ids.isin(selected_patients)
        selected_records_idcs = np.nonzero(selected_records_mask)[0]
        # set record masks
        record_masks[i, selected_records_idcs] = True
    return record_masks


def get_test_object(test_type: str):
    """Get the two-sample test used to compare a record's IN and OUT predictions.

    'energy' is the default: a nonparametric energy-distance test (hyppo's KSample with
    Dcorr), which makes no distributional assumption about the predicted probabilities.
    'hotelling' is Hotelling's T-squared -- considerably faster, but it assumes normality,
    which bounded probabilities near 0 or 1 do not satisfy.
    """
    if test_type == "hotelling":
        return hyppo.ksample.Hotelling()
    elif test_type == "energy":
        return hyppo.ksample.KSample(indep_test="Dcorr")
    else:
        raise ValueError(
            f"Unknown test type: {test_type}. Must be 'hotelling' or 'energy'."
        )


class RandomSubsetPredictionPartitioner:
    """
    Utility class to that allows storing the result of random subset training runs (predictions and inclusion masks) for efficient retrieval.
    Allows efficiently retrieving IN/OUT predictions for each record without loading all predictions into memory at once.

    Args:
        preds_list: A list of np.memmap or numpy arrays each of shape (n_records, n_classes) containing the predicted probabilities for all records for each run.
        inclusion_masks: A numpy array of shape (n_runs, n_records) containing the subset masks which indicate whether a record was used to training the respective
            model or not. If set to None, the partitioner will randomly split IN/OUT groups for each record when retrieving predictions (useful for sanity checks).
        transform_fn: A function to apply to the raw model outputs (logits) to obtain the predicted probabilities (e.g. a sofmax activation function).
            Default is keras.activations.sigmoid.
        dtype: The data type of the predictions. Default is np.float16.
    """

    def __init__(
        self,
        preds_list: List[np.ndarray],
        inclusion_masks: Optional[np.ndarray] = None,
        transform_fn: Callable = keras.activations.sigmoid,
        dtype: np.dtype = np.dtype(np.float16),
    ):
        self.n_runs = len(preds_list)
        assert (
            len(preds_list) == self.n_runs
        ), f"Expected pred_list length {self.n_runs}, got {len(preds_list)}"
        self.inclusion_masks = inclusion_masks
        self.preds_list = preds_list
        self.transform_fn = transform_fn
        self.n_classes = preds_list[0].shape[-1]
        self.n_records = self.preds_list[0].shape[0]
        if inclusion_masks is not None:
            assert (
                len(inclusion_masks.shape) == 2
            ), f"Expected 2D array of shape (n_runs, n_records), got {inclusion_masks.shape}"
            self.min_in_count = np.min(np.count_nonzero(inclusion_masks, axis=0))
            self.min_out_count = np.min(np.count_nonzero(~inclusion_masks, axis=0))
            self.min_n_count = min(self.min_in_count, self.min_out_count)
            print(
                f"... created PredictionPartioner with min IN count: {self.min_in_count}, min OUT count: {self.min_out_count}"
            )

    def __call__(
        self, record_index: int, reduce: Optional[Union[str, Callable]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the predictions for a given record index, separated into IN and OUT groups based on the inclusion masks.
        Args:
            record_index: The index of the record for which to retrieve the predictions.
            reduce: Whether to reduce the predictions across models by taking the mean. If None, returns predictions from all models.
        """
        # chose randomly if no inclusion masks are provided
        if self.inclusion_masks is None:
            mask = np.random.choice([True, False], size=(self.n_runs,))
        # otherwise use the provided masks to partition IN/OUT groups
        else:
            mask = self.inclusion_masks[:, record_index]
        assert mask.shape == (
            self.n_runs,
        ), f"Expected mask shape {(self.n_runs,)}, got {mask.shape}"
        in_count = np.count_nonzero(mask)
        out_count = len(mask) - in_count
        preds_in = np.array(
            list(
                map(
                    itemgetter(record_index),
                    itemgetter(*np.nonzero(mask)[0])(self.preds_list),
                )
            ),
            dtype=np.float16,
        )
        preds_out = np.array(
            list(
                map(
                    itemgetter(record_index),
                    itemgetter(*np.nonzero(~mask)[0])(self.preds_list),
                )
            ),
            dtype=np.float16,
        )
        # apply transform (activation) function
        if self.transform_fn is not None:
            preds_in = np.array(self.transform_fn(preds_in), dtype=preds_in.dtype)
            preds_out = np.array(self.transform_fn(preds_out), dtype=preds_out.dtype)
        # average over augmentation axis (if present)
        if len(preds_in.shape) == 3:
            preds_in = np.mean(preds_in, axis=1)
            preds_out = np.mean(preds_out, axis=1)
        assert preds_in.shape == (
            in_count,
            self.n_classes,
        ), f"Expected preds_in shape {(in_count, self.n_classes)}, got {preds_in.shape}"
        assert preds_out.shape == (
            out_count,
            self.n_classes,
        ), f"Expected preds_out shape {(out_count, self.n_classes)}, got {preds_out.shape}"
        if reduce == "mean":
            preds_in = np.mean(preds_in, axis=0)
            preds_out = np.mean(preds_out, axis=0)
        elif callable(reduce):
            preds_in = reduce(preds_in)
            preds_out = reduce(preds_out)
        elif reduce is not None:
            raise ValueError(
                f"Unknown reduce method: {reduce}. Must be 'mean' or None."
            )
        return preds_in, preds_out

    def get_many(
        self, record_indices: Union[List[int], np.ndarray], reduce: Optional[Union[Callable, str]] = None, homogenise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call the partitioner for multiple record indices at once.
            Args:
                record_indices: A list of record indices for which to retrieve the predictions.
                reduce: Whether to reduce the predictions across models by taking the mean. If None, returns predictions from all models.
            Returns:
                preds_in_arr: The IN predictions, stacked models-first: (min_n_count, len(record_indices))
                    when `reduce` selects a single class, (min_n_count, len(record_indices), n_classes)
                    when it does not. `homogenise` truncates both groups to `min_n_count` so the
                    per-record results can be stacked at all.
                preds_out_arr: The OUT predictions, same layout.
        """
        in_list, out_list = [], []
        for record_index in tqdm(record_indices, desc="retrieving IN/OUT preds", leave=False):
            preds_in, preds_out = self(record_index, reduce=reduce)
            if homogenise:
                # homogenise shape to form array
                preds_in = preds_in[: self.min_n_count]
                preds_out = preds_out[: self.min_n_count]
            in_list.append(preds_in)
            out_list.append(preds_out)
        return np.stack(in_list, axis=1, dtype=in_list[0].dtype), np.stack(
            out_list, axis=1, dtype=out_list[0].dtype
        )

    def get_unpartioned(
        self, record_index: int, reduce: Optional[str] = "mean"
    ) -> np.ndarray:
        """
        Retrieve the predictions for a given record index across all models, i.e. without partitioning into IN/OUT groups.
        Optionally applies reduction operation like mean.
            Args:
                record_index: The index of the record for which to retrieve the predictions.
                reduce: How to reduce the predictions across models. 'mean' averages them, a callable is
                    applied to the (n_runs, n_classes) array (e.g. `partial(np.take, indices=c, axis=-1)`
                    to select one class), None returns every model's prediction.
            Returns:
                preds: A numpy array containing the predictions for the specified record index, optionally reduced across models.
        """
        preds = np.array(
            list(
                map(
                    itemgetter(record_index),
                    itemgetter(*np.arange(self.n_runs))(self.preds_list),
                )
            ),
            dtype=np.float16,
        )
        if self.transform_fn is not None:
            preds = np.array(self.transform_fn(preds), dtype=preds.dtype)
        if len(preds.shape) == 3:
            preds = np.mean(preds, axis=1)
        if reduce == "mean":
            preds = np.mean(preds, axis=0)
        elif callable(reduce):
            preds = reduce(preds)
        elif reduce is not None:
            raise ValueError(
                f"Unknown reduce method: {reduce}. Must be 'mean', a callable or None."
            )
        return preds

    def get_many_unpartioned(
        self,
        record_indices: Union[List[int], np.ndarray],
        reduce: Optional[Union[Callable, str]] = None,
    ) -> np.ndarray:
        """
        Retrieve every model's prediction for several records at once, without partitioning.

        The IN/OUT split is left to the caller, which is what the run-label permutation test in
        `src.plotting.run_label_permutation_test` needs: it permutes the inclusion mask against a
        fixed matrix of per-model predictions, so the two have to be fetched separately and no
        model may be dropped. `get_many` cannot serve that purpose -- it splits the runs and
        truncates both groups to `min_n_count`.

            Args:
                record_indices: The record indices to retrieve predictions for.
                reduce: Passed through to `get_unpartioned`; typically
                    `partial(np.take, indices=class_idx, axis=-1)` to select a single class.
            Returns:
                preds: (n_runs, len(record_indices)) when `reduce` selects one class,
                    (n_runs, len(record_indices), n_classes) when `reduce` is None.
        """
        preds = [
            self.get_unpartioned(int(record_index), reduce=reduce)
            for record_index in tqdm(
                record_indices, desc="retrieving unpartitioned preds", leave=False
            )
        ]
        return np.stack(preds, axis=1, dtype=preds[0].dtype)

    def get_inclusion(
        self,
        record_indices: Union[List[int], np.ndarray],
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        The IN/OUT inclusion mask for several records, as a (n_runs, len(record_indices)) block.

        With `inclusion_masks=None` the partitioner is in its null-control mode, where `__call__`
        draws a fresh random split on every access. A permutation test needs one *fixed* mask to
        permute against, so this draws the whole block once from `rng` instead. Each record gets
        an independent Bernoulli(1/2) split over runs, matching `__call__`'s per-record draw.

            Args:
                record_indices: The record indices to retrieve the mask for.
                rng: Generator used only in the null-control mode.
            Returns:
                mask: (n_runs, len(record_indices)) boolean, True where the run trained on that
                    record's patient.
        """
        record_indices = np.asarray(record_indices, dtype=int)
        if self.inclusion_masks is None:
            rng = np.random.default_rng() if rng is None else rng
            return rng.random((self.n_runs, record_indices.shape[0])) < 0.5
        return np.asarray(self.inclusion_masks[:, record_indices], dtype=bool)

    def __len__(self):
        return self.n_records


def test_significantly_different(
    prediction_partioner: RandomSubsetPredictionPartitioner,
    test_fn_object: hyppo.ksample.KSample,
    alpha: float = 0.05,
):
    """Test every record for a difference between its IN and OUT model predictions.

    One multivariate two-sample test per record over the per-class predicted probability
    vectors (see `get_test_object`), followed by Benjamini-Hochberg FDR correction across
    all records -- the correction matters, since there is one test per record and the
    datasets run to millions of them.

    A rejected record is what this project calls memorised: whether the patient's earlier
    data was in the training set changed what the models predict about it.

    Args:
        prediction_partioner: Supplies (IN, OUT) predictions per record. Pass one built
            with `inclusion_masks=None` to get the random-partitioning null instead.
        test_fn_object: hyppo test object from `get_test_object`.
        alpha: FDR level.

    Returns:
        tuple:
            test_statistic_arr (np.ndarray): Per-record test statistic.
            rejected (np.ndarray): Boolean, per-record rejection after correction.
            p_val_arr (np.ndarray): Raw per-record p-values.
            adjusted_pvals (np.ndarray): FDR-corrected p-values.
    """
    p_vals, test_statistics = [], []

    n = len(prediction_partioner)
    # for each record, test whether predicticted probabilities differ significantly
    for i in tqdm(
        range(n), desc="Testing for significant differences in predictions", leave=False
    ):
        # retrieve IN/OUT predictions for this record from the prediction partitioner
        in_samples, out_samples = prediction_partioner(i)
        # test for significant difference
        statistic, pval = test_fn_object.test(in_samples, out_samples)
        p_vals.append(pval)
        test_statistics.append(statistic)
    p_val_arr = np.array(p_vals).squeeze()
    test_statistic_arr = np.array(test_statistics).squeeze()
    assert p_val_arr.shape == (
        n,
    ), f"Expected 1D array of shape {(n,)}, got {p_val_arr.shape}"
    # multiple comparison correction
    rejected, adjusted_pvals, _, _ = multipletests(
        pvals=p_val_arr, alpha=alpha, method="fdr_bh"
    )
    return test_statistic_arr, rejected, p_val_arr, adjusted_pvals
