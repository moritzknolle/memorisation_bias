"""The historical/future split policy: which of a patient's records the model may train on.

The whole project rests on a split that is *within* patient and *forward* in time. For
each patient, records are sorted chronologically and a tail is reserved for evaluation,
chosen so that at least `n_holdout_classes` disease labels first appear in that tail
while the patient still contributes at least one record to training. The models therefore
have seen the patient, but not that record and not that diagnosis -- which is what makes
a detected IN/OUT difference on it evidence of memorisation rather than of learning.

`generate_split_mask` applies the policy to a whole dataframe and is what the
preprocessing notebooks call; `split_policy_by_disease_labels` is the per-patient core.
Patients whose labels never change over time carry no signal for this design and are
assigned wholly to training (`check_subject_labels_valid`).

`n_holdout_classes` is reduced for patients who do not have enough distinct disease states.
"""

import pandas as pd
from tqdm import tqdm  # type: ignore
import numpy as np
from typing import List, Tuple

tqdm.pandas()


def _validate_split_policy_inputs(
    patient_labels: np.ndarray,
    n_holdout_classes: int,
    n_total_classes: int,
    verbose: bool = False,
) -> None:
    """Validate inputs for split_policy_by_disease_labels."""
    assert patient_labels.ndim == 2, "patient_labels must be 2D array"
    assert patient_labels.shape[1] == n_total_classes, (
        f"patient_labels must have {n_total_classes} columns, got: {patient_labels.shape[1]}"
    )
    assert n_holdout_classes >= 1, "n_holdout_classes must be at least 1"
    assert np.array_equal(patient_labels, patient_labels.astype(bool)), (
        f"patient_labels must be binary (0 or 1), got: \n{patient_labels}"
    )
    if verbose:
        print("input label array:\n", patient_labels)


def _compute_eval_mask(
    patient_labels: np.ndarray,
    n_holdout_classes: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Compute the evaluation mask for longitudinal split.

    Returns:
        Tuple of (eval_mask, n_holdout_classes_used)
    """
    # Check if patient has any "healthy" (all-zeros) records
    has_healthy_records = np.any(patient_labels.sum(axis=1) == 0)
    if verbose:
        print(f"Has healthy records: {has_healthy_records}")

    # Compute cumulative max to track which classes have been seen so far
    cumulative_classes_seen = np.maximum.accumulate(patient_labels, axis=0)
    cumulative_class_count = cumulative_classes_seen.sum(axis=1)
    n_classes_present = int(cumulative_class_count[-1])

    if verbose:
        print("cumulative disease presence matrix:\n", cumulative_classes_seen)
        print("cumulative classes present:\n", cumulative_class_count)

    # Count unique states (including healthy state if applicable)
    n_unique_states = n_classes_present
    if has_healthy_records:
        first_diseased_idx = np.where(cumulative_class_count > 0)[0]
        if len(first_diseased_idx) > 0 and first_diseased_idx[0] > 0:
            n_unique_states += 1
            if verbose:
                print(f"Patient has healthy->diseased transition, counting healthy as separate state. "
                      f"Total unique states: {n_unique_states}")

    classes_remaining = n_classes_present - cumulative_class_count
    if verbose:
        print("classes remaining (not yet seen):\n", classes_remaining)

    # Validation checks
    if n_classes_present < 1 and not has_healthy_records:
        raise ValueError(
            f"Found {n_classes_present} classes present in this patient's data. "
            f"Expected at least 1 unique class to be present. \nLabel array:\n{patient_labels}"
        )

    if n_unique_states < 2:
        raise ValueError(
            f"Found only {n_unique_states} unique state(s) in this patient's data. "
            f"Expected at least 2 unique states to create a valid longitudinal split. \nLabel array:\n{patient_labels}"
        )

    if cumulative_class_count[0] == n_classes_present > 0:
        if not (has_healthy_records and cumulative_class_count[0] == 0):
            raise ValueError(
                f"All {n_classes_present} unique classes appear in the first record. "
                f"Cannot create a valid longitudinal split where some classes are reserved for evaluation only. "
                f"\nLabel array:\n{patient_labels}"
            )

    # Adaptive threshold reduction if necessary
    if n_unique_states <= n_holdout_classes:
        if verbose:
            print(f"Warning: Only {n_unique_states} unique states present, <= n_holdout_classes={n_holdout_classes}.")
            print(f"Using n_holdout_classes={n_unique_states-1} instead.")
        n_holdout_classes = n_unique_states - 1

    # Compute eval mask and ensure at least one training record
    eval_mask = classes_remaining < n_holdout_classes
    warning_msgs = []
    while eval_mask.all():  # More concise than eval_mask.sum() == len(eval_mask)
        warning_msg = f"Warning: All records assigned to eval. Reducing n_holdout_classes to {n_holdout_classes-1}"
        warning_msgs.append(warning_msg)
        if verbose:
            print(warning_msg)
        n_holdout_classes -= 1
        eval_mask = classes_remaining < n_holdout_classes
        if n_holdout_classes == 0:
            for w in warning_msgs:
                print(w)
            raise ValueError("n_holdout_classes reduced to 0, cannot split data.")

    if verbose:
        print("evaluation mask:\n", eval_mask)

    return eval_mask, n_holdout_classes


def _validate_split_policy_output(
    patient_labels: np.ndarray,
    eval_mask: np.ndarray,
    n_holdout_classes: int,
    n_total_classes: int,
    verbose: bool = False,
) -> int:
    """
    Validate the split policy output and return the number of classes first appearing in eval.

    Returns:
        Number of classes that first appear in the eval set.
    """
    assert eval_mask.shape[0] == patient_labels.shape[0], "Output shape mismatch"

    # Find which classes appear in eval set
    classes_in_eval = np.any(patient_labels[eval_mask], axis=0)

    # For each class, find the first record where it appears and check if that record is in eval
    first_occurrences = np.argmax(patient_labels, axis=0)  # Index of first True for each class
    has_class = np.any(patient_labels, axis=0)  # Whether class appears at all
    classes_first_seen_in_eval = classes_in_eval & has_class & eval_mask[first_occurrences]

    n_classes_first_in_eval = np.sum(classes_first_seen_in_eval)
    if verbose:
        print(f"Classes that first appear in eval set: {np.where(classes_first_seen_in_eval)[0]}")

    assert n_classes_first_in_eval >= 1, (
        "No classes are reserved for longitudinal evaluation set. "
        f"At least 1 class should first appear in eval, but found {n_classes_first_in_eval}."
    )

    # Note: Multiple classes can first appear together in the same record
    if n_classes_first_in_eval > n_holdout_classes and verbose:
        print(f"Note: {n_classes_first_in_eval} classes first appear in eval set, "
              f"which exceeds the requested {n_holdout_classes}. "
              f"This occurs when multiple classes appear together in the first eval record.")

    return n_classes_first_in_eval


def split_policy_by_disease_labels(
    patient_labels: np.ndarray,
    n_holdout_classes: int,
    n_total_classes: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Given a 2D array of chronologically sorted binary disease labels (one-hot or multi-hot encoded),
    compute a boolean mask indicating which records should be included in longitudinal evaluation
    based on the number of cases encountered so far. The splits is performed in a way such that a
    specified number of n_heldout_classes disease classes appear only in the longitudinal evaluation set.

    Args:
        patient_labels (np.ndarray): 2D array of shape (n_records, n_total_classes) with binary labels.
        n_holdout_classes (int): Number of classes that should first appear in the evaluation set.
        n_total_classes (int): Total number of classes in the dataset.
        verbose (bool): If True, print additional information.

    Returns:
        Tuple[np.ndarray, int]:
            - eval_mask: 1D boolean array indicating inclusion in longitudinal evaluation set
            - n_classes_first_in_eval: Actual number of classes that first appear in eval set
    """
    _validate_split_policy_inputs(patient_labels, n_holdout_classes, n_total_classes, verbose)
    eval_mask, n_holdout_classes_used = _compute_eval_mask(patient_labels, n_holdout_classes, verbose)
    n_classes_first_in_eval = _validate_split_policy_output(
        patient_labels, eval_mask, n_holdout_classes_used, n_total_classes, verbose
    )
    return eval_mask, n_classes_first_in_eval


def check_subject_labels_valid(
    subject_df: pd.DataFrame, timestamp_col: str, label_cols: List[str]
) -> bool:
    """
    Check if a subject's dataframe has valid disease label progression over time. A valid progression means that there is at least one change in disease labels over time.
    Args:
        subject_df (pd.DataFrame): Dataframe containing records for a single subject.
        timestamp_col (str): The column name representing timestamps of the records.
        label_cols (List[str]): List of column names representing disease labels.
    Returns:
        bool: True if the subject has valid disease label progression, False otherwise.
    """
    assert (
        timestamp_col in subject_df.columns
    ), f"Timestamp column '{timestamp_col}' not found in dataframe."
    subject_df = subject_df.sort_values(
        timestamp_col, ascending=True
    )  # sort in chronological order
    valid = False
    n_records = len(subject_df)
    subject_label_arr = subject_df[label_cols].to_numpy(dtype=int)
    running_classes_present = np.sum(
        np.maximum.accumulate(subject_label_arr, axis=0), axis=1
    )
    if n_records > 1:
        # check there is at least one change in disease labels over time
        if np.max(running_classes_present) != np.min(running_classes_present):
            valid = True
    return valid


def generate_split_mask(
    dataframe: pd.DataFrame,
    patient_id_col: str,
    timestamp_col: str,
    label_cols: List[str],
    n_holdout_classes: int,
) -> np.ndarray:
    """
    Generate a boolean mask for splitting the dataframe into training and (longitudinal)-evaluation sets.

    This split policy ensures that all records in the evaluation set belong to patients that contribute at least one record to the training set.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing patient records.
        patient_id_col (str): The column name representing patient IDs.
        timestamp_col (str): The column name representing timestamps of the records.
        label_cols (List[str]): List of column names representing disease labels.
        n_holdout_classes (int): Number of classes to reserve for longitudinal evaluation.

    Returns:
        np.ndarray: A boolean mask where True indicates the record is in the evaluation set.

    Raises:
        AssertionError: If the specified patient ID or timestamp columns are not found in the dataframe,
                        or if the dataframe index is not a RangeIndex.
    """
    assert (
        patient_id_col in dataframe.columns
    ), f"Patient ID column '{patient_id_col}' not found in dataframe."
    assert (
        timestamp_col in dataframe.columns
    ), f"Timestamp column '{timestamp_col}' not found in dataframe."
    assert (
        dataframe.index == range(len(dataframe))
    ).all(), "Dataframe index must be a RangeIndex."

    long_eval_idcs = []
    # iterate over patients
    for subject_id, subject_df in tqdm(
        dataframe.groupby(patient_id_col), desc="processing patients"
    ):
        subject_df = subject_df.sort_values(
            timestamp_col, ascending=True
        )  # sort in chronological order
        valid = check_subject_labels_valid(
            subject_df=subject_df,
            timestamp_col=timestamp_col,
            label_cols=label_cols,
        )
        # we are only interested in splitting patients with valid disease progression
        # patients are considered valid when their disease labels change over time
        if valid:
            subject_label_arr = subject_df[label_cols].to_numpy(dtype=int)
            subject_split_mask, _ = split_policy_by_disease_labels(
                subject_label_arr,
                n_holdout_classes=n_holdout_classes,
                n_total_classes=len(label_cols),
                verbose=False,
            )
        else:
            # if not valid, assign all records to training set
            subject_split_mask = np.zeros(len(subject_df), dtype=bool)
        long_eval_idcs.extend(subject_df.index[subject_split_mask].tolist())

    # create global boolean mask for the full dataframe
    mask = np.zeros(len(dataframe), dtype=bool)
    mask[long_eval_idcs] = 1

    # verify that the split is valid
    # check every patient in the eval set also has at least one record in the training set
    eval_pids = set(dataframe[patient_id_col][mask])
    train_pids = set(dataframe[patient_id_col][~mask])
    assert len(eval_pids.intersection(train_pids)) == len(
        eval_pids
    ), f"Some patients in the evaluation set do not have records in the training set:{len(eval_pids.intersection(train_pids))} vs. {len(eval_pids)} \n{eval_pids - train_pids}"
    return mask
