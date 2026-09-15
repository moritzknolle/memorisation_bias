"""Training loop and leave-many-out bookkeeping.

* `train_and_eval` trains and evaluates one compiled keras model. With `use_dp=True` the model
  is wrapped by jax_privacy's `make_private`, with delta = 1/len(train_historical) and the noise
  multiplier calibrated for `epsilon` over the run's number of steps. `epsilon=inf` uses the
  same training step with the noise multiplier set to 0 (src/training/dp.py).
* `train_random_subset` trains one run of the leave-many-out experiment: it claims the next
  unclaimed patient subset from `logdir`, trains on those patients and saves the logits over
  the full historical, future and test splits.

`train_and_eval_sklearn` / `train_random_subset_sklearn` are the equivalents for sklearn models,
which save predicted probabilities instead of logits.

The bookkeeping files in `logdir` (`super_mask.npy`, `valid_idcs.pkl`, `completed_idcs.pkl`)
are created and claimed under a file lock (`claim_subset`), so any number of workers can share a
logdir, also when they start at the same time. An index is marked completed once its logits are
saved; indices of failed runs are handed out again when no unclaimed indices are left. A worker
raises StopIteration once all runs are complete.
"""

import fcntl
import math
import os
import pickle
import time
import traceback
from pathlib import Path
from typing import Callable, Optional, Tuple

import keras  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
from absl import flags  # type: ignore
from sklearn.base import BaseEstimator, clone  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

import wandb  # type: ignore

from ..data.datasets import BaseDataset, memmap_generator, subset_memmap_generator
from .logger import RetrainLogger, WandbLogger

AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    subset_indices: Optional[np.ndarray] = None,
    aug_fn: Optional[Callable] = None,
    drop_remainder: bool = True,
    repeat: bool = True,
    shuffle_buffer_size: int = 0,
    num_read_workers: int = 1,
):
    """
    Create a TensorFlow Dataset from NumPy arrays with optional augmentation and shuffling.

    Converts input/target arrays into a batched tf.data.Dataset optimized for training.
    Supports data augmentation, shuffling, and subset selection for memorisation auditing experiments.

    Args:
        inputs: Input data array, shape (n_samples, height, width, channels)
        targets: Target labels array, shape (n_samples, n_classes)
        batch_size: Number of samples per training batch
        shuffle: Whether to randomize sample order each epoch
        augment: Whether to apply data augmentation transforms
        subset_indices: Optional indices to select data subset (for leave-many-out training)
        aug_fn: Augmentation function to apply if augment=True
        drop_remainder: Whether to drop incomplete final batch (default True)
        shuffle_buffer_size: does nothing, only for API compatibility with
            prepare_dataset_memmap (in-memory datasets are shuffled globally)
        num_read_workers: does nothing, only for API compatibility with
            prepare_dataset_memmap (in-memory datasets need no reader threads)

    Returns:
        tf.data.Dataset: Batched, preprocessed dataset ready for training

    Note:
        Uses AUTOTUNE for optimal prefetching performance.
    """
    print("... preparing dataset as tf.data.Dataset")
    # shape checks
    assert inputs.shape[0] == targets.shape[0], "Found mismatching number of samples"
    assert (
        inputs.max() != 255.0
    ), f"Make sure inputs are normalised appropriately, found max(inputs)={inputs.max()}"
    if batch_size > len(inputs):
        print(
            f"... found batch size ({batch_size}) larger than dataset size ({inputs.shape}), setting batch size to dataset size"
        )
        batch_size = len(inputs)
    if subset_indices is not None:
        print(
            f"... using {len(subset_indices)/len(inputs)*100:.1f}% subset of the dataset"
        )
        inputs = inputs[subset_indices]
        targets = targets[subset_indices]
        assert len(inputs) == len(subset_indices) and len(targets) == len(
            subset_indices
        ), f"Mismatch between subset size and mask: {len(inputs)} != {len(subset_indices)} != {len(targets)}"
    print("... using in-memory dataset")
    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        ds = ds.shuffle(ds.cardinality(), reshuffle_each_iteration=True)

    if augment and aug_fn is not None:
        ds = ds.map(
            lambda x, y: (aug_fn(x), y),
            num_parallel_calls=AUTOTUNE,
        )
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return ds.prefetch(buffer_size=AUTOTUNE)


def prepare_dataset_sklearn(
    inputs: np.ndarray,
    targets: np.ndarray,
    shuffle: bool,
    subset_indices: Optional[np.ndarray] = None,
    flatten_input: bool = False,
    repeat: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare numpy arrays for sklearn model training.

    Converts input/target arrays into format suitable for sklearn models.
    Supports subset selection and optional image flattening for traditional sklearn models.

    Args:
        inputs: Input data array, shape (n_samples, height, width, channels) or (n_samples, features)
        targets: Target labels array, shape (n_samples, n_classes) or (n_samples,)
        shuffle: Whether to randomize sample order
        subset_indices: Optional indices to select data subset (for leave-many-out training)
        flatten_input: Whether to flatten input data for traditional sklearn models
        repeat: does nothing, only for API compatibility

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed input and target arrays
    """
    print("... preparing dataset for sklearn")
    # shape checks
    assert inputs.shape[0] == targets.shape[0], "Found mismatching number of samples"
    assert (
        inputs.max() != 255.0
    ), f"Make sure inputs are normalised appropriately, found max(inputs)={inputs.max()}"

    if inputs.dtype != np.float32:
        print("... converting inputs to np.float32")
        inputs = inputs.astype(np.float32)
    if targets.dtype != np.float32:
        print("... converting targets to np.float32")
        targets = targets.astype(np.float32)

    if subset_indices is not None:
        print(
            f"... using {len(subset_indices)/len(inputs)*100:.1f}% subset of the dataset"
        )
        inputs = inputs[subset_indices]
        targets = targets[subset_indices]
        assert len(inputs) == len(subset_indices) and len(targets) == len(
            subset_indices
        ), f"Mismatch between subset size and mask: {len(inputs)} != {len(subset_indices)} != {len(targets)}"

    # Flatten images for traditional sklearn models if requested
    if flatten_input and len(inputs.shape) > 2:
        original_shape = inputs.shape
        inputs = inputs.reshape(inputs.shape[0], -1)
        print(f"... flattened images from {original_shape} to {inputs.shape}")

    if shuffle:
        print("... shuffling dataset")
        indices = np.random.permutation(len(inputs))
        inputs = inputs[indices]
        targets = targets[indices]

    return inputs, targets


def prepare_dataset_memmap(
    inputs: np.memmap,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    subset_indices: Optional[np.ndarray] = None,
    aug_fn: Optional[Callable] = None,
    drop_remainder: bool = True,
    repeat: bool = True,
    shuffle_buffer_size: int = 0,
    num_read_workers: int = 1,
):
    """
    Create a TensorFlow Dataset from NumPy memmap arrays with optional augmentation.

    Converts input/target arrays into a batched tf.data.Dataset optimized for training.
    Supports data augmentation and subset selection for auditing experiments.

    Args:
        inputs: Input data array shape (n_samples, ..., n_channels)
        targets: Target labels array, shape (n_samples, n_classes)
        batch_size: Number of samples per training batch
        shuffle: Whether to randomize sample order each epoch
        augment: Whether to apply data augmentation transforms
        subset_indices: Optional indices to select data subset (for leave-many-out training)
        aug_fn: Augmentation function to apply if augment=True
        drop_remainder: Whether to drop incomplete final batch (default True)
        repeat: Whether to repeat dataset indefinitely (default True)
        shuffle_buffer_size: Size of the sliding shuffle buffer applied to the
            (sequentially read) memmap stream. 0 disables shuffling, which means
            samples are seen in on-disk order in every epoch. Reads stay
            sequential, so this randomises batch composition without hurting
            memmap read throughput. Costs shuffle_buffer_size * sample_nbytes of
            host memory.
        num_read_workers: Number of parallel generators reading the memmap. Values
            > 1 split the index range into that many contiguous shards and
            interleave them, so each shard is still read sequentially but the order
            across shards is non-deterministic.

    Returns:
        tf.data.Dataset: Batched, preprocessed dataset ready for training

    Note:
        Uses AUTOTUNE for optimal prefetching performance.
    """
    print("... preparing dataset as tf.data.Dataset")
    assert isinstance(inputs, np.memmap), "Expected inputs to be a np.memmap array"
    if batch_size > len(inputs):
        print(
            f"... found batch size ({batch_size}) larger than dataset size ({inputs.shape}), setting batch size to dataset size"
        )
        batch_size = len(inputs)
    if subset_indices is not None:
        print(
            f"... using {len(subset_indices)/len(inputs)*100:.1f}% subset of the dataset"
        )
        assert len(inputs) > len(
            subset_indices
        ), f"Expected subset size to be smaller than full dataset size, got {len(subset_indices)} >= {len(inputs)}"
        # Subset targets before passing to generator to avoid memory duplication
        targets_subset = targets[subset_indices]
        generator_fn = lambda: subset_memmap_generator(
            memmap_input_array=inputs,
            target_array_subset=targets_subset,
            subset_indices=subset_indices,
        )
    else:
        generator_fn = lambda: memmap_generator(
            memmap_input_array=inputs, target_array=targets
        )
    output_signature = (
        tf.TensorSpec(shape=inputs[0].shape, dtype=inputs.dtype),
        tf.TensorSpec(shape=targets[0].shape, dtype=targets.dtype),
    )
    if num_read_workers > 1:
        # one generator per contiguous shard of the index range, interleaved: reads stay sequential
        # within a shard while the (GIL-releasing) memmap copies of different shards overlap
        all_indices = (
            subset_indices
            if subset_indices is not None
            else np.arange(len(inputs), dtype=np.int64)
        )
        shard_bounds = np.linspace(0, len(all_indices), num_read_workers + 1).astype(np.int64)
        print(
            f"... using {num_read_workers} interleaved memmap generators "
            f"({len(all_indices) // num_read_workers} samples per shard)"
        )

        def shard_generator(shard_id):
            start, stop = shard_bounds[shard_id], shard_bounds[shard_id + 1]
            for i in range(start, stop):
                index = all_indices[i]
                yield inputs[index].copy(), targets[index].copy()

        ds = tf.data.Dataset.range(num_read_workers).interleave(
            lambda shard_id: tf.data.Dataset.from_generator(
                shard_generator, args=(shard_id,), output_signature=output_signature
            ),
            cycle_length=num_read_workers,
            num_parallel_calls=num_read_workers,
            deterministic=False,
        )
    else:
        print("... using memmap generator dataset")
        ds = tf.data.Dataset.from_generator(
            generator=generator_fn,
            output_signature=output_signature,
        )
    if shuffle:
        if shuffle_buffer_size > 0:
            print(
                f"... shuffling memmap stream with a sliding buffer of {shuffle_buffer_size} samples"
            )
            ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        else:
            print("WARNING: shuffling not supported for memmap datasets")

    if augment and aug_fn is not None:
        ds = ds.map(
            lambda x, y: (aug_fn(x), y),
            num_parallel_calls=AUTOTUNE,
        )
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return ds.prefetch(buffer_size=AUTOTUNE)


def print_dataset_stats(input_arr: np.ndarray, target_arr: np.ndarray, split: str):
    """
    Print dataset statistics for the given input and target arrays.
    Args:
        input_arr: np.ndarray, input data
        target_arr: np.ndarray, target data
        split: str, split name
    Returns:
        None
    """

    print(f"... {split} dataset stats")
    print(f"    input shape: {input_arr.shape} ({input_arr.dtype})")
    print(f"    target shape: {target_arr.shape} ({target_arr.dtype})")


def generate_masks(
    n_runs: int,
    dataset_size: int,
    subset_ratio: float,
    seed: int,
):
    """
    Generate random subset masks for the given number of runs and dataset size.
    Adapted from: https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/train.py

    Args:
        n_runs: int, number of runs
        dataset_size: int, size of the dataset
        subset_ratio: float, ratio of subset
        seed: int, random seed
    Returns:
        np.ndarray, random subset masks
    """
    np.random.seed(seed)
    randomness = np.random.uniform(0.0, 1.0, size=(n_runs, dataset_size))
    order = randomness.argsort(0)
    super_mask = order < int(subset_ratio * n_runs)
    if super_mask.shape[1] != dataset_size:
        raise ValueError(
            f"Found mismatching dataset size: {super_mask.shape[1]} != {dataset_size}"
        )
    if subset_ratio == 1.0:
        super_mask = np.ones_like(super_mask)
    return super_mask


def claim_subset(
    logdir: Path, n_patients: int, n_total_runs: int, subset_ratio: float, seed: int
) -> Tuple[int, np.ndarray]:
    """
    Claim the next unclaimed run of a logdir, creating the bookkeeping files (super_mask.npy,
    valid_idcs.pkl, completed_idcs.pkl) if they do not exist yet.

    Both happen under an exclusive lock on the logdir, so any number of workers can start on the
    same logdir at the same time. When no unclaimed runs are left, runs that have not completed
    are handed out again.

    Args:
        logdir: Path, directory holding the bookkeeping files
        n_patients: int, number of patients in the training split
        n_total_runs: int, total number of runs
        subset_ratio: float, ratio of patients per run
        seed: int, random seed for generating the subset masks
    Returns:
        Tuple[int, np.ndarray]: index of the claimed run and its patient subset mask
    Raises:
        StopIteration: if all runs are complete
    """
    super_mask_path = logdir / "super_mask.npy"
    valid_idcs_path = logdir / "valid_idcs.pkl"
    completed_idcs_path = logdir / "completed_idcs.pkl"
    with open(logdir / "bookkeeping.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # if the bookkeeping files exist, pick a run that hasn't been picked
        if (
            super_mask_path.exists()
            and valid_idcs_path.exists()
            and completed_idcs_path.exists()
        ):
            with open(valid_idcs_path, "r+b") as f:
                with open(completed_idcs_path, "r+b") as f2:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    super_mask = np.load(super_mask_path)
                    valid_idcs = pickle.load(f)
                    completed_idcs = pickle.load(f2)
                    if len(valid_idcs) == 0:
                        print("... No more valid subset masks left.")
                        if len(completed_idcs) < n_total_runs:
                            print("... checking for failed runs")
                            all_idcs = list(range(n_total_runs))
                            missing_idcs = list(set(all_idcs) - set(completed_idcs))
                            if len(missing_idcs) == 0:
                                print("... no failed runs found. Exiting")
                                raise StopIteration
                            else:
                                print(f"... found {len(missing_idcs)} failed runs, retrying")
                                valid_idcs = missing_idcs
                        else:
                            print("... done training. Exiting")
                            raise StopIteration
                    else:
                        print(
                            f"... processing run {valid_idcs[0]}, {len(valid_idcs)} runs left, {len(completed_idcs)} runs completed"
                        )
                    run_idx = valid_idcs.pop(0)
                    f.seek(0)
                    f.truncate()
                    # write the shortened index list before releasing the lock
                    pickle.dump(valid_idcs, f)
                    f.flush()
                    os.fsync(f.fileno())
                    fcntl.flock(f, fcntl.LOCK_UN)
        # otherwise, generate a new super mask and index files
        else:
            super_mask = generate_masks(
                n_runs=n_total_runs,
                dataset_size=n_patients,
                subset_ratio=subset_ratio,
                seed=seed,
            )
            valid_idcs = list(range(n_total_runs))
            run_idx = valid_idcs.pop(0)
            np.save(super_mask_path, super_mask)
            with open(valid_idcs_path, "w+b") as f:
                pickle.dump(valid_idcs, f)
            with open(completed_idcs_path, "w+b") as f:
                pickle.dump([], f)
    return run_idx, super_mask[run_idx]


def train_and_eval(
    compiled_model: keras.Model,
    train_historical_dataset: BaseDataset,
    val_dataset: BaseDataset,
    test_dataset: BaseDataset,
    batch_size: int,
    aug_fn: Callable,
    augment: bool,
    epochs: int,
    seed: int,
    target_metric: str,
    test_aug_fn: Optional[Callable] = None,
    subset_indices: Optional[np.ndarray] = None,
    callbacks: list = [],
    ckpt_file_path: Path = Path("./tmp/ckpt/"),
    track_data_stats: bool = True,
    overfit: bool = False,
    wandb_project_name: str = "",
    log_wandb: bool = False,
    verbose: int = 1,
    use_dp: bool = False,
    clipping_norm: Optional[float] = None,
    epsilon: Optional[float] = None,
    save_best_only: bool = False,
    microbatch_size: Optional[int] = None,
    shuffle_buffer_size: int = 0,
    num_read_workers: int = 1,
) -> Tuple[keras.Model, dict, dict, dict, dict]:
    """
    Train and evaluate a model using the given datasets and training parameters.

    Args:
        compiled_model: keras.Model, compiled model
        train_historical_dataset: BaseDataset, training dataset
        val_dataset: BaseDataset, validation dataset
        test_dataset: BaseDataset, testing dataset
        batch_size: int, batch size
        aug_fn: Callable, augmentation function
        augment: bool, whether to augment training data
        epochs: int, number of epochs to train
        seed: int, random seed
        target_metric: str, metric to monitor for checkpointing
        test_aug_fn: Optional[Callable], augmentation function for validation/test data
        subset_indices: Optional[np.ndarray], indices to select a subset of training data
        callbacks: list, list of callbacks
        ckpt_file_path: Path, directory to save model checkpoints
        track_data_stats: bool, whether to track data statistics
        overfit: bool, whether to overfit on first batch
        wandb_project_name: str, wandb project name
        log_wandb: bool, whether to log to wandb
        verbose: int, verbosity level
        use_dp: bool, whether to train with differential privacy
        clipping_norm: Optional[float], clipping norm for differential privacy
        epsilon: Optional[float], privacy budget for differential privacy
        save_best_only: bool, whether to save only the best checkpoint
        microbatch_size: Optional[int], size of the microbatches the per-sample
            gradients of a batch are computed in (DP training only). Trades
            parallelism for device memory, which allows large (effective) batch
            sizes. None means one microbatch per batch, i.e. no microbatching.
        shuffle_buffer_size: int, size of the sliding shuffle buffer for memmap
            (i.e. not in-memory) training datasets. 0 keeps on-disk order.
        num_read_workers: int, number of parallel memmap reader generators for the
            train/val/test streams (see prepare_dataset_memmap).

    Returns:
        Tuple[keras.Model, dict, dict, dict, dict]: trained model, training history,
            train metrics, val metrics, test metrics

    """
    print(f"... starting training using keras backend: {keras.backend.backend()}")
    if keras.backend.backend() == "jax":
        import jax  # type: ignore

        # Configure JAX compilation caching
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

        # Additional JAX configs to optimize compilation
        # Cache compilation results more aggressively
        try:
            cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            print(f"... jax compilation cache: {cache_dir}")
        except Exception:
            pass  # Older JAX versions may not support this

        print("... jax devices:", jax.devices())
    if log_wandb:
        if wandb_project_name == "":
            raise ValueError("Must provide a valid wandb project name")
        wandb.init(project=f"{wandb_project_name}")  # type: ignore
        wandb.config.update(flags.FLAGS)  # type: ignore
        wandb.config.update({"run_seed": seed})  # type: ignore
        wandb.config.update({"epochs": epochs})  # type: ignore
    if subset_indices is None:
        n_train = train_historical_dataset.__len__()
    else:
        n_train = len(subset_indices)
    n_val = len(val_dataset)
    print_dataset_stats(train_historical_dataset.inputs, train_historical_dataset.targets, "train")
    print_dataset_stats(val_dataset.inputs, val_dataset.targets, "val")
    print_dataset_stats(test_dataset.inputs, test_dataset.targets, "test")
    print(f"... training for {epochs} epochs ({epochs*(n_train//batch_size)} steps)")
    # prepare training and test datasets
    train_historical_dataset_prep_fn = (
        prepare_dataset if train_historical_dataset.fits_memory else prepare_dataset_memmap
    )
    train_ds = train_historical_dataset_prep_fn(
        inputs=train_historical_dataset.inputs,
        targets=train_historical_dataset.targets,
        subset_indices=subset_indices,
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
        aug_fn=aug_fn,
        shuffle_buffer_size=shuffle_buffer_size,
        num_read_workers=num_read_workers,
    )
    val_dataset_prep_fn = (
        prepare_dataset if val_dataset.fits_memory else prepare_dataset_memmap
    )
    # parallel reads reorder samples across shards, which is fine here: val_ds/test_ds only feed
    # evaluate() and fit(validation_data=...), whose metrics do not depend on sample order
    val_ds = val_dataset_prep_fn(
        inputs=val_dataset.inputs,
        targets=val_dataset.targets,
        batch_size=batch_size,
        shuffle=False,
        augment=test_aug_fn is not None,
        aug_fn=test_aug_fn,
        repeat=False,
        num_read_workers=num_read_workers,
    )
    test_dataset_prep_fn = (
        prepare_dataset if test_dataset.fits_memory else prepare_dataset_memmap
    )
    test_ds = test_dataset_prep_fn(
        inputs=test_dataset.inputs,
        targets=test_dataset.targets,
        batch_size=batch_size,
        shuffle=False,
        augment=test_aug_fn is not None,
        aug_fn=test_aug_fn,
        repeat=False,
        num_read_workers=num_read_workers,
    )
    # save a few random inputs to ./tmp for debugging purposes
    if track_data_stats:
        # create ./tmp/imgs (and ./tmp) if it does not exist yet
        tmp_dir = Path("./tmp/imgs")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            train_batch = next(iter(train_ds.take(1)))
            val_batch = next(iter(val_ds.take(1)))
            test_batch = next(iter(test_ds.take(1)))
            for i in range(16):
                train_sample = train_batch[0][i]
                test_sample = test_batch[0][i]
                if train_sample.shape[-1] in [1, 3]:
                    tf.keras.preprocessing.image.save_img(
                        f"{tmp_dir}/train_{i}.png", train_sample
                    )
                    tf.keras.preprocessing.image.save_img(
                        f"{tmp_dir}/test_{i}.png", test_sample
                    )
                elif test_sample.shape[-1] == 12:
                    lead_names = [
                        "I",
                        "II",
                        "III",
                        "aVR",
                        "aVL",
                        "aVF",
                        "V1",
                        "V2",
                        "V3",
                        "V4",
                        "V5",
                        "V6",
                    ]
                    fig, axs = plt.subplots(
                        12,
                        1,
                        figsize=(12, 12),
                        layout="tight",
                        sharex=True,
                        sharey=True,
                    )
                    for l, ax in enumerate(axs.flat):
                        ax.plot(train_sample[:, l])
                        ax.set_ylabel(f"{lead_names[l]}")
                    fig.savefig(f"{tmp_dir}/train_{i}.png")
                    plt.close(fig)
                    fig, axs = plt.subplots(
                        12,
                        1,
                        figsize=(12, 12),
                        layout="tight",
                        sharex=True,
                        sharey=True,
                    )
                    for l, ax in enumerate(axs.flat):
                        ax.plot(test_sample[:, l])
                        ax.set_ylabel(f"{lead_names[l]}")
                    fig.savefig(f"{tmp_dir}/test_{i}.png")
            train_batch_mean, train_batch_std, train_batch_min, train_batch_max = (
                tf.reduce_mean(train_batch[0]),
                tf.math.reduce_std(train_batch[0]),
                tf.reduce_min(train_batch[0]),
                tf.reduce_max(train_batch[0]),
            )
            test_batch_mean, test_batch_std, test_batch_min, test_batch_max = (
                tf.reduce_mean(test_batch[0]),
                tf.math.reduce_std(test_batch[0]),
                tf.reduce_min(test_batch[0]),
                tf.reduce_max(test_batch[0]),
            )
            val_batch_mean, val_batch_std, val_batch_min, val_batch_max = (
                tf.reduce_mean(val_batch[0]),
                tf.math.reduce_std(val_batch[0]),
                tf.reduce_min(val_batch[0]),
                tf.reduce_max(val_batch[0]),
            )
            print(
                f"... sample batch statistics (train): mean={train_batch_mean:.3f}, std={train_batch_std:.3f}, min={train_batch_min:.3f}, max={train_batch_max:.3f}"
            )
            print(
                f"... sample batch statistics (val): mean={val_batch_mean:.3f}, std={val_batch_std:.3f}, min={val_batch_min:.3f}, max={val_batch_max:.3f}"
            )
            print(
                f"... sample batch statistics (test): mean={test_batch_mean:.3f}, std={test_batch_std:.3f}, min={test_batch_min:.3f}, max={test_batch_max:.3f}"
            )
            if log_wandb:
                wandb.log(
                    {
                        "train_sample_batch_mean": train_batch_mean,
                        "train_sample_batch_std": train_batch_std,
                        "train_sample_batch_min": train_batch_min,
                        "train_sample_batch_max": train_batch_max,
                        "test_sample_batch_mean": test_batch_mean,
                        "test_sample_batch_std": test_batch_std,
                        "test_sample_batch_min": test_batch_min,
                        "test_sample_batch_max": test_batch_max,
                    }
                )  # type: ignore

        except Exception as e:
            print(f"... error while saving images: {e}")
            print(
                "... skipping image saving, this is probably due to the dataset not being image-based"
            )
    if overfit:

        def first_batch_only(dataset: tf.data.Dataset):
            batch = next(iter(dataset))
            total_steps = len(dataset)
            i = 0
            while i < total_steps:
                yield batch
                i += 1

        print("... delibaretely overfitting on first batch. Careful!")
        train_ds = first_batch_only(train_ds)
    print(f"... training on {n_train} samples, evaluating on {n_val} samples")
    if use_dp:
        from jax_privacy import keras_api # type: ignore

        from src.training.dp import is_nonprivate, make_dp_config

        train_steps = n_train // batch_size * epochs
        # epsilon=inf is the non-private arm: same train step, noise multiplier pinned to
        # 0 rather than calibrated. It goes through make_private() like every other arm so
        # that the eps axis varies sigma and nothing else -- see src/training/dp.py.
        params = make_dp_config(
            epsilon=epsilon,
            delta=1 / len(train_historical_dataset),
            clipping_norm=clipping_norm,
            batch_size=batch_size,
            gradient_accumulation_steps=1,
            rescale_to_unit_norm=False,
            train_steps=train_steps,
            train_size=n_train,
            seed=seed,
            microbatch_size=microbatch_size,
        )
        compiled_model = keras_api.make_private(compiled_model, params)
        if is_nonprivate(epsilon):
            print(
                "... epsilon=inf: DP-SGD train step with the noise switched off "
                f"(sigma=0), per-example clipping still at C={clipping_norm}"
            )
        print("DP params:", params)
        # update wandb config with dp params
        if log_wandb:
            print("... updating wandb config with DP params")
            if compiled_model._dp_params.noise_multiplier is None:
                # sometimes the jax_privacy keras_api does not correctly update the field so we recompute the noise multiplier
                noise_mult = (
                    compiled_model._dp_params.update_with_calibrated_noise_multiplier().noise_multiplier
                )
            else:
                noise_mult = compiled_model._dp_params.noise_multiplier
            dp_params = {
                # wandb's config serialiser has no JSON representation for inf, so the
                # non-private arm records its budget as the string "inf".
                "actual_epsilon": (
                    "inf"
                    if is_nonprivate(compiled_model._dp_params.epsilon)
                    else compiled_model._dp_params.epsilon
                ),
                "clipping_norm": compiled_model._dp_params.clipping_norm,
                "delta": compiled_model._dp_params.delta,
                "train_size": compiled_model._dp_params.train_size,
                "noise_multiplier": noise_mult,
                "train_steps": compiled_model._dp_params.train_steps,
                "effective_noise_scale": noise_mult * clipping_norm / batch_size,
            }
            wandb.config.update(dp_params)  # type: ignore
    else:
        print("... training without differential privacy (no clipping, no noise)")
    if log_wandb:
        # record the accelerator type in the run config
        try:
            import jax  # type: ignore

            device_kind = jax.devices()[0].device_kind
        except Exception as e:  # pragma: no cover - purely informational
            print(f"... could not determine device kind: {e}")
            device_kind = "unknown"
        wandb.config.update({"device_kind": str(device_kind)})  # type: ignore

    start_time = time.time()

    # model checks
    assert isinstance(compiled_model, keras.Model), "Model must be a keras model"
    assert (
        compiled_model.compiled
    ), f"Model must be compiled!, compilation status: {compiled_model.compiled}"
    print(compiled_model.summary())
    callbacks.append(keras.callbacks.TerminateOnNaN())
    wandb_callbacks = [WandbLogger(model=compiled_model)] if log_wandb else []
    if ckpt_file_path is not None:
        if log_wandb:
            # if using wandb, save model weights with wandb run id to avoid overwriting issues with multiple parallel runs
            ckpt_file_path = ckpt_file_path / f"model_{wandb.run.id}.weights.h5"  # type: ignore
        else:
            ckpt_file_path = ckpt_file_path / "model.weights.h5"
        print("... creating model checkpoint callback at ", ckpt_file_path)
        ckpt_callback = keras.callbacks.ModelCheckpoint(
            filepath=ckpt_file_path,
            save_weights_only=True,
            save_best_only=save_best_only,
            monitor=target_metric,
            mode="min" if "loss" in target_metric else "max",
            verbose=1,
        )
        callbacks.append(ckpt_callback)
    callbacks = wandb_callbacks + callbacks
    try:
        val_steps = n_val // batch_size
        val_steps = 1 if val_steps == 0 else val_steps
        training_history = compiled_model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            #verbose=verbose,
            callbacks=callbacks,
            steps_per_epoch=n_train // batch_size,
            validation_steps=val_steps,
        )
    except KeyboardInterrupt:
        if ckpt_file_path is not None:
            print(f"... deleting checkpoint file: {ckpt_file_path}")
            del ckpt_callback
            os.remove(ckpt_file_path)
    training_time = (time.time() - start_time) / 60  # training time in minutes
    print(f"... finished training in {training_time} mins")
    # evaluation
    if ckpt_file_path is not None:
        if os.path.exists(ckpt_file_path):
            # load best weights
            compiled_model.load_weights(ckpt_file_path)
            print(f"... deleting checkpoint file: {ckpt_file_path}")
            del ckpt_callback
            os.remove(ckpt_file_path)
        else:
            # no checkpoint is written when a run diverges and TerminateOnNaN stops it before the
            # first monitored metric is available; evaluate the final weights instead
            print(f"WARNING: no checkpoint found at {ckpt_file_path}, evaluating final weights")
    train_metrics = compiled_model.evaluate(
        train_ds, verbose=verbose, return_dict=True, steps=n_train // batch_size
    )
    val_metrics = compiled_model.evaluate(
        val_ds, verbose=verbose, return_dict=True, steps=val_steps
    )
    test_steps = len(test_dataset) // batch_size
    test_steps = 1 if test_steps == 0 else test_steps
    test_metrics = compiled_model.evaluate(
        test_ds,
        verbose=verbose,
        return_dict=True,
        steps=test_steps,
    )
    print(f"... test metrics: {test_metrics}")
    if log_wandb:
        train_metrics = {f"_{k}": v for k, v in train_metrics.items()} | {
            "training_time": training_time
        }
        val_metrics = {f"_val_{k}": v for k, v in val_metrics.items()} | {
            "training_time": training_time
        }
        test_metrics = {f"_test_{k}": v for k, v in test_metrics.items()} | {
            "training_time": training_time
        }
        wandb.log(train_metrics)  # type: ignore
        wandb.log(val_metrics)  # type: ignore
        wandb.log(test_metrics)  # type: ignore
    return (
        compiled_model,
        training_history,
        train_metrics,
        val_metrics,
        test_metrics,
    )


def train_random_subset(
    compiled_model: keras.Model,
    train_historical_dataset: BaseDataset,
    val_dataset: BaseDataset,
    train_future_dataset: BaseDataset,
    test_dataset: BaseDataset,
    patient_id_col: str,
    batch_size: int,
    aug_fn: Callable,
    augment: bool,
    epochs: int,
    seed: int,
    target_metric: str,
    logdir: Path,
    test_aug_fn: Optional[Callable] = None,
    ckpt_file_path: Path = Path("./tmp/ckpt/"),
    n_total_runs: int = 150,
    subset_ratio: float = 0.5,
    n_eval_views: int = 1,
    callbacks: list = [],
    track_data_stats: bool = True,
    overfit: bool = False,
    wandb_project_name: str = "",
    log_wandb: bool = False,
    verbose: int = 1,
    use_dp: bool = False,
    clipping_norm: Optional[float] = None,
    epsilon: Optional[float] = None,
    save_best_only: bool = False,
    microbatch_size: Optional[int] = None,
    shuffle_buffer_size: int = 0,
    num_read_workers: int = 1,
):
    """
    Train a model on a random subset of train_historical_dataset. Logs to wandb and saves logits and corresponding labels of train, long_eval and test dataset to logdir.

    Args:
        compiled_model: keras.Model, compiled model
        train_historical_dataset: BaseDataset, training dataset
        val_dataset: BaseDataset, validation dataset
        train_future_dataset: BaseDataset, future/long evaluation dataset
        test_dataset: BaseDataset, testing dataset
        patient_id_col: str, column name for patient IDs
        batch_size: int, batch size
        aug_fn: Callable, augmentation function
        augment: bool, whether to augment training data
        epochs: int, number of epochs to train
        seed: int, random seed
        target_metric: str, metric to monitor for checkpointing
        logdir: Path, directory to save logs
        test_aug_fn: Optional[Callable], augmentation function for validation/test data
        ckpt_file_path: Path, directory to save model checkpoints
        n_total_runs: int, total number of runs
        subset_ratio: float, ratio of subset
        n_eval_views: int, number of augmented views for evaluation
        callbacks: list, list of callbacks
        track_data_stats: bool, whether to track data statistics
        overfit: bool, whether to overfit on first batch
        wandb_project_name: str, wandb project name
        log_wandb: bool, whether to log to wandb
        verbose: int, verbosity level
        use_dp: bool, whether to train with differential privacy
        clipping_norm: Optional[float], clipping norm for differential privacy
        epsilon: Optional[float], privacy budget for differential privacy
        save_best_only: bool, whether to save only the best checkpoint
        microbatch_size: Optional[int], microbatch size for per-sample gradient
            computation (DP training only), see train_and_eval
        shuffle_buffer_size: int, size of the sliding shuffle buffer for memmap
            training datasets, see train_and_eval
        num_read_workers: int, number of parallel memmap reader generators for the
            training/evaluation streams, see train_and_eval. The logit dumps below
            always use a single reader because their row order has to match the
            dataframe.

    Returns:
        None
    """
    assert logdir is not None, "Must provide a logdir to save logs"
    assert isinstance(logdir, Path), "logdir must be a pathlib.Path object"
    keras.utils.clear_session(
        free_memory=True
    )  # clear keras session from a potential previous run
    logdir.mkdir(parents=True, exist_ok=True)
    # extract unique patient ids
    patient_ids = pd.Series(train_historical_dataset.dataframe[patient_id_col].unique())
    # claim the next run (index and patient subset mask)
    completed_idcs_path = logdir / "completed_idcs.pkl"
    next, subset_mask = claim_subset(
        logdir=logdir,
        n_patients=len(patient_ids),
        n_total_runs=n_total_runs,
        subset_ratio=subset_ratio,
        seed=seed,
    )
    assert len(subset_mask) == len(
        patient_ids
    ), "Subset mask shape must match number of patients"
    subset_idcs = np.nonzero(subset_mask)[0]
    selected_patients = patient_ids[subset_idcs]
    # convert patient-level subset mask to record-level mask
    record_pids = train_historical_dataset.dataframe[patient_id_col]
    record_subset_mask = train_historical_dataset.dataframe[patient_id_col].isin(selected_patients)
    record_subset_idcs = np.nonzero(record_subset_mask)[0]
    assert len(record_subset_idcs) < len(
        record_pids
    ), "Subset mask must be smaller than full dataset"
    # check whether record selection by patient worked correctly
    assert set(selected_patients) == set(
        train_historical_dataset.dataframe.iloc[record_subset_idcs][patient_id_col]
        .unique()
        .tolist()
    ), "Patient ids from selected subset do not match selected patients"
    if not isinstance(logdir, Path) and logdir is not None:
        logdir = Path(logdir)
    logger = RetrainLogger(
        logdir=logdir, patient_ids=patient_ids, patient_subset_mask=subset_mask
    )
    try:
        (
            compiled_model,
            training_history,
            train_metrics,
            val_metrics,
            test_metrics,
        ) = train_and_eval(
            compiled_model=compiled_model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            subset_indices=record_subset_idcs,
            batch_size=batch_size,
            aug_fn=aug_fn,
            test_aug_fn=test_aug_fn,
            augment=augment,
            epochs=epochs,
            seed=seed,
            target_metric=target_metric,
            ckpt_file_path=ckpt_file_path,
            callbacks=callbacks,
            track_data_stats=track_data_stats,
            overfit=overfit,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
            verbose=verbose,
            use_dp=use_dp,
            clipping_norm=clipping_norm,
            epsilon=epsilon,
            save_best_only=save_best_only,
            microbatch_size=microbatch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            num_read_workers=num_read_workers,
        )
    except Exception:
        # everything below depends on train_and_eval() having returned, so report the failure
        # where it happened and re-raise
        print("ERROR: training failed, aborting before the logit dump:")
        traceback.print_exc()
        raise
    # save logits and labels of train as well as long_eval dataset
    if log_wandb:
        print("... fininished training, saving logits and labels")
        augment_test = True if test_aug_fn is not None else False
        train_historical_dataset_prep_fn = (
            prepare_dataset if train_historical_dataset.fits_memory else prepare_dataset_memmap
        )
        train_ds = train_historical_dataset_prep_fn(
            inputs=train_historical_dataset.inputs,  # full training dataset
            targets=train_historical_dataset.targets,
            batch_size=batch_size,
            shuffle=False,
            augment=augment,
            aug_fn=aug_fn,
            drop_remainder=False,
            repeat=False,
        )
        train_future_dataset_prep_fn = (
            prepare_dataset
            if train_future_dataset.fits_memory
            else prepare_dataset_memmap
        )
        long_eval_ds = train_future_dataset_prep_fn(
            inputs=train_future_dataset.inputs,
            targets=train_future_dataset.targets,
            batch_size=batch_size,
            shuffle=False,
            augment=augment_test,
            aug_fn=test_aug_fn,
            drop_remainder=False,
            repeat=False,
        )
        test_dataset_prep_fn = (
            prepare_dataset if test_dataset.fits_memory else prepare_dataset_memmap
        )
        test_ds = test_dataset_prep_fn(
            inputs=test_dataset.inputs,
            targets=test_dataset.targets,
            batch_size=batch_size,
            shuffle=False,
            augment=augment_test,
            aug_fn=test_aug_fn,
            drop_remainder=False,
            repeat=False,
        )
        # set dtype policy to float32
        keras.config.set_dtype_policy("float32")
        # save logits and labels
        train_steps = (
            int(math.ceil(len(train_historical_dataset) / batch_size))
            if not train_historical_dataset.fits_memory
            else None
        )
        long_eval_steps = (
            int(math.ceil(len(train_future_dataset) / batch_size))
            if not train_future_dataset.fits_memory
            else None
        )
        test_steps = (
            int(math.ceil(len(test_dataset) / batch_size))
            if not test_dataset.fits_memory
            else None
        )
        if n_eval_views > 1:
            train_logits, long_eval_logits, test_logits = [], [], []
            for _ in range(n_eval_views):
                train_logits.append(
                    compiled_model.predict(train_ds, steps=train_steps)
                )
                long_eval_logits.append(
                    compiled_model.predict(long_eval_ds, steps=long_eval_steps)
                )
                test_logits.append(
                    compiled_model.predict(test_ds, steps=test_steps)
                )
                # check we didn't lose any samples
                assert len(train_logits[-1]) == len(
                    train_historical_dataset
                ), f"Mismatch in number of train logits, expected {len(train_historical_dataset)}, got {len(train_logits[-1])}"
                assert len(long_eval_logits[-1]) == len(
                    train_future_dataset
                ), f"Mismatch in number of long eval logits, expected {len(train_future_dataset)}, got {len(long_eval_logits[-1])}"
                assert len(test_logits[-1]) == len(
                    test_dataset
                ), f"Mismatch in number of test logits, expected {len(test_dataset)}, got {len(test_logits[-1])}"
            train_logit_arr = np.stack(train_logits, axis=1)
            long_eval_logit_arr = np.stack(long_eval_logits, axis=1)
            test_logit_arr = np.stack(test_logits, axis=1)
        else:
            # check we didn't lose any samples
            train_logit_arr = compiled_model.predict(train_ds, steps=train_steps)
            long_eval_logit_arr = compiled_model.predict(
                long_eval_ds, steps=long_eval_steps
            )
            test_logit_arr = compiled_model.predict(test_ds, steps=test_steps)
            assert len(train_logit_arr) == len(
                train_historical_dataset
            ), f"Mismatch in number of train logits, expected {len(train_historical_dataset)}, got {len(train_logit_arr)}"
            assert len(long_eval_logit_arr) == len(
                train_future_dataset
            ), f"Mismatch in number of long eval logits, expected {len(train_future_dataset)}, got {len(long_eval_logit_arr)}"
            assert len(test_logit_arr) == len(
                test_dataset
            ), f"Mismatch in number of test logits, expected {len(test_dataset)}, got {len(test_logit_arr)}"

        print(
            f"... saving logits to disk train={train_logit_arr.shape}, long_eval={long_eval_logit_arr.shape} and test={test_logit_arr.shape}"
        )
        success = logger.log(
            config=dict(wandb.config),  # type: ignore
            train_logits=train_logit_arr,
            long_eval_logits=long_eval_logit_arr,
            test_logits=test_logit_arr,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
        )
        if success:
            # add current index to completed indices
            with open(completed_idcs_path, "r+b") as f2:
                print(f"... succesfully completed run: {next}")
                # lock file to prevent race condition
                fcntl.flock(f2, fcntl.LOCK_EX)
                # re-load completed indices in case another process has modified it
                completed_idcs = pickle.load(f2)
                completed_idcs.append(next)
                f2.seek(0)
                f2.truncate()
                # same ordering as above: write, then release the lock
                pickle.dump(completed_idcs, f2)
                f2.flush()
                os.fsync(f2.fileno())
                # unlock file
                fcntl.flock(f2, fcntl.LOCK_UN)
    wandb.run.finish()  # type: ignore


def train_and_eval_sklearn(
    sklearn_model: BaseEstimator,
    train_historical_dataset: BaseDataset,
    val_dataset: BaseDataset,
    test_dataset: BaseDataset,
    seed: int,
    target_metric: str,  # for API compatibility only
    subset_indices: Optional[np.ndarray] = None,
    flatten_input: bool = False,
    track_data_stats: bool = True,
    wandb_project_name: str = "",
    log_wandb: bool = False,
    verbose: bool = True,
    evaluation_fn: Optional[Callable] = None,
) -> Tuple[object, dict, dict, dict, dict, bool]:
    """
    Train and evaluate a sklearn model using the given datasets and training parameters.

    Args:
        sklearn_model: sklearn model instance (will be cloned for training)
        train_historical_dataset: Training dataset
        test_dataset: Testing dataset
        seed: Random seed
        target_metric: Target metric name (for logging compatibility)
        subset_indices: Optional indices to select data subset
        flatten_input: Whether to flatten input data for traditional sklearn models
        track_data_stats: Whether to track data statistics
        wandb_project_name: wandb project name
        log_wandb: Whether to log to wandb
        verbose: verbosity setting
        evaluation_fn: Optional function to compute custom metrics. Signature: fn(model, X_train, y_train, X_test, y_test) -> dict

    Returns:
        Tuple[object, dict, dict, dict, bool]: trained model, training history (empty), train metrics, test metrics, failed flag
    """
    np.random.seed(seed)
    print(f"... starting sklearn training with model: {type(sklearn_model).__name__}")

    if log_wandb:
        if wandb_project_name == "":
            raise ValueError("Must provide a valid wandb project name")
        wandb.init(project=f"{wandb_project_name}")  # type: ignore
        wandb.config.update(flags.FLAGS)  # type: ignore
        wandb.config.update({"run_seed": seed})  # type: ignore
        wandb.config.update({"model_class": type(sklearn_model).__name__})  # type: ignore

    if subset_indices is None:
        n_train = train_historical_dataset.__len__()
    else:
        n_train = len(subset_indices)
    n_test = len(test_dataset)

    print_dataset_stats(train_historical_dataset.inputs, train_historical_dataset.targets, "train")
    print_dataset_stats(val_dataset.inputs, val_dataset.targets, "val")
    print_dataset_stats(test_dataset.inputs, test_dataset.targets, "test")

    # prepare training and test datasets - no augmentation for sklearn
    X_train, y_train = prepare_dataset_sklearn(
        inputs=train_historical_dataset.inputs,
        targets=train_historical_dataset.targets,
        subset_indices=subset_indices,
        shuffle=True,
        flatten_input=flatten_input,
    )
    X_val, y_val = prepare_dataset_sklearn(
        inputs=val_dataset.inputs,
        targets=val_dataset.targets,
        shuffle=False,
        flatten_input=flatten_input,
    )
    X_test, y_test = prepare_dataset_sklearn(
        inputs=test_dataset.inputs,
        targets=test_dataset.targets,
        shuffle=False,
        flatten_input=flatten_input,
    )

    print(f"... training on {n_train} samples, evaluating on {n_test} samples")

    # Clone the model to avoid modifying the original
    model = clone(sklearn_model)

    start_time = time.time()
    failed = False

    try:
        # Train the model
        print(
            f"... fitting sklearn model with data shape: inputs={X_train.shape}, targets={y_train.shape}"
        )
        model.fit(X_train, y_train)

    except Exception as e:
        print(f"... sklearn training failed: {e}")
        failed = True

    training_time = (time.time() - start_time) / 60  # training time in minutes
    print(f"... finished training in {training_time:.2f} mins")

    # Evaluation - compute custom metrics if evaluation function provided
    train_metrics, val_metrics, test_metrics = {}, {}, {}

    if not failed:
        if evaluation_fn is not None:
            try:
                print("... computing custom metrics")
                train_metrics = evaluation_fn(
                    model=model, data=(X_train, y_train), prefix="train"
                )
                print(f"train metrics: {train_metrics}") if verbose else None
                val_metrics = evaluation_fn(
                    model=model, data=(X_val, y_val), prefix="val"
                )
                print(f"val metrics: {val_metrics}") if verbose else None
                test_metrics = evaluation_fn(
                    model=model, data=(X_test, y_test), prefix="test"
                )
                print(f"test metrics: {test_metrics}") if verbose else None

            except Exception as e:
                print(f"... error computing custom metrics: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

        if verbose:
            print("... model training completed successfully")
            print(
                f"... train data shape: {X_train.shape}, test data shape: {X_test.shape}"
            )
            print(
                f"... train targets shape: {y_train.shape}, test targets shape: {y_test.shape}"
            )

    if log_wandb:
        wandb_metrics = {"training_time": training_time}
        if train_metrics:
            wandb_metrics.update(train_metrics)
        wandb.log(wandb_metrics)  # type: ignore

    return model, {}, train_metrics, val_metrics, test_metrics, failed


def train_random_subset_sklearn(
    sklearn_model: BaseEstimator,
    train_historical_dataset: BaseDataset,
    val_dataset: BaseDataset,
    train_future_dataset: BaseDataset,
    test_dataset: BaseDataset,
    patient_id_col: str,
    seed: int,
    target_metric: str,
    logdir: Path,
    flatten_input: bool = False,
    n_total_runs: int = 150,
    subset_ratio: float = 0.5,
    track_data_stats: bool = True,
    wandb_project_name: str = "",
    log_wandb: bool = False,
    verbose: bool = True,
    evaluation_fn: Optional[
        Callable[[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict]
    ] = None,
):
    """
    Train a sklearn model on a random subset of train_historical_dataset. Logs to wandb and saves predictions
    and corresponding labels of train, long_eval and test dataset to logdir.

    Args:
        sklearn_model: sklearn model instance
        train_historical_dataset: Training dataset
        train_future_dataset: Long evaluation dataset
        test_dataset: Testing dataset
        patient_id_col: Column name for patient IDs
        seed: Random seed
        target_metric: Target metric name
        logdir: Directory to save logs
        flatten_input: Whether to flatten input data for traditional sklearn models
        n_total_runs: Total number of runs
        subset_ratio: Ratio of subset
        track_data_stats: Whether to track data statistics
        wandb_project_name: wandb project name
        log_wandb: Whether to log to wandb
        verbose: Whether to print verbose logs
        evaluation_fn: Optional function to compute custom metrics. Signature: fn(model, X_train, y_train, X_test, y_test) -> dict

    Returns:
        object: Trained sklearn model, or None if training failed
    """
    assert logdir is not None, "Must provide a logdir to save logs"
    assert isinstance(logdir, Path), "logdir must be a pathlib.Path object"

    logdir.mkdir(parents=True, exist_ok=True)
    # extract unique patient ids
    patient_ids = pd.Series(train_historical_dataset.dataframe[patient_id_col].unique())
    # claim the next run (index and patient subset mask)
    valid_idcs_path = logdir / "valid_idcs.pkl"
    completed_idcs_path = logdir / "completed_idcs.pkl"
    next, subset_mask = claim_subset(
        logdir=logdir,
        n_patients=len(patient_ids),
        n_total_runs=n_total_runs,
        subset_ratio=subset_ratio,
        seed=seed,
    )

    assert len(subset_mask) == len(
        patient_ids
    ), "Subset mask shape must match number of patients"
    subset_idcs = np.nonzero(subset_mask)[0]
    selected_patients = patient_ids[subset_idcs]
    # convert patient-level subset mask to record-level mask
    record_pids = train_historical_dataset.dataframe[patient_id_col]
    record_subset_mask = train_historical_dataset.dataframe[patient_id_col].isin(selected_patients)
    record_subset_idcs = np.nonzero(record_subset_mask)[0]
    assert len(record_subset_idcs) < len(
        record_pids
    ), "Subset mask must be smaller than full dataset"
    # check whether record selection by patient worked correctly
    assert set(selected_patients) == set(
        train_historical_dataset.dataframe.iloc[record_subset_idcs][patient_id_col]
        .unique()
        .tolist()
    ), "Patient ids from selected subset do not match selected patients"

    if not isinstance(logdir, Path) and logdir is not None:
        logdir = Path(logdir)
    logger = RetrainLogger(
        logdir=logdir, patient_ids=patient_ids, patient_subset_mask=subset_mask
    )
    exception_raised = False

    try:
        (
            model,
            training_history,
            train_metrics,
            val_metrics,
            test_metrics,
            run_failed,
        ) = train_and_eval_sklearn(
            sklearn_model=sklearn_model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            subset_indices=record_subset_idcs,
            seed=seed,
            target_metric=target_metric,
            flatten_input=flatten_input,
            track_data_stats=track_data_stats,
            wandb_project_name=wandb_project_name,
            log_wandb=log_wandb,
            verbose=verbose,
            evaluation_fn=evaluation_fn,
        )
    except Exception as e:
        print(f"Error: {e}")
        exception_raised = True

    if exception_raised or run_failed:
        print(f"... run failed, adding {next} back to valid indices")
        # re-add supermask index to valid indices if training failed
        with open(valid_idcs_path, "r+b") as f:
            # lock file to prevent race condition
            fcntl.flock(f, fcntl.LOCK_EX)
            valid_idcs = pickle.load(f)
            valid_idcs.append(next)
            f.seek(0)
            f.truncate()
            # same ordering as above: write, then release the lock
            pickle.dump(valid_idcs, f)
            f.flush()
            os.fsync(f.fileno())
            # unlock file
            fcntl.flock(f, fcntl.LOCK_UN)
        wandb.run.finish()  # type: ignore
        raise RuntimeError("Run failed, model not returned")
    else:
        # save predictions and labels of train as well as long_eval dataset
        if log_wandb:
            # Prepare all datasets for prediction
            X_train_full, y_train_full = prepare_dataset_sklearn(
                inputs=train_historical_dataset.inputs,
                targets=train_historical_dataset.targets,
                shuffle=False,
                flatten_input=flatten_input,
            )
            X_long_eval, y_long_eval = prepare_dataset_sklearn(
                inputs=train_future_dataset.inputs,
                targets=train_future_dataset.targets,
                shuffle=False,
                flatten_input=flatten_input,
            )
            X_test, y_test = prepare_dataset_sklearn(
                inputs=test_dataset.inputs,
                targets=test_dataset.targets,
                shuffle=False,
                flatten_input=flatten_input,
            )

            print("... finished training, saving predicted probabilities and labels")
            assert hasattr(
                model, "predict_proba"
            ), "Model must support probability predictions via predict_proba"
            n_classes = y_train_full.shape[1]
            train_proba_list = model.predict_proba(X_train_full)
            long_eval_proba_list = model.predict_proba(X_long_eval)
            test_proba_list = model.predict_proba(X_test)
            # Stack probabilities into (n_records, n_classes) array - take positive class probability for each label
            train_preds = np.column_stack(
                [train_proba_list[i][:, 1] for i in range(len(train_proba_list))]
            )
            long_eval_preds = np.column_stack(
                [
                    long_eval_proba_list[i][:, 1]
                    for i in range(len(long_eval_proba_list))
                ]
            )
            test_preds = np.column_stack(
                [test_proba_list[i][:, 1] for i in range(len(test_proba_list))]
            )
            assert train_preds.shape == (
                X_train_full.shape[0],
                n_classes,
            ), f"Expected train predictions shape {(X_train_full.shape[0], n_classes)}, got {train_preds.shape}"
            assert long_eval_preds.shape == (
                X_long_eval.shape[0],
                n_classes,
            ), f"Expected long eval predictions shape {(X_long_eval.shape[0], n_classes)}, got {long_eval_preds.shape}"
            assert test_preds.shape == (
                X_test.shape[0],
                n_classes,
            ), f"Expected test predictions shape {(X_test.shape[0], n_classes)}, got {test_preds.shape}"
            print(
                f"... saving predictions to disk train={train_preds.shape}, long_eval={long_eval_preds.shape} and test={test_preds.shape}"
            )
            success = logger.log(
                config=dict(wandb.config),  # type: ignore
                train_logits=train_preds,
                long_eval_logits=long_eval_preds,
                test_logits=test_preds,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
            if success:
                # add current index to completed indices
                with open(completed_idcs_path, "r+b") as f2:
                    print(f"... successfully completed run: {next}")
                    # lock file to prevent race condition
                    fcntl.flock(f2, fcntl.LOCK_EX)
                    # re-load completed indices in case another process has modified it
                    completed_idcs = pickle.load(f2)
                    completed_idcs.append(next)
                    f2.seek(0)
                    f2.truncate()
                    # write the updated list before releasing the lock, as in the keras version
                    pickle.dump(completed_idcs, f2)
                    f2.flush()
                    os.fsync(f2.fileno())
                    # unlock file
                    fcntl.flock(f2, fcntl.LOCK_UN)
            wandb.run.finish()  # type: ignore
    return model  # Return the trained model
