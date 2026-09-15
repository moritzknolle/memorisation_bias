"""Training with DP-SGD on HEEDB.

Same modes and outputs as heedb.py, with DP-SGD (jax_privacy) applied in train_and_eval. See
mimic-ecg_dp.py for --eps, --eps=inf and --one_record_per_patient.

Usage:
    python heedb_dp.py --train_random_subset=True --eps=100 --logdir=<log_root>/dp/eps100
    python heedb_dp.py --train_random_subset=True --eps=inf --logdir=<log_root>/nonprivate
"""

import os
from pathlib import Path

# set keras backend to jax and enable compilation caching
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")

import keras  # type: ignore
import numpy as np
from absl import app, flags  # type: ignore

from src.data.dataset_factory import get_dataset
from src.training.models.model_factory import get_model
from src.training.training import train_and_eval, train_random_subset
from src.training.decay import MyCosineDecay, WarmupReduceLROnPlateau
from src.data.constants import HEEDB_CODE_DICT

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 20, "Number of training epochs.")
flags.DEFINE_float("lr", 3e-3, "Learning rate.")
flags.DEFINE_integer("batch_size", 8192, "Batch size. Must be divisible by --microbatch_size.")
flags.DEFINE_float(
    "eps",
    100.0,
    "epsilon of (eps, delta)-differential privacy. `inf` trains the non-private baseline: the "
    "same DP training step with per-example clipping at --C and the noise multiplier set to 0.",
)
flags.DEFINE_float("C", 1.0, "Clipping threshold for the per-example L2 gradient norm.")
flags.DEFINE_boolean(
    "one_record_per_patient",
    False,
    "Whether to use only one ECG per patient in the training set. Required for patient-level DP guarantees.",
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_boolean(
    "ema", True, "Whether to use exponential moving average for parameters."
)
flags.DEFINE_string("model", "vit1d_tiny_25_dp", "Name of the model to use.")
flags.DEFINE_enum("lr_schedule", "cosine", ["constant", "cosine"], "LR schedule.")
flags.DEFINE_float(
    "lr_warmup",
    0.1,
    "Relative fraction of steps to perform linear learning rate warmup.",
)
flags.DEFINE_float(
    "decay_steps",
    1.0,
    "Relative fraction of total steps until learning rate is decayed to 1/10 times the original value. A value smaller than one means faster decay and likewise a bigger value leads to slower decay.",
)
flags.DEFINE_float("ema_decay", 0.99, "EMA decay.")
flags.DEFINE_boolean(
    "mixed_precision",
    False,
    "Whether to perform mixed precision training. Disabled by default for DP training as float16 can cause divergence with DP noise injection.",
)
flags.DEFINE_enum(
    "precision",
    "default",
    ["default", "float32", "mixed_bfloat16", "mixed_float16"],
    "Global keras dtype policy. 'default' defers to --mixed_precision.",
)
flags.DEFINE_integer(
    "microbatch_size",
    128,
    "Number of samples per microbatch for per-example gradient computation. 0 means one "
    "microbatch per batch.",
)
flags.DEFINE_integer(
    "shuffle_buffer_size",
    0,
    "Size of the sliding shuffle buffer over the memmap training stream. 0 keeps on-disk order.",
)
flags.DEFINE_integer("num_read_workers", 1, "Number of parallel generators reading the memmap streams.")
flags.DEFINE_float(
    "train_subset_frac",
    1.0,
    "Fraction of the training records to use with --train_random_subset=False.",
)
flags.DEFINE_string("wandb_project", "heedb_dp", "Weights & Biases project to log to.")
flags.DEFINE_bool(
    "train_random_subset",
    False,
    "Whether to train on random subset for privacy auditing (True) or train on the full dataset (False).",
)
flags.DEFINE_bool(
    "use_dp",
    True,
    "Whether to train with DP-SGD. --use_dp=False trains without jax_privacy, using gradient "
    "accumulation to reach --batch_size. This is not the same as --eps=inf.",
)
flags.DEFINE_integer(
    "nondp_batch_size",
    512,
    "Physical batch size for --use_dp=False; gradients are accumulated over "
    "--batch_size/--nondp_batch_size steps.",
)
flags.DEFINE_string("data_root", "./data/raw/heedb/WFDB", "Path to the HEEDB WFDB files.")
flags.DEFINE_string("csv_root", "./data/csv", "Path to the split files.")
flags.DEFINE_integer(
    "n_runs", 200, "Number of leave-many-out re-training runs to perform."
)
flags.DEFINE_float(
    "subset_ratio", 0.5, "Ratio of the training data to use for each re-training run."
)
flags.DEFINE_string(
    "ckpt_file_path",
    "./tmp/ckpts/",
    "Path to root folder where the model checkpoint files are stored.",
)
flags.DEFINE_string(
    "save_root",
    "./data/npy/",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_string(
    "logdir",
    "./logs/heedb_dp/",
    "Path to logdir.",
)
flags.DEFINE_bool(
    "save_best_only",
    True,
    "Whether to save only the best model checkpoint based on the target metric.",
)
flags.DEFINE_integer(
    "n_threads",
    16,
    "Number of threads to use for parallel data loading and preprocessing.",
)
flags.DEFINE_integer(
    "chunk_size",
    4096,
    "Number of samples per chunk for multiprocessing.",
)
flags.DEFINE_boolean(
    "write_to_disk",
    False,
    "Whether to write the dataset to disk.",
)


def main(argv):
    np.random.seed(FLAGS.seed)
    if FLAGS.precision != "default":
        keras.mixed_precision.set_global_policy(FLAGS.precision)
    elif FLAGS.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"... keras dtype policy: {keras.mixed_precision.global_policy().name}")
    NUM_CLASSES = len(HEEDB_CODE_DICT)
    (
        train_historical_dataset, val_dataset, train_future_dataset, test_dataset,
    ) = get_dataset(
        dataset_name="heedb_p" if FLAGS.one_record_per_patient else "heedb",
        csv_root=Path(FLAGS.csv_root),
        data_root=Path(FLAGS.data_root),
        save_root=Path(FLAGS.save_root),
        use_cached=True,
        load_from_disk=True,
        write_to_disk=FLAGS.write_to_disk,
        n_threads=FLAGS.n_threads,
        chunk_size=FLAGS.chunk_size,
    )
    print(f"x_train.shape: {train_historical_dataset.inputs.shape}, y_train.shape: {train_historical_dataset.targets.shape}")
    print(f"x_val.shape: {val_dataset.inputs.shape}, y_val.shape: {val_dataset.targets.shape}")
    print(f"x_test.shape: {test_dataset.targets.shape}, y_test.shape: {test_dataset.targets.shape}")
    print(
        f"x_long_eval.shape: {train_future_dataset.inputs.shape}, y_long_eval.shape: {train_future_dataset.targets.shape}"
    )
    sample_batch = train_historical_dataset.inputs[0:FLAGS.batch_size]
    print(f"sample batch stats: {sample_batch.min()}, {sample_batch.max()}, {sample_batch.mean()}, {sample_batch.std()}")

    # optional proxy training budget: a fixed random subset of records, kept in on-disk
    # order so that memmap reads stay (strided) sequential
    train_subset_indices = None
    if not FLAGS.train_random_subset and FLAGS.train_subset_frac < 1.0:
        assert FLAGS.train_subset_frac > 0.0, "--train_subset_frac must be in (0, 1]"
        n_subset = int(FLAGS.train_subset_frac * len(train_historical_dataset))
        train_subset_indices = np.sort(
            np.random.default_rng(FLAGS.seed).choice(
                len(train_historical_dataset), size=n_subset, replace=False
            )
        )
        print(
            f"... training on a {FLAGS.train_subset_frac:.1%} record-level subset: "
            f"{n_subset} of {len(train_historical_dataset)} records"
        )
    # number of optimizer steps the run will actually perform. Has to match the number of
    # steps the DP accountant calibrates the noise for, otherwise the LR schedule is
    # stretched/compressed relative to the privacy budget.
    if FLAGS.train_random_subset:
        n_train_expected = int(FLAGS.subset_ratio * len(train_historical_dataset))
    elif train_subset_indices is not None:
        n_train_expected = len(train_subset_indices)
    else:
        n_train_expected = len(train_historical_dataset)
    STEPS = n_train_expected // FLAGS.batch_size * FLAGS.epochs
    microbatch_size = FLAGS.microbatch_size if FLAGS.microbatch_size > 0 else None
    if microbatch_size is not None:
        assert (
            FLAGS.batch_size % microbatch_size == 0
        ), f"batch size ({FLAGS.batch_size}) must be divisible by microbatch size ({microbatch_size})"
    print(f"... {STEPS} optimizer steps ({n_train_expected} train records, {FLAGS.epochs} epochs)")
    # non-private runs keep --batch_size as the *effective* batch and reach it by accumulating
    # gradients over smaller physical batches (see --nondp_batch_size), so the optimizer performs the
    # same STEPS updates on the same effective batch as the DP runs it is compared against.
    if FLAGS.use_dp:
        fit_batch_size, grad_accum_steps = FLAGS.batch_size, 1
    else:
        assert FLAGS.batch_size % FLAGS.nondp_batch_size == 0, (
            f"batch size ({FLAGS.batch_size}) must be divisible by the non-private physical batch "
            f"size ({FLAGS.nondp_batch_size})"
        )
        fit_batch_size = FLAGS.nondp_batch_size
        grad_accum_steps = FLAGS.batch_size // fit_batch_size
        print(
            f"... non-private baseline: {fit_batch_size} x {grad_accum_steps} accumulation steps = "
            f"{FLAGS.batch_size} effective batch"
        )
    # keras drives a LearningRateSchedule off the optimizer's raw batch counter (_iterations), which
    # is *not* divided by gradient_accumulation_steps, so the schedule has to be laid out over
    # physical batches. With no accumulation this is identical to STEPS.
    SCHEDULE_STEPS = n_train_expected // fit_batch_size * FLAGS.epochs

    def get_compiled_model():
        # create model, lr schedule and optimizer
        model = get_model(
            model_name=FLAGS.model,
            input_shape=(2_500, 12),
            num_classes=NUM_CLASSES,
            preprocessing_func=None,
            seed=FLAGS.seed,
        )
        schedule = MyCosineDecay(
            base_lr=FLAGS.lr,
            steps=int(FLAGS.decay_steps * SCHEDULE_STEPS),
            relative_lr_warmup_steps=FLAGS.lr_warmup,
        )
        opt = keras.optimizers.Adam(
            learning_rate=(schedule if FLAGS.lr_schedule == "cosine" else FLAGS.lr),
            weight_decay=0.0,
            use_ema=FLAGS.ema,
            ema_momentum=FLAGS.ema_decay,
            gradient_accumulation_steps=(
                grad_accum_steps if grad_accum_steps > 1 else None
            ),
        )
        # compile model
        model.compile(
            optimizer=opt,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.AUC(
                    multi_label=True,
                    from_logits=True,
                    name="macro_auroc",
                )
            ],
        )
        return model

    def get_callbacks(is_ema: bool):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, mode="min", verbose=1
            ),
        ]
        if is_ema:
            callbacks += [keras.callbacks.SwapEMAWeights(swap_on_epoch=True)]
        if FLAGS.lr_schedule == "constant":
            callbacks.append(
                WarmupReduceLROnPlateau(
                    warmup_steps=int(FLAGS.lr_warmup * SCHEDULE_STEPS),
                    initial_lr=1e-10,
                    max_lr=FLAGS.lr,
                    monitor="val_loss",
                    factor=0.1,
                    patience=1,
                    mode="min",
                    cooldown=0,
                    verbose=1,
                )
            )
        return callbacks

    if not FLAGS.train_random_subset:
        model = get_compiled_model()
        _ = train_and_eval(
            compiled_model=model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            subset_indices=train_subset_indices,
            batch_size=fit_batch_size,
            aug_fn=lambda x: x,
            augment=False,
            epochs=FLAGS.epochs,
            target_metric="val_macro_auroc",
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            seed=FLAGS.seed,
            log_wandb=FLAGS.log_wandb,
            wandb_project_name=FLAGS.wandb_project,
            save_best_only=FLAGS.save_best_only,
            use_dp=FLAGS.use_dp,
            clipping_norm=FLAGS.C,
            epsilon=FLAGS.eps,
            microbatch_size=microbatch_size,
            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
            num_read_workers=FLAGS.num_read_workers,
        )
        print("... training and evaluation complete.")
        return
    else:
        model = get_compiled_model()
        train_random_subset(
            compiled_model=model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_future_dataset=train_future_dataset,
            patient_id_col="BDSPPatientID",
            batch_size=fit_batch_size,
            aug_fn=lambda x: x,
            augment=False,
            epochs=FLAGS.epochs,
            target_metric="val_macro_auroc",
            seed=FLAGS.seed,
            logdir=Path(FLAGS.logdir),
            n_total_runs=FLAGS.n_runs,
            subset_ratio=FLAGS.subset_ratio,
            n_eval_views=1,
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            log_wandb=FLAGS.log_wandb,
            wandb_project_name=FLAGS.wandb_project,
            save_best_only=FLAGS.save_best_only,
            use_dp=FLAGS.use_dp,
            clipping_norm=FLAGS.C,
            epsilon=FLAGS.eps,
            microbatch_size=microbatch_size,
            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
            num_read_workers=FLAGS.num_read_workers,
        )


if __name__ == "__main__":
    app.run(main)
