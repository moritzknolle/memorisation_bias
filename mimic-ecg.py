"""Training on MIMIC-IV-ECG (12-lead ECG, 15 labels).

--train_random_subset=False trains one model on the full historical training split.
--train_random_subset=True trains one run of the leave-many-out experiment: it claims the next
unclaimed patient subset from --logdir, trains on it and saves the logits over the historical,
future and test splits. scripts/run_until_error.sh repeats this until all runs are complete.

Usage:
    python mimic-ecg.py --train_random_subset=True --logdir=<logdir>
"""

import os
from pathlib import Path

# set keras backend to jax and enable compilation caching
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import keras  # type: ignore
import numpy as np
from absl import app, flags  # type: ignore

from src.data.constants import MIMIC_ECG_LABELS
from src.data.dataset_factory import get_dataset
from src.training.models.model_factory import get_model
from src.training.training import train_and_eval, train_random_subset
from src.training.augment import (
    get_aug_fn,
)
from src.training.decay import MyCosineDecay

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 50, "Number of training epochs.")
flags.DEFINE_float("lr", 5e-3, "Learning rate.")
flags.DEFINE_float("wd", 0.5, "L2 weight decay.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_boolean(
    "ema", True, "Whether to use exponential moving average for parameters."
)
flags.DEFINE_string("model", "vit1d_small_25", "Name of the model to use.")
flags.DEFINE_enum("lr_schedule", "constant", ["constant", "cosine"], "LR schedule.")
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
flags.DEFINE_float("ema_decay", 0.95, "EMA decay.")
flags.DEFINE_integer("grad_accum_steps", 4, "Number of gradient accumulation steps.")
flags.DEFINE_float("dropout", 0.0, "Dropout rate.")
flags.DEFINE_integer("fs", 250, "Sampling frequency of the ECG signals (hz).")
flags.DEFINE_boolean(
    "mixed_precision",
    True,
    "Whether to perform mixed precision training to reduce training time.",
)
flags.DEFINE_bool(
    "train_random_subset",
    False,
    "Whether to train on random subset for privacy auditing (True) or train on the full dataset (False).",
)
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
flags.DEFINE_string("data_root", "./data/raw/mimic-ecg", "Path to the MIMIC-IV-ECG files.")
flags.DEFINE_string("csv_root", "./data/csv", "Path to the split files.")
flags.DEFINE_string(
    "save_root",
    "./data/npy/",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_string(
    "logdir",
    "./logs/mimic-ecg_test/",
    "Path to logdir.",
)
flags.DEFINE_bool(
    "save_best_only",
    True,
    "Whether to save only the best model checkpoint based on the target metric.",
)
flags.DEFINE_integer(
    "n_threads",
    64,
    "Number of threads to use for parallel data loading and preprocessing.",
)
flags.DEFINE_integer(
    "chunk_size",
    100,
    "Number of samples per chunk for multiprocessing.",
)
flags.DEFINE_boolean(
    "write_to_disk",
    False,
    "Whether to write the dataset to disk.",
)


def main(argv):
    np.random.seed(FLAGS.seed)
    if FLAGS.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    NUM_CLASSES = 15
    (
        train_historical_dataset,
        val_dataset,
        train_future_dataset,
        test_dataset,
    ) = get_dataset(
        dataset_name="mimic-ecg",
        csv_root=Path(FLAGS.csv_root),
        data_root=Path(FLAGS.data_root),
        save_root=Path(FLAGS.save_root),
        resolution=FLAGS.fs,
        use_cached=True,
        load_from_disk=True,
        write_to_disk=FLAGS.write_to_disk,
        n_threads=FLAGS.n_threads,
        chunk_size=FLAGS.chunk_size,
    )
    print(
        f"x_train.shape: {train_historical_dataset.inputs.shape}, y_train.shape: {train_historical_dataset.targets.shape}"
    )
    print(
        f"x_val.shape: {val_dataset.inputs.shape}, y_val.shape: {val_dataset.targets.shape}"
    )
    print(
        f"x_test.shape: {test_dataset.targets.shape}, y_test.shape: {test_dataset.targets.shape}"
    )
    print(
        f"x_long_eval.shape: {train_future_dataset.inputs.shape}, y_long_eval.shape: {train_future_dataset.targets.shape}"
    )
    sample_batch = train_historical_dataset.inputs[0:256]
    print(
        f"sample batch stats: {sample_batch.min()}, {sample_batch.max()}, {sample_batch.mean()}, {sample_batch.std()}"
    )

    STEPS = len(train_historical_dataset) // FLAGS.batch_size * FLAGS.epochs

    def get_compiled_model():
        # create model, lr schedule and optimizer
        model = get_model(
            model_name=FLAGS.model,
            input_shape=(int(FLAGS.fs * 10), 12),
            num_classes=NUM_CLASSES,
            preprocessing_func=None,
            seed=FLAGS.seed,
        )
        schedule = MyCosineDecay(
            base_lr=FLAGS.lr,
            steps=int(FLAGS.decay_steps * STEPS),
            relative_lr_warmup_steps=FLAGS.lr_warmup,
        )
        opt = keras.optimizers.AdamW(
            learning_rate=(schedule if FLAGS.lr_schedule == "cosine" else FLAGS.lr),
            weight_decay=FLAGS.wd,
            use_ema=FLAGS.ema,
            ema_momentum=FLAGS.ema_decay,
            global_clipnorm=1.0,
            gradient_accumulation_steps=(
                FLAGS.grad_accum_steps if FLAGS.grad_accum_steps > 1 else None
            ),
        )
        opt.exclude_from_weight_decay(var_names=["layernorm", "LayerNorm", "batch_normalization"])
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
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                mode="min",
                cooldown=2,
                verbose=1,
            ),
        ]
        if is_ema:
            callbacks += [keras.callbacks.SwapEMAWeights(swap_on_epoch=True)]
        return callbacks

    if not FLAGS.train_random_subset:
        model = get_compiled_model()
        _ = train_and_eval(
            compiled_model=model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=FLAGS.batch_size,
            aug_fn=get_aug_fn("none"),
            augment=False,
            epochs=FLAGS.epochs,
            target_metric="val_loss",
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            seed=FLAGS.seed,
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="mimic-ecg",
            save_best_only=FLAGS.save_best_only,
        )
        raise StopIteration("Evaluation only, stopping execution.")
    else:
        model = get_compiled_model()
        train_random_subset(
            compiled_model=model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            train_future_dataset=train_future_dataset,
            patient_id_col="subject_id",
            batch_size=FLAGS.batch_size,
            aug_fn=get_aug_fn("none"),
            augment=False,
            epochs=FLAGS.epochs,
            target_metric="val_loss",
            seed=FLAGS.seed,
            logdir=Path(FLAGS.logdir),
            n_total_runs=FLAGS.n_runs,
            subset_ratio=FLAGS.subset_ratio,
            n_eval_views=1,
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="mimic-ecg",
            save_best_only=FLAGS.save_best_only,
        )


if __name__ == "__main__":
    app.run(main)
