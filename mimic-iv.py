"""Training of a tabular ResNet on MIMIC-IV-ED (emergency department triage, 2 outcomes).

See mimic-ecg.py for the two modes selected by --train_random_subset. mimic-iv_sklearn.py trains
the sklearn models on the same splits. The dataset is held in memory and not cached.

Usage:
    python mimic-iv.py --train_random_subset=True --logdir=<logdir>
"""

import os
from pathlib import Path

# set keras backend to jax and enable compilation caching
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import keras  # type: ignore
import numpy as np
from absl import app, flags  # type: ignore
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report  # type: ignore
from typing import Tuple

from src.data.dataset_factory import get_dataset
from src.training.models.model_factory import get_model
from src.training.training import train_and_eval, train_random_subset
from src.training.augment import (
    get_aug_fn,
)
from src.training.decay import MyCosineDecay, WarmupReduceLROnPlateau

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 50, "Number of training epochs.")
flags.DEFINE_float("lr", 1e-2, "Learning rate.")
flags.DEFINE_float("wd", 1.0, "Decoupled weight decay.")
flags.DEFINE_integer("batch_size", 4096, "Batch size.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_boolean(
    "ema", True, "Whether to use exponential moving average for parameters."
)
flags.DEFINE_string("model", "tabresnet_500_5", "Name of the model to use.")
flags.DEFINE_float("dropout", 0.25, "Dropout rate inside the residual blocks of the tabular ResNet.")
flags.DEFINE_boolean(
    "input_norm",
    True,
    "Whether to standardise the input columns with a frozen Normalization layer fitted on the "
    "full historical training split.",
)
flags.DEFINE_string("wandb_project", "mimic-iv-ed", "Name of the wandb project to log to.")
flags.DEFINE_enum(
    "lr_schedule",
    "cosine",
    ["constant", "cosine"],
    "LR schedule: cosine decay, or linear warmup followed by reduction on plateau (constant).",
)
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
flags.DEFINE_integer("grad_accum_steps", 1, "Number of gradient accumulation steps.")
flags.DEFINE_boolean(
    "mixed_precision",
    True,
    "Whether to perform mixed precision training to reduce training time.",
)
flags.DEFINE_bool("train_random_subset", False, "Whether to train on random subset for privacy auditing (True) or train and evaluate on full dataset (False).")
flags.DEFINE_integer(
    "n_runs", 200, "Number of leave-many-out re-training runs to perform."
)
flags.DEFINE_float(
    "subset_ratio", 0.5, "Ratio of the training data to use for each re-training run."
)
flags.DEFINE_integer(
    "eval_views", 16, "Number of augmentations to query when saving train/test logits."
)
flags.DEFINE_string(
    "ckpt_file_path",
    "./tmp/ckpts/",
    "Path to root folder where the model checkpoint files are stored.",
)
flags.DEFINE_string("csv_root", "./data/csv", "Path to the split files.")
flags.DEFINE_string(
    "save_root",
    "./data/npy/",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_string(
    "logdir",
    "./logs/mimic-iv-ed/",
    "Path to logdir.",
)
flags.DEFINE_bool(
    "save_best_only",
    True,
    "Whether to save only the best model checkpoint based on the target metric.",
)
flags.DEFINE_boolean(
    "write_to_disk",
    False,
    "Whether to write the dataset to disk.",
)


def get_compiled_model(train_steps: int, num_classes: int = 2, input_stats=None):
    # create model, lr schedule and optimizer
    model = get_model(
        model_name=FLAGS.model,
        input_shape=(64,),
        batch_size=FLAGS.batch_size,
        num_classes=num_classes,
        preprocessing_func=None,
        seed=FLAGS.seed,
        dropout_rate=FLAGS.dropout,
        input_stats=input_stats,
    )
    schedule = MyCosineDecay(
        base_lr=FLAGS.lr,
        steps=int(FLAGS.decay_steps * train_steps),
        relative_lr_warmup_steps=FLAGS.lr_warmup,
    )   
    opt = keras.optimizers.AdamW(
        learning_rate=(
            schedule if FLAGS.lr_schedule == "cosine" else FLAGS.lr
        ),
        weight_decay=FLAGS.wd,
        use_ema=FLAGS.ema,
        ema_momentum=FLAGS.ema_decay,
        global_clipnorm=1.0,
        gradient_accumulation_steps=FLAGS.grad_accum_steps if FLAGS.grad_accum_steps > 1 else None,
    )
    opt.exclude_from_weight_decay(var_names=["layernorm", "LayerNorm", "batch_normalization"])
    # compile model with multi-label metrics
    model.compile(
        optimizer=opt,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(),
            keras.metrics.AUC(from_logits=True, multi_label=True, name='macro_auc'),
        ],
    )
    return model


def get_callbacks(is_ema: bool, train_steps:int):
    callbacks = []
    if is_ema:
        callbacks += [keras.callbacks.SwapEMAWeights(swap_on_epoch=True)]
    if FLAGS.lr_schedule == "constant":
        lr_callback = WarmupReduceLROnPlateau(
            warmup_steps=int(FLAGS.lr_warmup * train_steps),
            initial_lr=FLAGS.lr * 0.01,
            max_lr=FLAGS.lr,
            monitor="val_loss",
            factor=0.1,
            patience=5,
            mode="min",
            cooldown=3,
            verbose=1,
    )
        callbacks.append(lr_callback)
    return callbacks


def main(argv):
    if FLAGS.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")

    train_historical_dataset, val_dataset, train_future_dataset, test_dataset = get_dataset(
        dataset_name="mimic-iv-ed",
        csv_root=Path(FLAGS.csv_root),
        data_root=None,
        save_root=None,
        use_cached=True,
        load_from_disk=True,
        write_to_disk=FLAGS.write_to_disk,
    )
    # fitted on the full historical split rather than per run, so that IN and OUT models see
    # identical preprocessing. float32 because the inputs are float16 and raw values reach ~250.
    input_stats = None
    if FLAGS.input_norm:
        raw = np.asarray(train_historical_dataset.inputs, dtype=np.float32)
        mean = raw.mean(axis=0)
        variance = raw.var(axis=0)
        # constant columns would divide by ~0; Normalization guards with sqrt(var) but a
        # zero-variance feature still yields 0/0, so floor the variance at 1 (leaving such a
        # column mean-centred and otherwise untouched).
        variance = np.where(variance < 1e-6, 1.0, variance)
        input_stats = (mean, variance)
        print(f"... standardising inputs, {int((variance == 1.0).sum())}/{len(variance)} columns constant")

    if not FLAGS.train_random_subset:

        STEPS = len(train_historical_dataset) // FLAGS.batch_size * FLAGS.epochs
        model = get_compiled_model(train_steps=STEPS, input_stats=input_stats)
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
            callbacks=get_callbacks(FLAGS.ema, STEPS),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            seed=FLAGS.seed,
            log_wandb=FLAGS.log_wandb,
            wandb_project_name=FLAGS.wandb_project,
            save_best_only=FLAGS.save_best_only,
        )
    else:
        STEPS = int(len(train_historical_dataset)*FLAGS.subset_ratio) // FLAGS.batch_size * FLAGS.epochs
        model = get_compiled_model(train_steps=STEPS, input_stats=input_stats)
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
            callbacks=get_callbacks(FLAGS.ema, STEPS),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            log_wandb=FLAGS.log_wandb,
            wandb_project_name=FLAGS.wandb_project,
            save_best_only=FLAGS.save_best_only,
        )


if __name__ == "__main__":
    app.run(main)
