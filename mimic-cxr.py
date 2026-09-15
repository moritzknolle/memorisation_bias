"""Training on MIMIC-CXR (chest radiographs, 14 labels).

See mimic-ecg.py for the two modes selected by --train_random_subset. Images are cached per
--img_size. With augmentation, logits of --eval_views augmented views are saved per record and
averaged in the analysis.

Usage:
    python mimic-cxr.py --train_random_subset=True --logdir=<logdir>
"""

import os
from pathlib import Path

# set keras backend and enable compilation caching
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import keras  # type: ignore
import numpy as np # type: ignore
from absl import app, flags  # type: ignore

from src.data.constants import CXP_CHALLENGE_LABELS_IDX
from src.data.dataset_factory import get_dataset
from src.training.models.model_factory import get_model
from src.training.training import train_and_eval, train_random_subset
from src.training.augment import (
    get_aug_fn,
)
from src.training.decay import (
    MyCosineDecay,
    WarmupReduceLROnPlateau,
)

FLAGS = flags.FLAGS
flags.DEFINE_integer("epochs", 60, "Number of training epochs.")
flags.DEFINE_float("lr", 5e-4, "Learning rate.")
flags.DEFINE_float("wd", 0.01, "L2 weight decay.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_boolean(
    "ema", True, "Whether to use exponential moving average for parameters."
)
flags.DEFINE_string("model", "densenet121_imagenet", "Name of the model to use.")
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
flags.DEFINE_float("ema_decay", 0.9, "EMA decay.")
flags.DEFINE_integer("grad_accum_steps", 1, "Number of gradient accumulation steps.")
flags.DEFINE_string(
    "augment",
    "cxr_strong",
    "What type of data augmentations strength to apply. See `get_aug_fn` for options.",
)
flags.DEFINE_list("img_size", [256, 256], "Image size.")
flags.DEFINE_boolean(
    "mixed_precision",
    True,
    "Whether to perform mixed precision training to reduce training time.",
)
flags.DEFINE_bool(
    "train_random_subset",
    False,
    "Whether to train on random subset for memorisation detection (True) or train on the full dataset (False).",
)
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
flags.DEFINE_string(
    "data_root", "./data/raw/mimic-cxr/mimic-cxr-jpg", "Path to the MIMIC-CXR-JPG files."
)
flags.DEFINE_string("csv_root", "./data/csv", "Path to the split files.")
flags.DEFINE_string(
    "save_root",
    "./data/npy/",
    "Path to root folder where the memmap files are stored.",
)
flags.DEFINE_string(
    "logdir",
    "./logs/mimic-cxr-test/",
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
    1024,
    "Number of samples per chunk for multiprocessing.",
)
flags.DEFINE_boolean(
    "write_to_disk",
    False,
    "Whether to write the dataset to disk.",
)


def main(argv):
    assert len(FLAGS.img_size) == 2, "img_size must be a list of two integers"
    np.random.seed(FLAGS.seed)
    if FLAGS.mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    NUM_CLASSES = 14
    (
        train_historical_dataset,
        val_dataset,
        train_future_dataset,
        test_dataset,
    ) = get_dataset(
        dataset_name="mimic-cxr",
        csv_root=Path(FLAGS.csv_root),
        data_root=Path(FLAGS.data_root),
        save_root=Path(FLAGS.save_root),
        resolution=FLAGS.img_size[0],
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
        f"x_test.shape: {test_dataset.inputs.shape}, y_test.shape: {test_dataset.targets.shape}"
    )
    print(
        f"x_long_eval.shape: {train_future_dataset.inputs.shape}, y_long_eval.shape: {train_future_dataset.targets.shape}"
    )
    sample_batch = train_historical_dataset.inputs[:16]
    # crop sample batch if using cxr augmentations
    if "cxr" in FLAGS.augment:
        sample_batch = sample_batch[
            :,
            int(FLAGS.img_size[0] * 0.0625) : int(FLAGS.img_size[0] * 0.9375),
            int(FLAGS.img_size[1] * 0.0625) : int(FLAGS.img_size[1] * 0.9375),
            :,
        ]
    STEPS = len(train_historical_dataset) // FLAGS.batch_size * FLAGS.epochs

    def get_compiled_model():
        # create model, lr schedule and optimizer
        inp_shape = (int(FLAGS.img_size[0]), int(FLAGS.img_size[1]), 3)
        if "cxr" in FLAGS.augment:
            # center crop image size
            inp_shape = (int(FLAGS.img_size[0] * 0.875), int(FLAGS.img_size[1] * 0.875), 3)
        model = get_model(
            model_name=FLAGS.model,
            input_shape=inp_shape,
            num_classes=NUM_CLASSES,
            preprocessing_func=None,
            seed=FLAGS.seed,
            freeze_bn_layers=True,
        )
        model(sample_batch)  # build model

        if FLAGS.lr_schedule == "cosine":
            lr = MyCosineDecay(
                base_lr=FLAGS.lr,
                steps=int(FLAGS.decay_steps * STEPS),
                relative_lr_warmup_steps=FLAGS.lr_warmup,
            )
        else:
            lr = FLAGS.lr

        # Create base optimizer
        base_opt = keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=FLAGS.wd,
            use_ema=FLAGS.ema,
            ema_momentum=FLAGS.ema_decay,
            global_clipnorm=1.0,
            gradient_accumulation_steps=(
                FLAGS.grad_accum_steps if FLAGS.grad_accum_steps > 1 else None
            ),
        )
        base_opt.exclude_from_weight_decay(var_names=["layernorm", "LayerNorm", "batch_normalization"])

        # compile model
        cxp_label_weights = np.zeros(NUM_CLASSES)
        cxp_label_weights[CXP_CHALLENGE_LABELS_IDX] = 1
        model.compile(
            optimizer=base_opt,
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.AUC(
                    multi_label=True, from_logits=True, name="macro_auroc"
                ),  # macro AUROC over all classes
            ],
        )

        return model

    def get_callbacks(is_ema: bool):
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, mode="min", verbose=1
            ),
        ]
        if FLAGS.lr_schedule == "constant":
            callbacks.append(
                WarmupReduceLROnPlateau(
                    warmup_steps=int(FLAGS.lr_warmup * STEPS),
                    initial_lr=1e-10,
                    max_lr=FLAGS.lr,
                    monitor="val_loss",
                    factor=0.1,
                    patience=3,
                    mode="min",
                    cooldown=0,
                    verbose=1,
                )
            )
        elif FLAGS.lr_schedule == "cosine":
            raise NotImplementedError("LR scheduler callback for cosine decay is not implemented yet.")
        if is_ema:
            callbacks.append(keras.callbacks.SwapEMAWeights(swap_on_epoch=True))
        return callbacks

    if not FLAGS.train_random_subset:
        model = get_compiled_model()
        _ = train_and_eval(
            compiled_model=model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=FLAGS.batch_size,
            aug_fn=get_aug_fn(FLAGS.augment),
            test_aug_fn=get_aug_fn("center_crop") if "cxr" in FLAGS.augment else None,
            augment=True if FLAGS.augment != "None" else False,
            epochs=FLAGS.epochs,
            target_metric="val_loss",
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            seed=FLAGS.seed,
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="mimic-cxr",
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
            aug_fn=get_aug_fn(FLAGS.augment),
            augment=True if FLAGS.augment != "None" else False,
            test_aug_fn=get_aug_fn("center_crop") if "cxr" in FLAGS.augment else None,
            epochs=FLAGS.epochs,
            target_metric="val_loss",
            seed=FLAGS.seed,
            logdir=Path(FLAGS.logdir),
            n_total_runs=FLAGS.n_runs,
            subset_ratio=FLAGS.subset_ratio,
            n_eval_views=FLAGS.eval_views if FLAGS.augment != "none" else 1,
            callbacks=get_callbacks(FLAGS.ema),
            ckpt_file_path=Path(FLAGS.ckpt_file_path),
            log_wandb=FLAGS.log_wandb,
            wandb_project_name="mimic-cxr",
            save_best_only=FLAGS.save_best_only,
        )


if __name__ == "__main__":
    app.run(main)
