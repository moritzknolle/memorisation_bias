"""Training: one model, or one run of the leave-many-out experiment.

    training.py   train_and_eval() and train_random_subset(), including DP-SGD and the
                  file-locked subset bookkeeping shared by parallel workers.
    dp.py         DP-SGD configuration.
    logger.py     per-run outputs (logits, masks, info.json) and the wandb callback.
    models/       model factory; architectures are selected by name.
    augment.py    image augmentation (MIMIC-CXR only).
    decay.py      learning rate schedules.
"""
