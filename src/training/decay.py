"""Learning rate schedules shared by the entry points.

`MyCosineDecay` is cosine decay with linear warmup, driven off the optimizer's step
counter. Note that keras counts *physical* batches there, not optimizer updates, so with
gradient accumulation the schedule has to be laid out over batches -- see SCHEDULE_STEPS
in the DP entry points.

`WarmupReduceLROnPlateau` is the alternative used with --lr_schedule=constant: linear
warmup followed by keras' ReduceLROnPlateau, i.e. the LR is driven by the validation
metric rather than by a fixed curve.
"""

from keras import ops  # type: ignore
import numpy as np
import tensorflow as tf
import keras  # type: ignore
from keras.optimizers.schedules import LearningRateSchedule  # type: ignore
import re
from typing import Dict


# Custom cosine learning rate decay (Carlini et al., 2021)
class MyCosineDecay(LearningRateSchedule):
    """
    Custom cosine learning rate decay schedule.

    Args:
        base_lr (float): The base learning rate.
        steps (int): The total number of steps.
        name (str, optional): The name of the schedule. Defaults to "CosineDecay".
        relative_lr_warmup_steps (float, optional): The relative learning rate warmup steps. Defaults to 0.025.
    """

    def __init__(
        self,
        base_lr: float,
        steps: int,
        name: str = "CosineDecay",
        relative_lr_warmup_steps: float = 0.025,
    ):
        super().__init__()

        self.base_lr = base_lr
        self.steps = steps
        self.name = name
        self.warmup_factor = ops.convert_to_tensor(1 / relative_lr_warmup_steps)
        print(
            f"... custom cosine lr schedule: warmup steps={int(relative_lr_warmup_steps*steps)}, total decay steps={steps}"
        )
        if self.steps <= 0:
            raise ValueError(
                "Argument `steps` must be > 0. " f"Received: steps={self.steps}"
            )

    def _decay_function(
        self, step: int, steps: int, decay_from_lr: float, dtype=tf.float32
    ):
        """
        Compute the decayed learning rate.

        Args:
            step (int): The current step.
            steps (int): The total number of steps.
            decay_from_lr (float): The initial learning rate.
            dtype: The data type.

        Returns:
            tf.Tensor: The decayed learning rate.
        """
        completed_fraction = step / steps
        completed_fraction = ops.clip(completed_fraction, 0, 1)
        pi = ops.cast(np.pi, dtype=dtype)
        cosine_decayed = decay_from_lr * ops.cos(
            completed_fraction * (7 * pi) / (2 * 8)
        )
        cosine_decayed = cosine_decayed * ops.clip(
            completed_fraction * self.warmup_factor, 0, 1
        )  # modified so that linear lr warmup is slighlty longer
        return cosine_decayed

    def __call__(self, step: int):
        """
        Call the learning rate schedule.

        Args:
            step (int): The current step.

        Returns:
            tf.Tensor: The learning rate for the current step.
        """
        initial_learning_rate = ops.convert_to_tensor(self.base_lr)
        dtype = initial_learning_rate.dtype
        steps = ops.cast(self.steps, dtype)
        global_step_recomp = ops.cast(step, dtype)
        decayed_lr = self._decay_function(
            global_step_recomp, steps, initial_learning_rate, dtype
        )
        return decayed_lr

    def get_config(self):
        """
        Get the configuration of the learning rate schedule.

        Returns:
            dict: The configuration dictionary.
        """
        return {
            "base_lr": self.base_lr,
            "steps": self.steps,
            "name": self.name,
            "relative_lr_warmup_steps": self.warmup_factor,
        }


class WarmupReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
    """
    A custom callback that performs a linear warmup,
    then switches to standard ReduceLROnPlateau behavior.
    """

    def __init__(self, warmup_steps: int, initial_lr: float, max_lr: float, **kwargs):
        # Pass standard arguments to ReduceLROnPlateau (monitor, patience, etc.)
        super().__init__(**kwargs)
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.global_step = 0  # Internal counter for total steps across epochs

    def on_train_batch_begin(self, batch, logs=None):
        # Increment total steps (persists across epochs)
        self.global_step += 1

        # LINEAR WARMUP LOGIC (Per Step)
        if self.global_step <= self.warmup_steps:
            # Calculate linear progress: 0.0 at step 0, 1.0 at step N
            warmup_ratio = self.global_step / self.warmup_steps

            # Interpolate
            current_lr = (
                self.initial_lr + (self.max_lr - self.initial_lr) * warmup_ratio
            )

            # Assign (Keras 3 backend-agnostic way)
            self.model.optimizer.learning_rate.assign(current_lr)

    def on_epoch_end(self, epoch, logs=None):
        # PLATEAU LOGIC (Per Epoch)
        # We generally shouldn't reduce LR while we are still warming up.
        if self.global_step < self.warmup_steps:
            return

        # Once warmup is done, let standard Plateau logic take over
        super().on_epoch_end(epoch, logs)


