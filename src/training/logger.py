"""Saving of per-run outputs, and the wandb training callback.

`RetrainLogger` writes one directory per run, named after the wandb run id:

    <logdir>/<wandb_run_id>/
        train_logits.npy         model outputs over the historical (training) split
        long_eval_logits.npy     ... over the future split
        test_logits.npy          ... over the test split
        patient_subset_mask.npy  boolean mask of the patients the run was trained on
        patient_ids.pkl          patient ids, in the order of the mask
        info.json                wandb config, train/test metrics, timings

Logits are saved without activation function. Since the directory is named after the wandb run
id, `train_random_subset` only saves outputs with --log_wandb=True.

`WandbLogger` logs per-epoch metrics, learning rate and throughput.
"""

import json
import time
import uuid
import keras  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

import joblib
import numpy as np

import wandb

INFO_TEMPLATE = {
    "wandb_run_id": 0,
    "start_time": "",
    "end_time": "",
    "mac_address": "",
    "wandb_config": {},
    "train_metrics": {},
    "test_metrics": {},
}


class RetrainLogger:
    """
    A helper class for logging results and metrics of leave-many-out re-training runs.

    This logger is designed to track and save training and evaluation data, including logits,
    labels, and performance metrics. It integrates with Weights & Biases (wandb) for experiment
    tracking and saves all relevant data to the filesystem for later analysis.

    The logger creates a unique directory for each run based on the wandb run ID and saves
    all data in a structured format, including metadata about the run.

    Args:
        logdir (Path): The base directory where all logs will be saved.
        subset_idcs (np.ndarray): Array of indices indicating which records are included in the subset.
        subset_mask (np.ndarray): A boolean mask indicating which patients are included in the subset.

    Attributes:
        start_time (str): The start time of the retraining process in format "YYYY-MM-DD_HH:MM:SS".
        end_time (str): The end time of the retraining process in format "YYYY-MM-DD_HH:MM:SS".
        mac_address (str): The MAC address of the machine where the run was executed.
        wandb_run_id (str): The ID of the Weights & Biases run.
        config (dict): The configuration settings from wandb.
        base_dir (Path): The base directory where logs will be saved.
        log_dir (Path): The specific directory for the current run.
        subset_idcs (np.ndarray): Array of indices for record subset selection.
        subset_mask (np.ndarray): The boolean mask for patient subset selection.
    """

    def __init__(
        self,
        logdir: Path,
        patient_ids: pd.Series,
        patient_subset_mask: np.ndarray,
    ):
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.mac_address = self.get_mac_address()
        self.base_dir = logdir
        self.patient_subset_mask = patient_subset_mask
        self.base_dir = logdir
        self.patient_ids = patient_ids
        if not self.base_dir.exists():
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, config: dict, train_metrics: dict, test_metrics: dict):
        """
        Saves the metadata to a JSON file in the log directory.

        This method creates a comprehensive metadata file that includes:
        - Wandb run ID
        - Start and end times
        - Machine MAC address
        - Wandb configuration
        - Train and test metrics

        Args:
            config (dict): The wandb configuration to be saved.
            train_metrics (dict): The training metrics to be saved.
            test_metrics (dict): The test metrics to be saved.

        Returns:
            bool: True if the metadata is successfully saved, False otherwise.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Expected config as dictionary, found: {type(config)}")
        self.config = config
        train_metrics = {k: float(v) for k, v in train_metrics.items()}
        test_metrics = {k: float(v) for k, v in test_metrics.items()}
        try:
            info_dict = INFO_TEMPLATE
            info_dict["wandb_run_id"] = self.wandb_run_id
            info_dict["start_time"] = self.start_time
            info_dict["end_time"] = self.end_time
            info_dict["mac_address"] = self.mac_address
            info_dict["wandb_config"] = self.config
            info_dict["train_metrics"] = train_metrics
            info_dict["test_metrics"] = test_metrics
            with open(self.log_dir / "info.json", "w") as f:
                json.dump(info_dict, f, indent=4)
        except Exception as e:
            print(e)
            return False
        return True

    def maybe_create_log_dir(self):
        """
        Creates a log directory for the current run if it doesn't exist.

        This method checks if a wandb run is active, retrieves the run ID,
        and creates a directory structure for storing logs.

        Raises:
            ValueError: If wandb.init() has not been called before using this logger.
        """
        if wandb.run is None:
            raise ValueError(
                "You must call wandb.init() before starting a RetrainExperiment()"
            )
        self.wandb_run_id = wandb.run.id
        self.log_dir = self.base_dir / str(self.wandb_run_id)
        if not self.log_dir.exists():
            print(f"... creating log_dir at {self.log_dir}")
            self.log_dir.mkdir(parents=True)

    def log(
        self,
        config: dict,
        train_logits: np.ndarray,
        long_eval_logits: np.ndarray,
        test_logits: np.ndarray,
        train_metrics: dict,
        test_metrics: dict,
        train_features: Optional[np.ndarray] = None,
        long_eval_features: Optional[np.ndarray] = None,
    ):
        """
        Logs logits (raw predictions) and labels for the training and longitudinal evaluation data.
        This is a specialized method for longitudinal evaluation experiments.

        Args:
            config (dict): The wandb configuration.
            train_logits (np.ndarray): The model's raw predictions on the training data.
            long_eval_logits (np.ndarray): The model's raw predictions on the longitudinal evaluation data.
            test_logits (np.ndarray): The model's raw predictions on the test data (unseen patients).
            train_metrics (dict): The training metrics to be logged.
            test_metrics (dict): The test metrics to be logged.
            train_features (Optional[np.ndarray]): intermediate features of the training data.
            long_eval_features (Optional[np.ndarray]): intermediate features of the longitudinal evaluation data.

        Returns:
            bool: True if the data is successfully logged, False otherwise.

        Raises:
            AssertionError: If there's a shape mismatch between logits and labels.
        """
        self.maybe_create_log_dir()
        self.end_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if train_features is not None:
            assert len(train_logits) == len(train_features), "Careful! Shape mismatch"
        if long_eval_features is not None:
            assert len(long_eval_logits) == len(
                long_eval_features
            ), "Careful! Shape mismatch"
        try:
            np.save(self.log_dir / "train_logits.npy", train_logits)
            np.save(self.log_dir / "long_eval_logits.npy", long_eval_logits)
            np.save(self.log_dir / "test_logits.npy", test_logits)
            np.save(self.log_dir / "patient_subset_mask.npy", self.patient_subset_mask)
            joblib.dump(self.patient_ids, self.log_dir / "patient_ids.pkl")
            if train_features is not None:
                np.save(self.log_dir / "train_features.npy", train_features)
            if long_eval_features is not None:
                np.save(self.log_dir / "long_eval_features.npy", long_eval_features)
        except Exception as e:
            print(e)
            return False
        metadata_sucess = self.save_metadata(
            config=config, train_metrics=train_metrics, test_metrics=test_metrics
        )
        assert metadata_sucess, "Failed to save metadata"
        return True

    def get_mac_address(self):
        """
        Retrieves the MAC address of the machine.

        This method attempts to get the MAC address of the current machine using
        the uuid module. If it fails, it returns a default value.

        Returns:
            str: The MAC address of the machine in hexadecimal format, or a default
                 value if the address cannot be retrieved.
        """
        try:
            mac_id = hex(uuid.getnode())
        except Exception as e:
            print(e)
            mac_id = "0000000000"
        return mac_id


class WandbLogger(keras.callbacks.Callback):
    """
    Callback for efficiently logging performance metrics and learning rate to wandb during training.

    Args:
        model (keras.Model): The Keras model.
    """

    def __init__(self, model: keras.Model):
        super().__init__()
        assert isinstance(model, keras.Model)
        self.optimizer = model.optimizer
        self._epoch_start: float = 0.0
        self._batch_count: int = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None):
        self._epoch_start = time.time()
        self._batch_count = 0

    def on_train_batch_end(self, batch: int, logs: Optional[dict] = None):
        self._batch_count += 1

    def on_epoch_end(self, _: int, logs: Optional[dict] = None):
        """
        Log metrics and learning rate to wandb at the end of an epoch.

        Args:
            epoch (int): The current epoch.
            logs (Optional[dict], optional): The logs dictionary. Defaults to None.
        """
        elapsed = time.time() - self._epoch_start
        batches_per_sec = self._batch_count / elapsed if elapsed > 0 else 0.0
        lr = self.optimizer.learning_rate
        to_log = {"lr": lr} if logs is None else {**logs, "lr": lr._value}
        to_log["batches_per_sec"] = batches_per_sec
        try:
            wandb.log(to_log, commit=True)  # type: ignore
        except Exception as e:
            print(
                f"WARNING: ecountered Exception while trying to log wandb results: \n{to_log} \n {e}"
            )
