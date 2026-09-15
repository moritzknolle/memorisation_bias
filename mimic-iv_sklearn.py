"""Training of sklearn models on MIMIC-IV-ED (random forest, logistic regression, gradient boosting).

See mimic-ecg.py for the two modes selected by --train_random_subset. --model selects the base
estimator of a MultiOutputClassifier. Predicted probabilities rather than logits are saved; the
analysis detects this from `model_class` in info.json.

Usage:
    python mimic-iv_sklearn.py --model=rf --train_random_subset=True --logdir=<logdir>
"""

from pathlib import Path

import numpy as np
from absl import app, flags  # type: ignore
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report  # type: ignore
from typing import Tuple

from src.data.dataset_factory import get_dataset
from src.training.training import train_and_eval_sklearn, train_random_subset_sklearn

FLAGS = flags.FLAGS

# General training parameters
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("log_wandb", True, "Whether to log metrics to weights & biases.")
flags.DEFINE_bool("train_random_subset", False, "Whether to train on random subset for privacy auditing (True) or train and evaluate on full dataset (False).")
flags.DEFINE_string(
    "model",
    "rf",
    "Type of sklearn model to use: 'rf' (random forest), 'lr' (standardised logistic "
    "regression) or 'gb' (gradient boosting). See get_sklearn_model below.",
)

# Memorisation auditing parameters
flags.DEFINE_integer("n_runs", 200, "Number of leave-many-out re-training runs to perform.")
flags.DEFINE_float("subset_ratio", 0.5, "Ratio of the training data to use for each re-training run.")
flags.DEFINE_string("logdir", "./logs/mimic-iv-ed-sklearn/", "Path to logdir.")

# Data parameters
flags.DEFINE_string("csv_root", "./data/csv", "Path to the split files.")
flags.DEFINE_boolean(
    "write_to_disk",
    False,
    "Whether to write the dataset to disk.",
)


def get_sklearn_model(modeltype: str = "rf"):
    """Create and return a configured sklearn model for multi-label classification using MultiOutputClassifier."""
    from sklearn.multioutput import MultiOutputClassifier  # type: ignore

    if modeltype == "rf":
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        base_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=4,
            random_state=FLAGS.seed,
        )
    elif modeltype == "lr":
        from sklearn.linear_model import LogisticRegression # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        from sklearn.pipeline import make_pipeline  # type: ignore
        # scale features (age, vitals and counts span very different ranges) so that lbfgs converges
        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=3e-5,
                max_iter=5000,
                random_state=FLAGS.seed,
                verbose=True,
            ),
        )
    elif modeltype == "gb":
        from sklearn.ensemble import GradientBoostingClassifier
        base_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=FLAGS.seed,
        )
    else:
        raise ValueError(f"Unsupported model type: {modeltype}")

    # Wrap with MultiOutputClassifier for multi-label prediction
    model = MultiOutputClassifier(base_model)
    return model


def mimic_iv_evaluation_fn(model, data:Tuple[np.ndarray, np.ndarray], prefix: str = ""):
    """Evaluation function for MIMIC-IV-ED multi-label classification that returns metrics for logging."""
    X, y = data
    # Get predictions - shape (n, 2) for two binary labels
    pred = model.predict(X)

    # Get probabilities for AUC calculation
    # MultiOutputClassifier.predict_proba returns a list of arrays, one per output
    proba_list = model.predict_proba(X)
    # Stack probabilities into (n, 2) array - take positive class probability for each label
    proba = np.column_stack([proba_list[i][:, 1] for i in range(len(proba_list))])

    # Calculate metrics per label
    metrics = {}
    label_names = ["hospitalization", "critical"]

    for i, label_name in enumerate(label_names):
        acc = accuracy_score(y[:, i], pred[:, i])
        auc = roc_auc_score(y[:, i], proba[:, i])
        metrics[f"{prefix}_{label_name}_accuracy"] = acc
        metrics[f"{prefix}_{label_name}_auc"] = auc

    # Calculate macro-averaged metrics
    metrics[f"{prefix}_macro_accuracy"] = np.mean([metrics[f"{prefix}_{label_name}_accuracy"] for label_name in label_names])
    metrics[f"{prefix}_macro_auc"] = roc_auc_score(y, proba, average='macro')

    return metrics


def main(argv):
    # Load datasets
    train_historical_dataset, val_dataset, train_future_dataset, test_dataset = get_dataset(
        dataset_name="mimic-iv-ed",
        csv_root=Path(FLAGS.csv_root),
        data_root=None,
        save_root=None,
        use_cached=True,
        load_from_disk=True,
        write_to_disk=FLAGS.write_to_disk,
    )
    print(train_historical_dataset)
    # Create model
    sklearn_model = get_sklearn_model(modeltype=FLAGS.model)
    print(sklearn_model)
    if not FLAGS.train_random_subset:
        print("\n=== Training on Full Dataset ===")

        # Train and evaluate
        trained_model, training_history, train_metrics, val_metrics, test_metrics, failed = train_and_eval_sklearn(
            sklearn_model=sklearn_model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            seed=FLAGS.seed,
            target_metric="macro_auc",
            flatten_input=False,  # MIMIC-IV-ED is already tabular data
            track_data_stats=True,
            wandb_project_name="mimic-iv-ed-sklearn",
            log_wandb=FLAGS.log_wandb,
            verbose=True,
            evaluation_fn=mimic_iv_evaluation_fn,
        )

        if not failed:
            print(f"Training completed successfully! Metrics logged: {train_metrics}")

        else:
            print("Training failed!")

    else:
        print("\n=== Single Training Run on Random Subset ===")
        print(f"Subset ratio: {FLAGS.subset_ratio}")
        print(f"Log directory: {FLAGS.logdir}")
        trained_model = train_random_subset_sklearn(
            sklearn_model=sklearn_model,
            train_historical_dataset=train_historical_dataset,
            val_dataset=val_dataset,
            train_future_dataset=train_future_dataset,
            test_dataset=test_dataset,
            patient_id_col="subject_id",  # MIMIC-IV-ED patient identifier
            seed=FLAGS.seed,
            target_metric="macro_auc",
            logdir=Path(FLAGS.logdir),
            flatten_input=False,  # MIMIC-IV-ED is already tabular data
            n_total_runs=FLAGS.n_runs,
            subset_ratio=FLAGS.subset_ratio,
            track_data_stats=True,
            wandb_project_name="mimic-iv-ed-sklearn",
            log_wandb=FLAGS.log_wandb,
            verbose=True,
            evaluation_fn=mimic_iv_evaluation_fn,
        )


if __name__ == "__main__":
    app.run(main)