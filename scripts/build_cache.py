"""Write the input cache of a dataset (see README.md) without training.

Usage:
    python scripts/build_cache.py --dataset=mimic-ecg
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
# the repository root is one level up from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from absl import app, flags  # type: ignore

from src.data.dataset_factory import get_dataset
from src.utils import get_data_root

FLAGS = flags.FLAGS

# dataset name -> resolution used by the training scripts
DATASETS = {
    "mimic-cxr": 256,
    "mimic-ecg": 250,
    "mimic-ecg_p": 250,
    "heedb": None,
    "heedb_p": None,
}

flags.DEFINE_enum("dataset", None, list(DATASETS), "Dataset to cache.")
flags.DEFINE_string("data_root", None, "Path to the raw data. Default: get_data_root in src/utils.py.")
flags.DEFINE_string("csv_root", "./data/csv", "Path to the split files.")
flags.DEFINE_string("save_root", "./data/npy/", "Path to root folder where the cache is stored.")
flags.DEFINE_integer("n_threads", 16, "Number of processes used for preprocessing.")
flags.DEFINE_integer("chunk_size", 1024, "Number of records per process chunk.")
flags.mark_flag_as_required("dataset")


def main(argv):
    data_root = FLAGS.data_root or get_data_root(FLAGS.dataset.removesuffix("_p"))
    get_dataset(
        dataset_name=FLAGS.dataset,
        csv_root=Path(FLAGS.csv_root),
        data_root=Path(data_root),
        save_root=Path(FLAGS.save_root),
        resolution=DATASETS[FLAGS.dataset],
        use_cached=True,
        load_from_disk=True,
        write_to_disk=True,
        n_threads=FLAGS.n_threads,
        chunk_size=FLAGS.chunk_size,
    )
    print(f"... cache of {FLAGS.dataset} written to {FLAGS.save_root}")


if __name__ == "__main__":
    app.run(main)
