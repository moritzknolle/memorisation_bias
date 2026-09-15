"""One entry point for loading a dataset's four splits.

`get_dataset(name)` reads the split files written by the preprocessing notebooks from
`csv_root` and wraps each in the matching `BaseDataset` subclass, deciding per split
whether it is held in memory or memory-mapped.

Known names: 'mimic-cxr', 'mimic-ecg', 'mimic-iv-ed', 'heedb', plus the '_p' variants
('mimic-cxr_p', 'mimic-ecg_p', 'heedb_p'). The '_p' variants keep one record per patient
in the *training* split only -- one training example is then one patient, which is what
patient-level DP requires -- and deliberately leave the future split untouched, so both
levels are evaluated on exactly the same records.
"""

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
from typing import Tuple, Optional

from src.data.datasets import (
    CXRDataset,
    MIMICIVEDDataset,
    MIMICECGDataset,
    HEEDBDataset,
    BaseDataset,
)


def get_dataset(
    dataset_name: str,
    csv_root: Path,
    data_root: Path,
    save_root: Path,
    resolution: Optional[int]=None,
    use_cached: bool = False,
    load_from_disk: bool = False,
    write_to_disk: bool = False,
    overwrite_existing: bool = False,
    n_threads:int=64,
    chunk_size:int=100,
)-> Tuple[BaseDataset, BaseDataset, BaseDataset, BaseDataset]:  
    """
    Convenience function to retrieve a dataset by name. Raises a ValueError if the dataset is unknown.
        Args:
            dataset_name: str, name of the dataset (see the module docstring for the known names)
            csv_root: Path, directory holding the <dataset>_{train_historical,train_future,val,test}
                split files written by the preprocessing notebooks
            data_root: Path, root of the raw images/waveforms the splits refer to
            save_root: Path, root of the .npy/.mmp cache (a <dataset>/ subdirectory is created)
            resolution: Optional[int], image edge length for mimic-cxr, sampling rate in Hz for
                mimic-ecg. Ignored by datasets with a fixed input size
            use_cached: bool, whether to load and use the cached version of the dataset. If cache does not exist, it will be created.
            load_from_disk: bool, whether to load from disk
            write_to_disk: bool, whether to (re)build the cache and write it to disk
            overwrite_existing: bool, whether to overwrite existing cache files
            n_threads:int, number of threads to use for data loading and preprocessing
            chunk_size:int, number of samples per chunk for multiprocessing
        Returns:
            Tuple of four BaseDataset objects, in the order
            (train_historical, val, train_future, test). Note that val comes second --
            it is not the same order as the (historical, future, val, test) narrative
            order used elsewhere.
    """
    if dataset_name == "mimic-cxr":
        train_historical_df = pd.read_csv(csv_root / "mimic-cxr_train_historical.csv")
        train_future_df = pd.read_csv(csv_root / "mimic-cxr_train_future.csv")
        val_df = pd.read_csv(csv_root / "mimic-cxr_val.csv")
        test_df = pd.read_csv(csv_root / "mimic-cxr_test.csv")
        fits_memory = resolution <= 64
        train_historical_dataset = CXRDataset(
            df=train_historical_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="train",
            save_root=save_root,
            fits_memory=fits_memory,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        train_future_dataset = CXRDataset(
            df=train_future_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="train_future",
            save_root=save_root,
            fits_memory=fits_memory,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        val_dataset = CXRDataset(
            df=val_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="val",
            save_root=save_root,
            fits_memory=True,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        test_dataset = CXRDataset(
            df=test_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="test",
            save_root=save_root,
            fits_memory=False,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
    # one image per patient version of MIMIC-CXR (required for patient-level DP guarantees)
    elif dataset_name == "mimic-cxr_p":
        train_historical_df = pd.read_csv(csv_root / "mimic-cxr_train_historical.csv")
        train_future_df = pd.read_csv(csv_root / "mimic-cxr_train_future.csv")
        val_df = pd.read_csv(csv_root / "mimic-cxr_val.csv")
        test_df = pd.read_csv(csv_root / "mimic-cxr_test.csv")
        fits_memory = resolution <= 64
        train_historical_dataset = CXRDataset(
            df=train_historical_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="train",
            save_root=save_root,
            one_image_per_patient=True,
            fits_memory=fits_memory,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        train_future_dataset = CXRDataset(
            df=train_future_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="train_future",
            save_root=save_root,
            one_image_per_patient=False,  # keep multiple images per patient in the future dataset
            fits_memory=fits_memory,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        val_dataset = CXRDataset(
            df=val_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="val",
            save_root=save_root,
            fits_memory=True,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        test_dataset = CXRDataset(
            df=test_df,
            img_path=data_root / "files",
            name="mimic-cxr",
            img_size=[resolution, resolution],
            split="test",
            save_root=save_root,
            fits_memory=False,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
    elif dataset_name == "mimic-iv-ed":
        train_historical_df = pd.read_csv(csv_root / "mimic-iv-ed_train_historical.csv")
        train_future_df = pd.read_csv(csv_root / "mimic-iv-ed_train_future.csv")
        val_df = pd.read_csv(csv_root / "mimic-iv-ed_val.csv")
        test_df = pd.read_csv(csv_root / "mimic-iv-ed_test.csv")
        # Create train_future split from test for consistency
        train_historical_dataset = MIMICIVEDDataset(dataframe=train_historical_df)
        train_future_dataset = MIMICIVEDDataset(dataframe=train_future_df)
        val_dataset = MIMICIVEDDataset(dataframe=val_df)
        test_dataset = MIMICIVEDDataset(dataframe=test_df)
        # MIMIC-IV-ED does not support iterative loading of data, so we return the full dataset directly
        if use_cached:
            train_historical_dataset.inputs = train_historical_dataset.__get_all_inputs__()
            train_historical_dataset.targets = train_historical_dataset.__get_all_targets__()
            val_dataset.inputs = val_dataset.__get_all_inputs__()
            val_dataset.targets = val_dataset.__get_all_targets__()
            test_dataset.inputs = test_dataset.__get_all_inputs__()
            test_dataset.targets = test_dataset.__get_all_targets__()
            train_future_dataset.inputs = train_future_dataset.__get_all_inputs__()
            train_future_dataset.targets = train_future_dataset.__get_all_targets__()
            return train_historical_dataset, val_dataset, train_future_dataset, test_dataset
    elif dataset_name == "mimic-ecg":
        train_historical_df = pd.read_csv(csv_root / "mimic-ecg_train_historical.csv")
        train_future_df = pd.read_csv(csv_root / "mimic-ecg_train_future.csv")
        val_df = pd.read_csv(csv_root / "mimic-ecg_val.csv")
        test_df = pd.read_csv(csv_root / "mimic-ecg_test.csv")
        train_historical_dataset = MIMICECGDataset(
            df=train_historical_df,
            data_path=data_root,
            name="mimic-ecg",
            split="train",
            save_root=save_root,
            fits_memory=False,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        train_future_dataset = MIMICECGDataset(
            df=train_future_df,
            data_path=data_root,
            name="mimic-ecg",
            split="train_future",
            save_root=save_root,
            fits_memory=False,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        val_dataset = MIMICECGDataset(
            df=val_df,
            data_path=data_root,
            name="mimic-ecg",
            split="val",
            save_root=save_root,
            fits_memory=True,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        test_dataset = MIMICECGDataset(
            df=test_df,
            data_path=data_root,
            name="mimic-ecg",
            split="test",
            save_root=save_root,
            fits_memory=True,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
    # one ECG per patient version of MIMIC-ECG
    elif dataset_name == "mimic-ecg_p":
        train_historical_df = pd.read_csv(csv_root / "mimic-ecg_train_historical.csv")
        train_future_df = pd.read_csv(csv_root / "mimic-ecg_train_future.csv")
        val_df = pd.read_csv(csv_root / "mimic-ecg_val.csv")
        test_df = pd.read_csv(csv_root / "mimic-ecg_test.csv")
        train_historical_dataset = MIMICECGDataset(
            df=train_historical_df,
            data_path=data_root,
            name="mimic-ecg",
            split="train",
            save_root=save_root,
            one_record_per_patient=True,
            fits_memory=False,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        train_future_dataset = MIMICECGDataset(
            df=train_future_df,
            data_path=data_root,
            name="mimic-ecg",
            split="train_future",
            save_root=save_root,
            one_record_per_patient=False,  # we want to keep multiple ECGs per patient in the future dataset
            fits_memory=False,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        val_dataset = MIMICECGDataset(
            df=val_df,
            data_path=data_root,
            name="mimic-ecg",
            split="val",
            save_root=save_root,
            fits_memory=True,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
        test_dataset = MIMICECGDataset(
            df=test_df,
            data_path=data_root,
            name="mimic-ecg",
            split="test",
            save_root=save_root,
            fits_memory=True,
            fs=resolution,
            n_threads=n_threads,
            chunk_size=chunk_size,
        )
    elif dataset_name == "heedb":
        train_historical_df = pd.read_pickle(csv_root / "heedb_train_historical.pkl")
        train_future_df = pd.read_pickle(csv_root / "heedb_train_future.pkl")
        val_df = pd.read_pickle(csv_root / "heedb_val.pkl")
        test_df = pd.read_pickle(csv_root / "heedb_test.pkl")
        train_historical_dataset = HEEDBDataset(
            df=train_historical_df,
            data_path=data_root,
            name="heedb",
            split="train",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
        )
        print("Sanity check, train label shape:", train_historical_dataset.__get_all_targets__().shape)
        train_future_dataset = HEEDBDataset(
            df=train_future_df,
            data_path=data_root,
            name="heedb",
            split="train_future",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
        )
        val_dataset = HEEDBDataset(
            df=val_df,
            data_path=data_root,
            name="heedb",
            split="val",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
        )
        test_dataset = HEEDBDataset(
            df=test_df,
            data_path=data_root,
            name="heedb",
            split="test",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
        )
    # one ECG per patient version of HEEDB
    elif dataset_name == "heedb_p":
        train_historical_df = pd.read_pickle(csv_root / "heedb_train_historical.pkl")
        train_future_df = pd.read_pickle(csv_root / "heedb_train_future.pkl")
        val_df = pd.read_pickle(csv_root / "heedb_val.pkl")
        test_df = pd.read_pickle(csv_root / "heedb_test.pkl")
        train_historical_dataset = HEEDBDataset(
            df=train_historical_df,
            data_path=data_root,
            name="heedb",
            split="train",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
            one_record_per_patient=True,
        )
        print("Sanity check, train label shape:", train_historical_dataset.__get_all_targets__().shape)
        train_future_dataset = HEEDBDataset(
            df=train_future_df,
            data_path=data_root,
            name="heedb",
            split="train_future",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
            one_record_per_patient=False, # we want to keep multiple ECGs per patient in the future dataset
        )
        val_dataset = HEEDBDataset(
            df=val_df,
            data_path=data_root,
            name="heedb",
            split="val",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
        )
        test_dataset = HEEDBDataset(
            df=test_df,
            data_path=data_root,
            name="heedb",
            split="test",
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=False,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    if use_cached:
        to_process = [train_historical_dataset, val_dataset, train_future_dataset, test_dataset]
        # cache np.arrays into memory or as memmap file
        for dataset in to_process:
                dataset.cache_numpy(
                    load_from_disk=load_from_disk,
                    write_to_disk=overwrite_existing or write_to_disk,
                )
                dataset.check_shapes()
    return train_historical_dataset, val_dataset, train_future_dataset, test_dataset
