"""Dataset classes.

`BaseDataset` implements caching; subclasses implement `__getinput__` (load and preprocess one
record) and `__gettarget__`.

`cache_numpy()` preprocesses all records once, in parallel over `n_threads`, and writes them to
`save_root/<name>/<name>_<split>_<w>x<h>_inputs.{npy,mmp}` together with a `.json` file holding
shapes, dtypes and a SHA1 hash of the split dataframe. The hash is checked when loading, so a
changed split file raises an error instead of being paired with an outdated cache.
`fits_memory` selects between an in-memory `.npy` and a memory-mapped `.mmp` that is streamed
during training.

Subclasses:
    CXRDataset       chest radiographs, resized to img_size.
    MIMICECGDataset  10 s 12-lead ECGs from WFDB records, resampled to `fs` and z-scored.
    HEEDBDataset     the same for HEEDB, at 250 Hz.
    MIMICIVEDDataset tabular; held in memory without caching and implements only the parts of
                     the BaseDataset interface used by the training code.

The ECG classes take `one_record_per_patient`, which keeps each patient's last record and
appends '_p' to the dataset name so that its cache is kept separate.
"""

import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Generator, List, Tuple, Optional

import h5py  # type: ignore
import keras  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import scipy
from scipy.signal import medfilt, iirnotch, filtfilt, butter, resample
import wfdb  # type: ignore
from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

from .constants import (
    CXR_LABELS,
    HEEDB_CODE_DICT,
    MIMIC_ECG_LABELS,
    MIMIC_IV_ED_LABELS,
)


# Global worker function for multiprocessing (must be at module level to be picklable)
def _write_memmap_chunk_worker(args):
    """
    Worker function to write a chunk of data to a memmap file.
    This is a module-level function so it can be pickled for multiprocessing.

    Args:
        args: Tuple of (chunk_indices, memmap_path, example_shape, dtype, dataset_init_args)

    Yields:
        int: Number of samples processed (yields after each sample for progress tracking)
    """
    (
        chunk_indices,
        memmap_path,
        example_shape,
        dtype,
        dataset_class,
        dataset_init_args,
    ) = args

    # Recreate the dataset instance in the worker process
    dataset = dataset_class(**dataset_init_args)

    # Open the memmap file in read-write mode
    # Each process needs its own file handle
    memmap_array = np.memmap(
        filename=memmap_path,
        dtype=dtype,
        mode="r+",
        shape=(len(chunk_indices), *example_shape),
        offset=chunk_indices[0] * np.prod(example_shape) * np.dtype(dtype).itemsize,
    )

    # Write each sample in this chunk and yield progress
    samples_written = 0
    for local_idx, global_idx in enumerate(chunk_indices):
        memmap_array[local_idx] = dataset.__getinput__(global_idx)
        samples_written += 1

        # Flush periodically (every 100 samples) to avoid memory buildup
        if samples_written % 50 == 0:
            memmap_array.flush()

    # Final flush to ensure all data is written to disk
    memmap_array.flush()
    del memmap_array  # Close the file handle

    return samples_written


def subset_memmap_generator(
    memmap_input_array: np.memmap,
    target_array_subset: np.ndarray,
    subset_indices: np.ndarray,
) -> Generator:
    """
    Creates a generator that yields elements from a numpy.memmap based on a list of indices.
    This avoids loading the full subset of the array into memory.

    Args:
        memmap_input_array (np.memmap): The input memmap array (full array).
        target_array_subset (np.ndarray): The target array subset (already indexed by subset_indices).
        subset_indices (np.ndarray): List of indices to yield from the memmap array.

    Yields:
        Generator: Yields tuples of (input, target) for each index in subset_indices.
    """
    for i, index in enumerate(subset_indices):
        yield memmap_input_array[index].copy(), target_array_subset[i].copy()


def memmap_generator(
    memmap_input_array: np.memmap, target_array: np.ndarray
) -> Generator:
    """
    Creates a generator that yields elements from a numpy.memmap
    without loading the full array into memory.
    Args:
        memmap_input_array (np.memmap): The input memmap array.
        target_array (np.ndarray): The target array.
    Yields:
        Generator: Yields tuples of (input, target) for each index in the memmap array.
    """
    for index in range(len(memmap_input_array)):
        yield memmap_input_array[index].copy(), target_array[index].copy()


def preprocess_ecg(signal: np.ndarray, fs: int):
    """Preprocess a 12-lead ECG signal by applying:
        1. Notch filter to remove power-line interference at 50 Hz
        2. Bandpass filter between 0.67 Hz and 40 Hz to reduce high-frequency noise
        3. Median filter to remove baseline wander
    Credit for preprocessing code goes to ECGfounder (https://github.com/NickLJLee/ECGFounder/blob/67367071b3f97332ba0f933f17ec7402eba36d84/util.py#L28)
    Args:
        signal (np.ndarray): Input ECG signal of shape (num_samples, 12)
        fs (int): Sampling frequency of the ECG signal
    Returns:
        np.ndarray: Preprocessed ECG signal of shape (num_samples, 12)
    """
    num_channels = signal.shape[1]
    assert num_channels == 12, f"Input ECG signal must have 12 channels, found {num_channels}"
    filtered_signal = np.zeros_like(signal)
    # --- 1. Remove power-line interference ---
    b, a = iirnotch(50, 30, fs)
    for c in range(num_channels):
        filtered_signal[:, c] = filtfilt(b, a, signal[:, c])
    # --- 2. Simple bandpass filter ---
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(num_channels):
        filtered_signal[:, c] = filtfilt(b, a, filtered_signal[:, c])
    # --- 3. Remove baseline wander ---
    baseline = np.zeros_like(filtered_signal)
    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd
    for c in range(num_channels):
        baseline[:, c] = medfilt(filtered_signal[:, c], kernel_size=kernel_size)
    return filtered_signal - baseline


class BaseDataset:
    """
    Base class for datasets with caching and preprocessing capabilities.

    This class provides a common interface for all dataset implementations. It handles
    loading, preprocessing, caching, and retrieval of image data and corresponding labels.
    The class is designed to be subclassed for specific dataset types, with subclasses
    implementing the core data retrieval methods.

    Args:
        df (pd.DataFrame): DataFrame containing dataset metadata and labels
        img_path (Path): Directory containing the image files
        name (str): Name of the dataset for identification
        num_classes (int): Number of output classes for classification
        in_channels (int): Number of channels in the images (1 for grayscale, 3 for RGB)
        img_size (int): Target size for image resizing (assumes square images)
        split (str): Dataset split identifier (e.g., 'train', 'val', 'test')
        save_root (Path): Directory for caching preprocessed data
        n_threads (int, optional): Number of parallel threads for data loading. Defaults to 64.
        chunk_size (int, optional): Number of samples per chunk for multiprocessing. Defaults to 100.

    Attributes:
        dataframe (pd.DataFrame): The dataset metadata and labels
        img_path (Path): Directory containing the image files
        name (str): Dataset name
        num_classes (int): Number of output classes
        save_root (Path): Directory for cached data
        split (str): Dataset split identifier
        in_channels (int): Number of image channels
        img_size (Tuple): Target image size
        filename (str): Base filename for cached data
        n_threads (int): Number of threads to use for parallel processing
        chunk_size (int): Number of samples per chunk for multiprocessing
        inputs (np.ndarray): Cached input data
        targets (np.ndarray): Cached target data
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        num_classes: int,
        in_channels: int,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
        n_threads: int = 64,
        chunk_size: int = 100,
        fits_memory: bool = True,
        silent: bool = False,
        _worker_mode: bool = False,  # Internal flag for lightweight worker initialization
    ):
        if not silent:
            print(f"... Creating {name} dataset ({split})")
        super().__init__()
        self.dataframe = df
        self.data_path = img_path
        self.name = name
        self.num_classes = num_classes
        self.save_root = Path(save_root) / name
        self.split = split
        self.in_channels = in_channels
        self.img_size = img_size
        self.filename = f"{name}_{split}_{self.img_size[0]}x{self.img_size[1]}"
        self.n_threads = n_threads
        self.chunk_size = chunk_size
        self.fits_memory = fits_memory
        self._worker_mode = _worker_mode
        self.save_root.mkdir(parents=True, exist_ok=True)
        if not isinstance(self.data_path, Path):
            self.data_path = Path(self.data_path)
        if not self.data_path.exists() and not silent:
            print(f"Warning: Data path {self.data_path} does not exist")

    def __getinput__(self, index: int) -> np.ndarray:
        """
        Retrieve and preprocess the input data for a specific index.

        This method should be implemented by subclasses to load and preprocess
        the input data (typically images) for the given index.

        Args:
            index (int): Index of the data point to retrieve

        Returns:
            np.ndarray: Preprocessed input data
        """
        raise NotImplementedError()

    def __gettarget__(self, index: int) -> np.ndarray:
        """
        Retrieve the target data (labels) for a specific index.

        This method should be implemented by subclasses to load and format
        the target data for the given index.

        Args:
            index (int): Index of the data point to retrieve

        Returns:
            np.ndarray: Formatted target data
        """
        raise NotImplementedError()

    def __get_all_targets__(self) -> np.ndarray:
        """
        Retrieve all target data (labels) in the dataset at once.

        This method should be implemented by subclasses to load and format
        all target data.

        Returns:
            np.ndarray: Formatted target data for all samples
        """
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a data point and its label by index.

        This method combines the input and target retrieval methods to provide
        a complete data point.

        Args:
            index (int): Index of the data point to retrieve

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of (input, target)
        """
        inp = self.__getinput__(index)
        target = self.__gettarget__(index)
        return inp, target

    def _get_init_args(self):
        """
        Get the initialization arguments for recreating this dataset instance.
        This method should be overridden by subclasses if they have different init signatures.

        Returns:
            dict: Dictionary of initialization arguments
        """
        return {
            "df": self.dataframe,
            "img_path": self.data_path,
            "name": self.name,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "img_size": self.img_size,
            "split": self.split,
            "save_root": self.save_root.parent,  # Remove the name subfolder
            "n_threads": 1,  # Each worker only needs 1 thread
            "chunk_size": self.chunk_size,
            "fits_memory": self.fits_memory,
            "silent": True,  # Suppress print statements in worker processes
            "_worker_mode": True,  # Enable lightweight worker mode
        }

    def _write_memmap_parallel(
        self,
        final_memmap_path,
        example_shape,
        example_dtype,
        n_samples,
    ):
        """
        Write data to memmap file using multiprocessing.

        This method divides the dataset into chunks and processes them in parallel,
        with each worker writing directly to its designated portion of the memmap file.
        Progress is tracked per sample for accurate monitoring.

        Args:
            final_memmap_path: Path to the final memmap file
            example_shape: Shape of a single sample
            example_dtype: Data type of samples
            n_samples: Total number of samples
        """
        chunk_size = self.chunk_size
        # Calculate chunk size for each worker
        chunks = [
            list(range(i, min(i + chunk_size, n_samples)))
            for i in range(0, n_samples, chunk_size)
        ]
        print(
            f"... created chunks for multiprocessing: {len(chunks)} chunks of up to {chunk_size} samples each"
        )
        # Get initialization args for recreating dataset in workers
        dataset_init_args = self._get_init_args()
        dataset_class = self.__class__

        # Prepare arguments for each worker
        worker_args = [
            (
                chunk,
                final_memmap_path,
                example_shape,
                example_dtype,
                dataset_class,
                dataset_init_args,
            )
            for chunk in chunks
        ]

        # Process chunks in parallel with progress bar tracking chunks
        with mp.Pool(self.n_threads) as pool:
            # Use imap_unordered for better performance and track chunks completed
            with tqdm(
                total=len(chunks), desc="Writing memmap chunks", unit="chunks"
            ) as pbar:
                for _ in pool.imap_unordered(
                    _write_memmap_chunk_worker, worker_args, chunksize=1
                ):
                    pbar.update(1)

        print("... finished writing memmap file")

    def _load_from_scratch(self):
        """
        Load the dataset into memory by iterating through it.
        """
        success = False
        convenience_target_fn_exists = callable(
            getattr(self, "__get_all_targets__", None)
        )
        with mp.Pool(self.n_threads) as pool:
            if self.fits_memory:
                # load all inputs into memory
                print("... caching data into memory")
                inputs = list(
                    tqdm(
                        pool.imap(
                            self.__getinput__, range(self.__len__()), chunksize=32
                        ),
                        desc="Loading inputs",
                        total=self.__len__(),
                    )
                )
                self.inputs = np.array(inputs, dtype=inputs[0].dtype)
            else:
                print("... caching data into np.memmap file on disk")
                example = self.__getinput__(0)
                example_shape = example.shape
                example_dtype = example.dtype
                # Create the memmap file
                final_memmap_path = self.save_root / f"{self.filename}_inputs.mmp"
                self.inputs = np.memmap(
                    filename=final_memmap_path,
                    dtype=example_dtype,
                    mode="w+",
                    shape=(self.__len__(), *example_shape),
                )
                print(
                    f"... writing np.memmap file with shape {self.inputs.shape} and dtype {self.inputs.dtype}"
                )
                # Write memmap array content 
                self._write_memmap_parallel(
                    final_memmap_path=final_memmap_path,
                    example_shape=example_shape,
                    example_dtype=example_dtype,
                    n_samples=self.__len__(),
                )
            if not convenience_target_fn_exists:
                targets = list(
                    tqdm(
                        pool.imap(
                            self.__gettarget__, range(self.__len__()), chunksize=32
                        ),
                        desc="Loading targets",
                        total=self.__len__(),
                    )
                )
                self.targets = np.array(targets, dtype=targets[0].dtype)
            else:
                self.targets = self.__get_all_targets__()
        success = True

        return success

    def __len__(self) -> int:
        """
        Get the number of data points in the dataset.

        This method should be implemented by subclasses to return the total
        number of data points.

        Returns:
            int: The number of data points
        """
        raise NotImplementedError()

    def _write_metadata(self):
        """Write the cache sidecar: array shapes, dtypes and a SHA1 of the dataframe.

        The hash is what `_load_from_file` checks before trusting the cached arrays, so
        a re-run preprocessing notebook fails loudly instead of pairing new labels with
        stale inputs. Written atomically (temp file + rename) because parallel workers
        on a shared filesystem may reach this at the same time.
        """
        if self.inputs is None or self.targets is None:
            raise ValueError(
                "Cannot save dataset to disk, inputs and targets are None. Load the dataset first."
            )
        base_file_name = self.save_root / f"{self.filename}"
        df_hash = hashlib.sha1(
            pd.util.hash_pandas_object(self.dataframe).values
        ).hexdigest()
        # Write metadata JSON file using atomic write
        metadata = {
            "input_shape": self.inputs.shape,
            "target_shape": self.targets.shape,
            "df_hash": df_hash,
            "input_dtype": str(self.inputs.dtype),
            "target_dtype": str(self.targets.dtype),
        }
        metadata_file = f"{base_file_name}.json"
        temp_metadata_file = f"{metadata_file}.tmp.{os.getpid()}"
        try:
            with open(temp_metadata_file, "w") as file:
                json.dump(metadata, file)
            # Atomic rename (POSIX guarantees atomicity)
            os.rename(temp_metadata_file, metadata_file)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_metadata_file):
                os.remove(temp_metadata_file)
            raise

    def _load_from_file(self) -> bool:
        """
        Load a previously cached dataset from disk.

        This method attempts to load the dataset from cached .npy files and
        validates that the data is consistent with the current dataframe.

        Returns:
            bool: True if loading was successful, False otherwise

        Raises:
            ValueError: If the dataframe has changed since the cache was created
        """
        success = False
        base_file_name = self.save_root / f"{self.filename}"
        metadata_file = Path(f"{base_file_name}.json")
        print(f"... looking for dataset at: {self.save_root}")
        print(f"... checking metadata file: {metadata_file}")
        if metadata_file.is_file():
            # read metadata file
            with open(metadata_file, "r") as file:
                metadata = json.load(file)
                input_shape = tuple(metadata["input_shape"])
                target_shape = tuple(metadata["target_shape"])
            self.targets = self.__get_all_targets__()
            if target_shape != self.targets.shape:
                warnings.warn(
                    f"Target shape mismatch: {target_shape} vs. {self.targets.shape}"
                )
            # check if dataframe has changed since the dataset was created
            current_df_hash = hashlib.sha1(
                pd.util.hash_pandas_object(self.dataframe).values
            ).hexdigest()
            if metadata["df_hash"] != current_df_hash:
                raise ValueError(
                    f"Careful, hash mismatch! Underlying dataframe has changed since npy files were created: {metadata['df_hash']} vs. {current_df_hash}"
                )
            if self.fits_memory:
                # load .npy file
                input_file = Path(f"{base_file_name}_inputs.npy")
                # lock file to prevent race conditions
                with open(input_file, "rb") as f:
                    fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock for reads
                    self.inputs = np.load(f)  # Pass file handle, not path
                    fcntl.flock(f, fcntl.LOCK_UN)
                assert (
                    input_shape == self.inputs.shape
                ), f"Input shape mismatch: {input_shape} vs. {self.inputs.shape}"
            else:
                # load .mmp (npy memmap) file
                input_file = Path(f"{base_file_name}_inputs.mmp")
                # Acquire shared lock before creating memmap
                # Note: We keep a reference to the lock file to maintain the lock
                # while the memmap is being created and validated
                lock_file = open(input_file, "rb")
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_SH)  # Shared lock for reads
                    self.inputs = np.memmap(
                        input_file,
                        dtype=metadata["input_dtype"],
                        mode="r",
                        shape=input_shape,
                    )
                    assert (
                        input_shape == self.inputs.shape
                    ), f"Input shape mismatch: {input_shape} vs. {self.inputs.shape}"
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    lock_file.close()
            success = True

        print("... succesfully loaded dataset from disk") if success else 0
        return success

    def _write_arrays_to_disk(self) -> None:
        """Write a loaded dataset to disk (saves two .npz files and a metadata file)."""
        self._write_metadata()
        base_file_name = self.save_root / f"{self.filename}"
        if self.fits_memory:
            temp_file = f"{base_file_name}_inputs.tmp.{os.getpid()}.npy"
            try:
                np.save(temp_file, self.inputs)
                os.rename(temp_file, f"{base_file_name}_inputs.npy")
            except Exception as e:
                print(e)
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise
        else:
            memmap_file = Path(f"{base_file_name}_inputs.mmp")
            assert memmap_file.is_file(), f"Memmap file {memmap_file} does not exist"


    def cache_numpy(
        self, load_from_disk: bool = True, write_to_disk: bool = False
    ) -> None:
        """Load the dataset into memory, optionally loading from or writing to disk."""
        if load_from_disk:
            success = self._load_from_file()
            if not success:
                print("... loading dataset from disk failed, loading from scratch")
                success = self._load_from_scratch()
        else:
            success = self._load_from_scratch()
        if write_to_disk and success:
            self._write_arrays_to_disk()
            print("... sucessfully wrote dataset to disk")
        assert success, "loading dataset failed"


    def check_shapes(self) -> None:
        """
        Validate and potentially fix the shapes of the input and target arrays.

        This method ensures that the input arrays have the correct number of dimensions
        and that the channel dimension is in the correct position. It also verifies
        that the number of samples matches between inputs and targets.

        Raises:
            AssertionError: If the shapes are incompatible
        """
        # add dimension if necessary
        if len(self.targets.shape) == 1:
            self.targets = np.expand_dims(self.targets, axis=-1)
        if self.inputs is not None:
            # shape check
            assert (
                self.inputs.shape[0] == self.targets.shape[0]
            ), f"Input shape {self.inputs.shape} does not match target shape {self.targets.shape}"
        else:
            assert (
                self.__len__() == self.targets.shape[0]
            ), f"Dataset length {self.__len__()} does not match target shape {self.targets.shape}"


class CXRDataset(BaseDataset):
    """
    Dataset class for Chest X-Ray (CXR) images.

    This class handles loading and preprocessing of chest X-ray images and their
    associated labels. It supports different label strategies for handling uncertain
    annotations.

    Args:
        df (pd.DataFrame): DataFrame containing CXR metadata and labels
        img_path (Path): Directory containing the CXR images
        name (str): Name of the dataset
        img_size (tuple): Target size for image resizing
        split (str): Dataset split identifier (e.g., 'train', 'val', 'test')
        save_root (Path): Directory for caching preprocessed data
        fits_memory (bool, optional): whether the dataset fits into memory. Defaults to True.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_path: Path,
        name: str,
        img_size: Tuple[int, int],
        split: str,
        save_root: Path,
        imagenet_normalize:bool=True,
        one_image_per_patient: bool = False,
        fits_memory: bool = True,
        n_threads: int = 64,
        chunk_size: int = 100,
        silent: bool = False,
        _worker_mode: bool = False,
        _path_array: Optional[List[str]] = None,  # Lightweight path array for workers
    ):
        self.imagenet_normalize=imagenet_normalize
        if one_image_per_patient and not _worker_mode:
            df = (
                df.sort_values("timestamp")
                .drop_duplicates(subset="subject_id", keep="first")
                .reset_index(drop=True)
            )
            name = f"{name}_p"
        super().__init__(
            df=df,
            img_path=img_path,
            name=name,
            num_classes=14,
            in_channels=1,
            img_size=img_size,
            split=split,
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=fits_memory,
            silent=silent,
            _worker_mode=_worker_mode,
        )

        # Worker mode: use lightweight path array
        if _worker_mode and _path_array is not None:
            self._path_array = _path_array
        else:
            # Main process mode: use full dataframe
            self.dataframe = self.dataframe.fillna(0)
            if "Path" in self.dataframe.columns:
                self.dataframe.rename(columns={"Path": "path"}, inplace=True)
            assert "path" in self.dataframe.columns, "Dataframe must have a 'path' column"
            self._path_array = None

    def __getinput__(self, index: int) -> np.ndarray:
        """
        Load and preprocess a CXR image.

        This method loads the image from disk, resizes it to the target size,
        normalizes the pixel values to the range [0, 1], and adds a channel dimension.

        Args:
            index (int): Index of the image to load

        Returns:
            np.ndarray: Preprocessed image with shape (img_size[0], img_size[1], 1)

        Raises:
            AssertionError: If the data path doesn't exist or the data shape is incorrect
        """
        # Use lightweight path array in worker mode, dataframe otherwise
        if self._worker_mode and self._path_array is not None:
            path_str = self._path_array[index]
        else:
            path_str = self.dataframe["path"].iloc[index]

        data_path = self.data_path / path_str
        assert data_path.exists(), f"Image path {data_path} does not exist"
        # load in image
        img = Image.open(data_path).convert('RGB').resize(self.img_size)
        img = np.array(img) / 255.0 # scale to [0, 1]
        # imagenet normalisation
        if self.imagenet_normalize:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            img = (img-mean)/std
        assert img.shape == (
            self.img_size[0],
            self.img_size[1],
            3,
        ), f"Image shape {img.shape} is not ({self.img_size[0]}, {self.img_size[1]}, 3)"
        return img.astype(np.float16)

    def __gettarget__(self, index: int) -> np.ndarray:
        """
        Get the labels for a CXR image.

        This method retrieves the labels for the specified index and applies
        the configured label strategy for handling uncertain annotations.

        Args:
            index (int): Index of the image

        Returns:
            np.ndarray: Array of labels with shape (num_classes,)
        """
        target = self.dataframe.iloc[index][CXR_LABELS]
        target = np.array(target, dtype=np.float16)
        return target

    def __get_all_targets__(self) -> np.ndarray:
        """
        Get all labels in the dataset at once.

        This method retrieves all labels from the dataframe and applies
        the configured label strategy for handling uncertain annotations.

        Returns:
            np.ndarray: Array of all labels with shape (num_samples, num_classes)
        """
        targets = self.dataframe[CXR_LABELS]
        targets = np.array(targets, dtype=np.float16)
        return targets

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        if self._worker_mode and self._path_array is not None:
            return len(self._path_array)
        return len(self.dataframe)

    def _get_init_args(self):
        """Get initialization arguments for multiprocessing workers."""
        # Extract only the path column as a list for lightweight worker initialization
        path_array = self.dataframe["path"].tolist()

        return {
            "df": None,  # Don't pass the full dataframe to workers
            "img_path": self.data_path,
            "name": self.name,
            "img_size": self.img_size,
            "split": self.split,
            "save_root": self.save_root.parent,
            "fits_memory": self.fits_memory,
            "n_threads": 1,  # Each worker only needs 1 thread
            "chunk_size": self.chunk_size,
            "silent": True,
            "_worker_mode": True,
            "_path_array": path_array,  # Pass lightweight path array instead
        }


class MIMICIVEDDataset:
    """MIMIC-IV-ED emergency-department triage records, as a tabular dataset.

    Deliberately not a `BaseDataset`: the inputs are already numeric columns of the
    dataframe, so there is nothing to decode and no reason to cache anything to disk.
    It exposes just enough of the interface (`inputs`, `targets`, `fits_memory`,
    `__len__`, `__getitem__`, `__get_all_*__`) for the training and analysis code, and
    `get_dataset` fills `inputs`/`targets` eagerly rather than going through
    `cache_numpy`.

    The feature set is fixed in `self.input_variables`: demographics, prior ED/hospital/
    ICU utilisation counts, triage vitals, chief-complaint flags, and the Charlson and
    Elixhauser comorbidity indicators. Targets are the two outcomes in
    MIMIC_IV_ED_LABELS (hospitalisation, critical outcome).

    Args:
        dataframe (pd.DataFrame): One split's records, with the feature and label columns.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.dataframe = dataframe
        self.dataframe["sex_binarized"] = 0
        self.dataframe.loc[self.dataframe.sex == "F", "sex_binarized"] = 1
        self.fits_memory = True  # for API compatibility
        self.input_variables = [
            "age",
            "sex_binarized",
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
            
            "triage_temperature", "triage_heartrate", "triage_resprate", 
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", 
            "chiefcom_dizziness", 
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
            "cci_Cancer2", "cci_HIV", 
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression"
        ]

    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.dataframe)

    def __get_all_inputs__(self):
        """Returns all inputs in the dataset"""
        inputs = self.dataframe[self.input_variables].copy()
        inputs = np.array(inputs, dtype=np.float16)
        return inputs

    def __get_all_targets__(self):
        """Returns all targets in the dataset"""
        targets = self.dataframe[
           MIMIC_IV_ED_LABELS
        ].copy()  # Task 1, 2 & 3 from the paper
        targets = np.array(targets, dtype=np.float16)
        return targets

    def __getitem__(self, index: int) -> tuple:
        raise NotImplementedError(
            "MIMIC-IV dataset does not support __getitem__ method. Use __get_all_inputs__ and __get_all_targets__ instead."
        )


class MIMICECGDataset(BaseDataset):
    """MIMIC-IV-ECG: 10 s 12-lead ECGs read from WFDB records.

    Each record is resampled to `fs` Hz (so the cached array is (fs*10, 12)), optionally
    filtered by `preprocess_ecg` and z-scored per lead. Labels are the 15 statement
    groups in MIMIC_ECG_LABELS.

    Args:
        df (pd.DataFrame): One split's records; must carry the WFDB path, 'subject_id'
            and 'ecg_time' columns plus the label columns.
        data_path (Path): Root the record paths are resolved against.
        name (str): Dataset name; '_p' is appended when one_record_per_patient is set.
        split (str): 'train', 'train_future', 'val' or 'test'. Part of the cache filename.
        save_root (Path): Root of the .npy/.mmp cache.
        num_classes (int): Number of label columns. Defaults to len(MIMIC_ECG_LABELS).
        one_record_per_patient (bool): Keep only each patient's *last* ECG, which makes
            one training example one patient (needed for patient-level DP).
        fits_memory (bool): Whether the split is cached as an in-memory .npy rather than
            a streamed .mmp.
        fs (int): Target sampling rate in Hz.
        n_threads (int): Workers used when building the cache.
        chunk_size (int): Records per multiprocessing chunk when building the cache.
        silent (bool): Suppress construction logging.
        clean_signals (bool): Apply the baseline/notch/bandpass filtering of
            `preprocess_ecg` before normalisation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_path: Path,
        name: str,
        split: str,
        save_root: Path,
        num_classes:int=len(MIMIC_ECG_LABELS),
        one_record_per_patient: bool = False,
        fits_memory: bool = False,
        fs: int = 500,
        n_threads: int = 64,
        chunk_size: int = 100,
        silent: bool = False,
        clean_signals: bool = True,
        _worker_mode: bool = False,
        _path_array: Optional[List[str]] = None,  # Lightweight path array for workers
    ):
        self.clean_signals = clean_signals
        self.fs = fs
        if one_record_per_patient:
            df = df.sort_values("ecg_time").drop_duplicates(subset="subject_id", keep="last").reset_index(drop=True)
            name = f"{name}_p"
        super().__init__(
            df=df,
            img_path=data_path,
            name=name,
            num_classes=num_classes,
            in_channels=12,
            img_size=(fs, 0),
            split=split,
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=fits_memory,
            silent=silent,
            _worker_mode=_worker_mode,
        )

        # Worker mode: use lightweight path array
        if _worker_mode and _path_array is not None:
            self._path_array = _path_array
        else:
            self._path_array = None

    def normalise(self, signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Performs z-score normalisation on the ECG signal such that it has zero mean and unit variance.

        Args:
            signal (np.ndarray): The ECG signal to be normalised.
            eps (float): A small value to prevent division by zero.

        Returns:
            np.ndarray: The normalised ECG signal.
        """
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        return (signal - mean) / (std + eps)

    def __getinput__(self, index: int):
        # Use lightweight path array in worker mode, dataframe otherwise
        if self._worker_mode and self._path_array is not None:
            path_str = self._path_array[index]
        else:
            path_str = self.dataframe.iloc[index]["path"]

        path = self.data_path / path_str
        assert Path(f"{path}.dat").is_file(), f"ECG file {path} does not exist"
        record = wfdb.rdrecord(str(path))
        record_arr = np.array(record.p_signal, dtype=np.float32)
        # replace nan leads or nan values with zeros
        record_arr = np.nan_to_num(record_arr)
        # resample to desired resolution if needed
        if record.fs != self.fs:
            record_arr = scipy.signal.resample(record_arr, num=2_500, axis=0)
        if self.clean_signals:
            record_arr = preprocess_ecg(signal=record_arr, fs=record.fs)
        record_arr = self.normalise(record_arr)
        record_arr = record_arr.astype(np.float16)
        return record_arr

    def __gettarget__(self, index: int):
        target = self.dataframe.iloc[index][MIMIC_ECG_LABELS]
        assert (
            len(target) == self.num_classes
        ), f"Target length {len(target)} does not match num_classes {self.num_classes}"
        return np.array(target, dtype=np.float16)

    def __get_all_targets__(self):
        targets = np.array(self.dataframe[MIMIC_ECG_LABELS], dtype=np.float16)
        assert (
            targets.shape[1] == self.num_classes
        ), f"Targets shape {targets.shape} does not match num_classes {self.num_classes}"
        return targets

    def __len__(self) -> int:
        if self._worker_mode and self._path_array is not None:
            return len(self._path_array)
        return len(self.dataframe)

    def _get_init_args(self):
        """Get initialization arguments for multiprocessing workers."""
        # Extract only the path column as a list for lightweight worker initialization
        path_array = self.dataframe["path"].tolist()

        return {
            "df": None,  # Don't pass the full dataframe to workers
            "data_path": self.data_path,
            "name": self.name,
            "split": self.split,
            "save_root": self.save_root.parent,
            "num_classes": self.num_classes,
            "fits_memory": self.fits_memory,
            "fs": self.fs,
            "n_threads": 1,  # Each worker only needs 1 thread
            "chunk_size": self.chunk_size,
            "silent": True,
            "clean_signals": self.clean_signals,
            "_worker_mode": True,
            "_path_array": path_array,  # Pass lightweight path array instead
        }


class HEEDBDataset(BaseDataset):
    """Harvard-Emory ECG Database: 10 s 12-lead ECGs at a fixed 250 Hz.

    Same shape and preprocessing as `MIMICECGDataset`, but labels are the 36 ICD groups
    of HEEDB_CODE_DICT, held in 'icd_<code>' columns. With 6.1M training records this is
    the largest dataset here and never fits in memory, so `fits_memory` defaults to False
    and every split is streamed from its memmap.

    Args:
        df (pd.DataFrame): One split's records; must carry the WFDB filename,
            'BDSPPatientID' and 'ECGAcquisitionTime' columns plus the 'icd_*' labels.
        data_path (Path): Root the record filenames are resolved against.
        name (str): Dataset name; '_p' is appended when one_record_per_patient is set.
        split (str): 'train', 'train_future', 'val' or 'test'. Part of the cache filename.
        save_root (Path): Root of the .npy/.mmp cache.
        one_record_per_patient (bool): Keep only each patient's *last* ECG, which makes
            one training example one patient (needed for patient-level DP).
        fits_memory (bool): Whether the split is cached as an in-memory .npy rather than
            a streamed .mmp.
        n_threads (int): Workers used when building the cache.
        chunk_size (int): Records per multiprocessing chunk when building the cache.
        silent (bool): Suppress construction logging.
        clean_signals (bool): Apply the baseline/notch/bandpass filtering of
            `preprocess_ecg` before normalisation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_path: Path,
        name: str,
        split: str,
        save_root: Path,
        one_record_per_patient: bool = False,
        fits_memory: bool = False,
        n_threads: int = 64,
        chunk_size: int = 100,
        silent: bool = False,
        clean_signals: bool = True,
        _worker_mode: bool = False,
        _filename_array: Optional[List[str]] = None,  # Lightweight filename array for workers
    ):
        num_classes = len(HEEDB_CODE_DICT)
        self.clean_signals = clean_signals
        self.label_columns = [f"icd_{code}" for code in HEEDB_CODE_DICT.keys()]
        if one_record_per_patient:
            df = df.sort_values("ECGAcquisitionTime").drop_duplicates(subset="BDSPPatientID", keep="last").reset_index(drop=True)
            name=f"{name}_p"
        # if "age" not in df.columns:
        #     df.ECGAcquisitionTime = pd.to_datetime(df.ECGAcquisitionTime, errors='coerce')
        #     df.DateOfBirth = pd.to_datetime(df.DateOfBirth, errors='coerce')
        #     # replace 'redacted' with NaT
        #     df.DateOfBirth = df.DateOfBirth.replace('redacted', pd.NaT)
        #     df['age'] = df.ECGAcquisitionTime.dt.year - df.DateOfBirth.dt.year
        super().__init__(
            df=df,
            img_path=data_path,
            name=name,
            num_classes=num_classes,
            in_channels=12,
            img_size=(250, 0), # 250hz
            split=split,
            save_root=save_root,
            n_threads=n_threads,
            chunk_size=chunk_size,
            fits_memory=fits_memory,
            silent=silent,
            _worker_mode=_worker_mode,
        )

        # Worker mode: use lightweight filename array
        if _worker_mode and _filename_array is not None:
            self._filename_array = _filename_array
        else:
            self._filename_array = None

    def normalise(self, signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Performs z-score normalisation on the ECG signal such that it has zero mean and unit variance.

        Args:
            signal (np.ndarray): The ECG signal to be normalised.
            eps (float): A small value to prevent division by zero.

        Returns:
            np.ndarray: The normalised ECG signal.
        """
        mean = signal.mean(axis=0, keepdims=True)
        std = signal.std(axis=0, keepdims=True)
        return (signal - mean) / (std + eps)

    def __getinput__(self, index: int):
        # Use lightweight filename array in worker mode, dataframe otherwise
        if self._worker_mode and self._filename_array is not None:
            filename = self._filename_array[index]
        else:
            filename = self.dataframe.iloc[index]["FileName"]

        path = self.data_path / filename[1:]  # Remove leading "/" from filename
        assert Path(f"{path}.mat").is_file(), f"ECG file {path} does not exist"
        record = wfdb.rdrecord(str(path))
        record_arr = np.array(record.p_signal, dtype=np.float32)
        # replace nan leads or nan values with zeros
        record_arr = np.nan_to_num(record_arr)
        # resample to 250hz if needed
        if record.fs != 250:
            record_arr = scipy.signal.resample(record_arr, num=2_500, axis=0)
        if self.clean_signals:
            record_arr = preprocess_ecg(signal=record_arr, fs=250)
        # normalisation
        record_arr = self.normalise(record_arr)
        record_arr = record_arr.astype(np.float16)
        assert record_arr.shape == (
            2_500,
            12,
        ), f"ECG shape {record_arr.shape} is not (2500, 12)"
        return record_arr

    def __gettarget__(self, index: int):
        target = self.dataframe.iloc[index][self.label_columns]
        return np.array(target, dtype=np.float16)

    def __get_all_targets__(self):
        return np.array(self.dataframe[self.label_columns]).astype(np.float16)

    def __len__(self) -> int:
        if self._worker_mode and self._filename_array is not None:
            return len(self._filename_array)
        return len(self.dataframe)

    def _get_init_args(self):
        """Get initialization arguments for multiprocessing workers."""
        # Extract only the FileName column as a list for lightweight worker initialization
        filename_array = self.dataframe["FileName"].tolist()

        return {
            "df": None,
            "data_path": self.data_path,
            "name": self.name,
            "split": self.split,
            "save_root": self.save_root.parent,
            "fits_memory": self.fits_memory,
            "n_threads": 1,  # Each worker only needs 1 thread
            "chunk_size": self.chunk_size,
            "silent": True,
            "clean_signals": self.clean_signals,
            "_worker_mode": True,
            "_filename_array": filename_array,  # Pass lightweight filename array instead of dataframe
        }
