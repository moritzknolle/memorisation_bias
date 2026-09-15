"""Data layer: raw sources -> chronological splits -> cached arrays.

    notebooks/          preprocessing, one notebook per dataset. Writes the split files
                        data/csv/<dataset>_{train_historical,train_future,val,test}.
    data_splits.py      the historical/future split policy those notebooks apply.
    datasets.py         BaseDataset and its per-dataset subclasses, plus the .npy/.mmp cache.
    dataset_factory.py  get_dataset(name) -> the four splits. Start here.
    constants.py        label vocabularies; their order fixes the target column order.
"""
