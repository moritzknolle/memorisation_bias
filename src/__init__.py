"""Library code for the memorisation analysis. See README.md for usage.

    data/          datasets, historical/future split policy and the disk cache
    training/      training loop, leave-many-out bookkeeping, models
    memorisation/  IN/OUT partitioning and the per-record statistical test
    metrics.py     NaN-safe multi-label AUROC helpers
    plotting.py    plots and the receiver-operating-point comparison
    utils.py       per-dataset settings: paths, column names, label lists, colours
"""
