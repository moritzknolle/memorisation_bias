#!/bin/bash
# Trains all models of the paper: 200 leave-many-out runs per experiment (logdir layout in README.md).
#
# Run from the repository root with the conda environment activated. Write the input caches first.
# Each training line is independent and can be started on as many GPUs or machines as available at
# the same time, as long as they share the repository directory: the runs of an experiment are
# distributed via its logdir.
#
# Usage:
#   bash scripts/train_all.sh    # everything, one experiment after another on a single GPU

# input caches (once, before training)
python scripts/build_cache.py --dataset=mimic-cxr
python scripts/build_cache.py --dataset=mimic-ecg
python scripts/build_cache.py --dataset=mimic-ecg_p
python scripts/build_cache.py --dataset=heedb
python scripts/build_cache.py --dataset=heedb_p

# baseline models
bash scripts/run_until_error.sh python mimic-cxr.py --train_random_subset=True --n_runs=200 --logdir=logs/mimic-cxr
bash scripts/run_until_error.sh python mimic-ecg.py --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/vit_s
bash scripts/run_until_error.sh python heedb.py --train_random_subset=True --n_runs=200 --logdir=logs/heedb/vit_s
bash scripts/run_until_error.sh python mimic-iv_sklearn.py --model=rf --train_random_subset=True --n_runs=200 --logdir=logs/mimic-iv-ed-rf

# model comparison on MIMIC-IV-ED (with the random forest above)
bash scripts/run_until_error.sh python mimic-iv_sklearn.py --model=lr --train_random_subset=True --n_runs=200 --logdir=logs/mimic-iv-ed-lr
bash scripts/run_until_error.sh python mimic-iv.py --train_random_subset=True --n_runs=200 --logdir=logs/mimic-iv-ed-tabresnet

# differential privacy on MIMIC-ECG: record-level, patient-level, non-private baselines (eps=inf)
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=1 --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps1
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=10 --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps10
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=100 --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps100
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=1000 --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps1000
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=1 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps1_p
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=10 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps10_p
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=100 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps100_p
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=1000 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/dp/eps1000_p
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=inf --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/nonprivate
bash scripts/run_until_error.sh python mimic-ecg_dp.py --eps=inf --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/nonprivate_p

# differential privacy on HEEDB: record-level, patient-level, non-private baselines (eps=inf)
bash scripts/run_until_error.sh python heedb_dp.py --eps=1 --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps1
bash scripts/run_until_error.sh python heedb_dp.py --eps=10 --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps10
bash scripts/run_until_error.sh python heedb_dp.py --eps=100 --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps100
bash scripts/run_until_error.sh python heedb_dp.py --eps=1000 --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps1000
bash scripts/run_until_error.sh python heedb_dp.py --eps=1 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps1_p
bash scripts/run_until_error.sh python heedb_dp.py --eps=10 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps10_p
bash scripts/run_until_error.sh python heedb_dp.py --eps=100 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps100_p
bash scripts/run_until_error.sh python heedb_dp.py --eps=1000 --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/heedb/dp/eps1000_p
bash scripts/run_until_error.sh python heedb_dp.py --eps=inf --train_random_subset=True --n_runs=200 --logdir=logs/heedb/nonprivate
bash scripts/run_until_error.sh python heedb_dp.py --eps=inf --one_record_per_patient=True --train_random_subset=True --n_runs=200 --logdir=logs/heedb/nonprivate_p
