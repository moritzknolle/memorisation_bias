# memorisation_bias

Accompanying code repository of our paper "Memorisation bias in medical AI".
This repository allows us to quantify whether, and to what extent, the inclusion of a patient's
historical records in a model training dataset causes a significant change in the predictions
on their unseeen future data.

For each dataset, 200 models are trained on random 50% patient subsets. For every record,
the predictions of models trained on the patient's historical records (IN) are compared to those
of models that were not (OUT) using a multivariate two-sample test with Benjamini–Hochberg
correction. See the paper for details.

## Repository structure

```
├── mimic-cxr.py, mimic-ecg.py, heedb.py   training (keras)
├── mimic-iv.py, mimic-iv_sklearn.py       training on MIMIC-IV-ED (tabular ResNet, sklearn models)
├── mimic-ecg_dp.py, heedb_dp.py           training with DP-SGD
├── memorisation_analysis.py               memorisation analysis of one set of runs
├── dp_memorisation_analysis.py            memorisation analysis across privacy budgets
├── rop_comparison_plot.py                 sensitivity/specificity comparison figure
├── plot.py                                cross-dataset figures
├── requirements.txt                       pinned dependencies
├── scripts/
│   ├── train_all.sh                       commands to train all models
│   ├── build_cache.py                     write the input cache of a dataset
│   ├── run_until_error.sh                 train until all runs of a logdir are complete
│   └── run_analysis.sh                    run all analyses and create all figures
└── src/
    ├── data/                              datasets, historical/future split, preprocessing notebooks
    ├── training/                          training loop, DP-SGD, models
    ├── memorisation/                      IN/OUT partitioning and statistical testing
    ├── plotting.py                        plots, sensitivity/specificity comparison
    └── utils.py                           per-dataset settings
```

## Installation

Requires Python 3.12 and git. Keras runs on the JAX backend; TensorFlow is only used for the input
pipeline. GPU training requires an NVIDIA driver with CUDA 13 support.

```bash
conda create -n memorisation_bias python=3.12
conda activate memorisation_bias
pip install -r requirements.txt
pip install --no-deps "jax-privacy @ git+https://github.com/google-deepmind/jax_privacy.git@6cf09722521728cc5b2e6fb2db9846ee70bdc546"
```

`requirements.txt` pins all versions, including the commits of jax-privacy, optax and dp_accounting.

Training logs to Weights & Biases. Run directories are named after the wandb run ID, so
predictions are only saved with `--log_wandb=True` (the default). Run `wandb login` or set
`WANDB_MODE=offline`.

All commands are run from the repository root.

## Data

The datasets are publicly available for research purposes but require credentialed access. See links below.

| Dataset | Source | Location |
|---|---|---|
| MIMIC-CXR | [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) | `data/raw/mimic-cxr/mimic-cxr-jpg` |
| MIMIC-ECG | [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/) | `data/raw/mimic-ecg` |
| MIMIC-IV-ED | [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/), prepared with [mimic4ed-benchmark](https://github.com/nliulab/mimic4ed-benchmark) | `data/raw/mimic-iv-ed-benchmark` |
| HEEDB | [Harvard-Emory ECG Database](https://bdsp.io/content/heedb/) | `data/raw/heedb` (contains `WFDB/`) |

The MIMIC-CXR and MIMIC-ECG notebooks also read `patients.csv` from MIMIC-IV
(`data/raw/mimiciv/3.1/patients.csv`).

1. **Preprocessing.** Run the notebooks in `src/data/notebooks/` from that directory. Each writes
   the splits `data/csv/<dataset>_{train_historical,train_future,val,test}` (`.csv`, `.pkl` for HEEDB).
2. **Caching.** Preprocessed inputs are cached in `data/npy/` by
   `python scripts/build_cache.py --dataset=<dataset>` for `mimic-cxr`, `mimic-ecg`, `mimic-ecg_p`,
   `heedb` and `heedb_p` (the first lines of `scripts/train_all.sh`). MIMIC-IV-ED is not cached. Data
   paths can be changed with `--data_root`, `--csv_root` and `--save_root`.

`train_future` holds the future records (called `long_eval` in file names).

## Reproducing the results

With `--train_random_subset=True`, a training script trains one model on the next unclaimed patient
subset of `--logdir` and saves its predictions. The subsets of all `--n_runs` runs are drawn once, by
the first run on a logdir, and stored in `<logdir>/super_mask.npy`, a boolean matrix of shape
`(n_runs, n_patients)`. Each patient is included in exactly half of the runs (`--subset_ratio=0.5`),
so every record has the same number of IN and OUT models. `scripts/run_until_error.sh` repeats this until all
runs of the logdir are complete. The same command can be started on any number of GPUs or machines at
the same time, as long as they share the repository directory. The default flag values of the training
scripts are the settings used in the paper.

### Quick start: random forest on MIMIC-IV-ED

The random forest on MIMIC-IV-ED needs no GPU and no input cache, which makes it a good starting
point. On a 16-core workstation (Intel Xeon W-2245), each model trained in about one minute and the
subsequent memorisation detection analysis took about 25 minutes.

1. Prepare MIMIC-IV-ED with [mimic4ed-benchmark](https://github.com/nliulab/mimic4ed-benchmark) and
   place its `train.csv` and `test.csv` in `data/raw/mimic-iv-ed-benchmark/`.
2. Create the splits by running `src/data/notebooks/mimic-iv-ed.ipynb` from its directory.
3. Train the 200 models (start the command in several terminals to train in parallel):
   ```bash
   bash scripts/run_until_error.sh python mimic-iv_sklearn.py --model=rf --train_random_subset=True --n_runs=200 --logdir=logs/mimic-iv-ed-rf
   ```
4. Run the memorisation analysis:
   ```bash
   python memorisation_analysis.py --dataset=mimic-iv-ed --logdir=logs/mimic-iv-ed-rf --name=mimic-iv-ed_rf
   ```
   Results are written to `figs/mimic-iv-ed_rf/`. `files/mimic-iv-ed_rf_mem_data.csv` contains the
   test result (p-value, effect size, rejection) for every future record.

### All experiments

1. Preprocess all datasets (see [Data](#data)).
2. Train all models with the commands in `scripts/train_all.sh`. Its first lines write the input
   caches. `bash scripts/train_all.sh` runs everything one after another; to parallelise, start
   individual lines on several GPUs or machines, e.g.
   `CUDA_VISIBLE_DEVICES=1 bash scripts/run_until_error.sh python mimic-ecg.py --train_random_subset=True --n_runs=200 --logdir=logs/mimic-ecg/vit_s`.
3. Run all analyses and create all figures:
   ```bash
   LOG_ROOT=logs bash scripts/run_analysis.sh
   ```

### Log directories

Training writes to `--logdir` a set of bookkeeping files shared by all runs and one directory per
completed run, named after its wandb run ID:

```
<logdir>/
├── super_mask.npy               patient subset masks of all runs, (n_runs, n_patients)
├── valid_idcs.pkl               indices of runs not yet claimed
├── completed_idcs.pkl           indices of completed runs
├── bookkeeping.lock             lock for training in parallel
└── <wandb_run_id>/
    ├── train_logits.npy         model outputs on the historical split
    ├── long_eval_logits.npy     ... on the future split
    ├── test_logits.npy          ... on the test split
    ├── patient_subset_mask.npy  True for the patients this run was trained on
    ├── patient_ids.pkl          patient IDs of the historical split, in the order of the mask
    └── info.json                run metadata
```

Model outputs have shape `(n_records, n_classes)`, with records in the order of the split file in `data/csv/`.
They are logits, except for sklearn models, which save predicted probabilities. With test-time
augmentation (`mimic-cxr.py --eval_views`) there is an additional axis,
`(n_records, n_views, n_classes)`.

`info.json` contains:

| Key | Content |
|---|---|
| `wandb_run_id`, `start_time`, `end_time`, `mac_address` | run identification |
| `wandb_config` | all flags of the run, plus `run_seed`; sklearn runs also record `model_class` |
| `train_metrics` | metrics on the run's training subset |
| `test_metrics` | metrics on the test split, e.g. `_test_macro_auroc` |

The memorisation analysis reads the model outputs, `patient_subset_mask.npy` and `patient_ids.pkl`.
It checks `wandb_config` for `model_class` to decide whether an activation function must be
applied. The figure scripts read test AUROC from `test_metrics`.

The log directories of all experiments are organised as follows:

```
logs/
├── mimic-cxr/
├── mimic-iv-ed-{rf,lr,tabresnet}/
└── {mimic-ecg,heedb}/
    ├── vit_s/                        non-private
    ├── dp/eps{1,10,100,1000}[_p]/    DP-SGD, record-level [patient-level]
    └── nonprivate[_p]/               ε = ∞
```

### Analysis outputs

`scripts/run_analysis.sh` writes results and figures to `figs/`:

| Script | Output |
|---|---|
| `memorisation_analysis.py --dataset=<d> --logdir=<logdir>` | `figs/<d>/`: per-record test results (`files/*_mem_data*.csv`), sensitivity/specificity comparison, per-dataset plots |
| `rop_comparison_plot.py [--denovo] [--heedb_only]` | `figs/rop_comparison*.pdf` |
| `dp_memorisation_analysis.py --dataset=<d> --log_root=logs/<d>` | `figs/<d>_dp/`: memorisation and test AUROC across privacy budgets |
| `plot.py` | cross-dataset figures in `figs/` |

The last three scripts use the results of `memorisation_analysis.py`.
