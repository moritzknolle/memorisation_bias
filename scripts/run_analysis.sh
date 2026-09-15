#!/bin/bash
# Run all analyses and create all figures from the runs under LOG_ROOT (layout in README.md).
#
# Usage:
#   LOG_ROOT=logs bash scripts/run_analysis.sh
set -euo pipefail

LOG_ROOT="$(realpath "${LOG_ROOT:-logs}")"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# memorisation analysis, writes figs/<name>/
python memorisation_analysis.py --dataset=mimic-cxr --logdir="${LOG_ROOT}/mimic-cxr"
python memorisation_analysis.py --dataset=mimic-ecg --logdir="${LOG_ROOT}/mimic-ecg/vit_s"
python memorisation_analysis.py --dataset=heedb --logdir="${LOG_ROOT}/heedb/vit_s" --multiprocessing=False
for model in tabresnet rf lr; do
    python memorisation_analysis.py --dataset=mimic-iv-ed \
        --logdir="${LOG_ROOT}/mimic-iv-ed-${model}" --name="mimic-iv-ed_${model}"
done

# sensitivity/specificity comparison, writes figs/rop_comparison*.pdf
python rop_comparison_plot.py
python rop_comparison_plot.py --denovo
python rop_comparison_plot.py --heedb_only
python rop_comparison_plot.py --heedb_only --denovo

# differential privacy, writes figs/<dataset>_dp/
for dataset in mimic-ecg heedb; do
    python dp_memorisation_analysis.py --dataset="${dataset}" --log_root="${LOG_ROOT}/${dataset}"
done

# cross-dataset figures, writes figs/*.pdf
LOG_ROOT="${LOG_ROOT}" MPLBACKEND=Agg python plot.py
