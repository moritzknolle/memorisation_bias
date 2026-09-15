#!/bin/bash
# Repeat a command until it fails. A training script called with --train_random_subset=True trains
# one run per call and exits with an error once all runs of its logdir are complete.
#
# Usage:
#   bash scripts/run_until_error.sh python mimic-ecg.py --train_random_subset=True --logdir=<logdir>

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

counter=0
while "$@"; do
    counter=$((counter + 1))
    echo "... command succeeded, ${counter} run(s) completed"
    # sync wandb runs and delete local wandb directories
    wandb sync --clean --clean-old-hours 1 --clean-force
done
echo "... command failed after ${counter} completed run(s), exiting"
