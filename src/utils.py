"""Per-dataset settings: raw data paths, patient id and timestamp columns, label lists,
activation functions, and names and colours for figures. Unknown dataset names raise a
ValueError.

The colour palette at the bottom of the file is used by the DP analysis scripts.
"""

from pathlib import Path
from typing import Callable, Union
import numpy as np
import pandas as pd
from .data.constants import (
    CXR_LABELS,
    KERMANY_OCT_LABELS,
    MIMIC_ECG_LABELS,
    MIMIC_IV_ED_LABELS,
    HEEDB_CODE_DICT,
    CXR_SHORT_LABELS,
    MIMIC_ECG_LABELS_SHORT,
    MIMIC_IV_ED_LABELS,
    MIMIC_IV_ED_LABELS_Short,
)

import keras  # type: ignore


def fig_dir_exists(out_dir: Path):
    """Create an analysis output directory and its `files/` subdirectory.

    Figures go in `out_dir`, the per-record result CSVs in `out_dir/files/`.
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    files_dir = out_dir / "files"
    if not files_dir.exists():
        files_dir.mkdir(parents=True, exist_ok=True)


def get_data_root(dataset_name: str):
    """Default path to a dataset's raw images/waveforms (see README.md)."""
    if dataset_name == "mimic-cxr":
        data_root = Path("./data/raw/mimic-cxr/mimic-cxr-jpg")
    elif dataset_name == "mimic-iv-ed":
        data_root = Path("")
    elif dataset_name == "mimic-ecg":
        data_root = Path("./data/raw/mimic-ecg")
    elif dataset_name == "heedb":
        data_root = Path("./data/raw/heedb/WFDB")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data_root


def get_study_order_col(dataset_name: str):
    """Column to sort a patient's records into chronological order by.

    Usually the study timestamp, but for datasets without one a monotonically increasing
    study/image id stands in -- which is why this is separate from `get_study_date_col`.
    """
    if dataset_name == "chexpert":
        study_order_col = "study_id"
    elif dataset_name == "mimic-cxr":
        study_order_col = "timestamp"
    elif dataset_name == "kermany_oct":
        study_order_col = "img_id"
    elif dataset_name == "mimic-iv-ed":
        study_order_col = "outtime"
    elif dataset_name == "mimic-ecg":
        study_order_col = "ecg_time"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return study_order_col


def get_study_date_col(dataset_name: str):
    """Column holding an actual acquisition timestamp, or None if the dataset has none.

    Needed for the follow-up interval (`get_time_deltas`): the months between a future
    record and that patient's most recent training record. Datasets returning None cannot
    have that analysis run on them.
    """
    if dataset_name == "mimic-cxr":
        return "timestamp"
    elif dataset_name == "mimic-iv-ed":
        return "outtime"
    elif dataset_name == "mimic-ecg":
        return "ecg_time"
    elif dataset_name == "heedb":
        return "ECGAcquisitionTime"
    else:
        return None

def is_dataset_multilabel(dataset_name: str):
    """True for multi-label datasets (a record can carry several diagnoses at once)."""
    if dataset_name in ["mimic-cxr", "mimic-iv-ed", "mimic-ecg", "heedb"]:
        return True
    elif dataset_name in ["kermany_oct", "pinnacle"]:
        return False
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_act_fn(
    dataset_name: str, is_sklearn_model: bool = False
) -> Union[Callable, None]:
    """Activation turning a run's stored outputs into probabilities.

    Sigmoid for the multi-label datasets, softmax for the single-label ones. Returns None
    for sklearn models, which store `predict_proba` output rather than logits and must
    not be transformed a second time -- `detect_sklearn_model` in the analysis scripts
    infers that flag from the run config.
    """
    if is_sklearn_model:
        return None
    elif dataset_name in ["mimic-cxr", "mimic-iv-ed", "mimic-ecg", "heedb"]:
        return keras.activations.sigmoid
    elif dataset_name in ["pinnacle", "kermany_oct"]:
        return keras.activations.softmax
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")



def get_patient_col(dataset_name: str):
    """Column holding the patient id -- the unit the training subsets are drawn over."""
    if dataset_name in ["mimic-cxr", "mimic-iv-ed", "mimic-ecg"]:
        patient_col = "subject_id"
    elif dataset_name == "kermany_oct":
        patient_col = "patient_id"
    elif dataset_name == "pinnacle":
        patient_col = "PatientMatchID"
    elif dataset_name == "heedb":
        patient_col = "BDSPPatientID"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return patient_col



def get_label_list(dataset_name:str):
    """Label column names, in the column order of the dataset's target arrays."""
    if dataset_name in ["mimic-cxr", "chexpert"]:
        return CXR_LABELS
    elif dataset_name == "mimic-iv-ed":
        return MIMIC_IV_ED_LABELS
    elif dataset_name == "kermany_oct":
        return KERMANY_OCT_LABELS
    elif dataset_name == "mimic-ecg":
        return MIMIC_ECG_LABELS
    elif dataset_name == "heedb":
        return [f"icd_{c}" for c in HEEDB_CODE_DICT.keys()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
def get_label_list_short(dataset_name:str):
    """Display names for the labels, same order as `get_label_list`. For figure axes."""
    if dataset_name in ["mimic-cxr", "chexpert"]:
        return CXR_SHORT_LABELS
    elif dataset_name == "mimic-iv-ed":
        return MIMIC_IV_ED_LABELS_Short    
    elif dataset_name == "mimic-ecg":
        return MIMIC_ECG_LABELS_SHORT
    elif dataset_name == "heedb":
        return HEEDB_CODE_DICT.keys()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_dataset_str(dataset: str):
    """Display name of a dataset, for figure labels and legends."""
    if dataset == "mimic-cxr":
        return "MIMIC-CXR"
    elif dataset == "mimic-ecg":
        return "MIMIC-ECG"
    elif dataset == "mimic-iv-ed":
        return "MIMIC-IV-ED"
    elif dataset == "heedb":
        return "HEEDB"
    raise ValueError(f"Invalid dataset name., {dataset}")


def get_dataset_color_hex(dataset: str):
    """The colour a dataset is drawn in. One colour per dataset across every figure."""
    if dataset == "mimic-cxr":
        return "#786AAF"
    elif dataset == "mimic-ecg":
        return "#EA9A86"
    elif dataset == "mimic-iv-ed" or dataset == "mimic-iv-ed_rf":
        return "#C9B2E8"
    elif dataset == "heedb":
        return "#A4303F"
    else:
        raise ValueError(f"Invalid dataset name., {dataset}")


# --- DP figure palette -------------------------------------------------------
# One colour per DP level, used wherever a figure shows both levels side by side.
DP_COLOR_RECORD = "#18206F"
DP_COLOR_PATIENT = "#BD1E1E"

# Colour ramps for the eSF curves of the privacy budgets, one per DP level, from light
# (small eps) to dark (eps=inf), evenly spaced in OKLab lightness.
DP_ESF_RAMP = {
    "record": ["#B1B7E6", "#828BC3", "#5762A1", "#2E397F", "#18206F"],
    "patient": ["#EEA18D", "#CC705C", "#BD1E1E", "#8C0000", "#5D0000"],
}
# opacity of the lightest and the darkest curve
DP_ESF_ALPHA = (0.90, 1.00)

# privacy budgets in ramp order, so that a budget has the same colour in every figure
DP_ESF_EPSILONS = (1, 10, 100, 1000, np.inf)


def _lerp_hex(c0: str, c1: str, t: float) -> str:
    """Colour t of the way from c0 to c1."""
    a = np.array([int(c0[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(c1[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    return "#" + "".join(f"{int(round(v)):02X}" for v in a + (b - a) * t)


def dp_esf_styles(dp_level: str, epsilons) -> dict:
    """(colour, alpha) per epsilon for one eSF panel, keyed by epsilon.

    `epsilons` is the ascending list of budgets drawn in the panel, with the non-private
    baseline as np.inf. Budgets in DP_ESF_EPSILONS take their own step of the ramp, so a
    curve has the same colour in every figure; other budgets are spaced evenly along the ramp.
    """
    if dp_level not in DP_ESF_RAMP:
        raise ValueError(f"Invalid DP level, {dp_level}")
    ramp = DP_ESF_RAMP[dp_level]
    epsilons = list(epsilons)
    n = len(epsilons)
    if n == 0:
        return {}
    alphas = np.linspace(DP_ESF_ALPHA[0], DP_ESF_ALPHA[1], len(ramp))

    if all(eps in DP_ESF_EPSILONS for eps in epsilons):
        idx = [DP_ESF_EPSILONS.index(eps) for eps in epsilons]
        return {
            eps: (ramp[i], float(alphas[i])) for eps, i in zip(epsilons, idx)
        }

    if n == 1:
        return {epsilons[0]: (ramp[-1], DP_ESF_ALPHA[1])}
    styles = {}
    for i, eps in enumerate(epsilons):
        pos = i / (n - 1) * (len(ramp) - 1)
        lo = min(int(pos), len(ramp) - 2)
        t = pos - lo
        alpha = float(np.interp(pos, range(len(ramp)), alphas))
        styles[eps] = (_lerp_hex(ramp[lo], ramp[lo + 1], t), alpha)
    return styles