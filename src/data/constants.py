"""Label vocabularies, one per dataset.

The order of each list *is* the column order of that dataset's target arrays and of every
per-class array downstream (thresholds, AUROCs, the ROP comparison), so these lists must
not be reordered once a cache or a set of logit dumps exists.

    CXR_LABELS / CXR_SHORT_LABELS          14 findings, MIMIC-CXR and CheXpert
    CXP_CHALLENGE_LABELS[_IDX]             the 5 CheXpert competition findings
    MIMIC_ECG_LABELS / ..._SHORT           15 ECG statement groups
    MIMIC_IV_ED_LABELS[_Short]             2 ED outcomes
    HEEDB_CODE_DICT                        36 ICD codes -> disease names; the HEEDB label
                                           columns are named 'icd_<code>'
    KERMANY_OCT_LABELS                     retinal OCT, from an earlier experiment

The `*_SHORT` variants are display names for figures; `src/utils.py` maps a dataset name
to the right pair.
"""

from enum import Enum

# CheXpert and MIMIC-CXR labels
CXR_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CXR_SHORT_LABELS = [
    "NF",
    "EC",
    "Cm",
    "LO",
    "LL",
    "Ed",
    "Co",
    "Pn",
    "At",
    "Px",
    "PE",
    "PO",
    "Fr",
    "SD",
]
CXP_CHALLENGE_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
]

CXP_CHALLENGE_LABELS_IDX = [CXR_LABELS.index(l) for l in CXP_CHALLENGE_LABELS]
KERMANY_OCT_LABELS = ["Normal", "Drusen", "CNV", "DME"]
MIMIC_ECG_LABELS = [
    "sinus",
    "tachycardia",
    "bradycardia",
    "atrial fibrillation",
    "atrial flutter",
    "av block",
    "rbbb",
    "lbbb",
    "pacemaker",
    "infarct",
    "pvc",
    "abnormal ecg",
    "borderline ecg",
    "lateral st-t changes",
    "fascicular block",
]
MIMIC_ECG_LABELS_SHORT = [
    "Si",
    "Ta",
    "Br",
    "AF",
    "AFl",
    "AVB",
    "RBBB",
    "LBBB",
    "PM",
    "In",
    "PVC",
    "Abn",
    "Bor",
    "LST",
    "FB",
]
MIMIC_IV_ED_LABELS = ["outcome_hospitalization", "outcome_critical"]
MIMIC_IV_ED_LABELS_Short = ["Hosp", "Crit"]

HEEDB_CODE_DICT = {
    1684: 'Normal ECG',
    1699: 'Abnormal ECG',
    22: 'Normal sinus rhythm',
    21: 'Sinus bradycardia',
    161: 'Atrial fibrillation',
    23: 'Sinus tachycardia',
    372: 'Left axis deviation',
    231: 'Premature ventricular complexes',
    1693: 'Borderline ECG',
    440: 'Right bundle branch block',
    700: 'Septal infarct',
    1140: 'Non-specific t wave abnormality',
    222: 'Premature atrial complexes',
    740: 'Anterior infarct',
    460: 'Left bundle branch block',
    760: 'Lateral infarct',
    900: 'Non-specific ST abnormality',
    541: 'Left ventricular hypertrophy',
    162: 'Atrial flutter',
    470: 'Left anterior fascicular block',
    383: 'Right axis deviation',
    810: 'Anteroseptal infarct',
    820: 'Anterolateral infarct',
    350: 'Right atrial enlargement',
    780: 'Inferior infarct',
    480: 'Bi-fascicular block',
    471: 'Left posterior fascicular block',
    369: 'Bi-atrial enlargement',
    101: '1st degree AV block',
    801: 'Inferior-posterior infarct',
    266: 'Supraventricular tachycardia',
    235: 'Wide QRS tachycardia',
    304: 'Wolff-Parkinson-White',
    901: 'Acute pericarditis',
    802: 'Posterior infarct',
    570: 'Biventricular hypertrophy',
}

