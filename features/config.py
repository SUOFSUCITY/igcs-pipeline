"""Data paths, column mappings and feature groups."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PIPELINE_DATA_DIR", str(PROJECT_DIR / "data")))
RAW_DATA_DIR = DATA_DIR
IMAGES_TRAIN_DIR = DATA_DIR / "images" / "imagesTr"
IMAGES_TEST_DIR = DATA_DIR / "images" / "imagesTs"
MASKS_TRAIN_DIR = DATA_DIR / "masks" / "labelsTr"
MASKS_TEST_DIR = DATA_DIR / "masks" / "labelsTs"
CLINICAL_TRAIN_CSV = DATA_DIR / "clinical" / "clinical_info_train_v2.csv"
CLINICAL_TEST_CSV = DATA_DIR / "clinical" / "clinical_info_test_v2.csv"
LOCATION_DIR = DATA_DIR / "location"
LOCATION_TRAIN_CSV = LOCATION_DIR / "cv_oof_predictions.csv"
LOCATION_TEST_CSV = LOCATION_DIR / "test_predictions.csv"
FEATURES_DIR = Path(
    os.environ.get("PIPELINE_FEATURES_DIR", str(PROJECT_DIR / "outputs" / "features"))
)
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
COL_ID = "New_ID"
COL_SURGERY = "Surgery"
COL_PRIMARY_ENDPOINT = "90_day poor outcome 3_6"
COL_SECONDARY_ENDPOINT = "90_day poor outcome 4_6"
CLINICAL_COLS_MAP = {
    "Age": "age",
    "sex": "sex",
    "Admission GCS score": "GCS",
    "Admission SBP": "SBP",
    "Admission DBP": "DBP",
    "Time from onset to CT": "onset_to_ct_hours",
    "History of Hypertension ": "hypertension",
    "History of diabetes": "diabetes",
    "Smoking": "smoking",
    "Alcohol consumption": "alcohol",
    "IVH on first CT": "IVH",
    "SAH on first CT": "SAH",
}
LOCATION_COLS = ["p_basal", "p_brainstem", "p_cerebellum", "p_lobar", "p_thalamus"]
LOCATION_COLS_MAP = {
    "Prob_Basal": "p_basal",
    "Prob_Brainstem": "p_brainstem",
    "Prob_Cerebellum": "p_cerebellum",
    "Prob_Lobar": "p_lobar",
    "Prob_Thalamus": "p_thalamus",
}
LOCATION_LABEL_COL = "ICH location"
LOCATION_LABEL_MAP = {
    0: "basal",
    1: "brainstem",
    2: "cerebellum",
    3: "lobar",
    4: "thalamus",
}
IMAGING_VOLUME_COLS = ["hematoma_volume_ml", "hematoma_volume_log", "relative_volume"]
IMAGING_MORPHOLOGY_COLS = [
    "sphericity",
    "surface_area",
    "compactness",
    "elongation",
    "surface_volume_ratio",
]
IMAGING_DENSITY_COLS = [
    "density_mean",
    "density_std",
    "density_skewness",
    "density_kurtosis",
    "density_max",
    "density_range",
]
IMAGING_ALL_COLS = IMAGING_VOLUME_COLS + IMAGING_MORPHOLOGY_COLS + IMAGING_DENSITY_COLS
PASH_COLS = ["PASH_high_compactness", "PASH_low_dispersion", "PASH_fragmentation"]
INTERACTION_PAIRS = [
    ("p_brainstem", "hematoma_volume_log"),
    ("p_brainstem", "GCS"),
    ("p_lobar", "SBP"),
    ("p_lobar", "PASH_fragmentation"),
    ("p_basal", "hematoma_volume_log"),
    ("p_basal", "PASH_high_compactness"),
    ("p_thalamus", "hematoma_volume_log"),
    ("p_thalamus", "GCS"),
    ("p_cerebellum", "hematoma_volume_log"),
    ("p_brainstem", "PASH_low_dispersion"),
    ("density_mean", "GCS"),
    ("density_kurtosis", "GCS"),
    ("onset_to_ct_hours", "GCS"),
    ("GCS", "age"),
]
GCS_NONLINEAR_COLS = ["GCS_le8", "GCS_le12"]
CLINICAL_FEATURE_COLS = list(CLINICAL_COLS_MAP.values())
CT_OBJECTIVE_COLS = (
    ["hematoma_volume_log", "relative_volume"]
    + IMAGING_MORPHOLOGY_COLS
    + IMAGING_DENSITY_COLS
    + PASH_COLS
    + ["IVH", "SAH"]
)
INTER_NO_GCS_PAIRS = [
    (a, b) for a, b in INTERACTION_PAIRS if "GCS" not in a and "GCS" not in b
]
INTER_NO_GCS_COLS = [f"inter_{a}_{b}" for a, b in INTER_NO_GCS_PAIRS]
ABLATION_FEATURE_SETS = {
    "M0": ["GCS", "hematoma_volume_log", "age", "IVH"],
    "M_img": CT_OBJECTIVE_COLS,
    "M_loc": CT_OBJECTIVE_COLS + LOCATION_COLS,
    "M_age": CT_OBJECTIVE_COLS + LOCATION_COLS + ["age"],
    "M_inter": CT_OBJECTIVE_COLS + LOCATION_COLS + ["age"] + INTER_NO_GCS_COLS,
}
