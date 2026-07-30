"""Stable paths for the committed sentiment model artifacts."""

from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
MODEL_ARTIFACT_PATH = OUTPUTS_DIR / "logreg_model.pkl"
LABEL_ENCODER_ARTIFACT_PATH = OUTPUTS_DIR / "label_encoder.pkl"
