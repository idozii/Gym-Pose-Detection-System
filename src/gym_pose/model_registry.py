"""Offline, fail-closed research model registry for Fitness-AQA tasks.

Registry is intentionally NOT a production fitness-safety assistant.
It only consumes LOCAL artifacts made by this project; joblib uses pickle
and MUST NEVER be loaded from untrusted files. A missing model, missing
feature schema, or failed MediaPipe visibility gate returns abstention,
not a 'correct technique' or safe-to-train judgment.

Squat source-error detectors are reserved but not silently mapped to
the legacy notebook's unrelated correct/incorrect classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .barbell_row_torso4_features import (
    FEATURE_NAMES as ROW_FEATURE_NAMES,
    SCHEMA_VERSION as ROW_SCHEMA_VERSION,
    pose_to_torso4_features,
)
from .ohp_task_features import (
    FEATURES_BY_TASK as OHP_FEATURES_BY_TASK,
    SCHEMA_BY_TASK as OHP_SCHEMA_BY_TASK,
    pose_to_task_features,
)
from .features import (
    FEATURE_NAMES as SQUAT_FEATURE_NAMES,
    SCHEMA_VERSION as SQUAT_SCHEMA_VERSION,
    pose_to_features as pose_to_squat_features,
)

DEFAULT_PATHS = {
    ("squat", "error_knees_forward"): Path(
        "artifacts/fitness_aqa_squat_source_rf/rf_error_knees_forward_static_source.joblib"),
    ("squat", "error_knees_inward"): Path(
        "artifacts/fitness_aqa_squat_source_rf/rf_error_knees_inward_static_source.joblib"),
    ("ohp", "error_elbows"): Path(
        "artifacts/fitness_aqa_ohp_task_rf/error_elbows_static.joblib"),
    ("ohp", "error_knees"): Path(
        "artifacts/fitness_aqa_ohp_task_rf/error_knees_static.joblib"),
    ("barbell_row", "torso_angle"): Path(
        "artifacts/fitness_aqa_barbell_row_torso4_rf/barbell_row_torso_angle_torso4_static_rf.joblib"),
    ("barbell_row", "lumbar"): Path(
        "artifacts/fitness_aqa_barbell_row_torso4_rf/barbell_row_lumbar_torso4_static_rf.joblib"),
}
KNOWN_TASKS = (
    ("squat", "error_knees_forward"),
    ("squat", "error_knees_inward"),
    ("ohp", "error_elbows"),
    ("ohp", "error_knees"),
    ("barbell_row", "torso_angle"),
    ("barbell_row", "lumbar"),
)
EXPERIMENTAL_ONLY = True


@dataclass(frozen=True)
class ModelContract:
    exercise: str
    target: str
    schema_version: str
    feature_names: tuple[str, ...]
    threshold_key: str
    variant: str | None
    path: Path


def _contract(exercise: str, target: str, *, path: Path | None = None) -> ModelContract | None:
    key = (exercise, target)
    if key not in KNOWN_TASKS:
        raise ValueError(f"Unknown exercise/source label: {exercise}/{target}")
    if key not in DEFAULT_PATHS:
        return None
    model_path = Path(path) if path is not None else DEFAULT_PATHS[key]
    if exercise == "squat":
        # These exact 103 features are shared with the old notebook,
        # but its unrelated "correct"/"incorrect" LABELS are forbidden.
        return ModelContract(
            exercise, target, SQUAT_SCHEMA_VERSION, SQUAT_FEATURE_NAMES,
            "threshold_selected_on_train_groups_only", "static", model_path)
    if exercise == "barbell_row":
        return ModelContract(
            exercise, target, ROW_SCHEMA_VERSION, ROW_FEATURE_NAMES,
            "threshold_train_prefix_calibration", None, model_path)
    # M15B integrates exactly the OHP source 30-feature extraction schema.
    # Never coerce it to the earlier 111-column baseline, and never silently
    # serve a causal temporal 42-feature model in a static frame context.
    return ModelContract(
        exercise, target, OHP_SCHEMA_BY_TASK[target], OHP_FEATURES_BY_TASK[target],
        "threshold_selected_on_train_groups_only", "static", model_path)


def _abstain(exercise: str, target: str, reason: str) -> dict:
    return {
        "exercise": exercise, "target": target,
        "status": "abstain", "reason": reason,
        "error_probability": None, "threshold": None,
        "source_label_flag": None,
        "experimental_only": True,
    }


def _validate_bundle(bundle: Any, contract: ModelContract) -> dict:
    if not isinstance(bundle, dict):
        raise ValueError("Artifact must contain model provenance metadata")
    if bundle.get("exercise") != contract.exercise:
        raise ValueError("Model exercise metadata mismatch")
    if bundle.get("target") != contract.target:
        raise ValueError("Model source-label metadata mismatch")
    if bundle.get("schema_version") != contract.schema_version:
        raise ValueError("Model feature schema/version mismatch")
    if bundle.get("feature_names") != list(contract.feature_names):
        raise ValueError("Model feature order/names mismatch")
    if contract.variant is not None and bundle.get("variant") != contract.variant:
        raise ValueError("Only the STATIC source-error bundle is supported by the single-frame registry")
    if contract.exercise == "squat" and (
        bundle.get("annotation_semantics") != "publisher_named_error_interval_NOT_correctness"
        or bundle.get("classes") != [0, 1]
    ):
        raise ValueError("Source Squat model must attest named publisher knee-error labels (not correct/incorrect)")
    if bundle.get("source_train_only") is not True:
        raise ValueError("Model must be fit from source TRAIN only")
    if bundle.get("publisher_test_untouched") is not True:
        raise ValueError("Model must attest publisher TEST data was untouched")
    if contract.exercise == "barbell_row" and (
        bundle.get("no_direct_lumbar_spine_measurement") is not True
    ):
        raise ValueError("Model lacks mandatory non-clinical lumbar disclaimer")
    raw_threshold = bundle.get(contract.threshold_key)
    if (isinstance(raw_threshold, bool) or
        not isinstance(raw_threshold, (int, float, np.integer, np.floating)) or
        not np.isfinite(raw_threshold) or not 0 < float(raw_threshold) < 1):
        raise ValueError("Missing/invalid original train-only operating threshold")
    fitted = bundle.get("model")
    if fitted is None or not callable(getattr(fitted, "predict_proba", None)):
        raise ValueError("Model must provide a fitted probability estimator")
    if (not np.array_equal(getattr(fitted, "classes_", []), [0, 1])
        or int(getattr(fitted, "n_features_in_", -1)) != len(contract.feature_names)):
        raise ValueError("Model classes/features mismatch (source errors must be binary 0/1)")
    if hasattr(fitted, "feature_names_in_") and (
        list(fitted.feature_names_in_) != list(contract.feature_names)
    ):
        raise ValueError("Estimator fitted-column order mismatches source schema")
    if "classes" in bundle and not np.array_equal(bundle["classes"], [0, 1]):
        raise ValueError("Artifact classes must be [0,1] for source error labels")
    return bundle


class ModelRegistry:
    """Explicitly opt into loading trusted local research model artifacts.

    Construction never executes a pickle: loads only after the caller opts in.
    A task whose model/schema is absent may still appear as 'abstain' in
    reports and is NEVER silently replaced by a generic correctness model.
    """

    def __init__(self, *, paths: Mapping[tuple[str, str], str | Path] | None = None):
        self.paths = dict(DEFAULT_PATHS)
        if paths:
            for key, value in paths.items():
                if key not in DEFAULT_PATHS:
                    raise ValueError(f"No trained artifact contract for {key}")
                self.paths[key] = Path(value)
        self._models: dict[tuple[str, str], tuple[ModelContract, dict]] = {}

    def load_local(self, *, trust_local_pickle_files: bool = False) -> dict:
        if not trust_local_pickle_files:
            raise ValueError(
                "joblib artifacts use pickle; explicitly opt in to loading ONLY "
                "your own trusted local model files")
        import joblib
        result = {}
        for exercise, target in KNOWN_TASKS:
            key = (exercise, target)
            if key not in DEFAULT_PATHS:
                result[f"{exercise}/{target}"] = "not_trained_as_source_error_detector"
                continue
            contract = _contract(exercise, target, path=self.paths[key])
            if contract is None:
                result[f"{exercise}/{target}"] = "feature_schema_not_integrated"
                continue
            if not contract.path.is_file():
                result[f"{exercise}/{target}"] = "model_artifact_missing"
                continue
            # Never load URLs or download artifacts. Trust must be supplied
            # explicitly by caller; pickle's code execution is NOT sandboxed.
            bundle = _validate_bundle(joblib.load(contract.path), contract)
            self._models[key] = (contract, bundle)
            result[f"{exercise}/{target}"] = "loaded_experimental_only"
        return result

    def score_features(
        self, exercise: str, target: str,
        features: Mapping[str, float] | None,
    ) -> dict:
        key = (exercise, target)
        if key not in KNOWN_TASKS:
            raise ValueError(f"Unknown exercise/source label: {exercise}/{target}")
        if key not in self._models:
            return _abstain(exercise, target, "model_not_loaded_or_unavailable")
        if features is None:
            return _abstain(exercise, target, "missing_or_rejected_pose")
        contract, bundle = self._models[key]
        if not isinstance(features, Mapping):
            return _abstain(exercise, target, "invalid_pose_feature_map")
        if set(features) != set(contract.feature_names):
            return _abstain(exercise, target, "missing_or_mismatched_feature_schema")
        try:
            vector = np.asarray([features[name] for name in contract.feature_names],
                                dtype=float)
        except (ValueError, TypeError):
            return _abstain(exercise, target, "invalid_pose_feature_values")
        if not np.isfinite(vector).all():
            return _abstain(exercise, target, "nonfinite_pose_features")
        x = pd.DataFrame([vector], columns=list(contract.feature_names))
        probs = np.asarray(bundle["model"].predict_proba(x)[0], dtype=float)
        if (probs.shape != (2,) or not np.isfinite(probs).all()
            or (probs < 0).any() or (probs > 1).any()
            or not np.isclose(probs.sum(), 1., atol=1e-6)):
            return _abstain(exercise, target, "invalid_model_probability")
        # Preserve source classification semantics; no 'safe exercise'
        # judgment, clinical injury-prevention advice, or automatic alerts.
        threshold = float(bundle[contract.threshold_key])
        return {
            "exercise": exercise, "target": target,
            "status": "experimental_source_label_score",
            "error_probability": float(probs[1]),
            "threshold": threshold,
            "source_label_flag": bool(probs[1] >= threshold),
            "schema_version": contract.schema_version,
            "feature_count": len(contract.feature_names),
            "experimental_only": True,
            "warning": "Source-label association ONLY; NOT verified proper technique or medical spinal safety",
        }

    def score_landmarks(
        self, exercise: str, target: str, landmarks: Any,
        *, min_visibility: float = .5,
    ) -> dict:
        key = (exercise, target)
        if key not in KNOWN_TASKS:
            raise ValueError(f"Unknown exercise/source label: {exercise}/{target}")
        if key not in self._models:
            return _abstain(exercise, target, "model_not_loaded_or_unavailable")
        if exercise == "barbell_row":
            features = pose_to_torso4_features(
                landmarks, min_visibility=min_visibility)
        elif exercise == "ohp":
            features = pose_to_task_features(
                landmarks, task=target, min_visibility=min_visibility)
        elif exercise == "squat":
            features = pose_to_squat_features(
                landmarks, min_visibility=min_visibility)
        else:
            return _abstain(exercise, target, "source_error_model_not_integrated")
        return self.score_features(exercise, target, features)
