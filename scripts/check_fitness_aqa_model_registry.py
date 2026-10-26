#!/usr/bin/env python3
"""Print M15A's offline research model-availability registry.

Pass --trust-local-artifacts only for your own trusted joblib model
files: pickle deserialization can execute arbitrary Python code.
Nothing downloads models or opens publisher media/test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from gym_pose.model_registry import KNOWN_TASKS, DEFAULT_PATHS, ModelRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-local-artifacts", action="store_true",
                        help="Opt in to executing deserialization of your OWN locally trained joblib files")
    parser.add_argument("--squat-artifacts", type=Path,
                        default=Path("artifacts/fitness_aqa_squat_source_rf"))
    parser.add_argument("--ohp-artifacts", type=Path,
                        default=Path("artifacts/fitness_aqa_ohp_task_rf"))
    parser.add_argument("--row-artifacts", type=Path,
                        default=Path("artifacts/fitness_aqa_barbell_row_torso4_rf"))
    args = parser.parse_args()
    paths = {}
    for (exercise, target), path in DEFAULT_PATHS.items():
        paths[exercise, target] = (
            (args.squat_artifacts if exercise == "squat" else
             args.ohp_artifacts if exercise == "ohp" else args.row_artifacts)
            / path.name
        )
    registry = ModelRegistry(paths=paths)
    if args.trust_local_artifacts:
        try:
            status = registry.load_local(trust_local_pickle_files=True)
        except (OSError, ValueError, TypeError, ImportError) as error:
            parser.exit(1, f"Model registry validation failed: {error}\n")
    else:
        status = {
            f"{exercise}/{target}": (
                "unloaded_requires_explicit_trust")
            for exercise, target in KNOWN_TASKS
        }
    print(json.dumps({
        "status": "research_model_registry_only_NO_TEST",
        "publisher_test_accessed": False,
        "no_model_deployment": True,
        "artifacts_trusted_by_user": args.trust_local_artifacts,
        "models": status,
        "warning": "Do not use experimental source-label scores as reliable fitness-safety warnings",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
