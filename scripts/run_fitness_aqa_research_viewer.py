#!/usr/bin/env python3
"""Opt-in LOCAL experimental webcam viewer for six Fitness-AQA source labels.

Unlike the historical 'gym-pose webcam' correct/incorrect notebook, this
research viewer only shows explicit original publisher source-label scores
from exact source-matched models. No safety or injury claims and no media
recording; no publisher TEST clips/pixels used.
"""
from __future__ import annotations

import argparse
import json
import sys

from gym_pose.model_registry import ModelRegistry
from gym_pose.research_viewer import TASKS_BY_EXERCISE, run_live_viewer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", choices=TASKS_BY_EXERCISE, default="squat")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional local smoke-test frame cap; no images saved")
    parser.add_argument("--no-framing-guide", action="store_true",
                        help="Hide the dedicated live joint visibility side panel")
    parser.add_argument("--trust-local-artifacts", action="store_true",
                        help="Required: only your OWN trained joblib files (pickle can execute code)")
    args = parser.parse_args()
    if not args.trust_local_artifacts:
        parser.exit(2, "For your OWN local joblib artifacts, pass --trust-local-artifacts. "
                       "Untrusted pickle files can execute code.\n")
    registry = ModelRegistry()
    try:
        states = registry.load_local(trust_local_pickle_files=True)
        print(json.dumps({
            "status": "experimental_registry_preflight",
            "models": states,
            "warning": "Research-only source labels; never interpret as movement safety",
        }, indent=2))
        result = run_live_viewer(
            registry, exercise=args.exercise,
            camera_index=args.camera, max_frames=args.max_frames,
            show_framing_guide=not args.no_framing_guide)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as err:
        parser.exit(1, f"Local research viewer stopped: {err}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
