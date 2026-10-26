"""Command-line entry point: gym-pose [capture|extract|train|webcam]."""
import argparse
import json

from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Gym Pose modular squat baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--label", choices=["correct", "incorrect"], required=True)
    cap.add_argument("--count", type=int, default=100)
    cap.add_argument("--camera", type=int, default=0)
    ext = commands.add_parser("extract")
    ext.add_argument("--min-visibility", type=float, default=.5)
    fit = commands.add_parser("train")
    fit.add_argument("--group-column", choices=["subject_id", "session_id"])
    fit.add_argument("--seed", type=int, default=42)
    cam = commands.add_parser("webcam")
    cam.add_argument("--camera", type=int, default=0)
    cam.add_argument("--min-visibility", type=float, default=.5)
    args = parser.parse_args()
    if args.command == "capture":
        from .camera import capture
        result = {"captured": capture("data/raw", args.label, args.count, args.camera)}
    elif args.command == "extract":
        from .camera import extract
        result = extract("data/raw", "data/features/squat_features_v1.csv", args.min_visibility)
    elif args.command == "train":
        result = train("data/features/squat_features_v1.csv", "artifacts/squat_rf_v1.joblib",
                       group_column=args.group_column, seed=args.seed)
    else:
        from .camera import webcam
        webcam("artifacts/squat_rf_v1.joblib", args.camera, args.min_visibility)
        result = {"status": "camera_closed"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
