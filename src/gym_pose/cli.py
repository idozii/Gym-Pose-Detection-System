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
    template = commands.add_parser("manifest-template", help="Create an empty provenance CSV")
    template.add_argument("--output", default="data/manifest.csv")
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", default="data/manifest.csv")
    validate.add_argument("--image-root", default="data/raw")
    extract2 = commands.add_parser("extract-manifest")
    extract2.add_argument("--manifest", default="data/manifest.csv")
    extract2.add_argument("--image-root", default="data/raw")
    extract2.add_argument("--output", default="data/features/squat_provenance_v1.csv")
    extract2.add_argument("--min-visibility", type=float, default=.5)
    evaluate = commands.add_parser("benchmark")
    evaluate.add_argument("--features", default="data/features/squat_provenance_v1.csv")
    evaluate.add_argument("--output-dir", default="artifacts/benchmark")
    evaluate.add_argument("--seed", type=int, default=42)
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
    elif args.command == "manifest-template":
        from .dataset import manifest_template
        result = {"template": str(manifest_template(args.output))}
    elif args.command == "validate-manifest":
        from .dataset import summarize_manifest
        result = summarize_manifest(args.manifest, args.image_root)
    elif args.command == "extract-manifest":
        from .dataset import extract_manifest
        result = extract_manifest(args.manifest, args.image_root, args.output,
                                  min_visibility=args.min_visibility)
    elif args.command == "benchmark":
        from .evaluation import benchmark
        result = benchmark(args.features, args.output_dir, seed=args.seed)
    else:
        from .camera import webcam
        webcam("artifacts/squat_rf_v1.joblib", args.camera, args.min_visibility)
        result = {"status": "camera_closed"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
