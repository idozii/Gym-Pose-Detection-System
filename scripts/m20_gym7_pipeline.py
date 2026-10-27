"""M20 opt-in local research pipeline for NEW Push-up/Curl video-quality RFs.

Commands: scan, extract, train. Never accesses publisher TEST unless the
owner adds --evaluate-test. Models are NOT compatible with live 6-model API.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from gym_pose.gym7_research import scan_zip, extract_zip
from gym_pose.gym7_training import train_from_clip_csv


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    scan=sub.add_parser("scan",help="Create manually reviewable video/subject labels CSV")
    scan.add_argument("--zip",type=Path,required=True)
    scan.add_argument("--manifest",type=Path,default=Path("data/external/mendeley_gym7/m20_manifest.csv"))
    extract=sub.add_parser("extract",help="Process videos one at a time from ZIP")
    extract.add_argument("--zip",type=Path,required=True)
    extract.add_argument("--manifest",type=Path,default=Path("data/external/mendeley_gym7/m20_manifest.csv"))
    extract.add_argument("--output",type=Path,default=Path("data/features/m20_gym7/m20_clip_features.csv"))
    extract.add_argument("--sample-fps",type=float,default=2.)
    extract.add_argument("--max-frames",type=int,default=240)
    extract.add_argument("--max-video-mb",type=int,default=500)
    extract.add_argument("--limit",type=int,default=0,help="Pilot first N videos; zero means all")
    train=sub.add_parser("train",help="Train distinct clip-quality RFs (validation only by default)")
    train.add_argument("--features",type=Path,default=Path("data/features/m20_gym7/m20_clip_features.csv"))
    train.add_argument("--output-dir",type=Path,default=Path("artifacts/m20_gym7"))
    train.add_argument("--seed",type=int,default=42)
    train.add_argument("--trees",type=int,default=300)
    train.add_argument("--evaluate-test",action="store_true",
                       help="FINAL once-only TEST scoring, after all method decisions are frozen")
    opts=p.parse_args(argv)
    try:
        if opts.command=="scan":
            out=scan_zip(opts.zip,opts.manifest)
        elif opts.command=="extract":
            out=extract_zip(opts.zip,opts.manifest,opts.output,sample_fps=opts.sample_fps,
                            max_frames=opts.max_frames,max_video_mb=opts.max_video_mb,
                            limit=opts.limit)
        else:
            out=train_from_clip_csv(opts.features,opts.output_dir,seed=opts.seed,
                                    n_estimators=opts.trees,unlock_test=opts.evaluate_test)
    except (OSError,ValueError,RuntimeError,ImportError,EOFError) as error:
        print(f"M20 {opts.command} stopped: {type(error).__name__}: {error}",file=sys.stderr)
        return 1
    print(json.dumps(out,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
