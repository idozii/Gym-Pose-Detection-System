"""M28: frozen TRAIN-only Push-up temporal feature-family robustness.

No source ZIP, VAL/TEST video, VAL/TEST features, model artifacts or web
services are used. Compare four fixed RF feature sets on identical original
TRAIN clips and identical participant-heldout folds, then give descriptive
paired subject-bootstrap intervals using *fixed* OOF predictions.

This is post-M26 exploratory analysis, not independent validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from gym_pose.gym7_research import read_manifest
from gym_pose.gym7_training import load_clip_table
from gym_pose.gym7_m24_cv import _metrics, _read_table
from scripts.m23_train_only_pushup_features import FEATURE_COLUMNS, META, frozen_train_manifest
from scripts.m27_pushup_temporal_train import MOTION_COLS, _read_temporal

SEED = 42
TREES = 200
KINEMATIC = MOTION_COLS[:12]
QUALITY = MOTION_COLS[12:]
FEATURE_SETS = {
    "baseline_64": FEATURE_COLUMNS,
    "kinematic_76": (*FEATURE_COLUMNS, *KINEMATIC),
    "quality_proxy_68": (*FEATURE_COLUMNS, *QUALITY),
    "temporal_80": (*FEATURE_COLUMNS, *MOTION_COLS),
}
PAIRED_ALTERNATIVES = tuple(x for x in FEATURE_SETS if x != "baseline_64")


def verify_exact_train_cohort(base, motion, original_manifest, original_m20):
    eligible_rows, original_train_people = frozen_train_manifest(
        original_manifest, original_m20, seed=SEED
    )
    by_id = {r["video_id"]:r for r in eligible_rows}
    if len(by_id) != len(eligible_rows):
        raise ValueError("Duplicated publisher source video IDs")
    if not set(base.index) == set(motion.index):
        raise ValueError("M23/M27 must contain the identical original TRAIN video IDs")
    if len(base) < 10 or base.subject_id.nunique() < 5:
        raise ValueError("Insufficient original TRAIN subjects for reliable CV")
    for table in (base, motion):
        if not set(table.index).issubset(by_id):
            raise ValueError("M28 contains non-TRAIN or unknown source video ID")
        for vid in table.index:
            src = by_id[vid]
            if (src["subject_id"] not in original_train_people
                or any(table.at[vid,k] != src[k] for k in META)):
                raise ValueError("M28 original M20 TRAIN subject metadata mismatch")
    return sorted(base.index)


def _train_predict(frame, names, train_idx, held_idx):
    tr=frame.iloc[train_idx]
    te=frame.iloc[held_idx]
    if set(tr.subject_id) & set(te.subject_id):
        raise AssertionError("TRAIN/held-out source participant leakage")
    labels = tr.label.map({"good":0,"bad":1}).to_numpy(dtype=int)
    if set(labels) != {0,1}:
        raise ValueError("A TRAIN fold is missing an original publisher class")
    classifier = RandomForestClassifier(
        n_estimators=TREES, min_samples_leaf=2,
        class_weight="balanced", random_state=SEED, n_jobs=1
    )
    classifier.fit(tr.loc[:,names].to_numpy(dtype=float), labels)
    classes=list(classifier.classes_)
    if classes != [0,1]:
        raise ValueError("Unexpected source publisher class order")
    return classifier.predict_proba(te.loc[:,names].to_numpy(dtype=float))[:,1]


def _balanced(y,p):
    y=np.asarray(y,dtype=int)
    pred=np.asarray(p,dtype=float)>=.5
    if set(y)!={0,1}: return None
    return float(.5*(np.mean(pred[y==1])+np.mean(~pred[y==0])))


def _recall_bad(y,p):
    y=np.asarray(y,dtype=int)
    if not np.any(y==1):return None
    return float(np.mean((np.asarray(p,dtype=float)>=.5)[y==1]))


def paired_cluster_bootstrap(truth, prediction_maps, groups, *,
                             resamples=1000, seed=42):
    """Resample subjects, hold predictions fixed; intervals exploratory only."""
    if not 100<=resamples<=5000:
        raise ValueError("Bootstrap draws must be in 100..5000")
    ids=sorted(truth)
    if any(set(p)!=set(ids) for p in prediction_maps.values()):
        raise ValueError("All variants must score the same paired cohort")
    people=sorted({groups[i] for i in ids})
    if len(people)<5:raise ValueError("Need at least five TRAIN subjects")
    by_person={person:[i for i in ids if groups[i]==person] for person in people}
    rng=np.random.default_rng(seed)
    sampled={variant:{"balanced_accuracy":[],"recall_bad":[]}
             for variant in PAIRED_ALTERNATIVES}
    skipped=0
    for _ in range(resamples):
        selected=[vid for group in rng.choice(people,size=len(people),replace=True)
                  for vid in by_person[group]]
        y=[truth[i] for i in selected]
        if set(y)!={0,1}:
            skipped+=1
            continue
        reference=[prediction_maps["baseline_64"][i] for i in selected]
        base_bal=_balanced(y,reference)
        base_rec=_recall_bad(y,reference)
        for variant in PAIRED_ALTERNATIVES:
            scores=[prediction_maps[variant][i] for i in selected]
            sampled[variant]["balanced_accuracy"].append(_balanced(y,scores)-base_bal)
            sampled[variant]["recall_bad"].append(_recall_bad(y,scores)-base_rec)
    interval={}
    for variant in PAIRED_ALTERNATIVES:
        interval[variant]={}
        for metric,data in sampled[variant].items():
            if not data:
                interval[variant][metric]=None
                continue
            q=np.quantile(data,[.025,.5,.975])
            interval[variant][metric]={
                "delta_vs_baseline_64_lower_95":float(q[0]),
                "delta_vs_baseline_64_median":float(q[1]),
                "delta_vs_baseline_64_upper_95":float(q[2]),
                "fraction_positive":float(np.mean(np.asarray(data)>0))
            }
    return {
        "draws_requested":resamples,"usable_draws":resamples-skipped,
        "skipped_single_label_draws":skipped,
        "interval_type":"descriptive percentile interval; fixed OOF predictions, "
                        "participant bootstrap; no refitting and not independent validation",
        "paired_deltas":interval,
    }


def run_study(baseline_path, motion_path, manifest_path, original_path, *,
              folds=5, bootstrap=1000):
    if folds != 5:raise ValueError("Fixed M27 GroupKFold=5 is required")
    baseline=_read_table(baseline_path)
    motion=_read_temporal(motion_path)
    manifest=read_manifest(manifest_path)
    original=load_clip_table(original_path)
    ids=verify_exact_train_cohort(baseline,motion,manifest,original)
    base=baseline.loc[ids]
    data=base.loc[:,(*META,*FEATURE_COLUMNS)].join(
        motion.loc[ids,MOTION_COLS],how="inner"
    )
    if (len(data)!=len(ids) or data.subject_id.nunique()<5
        or not np.isfinite(data.loc[:,FEATURE_SETS["temporal_80"]].to_numpy(dtype=float)).all()):
        raise ValueError("Feature join lost clips or has nonfinite data")
    truth={vid:int(data.at[vid,"label"]=="bad") for vid in ids}
    subjects={vid:str(data.at[vid,"subject_id"]) for vid in ids}
    y=np.array([truth[vid] for vid in ids],dtype=int)
    strategies={
        "group5":GroupKFold(n_splits=folds),
        "loso":LeaveOneGroupOut(),
    }
    results={}
    uncertainty={}
    for strategy, cv in strategies.items():
        scores={name:{} for name in FEATURE_SETS}
        fold_count=0
        for train_idx, held_idx in cv.split(data,groups=data.subject_id):
            fold_count+=1
            for name, columns in FEATURE_SETS.items():
                probabilities=_train_predict(data,columns,train_idx,held_idx)
                for index,prob in zip(held_idx,probabilities):
                    vid=ids[index]
                    if vid in scores[name]:
                        raise AssertionError("Repeated original TRAIN OOF clip score")
                    scores[name][vid]=float(prob)
        if any(set(scored)!=set(ids) for scored in scores.values()):
            raise AssertionError("Unscored original TRAIN clip in CV")
        models={}
        for name,prediction in scores.items():
            paired=[(truth[vid],prediction[vid]) for vid in ids]
            models[name]={
                "all_paired":_metrics(paired),
                "by_view":{
                    view:_metrics([(truth[vid],prediction[vid]) for vid in ids
                                   if data.at[vid,"view"]==view])
                    for view in ("front","side","diagonal")
                }
            }
        results[strategy]={
            "folds":fold_count,
            "exact_same_clips":len(ids),
            "models":models,
        }
        uncertainty[strategy]=paired_cluster_bootstrap(
            truth,scores,subjects,resamples=bootstrap,seed=SEED
        )
    return {
        "study":"M28 original TRAIN-only temporal feature family ablation",
        "dataset":"Mendeley Gym7 v3",
        "original_source_split_seed":SEED,
        "paired_train_clip_count":len(ids),
        "train_people":int(data.subject_id.nunique()),
        "feature_sets":{name:len(cols) for name,cols in FEATURE_SETS.items()},
        "quality_proxy_features":list(QUALITY),
        "rf_trees":TREES,
        "rf_leaf_min":2,
        "threshold":.5,
        "results":results,
        "cluster_uncertainty":uncertainty,
        "original_M20_VAL_or_TEST_used":False,
        "raw_video_decoded":False,
        "caution":"Post-M26 exploratory TRAIN robustness only. Cluster intervals are "
                  "descriptive and should not guide repeated VAL or TEST tuning. "
                  "A pose-quality coverage win is not a validated form-correctness win."
    }


def _check_m27_repro(report, m27_path):
    """Re-run same GroupKFold and verify M27 reported baseline/temporal metrics."""
    if not m27_path.is_file():return "M27 original report absent; no reproducibility comparison"
    prev=json.loads(m27_path.read_text(encoding="utf-8"))
    for name in ("baseline_64","temporal_80"):
        earlier=prev["metrics"][name]["all_paired"]
        current=report["results"]["group5"]["models"][name]["all_paired"]
        for key in ("balanced_accuracy","f1_bad","recall_bad","average_precision_bad"):
            if not np.isclose(float(earlier[key]),float(current[key]),rtol=0,atol=1e-12):
                raise ValueError(f"M28 failed to reproduce original M27 {name}: {key}")
    return "Original M27 5-fold baseline and temporal metrics reproduced exactly"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upper",type=Path,default=Path(
        "data/features/m23_pushup_train_upper_v1/train_upper_body.csv"))
    parser.add_argument("--motion",type=Path,default=Path(
        "data/features/m27_pushup_train_motion_v1.csv"))
    parser.add_argument("--manifest",type=Path,default=Path(
        "data/external/mendeley_gym7/m20_manifest.csv"))
    parser.add_argument("--original",type=Path,default=Path(
        "data/features/m20_gym7/m20_clip_features.csv"))
    parser.add_argument("--m27-report",type=Path,default=Path(
        "data/processed/m27_pushup_temporal_train_cv.json"))
    parser.add_argument("--report",type=Path,default=Path(
        "data/processed/m28_pushup_temporal_robustness.json"))
    parser.add_argument("--bootstrap",type=int,default=1000)
    args=parser.parse_args(argv)
    try:
        if args.report.exists():
            raise FileExistsError("M28 report already exists; will not overwrite")
        report=run_study(args.upper,args.motion,args.manifest,args.original,
                         bootstrap=args.bootstrap)
        report["m27_reproducibility"]=_check_m27_repro(report,args.m27_report)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    except (OSError,ValueError,AssertionError,KeyError) as error:
        print(f"M28 stopped: {error}",file=sys.stderr)
        return 1
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
