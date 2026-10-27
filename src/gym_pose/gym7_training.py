"""Subject-disjoint Gym7 good/bad CLIP baseline - no independent-form claims.

This module is OFFLINE RESEARCH ONLY. Do not register new exercises in the
six-model public source-error registry without separate browser parity work.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit
from .gym7_research import EXERCISES, FEATURES, META, LABELS


def load_clip_table(path:Path) -> pd.DataFrame:
    df=pd.read_csv(path,keep_default_na=False)
    if tuple(df.columns)!=(*META,*FEATURES):
        raise ValueError("Expected exact Gym7 subject + 264 clip feature schema")
    if df.empty or df.video_id.duplicated().any():
        raise ValueError("Empty or duplicated videos are not eligible")
    if not df.exercise.isin(EXERCISES).all() or not df.label.isin(LABELS).all():
        raise ValueError("Only original good/bad Push-up/Curl labels are supported")
    if not df.subject_id.map(lambda x:isinstance(x,str) and bool(x.strip())).all():
        raise ValueError("Missing subject provenance is prohibited")
    if not df.view.isin(("front","side","diagonal")).all():
        raise ValueError("Invalid camera view")
    vals=df.loc[:,FEATURES].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("Nonfinite clip features")
    return df


def _complete_split(df, train_indices, val_indices, test_indices):
    splits={"train":df.iloc[train_indices],"val":df.iloc[val_indices],
            "test":df.iloc[test_indices]}
    groups={k:set(v.subject_id) for k,v in splits.items()}
    if not all(groups.values()) or any(groups[a]&groups[b] for a,b in
       (("train","val"),("train","test"),("val","test"))):
        return None
    for exercise in sorted(df.exercise.unique()):
        for part in splits.values():
            subset=part[part.exercise.eq(exercise)]
            if set(subset.label)!={"good","bad"}:
                return None
    return splits


def split_by_subject(df:pd.DataFrame, *,seed:int=42, attempts:int=250):
    """One cross-exercise subject split; all views and clips of one subject stay grouped."""
    if df.subject_id.nunique() < 12:
        raise ValueError("At least 12 unique verified people are required for M20 pilot")
    for trial in range(attempts):
        outer=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed+trial)
        tv, test=next(outer.split(df,groups=df.subject_id))
        train_val=df.iloc[tv]
        inner=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+trial+139)
        ta,va=next(inner.split(train_val,groups=train_val.subject_id))
        train,val=tv[ta],tv[va]
        complete=_complete_split(df,train,val,test)
        if complete is not None:
            return complete
    raise ValueError(
        "Not enough unique good/bad subjects per exercise for a 3-way "
        "person-separated split; collect more subjects or train one exercise."
    )


def _metrics(truth, probabilities, *,threshold=.5):
    truth=np.asarray(truth,dtype=int)
    pred=(np.asarray(probabilities)>=threshold).astype(int)
    if set(truth)!={0,1}:
        raise ValueError("Both labels must exist in every reported split")
    return {
        "n":int(len(truth)), "balanced_accuracy":float(balanced_accuracy_score(truth,pred)),
        "f1_bad":float(f1_score(truth,pred,zero_division=0)),
        "precision_bad":float(precision_score(truth,pred,zero_division=0)),
        "recall_bad":float(recall_score(truth,pred,zero_division=0)),
        "average_precision_bad":float(average_precision_score(truth,probabilities)),
        "confusion_matrix_good_bad":confusion_matrix(truth,pred,labels=[0,1]).tolist()
    }


def train_from_clip_csv(features:Path,output_dir:Path,*,seed:int=42,
                        unlock_test:bool=False,n_estimators:int=300):
    if n_estimators<25 or n_estimators>1000:
        raise ValueError("Keep RF complexity within bounded baseline values")
    df=load_clip_table(features)
    splits=split_by_subject(df,seed=seed)
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    # Intentionally prohibit rerunning an already evaluated held-out TEST on
    # the same output directory, even if model values are changed later.
    if unlock_test and (output_dir/"TEST_HAS_BEEN_EVALUATED.txt").exists():
        raise FileExistsError("Locked TEST results already exist in output directory")
    report={
        "dataset":"Mendeley Gym7 v3","doi":"10.17632/kgbb3yn47p.3",
        "feature_schema":"gym7-clip-264-v1",
        "task":"publisher GOOD/BAD video-level quality (not anatomical error detection)",
        "split":"SUBJECT_ID-disjoint cross exercise and cross viewpoint",
        "test_accessed":bool(unlock_test),"seed":seed,
        "split_subject_counts":{key:int(data.subject_id.nunique()) for key,data in splits.items()},
        "split_video_counts":{key:int(len(data)) for key,data in splits.items()},
        "models":{}
    }
    import joblib
    for exercise in sorted(df.exercise.unique()):
        sub={k:v[v.exercise.eq(exercise)] for k,v in splits.items()}
        fit=sub["train"]
        model=RandomForestClassifier(
            n_estimators=n_estimators,min_samples_leaf=2,
            random_state=seed,class_weight="balanced",n_jobs=1)
        X=fit.loc[:,FEATURES].to_numpy(dtype=float)
        y=fit.label.map({"good":0,"bad":1}).to_numpy(dtype=int)
        model.fit(X,y)
        if not np.array_equal(model.classes_,np.array([0,1])):
            raise ValueError("Train split unexpectedly missing a publisher quality class")
        exercise_report={"subject_counts":{k:int(v.subject_id.nunique()) for k,v in sub.items()},
                         "clip_counts":{k:int(len(v)) for k,v in sub.items()},
                         "evaluations":{}}
        for part in (("val","test") if unlock_test else ("val",)):
            data=sub[part]
            prob=model.predict_proba(data.loc[:,FEATURES].to_numpy(dtype=float))[:,1]
            actual=data.label.map({"good":0,"bad":1}).to_numpy(dtype=int)
            exercise_report["evaluations"][part]=_metrics(actual,prob)
            views={}
            for view,indices in data.groupby("view").groups.items():
                group=data.loc[indices]
                if set(group.label)=={"good","bad"}:
                    index=data.index.get_indexer(indices)
                    views[view]=_metrics(group.label.map({"good":0,"bad":1}),prob[index])
                else:
                    views[view]={"n":len(group),"note":"one class - no view score"}
            exercise_report["evaluations"][part]["by_view"]=views
        artifact={
            "model":model,"exercise":exercise,"dataset":"Mendeley Gym7 v3",
            "feature_schema":"gym7-clip-264-v1","feature_names":list(FEATURES),
            "quality_label_semantics":"publisher video good/bad, NOT fine-grained form errors",
            "classes":["good","bad"],"classes_encoded":[0,1],
            "split":"subject-disjoint","research_only":True,
            "web_deployable":False,
            "warning":"Cannot score a single browser landmark without temporal clip features"
        }
        model_path=output_dir/f"{exercise}_clip_quality_rf.joblib"
        if model_path.exists():
            raise FileExistsError("Model already exists; do not overwrite prior experiment")
        joblib.dump(artifact,model_path)
        report["models"][exercise]=exercise_report
    report_path=output_dir/"m20_research_report.json"
    if report_path.exists():
        raise FileExistsError("Report already exists")
    report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if unlock_test:
        (output_dir/"TEST_HAS_BEEN_EVALUATED.txt").write_text(
            "Publisher independent test has been inspected. Do not tune on TEST.\n",encoding="utf-8"
        )
    return report
