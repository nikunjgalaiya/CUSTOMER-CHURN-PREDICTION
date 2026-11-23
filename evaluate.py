"""Rebuild a baseline from source data; never load the historical pickle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,average_precision_score,confusion_matrix,
    f1_score,precision_score,recall_score,roc_auc_score)
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
FEATURES=["Subscription_Length_Months","Monthly_Bill","Total_Usage_GB"]

def load_data(path):
    frame=pd.read_excel(path)
    if any(c not in frame for c in FEATURES+["Churn"]):raise ValueError("Missing required feature or target")
    if frame["Churn"].isna().any() or set(frame["Churn"].unique())!={0,1}:
        raise ValueError("Churn must contain both binary labels 0 and 1")
    for col in FEATURES:
        frame[col]=pd.to_numeric(frame[col],errors="raise")
        if not np.isfinite(frame[col]).all() or (frame[col]<0).any():raise ValueError("Features must be finite non-negative values")
    if "CustomerID" in frame and frame["CustomerID"].duplicated().any():raise ValueError("Duplicate CustomerID")
    return frame

def scores(actual,probability,threshold=.5):
    predicted=(np.asarray(probability)>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(actual,predicted,labels=[0,1]).ravel()
    return {"roc_auc":float(roc_auc_score(actual,probability)),
      "average_precision":float(average_precision_score(actual,probability)),
      "accuracy":float(accuracy_score(actual,predicted)),
      "precision":float(precision_score(actual,predicted,zero_division=0)),
      "recall":float(recall_score(actual,predicted,zero_division=0)),
      "f1":float(f1_score(actual,predicted,zero_division=0)),
      "threshold":threshold,"confusion_matrix":{"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}}

def run(data=ROOT/"customer_churn_large_dataset.xlsx",output=ROOT/"reports"):
    frame=load_data(data);indices=np.arange(len(frame))
    train,test=train_test_split(indices,test_size=.2,stratify=frame["Churn"],random_state=42)
    x,y=frame[FEATURES],frame["Churn"]
    model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=500,random_state=42))
    folds=StratifiedKFold(5,shuffle=True,random_state=42)
    cv=cross_val_score(model,x.iloc[train],y.iloc[train],cv=folds,scoring="roc_auc",n_jobs=1)
    model.fit(x.iloc[train],y.iloc[train])
    dummy=DummyClassifier(strategy="prior").fit(x.iloc[train],y.iloc[train])
    probability=model.predict_proba(x.iloc[test])[:,1]
    baseline=dummy.predict_proba(x.iloc[test])[:,1]
    # Bootstrap the held-out pairs, not separate label/probability samples.
    rng=np.random.default_rng(42);aucs=[]
    labels=y.iloc[test].to_numpy()
    for _ in range(200):
        sample=rng.integers(0,len(test),len(test))
        if np.unique(labels[sample]).size==2:aucs.append(roc_auc_score(labels[sample],probability[sample]))
    ci=[float(v) for v in np.quantile(aucs,[.025,.975])]
    result={"source_sha256":hashlib.sha256(Path(data).read_bytes()).hexdigest(),
      "rows":len(frame),"train_rows":len(train),"holdout_rows":len(test),
      "features":FEATURES,"excluded_fields":["CustomerID","Name","Age","Gender","Location"],
      "split":{"method":"stratified random holdout","test_fraction":.2,"random_state":42},
      "training_cv":{"roc_auc_mean":float(cv.mean()),"roc_auc_std":float(cv.std()),"folds":5},
      "holdout":{"logistic_regression":scores(labels,probability),"prior_baseline":scores(labels,baseline)},
      "roc_auc_bootstrap_95_interval":ci,
      "decision":"Evaluate against baseline and uncertainty before a retention pilot; no realised churn reduction is claimed",
      "evidence_of_signal":bool(ci[0]>.5)}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/"metrics.json").write_text(json.dumps(result,indent=2))
    # Export anonymous row indices only; do not republish names or customer identifiers.
    pd.DataFrame({"source_row":test,"actual":labels,"probability":probability}).to_csv(output/"holdout_predictions.csv",index=False)
    cohort=frame.assign(tenure_group=pd.cut(frame["Subscription_Length_Months"],bins=[-1,6,12,24,float("inf")],labels=["0-6","7-12","13-24","25+"]))
    cohort.groupby("tenure_group",observed=False)["Churn"].agg(["count","mean"]).rename(columns={"count":"customers","mean":"churn_rate"}).to_csv(output/"tenure_cohorts.csv")
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=ROOT/"customer_churn_large_dataset.xlsx")
    p.add_argument("--output",type=Path,default=ROOT/"reports");a=p.parse_args()
    print(json.dumps(run(a.data,a.output),indent=2))
