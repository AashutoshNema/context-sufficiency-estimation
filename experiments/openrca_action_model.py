"""Leakage-safe candidate-ranking ablations for OpenRCA Telecom."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from openrca_context_control import Logistic


METRIC=[c for c in [
    "metric_rank_score","metric_anomaly","metric_change","metric_observations","metric_names",
    "metric_cpu_anomaly","metric_cpu_change","metric_network_anomaly","metric_network_change",
    "metric_memory_anomaly","metric_memory_change","metric_connection_anomaly","metric_connection_change",
    "metric_db_state_anomaly","metric_db_state_change","metric_disk_anomaly","metric_disk_change",
]]
TRACE=["trace_score","trace_calls","trace_latency","trace_max_latency","trace_failure_rate"]
APP=["app_calls","app_latency","app_failure_rate","app_services"]
MIDDLEWARE=["middleware_anomaly","middleware_change","middleware_entities","middleware_conflicts"]


def add_prior(train, target):
    positives=train[train.label==1]
    level_total=positives.groupby("level").size().to_dict()
    counts=positives.groupby(["level","candidate"]).size().to_dict()
    out=target.copy()
    out["prior_frequency"]=[counts.get((r.level,r.candidate),0)/max(level_total.get(r.level,0),1) for r in out.itertuples()]
    out["candidate_seen"]=[float((r.level,r.candidate) in counts) for r in out.itertuples()]
    return out


def feature_matrix(frame, columns):
    return np.column_stack([np.ones(len(frame))]+[pd.to_numeric(frame[c],errors="coerce").fillna(0).to_numpy(float) for c in columns])


def metrics(predictions):
    frame=pd.DataFrame(predictions)
    return {"cases":int(frame.case_id.nunique()),"top1":float(frame.groupby("case_id").head(1).label.mean()),"top3":float(frame.groupby("case_id").head(3).groupby("case_id").label.max().mean()),"mrr":float((1/frame[frame.label==1]["rank"]).mean())}


def run(data: Path, output: Path):
    raw=pd.read_csv(data); dates=sorted(raw.date.unique())
    sets={"prior":["prior_frequency","candidate_seen"],"metric":["prior_frequency","candidate_seen"]+METRIC,"trace":["prior_frequency","candidate_seen"]+TRACE,"metric_trace":["prior_frequency","candidate_seen"]+METRIC+TRACE,"all":["prior_frequency","candidate_seen"]+METRIC+TRACE+APP+MIDDLEWARE}
    predictions={name:[] for name in sets}; fold_rows=[]
    for date in dates:
        train_raw=raw[raw.date!=date]; test_raw=raw[raw.date==date]
        train=add_prior(train_raw,train_raw); test=add_prior(train_raw,test_raw)
        for name,columns in sets.items():
            model=Logistic(l2=2.0,steps=4000,rate=.03).fit(feature_matrix(train,columns),train.label.to_numpy())
            scored=test.copy(); scored["score"]=model.predict(feature_matrix(test,columns))
            scored=scored.sort_values(["case_id","score","candidate"],ascending=[True,False,True])
            scored["rank"]=scored.groupby("case_id").cumcount()+1
            for row in scored.itertuples(): predictions[name].append({"case_id":int(row.case_id),"date":row.date,"candidate":row.candidate,"label":int(row.label),"score":float(row.score),"rank":int(row.rank)})
        fold_rows.append({"test_date":date,"cases":int(test_raw.case_id.nunique()),"candidates":len(test_raw)})
    results={name:metrics(rows) for name,rows in predictions.items()}
    payload={"dataset":"OpenRCA Telecom raw source-specific candidate features","split":"leave-one-telemetry-date-out","folds":fold_rows,"feature_sets":{"prior":["historical component frequency"],"metric":METRIC,"trace":TRACE,"metric_trace":METRIC+TRACE,"all":METRIC+TRACE+APP+MIDDLEWARE},"results":results,"gate":{"requirement":"always-fetch top-3 must exceed prior-only top-3 before controller evaluation","passed":bool(results["all"]["top3"]>results["prior"]["top3"])},"notes":["Application and middleware features are incident-global and therefore cannot localize components without a topology mapping or candidate-specific interaction.","No root component, reason, root timestamp, or intervention identity is used as an input feature."]}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+"\n")
    pd.concat([pd.DataFrame(rows).assign(model=name) for name,rows in predictions.items()]).to_csv(output.with_suffix(".csv"),index=False)
    print(pd.DataFrame(results).T.to_string()); print("gate",payload["gate"]); print(f"wrote {output}")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=Path("artifacts/openrca_candidate_features.csv")); p.add_argument("--output",type=Path,default=Path("artifacts/openrca_action_model.json")); a=p.parse_args(); run(a.data,a.output)
