"""Extract leakage-safe, source-specific candidate features from OpenRCA Telecom."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from openrca_temporal_trace_baseline import SOURCE_BY_LEVEL, metric_features, metric_scores, trace_scores


CATEGORIES = {
    "cpu": r"cpu|processor|load|container_cpu",
    "network": r"network|send|recv|packet|queue|ping|tnsping",
    "memory": r"mem|memory|swap|container_mem",
    "connection": r"session|connect|login|client|proc_used",
    "db_state": r"on_off|hang|row_lock|dbtime",
    "disk": r"disk|tbs|physical_read|logic_read|redo",
}


def window_global_app(path: Path, windows):
    frame=pd.read_csv(path)
    frame["time"]=pd.to_datetime(frame.startTime,unit="ms")+pd.Timedelta(hours=8)
    for col in ["avg_time","num","succee_num","succee_rate"]: frame[col]=pd.to_numeric(frame[col],errors="coerce").fillna(0)
    rows=[]
    for start,end in windows:
        w=frame[frame.time.between(start,end,inclusive="left")]
        calls=float(w.num.sum()) if len(w) else 0.0
        rows.append({"app_calls":calls,"app_latency":float(np.average(w.avg_time,weights=w.num+1e-9)) if len(w) else 0.0,"app_failure_rate":float(1-w.succee_num.sum()/max(calls,1)) if len(w) else 0.0,"app_services":float(w.serviceName.nunique()) if len(w) else 0.0})
    return rows


def window_global_middleware(path: Path, windows):
    frame=metric_features(path)
    rows=[]
    for start,end in windows:
        w=frame[frame.time.between(start,end,inclusive="left")]
        rows.append({"middleware_anomaly":float(w.level_score.max()) if len(w) else 0.0,"middleware_change":float(w.change_score.max()) if len(w) else 0.0,"middleware_entities":float(w.cmdb_id.nunique()) if len(w) else 0.0,"middleware_conflicts":float((w.groupby("name").value_num.nunique()>1).sum()) if len(w) else 0.0})
    return rows


def candidate_metric_features(frame, start, end, candidate):
    w=frame[frame.time.between(start,end,inclusive="left") & (frame.cmdb_id.astype(str)==str(candidate))]
    out={"metric_anomaly":0.0,"metric_change":0.0,"metric_observations":0.0,"metric_names":0.0}
    if len(w):
        out.update(metric_anomaly=float(w.level_score.max()),metric_change=float(w.change_score.max()),metric_observations=float(len(w)),metric_names=float(w.name.nunique()))
    names=w.name.astype(str) if len(w) else pd.Series(dtype=str)
    for category,pattern in CATEGORIES.items():
        c=w[names.str.contains(pattern,case=False,regex=True,na=False)] if len(w) else w
        out[f"metric_{category}_anomaly"]=float(c.level_score.max()) if len(c) else 0.0
        out[f"metric_{category}_change"]=float(c.change_score.max()) if len(c) else 0.0
    return out


def run(root: Path, output: Path):
    telecom=root/"Telecom"; records=pd.read_csv(telecom/"record.csv"); records["root_time"]=pd.to_datetime(records.datetime); records["case_id"]=records.index
    rows=[]
    for date,group in records.groupby(records.root_time.dt.strftime("%Y-%m-%d"),sort=True):
        group=group.reset_index(drop=True); date_dir=telecom/"telemetry"/date.replace("-","_")
        windows=[(r.root_time.floor("30min"),r.root_time.floor("30min")+pd.Timedelta(minutes=30)) for r in group.itertuples()]
        print(f"extracting {date} n={len(group)}",flush=True)
        traces=trace_scores(date_dir/"trace"/"trace_span.csv",windows)
        metric_cache={level:metric_features(date_dir/"metric"/filename) for level,filename in SOURCE_BY_LEVEL.items()}
        app=window_global_app(date_dir/"metric"/"metric_app.csv",windows)
        middleware=window_global_middleware(date_dir/"metric"/"metric_middleware.csv",windows)
        for i,incident in enumerate(group.itertuples()):
            start,end=windows[i]; metric=metric_cache[incident.level]
            candidates=sorted(set(metric[metric.time.between(start,end,inclusive="left")].cmdb_id.astype(str))|{str(incident.component)})
            trace=traces[i]
            metric_rank=metric_scores(metric,start,end)
            for candidate in candidates:
                tf=trace.loc[candidate] if not trace.empty and candidate in trace.index else pd.Series(dtype=float)
                row={"case_id":int(incident.case_id),"date":date,"level":incident.level,"reason":incident.reason,"candidate":candidate,"label":int(candidate==incident.component),"metric_rank_score":float(metric_rank.get(candidate,0.0)),"trace_score":float(tf.get("score",0.0)),"trace_calls":float(tf.get("calls",0.0)),"trace_latency":float(tf.get("mean_latency",0.0)),"trace_max_latency":float(tf.get("max_latency",0.0)),"trace_failure_rate":float(tf.get("failure_rate",0.0)),**candidate_metric_features(metric,start,end,candidate),**app[i],**middleware[i]}
                rows.append(row)
    result=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).fillna(0); output.parent.mkdir(parents=True,exist_ok=True); result.to_csv(output,index=False)
    print(f"wrote {output} rows={len(result)} cases={result.case_id.nunique()} positives={int(result.label.sum())}")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,default=Path("openrca/dataset/openrca-telecom-full")); p.add_argument("--output",type=Path,default=Path("artifacts/openrca_candidate_features.csv")); a=p.parse_args(); run(a.dataset_root,a.output)
