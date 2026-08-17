"""Extract and evaluate explicit temporal versions from raw OpenRCA telemetry."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from openrca_temporal_trace_baseline import SOURCE_BY_LEVEL, metric_features, metric_scores, trace_scores
from openrca_candidate_features import candidate_metric_features, window_global_app, window_global_middleware
from openrca_action_model import Logistic, METRIC, TRACE, APP, MIDDLEWARE, add_prior, feature_matrix

VERSIONS=("previous_30m","current_10m","current_30m","history_60m")


def version_windows(root_time):
    start=root_time.floor("30min");end=start+pd.Timedelta(minutes=30)
    return {"previous_30m":(start-pd.Timedelta(minutes=30),start),"current_10m":(start,start+pd.Timedelta(minutes=10)),"current_30m":(start,end),"history_60m":(start-pd.Timedelta(minutes=30),end)}


def extract(root,output):
    telecom=root/"Telecom";records=pd.read_csv(telecom/"record.csv");records["root_time"]=pd.to_datetime(records.datetime);records["case_id"]=records.index;rows=[]
    for date,group in records.groupby(records.root_time.dt.strftime("%Y-%m-%d"),sort=True):
        group=group.reset_index(drop=True);date_dir=telecom/"telemetry"/date.replace("-","_");flat=[];keys=[]
        for incident in group.itertuples():
            for version,window in version_windows(incident.root_time).items():keys.append((incident.case_id,version));flat.append(window)
        print(f"version extraction {date} windows={len(flat)}",flush=True)
        traces=trace_scores(date_dir/"trace"/"trace_span.csv",flat);cache={level:metric_features(date_dir/"metric"/f) for level,f in SOURCE_BY_LEVEL.items()};apps=window_global_app(date_dir/"metric"/"metric_app.csv",flat);mids=window_global_middleware(date_dir/"metric"/"metric_middleware.csv",flat)
        incident_by_id={int(r.case_id):r for r in group.itertuples()}
        for i,(case_id,version) in enumerate(keys):
            incident=incident_by_id[int(case_id)];start,end=flat[i];metric=cache[incident.level];trace=traces[i];rank=metric_scores(metric,start,end);candidates=sorted(set(metric[metric.time.between(start,end,inclusive="left")].cmdb_id.astype(str))|{str(incident.component)})
            for candidate in candidates:
                tf=trace.loc[candidate] if not trace.empty and candidate in trace.index else pd.Series(dtype=float)
                rows.append({"case_id":int(case_id),"date":date,"version":version,"level":incident.level,"reason":incident.reason,"candidate":candidate,"label":int(candidate==incident.component),"metric_rank_score":float(rank.get(candidate,0)),"trace_score":float(tf.get("score",0)),"trace_calls":float(tf.get("calls",0)),"trace_latency":float(tf.get("mean_latency",0)),"trace_max_latency":float(tf.get("max_latency",0)),"trace_failure_rate":float(tf.get("failure_rate",0)),**candidate_metric_features(metric,start,end,candidate),**apps[i],**mids[i]})
    result=pd.DataFrame(rows).replace([np.inf,-np.inf],np.nan).fillna(0);output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False);print(f"wrote {output} rows={len(result)} cases={result.case_id.nunique()}")


def evaluate(data,output):
    raw=pd.read_csv(data);dates=sorted(raw.date.unique());columns=["prior_frequency","candidate_seen"]+METRIC+TRACE+APP+MIDDLEWARE;pred=[]
    for date in dates:
        train_base=raw[(raw.date!=date)&(raw.version=="current_30m")];test=raw[raw.date==date]
        train=add_prior(train_base,train_base);test=add_prior(train_base,test);model=Logistic(l2=2,steps=4000,rate=.03).fit(feature_matrix(train,columns),train.label.to_numpy());test=test.copy();test["score"]=model.predict(feature_matrix(test,columns));test=test.sort_values(["case_id","version","score","candidate"],ascending=[True,True,False,True]);test["rank"]=test.groupby(["case_id","version"]).cumcount()+1;pred.append(test[["case_id","date","version","candidate","label","score","rank"]])
    pred=pd.concat(pred,ignore_index=True);hits=pred[pred["rank"]<=3].groupby(["case_id","version"]).label.max().unstack(fill_value=0)
    version_results={v:float(hits[v].mean()) for v in VERSIONS}
    # Learned version selector: training-date empirical success by root level and
    # reason is unavailable at inference, so selection uses observable level only.
    meta=raw[["case_id","date","level"]].drop_duplicates();hits=hits.reset_index().merge(meta,on="case_id")
    selected=[]
    for date in dates:
        train=hits[hits.date!=date];test=hits[hits.date==date]
        utility=train.groupby("level")[list(VERSIONS)].mean()
        for row in test.itertuples():
            scores=utility.loc[row.level] if row.level in utility.index else train[list(VERSIONS)].mean();version=max(VERSIONS,key=lambda v:(scores[v]-.02*{"previous_30m":1,"current_10m":.5,"current_30m":1,"history_60m":2}[v],v));selected.append({"case_id":row.case_id,"date":date,"level":row.level,"selected_version":version,"hit":int(getattr(row,version)),"cost":{"previous_30m":1,"current_10m":.5,"current_30m":1,"history_60m":2}[version]})
    selected=pd.DataFrame(selected);oracle=hits[list(VERSIONS)].max(axis=1)
    payload={"dataset":"OpenRCA raw telemetry temporal versions","versions":{"previous_30m":"stale prior interval","current_10m":"partial current interval","current_30m":"complete current interval","history_60m":"previous plus complete current history"},"split":"leave-one-date-out action model and version selector","top3_by_version":version_results,"learned_selector":{"top3":float(selected.hit.mean()),"mean_cost":float(selected.cost.mean()),"selection_counts":selected.selected_version.value_counts().to_dict()},"always_all_versions":{"top3_oracle_union":float(oracle.mean()),"cost":4.5},"oracle_version":{"top3":float(oracle.mean()),"mean_minimum_cost":float(np.mean([min([c for v,c in {"previous_30m":1,"current_10m":.5,"current_30m":1,"history_60m":2}.items() if row[v]],default=4.5) for _,row in hits.iterrows()]))},"metrics":{"temporal_coverage_recall":version_results,"stale_action_rate":float(1-version_results["previous_30m"]),"partial_window_failure_rate":float(1-version_results["current_10m"])} }
    prediction_path=output.with_name(output.stem+"_predictions.csv")
    output.write_text(json.dumps(payload,indent=2)+"\n");pred.to_csv(prediction_path,index=False);print(json.dumps(payload,indent=2));print(f"wrote {output} and {prediction_path}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset-root",type=Path,default=Path("openrca/dataset/openrca-telecom-full"));p.add_argument("--features",type=Path,default=Path("artifacts/openrca_temporal_version_features.csv"));p.add_argument("--output",type=Path,default=Path("artifacts/openrca_temporal_versions.json"));p.add_argument("--extract",action="store_true");a=p.parse_args();
    if a.extract:extract(a.dataset_root,a.features)
    evaluate(a.features,a.output)
