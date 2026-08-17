"""ContextControl-v2 over leakage-safe out-of-fold OpenRCA modality models."""
from __future__ import annotations

import argparse, itertools, json
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import pandas as pd

from openrca_context_control import Logistic, auc, average_precision, ece

SOURCES=("metric","trace","app","middleware","metric_duplicate")
BASE_COST={"metric":1.0,"trace":2.0,"app":1.5,"middleware":1.5,"metric_duplicate":.8}
BASE_LATENCY={"metric":1.0,"trace":3.0,"app":2.0,"middleware":2.0,"metric_duplicate":1.0}


@dataclass(frozen=True)
class Source:
    name:str; scores:dict[str,float]; cost:float; latency:float; age:float=0.; entity_match:int=1; available:int=1; provenance:str=""


@dataclass
class Episode:
    case_id:int; date:str; level:str; truth:str; candidates:tuple[str,...]; model_scores:dict[str,dict[str,float]]; sources:dict[str,Source]; condition:str="canonical"


def load_cases(pred_path, feature_path):
    pred=pd.read_csv(pred_path); meta=pd.read_csv(feature_path)[["case_id","date","level","candidate","label"]].drop_duplicates()
    cases=[]
    for case_id,group in meta.groupby("case_id",sort=True):
        candidates=tuple(sorted(group.candidate.astype(str))); truth=str(group[group.label==1].iloc[0].candidate); row=group.iloc[0]
        model_scores={}
        for model in pred.model.unique():
            m=pred[(pred.case_id==case_id)&(pred.model==model)].set_index("candidate").score.to_dict(); model_scores[model]={c:float(m.get(c,0)) for c in candidates}
        sources={
            "metric":Source("metric",model_scores["metric"],1,1,provenance="metric"),
            "trace":Source("trace",model_scores["trace"],2,3,provenance="trace"),
            "app":Source("app",{},1.5,2,provenance="app"),
            "middleware":Source("middleware",{},1.5,2,provenance="middleware"),
            "metric_duplicate":Source("metric_duplicate",model_scores["metric"],.8,1,provenance="metric"),
        }
        cases.append(Episode(int(case_id),str(row.date),str(row.level),truth,candidates,model_scores,sources))
    return cases


def donor(cases,episode):
    pool=[x for x in cases if x.level==episode.level and x.case_id!=episode.case_id]
    return pool[episode.case_id%len(pool)] if pool else episode


def interventions(cases):
    out=[]
    for e in cases:
        out.append(e)
        s=dict(e.sources); s["trace"]=replace(s["trace"],available=0); out.append(replace(e,sources=s,condition="trace_unavailable"))
        s=dict(e.sources); s["trace"]=replace(s["trace"],cost=8,latency=12); out.append(replace(e,sources=s,condition="high_trace_cost"))
        d=donor(cases,e); s=dict(e.sources); s["metric"]=replace(s["metric"],scores=d.model_scores["metric"],age=1440,provenance="metric-stale"); s["metric_duplicate"]=replace(s["metric_duplicate"],scores=d.model_scores["metric"],age=1440,provenance="metric-stale"); out.append(replace(e,sources=s,condition="stale_metric"))
        s=dict(e.sources); s["metric"]=replace(s["metric"],scores=d.model_scores["metric"],entity_match=0,provenance="metric-other-entity"); s["metric_duplicate"]=replace(s["metric_duplicate"],scores=d.model_scores["metric"],entity_match=0,provenance="metric-other-entity"); out.append(replace(e,sources=s,condition="wrong_entity_metric"))
        s=dict(e.sources); s={k:replace(v,age=60) if k in {"metric","trace","metric_duplicate"} else v for k,v in s.items()}; out.append(replace(e,sources=s,condition="temporal_gap"))
    return out


def normalize(scores,candidates):
    x=np.array([scores.get(c,0) for c in candidates],float)
    if np.ptp(x)==0:return np.zeros(len(x))
    return (x-x.min())/np.ptp(x)


def scores_for(e,acquired):
    metric_names=[n for n in ("metric","metric_duplicate") if n in acquired and e.sources[n].available]
    trace_ok="trace" in acquired and e.sources["trace"].available
    app_mid={"app","middleware"}.issubset(acquired)
    canonical_metric=any(e.sources[n].age==0 and e.sources[n].entity_match and e.sources[n].provenance=="metric" for n in metric_names)
    canonical_trace=trace_ok and e.sources["trace"].age==0 and e.sources["trace"].entity_match
    if canonical_metric and canonical_trace and app_mid:return e.model_scores["all"]
    if canonical_metric and canonical_trace:return e.model_scores["metric_trace"]
    arrays=[]
    for name in metric_names: arrays.append(normalize(e.sources[name].scores,e.candidates))
    if trace_ok: arrays.append(normalize(e.sources["trace"].scores,e.candidates))
    if not arrays:return e.model_scores["prior"]
    x=np.mean(arrays,axis=0); return dict(zip(e.candidates,x))


def shortlist(e,acquired):
    scores=scores_for(e,acquired); return tuple(sorted(e.candidates,key=lambda c:(-scores.get(c,0),c))[:3])


def state_features(e,acquired):
    scores=scores_for(e,acquired); ordered=sorted([scores.get(c,0) for c in e.candidates],reverse=True); top=ordered+[0,0]
    src=[e.sources[n] for n in SOURCES if n in acquired and e.sources[n].available]
    return np.array([1,float(e.level=="node"),float(e.level=="pod"),float(e.level=="service"),*[float(n in acquired and e.sources[n].available) for n in SOURCES],len(src),sum(x.cost for x in src),sum(x.latency for x in src),max([x.age for x in src]or[0]),min([x.entity_match for x in src]or[1]),len({x.provenance for x in src}),float(len(src)>len({x.provenance for x in src})),top[0],top[0]-top[1],top[2]-top[3] if len(top)>3 else 0],float)


def registry_features(e,acquired,name):
    s=e.sources[name]; return np.concatenate([state_features(e,acquired),np.array([*[float(name==n) for n in SOURCES],s.cost,s.latency,s.age,s.entity_match,s.available,float(s.provenance in {e.sources[n].provenance for n in acquired})])])


def all_states(episodes):
    rows=[]
    for e in episodes:
        available=[n for n in SOURCES if e.sources[n].available]
        for size in range(len(available)+1):
            for subset in itertools.combinations(available,size):
                a=set(subset); rows.append((e,a,state_features(e,a),int(e.truth in shortlist(e,a))))
    return rows


def train_ranker(episodes):
    x=[];y=[]
    for e,a,_,_ in all_states(episodes):
        for n in SOURCES:
            if n in a or not e.sources[n].available:continue
            x.append(registry_features(e,a,n)); y.append(int(e.truth in shortlist(e,a|{n})))
    return Logistic(l2=2,steps=3000,rate=.04).fit(np.vstack(x),np.array(y))


def choose_source(ranker,e,a,cost_weight,guarded=False):
    options=[]
    for n in SOURCES:
        if n in a or not e.sources[n].available:continue
        if guarded and (e.sources[n].age>30 or not e.sources[n].entity_match):continue
        p=float(ranker.predict(registry_features(e,a,n))[0]);options.append((p-cost_weight*e.sources[n].cost,p,-e.sources[n].cost,n))
    return max(options)[-1] if options else None


def voi_source(gate,ranker,e,a,cost_weight=.03,guarded=False):
    current=float(gate.predict(state_features(e,a))[0]);options=[]
    for n in SOURCES:
        s=e.sources[n]
        if n in a or not s.available:continue
        if guarded and (s.age>30 or not s.entity_match):continue
        post=float(ranker.predict(registry_features(e,a,n))[0]);value=post-current-cost_weight*s.cost
        options.append((value,post,-s.cost,n))
    return max(options) if options else None


def threshold(model,states):
    y=np.array([r[3] for r in states]);p=model.predict(np.vstack([r[2] for r in states]));best=(-9,.5)
    for t in np.linspace(.1,.9,17):
        act=p>=t;utility=np.mean(y*act-1.5*(1-y)*act+.1*(~act))
        if utility>best[0]:best=(utility,float(t))
    return best[1]


def deployment_threshold(gate,ranker,episodes,rng,target_accuracy=.85):
    """Calibrate on trajectories produced by the deployment acquisition policy."""
    candidates=[]
    for value in range(30,91,10):
        rows=pd.DataFrame(policy(f"gate_t{value}",episodes,gate,ranker,value/100,rng))
        attempted=rows[rows.abstain==0]
        coverage=len(attempted)/max(len(rows),1)
        accuracy=float(attempted.correct.mean()) if len(attempted) else 0.0
        if len(attempted)>=10 and accuracy>=target_accuracy:
            candidates.append((coverage,-rows.cost.mean(),value/100))
    return max(candidates)[-1] if candidates else .90


def policy(name,episodes,gate,ranker,t,rng,cost_weight=.03):
    rows=[]
    for e in episodes:
        available=[n for n in SOURCES if e.sources[n].available];a=set();abstain=0
        if name=="always_fetch":a=set(available)
        elif name=="metric_only":a={"metric"}
        elif name=="trace_only":a={"trace"} if e.sources["trace"].available else set()
        elif name=="fixed_relevant":a={n for n in ("metric","trace") if e.sources[n].available}
        elif name=="random_top1":a={rng.choice(available)} if available else set()
        elif name in {"source_ranker","gate_plus_ranker","guarded_ranker","voi_ranker","guarded_voi"} or name.startswith("ranker_k") or name.startswith("ranker_cost") or name.startswith("gate_t"):
            if name.startswith("ranker_k"):
                limit=int(name.split("k")[-1]); gated=False; gate_threshold=t
            elif name.startswith("ranker_cost"):
                limit=2; gated=False; gate_threshold=t; cost_weight=float(name.split("cost")[-1])/100
            elif name.startswith("gate_t"):
                limit=len(available); gated=True; gate_threshold=float(name.split("t")[-1])/100
            else:
                limit=2 if name in {"source_ranker","guarded_ranker"} else len(available); gated=name=="gate_plus_ranker"; gate_threshold=t
            while len(a)<limit:
                p=float(gate.predict(state_features(e,a))[0])
                if gated and p>=gate_threshold:break
                if name in {"voi_ranker","guarded_voi"}:
                    option=voi_source(gate,ranker,e,a,cost_weight,guarded=name=="guarded_voi")
                    if option is None or option[0]<=0:break
                    n=option[-1]
                else:
                    n=choose_source(ranker,e,a,cost_weight,guarded=name=="guarded_ranker")
                if n is None:break
                a.add(n)
            abstain=int(gated and float(gate.predict(state_features(e,a))[0])<gate_threshold)
        elif name=="oracle":
            options=[]
            for size in range(len(available)+1):
                for ss in itertools.combinations(available,size):
                    aa=set(ss)
                    if e.truth in shortlist(e,aa):options.append((sum(e.sources[n].cost for n in aa),len(aa),aa))
            if options:a=min(options,key=lambda x:(x[0],x[1]))[2]
            else:a=set(available);abstain=1
        would_correct=int(e.truth in shortlist(e,a));correct=int(not abstain and would_correct);src=[e.sources[n] for n in a]
        rows.append({"policy":name,"case_id":e.case_id,"condition":e.condition,"correct":correct,"would_correct":would_correct,"unsafe_action":int(not abstain and not would_correct),"safe_abstain":int(abstain and not would_correct),"missed_safe_action":int(abstain and would_correct),"abstain":abstain,"calls":len(a),"cost":sum(x.cost for x in src),"latency":sum(x.latency for x in src),"selected":"|".join(sorted(a))})
    return rows


def run(predictions,features,output,seed):
    base=load_cases(predictions,features);episodes=interventions(base);dates=sorted({e.date for e in base});rng=np.random.default_rng(seed);rows=[];ys=[];ps=[];thresholds=[];correlation=[]
    names=["no_context","always_fetch","metric_only","trace_only","fixed_relevant","random_top1","source_ranker","guarded_ranker","voi_ranker","guarded_voi","ranker_cost10","ranker_cost20","gate_plus_ranker","ranker_k1","ranker_k2","ranker_k3","ranker_k4","ranker_k5","gate_t30","gate_t40","gate_t50","gate_t60","gate_t70","gate_t80","gate_t90","oracle"]
    for i,date in enumerate(dates):
        val={dates[(i-1)%len(dates)],dates[(i-2)%len(dates)]};train=[e for e in episodes if e.date not in val|{date}];valid=[e for e in episodes if e.date in val];test=[e for e in episodes if e.date==date]
        ts=all_states(train);gate=Logistic(l2=2,steps=3000,rate=.04).fit(np.vstack([r[2] for r in ts]),np.array([r[3] for r in ts]));ranker=train_ranker(train);t=deployment_threshold(gate,ranker,valid,rng);thresholds.append(t)
        test_states=all_states(test);ys.extend(r[3] for r in test_states);ps.extend(gate.predict(np.vstack([r[2] for r in test_states])))
        for e in [item for item in test if item.condition=="canonical"]:
            states={"metric":{"metric"},"metric_duplicate":{"metric","metric_duplicate"},"metric_trace":{"metric","trace"}}
            record={"case_id":e.case_id}
            for state_name,acquired in states.items():
                record[f"p_{state_name}"]=float(gate.predict(state_features(e,acquired))[0])
                record[f"y_{state_name}"]=int(e.truth in shortlist(e,acquired))
            correlation.append(record)
        for name in names:rows.extend(policy(name,test,gate,ranker,t,rng))
    detail=pd.DataFrame(rows);summary=detail.groupby("policy").agg(episodes=("correct","size"),acceptable_rate=("correct","mean"),coverage=("abstain",lambda x:1-x.mean()),unsafe_action_rate=("unsafe_action","mean"),safe_abstention_rate=("safe_abstain","mean"),missed_safe_action_rate=("missed_safe_action","mean"),mean_calls=("calls","mean"),mean_cost=("cost","mean"),mean_latency=("latency","mean")).reset_index();attempted=detail[detail.abstain==0].groupby("policy").correct.mean();summary["accuracy_attempted"]=summary.policy.map(attempted).fillna(0)
    by_condition=detail.groupby(["policy","condition"]).agg(acceptable_rate=("correct","mean"),coverage=("abstain",lambda x:1-x.mean()),mean_cost=("cost","mean")).reset_index();y=np.array(ys);p=np.array(ps)
    corr=pd.DataFrame(correlation)
    routing={}
    for policy_name in ["source_ranker","guarded_ranker","voi_ranker","guarded_voi","ranker_cost10","ranker_cost20","gate_plus_ranker"]:
        routing[policy_name]={}
        for condition in ["canonical","stale_metric","wrong_entity_metric","high_trace_cost","trace_unavailable","temporal_gap"]:
            g=detail[(detail.policy==policy_name)&(detail.condition==condition)]
            routing[policy_name][condition]={"metric_selected_rate":float(g.selected.str.contains("metric",regex=False).mean()),"trace_selected_rate":float(g.selected.str.contains("trace",regex=False).mean()),"acceptable_rate":float(g.correct.mean()),"mean_cost":float(g.cost.mean())}
    payload={"dataset":"OpenRCA Telecom out-of-fold modality action models","action":"top-3 root-component triage","split":"leave-one-date-out controller evaluation; two preceding dates calibrate","conditions":sorted(detail.condition.unique()),"sufficiency":{"AUROC":auc(y,p),"AUPRC":average_precision(y,p),"Brier":float(np.mean((p-y)**2)),"ECE":ece(y,p),"positive_rate":float(y.mean()),"median_threshold":float(np.median(thresholds))},"correlation_audit":{"cases":len(corr),"duplicate_confidence_delta":float((corr.p_metric_duplicate-corr.p_metric).mean()),"duplicate_accuracy_delta":float((corr.y_metric_duplicate-corr.y_metric).mean()),"independent_trace_confidence_delta":float((corr.p_metric_trace-corr.p_metric).mean()),"independent_trace_accuracy_delta":float((corr.y_metric_trace-corr.y_metric).mean())},"routing_audit":routing,"policies":summary.to_dict("records"),"by_condition":by_condition.to_dict("records")}
    output.write_text(json.dumps(payload,indent=2)+"\n");detail.to_csv(output.with_suffix(".csv"),index=False);print(summary.to_string(index=False));print(payload["sufficiency"]);print(f"wrote {output}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--predictions",type=Path,default=Path("artifacts/openrca_action_model.csv"));p.add_argument("--features",type=Path,default=Path("artifacts/openrca_candidate_features.csv"));p.add_argument("--output",type=Path,default=Path("artifacts/openrca_context_control_v2.json"));p.add_argument("--seed",type=int,default=42);a=p.parse_args();run(a.predictions,a.features,a.output,a.seed)
