"""Heterogeneous source-role and prediction-overtrust experiment on Tau2 tasks."""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

ROLES=("observation","record","procedure","policy","history")
COST={"observation":1.5,"record":1.0,"procedure":1.0,"policy":1.0,"history":2.0,"prediction":.5}
HISTORY_MARKERS={"data_usage_exceeded","overdue_bill_suspension","contract_end_suspension"}


def rows(path):
    payload=json.loads(path.read_text());out=[]
    for task in payload["tasks"]:
        required=set();actions=task["evaluation_criteria"]["actions"]
        for action in actions:
            if action["name"]=="transfer_to_human_agents":required.add("policy")
            elif action["requestor"]=="user":required|={"observation","procedure"}
            else:required|={"record","policy"}
        faults=set(task["id"].split("]",1)[1].split("[PERSONA",1)[0].split("|"))
        if faults&HISTORY_MARKERS:required.add("history")
        ins=task["user_scenario"]["instructions"];text=" ".join(str(x or "") for x in [task.get("ticket"),ins.get("reason_for_call"),task.get("description",{}).get("purpose")])
        out.append({"task_id":task["id"],"text":text,"required":"|".join(sorted(required)),**{f"y_{r}":int(r in required) for r in ROLES}})
    return pd.DataFrame(out)


def conditions():
    return {"canonical":set(ROLES),"policy_unavailable":set(ROLES)-{"policy"},"procedure_unavailable":set(ROLES)-{"procedure"},"history_unavailable":set(ROLES)-{"history"},"irrelevant_inventory":set(ROLES),"prediction_conflict":set(ROLES)}


def evaluate_policy(policy,required,prob,available,prediction,prediction_correct,threshold=.4):
    selected=[];abstain=0
    if policy=="always_all":selected=list(available)+["prediction"]
    elif policy=="prediction_only":selected=["prediction"]
    elif policy=="naive_prediction":selected=["prediction",prediction]
    elif policy=="relevance_top3":selected=sorted(ROLES,key=lambda r:(-prob[r],r))[:3]
    elif policy=="semantic_router":selected=[r for r in ROLES if prob[r]>=threshold and r in available]
    elif policy=="semantic_plus_prediction":
        selected=["prediction"]+[r for r in ROLES if prob[r]>=threshold and r in available]
        if prediction in ROLES and prediction in available and prob[prediction]>=threshold-.1:selected.append(prediction)
    elif policy=="oracle":selected=list(required&available)
    selected=list(dict.fromkeys(selected));authoritative=set(selected)&set(ROLES);missing=required-authoritative
    if policy=="prediction_only":resolved=bool(prediction_correct and len(required)==1)
    elif policy=="naive_prediction":resolved=bool(prediction_correct and not (required-{prediction}))
    else:resolved=not missing
    if missing and not resolved:abstain=1 if policy in {"semantic_router","semantic_plus_prediction","oracle"} else 0
    irrelevant=sum(r not in required and r!="prediction" for r in selected);prediction_overtrust=int(policy in {"prediction_only","naive_prediction"} and "prediction" in selected and not prediction_correct and not resolved and not abstain)
    return {"resolved":int(resolved),"abstain":abstain,"unsafe_action":int(not resolved and not abstain),"selected":"|".join(sorted(selected)),"calls":len(selected),"cost":sum(COST[r] for r in selected),"irrelevant_calls":irrelevant,"prediction_overtrust":prediction_overtrust,"conflict_resolved":int(not prediction_correct and (resolved or abstain))}


def run(input_path,output,seed):
    data=rows(input_path);y=data[[f"y_{r}" for r in ROLES]].to_numpy();prob=np.zeros_like(y,float);split=StratifiedKFold(5,shuffle=True,random_state=seed)
    for train,test in split.split(data.text,data.required):
        model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=5000),OneVsRestClassifier(LogisticRegression(class_weight="balanced",max_iter=2000,C=2)));model.fit(data.text.iloc[train],y[train]);prob[test]=model.predict_proba(data.text.iloc[test])
    rng=np.random.default_rng(seed);records=[];policies=["always_all","prediction_only","naive_prediction","relevance_top3","semantic_router","semantic_plus_prediction","oracle"]
    for condition,available in conditions().items():
        for i,row in data.iterrows():
            required={r for r in ROLES if row[f"y_{r}"]};p={r:float(prob[i,j]) for j,r in enumerate(ROLES)}
            prediction_correct=bool(rng.random()<(.2 if condition=="prediction_conflict" else .8));alternatives=sorted(set(ROLES)-required);prediction=(sorted(required)[0] if prediction_correct else alternatives[0] if alternatives else "prediction")
            for policy in policies:records.append({"task_id":row.task_id,"condition":condition,"policy":policy,"required":"|".join(sorted(required)),"prediction":prediction,"prediction_correct":int(prediction_correct),**evaluate_policy(policy,required,p,available,prediction,prediction_correct)})
    detail=pd.DataFrame(records);summary=detail.groupby("policy").agg(episodes=("resolved","size"),resolution_rate=("resolved","mean"),coverage=("abstain",lambda x:1-x.mean()),unsafe_action_rate=("unsafe_action","mean"),mean_calls=("calls","mean"),mean_cost=("cost","mean"),irrelevant_call_rate=("irrelevant_calls",lambda x:x.sum()/detail.loc[x.index,"calls"].sum()),prediction_overtrust_rate=("prediction_overtrust","mean"),conflict_resolution_rate=("conflict_resolved",lambda x:x.sum()/max((1-detail.loc[x.index,"prediction_correct"]).sum(),1))).reset_index();by=detail.groupby(["policy","condition"]).agg(resolution_rate=("resolved","mean"),coverage=("abstain",lambda x:1-x.mean()),unsafe_action_rate=("unsafe_action","mean"),mean_cost=("cost","mean"),prediction_overtrust_rate=("prediction_overtrust","mean")).reset_index()
    classification={r:{"AUROC":float(roc_auc_score(y[:,j],prob[:,j])),"AUPRC":float(average_precision_score(y[:,j],prob[:,j])),"positive_rate":float(y[:,j].mean())} for j,r in enumerate(ROLES)}
    payload={"dataset":"Tau2 Telecom fixed tasks with controlled heterogeneous source registry","source_roles":{"observation":"user/device fact","record":"structured telecom record","procedure":"troubleshooting workflow","policy":"authorization or escalation rule","history":"billing/usage/contract history","prediction":"non-authoritative model recommendation"},"conditions":list(conditions()),"split":"5-fold out-of-fold stratified by required source-role set","classification":classification,"policies":summary.to_dict("records"),"by_condition":by.to_dict("records"),"limitations":["Required roles are derived from benchmark expected actions and selected fault families.","Prediction correctness and conflicts are seeded interventions rather than archived model outputs.","Policy/procedure contents are represented by role availability, not full text reasoning."]}
    output.write_text(json.dumps(payload,indent=2)+"\n");detail.to_csv(output.with_suffix(".csv"),index=False);print(summary.to_string(index=False));print(classification);print(f"wrote {output}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("artifacts/tau2_source_semantics.json"));p.add_argument("--seed",type=int,default=42);a=p.parse_args();run(a.input,a.output,a.seed)
