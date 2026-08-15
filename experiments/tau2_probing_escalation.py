"""Tau2 Telecom probing, source-routing, and escalation experiment.

The hidden task and expected outcome are fixed.  Interventions change only
which acquisition channels are available or their interaction cost.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

ROLES=("user","system","human")
BASE_COST={"user":1.5,"system":1.0,"human":10.0}


def task_rows(path):
    payload=json.loads(path.read_text());rows=[]
    for task in payload["tasks"]:
        required=set()
        for action in task["evaluation_criteria"]["actions"]:
            if action["name"]=="transfer_to_human_agents":required.add("human")
            elif action["requestor"]=="user":required.add("user")
            else:required.add("system")
        instructions=task["user_scenario"]["instructions"]
        text=" ".join(str(x or "") for x in [task.get("ticket"),instructions.get("reason_for_call"),task.get("description",{}).get("purpose")])
        rows.append({"task_id":task["id"],"text":text,"required":"|".join(sorted(required)),**{f"y_{r}":int(r in required) for r in ROLES}})
    return pd.DataFrame(rows)


def conditions():
    return {
        "canonical":({r:1 for r in ROLES},dict(BASE_COST),1.0),
        "user_unavailable":({"user":0,"system":1,"human":1},dict(BASE_COST),1.0),
        "system_unavailable":({"user":1,"system":0,"human":1},dict(BASE_COST),1.0),
        "user_delayed":({r:1 for r in ROLES},{**BASE_COST,"user":4.0},1.0),
        "system_expensive":({r:1 for r in ROLES},{**BASE_COST,"system":5.0},1.0),
        "ambiguous_user":({r:1 for r in ROLES},dict(BASE_COST),.5),
    }


def execute(required,selected,availability,user_success):
    selected=list(selected);observed=set();turns=0;cost=0.;user_calls=0
    for role in selected:
        if not availability[role]:continue
        turns+=1;cost+=CURRENT_COST[role];user_calls+=role=="user"
        if role=="user" and not user_success:continue
        observed.add(role)
    missing=required-observed
    # Human escalation can resolve unavailable system/user requirements, but
    # only when it was explicitly selected.
    if "human" in observed:missing.clear()
    resolved=not missing
    useful=sum(role in required or role=="human" and bool(required-{"human"}) for role in observed)
    escalation=int("human" in selected)
    escalation_needed=int("human" in required or any(not availability[r] for r in required))
    return {"resolved":int(resolved),"safe_action":int(resolved),"calls":turns,"cost":cost,"turns":turns,"user_burden":user_calls,"probe_precision":useful/max(len(observed),1),"escalation":escalation,"escalation_correct":int(escalation and escalation_needed),"unresolved":int(not resolved)}


def choose(policy,prob,availability,cost,threshold=.4):
    ranked=sorted(ROLES,key=lambda r:(-(prob[r]/max(cost[r],1e-9)),r))
    if policy=="none":return []
    if policy=="always_all":return [r for r in ROLES if availability[r]]
    if policy.startswith("fixed_"):return [policy.split("_",1)[1]]
    if policy=="learned_top1":
        role=ranked[0];return [role] if availability[role] else ["human"]
    if policy=="learned_top2":
        selected=[]
        for role in ranked[:2]:selected.append(role if availability[role] else "human")
        return list(dict.fromkeys(selected))
    if policy.startswith("gate_"):
        t=float(policy.split("_")[1]);selected=[]
        for role in ranked:
            if prob[role]>=t:selected.append(role if availability[role] else "human")
        return list(dict.fromkeys(selected))
    raise ValueError(policy)


def run(input_path,output,seed):
    global CURRENT_COST
    data=task_rows(input_path);labels=data[[f"y_{r}" for r in ROLES]].to_numpy();strata=data.required
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed);prob=np.zeros_like(labels,dtype=float)
    for train,test in splitter.split(data.text,strata):
        model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=5000),OneVsRestClassifier(LogisticRegression(class_weight="balanced",max_iter=2000,C=2)))
        model.fit(data.text.iloc[train],labels[train]);prob[test]=model.predict_proba(data.text.iloc[test])
    policies=["none","always_all","fixed_user","fixed_system","fixed_human","learned_top1","learned_top2","gate_0.3","gate_0.4","gate_0.5","gate_0.6","oracle"]
    rng=np.random.default_rng(seed);rows=[]
    for condition,(availability,cost,user_reliability) in conditions().items():
        CURRENT_COST=cost
        for i,row in data.iterrows():
            user_success=bool(rng.random()<=user_reliability)
            required={r for r in ROLES if row[f"y_{r}"]};p={r:float(prob[i,j]) for j,r in enumerate(ROLES)}
            for policy in policies:
                if policy=="oracle":
                    selected=[]
                    for role in sorted(required):
                        selected.append(role if availability[role] else "human")
                    if "user" in selected and not user_success and availability["human"]:selected.append("human")
                    selected=list(dict.fromkeys(selected))
                else:selected=choose(policy,p,availability,cost)
                if (policy.startswith("learned") or policy.startswith("gate_")) and "user" in selected and not user_success and availability["human"]:
                    selected=list(dict.fromkeys(selected+["human"]))
                result=execute(required,selected,availability,user_success)
                rows.append({"task_id":row.task_id,"condition":condition,"policy":policy,"required":"|".join(sorted(required)),"selected":"|".join(selected),**result})
    detail=pd.DataFrame(rows);summary=detail.groupby("policy").agg(episodes=("resolved","size"),resolution_rate=("resolved","mean"),unsafe_or_unresolved_rate=("unresolved","mean"),mean_calls=("calls","mean"),mean_cost=("cost","mean"),mean_turns=("turns","mean"),user_burden=("user_burden","mean"),probe_precision=("probe_precision","mean"),escalation_rate=("escalation","mean"),escalation_precision=("escalation_correct",lambda x:x.sum()/max(detail.loc[x.index,"escalation"].sum(),1))).reset_index()
    by_condition=detail.groupby(["policy","condition"]).agg(resolution_rate=("resolved","mean"),mean_cost=("cost","mean"),probe_precision=("probe_precision","mean"),escalation_rate=("escalation","mean")).reset_index()
    classification={r:{"AUROC":float(roc_auc_score(labels[:,j],prob[:,j])),"AUPRC":float(average_precision_score(labels[:,j],prob[:,j])),"positive_rate":float(labels[:,j].mean())} for j,r in enumerate(ROLES)}
    payload={"dataset":"Tau2 Telecom tasks at pinned benchmark manifest revision","hidden_state":"original task and expected action roles","observation":"ticket, reason for call, and public task purpose","roles":{"user":"ask user/device to perform or report evidence","system":"query or act on structured telecom system","human":"support-agent escalation"},"conditions":list(conditions()),"split":"5-fold stratified out-of-fold by required role set","classification":classification,"policies":summary.to_dict("records"),"by_condition":by_condition.to_dict("records"),"limitations":["Role requirements are derived from expected action requestors, not free-form human answers.","Ambiguous user responses are simulated with a seeded 0.5 response-success probability.","Human escalation is modeled as resolving unavailable primary channels."]}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(payload,indent=2)+"\n");detail.to_csv(output.with_suffix(".csv"),index=False);print(summary.to_string(index=False));print(classification);print(f"wrote {output}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("artifacts/tau2_probing_escalation.json"));p.add_argument("--seed",type=int,default=42);a=p.parse_args();run(a.input,a.output,a.seed)
