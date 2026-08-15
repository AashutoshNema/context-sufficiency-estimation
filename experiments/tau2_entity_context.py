"""Explicit multi-asset/entity-scope interventions for Tau2 Telecom."""
from __future__ import annotations

import argparse, json, tomllib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def scalar_values(value):
    if isinstance(value,dict):
        for item in value.values():yield from scalar_values(item)
    elif isinstance(value,list):
        for item in value:yield from scalar_values(item)
    elif value is not None:yield str(value)


def prefix_texts(payload):
    output={}
    for sim in payload["simulations"]:
        prior=[];decision=0
        for message in sim.get("messages") or []:
            if message.get("role")=="assistant" and message.get("tool_calls"):
                user=" ".join(str(m.get("content") or "") for m in prior if m.get("role")=="user")
                tool=" ".join(str(m.get("content") or "") for m in prior if m.get("role")=="tool")
                output[(sim["id"],decision)]=(user,tool);decision+=1
            prior.append(message)
    return output


def ownership(db):
    customers={x["customer_id"]:set(x.get("line_ids") or []) for x in db.get("customers",[])}
    lines={x["line_id"] for x in db.get("lines",[])}
    return customers,lines


def enrich(rows,texts,customers,lines):
    enriched=[]
    for row in rows.itertuples():
        args=row.candidate_arguments or {};values=[x for x in scalar_values(args) if len(x)>=3];user,tool=texts.get((row.simulation_id,int(row.decision_index)),("",""));cid=str(args.get("customer_id",""));lid=str(args.get("line_id",""))
        valid=int((not cid or cid in customers) and (not lid or lid in lines) and (not cid or not lid or lid in customers.get(cid,set())))
        enriched.append({**row._asdict(),"arg_count":len(values),"user_visible":sum(v in user for v in values),"tool_visible":sum(v in tool for v in values),"all_visible":int(all(v in user or v in tool for v in values)),"ownership_valid":valid,"has_customer":int(bool(cid)),"has_line":int(bool(lid)),"same_customer_line":int(bool(cid and lid and lid in customers.get(cid,set()))),"wrong_entity":int(str(row.perturbation_type) in {"replace_customer_id","replace_line_id"})})
    return pd.DataFrame(enriched)


def target_entities(group):
    safe=group[group.counterfactual_safe==1];customers=set();lines=set()
    for args in safe.candidate_arguments:
        if args.get("customer_id"):customers.add(args["customer_id"])
        if args.get("line_id"):lines.add(args["line_id"])
    return customers,lines


def intervention(frame,name,customers):
    output=[]
    for _,group in frame.groupby(["simulation_id","decision_index"],sort=False):
        target_customers,target_lines=target_entities(group);keep=pd.Series(True,index=group.index)
        if name=="single_asset":keep=~group.perturbation_type.isin(["replace_customer_id","replace_line_id"])
        elif name=="connected_assets":
            def connected(r):
                if r.counterfactual_safe:return True
                args=r.candidate_arguments;cid=args.get("customer_id");lid=args.get("line_id")
                return bool(lid and any(lid in customers.get(tc,set()) for tc in target_customers) and (not cid or cid in target_customers))
            keep=group.apply(connected,axis=1)
        elif name=="independent_assets":
            def independent(r):
                if r.counterfactual_safe:return True
                args=r.candidate_arguments;cid=args.get("customer_id");lid=args.get("line_id")
                return bool((cid and cid not in target_customers) or (lid and all(lid not in customers.get(tc,set()) for tc in target_customers)))
            keep=group.apply(independent,axis=1)
        elif name=="cross_asset":keep=(group.has_customer.astype(bool)&group.has_line.astype(bool))|group.counterfactual_safe.astype(bool)
        selected=group[keep]
        if selected.counterfactual_safe.sum() and len(selected)>1:output.append(selected)
    out=pd.concat(output,ignore_index=True) if output else frame.iloc[:0].copy()
    if name=="ambiguous_asset":out[["user_visible","all_visible"]]=0
    if name=="missing_target_evidence":out[["tool_visible","all_visible"]]=0
    return out


def select_metrics(scored,model_name,condition):
    rows=[]
    for (sim,decision),group in scored.groupby(["simulation_id","decision_index"],sort=False):
        selected=group.sort_values(["score","candidate_index"],ascending=[False,True]).iloc[0]
        rows.append({"simulation_id":sim,"decision_index":decision,"condition":condition,"model":model_name,"safe":int(selected.counterfactual_safe),"wrong_asset":int((selected.wrong_entity or not selected.ownership_valid) and not selected.counterfactual_safe),"ownership_valid":int(selected.ownership_valid)})
    return rows


def run(results_path,rows_path,db_path,output,seed):
    payload=json.loads(results_path.read_text());texts=prefix_texts(payload);db=tomllib.loads(db_path.read_text());customers,lines=ownership(db);raw=pd.read_json(rows_path,lines=True);raw=enrich(raw,texts,customers,lines);raw=raw[raw.candidate_is_write==1].reset_index(drop=True)
    conditions=["canonical","single_asset","connected_assets","independent_assets","ambiguous_asset","missing_target_evidence","cross_asset"]
    canonical=intervention(raw,"canonical",customers);groups=canonical.task_id
    feature_sets={"mechanical":["context_token_estimate","prior_read_calls","candidate_args_total"] if "candidate_args_total" in canonical else ["context_token_estimate","prior_read_calls","arg_count"],"provenance":["context_token_estimate","prior_read_calls","arg_count","user_visible","tool_visible","all_visible"],"entity_aware":["context_token_estimate","prior_read_calls","arg_count","user_visible","tool_visible","all_visible","ownership_valid","has_customer","has_line","same_customer_line","wrong_entity"]}
    predictions=[];splitter=GroupKFold(n_splits=5)
    for fold,(train_idx,test_idx) in enumerate(splitter.split(canonical,canonical.counterfactual_safe,groups)):
        train_tasks=set(canonical.iloc[train_idx].task_id);test_tasks=set(canonical.iloc[test_idx].task_id)
        train=canonical[canonical.task_id.isin(train_tasks)]
        models={}
        for name,cols in feature_sets.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=2000,C=1,random_state=seed));model.fit(train[cols],train.counterfactual_safe);models[name]=(model,cols)
        for condition in conditions:
            test=intervention(raw[raw.task_id.isin(test_tasks)],condition,customers)
            for name,(model,cols) in models.items():
                scored=test.copy();scored["score"]=model.predict_proba(scored[cols])[:,1];predictions.extend(select_metrics(scored,name,condition))
            scored=test.copy();scored["score"]=scored.ownership_valid*100+scored.all_visible*10+scored.tool_visible+scored.user_visible;predictions.extend(select_metrics(scored,"hard_entity_gate",condition))
            scored=test.copy();scored["score"]=np.random.default_rng(seed+fold).random(len(scored));predictions.extend(select_metrics(scored,"random",condition))
            scored=test.copy();scored["score"]=scored.counterfactual_safe;predictions.extend(select_metrics(scored,"oracle",condition))
    detail=pd.DataFrame(predictions);summary=detail.groupby("model").agg(decisions=("safe","size"),safe_action_rate=("safe","mean"),wrong_asset_rate=("wrong_asset","mean"),ownership_valid_rate=("ownership_valid","mean")).reset_index();by_condition=detail.groupby(["model","condition"]).agg(decisions=("safe","size"),safe_action_rate=("safe","mean"),wrong_asset_rate=("wrong_asset","mean")).reset_index()
    report={"dataset":"Tau2 Telecom hard-counterfactual replay","split":"5-fold held-out task groups","conditions":conditions,"models":summary.to_dict("records"),"by_condition":by_condition.to_dict("records"),"integrity":{"hidden_world_fixed":True,"candidate_safety_labels_not_features":True,"interventions_modify_candidate inventory or observation visibility only":True},"limitations":["Customer graph has four customers and nine lines.","Connected-asset structure is ownership, not a general causal topology.","Visibility is exact scalar-value provenance in prior user/tool messages."]}
    output.write_text(json.dumps(report,indent=2)+"\n");detail.to_csv(output.with_suffix(".csv"),index=False);print(summary.to_string(index=False));print(f"wrote {output}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,required=True);p.add_argument("--rows",type=Path,required=True);p.add_argument("--db",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("artifacts/tau2_entity_context.json"));p.add_argument("--seed",type=int,default=42);a=p.parse_args();run(a.results,a.rows,a.db,a.output,a.seed)
