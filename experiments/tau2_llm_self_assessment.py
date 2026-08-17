"""Evaluate the archived Tau2 LLM's read/write choice as a sufficiency gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd


def run(rows_path,output):
    data=pd.read_json(rows_path,lines=True);records=[]
    for (simulation,decision),group in data.groupby(["simulation_id","decision_index"],sort=False):
        actual=group[(group.candidate_origin=="actual") & (group.candidate_is_write==1)]
        llm_act=int(len(actual)>0);llm_safe=int(actual.counterfactual_safe.max()) if llm_act else 0;oracle_sufficient=int(group.counterfactual_safe.max())
        first=group.iloc[0]
        records.append({"simulation_id":simulation,"decision_index":int(decision),"task_id":first.task_id,"task_family":first.task_family,"prior_reads":int(first.prior_read_calls),"llm_act":llm_act,"llm_safe":llm_safe,"oracle_sufficient":oracle_sufficient,"false_sufficient":int(llm_act and not llm_safe),"false_insufficient":int(not llm_act and oracle_sufficient)})
    detail=pd.DataFrame(records);acted=detail[detail.llm_act==1]
    metrics={"decision_prefixes":len(detail),"coverage":float(detail.llm_act.mean()),"safe_action_rate_all":float(detail.llm_safe.mean()),"safe_rate_when_acted":float(acted.llm_safe.mean()) if len(acted) else 0.,"false_sufficient_rate_all":float(detail.false_sufficient.mean()),"false_sufficient_rate_when_acted":float(acted.false_sufficient.mean()) if len(acted) else 0.,"false_insufficient_rate_all":float(detail.false_insufficient.mean()),"oracle_sufficient_rate":float(detail.oracle_sufficient.mean()),"unnecessary_query_rate_given_sufficient":float(detail.false_insufficient.sum()/max(detail.oracle_sufficient.sum(),1)),"mean_reads_before_act":float(acted.prior_reads.mean()) if len(acted) else 0.}
    by_family=detail.groupby("task_family").agg(prefixes=("llm_act","size"),coverage=("llm_act","mean"),safe_rate_all=("llm_safe","mean"),false_sufficient=("false_sufficient","mean"),false_insufficient=("false_insufficient","mean")).reset_index()
    payload={"dataset":"Pinned Tau2 GPT-4.1 Telecom trajectories and hard-counterfactual continuation labels","interpretation":"A write call is implicit LLM context-sufficient; another read or dialogue step is implicit insufficient.","metrics":metrics,"by_task_family":by_family.to_dict("records"),"limitations":["The archived model did not emit a numeric confidence score.","A safe counterfactual candidate existing does not mean the LLM knew which candidate was safe.","Dialogue-only turns are grouped with continued acquisition/abstention."]}
    output.write_text(json.dumps(payload,indent=2)+"\n");detail.to_csv(output.with_suffix(".csv"),index=False);print(json.dumps(metrics,indent=2));print(f"wrote {output}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--rows",type=Path,required=True);p.add_argument("--output",type=Path,default=Path("artifacts/tau2_llm_self_assessment.json"));a=p.parse_args();run(a.rows,a.output)
