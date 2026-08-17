"""Machine-check the locked north-star experiment claims."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def load(name):return json.loads((ROOT/"artifacts"/name).read_text())
def policies(payload):return {row["policy"]:row for row in payload["policies"]}


def main():
    action=load("openrca_action_model.json")
    assert action["gate"]["passed"]
    assert action["results"]["metric_trace"]["top3"]>action["results"]["prior"]["top3"]

    control=load("openrca_context_control_v2.json");p=policies(control)
    assert p["source_ranker"]["acceptable_rate"]>p["always_fetch"]["acceptable_rate"]
    assert p["source_ranker"]["mean_cost"]<p["always_fetch"]["mean_cost"]
    assert p["gate_t90"]["unsafe_action_rate"]<p["gate_t30"]["unsafe_action_rate"]
    assert control["correlation_audit"]["duplicate_accuracy_delta"]==0
    assert control["correlation_audit"]["independent_trace_accuracy_delta"]>0

    temporal=load("openrca_temporal_versions.json")
    assert temporal["top3_by_version"]["current_30m"]>temporal["top3_by_version"]["previous_30m"]
    assert temporal["learned_selector"]["top3"]>=temporal["top3_by_version"]["current_30m"]
    assert temporal["learned_selector"]["mean_cost"]<temporal["always_all_versions"]["cost"]

    probe=load("tau2_probing_escalation.json");p=policies(probe)
    assert p["gate_0.4"]["resolution_rate"]>=p["always_all"]["resolution_rate"]
    assert p["gate_0.4"]["mean_cost"]<p["always_all"]["mean_cost"]
    assert p["gate_0.4"]["escalation_precision"]>p["always_all"]["escalation_precision"]

    entity=load("tau2_entity_context.json");p={r["model"]:r for r in entity["models"]}
    assert p["entity_aware"]["safe_action_rate"]>=p["mechanical"]["safe_action_rate"]
    assert p["entity_aware"]["wrong_asset_rate"]==0

    semantics=load("tau2_source_semantics.json");p=policies(semantics)
    assert p["semantic_router"]["unsafe_action_rate"]==0
    assert p["prediction_only"]["prediction_overtrust_rate"]>0
    assert p["prediction_only"]["unsafe_action_rate"]>.9

    llm=load("tau2_llm_self_assessment.json")["metrics"]
    assert llm["false_sufficient_rate_when_acted"]>.4
    assert llm["unnecessary_query_rate_given_sufficient"]>.7

    manifest=load("experiment_manifest.json")
    assert manifest["global_rules"]["hidden_world_fixed_within_pair"]
    assert len(manifest["experiments"])>=7
    print("research artifact verification: PASS")
    print("verified: canonical action, source selection, risk-coverage, correlation, temporal coverage, probing/escalation, entity scope, source semantics, LLM baseline, manifest")


if __name__=="__main__":main()
