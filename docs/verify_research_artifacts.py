"""Check study structure and consistency of committed result summaries.

This does not rerun experiments from raw data. It verifies that each active
study owns its code and artifacts and that the repository-level status agrees
with selected machine-readable results.
"""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"


def load(study: str, name: str):
    return json.loads((STUDIES / study / "artifacts" / name).read_text())


def policies(payload):
    return {row["policy"]: row for row in payload["policies"]}


def verify_structure():
    active = [path for path in STUDIES.iterdir() if path.is_dir() and path.name != "_template"]
    assert active, "no active studies found"
    for study in active:
        assert (study / "README.md").is_file(), f"missing README: {study.name}"
        assert (study / "study_manifest.json").is_file(), f"missing study manifest: {study.name}"
        assert (study / "experiments").is_dir(), f"missing experiments/: {study.name}"
        assert (study / "artifacts").is_dir(), f"missing artifacts/: {study.name}"
    return active


def main():
    active = verify_structure()

    tau2 = load("tau2-selective-controller", "risk_controlled_controller_report_v3.json")["policies"]
    assert tau2["alpha_0.2"]["safe_rate_attempted"] < tau2["full_context"]["safe_rate_attempted"]

    action = load("openrca-source-selection", "openrca_action_model.json")
    assert action["results"]["metric_trace"]["top3"] > action["results"]["prior"]["top3"]

    control = load("openrca-source-selection", "openrca_context_control_v2.json")
    p = policies(control)
    assert p["source_ranker"]["acceptable_rate"] == p["fixed_relevant"]["acceptable_rate"]
    assert p["source_ranker"]["mean_cost"] < p["fixed_relevant"]["mean_cost"]
    assert control["correlation_audit"]["duplicate_accuracy_delta"] == 0

    temporal = load("openrca-temporal-context", "openrca_temporal_versions.json")
    assert temporal["top3_by_version"]["current_30m"] > temporal["top3_by_version"]["previous_30m"]
    assert temporal["learned_selector"]["mean_cost"] < temporal["always_all_versions"]["cost"]

    probe = policies(load("tau2-probing-escalation", "tau2_probing_escalation.json"))
    assert probe["gate_0.4"]["resolution_rate"] >= probe["always_all"]["resolution_rate"]
    assert probe["gate_0.4"]["mean_cost"] < probe["always_all"]["mean_cost"]

    entity = {row["model"]: row for row in load("tau2-entity-scope", "tau2_entity_context.json")["models"]}
    assert entity["entity_aware"]["wrong_asset_rate"] == 0

    semantics = policies(load("tau2-source-semantics", "tau2_source_semantics.json"))
    assert semantics["semantic_router"]["unsafe_action_rate"] == 0
    assert semantics["prediction_only"]["unsafe_action_rate"] > 0.9

    llm = load("tau2-llm-self-assessment", "tau2_llm_self_assessment.json")["metrics"]
    assert llm["false_sufficient_rate_when_acted"] > 0.4
    assert llm["unnecessary_query_rate_given_sufficient"] > 0.7

    print("research repository verification: PASS")
    print(f"verified structure for {len(active)} independent studies")
    print("verified active status: core controller NO-GO; supporting mechanism results internally consistent")


if __name__ == "__main__":
    main()
