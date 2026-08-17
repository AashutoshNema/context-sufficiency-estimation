"""Offline closed-loop selective context-calling evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from replay_common import (  # noqa: E402
    candidate_key,
    context_features,
    entity_pools,
    load_json,
    load_tool_types,
    perturbations,
    reference_candidates,
    source_events,
    provenance_features,
    evidence_features,
)
from tau2.data_model.tasks import Task


NUMERIC = [
    "context_messages", "context_chars", "context_token_estimate",
    "prior_assistant_messages", "prior_user_messages", "prior_tool_results",
    "prior_tool_calls", "prior_unique_tools", "prior_read_calls", "prior_write_calls",
    "candidate_is_write", "has_customer_lookup", "has_line_lookup", "has_status_check",
    "has_sim_check", "has_network_check", "has_action_tool", "decision_index",
]
CATEGORICAL = ["task_family", "candidate_tool"]
PROVENANCE = [
    "candidate_args_total", "candidate_args_visible_count", "candidate_args_visible_rate",
    "candidate_args_from_user_count", "candidate_args_from_tool_count",
    "candidate_args_from_read_count", "candidate_arg_source_count",
    "candidate_args_all_visible", "candidate_args_all_from_read",
    "candidate_arg_from_customer_lookup", "candidate_arg_from_line_details",
    "candidate_arg_from_bills", "candidate_arg_from_usage",
]
EVIDENCE = [
    "evidence_source_count", "evidence_relevant_source_count",
    "evidence_distractor_source_count", "evidence_relevance_rate",
    "evidence_structured_relevant_count", "evidence_value_corroboration_mean",
    "evidence_value_corroboration_min", "evidence_argument_conflict_count",
    "evidence_source_token_cost", "evidence_relevant_token_cost",
    "evidence_read_cost", "evidence_relevant_cost", "evidence_identity_sources",
    "evidence_entity_state_sources", "evidence_billing_sources",
    "evidence_usage_sources", "evidence_candidate_entity_scope",
    "evidence_fact_complete",
]


def candidates_for(task, actual, types, pools):
    base = []
    seen = set()
    if types.get(actual.get("name")) == "write":
        base.append(dict(actual))
        seen.add(candidate_key(actual))
    for reference in reference_candidates(task):
        if types.get(reference["name"]) == "write" and candidate_key(reference) not in seen:
            base.append(dict(reference))
            seen.add(candidate_key(reference))
    expanded = list(base)
    for candidate in base:
        for altered in perturbations(candidate, pools):
            if candidate_key(altered) not in seen:
                expanded.append(altered)
                seen.add(candidate_key(altered))
    return expanded


def model_for(train: pd.DataFrame, categorical: list[str], rich: bool):
    rich_numeric = NUMERIC + (PROVENANCE + EVIDENCE if rich else [])
    features = rich_numeric + categorical
    transformers = [("numeric", StandardScaler(), rich_numeric)]
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear")),
    ])
    model.fit(train[features], train["counterfactual_safe"])
    return model, features


def masked_prior(prior, events, keep):
    remove = set()
    for event in events:
        if event["source_event_index"] not in keep:
            remove.update(range(event["source_message_index"], event["source_end_index"]))
    return [message for index, message in enumerate(prior) if index not in remove]


def score_candidates(model, features, simulation, prior, candidates, types, decision_index):
    feature_rows = []
    for candidate in candidates:
        row = context_features(simulation, prior, candidate, types, decision_index)
        row.update(provenance_features(prior, candidate, types))
        row.update(evidence_features(prior, candidate, types))
        feature_rows.append(row)
    frame = pd.DataFrame(feature_rows)
    probabilities = model.predict_proba(frame[features])[:, 1]
    return frame, probabilities


def best_candidate(model, features, simulation, prior, candidates, types, decision_index, labels):
    frame, probabilities = score_candidates(model, features, simulation, prior, candidates, types, decision_index)
    index = int(np.argmax(probabilities))
    candidate = candidates[index]
    label = labels.get((simulation["id"], decision_index, candidate_key(candidate)), 0)
    return {
        "candidate": candidate,
        "probability": float(probabilities[index]),
        "safe": int(label),
        "context_tokens": int(frame.iloc[index]["context_token_estimate"]),
    }


def selective_choice(model, features, simulation, prior, events, candidates, types, decision_index, labels, threshold):
    selected: set[int] = set()
    while True:
        current_prior = masked_prior(prior, events, selected)
        current = best_candidate(model, features, simulation, current_prior, candidates, types, decision_index, labels)
        gains = []
        for event in events:
            event_index = event["source_event_index"]
            if event_index in selected:
                continue
            candidate_prior = masked_prior(prior, events, selected | {event_index})
            candidate_best = best_candidate(model, features, simulation, candidate_prior, candidates, types, decision_index, labels)
            gains.append((candidate_best["probability"] - current["probability"], event_index))
        if not gains:
            break
        gain, event_index = max(gains)
        if gain < threshold:
            break
        selected.add(event_index)
    final = best_candidate(model, features, simulation, masked_prior(prior, events, selected), candidates, types, decision_index, labels)
    final["selected_sources"] = sorted(selected)
    return final


def policy_metrics(rows):
    if not rows:
        return {"decision_rows": 0}
    return {
        "decision_rows": len(rows),
        "safe_rate": float(np.mean([row["safe"] for row in rows])),
        "mean_predicted_safe_probability": float(np.mean([row["probability"] for row in rows])),
        "mean_read_calls": float(np.mean([row["read_calls"] for row in rows])),
        "mean_context_tokens": float(np.mean([row["context_tokens"] for row in rows])),
        "p50_read_calls": float(np.percentile([row["read_calls"] for row in rows], 50)),
        "p90_read_calls": float(np.percentile([row["read_calls"] for row in rows], 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hard-rows", type=Path, required=True)
    parser.add_argument("--tool-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.00,0.01,0.02,0.05,0.10")
    args = parser.parse_args()

    payload = load_json(args.results)
    tasks = {item["id"]: Task.model_validate(item) for item in payload["tasks"]}
    types = load_tool_types(args.tool_source)
    pools = entity_pools()
    hard = pd.read_json(args.hard_rows, lines=True)
    features = NUMERIC + PROVENANCE + EVIDENCE + CATEGORICAL
    prefix_map = {}
    for simulation in payload["simulations"]:
        prior = []
        decision_index = 0
        for message in simulation.get("messages") or []:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                prefix_map[(simulation["id"], decision_index)] = list(prior)
                decision_index += 1
            prior.append(message)
    provenance_rows = []
    for _, row in hard.iterrows():
        prior = prefix_map.get((row["simulation_id"], int(row["decision_index"])), [])
        candidate = {"name": row["candidate_tool"], "arguments": row.get("candidate_arguments") or {}}
        provenance_rows.append(provenance_features(prior, candidate, types))
    evidence_rows = []
    for _, row in hard.iterrows():
        prior = prefix_map.get((row["simulation_id"], int(row["decision_index"])), [])
        candidate = {"name": row["candidate_tool"], "arguments": row.get("candidate_arguments") or {}}
        evidence_rows.append(evidence_features(prior, candidate, types))
    hard = pd.concat([hard.reset_index(drop=True), pd.DataFrame(provenance_rows), pd.DataFrame(evidence_rows)], axis=1)
    hard = hard.dropna(subset=features + ["counterfactual_safe", "task_id"])
    labels = {(row["simulation_id"], int(row["decision_index"]), candidate_key({"name": row["candidate_tool"], "arguments": row.get("candidate_arguments") or {}})): int(row["counterfactual_safe"]) for _, row in hard.iterrows()}
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(hard[features], hard["counterfactual_safe"], hard["task_id"]))
    train = hard.iloc[train_idx].copy()
    test_tasks = set(hard.iloc[test_idx]["task_id"])

    models = {}
    for name, categorical, rich in [
        ("mechanical_baseline", [], False),
        ("rich_context", CATEGORICAL, True),
    ]:
        models[name] = model_for(train, categorical, rich)

    validation = {}
    test_frame = hard.iloc[test_idx]
    for name, (model, model_features) in models.items():
        probabilities = model.predict_proba(test_frame[model_features])[:, 1]
        validation[name] = {
            "rows": len(test_frame),
            "positive_rate": float(test_frame["counterfactual_safe"].mean()),
            "auroc": float(roc_auc_score(test_frame["counterfactual_safe"], probabilities)),
            "auprc": float(average_precision_score(test_frame["counterfactual_safe"], probabilities)),
        }

    thresholds = [float(item) for item in args.thresholds.split(",")]
    policy_rows = defaultdict(lambda: defaultdict(list))
    processed = 0
    for simulation in payload["simulations"]:
        if simulation.get("task_id") not in test_tasks:
            continue
        task = tasks.get(simulation.get("task_id"))
        if task is None:
            continue
        prior = []
        decision_index = 0
        for message in simulation.get("messages") or []:
            calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not calls:
                prior.append(message)
                continue
            candidates = candidates_for(task, calls[0], types, pools)
            events = source_events(prior, types)
            if not candidates or not events:
                prior.append(message)
                decision_index += 1
                continue
            processed += 1
            for model_name, (model, model_features) in models.items():
                full = best_candidate(model, model_features, simulation, prior, candidates, types, decision_index, labels)
                no_read = best_candidate(model, model_features, simulation, masked_prior(prior, events, set()), candidates, types, decision_index, labels)
                for row_name, choice in [("full_context", full), ("no_read", no_read)]:
                    policy_rows[model_name][row_name].append({**choice, "read_calls": len(events) if row_name == "full_context" else 0})
                for threshold in thresholds:
                    choice = selective_choice(model, model_features, simulation, prior, events, candidates, types, decision_index, labels, threshold)
                    policy_rows[model_name][f"selective_{threshold:g}"].append({**choice, "read_calls": len(choice["selected_sources"])})
            decision_index += 1
            prior.append(message)

    report = {"heldout_task_groups": len(test_tasks), "decision_prefixes": processed, "training_rows": len(train), "validation": validation, "thresholds": thresholds, "models": {name: {policy: policy_metrics(rows) for policy, rows in policies.items()} for name, policies in policy_rows.items()}, "note": "Offline controller evaluation. Read calls are reconstructed source events; no new LLM calls are made. Candidate safety labels come from hard counterfactual continuation replay."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
