"""Calibrated, risk-controlled selective context-calling evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from replay_common import (  # noqa: E402
    candidate_key,
    context_features,
    entity_pools,
    evidence_features,
    load_json,
    load_tool_types,
    perturbations,
    provenance_features,
    reference_candidates,
    source_events,
)
from tau2.data_model.tasks import Task


NUMERIC = [
    "context_messages", "context_chars", "context_token_estimate",
    "prior_assistant_messages", "prior_user_messages", "prior_tool_results",
    "prior_tool_calls", "prior_unique_tools", "prior_read_calls", "prior_write_calls",
    "candidate_is_write", "has_customer_lookup", "has_line_lookup", "has_status_check",
    "has_sim_check", "has_network_check", "has_action_tool", "decision_index",
]
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
CATEGORICAL = ["task_family", "candidate_tool"]
RICH_FEATURES = NUMERIC + PROVENANCE + EVIDENCE + CATEGORICAL


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


def enrich_frame(frame, prefix_map, types):
    provenance_rows = []
    evidence_rows = []
    for _, row in frame.iterrows():
        prior = prefix_map.get((row["simulation_id"], int(row["decision_index"])), [])
        candidate = {"name": row["candidate_tool"], "arguments": row.get("candidate_arguments") or {}}
        provenance_rows.append(provenance_features(prior, candidate, types))
        evidence_rows.append(evidence_features(prior, candidate, types))
    return pd.concat(
        [frame.reset_index(drop=True), pd.DataFrame(provenance_rows), pd.DataFrame(evidence_rows)],
        axis=1,
    )


def fit_model(train):
    transformers = [("numeric", StandardScaler(), NUMERIC + PROVENANCE + EVIDENCE)]
    transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL))
    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear")),
    ])
    model.fit(train[RICH_FEATURES], train["counterfactual_safe"])
    return model


class CalibratedModel:
    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, frame):
        raw = self.base.predict_proba(frame[RICH_FEATURES])[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1.0 - calibrated, calibrated])


def calibrate(base, calibration):
    raw = base.predict_proba(calibration[RICH_FEATURES])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw, calibration["counterfactual_safe"].to_numpy())
    return CalibratedModel(base, calibrator)


def risk_threshold(probabilities, labels, alpha, confidence, minimum=25):
    """Choose a threshold on the same unit used at deployment: one choice per prefix."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    candidates = sorted(set(float(value) for value in probabilities), reverse=True)
    for threshold in candidates:
        selected = probabilities >= threshold
        n = int(selected.sum())
        if n < minimum:
            continue
        unsafe = int((1 - labels[selected]).sum())
        upper = float(beta.ppf(confidence, unsafe + 1, n - unsafe + 1))
        if upper <= alpha:
            return threshold, {"selected": n, "unsafe": unsafe, "unsafe_upper_bound": upper}
    return 1.0, {"selected": 0, "unsafe": 0, "unsafe_upper_bound": 1.0}


def decision_level_calibration(calibrated, calibration):
    """Collapse candidate rows to the candidate actually selected at each prefix."""
    scored = calibration.copy()
    scored["probability"] = calibrated.predict_proba(scored[RICH_FEATURES])[:, 1]
    selected = []
    for _, group in scored.groupby(["simulation_id", "decision_index"], sort=False):
        row = group.loc[group["probability"].idxmax()]
        selected.append(row)
    selected = pd.DataFrame(selected)
    return selected["probability"].to_numpy(), selected["counterfactual_safe"].to_numpy()


def masked_prior(prior, events, keep):
    remove = set()
    for event in events:
        if event["source_event_index"] not in keep:
            remove.update(range(event["source_message_index"], event["source_end_index"]))
    return [message for index, message in enumerate(prior) if index not in remove]


def score_candidates(model, simulation, prior, candidates, types, decision_index):
    rows = []
    for candidate in candidates:
        row = context_features(simulation, prior, candidate, types, decision_index)
        row.update(provenance_features(prior, candidate, types))
        row.update(evidence_features(prior, candidate, types))
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, model.predict_proba(frame[RICH_FEATURES])[:, 1]


def choose(model, simulation, prior, candidates, types, decision_index, labels, threshold, require_complete=True):
    frame, probabilities = score_candidates(model, simulation, prior, candidates, types, decision_index)
    eligible = np.ones(len(candidates), dtype=bool)
    if require_complete:
        eligible &= frame["evidence_fact_complete"].to_numpy(dtype=bool)
        eligible &= frame["evidence_argument_conflict_count"].to_numpy() == 0
    eligible &= probabilities >= threshold
    if not eligible.any():
        return {"abstain": 1, "probability": float(probabilities.max()) if len(probabilities) else 0.0, "read_calls": 0, "context_tokens": int(frame["context_token_estimate"].min()) if len(frame) else 0, "safe": 0, "selected_sources": []}
    eligible_indices = np.flatnonzero(eligible)
    index = int(eligible_indices[np.argmax(probabilities[eligible])])
    candidate = candidates[index]
    label = labels.get((simulation["id"], decision_index, candidate_key(candidate)), 0)
    return {"abstain": 0, "candidate": candidate, "probability": float(probabilities[index]), "read_calls": 0, "context_tokens": int(frame.iloc[index]["context_token_estimate"]), "safe": int(label), "selected_sources": []}


def best_unconstrained(model, simulation, prior, candidates, types, decision_index, labels):
    frame, probabilities = score_candidates(model, simulation, prior, candidates, types, decision_index)
    index = int(np.argmax(probabilities))
    candidate = candidates[index]
    label = labels.get((simulation["id"], decision_index, candidate_key(candidate)), 0)
    return {
        "candidate": candidate,
        "probability": float(probabilities[index]),
        "safe": int(label),
        "context_tokens": int(frame.iloc[index]["context_token_estimate"]),
        "fact_complete": int(frame.iloc[index]["evidence_fact_complete"]),
        "conflict_count": int(frame.iloc[index]["evidence_argument_conflict_count"]),
    }


def acquire(model, simulation, prior, events, candidates, types, decision_index, labels, threshold, max_reads):
    selected = set()
    while True:
        current_prior = masked_prior(prior, events, selected)
        current = choose(model, simulation, current_prior, candidates, types, decision_index, labels, threshold)
        if not current["abstain"] or len(selected) >= min(max_reads, len(events)):
            current["selected_sources"] = sorted(selected)
            current["read_calls"] = len(selected)
            return current
        gains = []
        current_unconstrained = best_unconstrained(model, simulation, current_prior, candidates, types, decision_index, labels)
        for event in events:
            if event["source_event_index"] in selected:
                continue
            candidate_prior = masked_prior(prior, events, selected | {event["source_event_index"]})
            candidate_choice = best_unconstrained(model, simulation, candidate_prior, candidates, types, decision_index, labels)
            candidate_score = candidate_choice["probability"]
            probability_gain = candidate_score - current_unconstrained["probability"]
            completeness_gain = candidate_choice["fact_complete"] - current_unconstrained["fact_complete"]
            conflict_gain = min(current_unconstrained["conflict_count"], 1) - min(candidate_choice["conflict_count"], 1)
            gains.append((completeness_gain, conflict_gain, probability_gain, candidate_score, -event["source_event_index"], event["source_event_index"]))
        if not gains:
            current["selected_sources"] = sorted(selected)
            current["read_calls"] = len(selected)
            return current
        selected.add(max(gains)[-1])


def metrics(rows):
    attempted = [row for row in rows if not row["abstain"]]
    return {
        "decision_rows": len(rows),
        "coverage": float(len(attempted) / max(1, len(rows))),
        "safe_rate_all": float(np.mean([row["safe"] for row in rows])),
        "safe_rate_attempted": float(np.mean([row["safe"] for row in attempted])) if attempted else 0.0,
        "mean_read_calls": float(np.mean([row["read_calls"] for row in rows])),
        "mean_context_tokens": float(np.mean([row["context_tokens"] for row in rows])),
        "p90_read_calls": float(np.percentile([row["read_calls"] for row in rows], 90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hard-rows", type=Path, required=True)
    parser.add_argument("--tool-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", default="0.10,0.20,0.30")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--max-reads", type=int, default=99)
    args = parser.parse_args()

    payload = load_json(args.results)
    tasks = {item["id"]: Task.model_validate(item) for item in payload["tasks"]}
    types = load_tool_types(args.tool_source)
    pools = entity_pools()
    hard = pd.read_json(args.hard_rows, lines=True)

    prefix_map = {}
    for simulation in payload["simulations"]:
        prior = []
        decision_index = 0
        for message in simulation.get("messages") or []:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                prefix_map[(simulation["id"], decision_index)] = list(prior)
                decision_index += 1
            prior.append(message)
    hard = enrich_frame(hard, prefix_map, types)
    hard = hard.dropna(subset=RICH_FEATURES + ["counterfactual_safe", "task_id"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_cal_idx, test_idx = next(splitter.split(hard[RICH_FEATURES], hard["counterfactual_safe"], hard["task_id"]))
    train_cal = hard.iloc[train_cal_idx]
    test = hard.iloc[test_idx]
    cal_splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=43)
    train_rel, cal_rel = next(cal_splitter.split(train_cal[RICH_FEATURES], train_cal["counterfactual_safe"], train_cal["task_id"]))
    train = train_cal.iloc[train_rel].copy()
    calibration = train_cal.iloc[cal_rel].copy()
    test_tasks = set(test["task_id"])
    base = fit_model(train)
    calibrated = calibrate(base, calibration)

    calibration_probabilities, calibration_labels = decision_level_calibration(calibrated, calibration)
    thresholds = {}
    for alpha in [float(item) for item in args.alpha.split(",")]:
        threshold, info = risk_threshold(calibration_probabilities, calibration_labels, alpha, args.confidence, minimum=10)
        thresholds[f"alpha_{alpha:g}"] = {"threshold": threshold, **info}

    validation_prob = calibrated.predict_proba(test[RICH_FEATURES])[:, 1]
    validation = {
        "rows": len(test),
        "positive_rate": float(test["counterfactual_safe"].mean()),
        "auroc": float(roc_auc_score(test["counterfactual_safe"], validation_prob)),
        "auprc": float(average_precision_score(test["counterfactual_safe"], validation_prob)),
    }
    labels = {(row["simulation_id"], int(row["decision_index"]), candidate_key({"name": row["candidate_tool"], "arguments": row.get("candidate_arguments") or {}})): int(row["counterfactual_safe"]) for _, row in hard.iterrows()}
    policy_rows = defaultdict(list)
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
            full = choose(calibrated, simulation, prior, candidates, types, decision_index, labels, 0.0, require_complete=False)
            full["read_calls"] = len(events)
            no_read = choose(calibrated, simulation, masked_prior(prior, events, set()), candidates, types, decision_index, labels, 0.0, require_complete=False)
            no_read["read_calls"] = 0
            for policy, row in [("full_context", full), ("no_read", no_read)]:
                policy_rows[policy].append({**row, "policy": policy})
            fixed_prior = masked_prior(prior, events, {events[0]["source_event_index"]})
            fixed_top1 = choose(calibrated, simulation, fixed_prior, candidates, types, decision_index, labels, 0.0, require_complete=False)
            fixed_top1["policy"] = "fixed_top1"
            fixed_top1["read_calls"] = 1
            fixed_top1["selected_sources"] = [events[0]["source_event_index"]]
            policy_rows["fixed_top1"].append(fixed_top1)
            for name, details in thresholds.items():
                choice = acquire(calibrated, simulation, prior, events, candidates, types, decision_index, labels, details["threshold"], args.max_reads)
                choice["policy"] = name
                policy_rows[name].append(choice)
            prior.append(message)
            decision_index += 1

    report = {
        "heldout_task_groups": len(test_tasks),
        "decision_prefixes": processed,
        "train_task_groups": int(train["task_id"].nunique()),
        "calibration_task_groups": int(calibration["task_id"].nunique()),
        "test_task_groups": int(test["task_id"].nunique()),
        "calibration_method": "isotonic regression plus decision-prefix selection calibration",
        "calibration_decision_prefixes": int(len(calibration_probabilities)),
        "risk_control": {"confidence": args.confidence, "alpha_thresholds": thresholds},
        "validation": validation,
        "policies": {name: metrics(rows) for name, rows in policy_rows.items()},
        "note": "Risk-controlled offline evaluation. Abstention means the controller would request more context or escalate rather than issue a write. Labels remain hard-counterfactual continuation outcomes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
