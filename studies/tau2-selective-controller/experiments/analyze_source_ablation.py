"""Evaluate safety prediction and candidate selection after source removal."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = [
    "context_messages", "context_chars", "context_token_estimate",
    "prior_assistant_messages", "prior_user_messages", "prior_tool_results",
    "prior_tool_calls", "prior_unique_tools", "prior_read_calls", "prior_write_calls",
    "candidate_is_write", "has_customer_lookup", "has_line_lookup", "has_status_check",
    "has_sim_check", "has_network_check", "has_action_tool", "decision_index",
]
CATEGORICAL = ["task_family", "candidate_tool"]


def metric(y, p):
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def fit(train: pd.DataFrame, categorical: list[str]):
    features = NUMERIC + categorical
    transformers = [("numeric", StandardScaler(), NUMERIC)]
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    model = Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear")),
    ])
    model.fit(train[features], train["counterfactual_safe"])
    return model, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-rows", type=Path, required=True)
    parser.add_argument("--ablated-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hard-only", action="store_true")
    args = parser.parse_args()

    full = pd.read_json(args.full_rows, lines=True)
    masked = pd.read_json(args.ablated_rows, lines=True)
    if args.hard_only:
        full = full[full["is_hard_counterfactual"] == 1].copy()
        masked = masked[masked["is_hard_counterfactual"] == 1].copy()
    features = NUMERIC + CATEGORICAL
    full = full.dropna(subset=features + ["counterfactual_safe", "task_id"])
    masked = masked.dropna(subset=features + ["counterfactual_safe", "task_id", "full_row_key"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(full[features], full["counterfactual_safe"], full["task_id"]))
    train = full.iloc[train_idx].copy()
    test = full.iloc[test_idx].copy()
    test_tasks = set(test["task_id"])
    masked_test = masked[masked["task_id"].isin(test_tasks)].copy().reset_index(drop=True)

    reports = {}
    source_effects = {}
    selection = {}
    for name, categorical in [("full_features", CATEGORICAL), ("context_only", [])]:
        model, model_features = fit(train, categorical)
        full_p = model.predict_proba(test[NUMERIC + categorical])[:, 1]
        masked_p = model.predict_proba(masked_test[NUMERIC + categorical])[:, 1]
        reports[name] = {"full_test": metric(test["counterfactual_safe"].to_numpy(), full_p), "source_ablated_test": metric(masked_test["counterfactual_safe"].to_numpy(), masked_p)}

        def full_key(row):
            return json.dumps((row["simulation_id"], int(row["decision_index"]), row["candidate_tool"], json.dumps(row["candidate_arguments"] or {}, sort_keys=True)))

        full_lookup = dict(zip(test.apply(full_key, axis=1), full_p))
        paired = masked_test.copy()
        paired["full_probability"] = paired["full_row_key"].map(lambda value: full_lookup.get(value, np.nan))
        paired = paired.dropna(subset=["full_probability"])
        paired["delta_masked_minus_full"] = masked_p[paired.index.to_numpy()] - paired["full_probability"].to_numpy()
        effects = []
        for source, group in paired.groupby("source_tool"):
            delta = group["delta_masked_minus_full"]
            effects.append({"source_tool": source, "rows": int(len(group)), "mean_delta_safe_probability": float(delta.mean()), "mean_abs_delta": float(delta.abs().mean()), "risk_increase_rate": float((delta < 0).mean())})
        source_effects[name] = effects

        # Compare the candidate chosen by the model before and after each source is removed.
        paired["masked_probability"] = masked_p[paired.index.to_numpy()]
        for key, group in paired.groupby(["simulation_id", "decision_index", "source_event_index"]):
            best_full = group.loc[group["full_probability"].idxmax()]
            best_masked = group.loc[group["masked_probability"].idxmax()]
            selection.setdefault(name, []).append({"full_safe": int(best_full["counterfactual_safe"]), "masked_safe": int(best_masked["counterfactual_safe"]), "source_tool": best_masked["source_tool"]})

    selection_report = {}
    for name, items in selection.items():
        selection_report[name] = {
            "groups": len(items),
            "full_selected_safe_rate": float(np.mean([item["full_safe"] for item in items])),
            "masked_selected_safe_rate": float(np.mean([item["masked_safe"] for item in items])),
            "selection_changed_safe_outcome_rate": float(np.mean([item["full_safe"] != item["masked_safe"] for item in items])),
        }
    report = {"full_rows": len(full), "masked_rows": len(masked), "train_task_groups": int(train["task_id"].nunique()), "test_task_groups": len(test_tasks), "models": reports, "source_effects": source_effects, "selection": selection_report, "note": "Ablation changes observable context features while preserving hidden-state labels. This is an offline source-value proxy, not a live reacquisition controller."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
