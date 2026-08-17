"""Generate paired rows after removing one prior read result from context."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from replay_common import *  # noqa: F403
from tau2.data_model.tasks import Task


def row_key(simulation_id, decision_index, candidate):
    return (simulation_id, decision_index, candidate["name"], json.dumps(candidate.get("arguments") or {}, sort_keys=True))


def candidates_for(task, actual, types, pools):
    base = []
    seen = set()
    if types.get(actual.get("name")) == "write":
        base.append((dict(actual), "actual"))
        seen.add(candidate_key(actual))
    for ref in reference_candidates(task):
        if types.get(ref["name"]) == "write" and candidate_key(ref) not in seen:
            base.append((dict(ref), "reference"))
            seen.add(candidate_key(ref))
    expanded = list(base)
    for candidate, _ in base:
        for altered in perturbations(candidate, pools):
            if candidate_key(altered) not in seen:
                expanded.append((altered, "perturbed"))
                seen.add(candidate_key(altered))
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--hard-rows", type=Path, required=True)
    parser.add_argument("--tool-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-only", action="store_true")
    args = parser.parse_args()

    payload = load_json(args.results)
    tasks = {item["id"]: Task.model_validate(item) for item in payload["tasks"]}
    types = load_tool_types(args.tool_source)
    pools = entity_pools()
    labels = {}
    for line in args.hard_rows.open():
        row = json.loads(line)
        if args.hard_only and row.get("is_hard_counterfactual") != 1:
            continue
        key = row_key(row["simulation_id"], row["decision_index"], {"name": row["candidate_tool"], "arguments": row.get("candidate_arguments") or {}})
        labels[key] = row

    rows = []
    prefixes = 0
    missing = 0
    for simulation in payload["simulations"]:
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
            events = source_events(prior, types)
            candidates = candidates_for(task, calls[0], types, pools)
            if args.hard_only:
                candidates = [(candidate, origin) for candidate, origin in candidates if labels.get(row_key(simulation["id"], decision_index, candidate), {}).get("is_hard_counterfactual") == 1]
            if events and candidates:
                prefixes += 1
            for event in events:
                masked = prior[:event["source_message_index"]] + prior[event["source_end_index"]:]
                for candidate, origin in candidates:
                    label = labels.get(row_key(simulation["id"], decision_index, candidate))
                    if label is None:
                        missing += 1
                        continue
                    row = context_features(simulation, masked, candidate, types, decision_index)
                    row.update({
                        "source_event_index": event["source_event_index"],
                        "source_tool": event["source_tool"],
                        "source_call": event["source_call"],
                        "full_context_messages": label["context_messages"],
                        "full_context_chars": label["context_chars"],
                        "full_prior_read_calls": label["prior_read_calls"],
                        "counterfactual_safe": label["counterfactual_safe"],
                        "execution_valid": label["execution_valid"],
                        "is_hard_counterfactual": label["is_hard_counterfactual"],
                        "candidate_origin": label["candidate_origin"],
                        "perturbation_type": label["perturbation_type"],
                        "full_row_key": json.dumps(row_key(simulation["id"], decision_index, candidate)),
                    })
                    rows.append(row)
            decision_index += 1
            prior.append(message)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "telecom_source_ablated_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"ablated_rows": len(rows), "prefixes": prefixes, "source_tools": dict(Counter(row["source_tool"] for row in rows)), "missing_labels": missing, "hard_only": args.hard_only}
    (args.output_dir / "telecom_source_ablation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
