"""Generate hard telecom candidates and replay them from observed prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

from replay_common import *  # noqa: F403
from tau2.data_model.tasks import Task

logger.remove()
logger.add(sys.stderr, level="WARNING")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tool-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = load_json(args.results)
    tasks = {item["id"]: Task.model_validate(item) for item in payload["tasks"]}
    types = load_tool_types(args.tool_source)
    pools = entity_pools()
    rows = []
    prefixes = 0
    errors = 0
    for simulation in payload["simulations"]:
        task = tasks.get(simulation.get("task_id"))
        if task is None:
            continue
        prior = []
        decision_index = 0
        for message_index, message in enumerate(simulation.get("messages") or []):
            calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not calls:
                prior.append(message)
                continue
            actual = calls[0]
            references = [item for item in reference_candidates(task) if types.get(item["name"]) == "write"]
            candidates = []
            seen = set()
            if types.get(actual.get("name")) == "write":
                candidates.append((dict(actual), "actual")); seen.add(candidate_key(actual))
            for ref in references:
                if candidate_key(ref) not in seen:
                    candidates.append((dict(ref), "reference")); seen.add(candidate_key(ref))
            if not candidates:
                prior.append(message); decision_index += 1; continue
            expanded = list(candidates)
            for base, _ in candidates:
                for altered in perturbations(base, pools):
                    if candidate_key(altered) not in seen:
                        expanded.append((altered, "perturbed")); seen.add(candidate_key(altered))
            prefixes += 1
            for index, (candidate, origin) in enumerate(expanded):
                try:
                    env = fresh_at_prefix(task, prior)
                    before = db_signature(env)
                    valid, response = execute(env, candidate)
                    after = db_signature(env)
                    replay_suffix(env, (simulation.get("messages") or [])[message_index + 1:])
                    safe = valid and assertion_match(env, task)
                    row = context_features(simulation, prior, candidate, types, decision_index)
                    row.update({"candidate_index": index, "candidate_origin": origin, "perturbation_type": candidate.get("perturbation_type", "none"), "perturbed_argument": candidate.get("perturbed_argument"), "original_value": candidate.get("original_value"), "alternate_value": candidate.get("alternate_value"), "is_hard_counterfactual": int(origin == "perturbed"), "execution_valid": int(valid), "state_changed": int(before != after), "response_chars": len(response or ""), "counterfactual_safe": int(safe), "replay_error": None})
                    rows.append(row)
                except Exception as exc:
                    errors += 1
                    row = context_features(simulation, prior, candidate, types, decision_index)
                    row.update({"candidate_index": index, "candidate_origin": origin, "perturbation_type": candidate.get("perturbation_type", "none"), "is_hard_counterfactual": int(origin == "perturbed"), "execution_valid": 0, "state_changed": 0, "counterfactual_safe": 0, "replay_error": str(exc)})
                    rows.append(row)
            decision_index += 1
            prior.append(message)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "telecom_hard_counterfactual_rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {"candidate_rows": len(rows), "decision_prefixes": prefixes, "candidate_origins": dict(Counter(row["candidate_origin"] for row in rows)), "perturbation_types": dict(Counter(row["perturbation_type"] for row in rows)), "execution_valid_rate": sum(row["execution_valid"] for row in rows) / max(1, len(rows)), "safe_rate": sum(row["counterfactual_safe"] for row in rows) / max(1, len(rows)), "hard_execution_valid_rate": sum(row["execution_valid"] for row in rows if row["is_hard_counterfactual"]) / max(1, sum(row["is_hard_counterfactual"] for row in rows)), "hard_safe_rate": sum(row["counterfactual_safe"] for row in rows if row["is_hard_counterfactual"]) / max(1, sum(row["is_hard_counterfactual"] for row in rows)), "errors": errors}
    (args.output_dir / "telecom_hard_counterfactual_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
