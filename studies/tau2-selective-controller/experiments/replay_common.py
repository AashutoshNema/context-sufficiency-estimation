"""Shared utilities for telecom counterfactual experiments."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.domains.telecom.environment import get_environment


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_tool_types(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "is_tool" and dec.args:
                arg = dec.args[0]
                if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name) and arg.value.id == "ToolType":
                    result[node.name] = arg.attr.lower()
    return result


def parse_message(raw: dict[str, Any]):
    classes = {"assistant": AssistantMessage, "user": UserMessage, "tool": ToolMessage, "system": SystemMessage}
    return classes[raw["role"]].model_validate(raw)


def fresh_at_prefix(task, prior: list[dict[str, Any]]):
    env = get_environment()
    initial = task.initial_state
    env.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=[parse_message(item) for item in prior],
        strict=False,
    )
    return env


def execute(env, candidate: dict[str, Any]) -> tuple[bool, str | None]:
    response = env.get_response(ToolCall.model_validate(candidate))
    return not response.error, response.content


def db_signature(env):
    return env.get_db_hash(), env.get_user_db_hash()


def replay_suffix(env, suffix: list[dict[str, Any]]) -> None:
    for message in suffix:
        if message.get("role") not in {"assistant", "user"}:
            continue
        for call in message.get("tool_calls") or []:
            execute(env, call)


def assertion_match(env, task) -> bool:
    return all(env.run_env_assertion(item, raise_assertion_error=False) for item in (task.evaluation_criteria.env_assertions or []))


def context_features(simulation, prior, candidate, tool_types, decision_index):
    prior_tools = [call.get("name") for message in prior for call in (message.get("tool_calls") or []) if call.get("name")]
    context = "\n".join(str(message.get("content") or "") for message in prior)
    tool = candidate["name"]
    return {
        "simulation_id": simulation.get("id"),
        "task_id": simulation.get("task_id"),
        "trial": simulation.get("trial"),
        "decision_index": decision_index,
        "task_family": (simulation.get("task_id") or "").split("]", 1)[0] + "]",
        "candidate_tool": tool,
        "candidate_tool_type": tool_types.get(tool, "unknown"),
        "candidate_is_write": int(tool_types.get(tool) == "write"),
        "candidate_arguments": candidate.get("arguments") or {},
        "context_messages": len(prior),
        "context_chars": len(context),
        "context_token_estimate": (len(context) + 3) // 4,
        "prior_assistant_messages": sum(item.get("role") == "assistant" for item in prior),
        "prior_user_messages": sum(item.get("role") == "user" for item in prior),
        "prior_tool_results": sum(item.get("role") == "tool" for item in prior),
        "prior_tool_calls": len(prior_tools),
        "prior_unique_tools": len(set(prior_tools)),
        "prior_read_calls": sum(tool_types.get(name) == "read" for name in prior_tools),
        "prior_write_calls": sum(tool_types.get(name) == "write" for name in prior_tools),
        "has_customer_lookup": int(any("customer" in name for name in prior_tools)),
        "has_line_lookup": int(any("detail" in name or "line" in name for name in prior_tools)),
        "has_status_check": int(any("status" in name or "signal" in name for name in prior_tools)),
        "has_sim_check": int(any("sim" in name for name in prior_tools)),
        "has_network_check": int(any("network" in name or "speed" in name for name in prior_tools)),
        "has_action_tool": int(any(token in name for name in prior_tools for token in ("toggle", "enable", "disable", "refuel", "reset", "unlock"))),
    }


def provenance_features(prior, candidate, tool_types):
    """Record where candidate argument values are observable in the prefix."""
    values = [
        str(value)
        for value in (candidate.get("arguments") or {}).values()
        if value not in (None, "", [], {})
    ]
    user_text = "\n".join(str(item.get("content") or "") for item in prior if item.get("role") == "user")
    tool_text_by_name = {}
    last_tool_name = None
    for item in prior:
        if item.get("role") == "assistant" and item.get("tool_calls"):
            last_tool_name = item["tool_calls"][0].get("name")
        elif item.get("role") == "tool":
            tool_text_by_name.setdefault(last_tool_name, []).append(str(item.get("content") or ""))
    tool_text_by_name = {name: "\n".join(texts) for name, texts in tool_text_by_name.items() if name}
    visible = [value for value in values if any(value in text for text in [user_text, *tool_text_by_name.values()])]
    from_user = [value for value in values if value in user_text]
    from_tools = [value for value in values if any(value in text for text in tool_text_by_name.values())]
    from_reads = [value for value in values if any(tool_types.get(name) == "read" and value in text for name, text in tool_text_by_name.items())]
    source_names = {name for name, text in tool_text_by_name.items() if tool_types.get(name) == "read" and any(value in text for value in values)}
    return {
        "candidate_args_total": len(values),
        "candidate_args_visible_count": len(visible),
        "candidate_args_visible_rate": len(visible) / max(1, len(values)),
        "candidate_args_from_user_count": len(from_user),
        "candidate_args_from_tool_count": len(from_tools),
        "candidate_args_from_read_count": len(from_reads),
        "candidate_arg_source_count": len(source_names),
        "candidate_args_all_visible": int(len(visible) == len(values)),
        "candidate_args_all_from_read": int(len(from_reads) == len(values)),
        "candidate_arg_from_customer_lookup": int("get_customer_by_phone" in source_names or "get_customer_by_id" in source_names),
        "candidate_arg_from_line_details": int("get_details_by_id" in source_names),
        "candidate_arg_from_bills": int("get_bills_for_customer" in source_names),
        "candidate_arg_from_usage": int("get_data_usage" in source_names),
    }


SOURCE_ROLES = {
    "get_customer_by_phone": "identity",
    "get_customer_by_id": "identity",
    "get_details_by_id": "entity_state",
    "get_bills_for_customer": "billing",
    "get_data_usage": "usage",
}
SOURCE_COSTS = {
    "get_customer_by_phone": 1.0,
    "get_customer_by_id": 1.0,
    "get_details_by_id": 1.2,
    "get_bills_for_customer": 1.5,
    "get_data_usage": 1.3,
}


def _read_records(prior, tool_types):
    records = []
    pending = None
    for item in prior:
        if item.get("role") == "assistant" and item.get("tool_calls"):
            call = item["tool_calls"][0]
            if tool_types.get(call.get("name")) == "read":
                pending = call.get("name")
        elif item.get("role") == "tool" and pending:
            text = str(item.get("content") or "")
            try:
                parsed = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                parsed = None
            records.append({"name": pending, "text": text, "parsed": parsed})
            pending = None
    return records


def evidence_features(prior, candidate, tool_types):
    """Structured, observable evidence features for source quality and relevance."""
    records = _read_records(prior, tool_types)
    arguments = candidate.get("arguments") or {}
    values = [str(value) for value in arguments.values() if value not in (None, "", [], {})]
    relevant = [record for record in records if any(value in record["text"] for value in values)]
    structured_relevant = [record for record in relevant if record["parsed"] is not None]
    source_names = {record["name"] for record in relevant}
    role_counts = {role: sum(SOURCE_ROLES.get(record["name"]) == role for record in records) for role in ("identity", "entity_state", "billing", "usage")}
    value_counts = {value: sum(value in record["text"] for record in records) for value in values}
    conflicting_argument_values = 0
    for key, value in arguments.items():
        observed = set()
        for record in records:
            parsed = record["parsed"]
            if isinstance(parsed, dict) and key in parsed:
                observed.add(str(parsed[key]))
        if observed and str(value) not in observed:
            conflicting_argument_values += 1
        if len(observed) > 1:
            conflicting_argument_values += len(observed) - 1
    source_tokens = sum((len(record["text"]) + 3) // 4 for record in records)
    relevant_tokens = sum((len(record["text"]) + 3) // 4 for record in relevant)
    return {
        "evidence_source_count": len(records),
        "evidence_relevant_source_count": len(relevant),
        "evidence_distractor_source_count": len(records) - len(relevant),
        "evidence_relevance_rate": len(relevant) / max(1, len(records)),
        "evidence_structured_relevant_count": len(structured_relevant),
        "evidence_value_corroboration_mean": sum(value_counts.values()) / max(1, len(values)),
        "evidence_value_corroboration_min": min(value_counts.values(), default=0),
        "evidence_argument_conflict_count": conflicting_argument_values,
        "evidence_source_token_cost": source_tokens,
        "evidence_relevant_token_cost": relevant_tokens,
        "evidence_read_cost": sum(SOURCE_COSTS.get(record["name"], 1.0) for record in records),
        "evidence_relevant_cost": sum(SOURCE_COSTS.get(record["name"], 1.0) for record in relevant),
        "evidence_identity_sources": role_counts["identity"],
        "evidence_entity_state_sources": role_counts["entity_state"],
        "evidence_billing_sources": role_counts["billing"],
        "evidence_usage_sources": role_counts["usage"],
        "evidence_candidate_entity_scope": int(len(relevant) > 0 and len(value_counts) == len(values)),
        "evidence_fact_complete": int(len(structured_relevant) > 0 and len(value_counts) == len(values)),
    }


def reference_candidates(task):
    output = []
    seen = set()
    for action in task.evaluation_criteria.actions or []:
        if action.requestor != "assistant":
            continue
        candidate = {"name": action.name, "arguments": action.arguments, "requestor": "assistant"}
        key = candidate_key(candidate)
        if key not in seen:
            output.append(candidate)
            seen.add(key)
    return output


def candidate_key(candidate):
    return json.dumps({"name": candidate["name"], "arguments": candidate.get("arguments") or {}}, sort_keys=True)


def entity_pools():
    db = get_environment().tools.db
    customers = [item.customer_id for item in db.customers]
    lines = [item.line_id for item in db.lines]
    bills = [item.bill_id for item in db.bills]
    devices = [item.device_id for item in db.devices]
    phones = [item.phone_number for item in db.lines] + [item.phone_number for item in db.customers]
    return {"customer_id": customers, "line_id": lines, "bill_id": bills, "device_id": devices, "phone_number": list(dict.fromkeys(phones)), "id": customers + lines + bills + devices}


def perturbations(candidate, pools):
    output = []
    for key, value in (candidate.get("arguments") or {}).items():
        alternatives = []
        if key in {"customer_id", "line_id", "bill_id", "device_id", "phone_number", "id"} and isinstance(value, str):
            alternatives = [item for item in pools.get(key, []) if item != value]
            if key == "id":
                alternatives = [item for item in alternatives if item[:1] == value[:1]]
            alternatives = alternatives[:3]
        elif key == "gb_amount" and isinstance(value, (int, float)):
            alternatives = [item for item in dict.fromkeys([1.0, round(float(value) / 2, 2), round(float(value) + 1, 2)]) if item > 0 and item != value]
        for alternate in alternatives:
            args = dict(candidate.get("arguments") or {})
            args[key] = alternate
            output.append({"name": candidate["name"], "arguments": args, "requestor": "assistant", "perturbation_type": f"replace_{key}", "original_value": value, "alternate_value": alternate})
    return output


def source_events(prior, tool_types):
    events = []
    for index, message in enumerate(prior):
        calls = message.get("tool_calls") or []
        reads = [call for call in calls if tool_types.get(call.get("name")) == "read"]
        if message.get("role") != "assistant" or not reads:
            continue
        end = index + 1
        if end < len(prior) and prior[end].get("role") == "tool":
            end += 1
        events.append({"source_event_index": len(events), "source_message_index": index, "source_end_index": end, "source_tool": reads[0]["name"], "source_call": reads[0]})
    return events
