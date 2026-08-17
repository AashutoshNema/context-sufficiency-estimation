# Study: Tau2 selective context controller

## Question

Can a controller begin without prior reads, selectively reacquire telecom sources, and match full-context action safety with fewer calls and tokens?

## Status

**NO-GO.** The representation predicts candidate safety, but the acquisition policy does not convert that signal into safe selective action.

On 23 held-out task groups and 380 decision prefixes, full context reaches 34.2% safe selected actions. The risk-controlled selective policy reaches 13.8% safety when attempted at 51.3% coverage. See [`artifacts/experiment_2_risk_control_report.md`](artifacts/experiment_2_risk_control_report.md).

## Independent scope

This study owns the original hard-counterfactual, source-ablation, closed-loop, rich-feature, and decision-level risk-control experiments. Later Tau2 mechanism studies do not modify or supersede this result.

## Inputs

- Tau2 benchmark revision `668d3bcd135c02aa3438f987ef45735b7c163ee3`.
- GPT-4.1 Telecom result with 456 trajectories over 114 tasks.
- Study scope is recorded in [`study_manifest.json`](study_manifest.json).
- Input and sparse-checkout details are pinned in [`benchmark_manifest.json`](benchmark_manifest.json).

Raw benchmark data and generated JSONL rows are excluded from Git.

## Reproduce

Run from the repository root:

```bash
python3 studies/tau2-selective-controller/experiments/prepare_benchmark.py
uv sync --frozen --project .benchmark/tau2-bench --extra experiments
uv run --project .benchmark/tau2-bench python \
  studies/tau2-selective-controller/experiments/replicate_telecom.py
```

Generated outputs go to `studies/tau2-selective-controller/artifacts/reproduction/`.

## Key artifacts

- [`experiment_2_report.md`](artifacts/experiment_2_report.md) — original closed-loop result.
- [`experiment_2_rich_retry_report.md`](artifacts/experiment_2_rich_retry_report.md) — richer evidence representation.
- [`experiment_2_risk_control_report.md`](artifacts/experiment_2_risk_control_report.md) — corrected decision-level risk control.
- [`source_ablation_report.md`](artifacts/source_ablation_report.md) — paired source-removal analysis.

## Falsification outcome

The selective controller fails the required full-context comparison. Reduced reads alone are not evidence of context sufficiency when action safety falls.
