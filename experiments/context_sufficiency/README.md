# Telecom Context Sufficiency Experiments

The scripts in this directory replay completed telecom trajectories against reconstructed hidden
environment state. The hard-candidate pass creates schema-valid entity and amount perturbations;
the source-ablation pass removes one prior read call and its result from the observable context.

Run hard candidates:

```bash
python3 experiments/context_sufficiency/hard_counterfactuals.py \
  --results data/tau2/results/final/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json \
  --tool-source src/tau2/domains/telecom/tools.py \
  --output-dir experiments/context_sufficiency/artifacts/hard_counterfactual_full
```

Run source ablation:

```bash
python3 experiments/context_sufficiency/source_ablation.py \
  --results data/tau2/results/final/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json \
  --hard-rows experiments/context_sufficiency/artifacts/hard_counterfactual_full/telecom_hard_counterfactual_rows.jsonl \
  --tool-source src/tau2/domains/telecom/tools.py \
  --output-dir experiments/context_sufficiency/artifacts/source_ablation_full \
  --hard-only
```

The current source-ablation report is in `artifacts/source_ablation_report.md`.

Run the closed-loop controller evaluation:

```bash
python3 experiments/context_sufficiency/closed_loop_controller.py \
  --results data/tau2/results/final/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json \
  --hard-rows experiments/context_sufficiency/artifacts/hard_counterfactual_full/telecom_hard_counterfactual_rows.jsonl \
  --tool-source src/tau2/domains/telecom/tools.py \
  --output experiments/context_sufficiency/artifacts/closed_loop_controller_provenance_report.json
```

The current decision report is `artifacts/experiment_2_report.md`.

The dimension-aware retry is reported in `artifacts/experiment_2_rich_retry_report.md`; its output
is `artifacts/closed_loop_controller_rich_report.json`.

Run the corrected decision-level risk-control evaluation:

```bash
python3 experiments/context_sufficiency/risk_controlled_controller.py \
  --results data/tau2/results/final/gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json \
  --hard-rows experiments/context_sufficiency/artifacts/hard_counterfactual_full/telecom_hard_counterfactual_rows.jsonl \
  --tool-source src/tau2/domains/telecom/tools.py \
  --output experiments/context_sufficiency/artifacts/risk_controlled_controller_report_v3.json \
  --alpha 0.20,0.30,0.40,0.50
```

The corrected result is in `artifacts/experiment_2_risk_control_report.md`.
