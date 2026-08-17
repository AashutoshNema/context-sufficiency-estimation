# Study: OpenRCA source selection

## Question

Can a learned source-acquisition policy improve top-3 root-component triage reliability or preserve it at materially lower cost than broad and strong static context policies?

## Status

**Useful action signal; adaptive advantage not established.**

The supervised leave-one-date-out metric-plus-trace model reaches 84.3% top-3 accuracy. The learned router also reaches 84.3% on canonical incidents, but so does the static metric-plus-trace policy. Across all interventions, learned and static relevant-source policies both reach 79.1% acceptable actions.

The learned policy's lower assigned cost relies heavily on an exact duplicate of the metric source priced below the original. This is not sufficient evidence of meaningful per-incident adaptation.

## Two distinct action models

| Formulation | Top-1 | Top-3 | Interpretation |
|---|---:|---:|---|
| Unsupervised anomaly heuristic | 13.7% | 27.5% | Fixed metric/trace scoring without supervised training. |
| Supervised candidate ranker | 60.8% | 84.3% | Leave-one-date-out logistic ranking with historical priors and richer features. |

The results are not contradictory, but they must not be presented as the same “metrics + traces” model.

## Independent scope

This directory owns the complete OpenRCA source-selection test: raw telemetry baselines, candidate extraction, supervised action-model gate, and v1/v2 acquisition policies. It does not inherit a positive conclusion from the Tau2 studies.

## Layout

- `experiments/` — telemetry baselines, feature extraction, candidate ranking, and controllers.
- `artifacts/HEURISTIC_RESULTS.md` — unsupervised baseline.
- `artifacts/OPENRCA_ACTION_MODEL.md` — supervised action-model gate.
- `artifacts/OPENRCA_CONTEXT_CONTROL_V2.md` — source-selection and stopping results.
- `artifacts/*.json` — machine-readable summaries.

Raw OpenRCA telemetry and generated candidate/prediction CSVs are excluded from Git.

## Next required test

Remove the duplicate source and compare learned routing with cost-matched static, per-level, and fixed-relevant policies. A GO requires a real reliability gain or a material real-cost reduction without relying on synthetic duplicate pricing.
