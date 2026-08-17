# OpenRCA leakage-safe canonical action model

## Purpose

Establish whether source context can support a competent action model before evaluating selective acquisition. The action is a top-3 root-component triage shortlist.

## Method

Candidate-level features were extracted from raw OpenRCA Telecom telemetry for all 51 incidents:

- robust level and temporal-change features by metric family;
- trace volume, latency, maximum latency, and failure rate;
- application call, latency, and failure summaries;
- middleware anomaly, change, and entity summaries;
- a historical component-frequency prior computed only from training dates.

Each test date is held out in full. Candidate-ranking logistic models are trained on the other dates. Root component, reason, root timestamp, and intervention identity are not input features.

## Results

| Context | Top-1 | Top-3 | MRR |
|---|---:|---:|---:|
| Historical prior only | 27.5% | 54.9% | 0.495 |
| Prior + metrics | 37.3% | 64.7% | 0.568 |
| Prior + traces | 58.8% | 80.4% | 0.717 |
| Prior + metrics + traces | **60.8%** | **84.3%** | **0.739** |
| All extracted features | 49.0% | 76.5% | 0.653 |

Metrics plus traces improve top-3 accuracy by 29.4 percentage points over the prior. In paired case-level comparison there are 15 gains, zero losses, and 36 ties; the bootstrap 95% interval for improvement is 17.6–43.1 percentage points and the exact paired binomial p-value is 0.000061.

Metric-plus-trace top-3 accuracy by root level is 70.0% for node (14/20), 89.5% for pod (17/19), and 100% for service (12/12).

## Gate decision

The canonical action-model gate passes: relevant full context materially and significantly outperforms the leakage-safe no-context prior. OpenRCA is now suitable for a selective metric/trace acquisition experiment for component triage.

Application and middleware summaries reduce held-out performance when added without topology mappings. They are useful as explicit irrelevant or unmapped sources in the context-control experiment, but cannot yet support a positive component-localization claim.
