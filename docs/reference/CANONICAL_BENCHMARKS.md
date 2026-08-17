# Canonical benchmark performance

## Tau2 Telecom

The pinned GPT-4.1 result contains 456 original trajectories over 114 tasks and four trials per task.

| Metric | Canonical result |
|---|---:|
| End-to-end task success | 34.2% |
| Mean assistant tool calls | 6.67 |
| Mean billed prompt + completion tokens | 218,525 |
| Mean agent cost | 0.110 |
| Mean simulated-user cost | 0.086 |
| Mean combined model cost | 0.196 |
| Mean duration | 118.6 seconds |
| User-stop termination | 453/456 |
| Maximum-step termination | 3/456 |

The end-to-end result is the mean official Tau2 reward: 156 successful rewards across 456 trajectories. The pinned raw result is intentionally excluded from Git, so this aggregate must be reproduced from the benchmark revision and result file listed below. It is distinct from the controller safety metric in the next table. The two percentages are both 34.2% because 156/456 and 130/380 reduce to the same fraction.

On the controller’s 23 held-out task groups and 380 write-decision prefixes, the reproduced policies achieve:

| Context condition | Safe selected action | Reads | Context tokens |
|---|---:|---:|---:|
| No read | 31.1% | 0.00 | 1,362 |
| Fixed top-1 source | 30.3% | 1.00 | 1,506 |
| Full context | 34.2% | 4.27 | 1,800 |

The canonical benchmark is difficult and broad retrieval provides only a small safety gain over no reads. This is the baseline that the Tau2 context controller must preserve or improve.

## OpenRCA Telecom

The original archived OpenRCA agent has mean official score 0.281 and strict all-criteria accuracy 12/51 (23.5%). The locked transparent heuristic baseline has official mean score 0.180 and strict accuracy 6/51 (11.8%).

For the narrower, explicitly action-conditioned component-triage task used by ContextControl-v2, the leakage-safe canonical model achieves:

| Context | Top-1 | Top-3 |
|---|---:|---:|
| Historical prior | 27.5% | 54.9% |
| Metrics | 37.3% | 64.7% |
| Traces | 58.8% | 80.4% |
| Metrics + traces | **60.8%** | **84.3%** |
| All extracted sources without topology mapping | 49.0% | 76.5% |

These are supervised leave-one-date-out candidate-ranking results. They are not the same formulation as the earlier unsupervised anomaly heuristic, which reached 13.7% top-1 and 27.5% top-3. The source-selection study documents both formulations explicitly.

The two benchmarks answer different action questions. Tau2 measures end-to-end support action safety; OpenRCA v2 measures a top-3 component shortlist. Their percentages must not be compared as if they were the same endpoint.

## Reproducibility

- Tau2 revision: `668d3bcd135c02aa3438f987ef45735b7c163ee3`.
- Tau2 original result: `gpt-4.1-2025-04-14_telecom_default_gpt-4.1-2025-04-14_4trials.json`.
- OpenRCA split: all 51 Telecom incidents, evaluated leave-one-telemetry-date-out.
