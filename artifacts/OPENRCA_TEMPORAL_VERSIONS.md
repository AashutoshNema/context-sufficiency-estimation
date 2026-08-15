# OpenRCA temporal version and coverage experiment

## Research question

Can a context controller distinguish stale, incomplete, current, and extended historical telemetry, and acquire enough temporal coverage without always retrieving every available interval?

Track: `temporal/telemetry` and `cost-efficiency`.

## Method

Four observations were extracted directly from raw telemetry for each of the 51 fixed incidents:

- previous 30 minutes: stale interval;
- first 10 minutes of the incident interval: incomplete current coverage;
- complete current 30 minutes;
- 60-minute history combining previous and complete current intervals.

Candidate-level metric, temporal-change, trace, application, and middleware features were recomputed separately for every interval. The action model and temporal selector use leave-one-telemetry-date-out evaluation. Root component, reason, root timestamp, and intervention identity are not inputs.

## Results

| Temporal observation | Top-3 component accuracy | Cost |
|---|---:|---:|
| Previous 30 minutes | 52.9% | 1.0 |
| Partial current 10 minutes | 60.8% | 0.5 |
| Complete current 30 minutes | **76.5%** | 1.0 |
| Combined 60-minute history | 70.6% | 2.0 |
| Learned temporal selector | **76.5%** | **0.88** |
| Retrieve all intervals / oracle union | 78.4% | 4.5 |

The learned selector chooses the complete current interval for 39 incidents and the cheaper partial interval for 12. It matches complete-current accuracy at 11.8% lower cost and stays within 2.0 points of the all-interval union at 80.4% lower cost.

Complete current coverage improves over stale telemetry by 23.5 percentage points (13 paired gains, one loss; bootstrap 95% interval 11.8–37.3). It improves over the partial current interval by 15.7 points (nine gains, one loss; interval 3.9–27.5). Extended history is not monotonically better: it is 5.9 points below complete current coverage.

## Interpretation

This is positive evidence for temporal coverage control. Freshness alone is insufficient because a fresh but partial interval loses 15.7 points. More history is also not automatically better because the 60-minute interval dilutes the incident signal. The controller must reason jointly about valid time, coverage completeness, and action requirements.

Limitations:

- intervals are deterministic telemetry slices rather than naturally versioned records;
- all tasks are incident-localization tasks rather than separately labeled current-state and trend-detection tasks;
- root-occurrence-time prediction remains weak and is not claimed as solved;
- the temporal selector uses observable incident level but not a learned semantic task parser.

Scope-valid claim:

> On OpenRCA component triage, selecting the action-appropriate temporal interval preserves complete-window reliability at substantially lower acquisition cost than retrieving every available interval.
