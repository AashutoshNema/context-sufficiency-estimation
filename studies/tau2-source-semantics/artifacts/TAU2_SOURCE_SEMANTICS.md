# Tau2 heterogeneous source-semantics experiment

## Research question

Can a context controller route evidence according to source role and avoid treating a prediction as equivalent to an observation, authoritative record, procedure, policy, or history?

Track: `heterogeneous grounding sources`.

## Method

The 114 fixed Tau2 Telecom tasks are mapped to five authoritative evidence roles from their original expected actions and fault families:

- observation: user/device fact;
- structured record: telecom database state;
- procedure: troubleshooting workflow;
- policy: authorization or escalation rule;
- history: billing, usage, or contract history.

A sixth source is a non-authoritative model prediction. Prediction correctness is a seeded intervention: 80% normally and 20% in the explicit conflict condition.

The router observes only ticket and public task text. Five-fold out-of-fold role predictions are evaluated under canonical, policy-unavailable, procedure-unavailable, history-unavailable, irrelevant-inventory, and prediction-conflict conditions.

## Role discrimination

| Role | AUROC | AUPRC |
|---|---:|---:|
| Observation | 0.931 | 0.986 |
| Structured record | 0.741 | 0.782 |
| Procedure | 0.931 | 0.986 |
| Policy | 0.605 | 0.842 |
| History | 0.516 | 0.642 |

History routing is effectively unlearned by AUROC and remains a weakness.

## Aggregate results

Across 684 task/condition episodes:

| Policy | Resolution | Coverage | Unsafe-action rate | Mean cost | Irrelevant-call rate | Prediction overtrust |
|---|---:|---:|---:|---:|---:|---:|
| Always all sources | 62.6% | 100% | 37.4% | 6.33 | 21.5% | 0% |
| Relevance top-3 | 35.1% | 100% | 64.9% | 3.63 | 15.8% | 0% |
| Prediction only | 3.2% | 100% | 96.8% | 0.50 | 0% | 29.2% |
| Naive prediction-as-evidence | 3.2% | 100% | 96.8% | 2.05 | 8.3% | 29.2% |
| Semantic router | 52.9% | 52.9% | **0%** | 4.82 | 16.3% | **0%** |
| Semantic router plus prediction | 52.9% | 52.9% | **0%** | 5.33 | 12.9% | **0%** |
| Oracle | 62.6% | 62.6% | 0% | 4.25 | 0% | 0% |

On canonical tasks, always-all resolves 100% at cost 7.00. The semantic router resolves 82.5%, abstains on the remainder, has zero unsafe actions, and costs 5.42. In the prediction-conflict condition it maintains the same 82.5% resolution and zero overtrust; prediction-only falls to 0.9% resolution.

Unavailable authoritative sources explain why always-all resolves only 62.6% in aggregate while still acting unsafely. The semantic router converts those cases into abstentions rather than pretending that a recommendation replaces missing evidence.

## Interpretation

This experiment supports the source-semantics part of the research claim: explicit evidentiary roles prevent model recommendations from being treated as facts and turn missing authoritative evidence into safe abstention. Generic relevance ranking is not enough.

It is not a full textual grounding evaluation. Policy and procedure contents are represented by source-role availability, prediction conflicts are synthetic, and the history router is weak. The next version should retrieve and reason over actual policy/manual passages and archived model outputs.

Scope-valid claim:

> Explicit source semantics prevent prediction overtrust and unsafe action when authoritative observations, records, procedures, policies, or history are unavailable.
