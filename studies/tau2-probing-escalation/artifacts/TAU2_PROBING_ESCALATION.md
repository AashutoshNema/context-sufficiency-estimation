# Tau2 Telecom probing and escalation replay

## North-star subclaim

Can a context controller route an information need to the user/device, structured telecom system, or human support channel, update after an ambiguous response, and preserve task resolution at lower interaction cost than broad querying or unconditional escalation?

Tracks: `probing/escalation`, `heterogeneous grounding`, and `cost-efficiency`.

## Method

The experiment uses 114 fixed Tau2 Telecom tasks from the pinned benchmark revision. Required source roles are derived from the benchmark’s original expected actions:

- `user`: user/device action or observation;
- `system`: structured telecom-system action;
- `human`: support-agent escalation.

The controller observes only the ticket, reason for call, and public task purpose. Expected actions, task identifiers, hidden state, and intervention metadata are excluded from model input. A TF-IDF logistic role router is evaluated with five-fold out-of-fold predictions stratified by required role set.

Each task is replayed under six paired conditions: canonical, user unavailable, system unavailable, delayed user, expensive system, and ambiguous user response. Ambiguous responses use a seeded 0.5 success probability shared across policies. Sequential policies may escalate after an unsuccessful user probe.

## Role-routing discrimination

| Role | Positive rate | AUROC | AUPRC |
|---|---:|---:|---:|
| User/device | 82.5% | 0.935 | 0.986 |
| Structured system | 62.3% | 0.755 | 0.771 |
| Human support | 17.5% | 0.935 | 0.615 |

## Aggregate policy results

Across 684 paired episodes:

| Policy | Resolution | Mean cost | Calls/turns | Probe precision | Escalation rate | Escalation precision |
|---|---:|---:|---:|---:|---:|---:|
| Always query all channels | 100% | 13.17 | 2.67 | 83.1% | 100% | 41.7% |
| Always escalate | 100% | 10.00 | 1.00 | 100% | 100% | 41.7% |
| Learned top-2 roles | 98.4% | 8.15 | 2.04 | 83.1% | 55.0% | 72.9% |
| Gated sequential router (0.4) | **100%** | **8.05** | **1.81** | **92.7%** | **56.9%** | **73.3%** |
| Oracle | 100% | 7.21 | 1.70 | 100% | 49.0% | 85.1% |

The gated router matches broad retrieval and unconditional escalation on resolution while reducing cost by 38.9% versus always-all and 19.5% versus always-escalate. It also raises escalation precision from 41.7% to 73.3%.

On canonical tasks, the gated router resolves 100% at cost 4.41, compared with 12.50 for always-all, 10.00 for always-escalate, and 3.61 for the oracle. In unavailable-channel conditions it escalates on 100% of episodes and preserves resolution.

## Interpretation

This is positive evidence that heterogeneous source routing and sequential fallback can improve the reliability–interaction-cost frontier. The result also confirms why unconditional escalation must be included: when human cost was initially modeled too cheaply, always-escalate dominated and correctly falsified the first environment specification.

The result is not a complete human-interaction validation. Required roles are derived from benchmark action requestors, ambiguous answers are simulated, and human support is modeled as resolving the task. A future live-agent evaluation must measure free-form question quality, response ambiguity, repeated probing, and unsuccessful escalation.

Scope-valid claim:

> On fixed Tau2 Telecom tasks with controlled channel availability and cost, a learned sequential router preserves task-role resolution while reducing broad retrieval and unnecessary escalation.
