# OpenRCA ContextControl-v1

## North-star subclaim

Can a controller decide whether to acquire metric, trace, application, or middleware context before returning a top-3 root-component triage shortlist, while preserving reliability at lower acquisition cost than always-fetch?

Track: `controller` plus paired `intervention` evaluation.

## POMDP mapping

- Hidden state: the fixed OpenRCA incident and labeled root component.
- Initial observation: incident level and a train-fold component prior; no incident telemetry.
- Sources: metric, trace, application, middleware, and a correlated metric duplicate.
- Actions: acquire a source, stop and return a top-3 shortlist, or abstain.
- Observation update: source candidates plus freshness, entity scope, provenance group, cost, and latency.
- Utility evidence: acceptable shortlist, unsafe attempted shortlist, abstention, calls, cost, and latency.
- Stopping: logistic sufficiency threshold or acquisition limit.

## Integrity and split

All 51 incidents are evaluated out of date using leave-one-telemetry-date-out folds. For every test date, the preceding two dates calibrate the stopping threshold and the remaining eight dates train the models. Root-component labels are used only for training targets and evaluation. Intervention identity is not an input feature.

Paired conditions preserve the hidden incident and alter only observation/access conditions:

- canonical;
- trace unavailable;
- stale metric observation;
- wrong-entity metric observation;
- expensive/slow trace;
- temporal gap.

The sufficiency gate and source ranker are separate L2-logistic models. The source ranker sees the current evidence state and source-registry metadata before acquisition; it does not see the hidden source contents.

## Cross-validated results

The table aggregates 306 paired test episodes: 51 incidents under six conditions.

| Policy | Acceptable rate | Coverage | Accuracy when attempted | Mean calls | Mean cost |
|---|---:|---:|---:|---:|---:|
| No context | 56.9% | 100.0% | 56.9% | 0.00 | 0.00 |
| Always fetch | 23.5% | 100.0% | 23.5% | 4.83 | 7.47 |
| Fixed metric | 20.9% | 100.0% | 20.9% | 1.00 | 1.00 |
| Freshness-only | 21.6% | 100.0% | 21.6% | 1.00 | 0.80 |
| Random top-1 | 42.2% | 100.0% | 42.2% | 1.00 | 1.56 |
| Source ranker, two calls | 56.9% | 100.0% | 56.9% | 2.00 | 3.00 |
| Gate plus source ranker | 43.1% | 82.4% | 52.4% | 0.85 | 1.32 |
| Oracle cheapest sufficient subset | 82.0% | 82.0% | 100.0% | 1.13 | 1.65 |

On the 51 canonical episodes alone, no-context achieves 56.9%, always-fetch 27.5%, gate plus ranker 43.1%, and the oracle 84.3%.

The held-out sufficiency estimator has AUROC 0.752, AUPRC 0.452, Brier score 0.201, and ECE 0.167. It discriminates some sufficient states but is not calibrated well enough to gate safe action.

## Falsification result

This experiment triggers a north-star falsification warning: the no-context level-conditioned prior dominates every non-oracle telemetry policy in both reliability and cost. The current OpenRCA summaries displace useful prior candidates more often than they add decisive evidence. The gate reduces attempted coverage without improving attempted accuracy over no-context.

Therefore OpenRCA ContextControl-v1 does **not** validate selective context acquisition. It demonstrates that the replay machinery and oracle opportunity exist, but the current action model and source representation are too weak for a valid positive controller result.

## Scope and required correction

This result applies only to component top-3 triage from derived source summaries. It does not cover full component/reason/time RCA, raw source-content reasoning, human probing, authorization, policies, or natural conflicts.

Before another OpenRCA controller claim is attempted:

1. build a stronger leakage-safe canonical action model over raw source-specific telemetry;
2. add topology mappings so application, middleware, and trace entities can support root components;
3. require the always-fetch action baseline to outperform the no-context prior;
4. train source value under the same masked sequential policy used at evaluation;
5. calibrate stopping at the decision level and report risk-coverage curves.

Until gate 3 is met, OpenRCA remains a temporal/source-acquisition stress test rather than positive evidence for the research claim.
