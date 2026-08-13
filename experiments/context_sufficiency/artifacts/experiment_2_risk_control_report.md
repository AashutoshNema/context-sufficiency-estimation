# Experiment 2 Correction: Decision-Level Risk Control

## Correction made

The prior policy calibrated safety per candidate row, then selected the highest-scoring candidate from a prefix. That calibration unit was wrong: deployment makes one maximum-score decision over a candidate set, which creates selection optimism.

This retry:

- calibrates the rich estimator with isotonic regression;
- collapses the calibration set to the candidate selected at each decision prefix;
- chooses a threshold using a one-sided beta upper bound on unsafe selected prefixes;
- requires structured evidence completeness and no observed argument conflict;
- ranks additional sources using evidence-completeness gain, conflict reduction, and unconstrained score gain;
- adds a fixed one-source baseline.

The train, calibration, and test partitions are separated by task group. Evaluation uses 23 held-out task groups and 380 decision prefixes.

## Dimensions represented

The telecom replay supports mechanical interventions for fact completeness, task-source relevance, entity scope, source semantics, provenance, source conflict/corroboration, acquisition cost, source masking/availability, context growth, and redundancy. Temporal freshness, version coverage, modality, authorization, human probing, and cross-asset dependency are not natively represented in this dataset and are not claimed as measured here.

## Held-out results

| Policy | Coverage | Safe rate | Safe rate when attempted | Mean reads | Mean tokens |
|---|---:|---:|---:|---:|---:|
| No read | 100.0% | 31.1% | 31.1% | 0.00 | 1,362 |
| Fixed top-1 source | 100.0% | 30.3% | 30.3% | 1.00 | 1,506 |
| Full context | 100.0% | **34.2%** | **34.2%** | 4.27 | 1,800 |
| Risk-controlled selective, alpha 0.20 | 51.3% | 7.1% | 13.8% | 1.56 | 1,557 |
| Risk-controlled selective, alpha 0.30-0.50 | 51.3% | 7.1% | 13.8% | 1.56 | 1,557 |

The rich estimator itself remains useful:

- AUROC: `0.843`
- AUPRC: `0.284`
- positive rate: `0.061`

Decision-level calibration was too sparse to support a low-risk threshold: at alpha `0.20`, no calibration prefix met the minimum sample requirement; at alpha `0.30-0.50`, only 11 prefixes were selected at threshold `1.0`. The held-out selective policy was still overconfident after source masking.

## Interpretation

This correction strengthens the negative result for the current controller. The representation has signal, but the present acquisition policy does not convert that signal into safe decisions. It should not be presented as improving safety or reducing context cost.

The defensible claim is narrower:

> Mechanically observable evidence structure improves held-out discrimination of whether a candidate action is safe, but candidate-level risk calibration and greedy source acquisition do not yet provide reliable selective context calling.

The next controller experiment should train source value and action safety as separate decision-level objectives, with calibration data generated under the same masked-context policy used at deployment. Until that is done, the controller portion remains a no-go while the sufficiency-estimation portion remains promising.
