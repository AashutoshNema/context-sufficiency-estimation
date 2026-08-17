# Context sufficiency research validity report

## Decision

**Conditional GO for the controlled-replay claim; NO-GO for an unrestricted production-general claim.**

The completed experiments support this statement:

> Across fixed-world telecom replays, an action-conditioned controller can improve the reliability–efficiency frontier by selecting temporally complete, entity-correct, semantically authoritative evidence and by probing or escalating selectively rather than acquiring all available context.

The evidence does not yet support claiming that the controller is universally calibrated, that it solves full OpenRCA reason/time RCA, or that it generalizes to live users and arbitrary enterprise sources.

## Evidence by north-star track

| Track | Main held-out result | Decision |
|---|---|---|
| Canonical | Tau2 task success 34.2%; OpenRCA metric+trace top-3 84.3% | baseline established |
| Observation interventions | Seven manifested paired suites with fixed hidden worlds and seed 42 | covered |
| Temporal | Learned interval selector preserves 76.5% accuracy at cost 0.88 versus 4.5 for all intervals | positive |
| Correlated/disjoint sources | Duplicate evidence adds 0 accuracy and no confidence; independent traces add 19.6 points | positive |
| Multi-asset/entity | Entity-aware safety 84.1%, wrong-asset rate 0%; mechanical wrong-asset rate 18% under independent distractors | positive |
| Probing/escalation | 100% role resolution at cost 8.05 versus 13.17 always-all and 10.00 always-escalate | positive controlled replay |
| Heterogeneous semantics | Semantic router has 0 unsafe actions and 0 prediction overtrust; prediction-only is unsafe on 96.8% | positive risk control |
| Cost frontier | Learned two-source OpenRCA policy 79.1% at cost 3.18 versus always-fetch 73.9% at cost 7.47 | positive |

## Core controller results

On canonical OpenRCA component triage, learned source selection reaches 84.3% at cost 2.56 versus 76.5% at cost 6.80 for always-fetch. Across stale, unavailable, costly, wrong-entity, and temporal-gap conditions, the two-source ranker reaches 79.1% at cost 3.18 versus 73.9% at cost 7.47.

The sufficiency gate creates a genuine risk–coverage curve: attempted accuracy rises from 71.5% at 95.1% coverage to 98.9% at 29.7% coverage, while unsafe-action rate falls from 27.1% to 0.3%. Its ECE is 0.143, so high-coverage stopping is not yet deployment-calibrated.

The archived GPT-4.1 agent is not an adequate sufficiency gate: 46.3% of its attempted writes are false-sufficient and it continues gathering context on 77.4% of prefixes where a safe write candidate already exists.

## Falsification audit

| North-star falsification condition | Evidence | Result |
|---|---|---|
| Always-fetch dominates at equal cost | Selective OpenRCA and Tau2 probing policies are both cheaper and at least as reliable | not triggered |
| Logistic regression equals every sequential controller | Simple models are sufficient for several controlled tasks; no advantage from a more complex POMDP model is established | complexity claim intentionally not made |
| Retrieval improves but actions do not | OpenRCA action correctness improves 5.2 points aggregate and 7.8 points canonical over always-fetch | not triggered |
| Method works only on synthetic interventions | Canonical raw OpenRCA metric/trace and raw temporal-window gains are positive | not triggered |
| Calls increase without reducing incorrect actions | Two-source ranker uses fewer calls and lowers incorrect actions; k>2 degrades and is rejected | not triggered |
| Cannot distinguish stale/current context | Raw temporal accuracy: stale 52.9%, partial 60.8%, complete current 76.5%; guarded policy rejects stale sources | not triggered with explicit guards |
| Correlated sources are double-counted | Duplicate source changes accuracy by 0 and confidence by -0.013 | not triggered in tested duplicate condition |
| Acts when required asset/time/policy is missing | Entity and semantic hard gates reduce these cases to abstention; unconstrained ranker fails and is not accepted | not triggered for guarded policy |

## What is validated

- Context sufficiency is action-conditioned: the useful source set changes with component triage, temporal coverage, entity ownership, source role, and interaction channel.
- More context is not monotonically better: OpenRCA ranker accuracy peaks at two sources and declines through k=5.
- Explicit hard requirements matter: freshness, entity ownership, and source authority cannot be replaced by an averaged confidence score.
- Source selection and stopping are separate objectives: source ranking is strong, while high-coverage stopping remains calibration-limited.
- Selective probing can reduce user/support burden and unnecessary escalation while preserving controlled task resolution.

## What remains unvalidated

- live free-form user or technician responses;
- production source prices, latency distributions, permissions, and failure modes;
- large causal asset graphs and natural cross-document coreference;
- full-text policy/procedure reasoning rather than role-level availability;
- strong history-source routing (history AUROC is 0.516);
- full OpenRCA component/reason/time diagnosis; official time accuracy remains 0/31;
- generalization beyond the two telecom environments.

## Final scope-correct claim

> An action-oriented context controller can improve reliability and efficiency in controlled, partially observed telecom environments by estimating action sufficiency, enforcing temporal/entity/source constraints, and selectively acquiring the most valuable system or human evidence before acting. Broader production generalization and high-coverage calibration remain open research questions.
