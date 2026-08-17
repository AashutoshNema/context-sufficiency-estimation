# Tau2 multi-asset and entity-scope experiment

## Research question

Can an action-conditioned controller preserve customer/line scope and avoid wrong-asset actions when context contains independent distractors, connected assets, ambiguous identity evidence, or missing target evidence?

Track: `multi-asset/entity-aware context`.

## Method

The experiment uses the pinned Tau2 Telecom hard-counterfactual replay and its fixed hidden environment. Candidate actions are evaluated under seven paired inventories:

- canonical all candidates;
- single asset;
- connected lines belonging to the target customer;
- independent customer/line distractors;
- ambiguous user identity evidence;
- missing structured target evidence;
- customer-plus-line cross-asset actions.

Customer-to-line ownership is loaded from the benchmark database. Candidate argument provenance is computed from prior user and tool messages. Models are evaluated out of task group with five-fold GroupKFold; safety labels and intervention types are not input features.

## Aggregate results

Across 5,682 held-out decision/condition groups:

| Policy | Safe-action rate | Wrong-asset rate | Ownership-valid rate |
|---|---:|---:|---:|
| Random candidate | 25.9% | 58.2% | 61.1% |
| Mechanical context | 82.0% | 2.8% | 97.3% |
| Provenance model | 71.1% | 1.1% | 99.0% |
| Hard entity gate | 71.8% | **0%** | **100%** |
| Entity-aware model | **84.1%** | **0%** | **100%** |
| Oracle | 100% | 0% | 100% |

The entity-aware model improves aggregate safety by 2.1 percentage points over the mechanical model while eliminating wrong-asset actions.

The independent-asset condition is the decisive stress test:

- mechanical safety: 82.0%;
- mechanical wrong-asset rate: 18.0%;
- entity-aware safety: 100%;
- entity-aware wrong-asset rate: 0%.

In connected and cross-asset conditions, the mechanical model is slightly more accurate (82.1% versus 79.9%), but the entity-aware model maintains perfect ownership validity. Hard gating alone prevents wrong-asset actions but reduces safety, showing that entity constraints and learned action ranking are complementary.

## Interpretation

This is positive evidence for explicit entity and asset coverage. Aggregate context features can appear strong while still selecting a different customer or line under independent distractors. A hard ownership constraint removes that failure mode, and a learned entity-aware ranker recovers most of the action performance lost by gating alone.

Limitations:

- the benchmark graph contains four customers and nine lines;
- connectedness is account ownership rather than a general causal or network topology;
- provenance uses exact scalar-value matches;
- ambiguous references are produced by visibility masking rather than natural coreference dialogue.

Scope-valid claim:

> Explicit entity scope and ownership constraints prevent wrong-asset actions under multi-asset context without sacrificing aggregate action reliability.
