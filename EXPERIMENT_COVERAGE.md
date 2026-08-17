# North-Star Experiment Coverage Audit

Status as of 2026-08-14. An experiment is `complete` only when its code, held-out results, baselines, leakage policy, and scope-limited conclusion are reproducible.

| Track | Current evidence | Status | Missing proof |
|---|---|---|---|
| 1. Canonical benchmark | Consolidated Tau2 456-trajectory task success/calls/tokens/cost/latency plus held-out controller split and OpenRCA canonical action baselines | complete |
| 2. Observation interventions | Manifested paired suite covers hidden, stale, partial, unavailable, delayed, expensive, irrelevant, ambiguous, wrong-entity, and conflicting observations with fixed worlds and published seed | complete for controlled replay |
| 3. Temporal/telemetry | Four raw temporal intervals per incident; stale/partial/current/history accuracy and selector cost measured; official root-time inference remains 0/31 and is a locked negative result | complete for available OpenRCA temporal endpoints | Better root-time/reason models are future method work, not missing evaluation |
| 4. Disjoint/correlated sources | Explicit correlated metric duplicate versus independent trace audit; duplicate adds 0 accuracy and -0.013 confidence while trace adds 19.6 points and +0.241 confidence | complete for controlled replay |
| 5. Multi-asset/entity | Tau2 single, connected, independent, ambiguous, missing-evidence, and cross-asset conditions over 5,682 held-out decisions; entity-aware model eliminates wrong-asset actions | complete for Tau2 ownership scope | Larger causal topologies and natural coreference remain external-validity extensions |
| 6. Probing/escalation | Tau2 role-routing replay over 114 tasks × six paired conditions; user/system/human actions, ambiguity, availability, cost, sequential fallback, and unconditional-escalation baseline | complete for controlled replay | Free-form live-human behavior remains external validity |
| 7. Heterogeneous grounding | Controlled Tau2 observation, record, procedure, policy, history, and prediction roles; semantic router has zero unsafe actions and prediction overtrust | complete for role-level replay | Full-text policy/manual reasoning and archived prediction sources remain external-validity extensions |
| 8. Cost/efficiency frontier | OpenRCA k=1..5 budgets and 0.30..0.90 risk–coverage sweep plus Tau2 calls/tokens/model cost/latency and user-interaction/escalation costs | complete for controlled replay | Production monetary and human-burden calibration remains external validity |

## Baseline coverage

| Baseline | Status |
|---|---|
| No additional context | implemented in Tau2 |
| Always fetch all | implemented in Tau2 |
| Fixed top-k | k=1 through k=5 implemented in OpenRCA v2; top-1 in Tau2 |
| Freshness-only | implemented in OpenRCA v1 and explicit temporal-version selector |
| Relevance-only | isolated as relevance top-3 in heterogeneous-source replay |
| LLM self-assessed sufficiency | implemented from archived GPT-4.1 read/write behavior over 2,007 prefixes; 46.3% false-sufficient when acting and 77.4% unnecessary-query rate among sufficient prefixes |
| Logistic sufficiency gate | implemented in both benchmarks; OpenRCA AUROC 0.748 with explicit risk–coverage curve |
| Source ranker without gate | implemented separately in OpenRCA v2; k=2 beats always-fetch in reliability and cost |
| Gate plus source ranker | deployment-trajectory calibrated in OpenRCA v2; positive risk reduction but low coverage |
| Oracle selector | implemented in OpenRCA v2 |
| Sequential/POMDP controller | implemented in OpenRCA v2; source selection positive and stopping exposes calibrated risk–coverage trade-off |

## Current defensible claim

Across two controlled telecom replay environments, action-conditioned source selection improves the reliability–efficiency frontier, explicit temporal/entity/source-role constraints prevent identifiable unsafe failures, and sufficiency gating provides a tunable risk–coverage trade-off. The evidence supports the claim for controlled replay, not yet for unconstrained production agents or live human interaction.
