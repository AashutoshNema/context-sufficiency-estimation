# Context Sufficiency Estimation

This repository studies a practical question for tool-using agents:

> Can an agent determine whether its current evidence is sufficient for a specific action, identify what is missing, acquire the most valuable additional context, and act safely at lower cost than broad retrieval?

The corrected validation branch evaluates that question in two controlled telecom replay environments:

- **OpenRCA Telecom** for telemetry, traces, temporal completeness, source selection, and component-triage actions.
- **Tau2 Telecom** for probing, escalation, entity ownership, source semantics, and end-to-end support actions.

## Current research conclusion

**Conditional GO for the controlled-replay claim; NO-GO for an unrestricted production-general claim.**

The experiments support the following scope-correct statement:

> An action-oriented context controller can improve reliability and efficiency in controlled, partially observed telecom environments by estimating action sufficiency, enforcing temporal, entity, and source constraints, and selectively acquiring the most valuable system or human evidence before acting. Broader production generalization and high-coverage calibration remain open research questions.

The evidence does **not** establish universal calibration, full OpenRCA component/reason/time diagnosis, or generalization to live users and arbitrary enterprise sources.

## Headline results

| Experiment | Selective policy | Broad-context comparison | Main finding |
|---|---:|---:|---|
| OpenRCA canonical component triage | 84.3% top-3 at cost 2.56 | 76.5% at cost 6.80 | Selecting two useful sources is more accurate and cheaper than using all extracted sources. |
| OpenRCA intervention suite | 79.1% acceptable action at cost 3.18 | 73.9% at cost 7.47 | Selective acquisition remains favorable under stale, unavailable, costly, wrong-entity, and temporal-gap conditions. |
| OpenRCA temporal selection | 76.5% accuracy at cost 0.88 | 78.4% at cost 4.50 | A learned interval selector preserves most of the available accuracy at substantially lower cost. |
| Tau2 probing and escalation | 100% controlled resolution at cost 8.05 | 100% at cost 13.17 | Conditional probing and escalation reduce interaction cost while preserving replay resolution. |
| Tau2 entity constraints | 84.1% safe action, 0% wrong asset | Mechanical selection reaches 82.0% with entity errors | Explicit ownership constraints prevent identifiable cross-asset failures. |
| Tau2 source semantics | 0 unsafe actions for the guarded semantic router | Prediction-only is unsafe on 96.8% | Source authority must be represented explicitly rather than averaged into a generic relevance score. |

More context is not monotonically better: OpenRCA performance peaks at two selected sources and declines as additional sources are added. Source selection is currently stronger than stopping calibration; the sufficiency gate provides a useful risk-coverage trade-off but is not ready for high-coverage deployment.

## Recommended reading order

1. [`RESEARCH_GUARDRAILS.md`](RESEARCH_GUARDRAILS.md) — north-star question, POMDP framing, required comparisons, and anti-drift gates.
2. [`artifacts/RESEARCH_VALIDITY_REPORT.md`](artifacts/RESEARCH_VALIDITY_REPORT.md) — executive decision, supporting evidence, falsification audit, and limitations.
3. [`EXPERIMENT_COVERAGE.md`](EXPERIMENT_COVERAGE.md) — coverage of all eight research tracks and remaining external-validity work.
4. [`artifacts/CANONICAL_BENCHMARKS.md`](artifacts/CANONICAL_BENCHMARKS.md) — original Tau2 and OpenRCA baselines.
5. Detailed OpenRCA reports:
   - [`OPENRCA_ACTION_MODEL.md`](artifacts/OPENRCA_ACTION_MODEL.md)
   - [`OPENRCA_CONTEXT_CONTROL_V2.md`](artifacts/OPENRCA_CONTEXT_CONTROL_V2.md)
   - [`OPENRCA_TEMPORAL_VERSIONS.md`](artifacts/OPENRCA_TEMPORAL_VERSIONS.md)
   - [`OPENRCA_TELECOM_RESEARCH_LOCK.md`](artifacts/OPENRCA_TELECOM_RESEARCH_LOCK.md)
6. Complementary Tau2 reports:
   - [`TAU2_PROBING_ESCALATION.md`](artifacts/TAU2_PROBING_ESCALATION.md)
   - [`TAU2_ENTITY_CONTEXT.md`](artifacts/TAU2_ENTITY_CONTEXT.md)
   - [`TAU2_SOURCE_SEMANTICS.md`](artifacts/TAU2_SOURCE_SEMANTICS.md)
   - [`TAU2_LLM_SELF_ASSESSMENT.md`](artifacts/TAU2_LLM_SELF_ASSESSMENT.md)

## Experiment design

The work treats context gathering as an action-conditioned, cost-aware sequential decision problem. Each controller experiment identifies:

- a hidden world state;
- the agent's initial partial observation;
- available evidence sources and their costs;
- acquisition, abstention, escalation, and task actions;
- evidence or belief updates;
- stopping conditions; and
- utility combining action reliability with acquisition cost.

Required comparisons include no additional context, always-fetch, fixed top-k, a logistic sufficiency gate, a source ranker without gating, gate plus ranker, and an oracle selector where feasible. Results report action quality and efficiency together rather than treating retrieval quality as the endpoint.

## Repository structure

```text
.
├── RESEARCH_GUARDRAILS.md       # Research contract and anti-drift gates
├── EXPERIMENT_COVERAGE.md       # Track and baseline coverage audit
├── artifacts/
│   ├── *.md                     # Human-readable experiment reports
│   ├── *.json                   # Machine-readable result summaries
│   └── experiment_manifest.json # Reproducibility manifest
└── experiments/
    ├── openrca_*.py             # OpenRCA baselines and controllers
    ├── tau2_*.py                # Tau2 controlled replay experiments
    └── verify_research_artifacts.py
```

Raw OpenRCA telemetry and large generated replay tables are intentionally excluded from Git. The committed JSON files contain the lightweight result summaries needed by the artifact verifier.

## Verify the committed results

From the repository root, run:

```bash
python3 experiments/verify_research_artifacts.py
```

Expected output:

```text
research artifact verification: PASS
verified: canonical action, source selection, risk-coverage, correlation, temporal coverage, probing/escalation, entity scope, source semantics, LLM baseline, manifest
```

This checks the internal consistency of the committed reports and machine-readable outputs. Re-running raw-data extraction requires the separately obtained OpenRCA and pinned Tau2 datasets.

## What remains open

- calibration at high action coverage;
- live free-form user and technician responses;
- production source prices, latency, permissions, and failures;
- larger causal asset graphs and natural cross-document coreference;
- full-text policy and procedure reasoning;
- stronger history-source routing; and
- full OpenRCA component/reason/time diagnosis.

These are external-validity and method-development steps. They should not be presented as already validated by the controlled replay results in this branch.
