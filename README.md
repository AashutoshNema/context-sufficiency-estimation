# Context Sufficiency Estimation

This repository contains independent exploratory studies of one research question:

> Can an agent determine whether its current evidence is sufficient for a specific action, identify what is missing, acquire valuable context, and act safely at lower cost than broad retrieval?

## Current status

**NO-GO for the core learned on-demand context controller.**

The original Tau2 selective controller remains below the full-context baseline. OpenRCA establishes that metric and trace evidence can support component triage, but the learned source router does not outperform a strong static relevant-source policy in action accuracy. The repository contains useful supporting evidence for sufficiency prediction, temporal selection, entity constraints, source semantics, and probing/escalation, but these results do not yet validate the central adaptive-controller claim.

Read [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) for the evidence and exact scope.

## Repository organization

Every research question is isolated as a study:

```text
studies/<study-name>/
├── README.md       # question, hypothesis, status, method, and limitations
├── study_manifest.json # machine-readable scope and falsification contract
├── experiments/    # code owned by this study
└── artifacts/      # reports and machine-readable outputs owned by this study
```

Studies do not inherit another study's conclusion. If a study needs preprocessing or a baseline from earlier work, it freezes a local copy or declares the input explicitly. Cross-study synthesis belongs in `docs/`, never inside a study's result artifacts.

## Study index

| Study | Question | Current result |
|---|---|---|
| [`tau2-selective-controller`](studies/tau2-selective-controller/) | Can selective reacquisition match full context with fewer reads? | **NO-GO** |
| [`openrca-source-selection`](studies/openrca-source-selection/) | Can learned source selection beat broad and strong static context policies? | Signal positive; adaptive advantage **not established** |
| [`openrca-temporal-context`](studies/openrca-temporal-context/) | Which telemetry interval is sufficient for component triage? | Positive supporting result |
| [`tau2-probing-escalation`](studies/tau2-probing-escalation/) | Can an agent selectively probe users/systems and escalate? | Positive controlled replay |
| [`tau2-entity-scope`](studies/tau2-entity-scope/) | Do entity constraints prevent wrong-asset actions? | Positive controlled replay |
| [`tau2-source-semantics`](studies/tau2-source-semantics/) | Do explicit source roles prevent unsafe reliance on predictions? | Positive controlled replay |
| [`tau2-llm-self-assessment`](studies/tau2-llm-self-assessment/) | Can an archived LLM reliably self-assess context sufficiency? | Negative baseline |

Use [`studies/_template/`](studies/_template/) to begin a new study.

## Project-wide documents

- [`docs/RESEARCH_GUARDRAILS.md`](docs/RESEARCH_GUARDRAILS.md) — the research contract and anti-drift gates.
- [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md) — the active evidence-based verdict.
- [`docs/EXPERIMENT_COVERAGE.md`](docs/EXPERIMENT_COVERAGE.md) — historical coverage inventory, not a validity verdict.
- [`docs/reference/CANONICAL_BENCHMARKS.md`](docs/reference/CANONICAL_BENCHMARKS.md) — cross-study benchmark reference.
- [`docs/archive/`](docs/archive/) — superseded synthesis documents retained for audit history.

## Start a new study

1. Copy `studies/_template` to a new hypothesis-focused name.
2. Write the falsifiable question before adding code.
3. Declare raw inputs, split policy, costs, and leakage controls.
4. Generate outputs only inside that study's `artifacts/` directory.
5. Compare against no-context, broad-context, and strong static baselines.
6. Record a local GO/NO-GO result without changing another study's conclusion.

## Verify the repository

From the repository root:

```bash
python3 docs/verify_research_artifacts.py
```

The verifier checks the reorganized artifact locations and selected internal claims. It is an artifact-consistency check, not a substitute for rerunning experiments from raw data.

Raw OpenRCA telemetry, the pinned Tau2 checkout, and large generated replay tables are intentionally excluded from Git.
