# Study: <short hypothesis name>

## Question

State one falsifiable question. Do not inherit a GO/NO-GO conclusion from another study.

## Hypothesis

Describe the expected reliability and efficiency change.

## Falsification condition

Define the result that will reject the hypothesis before running the experiment.

## Status

**PLANNED.** Replace with GO, NO-GO, or INCONCLUSIVE after evaluation.

## Independent scope

- Declare raw or pinned inputs and hashes.
- Freeze any required preprocessing inside this study.
- Do not import executable code from another study.
- Prior results may appear only as explicitly labeled baselines.
- Write all generated outputs under this study's `artifacts/` directory.

## POMDP mapping

- Hidden world state:
- Initial observation:
- Available sources and costs:
- Acquisition actions:
- Returned observations:
- Belief/evidence update:
- Task action, abstention, and escalation:
- Utility:
- Stopping condition:

## Evaluation

- Dataset and split:
- Leakage controls:
- No-context baseline:
- Broad-context baseline:
- Strong static baseline:
- Learned policy:
- Reliability metric:
- Efficiency metric:

## Result

Record the result without editing another study's conclusion.

## Reproduce

Add exact commands that run from this study directory.
