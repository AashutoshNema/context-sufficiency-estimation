# Study: Tau2 LLM self-assessment baseline

## Question

Can archived LLM read/write behavior act as an implicit context-sufficiency gate?

## Status

**Negative baseline.** Among attempted writes, 46.3% are false-sufficient. Among prefixes with a safe candidate, 77.4% continue gathering context or defer action.

This baseline is descriptive of the archived GPT-4.1 trajectories; it is not a prompted confidence-estimation experiment.

## Layout

- `experiments/tau2_llm_self_assessment.py`
- `artifacts/TAU2_LLM_SELF_ASSESSMENT.md`
- `artifacts/tau2_llm_self_assessment.json`
