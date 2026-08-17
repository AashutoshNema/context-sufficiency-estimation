# Study: Tau2 source semantics

## Question

Do explicit source roles and authority constraints prevent unsafe action when observations, records, procedures, policies, history, or predictions conflict?

## Status

**Positive role-level controlled replay.** The guarded semantic router has zero unsafe actions in the tested conditions, while prediction-only selection is unsafe on 96.8% of episodes.

The study tests role availability, not full-text policy or manual reasoning.

## Layout

- `experiments/tau2_source_semantics.py`
- `artifacts/TAU2_SOURCE_SEMANTICS.md`
- `artifacts/tau2_source_semantics.json`
