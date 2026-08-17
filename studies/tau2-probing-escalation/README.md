# Study: Tau2 probing and escalation

## Question

Can a policy choose between user probes, system probes, and human escalation while preserving controlled task resolution at lower interaction cost than unconditional acquisition?

## Status

**Positive controlled replay; not validated with live humans.** The gated policy preserves 100% replay resolution at mean synthetic cost 8.05 versus 13.17 for always-all and 10.00 for always-human escalation.

## Independence

This study uses fixed Tau2 tasks and locally defined intervention roles. Its result is a mechanism test and does not repair the Tau2 selective-controller NO-GO.

## Layout

- `experiments/tau2_probing_escalation.py`
- `artifacts/TAU2_PROBING_ESCALATION.md`
- `artifacts/tau2_probing_escalation.json`
