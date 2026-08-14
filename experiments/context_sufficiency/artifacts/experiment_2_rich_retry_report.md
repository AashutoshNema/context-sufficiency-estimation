# Experiment 2 Retry: Dimension-Aware Context Representation

## Dimensions added

The retry augmented the previous count/provenance features with observable evidence features:

- fact completeness: whether candidate arguments occur in structured read results;
- task-source relevance: relevant versus distractor read results;
- entity scope: whether all candidate arguments are jointly supported by evidence;
- source semantics: identity, entity state, billing, and usage source roles;
- provenance: which read source supplied candidate values;
- source conflict and corroboration: repeated or inconsistent observed values;
- acquisition cost: source-specific read-cost proxy and token cost;
- source availability: full versus masked source observations.

The hidden environment, task labels, and hard counterfactual candidates were unchanged.

## Held-out results

Evaluation used 23 held-out task groups and 380 decision prefixes.

| Model/policy | AUROC | Safe rate | Mean reads | Mean context tokens |
|---|---:|---:|---:|---:|
| Mechanical baseline, full context | 0.627 | 22.1% | 4.27 | 1,800 |
| Rich context, full context | **0.856** | **33.7%** | 4.27 | 1,800 |
| Rich context, no read | | 32.6% | 0.00 | 1,362 |
| Rich context, selective, threshold 0.00 | | 26.8% | 1.59 | 1,525 |
| Rich context, selective, threshold 0.02 | | 27.1% | 0.95 | 1,467 |
| Rich context, selective, threshold 0.05 | | 27.1% | 0.93 | 1,467 |
| Rich context, selective, threshold 0.10 | | 26.6% | 0.61 | 1,432 |

The rich model's held-out classifier performance was:

- AUROC: `0.856`
- AUPRC: `0.304`
- Positive baseline: `0.061`

The prior mechanical model achieved AUROC `0.627` and AUPRC `0.140` on the same split.

## Interpretation

This retry gives a meaningful partial positive result. The missing dimensions were not cosmetic:
structured evidence features substantially improved held-out safety discrimination, and retaining
full context improved selected-candidate safety by 11.6 percentage points over the mechanical
baseline.

The selective controller is still a no-go. Its best safe rate, `27.1%`, is below full context at
`33.7%`. Greedy source selection is overconfident: it chooses sources that raise predicted safety
but do not reliably preserve the true hard-candidate outcome.

Therefore:

- **Context sufficiency estimation:** promising and now materially supported;
- **Selective data-calling controller:** not yet validated;
- **Overall research idea:** alive, but the controller requires calibration and risk control.

## Next required correction

Replace greedy probability-gain acquisition with a calibrated risk-controlled policy:

1. calibrate the rich estimator on held-out task groups;
2. optimize a lower confidence bound or conformal risk bound, not raw probability gain;
3. impose a minimum evidence-completeness condition before selecting a write;
4. compare against a full-context oracle and a fixed top-1 source policy.

The paper should not claim selective retrieval improves performance until this corrected controller
matches full-context safety at materially lower read/token cost.
