# Experiment 2: Closed-Loop Selective Context Calling

## Question

Can a controller start with no prior read results, selectively reacquire sources, and improve
hard-candidate safety while using fewer reads and fewer context tokens than full retrieval?

## Method

The controller was trained on full-context telecom hard-candidate rows. It scores candidate writes,
then greedily reacquires a source when adding that source increases the best candidate's predicted
safety probability above a fixed threshold. Thresholds were evaluated at `0.00`, `0.05`, and `0.10`.

Evaluation used 23 held-out task groups and 380 decision prefixes. Baselines were:

- `full_context`: retain every prior read result;
- `no_read`: remove every prior read result;
- `selective`: start with no reads and reacquire sources greedily.

The controller used both coarse context features and argument-provenance features identifying
whether candidate arguments appeared in user text or a read result.

## Results

| Policy | Model | Safe rate | Mean reads | Mean context tokens |
|---|---|---:|---:|---:|
| Full context | Context-only | 22.1% | 4.27 | 1,800 |
| No read | Context-only | 22.1% | 0.00 | 1,362 |
| Selective, threshold 0.00 | Context-only | 22.1% | 1.33 | 1,601 |
| Selective, threshold 0.05 | Context-only | 22.1% | 0.13 | 1,423 |
| Selective, threshold 0.10 | Context-only | 22.1% | 0.02 | 1,369 |
| Full context | Full features | 39.2% | 4.27 | 1,800 |
| No read | Full features | 39.2% | 0.00 | 1,362 |
| Selective, threshold 0.00 | Full features | 39.2% | 0.82 | 1,443 |
| Selective, threshold 0.05 | Full features | 39.2% | 0.00 | 1,362 |
| Selective, threshold 0.10 | Full features | 39.2% | 0.00 | 1,362 |

The exact safe rate across policies is not a rounding artifact: source removal did not change the
safety outcome of the model-selected candidate in any selection group.

## Go / No-Go

**Result: no evidence of a working closed-loop data-calling policy yet.** The controller can reduce
read calls, but it does so without measurable safety loss or gain because the candidate ranking is
not responding to source content. Argument-provenance features also failed to change the ranking.

This does not disprove the broader research idea. It does disprove the current offline controller
formulation as a sufficient demonstration. The missing component is semantic source utility: the
controller must represent what fact a source contributed and whether that fact supports a candidate
action, rather than only whether an argument string appeared somewhere in the prefix.

## Decision

Do not claim that selective context calling improves agent performance based on telecom so far. The
next viable revision requires source-content representations or an LLM/judge-based fact-to-action
compatibility score, followed by a fresh controller evaluation. Without that additional signal,
the method is primarily a token-reduction heuristic, not context sufficiency estimation.
