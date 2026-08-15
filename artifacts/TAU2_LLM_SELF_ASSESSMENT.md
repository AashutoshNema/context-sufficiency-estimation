# Tau2 LLM self-assessed sufficiency baseline

## Definition

The archived GPT-4.1 Telecom agent’s next-step behavior is treated as an implicit sufficiency judgment:

- issuing a write means “current context is sufficient to act”;
- continuing with reads or dialogue means “context is not yet sufficient.”

The judgment is evaluated against the same hard-counterfactual continuation labels used by the learned controllers. A prefix is oracle-sufficient when at least one available candidate write is safe in replay.

## Results

Across 2,007 decision prefixes:

| Metric | Result |
|---|---:|
| Action coverage | 17.4% |
| Safe rate when the LLM acted | 53.7% |
| False-sufficient rate when acted | 46.3% |
| Oracle-sufficient prefixes | 43.3% |
| Unnecessary-query rate among oracle-sufficient prefixes | 77.4% |
| Safe actions over all prefixes | 9.4% |
| Mean prior reads before acting | 4.87 |

## Interpretation

Unstructured LLM self-assessment is neither sufficiently safe nor efficient in this replay. Almost half of attempted writes are unsafe, while more than three quarters of prefixes that already contain a safe candidate continue gathering context or defer action.

This supports using an explicit sufficiency controller rather than relying on the base agent’s implicit confidence. It does not prove that a prompted numeric self-confidence baseline would behave identically because the archive contains decisions, not confidence scores.
