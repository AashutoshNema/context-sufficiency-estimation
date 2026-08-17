# Telecom Source Ablation Experiment

## Setup

For each write-decision prefix, one prior read call and its tool result were removed from the
observable context. The hidden environment state and counterfactual safety label were kept fixed.
This separates the value of observable context from the value of changing the environment itself.

Two datasets were produced:

- `hard-only`: 116,791 rows over 1,723 prefixes, containing only perturbed candidates;
- `all-candidates`: 134,949 rows, adding exact and reference candidates for policy selection.

The classifier was trained only on full-context rows and evaluated on held-out task groups.

## Safety prediction

On hard candidates, the full-context context-only model achieved AUROC `0.682` and AUPRC `0.100`.
After source removal, AUROC was `0.770` and AUPRC `0.094`. The changed class balance and repeated
source-specific rows make these aggregate metrics unsuitable as a direct claim of improvement;
the paired probability shift is the meaningful ablation statistic.

## Paired source effects

Negative `delta_safe_probability` means removing the source made the model more pessimistic about
candidate safety. Positive values mean the model became more optimistic after removal.

| Source removed | Mean delta, context-only | Mean delta, full features | Interpretation |
|---|---:|---:|---|
| `get_bills_for_customer` | -0.089 | -0.043 | bill context materially raises predicted safety |
| `get_customer_by_id` | -0.009 | -0.007 | small negative effect |
| `get_customer_by_phone` | -0.008 | +0.053 | model is unstable across feature sets |
| `get_data_usage` | +0.033 | +0.010 | removal makes the model more optimistic |
| `get_details_by_id` | +0.066 | +0.033 | removal makes the model more optimistic |

The source-dependent shifts prove that the current estimator is sensitive to context availability,
but their inconsistent direction is a warning: the features are mostly counts and coarse lookup
flags, not the semantic contents of the retrieved source.

## Candidate selection

Using all candidates, the model-selected candidate was safe in 16.8% of groups with context-only
features and 37.0% with candidate-tool features. Removing a source did not change the selected
candidate's safety outcome in the current setup.

This is not yet a live data-calling result. No source was reacquired, and the controller does not
read the content of a source. The experiment establishes the ablation harness and exposes the next
required improvement: source-content and argument-provenance features.

## Next implementation

Add features that identify which source introduced each candidate argument, then train a calibrated
source-value model. A source should be reacquired only when its removal increases estimated risk
above a risk-controlled threshold. The final evaluation should report task success, hard-candidate
safety, number of read calls, and token cost under full context, ablated context, and selective
reacquisition.
