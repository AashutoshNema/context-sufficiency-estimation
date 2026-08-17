# Study: OpenRCA temporal context

## Question

Which telemetry interval is sufficient for top-3 component triage, and can a selector avoid broad temporal retrieval without losing action quality?

## Status

**Positive supporting result for the available endpoint.** Complete current telemetry reaches 76.5% top-3 accuracy versus 52.9% for the previous stale interval and 60.8% for a partial current interval. The learned selector preserves 76.5% at mean synthetic cost 0.88.

This does not validate root-cause time inference; official root-time accuracy remains a locked negative result.

## Independence

The study contains frozen local copies of the preprocessing and candidate-model helpers it needs. It does not import code or conclusions from another study directory.

## Layout

- `experiments/openrca_temporal_versions.py` — extraction and evaluation.
- `experiments/openrca_*` — frozen helper snapshot.
- `artifacts/OPENRCA_TEMPORAL_VERSIONS.md` — human-readable result.
- `artifacts/openrca_temporal_versions.json` — machine-readable result.

Raw temporal feature and prediction CSVs are intentionally excluded from Git.
