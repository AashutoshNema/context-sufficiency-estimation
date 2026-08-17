# OpenRCA Telecom trace and temporal-feature validation

## Scope

This artifact records the full OpenRCA 1.0 Telecom validation run. It covers 51 labeled incidents across 15 telemetry dates. The raw Telecom telemetry is not vendored here; the local working copy was approximately 16 GB.

## Experiment

The component-ranking baseline uses each incident's observed 30-minute window rather than the labeled root-cause timestamp. It combines:

- robust anomaly scores from component-mapped node, container, and service metrics;
- temporal change scores from metric deltas; and
- trace call volume, latency, and failure-rate features for pod-level candidates.

## Results

| Method | Top-1 component accuracy | Top-3 component accuracy |
| --- | ---: | ---: |
| Temporal metrics | 9.8% (5/51) | 19.6% (10/51) |
| Metrics + traces | 13.7% (7/51) | 27.5% (14/51) |

Trace features improved Top-1 accuracy by 3.9 percentage points and Top-3 accuracy by 7.9 percentage points. The effect is positive but modest; traces work best as supporting evidence rather than as the dominant ranking signal.

## Interpretation

The hypothesis that traces and temporal features improve root-cause localization is directionally supported. This is a component-localization result, not a full OpenRCA benchmark score: the official evaluation also requires root-cause time and reason. App and middleware telemetry were present in the full Telecom download but require topology mapping before they can be fairly attributed to root-cause components.

## Provenance

- Dataset: [Microsoft OpenRCA](https://github.com/microsoft/OpenRCA)
- Public mirror used for download: [cdreetz/OpenRCA](https://huggingface.co/datasets/cdreetz/OpenRCA)
- System: Telecom
- Cases: 51
- Local raw data: intentionally excluded from Git
