"""Temporal metric + trace baseline for a local OpenRCA Telecom slice."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


SOURCE_BY_LEVEL = {
    "node": "metric_node.csv",
    "pod": "metric_container.csv",
    "service": "metric_service.csv",
}


def metric_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["value_num"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["cmdb_id", "value_num"])
    frame["time"] = pd.to_datetime(frame["timestamp"], unit="ms") + pd.to_timedelta(8, unit="h")
    frame = frame.sort_values(["cmdb_id", "name", "time"])
    stats = frame.groupby(["cmdb_id", "name"])["value_num"].agg(
        median="median",
        mad=lambda values: np.median(np.abs(values - np.median(values))),
    )
    frame = frame.join(stats, on=["cmdb_id", "name"])
    frame["level_score"] = ((frame["value_num"] - frame["median"]) / (1.4826 * frame["mad"].replace(0, np.nan))).abs()
    frame["delta"] = frame.groupby(["cmdb_id", "name"])["value_num"].diff().abs()
    delta_mad = frame.groupby(["cmdb_id", "name"])["delta"].transform(
        lambda values: np.nanmedian(np.abs(values - np.nanmedian(values)))
    )
    frame["change_score"] = frame["delta"] / (1.4826 * delta_mad.replace(0, np.nan))
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def rank_normalize(series: pd.Series) -> pd.Series:
    if series.empty or series.max() == series.min():
        return pd.Series(0.0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def metric_scores(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    window = frame[frame["time"].between(start, end, inclusive="left")]
    if window.empty:
        return pd.Series(dtype=float)
    grouped = window.groupby("cmdb_id").agg(
        anomaly=("level_score", "max"),
        change=("change_score", "max"),
    )
    return rank_normalize(grouped["anomaly"]) + rank_normalize(grouped["change"])


def trace_scores(path: Path, windows: list[tuple[pd.Timestamp, pd.Timestamp]]) -> dict[int, pd.DataFrame]:
    sums: dict[int, dict[str, dict[str, float]]] = {i: {} for i in range(len(windows))}
    usecols = ["startTime", "elapsedTime", "success", "cmdb_id"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        chunk["time"] = pd.to_datetime(chunk["startTime"], unit="ms") + pd.to_timedelta(8, unit="h")
        chunk["elapsedTime"] = pd.to_numeric(chunk["elapsedTime"], errors="coerce").fillna(0)
        chunk["failure"] = (~chunk["success"].astype(bool)).astype(int)
        for i, (start, end) in enumerate(windows):
            selected = chunk[chunk["time"].between(start, end, inclusive="left")]
            if selected.empty:
                continue
            aggregate = selected.groupby("cmdb_id").agg(
                calls=("cmdb_id", "size"),
                mean_latency=("elapsedTime", "mean"),
                max_latency=("elapsedTime", "max"),
                failures=("failure", "sum"),
            )
            for component, row in aggregate.iterrows():
                current = sums[i].setdefault(component, {"calls": 0, "latency_sum": 0, "max_latency": 0, "failures": 0})
                current["calls"] += row["calls"]
                current["latency_sum"] += row["mean_latency"] * row["calls"]
                current["max_latency"] = max(current["max_latency"], row["max_latency"])
                current["failures"] += row["failures"]

    outputs = {}
    for i, values in sums.items():
        if not values:
            outputs[i] = pd.DataFrame()
            continue
        result = pd.DataFrame.from_dict(values, orient="index")
        result["mean_latency"] = result["latency_sum"] / result["calls"].replace(0, np.nan)
        result["failure_rate"] = result["failures"] / result["calls"].replace(0, np.nan)
        result["score"] = (
            rank_normalize(result["mean_latency"].fillna(0))
            + rank_normalize(result["max_latency"].fillna(0))
            + rank_normalize(result["failure_rate"].fillna(0))
        )
        outputs[i] = result
    return outputs


def main(dataset_root: Path, date: str, trace_weight: float) -> None:
    telecom = dataset_root / "Telecom"
    records = pd.read_csv(telecom / "record.csv")
    records["root_time"] = pd.to_datetime(records["datetime"])
    records = records[records["root_time"].dt.strftime("%Y-%m-%d") == date].reset_index(drop=True)
    windows = [(row.root_time.floor("30min"), row.root_time.floor("30min") + pd.to_timedelta(30, unit="m")) for row in records.itertuples()]
    trace = trace_scores(telecom / "telemetry" / date.replace("-", "_") / "trace" / "trace_span.csv", windows)
    rows = []
    metric_cache = {level: metric_features(telecom / "telemetry" / date.replace("-", "_") / "metric" / filename) for level, filename in SOURCE_BY_LEVEL.items()}
    for i, row in enumerate(records.itertuples()):
        metric = metric_scores(metric_cache[row.level], *windows[i])
        candidates = set(metric.index)
        trace_frame = trace[i]
        if row.level == "pod" and not trace_frame.empty:
            candidates.update(trace_frame.index)
        scores = pd.DataFrame(index=sorted(candidates))
        scores["metric"] = metric.reindex(scores.index).fillna(0)
        scores["trace"] = trace_frame["score"].reindex(scores.index).fillna(0) if row.level == "pod" and not trace_frame.empty else 0
        scores["combined"] = scores["metric"] + trace_weight * scores["trace"]
        ranked = {name: list(scores[name].sort_values(ascending=False).index) for name in ["metric", "combined"]}
        rows.append({
            "datetime": row.datetime, "level": row.level, "reason": row.reason,
            "actual_component": row.component,
            "metric_top1": ranked["metric"][0] if ranked["metric"] else None,
            "metric_top3": ranked["metric"][:3],
            "combined_top1": ranked["combined"][0] if ranked["combined"] else None,
            "combined_top3": ranked["combined"][:3],
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise SystemExit(f"No labeled incidents found for {date}")
    for prefix in ["metric", "combined"]:
        result[f"{prefix}_top3_hit"] = result.apply(lambda row: row.actual_component in row[f"{prefix}_top3"], axis=1)
        print(f"{prefix}_top1_accuracy={(result[f'{prefix}_top1'] == result.actual_component).mean():.3f}")
        print(f"{prefix}_top3_accuracy={result[f'{prefix}_top3_hit'].mean():.3f}")
    print(f"records={len(result)} date={date}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("openrca/dataset/openrca-telecom-full"))
    parser.add_argument("--date", default="2020-04-11")
    parser.add_argument("--trace-weight", type=float, default=1.0)
    main(**vars(parser.parse_args()))
