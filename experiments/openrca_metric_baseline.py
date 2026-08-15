"""Level-aware metric anomaly baseline for a local OpenRCA Telecom slice."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


SOURCE_BY_LEVEL = {
    "node": "metric_node.csv",
    "pod": "metric_container.csv",
    "service": "metric_service.csv",
}


def robust_scores(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["value_num"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["cmdb_id", "value_num"])
    frame["time"] = pd.to_datetime(frame["timestamp"], unit="ms") + pd.to_timedelta(8, unit="h")
    stats = frame.groupby(["cmdb_id", "name"])["value_num"].agg(
        median="median",
        mad=lambda values: np.median(np.abs(values - np.median(values))),
    )
    frame = frame.join(stats, on=["cmdb_id", "name"])
    denominator = 1.4826 * frame["mad"].replace(0, np.nan)
    frame["score"] = ((frame["value_num"] - frame["median"]) / denominator).abs()
    frame["score"] = frame["score"].replace([np.inf, -np.inf], np.nan).fillna(0)
    return frame


def main(dataset_root: Path, date: str, window_minutes: int) -> None:
    telecom = dataset_root / "Telecom"
    records = pd.read_csv(telecom / "record.csv")
    records = records[records["datetime"].str.startswith(date)].reset_index(drop=True)
    telemetry = telecom / "telemetry" / date.replace("-", "_") / "metric"

    results = []
    for record in records.itertuples(index=False):
        filename = SOURCE_BY_LEVEL.get(record.level)
        if filename is None:
            continue
        metrics = robust_scores(pd.read_csv(telemetry / filename))
        target_time = pd.Timestamp(record.datetime)
        window = metrics[
            metrics["time"].between(
                target_time - pd.to_timedelta(window_minutes, unit="m"),
                target_time + pd.to_timedelta(window_minutes, unit="m"),
            )
        ]
        ranked = list(window.groupby("cmdb_id")["score"].max().sort_values(ascending=False).index)
        results.append(
            {
                "datetime": record.datetime,
                "level": record.level,
                "reason": record.reason,
                "actual_component": record.component,
                "top1": ranked[0] if ranked else None,
                "top3": ranked[:3],
                "rank": ranked.index(record.component) + 1 if record.component in ranked else None,
            }
        )

    result_frame = pd.DataFrame(results)
    if result_frame.empty:
        raise SystemExit(f"No records found for {date}")
    result_frame["top3_hit"] = result_frame.apply(
        lambda row: row["actual_component"] in row["top3"], axis=1
    )
    print(f"date={date} records={len(result_frame)} window_minutes={window_minutes}")
    print(f"top1_accuracy={(result_frame['top1'] == result_frame['actual_component']).mean():.3f}")
    print(f"top3_accuracy={result_frame['top3_hit'].mean():.3f}")
    print(f"component_observed={result_frame['rank'].notna().mean():.3f}")
    print(result_frame.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("openrca/dataset/openrca-lite"))
    parser.add_argument("--date", default="2020-04-11")
    parser.add_argument("--window-minutes", type=int, default=1)
    main(**vars(parser.parse_args()))
