"""Run the complete telecom context-sufficiency replication pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_benchmark import prepare


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
MANIFEST = json.loads((HERE.parent / "benchmark_manifest.json").read_text())


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, default=Path(".benchmark/tau2-bench"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parent / "artifacts/reproduction",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.skip_prepare:
        result_file = benchmark_dir / MANIFEST["result_file"]
    else:
        prepared = prepare(benchmark_dir)
        result_file = Path(prepared["result_file"])

    tool_source = benchmark_dir / "src/tau2/domains/telecom/tools.py"
    if not result_file.exists() or not tool_source.exists():
        raise FileNotFoundError("Prepared benchmark is missing the telecom source or result file")

    output_dir.mkdir(parents=True, exist_ok=True)
    hard_dir = output_dir / "hard_counterfactual_full"
    source_dir = output_dir / "source_ablation_full"
    env = os.environ.copy()
    env["TAU2_DATA_DIR"] = str(benchmark_dir / "data")
    env["PYTHONPATH"] = os.pathsep.join([
        str(benchmark_dir / "src"),
        env.get("PYTHONPATH", ""),
    ]).rstrip(os.pathsep)

    run([
        sys.executable,
        str(HERE / "hard_counterfactuals.py"),
        "--results", str(result_file),
        "--tool-source", str(tool_source),
        "--output-dir", str(hard_dir),
    ], env)
    hard_rows = hard_dir / "telecom_hard_counterfactual_rows.jsonl"

    run([
        sys.executable,
        str(HERE / "source_ablation.py"),
        "--results", str(result_file),
        "--hard-rows", str(hard_rows),
        "--tool-source", str(tool_source),
        "--output-dir", str(source_dir),
        "--hard-only",
    ], env)
    run([
        sys.executable,
        str(HERE / "closed_loop_controller.py"),
        "--results", str(result_file),
        "--hard-rows", str(hard_rows),
        "--tool-source", str(tool_source),
        "--output", str(output_dir / "closed_loop_controller_report.json"),
    ], env)
    run([
        sys.executable,
        str(HERE / "risk_controlled_controller.py"),
        "--results", str(result_file),
        "--hard-rows", str(hard_rows),
        "--tool-source", str(tool_source),
        "--output", str(output_dir / "risk_controlled_controller_report.json"),
        "--alpha", "0.20,0.30,0.40,0.50",
    ], env)

    print(json.dumps({
        "benchmark_dir": str(benchmark_dir),
        "benchmark_ref": MANIFEST["ref"],
        "result_file": str(result_file),
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
