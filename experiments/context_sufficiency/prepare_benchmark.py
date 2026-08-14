"""Prepare a sparse, pinned checkout of the upstream tau2 benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "benchmark_manifest.json"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def output(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def prepare(destination: Path) -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text())
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    created = False
    if not (destination / ".git").exists():
        run([
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            manifest["repository"],
            str(destination),
        ])
        created = True

    remote = output(["git", "config", "--get", "remote.origin.url"], destination)
    if remote != manifest["repository"]:
        raise RuntimeError(f"Existing checkout has unexpected origin: {remote}")

    if not created and output(["git", "status", "--porcelain"], destination):
        raise RuntimeError("Benchmark checkout has local changes; refusing to overwrite it")

    run(["git", "fetch", "--depth", "1", "origin", manifest["ref"]], destination)
    run(["git", "sparse-checkout", "init", "--no-cone"], destination)
    run(["git", "sparse-checkout", "set", "--no-cone", *manifest["sparse_paths"]], destination)
    run(["git", "checkout", "--detach", manifest["ref"]], destination)

    missing = [
        path
        for path in manifest["sparse_paths"]
        if not (destination / path).exists()
    ]
    if missing:
        raise RuntimeError(f"Sparse checkout is missing expected paths: {missing}")

    return {
        "path": str(destination),
        "repository": manifest["repository"],
        "ref": manifest["ref"],
        "result_file": str(destination / manifest["result_file"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(".benchmark/tau2-bench"),
        help="Local benchmark checkout directory",
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
