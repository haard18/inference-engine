#!/usr/bin/env python3
"""Compare fresh-process GGUF generation with a local llama.cpp build."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], timeout: float, final_newlines: int) -> tuple[float, str]:
    start = time.perf_counter()
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"{Path(command[0]).name} exited with {result.returncode}: "
            f"{result.stderr[-2000:]}"
        )
    suffix = "\n" * final_newlines
    if not result.stdout.endswith(suffix):
        raise RuntimeError(f"{Path(command[0]).name} changed its output format")
    return elapsed, result.stdout[: -final_newlines]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("engine_probe", type=Path)
    parser.add_argument("llama_completion", type=Path)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--metal", action="store_true")
    args = parser.parse_args()
    if args.tokens <= 0 or args.trials <= 0 or args.timeout <= 0 or not args.prompt:
        parser.error("tokens, trials, timeout, and prompt must be positive or nonempty")
    for path in (args.model, args.engine_probe, args.llama_completion):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    model = str(args.model.resolve())
    engine = [str(args.engine_probe.resolve())]
    if args.metal:
        engine.append("--metal")
    engine.extend((model, args.prompt, str(args.tokens)))
    llama = [
        str(args.llama_completion.resolve()),
        "-m", model,
        "-p", args.prompt,
        "-n", str(args.tokens),
        "-ngl", "99" if args.metal else "0",
        "--temp", "0",
        "--repeat-penalty", "1",
        "--no-warmup",
        "--no-display-prompt",
        "-no-cnv",
        "--simple-io",
    ]
    commands = {"engine": engine, "llama_cpp": llama}
    final_newlines = {"engine": 1, "llama_cpp": 2}
    samples: dict[str, list[float]] = {name: [] for name in commands}
    texts: dict[str, list[str]] = {name: [] for name in commands}
    try:
        for name, command in commands.items():
            run(command, args.timeout, final_newlines[name])
        for index in range(args.trials):
            order = ("engine", "llama_cpp") if index % 2 == 0 else ("llama_cpp", "engine")
            for name in order:
                elapsed, generated = run(commands[name], args.timeout, final_newlines[name])
                samples[name].append(round(elapsed, 6))
                texts[name].append(generated)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        parser.exit(1, f"reference run failed: {error}\n")

    results = {}
    for name in commands:
        digests = {
            hashlib.sha256(generated.encode()).hexdigest()
            for generated in texts[name]
        }
        results[name] = {
            "samples_seconds": samples[name],
            "median_wall_seconds": statistics.median(samples[name]),
            "output_digests": sorted(digests),
            "consistent_output": len(digests) == 1,
        }
    result = {
        "schema_version": 1,
        "model": model,
        "model_sha256": digest_file(args.model),
        "backend": "metal" if args.metal else "cpu",
        "prompt": args.prompt,
        "generated_tokens_requested": args.tokens,
        "warmups_per_engine": 1,
        "timing_scope": "fresh process, including model load and generation",
        "commands": commands,
        "results": results,
        "text_matches": texts["engine"][0] == texts["llama_cpp"][0]
        and results["engine"]["consistent_output"]
        and results["llama_cpp"]["consistent_output"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
