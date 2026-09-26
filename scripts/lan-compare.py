#!/usr/bin/env python3
"""Validate repeated complete, pooled, and split serving measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SCENARIOS = ("whole", "pool", "split")
MATCH_FIELDS = (
    "model",
    "system_prompt_sha256",
    "mode",
    "requested",
    "concurrency",
    "max_completion_tokens",
    "warmup_requests",
)


def positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive")
    return float(value)


def read_trial(path: Path, min_requests: int) -> dict[str, Any]:
    if path.stat().st_size > 1024 * 1024:
        raise ValueError(f"{path}: benchmark report exceeds 1 MiB")
    report = json.loads(path.read_text())
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported benchmark report")
    if report.get("mode") not in ("ordinary", "stream"):
        raise ValueError(f"{path}: invalid benchmark mode")
    for field in ("model", "endpoint"):
        if not isinstance(report.get(field), str) or not report[field]:
            raise ValueError(f"{path}: missing {field}")
    system_prompt_sha256 = report.get("system_prompt_sha256")
    if system_prompt_sha256 is not None and (
        not isinstance(system_prompt_sha256, str)
        or len(system_prompt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in system_prompt_sha256)
    ):
        raise ValueError(f"{path}: invalid system prompt digest")
    for field in ("requested", "concurrency", "max_completion_tokens", "warmup_requests"):
        positive_number(report.get(field), f"{path}: {field}")
    requested = report["requested"]
    if not isinstance(requested, int) or requested < min_requests:
        raise ValueError(f"{path}: fewer than {min_requests} requests")
    if report.get("successful") != requested:
        raise ValueError(f"{path}: not every request completed")
    if report.get("transport_failures") or report.get("application_errors"):
        raise ValueError(f"{path}: benchmark reported failures")
    if report.get("http_statuses") != {"200": requested}:
        raise ValueError(f"{path}: benchmark had a non-200 response")
    digests = report.get("completion_digests")
    if not isinstance(digests, dict) or len(digests) != 1:
        raise ValueError(f"{path}: completed requests produced different text")
    digest, count = next(iter(digests.items()))
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or count != requested
    ):
        raise ValueError(f"{path}: invalid completion digest")
    for field in ("elapsed_seconds", "requests_per_second", "p50_ms", "p95_ms"):
        positive_number(report.get(field), f"{path}: {field}")
    if report["p95_ms"] < report["p50_ms"]:
        raise ValueError(f"{path}: p95 is below p50")
    if report["mode"] == "stream":
        positive_number(report.get("first_content_p50_ms"), f"{path}: first content p50")
        positive_number(report.get("first_content_p95_ms"), f"{path}: first content p95")
        if report.get("responses_with_content") != requested:
            raise ValueError(f"{path}: some streams had no visible content")
    return report


def summarize(paths: list[Path], min_requests: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports = [read_trial(path, min_requests) for path in paths]
    summary: dict[str, Any] = {
        "trials": len(reports),
        "sources": [str(path) for path in paths],
        "median_requests_per_second": statistics.median(
            report["requests_per_second"] for report in reports
        ),
        "median_p50_ms": statistics.median(report["p50_ms"] for report in reports),
        "median_p95_ms": statistics.median(report["p95_ms"] for report in reports),
        "requests_by_device": {},
    }
    if reports[0]["mode"] == "stream":
        summary["median_first_content_p50_ms"] = statistics.median(
            report["first_content_p50_ms"] for report in reports
        )
        summary["median_first_content_p95_ms"] = statistics.median(
            report["first_content_p95_ms"] for report in reports
        )
    else:
        summary["median_completion_tokens_per_second"] = statistics.median(
            positive_number(report.get("completion_tokens_per_second"), "token rate")
            for report in reports
        )
    owners: dict[str, int] = {}
    for report in reports:
        allocations = report.get("requests_by_device")
        if not isinstance(allocations, dict):
            raise ValueError("invalid requests_by_device")
        total = 0
        for owner, count in allocations.items():
            if not isinstance(owner, str) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid requests_by_device")
            total += count
            owners[owner] = owners.get(owner, 0) + count
        if total != report["requested"]:
            raise ValueError("requests_by_device does not cover every request")
    summary["requests_by_device"] = owners
    return summary, reports


def compare(groups: dict[str, list[Path]], min_trials: int, min_requests: int) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    reports_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for scenario in SCENARIOS:
        paths = groups[scenario]
        if len(paths) < min_trials:
            raise ValueError(f"{scenario}: need at least {min_trials} trials")
        summaries[scenario], reports_by_scenario[scenario] = summarize(paths, min_requests)

    baseline = reports_by_scenario["whole"][0]
    expected = tuple(baseline.get(field) for field in MATCH_FIELDS)
    digest = next(iter(baseline["completion_digests"]))
    for scenario, reports in reports_by_scenario.items():
        for report in reports:
            if tuple(report.get(field) for field in MATCH_FIELDS) != expected:
                raise ValueError(f"{scenario}: model, mode, or load settings differ")
            if next(iter(report["completion_digests"])) != digest:
                raise ValueError(f"{scenario}: generated text differs from complete serving")
    if baseline["concurrency"] >= 2:
        owners = summaries["pool"]["requests_by_device"]
        if len([owner for owner, count in owners.items() if owner != "unknown" and count > 0]) < 2:
            raise ValueError("pool: fewer than two devices handled requests")

    whole_rate = summaries["whole"]["median_requests_per_second"]
    return {
        "model": baseline["model"],
        "system_prompt_sha256": baseline.get("system_prompt_sha256"),
        "mode": baseline["mode"],
        "requested_per_trial": baseline["requested"],
        "concurrency": baseline["concurrency"],
        "max_completion_tokens": baseline["max_completion_tokens"],
        "completion_digest": digest,
        "scenarios": summaries,
        "pool_rate_vs_whole": summaries["pool"]["median_requests_per_second"] / whole_rate,
        "split_rate_vs_whole": summaries["split"]["median_requests_per_second"] / whole_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for scenario in SCENARIOS:
        parser.add_argument(f"--{scenario}", nargs="+", type=Path, required=True)
    parser.add_argument("--min-trials", type=int, default=3)
    parser.add_argument("--min-requests", type=int, default=30)
    args = parser.parse_args()
    if args.min_trials <= 0 or args.min_requests <= 0:
        parser.error("minimum trials and requests must be positive")
    try:
        result = compare(
            {scenario: getattr(args, scenario) for scenario in SCENARIOS},
            args.min_trials,
            args.min_requests,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
