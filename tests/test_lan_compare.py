"""Checks that the LAN comparison rejects results that cannot support a claim."""

from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from typing import Any


COMPARE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "lan-compare.py"),
    run_name="lan_compare_test",
)["compare"]


def report(scenario: str, rate: float, digest: str = "a" * 64) -> dict[str, Any]:
    owners = {"local": 15, "remote": 15} if scenario == "pool" else {"local": 30}
    return {
        "schema_version": 1,
        "endpoint": "127.0.0.1:8080",
        "model": "local-smollm2",
        "mode": "ordinary",
        "requested": 30,
        "concurrency": 2,
        "max_completion_tokens": 8,
        "warmup_requests": 2,
        "successful": 30,
        "transport_failures": {},
        "application_errors": {},
        "http_statuses": {"200": 30},
        "completion_digests": {digest: 30},
        "requests_by_device": owners,
        "elapsed_seconds": 30 / rate,
        "requests_per_second": rate,
        "completion_tokens_per_second": rate * 8,
        "p50_ms": 1000 / rate,
        "p95_ms": 1500 / rate,
    }


class LanCompareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def groups(self) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for scenario, rate in (("whole", 1.0), ("pool", 2.0), ("split", 0.8)):
            paths = []
            for index in range(3):
                path = self.root / f"{scenario}-{index}.json"
                path.write_text(json.dumps(report(scenario, rate)))
                paths.append(path)
            groups[scenario] = paths
        return groups

    def test_comparison_requires_matching_outputs_and_both_pool_devices(self) -> None:
        result = COMPARE(self.groups(), 3, 30)
        self.assertEqual(result["completion_digest"], "a" * 64)
        self.assertEqual(result["pool_rate_vs_whole"], 2.0)
        self.assertEqual(result["split_rate_vs_whole"], 0.8)

    def test_comparison_rejects_different_generated_text(self) -> None:
        groups = self.groups()
        groups["split"][0].write_text(json.dumps(report("split", 0.8, "b" * 64)))
        with self.assertRaisesRegex(ValueError, "generated text differs"):
            COMPARE(groups, 3, 30)

    def test_comparison_rejects_different_system_prompts(self) -> None:
        groups = self.groups()
        changed = report("split", 0.8)
        changed["system_prompt_sha256"] = "b" * 64
        groups["split"][0].write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "load settings differ"):
            COMPARE(groups, 3, 30)

    def test_comparison_rejects_failed_or_one_device_pool_runs(self) -> None:
        groups = self.groups()
        failed = report("split", 0.8)
        failed["successful"] = 29
        groups["split"][0].write_text(json.dumps(failed))
        with self.assertRaisesRegex(ValueError, "not every request"):
            COMPARE(groups, 3, 30)

        groups = self.groups()
        for path in groups["pool"]:
            single_device = report("pool", 2.0)
            single_device["requests_by_device"] = {"local": 30}
            path.write_text(json.dumps(single_device))
        with self.assertRaisesRegex(ValueError, "fewer than two devices"):
            COMPARE(groups, 3, 30)


if __name__ == "__main__":
    unittest.main()
