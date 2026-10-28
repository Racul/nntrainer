#!/usr/bin/env python3
"""Validate benchmark log parsing and refusal to summarize invalid trials."""
import unittest
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from manifold_kv_eval import execute_trial, parse_metrics, summarize


class EvaluationMetricsTest(unittest.TestCase):
    log = """prefill: 512 tokens, 21.50 ms, 23813.95 TPS
generation: 128 tokens, 17.25 ms, 7420.29 TPS
total: 38.75 ms
peak memory: 2048 KB
[e2e time]: 100 ms
PID - 42 :: 2048 kB
1 1024 kB
2 2048 kB
"""

    def test_float_and_legacy_integer_metrics(self):
        values = parse_metrics(self.log, 512, 128, True)
        self.assertEqual(values["target_pid"], 42)
        self.assertEqual(values["sampled_peak_kib"], 2048)
        self.assertEqual(values["prefill_ms"], 21.5)
        self.assertFalse(values["early_eos"])
        legacy = self.log.replace("21.50", "21").replace("17.25", "17")
        self.assertEqual(parse_metrics(legacy, 512, 128, False)["prefill_ms"], 21)

    def test_rejects_truncation_and_bad_memory(self):
        with self.assertRaisesRegex(ValueError, "truncated"):
            parse_metrics(self.log, 4096, 128, False)
        with self.assertRaisesRegex(ValueError, "RSS"):
            parse_metrics(self.log.replace("2 2048 kB", "2 1024 kB"), 512, 128, True)
        with self.assertRaisesRegex(ValueError, "RSS"):
            parse_metrics(self.log.replace("1 1024 kB", "1 0 kB"), 512, 128, True)

    def test_early_eos_and_incomplete_report(self):
        values = parse_metrics(self.log.replace("generation: 128", "generation: 5"), 512, 128, False)
        self.assertTrue(values["early_eos"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize([], [512], 5)

    def test_manifold_table_units_means_and_changes(self):
        rows = []
        for phase in ("latency", "memory"):
            for mode, latency, memory in (("baseline", 2000, 3072), ("manifold_kv", 1000, 2048)):
                for delta in (-100, 100):
                    rows.append(dict(valid=True, context=512, phase=phase, mode=mode,
                                     total_ms=latency + delta, e2e_ms=latency + delta,
                                     prefill_tps=100, generation_tps=10,
                                     sampled_peak_kib=memory + delta))
        report = summarize(rows, [512], 2)
        self.assertIn("ManifoldKV latency (s)", report)
        self.assertIn("| 512 | 2.000 | 1.000 | -50.0% | 3.0 | 2.0 | -33.3% |", report)
        self.assertNotIn("StreamingLLM", report)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            summarize(rows[:-1], [512], 2)

    @unittest.skipUnless(hasattr(os, "killpg"), "requires POSIX process groups")
    def test_timeout_stops_inference_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "child-survived"
            child = "import time; from pathlib import Path; time.sleep(1); Path(" + repr(str(marker)) + ").touch()"
            parent = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', " + repr(child) + "]); time.sleep(3)"
            with self.assertRaises(subprocess.TimeoutExpired):
                execute_trial([sys.executable, "-c", parent], tmp, os.environ.copy(), 0.2)
            time.sleep(1.1)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
