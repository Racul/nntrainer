#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of the trial target interface and the Android target.

No device, model or LongBench data is needed:

  python -m unittest eval/kvcache/test_android_eval.py
"""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kvcache_eval as kv  # noqa: E402

MYKV = ["--setting", 'mykv={"mykv": {"ratio": 0.5}}', "--expect", r"mykv=^\[MyKV\] ratio=0\.5$"]


class StubCaseBuilder:
    """One fixed prompt per length, each a truncation of one long document."""

    def __init__(self, model, official):
        pass

    def build(self, task, index, example, max_input):
        return dict(task=task, index=index, prompt=f"Summarize report {index}.\nLimit {max_input}.",
                    tokens=max_input - 4, prompt_ids_sha256=f"{max_input:064x}", truncated=True,
                    original_content_tokens=10000, max_new_tokens=512, answers=[], all_classes=None, length=0)


def method_lines(config):
    """Log lines a fake method prints when its nntr_config.json key is set."""
    lines = []
    if "mykv" in config:
        lines.append(f"[MyKV] ratio={config['mykv']['ratio']}")
    if "snapkv" in config and config["init_seq_len"] - 1 > config["snapkv"]["budget"]:
        capacity = config["snapkv"]["budget"] + config["num_to_generate"]
        lines.append(f"[SnapKV cache]: 100 -> 50 bytes, capacity {capacity} tokens")
    return lines


def model_output(config, lines=(), memory=False):
    """What nntr_causallm prints for one nntr_config.json (and mem_usage.sh around it)."""
    tokens, steps = config["init_seq_len"] - 1, config["num_to_generate"]
    text = (f"/model\n{config['sample_input']}\nA short summary.{kv.TRAILER}===================\n"
            f"prefill: {tokens} tokens, {2 * tokens} ms, 500 TPS\n"
            f"generation: {steps} tokens, {10 * steps} ms, 100 TPS\n"
            f"total: {2 * tokens + 10 * steps} ms\n"
            "==========================================================\n"
            f"[e2e time]: {2 * tokens + 10 * steps + 300} ms\n"
            "Max Resident Set Size: 2800000 KB\n" + "".join(line + "\n" for line in lines))
    if memory:
        text += "\nPID - 4242 :: 2807052 kB\n-------------------------------------\n1 3012 kB\n2 2807052 kB\n\n"
    return text


class Workspace:
    """A model directory, LongBench stand-ins and executables in a temporary folder."""

    def __init__(self, root):
        self.root = Path(root)
        model = self.model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
        (model / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
        (model / "nntr_config.json").write_text(json.dumps(dict(
            model_file_name="weights.bin", tokenizer_file="tokenizer.json", sample_input="Hello",
            init_seq_len=32, max_seq_len=256, num_to_generate=32, batch_size=1)))
        (model / "weights.bin").write_bytes(b"weights")
        (model / "tokenizer.json").write_text("{}")
        (self.root / "data").mkdir()
        (self.root / "data/gov_report.jsonl").write_text('{"context": "report"}\n')
        (self.root / "official/config").mkdir(parents=True)
        (self.root / "official/config/dataset2prompt.json").write_text("{}")

    def binary(self, name, content=b"executable"):
        path = self.root / "bin" / name / "nntr_causallm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def perf_args(self, *options, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(parents=list(kv.perf_parser_parents()))
        args = parser.parse_args(["--model", str(self.model), "--official", str(self.root / "official"),
                                  "--data", str(self.root / "data"), "--output", str(self.root / "run"),
                                  "--interval", "0", *options])
        kv.resolve_paths(args)
        kv.check_perf_args(parser, args)
        return args


def quiet(function, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def read_run(out):
    manifest = json.loads((out / "manifest.json").read_text())
    path = out / "trials.jsonl"
    return manifest, [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


class FakeTarget:
    """A target that adds two per-trial fields, a note and a volatile value."""
    kind = "fake"

    def __init__(self, interrupt_at=None):
        self.runs, self.interrupt_at, self.closed = [], interrupt_at, False

    def check(self, kind):
        self.checked = kind

    def builds(self, binaries):
        return {label: dict(options={}, compiler="fake") for label in binaries}

    def describe(self, protocol):
        return dict(target="fake", extra_columns=["temp_start_c", "fake_flag"], environment_note="Fake target.")

    def volatile(self):
        return dict(battery_level=50)

    def open(self, protocol):
        return FakeSession(self)


class FakeSession:
    def __init__(self, target):
        self.target, self.files = target, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.target.closed = True

    def put(self, name, value):
        self.files[name] = value

    def before_trial(self, manifest):
        manifest.setdefault("target_notes", ["Set by the first trial."])
        return dict(temp_start_c=30.0 + len(self.target.runs))

    def run(self, label, memory, stdout_path, stderr_path):
        if len(self.target.runs) == self.target.interrupt_at:
            raise KeyboardInterrupt()
        config = self.files["nntr_config.json"]
        self.target.runs.append((label, memory, config["init_seq_len"] - 1))
        text = model_output(config, method_lines(config), memory)
        stdout_path.write_text(text)
        stderr_path.write_text("")
        return 0, 100, 0.5, text

    def after_trial(self):
        return dict(fake_flag=True)


class SeamLoop(unittest.TestCase):
    """T1: the generic trial loop runs through any target."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)
        self.args = self.ws.perf_args("--contexts", "512,1024", "--repeat", "1",
                                      "--condition", f"baseline={self.ws.binary('main')}",
                                      "--condition", f"mykv={self.ws.binary('mykv')}", *MYKV)
        patcher = mock.patch.object(kv, "CaseBuilder", StubCaseBuilder)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_fields_notes_and_columns(self):
        target = FakeTarget()
        quiet(kv.evaluate_perf, self.args, target)
        manifest, records = read_run(self.args.output)
        self.assertEqual(target.checked, "perf")
        self.assertTrue(target.closed)
        self.assertEqual(manifest["status"], "COMPLETE")
        self.assertEqual(len(records), 2 + 2 * 2 * 2)
        self.assertTrue(all(r["valid"] and r["fake_flag"] is True and "temp_start_c" in r for r in records))
        self.assertEqual(manifest["target_notes"], ["Set by the first trial."])
        self.assertEqual(manifest["target_state"][0]["battery_level"], 50)
        self.assertEqual(manifest["protocol"]["target"], "fake")
        with (self.args.output / "trials.csv").open() as stream:
            header = next(csv.reader(stream))
        self.assertEqual(header, list(kv.PERF_COLUMNS) + ["temp_start_c", "fake_flag"])
        summary = (self.args.output / "summary.md").read_text()
        self.assertIn("Fake target.\nSet by the first trial.\nInvalid trials: 0.", summary)
        self.assertNotIn("not controlled or recorded on WSL", summary)
        self.assertIn("|512|temp_start_c|", summary)
        self.assertNotIn("|fake_flag|", summary)  # booleans are not averaged

    def test_resume_skips_completed_trials(self):
        with self.assertRaises(KeyboardInterrupt):
            quiet(kv.evaluate_perf, self.args, FakeTarget(interrupt_at=3))
        manifest, records = read_run(self.args.output)
        self.assertEqual((manifest["status"], len(records)), ("INTERRUPTED", 3))
        target = FakeTarget()
        quiet(kv.evaluate_perf, self.args, target)
        manifest, records = read_run(self.args.output)
        self.assertEqual((manifest["status"], len(records), len(target.runs)), ("COMPLETE", 10, 7))
        self.assertEqual(len(manifest["target_state"]), 2)
        self.assertEqual(len({(r["phase"], r["context"], r["condition"], r["trial"]) for r in records}), 10)


if __name__ == "__main__":
    unittest.main()
