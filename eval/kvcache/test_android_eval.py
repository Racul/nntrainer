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
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import android_eval as ae  # noqa: E402
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

    def perf_args(self, *options, parser=None, command=(), output="run"):
        if parser is None:
            parser = argparse.ArgumentParser(parents=list(kv.perf_parser_parents()))
        args = parser.parse_args([*command, "--model", str(self.model), "--official", str(self.root / "official"),
                                  "--data", str(self.root / "data"), "--output", str(self.root / output),
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


# ---------------------------------------------------------------------------
# Android target
# ---------------------------------------------------------------------------
def write_meson(build_dir, options, compiler):
    info = build_dir / "meson-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "intro-buildoptions.json").write_text(json.dumps([dict(name=k, value=v) for k, v in options.items()]))
    (info / "intro-compilers.json").write_text(json.dumps({"host": {"cpp": {"full_version": compiler}}}))


def make_build(root, causallm=b"causallm-main", qnn=False, opencl=False, fp16=True):
    """build_android.sh outputs with Meson metadata in a fake checkout, and an NDK."""
    repo, ndk = root / "repo", root / "ndk"
    for name, path in ae.BUILD_OUTPUTS.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_bytes(causallm if name == "libcausallm.so" else name.encode())
    engine = {key: False for key in kv.BUILD_KEYS}
    engine.update({"platform": "android", "buildtype": "release", "enable-fp16": fp16, "enable-opencl": opencl,
                   "thread-backend": "omp", "nntr-num-threads": 7, "openblas-num-threads": 1})
    write_meson(repo / "builddir", engine, "clang 17.0.2")
    write_meson(repo / "builddir_app/cpu", {"platform": "android", "enable-qnn": qnn, "buildtype": "release",
                                            "enable-api": True}, "clang 17.0.2")
    (ndk / ae.LIBCXX_IN_NDK).parent.mkdir(parents=True, exist_ok=True)
    (ndk / ae.LIBCXX_IN_NDK).write_bytes(b"libc++")
    (ndk / "source.properties").write_text("Pkg.Desc = Android NDK\nPkg.Revision = 26.3.11579264\n")
    return repo, ndk


def done(stdout="", code=0, stderr=""):
    return subprocess.CompletedProcess([], code, stdout, stderr)


GETPROP = {"ro.product.manufacturer": "Fake", "ro.product.model": "Phone 1", "ro.soc.model": "SM8650",
           "ro.product.cpu.abi": "arm64-v8a", "ro.build.version.release": "15", "ro.build.version.sdk": "35",
           "ro.build.fingerprint": "fake/phone/1:15/AP1A/1:user/release-keys"}


class FakeAdb(ae.AndroidDevice):
    """A device in a temporary folder. push copies files, shell answers the few
    commands the target sends, and a trial prints what nntr_causallm would for
    the pushed nntr_config.json, with CRLF line ends as adb.exe may."""

    def __init__(self, root, serials=("FAKE01",), temps=(36.0,)):
        super().__init__("adb")
        self.root, self.serials, self.temps = Path(root), list(serials), list(temps)
        self.level, self.mem_total, self.props = 80, 7_600_000, dict(GETPROP)
        self.pushes, self.killed, self.running, self.runs = [], [], [], []
        self.interrupt_at, self.corrupt, self.lines = None, False, method_lines

    def local(self, remote):
        return self.root / remote.lstrip("/")

    def _adb(self, *args, timeout=300):
        if args[0] == "devices":
            return done("List of devices attached\n" + "".join(f"{s}\tdevice\n" for s in self.serials))
        if args[0] == "push":
            source, remote = args[1:]
            self.local(remote).parent.mkdir(parents=True, exist_ok=True)
            self.local(remote).write_bytes(b"corrupt" if self.corrupt else Path(source).read_bytes())
            self.pushes.append(remote)
            return done(f"{source}: 1 file pushed")
        return self.answer(args[1])

    def answer(self, command):
        if "df -k" in command:
            return done("Filesystem 1K-blocks Used Available Use% Mounted on\n"
                        "/dev/block/dm-50 100000000 40000000 60000000 40% /data\n")
        name, *rest = shlex.split(command.replace(" 2>/dev/null", ""))
        if name == "getprop":
            return done("".join(f"[{key}]: [{value}]\n" for key, value in self.props.items()))
        if name == "cat" and rest[0].endswith("/temp"):
            temp = self.temps.pop(0) if len(self.temps) > 1 else self.temps[0]
            return done(f"{temp}\n" if isinstance(temp, str) else f"{round(temp * 1000)}\n")
        if name == "cat" and rest[0].endswith("/type"):
            return done("cpu-0-0\n")
        if name == "cat" and rest[0] == "/proc/meminfo":
            return done(f"MemTotal:        {self.mem_total} kB\n")
        if name == "dumpsys":
            return done("Current Battery Service state:\n  AC powered: false\n  USB powered: true\n"
                        f"  level: {self.level}\n  temperature: 312\n")
        if name == "sha256sum":
            return done("".join(f"{kv.sha(self.local(p))}  {p}\n" for p in rest if self.local(p).is_file()))
        if name == "mkdir":
            for path in rest[1:]:
                self.local(path).mkdir(parents=True, exist_ok=True)
            return done()
        if name == "chmod":
            return done() if all(self.local(p).is_file() for p in rest[1:]) else done(code=1, stderr="no such file")
        if name == "pidof":
            return done(" ".join(self.running) + "\n", code=0) if self.running else done(code=1)
        if name == "pkill":
            self.killed.append(rest[-1])
            return done()
        if name == "ls":
            folder = self.local(rest[0])
            return done("".join(p.name + "\n" for p in sorted(folder.iterdir()))) if folder.is_dir() else done(code=1)
        raise AssertionError(f"unexpected device command: {command}")

    def run(self, command, stdout_path, stderr_path):
        if len(self.runs) == self.interrupt_at:
            raise KeyboardInterrupt()
        words = shlex.split(command)
        binary = next(word for word in words if word.endswith("/nntr_causallm"))
        env = dict(word.split("=", 1) for word in words if re.fullmatch(r"[A-Z_]+=.*", word))
        config = json.loads((self.local(words[1]) / "nntr_config.json").read_text())
        memory = any(word.endswith("/mem_usage.sh") for word in words)
        self.runs.append(dict(binary=binary, env=env, memory=memory, config=config))
        needed = ([binary, env["LD_LIBRARY_PATH"] + "/libcausallm.so"]
                  + [config[key] for key in kv.ASSET_KEYS if config.get(key)])
        missing = [path for path in needed if not self.local(path).is_file()]
        text = (f"missing on device: {missing}\n" if missing else
                "device_pid=777\n" + model_output(config, self.lines(config), memory)).replace("\n", "\r\n")
        stdout_path.write_bytes(text.encode())
        stderr_path.write_bytes(b"")
        return int(bool(missing)), 4321, 0.25, text


class StageCopiesBuild(unittest.TestCase):
    """S1, S2: stage copies the six outputs and records how they were built."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo, self.ndk = make_build(self.root)

    def test_copies_hashes_and_refuses(self):
        stage = ae.stage_build(self.repo, "main", "a" * 40, self.ndk)
        self.assertEqual(sorted(p.name for p in stage.iterdir()), sorted(ae.STAGED_FILES + ["build-info.json"]))
        info = ae.load_stage(stage / "nntr_causallm")
        self.assertEqual(info["files"]["libcausallm.so"], kv.sha(self.repo / ae.BUILD_OUTPUTS["libcausallm.so"]))
        self.assertEqual(info["files"]["libc++_shared.so"], kv.sha(self.ndk / ae.LIBCXX_IN_NDK))
        self.assertEqual((self.repo / "build-android/.gitignore").read_text(), "*\n")
        with self.assertRaisesRegex(ValueError, "--replace"):
            ae.stage_build(self.repo, "main", "a" * 40, self.ndk)
        make_build(self.root, causallm=b"rebuilt")
        ae.stage_build(self.repo, "main", "c" * 40, self.ndk, replace=True)
        self.assertEqual(ae.load_stage(stage / "nntr_causallm")["revision"], "c" * 40)
        (stage / "libcausallm.so").write_bytes(b"edited")
        with self.assertRaisesRegex(ValueError, "changed after staging"):
            ae.load_stage(stage / "nntr_causallm")
        (self.repo / ae.BUILD_OUTPUTS["libnntrainer.so"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing: .*libnntrainer.so"):
            ae.stage_build(self.repo, "other", "a" * 40, self.ndk)
        self.assertEqual(sorted(p.name for p in (self.repo / "build-android").iterdir()), [".gitignore", "main"])
        with self.assertRaisesRegex(ValueError, "Not a staged Android build"):
            ae.load_stage(self.repo / ae.BUILD_OUTPUTS["nntr_causallm"])

    def test_records_options(self):
        info = ae.load_stage(ae.stage_build(self.repo, "main", "a" * 40, self.ndk) / "nntr_causallm")
        self.assertEqual(info["revision"], "a" * 40)
        self.assertEqual(info["ndk"], "26.3.11579264")
        self.assertEqual(info["engine"]["compiler"], "clang 17.0.2")
        self.assertEqual(set(info["engine"]["options"]), set(kv.BUILD_KEYS))
        self.assertEqual((info["engine"]["options"]["platform"], info["engine"]["options"]["enable-fp16"]),
                         ("android", True))
        self.assertEqual(info["app"], dict(options={"platform": "android", "enable-qnn": False, "buildtype": "release"},
                                           compiler="clang 17.0.2"))
        self.assertEqual(set(info["built_at"]), set(ae.STAGED_FILES))


class AndroidCase(unittest.TestCase):
    """A staged main build, a staged method build (other libcausallm.so) and a fake device."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ws = Workspace(self.root)
        self.repo, self.ndk = make_build(self.root)
        self.main = ae.stage_build(self.repo, "main", "a" * 40, self.ndk) / "nntr_causallm"
        make_build(self.root, causallm=b"causallm-mykv")
        self.mykv = ae.stage_build(self.repo, "mykv", "b" * 40, self.ndk) / "nntr_causallm"
        self.adb = FakeAdb(self.root / "device")
        patcher = mock.patch.object(kv, "CaseBuilder", StubCaseBuilder)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, *options, output="run"):
        args = self.ws.perf_args("--contexts", "512", *options, parser=ae.build_parser(), command=["perf"],
                                 output=output)
        ae.fill_revisions(args)
        return args

    def standard(self, output="run"):
        return self.args("--condition", f"baseline={self.main}", "--condition", f"mykv={self.mykv}", *MYKV,
                         output=output)

    def run_perf(self, args, adb=None):
        return quiet(kv.evaluate_perf, args, ae.AndroidTarget(adb or self.adb))


class AndroidBuildCheck(AndroidCase):
    """T2: CPU-only Android builds, generic build comparison, staged revisions."""

    def test_rejects_device_backends(self):
        for name, options in (("qnn", dict(qnn=True)), ("opencl", dict(opencl=True))):
            make_build(self.root, **options)
            binary = ae.stage_build(self.repo, name, "a" * 40, self.ndk) / "nntr_causallm"
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                ae.AndroidTarget(self.adb).builds({name: binary})
        builds = ae.AndroidTarget(self.adb).builds({"baseline": self.main})
        self.assertFalse(builds["baseline"]["engine"]["options"]["enable-transformer"])

    def test_engine_option_difference(self):
        make_build(self.root, fp16=False)
        nofp16 = ae.stage_build(self.repo, "nofp16", "b" * 40, self.ndk) / "nntr_causallm"
        conditions = ["--condition", f"baseline={self.main}", "--condition", f"nofp16={nofp16}",
                      "--unchecked", "nofp16", "--check"]
        with self.assertRaisesRegex(ValueError, "Build options or compiler differ"):
            self.run_perf(self.args(*conditions))
        self.run_perf(self.args(*conditions, "--allow-build-difference"))

    def test_revisions_come_from_the_stage(self):
        args = self.standard()
        self.assertEqual(args.revision, ["baseline=" + "a" * 40, "mykv=" + "b" * 40])
        self.assertEqual(args.repeat, 1)
        with mock.patch.object(ae, "git_revision", return_value="d" * 40):
            with self.assertRaisesRegex(ValueError, "not the staged build's revision"):
                self.args("--condition", f"baseline={self.main}", "--revision", "baseline=HEAD")
        with self.assertRaisesRegex(ValueError, "without a --condition"):
            self.args("--condition", f"baseline={self.main}", "--revision", "other=HEAD")


class AndroidRun(AndroidCase):
    """T3–T8, T12: deployment, remote runs, interruption and resume."""

    def test_run_deploys_and_normalizes_output(self):
        args = self.standard()
        self.run_perf(args)
        manifest, records = read_run(args.output)
        protocol = manifest["protocol"]
        self.assertEqual((manifest["status"], len(records)), ("COMPLETE", 2 + 2 * 2))
        self.assertTrue(all(r["valid"] and r["pid"] == 777 for r in records), records[0]["failure"])
        bins = {label: f"{ae.DEVICE_DIR}/bin/{protocol['bundles'][label]['id']}" for label in ("baseline", "mykv")}
        model_dir = f"{ae.DEVICE_DIR}/models/{protocol['model_files']['id']}"
        deployed = {*(f"{bins[label]}/{name}" for label in bins for name in ae.STAGED_FILES),
                    f"{model_dir}/weights.bin", f"{model_dir}/tokenizer.json", f"{ae.DEVICE_DIR}/mem_usage.sh"}
        self.assertEqual({p for p in self.adb.pushes if "/trial/" not in p}, deployed)
        first = self.adb.runs[0]
        self.assertEqual(first["config"]["model_file_name"], f"{model_dir}/weights.bin")
        self.assertEqual(first["env"]["LD_LIBRARY_PATH"], str(Path(first["binary"]).parent))
        self.assertEqual({k: first["env"][k] for k in kv.THREADS}, kv.THREADS)
        self.assertEqual(sum(run["memory"] for run in self.adb.runs), 2)
        self.assertIn(b"\r\n", (args.output / records[0]["stdout_log"]).read_bytes())
        memory = [r for r in records if r["phase"] == "memory"]
        self.assertTrue(all(r["sampled_peak_rss_kib"] == 2807052 and r["target_pid"] == 4242 for r in memory))
        self.assertTrue(all(r["battery_level"] == 80 and r["usb_powered"] is True and r["temp_start_c"] == 36.0
                            for r in records))
        self.assertEqual(manifest["target_state"][0]["device_mem_total_kib"], 7_600_000)
        self.assertEqual((protocol["device"]["model"], protocol["cpu_affinity"]), ("Phone 1", "not pinned"))
        with (args.output / "trials.csv").open() as stream:
            self.assertEqual(next(csv.reader(stream))[-len(ae.TRIAL_COLUMNS):], ae.TRIAL_COLUMNS)
        summary = (args.output / "summary.md").read_text()
        self.assertIn("Android device Phone 1 (SM8650)", summary)
        self.assertIn("|512|temp_start_c|36.00 ± 0.00|36.00 ± 0.00|", summary)

        # A second run pushes only its configuration files.
        self.adb.pushes.clear()
        self.run_perf(self.standard(output="again"))
        self.assertTrue(self.adb.pushes and all("/trial/" in p for p in self.adb.pushes))

    def test_bundle_depends_on_every_file(self):
        args = self.standard()
        quiet(kv.evaluate_perf, args, ae.AndroidTarget(self.adb))
        bundles = read_run(args.output)[0]["protocol"]["bundles"]
        self.assertNotEqual(bundles["baseline"]["id"], bundles["mykv"]["id"])
        self.assertEqual({k for k in bundles["baseline"]["files"]
                          if bundles["baseline"]["files"][k] != bundles["mykv"]["files"][k]}, {"libcausallm.so"})

    def test_push_verification(self):
        self.adb.corrupt = True
        with self.assertRaisesRegex(RuntimeError, "Pushed files differ"):
            self.run_perf(self.standard())

    def test_remote_command(self):
        bin_dir, trial, script = "/d/bin/abc", "/d/trial", "/d/mem_usage.sh"
        env = ("env LD_LIBRARY_PATH=/d/bin/abc NNTR_ENGINE=cpu NNTR_NUM_THREADS=4 OMP_NUM_THREADS=4 "
               "OPENBLAS_NUM_THREADS=1")
        self.assertEqual(ae.remote_command(bin_dir, trial, False, script),
                         f"cd /d/trial && echo device_pid=$$ && exec {env} /d/bin/abc/nntr_causallm /d/trial")
        self.assertEqual(ae.remote_command(bin_dir, trial, True, script),
                         f"cd /d/trial && echo device_pid=$$ && exec {env} sh /d/mem_usage.sh "
                         "/d/bin/abc/nntr_causallm /d/trial")
        self.assertIn("'/d/my dir'", ae.remote_command("/d/my dir", trial, False, script))

    def test_push_path_conversion(self):
        local = self.root / "file"
        with mock.patch.object(ae.subprocess, "check_output", return_value="\\\\wsl.localhost\\file\n") as wslpath:
            self.assertEqual(ae.AndroidDevice("/mnt/c/sdk/adb.EXE").host_path(local), "\\\\wsl.localhost\\file")
            wslpath.assert_called_once_with(["wslpath", "-w", str(local)], text=True)
            self.assertEqual(ae.AndroidDevice("adb").host_path(local), str(local))
            wslpath.assert_called_once()

    def test_interrupt_kills_remote_trial(self):
        self.adb.interrupt_at = 2
        args = self.standard()
        with self.assertRaises(KeyboardInterrupt):
            self.run_perf(args)
        manifest, records = read_run(args.output)
        self.assertEqual((manifest["status"], len(records)), ("INTERRUPTED", 2))
        label = manifest["active"]["condition"]
        self.assertEqual(self.adb.killed,
                         [f"{ae.DEVICE_DIR}/bin/{manifest['protocol']['bundles'][label]['id']}/nntr_causallm"])

    def test_preflight(self):
        self.adb.running = ["1234"]
        with self.assertRaisesRegex(RuntimeError, "already running"):
            self.run_perf(self.standard())
        for serials, message in (((), "No authorized Android device"), (("A", "B"), "--serial")):
            with self.assertRaisesRegex(RuntimeError, message):
                self.run_perf(self.standard(), FakeAdb(self.root / "device", serials=serials))
        self.adb.running, self.adb.props["ro.product.cpu.abi"] = [], "armeabi-v7a"
        with self.assertRaisesRegex(RuntimeError, "arm64-v8a"):
            self.run_perf(self.standard())

    def test_resume_ignores_volatile_values(self):
        self.adb.interrupt_at = 3
        args = self.standard()
        with self.assertRaises(KeyboardInterrupt):
            self.run_perf(args)
        self.adb.interrupt_at, self.adb.level, self.adb.mem_total, self.adb.temps = None, 55, 7_500_000, [33.0]
        host = dict(kv.environment(), mem_total_kib=1)
        with mock.patch.object(kv, "environment", return_value=host):
            self.run_perf(args)
        manifest, records = read_run(args.output)
        self.assertEqual((manifest["status"], len(records)), ("COMPLETE", 6))
        self.assertEqual([s["battery_level"] for s in manifest["target_state"]], [80, 55])
        make_build(self.root, causallm=b"causallm-mykv-v2")
        ae.stage_build(self.repo, "mykv", "b" * 40, self.ndk, replace=True)
        with self.assertRaisesRegex(ValueError, r"protocol changed: bundles$"):
            self.run_perf(args)


class GenericConditions(AndroidCase):
    """T13: presets and custom conditions are judged as on Linux."""

    def conditions(self, output="run"):
        return self.args("--condition", f"baseline={self.main}", "--condition", f"snapkv={self.mykv}",
                         "--condition", f"mykv={self.mykv}", *MYKV, output=output)

    def test_preset_and_custom_conditions(self):
        args = self.conditions()
        self.run_perf(args)
        manifest, records = read_run(args.output)
        self.assertEqual((manifest["status"], len(records)), ("COMPLETE", 3 + 3 * 2))
        self.assertTrue(all(r["valid"] for r in records))
        self.assertEqual(manifest["protocol"]["condition_specs"]["snapkv"]["activation"], "preset")

    def test_baseline_with_method_output_is_invalid(self):
        self.adb.lines = lambda config: method_lines(config) + ["[MyKV] ratio=0.5"]
        args = self.conditions()
        with self.assertRaisesRegex(RuntimeError, "Invalid trial function-512-baseline-00: baseline log matches"):
            self.run_perf(args)


if __name__ == "__main__":
    unittest.main()
