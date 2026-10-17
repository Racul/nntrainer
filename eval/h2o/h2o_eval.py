#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate existing Linux or native Windows CPU baseline and H2O executables.

Default protocol: 512/2048/4096 prompt tokens, at most 128 decode steps,
cache capacity 4224, budget/recent 256/128, and five trials per phase.
Latency and resident-memory measurements use separate processes. Model JSON files
are copied to a temporary directory; original sampling and EOS settings remain.
No build, installation, cooling control, or output-quality scoring is performed.
"""

import argparse
import csv
import ctypes
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
WINDOWS = sys.platform == "win32"
PLATFORM = "windows" if WINDOWS else "linux"
MEMORY_KEY = "sampled_peak_working_set_kib" if WINDOWS else "sampled_peak_rss_kib"
MEMORY_SCRIPT = REPO / "Applications/utils/mem_usage.sh"
MEMORY_SCRIPT_SHA = "f075e1f0d535128da1d6c2e7dbc5827bf442ff5d3da01be3ca1232074a9d6b62"
BUILD_KEYS = (
    "platform", "buildtype", "enable-transformer", "enable-fp16", "enable-blas",
    "thread-backend", "nntr-num-threads", "openblas-num-threads",
    "enable-opencl", "enable-npu", "enable-htp", "enable-cuda",
)
COLUMNS = (
    "phase context mode trial position prefill_tokens generation_tokens early_eos "
    "prefill_ms prefill_tps generation_ms generation_tps total_ms e2e_ms "
    "sampled_peak_working_set_kib observed_os_peak_working_set_kib "
    "sampled_peak_rss_kib os_peak_rss_kib samples target_pid duration_s started_at valid failure log memory_log"
).split()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class WindowsMemory:
    """Read the target PID's resident working set through PSAPI, in bytes.

    Reference: https://learn.microsoft.com/en-us/windows/win32/api/psapi/
    nf-psapi-getprocessmemoryinfo. Explicit widths preserve the Windows ABI.
    """

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    def __init__(self, pid):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        self.kernel.OpenProcess.restype = ctypes.c_void_p
        self.kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel.CloseHandle.restype = ctypes.c_int
        self.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(self.Counters), ctypes.c_uint32,
        ]
        self.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        # PROCESS_QUERY_LIMITED_INFORMATION; no administrator privileges needed.
        self.handle = self.kernel.OpenProcess(0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def read(self):
        counters = self.Counters()
        counters.cb = ctypes.sizeof(counters)
        if not self.psapi.GetProcessMemoryInfo(
                self.handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.WorkingSetSize, counters.PeakWorkingSetSize

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def execute_windows(argv, env, log, memory_log=None, sample_interval=0.05):
    """Capture raw output, observe only the model PID, and clean up on failure."""
    process = observer = None
    samples = []
    start = time.monotonic()
    with log.open("wb") as stream:
        try:
            process = subprocess.Popen(
                argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
            )
            if memory_log is None:
                code = process.wait()
            else:
                observer = WindowsMemory(process.pid)
                with memory_log.open("w", newline="", encoding="utf-8") as memory:
                    writer = csv.writer(memory)
                    writer.writerow(["elapsed_s", "working_set_bytes", "os_peak_working_set_bytes"])
                    while process.poll() is None:
                        try:
                            current, peak = observer.read()
                        except OSError:
                            # Exit may race the last read. Other errors stop the run.
                            if process.poll() is None:
                                raise
                            break
                        if current > 0:
                            samples.append((current, peak))
                            writer.writerow([round(time.monotonic() - start, 6), current, peak])
                            memory.flush()
                        time.sleep(sample_interval)
                code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if observer is not None:
                observer.close()
    metrics = dict(target_pid=process.pid, duration_s=round(time.monotonic() - start, 3))
    if memory_log is not None:
        metrics.update(
            samples=len(samples),
            sampled_peak_working_set_kib=max((s[0] for s in samples), default=0) / 1024,
            observed_os_peak_working_set_kib=max((s[1] for s in samples), default=0) / 1024,
        )
    return code, log.read_text(encoding="utf-8", errors="replace"), metrics


def execute(argv, env, log, memory_log=None, sample_interval=0.05):
    if WINDOWS:
        return execute_windows(argv, env, log, memory_log, sample_interval)
    process = None
    start = time.monotonic()
    with log.open("wb") as stream:
        try:
            process = subprocess.Popen(argv, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    text = log.read_text(encoding="utf-8", errors="replace")
    metrics = dict(target_pid=process.pid, duration_s=round(time.monotonic() - start, 3))
    if memory_log is not None:
        peak = re.search(r"^PID - (\d+)\s*:+\s*(\d+) kB", text, re.M)
        values = [int(v) for v in re.findall(r"^\d+ (\d+) kB(?: |$)", text, re.M)]
        if (not peak or not values or min(values) <= 0 or int(peak[2]) != max(values)
                or re.search(r"^\d+\s+kB|/smaps.*(?:denied|No such|cannot)", text, re.M | re.I)):
            metrics["memory_error"] = "missing or inconsistent sampled RSS"
        metrics.update(samples=len(values), sampled_peak_rss_kib=int(peak[2]) if peak else None)
        if peak:
            metrics["target_pid"] = int(peak[1])
    peak = re.search(r"^(?:peak memory:|Max Resident Set Size:) (\d+) KB", text, re.M)
    if peak:
        metrics["os_peak_rss_kib"] = int(peak[1])
    return code, text, metrics


def check_condition(text, mode, settings):
    marker = f"[H2O] h2o_budget={settings['budget']} h2o_recent={settings['recent']}"
    if mode == "h2o" and marker not in text:
        return ["missing H2O activation marker; verify the deployed executable/library"]
    if mode == "baseline" and "[H2O]" in text:
        return ["baseline unexpectedly enabled H2O"]
    return []


def parse_metrics(text, returncode, context, max_decode):
    metrics, errors = {}, []
    if returncode:
        errors.append(f"model exit {returncode}")
    for stage in ("prefill", "generation"):
        match = re.search(
            r"^" + stage + r": (\d+) tokens, (\d+) ms, (\S+) TPS\s*$", text, re.M,
        )
        if not match:
            errors.append("missing " + stage + " output")
            continue
        metrics.update({stage + "_tokens": int(match[1]), stage + "_ms": int(match[2])})
        try:
            tps = float(match[3])
        except ValueError:
            tps = math.nan
        # A zero-ms stage can print inf. Keep that in the raw log, not statistics.
        metrics[stage + "_tps"] = tps if math.isfinite(tps) and tps >= 0 else None
    for key, pattern in (("total_ms", r"^total: (\d+) ms\s*$"),
                         ("e2e_ms", r"^\[e2e time\]: (\d+) ms\s*$")):
        match = re.search(pattern, text, re.M)
        if match:
            metrics[key] = int(match[1])
        else:
            errors.append("missing " + key)
    if metrics.get("prefill_tokens") != context:
        errors.append("actual prompt token count differs from requested count")
    generated = metrics.get("generation_tokens")
    if generated is None or not 0 <= generated <= max_decode:
        errors.append("invalid decode count")
    else:
        metrics["early_eos"] = generated < max_decode
    if re.search(r"FATAL ERROR|truncat|heuristic|falling back", text, re.I):
        errors.append("fatal/truncation/fallback output")
    return metrics, errors


def load_build(binary):
    if not binary.is_file() or (WINDOWS and binary.suffix.lower() != ".exe") or (not WINDOWS and not os.access(binary, os.X_OK)):
        raise ValueError(f"Build the executable first: {binary}")
    for root in binary.parents:
        info = root / "meson-info/intro-buildoptions.json"
        if info.is_file():
            options = {item["name"]: item["value"] for item in json.loads(info.read_text(encoding="utf-8"))}
            missing = set(BUILD_KEYS) - options.keys()
            if missing:
                raise ValueError(f"Missing Meson build options in {info}: {sorted(missing)}")
            return root, options
    raise ValueError(f"No Meson build metadata found above {binary}")


def prepare(args):
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("Install tokenizers in the selected Python: python -m pip install tokenizers") from error
    configs = {name: json.loads((args.model / name).read_text(encoding="utf-8-sig"))
               for name in ("config.json", "generation_config.json", "nntr_config.json")}
    nntr = configs["nntr_config.json"]
    if configs["config.json"].get("model_type") != "qwen3" or nntr.get("model_tensor_type") != "FP32-FP32":
        raise ValueError("Expected a Qwen3 FP32-FP32 model")
    position_limit = configs["config.json"].get("max_position_embeddings")
    if position_limit is not None and max(args.prompts) + args.n_gen > position_limit:
        raise ValueError("Prompt + n-gen exceeds the model's RoPE position limit")
    if any(key in nntr for key in ("chat_input", "system_prompt")) or nntr.get("skip_prefill"):
        raise ValueError("Use plain sample_input with a fresh, full prefill")
    asset_hashes = {}
    for key in ("model_file_name", "tokenizer_file", "embedding_file_name"):
        if nntr.get(key):
            path = Path(nntr[key])
            path = (path if path.is_absolute() else args.model / path).resolve()
            nntr[key] = str(path)
            asset_hashes[key] = sha(path)
    tokenizer = Tokenizer.from_file(nntr["tokenizer_file"])
    base = tokenizer.decode([5555])
    prompts = {str(n): base * n for n in args.prompts}
    for length, prompt in prompts.items():
        if len(tokenizer.encode(prompt, add_special_tokens=False).ids) != int(length):
            raise ValueError(f"Tokenizer length mismatch for {length} tokens")
    modes = ("baseline", "h2o") if args.condition == "both" else (args.condition,)
    binaries = {mode: getattr(args, mode + "_bin") for mode in modes}
    builds = {mode: load_build(binary) for mode, binary in binaries.items()}
    selected = {mode: {key: options[key] for key in BUILD_KEYS} for mode, (_, options) in builds.items()}
    if len(selected) == 2 and selected["baseline"] != selected["h2o"]:
        raise ValueError("Baseline and H2O Meson build options differ: " + json.dumps(selected))
    options = selected[modes[0]]
    if (options["platform"] != ("windows" if WINDOWS else "none") or not options["enable-transformer"]
            or options["nntr-num-threads"] != 4 or options["openblas-num-threads"] != 1
            or any(options[key] for key in ("enable-opencl", "enable-npu", "enable-htp", "enable-cuda"))):
        raise ValueError("Build for CPU with transformer enabled, nntr-num-threads=4, openblas-num-threads=1")
    envs, dll_hashes = {}, {}
    for mode, (root, _) in builds.items():
        # Discover only this build's DLLs, including nested Meson subdirectories.
        other = builds.get("h2o" if mode == "baseline" else "baseline", (root,))[0]
        dlls = sorted(path for path in root.rglob("*.dll")
                      if path.is_file() and not (other != root and other in path.parents)) if WINDOWS else []
        dll_hashes[mode] = {str(path): sha(path) for path in dlls}
        dirs = list(dict.fromkeys([str(binaries[mode].parent), str(root)] + [str(p.parent) for p in dlls]))
        envs[mode] = dict(os.environ, NNTR_ENGINE="cpu", OMP_NUM_THREADS="4", NNTR_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
        envs[mode]["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
    protocol = dict(
        conditions=list(modes), context_lengths=args.prompts,
        max_seq_len=args.max_seq_len, max_decode_steps=args.n_gen,
        h2o=dict(budget=args.budget, recent=args.recent), batch_size=1,
        repetitions=args.repeat, interval_seconds=args.interval, sample_interval_seconds=args.sample_interval,
        cpu_affinity="not pinned" if WINDOWS else args.cpus, threads=dict(NNTR_NUM_THREADS=4, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1),
        configs=configs, prompts=prompts, asset_sha256=asset_hashes,
        binaries={mode: str(path) for mode, path in binaries.items()},
        binary_sha256={mode: sha(path) for mode, path in binaries.items()},
        runtime_dll_sha256=dll_hashes, build_options=selected,
        environment=dict(os=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                         logical_cpus=os.cpu_count(), python=sys.version,
                         runtime_path={mode: env["PATH"] for mode, env in envs.items()}),
    )
    return protocol, envs


def summary(rows, contexts, platform_name=PLATFORM, memory_key=MEMORY_KEY,
            modes=("baseline", "h2o")):
    comparison = len(modes) == 2
    title = " / ".join("H2O" if mode == "h2o" else "baseline" for mode in modes)
    report = [f"# {platform_name.capitalize()} {title}", "", "Mean ± sample SD; valid trials only.", "",
              "|Input|Metric|Baseline|H2O|Change (%)|n per mode|" if comparison
              else f"|Input|Metric|{title.capitalize()}|n|",
              "|---|---|---|---|---|---|" if comparison else "|---|---|---|---|"]
    for context in contexts:
        for phase, key in (("latency", "total_ms"), ("latency", "e2e_ms"),
                           ("latency", "prefill_tps"), ("latency", "generation_tps"),
                           ("memory", memory_key)):
            values = [[row[key] for row in rows if row.get("valid") and row["phase"] == phase
                       and row["context"] == context and row["mode"] == mode and row.get(key) is not None]
                      for mode in modes]
            cells = [f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}" if len(v) > 1
                     else f"{v[0]:.2f} ± N/A" if v else "N/A" for v in values]
            change = "N/A"
            if comparison and all(values) and statistics.mean(values[0]):
                delta = (statistics.mean(values[1]) / statistics.mean(values[0]) - 1) * 100
                change = f"{delta if key.endswith('tps') else -delta:+.2f}"
            if comparison:
                report.append(f"|{context}|{key}|{cells[0]}|{cells[1]}|{change}|{len(values[0])}/{len(values[1])}|")
            else:
                report.append(f"|{context}|{key}|{cells[0]}|{len(values[0])}|")
    report += ["", "Positive change means lower time/resident memory or higher TPS." if comparison
               else "Only the selected condition was evaluated.",
               "Latency and memory use separate processes; function checks are excluded.",
               "Windows: sampled Working Set via PSAPI; Linux/Android: sampled RSS via original mem_usage.sh.",
               "OS peaks are recorded separately. Sampling can miss spikes, especially runs under 30 seconds.",
               ("KV allocation sizes are unchanged; retained cache lengths are bounded by the H2O budget."
                if "h2o" in modes else "H2O and SnapKV were disabled for all baseline trials."),
               "Sampling/EOS are unchanged. Compare actual decode counts in CSV before claiming equal work.",
               ("Android cooling is checked between contexts; no frequency control or quality scoring is performed."
                if platform_name == "android" else "No thermal/frequency control or quality scoring is performed."),
               "Nonfinite TPS is omitted."]
    return "\n".join(report) + "\n"


def save(out, manifest, platform_name=PLATFORM, memory_key=MEMORY_KEY):
    pending = out / (platform_name + "-comparison.tmp")
    pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(out / (platform_name + "-comparison.json"))
    with (out / (platform_name + "-comparison.csv.tmp")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["trials"])
    (out / (platform_name + "-comparison.csv.tmp")).replace(out / (platform_name + "-comparison.csv"))
    (out / "summary.md").write_text(summary(manifest["trials"], manifest["protocol"]["context_lengths"], platform_name, memory_key,
                                           manifest["protocol"].get("conditions", ("baseline", "h2o"))), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", type=Path, default=REPO / "models/qwen3-fp32")
    parser.add_argument("--baseline-bin", type=Path, default=os.environ.get("BASELINE_BIN", REPO / ("build-causallm-win-baseline/Applications/CausalLM/nntr_causallm.exe" if WINDOWS else "build-baseline/Applications/CausalLM/nntr_causallm")))
    parser.add_argument("--h2o-bin", type=Path, default=os.environ.get("H2O_BIN", REPO / ("build-causallm-win/Applications/CausalLM/nntr_causallm.exe" if WINDOWS else "build/Applications/CausalLM/nntr_causallm")))
    parser.add_argument("--condition", choices=("baseline", "h2o", "both"), default="both",
                        help="evaluate a single condition or compare both; only selected binaries are required")
    parser.add_argument("--check", action="store_true", help="inspect inputs/API; do not run inference or create results")
    parser.add_argument("--resume", type=Path, help="resume a matching interrupted run with only valid completed trials")
    parser.add_argument("--output", type=Path, help="new directory; default logs/h2o/<platform>/<timestamp>")
    parser.add_argument("--prompts", default="512,2048,4096", help="comma-separated prompt token counts")
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=4224)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--recent", type=int, default=128)
    parser.add_argument("--cpus", default=os.environ.get("EVAL_CPUS", "0,2,4,6"), help="Linux taskset affinity; unused on Windows")
    parser.add_argument("--repeat", type=int, default=5, help="trials per phase/context/mode; excludes function checks")
    parser.add_argument("--interval", type=float, default=1, help="seconds between trials")
    parser.add_argument("--sample-interval", type=float, default=0.05, help="working-set polling period in seconds")
    args = parser.parse_args()
    if sys.platform != "win32" and not sys.platform.startswith("linux"):
        parser.error("Supported hosts: Linux and native Windows")
    try:
        args.prompts = [int(n) for n in args.prompts.split(",")]
    except ValueError:
        parser.error("--prompts must contain comma-separated integers")
    if (not args.prompts or min(args.prompts) < 1 or len(set(args.prompts)) != len(args.prompts)
            or args.n_gen < 1 or max(args.prompts) + args.n_gen > args.max_seq_len):
        parser.error("positive unique prompts and prompt + n-gen <= max-seq-len are required")
    if args.condition != "baseline" and not 0 < args.recent < args.budget:
        parser.error("H2O requires 0 < recent < budget")
    if (args.repeat < 1 or not math.isfinite(args.interval) or args.interval < 0
            or not math.isfinite(args.sample_interval) or args.sample_interval <= 0):
        parser.error("repeat/sample-interval must be positive; interval must be nonnegative")
    if args.resume and args.output:
        parser.error("--resume and --output cannot be combined")
    for key in ("model", "baseline_bin", "h2o_bin"):
        setattr(args, key, getattr(args, key).resolve())
    if WINDOWS:
        observer = WindowsMemory(os.getpid())
        try:
            observer.read()
        finally:
            observer.close()
    else:
        if sha(MEMORY_SCRIPT) != MEMORY_SCRIPT_SHA:
            raise ValueError("The original mem_usage.sh changed")
        subprocess.run(["taskset", "-c", args.cpus, "true"], check=True)
    protocol, envs = prepare(args)
    protocol.update(policy="baseline with H2O and SnapKV disabled" if args.condition == "baseline"
                    else "H2O with GQA scores averaged per KV head",
                    cache_allocation="unchanged", quality="not evaluated",
                    evaluation_script_sha256=sha(Path(__file__)))
    if not WINDOWS:
        libraries = {}
        for mode, binary in protocol["binaries"].items():
            text = subprocess.check_output(["ldd", binary], env=envs[mode], text=True)
            if "not found" in text:
                raise RuntimeError("Missing runtime library: " + text)
            libraries[mode] = {p: sha(Path(p)) for p in re.findall(r"=> (\S+)", text) if Path(p).is_file()}
        protocol.update(runtime_library_sha256=libraries, memory_script_sha256=MEMORY_SCRIPT_SHA)
    print(f"Model: {args.model}")
    for mode, binary in protocol["binaries"].items():
        print(f"{mode}: {binary}")
    print("NNTR/OMP=4, OpenBLAS=1; latency/memory are separate; quality not evaluated.")
    if args.check:
        print("Preflight passed; no inference performed. Function checks will test executable startup.")
        return 0
    out = (args.resume or args.output or REPO / "logs/h2o" / PLATFORM / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    if args.resume:
        manifest = json.loads((out / (PLATFORM + "-comparison.json")).read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            raise ValueError("Cannot resume: settings, assets, binaries, DLLs, or environment changed")
        if any(not row.get("valid") for row in manifest["trials"]):
            raise ValueError("Inspect invalid trials before resuming; they are not silently rerun")
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = dict(protocol=protocol, trials=[], started_at=datetime.now().astimezone().isoformat(),
                        binary_revision="Not inferred from host checkout; verify build provenance separately")
        try:
            manifest["source_head_at_eval"] = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            manifest["source_head_at_eval"] = None
    (out / "logs").mkdir(exist_ok=True)
    manifest.update(status="RUNNING", controller_pid=os.getpid())
    manifest.pop("error", None)
    manifest.setdefault("invocations", []).append(dict(at=datetime.now().astimezone().isoformat(),
                                                     argv=sys.argv, script_sha256=sha(Path(__file__))))
    completed = {(r["phase"], r["context"], r["mode"], r["trial"]) for r in manifest["trials"]}
    print("Output:", out, flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="nntr-h2o-eval-") as tmp:
            model = Path(tmp)
            for name in ("config.json", "generation_config.json"):
                (model / name).write_text(json.dumps(protocol["configs"][name], ensure_ascii=False), encoding="utf-8")

            def run(phase, context, mode, trial, position):
                if (phase, context, mode, trial) in completed:
                    return
                if manifest["trials"]:
                    time.sleep(args.interval)
                config = dict(protocol["configs"]["nntr_config.json"], init_seq_len=context,
                              max_seq_len=args.max_seq_len, num_to_generate=args.n_gen,
                              batch_size=1, sample_input=protocol["prompts"][str(context)])
                config.pop("h2o", None)
                config.pop("snapkv", None)
                if mode == "h2o":
                    config["h2o"] = protocol["h2o"]
                (model / "nntr_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                stem = f"{phase}-{context}-{mode}-{trial:02d}"
                log = out / "logs" / (stem + ".log")
                memory_log = out / "logs" / (stem + "-working-set.csv") if phase == "memory" and WINDOWS else log if phase == "memory" else None
                row = dict(phase=phase, context=context, mode=mode, trial=trial, position=position,
                           started_at=datetime.now().astimezone().isoformat(), status="RUNNING", valid=False,
                           config=config, command=[protocol["binaries"][mode], str(model)],
                           log=str(log.relative_to(out)), memory_log=str(memory_log.relative_to(out)) if memory_log else "")
                if not WINDOWS:
                    row["command"] = ["taskset", "-c", args.cpus] + (["bash", str(MEMORY_SCRIPT)] if phase == "memory" else []) + row["command"]
                print("START", stem, flush=True)
                manifest["active_trial"] = row
                save(out, manifest)
                code, text, observed = execute(row["command"], envs[mode], log, memory_log, args.sample_interval)
                metrics, errors = parse_metrics(text, code, context, args.n_gen)
                errors += check_condition(text, mode, protocol["h2o"])
                if observed.get("memory_error"):
                    errors.append(observed.pop("memory_error"))
                if memory_log and (not observed["samples"] or (observed.get(MEMORY_KEY) or 0) <= 0):
                    errors.append("no positive resident-memory samples")
                row.update(observed, **metrics, valid=not errors, failure="; ".join(errors),
                           status="INVALID" if errors else "COMPLETE")
                manifest["trials"].append(row)
                manifest.pop("active_trial", None)
                save(out, manifest)
                print(f"DONE {stem}: valid={row['valid']}, decode={row.get('generation_tokens')}", flush=True)
                if errors:
                    raise RuntimeError(f"Invalid trial: {row['failure']}; inspect {log}")

            for position, mode in enumerate(protocol["conditions"], 1):
                run("function", args.prompts[0], mode, 0, position)
            for context in args.prompts:
                for phase in ("latency", "memory"):
                    for trial in range(1, args.repeat + 1):
                        modes = protocol["conditions"] if trial % 2 else list(reversed(protocol["conditions"]))
                        for position, mode in enumerate(modes, 1):
                            run(phase, context, mode, trial, position)
        manifest["status"] = "COMPLETE"
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        save(out, manifest)
    print("Results:", out / "summary.md")
    return 0


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed trials and raw logs were retained.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
