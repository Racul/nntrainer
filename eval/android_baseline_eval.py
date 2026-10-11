#!/usr/bin/env python3
"""Evaluate an already deployed Android baseline model.

run/memory/both use current settings. sweep temporarily prepares 512/2048/4096
input tokens, max decode 128, batch 1, cache 4224, and runs each phase five times.
sweep requires the existing tokenizers package; sampling/EOS remain unchanged.
The original device config is backed up and restored, including on errors.
This project helper is separate from the README's benchmark_android.py CLI.
Latency uses run_causallm.sh; memory uses the unchanged repository mem_usage.sh.

Full evaluation: ./eval/android_baseline_eval.sh sweep
Read-only preflight: ./eval/android_baseline_eval.sh sweep --check
Raw logs, exact trial configs, and statistics: logs/android_baseline/<timestamp>.
Cooling follows the repo's thermal_zone0 policy between contexts; read errors
and cooling timeouts stop the run. Invalid RSS trials remain in the report.
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[1]
DEVICE_DIR = "/data/local/tmp/nntrainer/causallm"
MEMORY_SCRIPT = REPO / "Applications/utils/mem_usage.sh"


def command(argv):
    return subprocess.check_output(argv, text=True, encoding="utf-8").strip()


def find_adb(explicit):
    if explicit:
        return explicit
    # Prefer Windows ADB in WSL: the phone remains attached to Windows USB.
    windows = shutil.which("adb.exe")
    candidates = list(Path("/mnt/c/Users").glob(
        "*/AppData/Local/Android/Sdk/platform-tools/adb.exe"))
    if windows:
        return windows
    if len(candidates) == 1:
        return str(candidates[0])
    native = shutil.which("adb")
    if native and not candidates:
        return native
    raise RuntimeError("Specify --adb /path/to/adb.exe (or set ADB_EXE).")


class Adb:
    def __init__(self, executable, serial):
        self.executable = executable
        self.serial = serial
        if not self.serial:
            devices = command([executable, "devices"])
            rows = [line.split() for line in devices.splitlines()[1:] if line.strip()]
            if len(rows) != 1 or rows[0][1] != "device":
                raise RuntimeError("Connect/authorize one device, or specify --serial.\n" + devices)
            self.serial = rows[0][0]
        if self.run("get-state").strip() != "device":
            raise RuntimeError("The selected device is not ready.")

    def run(self, *args):
        return command([self.executable, "-s", self.serial, *args])

    def shell(self, script):
        return self.run("shell", script)

    def push(self, source, destination):
        source = str(Path(source).resolve())
        if self.executable.lower().endswith(".exe") and sys.platform != "win32":
            source = command(["wslpath", "-w", source])
        return self.run("push", source, destination)

    def pull(self, source, destination):
        destination = str(Path(destination).resolve())
        if self.executable.lower().endswith(".exe") and sys.platform != "win32":
            destination = command(["wslpath", "-w", destination])
        return self.run("pull", source, destination)

    def sha(self, path):
        return self.shell("sha256sum " + shlex.quote(path)).split()[0]

    def capture(self, script, path):
        # Keep the raw output even when ADB or the child fails.
        argv = [self.executable, "-s", self.serial, "shell", script]
        with path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode, path.read_text(encoding="utf-8", errors="replace")


def read_configs(adb, model):
    configs = {}
    for name in ("config.json", "nntr_config.json", "generation_config.json"):
        path = shlex.quote(model + "/" + name)
        if name == "generation_config.json":
            text = adb.shell(f"if [ -f {path} ]; then cat {path}; else echo '{{}}'; fi")
        else:
            text = adb.shell("cat " + path)
        configs[name] = json.loads(text)
    return configs


def parse_metrics(text, memory, returncode):
    metrics, errors = {}, []
    if returncode:
        errors.append(f"ADB exit {returncode}")
    for stage in ("prefill", "generation"):
        match = re.search(r"^" + stage + r": (\d+) tokens, (\d+) ms, (\S+) TPS", text, re.M)
        if match:
            metrics[stage + "_tokens"] = int(match[1])
            metrics[stage + "_ms"] = int(match[2])
            # Zero-duration short stages can print inf; retain raw TPS in the log.
        else:
            errors.append("missing " + stage + " output")
    for key, pattern in (
        ("total_ms", r"^total: (\d+) ms"),
        ("e2e_ms", r"^\[e2e time\]: (\d+) ms"),
        ("os_peak_rss_kib", r"^Max Resident Set Size: (\d+) KB"),
    ):
        match = re.search(pattern, text, re.M)
        if match:
            metrics[key] = int(match[1])
        elif key != "os_peak_rss_kib":
            errors.append("missing " + key)
    if re.search(r"FATAL ERROR|Permission denied|truncat|falling back", text, re.I):
        errors.append("fatal, permission, truncation, or fallback output")
    if memory:
        peak = re.search(r"^PID - (\d+)\s*:+\s*(\d+) kB", text, re.M)
        values = [int(n) for n in re.findall(r"^\d+ (\d+) kB(?: |$)", text, re.M)]
        empty = len(re.findall(r"^\d+\s+kB(?: |$)", text, re.M))
        metrics.update(rss_samples=len(values), empty_rss_samples=empty)
        if peak:
            metrics.update(target_pid=int(peak[1]), sampled_peak_rss_kib=int(peak[2]))
        if not peak or not values or min(values) <= 0 or int(peak[2]) != max(values):
            errors.append("missing or inconsistent sampled RSS")
        if empty or re.search(r"/smaps.*(?:denied|No such|cannot)", text, re.I):
            errors.append("empty RSS sample or smaps read error")
    return metrics, errors


def prepare_sweep(adb, model, config, args):
    """Same token-5555 repetition as benchmark_android.generate_sample_input.

    Use the device tokenizer and verify actual lengths; never use the heuristic
    fallback. This does not write any device configuration.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("sweep requires tokenizers; use nntrainer/.venv/bin/python") from error
    if any(key in config for key in ("chat_input", "system_prompt")):
        raise RuntimeError("Sweep requires plain sample_input without chat_input/system_prompt")
    tokenizer_path = config["tokenizer_file"]
    if not PurePosixPath(tokenizer_path).is_absolute():
        tokenizer_path = model + "/" + tokenizer_path
    with tempfile.TemporaryDirectory(prefix="nntr-tokenizer-") as tmp:
        local = Path(tmp) / "tokenizer.json"
        adb.pull(tokenizer_path, local)
        tokenizer_sha = hashlib.sha256(local.read_bytes()).hexdigest()
        if tokenizer_sha != adb.sha(tokenizer_path):
            raise RuntimeError("Tokenizer transfer hash mismatch")
        tokenizer = Tokenizer.from_file(str(local))
    base = tokenizer.decode([5555])
    prepared = {}
    for length in args.prompts:
        prompt = base * length
        actual = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        if actual != length:
            raise RuntimeError(f"Requested {length} prompt tokens, tokenizer produced {actual}")
        prepared[length] = dict(config, sample_input=prompt, init_seq_len=length,
                                num_to_generate=args.n_gen, batch_size=1,
                                max_seq_len=args.max_seq_len)
    return prepared, tokenizer_sha


def wait_for_cooling(adb, target, max_wait):
    """Use the repo CLI's zone/threshold/polling; fail on read error or timeout."""
    deadline = time.monotonic() + max_wait
    while True:
        value = float(adb.shell("cat /sys/class/thermal/thermal_zone0/temp")) / 1000
        if not 0 < value < 150:
            raise RuntimeError(f"Invalid thermal_zone0 temperature: {value}")
        print(f"Cooling: thermal_zone0={value:.1f} C, target={target:.1f} C", flush=True)
        if value <= target:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Cooling timed out; stopping instead of measuring hot")
        time.sleep(min(10, max(0, deadline - time.monotonic())))


def summarize(trials):
    groups = {}
    for row in trials:
        key = (row.get("prompt_tokens_requested"), row["phase"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (context, phase), rows in groups.items():
        valid = [row for row in rows if row.get("valid")]
        item = dict(prompt_tokens=context, phase=phase, attempted=len(rows), valid=len(valid))
        keys = ("sampled_peak_rss_kib",) if phase == "memory" else ("total_ms", "e2e_ms")
        for key in keys:
            values = [row["metrics"][key] for row in valid]
            item[key] = dict(raw=values, mean=statistics.mean(values) if values else None,
                             sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        item["actual_generation_tokens"] = [row["metrics"].get("generation_tokens") for row in rows if "metrics" in row]
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=("check", "run", "memory", "both", "sweep"), default="check")
    parser.add_argument("--check", action="store_true", help="only inspect the device; no inference or output directory")
    parser.add_argument("--adb", default=os.environ.get("ADB_EXE"), help="ADB executable; Windows adb.exe preferred in WSL")
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))
    parser.add_argument("--model", default=DEVICE_DIR + "/models/qwen3-fp32", help="existing Android model directory")
    parser.add_argument("--repeat", type=int, help="trials per phase/context (default: sweep 5, otherwise 1; warm-up 0)")
    parser.add_argument("--interval", type=float, default=1, help="seconds between trials (default: 1; sweep also cools between contexts)")
    parser.add_argument("--prompts", default="512,2048,4096", help="sweep input token counts, comma-separated")
    parser.add_argument("--n-gen", type=int, default=128, help="sweep maximum decode steps (sampling/EOS unchanged)")
    parser.add_argument("--max-seq-len", type=int, default=4224, help="sweep cache capacity")
    parser.add_argument("--cool-to", type=float, default=35, help="sweep cooling threshold between contexts, thermal_zone0 C")
    parser.add_argument("--max-cool-wait", type=int, default=300, help="maximum seconds for cooling between contexts")
    parser.add_argument("--output", type=Path, help="new result directory (default: logs/android_baseline/<timestamp>)")
    parser.add_argument("--expected-binary-sha256", help="optional trusted baseline executable hash to verify")
    args = parser.parse_args()
    sweep = args.mode == "sweep"
    args.repeat = args.repeat if args.repeat is not None else (5 if sweep else 1)
    try:
        args.prompts = [int(value) for value in args.prompts.split(",")]
    except ValueError:
        parser.error("--prompts must contain comma-separated integers")
    if (not args.prompts or min(args.prompts) < 1 or len(set(args.prompts)) != len(args.prompts)
            or args.n_gen < 1 or max(args.prompts) + args.n_gen > args.max_seq_len):
        parser.error("prompts must be positive/unique and prompt + n-gen must fit max-seq-len")
    if args.repeat < 1 or not math.isfinite(args.interval) or args.interval < 0:
        parser.error("--repeat must be positive and --interval nonnegative")
    if not math.isfinite(args.cool_to) or not 0 < args.cool_to < 150 or args.max_cool_wait < 0:
        parser.error("--cool-to must be in (0, 150) and --max-cool-wait nonnegative")
    if not args.model.startswith(DEVICE_DIR + "/") or ".." in PurePosixPath(args.model).parts:
        parser.error("--model must be an absolute Android path under " + DEVICE_DIR)

    adb = Adb(find_adb(args.adb), args.serial)
    model = args.model.rstrip("/")
    configs = read_configs(adb, model)
    config = configs["nntr_config.json"]
    if config.get("snapkv"):
        raise RuntimeError("The device config enables SnapKV; select the baseline model config.")
    # Use the installer's wrapper unchanged. It fixes NNTR_NUM_THREADS at four.
    wrapper = adb.shell("cat " + DEVICE_DIR + "/run_causallm.sh")
    if "export NNTR_NUM_THREADS=4" not in wrapper:
        raise RuntimeError("Expected the existing install_android.sh wrapper with NNTR_NUM_THREADS=4.")
    adb.shell(f"test -x {DEVICE_DIR}/nntrainer_causallm && test -x {DEVICE_DIR}/run_causallm.sh")
    binary_sha = adb.sha(DEVICE_DIR + "/nntrainer_causallm")
    if args.expected_binary_sha256 and binary_sha != args.expected_binary_sha256:
        raise RuntimeError("Deployed executable differs from the expected baseline hash.")
    print(f"Device: {adb.serial}\nADB: {adb.executable}\nModel: {model}\nBinary SHA256: {binary_sha}")
    print(json.dumps(config, ensure_ascii=False, indent=2))
    prepared, tokenizer_sha = prepare_sweep(adb, model, config, args) if sweep else ({None: config}, None)
    print(f"NNTR/OMP=4, warm-up=0, repetitions={args.repeat}; quality not evaluated.")
    if sweep:
        print(f"Sweep prompts={args.prompts}, max decode={args.n_gen}, cache={args.max_seq_len}; sampling/EOS unchanged.")
    if args.mode == "check" or args.check:
        print("Preflight passed. A source checkout alone does not prove the deployed binary revision.")
        return 0

    out = (args.output or REPO / "logs/android_baseline" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = out / "results.json"
    manifest = dict(started_at=datetime.now().astimezone().isoformat(), status="PREPARING",
                    invocation=sys.argv, python_version=sys.version,
                    adb=adb.executable, serial=adb.serial, model=model, configs=configs,
                    binary_sha256=binary_sha, expected_binary_sha256=args.expected_binary_sha256,
                    binary_revision="Not inferred from the host checkout; verify build provenance separately",
                    warmup=0, repetitions=args.repeat, interval_seconds=args.interval,
                    threads=dict(NNTR_NUM_THREADS=4, OMP_NUM_THREADS=4),
                    quality="not evaluated", automatic_cooling=sweep, trials=[],
                    protocol=dict(mode=args.mode, prompt_tokens=args.prompts if sweep else None,
                                  max_decode=args.n_gen if sweep else config.get("num_to_generate"),
                                  cache_capacity=args.max_seq_len if sweep else config.get("max_seq_len"),
                                  cooling="between contexts after the first; no per-trial cooling" if sweep else "none",
                                  cool_to_c=args.cool_to if sweep else None,
                                  max_cool_wait_s=args.max_cool_wait if sweep else None),
                    tokenizer_sha256=tokenizer_sha, config_restored=None)

    device_config = model + "/nntr_config.json"
    device_backup = device_config + ".android_baseline_backup"
    backup_created = False
    original_sha = None

    def save():
        pending = manifest_path.with_suffix(".tmp")
        pending.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.replace(manifest_path)

    try:
        save()
        manifest["source_head_at_eval"] = command(["git", "-C", str(REPO), "rev-parse", "HEAD"])
        manifest["source_status"] = command(["git", "-C", str(REPO), "status", "--short"])
        manifest["device"] = adb.shell("getprop ro.product.model; getprop ro.soc.model; getprop ro.build.fingerprint; uname -a; cat /proc/cpuinfo; cat /proc/meminfo")
        manifest["runtime_hashes"] = adb.shell(f"sha256sum {DEVICE_DIR}/*.so")
        weight = config["model_file_name"]
        if not PurePosixPath(weight).is_absolute():
            weight = model + "/" + weight
        manifest["weight_sha256"] = adb.sha(weight)
        for name in ("compiler.txt", "model-revision.txt", "nntrainer-commit.txt"):
            path = REPO / "logs" / name
            if path.is_file():
                manifest.setdefault("existing_build_records_unverified", {})[name] = path.read_text()
        build_options = REPO / "builddir/meson-info/intro-buildoptions.json"
        if build_options.is_file():
            manifest["host_build_options_unverified"] = json.loads(build_options.read_text())
        manifest["run_wrapper"] = wrapper
        manifest["evaluation_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        phases = ("run", "memory") if args.mode in ("both", "sweep") else (args.mode,)
        if "memory" in phases:
            script_sha = hashlib.sha256(MEMORY_SCRIPT.read_bytes()).hexdigest()
            print(adb.push(MEMORY_SCRIPT, DEVICE_DIR + "/mem_usage.sh"))
            if adb.sha(DEVICE_DIR + "/mem_usage.sh") != script_sha:
                raise RuntimeError("Transferred mem_usage.sh hash mismatch")
            manifest["memory_script_sha256"] = script_sha
        manifest["status"] = "RUNNING"
        if sweep:
            if read_configs(adb, model) != configs:
                raise RuntimeError("Device settings changed during preparation")
            # Refuse to overwrite a backup from an interrupted controller.
            adb.shell(f"test ! -e {shlex.quote(device_backup)}")
            original_sha = adb.sha(device_config)
            adb.pull(device_config, out / "nntr_config.original.json")
            if hashlib.sha256((out / "nntr_config.original.json").read_bytes()).hexdigest() != original_sha:
                raise RuntimeError("Original config backup hash mismatch")
            adb.shell(f"cp {shlex.quote(device_config)} {shlex.quote(device_backup)}")
            if adb.sha(device_backup) != original_sha:
                raise RuntimeError("Device config backup hash mismatch; original config was not changed")
            backup_created = True
            manifest["original_config_sha256"] = original_sha
            manifest["device_config_backup"] = device_backup
            manifest["thermal_zone0_type"] = adb.shell("cat /sys/class/thermal/thermal_zone0/type")
        save()
        print("Output:", out)
        for context_index, (context, trial_config) in enumerate(prepared.items()):
            expected_configs = dict(configs, **{"nntr_config.json": trial_config})
            if sweep:
                if context_index:
                    wait_for_cooling(adb, args.cool_to, args.max_cool_wait)
                    time.sleep(2)  # Existing repo CLI's pause after cooling.
                local_config = out / f"nntr_config-{context}.json"
                local_config.write_text(json.dumps(trial_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                adb.push(local_config, device_config + ".android_baseline_next")
                adb.shell(f"mv {shlex.quote(device_config + '.android_baseline_next')} {shlex.quote(device_config)}")
                if read_configs(adb, model) != expected_configs:
                    raise RuntimeError("Trial config verification failed")
            for phase in phases:
                # Each context has separate latency and memory repetitions.
                for trial in range(1, args.repeat + 1):
                    if manifest["trials"]:
                        time.sleep(args.interval)
                    if read_configs(adb, model) != expected_configs:
                        raise RuntimeError("Device model settings changed during evaluation")
                    state = adb.shell("dumpsys battery; dumpsys thermalservice")
                    stem = f"{context}-{phase}-{trial:02d}" if sweep else f"{phase}-{trial:02d}"
                    (out / (stem + "-device.txt")).write_text(state, encoding="utf-8")
                    if phase == "memory":
                        script = (f"cd {DEVICE_DIR} && export LD_LIBRARY_PATH={DEVICE_DIR}:$LD_LIBRARY_PATH && "
                                  "export NNTR_NUM_THREADS=4 OMP_NUM_THREADS=4 && "
                                  "sh ./mem_usage.sh ./nntrainer_causallm " + shlex.quote(model))
                    else:
                        script = f"cd {DEVICE_DIR} && OMP_NUM_THREADS=4 ./run_causallm.sh " + shlex.quote(model)
                    log = out / (stem + ".log")
                    record = dict(phase=phase, trial=trial, command=script, log=log.name,
                                  prompt_tokens_requested=context,
                                  started_at=datetime.now().astimezone().isoformat(), status="RUNNING")
                    manifest["trials"].append(record)
                    save()
                    print(f"START {stem}", flush=True)
                    code, text = adb.capture(script, log)
                    metrics, errors = parse_metrics(text, phase == "memory", code)
                    if sweep:
                        if metrics.get("prefill_tokens") != context:
                            errors.append("actual prompt token count differs from requested count")
                        generated = metrics.get("generation_tokens")
                        if generated is None or not 0 <= generated <= args.n_gen:
                            errors.append("invalid decode count")
                        record["ended_before_generation_limit"] = generated is not None and generated < args.n_gen
                    record.update(metrics=metrics, errors=errors, valid=not errors,
                                  status="INVALID" if errors else "COMPLETE")
                    record["settings_unchanged"] = read_configs(adb, model) == expected_configs
                    if not record["settings_unchanged"]:
                        errors.append("device settings changed during trial")
                        record.update(valid=False, status="INVALID")
                    if phase == "memory" and metrics.get("e2e_ms", 0) < 30000:
                        record["sampling_note"] = "Under 30 seconds; sparse RSS sampling can miss peaks."
                    save()
                    print(json.dumps(record, ensure_ascii=False, indent=2))
                    # Keep invalid RSS trials in the report; never substitute OS RSS
                    # or silently rerun them. Stop on inference/configuration errors.
                    if any(error not in ("missing or inconsistent sampled RSS", "empty RSS sample or smaps read error") for error in errors):
                        raise RuntimeError("Invalid trial; inspect the raw log: " + str(log))
        manifest["status"] = "COMPLETE" if all(row["valid"] for row in manifest["trials"]) else "COMPLETE_WITH_INVALID_TRIALS"
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        if backup_created:
            try:
                adb.shell(f"cp {shlex.quote(device_backup)} {shlex.quote(device_config)}")
                if adb.sha(device_config) != original_sha:
                    raise RuntimeError("Restored device configuration hash mismatch")
                manifest["config_restored"] = True
                adb.shell(f"rm {shlex.quote(device_backup)}")
            except BaseException as error:
                manifest.update(status="RESTORE_FAILED", config_restored=False, restore_error=str(error))
                print(f"Restore failed. Original config: {out / 'nntr_config.original.json'}; device backup: {device_backup}", file=sys.stderr)
        manifest["summary"] = summarize(manifest["trials"])
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        save()
        print("Saved:", manifest_path)
    return 0 if manifest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Inspect device processes before retrying.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
