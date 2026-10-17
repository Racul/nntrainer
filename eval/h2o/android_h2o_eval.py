#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare baseline/H2O on an already deployed Android CPU Qwen3 build.

Defaults: 512/2048/4096 input tokens, max decode 128, cache 4224,
budget/recent 256/128, five latency and five memory trials per condition.
Both conditions use the same H2O executable; baseline has H2O disabled.
Original configuration is backed up and restored, including on errors.
No deployment/build or quality scoring is performed. Sampling/EOS stay intact.
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
import subprocess
import sys
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
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


from h2o_eval import MEMORY_SCRIPT_SHA, check_condition, parse_metrics, save, sha


def parse_memory(text):
    peak = re.search(r"^PID - (\d+)\s*:+\s*(\d+) kB", text, re.M)
    values = [int(v) for v in re.findall(r"^\d+ (\d+) kB(?: |$)", text, re.M)]
    metrics = dict(samples=len(values))
    if peak:
        metrics.update(target_pid=int(peak[1]), sampled_peak_rss_kib=int(peak[2]))
    errors = []
    if (not peak or not values or min(values) <= 0 or int(peak[2]) != max(values)
            or re.search(r"^\d+\s+kB|/smaps.*(?:denied|No such|cannot)", text, re.M | re.I)):
        errors.append("missing or inconsistent sampled RSS")
    return metrics, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="inspect only; no inference or device config writes")
    parser.add_argument("--adb", default=os.environ.get("ADB_EXE"))
    parser.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"))
    parser.add_argument("--model", default=DEVICE_DIR + "/models/qwen3-fp32")
    parser.add_argument("--condition", choices=("both", "baseline", "h2o"), default="both")
    parser.add_argument("--phase", choices=("both", "latency", "memory"), default="both")
    parser.add_argument("--prompts", default="512,2048,4096")
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=4224)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--recent", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--cool-to", type=float, default=35)
    parser.add_argument("--max-cool-wait", type=int, default=300)
    parser.add_argument("--expected-binary-sha256", help="trusted hash of the deployed H2O executable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        args.prompts = [int(n) for n in args.prompts.split(",")]
    except ValueError:
        parser.error("--prompts must contain comma-separated integers")
    if (not args.prompts or min(args.prompts) < 1 or len(set(args.prompts)) != len(args.prompts)
            or args.n_gen < 1 or max(args.prompts) + args.n_gen > args.max_seq_len):
        parser.error("positive unique prompts and prompt + n-gen <= max-seq-len are required")
    if not 0 < args.recent < args.budget:
        parser.error("H2O requires 0 < recent < budget")
    if args.repeat < 1 or not math.isfinite(args.interval) or args.interval < 0:
        parser.error("repeat must be positive and interval finite/nonnegative")
    if not math.isfinite(args.cool_to) or not 0 < args.cool_to < 150 or args.max_cool_wait < 0:
        parser.error("cool-to must be in (0, 150); max-cool-wait must be nonnegative")
    if not args.model.startswith(DEVICE_DIR + "/") or ".." in PurePosixPath(args.model).parts:
        parser.error("model must be an absolute device path under " + DEVICE_DIR)
    adb = Adb(find_adb(args.adb), args.serial)
    model = args.model.rstrip("/")
    configs = read_configs(adb, model)
    config = configs["nntr_config.json"]
    if configs["config.json"].get("model_type") != "qwen3" or config.get("model_tensor_type") != "FP32-FP32" or config.get("skip_prefill"):
        raise ValueError("Expected Qwen3 FP32-FP32 with full prefill")
    limit = configs["config.json"].get("max_position_embeddings")
    if limit is not None and max(args.prompts) + args.n_gen > limit:
        raise ValueError("Prompt + generation exceeds the model RoPE limit")
    if sha(MEMORY_SCRIPT) != MEMORY_SCRIPT_SHA:
        raise ValueError("The original mem_usage.sh changed")
    wrapper = adb.shell("cat " + DEVICE_DIR + "/run_causallm.sh")
    if "export NNTR_NUM_THREADS=4" not in wrapper:
        raise RuntimeError("Expected install_android.sh wrapper with NNTR_NUM_THREADS=4")
    adb.shell(f"test -x {DEVICE_DIR}/nntrainer_causallm && test -x {DEVICE_DIR}/run_causallm.sh")
    binary_sha = adb.sha(DEVICE_DIR + "/nntrainer_causallm")
    if args.expected_binary_sha256 and binary_sha != args.expected_binary_sha256:
        raise RuntimeError("Deployed executable differs from the expected H2O hash")
    prepared, tokenizer_sha = prepare_sweep(adb, model, config, args)
    print(f"Device: {adb.serial}\nADB: {adb.executable}\nModel: {model}\nBinary SHA256: {binary_sha}")
    if args.check:
        print("Preflight passed; no inference/config writes. Function checks will verify H2O activation.")
        return 0
    conditions = ("baseline", "h2o") if args.condition == "both" else (args.condition,)
    phases = ("latency", "memory") if args.phase == "both" else (args.phase,)
    out = (args.output or REPO / "logs/h2o/android" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    manifest = dict(started_at=datetime.now().astimezone().isoformat(), status="PREPARING", trials=[],
                    configs=configs, binary_sha256=binary_sha, tokenizer_sha256=tokenizer_sha,
                    adb=adb.executable, serial=adb.serial, model=model, invocation=sys.argv,
                    binary_revision="Same H2O binary for both modes; verify build provenance separately",
                    evaluation_script_sha256=sha(Path(__file__)), memory_script_sha256=MEMORY_SCRIPT_SHA,
                    evaluation_common_script_sha256=sha(Path(__file__).with_name("h2o_eval.py")),
                    protocol=dict(context_lengths=args.prompts, max_seq_len=args.max_seq_len,
                                  max_decode_steps=args.n_gen, repetitions=args.repeat, conditions=conditions,
                                  h2o=dict(budget=args.budget, recent=args.recent), cache_allocation="unchanged",
                                  quality="not evaluated", warmup=0, function_checks="excluded",
                                  threads=dict(NNTR_NUM_THREADS=4, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1),
                                  cooling="between contexts after the first; no per-trial cooling",
                                  cool_to=args.cool_to, max_cool_wait=args.max_cool_wait), config_restored=None)
    device_config = model + "/nntr_config.json"
    device_backup = device_config + ".h2o_eval_backup"
    device_next = device_config + ".h2o_eval_next"
    backup_created = False
    original_sha = adb.sha(device_config)

    def persist():
        save(out, manifest, "android", "sampled_peak_rss_kib")

    try:
        persist()
        manifest["source_head_at_eval"] = command(["git", "-C", str(REPO), "rev-parse", "HEAD"])
        manifest["device"] = adb.shell("getprop ro.product.model; getprop ro.soc.model; getprop ro.build.fingerprint; uname -a; cat /proc/cpuinfo; cat /proc/meminfo")
        manifest["runtime_library_sha256"] = adb.shell(f"sha256sum {DEVICE_DIR}/*.so")
        weight = config["model_file_name"]
        if not PurePosixPath(weight).is_absolute():
            weight = model + "/" + weight
        manifest["weight_sha256"] = adb.sha(weight)
        adb.shell(f"test ! -e {shlex.quote(device_backup)} && test ! -e {shlex.quote(device_next)}")
        if read_configs(adb, model) != configs:
            raise RuntimeError("Device config changed during preparation")
        adb.pull(device_config, out / "nntr_config.original.json")
        if sha(out / "nntr_config.original.json") != original_sha:
            raise RuntimeError("Original config backup hash mismatch")
        adb.shell(f"cp {shlex.quote(device_config)} {shlex.quote(device_backup)}")
        if adb.sha(device_backup) != original_sha:
            raise RuntimeError("Device config backup hash mismatch")
        backup_created = True
        manifest.update(original_config_sha256=original_sha, device_config_backup=device_backup)
        if "memory" in phases:
            adb.push(MEMORY_SCRIPT, DEVICE_DIR + "/mem_usage.sh")
            if adb.sha(DEVICE_DIR + "/mem_usage.sh") != MEMORY_SCRIPT_SHA:
                raise RuntimeError("Transferred mem_usage.sh hash mismatch")
        manifest["thermal_zone0_type"] = adb.shell("cat /sys/class/thermal/thermal_zone0/type")
        manifest["status"] = "RUNNING"
        expected_configs = configs

        def run(phase, context, mode, trial, position):
            nonlocal expected_configs
            if manifest["trials"]:
                time.sleep(args.interval)
            if read_configs(adb, model) != expected_configs:
                raise RuntimeError("Device config changed during evaluation")
            trial_config = dict(prepared[context])
            trial_config.pop("snapkv", None)
            trial_config.pop("h2o", None)
            if mode == "h2o":
                trial_config["h2o"] = manifest["protocol"]["h2o"]
            stem = f"{phase}-{context}-{mode}-{trial:02d}"
            local_config = out / (stem + "-config.json")
            local_config.write_text(json.dumps(trial_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            adb.push(local_config, device_next)
            adb.shell(f"mv {shlex.quote(device_next)} {shlex.quote(device_config)}")
            expected_configs = dict(configs, **{"nntr_config.json": trial_config})
            if read_configs(adb, model) != expected_configs:
                raise RuntimeError("Trial config verification failed")
            state = adb.shell("dumpsys battery; dumpsys thermalservice")
            (out / (stem + "-device.txt")).write_text(state, encoding="utf-8")
            env = f"cd {DEVICE_DIR} && export NNTR_ENGINE=cpu NNTR_NUM_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 && "
            if phase == "memory":
                script = env + f"export LD_LIBRARY_PATH={DEVICE_DIR}:$LD_LIBRARY_PATH && sh ./mem_usage.sh ./nntrainer_causallm " + shlex.quote(model)
            else:
                script = env + "./run_causallm.sh " + shlex.quote(model)
            log = out / "logs" / (stem + ".log")
            row = dict(phase=phase, context=context, mode=mode, trial=trial, position=position, config=trial_config,
                       started_at=datetime.now().astimezone().isoformat(), status="RUNNING", valid=False,
                       command=script, log=str(log.relative_to(out)))
            manifest["active_trial"] = row
            persist()
            print("START", stem, flush=True)
            code, text = adb.capture(script, log)
            metrics, errors = parse_metrics(text, code, context, args.n_gen)
            errors += check_condition(text, mode, manifest["protocol"]["h2o"])
            if phase == "memory":
                observed, memory_errors = parse_memory(text)
                metrics.update(observed)
                errors += memory_errors
            if read_configs(adb, model) != expected_configs:
                errors.append("device settings changed during trial")
            row.update(metrics, valid=not errors, failure="; ".join(errors), status="INVALID" if errors else "COMPLETE")
            manifest["trials"].append(row)
            manifest.pop("active_trial", None)
            persist()
            if errors:
                raise RuntimeError("Invalid trial; inspect " + str(log) + ": " + row["failure"])

        for position, mode in enumerate(conditions, 1):
            run("function", args.prompts[0], mode, 0, position)
        for index, context in enumerate(args.prompts):
            if index:
                wait_for_cooling(adb, args.cool_to, args.max_cool_wait)
                time.sleep(2)
            for phase in phases:
                for trial in range(1, args.repeat + 1):
                    modes = conditions if trial % 2 else tuple(reversed(conditions))
                    for position, mode in enumerate(modes, 1):
                        run(phase, context, mode, trial, position)
        manifest["status"] = "COMPLETE"
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        if backup_created:
            try:
                adb.shell(f"cp {shlex.quote(device_backup)} {shlex.quote(device_config)}")
                if adb.sha(device_config) != original_sha:
                    raise RuntimeError("Restored device config hash mismatch")
                manifest["config_restored"] = True
                adb.shell(f"rm {shlex.quote(device_backup)}")
            except BaseException as error:
                manifest.update(status="RESTORE_FAILED", config_restored=False, restore_error=str(error))
                print("Restore failed. Original config:", out / "nntr_config.original.json", "Device backup:", device_backup, file=sys.stderr)
        manifest["finished_at"] = datetime.now().astimezone().isoformat()
        persist()
    print("Results:", out / "summary.md")
    return 0 if manifest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; inspect device processes before retrying.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
