#!/usr/bin/env python3
"""Compare dense attention and ManifoldKV using the CausalLM executable.

The protocol follows origin/snapkv: separate latency/RSS processes, alternating
order, five trials, 512/1024/2048 input tokens and 128 decode steps. Model JSON
files are copied into temporary directories; original configurations stay intact.
"""

import argparse
import csv
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import statistics
import subprocess
import tempfile
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
MEMORY_SCRIPT = REPO / "Applications/utils/mem_usage.sh"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def execute_trial(argv, cwd, env, timeout):
    """Keep a memory wrapper and its inference child in one cancellable group."""
    with subprocess.Popen(argv, cwd=cwd, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          start_new_session=True) as process:
        finished = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            finished = True
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        finally:
            if not finished:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.communicate()


def parse_metrics(stdout, context, decode, memory):
    result = {}
    for phase in ("prefill", "generation"):
        match = re.search(
            rf"^{phase}: (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS\s*$",
            stdout, re.MULTILINE,
        )
        if not match:
            raise ValueError(f"missing {phase} metrics")
        result[f"{phase}_tokens"] = int(match[1])
        result[f"{phase}_ms"] = float(match[2])
        result[f"{phase}_tps"] = float(match[3])
    for key, pattern in (
        ("total_ms", r"^total: ([\d.]+) ms"),
        ("e2e_ms", r"^\[e2e time\]: ([\d.]+) ms"),
        ("os_peak_kib", r"^peak memory: (\d+) KB"),
    ):
        match = re.search(pattern, stdout, re.MULTILINE)
        if not match:
            raise ValueError(f"missing {key}")
        result[key] = float(match[1])
    if result["prefill_tokens"] != context:
        raise ValueError("input was truncated or tokenizer length differed")
    if not 0 < result["generation_tokens"] <= decode:
        raise ValueError("invalid generation count")
    result["early_eos"] = result["generation_tokens"] < decode
    if memory:
        peak = re.search(r"PID - (\d+)\s*:+\s*(\d+) kB", stdout)
        samples = [int(x) for x in re.findall(r"^\d+ (\d+) kB(?: |$)", stdout, re.MULTILINE)]
        if not peak or not samples or min(samples) <= 0 or int(peak[2]) != max(samples):
            raise ValueError("missing or inconsistent target-process RSS samples")
        result.update(target_pid=int(peak[1]), sampled_peak_kib=int(peak[2]), samples=len(samples))
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("non-finite metrics")
    return result


def summarize(rows, contexts, repeats):
    label = "ManifoldKV"
    report = [
        f"# Dense attention / {label}", "",
        "Mean ± sample standard deviation; latency and memory use separate processes.", "",
        f"| 입력 tokens | Baseline latency (s) | {label} latency (s) | Latency 변화 | Baseline peak memory (MiB) | {label} peak memory (MiB) | Memory 변화 |",
        "| --------: | -------------------: | ----------------------: | -----------: | -------------------------: | -----------------------------: | ----------: |",
    ]
    details = ["", f"|Input|Metric|Baseline|{label}|Change (%)|", "|---|---|---|---|---|"]
    for context in contexts:
        means_by_metric = {}
        for phase, metric in (
            ("latency", "total_ms"), ("latency", "e2e_ms"),
            ("latency", "prefill_tps"), ("latency", "generation_tps"),
            ("memory", "sampled_peak_kib"),
        ):
            groups = [[r for r in rows if r["valid"] and r["context"] == context
                       and r["phase"] == phase and r["mode"] == mode]
                      for mode in ("baseline", "manifold_kv")]
            if any(len(group) != repeats for group in groups):
                raise ValueError("cannot summarize incomplete or invalid trials")
            values = [[r[metric] for r in group] for group in groups]
            means = [statistics.mean(group) for group in values]
            means_by_metric[metric] = means
            change = (means[1] / means[0] - 1) * 100 if means[0] else float("nan")
            cells = [f"{statistics.mean(g):.2f} ± {statistics.stdev(g) if len(g) > 1 else 0:.2f}"
                     for g in values]
            details.append(f"|{context}|{metric}|{cells[0]}|{cells[1]}|{change:+.2f}|")
        latency = [value / 1000 for value in means_by_metric["total_ms"]]
        memory = [value / 1024 for value in means_by_metric["sampled_peak_kib"]]
        report.append(f"| {context} | {latency[0]:.3f} | {latency[1]:.3f} | "
                      f"{(latency[1] / latency[0] - 1) * 100:+.1f}% | "
                      f"{memory[0]:,.1f} | {memory[1]:,.1f} | "
                      f"{(memory[1] / memory[0] - 1) * 100:+.1f}% |")
    report += details
    report += [
        "", f"Change is ({label} / baseline − 1) × 100. Inspect early_eos and actual token counts.",
        "Latency uses the reported total (prefill + generation); memory uses sampled target-process peak RSS.",
        "The legacy dense driver may emit an extra token from prefill outside its reported decode count.",
        "These latency/RSS measurements do not measure output quality.",
    ]
    report.append("ManifoldKV compresses the prompt after dense prefill and appends generated tokens without further eviction.")
    return "\n".join(report) + "\n"


def main():
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local directory with model JSON, weights and tokenizer")
    parser.add_argument("--platform", choices=("linux", "android"), default="linux")
    parser.add_argument("--binary", required=True, help="local executable, or deployed Android executable path")
    parser.add_argument("--baseline-bin", help="optional separate dense executable; defaults to --binary")
    parser.add_argument("--contexts", type=positive, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--max-seq-len", type=positive, default=4225)
    parser.add_argument("--decode", type=positive, default=128)
    parser.add_argument("--repeat", type=positive, default=5)
    parser.add_argument("--budget", type=positive, default=256, help="ManifoldKV retained prompt tokens per KV head")
    parser.add_argument("--window-size", type=int, default=0, help="ManifoldKV local centroid window; 0 uses the global centroid")
    parser.add_argument("--prompt-token-id", type=int, default=5555)
    parser.add_argument("--cpus", help="CPU affinity list, e.g. 0,2,4,6; omitted uses OS scheduling")
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--timeout", type=positive, default=3600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="read-only preflight without inference")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--serial", help="required when more than one Android device is connected")
    parser.add_argument("--device-model", help="deployed model directory on Android")
    parser.add_argument("--temperature-path", default="/sys/class/thermal/thermal_zone0/temp")
    parser.add_argument("--cool-below", type=positive, default=40000, help="Android cooling threshold in millidegrees C")
    args = parser.parse_args()
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("contexts must be distinct")
    if args.decode > 2147483647 or args.max_seq_len > 2147483647:
        parser.error("decode and max-seq-len must fit the CausalLM integer settings")
    if args.interval < 0 or not math.isfinite(args.interval):
        parser.error("interval must be non-negative")
    if args.budget > args.max_seq_len or args.window_size < 0:
        parser.error("ManifoldKV budget must fit max_seq_len and window size must be non-negative")
    if max(args.contexts) + args.decode + 1 > args.max_seq_len:
        parser.error("baseline prompt + decode + one reserved row must fit max_seq_len")
    if args.platform == "android" and not args.device_model:
        parser.error("Android requires --device-model")

    from tokenizers import Tokenizer

    assets = args.model.resolve()
    configs = {name: json.loads((assets / name).read_text()) for name in
               ("config.json", "generation_config.json", "nntr_config.json")}
    config = dict(configs["nntr_config.json"])
    if configs["config.json"].get("model_type") != "qwen3":
        raise ValueError("evaluation currently supports Qwen3")
    if configs["config.json"]["max_position_embeddings"] < args.max_seq_len:
        raise ValueError("max_seq_len exceeds the model's position table")
    if any(k in config for k in ("chat_input", "system_prompt")) or config.get("skip_prefill", False):
        raise ValueError("use a plain benchmark config without chat/system prompt or skipped prefill")
    files = {}
    for name in ("model_file_name", "tokenizer_file", "embedding_file_name"):
        if name in config:
            path = Path(config[name])
            files[name] = path if path.is_absolute() else assets / path
            if not files[name].is_file():
                raise ValueError(f"missing model asset: {files[name]}")
    tokenizer = Tokenizer.from_file(str(files["tokenizer_file"]))
    prompts = {n: tokenizer.decode([args.prompt_token_id]) * n for n in args.contexts}
    for n, prompt in prompts.items():
        if len(tokenizer.encode(prompt, add_special_tokens=True).ids) != n:
            raise ValueError(f"prompt token {args.prompt_token_id} does not produce exactly {n} tokens")
    env = os.environ.copy()
    for key in ("NNTR_MAX_SEQ_LEN", "NNTR_INIT_SEQ_LEN", "NNTR_KV_INT8"):
        if key in env:
            raise ValueError(f"unset {key} before evaluating")
    env.update(NNTR_ENGINE="cpu", NNTR_NUM_THREADS=str(args.threads),
               OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1")
    binaries = {"baseline": args.baseline_bin or args.binary, "manifold_kv": args.binary}
    adb = [args.adb] + (["-s", args.serial] if args.serial else [])

    def command(argv):
        return subprocess.check_output(argv, env=env, text=True, timeout=30).strip()

    def shell(script):
        return command(adb + ["shell", script])

    if args.platform == "linux":
        binaries = {mode: str(Path(path).resolve()) for mode, path in binaries.items()}
        for path in binaries.values():
            if not os.access(path, os.X_OK):
                raise ValueError(f"executable not found: {path}")
        if args.cpus:
            command(["taskset", "-c", args.cpus, "true"])
        binary_hashes = {mode: sha256(path) for mode, path in binaries.items()}
        library_info = {mode: command(["ldd", path]) for mode, path in binaries.items()}
        if any("not found" in info for info in library_info.values()):
            raise ValueError("missing runtime library")
        for name, path in files.items():
            config[name] = str(path.resolve())
    else:
        if command(adb + ["get-state"]) != "device":
            raise ValueError("Android device is not ready")
        binary_hashes = {mode: shell("sha256sum " + shlex.quote(path)).split()[0]
                         for mode, path in binaries.items()}
        library_info = {"device": shell("uname -a")}
        for name, path in files.items():
            device_path = str(PurePosixPath(args.device_model) / path.name)
            if shell("sha256sum " + shlex.quote(device_path)).split()[0] != sha256(path):
                raise ValueError(f"device asset differs from local asset: {device_path}")
            config[name] = device_path
        shell("cat " + shlex.quote(args.temperature_path))
        if args.cpus:
            shell("taskset -c " + shlex.quote(args.cpus) + " true")
    cache = f"prompt budget={args.budget}, window={args.window_size}"
    print(f"Preflight passed: {args.platform}, policy=manifold_kv, contexts={args.contexts}, {cache}", flush=True)
    if args.check:
        return

    output = (args.output or REPO / "logs/manifold_kv" /
              datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="RUNNING", protocol=vars(args) | {"policy": "manifold_kv", "model": str(assets), "output": str(output)},
                    binary_sha256=binary_hashes, runtime_libraries=library_info,
                    model_sha256={k: sha256(v) for k, v in files.items()},
                    source_head=command(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
                    memory_script_sha256=sha256(MEMORY_SCRIPT), configs=configs, trials=[],
                    quality="not evaluated", environment={k: env[k] for k in
                        ("NNTR_ENGINE", "NNTR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    manifest["protocol"]["output"] = str(output)
    manifest["protocol"]["model"] = str(assets)
    if args.platform == "linux":
        manifest["runtime_library_sha256"] = {
            mode: {path: sha256(path) for path in re.findall(r"=> (/\S+)", info)}
            for mode, info in library_info.items()
        }
        manifest["build_options"] = {}
        for mode, binary in binaries.items():
            options = Path(binary).parent.parent.parent / "meson-info/intro-buildoptions.json"
            if options.is_file():
                manifest["build_options"][mode] = json.loads(options.read_text())
    remote = str(PurePosixPath(args.device_model) / (".manifold-eval-" + uuid.uuid4().hex)) if args.device_model else None
    remote_created = False
    rows = []

    def save():
        (output / "results.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        if rows:
            columns = list(dict.fromkeys(key for row in rows for key in row))
            with (output / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

    def cool():
        deadline = time.monotonic() + 300
        while True:
            temperature = int(shell("cat " + shlex.quote(args.temperature_path)))
            if temperature <= args.cool_below:
                return temperature
            if time.monotonic() >= deadline:
                raise RuntimeError("Android cooling timed out")
            time.sleep(5)

    try:
        save()
        with tempfile.TemporaryDirectory(prefix="nntr-manifold-eval-") as tmp:
            staging = Path(tmp)
            for name in ("config.json", "generation_config.json"):
                (staging / name).write_text(json.dumps(configs[name], indent=2) + "\n")
            if args.platform == "android":
                shell("mkdir " + shlex.quote(remote))
                remote_created = True
                for name in ("config.json", "generation_config.json"):
                    command(adb + ["push", str(staging / name), remote + "/" + name])
                command(adb + ["push", str(MEMORY_SCRIPT), remote + "/mem_usage.sh"])
                # exec keeps the recorded PID attached to the actual inference
                # process, even when mem_usage.sh starts it as a child.
                launcher = staging / "launch.sh"
                launcher.write_text("#!/system/bin/sh\necho $$ > " +
                                    shlex.quote(remote + "/inference.pid") +
                                    '\nexec "$@"\n')
                command(adb + ["push", str(launcher), remote + "/launch.sh"])
            for context_index, context in enumerate(args.contexts):
                for phase in ("latency", "memory"):
                    for trial in range(1, args.repeat + 1):
                        modes = list(binaries) if trial % 2 else list(reversed(binaries))
                        for mode in modes:
                            nntr = dict(config, batch_size=1, init_seq_len=context,
                                        max_seq_len=args.max_seq_len, num_to_generate=args.decode,
                                        sample_input=prompts[context])
                            nntr.pop("snapkv", None)
                            nntr.pop("streaming_llm", None)
                            nntr.pop("manifold_kv", None)
                            if mode == "manifold_kv":
                                nntr["manifold_kv"] = dict(budget=args.budget, window_size=args.window_size)
                            (staging / "nntr_config.json").write_text(json.dumps(nntr, indent=2) + "\n")
                            prefix = ["taskset", "-c", args.cpus] if args.cpus else []
                            temperature = "N/A"
                            if args.platform == "linux":
                                argv = prefix + (["bash", str(MEMORY_SCRIPT)] if phase == "memory" else [])
                                argv += [binaries[mode], str(staging)]
                                cwd = REPO
                            else:
                                temperature = cool()
                                command(adb + ["push", str(staging / "nntr_config.json"), remote + "/nntr_config.json"])
                                argv_remote = prefix + (["sh", remote + "/mem_usage.sh"] if phase == "memory" else [])
                                argv_remote += ["sh", remote + "/launch.sh", binaries[mode], remote]
                                script = "cd " + shlex.quote(str(PurePosixPath(binaries[mode]).parent)) + " && "
                                script += "export NNTR_ENGINE=cpu NNTR_NUM_THREADS=" + str(args.threads)
                                script += " OMP_NUM_THREADS=" + str(args.threads) + " OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=. && "
                                script += shlex.join(argv_remote)
                                argv = adb + ["shell", script]
                                cwd = REPO
                            name = f"{context}-{phase}-{trial}-{mode}"
                            print("START", name, flush=True)
                            started = time.monotonic()
                            completed = execute_trial(argv, cwd, env, args.timeout)
                            (output / (name + ".log")).write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr)
                            row = dict(context=context, phase=phase, trial=trial, mode=mode,
                                       temperature=temperature, wall_ms=(time.monotonic() - started) * 1000,
                                       valid=False, failure="", log=name + ".log")
                            manifest["trials"].append(dict(name=name, command=argv, config=nntr))
                            try:
                                if completed.returncode:
                                    raise ValueError(f"process exited {completed.returncode}")
                                marker = (f"ManifoldKV: prompt_budget={args.budget}, "
                                          f"window_size={args.window_size}, "
                                          f"retained_prompt_tokens={min(context, args.budget)}, "
                                          f"allocated_cache_rows={min(args.max_seq_len, args.budget + args.decode)}")
                                if mode == "manifold_kv" and marker not in completed.stdout:
                                    raise ValueError("executable did not confirm ManifoldKV activation")
                                if re.search(r"FATAL ERROR|truncating|falling back", completed.stdout + completed.stderr, re.IGNORECASE):
                                    raise ValueError("fatal, truncation or fallback marker")
                                row.update(parse_metrics(completed.stdout, context, args.decode, phase == "memory"), valid=True)
                            except ValueError as error:
                                row["failure"] = str(error)
                            rows.append(row)
                            save()
                            if not row["valid"]:
                                raise RuntimeError(f"invalid trial {name}: {row['failure']}")
                            print(f"DONE {name}: total={row['total_ms'] / 1000:.3f}s, "
                                  f"peak={row.get('sampled_peak_kib', row['os_peak_kib']) / 1024:.1f} MiB, "
                                  f"decode={row['generation_tokens']}", flush=True)
                            time.sleep(args.interval)
                completed_contexts = args.contexts[:context_index + 1]
                (output / "summary.md").write_text(summarize(rows, completed_contexts, args.repeat))
        (output / "summary.md").write_text(summarize(rows, args.contexts, args.repeat))
        manifest["status"] = "COMPLETE"
        print("Results:", output)
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        save()
        if args.platform == "android" and remote_created:
            # A lost adb shell connection does not reliably terminate its
            # remote child. Limit cleanup to the PID this staging run owns.
            shell("if [ -f " + shlex.quote(remote + "/inference.pid") + " ]; then "
                  "read worker < " + shlex.quote(remote + "/inference.pid") + "; "
                  "case $worker in ''|*[!0-9]*) ;; *) "
                  "if tr '\\0' ' ' < /proc/$worker/cmdline 2>/dev/null | "
                  "grep -F -- " + shlex.quote(remote) + " >/dev/null; then kill $worker; fi ;; esac; fi; "
                  "rm -rf " + shlex.quote(remote))


if __name__ == "__main__":
    main()
