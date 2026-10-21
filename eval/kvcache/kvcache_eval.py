#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate NNTrainer CausalLM methods against an unchanged baseline.

Covers the project metrics: end-to-end latency, sampled peak RSS and
throughput on fixed prompts (perf), and output quality on LongBench v1
(quality). Each executable is built from its own branch; this script only
runs, validates and reports them. Methods are not built in: a condition is
LABEL=BINARY plus the nntr_config.json keys it adds (--setting) and the log
lines that prove it ran (--expect). SnapKV, H2O and PyramidKV are presets.

  prepare  tokenize the official LongBench v1 dataset into a fixed cases file
  quality  generate and score LongBench predictions for every condition
  perf     latency and memory trials on a real LongBench prompt per length
  report   rewrite summary.md of an existing quality or perf run
"""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
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
TASKS = (
    "narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa "
    "musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum "
    "lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
).split()
# Official LongBench runs these tasks without chat wrapping, and scores only
# the first line of the prediction for FIRST_LINE tasks.
PLAIN = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
TRAILER = "\n\n=================[ LLM with NNTrainer ]"
BUILD_KEYS = (
    "platform", "buildtype", "enable-transformer", "enable-fp16", "enable-blas",
    "thread-backend", "nntr-num-threads", "openblas-num-threads",
    "enable-opencl", "enable-npu", "enable-htp", "enable-cuda",
)
DEVICE_KEYS = ("enable-opencl", "enable-npu", "enable-htp", "enable-cuda")
THREADS = dict(NNTR_NUM_THREADS="4", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
# Built-in presets: default settings under the nntr_config.json key of the
# same name, and a pattern the branch logs only while the method runs. Any
# other method is described on the command line (--setting / --expect).
PRESETS = {
    "snapkv": dict(settings=dict(budget=256, window=16, kernel=5, pooling="maxpool"), marker=r"\[SnapKV"),
    "h2o": dict(settings=dict(budget=256, recent=128), marker=r"\[H2O\]"),
    "pyramidkv": dict(settings=dict(budget=256, window=64, kernel=5, beta=20), marker=r"\[PyramidKV"),
}
# nntr_config.json keys the harness sets itself; a --setting may not change them.
CONTROLLED_KEYS = ("sample_input", "init_seq_len", "max_seq_len", "num_to_generate", "batch_size",
                   "model_file_name", "tokenizer_file", "embedding_file_name",
                   "chat_input", "system_prompt", "skip_prefill")
# Labels name log files and table columns.
LABEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
# The agreed memory metric is the unmodified repository script.
MEMORY_SCRIPT = REPO / "Applications/utils/mem_usage.sh"
MEMORY_SCRIPT_SHA = "f075e1f0d535128da1d6c2e7dbc5827bf442ff5d3da01be3ca1232074a9d6b62"
# Performance inputs are a real LongBench prompt truncated to each length. A
# long summarization case fills every length and rarely stops before 128 steps.
PERF_CASE = "gov_report:0"
PERF_COLUMNS = (
    "phase context tokens condition trial position prefill_tokens generation_tokens early_eos "
    "prefill_ms prefill_tps generation_ms generation_tps total_ms e2e_ms sampled_peak_rss_kib "
    "samples target_pid os_peak_rss_kib reported_peak_kib duration_s started_at valid failure stdout_log"
).split()
# (phase, key, label, direction of improvement); time/memory use the
# reduction rate and TPS the increase rate, both positive when better.
PERF_METRICS = (
    ("latency", "total_ms", "total (ms)", "lower"),
    ("latency", "e2e_ms", "e2e incl. loading (ms)", "lower"),
    ("latency", "prefill_tps", "prefill TPS", "higher"),
    ("latency", "generation_tps", "decode TPS", "higher"),
    ("memory", "sampled_peak_rss_mib", "sampled peak RSS (MiB)", "lower"),
    ("latency", "os_peak_rss_mib", "OS peak RSS, reference (MiB)", "lower"),
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_record(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class CaseBuilder:
    """LongBench v1 prompt construction, exactly as on the pyramidkv branch."""

    def __init__(self, model, official):
        from tokenizers import Tokenizer
        from jinja2 import Environment, StrictUndefined
        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        tokenizer_cfg = json.loads((model / "tokenizer_config.json").read_text())
        self.template = Environment(undefined=StrictUndefined).from_string(tokenizer_cfg["chat_template"])
        self.prompts = json.loads((official / "config/dataset2prompt.json").read_text())
        self.limits = json.loads((official / "config/dataset2maxlen.json").read_text())

    def build(self, task, index, example, max_input):
        tokenizer = self.tokenizer
        raw = self.prompts[task].format(**example)
        original = tokenizer.encode(raw, add_special_tokens=False).ids

        def wrap(text):
            return text if task in PLAIN else self.template.render(
                messages=[dict(role="user", content=text)], tools=None,
                add_generation_prompt=True, enable_thinking=False)

        # Official middle truncation, reserving the template overhead,
        # then checking the final token array against the limit.
        allowance = max_input - len(tokenizer.encode(wrap(""), add_special_tokens=False).ids) - 4
        if allowance < 2:
            raise ValueError("Context limit cannot hold the prompt template")
        was_truncated = len(original) > allowance
        while True:
            content = (tokenizer.decode(original[:allowance // 2], skip_special_tokens=True)
                       + tokenizer.decode(original[-(allowance - allowance // 2):], skip_special_tokens=True)
                       if len(original) > allowance else raw)
            prompt = wrap(content)
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if len(ids) <= max_input:
                break
            allowance -= len(ids) - max_input + 2
        return dict(task=task, index=index, prompt=prompt, tokens=len(ids),
                    prompt_ids_sha256=ids_sha(ids),
                    original_content_tokens=len(original), truncated=was_truncated,
                    max_new_tokens=self.limits[task], answers=example["answers"],
                    all_classes=example["all_classes"], length=example["length"])


def prepare_cases(args):
    builder = CaseBuilder(args.model, args.official)
    if args.cases.exists():
        raise ValueError("Refusing to replace existing cases")
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    counts, truncated, source_hashes = {}, {}, {}
    with args.cases.open("w", encoding="utf-8") as stream:
        for task in TASKS:
            path = args.data / (task + ".jsonl")
            source_hashes[task] = sha(path)
            counts[task] = truncated[task] = 0
            for index, line in enumerate(path.open(encoding="utf-8")):
                case = builder.build(task, index, json.loads(line), args.max_input)
                was_truncated = case["truncated"]
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
                counts[task] += 1
                truncated[task] += int(was_truncated)
            print(task, counts[task], "prepared", flush=True)
    if sum(counts.values()) != 4750:
        raise ValueError(f"Expected all 4750 LongBench v1 examples, got {counts}")
    write_json(args.cases.with_suffix(".manifest.json"), dict(
        counts=counts, total=sum(counts.values()), truncated=truncated,
        max_input_tokens=args.max_input, cases_sha256=sha(args.cases),
        dataset_sha256=source_hashes,
        official_sources={str(p.relative_to(args.official)): sha(p) for p in args.official.rglob("*.py")},
        official_config_sha256={str(p.relative_to(args.official)): sha(p)
                                for p in (args.official / "config").glob("*.json")},
        tokenizer_sha256=sha(args.model / "tokenizer.json"),
        tokenizer_config_sha256=sha(args.model / "tokenizer_config.json"),
        chat_policy="Qwen3 official template, enable_thinking=false; official six plain-prompt tasks",
        script_sha256=sha(Path(__file__))))


def load_scorer(official):
    sys.path.insert(0, str(official))
    spec = importlib.util.spec_from_file_location("official_longbench_eval", official / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if set(module.dataset2metric) != set(TASKS):
        raise ValueError("Unexpected official LongBench task metrics")
    return module


def special_tokens(tokenizer_file):
    """Strings of the tokenizer's special tokens, e.g. <|im_end|>."""
    added = json.loads(Path(tokenizer_file).read_text(encoding="utf-8")).get("added_tokens", [])
    return sorted({token["content"] for token in added if token.get("special")}, key=len, reverse=True)


def clean_prediction(pred, specials):
    """NNTrainer prints the EOS token text; the official prediction script
    decodes with skip_special_tokens=True, so drop special tokens before
    scoring (e.g. "unanswerable<|im_end|>" would otherwise score 0)."""
    for token in specials:
        pred = pred.replace(token, "")
    return pred


def score_prediction(scorer, case, pred):
    scored = pred.lstrip("\n").split("\n")[0] if case["task"] in FIRST_LINE else pred
    return max(scorer.dataset2metric[case["task"]](scored, answer, all_classes=case["all_classes"])
               for answer in case["answers"])


def pyramidkv_retained(settings, layer, length, layers):
    """Official PyramidKV integer schedule, as checked on the pyramidkv branch."""
    prefix = settings["budget"] - settings["window"]
    if length < settings["budget"]:
        return length
    if length < 2 * prefix or layers == 1:
        return settings["budget"]
    high = min(2 * prefix - prefix // settings["beta"], length - settings["window"])
    low = 2 * prefix - high
    return high - layer * ((high - low) // (layers - 1)) + settings["window"]


def preset_errors(method, settings, text, tokens, decode_steps):
    """Detailed checks of what each built-in method's branch logs when active."""
    if method == "snapkv":
        # SnapKV reports its compacted cache only when the prompt exceeds the
        # budget; the capacity reveals the budget actually applied.
        match = re.search(r"^\[SnapKV cache\]: \d+ -> \d+ bytes, capacity (\d+) tokens$", text, re.M)
        if bool(match) != (tokens > settings["budget"]):
            return ["SnapKV compression marker does not match the prompt length"]
        if match and int(match[1]) != settings["budget"] + decode_steps:
            return ["SnapKV capacity differs from budget + decode steps"]
    elif method == "h2o":
        if f"[H2O] h2o_budget={settings['budget']} h2o_recent={settings['recent']}" not in text:
            return ["missing H2O activation marker"]
    elif method == "pyramidkv":
        head = " ".join(f"pyramidkv_{key}={settings[key]}" for key in ("budget", "window", "kernel", "beta"))
        if "[PyramidKV] " + head not in text:
            return ["missing PyramidKV activation marker"]
        layers = re.findall(r"\[PyramidKV layer=(\d+)\] prompt=(\d+) retained=(\d+)", text)
        if not layers or len({layer for layer, _, _ in layers}) != len(layers):
            return ["missing or duplicate per-layer PyramidKV markers"]
        for layer, length, retained in layers:
            expected = pyramidkv_retained(settings, int(layer), int(length), len(layers))
            if int(length) != tokens or int(retained) != expected:
                return [f"incorrect PyramidKV layer {layer}: prompt={length}, retained={retained}, expected={expected}"]
    return []


def activation_errors(protocol, label, text, tokens, decode_steps):
    """Prove from the condition's own log that its method ran, and that the
    baseline shows none of the patterns any condition expects."""
    spec = protocol["condition_specs"][label]
    if label == "baseline":
        found = [p for p in protocol["baseline_forbidden"] if re.search(p, text, re.M)]
        return ["baseline log matches method pattern(s): " + ", ".join(found)] if found else []
    errors = []
    if spec["preset"]:
        errors += preset_errors(spec["preset"], spec["setting"][spec["preset"]], text, tokens, decode_steps)
    errors += [f"missing expected log line /{p}/" for p in spec["expect"] if not re.search(p, text, re.M)]
    return errors


def method_log_patterns(spec):
    """Log lines a method prints on purpose; never read as warnings."""
    return spec["expect"] + ([PRESETS[spec["preset"]]["marker"]] if spec["preset"] else [])


def parse_metrics(text, returncode, tokens, decode_steps, method_lines=()):
    metrics, errors = {}, []
    if returncode:
        errors.append(f"model exit {returncode}")
    for stage in ("prefill", "generation"):
        match = re.search(r"^" + stage + r": (\d+) tokens, (\d+) ms, \S+ TPS\s*$", text, re.M)
        if not match:
            errors.append("missing " + stage + " output")
            continue
        count, ms = int(match[1]), int(match[2])
        # The log prints inf for a zero-ms stage; keep TPS finite or empty.
        metrics.update({stage + "_tokens": count, stage + "_ms": ms,
                        stage + "_tps": count * 1000 / ms if ms else None})
    for key, pattern in (("total_ms", r"^total: (\d+) ms\s*$"), ("e2e_ms", r"^\[e2e time\]: (\d+) ms\s*$")):
        match = re.search(pattern, text, re.M)
        if match:
            metrics[key] = int(match[1])
        else:
            errors.append("missing " + key)
    # Reference only: "peak memory:" and ru_maxrss as printed by CausalLM.
    for key, pattern in (("reported_peak_kib", r"^peak memory: (\d+) KB"),
                         ("os_peak_rss_kib", r"^Max Resident Set Size: (\d+) KB")):
        match = re.search(pattern, text, re.M)
        if match:
            metrics[key] = int(match[1])
    if metrics.get("prefill_tokens") != tokens:
        errors.append("actual prompt token count differs from the requested input")
    generated = metrics.get("generation_tokens")
    if generated is None or not 0 <= generated <= decode_steps:
        errors.append("invalid decode count")
    else:
        metrics["early_eos"] = generated < decode_steps
    # A method's own activation lines may legitimately say "truncated" etc.
    scanned = "\n".join(line for line in text.splitlines()
                        if not any(re.search(p, line) for p in method_lines))
    warning = re.search(r".*(?:FATAL ERROR|truncat|heuristic|falling back).*", scanned, re.I)
    if warning:
        errors.append("fatal/truncation/fallback output: " + warning[0].strip()[:160])
    return metrics, errors


def parse_memory(stdout, stderr):
    """Validate the original mem_usage.sh log: target PID, positive samples, peak."""
    peak = re.search(r"^PID - (\d+)\s*:+\s*(\d+) kB", stdout, re.M)
    values = [int(v) for v in re.findall(r"^\d+ (\d+) kB(?: |$)", stdout, re.M)]
    memory = dict(samples=len(values), target_pid=int(peak[1]) if peak else None,
                  sampled_peak_rss_kib=int(peak[2]) if peak else None)
    errors = []
    if not peak or not values or min(values) <= 0 or int(peak[2]) != max(values):
        errors.append("missing or inconsistent sampled RSS")
    if re.search(r"/smaps.*(?:denied|No such|cannot)", stderr, re.I):
        errors.append("smaps read error")
    return memory, errors


def split_output(stdout, prompt):
    """Return the generated text and the log text outside the prompt/prediction.

    Model warnings are searched only in the latter, so a prediction that
    happens to contain a word like "truncated" is not mistaken for one.
    """
    stdout = stdout.replace("\r\n", "\n")
    marker = prompt.replace("\r\n", "\n") + "\n"
    if marker not in stdout:
        raise ValueError("Cannot find exact echoed prompt in model stdout")
    start = stdout.index(marker) + len(marker)
    end = stdout.rfind(TRAILER)
    if end < start:
        raise ValueError("Cannot find completed model timing trailer")
    return stdout[start:end], stdout[:stdout.index(marker)] + stdout[end:]


def run_model(command, env, stdout_path, stderr_path):
    process = None
    start = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err,
                                       **({} if WINDOWS else {"start_new_session": True}))
            code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                if WINDOWS:
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    # Undo the Windows CRT's LF -> CRLF translation before exact prompt matching.
    stdout = stdout_path.read_bytes().decode("utf-8", errors="strict")
    return code, process.pid, time.monotonic() - start, stdout.replace("\r\n", "\n") if WINDOWS else stdout


def load_build(binary):
    """Return the Meson options and C++ compiler that built an executable."""
    if not binary.is_file():
        raise ValueError(f"Build the executable first: {binary}")
    for root in binary.parents:
        info = root / "meson-info"
        if (info / "intro-buildoptions.json").is_file():
            options = {item["name"]: item["value"]
                       for item in json.loads((info / "intro-buildoptions.json").read_text(encoding="utf-8"))}
            compilers = info / "intro-compilers.json"
            compiler = (json.loads(compilers.read_text(encoding="utf-8")).get("host", {}).get("cpp", {})
                        .get("full_version") if compilers.is_file() else None)
            # Older trees predate some device options; absent means not built.
            return dict(options={key: options.get(key, False if key in DEVICE_KEYS else None)
                                 for key in BUILD_KEYS}, compiler=compiler)
    raise ValueError(f"No Meson build metadata found above {binary}")


def parse_pairs(values, what):
    pairs = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not rest or name in pairs:
            raise ValueError(f"Expected unique NAME=VALUE for {what}: {value}")
        pairs[name] = rest
    return pairs


def environment():
    info = dict(os=platform.platform(), machine=platform.machine(), logical_cpus=os.cpu_count(),
                python=sys.version.split()[0], temperature="N/A", frequency="N/A")
    if not WINDOWS:
        cpu = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(errors="replace"), re.M)
        mem = re.search(r"^MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        info.update(cpu=cpu[1].strip() if cpu else None, mem_total_kib=int(mem[1]) if mem else None)
    return info


def condition_specs(args, base_config):
    """Describe every condition without built-in knowledge of its method.

    A condition is LABEL=BINARY. A label naming a preset (or given --preset)
    gets that method's defaults and log checks. Any other method adds its
    nntr_config.json keys with --setting and proves it ran with --expect
    patterns; --unchecked accepts a run with no such proof, and says so.
    """
    binaries = {}
    for value in args.condition:
        label, sep, path = value.partition("=")
        if not sep or not path or label in binaries:
            raise ValueError(f"Expected unique LABEL=BINARY for --condition: {value}")
        if not LABEL_PATTERN.fullmatch(label):
            raise ValueError(f"Condition label must match {LABEL_PATTERN.pattern}: {label}")
        binaries[label] = Path(path).resolve()
    labels = (["baseline"] if "baseline" in binaries else []) + [label for label in binaries if label != "baseline"]
    presets = parse_pairs(args.preset, "--preset")
    settings = {label: json.loads(value) for label, value in parse_pairs(args.setting, "--setting").items()}
    expects = {}
    for value in args.expect:
        label, sep, pattern = value.partition("=")
        if not sep or not pattern:
            raise ValueError(f"Expected LABEL=REGEX for --expect: {value}")
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"Invalid --expect pattern for {label}: {error}") from error
        expects.setdefault(label, []).append(pattern)
    named = set(presets) | set(settings) | set(expects) | set(args.unchecked)
    if named - set(binaries):
        raise ValueError("Options name labels without a --condition: " + ", ".join(sorted(named - set(binaries))))
    if "baseline" in named:
        raise ValueError("baseline runs the model's own configuration; give it no preset, setting or expectation")
    specs = {"baseline": dict(preset=None, setting={}, expect=[], activation="baseline")} if "baseline" in binaries else {}
    for label in labels[1:] if "baseline" in binaries else labels:
        preset = presets.get(label, label if label in PRESETS else None)
        if preset is not None and preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset}; built-in presets: {', '.join(PRESETS)}")
        setting = settings.get(label, {})
        if not isinstance(setting, dict):
            raise ValueError(f"--setting {label} must be a JSON object of nntr_config.json keys")
        blocked = sorted(set(setting) & set(CONTROLLED_KEYS))
        if blocked:
            raise ValueError(f"--setting {label} cannot change harness-controlled keys: {', '.join(blocked)}")
        setting = dict(setting)
        if preset:
            override = setting.get(preset, {})
            if not isinstance(override, dict):
                raise ValueError(f"--setting {label}: {preset} must be a JSON object")
            setting[preset] = dict(PRESETS[preset]["settings"], **override)
        patterns = expects.get(label, [])
        if not preset and not patterns and label not in args.unchecked:
            raise ValueError(f"{label}: add --expect {label}=REGEX for a log line the method prints only when "
                             f"active, or --unchecked {label} to run without that proof")
        activation = "+".join(kind for kind, used in (("preset", preset), ("expect", patterns)) if used) or "unchecked"
        specs[label] = dict(preset=preset, setting=setting, expect=patterns, activation=activation)
    # Keys a method adds are removed from every other condition, baseline
    # included; overrides of keys the model already has stay per condition.
    method_keys = sorted(set(PRESETS) | {key for spec in specs.values() for key in spec["setting"]
                                         if key not in base_config})
    forbidden = sorted({preset["marker"] for preset in PRESETS.values()}
                       | {pattern for spec in specs.values() for pattern in spec["expect"]})
    return binaries, labels, specs, method_keys, forbidden


def prepare_conditions(args):
    """Shared setup: executables, method settings, model assets, builds, runtime."""
    configs = {name: json.loads((args.model / name).read_text(encoding="utf-8-sig"))
               for name in ("config.json", "generation_config.json", "nntr_config.json")}
    nntr = configs["nntr_config.json"]
    binaries, labels, specs, method_keys, forbidden = condition_specs(args, nntr)
    if any(key in nntr for key in ("chat_input", "system_prompt")) or nntr.get("skip_prefill"):
        raise ValueError("Use plain sample_input with a fresh, full prefill")
    assets = {}
    for key in ("model_file_name", "tokenizer_file", "embedding_file_name"):
        if nntr.get(key):
            path = Path(nntr[key])
            nntr[key] = str((path if path.is_absolute() else args.model / path).resolve())
            assets[key] = sha(Path(nntr[key]))

    builds = {name: load_build(binary) for name, binary in binaries.items()}
    build_difference = len({json.dumps(build, sort_keys=True) for build in builds.values()}) != 1
    if build_difference and not args.allow_build_difference:
        raise ValueError("Build options or compiler differ between conditions (--allow-build-difference compares "
                         "them anyway and marks the run): " + json.dumps(builds))
    for name, build in builds.items():
        if not build["options"]["enable-transformer"] or any(build["options"][key] for key in DEVICE_KEYS):
            raise ValueError(f"Build {name} for CPU with enable-transformer=true")

    envs = {}
    for name, binary in binaries.items():
        env = dict(os.environ, NNTR_ENGINE="cpu", **THREADS)
        env["PATH"] = os.pathsep.join([str(binary.parent), env.get("PATH", "")])
        envs[name] = env
    libraries = {}
    if not WINDOWS:
        subprocess.run(["taskset", "-c", args.cpus, "true"], check=True)
        for name, binary in binaries.items():
            loaded = subprocess.check_output(["ldd", str(binary)], env=envs[name], text=True)
            if "not found" in loaded:
                raise RuntimeError("Missing runtime libraries: " + loaded)
            libraries[name] = {p: sha(Path(p)) for p in re.findall(r"=> (\S+)", loaded) if Path(p).is_file()}
    protocol = dict(
        conditions=labels, condition_specs=specs, method_keys=method_keys, baseline_forbidden=forbidden,
        declared_revisions=parse_pairs(args.revision, "--revision"),
        batch_size=1, threads=THREADS, engine="cpu",
        cpu_affinity="not pinned" if WINDOWS else args.cpus,
        model_configs=configs, asset_sha256=assets,
        binaries={name: str(path) for name, path in binaries.items()},
        binary_sha256={name: sha(path) for name, path in binaries.items()},
        runtime_library_sha256=libraries, builds=builds, build_difference=build_difference,
        environment=environment(), script_sha256=sha(Path(__file__)))
    return protocol, envs


def open_run(out, protocol, records_name):
    """Start a run directory, or resume one whose protocol is unchanged."""
    out.mkdir(parents=True, exist_ok=True)
    manifest_path, records_path = out / "manifest.json", out / records_name
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["protocol"] != protocol:
            changed = sorted(k for k in set(protocol) | set(manifest["protocol"])
                             if protocol.get(k) != manifest["protocol"].get(k))
            raise ValueError("Cannot resume; protocol changed: " + ", ".join(changed))
        records = [json.loads(line) for line in records_path.open(encoding="utf-8")] if records_path.exists() else []
        if any(not r["valid"] for r in records):
            raise ValueError("Inspect invalid records before resuming")
    elif records_path.exists():
        raise ValueError("Records exist without a manifest")
    else:
        manifest, records = dict(protocol=protocol, started_at=now()), []
    return manifest_path, records_path, manifest, records


def model_config(protocol, label, **fields):
    """Write-ready nntr_config.json for one condition; other methods stay off."""
    config = dict(protocol["model_configs"]["nntr_config.json"], batch_size=1, **fields)
    for key in protocol["method_keys"] + ["chat_input", "system_prompt"]:
        config.pop(key, None)
    config.update(protocol["condition_specs"][label]["setting"])
    return config


def condition_lines(protocol):
    """Summary table of what each condition ran and how activation was proven."""
    specs = protocol.get("condition_specs")
    if not specs:
        return []
    lines = ["", "|Condition|Preset|nntr_config.json additions|Activation check|", "|---|---|---|---|"]
    for label in protocol["conditions"]:
        spec = specs[label]
        check = ("no method pattern may appear" if spec["activation"] == "baseline" else
                 "**UNCHECKED: activation not proven**" if spec["activation"] == "unchecked" else
                 " + ".join((["preset log checks"] if spec["preset"] else [])
                            + [f"`/{p}/`" for p in spec["expect"]]))
        setting = json.dumps(spec["setting"], ensure_ascii=False, separators=(",", ":")) if spec["setting"] else ""
        cells = [label, spec["preset"] or "", f"`{setting}`" if setting else "", check]
        lines.append("|" + "|".join(cell.replace("|", "\\|") for cell in cells) + "|")
    if protocol.get("build_difference"):
        lines += ["", "**Warning:** build options or compilers differ between conditions "
                      "(--allow-build-difference); see `builds` in manifest.json."]
    return lines


def log_paths(out, folder, stem):
    logs = out / "logs" / folder
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = logs / (stem + ".stdout.log"), logs / (stem + ".stderr.log")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    for partial in (stdout_path, stderr_path):
        if partial.exists():
            partial.rename(partial.with_name(partial.name + ".interrupted-" + stamp))
    return stdout_path, stderr_path


def mean_sd(values):
    if not values:
        return "N/A"
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}" if len(values) > 1 \
        else f"{values[0]:.2f} ± N/A"


# ---------------------------------------------------------------------------
# Quality: LongBench v1
# ---------------------------------------------------------------------------
def select_cases(path, tasks, samples_per_task):
    """Fixed dataset indices per task, independent of any observed score."""
    return [case for case in map(json.loads, path.open(encoding="utf-8"))
            if case["task"] in tasks and (samples_per_task is None or case["index"] < samples_per_task)]


def task_scores(records, task, label):
    selected = sorted((r for r in records if r["valid"] and r["task"] == task and r["condition"] == label),
                      key=lambda r: r["index"])
    if not selected:
        return None, 0
    return round(100 * sum(r["quality_score"] for r in selected) / len(selected), 2), len(selected)


def write_quality_summary(out, manifest, records):
    protocol = manifest["protocol"]
    labels, counts = protocol["conditions"], protocol["selection"]["counts"]
    deltas = [label for label in labels if label != "baseline"] if "baseline" in labels else []
    full = protocol["selection"]["scope"] == "FULL_DATASET"
    lines = ["# LongBench v1", "",
             "Full dataset evaluation." if full else
             "SUBSET: fixed dataset indices only; these are not full LongBench scores.", "",
             f"Input limit {protocol['max_input_tokens']} tokens. "
             f"Status: {manifest.get('status')}; valid predictions {len(records)}/{manifest['expected']}.", "",
             "|Task|" + "|".join(labels) + "".join(f"|{m} − baseline (pp)" for m in deltas) + "|n / expected|",
             "|---" * (1 + len(labels) + len(deltas)) + "|---:|"]
    complete = {}
    for task in counts:
        scores = {label: task_scores(records, task, label) for label in labels}
        cells = [f"{s:.2f}" if s is not None else "pending" for s, _ in scores.values()]
        for method in deltas:
            ready = scores[method][0] is not None and scores["baseline"][0] is not None
            cells.append(f"{scores[method][0] - scores['baseline'][0]:+.2f}" if ready else "")
        ns = "/".join(str(n) for _, n in scores.values())
        lines.append(f"|{task}|" + "|".join(cells) + f"|{ns} / {counts[task]}|")
        if all(n == counts[task] for _, n in scores.values()):
            complete[task] = {label: s for label, (s, _) in scores.items()}
    if complete:
        means = {label: sum(s[label] for s in complete.values()) / len(complete) for label in labels}
        cells = [f"**{means[label]:.2f}**" for label in labels]
        cells += [f"**{means[m] - means['baseline']:+.2f}**" for m in deltas]
        lines.append(f"|**Mean of {len(complete)} complete tasks**|" + "|".join(cells) + "||")
    invalid = [r for r in records if not r["valid"]]
    lines += ["", "Scores use each task's official metric on a 0-100 scale; the mean weights tasks equally.",
              "Greedy decoding, one prediction per case and condition. Per-case latency is recorded in",
              "records.jsonl but varies with early EOS; use `perf` for speed and memory claims.",
              f"Invalid predictions: {len(invalid)}."] + condition_lines(protocol)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "scores.json", {task: {label: dict(zip(("score", "n"), task_scores(records, task, label)))
                                            for label in labels} for task in counts})


def evaluate_quality(args):
    from tokenizers import Tokenizer
    cases_manifest = json.loads(args.cases.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if sha(args.cases) != cases_manifest["cases_sha256"]:
        raise ValueError("Cases file differs from its manifest")
    protocol, envs = prepare_conditions(args)
    if protocol["asset_sha256"]["tokenizer_file"] != cases_manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the one used to prepare the cases")
    tasks = TASKS if args.tasks is None else args.tasks.split(",")
    if not tasks or set(tasks) - set(TASKS) or len(set(tasks)) != len(tasks):
        raise ValueError("--tasks must name unique LongBench v1 tasks")
    if args.samples_per_task is not None and args.samples_per_task < 1:
        raise ValueError("--samples-per-task must be positive")
    cases = select_cases(args.cases, tasks, args.samples_per_task)
    counts = {task: sum(case["task"] == task for case in cases) for task in tasks}
    identities = [[case["task"], case["index"], case["prompt_ids_sha256"]] for case in cases]
    full = args.tasks is None and args.samples_per_task is None
    try:
        import Levenshtein  # noqa: F401  (fuzzywuzzy picks a faster, slightly different ratio)
        fuzz_backend = "python-Levenshtein"
    except ImportError:
        fuzz_backend = "difflib.SequenceMatcher"
    protocol.update(
        kind="quality", benchmark="LongBench v1",
        selection=dict(scope="FULL_DATASET" if full else "FIXED_SUBSET", tasks=tasks,
                       samples_per_task=args.samples_per_task, counts=counts, total=len(cases),
                       identity_sha256=ids_sha(identities)),
        cases=str(args.cases), cases_manifest=cases_manifest,
        max_input_tokens=cases_manifest["max_input_tokens"],
        max_seq_len=cases_manifest["max_input_tokens"] + 513,
        generation="greedy (do_sample=false, top_k=1, seed=42); official per-task max_new_tokens",
        first_token="init_seq_len=prompt_tokens+1 registers the prefill output; num_to_generate=max_new_tokens-1",
        samsum_eos="newline added to EOS, as in the official prediction script",
        scorer_sha256={name: sha(args.official / name) for name in ("eval.py", "metrics.py")},
        fuzzywuzzy_backend=fuzz_backend,
        prediction_cleanup="special tokens removed before scoring, as skip_special_tokens=True",
        special_tokens=special_tokens(protocol["model_configs"]["nntr_config.json"]["tokenizer_file"]))
    scorer = load_scorer(args.official)
    labels = protocol["conditions"]
    if args.check:
        print(f"Quality preflight passed: {len(cases)} cases x {len(labels)} conditions ({', '.join(labels)}); "
              f"input limit {protocol['max_input_tokens']}; scope={protocol['selection']['scope']}; no inference")
        return
    out = args.output
    manifest_path, records_path, manifest, records = open_run(out, protocol, "records.jsonl")
    done = {(r["task"], r["index"], r["condition"]) for r in records}
    manifest.update(status="RUNNING", controller_pid=os.getpid(), completed=len(records),
                    expected=len(cases) * len(labels))
    write_json(manifest_path, manifest)
    write_quality_summary(out, manifest, records)
    tokenizer = Tokenizer.from_file(protocol["model_configs"]["nntr_config.json"]["tokenizer_file"])
    newline = tokenizer.encode("\n", add_special_tokens=False).ids[-1]
    try:
        with tempfile.TemporaryDirectory(prefix="nntr-kvcache-quality-") as tmp:
            model = Path(tmp)
            (model / "config.json").write_text(json.dumps(protocol["model_configs"]["config.json"]), encoding="utf-8")
            for case in cases:
                ids = tokenizer.encode(case["prompt"], add_special_tokens=False).ids
                if len(ids) != case["tokens"] or ids_sha(ids) != case["prompt_ids_sha256"]:
                    raise ValueError("LongBench prompt tokens changed")
                # Rotate the condition order per sample to spread drift evenly.
                shift = case["index"] % len(labels)
                for label in labels[shift:] + labels[:shift]:
                    if (case["task"], case["index"], label) in done:
                        continue
                    generation = dict(protocol["model_configs"]["generation_config.json"],
                                      do_sample=False, top_k=1, top_p=1.0, temperature=1.0, seed=42)
                    if case["task"] == "samsum":
                        eos = generation["eos_token_id"]
                        eos = eos if isinstance(eos, list) else [eos]
                        generation["eos_token_id"] = list(dict.fromkeys(eos + [newline]))
                    decode_steps = case["max_new_tokens"] - 1
                    config = model_config(protocol, label, sample_input=case["prompt"],
                                          init_seq_len=case["tokens"] + 1, max_seq_len=protocol["max_seq_len"],
                                          num_to_generate=decode_steps)
                    (model / "nntr_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                    (model / "generation_config.json").write_text(json.dumps(generation), encoding="utf-8")
                    stem = f"{case['index']:04d}-{label}"
                    stdout_path, stderr_path = log_paths(out, case["task"], stem)
                    command = [protocol["binaries"][label], str(model)]
                    if not WINDOWS:
                        command = ["taskset", "-c", args.cpus] + command
                    started_at = now()
                    manifest["active"] = dict(task=case["task"], index=case["index"], condition=label,
                                              started_at=started_at)
                    write_json(manifest_path, manifest)
                    print("START", case["task"], stem, "tokens", case["tokens"], flush=True)
                    code, pid, duration, stdout = run_model(command, envs[label], stdout_path, stderr_path)
                    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                    pred, errors = "", []
                    try:
                        pred, log_text = split_output(stdout, case["prompt"])
                    except ValueError as error:
                        log_text = stdout
                        errors.append(str(error))
                    log_text += "\n" + stderr
                    metrics, metric_errors = parse_metrics(
                        log_text, code, case["tokens"], decode_steps,
                        method_log_patterns(protocol["condition_specs"][label]))
                    errors += metric_errors
                    errors += activation_errors(protocol, label, log_text, case["tokens"], decode_steps)
                    record = {k: v for k, v in case.items() if k != "prompt"}
                    scored = clean_prediction(pred, protocol["special_tokens"])
                    record.update(condition=label, pred=scored, raw_pred=pred, valid=not errors,
                                  failure="; ".join(errors),
                                  quality_score=None if errors else score_prediction(scorer, case, scored),
                                  pid=pid, duration_s=round(duration, 3), started_at=started_at,
                                  stdout_log=str(stdout_path.relative_to(out)),
                                  stderr_log=str(stderr_path.relative_to(out)), **metrics)
                    append_record(records_path, record)
                    records.append(record)
                    manifest.pop("active", None)
                    manifest["completed"] = len(records)
                    write_json(manifest_path, manifest)
                    write_quality_summary(out, manifest, records)
                    print("DONE", case["task"], stem, f"valid={not errors}",
                          f"score={record['quality_score']}", f"{len(records)}/{manifest['expected']}", flush=True)
                    if errors:
                        raise RuntimeError("Invalid LongBench run: " + record["failure"])
        # Cross-check the incremental per-case scores with the official scorer.
        for task in protocol["selection"]["counts"]:
            for label in labels:
                selected = sorted((r for r in records if r["task"] == task and r["condition"] == label),
                                  key=lambda r: r["index"])
                if len(selected) != protocol["selection"]["counts"][task]:
                    raise RuntimeError(f"Incomplete task/condition: {task}/{label}")
                official = scorer.scorer(task, [r["pred"] for r in selected], [r["answers"] for r in selected],
                                         selected[0]["all_classes"])
                if official != task_scores(records, task, label)[0]:
                    raise RuntimeError(f"Incremental score differs from the official scorer: {task}/{label}")
        manifest["status"] = "COMPLETE"
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        write_quality_summary(out, manifest, records)
    print("Results:", out / "summary.md")


# ---------------------------------------------------------------------------
# Performance: latency and sampled peak RSS (metric.md protocol)
# ---------------------------------------------------------------------------
def perf_values(records, phase, context, label, key):
    values = []
    for r in records:
        if not (r["valid"] and r["phase"] == phase and r["context"] == context and r["condition"] == label):
            continue
        # MiB views of the kB fields: MiB = kB / 1024.
        raw = r.get(key[:-4] + "_kib") if key.endswith("_mib") else r.get(key)
        if raw is not None:
            values.append(raw / 1024 if key.endswith("_mib") else raw)
    return values


def write_perf_summary(out, manifest, records):
    protocol = manifest["protocol"]
    labels = protocol["conditions"]
    deltas = [label for label in labels if label != "baseline"] if "baseline" in labels else []
    source = protocol["perf_input"]
    actual = ", ".join(f"{c}→{source['prompts'][str(c)]['tokens']}" for c in protocol["contexts"])
    lines = ["# Latency and memory", "",
             f"Input: LongBench v1 {source['task']} #{source['index']}, official prompt and middle truncation "
             f"(limit→actual tokens: {actual}); at most {protocol['n_gen']} decode steps, "
             f"max_seq_len {protocol['max_seq_len']}, original sampling/EOS, "
             f"{protocol['repetitions']} latency + {protocol['repetitions']} memory trials per input and condition.",
             f"Status: {manifest.get('status')}; valid trials {sum(r['valid'] for r in records)}/{manifest['expected']}.", "",
             "Mean ± sample SD over valid trials; function checks are excluded.", "",
             "|Input|Metric|" + "|".join(labels) + "".join(f"|{m} vs baseline (%)" for m in deltas) + "|n|",
             "|---|---" + "|---" * (len(labels) + len(deltas)) + "|---|"]
    for context in protocol["contexts"]:
        for phase, key, name, better in PERF_METRICS:
            values = {label: perf_values(records, phase, context, label, key) for label in labels}
            cells = [mean_sd(values[label]) for label in labels]
            for method in deltas:
                base, value = values["baseline"], values[method]
                if base and value and statistics.mean(base):
                    b, m = statistics.mean(base), statistics.mean(value)
                    cells.append(f"{((b - m) if better == 'lower' else (m - b)) / b * 100:+.2f}")
                else:
                    cells.append("N/A")
            ns = "/".join(str(len(values[label])) for label in labels)
            lines.append(f"|{context}|{name}|" + "|".join(cells) + f"|{ns}|")
        eos = [str(sum(1 for r in records if r["valid"] and r["phase"] == "latency" and r["context"] == context
                       and r["condition"] == label and r.get("early_eos"))) for label in labels]
        lines.append(f"|{context}|early EOS (latency trials)|" + "|".join(eos) + "|" * len(deltas) + "||")
    lines += ["",
              "Change follows metric.md: time and memory use (baseline − method) / baseline × 100, TPS uses",
              "(method − baseline) / baseline × 100, so positive is an improvement in both.",
              "Latency rows come from latency trials; sampled peak RSS comes from separate trials run under the",
              "original Applications/utils/mem_usage.sh (smaps Rss sum, 1 s sampling, can miss spikes, fewer",
              "samples below 30 s). OS peak RSS is CausalLM's ru_maxrss line, a reference only.",
              "Equal-work comparisons need equal decode counts; inspect early EOS and trials.csv.",
              "Temperature and CPU frequency are not controlled or recorded on WSL (N/A).",
              f"Invalid trials: {sum(not r['valid'] for r in records)}."] + condition_lines(protocol)
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out / "trials.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PERF_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (out / "trials.csv.tmp").replace(out / "trials.csv")


def evaluate_perf(args):
    if WINDOWS:
        raise ValueError("perf measures memory with the original mem_usage.sh and supports Linux only")
    if sha(MEMORY_SCRIPT) != MEMORY_SCRIPT_SHA:
        raise ValueError("The original Applications/utils/mem_usage.sh changed")
    protocol, envs = prepare_conditions(args)
    task, _, index = args.case.partition(":")
    if task not in TASKS or not index.isdigit():
        raise ValueError("--case must be TASK:INDEX of a LongBench v1 task")
    path = args.data / (task + ".jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    if int(index) >= len(lines):
        raise ValueError(f"{task} has no index {index}")
    example = json.loads(lines[int(index)])
    builder = CaseBuilder(args.model, args.official)
    if protocol["asset_sha256"]["tokenizer_file"] != sha(args.model / "tokenizer.json"):
        raise ValueError("Model tokenizer_file differs from the model directory's tokenizer.json")
    cases = {context: builder.build(task, int(index), example, context) for context in args.contexts}
    for context, case in cases.items():
        # Each length must be a truncation of the same longer document.
        if not case["truncated"]:
            raise ValueError(f"{args.case} has only {case['original_content_tokens']} content tokens; "
                             f"choose a case longer than {context}")
    # NNTrainer registers the prefill's first token only when the prompt is
    # shorter than init_seq_len (causal_lm.cpp run()); +1 keeps n_gen decode steps.
    # The cache capacity, and so the allocation, stays 2048 + n_gen (2176) when
    # a subset of lengths is measured, so memory remains comparable.
    max_seq_len = args.max_seq_len or max(max(args.contexts), 2048) + args.n_gen
    if max(case["tokens"] for case in cases.values()) + 1 + args.n_gen > max_seq_len:
        raise ValueError("--max-seq-len must hold the longest prompt + 1 + --n-gen")
    protocol.update(
        kind="perf", contexts=args.contexts, n_gen=args.n_gen, max_seq_len=max_seq_len,
        repetitions=args.repeat, interval_seconds=args.interval, warmup=0,
        perf_input=dict(benchmark="LongBench v1", task=task, index=int(index), dataset_sha256=sha(path),
                        prompt_config_sha256=sha(args.official / "config/dataset2prompt.json"),
                        prompts={str(c): {k: case[k] for k in ("tokens", "prompt_ids_sha256", "truncated",
                                                               "original_content_tokens")}
                                 for c, case in cases.items()}),
        first_token="init_seq_len=prompt_tokens+1 registers the prefill output",
        generation="original model sampling and EOS", memory_script_sha256=MEMORY_SCRIPT_SHA)
    prompts = {context: case["prompt"] for context, case in cases.items()}
    labels = protocol["conditions"]
    plan = [("function", min(args.contexts), label, 0, position) for position, label in enumerate(labels, 1)]
    for context in args.contexts:
        for phase in ("latency", "memory"):
            for trial in range(1, args.repeat + 1):
                # Rotate the condition order per trial (alternation for two).
                shift = (trial - 1) % len(labels)
                order = labels[shift:] + labels[:shift]
                plan += [(phase, context, label, trial, position) for position, label in enumerate(order, 1)]
    if args.check:
        print(f"Perf preflight passed: inputs {args.contexts}, n_gen {args.n_gen}, max_seq_len {max_seq_len}, "
              f"{len(plan)} trials ({', '.join(labels)}); no inference")
        return
    out = args.output
    manifest_path, records_path, manifest, records = open_run(out, protocol, "trials.jsonl")
    done = {(r["phase"], r["context"], r["condition"], r["trial"]) for r in records}
    manifest.update(status="RUNNING", controller_pid=os.getpid(), completed=len(records), expected=len(plan))
    write_json(manifest_path, manifest)
    write_perf_summary(out, manifest, records)
    try:
        with tempfile.TemporaryDirectory(prefix="nntr-kvcache-perf-") as tmp:
            model = Path(tmp)
            for name in ("config.json", "generation_config.json"):
                (model / name).write_text(json.dumps(protocol["model_configs"][name]), encoding="utf-8")
            for phase, context, label, trial, position in plan:
                if (phase, context, label, trial) in done:
                    continue
                if records:
                    time.sleep(args.interval)
                tokens = cases[context]["tokens"]
                config = model_config(protocol, label, sample_input=prompts[context], init_seq_len=tokens + 1,
                                      max_seq_len=max_seq_len, num_to_generate=args.n_gen)
                (model / "nntr_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                stem = f"{phase}-{context}-{label}-{trial:02d}"
                stdout_path, stderr_path = log_paths(out, "perf", stem)
                # Memory trials run the executable under the original script,
                # which samples the model PID itself; latency trials run it alone.
                command = (["taskset", "-c", args.cpus] + (["bash", str(MEMORY_SCRIPT)] if phase == "memory" else [])
                           + [protocol["binaries"][label], str(model)])
                started_at = now()
                manifest["active"] = dict(phase=phase, context=context, condition=label, trial=trial,
                                          started_at=started_at)
                write_json(manifest_path, manifest)
                print("START", stem, flush=True)
                code, pid, duration, stdout = run_model(command, envs[label], stdout_path, stderr_path)
                stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                errors = []
                try:
                    _, log_text = split_output(stdout, prompts[context])
                except ValueError as error:
                    log_text = stdout
                    errors.append(str(error))
                log_text += "\n" + stderr
                metrics, metric_errors = parse_metrics(log_text, code, tokens, args.n_gen,
                                                       method_log_patterns(protocol["condition_specs"][label]))
                errors += metric_errors
                errors += activation_errors(protocol, label, log_text, tokens, args.n_gen)
                if phase == "memory":
                    memory, memory_errors = parse_memory(stdout, stderr)
                    metrics.update(memory)
                    errors += memory_errors
                record = dict(phase=phase, context=context, tokens=tokens, condition=label, trial=trial,
                              position=position,
                              started_at=started_at, pid=pid, duration_s=round(duration, 3),
                              valid=not errors, failure="; ".join(errors),
                              stdout_log=str(stdout_path.relative_to(out)),
                              stderr_log=str(stderr_path.relative_to(out)), **metrics)
                append_record(records_path, record)
                records.append(record)
                manifest.pop("active", None)
                manifest["completed"] = len(records)
                write_json(manifest_path, manifest)
                write_perf_summary(out, manifest, records)
                print(f"DONE {stem}: valid={not errors} decode={metrics.get('generation_tokens')} "
                      f"total={metrics.get('total_ms')} ms RSS={metrics.get('sampled_peak_rss_kib')} kB "
                      f"{len(records)}/{manifest['expected']}", flush=True)
                if errors:
                    raise RuntimeError(f"Invalid trial {stem}: {record['failure']}")
        manifest["status"] = "COMPLETE"
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        manifest["updated_at"] = now()
        write_json(manifest_path, manifest)
        write_perf_summary(out, manifest, records)
    print("Results:", out / "summary.md")


def report(args):
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    perf = manifest["protocol"].get("kind") == "perf"
    path = args.output / ("trials.jsonl" if perf else "records.jsonl")
    records = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    (write_perf_summary if perf else write_quality_summary)(args.output, manifest, records)
    print((args.output / "summary.md").read_text(encoding="utf-8"), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--model", type=Path, required=True, help="model directory with nntr_config.json")
    longbench = argparse.ArgumentParser(add_help=False)
    longbench.add_argument("--official", type=Path, required=True,
                           help="official LongBench v1 directory containing eval.py, metrics.py and config/")
    longbench.add_argument("--cases", type=Path, required=True, help="prepared cases JSONL")
    conditions = argparse.ArgumentParser(add_help=False)
    conditions.add_argument("--condition", action="append", required=True, metavar="LABEL=BINARY",
                            help="repeatable; LABEL is free ([A-Za-z0-9._-]), 'baseline' is the reference, "
                                 "and a preset name (" + ", ".join(PRESETS) + ") applies that preset")
    conditions.add_argument("--setting", action="append", default=[], metavar="LABEL=JSON",
                            help="nntr_config.json keys this condition adds, e.g. capkv-r50='{\"capkv\": "
                                 "{\"ratio\": 0.5}}'; for a preset, its key merges with the preset defaults")
    conditions.add_argument("--expect", action="append", default=[], metavar="LABEL=REGEX",
                            help="repeatable; a log line the method prints only when active (multiline regex); "
                                 "must appear for LABEL and must not appear for baseline")
    conditions.add_argument("--preset", action="append", default=[], metavar="LABEL=PRESET",
                            help="apply a preset's defaults and log checks to a differently named label")
    conditions.add_argument("--unchecked", action="append", default=[], metavar="LABEL",
                            help="run a custom condition without --expect; marked UNCHECKED in the summary")
    conditions.add_argument("--allow-build-difference", action="store_true",
                            help="compare executables whose Meson options or compiler differ; the run is marked")
    conditions.add_argument("--revision", action="append", default=[], metavar="LABEL=REV",
                            help="record the source revision each executable was built from")
    conditions.add_argument("--cpus", default="0,2,4,6", help="Linux taskset affinity")
    conditions.add_argument("--output", type=Path, help="run directory; an existing one is resumed")
    conditions.add_argument("--check", action="store_true", help="validate inputs without inference")

    prepare = commands.add_parser("prepare", parents=[model, longbench], help="tokenize the official dataset once")
    prepare.add_argument("--data", type=Path, required=True, help="extracted v1 JSONL directory, not LongBench-E")
    prepare.add_argument("--max-input", type=int, default=2048, help="prompt token limit (default 2048)")

    quality = commands.add_parser("quality", parents=[model, longbench, conditions],
                                  help="generate and score LongBench predictions")
    quality.add_argument("--tasks", help="comma-separated task subset (default: all 21)")
    quality.add_argument("--samples-per-task", type=int, help="fixed subset: dataset indices 0..N-1 of each task")

    perf = commands.add_parser("perf", parents=[model, conditions], help="latency and memory trials")
    perf.add_argument("--official", type=Path, required=True,
                      help="official LongBench v1 directory containing eval.py, metrics.py and config/")
    perf.add_argument("--data", type=Path, required=True, help="extracted v1 JSONL directory, not LongBench-E")
    perf.add_argument("--case", default=PERF_CASE, help=f"LongBench TASK:INDEX used as input (default {PERF_CASE})")
    perf.add_argument("--contexts", default="512,1024,2048", help="comma-separated prompt token limits")
    perf.add_argument("--n-gen", type=int, default=128, help="maximum decode steps (default 128)")
    perf.add_argument("--max-seq-len", type=int, help="cache capacity (default: max(2048, longest limit) + n-gen)")
    perf.add_argument("--repeat", type=int, default=3, help="latency and memory trials each (default 3)")
    perf.add_argument("--interval", type=float, default=1.0, help="seconds between trials (default 1)")

    summary = commands.add_parser("report", help="rewrite summary.md from records")
    summary.add_argument("output", type=Path)

    args = parser.parse_args()
    for field in ("model", "official", "cases", "data", "output"):
        if getattr(args, field, None) is not None:
            setattr(args, field, getattr(args, field).resolve())
    if args.command in ("quality", "perf") and args.output is None and not args.check:
        parser.error(args.command + " requires --output unless --check is given")
    if args.command == "prepare":
        if args.max_input < 2:
            parser.error("--max-input must be at least 2")
        prepare_cases(args)
    elif args.command == "quality":
        evaluate_quality(args)
    elif args.command == "perf":
        try:
            args.contexts = [int(n) for n in args.contexts.split(",")]
        except ValueError:
            parser.error("--contexts must be comma-separated integers")
        if (not args.contexts or min(args.contexts) < 1 or len(set(args.contexts)) != len(args.contexts)
                or args.n_gen < 1 or args.repeat < 1 or args.interval < 0):
            parser.error("positive unique contexts, n-gen and repeat, and a nonnegative interval are required")
        evaluate_perf(args)
    else:
        report(args)


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, terminate)
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed records and raw logs were retained.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print("Error:", error, file=sys.stderr)
        sys.exit(1)
