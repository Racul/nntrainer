#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LongBench v1 output-quality evaluation for NNTrainer CausalLM executables.

Compares a baseline executable with KV-cache method executables (SnapKV, H2O,
PyramidKV) using the protocol of eval/pyramidkv/longbench_eval.py from the
pyramidkv branch: official prompts and metrics, middle truncation, the Qwen3
non-thinking chat template and greedy decoding. Each executable is built from
its own branch; this script only runs and scores them.

  prepare  tokenize the official dataset into a fixed cases file
  run      generate and score predictions for every condition
  report   rewrite summary.md of an existing run
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
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
# Default settings, written under the method's own nntr_config.json key.
METHODS = {
    "snapkv": dict(budget=256, window=16, kernel=5, pooling="maxpool"),
    "h2o": dict(budget=256, recent=128),
    "pyramidkv": dict(budget=256, window=64, kernel=5, beta=20),
}
MARKERS = {"snapkv": "[SnapKV", "h2o": "[H2O]", "pyramidkv": "[PyramidKV"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare_cases(args):
    """Build prompts exactly as the pyramidkv branch does, so cases match."""
    from tokenizers import Tokenizer
    from jinja2 import Environment, StrictUndefined
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer_cfg = json.loads((args.model / "tokenizer_config.json").read_text())
    template = Environment(undefined=StrictUndefined).from_string(tokenizer_cfg["chat_template"])
    prompts = json.loads((args.official / "config/dataset2prompt.json").read_text())
    limits = json.loads((args.official / "config/dataset2maxlen.json").read_text())
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
                example = json.loads(line)
                raw = prompts[task].format(**example)
                original = tokenizer.encode(raw, add_special_tokens=False).ids

                def wrap(text):
                    return text if task in PLAIN else template.render(
                        messages=[dict(role="user", content=text)], tools=None,
                        add_generation_prompt=True, enable_thinking=False)

                # Official middle truncation, reserving the template overhead,
                # then checking the final token array against the limit.
                allowance = args.max_input - len(tokenizer.encode(wrap(""), add_special_tokens=False).ids) - 4
                if allowance < 2:
                    raise ValueError("Context limit cannot hold the prompt template")
                was_truncated = len(original) > allowance
                while True:
                    content = (tokenizer.decode(original[:allowance // 2], skip_special_tokens=True)
                               + tokenizer.decode(original[-(allowance - allowance // 2):], skip_special_tokens=True)
                               if len(original) > allowance else raw)
                    prompt = wrap(content)
                    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                    if len(ids) <= args.max_input:
                        break
                    allowance -= len(ids) - args.max_input + 2
                case = dict(task=task, index=index, prompt=prompt, tokens=len(ids),
                            prompt_ids_sha256=ids_sha(ids),
                            original_content_tokens=len(original), truncated=was_truncated,
                            max_new_tokens=limits[task], answers=example["answers"],
                            all_classes=example["all_classes"], length=example["length"])
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


def activation_errors(method, settings, text, tokens, decode_steps):
    """Check from each branch's own log lines that only the requested policy ran."""
    others = sorted(m for m, marker in MARKERS.items() if m != method and marker in text)
    if others:
        return ["unexpected activation marker: " + ", ".join(others)]
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


def parse_metrics(text, returncode, tokens, decode_steps):
    metrics, errors = {}, []
    if returncode:
        errors.append(f"model exit {returncode}")
    for stage in ("prefill", "generation"):
        match = re.search(r"^" + stage + r": (\d+) tokens, (\d+) ms, \S+ TPS\s*$", text, re.M)
        if not match:
            errors.append("missing " + stage + " output")
            continue
        metrics.update({stage + "_tokens": int(match[1]), stage + "_ms": int(match[2])})
    for key, pattern in (("total_ms", r"^total: (\d+) ms\s*$"), ("e2e_ms", r"^\[e2e time\]: (\d+) ms\s*$")):
        match = re.search(pattern, text, re.M)
        if match:
            metrics[key] = int(match[1])
        else:
            errors.append("missing " + key)
    if metrics.get("prefill_tokens") != tokens:
        errors.append("actual prompt token count differs from the prepared case")
    generated = metrics.get("generation_tokens")
    if generated is None or not 0 <= generated <= decode_steps:
        errors.append("invalid decode count")
    else:
        metrics["early_eos"] = generated < decode_steps
    if re.search(r"FATAL ERROR|truncat|heuristic|falling back", text, re.I):
        errors.append("fatal/truncation/fallback output")
    return metrics, errors


def split_output(stdout, prompt):
    """Return the prediction and the log text outside the prompt/prediction.

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
    if not binary.is_file():
        raise ValueError(f"Build the executable first: {binary}")
    for root in binary.parents:
        info = root / "meson-info/intro-buildoptions.json"
        if info.is_file():
            options = {item["name"]: item["value"] for item in json.loads(info.read_text(encoding="utf-8"))}
            # Older trees predate some device options; absent means not built.
            return {key: options.get(key, False if key in DEVICE_KEYS else None) for key in BUILD_KEYS}
    raise ValueError(f"No Meson build metadata found above {binary}")


def parse_pairs(values, what):
    pairs = {}
    for value in values:
        name, sep, rest = value.partition("=")
        if not sep or not rest or name in pairs:
            raise ValueError(f"Expected unique NAME=VALUE for {what}: {value}")
        pairs[name] = rest
    return pairs


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


def write_summary(out, manifest, records):
    protocol = manifest["protocol"]
    labels, counts = protocol["conditions"], protocol["selection"]["counts"]
    methods = [label for label in labels if label != "baseline"]
    deltas = methods if "baseline" in labels else []
    full = protocol["selection"]["scope"] == "FULL_DATASET"
    lines = ["# LongBench v1", "",
             "Full dataset evaluation." if full else
             "SUBSET: fixed dataset indices only; these are not full LongBench scores.", "",
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
    lines += ["", "Scores use each task's official metric on a 0-100 scale; the mean weights tasks equally.",
              "Greedy decoding, one prediction per case and condition. Per-case latency is recorded in",
              "records.jsonl but varies with early EOS; use the separate performance protocol for speed claims."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "scores.json", {task: {label: dict(zip(("score", "n"), task_scores(records, task, label)))
                                            for label in labels} for task in counts})


def build_protocol(args):
    cases_manifest = json.loads(args.cases.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if sha(args.cases) != cases_manifest["cases_sha256"]:
        raise ValueError("Cases file differs from its manifest")
    binaries = {name: Path(path).resolve() for name, path in parse_pairs(args.condition, "--condition").items()}
    unknown = set(binaries) - {"baseline"} - set(METHODS)
    if unknown:
        raise ValueError("Unknown condition(s): " + ", ".join(sorted(unknown)))
    settings = {}
    overrides = {name: json.loads(value) for name, value in parse_pairs(args.config, "--config").items()}
    if set(overrides) - (set(binaries) - {"baseline"}):
        raise ValueError("--config names a method that is not a condition")
    for name in binaries:
        if name != "baseline":
            settings[name] = dict(METHODS[name], **overrides.get(name, {}))
    labels = sorted(binaries, key=lambda name: (name != "baseline", name))

    configs = {name: json.loads((args.model / name).read_text(encoding="utf-8-sig"))
               for name in ("config.json", "generation_config.json", "nntr_config.json")}
    nntr = configs["nntr_config.json"]
    if any(key in nntr for key in ("chat_input", "system_prompt")) or nntr.get("skip_prefill"):
        raise ValueError("Use plain sample_input with a fresh, full prefill")
    assets = {}
    for key in ("model_file_name", "tokenizer_file", "embedding_file_name"):
        if nntr.get(key):
            path = Path(nntr[key])
            nntr[key] = str((path if path.is_absolute() else args.model / path).resolve())
            assets[key] = sha(Path(nntr[key]))
    if assets["tokenizer_file"] != cases_manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the one used to prepare the cases")

    builds = {name: load_build(binary) for name, binary in binaries.items()}
    if len({json.dumps(options, sort_keys=True) for options in builds.values()}) != 1:
        raise ValueError("Meson build options differ between conditions: " + json.dumps(builds))
    options = next(iter(builds.values()))
    if not options["enable-transformer"] or any(options[key] for key in DEVICE_KEYS):
        raise ValueError("Build every condition for CPU with enable-transformer=true")

    tasks = TASKS if args.tasks is None else args.tasks.split(",")
    if not tasks or set(tasks) - set(TASKS) or len(set(tasks)) != len(tasks):
        raise ValueError("--tasks must name unique LongBench v1 tasks")
    if args.samples_per_task is not None and args.samples_per_task < 1:
        raise ValueError("--samples-per-task must be positive")
    cases = select_cases(args.cases, tasks, args.samples_per_task)
    counts = {task: sum(case["task"] == task for case in cases) for task in tasks}
    identities = [[case["task"], case["index"], case["prompt_ids_sha256"]] for case in cases]
    full = args.tasks is None and args.samples_per_task is None

    envs = {}
    for name, binary in binaries.items():
        env = dict(os.environ, NNTR_ENGINE="cpu", **THREADS)
        env["PATH"] = os.pathsep.join([str(binary.parent), env.get("PATH", "")])
        envs[name] = env
    libraries = {}
    if not WINDOWS:
        for name, binary in binaries.items():
            loaded = subprocess.check_output(["ldd", str(binary)], env=envs[name], text=True)
            if "not found" in loaded:
                raise RuntimeError("Missing runtime libraries: " + loaded)
            libraries[name] = {p: sha(Path(p)) for p in re.findall(r"=> (\S+)", loaded) if Path(p).is_file()}
    try:
        import Levenshtein  # noqa: F401  (fuzzywuzzy picks a faster, slightly different ratio)
        fuzz_backend = "python-Levenshtein"
    except ImportError:
        fuzz_backend = "difflib.SequenceMatcher"

    protocol = dict(
        benchmark="LongBench v1", conditions=labels, methods=settings,
        declared_revisions=parse_pairs(args.revision, "--revision"),
        selection=dict(scope="FULL_DATASET" if full else "FIXED_SUBSET", tasks=tasks,
                       samples_per_task=args.samples_per_task, counts=counts, total=len(cases),
                       identity_sha256=ids_sha(identities)),
        cases=str(args.cases), cases_manifest=cases_manifest,
        max_input_tokens=cases_manifest["max_input_tokens"],
        max_seq_len=cases_manifest["max_input_tokens"] + 513,
        generation="greedy (do_sample=false, top_k=1, seed=42); official per-task max_new_tokens",
        first_token="init_seq_len=prompt_tokens+1 registers the prefill output; num_to_generate=max_new_tokens-1",
        samsum_eos="newline added to EOS, as in the official prediction script",
        batch_size=1, threads=THREADS, engine="cpu",
        cpu_affinity="not pinned" if WINDOWS else args.cpus,
        model_configs=configs, asset_sha256=assets,
        binaries={name: str(path) for name, path in binaries.items()},
        binary_sha256={name: sha(path) for name, path in binaries.items()},
        runtime_library_sha256=libraries, build_options=options,
        scorer_sha256={name: sha(args.official / name) for name in ("eval.py", "metrics.py")},
        fuzzywuzzy_backend=fuzz_backend,
        environment=dict(os=platform.platform(), machine=platform.machine(),
                         logical_cpus=os.cpu_count(), python=sys.version.split()[0]),
        script_sha256=sha(Path(__file__)))
    return protocol, envs, cases


def evaluate(args):
    from tokenizers import Tokenizer
    protocol, envs, cases = build_protocol(args)
    scorer = load_scorer(args.official)
    if not WINDOWS:
        subprocess.run(["taskset", "-c", args.cpus, "true"], check=True)
    if args.check:
        print(f"LongBench preflight passed: {protocol['selection']['total']} cases x "
              f"{len(protocol['conditions'])} conditions ({', '.join(protocol['conditions'])}); "
              f"scope={protocol['selection']['scope']}; no inference performed")
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest_path, records_path = out / "manifest.json", out / "records.jsonl"
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
        manifest, records = dict(protocol=protocol, started_at=datetime.now().astimezone().isoformat()), []
    done = {(r["task"], r["index"], r["condition"]) for r in records}
    labels = protocol["conditions"]
    manifest.update(status="RUNNING", controller_pid=os.getpid(), completed=len(records),
                    expected=protocol["selection"]["total"] * len(labels))
    write_json(manifest_path, manifest)
    write_summary(out, manifest, records)
    tokenizer = Tokenizer.from_file(protocol["model_configs"]["nntr_config.json"]["tokenizer_file"])
    newline = tokenizer.encode("\n", add_special_tokens=False).ids[-1]
    try:
        with tempfile.TemporaryDirectory(prefix="nntr-longbench-") as tmp:
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
                    config = dict(protocol["model_configs"]["nntr_config.json"], sample_input=case["prompt"],
                                  init_seq_len=case["tokens"] + 1, max_seq_len=protocol["max_seq_len"],
                                  num_to_generate=decode_steps, batch_size=1)
                    for field in list(METHODS) + ["chat_input", "system_prompt"]:
                        config.pop(field, None)
                    if label != "baseline":
                        config[label] = protocol["methods"][label]
                    (model / "nntr_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                    (model / "generation_config.json").write_text(json.dumps(generation), encoding="utf-8")
                    logs = out / "logs" / case["task"]
                    logs.mkdir(parents=True, exist_ok=True)
                    stem = f"{case['index']:04d}-{label}"
                    stdout_path, stderr_path = logs / (stem + ".stdout.log"), logs / (stem + ".stderr.log")
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    for partial in (stdout_path, stderr_path):
                        if partial.exists():
                            partial.rename(partial.with_name(partial.name + ".interrupted-" + stamp))
                    command = [protocol["binaries"][label], str(model)]
                    if not WINDOWS:
                        command = ["taskset", "-c", args.cpus] + command
                    started_at = datetime.now().astimezone().isoformat()
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
                    metrics, metric_errors = parse_metrics(log_text, code, case["tokens"], decode_steps)
                    errors += metric_errors
                    errors += activation_errors(label if label != "baseline" else None,
                                                protocol["methods"].get(label), log_text,
                                                case["tokens"], decode_steps)
                    record = {k: v for k, v in case.items() if k != "prompt"}
                    record.update(condition=label, pred=pred, valid=not errors, failure="; ".join(errors),
                                  quality_score=None if errors else score_prediction(scorer, case, pred),
                                  pid=pid, duration_s=round(duration, 3), started_at=started_at,
                                  stdout_log=str(stdout_path.relative_to(out)),
                                  stderr_log=str(stderr_path.relative_to(out)), **metrics)
                    with records_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    records.append(record)
                    manifest.pop("active", None)
                    manifest["completed"] = len(records)
                    write_json(manifest_path, manifest)
                    write_summary(out, manifest, records)
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
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(manifest_path, manifest)
        write_summary(out, manifest, records)
    print("Results:", out / "summary.md")


def report(args):
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    path = args.output / "records.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    write_summary(args.output, manifest, records)
    print((args.output / "summary.md").read_text(encoding="utf-8"), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", type=Path, required=True, help="model directory with nntr_config.json")
    shared.add_argument("--official", type=Path, required=True,
                        help="official LongBench v1 directory containing eval.py, metrics.py and config/")
    shared.add_argument("--cases", type=Path, required=True, help="prepared cases JSONL")

    prepare = commands.add_parser("prepare", parents=[shared], help="tokenize the official dataset once")
    prepare.add_argument("--data", type=Path, required=True, help="extracted v1 JSONL directory, not LongBench-E")
    prepare.add_argument("--max-input", type=int, default=8192)

    run = commands.add_parser("run", parents=[shared], help="generate and score predictions")
    run.add_argument("--condition", action="append", required=True, metavar="NAME=BINARY",
                     help="repeatable; NAME is baseline, " + ", ".join(METHODS))
    run.add_argument("--config", action="append", default=[], metavar="NAME=JSON",
                     help="override method settings, e.g. snapkv='{\"budget\": 512}'")
    run.add_argument("--revision", action="append", default=[], metavar="NAME=REV",
                     help="record the source revision each executable was built from")
    run.add_argument("--tasks", help="comma-separated task subset (default: all 21)")
    run.add_argument("--samples-per-task", type=int, help="fixed subset: dataset indices 0..N-1 of each task")
    run.add_argument("--cpus", default="0,2,4,6", help="Linux taskset affinity")
    run.add_argument("--output", type=Path, help="run directory; an existing one is resumed")
    run.add_argument("--check", action="store_true", help="validate inputs without inference")

    summary = commands.add_parser("report", help="rewrite summary.md from records")
    summary.add_argument("output", type=Path)

    args = parser.parse_args()
    for field in ("model", "official", "cases", "data", "output"):
        if getattr(args, field, None) is not None:
            setattr(args, field, getattr(args, field).resolve())
    if args.command == "prepare":
        if args.max_input < 2:
            parser.error("--max-input must be at least 2")
        prepare_cases(args)
    elif args.command == "run":
        if args.output is None and not args.check:
            parser.error("run requires --output unless --check is given")
        evaluate(args)
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
