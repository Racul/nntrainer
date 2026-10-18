#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run LongBench v1, or a declared fixed subset, on native NNTrainer binaries.

Prepare shared cases with --prepare, then run each platform's two conditions.
Official source, data, model and executable paths are mandatory local inputs.
Quality uses greedy decoding once per sample/condition, separately from the
three-repetition latency/memory experiment in pyramidkv_eval.py.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time

import pyramidkv_eval as common

TASKS = (
    "narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa 2wikimqa "
    "musique dureader gov_report qmsum multi_news vcsum trec triviaqa samsum "
    "lsht passage_count passage_retrieval_en passage_retrieval_zh lcc repobench-p"
).split()
PLAIN = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare_cases(args):
    from tokenizers import Tokenizer
    from jinja2 import Environment, StrictUndefined
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    tokenizer_cfg = json.loads((args.model / "tokenizer_config.json").read_text())
    template = Environment(undefined=StrictUndefined).from_string(tokenizer_cfg["chat_template"])
    prompts = json.loads((args.official / "config/dataset2prompt.json").read_text())
    limits = json.loads((args.official / "config/dataset2maxlen.json").read_text())
    args.cases.parent.mkdir(parents=True, exist_ok=True)
    if args.cases.exists():
        raise ValueError("Refusing to replace existing cases")
    counts, truncated, source_hashes = {}, {}, {}
    with args.cases.open("w", encoding="utf-8") as stream:
        for task in TASKS:
            path = args.data / (task + ".jsonl")
            source_hashes[task] = common.sha(path)
            counts[task] = truncated[task] = 0
            for index, line in enumerate(path.open(encoding="utf-8")):
                example = json.loads(line)
                raw = prompts[task].format(**example)
                original = tokenizer.encode(raw, add_special_tokens=False).ids

                def wrap(text):
                    return text if task in PLAIN else template.render(
                        messages=[dict(role="user", content=text)], tools=None,
                        add_generation_prompt=True, enable_thinking=False)

                # Official middle truncation, reserving the model template's
                # overhead, then checking the actual final token array.
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
                            prompt_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                            original_content_tokens=len(original), truncated=was_truncated,
                            max_new_tokens=limits[task], answers=example["answers"],
                            all_classes=example["all_classes"], length=example["length"])
                stream.write(json.dumps(case, ensure_ascii=False) + "\n")
                counts[task] += 1
                truncated[task] += int(was_truncated)
            print(task, counts[task], "prepared", flush=True)
    if len(counts) != 21 or sum(counts.values()) != 4750:
        raise ValueError(f"Expected all 4750 examples in 21 tasks, got {counts}")
    write_json(args.cases.with_suffix(".manifest.json"), dict(
        counts=counts, total=sum(counts.values()), truncated=truncated,
        max_input_tokens=args.max_input, cases_sha256=common.sha(args.cases),
        dataset_sha256=source_hashes, official_sources={str(p.relative_to(args.official)): common.sha(p)
            for p in args.official.rglob("*.py")},
        official_config_sha256={str(p.relative_to(args.official)): common.sha(p)
            for p in (args.official / "config").glob("*.json")},
        tokenizer_sha256=common.sha(args.model / "tokenizer.json"),
        tokenizer_config_sha256=common.sha(args.model / "tokenizer_config.json"),
        chat_policy="Qwen3 official template, enable_thinking=false; official six plain-prompt tasks",
        script_sha256=common.sha(Path(__file__))))


def load_scorer(official):
    sys.path.insert(0, str(official))
    spec = importlib.util.spec_from_file_location("official_longbench_eval", official / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if set(module.dataset2metric) != set(TASKS):
        raise ValueError("Unexpected official LongBench task metrics")
    return module


def run_model(command, env, stdout_path, stderr_path, on_start=None):
    process = None
    start = time.monotonic()
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        try:
            process = subprocess.Popen(command, cwd=common.REPO, env=env,
                                       stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                       **({} if common.WINDOWS else {"start_new_session": True}))
            if on_start is not None:
                on_start(process.pid)
            code = process.wait()
        finally:
            if process is not None and process.poll() is None:
                if common.WINDOWS:
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return code, process.pid, time.monotonic() - start


def extract_prediction(stdout, prompt):
    stdout = stdout.replace("\r\n", "\n")
    prompt = prompt.replace("\r\n", "\n")
    marker = prompt + "\n"
    if marker not in stdout:
        raise ValueError("Cannot find exact echoed prompt in model stdout")
    start = stdout.index(marker) + len(marker)
    end = stdout.rfind("\n\n=================[ LLM with NNTrainer ]")
    if end < start:
        raise ValueError("Cannot find completed model timing trailer")
    return stdout[start:end]


def select_cases(path, counts, samples_per_task):
    """Select fixed dataset indices, independently of any observed scores."""
    if samples_per_task is None:
        return dict(counts), None
    if samples_per_task < 1 or samples_per_task > min(counts.values()):
        raise ValueError("samples-per-task must fit every official task")
    selected = {}
    for line in path.open(encoding="utf-8"):
        case = json.loads(line)
        key = case['task'], case['index']
        if case['task'] in TASKS and 0 <= case['index'] < samples_per_task:
            if key in selected:
                raise ValueError("Duplicate selected case")
            selected[key] = case
    expected = {(task, index) for task in TASKS for index in range(samples_per_task)}
    if set(selected) != expected:
        raise ValueError("Missing fixed-subset cases")
    return {task: samples_per_task for task in TASKS}, selected


def validate_record_ids(records, conditions, case_counts):
    keys = [(r['task'], r['index'], r['condition']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate prediction records")
    if any(task not in case_counts or not 0 <= index < case_counts[task] or mode not in conditions
           for task, index, mode in keys):
        raise ValueError("Prediction records fall outside the selected cases")


def reuse_records(source, out, protocol, selected, scorer):
    """Validate original predictions/logs before importing selected records."""
    if source.resolve() == out.resolve():
        raise ValueError("Reuse source and output must differ")
    old_path, records_path = source / 'manifest.json', source / 'records.jsonl'
    old = json.loads(old_path.read_text(encoding='utf-8'))
    # The new script adds selection/import only. All inference, scoring,
    # model, binary, runtime and environment settings must still match.
    ignored = {'script_sha256', 'sample_selection'}
    previous = {k: v for k, v in old['protocol'].items() if k not in ignored}
    current = {k: v for k, v in protocol.items() if k not in ignored}
    if previous != current:
        different = sorted(k for k in set(previous) | set(current) if previous.get(k) != current.get(k))
        raise ValueError('Cannot reuse different inference/scoring protocol: ' + ', '.join(different))
    rows = [json.loads(line) for line in records_path.open(encoding='utf-8')]
    if len(rows) != old['completed'] or any(not r.get('valid') for r in rows):
        raise ValueError('Reuse source is incomplete or contains invalid records')
    validate_record_ids(rows, protocol['conditions'], old['protocol']['cases_manifest']['counts'])
    proof = dict(source=str(source), manifest_sha256=common.sha(old_path),
                 records_sha256=common.sha(records_path),
                 original_script_sha256=old['protocol']['script_sha256'])
    result = []
    for row in rows:
        key = row['task'], row['index']
        if key not in selected:
            continue
        case = selected[key]
        if any(row[k] != v for k, v in case.items() if k != 'prompt'):
            raise ValueError('Reused record does not match its prepared case')
        relative = Path('logs') / case['task'] / f"{case['index']:04d}-{row['condition']}"
        stdout_path = source / (str(relative) + '.stdout.log')
        stderr_path = source / (str(relative) + '.stderr.log')
        stdout = stdout_path.read_text(encoding='utf-8')
        stderr = stderr_path.read_text(encoding='utf-8')
        metrics, errors = common.parse_metrics(stdout + '\n' + stderr, 0,
                                               case['tokens'], case['max_new_tokens'] - 1)
        errors += common.check_condition(stderr, row['condition'], protocol['pyramidkv'])
        if errors or any(row[k] != v for k, v in metrics.items()):
            raise ValueError('Reused metrics/activation differ from original raw logs')
        pred = extract_prediction(stdout, case['prompt'])
        scored = pred.lstrip('\n').split('\n')[0] if case['task'] in {'trec', 'triviaqa', 'samsum', 'lsht'} else pred
        score = max(scorer.dataset2metric[case['task']](scored, answer,
                    all_classes=case['all_classes']) for answer in case['answers'])
        if pred != row['pred'] or not math.isclose(score, row['quality_score'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Reused prediction/score differs from official rescoring')
        copied = dict(row, reuse_provenance=dict(proof, stdout_sha256=common.sha(stdout_path),
                                               stderr_sha256=common.sha(stderr_path)))
        for src in (stdout_path, stderr_path):
            dst = out / src.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and common.sha(dst) != common.sha(src):
                raise ValueError('Refusing to replace different imported raw logs')
            if not dst.exists():
                shutil.copy2(src, dst)
        result.append(copied)
    return result, dict(proof, imported=len(result))


def save_scores(out, records, scorer, conditions, case_counts, samples_per_task=None):
    scope = (f"SUBSET: fixed first {samples_per_task} sample(s) per task; these are not full-dataset scores."
             if samples_per_task is not None else "Full dataset evaluation.")
    report = ["# LongBench v1", "", scope, "",
              "Run incomplete until every selected task/condition reaches its expected count.", "",
              "|Task|Baseline score|PyramidKV score|Baseline n|PyramidKV n|Expected n|",
              "|---|---:|---:|---:|---:|---:|"]
    result = {}
    for task in TASKS:
        result[task] = {}
        cells = []
        counts = []
        for mode in ("baseline", "pyramidkv"):
            selected = [r for r in records if r.get("valid") and r["task"] == task and r["condition"] == mode]
            score = round(100 * sum(r["quality_score"] for r in selected) / len(selected), 2) if selected else None
            result[task][mode] = dict(score=score, n=len(selected), expected=case_counts[task])
            cells.append(f"{score:.2f}" if score is not None else "pending")
            counts.append(str(len(selected)))
        report.append(f"|{task}|{cells[0]}|{cells[1]}|{counts[0]}|{counts[1]}|{case_counts[task]}|")
    write_json(out / "scores.json", result)
    (out / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def evaluate(args):
    from tokenizers import Tokenizer
    args.prompts = [1]
    args.n_gen = 1
    args.max_seq_len = args.max_input + 513
    args.repeat = 1
    args.interval = 0
    args.sample_interval = 0.05
    # Reuse asset, build-option, DLL and platform validation from the latency
    # helper. Its synthetic prompt is not used for LongBench.
    protocol, envs = common.prepare(args)
    protocol.pop("prompts")
    protocol.pop("context_lengths")
    cases_manifest = json.loads(args.cases.with_suffix(".manifest.json").read_text())
    if cases_manifest["max_input_tokens"] != args.max_input or common.sha(args.cases) != cases_manifest["cases_sha256"]:
        raise ValueError("Prepared cases do not match the chosen context limit/hash")
    if protocol["asset_sha256"]["tokenizer_file"] != cases_manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the prepared prompt tokens")
    case_counts, selected = select_cases(args.cases, cases_manifest['counts'], args.samples_per_task)
    if selected is not None:
        identities = [[task, index, case['prompt_ids_sha256']] for (task, index), case in selected.items()]
        protocol['sample_selection'] = dict(kind='fixed_first_per_task', samples_per_task=args.samples_per_task,
            counts=case_counts, total=len(selected), case_ids=identities,
            identity_sha256=hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest())
    protocol.update(benchmark="LongBench v1", cases_manifest=cases_manifest,
                    cases=str(args.cases), max_input_tokens=args.max_input,
                    quality_generation="greedy, official task max_new_tokens including first prefill output",
                    quality_repetitions=1, model_seed=42,
                    stdout_parser="separate stdout/stderr; exact echoed prompt and completed timing trailer",
                    first_token="init_seq_len=actual_prompt_tokens+1 to register the prefill output",
                    script_sha256=common.sha(Path(__file__)),
                    helper_sha256=common.sha(Path(common.__file__)),
                    metrics_phase="quality with per-example latency; no sampled peak-memory claim")
    if not common.WINDOWS:
        for env in envs.values():
            env["NNTR_LOG_DIR"] = str(args.output)
    else:
        for env in envs.values():
            env["PYTHONIOENCODING"] = "utf-8"
    if not common.WINDOWS:
        libraries = {}
        for mode, binary in protocol['binaries'].items():
            loaded = subprocess.check_output(['ldd', binary], env=envs[mode], text=True)
            if 'not found' in loaded:
                raise RuntimeError('Missing runtime libraries: ' + loaded)
            libraries[mode] = {p: common.sha(Path(p)) for p in re.findall(r'=> (\S+)', loaded) if Path(p).is_file()}
        protocol['runtime_library_sha256'] = libraries
    protocol['scorer_source_sha256'] = {name: common.sha(args.official / name) for name in ('eval.py', 'metrics.py')}
    scorer = load_scorer(args.official)
    if args.check:
        if args.reuse_from:
            with tempfile.TemporaryDirectory(prefix='nntr-longbench-reuse-check-') as tmp:
                reused, _ = reuse_records(args.reuse_from, Path(tmp), protocol, selected, scorer)
            print(f'Validated {len(reused)} reusable predictions against original logs and official scores')
        print(f"LongBench preflight passed: {sum(case_counts.values())} cases per condition; "
              f"scope={'FULL' if selected is None else 'SUBSET'}; no inference performed")
        if selected is not None:
            print('selection_sha256=' + protocol['sample_selection']['identity_sha256'])
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    records_path = out / "records.jsonl"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol"] != protocol:
            raise ValueError("Cannot resume: protocol/assets/binary/runtime/cases changed")
        records = [json.loads(l) for l in records_path.open(encoding="utf-8")] if records_path.exists() else []
        if any(not r.get("valid") for r in records):
            raise ValueError("Inspect invalid records before resuming")
    else:
        if records_path.exists():
            raise ValueError("Records exist without matching manifest")
        manifest = dict(protocol=protocol, started_at=datetime.now().astimezone().isoformat())
        records = []
        if args.reuse_from:
            records, manifest['reuse'] = reuse_records(args.reuse_from, out, protocol, selected, scorer)
            temporary = records_path.with_suffix('.jsonl.tmp')
            with temporary.open('w', encoding='utf-8') as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(records_path)
    validate_record_ids(records, protocol['conditions'], case_counts)
    done = {(r["task"], r["index"], r["condition"]) for r in records}
    manifest.update(status="RUNNING", controller_pid=os.getpid(), completed=len(records),
                    expected=sum(case_counts.values()) * len(protocol["conditions"]),
                    evaluation_scope='FULL_DATASET' if selected is None else 'FIXED_SUBSET',
                    full_dataset_complete=False)
    write_json(manifest_path, manifest)
    save_scores(out, records, scorer, protocol['conditions'], case_counts, args.samples_per_task)
    tokenizer = Tokenizer.from_file(protocol["configs"]["nntr_config.json"]["tokenizer_file"])
    try:
        with tempfile.TemporaryDirectory(prefix="nntr-pyramidkv-longbench-") as tmp:
            model = Path(tmp)
            (model / "config.json").write_text(json.dumps(protocol["configs"]["config.json"]), encoding="utf-8")
            for line in args.cases.open(encoding="utf-8"):
                case = json.loads(line)
                if selected is not None and (case['task'], case['index']) not in selected:
                    continue
                modes = protocol["conditions"] if case["index"] % 2 == 0 else list(reversed(protocol["conditions"]))
                for mode in modes:
                    key = case["task"], case["index"], mode
                    if key in done:
                        continue
                    ids = tokenizer.encode(case["prompt"], add_special_tokens=False).ids
                    if len(ids) != case["tokens"] or hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != case["prompt_ids_sha256"]:
                        raise ValueError("LongBench prompt tokens changed")
                    generation = dict(protocol["configs"]["generation_config.json"], do_sample=False,
                                      top_k=1, top_p=1.0, temperature=1.0, seed=42)
                    if case["task"] == "samsum":
                        eos = generation["eos_token_id"]
                        eos = eos if isinstance(eos, list) else [eos]
                        generation["eos_token_id"] = list(dict.fromkeys(eos + [tokenizer.encode("\n", add_special_tokens=False).ids[-1]]))
                    config = dict(protocol["configs"]["nntr_config.json"], sample_input=case["prompt"],
                                  init_seq_len=case["tokens"] + 1,
                                  max_seq_len=args.max_input + 513,
                                  num_to_generate=case["max_new_tokens"] - 1, batch_size=1)
                    for field in ("h2o", "snapkv", "pyramidkv", "chat_input", "system_prompt"):
                        config.pop(field, None)
                    if mode == "pyramidkv":
                        config[mode] = protocol[mode]
                    (model / "nntr_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
                    (model / "generation_config.json").write_text(json.dumps(generation), encoding="utf-8")
                    logs = out / "logs" / case["task"]
                    logs.mkdir(parents=True, exist_ok=True)
                    stem = f"{case['index']:04d}-{mode}"
                    stdout_path, stderr_path = logs / (stem + ".stdout.log"), logs / (stem + ".stderr.log")
                    if stdout_path.exists() or stderr_path.exists():
                        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                        for partial in (stdout_path, stderr_path):
                            if partial.exists():
                                partial.rename(partial.with_name(partial.name + '.interrupted-' + stamp))
                    command = [protocol["binaries"][mode], str(model)]
                    if not common.WINDOWS:
                        command = ["taskset", "-c", args.cpus] + command
                    active = dict(task=case["task"], index=case["index"], condition=mode,
                                  started_at=datetime.now().astimezone().isoformat(), command=command,
                                  config=config, generation_config=generation,
                                  stdout=str(stdout_path.relative_to(out)), stderr=str(stderr_path.relative_to(out)))
                    manifest["active"] = active
                    write_json(manifest_path, manifest)
                    print("START", stem, case["task"], "tokens", case["tokens"], flush=True)
                    def started(pid):
                        manifest['active']['pid'] = pid
                        write_json(manifest_path, manifest)
                    code, pid, duration = run_model(command, envs[mode], stdout_path, stderr_path, started)
                    stdout = stdout_path.read_text(encoding="utf-8", errors="strict")
                    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
                    metrics, errors = common.parse_metrics(stdout + "\n" + stderr, code, case["tokens"], case["max_new_tokens"] - 1)
                    errors += common.check_condition(stderr, mode, protocol["pyramidkv"])
                    pred = ""
                    try:
                        pred = extract_prediction(stdout, case["prompt"])
                    except ValueError as error:
                        errors.append(str(error))
                    quality_score = None
                    if not errors:
                        scored_pred = pred.lstrip('\n').split('\n')[0] if case['task'] in {'trec', 'triviaqa', 'samsum', 'lsht'} else pred
                        quality_score = max(scorer.dataset2metric[case['task']](scored_pred, answer, all_classes=case['all_classes']) for answer in case['answers'])
                    record = dict(case, condition=mode, pred=pred, quality_score=quality_score, valid=not errors,
                                  failure="; ".join(errors), pid=pid, duration_s=duration,
                                  command=command, started_at=active["started_at"], **metrics)
                    record.pop("prompt")
                    with records_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    records.append(record)
                    done.add(key)
                    manifest.pop("active", None)
                    manifest["completed"] = len(records)
                    write_json(manifest_path, manifest)
                    save_scores(out, records, scorer, protocol["conditions"], case_counts, args.samples_per_task)
                    print("DONE", case["task"], stem, f"valid={not errors}", f"{len(records)}/{manifest['expected']}", flush=True)
                    if errors:
                        raise RuntimeError("Invalid LongBench run: " + "; ".join(errors))
        if len(records) != manifest["expected"]:
            raise RuntimeError("Incomplete selected-case evaluation")
        for task in TASKS:
            for mode in protocol['conditions']:
                selected = [r for r in records if r['task'] == task and r['condition'] == mode]
                if len(selected) != case_counts[task]:
                    raise RuntimeError('Incomplete task/condition')
                official_score = scorer.scorer(task, [r['pred'] for r in selected], [r['answers'] for r in selected], selected[0]['all_classes'])
                mean_score = round(100 * sum(r['quality_score'] for r in selected) / len(selected), 2)
                if official_score != mean_score:
                    raise RuntimeError('Incremental score differs from official scorer')
        manifest["status"] = "COMPLETE"
        manifest['full_dataset_complete'] = args.samples_per_task is None
    except BaseException as error:
        manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        manifest["updated_at"] = datetime.now().astimezone().isoformat()
        write_json(manifest_path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--max-input", type=int, default=8192)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--samples-per-task", type=int, help="evaluate a declared subset: fixed first N cases per task")
    parser.add_argument("--reuse-from", type=Path, help="import matching valid selected predictions and raw logs from an earlier run")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-bin", type=Path)
    parser.add_argument("--pyramidkv-bin", type=Path)
    parser.add_argument("--condition", choices=("baseline", "pyramidkv", "both"), default="both")
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--beta", type=int, default=20)
    parser.add_argument("--cpus", default="0,2,4,6")
    args = parser.parse_args()
    for field in ("model", "official", "data", "cases", "output", "baseline_bin", "pyramidkv_bin", "reuse_from"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if args.max_input < 2 or not 0 < args.window < args.budget or args.beta < 1 or args.kernel < 1 or not args.kernel % 2:
        parser.error("Invalid input/budget/window/kernel/beta")
    if args.samples_per_task is not None and args.samples_per_task < 1:
        parser.error('samples-per-task must be positive')
    if args.reuse_from and (args.samples_per_task is None or args.prepare):
        parser.error('reuse-from requires evaluation with samples-per-task')
    if args.prepare:
        if args.data is None:
            parser.error("--prepare requires --data")
        prepare_cases(args)
    else:
        if args.output is None or args.baseline_bin is None or args.pyramidkv_bin is None:
            parser.error("Evaluation requires --output and both binary paths")
        evaluate(args)


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, terminate)
    main()
