#!/usr/bin/env python3
"""기존 CausalLM 실행 파일로 Linux baseline/SnapKV를 비교한다.

순서: 사전 검사 → 별도 동작 확인 → 시간 측정 → 메모리 측정 → CSV/요약 저장.
입력 길이마다 baseline/SnapKV를 각각 5회 실행하며 두 측정은 별도 프로세스다.
모델 원본 JSON은 보존한다. Android 실행·온도 제어·품질 채점은 수행하지 않는다.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import tempfile
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# 1. 실행 옵션: 기본은 전체 측정, --check는 검사만, --resume은 중단 기록 재개
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(prog="snapkv_eval.sh", description="Linux CPU: 512/2048/4096 input tokens, "
    "cache 4224, max decode 128, original sampling/EOS, 5 latency + 5 memory trials.")
parser.add_argument("model", nargs="?", help="directory containing the three model JSON files")
parser.add_argument("--check", action="store_true", help="check inputs without running inference")
parser.add_argument("--resume", type=Path, help="continue a matching interrupted evaluation")
args = parser.parse_args()
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# 2. 경로와 실행 환경: 모델·결과는 저장소 안, 두 조건의 CPU/thread 설정 동일
# ---------------------------------------------------------------------------
repo = Path(__file__).resolve().parents[1]
build = repo / "build-linux"
assets = Path(args.model).resolve() if args.model else repo / "models/qwen3-fp32"
binaries = {
    "baseline": Path(os.environ.get("BASELINE_BIN", build / "reference/Applications/CausalLM/nntr_causallm")).resolve(),
    "snapkv": Path(os.environ.get("SNAPKV_BIN", build / "Applications/CausalLM/nntr_causallm")).resolve(),
}
cpus = os.environ.get("EVAL_CPUS", "0,2,4,6")
script = repo / "Applications/utils/mem_usage.sh"
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
out = args.resume or Path(os.environ.get("EVAL_OUTPUT", repo / "logs/snapkv" / stamp))
out = out.resolve()
env = os.environ.copy()
env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", NNTR_NUM_THREADS="4")
blas = build / "validation/deps/root/usr/lib/x86_64-linux-gnu/openblas-pthread"
if blas.is_dir():
    env["LD_LIBRARY_PATH"] = str(blas) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

# ---------------------------------------------------------------------------
# 3. 공통 도구: 파일 hash와 환경 정보 수집; 별도 측정 프로그램은 만들지 않음
# ---------------------------------------------------------------------------
def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def command(argv):
    return subprocess.check_output(argv, env=env, text=True).strip()

# ---------------------------------------------------------------------------
# 4. 사전 검사: 모델 설정, 정확한 입력 token 수, 원본 mem_usage.sh, 빌드 조건
#    문제가 있으면 측정을 시작하지 않는다. --check는 이 구역까지만 실행한다.
# ---------------------------------------------------------------------------
configs = {name: json.loads((assets / name).read_text()) for name in
           ("config.json", "generation_config.json", "nntr_config.json")}
nntr = configs["nntr_config.json"]
for key in ("model_file_name", "tokenizer_file", "embedding_file_name"):
    if key in nntr and not Path(nntr[key]).is_absolute():
        nntr[key] = str((assets / nntr[key]).resolve())
assert nntr["model_tensor_type"] == "FP32-FP32", "Expected the FP32 baseline model"
assert not any(key in nntr for key in ("chat_input", "system_prompt")), "Use the plain benchmark input configuration"
tokenizer = Tokenizer.from_file(nntr["tokenizer_file"])
prompts = {str(n): tokenizer.decode([5555]) * n for n in (512, 2048, 4096)}
for n, prompt in prompts.items():
    assert len(tokenizer.encode(prompt, add_special_tokens=False).ids) == int(n), "Tokenizer length mismatch"
assert sha(script) == "f075e1f0d535128da1d6c2e7dbc5827bf442ff5d3da01be3ca1232074a9d6b62", "mem_usage.sh changed"
command(["taskset", "-c", cpus, "true"])
ldd = {}
options = {}
libraries = {}
for mode, binary in binaries.items():
    assert os.access(binary, os.X_OK), f"Build the {mode} executable first: {binary}"
    ldd[mode] = command(["ldd", str(binary)])
    assert "not found" not in ldd[mode], f"Missing {mode} runtime library"
    libraries[mode] = {str(Path(p)): sha(Path(p)) for p in re.findall(r"=> (\S+)", ldd[mode]) if str(build) in p}
    info = binary.parents[2] / "meson-info/intro-buildoptions.json"
    options[mode] = {x["name"]: x["value"] for x in json.loads(info.read_text())}
for key in ("buildtype", "enable-transformer", "enable-fp16", "enable-blas", "thread-backend",
            "nntr-num-threads", "openblas-num-threads", "enable-opencl", "enable-npu", "enable-htp"):
    assert options["baseline"][key] == options["snapkv"][key], f"Build mismatch: {key}"
assert options["snapkv"]["nntr-num-threads"] == 4
assert options["snapkv"]["openblas-num-threads"] == 1
print(f"Model: {assets}\nBaseline: {binaries['baseline']}\nSnapKV: {binaries['snapkv']}\nOutput: {out}")
print("Temperature/frequency/charging: N/A in WSL; Android cooling is not verified. Quality is not evaluated.")
if args.check:
    print("Preflight passed; no inference performed.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 5. 측정 조건과 기록 준비: 원본 sampling/EOS 유지, 새 기록 생성 또는 재개
#    재개 시 설정·모델·실행 파일·라이브러리 hash가 같아야 기존 값을 재사용한다.
# ---------------------------------------------------------------------------
protocol = dict(context_lengths=[512, 2048, 4096], max_seq_len=4224, max_decode_steps=128,
    snapkv=dict(budget=256, window=16, kernel=5), cpu_affinity=cpus, batch_size=1,
    latency_repetitions=5, memory_repetitions=5, generation_config=configs["generation_config.json"],
    binary_sha256={mode: sha(path) for mode, path in binaries.items()},
    model_weight_sha256=sha(Path(nntr["model_file_name"])), memory_script_sha256=sha(script))
manifest_path = out / "linux-comparison.json"
csv_path = out / "linux-comparison.csv"
rows = []
if args.resume:
    manifest = json.loads(manifest_path.read_text())
    for key, value in protocol.items():
        assert manifest[key] == value, f"Cannot resume: {key} changed"
    for mode in binaries:
        assert sorted(manifest["runtime_library_sha256"][mode].values()) == sorted(libraries[mode].values()), "Runtime libraries changed"
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["valid"] == "True" for row in rows), "Inspect invalid trials before resuming"
else:
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(protocol, started_at=datetime.now().astimezone().isoformat(), prompts=prompts,
        architecture_config=configs["config.json"], environment=dict(os=Path("/etc/os-release").read_text(),
        kernel=command(["uname", "-a"]), cpu=command(["lscpu"]), compiler=command(["c++", "--version"]),
        build_options=options, env={k: env.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NNTR_NUM_THREADS", "LD_LIBRARY_PATH")}),
        source_head_at_eval=command(["git", "-C", str(repo), "rev-parse", "HEAD"]),
        baseline_source_commit="2d1e4974ae84e1e6d37784416bdfb52b647919a6 (expected prebuilt baseline)",
        runtime_library_sha256=libraries, temperature="N/A", frequency="N/A", charging="N/A",
        quality="Not evaluated; agreed dataset, score and threshold are undefined", configs={}, commands={})
(out / "logs").mkdir(exist_ok=True)
manifest.update(status="RUNNING", controller_pid=os.getpid(), completed_trials=len(rows))
manifest.setdefault("invocations", []).append(dict(at=datetime.now().astimezone().isoformat(), repo=str(repo),
    resumed=bool(args.resume), script_sha256=sha(Path(__file__)), launcher_sha256=sha(repo / "eval/snapkv_eval.sh"), ldd=ldd, env=env.get("LD_LIBRARY_PATH")))

# ---------------------------------------------------------------------------
# 6. 실행 상태와 CSV 열: 각 trial 직후 저장하여 중단돼도 완료 기록을 보존
# ---------------------------------------------------------------------------
def save():
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

columns = "phase context mode trial position prefill_tokens generation_tokens early_eos prefill_ms prefill_tps generation_ms generation_tps total_ms e2e_ms os_peak_kib sampled_peak_kib samples target_pid duration_s started_at temperature frequency_unavailable valid failure log".split()
completed = {(r["phase"], int(r["context"]), r["mode"], int(r["trial"])) for r in rows}
current = None
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
save()
# ---------------------------------------------------------------------------
# 7. 추론 실행: 임시 JSON만 변경하고 원본 모델 설정은 그대로 둔다.
# ---------------------------------------------------------------------------
try:
    with tempfile.TemporaryDirectory(prefix="nntr-eval-") as tmp, csv_path.open("a" if args.resume else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not args.resume:
            writer.writeheader()
        model = Path(tmp)
        for name in ("config.json", "generation_config.json"):
            (model / name).write_text(json.dumps(configs[name], indent=2) + "\n")

        def run(phase, context, mode, trial, position):
            """한 번 실행한 뒤 완료·token 수·RSS를 검증하고 원시값을 저장한다."""
            global current
            if (phase, context, mode, trial) in completed:
                return
            # 7-1. 조건 준비: baseline에는 SnapKV 설정 없음, SnapKV에는 budget 256.
            config = dict(nntr, init_seq_len=context, max_seq_len=4224, num_to_generate=128,
                          batch_size=1, sample_input=prompts[str(context)])
            config.pop("snapkv", None)
            if mode == "snapkv":
                config["snapkv"] = protocol["snapkv"]
            (model / "nntr_config.json").write_text(json.dumps(config, indent=2) + "\n")
            manifest["configs"][f"{context}-{mode}"] = dict(configs, **{"nntr_config.json": config})
            # 7-2. 실행 경로 분리. memory는 원본 스크립트가 실제 모델 PID를 관측.
            # latency는 메모리 감시 없이 CausalLM을 직접 실행한다.
            argv = ["taskset", "-c", cpus]
            if phase == "memory":
                argv += ["bash", str(script)]
            argv += [str(binaries[mode]), str(model)]
            name = f"metric-{phase}-{context}-{mode}-{trial}"
            log = out / "logs" / (name + ".log")
            manifest["commands"][name] = argv
            save()
            row = dict.fromkeys(columns, "")
            row.update(phase=phase, context=context, mode=mode, trial=trial, position=position,
                started_at=datetime.now().astimezone().isoformat(), temperature="N/A", log=str(log.relative_to(out)))
            start = time.monotonic()
            print("START", name, row["started_at"], flush=True)
            current = subprocess.Popen(argv, cwd=repo, env=env, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
            stdout, stderr = current.communicate()
            code = current.returncode
            current = None
            log.write_text(stdout + "\n--- STDERR ---\n" + stderr)
            row.update(duration_s=round(time.monotonic() - start, 3), frequency_unavailable="scaling_cur_freq" in stderr)
            # 7-3. 추론 로그 검증. total은 로딩 제외, e2e는 로딩 포함 시간(ms).
            # peak memory는 OS 참고값이며 주 메모리 지표로 대체하지 않는다.
            failures = [] if code == 0 else [f"wrapper exit {code}"]
            for kind in ("prefill", "generation"):
                match = re.search(r"^" + kind + r": (\d+) tokens, (\d+) ms, ([\d.]+) TPS\s*$", stdout, re.M)
                if match:
                    row.update({kind + "_tokens": int(match[1]), kind + "_ms": int(match[2]), kind + "_tps": float(match[3])})
                else:
                    failures.append("missing " + kind)
            for key, pattern in (("total_ms", r"^total: (\d+) ms"), ("e2e_ms", r"^\[e2e time\]: (\d+) ms"), ("os_peak_kib", r"^peak memory: (\d+) KB")):
                match = re.search(pattern, stdout, re.M)
                if match:
                    row[key] = int(match[1])
                else:
                    failures.append("missing " + key)
            if row["prefill_tokens"] != context:
                failures.append("input length mismatch")
            count = row["generation_tokens"]
            if isinstance(count, int) and 0 < count <= 128:
                row["early_eos"] = count < 128
            else:
                failures.append("invalid generation count")
            if re.search(r"FATAL ERROR|truncat|heuristic|falling back", stdout + stderr, re.I):
                failures.append("fatal/truncation/fallback marker")
            # 7-4. 원본 mem_usage.sh의 PID·양수 RSS 표본·sampled peak를 확인.
            # 스크립트 exit code만으로 자식 추론의 성공을 판단하지 않는다.
            if phase == "memory":
                peak = re.search(r"PID - (\d+)\s*:+\s*(\d+) kB", stdout)
                samples = [int(value) for value in re.findall(r"^\d+ (\d+) kB(?: |$)", stdout, re.M)]
                row["samples"] = len(samples)
                if peak:
                    row.update(target_pid=int(peak[1]), sampled_peak_kib=int(peak[2]))
                if not peak or not samples or min(samples) <= 0 or row["sampled_peak_kib"] != max(samples):
                    failures.append("invalid sampled RSS")
                if re.search(r"/smaps.*(?:denied|No such|cannot)", stderr, re.I):
                    failures.append("smaps read error")
            # 7-5. raw 로그/CSV/진행 상태 저장. 실패하면 원인을 남기고 중단한다.
            row.update(valid=not failures, failure="; ".join(failures))
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            manifest["completed_trials"] = len(rows)
            save()
            print(f"DONE {name}: valid={row['valid']}, decode={count}, total={row['total_ms']} ms, RSS={row['sampled_peak_kib']} kB", flush=True)
            if failures:
                raise RuntimeError(row["failure"])
            time.sleep(1)

        # 7-6. 동작 확인은 집계 제외. 각 길이의 latency 5회 후 memory 5회.
        # 반복마다 baseline/SnapKV 선후 순서를 교대하고 trial 사이 1초 대기.
        for position, mode in enumerate(binaries, 1):
            run("function", 512, mode, 0, position)
        for context in protocol["context_lengths"]:
            for phase in ("latency", "memory"):
                for trial in range(1, 6):
                    modes = list(binaries) if trial % 2 else list(reversed(binaries))
                    for position, mode in enumerate(modes, 1):
                        run(phase, context, mode, trial, position)
    manifest.update(status="COMPLETE", finished_at=datetime.now().astimezone().isoformat())
except BaseException as error:
    manifest.update(status="INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
    raise
finally:
    # Ctrl-C/오류 때 남은 자식 프로세스를 정리하고 완료된 기록은 보존한다.
    if current is not None and current.poll() is None:
        os.killpg(current.pid, signal.SIGTERM)
        try:
            current.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(current.pid, signal.SIGKILL)
            current.wait()
    save()

# ---------------------------------------------------------------------------
# 8. 결과 집계: latency는 latency 행만, 주 memory는 memory 행만 사용한다.
#    평균 ± 표본 표준편차와 변화율을 저장한다. EOS/품질 한계를 함께 명시한다.
# ---------------------------------------------------------------------------
report = ["# Linux baseline / SnapKV", "", "Mean ± sample SD. Raw values: linux-comparison.csv; conditions: linux-comparison.json.", "",
          "|Input|Metric|Baseline|SnapKV|Change (%)|n per mode|", "|---|---|---|---|---|---|"]
for context in protocol["context_lengths"]:
    for phase, key in (("latency", "total_ms"), ("latency", "e2e_ms"), ("latency", "prefill_tps"), ("latency", "generation_tps"), ("memory", "sampled_peak_kib")):
        values = [[float(r[key]) for r in rows if r["phase"] == phase and int(r["context"]) == context and r["mode"] == mode] for mode in binaries]
        means = [statistics.mean(v) for v in values]
        change = (means[0] - means[1]) / means[0] * 100
        if key.endswith("tps"):
            change = -change
        cells = [f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}" for v in values]
        report.append(f"|{context}|{key}|{cells[0]}|{cells[1]}|{change:+.2f}|{len(values[0])}/{len(values[1])}|")
report += ["", "Positive change means lower time/RSS or higher TPS. Latency and memory use separate runs.",
           "Actual decode counts must match for fixed-work comparisons; inspect early_eos in CSV.",
           "Temperature, frequency and charging are N/A on WSL. Quality is unevaluated; these numbers do not establish optimization success."]
(out / "summary.md").write_text("\n".join(report) + "\n")
print("Results:", out / "summary.md")
