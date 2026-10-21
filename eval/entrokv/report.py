#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit complete three-repetition performance runs and render Korean tables."""
import argparse
import json
from pathlib import Path
import statistics


def platform_report(root, platform):
    path = root / ("performance-" + platform) / (platform + "-comparison.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    if data["status"] != "COMPLETE" or protocol["repetitions"] != 3:
        raise ValueError(f"{platform}: complete three-repetition run required")
    rows = [r for r in data["trials"] if r["phase"] != "function"]
    expected = {(phase, n, mode, trial) for phase in ("latency", "memory")
                for n in protocol["context_lengths"] for mode in ("baseline", "entrokv")
                for trial in (1, 2, 3)}
    actual = [(r["phase"], r["context"], r["mode"], r["trial"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected or any(not r["valid"] for r in data["trials"]):
        raise ValueError(f"{platform}: missing, duplicate or invalid trials")
    key = "sampled_peak_working_set_kib" if platform == "windows" else "sampled_peak_rss_kib"
    title = "Windows native" if platform == "windows" else "Linux (WSL2)"
    report = [f"### {title}", "", "Baseline 3회·EntroKV 3회 평균; latency와 memory는 별도 프로세스.", "",
              "| 입력 tokens | Baseline latency (s) | EntroKV latency (s) | Latency 변화 | Baseline peak memory (MiB) | EntroKV peak memory (MiB) | Memory 변화 |",
              "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    values = []
    for n in protocol["context_lengths"]:
        result = dict(input_tokens=n)
        for mode in ("baseline", "entrokv"):
            for phase, metric, scale in (("latency", "total_ms", 1000), ("memory", key, 1024)):
                selected = [r for r in rows if r["context"] == n and r["mode"] == mode and r["phase"] == phase]
                sample = [r[metric] / scale for r in selected]
                result[mode + "_" + phase] = dict(mean=statistics.mean(sample), sd=statistics.stdev(sample), n=len(sample),
                    measured=sample, actual_decode_tokens=[r["generation_tokens"] for r in selected],
                    samples=[r.get("samples") for r in selected] if phase == "memory" else None)
        b, m = result["baseline_latency"]["mean"], result["entrokv_latency"]["mean"]
        bm, mm = result["baseline_memory"]["mean"], result["entrokv_memory"]["mean"]
        result.update(latency_change_pct=100 * (m / b - 1), memory_change_pct=100 * (mm / bm - 1))
        values.append(result)
        report.append(f"| {n} | {b:.3f} | {m:.3f} | {result['latency_change_pct']:+.1f}% | {bm:,.1f} | {mm:,.1f} | {result['memory_change_pct']:+.1f}% |")
    overall = {}
    for phase in ("latency", "memory"):
        baseline = statistics.mean(r["baseline_" + phase]["mean"] for r in values)
        method = statistics.mean(r["entrokv_" + phase]["mean"] for r in values)
        overall[phase] = dict(baseline=baseline, entrokv=method, change_pct=100 * (method / baseline - 1))
    report += ["", f"입력별 평균에 같은 가중치를 준 전체 변화: latency {overall['latency']['change_pct']:+.1f}%, peak memory {overall['memory']['change_pct']:+.1f}%.",
               "", f"메모리: {'PSAPI current Working Set, 50 ms 표본 최대' if platform == 'windows' else '원본 mem_usage.sh의 모델 PID smaps RSS 표본 최대'}. OS peak와 구분합니다."]
    decode = {n: {mode: sorted({r["generation_tokens"] for r in rows if r["context"] == n and r["mode"] == mode})
                  for mode in ("baseline", "entrokv")} for n in protocol["context_lengths"]}
    report += ["", "실제 decode 수: `" + json.dumps(decode) + "`. EOS로 작업량이 다르면 동일 작업량의 speedup으로 해석하지 않습니다."]
    return "\n".join(report), dict(rows=values, overall=overall, actual_decode=decode, raw=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    reports, result = [], {}
    for platform in ("linux", "windows"):
        report, result[platform] = platform_report(args.results, platform)
        reports.append(report)
    header = "# EntroKV 대 main baseline\n\nQwen3-0.6B, FP32 weights/activations, FP16-bit UINT16 KV, batch 1, capacity 4225, 최대 decode 128. beta 0.3/alpha 1/window 32/kernel 7, GQA mean entropy; Eq.10 normalization off.\n\n"
    footer = "\n\nKV allocation 크기는 유지하며, head별 prefill 캐시만 압축하고 최대 quota까지 padding한 뒤 decode 토큰을 추가합니다. EntroKV는 32-query 블록의 host attention을 사용합니다. 사용자 요청에 따라 LongBench는 실행하지 않았으며 품질 점수는 측정하지 않았습니다.\n"
    (args.results / "performance-summary.md").write_text(header + "\n\n".join(reports) + footer, encoding="utf-8")
    (args.results / "performance-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.results / "performance-summary.md")


if __name__ == "__main__":
    main()
