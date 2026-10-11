#!/usr/bin/env python3
"""Run Android SnapKV with the baseline evaluator's measurement protocol.

Examples:
  ./eval/android_snapkv_eval.sh check
  ./eval/android_snapkv_eval.sh sweep
  ./eval/android_snapkv_eval.sh sweep --budget 256 --window 16 --kernel 5

Requires a SnapKV-enabled Android CausalLM build already deployed to the device.
Default contexts are 512/1024/2048 tokens; cache capacity remains 4224.
The shared evaluator temporarily adds snapkv settings, records raw latency/RSS
and exact configs, and restores the original device config. Outputs are placed
under nntrainer/logs/android_snapkv/. Sampling/EOS remain unchanged.
"""

from android_baseline_eval import cli


if __name__ == "__main__":
    cli("snapkv")
