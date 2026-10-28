// SPDX-License-Identifier: Apache-2.0
/**
 * @file manifold_kv.cpp
 * @brief Qwen3 inference with one-shot ManifoldKV prompt compression.
 */
#include <api/streamer.h>
#include <causal_lm.h>
#include <footprint_sampler.h>
#include <mha_core.h>

#include <chrono>

namespace causallm {

void CausalLM::runManifoldKV(const WSTR &prompt, bool do_sample,
                             const WSTR &system_prompt, const WSTR &tail_prompt,
                             bool log_output) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  struct EndGuard {
    BaseStreamer *streamer;
    ~EndGuard() { streamer_end(streamer); }
  } end_guard{streamer_};
  if (global_token_len || USE_KVCACHE)
    throw std::invalid_argument(
      "ManifoldKV needs a fresh prompt; call setSessionTokenPosition(0) before "
      "another run");
  has_run_ = false;
  prepareStopRequestForRun();
  beginSamplingRun();
  output_list.assign(1, "");
  pending_ids_.clear();
  ensureTokenizer();
  const auto tokens =
    tokenizer->Encode(system_prompt + prompt + tail_prompt, true);
  const auto position_limit = std::min(MAX_SEQ_LEN, MAX_POSITION_EMBEDDINGS);
  if (INIT_SEQ_LEN <= 0 || tokens.empty() ||
      tokens.size() > static_cast<unsigned int>(INIT_SEQ_LEN) ||
      tokens.size() >= position_limit)
    throw std::invalid_argument(
      "ManifoldKV requires a nonempty prompt fitting init_seq_len and leaving "
      "a generation position");
  manifold_prompt_length = static_cast<unsigned int>(tokens.size());
  const auto budget =
    std::min(position_limit - manifold_prompt_length,
             NUM_TO_GENERATE > 0 ? static_cast<unsigned int>(NUM_TO_GENERATE)
                                 : position_limit - manifold_prompt_length);
  allocateAndBindKVCache();
  unsigned int attention_layers = 0;
  model->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      if (layer.getType() == MHACoreLayer::type) {
        layer.setProperty({"manifold_kv_prompt_length=" +
                           std::to_string(manifold_prompt_length)});
        ++attention_layers;
      }
    },
    nullptr);
  if (attention_layers != static_cast<unsigned int>(NUM_LAYERS))
    throw std::invalid_argument("ManifoldKV needs an MHA cache in every layer");

  std::vector<float> input_sample(MAX_SEQ_LEN, 0.0f);
  for (size_t i = 0; i < tokens.size(); ++i)
    input_sample[i] = static_cast<float>(tokens[i]);
  // Match the model's lexicographic placeholder order, including layer 10+.
  std::vector<std::pair<std::string, float *>> caches;
  for (int i = 0; i < NUM_LAYERS; ++i) {
    caches.emplace_back(
      "cache_k_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
    caches.emplace_back(
      "cache_v_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
  }
  std::sort(caches.begin(), caches.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<float *> input{input_sample.data()};
  for (const auto &entry : caches)
    input.push_back(entry.second);
  auto release = [this](float *p) { releaseLogitsBuf(p); };
  using Logits = std::unique_ptr<float, decltype(release)>;
  auto forward = [&](unsigned int rows) {
    const auto result = incrementalInference(1, input, rows, global_token_len,
                                             global_token_len + rows);
    global_token_len += rows;
    return Logits(result.at(0), release);
  };
  Logits logits(nullptr, release);
  unsigned int prefilled = 0, generated = 0;
  std::vector<unsigned int> history;
  if (REPETITION_PENALTY != 1.0f && REPETITION_WINDOW) {
    const auto first =
      tokens.size() > REPETITION_WINDOW ? tokens.size() - REPETITION_WINDOW : 0;
    for (size_t i = first; i < tokens.size(); ++i)
      history.push_back(static_cast<unsigned int>(tokens[i]));
  }
  std::vector<bool> eos(1, false);
  std::vector<double> decode_times;
  auto elapsed_ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto prefill_end = start;
  double ttft_ms = 0;
  try {
    if (!stop_requested_.load(std::memory_order_acquire)) {
      logits.reset(forward(manifold_prompt_length).release());
      prefilled = manifold_prompt_length;
    }
    prefill_end = Clock::now();
    while (prefilled && generated < budget &&
           !stop_requested_.load(std::memory_order_acquire)) {
      const auto step_start = Clock::now();
      const auto ids = generate(logits.get(), do_sample,
                                history.empty() ? 1.0f : REPETITION_PENALTY,
                                history.empty() ? nullptr : history.data(),
                                static_cast<unsigned int>(history.size()));
      registerOutputs(tokenizer, ids, generated, eos, log_output);
      ++generated;
      if (generated == 1)
        ttft_ms = elapsed_ms(start, Clock::now());
      if (REPETITION_PENALTY != 1.0f && REPETITION_WINDOW) {
        if (history.size() == REPETITION_WINDOW)
          history.erase(history.begin());
        history.push_back(ids[0]);
      }
      input_sample[0] = static_cast<float>(ids[0]);
      // Commit the final/EOS token too, keeping cache and position consistent.
      logits.reset(forward(1).release());
      decode_times.push_back(elapsed_ms(step_start, Clock::now()));
      if (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(), ids[0]) !=
          EOS_TOKEN_ID.end())
        break;
    }
  } catch (...) {
    setSessionTokenPosition(0);
    throw;
  }
  const auto finish = Clock::now();
  const auto initialization_ms = performance_metrics.initialization_duration_ms;
  performance_metrics = {};
  performance_metrics.initialization_duration_ms = initialization_ms;
  performance_metrics.prefill_tokens = prefilled;
  performance_metrics.generation_tokens = generated;
  performance_metrics.prefill_duration_ms = elapsed_ms(start, prefill_end);
  performance_metrics.generation_duration_ms = elapsed_ms(prefill_end, finish);
  performance_metrics.total_duration_ms = elapsed_ms(start, finish);
  const size_t maxrss = getPeakMemoryKb();
  performance_metrics.peak_rss_kb = maxrss;
  performance_metrics.peak_memory_kb =
    resolvePeakMemoryKb(FootprintSampler::get().finish(), maxrss);
  performance_metrics.ttft_ms = ttft_ms;
  if (!decode_times.empty())
    performance_metrics.first_token_ms = decode_times.front();
  if (decode_times.size() > 1) {
    std::sort(decode_times.begin() + 1, decode_times.end());
    const size_t n = decode_times.size() - 1;
    performance_metrics.decode_steady_ms =
      n % 2 ? decode_times[1 + n / 2]
            : (decode_times[n / 2] + decode_times[1 + n / 2]) / 2;
    performance_metrics.decode_steady_tokens = static_cast<unsigned int>(n);
  }
  if (log_output) {
    const auto tps = [](unsigned int count, double ms) {
      return ms > 0 ? count * 1000.0 / ms : 0.0;
    };
    std::cout << "\n\nManifoldKV: prompt_budget=" << manifold_kv.budget
              << ", window_size=" << manifold_kv.window_size
              << ", retained_prompt_tokens="
              << std::min(prefilled, manifold_kv.budget)
              << ", allocated_cache_rows=" << getKVCacheRows(0)
              << "\nprefill: " << prefilled << " tokens, "
              << performance_metrics.prefill_duration_ms << " ms, "
              << tps(prefilled, performance_metrics.prefill_duration_ms)
              << " TPS\n"
              << "generation: " << generated << " tokens, "
              << performance_metrics.generation_duration_ms << " ms, "
              << tps(generated, performance_metrics.generation_duration_ms)
              << " TPS\n"
              << "total: " << performance_metrics.total_duration_ms << " ms\n"
              << "peak memory: " << performance_metrics.peak_memory_kb
              << " KB\n";
  }
  has_run_ = prefilled != 0;
}

} // namespace causallm
