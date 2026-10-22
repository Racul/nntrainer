// SPDX-License-Identifier: Apache-2.0
/**
 * @file streaming_llm.cpp
 * @brief Bounded-cache Qwen3 streaming inference driver.
 */
#include <api/streamer.h>
#include <causal_lm.h>
#include <footprint_sampler.h>
#include <mha_core.h>

#include <chrono>
#include <climits>
#include <iostream>

namespace causallm {

void CausalLM::runStreamingLLM(const WSTR &prompt, bool do_sample,
                               const WSTR &system_prompt,
                               const WSTR &tail_prompt, bool log_output) {
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  struct EndGuard {
    BaseStreamer *streamer;
    ~EndGuard() { streamer_end(streamer); }
  } end_guard{streamer_};
  if (INIT_SEQ_LEN <= 0 || MAX_SEQ_LEN < INIT_SEQ_LEN || USE_KVCACHE)
    throw std::invalid_argument(
      "Invalid StreamingLLM input plane or saved-cache resume");

  has_run_ = false;
  prepareStopRequestForRun();
  beginSamplingRun();
  output_list.assign(1, "");
  pending_ids_.clear();
  ensureTokenizer();
  const auto tokens =
    tokenizer->Encode(system_prompt + prompt + tail_prompt, true);
  if (tokens.empty())
    throw std::invalid_argument("StreamingLLM requires a non-empty prompt");
  const unsigned int budget = NUM_TO_GENERATE > 0
                                ? static_cast<unsigned int>(NUM_TO_GENERATE)
                                : static_cast<unsigned int>(MAX_SEQ_LEN);
  // The absolute counter only orders cache updates; positions used for RoPE
  // stay inside capacity. Check arithmetic before touching any layer state.
  if (tokens.size() > UINT_MAX - global_token_len ||
      budget > UINT_MAX - global_token_len - tokens.size())
    throw std::invalid_argument(
      "StreamingLLM absolute token counter exhausted");

  allocateAndBindKVCache();
  unsigned int attention_layers = 0;
  model->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      if (layer.getType() == MHACoreLayer::type) {
        layer.setProperty({"streaming_llm_sink_tokens=" +
                             std::to_string(streaming_llm.sink_tokens),
                           "streaming_llm_recent_tokens=" +
                             std::to_string(streaming_llm.recent_tokens)});
        ++attention_layers;
      }
    },
    nullptr);
  if (attention_layers != static_cast<unsigned int>(NUM_LAYERS))
    throw std::invalid_argument(
      "StreamingLLM needs an MHA cache in every layer");

  if (log_output)
    std::cout << "StreamingLLM: sink_tokens=" << streaming_llm.sink_tokens
              << ", recent_tokens=" << streaming_llm.recent_tokens
              << ", cache_tokens=" << streaming_llm.capacity() << '\n';

  std::vector<float> input_sample(static_cast<size_t>(MAX_SEQ_LEN), 0.0f);
  std::vector<float *> input{input_sample.data()};
  for (int i = 0; i < NUM_LAYERS; ++i) {
    input.push_back(
      reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
    input.push_back(
      reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
  }
  // Always return pooled logits, including when sampling or callbacks throw.
  auto release = [this](float *p) { releaseLogitsBuf(p); };
  using Logits = std::unique_ptr<float, decltype(release)>;
  auto forward = [&](unsigned int rows) {
    auto result = incrementalInference(1, input, rows, global_token_len,
                                       global_token_len + rows);
    global_token_len += rows;
    return Logits(result.at(0), release);
  };
  Logits logits(nullptr, release);
  unsigned int prefilled = 0;
  unsigned int generated = 0;
  std::vector<unsigned int> recent_ids;
  std::vector<bool> eos(1, false);
  std::vector<double> decode_times;
  auto elapsed_ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto prefill_end = start;
  double ttft_ms = 0;
  try {
    while (prefilled < tokens.size() &&
           !stop_requested_.load(std::memory_order_acquire)) {
      const auto rows = static_cast<unsigned int>(
        std::min<size_t>(INIT_SEQ_LEN, tokens.size() - prefilled));
      for (unsigned int i = 0; i < rows; ++i)
        input_sample[i] = static_cast<float>(tokens[prefilled + i]);
      logits.reset(forward(rows).release());
      prefilled += rows;
    }
    prefill_end = Clock::now();
    if (prefilled == tokens.size()) {
      for (; generated < budget &&
             !stop_requested_.load(std::memory_order_acquire);) {
        const auto step_start = Clock::now();
        auto ids = generate(logits.get(), do_sample,
                            recent_ids.empty() ? 1.0f : REPETITION_PENALTY,
                            recent_ids.empty() ? nullptr : recent_ids.data(),
                            static_cast<unsigned int>(recent_ids.size()));
        registerOutputs(tokenizer, ids, generated % MAX_SEQ_LEN, eos,
                        log_output);
        ++generated;
        if (generated == 1)
          ttft_ms = elapsed_ms(start, Clock::now());
        if (REPETITION_PENALTY != 1.0f && REPETITION_WINDOW) {
          if (recent_ids.size() == REPETITION_WINDOW)
            recent_ids.erase(recent_ids.begin());
          recent_ids.push_back(ids[0]);
        }
        // Commit even the final/EOS token. The next run can append another
        // turn to the same stream without losing its last generated token.
        input_sample[0] = static_cast<float>(ids[0]);
        logits.reset(forward(1).release());
        decode_times.push_back(elapsed_ms(step_start, Clock::now()));
        if (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(), ids[0]) !=
            EOS_TOKEN_ID.end())
          break;
      }
    }
  } catch (...) {
    // A partially forwarded graph cannot be resumed coherently. Starting
    // again at zero overwrites the raw-cache prefix in every layer.
    setSessionTokenPosition(0);
    throw;
  }
  const auto finish = Clock::now();
  const size_t maxrss = getPeakMemoryKb();
  const size_t honest = FootprintSampler::get().finish();
  const auto initialization_ms = performance_metrics.initialization_duration_ms;
  performance_metrics = {};
  performance_metrics.initialization_duration_ms = initialization_ms;
  performance_metrics.prefill_tokens = prefilled;
  performance_metrics.generation_tokens = generated;
  performance_metrics.prefill_duration_ms = elapsed_ms(start, prefill_end);
  performance_metrics.generation_duration_ms = elapsed_ms(prefill_end, finish);
  performance_metrics.total_duration_ms = elapsed_ms(start, finish);
  performance_metrics.peak_rss_kb = maxrss;
  performance_metrics.peak_memory_kb = resolvePeakMemoryKb(honest, maxrss);
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
    std::cout << "\n\nprefill: " << prefilled << " tokens, "
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
  has_run_ = true;
}

} // namespace causallm
