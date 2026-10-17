// SPDX-License-Identifier: Apache-2.0
/**
 * @file h2o.h
 * @date 11 October 2026
 * @brief Layer-local heavy-hitter and recent-token KV eviction
 */

#ifndef __CAUSALLM_H2O_H__
#define __CAUSALLM_H2O_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <tensor.h>

namespace causallm {

/** @brief Total retained KV budget, including the recent-token quota. */
struct H2OConfig {
  unsigned int budget = 0;
  unsigned int recent = 128;

  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("H2O expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("H2O integer exceeds unsigned int");
    return static_cast<unsigned int>(number);
  }

  void validate() const {
    if (budget && (!recent || recent >= budget))
      throw std::invalid_argument("H2O requires 0 < recent < budget");
  }
};

/**
 * @brief Accumulate causal softmax attention, keep heavy hitters plus recent
 * KV.
 * @details Follows H2OKVCache_LayerWise in the official real-drop
 * implementation:
 * https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_real_drop/modify_llama.py
 * Full prefill seeds scores; every decode adds scores and evicts again. GQA
 * query-head scores are averaged per shared KV head, a variant of the original
 * independent query-head selection. K/V and scores move together in temporal
 * order. Absolute RoPE positions, allocation sizes and cache shapes stay
 * intact.
 */
class H2OState {
public:
  bool enabled() const { return config_.budget != 0; }

  /** @brief Pass prompt_length last to reset the session and accumulated
   * scores.
   */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key != "h2o_budget" && key != "h2o_recent" &&
        key != "h2o_prompt_length")
      return false;
    const auto number = H2OConfig::parseUnsigned(value);
    if (key == "h2o_budget")
      config_.budget = number;
    else if (key == "h2o_recent")
      config_.recent = number;
    else {
      config_.validate();
      if (enabled() && !number)
        throw std::invalid_argument("H2O requires a nonempty prompt");
      prompt_length_ = number;
      cached_tokens_ = next_logical_position_ = 0;
      scores_.clear();
    }
    return true;
  }

  /** @brief Validate logical/RoPE and physical bounds before writing any KV. */
  void checkStep(unsigned int logical_from, unsigned int step_size,
                 unsigned int capacity, unsigned int max_position) const {
    config_.validate();
    if (!enabled() || !prompt_length_ ||
        logical_from != next_logical_position_ ||
        step_size != (next_logical_position_ ? 1u : prompt_length_))
      throw std::invalid_argument("H2O requires one full prefill followed by "
                                  "consecutive single tokens");
    const auto limit = std::min(capacity, max_position);
    if (logical_from > limit || step_size > limit - logical_from ||
        cached_tokens_ > capacity || step_size > capacity - cached_tokens_)
      throw std::out_of_range("H2O step exceeds sequence or RoPE capacity");
  }

  /**
   * @brief Use existing RoPE and attention kernels over a compact cache view.
   * @details Explicit probabilities are needed for online scoring. The
   * callbacks keep model/kernel dependencies outside this policy and run only
   * when enabled.
   */
  template <typename Store, typename Attend>
  void forwardStep(nntrainer::Tensor &query, nntrainer::Tensor &key,
                   nntrainer::Tensor &value, unsigned int logical_from,
                   unsigned int step_size, unsigned int max_position,
                   unsigned int heads_q, unsigned int heads_kv,
                   unsigned int head_dim, Store store, Attend attend) {
    checkStep(logical_from, step_size, key.height(), max_position);
    validateCache(key, value, heads_q, heads_kv, head_dim);
    const unsigned int length = cached_tokens_ + step_size;
    const size_t rows = static_cast<size_t>(step_size) * cached_tokens_ +
                        static_cast<size_t>(step_size) * (step_size + 1ull) / 2;
    if (rows > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("H2O probability tensor exceeds dimension range");
    auto step_dim = key.getDim();
    step_dim.height(step_size);
    auto key_step = key.getSharedDataTensor(
      step_dim, static_cast<size_t>(cached_tokens_) * key.width(), true);
    auto value_step = value.getSharedDataTensor(
      step_dim, static_cast<size_t>(cached_tokens_) * value.width(), true);
    store(key_step, value_step);
    auto cached_dim = key.getDim();
    cached_dim.height(length);
    auto cached_key = key.getSharedDataTensor(cached_dim, 0, true);
    auto cached_value = value.getSharedDataTensor(cached_dim, 0, true);
    nntrainer::Tensor probabilities(1, 1, static_cast<unsigned int>(rows),
                                    heads_q, query.getTensorType());
    attend(cached_key, cached_value, probabilities, cached_tokens_);
    finishStep(probabilities, key, value, logical_from, step_size, heads_q,
               heads_kv, head_dim);
  }

  /** @brief Update scores and compact only after attention has consumed V. */
  void finishStep(const nntrainer::Tensor &probabilities,
                  nntrainer::Tensor &key, nntrainer::Tensor &value,
                  unsigned int logical_from, unsigned int step_size,
                  unsigned int heads_q, unsigned int heads_kv,
                  unsigned int head_dim) {
    checkStep(logical_from, step_size, key.height(), key.height());
    validateCache(key, value, heads_q, heads_kv, head_dim);
    const unsigned int length = cached_tokens_ + step_size;
    const size_t rows = static_cast<size_t>(step_size) * cached_tokens_ +
                        static_cast<size_t>(step_size) * (step_size + 1ull) / 2;
    if (probabilities.batch() != 1 || probabilities.channel() != 1 ||
        probabilities.width() != heads_q || probabilities.height() != rows)
      throw std::invalid_argument("H2O causal probability shape mismatch");
    if (scores_.empty())
      scores_.resize(heads_kv);
    if (scores_.size() != heads_kv)
      throw std::invalid_argument("H2O head count changed during inference");
    const unsigned int group = heads_q / heads_kv;
    const size_t element_size = key.getDataType() == nntrainer::Tdatatype::FP32
                                  ? sizeof(float)
                                  : sizeof(uint16_t);
    std::vector<unsigned int> indices, positions;
    if (length > config_.budget) {
      indices.resize(length - config_.recent);
      positions.resize(config_.budget);
    }
    for (unsigned int head = 0; head < heads_kv; ++head) {
      auto &scores = scores_[head];
      scores.resize(length, 0.0f);
      size_t row = 0;
      for (size_t q = cached_tokens_; q < length; ++q) {
        for (size_t j = 0; j <= q; ++j) {
          float vote = 0.0f;
          for (unsigned int g = 0; g < group; ++g)
            vote += probability(probabilities,
                                (row + j) * heads_q + head * group + g);
          scores[j] += vote / group;
        }
        row += q + 1;
      }
      for (const auto score : scores)
        if (!std::isfinite(score) || score < 0.0f)
          throw std::invalid_argument("H2O received invalid attention scores");
      if (length <= config_.budget)
        continue;
      const unsigned int heavy = config_.budget - config_.recent;
      std::iota(indices.begin(), indices.end(), 0u);
      std::partial_sort(indices.begin(), indices.begin() + heavy, indices.end(),
                        [&](unsigned int a, unsigned int b) {
                          return scores[a] == scores[b] ? a < b
                                                        : scores[a] > scores[b];
                        });
      std::sort(indices.begin(), indices.begin() + heavy);
      std::copy_n(indices.begin(), heavy, positions.begin());
      std::iota(positions.begin() + heavy, positions.end(),
                length - config_.recent);
      compactHead(key, positions, head, head_dim, element_size);
      compactHead(value, positions, head, head_dim, element_size);
      for (size_t dst = 0; dst < positions.size(); ++dst)
        scores[dst] = scores[positions[dst]];
      scores.resize(config_.budget);
    }
    cached_tokens_ = std::min(length, config_.budget);
    next_logical_position_ = logical_from + step_size;
  }

private:
  static void validateCache(const nntrainer::Tensor &key,
                            const nntrainer::Tensor &value,
                            unsigned int heads_q, unsigned int heads_kv,
                            unsigned int head_dim) {
    using DType = nntrainer::Tdatatype;
    if (!heads_q || !heads_kv || heads_q % heads_kv || !head_dim ||
        key.batch() != 1 || key.channel() != 1 ||
        key.getDim() != value.getDim() ||
        key.width() != static_cast<size_t>(heads_kv) * head_dim ||
        key.getDataType() != value.getDataType())
      throw std::invalid_argument("H2O cache shape mismatch");
    const auto dtype = key.getDataType();
    if (dtype != DType::FP32 && dtype != DType::FP16 && dtype != DType::UINT16)
      throw std::invalid_argument("H2O requires FP32 or 16-bit KV storage");
  }

  static float probability(const nntrainer::Tensor &tensor, size_t index) {
    if (tensor.getDataType() == nntrainer::Tdatatype::FP32)
      return tensor.getData<float>()[index];
#ifdef ENABLE_FP16
    if (tensor.getDataType() == nntrainer::Tdatatype::FP16)
      return static_cast<float>(tensor.getData<_FP16>()[index]);
#endif
    throw std::invalid_argument("H2O requires FP32 or FP16 probabilities");
  }

  /** @brief Ascending source indices preserve all K/V bits during compaction.
   */
  static void compactHead(nntrainer::Tensor &cache,
                          const std::vector<unsigned int> &positions,
                          unsigned int head, unsigned int head_dim,
                          size_t element_size) {
    auto *data = cache.getData<unsigned char>();
    const size_t row_bytes = cache.width() * element_size;
    const size_t head_bytes = head_dim * element_size;
    for (size_t dst = 0; dst < positions.size(); ++dst)
      std::memmove(data + dst * row_bytes + head * head_bytes,
                   data + positions[dst] * row_bytes + head * head_bytes,
                   head_bytes);
  }

  H2OConfig config_;
  unsigned int prompt_length_ = 0;
  unsigned int cached_tokens_ = 0;
  unsigned int next_logical_position_ = 0;
  std::vector<std::vector<float>> scores_;
};

} // namespace causallm

#endif // __CAUSALLM_H2O_H__
