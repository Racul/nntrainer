// SPDX-License-Identifier: Apache-2.0
/**
 * @file snapkv.h
 * @date 11 October 2026
 * @brief Prompt KV selection with observation-window votes shared per KV head
 */

#ifndef __CAUSALLM_SNAPKV_H__
#define __CAUSALLM_SNAPKV_H__

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

/** @brief Runtime-only settings; budget includes the observation window. */
struct SnapKVConfig {
  unsigned int budget = 0;
  unsigned int window = 16;
  unsigned int kernel = 5;

  /** @brief Parse a nonnegative integer without truncation or wraparound. */
  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("SnapKV expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("SnapKV integer exceeds unsigned int");
    return static_cast<unsigned int>(number);
  }

  /** @brief A zero budget disables SnapKV. */
  void validate() const {
    if (budget && (!window || window >= budget || !kernel || !(kernel % 2)))
      throw std::invalid_argument(
        "SnapKV requires 0 < window < budget and an odd positive kernel");
  }
};

/**
 * @brief Layer-local single-prompt compression. Absolute RoPE positions remain
 * unchanged. GQA query-head votes are averaged within each KV head; this
 * differs from the original SnapKV implementation's independently expanded
 * query heads. Cache allocation and tensor shapes are preserved.
 */
class SnapKVState {
public:
  bool enabled() const { return config_.budget != 0; }

  /** @brief Consume settings, with prompt_length last to start a new session.
   */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key != "snapkv_budget" && key != "snapkv_window" &&
        key != "snapkv_kernel" && key != "snapkv_prompt_length")
      return false;
    const auto number = SnapKVConfig::parseUnsigned(value);
    if (key == "snapkv_budget")
      config_.budget = number;
    else if (key == "snapkv_window")
      config_.window = number;
    else if (key == "snapkv_kernel")
      config_.kernel = number;
    else {
      config_.validate();
      if (enabled() && !number)
        throw std::invalid_argument("SnapKV requires a nonempty prompt");
      prompt_length_ = number;
      removed_tokens_ = next_logical_position_ = 0;
      prompt_finished_ = false;
    }
    return true;
  }

  /** @brief Validate before writing K/V or applying RoPE. */
  void checkStep(unsigned int logical_from, unsigned int step_size,
                 unsigned int capacity, unsigned int max_position) const {
    if (!enabled())
      return;
    config_.validate();
    if (!prompt_length_ || logical_from != next_logical_position_ ||
        step_size != (prompt_finished_ ? 1u : prompt_length_))
      throw std::invalid_argument("SnapKV requires one full prefill followed "
                                  "by consecutive single tokens");
    const auto limit = std::min(capacity, max_position);
    if (logical_from > limit || step_size > limit - logical_from)
      throw std::out_of_range("SnapKV step exceeds sequence or RoPE capacity");
  }

  /** @brief Map a logical token position to its compact cache offset. */
  unsigned int physicalPosition(unsigned int logical_position) const {
    if (!enabled())
      return logical_position;
    if (logical_position < removed_tokens_)
      throw std::out_of_range("SnapKV cannot rewind a compressed cache");
    return logical_position - removed_tokens_;
  }

  /** @brief Call after attention has consumed the original V cache. */
  void finishStep(const nntrainer::Tensor &probabilities,
                  nntrainer::Tensor &key, nntrainer::Tensor &value,
                  unsigned int logical_from, unsigned int step_size,
                  unsigned int heads_q, unsigned int heads_kv,
                  unsigned int head_dim) {
    if (!enabled())
      return;
    checkStep(logical_from, step_size, key.height(), key.height());
    if (!prompt_finished_ && prompt_length_ > config_.budget) {
      compress(probabilities, key, value, heads_q, heads_kv, head_dim);
      removed_tokens_ = prompt_length_ - config_.budget;
    }
    prompt_finished_ = true;
    next_logical_position_ = logical_from + step_size;
  }

private:
  /** @brief Read probabilities in their original precision, accumulate in FP32.
   */
  static float probability(const nntrainer::Tensor &tensor, size_t index) {
    if (tensor.getDataType() == nntrainer::Tdatatype::FP32)
      return tensor.getData<float>()[index];
#ifdef ENABLE_FP16
    if (tensor.getDataType() == nntrainer::Tdatatype::FP16)
      return static_cast<float>(tensor.getData<_FP16>()[index]);
#endif
    throw std::invalid_argument("SnapKV requires FP32 or FP16 probabilities");
  }

  /** @brief Move one head in ascending source order, preserving K/V bitwise. */
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

  void compress(const nntrainer::Tensor &probs, nntrainer::Tensor &key,
                nntrainer::Tensor &value, unsigned int heads_q,
                unsigned int heads_kv, unsigned int head_dim) const {
    using DType = nntrainer::Tdatatype;
    const size_t length = prompt_length_;
    if (!heads_kv || !heads_q || heads_q % heads_kv || !head_dim ||
        key.batch() != 1 || key.channel() != 1 ||
        key.getDim() != value.getDim() ||
        key.width() != static_cast<size_t>(heads_kv) * head_dim ||
        probs.batch() != 1 || probs.channel() != 1 ||
        probs.width() != heads_q || probs.height() != length * (length + 1) / 2)
      throw std::invalid_argument(
        "SnapKV cache or causal probability shape mismatch");
    const auto dtype = key.getDataType();
    if (dtype != DType::FP32 && dtype != DType::FP16 && dtype != DType::UINT16)
      throw std::invalid_argument(
        "SnapKV cache must use FP32 or 16-bit storage");
    const size_t element_size =
      dtype == DType::FP32 ? sizeof(float) : sizeof(uint16_t);
    const unsigned int prefix = prompt_length_ - config_.window;
    const unsigned int keep_prefix = config_.budget - config_.window;
    const unsigned int group = heads_q / heads_kv;
    std::vector<float> votes(prefix), pooled(prefix);
    std::vector<unsigned int> indices(prefix), positions(config_.budget);

    for (unsigned int head = 0; head < heads_kv; ++head) {
      std::fill(votes.begin(), votes.end(), 0.0f);
      for (size_t q = prefix; q < length; ++q) {
        const size_t row = q * (q + 1) / 2;
        for (unsigned int j = 0; j < prefix; ++j)
          for (unsigned int g = 0; g < group; ++g)
            votes[j] +=
              probability(probs, (row + j) * heads_q + head * group + g);
      }
      for (auto &vote : votes) {
        vote /= group;
        if (!std::isfinite(vote))
          throw std::invalid_argument(
            "SnapKV received nonfinite attention votes");
      }
      const size_t radius = config_.kernel / 2;
      for (size_t j = 0; j < prefix; ++j) {
        const size_t start = j > radius ? j - radius : 0;
        const size_t end = std::min<size_t>(prefix, j + radius + 1);
        pooled[j] =
          *std::max_element(votes.begin() + start, votes.begin() + end);
      }
      std::iota(indices.begin(), indices.end(), 0u);
      std::partial_sort(indices.begin(), indices.begin() + keep_prefix,
                        indices.end(), [&](unsigned int a, unsigned int b) {
                          return pooled[a] == pooled[b] ? a < b
                                                        : pooled[a] > pooled[b];
                        });
      std::sort(indices.begin(), indices.begin() + keep_prefix);
      std::copy_n(indices.begin(), keep_prefix, positions.begin());
      std::iota(positions.begin() + keep_prefix, positions.end(), prefix);
      compactHead(key, positions, head, head_dim, element_size);
      compactHead(value, positions, head, head_dim, element_size);
    }
  }

  SnapKVConfig config_;
  unsigned int prompt_length_ = 0;
  unsigned int removed_tokens_ = 0;
  unsigned int next_logical_position_ = 0;
  bool prompt_finished_ = false;
};

} // namespace causallm

#endif // __CAUSALLM_SNAPKV_H__
