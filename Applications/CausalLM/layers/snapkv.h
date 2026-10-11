// SPDX-License-Identifier: Apache-2.0
/**
 * @file snapkv.h
 * @date 11 October 2026
 * @brief Head-wise prompt KV selection following SnapKVCluster.update_kv
 * @note C++ adaptation of FasterDecoding/SnapKV (Apache-2.0), commit e216ddc8,
 *       snapkv/monkeypatch/snapkv_utils.py and llama_hijack_4_37.py.
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
  std::string pooling = "avgpool";

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
    if (budget && (!window || window >= budget || !kernel || !(kernel % 2) ||
                   (pooling != "avgpool" && pooling != "maxpool")))
      throw std::invalid_argument(
        "SnapKV requires 0 < window < budget, an odd positive kernel, and "
        "avgpool or maxpool");
  }
};

/**
 * @brief Layer-local single-prompt compression. Absolute RoPE positions remain
 * unchanged. Each query head independently selects prompt positions, including
 * GQA heads sharing the original K/V. The host reserves decode space after
 * prefill and releases the full prompt cache.
 */
class SnapKVState {
public:
  bool enabled() const { return config_.budget != 0; }
  bool compressed() const { return removed_tokens_ != 0; }

  /** @brief Repeat a newly appended GQA token after applying its absolute RoPE.
   * The original KV heads occupy the start of the expanded single-token row.
   */
  static void repeatKV(nntrainer::Tensor &step, unsigned int heads_kv,
                       unsigned int head_dim) {
    if (!head_dim || !heads_kv || step.width() % head_dim)
      throw std::invalid_argument("SnapKV repeatKV head shape mismatch");
    const unsigned int heads_q = step.width() / head_dim;
    if (step.height() != 1 || !heads_q || heads_q % heads_kv)
      throw std::invalid_argument("SnapKV repeatKV requires one GQA token");
    const size_t head_bytes = step.bytes() / heads_q;
    auto *data = step.getData<unsigned char>();
    const unsigned int group = heads_q / heads_kv;
    for (unsigned int head = heads_q; head-- > 0;)
      std::memmove(data + head * head_bytes, data + (head / group) * head_bytes,
                   head_bytes);
  }

  /** @brief Consume settings, with prompt_length last to start a new session.
   */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key == "snapkv_pooling") {
      config_.pooling = value;
      return true;
    }
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
    const auto physical_from = physicalPosition(logical_from);
    if (logical_from > max_position ||
        step_size > max_position - logical_from || physical_from > capacity ||
        step_size > capacity - physical_from)
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
    // The caller already checked the absolute RoPE limit. Cache capacity is
    // physical and can be smaller than the absolute position after eviction.
    checkStep(logical_from, step_size, key.height(),
              std::numeric_limits<unsigned int>::max());
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

  /** @brief Gather one query head from its original shared KV head. */
  static void gatherHead(const nntrainer::Tensor &source,
                         nntrainer::Tensor &destination,
                         const std::vector<unsigned int> &positions,
                         unsigned int head, unsigned int group,
                         unsigned int head_dim, size_t element_size) {
    const auto *src = source.getData<unsigned char>();
    auto *dst = destination.getData<unsigned char>();
    const size_t head_bytes = head_dim * element_size;
    for (size_t row = 0; row < positions.size(); ++row)
      std::memcpy(dst + row * destination.width() * element_size +
                    head * head_bytes,
                  src + positions[row] * source.width() * element_size +
                    (head / group) * head_bytes,
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

    auto compact_dim = key.getDim();
    compact_dim.height(config_.budget);
    compact_dim.width(heads_q * head_dim);
    if (compact_dim.getDataLen() > key.size())
      throw std::invalid_argument(
        "SnapKV prompt storage cannot stage expanded heads");
    nntrainer::Tensor compact_key(compact_dim, true);
    nntrainer::Tensor compact_value(compact_dim, true);
    // Equivalent to repeat_kv -> per-query-head gather, without materializing
    // an expanded full-length prompt cache.
    for (unsigned int head = 0; head < heads_q; ++head) {
      std::fill(votes.begin(), votes.end(), 0.0f);
      for (size_t q = prefix; q < length; ++q) {
        const size_t row = q * (q + 1) / 2;
        for (unsigned int j = 0; j < prefix; ++j)
          votes[j] += probability(probs, (row + j) * heads_q + head);
      }
      for (auto &vote : votes) {
#ifdef ENABLE_FP16
        // torch.sum returns the input dtype after accumulation.
        if (probs.getDataType() == DType::FP16)
          vote = static_cast<float>(static_cast<_FP16>(vote));
#endif
        if (!std::isfinite(vote))
          throw std::invalid_argument(
            "SnapKV received nonfinite attention votes");
      }
      const size_t radius = config_.kernel / 2;
      for (size_t j = 0; j < prefix; ++j) {
        const size_t start = j > radius ? j - radius : 0;
        const size_t end = std::min<size_t>(prefix, j + radius + 1);
        pooled[j] =
          config_.pooling == "maxpool"
            ? *std::max_element(votes.begin() + start, votes.begin() + end)
            : std::accumulate(votes.begin() + start, votes.begin() + end,
                              0.0f) /
                config_.kernel; // avg_pool1d includes padded zeros
#ifdef ENABLE_FP16
        if (probs.getDataType() == DType::FP16)
          pooled[j] = static_cast<float>(static_cast<_FP16>(pooled[j]));
#endif
      }
      std::iota(indices.begin(), indices.end(), 0u);
      const auto greater = [&](unsigned int a, unsigned int b) {
        return pooled[a] > pooled[b];
      };
      // Follow torch.topk's CPU partition/sort strategy, retaining score order.
      // Tied indices are not guaranteed across PyTorch backends/platforms.
      if (static_cast<size_t>(keep_prefix) * 64 <= prefix) {
        std::partial_sort(indices.begin(), indices.begin() + keep_prefix,
                          indices.end(), greater);
      } else {
        std::nth_element(indices.begin(), indices.begin() + keep_prefix - 1,
                         indices.end(), greater);
        std::sort(indices.begin(), indices.begin() + keep_prefix - 1, greater);
      }
      std::copy_n(indices.begin(), keep_prefix, positions.begin());
      std::iota(positions.begin() + keep_prefix, positions.end(), prefix);
      gatherHead(key, compact_key, positions, head, group, head_dim,
                 element_size);
      gatherHead(value, compact_value, positions, head, group, head_dim,
                 element_size);
    }
    // Stage the gathered Hq rows in the host-owned slab. NNTrainer clears
    // external input bindings after prefill; changing only those views would
    // lose the new buffer. The host copies this packed prefix into the small
    // Hq cache and rebinds all views before the next attention call.
    std::memcpy(key.getData(), compact_key.getData(), compact_key.bytes());
    std::memcpy(value.getData(), compact_value.getData(),
                compact_value.bytes());
  }

  SnapKVConfig config_;
  unsigned int prompt_length_ = 0;
  unsigned int removed_tokens_ = 0;
  unsigned int next_logical_position_ = 0;
  bool prompt_finished_ = false;
};

} // namespace causallm

#endif // __CAUSALLM_SNAPKV_H__
