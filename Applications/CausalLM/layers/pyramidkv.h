// SPDX-License-Identifier: Apache-2.0
/**
 * @file pyramidkv.h
 * @date 17 October 2026
 * @brief Layer-dependent prompt KV selection for CPU causal attention
 */
#ifndef __CAUSALLM_PYRAMIDKV_H__
#define __CAUSALLM_PYRAMIDKV_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <tensor.h>

namespace causallm {

/** @brief Prompt budget includes the observation window; zero disables it. */
struct PyramidKVConfig {
  unsigned int budget = 0, window = 64, kernel = 5, beta = 20;

  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("PyramidKV expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("PyramidKV integer exceeds unsigned int");
    return static_cast<unsigned int>(number);
  }

  void validate() const {
    if (budget &&
        (!window || window >= budget || !kernel || !(kernel % 2) || !beta ||
         budget > std::numeric_limits<unsigned int>::max() / 2))
      throw std::invalid_argument("PyramidKV requires 0 < window < budget, odd "
                                  "positive kernel and positive beta");
  }

  /**
   * @brief Official integer schedule, including the short-prompt fallback.
   * @details Lower layers keep more prefix entries. Integer division matches
   * PyramidKVCluster.update_kv; rounding can make the mean exceed budget.
   */
  unsigned int retained(unsigned int length, unsigned int layer,
                        unsigned int layers) const {
    validate();
    if (!budget || !layers || layer >= layers)
      throw std::invalid_argument("Invalid PyramidKV layer configuration");
    if (length < budget)
      return length;
    const auto prefix = budget - window;
    if (length < 2 * prefix || layers == 1)
      return budget;
    auto low = prefix / beta;
    auto high = 2 * prefix - low;
    if (high >= length - window) {
      high = length - window;
      low = 2 * prefix - high;
    }
    return high - layer * ((high - low) / (layers - 1)) + window;
  }
};

/**
 * @brief Compress once after full prefill; append all subsequent decode KV.
 * @details Follows https://arxiv.org/abs/2406.02069 and the official
 * PyramidKVCluster: last-window causal softmax votes, GQA mean reduction,
 * zero-padded average pooling, per-layer top-k, and all recent window tokens.
 * Sorted selected positions preserve temporal order and K/V bits. Absolute
 * RoPE positions and allocated tensor shapes are unchanged. Only the window
 * probability rows are materialized, after the original prefill attention.
 */
class PyramidKVState {
public:
  bool enabled() const { return config_.budget != 0; }

  /** @brief Set prompt_length last to reset a session. */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key.compare(0, 10, "pyramidkv_") != 0)
      return false;
    const auto n = PyramidKVConfig::parseUnsigned(value);
    if (key == "pyramidkv_budget")
      config_.budget = n;
    else if (key == "pyramidkv_window")
      config_.window = n;
    else if (key == "pyramidkv_kernel")
      config_.kernel = n;
    else if (key == "pyramidkv_beta")
      config_.beta = n;
    else if (key == "pyramidkv_layer")
      layer_ = n;
    else if (key == "pyramidkv_layers")
      layers_ = n;
    else if (key == "pyramidkv_prompt_length") {
      config_.validate();
      if (enabled() && (!n || !layers_ || layer_ >= layers_))
        throw std::invalid_argument("Invalid PyramidKV session");
      prompt_ = n;
      removed_ = next_ = 0;
    } else
      return false;
    return true;
  }

  void checkStep(unsigned int from, unsigned int step, unsigned int capacity,
                 unsigned int max_position) const {
    config_.validate();
    if (!enabled() || !prompt_ || from != next_ ||
        step != (next_ ? 1u : prompt_))
      throw std::invalid_argument("PyramidKV requires one full prefill "
                                  "followed by consecutive single tokens");
    const auto limit = std::min(capacity, max_position);
    if (from > limit || step > limit - from)
      throw std::out_of_range(
        "PyramidKV step exceeds sequence or RoPE capacity");
  }

  unsigned int physicalPosition(unsigned int logical) const {
    if (logical < removed_)
      throw std::out_of_range("PyramidKV cannot rewind compressed KV");
    return logical - removed_;
  }

  /** @brief Score only the observation rows using existing attention kernels.
   */
  template <typename Score>
  void finishPrefill(nntrainer::Tensor &rotated_query, nntrainer::Tensor &key,
                     nntrainer::Tensor &value, unsigned int heads_q,
                     unsigned int heads_kv, unsigned int head_dim,
                     Score score) {
    checkStep(0, prompt_, key.height(), key.height());
    validateCache(key, value, heads_q, heads_kv, head_dim);
    if (rotated_query.getDataType() != nntrainer::Tdatatype::FP32 ||
        rotated_query.batch() != 1 || rotated_query.channel() != 1 ||
        rotated_query.height() != prompt_ ||
        rotated_query.width() != static_cast<size_t>(heads_q) * head_dim)
      throw std::invalid_argument(
        "PyramidKV prefill query shape or dtype mismatch");
    const auto keep = config_.retained(prompt_, layer_, layers_);
    if (keep < prompt_) {
      const auto prefix = prompt_ - config_.window;
      auto dim = rotated_query.getDim();
      dim.height(config_.window);
      auto query = rotated_query.getSharedDataTensor(
        dim, static_cast<size_t>(prefix) * rotated_query.width(), true);
      const uint64_t rows =
        static_cast<uint64_t>(config_.window) * prefix +
        static_cast<uint64_t>(config_.window) * (config_.window + 1ull) / 2;
      if (rows > std::numeric_limits<unsigned int>::max())
        throw std::out_of_range("PyramidKV probability tensor too large");
      nntrainer::Tensor probabilities(1, 1, static_cast<unsigned int>(rows),
                                      heads_q, rotated_query.getTensorType());
      score(query, key, probabilities, prefix, config_.window);
      compact(probabilities, key, value, heads_q, heads_kv, head_dim, keep);
    }
    removed_ = prompt_ - keep;
    next_ = prompt_;
    std::clog << "[PyramidKV layer=" << layer_ << "] prompt=" << prompt_
              << " retained=" << keep << std::endl;
  }

  /** @brief Decode over compact KV, retaining each new token without eviction.
   */
  template <typename Store, typename Attend>
  void decodeStep(nntrainer::Tensor &query, nntrainer::Tensor &key,
                  nntrainer::Tensor &value, unsigned int logical,
                  unsigned int max_position, unsigned int heads_q,
                  unsigned int heads_kv, unsigned int head_dim, Store store,
                  Attend attend) {
    checkStep(logical, 1, key.height(), max_position);
    validateCache(key, value, heads_q, heads_kv, head_dim);
    if (query.getDataType() != nntrainer::Tdatatype::FP32 ||
        query.batch() != 1 || query.channel() != 1 || query.height() != 1 ||
        query.width() != static_cast<size_t>(heads_q) * head_dim)
      throw std::invalid_argument(
        "PyramidKV decode query shape or dtype mismatch");
    const auto physical = physicalPosition(logical);
    auto dim = key.getDim();
    dim.height(1);
    auto k = key.getSharedDataTensor(
      dim, static_cast<size_t>(physical) * key.width(), true);
    auto v = value.getSharedDataTensor(
      dim, static_cast<size_t>(physical) * value.width(), true);
    store(k, v);
    dim.height(physical + 1);
    k = key.getSharedDataTensor(dim, 0, true);
    v = value.getSharedDataTensor(dim, 0, true);
    nntrainer::Tensor probabilities(1, 1, physical + 1, heads_q,
                                    query.getTensorType());
    attend(k, v, probabilities, physical);
    next_ = logical + 1;
  }

private:
  static void validateCache(const nntrainer::Tensor &key,
                            const nntrainer::Tensor &value, unsigned int hq,
                            unsigned int hk, unsigned int d) {
    if (!hq || !hk || hq % hk || !d || key.batch() != 1 || key.channel() != 1 ||
        key.getDim() != value.getDim() ||
        key.width() != static_cast<size_t>(hk) * d ||
        (key.getDataType() != nntrainer::Tdatatype::FP32 &&
         key.getDataType() != nntrainer::Tdatatype::UINT16))
      throw std::invalid_argument(
        "PyramidKV requires matching FP32 or UINT16 KV caches");
  }

  void compact(const nntrainer::Tensor &probs, nntrainer::Tensor &key,
               nntrainer::Tensor &value, unsigned int hq, unsigned int hk,
               unsigned int d, unsigned int keep) const {
    const auto prefix = prompt_ - config_.window;
    const auto group = hq / hk;
    const auto *p = probs.getData<float>();
    std::vector<float> votes(prefix), pooled(prefix);
    std::vector<unsigned int> indices(prefix), positions(keep);
    for (unsigned int h = 0; h < hk; ++h) {
      std::fill(votes.begin(), votes.end(), 0.0f);
      size_t row = 0;
      for (size_t q = prefix; q < prompt_; ++q) {
        for (size_t j = 0; j < prefix; ++j)
          for (unsigned int g = 0; g < group; ++g) {
            const float vote = p[(row + j) * hq + h * group + g];
            if (!std::isfinite(vote) || vote < 0)
              throw std::invalid_argument(
                "Invalid PyramidKV attention probabilities");
            votes[j] += vote / group;
          }
        row += q + 1;
      }
      const size_t radius = config_.kernel / 2;
      for (size_t j = 0; j < prefix; ++j) {
        const size_t start = j > radius ? j - radius : 0;
        const size_t end = std::min<size_t>(prefix, j + radius + 1);
        pooled[j] =
          std::accumulate(votes.begin() + start, votes.begin() + end, 0.0f) /
          config_.kernel;
      }
      const auto old_keep = keep - config_.window;
      std::iota(indices.begin(), indices.end(), 0u);
      std::partial_sort(indices.begin(), indices.begin() + old_keep,
                        indices.end(), [&](unsigned int a, unsigned int b) {
                          return pooled[a] == pooled[b] ? a < b
                                                        : pooled[a] > pooled[b];
                        });
      std::sort(indices.begin(), indices.begin() + old_keep);
      std::copy_n(indices.begin(), old_keep, positions.begin());
      std::iota(positions.begin() + old_keep, positions.end(), prefix);
      const size_t bytes = key.getDataType() == nntrainer::Tdatatype::FP32
                             ? sizeof(float)
                             : sizeof(uint16_t);
      for (auto *cache : {&key, &value}) {
        auto *data = cache->getData<unsigned char>();
        for (size_t dst = 0; dst < positions.size(); ++dst)
          std::memmove(data + (dst * cache->width() + h * d) * bytes,
                       data + (positions[dst] * cache->width() + h * d) * bytes,
                       d * bytes);
      }
    }
  }

  PyramidKVConfig config_;
  unsigned int layer_ = 0, layers_ = 0, prompt_ = 0, removed_ = 0, next_ = 0;
};
} // namespace causallm
#endif // __CAUSALLM_PYRAMIDKV_H__
