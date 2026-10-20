// SPDX-License-Identifier: Apache-2.0
/**
 * @file obcache.h
 * @date 20 October 2026
 * @brief Output-aware prompt KV pruning for CPU causal attention
 */
#ifndef __CAUSALLM_OBCACHE_H__
#define __CAUSALLM_OBCACHE_H__

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

#include <fp16.h>
#include <tensor.h>

namespace causallm {

/** @brief Uniform prompt budget includes the protected observation window. */
struct OBCacheConfig {
  unsigned int budget = 0, window = 64, kernel = 5;
  std::string score = "key";

  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("OBCache expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("OBCache integer exceeds unsigned int");
    return static_cast<unsigned int>(number);
  }

  void validate() const {
    if (budget &&
        (!window || window >= budget || !kernel || !(kernel % 2) ||
         kernel > static_cast<unsigned int>(std::numeric_limits<int>::max()) ||
         (score != "key" && score != "value" && score != "joint")))
      throw std::invalid_argument(
        "OBCache requires 0 < window < budget, an odd positive kernel and "
        "score=key, value or joint");
  }
};

/**
 * @brief OBCacheSnapKV: score and prune once, after the full prefill output.
 * @details Follows arXiv:2510.07651, equations (5)-(7), and the official
 * OBCacheSnapKVPress with gqa_aggregation=pre_redc: average probabilities and
 * raw scaled logits within a KV group BEFORE evaluating the nonlinear score.
 * Recompute O = mean(A) V, pool prefix scores, retain top budget-window and
 * every recent window token per KV head. Decode appends without eviction.
 * Only one causal query row of logits/probabilities is materialized at a time.
 * Cache allocations and absolute RoPE positions are unchanged.
 */
class OBCacheState {
public:
  bool enabled() const { return config_.budget != 0; }

  /** @brief Set prompt_length last to start a fresh session. */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key.compare(0, 8, "obcache_") != 0)
      return false;
    if (key == "obcache_score") {
      config_.score = value;
      return true;
    }
    const auto n = OBCacheConfig::parseUnsigned(value);
    if (key == "obcache_budget")
      config_.budget = n;
    else if (key == "obcache_window")
      config_.window = n;
    else if (key == "obcache_kernel")
      config_.kernel = n;
    else if (key == "obcache_layer")
      layer_ = n;
    else if (key == "obcache_prompt_length") {
      config_.validate();
      if (enabled() && !n)
        throw std::invalid_argument("OBCache requires a nonempty prompt");
      prompt_ = n;
      removed_ = next_ = 0;
    } else
      throw std::invalid_argument("Unknown OBCache property: " + key);
    return true;
  }

  void checkStep(unsigned int from, unsigned int step, unsigned int capacity,
                 unsigned int max_position) const {
    config_.validate();
    if (!enabled() || !prompt_ || from != next_ ||
        step != (next_ ? 1u : prompt_))
      throw std::invalid_argument(
        "OBCache requires one full prefill followed by consecutive single "
        "tokens; resumed and chunked prefill are unsupported");
    const auto limit = std::min(capacity, max_position);
    if (from >= limit || step > limit - from)
      throw std::out_of_range("OBCache step exceeds sequence or RoPE capacity");
  }

  unsigned int physicalPosition(unsigned int logical) const {
    if (logical < removed_)
      throw std::out_of_range("OBCache cannot rewind compressed KV");
    return logical - removed_;
  }

  /** @brief Validate storage before the original prefill writes any cache. */
  void checkCache(const nntrainer::Tensor &key, const nntrainer::Tensor &value,
                  unsigned int hq, unsigned int hk, unsigned int d) const {
    validateCache(key, value, hq, hk, d);
  }

  /** @brief Callback fills raw QK/sqrt(d) and causal softmax for one row. */
  template <typename ScoreRow>
  void finishPrefill(nntrainer::Tensor &rotated_query, nntrainer::Tensor &key,
                     nntrainer::Tensor &value, unsigned int hq, unsigned int hk,
                     unsigned int d, ScoreRow score_row) {
    checkStep(0, prompt_, key.height(), key.height());
    validateCache(key, value, hq, hk, d);
    validateQuery(rotated_query, prompt_, hq, d);
    const auto keep = std::min(prompt_, config_.budget);
    if (keep < prompt_) {
      const auto prefix = prompt_ - config_.window;
      std::vector<double> scores(static_cast<size_t>(hk) * prefix, 0.0);
      // Score every head before moving any values used by later queries.
      for (unsigned int pos = prefix; pos < prompt_; ++pos) {
        auto dim = rotated_query.getDim();
        dim.height(1);
        auto query = rotated_query.getSharedDataTensor(
          dim, static_cast<size_t>(pos) * rotated_query.width(), true);
        nntrainer::Tensor logits(1, 1, pos + 1, hq,
                                 rotated_query.getTensorType());
        nntrainer::Tensor probabilities(1, 1, pos + 1, hq,
                                        rotated_query.getTensorType());
        score_row(query, key, logits, probabilities, pos);
        accumulateScores(logits, probabilities, value, pos + 1, prefix, hq, hk,
                         d, scores);
      }
      compact(scores, key, value, prefix, hk, d, keep);
    }
    removed_ = prompt_ - keep;
    next_ = prompt_;
    std::clog << "[OBCache layer=" << layer_ << "] prompt=" << prompt_
              << " retained=" << keep << " score=" << config_.score
              << " gqa=pre_redc" << std::endl;
  }

  /** @brief Write at the physical tail, using logical positions for RoPE. */
  template <typename Store, typename Attend>
  void decodeStep(nntrainer::Tensor &query, nntrainer::Tensor &key,
                  nntrainer::Tensor &value, unsigned int logical,
                  unsigned int max_position, unsigned int hq, unsigned int hk,
                  unsigned int d, Store store, Attend attend) {
    checkStep(logical, 1, key.height(), max_position);
    validateCache(key, value, hq, hk, d);
    validateQuery(query, 1, hq, d);
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
    nntrainer::Tensor probabilities(1, 1, physical + 1, hq,
                                    query.getTensorType());
    attend(k, v, probabilities, physical);
    next_ = logical + 1;
  }

private:
  static void validateQuery(const nntrainer::Tensor &q, unsigned int rows,
                            unsigned int hq, unsigned int d) {
    if (q.getDataType() != nntrainer::Tdatatype::FP32 || q.batch() != 1 ||
        q.channel() != 1 || q.height() != rows ||
        q.getFormat() != nntrainer::Tformat::NCHW ||
        q.width() != static_cast<size_t>(hq) * d)
      throw std::invalid_argument("OBCache requires FP32 batch-one queries");
  }

  static void validateCache(const nntrainer::Tensor &k,
                            const nntrainer::Tensor &v, unsigned int hq,
                            unsigned int hk, unsigned int d) {
    if (!hq || !hk || hq % hk || !d || k.batch() != 1 || k.channel() != 1 ||
        k.getDim() != v.getDim() || k.getDataType() != v.getDataType() ||
        k.getFormat() != nntrainer::Tformat::NCHW ||
        v.getFormat() != nntrainer::Tformat::NCHW ||
        k.width() != static_cast<size_t>(hk) * d ||
        (k.getDataType() != nntrainer::Tdatatype::FP32 &&
         k.getDataType() != nntrainer::Tdatatype::UINT16 &&
         k.getDataType() != nntrainer::Tdatatype::FP16))
      throw std::invalid_argument("OBCache requires matching FP32 or half KV");
  }

  static float readValue(const nntrainer::Tensor &v, size_t index) {
    const float result =
      v.getDataType() == nntrainer::Tdatatype::FP32
        ? v.getData<float>()[index]
        : nntrainer::compute_fp16_to_fp32(v.getData<uint16_t>()[index]);
    if (!std::isfinite(result))
      throw std::invalid_argument("Nonfinite OBCache value state");
    return result;
  }

  void accumulateScores(const nntrainer::Tensor &logits,
                        const nntrainer::Tensor &probabilities,
                        const nntrainer::Tensor &value, unsigned int rows,
                        unsigned int prefix, unsigned int hq, unsigned int hk,
                        unsigned int d, std::vector<double> &scores) const {
    const auto *z = logits.getData<float>();
    const auto *a = probabilities.getData<float>();
    const auto group = hq / hk;
    std::vector<double> attention(rows), mean_logits(rows), output(d);
    for (unsigned int h = 0; h < hk; ++h) {
      std::fill(output.begin(), output.end(), 0.0);
      for (size_t j = 0; j < rows; ++j) {
        double aj = 0, zj = 0;
        for (unsigned int g = 0; g < group; ++g) {
          const auto index = j * hq + h * group + g;
          if (!std::isfinite(a[index]) || a[index] < 0 || a[index] > 1 ||
              !std::isfinite(z[index]))
            throw std::invalid_argument("Invalid OBCache attention/logit");
          aj += a[index] / group;
          zj += z[index] / group;
        }
        attention[j] = aj;
        mean_logits[j] = zj;
        if (config_.score != "value")
          for (unsigned int x = 0; x < d; ++x)
            output[x] += aj * readValue(value, j * value.width() + h * d + x);
      }
      for (size_t j = 0; j < prefix; ++j) {
        double norm = 0;
        for (unsigned int x = 0; x < d; ++x) {
          const double v = readValue(value, j * value.width() + h * d + x);
          // Joint = A^2 ||V + Z(V-O)||^2, algebraically equation (7).
          const double delta = config_.score == "value" ? v
                               : config_.score == "key"
                                 ? mean_logits[j] * (v - output[x])
                                 : v + mean_logits[j] * (v - output[x]);
          norm += delta * delta;
        }
        auto &s = scores[static_cast<size_t>(h) * prefix + j];
        s += attention[j] * attention[j] * norm;
        if (!std::isfinite(s))
          throw std::invalid_argument("Nonfinite OBCache pruning score");
      }
    }
  }

  void compact(const std::vector<double> &scores, nntrainer::Tensor &key,
               nntrainer::Tensor &value, unsigned int prefix, unsigned int hk,
               unsigned int d, unsigned int keep) const {
    std::vector<double> pooled(prefix);
    std::vector<unsigned int> indices(prefix), positions(keep);
    for (unsigned int h = 0; h < hk; ++h) {
      const size_t radius = config_.kernel / 2;
      const auto start = scores.begin() + static_cast<size_t>(h) * prefix;
      for (size_t j = 0; j < prefix; ++j) {
        const auto lo = j > radius ? j - radius : 0;
        const auto hi = j + std::min<size_t>(radius + 1, prefix - j);
        pooled[j] =
          std::accumulate(start + lo, start + hi, 0.0) / config_.kernel;
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

  OBCacheConfig config_;
  unsigned int layer_ = 0, prompt_ = 0, removed_ = 0, next_ = 0;
};
} // namespace causallm
#endif // __CAUSALLM_OBCACHE_H__
