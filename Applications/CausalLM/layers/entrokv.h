// SPDX-License-Identifier: Apache-2.0
/**
 * @file entrokv.h
 * @brief Entro-SnapKV allocation, observation and token-major cache compaction.
 * @see https://proceedings.mlr.press/v306/gao26a.html (Eq. 8-9, Appendix E.1)
 */

#ifndef __CAUSALLM_ENTROKV_H__
#define __CAUSALLM_ENTROKV_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <tensor.h>

namespace causallm {

/** @brief beta=0 disables EntroKV. No model calibration constant is invented.
 */
struct EntroKVConfig {
  double beta = 0.0;
  double alpha = 1.0;
  double baseline_entropy = 0.0;
  bool calibrated = false;
  unsigned int window = 32;
  unsigned int kernel = 7;

  bool enabled() const { return beta > 0.0; }

  void validate() const {
    if (!std::isfinite(beta) || beta < 0.0 || beta > 1.0 ||
        !std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0 ||
        !std::isfinite(baseline_entropy) || baseline_entropy < 0.0 ||
        baseline_entropy > 1.0 || window == 0 || kernel == 0 ||
        kernel % 2 == 0 || (enabled() && alpha < 1.0 && !calibrated))
      throw std::invalid_argument(
        "EntroKV: invalid allocation/window settings");
  }

  /** @brief Strict runtime-property parsing, including full consumption. */
  static double real(const std::string &s) {
    size_t end = 0;
    const double v = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(v))
      throw std::invalid_argument("EntroKV: expected a finite number");
    return v;
  }

  static unsigned int integer(const std::string &s) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("EntroKV: expected an unsigned integer");
    const auto v = std::stoull(s);
    if (v > std::numeric_limits<unsigned int>::max())
      throw std::invalid_argument("EntroKV: integer exceeds unsigned range");
    return static_cast<unsigned int>(v);
  }
};

/**
 * @brief One full prefill followed by single-token decode, batch 1.
 * @details Quotas vary by KV head. Each head is compacted into its own prefix;
 * unused rows up to the largest quota are zeroed and masked in decode. New
 * tokens share the region after that largest quota. RoPE stays in logical
 * coordinates. The owning KVCacheManager's allocation is unchanged.
 */
class EntroKVState {
public:
  const EntroKVConfig &config() const { return config_; }
  bool enabled() const { return config_.enabled(); }
  unsigned int promptLength() const { return prompt_; }
  unsigned int retainedSpan() const { return span_; }
  const std::vector<unsigned int> &quotas() const { return quotas_; }
  const std::vector<double> &entropies() const { return entropy_; }

  /** @brief Cases whose quotas are known to retain every prompt token. */
  bool keepsFullCache() const {
    return prompt_ <= config_.window ||
           (config_.beta == 1.0 && config_.alpha == 0.0 &&
            config_.baseline_entropy == 1.0);
  }

  /** @brief Preserve the original attention path when no eviction is needed. */
  void finishUncompressed(unsigned int kv_heads) {
    if (started_ || !keepsFullCache() || kv_heads == 0 || prompt_ == 0)
      throw std::invalid_argument("EntroKV: invalid uncompressed transition");
    quotas_.assign(kv_heads, prompt_);
    span_ = prompt_;
    next_ = prompt_;
    started_ = true;
  }

  /** @brief Configure atomically before inference; cache_index is independent.
   */
  void configure(const EntroKVConfig &cfg, unsigned int prompt) {
    cfg.validate();
    if (started_)
      throw std::invalid_argument(
        "EntroKV: cannot reconfigure an active cache");
    if (cfg.enabled() && prompt == 0)
      throw std::invalid_argument("EntroKV: prompt_length must be positive");
    config_ = cfg;
    prompt_ = prompt;
    span_ = prompt;
    next_ = 0;
    quotas_.clear();
    entropy_.clear();
    votes_.clear();
    observed_ = 0;
  }

  void checkStep(unsigned int logical, unsigned int rows, unsigned int capacity,
                 unsigned int rope_capacity) const {
    if (logical != next_ || rows == 0 ||
        (!started_ && (logical != 0 || rows != prompt_)) ||
        (started_ && rows != 1))
      throw std::invalid_argument("EntroKV: requires one complete prefill then "
                                  "consecutive 1-token decode");
    if (logical > capacity || rows > capacity - logical ||
        logical > rope_capacity || rows > rope_capacity - logical)
      throw std::invalid_argument("EntroKV: logical cache/RoPE range exceeded");
  }

  unsigned int physicalPosition(unsigned int logical) const {
    if (!started_)
      return logical;
    if (logical < prompt_)
      throw std::invalid_argument("EntroKV: resume/rewind is unsupported");
    return span_ + logical - prompt_;
  }

  /**
   * @brief Reduce FP32 causal softmax blocks, without another QK matmul.
   * @details Layout: [query][visible key][query head], triangular row packing.
   * Entropy uses all visible keys, normalized by log(full prompt length).
   * Prefix votes exclude the always-kept recent window. GQA uses mean pooling.
   */
  void observe(const nntrainer::Tensor &probabilities, unsigned int from,
               unsigned int rows, unsigned int query_heads,
               unsigned int kv_heads) {
    if (query_heads == 0 || kv_heads == 0 || query_heads % kv_heads != 0 ||
        from > prompt_ || rows > prompt_ - from || started_ ||
        probabilities.getDataType() != ml::train::TensorDim::DataType::FP32)
      throw std::invalid_argument("EntroKV: invalid observation geometry/type");
    const unsigned int window = std::min(config_.window, prompt_);
    const unsigned int prefix = prompt_ - window;
    const unsigned int group = query_heads / kv_heads;
    const size_t count = (triangle(from + rows) - triangle(from)) * query_heads;
    if (probabilities.size() != count)
      throw std::invalid_argument("EntroKV: observation buffer size mismatch");
    if (entropy_.empty()) {
      entropy_.assign(kv_heads, 0.0);
      votes_.assign(static_cast<size_t>(kv_heads) * prefix, 0.0);
    } else if (entropy_.size() != kv_heads)
      throw std::invalid_argument("EntroKV: head geometry changed");

    const float *p = probabilities.getData<float>();
    const double scale = 1.0 / (static_cast<double>(window) * group);
    for (unsigned int q = std::max(from, prefix); q < from + rows; ++q) {
      const size_t base = (triangle(q) - triangle(from)) * query_heads;
      for (unsigned int j = 0; j <= q; ++j) {
        for (unsigned int h = 0; h < query_heads; ++h) {
          const double a = p[base + static_cast<size_t>(j) * query_heads + h];
          if (!std::isfinite(a) || a < 0.0 || a > 1.00001)
            throw std::runtime_error("EntroKV: invalid attention probability");
          const unsigned int kv = h / group;
          if (a > 0.0 && prompt_ > 1)
            entropy_[kv] -= a * std::log(a) * scale / std::log(prompt_);
          if (j < prefix)
            votes_[static_cast<size_t>(kv) * prefix + j] += a * scale;
        }
      }
      ++observed_;
    }
  }

  /** @brief Evict only after every prefill attention output has consumed V. */
  void finishPrefill(nntrainer::Tensor &key, nntrainer::Tensor &value,
                     unsigned int kv_heads, unsigned int head_dim) {
    const unsigned int window = std::min(config_.window, prompt_);
    const unsigned int prefix = prompt_ - window;
    if (started_ || observed_ != window || entropy_.size() != kv_heads ||
        key.getDim() != value.getDim() || key.batch() != 1 ||
        key.channel() != 1 || key.width() != kv_heads * head_dim ||
        key.height() < prompt_ || head_dim == 0)
      throw std::invalid_argument("EntroKV: incomplete prefill/cache geometry");
    const auto dtype = key.getDataType();
    if (dtype != ml::train::TensorDim::DataType::FP32 &&
        dtype != ml::train::TensorDim::DataType::FP16 &&
        dtype != ml::train::TensorDim::DataType::UINT16)
      throw std::invalid_argument("EntroKV: unsupported cache dtype");
    quotas_.resize(kv_heads);
    std::vector<std::vector<unsigned int>> positions(kv_heads);
    span_ = 0;
    for (unsigned int h = 0; h < kv_heads; ++h) {
      entropy_[h] = std::clamp(entropy_[h], 0.0, 1.0);
      // Appendix E.1 Eq. 8, with the optional Eq. 10 normalization disabled.
      const double ratio =
        config_.beta * ((1.0 - config_.alpha) * config_.baseline_entropy +
                        config_.alpha * entropy_[h]);
      quotas_[h] = std::max(
        window, static_cast<unsigned int>(std::floor(prompt_ * ratio)));
      span_ = std::max(span_, quotas_[h]);
      std::vector<double> pooled(prefix);
      const size_t radius = config_.kernel / 2;
      for (unsigned int j = 0; j < prefix; ++j) {
        const size_t begin = j > radius ? j - radius : 0;
        const size_t end = std::min<size_t>(prefix, size_t(j) + radius + 1);
        pooled[j] =
          *std::max_element(votes_.begin() + size_t(h) * prefix + begin,
                            votes_.begin() + size_t(h) * prefix + end);
      }
      auto &keep = positions[h];
      keep.resize(prefix);
      std::iota(keep.begin(), keep.end(), 0u);
      const size_t selected = quotas_[h] - window;
      std::partial_sort(keep.begin(), keep.begin() + selected, keep.end(),
                        [&](unsigned int a, unsigned int b) {
                          return pooled[a] == pooled[b] ? a < b
                                                        : pooled[a] > pooled[b];
                        });
      keep.resize(selected);
      std::sort(keep.begin(), keep.end());
      for (unsigned int j = prefix; j < prompt_; ++j)
        keep.push_back(j);
    }
    if (dtype == ml::train::TensorDim::DataType::FP32)
      compact<float>(key, value, positions, head_dim);
    else
      compact<uint16_t>(key, value, positions, head_dim);
    // Release observation scratch; keep only per-head quotas/entropy at decode.
    std::vector<double>().swap(votes_);
    next_ = prompt_;
    started_ = true;
  }

  /** @brief Mask padding between a head's prefix and the shared decode tail. */
  void maskDecode(nntrainer::Tensor &scores, unsigned int query_heads) const {
    if (!started_ || quotas_.empty() || query_heads % quotas_.size() != 0 ||
        scores.getDataType() != ml::train::TensorDim::DataType::FP32 ||
        scores.size() < static_cast<size_t>(span_) * query_heads)
      throw std::invalid_argument("EntroKV: invalid decode score geometry");
    float *p = scores.getData<float>();
    const size_t group = query_heads / quotas_.size();
    for (unsigned int h = 0; h < query_heads; ++h)
      for (unsigned int j = quotas_[h / group]; j < span_; ++j)
        p[static_cast<size_t>(j) * query_heads + h] =
          -std::numeric_limits<float>::infinity();
  }

  void finishDecode() { ++next_; }

  static size_t triangle(size_t n) { return n * (n + 1) / 2; }

private:
  template <typename T>
  void compact(nntrainer::Tensor &key, nntrainer::Tensor &value,
               const std::vector<std::vector<unsigned int>> &positions,
               unsigned int head_dim) {
    T *k = key.getData<T>();
    T *v = value.getData<T>();
    const size_t width = key.width();
    for (size_t h = 0; h < positions.size(); ++h) {
      for (size_t dst = 0; dst < positions[h].size(); ++dst) {
        const size_t src = positions[h][dst];
        if (src == dst)
          continue;
        // Sorted indices satisfy src >= dst; head slices never overlap.
        std::copy_n(k + src * width + h * head_dim, head_dim,
                    k + dst * width + h * head_dim);
        std::copy_n(v + src * width + h * head_dim, head_dim,
                    v + dst * width + h * head_dim);
      }
      for (size_t dst = positions[h].size(); dst < span_; ++dst) {
        std::fill_n(k + dst * width + h * head_dim, head_dim, T{});
        std::fill_n(v + dst * width + h * head_dim, head_dim, T{});
      }
    }
  }

  EntroKVConfig config_;
  unsigned int prompt_ = 0;
  unsigned int span_ = 0;
  unsigned int next_ = 0;
  unsigned int observed_ = 0;
  bool started_ = false;
  std::vector<unsigned int> quotas_;
  std::vector<double> entropy_;
  std::vector<double> votes_;
};

} // namespace causallm

#endif // __CAUSALLM_ENTROKV_H__
