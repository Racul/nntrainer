// SPDX-License-Identifier: Apache-2.0
/**
 * @file manifoldkv.h
 * @brief ManifoldKV Algorithms 1/2, arXiv:2602.08343v1.
 */
#ifndef __CAUSALLM_MANIFOLDKV_H__
#define __CAUSALLM_MANIFOLDKV_H__
#include "compact_kv.h"
#include <iostream>

namespace causallm {
struct ManifoldKVConfig {
  double keep_ratio = 1.0;
  unsigned int centroid_window = 0;
  void validate() const {
    if (!std::isfinite(keep_ratio) || keep_ratio <= 0 || keep_ratio > 1)
      throw std::invalid_argument("ManifoldKV keep_ratio must be in (0, 1]");
  }
};

class ManifoldKVState {
public:
  ManifoldKVConfig config;
  CompactKVCache cache;
  bool active = false;
  bool enabled() const { return active; }

  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key.compare(0, 11, "manifoldkv_") != 0)
      return false;
    if (key == "manifoldkv_enabled")
      active = kvUnsigned(value) != 0;
    else if (key == "manifoldkv_keep_ratio")
      config.keep_ratio = kvRatio(value);
    else if (key == "manifoldkv_centroid_window")
      config.centroid_window = kvUnsigned(value);
    else if (key == "manifoldkv_layer")
      cache.layer = kvUnsigned(value);
    else if (key == "manifoldkv_prompt_length") {
      config.validate();
      cache.reset(kvUnsigned(value));
    } else
      throw std::invalid_argument("Unknown ManifoldKV property: " + key);
    return true;
  }

  /** @brief Raw Euclidean distance, without normalization or attention scores.
   */
  static std::vector<double> scores(const std::vector<float> &keys,
                                    unsigned int n, unsigned int d,
                                    unsigned int window) {
    if (!n || !d || keys.size() != static_cast<size_t>(n) * d)
      throw std::invalid_argument("Invalid ManifoldKV key geometry");
    kvFinite(keys);
    std::vector<double> result(n), mean(d);
    const auto width = window ? window : n;
    for (unsigned int start = 0; start < n;) {
      const auto end = start + std::min(width, n - start);
      std::fill(mean.begin(), mean.end(), 0.0);
      for (unsigned int t = start; t < end; ++t)
        for (unsigned int j = 0; j < d; ++j)
          mean[j] += keys[static_cast<size_t>(t) * d + j] / (end - start);
      for (unsigned int t = start; t < end; ++t) {
        double sum = 0;
        for (unsigned int j = 0; j < d; ++j) {
          const double delta = keys[static_cast<size_t>(t) * d + j] - mean[j];
          sum += delta * delta;
        }
        result[t] = std::sqrt(sum);
      }
      start = end;
    }
    return result;
  }

  void afterAttention(nntrainer::Tensor &, unsigned int hq, unsigned int hk,
                      unsigned int d, unsigned int step, unsigned int max_new) {
    (void)hq;
    cache.live += step;
    if (cache.next == 0) {
      const auto keep = std::max(1u, static_cast<unsigned int>(std::floor(
                                       config.keep_ratio * cache.live)));
      std::vector<std::vector<unsigned int>> ids(hk);
      for (unsigned int h = 0; h < hk; ++h) {
        std::vector<float> k, v;
        cache.head(h, hk, d, k, v);
        ids[h] = kvTop(scores(k, cache.live, d, config.centroid_window), keep);
      }
      cache.compact(ids, d, keep + max_new);
      std::clog << "[ManifoldKV layer=" << cache.layer
                << "] prompt=" << cache.prompt << " retained=" << cache.live
                << " allocated_kv_bytes=" << cache.bytes() << std::endl;
    }
    cache.next += step;
  }
};
} // namespace causallm
#endif
