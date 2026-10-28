// SPDX-License-Identifier: Apache-2.0
/**
 * @file dropkv.h
 * @brief DropKV CPU reference scorer, arXiv:2610.22130v1.
 */
#ifndef __CAUSALLM_DROPKV_H__
#define __CAUSALLM_DROPKV_H__
#include "compact_kv.h"
#include <cpu_backend.h>
#include <iostream>

namespace causallm {
struct DropKVConfig {
  double keep_ratio = 0.1;
  unsigned int window = 8, kernel = 11;
  std::string pooling = "max";
  void validate() const {
    if (!std::isfinite(keep_ratio) || keep_ratio <= 0 || keep_ratio > 1 ||
        !window || !kernel || !(kernel % 2) || kernel > 65535 ||
        (pooling != "max" && pooling != "avg"))
      throw std::invalid_argument("DropKV requires keep_ratio in (0,1], "
                                  "positive window, odd kernel <= 65535");
  }
};

class DropKVState {
public:
  DropKVConfig config;
  CompactKVCache cache;
  bool active = false;
  bool enabled() const { return active; }
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key.compare(0, 7, "dropkv_") != 0)
      return false;
    if (key == "dropkv_pooling")
      config.pooling = value;
    else if (key == "dropkv_enabled")
      active = kvUnsigned(value) != 0;
    else if (key == "dropkv_keep_ratio")
      config.keep_ratio = kvRatio(value);
    else if (key == "dropkv_window")
      config.window = kvUnsigned(value);
    else if (key == "dropkv_kernel")
      config.kernel = kvUnsigned(value);
    else if (key == "dropkv_layer")
      cache.layer = kvUnsigned(value);
    else if (key == "dropkv_prompt_length") {
      config.validate();
      cache.reset(kvUnsigned(value));
    } else
      throw std::invalid_argument("Unknown DropKV property: " + key);
    return true;
  }

  /**
   * @brief Sum squared single-token output perturbations, then mean over GQA.
   * @details Q is [group,window,dim], K/V [sequence,dim]. One attention row is
   * materialized at a time; values are converted once per KV head. The CPU
   * port uses FP32 Q and half K/V, without GPU/BF16 probability rounding.
   */
  static std::vector<double> scores(const std::vector<float> &q,
                                    const std::vector<float> &k,
                                    const std::vector<float> &v, unsigned int n,
                                    unsigned int d, unsigned int group,
                                    unsigned int window) {
    if (!n || !d || !group || !window || window > n ||
        k.size() != static_cast<size_t>(n) * d || v.size() != k.size() ||
        q.size() != static_cast<size_t>(group) * window * d)
      throw std::invalid_argument("Invalid DropKV score geometry");
    kvFinite(q);
    kvFinite(k);
    kvFinite(v);
    std::vector<double> result(n, 0.0), norm(n, 0.0);
    std::vector<float> p(n), output(d), dot(n);
    for (unsigned int t = 0; t < n; ++t)
      for (unsigned int j = 0; j < d; ++j) {
        const double a = v[static_cast<size_t>(t) * d + j];
        norm[t] += a * a;
      }
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));
    for (unsigned int h = 0; h < group; ++h)
      for (unsigned int t = 0; t < window; ++t) {
        const auto *query =
          q.data() + (static_cast<size_t>(h) * window + t) * d;
        const auto visible = n - window + t + 1; // Tail-aligned causal mask.
        nntrainer::sgemv(0, false, visible, d, static_cast<float>(scale),
                         k.data(), d, query, 1, 0.0f, p.data(), 1);
        double largest = -std::numeric_limits<double>::infinity();
        for (unsigned int i = 0; i < visible; ++i)
          largest = std::max(largest, static_cast<double>(p[i]));
        double sum = 0;
        for (unsigned int i = 0; i < visible; ++i) {
          p[i] = std::exp(p[i] - largest);
          sum += p[i];
        }
        if (!std::isfinite(sum) || sum <= 0)
          throw std::invalid_argument("Non-finite DropKV attention logits");
        for (unsigned int i = 0; i < visible; ++i)
          p[i] /= sum;
        nntrainer::sgemv(0, true, visible, d, 1.0f, v.data(), d, p.data(), 1,
                         0.0f, output.data(), 1);
        nntrainer::sgemv(0, false, visible, d, 1.0f, v.data(), d, output.data(),
                         1, 0.0f, dot.data(), 1);
        const double onorm =
          std::inner_product(output.begin(), output.end(), output.begin(), 0.0);
        for (unsigned int i = 0; i < visible; ++i) {
          const double distance = std::max(0.0, onorm + norm[i] - 2 * dot[i]);
          const double ratio = p[i] / (1.0 - p[i] + 1e-6);
          result[i] += ratio * ratio * distance / group;
        }
      }
    return result;
  }

  /** @brief Paper Algorithm 1 uses max; standalone code uses padded average. */
  static std::vector<double> pool(const std::vector<double> &s,
                                  unsigned int kernel,
                                  const std::string &kind = "max") {
    if (!kernel || !(kernel % 2) || kernel > 65535 ||
        (kind != "max" && kind != "avg"))
      throw std::invalid_argument("DropKV pooling kernel must be positive odd");
    std::vector<double> result(s.size());
    const auto radius = kernel / 2;
    for (size_t i = 0; i < s.size(); ++i) {
      const auto start = i > radius ? i - radius : 0;
      const auto end = std::min(s.size(), i + radius + 1);
      if (kind == "max")
        result[i] = -std::numeric_limits<double>::infinity();
      for (size_t j = start; j < end; ++j)
        if (kind == "max")
          result[i] = std::max(result[i], s[j]);
        else
          result[i] += s[j] / kernel;
    }
    return result;
  }

  void afterAttention(nntrainer::Tensor &query, unsigned int hq,
                      unsigned int hk, unsigned int d, unsigned int step,
                      unsigned int max_new) {
    cache.live += step;
    if (cache.next == 0) {
      const auto w = std::min(config.window, cache.prompt);
      const auto keep =
        std::min(cache.prompt,
                 std::max(w, static_cast<unsigned int>(
                               std::floor(config.keep_ratio * cache.prompt))));
      const auto group = hq / hk;
      std::vector<std::vector<unsigned int>> ids(hk);
      const auto *qp = query.getData<float>();
      for (unsigned int h = 0; h < hk; ++h) {
        std::vector<float> k, v, q(static_cast<size_t>(group) * w * d);
        cache.head(h, hk, d, k, v);
        for (unsigned int g = 0; g < group; ++g)
          for (unsigned int t = 0; t < w; ++t)
            std::copy_n(qp +
                          static_cast<size_t>(cache.prompt - w + t) * hq * d +
                          (h * group + g) * d,
                        d, q.data() + (static_cast<size_t>(g) * w + t) * d);
        auto s = pool(scores(q, k, v, cache.prompt, d, group, w), config.kernel,
                      config.pooling);
        for (unsigned int t = cache.prompt - w; t < cache.prompt; ++t)
          s[t] = std::numeric_limits<double>::infinity();
        ids[h] = kvTop(s, keep);
      }
      cache.compact(ids, d, keep + max_new);
      std::clog << "[DropKV layer=" << cache.layer
                << "] prompt=" << cache.prompt << " retained=" << cache.live
                << " window=" << w
                << " gqa=post_score_mean pool=" << config.pooling
                << " allocated_kv_bytes=" << cache.bytes() << std::endl;
    }
    cache.next += step;
  }
};
} // namespace causallm
#endif
