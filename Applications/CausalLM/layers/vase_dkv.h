// SPDX-License-Identifier: Apache-2.0
/**
 * @file vase_dkv.h
 * @brief VaSE-DKV (cur_resample_gauss), arXiv:2606.03928v1.
 */
#ifndef __CAUSALLM_VASE_DKV_H__
#define __CAUSALLM_VASE_DKV_H__
#include "compact_kv.h"
#include <cpu_backend.h>
#include <iostream>
#include <random>

namespace causallm {
struct VaSEDKVConfig {
  unsigned int budget = 1024, buffer = 64, rank = 20, seed = 42;
  void validate() const {
    if (!buffer || !rank || rank > 1024 || budget <= buffer ||
        budget - buffer < 4 || budget % buffer ||
        budget > std::numeric_limits<unsigned int>::max() - buffer)
      throw std::invalid_argument(
        "VaSE-DKV requires budget divisible by buffer, budget-buffer >= 4, "
        "positive buffer and rank <= 1024");
  }
};

class VaSEDKVState {
public:
  VaSEDKVConfig config;
  CompactKVCache cache;
  bool active = false;
  unsigned int evictions = 0;
  std::mt19937 rng;
  bool enabled() const { return active; }

  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key.compare(0, 9, "vase_dkv_") != 0)
      return false;
    if (key == "vase_dkv_enabled")
      active = kvUnsigned(value) != 0;
    else if (key == "vase_dkv_budget")
      config.budget = kvUnsigned(value);
    else if (key == "vase_dkv_buffer")
      config.buffer = kvUnsigned(value);
    else if (key == "vase_dkv_rank")
      config.rank = kvUnsigned(value);
    else if (key == "vase_dkv_seed")
      config.seed = kvUnsigned(value);
    else if (key == "vase_dkv_layer")
      cache.layer = kvUnsigned(value);
    else if (key == "vase_dkv_prompt_length") {
      config.validate();
      cache.reset(kvUnsigned(value));
      evictions = 0;
      rng.seed(config.seed + cache.layer);
    } else
      throw std::invalid_argument("Unknown VaSE-DKV property: " + key);
    return true;
  }

  /** @brief Independent N(0,1/r) matrix, shared by all KV heads as upstream. */
  std::vector<float> projection(unsigned int dim) {
    std::vector<float> g(static_cast<size_t>(dim) * config.rank);
    const double scale = 1.0 / std::sqrt(static_cast<double>(config.rank));
    for (size_t i = 0; i < g.size(); i += 2) {
      const double u1 = (static_cast<double>(rng()) + .5) / 4294967296.0;
      const double u2 = (static_cast<double>(rng()) + .5) / 4294967296.0;
      const double radius = std::sqrt(-2 * std::log(u1)) * scale;
      const double angle = 6.2831853071795864769 * u2;
      g[i] = static_cast<float>(radius * std::cos(angle));
      if (i + 1 < g.size())
        g[i + 1] = static_cast<float>(radius * std::sin(angle));
    }
    return g;
  }

  /** @brief CurDKV product of projected squared K/V row norms. */
  static std::vector<double> scores(const std::vector<float> &k,
                                    const std::vector<float> &v,
                                    const std::vector<float> &g, unsigned int n,
                                    unsigned int d, unsigned int rank) {
    if (!n || !d || !rank || k.size() != static_cast<size_t>(n) * d ||
        v.size() != k.size() || g.size() != static_cast<size_t>(d) * rank)
      throw std::invalid_argument("Invalid VaSE-DKV projection geometry");
    kvFinite(k);
    kvFinite(v);
    kvFinite(g);
    std::vector<double> result(n);
    std::vector<float> kg(static_cast<size_t>(n) * rank), vg(kg.size());
    // Two matrix products use the same portable CPU BLAS backend as attention.
    nntrainer::sgemm(0, false, false, n, rank, d, 1.0f, k.data(), d, g.data(),
                     rank, 0.0f, kg.data(), rank);
    nntrainer::sgemm(0, false, false, n, rank, d, 1.0f, v.data(), d, g.data(),
                     rank, 0.0f, vg.data(), rank);
    for (unsigned int t = 0; t < n; ++t) {
      double knorm = 0, vnorm = 0;
      for (unsigned int r = 0; r < rank; ++r) {
        const double kp = kg[static_cast<size_t>(t) * rank + r];
        const double vp = vg[static_cast<size_t>(t) * rank + r];
        knorm += kp * kp;
        vnorm += vp * vp;
      }
      result[t] = knorm * vnorm;
    }
    const double sum = std::accumulate(result.begin(), result.end(), 0.0);
    if (!std::isfinite(sum))
      throw std::invalid_argument("Non-finite VaSE-DKV projection score");
    if (sum > 0)
      for (auto &s : result)
        s /= sum;
    // Exact sink protection; chronological compaction keeps these at rows 0..3.
    for (unsigned int t = 0; t < std::min(4u, n); ++t)
      result[t] = std::numeric_limits<double>::infinity();
    return result;
  }

  void afterAttention(nntrainer::Tensor &, unsigned int hq, unsigned int hk,
                      unsigned int d, unsigned int step, unsigned int max_new) {
    (void)hq;
    (void)max_new;
    cache.live += step;
    if (cache.next == 0) {
      const auto base = std::max(cache.prompt, config.budget);
      if (base > std::numeric_limits<unsigned int>::max() - config.buffer)
        throw std::out_of_range("VaSE-DKV capacity overflow");
      cache.reserve(hk, d, base + config.buffer);
      std::clog << "[VaSE-DKV layer=" << cache.layer
                << "] prompt=" << cache.prompt << " retained=" << cache.live
                << " budget=" << config.budget << " buffer=" << config.buffer
                << " allocated_kv_bytes=" << cache.bytes() << std::endl;
    } else if (cache.live > config.budget && cache.live % config.buffer == 0) {
      const auto candidates = cache.live - config.buffer;
      const auto historical = config.budget - config.buffer;
      const auto g = projection(d); // Resampled ON EVERY eviction, not cached.
      std::vector<std::vector<unsigned int>> ids(hk);
      for (unsigned int h = 0; h < hk; ++h) {
        std::vector<float> k, v;
        cache.head(h, hk, d, k, v);
        k.resize(static_cast<size_t>(candidates) * d);
        v.resize(k.size());
        ids[h] = kvTop(scores(k, v, g, candidates, d, config.rank), historical);
        for (unsigned int t = candidates; t < cache.live; ++t)
          ids[h].push_back(t);
      }
      cache.compact(ids, d, config.budget + config.buffer);
      ++evictions;
      std::clog << "[VaSE-DKV layer=" << cache.layer
                << "] eviction=" << evictions
                << " logical=" << cache.next + step
                << " retained=" << cache.live << " rank=" << config.rank
                << " allocated_kv_bytes=" << cache.bytes() << std::endl;
    }
    cache.next += step;
  }
};
} // namespace causallm
#endif
