// SPDX-License-Identifier: Apache-2.0
/**
 * @file capkv.h
 * @date 21 October 2026
 * @brief Capacity-aware prompt KV eviction (CapKV) for CPU causal attention
 * @note C++ adaptation of CapPress and CapacityLoggerMixin from the official
 *       CapKV repository (https://github.com/jiamingyy/CapKV, cf9c894,
 *       cap_press.py and record_mixin.py) for "Rethinking KV Cache Eviction
 *       via a Unified Information-Theoretic Objective" (arXiv 2604.25975).
 */
#ifndef __CAUSALLM_CAPKV_H__
#define __CAUSALLM_CAPKV_H__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <fp16.h>
#include <tensor.h>

namespace causallm {

/**
 * @brief Runtime settings. Exactly one of budget or ratio enables CapKV; both
 * zero leave the existing path unchanged.
 */
struct CapKVConfig {
  unsigned int budget =
    0;                   /**< kept prompt entries per KV head, sinks included */
  double ratio = 0.0;    /**< or compression ratio: keep int(L * (1 - ratio)) */
  double tau = 5.0;      /**< query-key alignment temperature (paper: 5) */
  unsigned int sink = 4; /**< leading tokens that are always kept */
  unsigned int future = 512;    /**< future positions averaged into the RoPE */
  bool record_capacity = false; /**< log K/U/KU capacity before and after */

  bool enabled() const { return budget != 0 || ratio != 0.0; }

  /** @brief Parse a nonnegative integer without truncation or wraparound. */
  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("CapKV expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("CapKV integer exceeds unsigned int");
    return static_cast<unsigned int>(number);
  }

  /** @brief Parse a finite number written entirely as one value. */
  static double parseNumber(const std::string &value) {
    std::size_t used = 0;
    double number = 0.0;
    try {
      number = std::stod(value, &used);
    } catch (const std::exception &) {
      used = 0;
    }
    if (value.empty() || used != value.size() || !std::isfinite(number))
      throw std::invalid_argument("CapKV expects a finite number");
    return number;
  }

  static bool parseBool(const std::string &value) {
    if (value == "true" || value == "1")
      return true;
    if (value == "false" || value == "0")
      return false;
    throw std::invalid_argument("CapKV expects true or false");
  }

  void validate() const {
    if (enabled() && ((budget != 0) == (ratio != 0.0) || ratio < 0.0 ||
                      ratio >= 1.0 || tau < 0.0 || !future))
      throw std::invalid_argument(
        "CapKV requires exactly one of budget > 0 or 0 < ratio < 1, "
        "tau >= 0 and future >= 1");
  }

  /** @brief Kept entries for a prompt, as KVPress ScorerPress::compress. */
  unsigned int keep(unsigned int length) const {
    const auto kept = budget ? budget
                             : static_cast<unsigned int>(
                                 static_cast<double>(length) * (1.0 - ratio));
    return std::min(kept, length);
  }
};

/** @brief Simplified capacity proxies of record_mixin.py, mean over KV heads.
 */
struct CapKVCapacity {
  double k = 0.0;  /**< K-capacity, log det of the key Gram */
  double u = 0.0;  /**< U-capacity, log det of the value Gram */
  double ku = 0.0; /**< KU-capacity of the channel U^T K / sqrt(T) */
};

/**
 * @brief Layer-local single-prompt compression after the full prefill.
 * @details Every KV head keeps the same number of prompt entries, so the
 * compacted cache is a contiguous [kept, Hkv * D] prefix of the existing
 * buffer. Absolute RoPE positions are unchanged; decode tokens are appended at
 * physical row kept + t, which the attention layer maps with removed().
 */
class CapKVState {
public:
  bool enabled() const { return config_.enabled(); }
  unsigned int removed() const { return removed_; }

  /** @brief Consume settings; capkv_prompt_length last starts a session. */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key.compare(0, 6, "capkv_") != 0)
      return false;
    if (key == "capkv_budget")
      config_.budget = CapKVConfig::parseUnsigned(value);
    else if (key == "capkv_ratio")
      config_.ratio = CapKVConfig::parseNumber(value);
    else if (key == "capkv_tau")
      config_.tau = CapKVConfig::parseNumber(value);
    else if (key == "capkv_sink")
      config_.sink = CapKVConfig::parseUnsigned(value);
    else if (key == "capkv_future")
      config_.future = CapKVConfig::parseUnsigned(value);
    else if (key == "capkv_record_capacity")
      config_.record_capacity = CapKVConfig::parseBool(value);
    else if (key == "capkv_layer")
      layer_ = CapKVConfig::parseUnsigned(value);
    else if (key == "capkv_prompt_length") {
      config_.validate();
      const auto length = CapKVConfig::parseUnsigned(value);
      if (enabled() && !length)
        throw std::invalid_argument("CapKV requires a nonempty prompt");
      prompt_ = length;
      removed_ = next_ = observed_rows_ = 0;
      query_sum_.clear();
    } else
      return false;
    return true;
  }

  /** @brief Validate a step before K/V are written or RoPE is applied. */
  void checkStep(unsigned int from, unsigned int step, unsigned int capacity,
                 unsigned int max_position) const {
    if (!enabled())
      return;
    if (!prompt_ || from != next_ || step != (next_ ? 1u : prompt_))
      throw std::invalid_argument("CapKV requires one full prefill followed "
                                  "by consecutive single tokens");
    if (from > max_position || step > max_position - from)
      throw std::out_of_range("CapKV step exceeds the RoPE position limit");
    const auto row = from - removed_;
    if (row > capacity || step > capacity - row)
      throw std::out_of_range("CapKV step exceeds the cache capacity");
  }

  /** @brief Record the logical position after a completed step. */
  void advance(unsigned int from, unsigned int step) {
    if (enabled())
      next_ = from + step;
  }

  /**
   * @brief Sum the pre-RoPE prefill queries of rows [sink, L) per head.
   * @note Call before attention rotates the query in place.
   */
  void observeQueries(const nntrainer::Tensor &query, unsigned int heads_q,
                      unsigned int head_dim) {
    if (!enabled())
      return;
    const size_t width = static_cast<size_t>(heads_q) * head_dim;
    if (query.getDataType() != nntrainer::Tdatatype::FP32 ||
        query.batch() != 1 || query.channel() != 1 ||
        query.height() != prompt_ || query.width() != width)
      throw std::invalid_argument("CapKV requires one FP32 prompt query");
    query_sum_.assign(width, 0.0);
    const float *data = query.getData<float>();
    for (size_t row = config_.sink; row < prompt_; ++row)
      for (size_t i = 0; i < width; ++i)
        query_sum_[i] += data[row * width + i];
    observed_rows_ = prompt_ > config_.sink ? prompt_ - config_.sink : 0;
  }

  /**
   * @brief Score, select and compact the prompt cache after prefill attention.
   * @param key, value full cache tensors [1, 1, capacity, Hkv * D]
   * @param rope_theta base of the default RoPE used by this layer
   */
  void finishPrefill(nntrainer::Tensor &key, nntrainer::Tensor &value,
                     unsigned int heads_q, unsigned int heads_kv,
                     unsigned int head_dim, float rope_theta) {
    if (!enabled())
      return;
    const auto start = std::chrono::steady_clock::now();
    const unsigned int length = prompt_;
    const unsigned int kept = config_.keep(length);
    if (kept >= length) {
      log(length, length, start);
      return;
    }
    if (kept <= config_.sink || observed_rows_ != length - config_.sink)
      throw std::invalid_argument(
        "CapKV must keep more than the sink tokens of an observed prompt");
    if (!heads_kv || !heads_q || heads_q % heads_kv || !head_dim ||
        head_dim % 2 || key.getDim() != value.getDim() || key.batch() != 1 ||
        key.channel() != 1 ||
        key.width() != static_cast<size_t>(heads_kv) * head_dim ||
        key.height() < length)
      throw std::invalid_argument("CapKV cache shape mismatch");

    const unsigned int group = heads_q / heads_kv;
    const unsigned int order =
      static_cast<unsigned int>(key.getDim().getStorageOrder());
    const auto anchor = anchors(query_sum_, observed_rows_, heads_q, head_dim,
                                rope_theta, length, config_.future);
    std::vector<std::vector<unsigned int>> selected(heads_kv);
    std::vector<float> keys, values, kept_keys, kept_values;
    CapKVCapacity pre, post;
    for (unsigned int head = 0; head < heads_kv; ++head) {
      loadHead(key, length, head, head_dim, keys);
      loadHead(value, length, head, head_dim, values);
      const auto scores = scoreKVHead(
        anchor.data() + static_cast<size_t>(head) * group * head_dim, group,
        keys.data(), values.data(), length, head_dim, config_.sink, config_.tau,
        order);
      selected[head] = select(scores, config_.sink, kept);
      if (config_.record_capacity) {
        accumulate(
          pre, capacity(keys.data(), values.data(), length, head_dim, order));
        gather(keys, selected[head], head_dim, kept_keys);
        gather(values, selected[head], head_dim, kept_values);
        accumulate(post, capacity(kept_keys.data(), kept_values.data(), kept,
                                  head_dim, order));
      }
    }
    compact(key, selected, head_dim);
    compact(value, selected, head_dim);
    removed_ = length - kept;
    log(length, kept, start);
    if (config_.record_capacity) {
      std::ostringstream line;
      line << std::setprecision(10) << "[CapKV capacity] layer=" << layer_
           << " pre_k=" << pre.k / heads_kv << " pre_u=" << pre.u / heads_kv
           << " pre_ku=" << pre.ku / heads_kv << " post_k=" << post.k / heads_kv
           << " post_u=" << post.u / heads_kv
           << " post_ku=" << post.ku / heads_kv;
      std::clog << line.str() << std::endl;
    }
  }

  /**
   * @brief CapPress._query_anchor for every query head: the mean pre-RoPE
   * query rotated by the RoPE averaged over positions [length, length+future).
   * @param sum per-head sum of observed pre-RoPE queries, [heads_q * D]
   */
  static std::vector<float> anchors(const std::vector<double> &sum,
                                    unsigned int rows, unsigned int heads_q,
                                    unsigned int head_dim, float rope_theta,
                                    unsigned int length, unsigned int future) {
    if (!rows || sum.size() != static_cast<size_t>(heads_q) * head_dim)
      throw std::invalid_argument("CapKV anchor needs observed queries");
    const unsigned int half = head_dim / 2;
    std::vector<double> mean_cos(half, 0.0), mean_sin(half, 0.0);
    for (unsigned int i = 0; i < half; ++i) {
      // Same inverse frequency as MHACoreLayer::_compute_default_parameters.
      const double inv_freq = static_cast<float>(
        1.0 / std::pow(rope_theta, (2 * i) / static_cast<float>(head_dim)));
      for (unsigned int p = 0; p < future; ++p) {
        const double angle = static_cast<double>(length + p) * inv_freq;
        mean_cos[i] += std::cos(angle);
        mean_sin[i] += std::sin(angle);
      }
      mean_cos[i] /= future;
      mean_sin[i] /= future;
    }
    // Rotate-half layout, as compute_rotary_emb_value: pairs (i, i + D/2).
    std::vector<float> out(sum.size());
    for (unsigned int head = 0; head < heads_q; ++head) {
      const double *m = sum.data() + static_cast<size_t>(head) * head_dim;
      float *o = out.data() + static_cast<size_t>(head) * head_dim;
      for (unsigned int i = 0; i < half; ++i) {
        const double a = m[i] / rows, b = m[i + half] / rows;
        o[i] = static_cast<float>(mean_cos[i] * a - mean_sin[i] * b);
        o[i + half] = static_cast<float>(mean_sin[i] * a + mean_cos[i] * b);
      }
    }
    return out;
  }

  /**
   * @brief CapPress.score for one KV head: mean over its query heads of
   * w_i * v_i^T A^{-1} v_i with A = I + sum_i (w_i + eps) v_i v_i^T and
   * w_i = exp(tau * (cos(mu, k_i) - max)).
   * @param anchors group query-head anchors, [group * D]
   * @param keys, values rows [0, length) of this KV head, [length * D]
   * @return scores of positions [sink, length)
   */
  static std::vector<float>
  scoreKVHead(const float *anchors, unsigned int group, const float *keys,
              const float *values, unsigned int length, unsigned int head_dim,
              unsigned int sink, double tau, unsigned int order) {
    constexpr double eps = 1e-6, norm_eps = 1e-12;
    const unsigned int n = length - sink, d = head_dim;
    const float *k = keys + static_cast<size_t>(sink) * d;
    const float *v = values + static_cast<size_t>(sink) * d;
    std::vector<double> key_norm(n), weight(n), total(n, 0.0);
    for (unsigned int i = 0; i < n; ++i)
      key_norm[i] = std::max(norm(k + static_cast<size_t>(i) * d, d), norm_eps);
    std::vector<float> scaled(static_cast<size_t>(n) * d), gram(d * d),
      inverse(d * d), solved(static_cast<size_t>(n) * d);
    for (unsigned int q = 0; q < group; ++q) {
      const float *mu = anchors + static_cast<size_t>(q) * d;
      const double mu_norm = std::max(norm(mu, d), norm_eps);
      double max_logit = -std::numeric_limits<double>::infinity();
      for (unsigned int i = 0; i < n; ++i) {
        double dot = 0.0;
        for (unsigned int c = 0; c < d; ++c)
          dot += static_cast<double>(mu[c]) * k[static_cast<size_t>(i) * d + c];
        const double s = std::clamp(dot / (mu_norm * key_norm[i]), -1.0, 1.0);
        weight[i] = tau * s;
        max_logit = std::max(max_logit, weight[i]);
      }
      for (unsigned int i = 0; i < n; ++i) {
        weight[i] = std::exp(weight[i] - max_logit);
        const auto root = static_cast<float>(std::sqrt(weight[i] + eps));
        for (unsigned int c = 0; c < d; ++c)
          scaled[static_cast<size_t>(i) * d + c] =
            v[static_cast<size_t>(i) * d + c] * root;
      }
      // A = I + scaled^T scaled; leverage = |L^{-1} v_i|^2 with A = L L^T.
      nntrainer::sgemm(order, true, false, d, d, n, 1.0f, scaled.data(), d,
                       scaled.data(), d, 0.0f, gram.data(), d);
      std::vector<double> a(gram.begin(), gram.end());
      for (unsigned int c = 0; c < d; ++c)
        a[static_cast<size_t>(c) * d + c] += 1.0;
      if (!cholesky(a, d)) {
        // FP32 rounding of a large, nearly singular Gram can outweigh the
        // identity; rebuild this matrix in double before giving up.
        crossGram(scaled.data(), scaled.data(), n, d, 1.0, a);
        for (unsigned int c = 0; c < d; ++c)
          a[static_cast<size_t>(c) * d + c] += 1.0;
        if (!cholesky(a, d))
          throw std::runtime_error("CapKV capacity matrix is not positive");
      }
      invertLower(a, d, inverse);
      nntrainer::sgemm(order, false, true, n, d, d, 1.0f, v, d, inverse.data(),
                       d, 0.0f, solved.data(), d);
      for (unsigned int i = 0; i < n; ++i) {
        const double lev = norm(solved.data() + static_cast<size_t>(i) * d, d);
        total[i] += weight[i] * lev * lev;
      }
    }
    std::vector<float> scores(n);
    for (unsigned int i = 0; i < n; ++i) {
      scores[i] = static_cast<float>(total[i] / group);
      if (!std::isfinite(scores[i]))
        throw std::runtime_error("CapKV produced a nonfinite score");
    }
    return scores;
  }

  /**
   * @brief Positions kept by one KV head in ascending order: all sinks, then
   * the top kept - sink scores, earlier positions first on ties.
   */
  static std::vector<unsigned int> select(const std::vector<float> &scores,
                                          unsigned int sink,
                                          unsigned int kept) {
    if (kept < sink || kept - sink > scores.size())
      throw std::invalid_argument("CapKV selection exceeds the prompt");
    std::vector<unsigned int> order(scores.size());
    std::iota(order.begin(), order.end(), 0u);
    const auto better = [&](unsigned int a, unsigned int b) {
      return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    };
    const auto extra = kept - sink;
    std::partial_sort(order.begin(), order.begin() + extra, order.end(),
                      better);
    std::vector<unsigned int> positions(sink);
    std::iota(positions.begin(), positions.end(), 0u);
    for (unsigned int i = 0; i < extra; ++i)
      positions.push_back(order[i] + sink);
    std::sort(positions.begin(), positions.end());
    return positions;
  }

  /**
   * @brief record_mixin capacity proxies of one KV head, rows [0, rows):
   * K and U: log det(I + (X^T X + (1 + eps) I)) (_logdet_I_plus_XtX), and
   * KU: log det(I + C C^T), C = U^T K / sqrt(T) (_logdet_total_capacity).
   * @note Diagnostic only; Gram matrices are accumulated in double because
   * real keys make X^T X large and nearly singular.
   */
  static CapKVCapacity capacity(const float *keys, const float *values,
                                unsigned int rows, unsigned int head_dim,
                                unsigned int) {
    constexpr double eps = 1e-6;
    const unsigned int d = head_dim;
    std::vector<double> gram, channel, product(static_cast<size_t>(d) * d);
    CapKVCapacity result;
    crossGram(keys, keys, rows, d, 1.0, gram);
    result.k = logdetShifted(gram, d, 2.0 + eps);
    crossGram(values, values, rows, d, 1.0, gram);
    result.u = logdetShifted(gram, d, 2.0 + eps);
    crossGram(values, keys, rows, d, 1.0 / std::sqrt(rows), channel);
    for (unsigned int r = 0; r < d; ++r)
      for (unsigned int c = 0; c < d; ++c) {
        double sum = 0.0;
        for (unsigned int k = 0; k < d; ++k)
          sum += channel[static_cast<size_t>(r) * d + k] *
                 channel[static_cast<size_t>(c) * d + k];
        product[static_cast<size_t>(r) * d + c] = sum;
      }
    result.ku = logdetShifted(product, d, 1.0);
    return result;
  }

  /**
   * @brief log det(shift * I + M) for symmetric PSD M, as _spd_logdet_mean:
   * symmetrize, then Cholesky with jitter eps, growing tenfold on failure.
   */
  static double logdetShifted(const std::vector<double> &matrix, unsigned int n,
                              double shift) {
    double jitter = 1e-6;
    for (int attempt = 0; attempt < 6; ++attempt, jitter *= 10.0) {
      std::vector<double> a(static_cast<size_t>(n) * n);
      for (unsigned int r = 0; r < n; ++r)
        for (unsigned int c = 0; c < n; ++c)
          a[static_cast<size_t>(r) * n + c] =
            0.5 * (matrix[static_cast<size_t>(r) * n + c] +
                   matrix[static_cast<size_t>(c) * n + r]) +
            (r == c ? shift + jitter : 0.0);
      if (cholesky(a, n)) {
        double logdet = 0.0;
        for (unsigned int i = 0; i < n; ++i)
          logdet += 2.0 * std::log(a[static_cast<size_t>(i) * n + i]);
        return logdet;
      }
    }
    throw std::runtime_error("CapKV capacity log-determinant failed");
  }

  /** @brief In-place lower Cholesky factor of a row-major SPD matrix. */
  static bool cholesky(std::vector<double> &a, unsigned int n) {
    for (unsigned int j = 0; j < n; ++j) {
      double diagonal = a[static_cast<size_t>(j) * n + j];
      for (unsigned int k = 0; k < j; ++k)
        diagonal -=
          a[static_cast<size_t>(j) * n + k] * a[static_cast<size_t>(j) * n + k];
      if (!(diagonal > 0.0) || !std::isfinite(diagonal))
        return false;
      const double root = std::sqrt(diagonal);
      a[static_cast<size_t>(j) * n + j] = root;
      for (unsigned int i = j + 1; i < n; ++i) {
        double value = a[static_cast<size_t>(i) * n + j];
        for (unsigned int k = 0; k < j; ++k)
          value -= a[static_cast<size_t>(i) * n + k] *
                   a[static_cast<size_t>(j) * n + k];
        a[static_cast<size_t>(i) * n + j] = value / root;
      }
      for (unsigned int i = 0; i < j; ++i)
        a[static_cast<size_t>(i) * n + j] = 0.0;
    }
    return true;
  }

private:
  /** @brief out = scale * A^T B in double for row-major A, B of [rows, d]. */
  static void crossGram(const float *a, const float *b, unsigned int rows,
                        unsigned int d, double scale,
                        std::vector<double> &out) {
    out.assign(static_cast<size_t>(d) * d, 0.0);
    for (unsigned int i = 0; i < rows; ++i) {
      const float *x = a + static_cast<size_t>(i) * d;
      const float *y = b + static_cast<size_t>(i) * d;
      for (unsigned int r = 0; r < d; ++r) {
        const double xr = x[r];
        double *row = out.data() + static_cast<size_t>(r) * d;
        for (unsigned int c = 0; c < d; ++c)
          row[c] += xr * y[c];
      }
    }
    for (auto &value : out)
      value *= scale;
  }

  static double norm(const float *x, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; ++i)
      sum += static_cast<double>(x[i]) * x[i];
    return std::sqrt(sum);
  }

  /** @brief Inverse of a lower-triangular Cholesky factor, row-major. */
  static void invertLower(const std::vector<double> &l, unsigned int n,
                          std::vector<float> &inverse) {
    std::vector<double> x(static_cast<size_t>(n) * n, 0.0);
    for (unsigned int c = 0; c < n; ++c) {
      x[static_cast<size_t>(c) * n + c] =
        1.0 / l[static_cast<size_t>(c) * n + c];
      for (unsigned int r = c + 1; r < n; ++r) {
        double value = 0.0;
        for (unsigned int k = c; k < r; ++k)
          value -= l[static_cast<size_t>(r) * n + k] *
                   x[static_cast<size_t>(k) * n + c];
        x[static_cast<size_t>(r) * n + c] =
          value / l[static_cast<size_t>(r) * n + r];
      }
    }
    inverse.assign(x.begin(), x.end());
  }

  /** @brief Rows [0, rows) of one KV head as FP32, from the cache storage. */
  static void loadHead(const nntrainer::Tensor &cache, unsigned int rows,
                       unsigned int head, unsigned int head_dim,
                       std::vector<float> &out) {
    const size_t width = cache.width();
    const size_t offset = static_cast<size_t>(head) * head_dim;
    out.resize(static_cast<size_t>(rows) * head_dim);
    const auto dtype = cache.getDataType();
    for (size_t r = 0; r < rows; ++r) {
      float *dst = out.data() + r * head_dim;
      const size_t base = r * width + offset;
      if (dtype == nntrainer::Tdatatype::FP32) {
        std::memcpy(dst, cache.getData<float>() + base,
                    head_dim * sizeof(float));
      } else if (dtype == nntrainer::Tdatatype::UINT16) {
        // FP16 bits stored when ENABLE_FP16 is off.
        const uint16_t *src = cache.getData<uint16_t>() + base;
        for (unsigned int c = 0; c < head_dim; ++c)
          dst[c] = nntrainer::compute_fp16_to_fp32(src[c]);
#ifdef ENABLE_FP16
      } else if (dtype == nntrainer::Tdatatype::FP16) {
        const _FP16 *src = cache.getData<_FP16>() + base;
        for (unsigned int c = 0; c < head_dim; ++c)
          dst[c] = static_cast<float>(src[c]);
#endif
      } else {
        throw std::invalid_argument("CapKV cache must be FP32 or 16-bit");
      }
    }
  }

  static void gather(const std::vector<float> &rows,
                     const std::vector<unsigned int> &positions,
                     unsigned int head_dim, std::vector<float> &out) {
    out.resize(positions.size() * head_dim);
    for (size_t i = 0; i < positions.size(); ++i)
      std::memcpy(out.data() + i * head_dim,
                  rows.data() + static_cast<size_t>(positions[i]) * head_dim,
                  head_dim * sizeof(float));
  }

  /**
   * @brief Move each KV head's kept rows to the front of the buffer. Positions
   * ascend, so every source row is read before it can be overwritten.
   */
  static void compact(nntrainer::Tensor &cache,
                      const std::vector<std::vector<unsigned int>> &selected,
                      unsigned int head_dim) {
    const size_t element = cache.getDataType() == nntrainer::Tdatatype::FP32
                             ? sizeof(float)
                             : sizeof(uint16_t);
    const size_t row_bytes = cache.width() * element;
    const size_t head_bytes = head_dim * element;
    auto *data = cache.getData<unsigned char>();
    for (size_t head = 0; head < selected.size(); ++head)
      for (size_t row = 0; row < selected[head].size(); ++row)
        if (selected[head][row] != row)
          std::memmove(data + row * row_bytes + head * head_bytes,
                       data + selected[head][row] * row_bytes +
                         head * head_bytes,
                       head_bytes);
  }

  static void accumulate(CapKVCapacity &sum, const CapKVCapacity &value) {
    sum.k += value.k;
    sum.u += value.u;
    sum.ku += value.ku;
  }

  void log(unsigned int length, unsigned int kept,
           std::chrono::steady_clock::time_point start) const {
    const auto ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
    std::clog << "[CapKV layer=" << layer_ << "] prompt=" << length
              << " kept=" << kept << " score_ms=" << ms << std::endl;
  }

  CapKVConfig config_;
  std::vector<double> query_sum_;
  unsigned int observed_rows_ = 0;
  unsigned int layer_ = 0;
  unsigned int prompt_ = 0;
  unsigned int removed_ = 0;
  unsigned int next_ = 0;
};

} // namespace causallm

#endif // __CAUSALLM_CAPKV_H__
