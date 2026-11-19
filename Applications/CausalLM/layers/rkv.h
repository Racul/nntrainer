// SPDX-License-Identifier: Apache-2.0
/**
 * @file rkv.h
 * @date 18 November 2026
 * @brief R-KV redundancy-aware KV cache compression for CPU causal attention
 * @note C++ implementation of "R-KV: Redundancy-aware KV Cache Compression
 *       for Reasoning Models" (Cai et al., NeurIPS 2025, arXiv 2505.24133)
 *       after its official HuggingFace code (github.com/Zefan-Cai/R-KV:
 *       rkv/compression/r1_kv.py, rkv/utils.py, rkv/modeling.py):
 *       decoding-time compression with a budget and a buffer (Sec. 3.1),
 *       the observation query window (Sec. 3.2), importance from the
 *       observation attention with GQA pooling (Eq. 1-4, App. A.2),
 *       redundancy from key cosine similarity with the recent-copy retention
 *       (Eq. 5-6) and the joint selection Z = lambda I - (1 - lambda) R
 *       (Eq. 7, Alg. 1). The redundancy equals cal_similarity() without the
 *       n x n matrix: column sums come from the summed unit keys and only the
 *       retained entries are searched, newest first.
 */
#ifndef __CAUSALLM_RKV_H__
#define __CAUSALLM_RKV_H__

#include <algorithm>
#include <chrono>
#include <climits>
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

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <fp16.h>
#include <tensor.h>
#include <thread_manager.h>

namespace causallm {

/** @brief Keys per transposed block of the similarity kernels. */
constexpr unsigned int kRKVLanes = 8;

/** @brief Where the recent-copy retention is applied (M4). */
enum class RKVRetainAxis {
  Code, /**< cal_similarity(): zero S[i, j*] before the column mean */
  Paper /**< Eq. 6 / Alg. 1: drop S[i, j*] from token i's own mean */
};

/** @brief Selection unit of the joint score (M5). */
enum class RKVHeadSelect {
  PerKVHead, /**< r1_kv.py: top-k per KV head */
  Shared     /**< Alg. 1: one top-k of the head-mean score for every head */
};

/** @brief GQA reduction of the observation logits (M3). */
enum class RKVGqaPool { Max, Mean };

/**
 * @brief Runtime settings of the "rkv" nntr_config object. Exactly one of
 * ratio or budget enables R-KV; ratio 0 keeps every row (page release only).
 */
struct RKVConfig {
  double ratio = -1.0;       /**< < 0: unset, 0: keep, else 1 - B / prompt */
  unsigned int budget = 0;   /**< 0: unset, else the fixed B_budget */
  unsigned int window = 8;   /**< alpha: observation queries and kept tail */
  unsigned int buffer = 128; /**< B_buffer: decode re-compression interval */
  unsigned int kernel = 7;   /**< importance max-pool size (odd) */
  RKVGqaPool gqa_pool = RKVGqaPool::Max;
  double threshold = 0.5;  /**< T: similar-token threshold */
  unsigned int retain = 1; /**< beta: recent similar tokens retained */
  RKVRetainAxis retain_axis = RKVRetainAxis::Code;
  double lambda = 0.1; /**< importance weight of Z */
  RKVHeadSelect heads = RKVHeadSelect::PerKVHead;
  bool decode_compress = true; /**< false: compress after the prefill only */

  bool enabled() const { return ratio >= 0.0 || budget != 0; }
  bool keepOnly() const { return ratio == 0.0; }

  /** @brief B for a prompt: the fixed budget, or max(window + 1, t (1 - r)).
   */
  unsigned int budgetFor(unsigned int prompt) const {
    if (budget)
      return budget;
    const auto kept =
      static_cast<unsigned int>(static_cast<double>(prompt) * (1.0 - ratio));
    return std::max(window + 1, kept);
  }

  static std::string stripQuotes(const std::string &text) {
    if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
      return text.substr(1, text.size() - 2);
    return text;
  }

  /** @brief Parse a finite number. */
  static double parseNumber(const std::string &text) {
    size_t used = 0;
    double number = 0.0;
    try {
      number = std::stod(text, &used);
    } catch (const std::exception &) {
      used = 0;
    }
    if (text.empty() || used != text.size() || !std::isfinite(number))
      throw std::invalid_argument("R-KV expects a number, got " + text);
    return number;
  }

  /** @brief Parse an unsigned int without truncation or wraparound. */
  static unsigned int parseUnsigned(const std::string &text) {
    if (text.empty() || text.size() > 10 ||
        text.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("R-KV expects an unsigned integer, got " +
                                  text);
    const unsigned long long value = std::stoull(text);
    if (value > UINT_MAX)
      throw std::invalid_argument("R-KV integer out of range: " + text);
    return static_cast<unsigned int>(value);
  }

  static bool parseBool(const std::string &text) {
    if (text == "true" || text == "1")
      return true;
    if (text == "false" || text == "0")
      return false;
    throw std::invalid_argument("R-KV expects a boolean, got " + text);
  }

  /** @brief Shortest text that parses back to the same double. */
  static std::string number(double value) {
    std::ostringstream text;
    text << std::setprecision(17) << value;
    const std::string exact = text.str();
    for (int digits = 1; digits < 17; ++digits) {
      std::ostringstream shorter;
      shorter << std::setprecision(digits) << value;
      if (std::stod(shorter.str()) == value)
        return shorter.str();
    }
    return exact;
  }

  static const char *name(RKVRetainAxis axis) {
    return axis == RKVRetainAxis::Code ? "code" : "paper";
  }
  static const char *name(RKVHeadSelect heads) {
    return heads == RKVHeadSelect::PerKVHead ? "per_kv_head" : "shared";
  }
  static const char *name(RKVGqaPool pool) {
    return pool == RKVGqaPool::Max ? "max" : "mean";
  }

  /** @brief Apply one setting given as its JSON key and textual value. */
  void set(const std::string &key, const std::string &raw) {
    const std::string value = stripQuotes(raw);
    if (key == "ratio") {
      ratio = parseNumber(value);
      if (!(ratio >= 0.0 && ratio < 1.0))
        throw std::invalid_argument("R-KV ratio must be in [0, 1), got " +
                                    value);
    } else if (key == "budget") {
      budget = parseUnsigned(value);
      if (!budget)
        throw std::invalid_argument("R-KV budget must be positive");
    } else if (key == "window") {
      window = parseUnsigned(value);
    } else if (key == "buffer") {
      buffer = parseUnsigned(value);
    } else if (key == "kernel") {
      kernel = parseUnsigned(value);
    } else if (key == "gqa_pool") {
      if (value == "max")
        gqa_pool = RKVGqaPool::Max;
      else if (value == "mean")
        gqa_pool = RKVGqaPool::Mean;
      else
        throw std::invalid_argument("R-KV gqa_pool must be max or mean");
    } else if (key == "threshold") {
      threshold = parseNumber(value);
    } else if (key == "retain") {
      retain = parseUnsigned(value);
    } else if (key == "retain_axis") {
      if (value == "code")
        retain_axis = RKVRetainAxis::Code;
      else if (value == "paper")
        retain_axis = RKVRetainAxis::Paper;
      else
        throw std::invalid_argument("R-KV retain_axis must be code or paper");
    } else if (key == "lambda") {
      lambda = parseNumber(value);
    } else if (key == "heads") {
      if (value == "per_kv_head")
        heads = RKVHeadSelect::PerKVHead;
      else if (value == "shared")
        heads = RKVHeadSelect::Shared;
      else
        throw std::invalid_argument("R-KV heads must be per_kv_head or shared");
    } else if (key == "decode_compress") {
      decode_compress = parseBool(value);
    } else {
      throw std::invalid_argument("Unknown R-KV setting " + key);
    }
  }

  /** @brief Reject settings outside the ranges of the paper and the code. */
  void validate() const {
    if (!enabled())
      throw std::invalid_argument("R-KV requires ratio or budget");
    if (ratio >= 0.0 && budget)
      throw std::invalid_argument("R-KV takes ratio or budget, not both");
    if (!window || !buffer)
      throw std::invalid_argument("R-KV window and buffer must be positive");
    if (budget && budget <= window)
      throw std::invalid_argument("R-KV budget must exceed the window");
    if (kernel % 2 == 0)
      throw std::invalid_argument("R-KV kernel must be odd");
    if (!retain)
      throw std::invalid_argument("R-KV retain must be positive");
    if (!(lambda >= 0.0 && lambda <= 1.0))
      throw std::invalid_argument("R-KV lambda must be in [0, 1]");
    if (!(threshold >= -1.0 && threshold <= 1.0))
      throw std::invalid_argument("R-KV threshold must be in [-1, 1]");
  }

  /** @brief One-line form for a layer property ("key:value;..."). */
  std::string serialize() const {
    std::ostringstream text;
    if (ratio >= 0.0)
      text << "ratio:" << number(ratio) << ';';
    if (budget)
      text << "budget:" << budget << ';';
    text << "window:" << window << ";buffer:" << buffer << ";kernel:" << kernel
         << ";gqa_pool:" << name(gqa_pool) << ";threshold:" << number(threshold)
         << ";retain:" << retain << ";retain_axis:" << name(retain_axis)
         << ";lambda:" << number(lambda) << ";heads:" << name(heads)
         << ";decode_compress:" << (decode_compress ? 1 : 0);
    return text.str();
  }

  static RKVConfig parse(const std::string &serialized) {
    RKVConfig config;
    std::stringstream stream(serialized);
    std::string item;
    while (std::getline(stream, item, ';')) {
      const auto colon = item.find(':');
      if (colon == std::string::npos)
        throw std::invalid_argument("Malformed R-KV setting " + item);
      config.set(item.substr(0, colon), item.substr(colon + 1));
    }
    config.validate();
    return config;
  }

  /** @brief Body of the "[R-KV] ..." run log line. */
  std::string describe(unsigned int prompt) const {
    std::ostringstream text;
    if (keepOnly()) {
      text << "keep prompt=" << prompt;
      return text.str();
    }
    if (ratio >= 0.0)
      text << "ratio=" << ratio << ' ';
    text << "budget=" << budgetFor(prompt) << " window=" << window
         << " buffer=" << buffer << " kernel=" << kernel << " lambda=" << lambda
         << " threshold=" << threshold << " retain=" << retain
         << " axis=" << name(retain_axis) << " heads=" << name(heads)
         << " pool=" << name(gqa_pool)
         << " decode=" << (decode_compress ? 1 : 0) << " prompt=" << prompt;
    return text.str();
  }
};

/* ------------------------------------------------------------------------ */
/* Pure helpers                                                             */
/* ------------------------------------------------------------------------ */

/**
 * @brief Read-only view of a [rows, width] cache or query buffer of FP32,
 * FP16 bits in UINT16 (ENABLE_FP16 off) or _FP16 storage.
 */
struct RKVView {
  const unsigned char *data = nullptr;
  nntrainer::Tdatatype dtype = nntrainer::Tdatatype::FP32;
  size_t width = 0; /**< elements per row */

  static RKVView of(const nntrainer::Tensor &tensor) {
    RKVView view;
    view.data = tensor.getData<unsigned char>();
    view.dtype = tensor.getDataType();
    view.width = tensor.width();
    if (view.dtype != nntrainer::Tdatatype::FP32 && !view.half())
      throw std::invalid_argument("R-KV needs FP32 or 16-bit tensors");
    return view;
  }

  bool half() const {
    return dtype == nntrainer::Tdatatype::UINT16
#ifdef ENABLE_FP16
           || dtype == nntrainer::Tdatatype::FP16
#endif
      ;
  }

  size_t elementSize() const { return half() ? 2 : 4; }

  float at(size_t row, size_t column) const {
    const size_t index = row * width + column;
    if (!half())
      return reinterpret_cast<const float *>(data)[index];
    return nntrainer::compute_fp16_to_fp32(
      reinterpret_cast<const uint16_t *>(data)[index]);
  }
};

/**
 * @brief out[l] = sum_d x[d] * block[d * 8 + l]: dot products of x with the
 * 8 keys of one transposed [dim][8] block.
 */
inline void rkvBlockDot(const float *x, const float *block, unsigned int dim,
                        float *out) {
#if defined(__AVX2__)
  __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                   _mm256_setzero_ps(), _mm256_setzero_ps()};
  unsigned int d = 0;
  for (; d + 4 <= dim; d += 4)
    for (unsigned int u = 0; u < 4; ++u) {
      const __m256 xv = _mm256_set1_ps(x[d + u]);
      const __m256 kv = _mm256_loadu_ps(block + (d + u) * kRKVLanes);
#if defined(__FMA__)
      acc[u] = _mm256_fmadd_ps(xv, kv, acc[u]);
#else
      acc[u] = _mm256_add_ps(acc[u], _mm256_mul_ps(xv, kv));
#endif
    }
  for (; d < dim; ++d)
    acc[0] = _mm256_add_ps(
      acc[0], _mm256_mul_ps(_mm256_set1_ps(x[d]),
                            _mm256_loadu_ps(block + d * kRKVLanes)));
  _mm256_storeu_ps(out, _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]),
                                      _mm256_add_ps(acc[2], acc[3])));
#elif defined(__ARM_NEON)
  float32x4_t lo[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
  float32x4_t hi[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
  unsigned int d = 0;
  for (; d + 2 <= dim; d += 2)
    for (unsigned int u = 0; u < 2; ++u) {
      const float *row = block + (d + u) * kRKVLanes;
#if defined(__aarch64__)
      lo[u] = vfmaq_n_f32(lo[u], vld1q_f32(row), x[d + u]);
      hi[u] = vfmaq_n_f32(hi[u], vld1q_f32(row + 4), x[d + u]);
#else
      lo[u] = vmlaq_n_f32(lo[u], vld1q_f32(row), x[d + u]);
      hi[u] = vmlaq_n_f32(hi[u], vld1q_f32(row + 4), x[d + u]);
#endif
    }
  for (; d < dim; ++d) {
    const float *row = block + d * kRKVLanes;
    lo[0] = vaddq_f32(lo[0], vmulq_n_f32(vld1q_f32(row), x[d]));
    hi[0] = vaddq_f32(hi[0], vmulq_n_f32(vld1q_f32(row + 4), x[d]));
  }
  vst1q_f32(out, vaddq_f32(lo[0], lo[1]));
  vst1q_f32(out + 4, vaddq_f32(hi[0], hi[1]));
#else
  float acc[kRKVLanes] = {0.0f};
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int l = 0; l < kRKVLanes; ++l)
      acc[l] += x[d] * block[d * kRKVLanes + l];
  std::memcpy(out, acc, sizeof(acc));
#endif
}

/**
 * @brief Keys of every KV head in FP32, transposed in blocks of 8 rows:
 * [heads][ceil(rows / 8)][dim][8], zero padded. One copy serves the
 * importance (raw keys) and then the redundancy (normalized in place).
 */
struct RKVPacked {
  unsigned int rows = 0;
  unsigned int heads = 0;
  unsigned int dim = 0;
  unsigned int blocks = 0;
  std::vector<float> data;

  size_t headSize() const {
    return static_cast<size_t>(blocks) * dim * kRKVLanes;
  }
  float *head(unsigned int g) { return data.data() + g * headSize(); }
  const float *head(unsigned int g) const {
    return data.data() + g * headSize();
  }
  float at(unsigned int g, unsigned int row, unsigned int d) const {
    return head(
      g)[(static_cast<size_t>(row / kRKVLanes) * dim + d) * kRKVLanes +
         row % kRKVLanes];
  }

  /** @brief Pack rows [0, n) of a [rows, heads_kv * head_dim] cache. */
  void pack(const RKVView &key, unsigned int n, unsigned int heads_kv,
            unsigned int head_dim) {
    if (key.width != static_cast<size_t>(heads_kv) * head_dim)
      throw std::invalid_argument("R-KV key width mismatch");
    rows = n;
    heads = heads_kv;
    dim = head_dim;
    blocks = (n + kRKVLanes - 1) / kRKVLanes;
    data.assign(heads * headSize(), 0.0f);
    nntrainer::ThreadManager::Global().parallel_for(0, heads, [&](size_t g) {
      float *dst = head(static_cast<unsigned int>(g));
      for (unsigned int r = 0; r < n; ++r) {
        float *col = dst +
                     static_cast<size_t>(r / kRKVLanes) * dim * kRKVLanes +
                     r % kRKVLanes;
        for (unsigned int d = 0; d < dim; ++d)
          col[d * kRKVLanes] = key.at(r, g * dim + d);
      }
    });
  }
};

/**
 * @brief F.max_pool1d(x, kernel, stride=1, padding=kernel // 2) of one row:
 * the padding never wins (it is -inf).
 */
inline void rkvMaxPool1d(const float *in, unsigned int n, unsigned int kernel,
                         float *out) {
  const unsigned int half = kernel / 2;
  for (unsigned int c = 0; c < n; ++c) {
    const unsigned int begin = c > half ? c - half : 0;
    const unsigned int end = std::min(n, c + half + 1);
    float top = in[begin];
    for (unsigned int i = begin + 1; i < end; ++i)
      top = std::max(top, in[i]);
    out[c] = top;
  }
}

/**
 * @brief M3 on packed raw keys. For every KV head: the logits of the a
 * observation queries over the cand candidate keys (/ sqrt(D)), pooled over
 * the GQA group, softmax over the candidates, mean over the observations,
 * then the 1D max-pool (r1_kv.py with compute_attention_scores()).
 * @param obs_q rotated observation queries [a][heads_q * dim], FP32
 * @param out [heads * cand]
 */
inline void rkvImportancePacked(const float *obs_q, unsigned int a,
                                unsigned int heads_q, const RKVPacked &key,
                                unsigned int cand, RKVGqaPool pool,
                                unsigned int kernel, std::vector<float> &out) {
  const unsigned int heads_kv = key.heads;
  const unsigned int dim = key.dim;
  if (!a || !heads_kv || heads_q % heads_kv || cand > key.rows ||
      kernel % 2 == 0)
    throw std::invalid_argument("R-KV importance shape mismatch");
  out.assign(static_cast<size_t>(heads_kv) * cand, 0.0f);
  if (!cand)
    return;
  const unsigned int gqa = heads_q / heads_kv;
  const size_t q_width = static_cast<size_t>(heads_q) * dim;
  const unsigned int blocks = (cand + kRKVLanes - 1) / kRKVLanes;
  const float root = std::sqrt(static_cast<float>(dim));

  nntrainer::ThreadManager::Global().parallel_for(0, heads_kv, [&](size_t g) {
    const float *kt = key.head(static_cast<unsigned int>(g));
    std::vector<float> pooled(cand);
    std::vector<double> mean(cand, 0.0);
    float lane[kRKVLanes];
    for (unsigned int s = 0; s < a; ++s) {
      for (unsigned int r = 0; r < gqa; ++r) {
        const float *x = obs_q + s * q_width + (g * gqa + r) * dim;
        for (unsigned int b = 0; b < blocks; ++b) {
          rkvBlockDot(x, kt + static_cast<size_t>(b) * dim * kRKVLanes, dim,
                      lane);
          for (unsigned int l = 0; l < kRKVLanes; ++l) {
            const unsigned int c = b * kRKVLanes + l;
            if (c >= cand)
              break;
            const float v = lane[l] / root;
            if (r == 0)
              pooled[c] = v;
            else if (pool == RKVGqaPool::Max)
              pooled[c] = std::max(pooled[c], v);
            else
              pooled[c] += v;
          }
        }
      }
      if (pool == RKVGqaPool::Mean)
        for (auto &v : pooled)
          v /= static_cast<float>(gqa);
      const float top = *std::max_element(pooled.begin(), pooled.end());
      double total = 0.0;
      for (auto &v : pooled) {
        v = std::exp(v - top);
        total += v;
      }
      for (unsigned int c = 0; c < cand; ++c)
        mean[c] += pooled[c] / total;
    }
    for (unsigned int c = 0; c < cand; ++c)
      pooled[c] = static_cast<float>(mean[c] / a);
    rkvMaxPool1d(pooled.data(), cand, kernel, out.data() + g * cand);
  });
}

/** @brief M3 on a cache view: packs rows [0, cand) and scores them. */
inline void rkvImportance(const float *obs_q, unsigned int a,
                          unsigned int heads_q, const RKVView &key,
                          unsigned int cand, unsigned int heads_kv,
                          unsigned int head_dim, RKVGqaPool pool,
                          unsigned int kernel, std::vector<float> &out) {
  RKVPacked packed;
  packed.pack(key, cand, heads_kv, head_dim);
  rkvImportancePacked(obs_q, a, heads_q, packed, cand, pool, kernel, out);
}

/**
 * @brief M4 on packed keys, which are normalized in place. Equals
 * cal_similarity(threshold, retain_direction="last") for beta = 1 on the
 * code axis: k / (|k| + 1e-8), S = K K^T with a zero diagonal, S[i, j*] = 0
 * for the largest j* with S[i, j*] > T (j* = 0 when there is none), then the
 * softmax over j of the column means. Column sums come from k_j . sum_i k_i;
 * each row searches its newest similar keys and stops after beta of them.
 * @param out [heads * rows]
 */
inline void rkvRedundancyPacked(RKVPacked &key, float threshold,
                                unsigned int beta, RKVRetainAxis axis,
                                std::vector<float> &out) {
  const unsigned int n = key.rows;
  const unsigned int heads = key.heads;
  const unsigned int dim = key.dim;
  if (!beta)
    throw std::invalid_argument("R-KV retain must be positive");
  out.assign(static_cast<size_t>(heads) * n, 0.0f);
  if (!n)
    return;
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t cells = static_cast<size_t>(heads) * n;
  std::vector<double> col(cells);

  // Unit keys in place and the uncorrected column sums sum_{i != j} S_ij.
  tm.parallel_for(0, heads, [&](size_t g) {
    float *kt = key.head(static_cast<unsigned int>(g));
    std::vector<double> sum(dim, 0.0);
    for (unsigned int j = 0; j < n; ++j) {
      float *k = kt + static_cast<size_t>(j / kRKVLanes) * dim * kRKVLanes +
                 j % kRKVLanes;
      double squares = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
        squares += static_cast<double>(k[d * kRKVLanes]) * k[d * kRKVLanes];
      const float den = static_cast<float>(std::sqrt(squares)) + 1e-8f;
      for (unsigned int d = 0; d < dim; ++d) {
        k[d * kRKVLanes] /= den;
        sum[d] += k[d * kRKVLanes];
      }
    }
    for (unsigned int j = 0; j < n; ++j) {
      const float *k = kt +
                       static_cast<size_t>(j / kRKVLanes) * dim * kRKVLanes +
                       j % kRKVLanes;
      double dot = 0.0;
      double self = 0.0;
      for (unsigned int d = 0; d < dim; ++d) {
        const double v = k[d * kRKVLanes];
        dot += v * sum[d];
        self += v * v;
      }
      col[g * n + j] = dot - self;
    }
  });

  // Newest similar keys of every row, searched block by block from the end.
  const unsigned int threads = tm.getComputeThreadCount();
  const unsigned int chunks =
    std::max(1u, std::min(n, (4u * threads + heads - 1) / heads));
  std::vector<unsigned int> hit_j(cells * beta);
  std::vector<float> hit_s(cells * beta);
  std::vector<unsigned int> hits(cells);
  std::vector<float> first(cells); // S_i0, read when a row has no hit
  tm.parallel_for(0, static_cast<size_t>(heads) * chunks, [&](size_t u) {
    const unsigned int g = static_cast<unsigned int>(u / chunks);
    const unsigned int start = static_cast<unsigned int>(u % chunks);
    const float *kt = key.head(g);
    std::vector<float> x(dim);
    float lane[kRKVLanes];
    for (unsigned int i = start; i < n; i += chunks) {
      const float *ki = kt +
                        static_cast<size_t>(i / kRKVLanes) * dim * kRKVLanes +
                        i % kRKVLanes;
      for (unsigned int d = 0; d < dim; ++d)
        x[d] = ki[d * kRKVLanes];
      const size_t cell = static_cast<size_t>(g) * n + i;
      unsigned int found = 0;
      float s0 = 0.0f;
      for (unsigned int b = key.blocks; b-- > 0 && found < beta;) {
        rkvBlockDot(x.data(), kt + static_cast<size_t>(b) * dim * kRKVLanes,
                    dim, lane);
        for (unsigned int l = kRKVLanes; l-- > 0;) {
          const unsigned int j = b * kRKVLanes + l;
          if (j >= n)
            continue;
          const float sim = j == i ? 0.0f : lane[l];
          if (j == 0)
            s0 = sim;
          if (sim > threshold) {
            hit_j[cell * beta + found] = j;
            hit_s[cell * beta + found] = sim;
            if (++found == beta)
              break;
          }
        }
      }
      hits[cell] = found;
      first[cell] = s0;
    }
  });

  // Retention, column means and the softmax over all rows.
  tm.parallel_for(0, heads, [&](size_t g) {
    double *c = col.data() + g * n;
    for (unsigned int i = 0; i < n; ++i) {
      const size_t cell = g * n + i;
      if (axis == RKVRetainAxis::Code) {
        if (!hits[cell])
          c[0] -= first[cell];
        for (unsigned int b = 0; b < hits[cell]; ++b)
          c[hit_j[cell * beta + b]] -= hit_s[cell * beta + b];
      } else {
        for (unsigned int b = 0; b < hits[cell]; ++b)
          c[i] -= hit_s[cell * beta + b];
      }
    }
    double top = -std::numeric_limits<double>::infinity();
    for (unsigned int j = 0; j < n; ++j)
      top = std::max(top, c[j] / n);
    double total = 0.0;
    for (unsigned int j = 0; j < n; ++j) {
      c[j] = std::exp(c[j] / n - top);
      total += c[j];
    }
    for (unsigned int j = 0; j < n; ++j)
      out[g * n + j] = static_cast<float>(c[j] / total);
  });
}

/** @brief M4 on a cache view: packs rows [0, n) and scores them. */
inline void rkvRedundancy(const RKVView &key, unsigned int n,
                          unsigned int heads_kv, unsigned int head_dim,
                          float threshold, unsigned int beta,
                          RKVRetainAxis axis, std::vector<float> &out) {
  RKVPacked packed;
  packed.pack(key, n, heads_kv, head_dim);
  rkvRedundancyPacked(packed, threshold, beta, axis, out);
}

/**
 * @brief M5. Z = lambda I - (1 - lambda) R over the candidates [0, n -
 * window); each head keeps its top (budget - window) candidates (ties: the
 * smaller index) and the last window rows, in ascending order. Shared picks
 * one set from the head-mean Z (Alg. 1).
 * @param importance [heads_kv * (n - window)], unused when lambda == 0
 * @param redundancy [heads_kv * n], unused when lambda == 1
 */
inline std::vector<std::vector<unsigned int>>
rkvSelect(const std::vector<float> &importance,
          const std::vector<float> &redundancy, unsigned int n,
          unsigned int window, unsigned int heads_kv, unsigned int budget,
          double lambda, RKVHeadSelect heads) {
  std::vector<std::vector<unsigned int>> keep(heads_kv);
  if (n <= budget || window >= budget) {
    for (auto &rows : keep) {
      rows.resize(n);
      std::iota(rows.begin(), rows.end(), 0u);
    }
    return keep;
  }
  const unsigned int cand = n - window;
  const unsigned int pick = budget - window;
  const bool use_i = lambda > 0.0;
  const bool use_r = lambda < 1.0;
  if ((use_i && importance.size() != static_cast<size_t>(heads_kv) * cand) ||
      (use_r && redundancy.size() != static_cast<size_t>(heads_kv) * n))
    throw std::invalid_argument("R-KV score size mismatch");
  const float wi = static_cast<float>(lambda);
  const float wr = static_cast<float>(1.0 - lambda);
  auto score = [&](unsigned int g, unsigned int c) {
    float z = 0.0f;
    if (use_i)
      z += importance[static_cast<size_t>(g) * cand + c] * wi;
    if (use_r)
      z -= redundancy[static_cast<size_t>(g) * n + c] * wr;
    return z;
  };
  auto top = [&](const std::vector<float> &z) {
    std::vector<unsigned int> order(cand);
    std::iota(order.begin(), order.end(), 0u);
    std::nth_element(order.begin(), order.begin() + pick, order.end(),
                     [&](unsigned int x, unsigned int y) {
                       return z[x] > z[y] || (z[x] == z[y] && x < y);
                     });
    order.resize(pick);
    std::sort(order.begin(), order.end());
    for (unsigned int r = cand; r < n; ++r)
      order.push_back(r);
    return order;
  };
  std::vector<float> z(cand);
  if (heads == RKVHeadSelect::Shared) {
    for (unsigned int c = 0; c < cand; ++c) {
      double sum = 0.0;
      for (unsigned int g = 0; g < heads_kv; ++g)
        sum += score(g, c);
      z[c] = static_cast<float>(sum / heads_kv);
    }
    const auto rows = top(z);
    for (auto &kept : keep)
      kept = rows;
  } else {
    for (unsigned int g = 0; g < heads_kv; ++g) {
      for (unsigned int c = 0; c < cand; ++c)
        z[c] = score(g, c);
      keep[g] = top(z);
    }
  }
  return keep;
}

/**
 * @brief Move each KV head's kept rows of a [rows, Hkv * head_bytes] buffer
 * to the front: row r of head g becomes old row keep[g][r]. Kept rows ascend,
 * so every source row is read before it can be overwritten.
 */
inline void rkvCompact(unsigned char *data, size_t row_bytes, size_t head_bytes,
                       const std::vector<std::vector<unsigned int>> &keep) {
  for (size_t g = 0; g < keep.size(); ++g) {
    const auto &rows = keep[g];
    for (size_t r = 0; r < rows.size(); ++r) {
      if (r && rows[r] <= rows[r - 1])
        throw std::invalid_argument("R-KV kept rows must ascend");
      if (rows[r] != r)
        std::memmove(data + r * row_bytes + g * head_bytes,
                     data + rows[r] * row_bytes + g * head_bytes, head_bytes);
    }
  }
}

/**
 * @brief Return the physical pages of rows [first_row, rows) of a contiguous
 * [rows, row_bytes] buffer to the system. The rows read as zero on Linux and
 * are faulted in again when written. Only whole pages are released.
 * @note No-op on Windows and in CUDA builds, whose KV may be managed memory.
 * @return released bytes
 */
inline size_t rkvReleaseRows(void *base, size_t row_bytes, size_t first_row,
                             size_t rows) {
#if defined(_WIN32) || (defined(ENABLE_CUDA) && ENABLE_CUDA == 1)
  (void)base;
  (void)row_bytes;
  (void)first_row;
  (void)rows;
  return 0;
#else
  const long page = ::sysconf(_SC_PAGESIZE);
  if (page <= 0 || base == nullptr || first_row >= rows)
    return 0;
  const uintptr_t mask = static_cast<uintptr_t>(page) - 1;
  const auto start = reinterpret_cast<uintptr_t>(base);
  const uintptr_t begin = (start + first_row * row_bytes + mask) & ~mask;
  const uintptr_t end = (start + rows * row_bytes) & ~mask;
  if (end <= begin || ::madvise(reinterpret_cast<void *>(begin), end - begin,
                                MADV_DONTNEED) != 0)
    return 0;
  return end - begin;
#endif
}

/** @brief rkvReleaseRows() on a [1, 1, rows, width] cache tensor. */
inline size_t rkvReleaseRows(nntrainer::Tensor &cache, size_t first_row) {
  if (cache.batch() != 1 || cache.channel() != 1 || cache.height() == 0)
    return 0;
  return rkvReleaseRows(cache.getData<unsigned char>(),
                        cache.bytes() / cache.height(), first_row,
                        cache.height());
}

/* ------------------------------------------------------------------------ */
/* Per-layer state                                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Layer-local R-KV state (M1, M2). After each step's attention it
 * records the rotated queries of the observation window and, after the
 * prefill and at every decode position divisible by the buffer, compresses
 * the cache to the budget: each KV head's kept rows move to the front in
 * position order and the freed rows are released. Physical row of an
 * absolute position u is u - removed(); RoPE keeps the absolute position.
 */
class RKVState {
public:
  bool enabled() const { return config_.enabled(); }
  const RKVConfig &config() const { return config_; }

  /**
   * @brief Consume "rkv" (serialized config) and "rkv_run"
   * ("layer,prompt"); both reset the per-run state.
   */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key == "rkv") {
      config_ = RKVConfig::parse(value);
    } else if (key == "rkv_run") {
      const auto comma = value.find(',');
      if (comma == std::string::npos)
        throw std::invalid_argument("rkv_run needs layer,prompt");
      layer_ = RKVConfig::parseUnsigned(value.substr(0, comma));
      prompt_ = RKVConfig::parseUnsigned(value.substr(comma + 1));
      if (!prompt_)
        throw std::invalid_argument("Invalid rkv_run " + value);
    } else {
      return false;
    }
    resetRun();
    return true;
  }

  /** @brief Validate a step before K/V are written or RoPE is applied and
   * record the next absolute position. */
  void checkStep(unsigned int from, unsigned int step, size_t capacity,
                 unsigned int max_position) {
    if (!enabled())
      return;
    if (!prompt_ || from != next_ || step != (next_ ? 1u : prompt_))
      throw std::invalid_argument("R-KV requires one full prefill followed "
                                  "by consecutive single tokens");
    if (from > max_position || step > max_position - from)
      throw std::out_of_range("R-KV step exceeds the RoPE position limit");
    const size_t row = from - removed_;
    if (row > capacity || step > capacity - row)
      throw std::out_of_range("R-KV step exceeds the cache capacity");
    next_ = from + step;
  }

  /**
   * @brief After a step's attention (which read the uncompressed cache):
   * update the observation queries, then compress when due.
   * @param rotated_q this step's query after RoPE [1, 1, step, Hq * D]
   * @param key, value full cache tensors [1, 1, capacity, Hkv * D]
   * @param from absolute position of the step's first token
   */
  void afterAttention(const nntrainer::Tensor &rotated_q,
                      nntrainer::Tensor &key, nntrainer::Tensor &value,
                      unsigned int heads_q, unsigned int heads_kv,
                      unsigned int head_dim, unsigned int from,
                      unsigned int step);

  size_t removed() const { return removed_; }
  unsigned int rows() const { return rows_; }
  unsigned int budget() const { return budget_; }
  unsigned int events() const { return events_; }
  unsigned int layer() const { return layer_; }

  /** @brief Absolute position of every physical row, per KV head. */
  const std::vector<std::vector<unsigned int>> &positions() const {
    return pos_;
  }

  /** @brief Observation queries [rows][Hq * D], oldest first. */
  std::vector<float> observations() const {
    const size_t width = static_cast<size_t>(heads_q_) * head_dim_;
    std::vector<float> out(ring_rows_ * width);
    const unsigned int window = config_.window;
    const unsigned int oldest = (ring_head_ + window - ring_rows_) % window;
    for (unsigned int r = 0; r < ring_rows_; ++r)
      std::memcpy(out.data() + r * width,
                  ring_.data() + ((oldest + r) % window) * width,
                  width * sizeof(float));
    return out;
  }

private:
  void resetRun() {
    next_ = 0;
    removed_ = 0;
    rows_ = 0;
    events_ = 0;
    budget_ = enabled() && prompt_ ? config_.budgetFor(prompt_) : 0;
    heads_q_ = heads_kv_ = head_dim_ = 0;
    ring_.clear();
    ring_rows_ = ring_head_ = 0;
    pos_.clear();
  }

  void pushQueries(const nntrainer::Tensor &rotated_q, unsigned int step);
  void compress(nntrainer::Tensor &key, nntrainer::Tensor &value,
                unsigned int n, unsigned int pos, bool prefill);

  RKVConfig config_;
  unsigned int layer_ = 0;
  unsigned int prompt_ = 0;
  unsigned int budget_ = 0;
  unsigned int next_ = 0;
  unsigned int rows_ = 0;
  unsigned int events_ = 0;
  size_t removed_ = 0;
  unsigned int heads_q_ = 0;
  unsigned int heads_kv_ = 0;
  unsigned int head_dim_ = 0;
  std::vector<float> ring_; /**< [window][Hq * D], circular */
  unsigned int ring_rows_ = 0;
  unsigned int ring_head_ = 0;                 /**< next slot */
  std::vector<std::vector<unsigned int>> pos_; /**< [Hkv][rows] */
};

inline void RKVState::afterAttention(
  const nntrainer::Tensor &rotated_q, nntrainer::Tensor &key,
  nntrainer::Tensor &value, unsigned int heads_q, unsigned int heads_kv,
  unsigned int head_dim, unsigned int from, unsigned int step) {
  if (!enabled())
    return;
  if (from == 0) {
    heads_q_ = heads_q;
    heads_kv_ = heads_kv;
    head_dim_ = head_dim;
    pos_.assign(heads_kv, {});
  }
  if (!heads_q || !heads_kv || heads_q % heads_kv || !head_dim ||
      heads_q != heads_q_ || heads_kv != heads_kv_ || head_dim != head_dim_ ||
      rotated_q.batch() != 1 || rotated_q.channel() != 1 ||
      rotated_q.height() != step ||
      rotated_q.width() != static_cast<size_t>(heads_q) * head_dim ||
      key.getDim() != value.getDim() || key.batch() != 1 ||
      key.channel() != 1 ||
      key.width() != static_cast<size_t>(heads_kv) * head_dim)
    throw std::invalid_argument("R-KV query/cache shape mismatch");

  const unsigned int n = static_cast<unsigned int>(from + step - removed_);
  rows_ = n;
  if (config_.keepOnly())
    return;
  pushQueries(rotated_q, step);
  for (auto &rows : pos_)
    for (unsigned int p = from; p < from + step; ++p)
      rows.push_back(p);

  const bool due =
    from == 0 || (config_.decode_compress && from % config_.buffer == 0);
  if (due && n > budget_)
    compress(key, value, n, from + step - 1, from == 0);
}

inline void RKVState::pushQueries(const nntrainer::Tensor &rotated_q,
                                  unsigned int step) {
  const unsigned int window = config_.window;
  const size_t width = static_cast<size_t>(heads_q_) * head_dim_;
  if (ring_.size() != window * width) {
    ring_.assign(window * width, 0.0f);
    ring_rows_ = ring_head_ = 0;
  }
  const RKVView q = RKVView::of(rotated_q);
  for (unsigned int r = step > window ? step - window : 0; r < step; ++r) {
    float *slot = ring_.data() + ring_head_ * width;
    for (size_t c = 0; c < width; ++c)
      slot[c] = q.at(r, c);
    ring_head_ = (ring_head_ + 1) % window;
    ring_rows_ = std::min(window, ring_rows_ + 1);
  }
}

inline void RKVState::compress(nntrainer::Tensor &key, nntrainer::Tensor &value,
                               unsigned int n, unsigned int pos, bool prefill) {
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const unsigned int window = config_.window;
  // n > budget > window rows were processed, so the window is full.
  if (ring_rows_ != window || n <= window)
    throw std::logic_error("R-KV observation window is not full");
  const unsigned int cand = n - window;
  const auto start = clock::now();

  RKVPacked packed;
  packed.pack(RKVView::of(key), n, heads_kv_, head_dim_);
  std::vector<float> importance;
  if (config_.lambda > 0.0) {
    const auto obs = observations();
    rkvImportancePacked(obs.data(), window, heads_q_, packed, cand,
                        config_.gqa_pool, config_.kernel, importance);
  }
  const auto scored = clock::now();
  std::vector<float> redundancy;
  if (config_.lambda < 1.0)
    rkvRedundancyPacked(packed, static_cast<float>(config_.threshold),
                        config_.retain, config_.retain_axis, redundancy);
  const auto similar = clock::now();
  packed = RKVPacked();

  const auto keep = rkvSelect(importance, redundancy, n, window, heads_kv_,
                              budget_, config_.lambda, config_.heads);

  const RKVView view = RKVView::of(key);
  if (RKVView::of(value).dtype != view.dtype)
    throw std::invalid_argument("R-KV key and value types differ");
  const size_t element = view.elementSize();
  const size_t row_bytes = view.width * element;
  const size_t head_bytes = static_cast<size_t>(head_dim_) * element;
  unsigned char *key_data = key.getData<unsigned char>();
  unsigned char *value_data = value.getData<unsigned char>();
  rkvCompact(key_data, row_bytes, head_bytes, keep);
  rkvCompact(value_data, row_bytes, head_bytes, keep);
  for (unsigned int g = 0; g < heads_kv_; ++g) {
    std::vector<unsigned int> kept(keep[g].size());
    for (size_t r = 0; r < keep[g].size(); ++r)
      kept[r] = pos_[g][keep[g][r]];
    pos_[g] = std::move(kept);
  }
  const unsigned int kept = static_cast<unsigned int>(keep[0].size());
  removed_ += n - kept;
  rows_ = kept;
  ++events_;
  rkvReleaseRows(key_data, row_bytes, kept, n);
  rkvReleaseRows(value_data, row_bytes, kept, n);
  const auto done = clock::now();

  std::ostringstream line;
  line << "[R-KV layer=" << layer_
       << "] event=" << (prefill ? "prefill" : "decode") << " pos=" << pos
       << " rows=" << n << " kept=" << kept
       << " importance_ms=" << ms(start, scored)
       << " redundancy_ms=" << ms(scored, similar)
       << " compact_ms=" << ms(similar, done);
  std::clog << line.str() << std::endl;
}

} // namespace causallm

#endif // __CAUSALLM_RKV_H__
