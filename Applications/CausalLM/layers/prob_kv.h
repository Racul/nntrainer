// SPDX-License-Identifier: Apache-2.0
/**
 * @file prob_kv.h
 * @date 27 October 2026
 * @brief Probabilistic prompt KV eviction with decode-time importance
 *        correction (ProbKV) for CPU causal attention
 * @note C++ implementation of "A Probabilistic Interpretation of KV Cache
 *       Eviction" (Geh et al., arXiv 2608.28293), which has no official code:
 *       probabilistic eviction (Sec. 4), IMPORTANCEATTENTION (Alg. 1) applied
 *       as an additive logit bias, the importance weight temperature
 *       (Eq. 10), the GQA group mixture (Sec. 5), EXPECTED SAMPLES (Alg. 2)
 *       per head and globally (Eq. 34), the proposals of App. D and H with
 *       the adjustments of App. D.1, and the MAE diagnostic of Sec. 6.
 */
#ifndef __CAUSALLM_PROB_KV_H__
#define __CAUSALLM_PROB_KV_H__

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cpu_backend.h>
#include <fp16.h>
#include <tensor.h>
#include <thread_manager.h>

namespace causallm {

/** @brief Same row tile as mha_core.cpp's compute_kcaches calls. */
constexpr int kProbKVTile = 4;
/** @brief Binary search bound of Alg. 2 (alpha = 1e12). */
constexpr uint64_t kProbKVMaxSamples = 1000000000000ULL;

/** @brief How the kept prompt entries are chosen. */
enum class ProbKVMode {
  Sample, /**< probabilistic eviction: unique entries of m draws (Sec. 4) */
  TopK    /**< zero-variance top-k of the same score, no correction */
};

/** @brief Where the number of samples m is fixed. */
enum class ProbKVScope {
  Global, /**< one m for every KV head of every layer (App. F.1, Eq. 34) */
  Head    /**< one m per KV head (Alg. 2, Sec. 6) */
};

/** @brief Proposal distribution (App. H and App. D). */
enum class ProbKVProposal {
  Min,
  MinH,
  H2O,
  H2OH,
  KNorm,
  TOVA,
  SnapKV,
  Streaming
};

/**
 * @brief Runtime settings of the "probkv" nntr_config object. active is set
 * once the object is given; ratio 0 keeps every row (page release only).
 */
struct ProbKVConfig {
  bool active = false;
  double ratio = 0.0; /**< target compression ratio r in [0, 1) */
  ProbKVMode mode = ProbKVMode::Sample;
  double tau = 1.0; /**< importance weight temperature, +inf: uncorrected */
  ProbKVProposal proposal = ProbKVProposal::MinH;
  ProbKVScope scope = ProbKVScope::Global;
  std::vector<double> group_alpha;   /**< mixture weights; empty: uniform */
  double proposal_temperature = 0.0; /**< D.1 softmax temperature, 0: off */
  double uniform_mix = 1.0;          /**< D.1 alpha, 1: off */
  unsigned int snapkv_window = 64;
  unsigned int snapkv_kernel = 5;
  bool snapkv_maxpool = false;
  unsigned int streaming_sink = 4;
  double streaming_sink_mass = 0.5;
  uint64_t seed = 0;
  bool mae = false; /**< log the Sec. 6 attention MAE on the first decode */
  bool ratio_set = false;
  bool tau_set = false;
  bool scope_set = false;

  bool enabled() const { return active; }

  /** @brief Parse a finite number, or +inf when allow_inf. */
  static double parseNumber(const std::string &text, bool allow_inf = false) {
    std::string value = text;
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
      value = value.substr(1, value.size() - 2);
    if (allow_inf && (value == "inf" || value == "Infinity"))
      return std::numeric_limits<double>::infinity();
    size_t used = 0;
    double number = 0.0;
    try {
      number = std::stod(value, &used);
    } catch (const std::exception &) {
      used = 0;
    }
    if (value.empty() || used != value.size() || !std::isfinite(number))
      throw std::invalid_argument("ProbKV expects a number, got " + text);
    return number;
  }

  /** @brief Parse a nonnegative integer without truncation or wraparound. */
  static uint64_t parseUnsigned(const std::string &text) {
    if (text.empty() ||
        text.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("ProbKV expects an unsigned integer, got " +
                                  text);
    return std::stoull(text);
  }

  static bool parseBool(const std::string &text) {
    if (text == "true" || text == "1")
      return true;
    if (text == "false" || text == "0")
      return false;
    throw std::invalid_argument("ProbKV expects a boolean, got " + text);
  }

  static std::string stripQuotes(const std::string &text) {
    if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
      return text.substr(1, text.size() - 2);
    return text;
  }

  static const char *name(ProbKVProposal proposal) {
    switch (proposal) {
    case ProbKVProposal::Min:
      return "min";
    case ProbKVProposal::MinH:
      return "min_h";
    case ProbKVProposal::H2O:
      return "h2o";
    case ProbKVProposal::H2OH:
      return "h2o_h";
    case ProbKVProposal::KNorm:
      return "knorm";
    case ProbKVProposal::TOVA:
      return "tova";
    case ProbKVProposal::SnapKV:
      return "snapkv";
    case ProbKVProposal::Streaming:
      return "streaming";
    }
    return "?";
  }

  /** @brief Apply one setting given as its JSON key and textual value. */
  void set(const std::string &key, const std::string &raw) {
    const std::string value = stripQuotes(raw);
    active = true;
    if (key == "ratio") {
      ratio = parseNumber(value);
      ratio_set = true;
    } else if (key == "mode") {
      if (value == "sample")
        mode = ProbKVMode::Sample;
      else if (value == "topk")
        mode = ProbKVMode::TopK;
      else
        throw std::invalid_argument("Unknown ProbKV mode: " + value);
    } else if (key == "tau") {
      tau = parseNumber(value, true);
      tau_set = true;
    } else if (key == "proposal") {
      static const ProbKVProposal all[] = {
        ProbKVProposal::Min,    ProbKVProposal::MinH,     ProbKVProposal::H2O,
        ProbKVProposal::H2OH,   ProbKVProposal::KNorm,    ProbKVProposal::TOVA,
        ProbKVProposal::SnapKV, ProbKVProposal::Streaming};
      bool found = false;
      for (const auto candidate : all)
        if (value == name(candidate)) {
          proposal = candidate;
          found = true;
        }
      if (!found)
        throw std::invalid_argument("Unknown ProbKV proposal: " + value);
    } else if (key == "scope") {
      if (value == "global")
        scope = ProbKVScope::Global;
      else if (value == "head")
        scope = ProbKVScope::Head;
      else
        throw std::invalid_argument("Unknown ProbKV scope: " + value);
      scope_set = true;
    } else if (key == "group_alpha") {
      std::string list = value;
      list.erase(
        std::remove_if(list.begin(), list.end(),
                       [](char c) { return c == '[' || c == ']' || c == ' '; }),
        list.end());
      group_alpha.clear();
      std::stringstream stream(list);
      std::string item;
      while (std::getline(stream, item, ','))
        group_alpha.push_back(parseNumber(item));
    } else if (key == "proposal_temperature") {
      proposal_temperature = parseNumber(value);
    } else if (key == "uniform_mix") {
      uniform_mix = parseNumber(value);
    } else if (key == "snapkv_window") {
      snapkv_window = static_cast<unsigned int>(parseUnsigned(value));
    } else if (key == "snapkv_kernel") {
      snapkv_kernel = static_cast<unsigned int>(parseUnsigned(value));
    } else if (key == "snapkv_pool") {
      if (value == "avg")
        snapkv_maxpool = false;
      else if (value == "max")
        snapkv_maxpool = true;
      else
        throw std::invalid_argument("Unknown ProbKV snapkv_pool: " + value);
    } else if (key == "streaming_sink") {
      streaming_sink = static_cast<unsigned int>(parseUnsigned(value));
    } else if (key == "streaming_sink_mass") {
      streaming_sink_mass = parseNumber(value);
    } else if (key == "seed") {
      seed = parseUnsigned(value);
    } else if (key == "mae") {
      mae = parseBool(value);
    } else {
      throw std::invalid_argument("Unknown probkv setting: " + key);
    }
  }

  /** @brief Check ranges and apply the top-k defaults (head scope, no
   * correction). */
  void validate() {
    if (!active)
      return;
    if (!ratio_set)
      throw std::invalid_argument("probkv requires a ratio");
    if (!(ratio >= 0.0 && ratio < 1.0))
      throw std::invalid_argument("ProbKV ratio must be in [0, 1)");
    if (!(tau > 0.0))
      throw std::invalid_argument("ProbKV tau must be positive or inf");
    if (mode == ProbKVMode::TopK) {
      if (tau_set)
        throw std::invalid_argument("ProbKV topk has no correction; drop tau");
      if (scope_set && scope == ProbKVScope::Global)
        throw std::invalid_argument("ProbKV topk keeps a uniform k per head");
      scope = ProbKVScope::Head;
      tau = std::numeric_limits<double>::infinity();
    }
    if (!group_alpha.empty()) {
      double sum = 0.0;
      for (const double a : group_alpha) {
        if (!(a >= 0.0))
          throw std::invalid_argument("ProbKV group_alpha must be >= 0");
        sum += a;
      }
      if (std::fabs(sum - 1.0) > 1e-6)
        throw std::invalid_argument("ProbKV group_alpha must sum to 1");
    }
    if (!(proposal_temperature >= 0.0))
      throw std::invalid_argument("ProbKV proposal_temperature must be >= 0");
    if (!(uniform_mix >= 0.0 && uniform_mix <= 1.0))
      throw std::invalid_argument("ProbKV uniform_mix must be in [0, 1]");
    if (snapkv_window == 0 || snapkv_kernel == 0 || snapkv_kernel % 2 == 0)
      throw std::invalid_argument(
        "ProbKV snapkv needs a window and an odd kernel");
    if (!(streaming_sink_mass >= 0.0 && streaming_sink_mass < 1.0))
      throw std::invalid_argument(
        "ProbKV streaming_sink_mass must be in [0, 1)");
  }

  static std::string number(double value) {
    if (std::isinf(value))
      return "inf";
    std::ostringstream text;
    text << std::setprecision(15) << value;
    return text.str();
  }

  /**
   * @brief One-token encoding for a layer property: "key:value;...". The
   * property parser rejects a second '=' and any space.
   */
  std::string serialize() const {
    std::ostringstream text;
    text << "ratio:" << number(ratio)
         << ";mode:" << (mode == ProbKVMode::Sample ? "sample" : "topk")
         << ";tau:" << number(tau) << ";proposal:" << name(proposal)
         << ";scope:" << (scope == ProbKVScope::Global ? "global" : "head");
    if (!group_alpha.empty()) {
      text << ";group_alpha:";
      for (size_t i = 0; i < group_alpha.size(); ++i)
        text << (i ? "," : "") << number(group_alpha[i]);
    }
    text << ";proposal_temperature:" << number(proposal_temperature)
         << ";uniform_mix:" << number(uniform_mix)
         << ";snapkv_window:" << snapkv_window
         << ";snapkv_kernel:" << snapkv_kernel
         << ";snapkv_pool:" << (snapkv_maxpool ? "max" : "avg")
         << ";streaming_sink:" << streaming_sink
         << ";streaming_sink_mass:" << number(streaming_sink_mass)
         << ";seed:" << seed << ";mae:" << (mae ? "true" : "false");
    return text.str();
  }

  /** @brief Inverse of serialize(); validates the result. */
  static ProbKVConfig parse(const std::string &text) {
    ProbKVConfig config;
    std::stringstream stream(text);
    std::string item;
    bool topk = false;
    while (std::getline(stream, item, ';')) {
      const auto colon = item.find(':');
      if (colon == std::string::npos)
        throw std::invalid_argument("Malformed probkv property: " + text);
      const auto key = item.substr(0, colon);
      const auto value = item.substr(colon + 1);
      // serialize() always writes the top-k defaults; accept them back.
      if (key == "mode" && value == "topk")
        topk = true;
      if (topk && (key == "tau" || key == "scope"))
        continue;
      config.set(key, value);
    }
    config.validate();
    return config;
  }

  /** @brief Human-readable settings for the activation log line. */
  std::string describe() const {
    std::ostringstream text;
    text << "ratio=" << number(ratio)
         << " mode=" << (mode == ProbKVMode::Sample ? "sample" : "topk")
         << " proposal=" << name(proposal)
         << " scope=" << (scope == ProbKVScope::Global ? "global" : "head")
         << " tau=" << number(tau);
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
struct ProbKVView {
  const unsigned char *data = nullptr;
  nntrainer::Tdatatype dtype = nntrainer::Tdatatype::FP32;
  size_t width = 0; /**< elements per row */

  static ProbKVView of(const nntrainer::Tensor &tensor) {
    ProbKVView view;
    view.data = tensor.getData<unsigned char>();
    view.dtype = tensor.getDataType();
    view.width = tensor.width();
    if (view.dtype != nntrainer::Tdatatype::FP32 && !view.half())
      throw std::invalid_argument("ProbKV needs FP32 or 16-bit tensors");
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

/** @brief Eq. 38, 0-indexed: h_j = 1 / ((t - j) * H_t), j = 0..t-1. */
inline std::vector<double> harmonicPrior(unsigned int t) {
  double harmonic = 0.0;
  for (unsigned int k = 1; k <= t; ++k)
    harmonic += 1.0 / k;
  std::vector<double> prior(t);
  for (unsigned int j = 0; j < t; ++j)
    prior[j] = 1.0 / ((t - j) * harmonic);
  return prior;
}

/**
 * @brief acc[h * t + i] += sum_{j >= row_begin} w_j softmax_{i <= j}(q_j k_i /
 * sqrt(D)) for every query head h: the weighted column sums of the causal
 * prompt attention. q is the rotated prompt query [t, Hq * D] in FP32.
 * @param weight per-row weights indexed by j; empty means 1 for every row.
 * @note Rows are interleaved over ThreadManager workers so the causal cost is
 * balanced; every worker accumulates into its own buffer.
 */
inline void accumulateAttention(const float *q, const ProbKVView &k,
                                unsigned int t, unsigned int heads_q,
                                unsigned int heads_kv, unsigned int head_dim,
                                unsigned int row_begin,
                                const std::vector<double> &weight,
                                std::vector<double> &acc) {
  acc.assign(static_cast<size_t>(heads_q) * t, 0.0);
  if (row_begin >= t)
    return;
  const unsigned int gqa = heads_q / heads_kv;
  const size_t q_width = static_cast<size_t>(heads_q) * head_dim;
  const unsigned int rows = t - row_begin;
  auto &tm = nntrainer::ThreadManager::Global();
  const unsigned int units = std::min<unsigned int>(
    rows, std::max<unsigned int>(1u, 4u * (tm.getComputeThreadCount() + 1u)));
  std::vector<std::vector<double>> partial(units);
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  tm.parallel_for(0, units, [&](size_t u) {
    auto &mine = partial[u];
    mine.assign(static_cast<size_t>(heads_q) * t, 0.0);
    std::vector<float> scores(static_cast<size_t>(t) * heads_q);
    for (unsigned int j = row_begin + static_cast<unsigned int>(u); j < t;
         j += units) {
      const double w = weight.empty() ? 1.0 : weight[j];
      if (w == 0.0)
        continue;
      const float *qj = q + static_cast<size_t>(j) * q_width;
      if (k.half()) {
        // Same kernel and [row * Hq + h] layout as the decode attention.
        nntrainer::compute_kcaches<uint16_t>(
          qj, reinterpret_cast<const uint16_t *>(k.data), scores.data(),
          static_cast<int>(j + 1), static_cast<int>(heads_kv),
          static_cast<int>(head_dim), static_cast<int>(gqa), kProbKVTile);
      } else {
        const float *kd = reinterpret_cast<const float *>(k.data);
        for (unsigned int i = 0; i <= j; ++i)
          for (unsigned int h = 0; h < heads_q; ++h) {
            const float *qh = qj + static_cast<size_t>(h) * head_dim;
            const float *ki =
              kd + static_cast<size_t>(i) * k.width + (h / gqa) * head_dim;
            float sum = 0.0f;
            for (unsigned int d = 0; d < head_dim; ++d)
              sum += qh[d] * ki[d];
            scores[static_cast<size_t>(i) * heads_q + h] = sum * scale;
          }
      }
      for (unsigned int h = 0; h < heads_q; ++h) {
        float top = -std::numeric_limits<float>::infinity();
        for (unsigned int i = 0; i <= j; ++i)
          top = std::max(top, scores[static_cast<size_t>(i) * heads_q + h]);
        double total = 0.0;
        for (unsigned int i = 0; i <= j; ++i) {
          float &s = scores[static_cast<size_t>(i) * heads_q + h];
          s = std::exp(s - top);
          total += s;
        }
        const double factor = w / total;
        double *row = mine.data() + static_cast<size_t>(h) * t;
        for (unsigned int i = 0; i <= j; ++i)
          row[i] += factor * scores[static_cast<size_t>(i) * heads_q + h];
      }
    }
  });
  for (const auto &mine : partial)
    for (size_t i = 0; i < acc.size(); ++i)
      acc[i] += mine[i];
}

/** @brief L2 norm of each (row, KV head) slice: out[g * t + i]. */
inline std::vector<double> rowNorms(const ProbKVView &cache, unsigned int t,
                                    unsigned int heads_kv,
                                    unsigned int head_dim) {
  std::vector<double> out(static_cast<size_t>(heads_kv) * t);
  for (unsigned int g = 0; g < heads_kv; ++g)
    for (unsigned int i = 0; i < t; ++i) {
      double sum = 0.0;
      for (unsigned int d = 0; d < head_dim; ++d) {
        const double x = cache.at(i, static_cast<size_t>(g) * head_dim + d);
        sum += x * x;
      }
      out[static_cast<size_t>(g) * t + i] = std::sqrt(sum);
    }
  return out;
}

/** @brief Normalize a nonnegative row in place; uniform if it sums to 0. */
inline void normalizeRow(double *row, unsigned int t) {
  double sum = 0.0;
  for (unsigned int i = 0; i < t; ++i)
    sum += row[i];
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    for (unsigned int i = 0; i < t; ++i)
      row[i] = 1.0 / t;
    return;
  }
  for (unsigned int i = 0; i < t; ++i)
    row[i] /= sum;
}

/** @brief Softmax of a row in place (App. D K-norm and D.1 temperature). */
inline void softmaxRow(double *row, unsigned int t, double temperature) {
  double top = -std::numeric_limits<double>::infinity();
  for (unsigned int i = 0; i < t; ++i)
    top = std::max(top, row[i] / temperature);
  double sum = 0.0;
  for (unsigned int i = 0; i < t; ++i) {
    row[i] = std::exp(row[i] / temperature - top);
    sum += row[i];
  }
  for (unsigned int i = 0; i < t; ++i)
    row[i] /= sum;
}

/**
 * @brief Per-query-head proposals [Hq * t], each row summing to 1:
 * pi_min, pi_min-h, pi_H2O, pi_H2O-h (App. H, Eq. 35-36), the App. D
 * proposals of K-norm, TOVA, SnapKV and StreamingLLM, then the App. D.1
 * temperature and uniform mixture.
 * @param q rotated prompt query [t, Hq * D] in FP32
 */
inline std::vector<double>
buildHeadProposals(const ProbKVConfig &config, const float *q,
                   const ProbKVView &k, const ProbKVView &v, unsigned int t,
                   unsigned int heads_q, unsigned int heads_kv,
                   unsigned int head_dim) {
  const unsigned int gqa = heads_q / heads_kv;
  std::vector<double> score;
  bool softmax_score = false; // K-norm: the score is a logit, not a mass
  switch (config.proposal) {
  case ProbKVProposal::Min:
  case ProbKVProposal::TOVA:
    accumulateAttention(q, k, t, heads_q, heads_kv, head_dim, t - 1, {}, score);
    break;
  case ProbKVProposal::MinH:
  case ProbKVProposal::H2OH:
    accumulateAttention(q, k, t, heads_q, heads_kv, head_dim, 0,
                        harmonicPrior(t), score);
    break;
  case ProbKVProposal::H2O:
    accumulateAttention(q, k, t, heads_q, heads_kv, head_dim, 0, {}, score);
    break;
  case ProbKVProposal::SnapKV: {
    const unsigned int window = std::min(config.snapkv_window, t);
    accumulateAttention(q, k, t, heads_q, heads_kv, head_dim, t - window, {},
                        score);
    const unsigned int prefix = t - window;
    const int half = static_cast<int>(config.snapkv_kernel / 2);
    std::vector<double> pooled(t);
    for (unsigned int h = 0; h < heads_q; ++h) {
      double *row = score.data() + static_cast<size_t>(h) * t;
      for (unsigned int i = 0; i < t; ++i)
        row[i] /= window; // mean over the observation window
      double best = 0.0;
      for (unsigned int i = 0; i < prefix; ++i) {
        // avg_pool1d(padding=kernel/2, count_include_pad) or max_pool1d
        double acc = config.snapkv_maxpool
                       ? -std::numeric_limits<double>::infinity()
                       : 0.0;
        for (int o = -half; o <= half; ++o) {
          const int x = static_cast<int>(i) + o;
          const double value =
            (x >= 0 && x < static_cast<int>(prefix)) ? row[x] : 0.0;
          acc = config.snapkv_maxpool ? std::max(acc, value) : acc + value;
        }
        pooled[i] = config.snapkv_maxpool ? acc : acc / config.snapkv_kernel;
        best = std::max(best, pooled[i]);
      }
      // The observation window is always kept by SnapKV: give it the max.
      for (unsigned int i = prefix; i < t; ++i)
        pooled[i] = prefix ? best : 1.0;
      std::copy(pooled.begin(), pooled.end(), row);
    }
    break;
  }
  case ProbKVProposal::KNorm: {
    // K-norm keeps low key norms (Devoto et al., KVPress KnormPress).
    const auto norms = rowNorms(k, t, heads_kv, head_dim);
    score.resize(static_cast<size_t>(heads_q) * t);
    for (unsigned int h = 0; h < heads_q; ++h)
      for (unsigned int i = 0; i < t; ++i)
        score[static_cast<size_t>(h) * t + i] =
          -norms[static_cast<size_t>(h / gqa) * t + i];
    softmax_score = true;
    break;
  }
  case ProbKVProposal::Streaming: {
    // A mixture of a monotonically increasing distribution with one on the
    // first sink entries (App. D).
    score.resize(static_cast<size_t>(heads_q) * t);
    const unsigned int sink = std::min(config.streaming_sink, t);
    const double mass = sink ? config.streaming_sink_mass : 0.0;
    const double ramp = 0.5 * static_cast<double>(t) * (t + 1);
    for (unsigned int h = 0; h < heads_q; ++h)
      for (unsigned int i = 0; i < t; ++i)
        score[static_cast<size_t>(h) * t + i] =
          (1.0 - mass) * (i + 1) / ramp + (i < sink ? mass / sink : 0.0);
    break;
  }
  }

  if (config.proposal == ProbKVProposal::Min ||
      config.proposal == ProbKVProposal::MinH) {
    const auto norms = rowNorms(v, t, heads_kv, head_dim);
    for (unsigned int h = 0; h < heads_q; ++h)
      for (unsigned int i = 0; i < t; ++i)
        score[static_cast<size_t>(h) * t + i] *=
          norms[static_cast<size_t>(h / gqa) * t + i];
  }
  if (config.proposal == ProbKVProposal::TOVA) {
    // TOVA averages the last query's distribution over the layer's heads.
    std::vector<double> mean(t, 0.0);
    for (unsigned int h = 0; h < heads_q; ++h)
      for (unsigned int i = 0; i < t; ++i)
        mean[i] += score[static_cast<size_t>(h) * t + i] / heads_q;
    for (unsigned int h = 0; h < heads_q; ++h)
      std::copy(mean.begin(), mean.end(),
                score.begin() + static_cast<size_t>(h) * t);
  }

  for (unsigned int h = 0; h < heads_q; ++h) {
    double *row = score.data() + static_cast<size_t>(h) * t;
    if (config.proposal_temperature > 0.0)
      softmaxRow(row, t, config.proposal_temperature);
    else if (softmax_score)
      softmaxRow(row, t, 1.0);
    else
      normalizeRow(row, t);
    if (config.uniform_mix < 1.0)
      for (unsigned int i = 0; i < t; ++i)
        row[i] = config.uniform_mix * row[i] + (1.0 - config.uniform_mix) / t;
  }
  return score;
}

/** @brief Sec. 5 GQA mixture pi_g = sum_j alpha_j pi_{g * gqa + j}. */
inline std::vector<double> mixGroupProposal(const std::vector<double> &head_pi,
                                            unsigned int heads_q,
                                            unsigned int heads_kv,
                                            unsigned int t,
                                            const std::vector<double> &alpha) {
  const unsigned int gqa = heads_q / heads_kv;
  if (!alpha.empty() && alpha.size() != gqa)
    throw std::invalid_argument("ProbKV group_alpha needs one weight per "
                                "query head of a KV group");
  std::vector<double> group(static_cast<size_t>(heads_kv) * t, 0.0);
  for (unsigned int g = 0; g < heads_kv; ++g)
    for (unsigned int j = 0; j < gqa; ++j) {
      const double a = alpha.empty() ? 1.0 / gqa : alpha[j];
      const double *src = head_pi.data() + static_cast<size_t>(g * gqa + j) * t;
      double *dst = group.data() + static_cast<size_t>(g) * t;
      for (unsigned int i = 0; i < t; ++i)
        dst[i] += a * src[i];
    }
  return group;
}

/** @brief log1p(-pi_i), cached for expectedUnique(). */
inline std::vector<double> logComplement(const double *pi, size_t n) {
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = std::log1p(-std::min(pi[i], 1.0));
  return out;
}

/** @brief Eq. 25: sum_i 1 - (1 - pi_i)^m from cached log1p(-pi_i). */
inline double expectedUnique(const double *log_complement, size_t n,
                             uint64_t m) {
  if (m == 0)
    return 0.0;
  const double draws = static_cast<double>(m);
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i)
    sum += log_complement[i] == -std::numeric_limits<double>::infinity()
             ? 1.0
             : -std::expm1(draws * log_complement[i]);
  return sum;
}

/**
 * @brief Alg. 2: binary search on [0, 1e12] for m with
 * |expected(m) - target| <= 0.5. expected must be non-decreasing.
 * @return the found m, or the first m above the target when unreachable
 */
inline uint64_t solveSamples(const std::function<double(uint64_t)> &expected,
                             double target) {
  uint64_t low = 0, high = kProbKVMaxSamples;
  while (low <= high) {
    const uint64_t mid = low + (high - low) / 2;
    const double e = expected(mid);
    if (std::fabs(e - target) <= 0.5)
      return mid;
    if (e < target)
      low = mid + 1;
    else if (mid == 0)
      return 0;
    else
      high = mid - 1;
  }
  return std::min(low, kProbKVMaxSamples);
}

/** @brief M1: multinomial counts of m draws by chained binomials, O(t). */
inline std::vector<uint64_t> sampleCounts(const double *pi, unsigned int t,
                                          uint64_t m, std::mt19937_64 &rng) {
  std::vector<uint64_t> counts(t, 0);
  std::vector<double> suffix(t + 1, 0.0);
  for (unsigned int i = t; i-- > 0;)
    suffix[i] = suffix[i + 1] + pi[i];
  unsigned long long remaining = m;
  for (unsigned int i = 0; i < t && remaining; ++i) {
    double p = suffix[i] > 0.0 ? pi[i] / suffix[i] : 0.0;
    if (i + 1 == t || p >= 1.0) {
      counts[i] = remaining;
      break;
    }
    if (!(p > 0.0))
      continue;
    std::binomial_distribution<unsigned long long> draw(remaining, p);
    counts[i] = draw(rng);
    remaining -= counts[i];
  }
  return counts;
}

/** @brief M10: indices of the k largest scores (ties: lower index), sorted. */
inline std::vector<unsigned int> topkKeep(const double *score, unsigned int t,
                                          unsigned int k) {
  std::vector<unsigned int> index(t);
  std::iota(index.begin(), index.end(), 0u);
  k = std::min(k, t);
  std::partial_sort(index.begin(), index.begin() + k, index.end(),
                    [score](unsigned int a, unsigned int b) {
                      return score[a] > score[b] ||
                             (score[a] == score[b] && a < b);
                    });
  index.resize(k);
  std::sort(index.begin(), index.end());
  return index;
}

/** @brief Alg. 1 line 2 with Eq. 10: (1/tau) log(c / (m pi)); 0 at inf. */
inline float logWeight(uint64_t count, uint64_t m, double pi, double tau) {
  if (std::isinf(tau))
    return 0.0f;
  return static_cast<float>((std::log(static_cast<double>(count)) -
                             std::log(static_cast<double>(m)) - std::log(pi)) /
                            tau);
}

/**
 * @brief Alg. 1 reference for one head. Rows [0, t) are eviction-time entries
 * scaled by weight[i] (0: evicted), rows [t, n) decode-time entries.
 */
inline void importanceAttention(const float *q, const float *k, const float *v,
                                unsigned int n, unsigned int t, unsigned int d,
                                const std::vector<double> &weight, float *out) {
  std::vector<double> s(n);
  double top = -std::numeric_limits<double>::infinity();
  for (unsigned int i = 0; i < n; ++i) {
    double dot = 0.0;
    for (unsigned int x = 0; x < d; ++x)
      dot += static_cast<double>(q[x]) * k[static_cast<size_t>(i) * d + x];
    s[i] = dot / std::sqrt(static_cast<double>(d));
    if (i >= t || weight[i] > 0.0)
      top = std::max(top, s[i]);
  }
  std::vector<double> num(d, 0.0);
  double z = 0.0;
  for (unsigned int i = 0; i < n; ++i) {
    const double w = i < t ? weight[i] : 1.0;
    if (!(w > 0.0))
      continue;
    const double p = w * std::exp(s[i] - top);
    z += p;
    for (unsigned int x = 0; x < d; ++x)
      num[x] += p * v[static_cast<size_t>(i) * d + x];
  }
  for (unsigned int x = 0; x < d; ++x)
    out[x] = static_cast<float>(num[x] / z);
}

/**
 * @brief Return the physical pages of rows [first_row, rows) of a contiguous
 * [rows, row_bytes] buffer to the system. The rows read as zero on Linux and
 * are faulted in again when written. Only whole pages are released.
 * @note No-op on Windows and in CUDA builds, whose KV may be managed memory.
 * @return released bytes
 */
inline size_t releaseRows(void *base, size_t row_bytes, size_t first_row,
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

/** @brief releaseRows() on a [1, 1, rows, width] cache tensor. */
inline size_t releaseRows(nntrainer::Tensor &cache, size_t first_row) {
  if (cache.batch() != 1 || cache.channel() != 1 || cache.height() == 0)
    return 0;
  return releaseRows(cache.getData<unsigned char>(),
                     cache.bytes() / cache.height(), first_row, cache.height());
}

/* ------------------------------------------------------------------------ */
/* Per-layer state and the global coordinator                               */
/* ------------------------------------------------------------------------ */

class ProbKVRegistry;

/**
 * @brief Layer-local ProbKV state. After the prompt attention it builds the
 * proposals, selects the kept entries per KV head (sampling or top-k), moves
 * each head's kept rows to the front of the cache, and keeps the Alg. 1
 * correction as an additive logit bias of [Hkv, R] where R is the largest
 * kept count of the layer: log w for kept rows, -inf where a head kept fewer
 * than R. Decode rows (physical row >= R) are exact and get no bias.
 */
class ProbKVState {
public:
  bool enabled() const { return config_.enabled(); }
  const ProbKVConfig &config() const { return config_; }

  /**
   * @brief Consume "probkv" (serialized config) and "probkv_run"
   * ("layer,layers,prompt,session"); both reset the per-run state.
   */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key == "probkv") {
      config_ = ProbKVConfig::parse(value);
    } else if (key == "probkv_run") {
      std::stringstream stream(value);
      std::string item;
      std::vector<std::string> parts;
      while (std::getline(stream, item, ','))
        parts.push_back(item);
      if (parts.size() != 4)
        throw std::invalid_argument("probkv_run needs layer,layers,prompt,id");
      layer_ = static_cast<unsigned int>(ProbKVConfig::parseUnsigned(parts[0]));
      layers_ =
        static_cast<unsigned int>(ProbKVConfig::parseUnsigned(parts[1]));
      prompt_ =
        static_cast<unsigned int>(ProbKVConfig::parseUnsigned(parts[2]));
      session_ = parts[3];
      if (!layers_ || layer_ >= layers_ || !prompt_)
        throw std::invalid_argument("Invalid probkv_run " + value);
    } else {
      return false;
    }
    resetRun();
    return true;
  }

  /** @brief Validate a step before K/V are written or RoPE is applied and
   * record the next logical position. */
  void checkStep(unsigned int from, unsigned int step, size_t capacity,
                 unsigned int max_position) {
    if (!enabled())
      return;
    if (!prompt_ || from != next_ || step != (next_ ? 1u : prompt_))
      throw std::invalid_argument("ProbKV requires one full prefill followed "
                                  "by consecutive single tokens");
    if (from > max_position || step > max_position - from)
      throw std::out_of_range("ProbKV step exceeds the RoPE position limit");
    const size_t row = from - removed_;
    if (row > capacity || step > capacity - row)
      throw std::out_of_range("ProbKV step exceeds the cache capacity");
    next_ = from + step;
  }

  /**
   * @brief After the prompt attention: build the group proposals, then evict
   * now (head scope and top-k) or hand the layer to the global solve.
   * @param rotated_q prompt query after RoPE [1, 1, t, Hq * D]
   * @param key, value full cache tensors [1, 1, capacity, Hkv * D]
   */
  void finishPrefill(const nntrainer::Tensor &rotated_q, nntrainer::Tensor &key,
                     nntrainer::Tensor &value, unsigned int heads_q,
                     unsigned int heads_kv, unsigned int head_dim,
                     ProbKVRegistry &registry);

  /** @brief Global scope: sample m draws for every KV head and compact. */
  void evictGlobal(uint64_t m) { select(std::vector<uint64_t>(heads_kv_, m)); }

  /** @brief Group proposals [Hkv * t] kept for the global solve. */
  const std::vector<double> &groupProposals() const { return group_pi_; }
  unsigned int prompt() const { return prompt_; }
  unsigned int layer() const { return layer_; }
  /** @brief Sum over KV heads of kept entries (after eviction). */
  size_t keptTotal() const {
    return std::accumulate(kept_.begin(), kept_.end(), size_t{0});
  }

  size_t removed() const { return removed_; }
  unsigned int rows() const { return rows_; }
  /** @brief R when decode attention needs the bias, else 0. */
  unsigned int biasRows() const { return bias_active_ ? rows_ : 0; }
  /** @brief The R bias entries of one KV head. */
  const float *bias(unsigned int kv_head) const {
    return bias_.data() + static_cast<size_t>(kv_head) * rows_;
  }

  /** @brief Add the bias to per-row scores laid out as [row * Hq + h]. */
  void addBias(float *scores, unsigned int rows, unsigned int heads_q,
               unsigned int gqa) const {
    const unsigned int end = std::min(rows, biasRows());
    for (unsigned int r = 0; r < end; ++r)
      for (unsigned int h = 0; h < heads_q; ++h)
        scores[static_cast<size_t>(r) * heads_q + h] +=
          bias_[static_cast<size_t>(h / gqa) * rows_ + r];
  }
#ifdef ENABLE_FP16
  void addBias(_FP16 *scores, unsigned int rows, unsigned int heads_q,
               unsigned int gqa) const {
    const unsigned int end = std::min(rows, biasRows());
    for (unsigned int r = 0; r < end; ++r)
      for (unsigned int h = 0; h < heads_q; ++h) {
        _FP16 &s = scores[static_cast<size_t>(r) * heads_q + h];
        s = static_cast<_FP16>(static_cast<float>(s) +
                               bias_[static_cast<size_t>(h / gqa) * rows_ + r]);
      }
  }
#endif

  /** @brief M11 diagnostic pending for the first decode step. */
  bool maeArmed() const { return mae_armed_; }

  /**
   * @brief M11 (Sec. 6, Eq. 11): mean absolute error of the first decode
   * token's attention output against the full prompt cache, for the
   * uncorrected, corrected and top-k kept sets. Frees the prompt copy.
   * @param rotated_q decode query after RoPE [1, 1, 1, Hq * D]
   * @return head-mean MAE {uncorrected, corrected, top-k}
   */
  std::array<double, 3> maeReport(const nntrainer::Tensor &rotated_q,
                                  const nntrainer::Tensor &key,
                                  const nntrainer::Tensor &value);

private:
  void resetRun() {
    next_ = 0;
    removed_ = 0;
    rows_ = 0;
    bias_active_ = false;
    bias_.clear();
    kept_.clear();
    group_pi_.clear();
    key_ = value_ = nullptr;
    mae_armed_ = false;
    mae_k_.clear();
    mae_v_.clear();
    mae_keep_.clear();
    mae_topk_.clear();
  }

  /** @brief Choose kept rows per KV head (m[g] draws, or top-k), compact,
   * fill the bias and release the evicted pages. */
  void select(const std::vector<uint64_t> &m);

  ProbKVConfig config_;
  unsigned int layer_ = 0;
  unsigned int layers_ = 0;
  unsigned int prompt_ = 0;
  std::string session_;
  unsigned int next_ = 0;

  unsigned char *key_ = nullptr;
  unsigned char *value_ = nullptr;
  ProbKVView key_view_, value_view_;
  size_t capacity_ = 0;
  unsigned int heads_q_ = 0, heads_kv_ = 0, head_dim_ = 0;
  double proposal_ms_ = 0.0;

  std::vector<double> group_pi_; /**< [Hkv * t] */
  std::vector<size_t> kept_;     /**< per KV head */
  size_t removed_ = 0;
  unsigned int rows_ = 0; /**< R */
  bool bias_active_ = false;
  std::vector<float> bias_; /**< [Hkv * R] */

  bool mae_armed_ = false;
  std::vector<unsigned char> mae_k_,
    mae_v_; /**< prompt rows before compaction */
  std::vector<std::vector<unsigned int>> mae_keep_, mae_topk_;
};

/**
 * @brief Global-scope coordinator (App. F.1, Eq. 34): one m such that the
 * mean expected number of kept entries over every KV head of every layer is
 * (1 - r) t. Layers register after their prompt attention; the last one
 * triggers the solve and the eviction of all, still inside the prefill.
 */
class ProbKVRegistry {
public:
  void add(const std::string &session, unsigned int layer, unsigned int layers,
           double ratio, ProbKVState *state) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (session != session_ || states_.size() != layers) {
      session_ = session;
      states_.assign(layers, nullptr);
    }
    if (layer >= states_.size() || states_[layer] != nullptr)
      throw std::invalid_argument("ProbKV layer registered twice: " +
                                  std::to_string(layer));
    states_[layer] = state;
    if (std::find(states_.begin(), states_.end(), nullptr) != states_.end())
      return;

    const auto start = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> logs;
    size_t heads = 0;
    unsigned int t = 0;
    for (auto *s : states_) {
      const auto &pi = s->groupProposals();
      t = s->prompt();
      heads += pi.size() / t;
      logs.push_back(logComplement(pi.data(), pi.size()));
    }
    const double target = (1.0 - ratio) * t;
    auto &tm = nntrainer::ThreadManager::Global();
    std::vector<double> partial(logs.size());
    const uint64_t m = std::max<uint64_t>(
      1,
      solveSamples(
        [&](uint64_t draws) {
          tm.parallel_for(0, logs.size(), [&](size_t i) {
            partial[i] = expectedUnique(logs[i].data(), logs[i].size(), draws);
          });
          return std::accumulate(partial.begin(), partial.end(), 0.0) / heads;
        },
        target));
    const double solve_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
    size_t kept = 0, rows = 0;
    for (auto *s : states_) {
      s->evictGlobal(m);
      kept += s->keptTotal();
      rows += s->rows();
    }
    last_m_ = m;
    std::clog << "[ProbKV global] m=" << m
              << " target_r=" << ProbKVConfig::number(ratio) << " effective_r="
              << 1.0 - static_cast<double>(kept) / (heads * t) << " physical_r="
              << 1.0 - static_cast<double>(rows) / (states_.size() * t)
              << " solve_ms=" << solve_ms << std::endl;
    states_.assign(layers, nullptr);
  }

  uint64_t lastSamples() const { return last_m_; }

private:
  std::mutex mutex_;
  std::string session_;
  std::vector<ProbKVState *> states_;
  uint64_t last_m_ = 0;
};

inline void
ProbKVState::finishPrefill(const nntrainer::Tensor &rotated_q,
                           nntrainer::Tensor &key, nntrainer::Tensor &value,
                           unsigned int heads_q, unsigned int heads_kv,
                           unsigned int head_dim, ProbKVRegistry &registry) {
  if (!enabled())
    return;
  const unsigned int t = prompt_;
  if (!heads_q || !heads_kv || heads_q % heads_kv || !head_dim ||
      rotated_q.batch() != 1 || rotated_q.channel() != 1 ||
      rotated_q.height() != t ||
      rotated_q.width() != static_cast<size_t>(heads_q) * head_dim ||
      key.getDim() != value.getDim() || key.batch() != 1 ||
      key.channel() != 1 ||
      key.width() != static_cast<size_t>(heads_kv) * head_dim ||
      key.height() < t)
    throw std::invalid_argument("ProbKV prompt/cache shape mismatch");
  heads_q_ = heads_q;
  heads_kv_ = heads_kv;
  head_dim_ = head_dim;
  key_ = key.getData<unsigned char>();
  value_ = value.getData<unsigned char>();
  key_view_ = ProbKVView::of(key);
  value_view_ = ProbKVView::of(value);
  capacity_ = key.height();

  if (config_.ratio == 0.0) {
    // Keep control: nothing evicted, decode is the unchanged path.
    kept_.assign(heads_kv_, t);
    rows_ = t;
    std::clog << "[ProbKV layer=" << layer_ << "] prompt=" << t << " rows=" << t
              << " kept=all" << std::endl;
    return;
  }

  const auto start = std::chrono::steady_clock::now();
  const ProbKVView qv = ProbKVView::of(rotated_q);
  std::vector<float> q32(static_cast<size_t>(t) * qv.width);
  for (size_t r = 0; r < t; ++r)
    for (size_t c = 0; c < qv.width; ++c)
      q32[r * qv.width + c] = qv.at(r, c);
  const auto head_pi =
    buildHeadProposals(config_, q32.data(), key_view_, value_view_, t, heads_q,
                       heads_kv, head_dim);
  group_pi_ =
    mixGroupProposal(head_pi, heads_q, heads_kv, t, config_.group_alpha);
  proposal_ms_ = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - start)
                   .count();

  if (config_.mode == ProbKVMode::TopK) {
    select({});
  } else if (config_.scope == ProbKVScope::Head) {
    std::vector<uint64_t> m(heads_kv_);
    for (unsigned int g = 0; g < heads_kv_; ++g) {
      const auto logs =
        logComplement(group_pi_.data() + static_cast<size_t>(g) * t, t);
      m[g] = std::max<uint64_t>(
        1,
        solveSamples(
          [&](uint64_t draws) { return expectedUnique(logs.data(), t, draws); },
          (1.0 - config_.ratio) * t));
    }
    select(m);
  } else {
    registry.add(session_, layer_, layers_, config_.ratio, this);
  }
}

inline void ProbKVState::select(const std::vector<uint64_t> &m) {
  const auto start = std::chrono::steady_clock::now();
  const unsigned int t = prompt_;
  const bool topk = config_.mode == ProbKVMode::TopK;
  const unsigned int k_topk = std::max<unsigned int>(
    1u, static_cast<unsigned int>(std::floor(t * (1.0 - config_.ratio))));

  std::vector<std::vector<unsigned int>> keep(heads_kv_);
  std::vector<std::vector<float>> logw(heads_kv_);
  for (unsigned int g = 0; g < heads_kv_; ++g) {
    const double *pi = group_pi_.data() + static_cast<size_t>(g) * t;
    if (topk) {
      keep[g] = topkKeep(pi, t, k_topk);
      logw[g].assign(keep[g].size(), 0.0f);
      continue;
    }
    std::seed_seq seq{static_cast<uint32_t>(config_.seed),
                      static_cast<uint32_t>(config_.seed >> 32),
                      static_cast<uint32_t>(layer_), static_cast<uint32_t>(g)};
    std::mt19937_64 rng(seq);
    const auto counts = sampleCounts(pi, t, m[g], rng);
    for (unsigned int i = 0; i < t; ++i)
      if (counts[i]) {
        keep[g].push_back(i);
        logw[g].push_back(logWeight(counts[i], m[g], pi[i], config_.tau));
      }
  }

  if (config_.mae) {
    // Sec. 6 compares at the same per-head size: top-k of the same scores.
    const size_t row_bytes = key_view_.width * key_view_.elementSize();
    mae_k_.assign(key_, key_ + static_cast<size_t>(t) * row_bytes);
    mae_v_.assign(value_, value_ + static_cast<size_t>(t) * row_bytes);
    mae_keep_ = keep;
    mae_topk_.resize(heads_kv_);
    for (unsigned int g = 0; g < heads_kv_; ++g)
      mae_topk_[g] = topkKeep(group_pi_.data() + static_cast<size_t>(g) * t, t,
                              static_cast<unsigned int>(keep[g].size()));
    mae_armed_ = true;
  }

  unsigned int rows = 0;
  kept_.assign(heads_kv_, 0);
  for (unsigned int g = 0; g < heads_kv_; ++g) {
    kept_[g] = keep[g].size();
    rows = std::max<unsigned int>(rows, keep[g].size());
  }

  // Move each KV head's kept rows to the front; positions ascend, so every
  // source row is read before it can be overwritten.
  const size_t element = key_view_.elementSize();
  const size_t row_bytes = key_view_.width * element;
  const size_t head_bytes = static_cast<size_t>(head_dim_) * element;
  for (unsigned char *data : {key_, value_})
    for (unsigned int g = 0; g < heads_kv_; ++g)
      for (size_t r = 0; r < keep[g].size(); ++r)
        if (keep[g][r] != r)
          std::memmove(data + r * row_bytes + g * head_bytes,
                       data + keep[g][r] * row_bytes + g * head_bytes,
                       head_bytes);

  bias_.assign(static_cast<size_t>(heads_kv_) * rows,
               -std::numeric_limits<float>::infinity());
  bias_active_ = false;
  for (unsigned int g = 0; g < heads_kv_; ++g) {
    if (keep[g].size() < rows)
      bias_active_ = true;
    for (size_t r = 0; r < keep[g].size(); ++r) {
      bias_[static_cast<size_t>(g) * rows + r] = logw[g][r];
      if (logw[g][r] != 0.0f)
        bias_active_ = true;
    }
  }
  rows_ = rows;
  removed_ = t - rows;
  releaseRows(key_, row_bytes, rows, capacity_);
  releaseRows(value_, row_bytes, rows, capacity_);

  const double evict_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
  std::ostringstream line;
  line << "[ProbKV layer=" << layer_ << "] prompt=" << t << " rows=" << rows
       << " kept=";
  for (unsigned int g = 0; g < heads_kv_; ++g)
    line << (g ? "," : "") << kept_[g];
  line << " m=";
  if (topk)
    line << "topk";
  else
    for (unsigned int g = 0; g < heads_kv_; ++g) {
      line << (g ? "," : "") << m[g];
      if (config_.scope == ProbKVScope::Global)
        break;
    }
  line << " proposal_ms=" << proposal_ms_ << " evict_ms=" << evict_ms;
  std::clog << line.str() << std::endl;
}

inline std::array<double, 3>
ProbKVState::maeReport(const nntrainer::Tensor &rotated_q,
                       const nntrainer::Tensor &key,
                       const nntrainer::Tensor &value) {
  if (!mae_armed_)
    return {0.0, 0.0, 0.0};
  mae_armed_ = false;
  const unsigned int t = prompt_, d = head_dim_, gqa = heads_q_ / heads_kv_;
  ProbKVView full_k = key_view_, full_v = value_view_;
  full_k.data = mae_k_.data();
  full_v.data = mae_v_.data();
  const ProbKVView cur_k = ProbKVView::of(key), cur_v = ProbKVView::of(value);
  const ProbKVView qv = ProbKVView::of(rotated_q);

  // Gather rows of one KV head into FP32 [n, d] buffers.
  auto gather = [&](const ProbKVView &cache,
                    const std::vector<unsigned int> &rs, unsigned int g,
                    std::vector<float> &out) {
    out.resize((rs.size() + 1) * d);
    for (size_t r = 0; r < rs.size(); ++r)
      for (unsigned int x = 0; x < d; ++x)
        out[r * d + x] = cache.at(rs[r], static_cast<size_t>(g) * d + x);
  };
  auto append_new = [&](const ProbKVView &cache, unsigned int g, size_t at,
                        std::vector<float> &out) {
    for (unsigned int x = 0; x < d; ++x)
      out[at * d + x] = cache.at(rows_, static_cast<size_t>(g) * d + x);
  };

  double err_unc = 0.0, err_cor = 0.0, err_top = 0.0;
  std::vector<unsigned int> all(t);
  std::iota(all.begin(), all.end(), 0u);
  std::vector<float> kf, vf, ks, vs, kt, vt, q(d), o(d), ou(d), oc(d), ot(d);
  for (unsigned int h = 0; h < heads_q_; ++h) {
    const unsigned int g = h / gqa;
    for (unsigned int x = 0; x < d; ++x)
      q[x] = qv.at(0, static_cast<size_t>(h) * d + x);
    gather(full_k, all, g, kf);
    gather(full_v, all, g, vf);
    append_new(cur_k, g, t, kf);
    append_new(cur_v, g, t, vf);
    importanceAttention(q.data(), kf.data(), vf.data(), t + 1, t, d,
                        std::vector<double>(t, 1.0), o.data());

    const auto &keep = mae_keep_[g];
    const unsigned int n = static_cast<unsigned int>(keep.size());
    gather(full_k, keep, g, ks);
    gather(full_v, keep, g, vs);
    append_new(cur_k, g, n, ks);
    append_new(cur_v, g, n, vs);
    std::vector<double> ones(n, 1.0), weights(n);
    for (unsigned int r = 0; r < n; ++r)
      weights[r] = std::exp(bias_[static_cast<size_t>(g) * rows_ + r]);
    importanceAttention(q.data(), ks.data(), vs.data(), n + 1, n, d, ones,
                        ou.data());
    importanceAttention(q.data(), ks.data(), vs.data(), n + 1, n, d, weights,
                        oc.data());

    const auto &top = mae_topk_[g];
    const unsigned int nt = static_cast<unsigned int>(top.size());
    gather(full_k, top, g, kt);
    gather(full_v, top, g, vt);
    append_new(cur_k, g, nt, kt);
    append_new(cur_v, g, nt, vt);
    importanceAttention(q.data(), kt.data(), vt.data(), nt + 1, nt, d,
                        std::vector<double>(nt, 1.0), ot.data());

    for (unsigned int x = 0; x < d; ++x) {
      err_unc += std::fabs(o[x] - ou[x]) / d;
      err_cor += std::fabs(o[x] - oc[x]) / d;
      err_top += std::fabs(o[x] - ot[x]) / d;
    }
  }
  std::clog << "[ProbKV mae] layer=" << layer_ << " unc=" << err_unc / heads_q_
            << " cor=" << err_cor / heads_q_ << " top=" << err_top / heads_q_
            << std::endl;
  mae_k_.clear();
  mae_k_.shrink_to_fit();
  mae_v_.clear();
  mae_v_.shrink_to_fit();
  return {err_unc / heads_q_, err_cor / heads_q_, err_top / heads_q_};
}

} // namespace causallm

#endif // __CAUSALLM_PROB_KV_H__
