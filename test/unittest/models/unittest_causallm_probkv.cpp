// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_causallm_probkv.cpp
 * @date   27 October 2026
 * @brief  Unit tests for ProbKV probabilistic prompt KV eviction
 * @see    Geh et al., "A Probabilistic Interpretation of KV Cache Eviction",
 *         arXiv 2608.28293
 * @bug    No known bugs except for NYI items
 */

#include <prob_kv.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using causallm::ProbKVConfig;
using causallm::ProbKVState;
using causallm::ProbKVView;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

/** @brief Uniform random floats in [-scale, scale]. */
std::vector<float> randomValues(size_t n, uint32_t seed, float scale = 1.0f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-scale, scale);
  std::vector<float> out(n);
  for (auto &x : out)
    x = dist(rng);
  return out;
}

/** @brief A [1, 1, rows, width] tensor holding values (zero padded). */
nntrainer::Tensor
makeTensor(const std::vector<float> &values, size_t rows, size_t width,
           nntrainer::Tdatatype dtype = nntrainer::Tdatatype::FP32) {
  nntrainer::Tensor tensor(
    nntrainer::TensorDim(1, 1, rows, width, {nntrainer::Tformat::NCHW, dtype}),
    true);
  for (size_t i = 0; i < rows * width; ++i) {
    const float value = i < values.size() ? values[i] : 0.0f;
    if (dtype == nntrainer::Tdatatype::FP32)
      tensor.getData<float>()[i] = value;
    else
      tensor.getData<uint16_t>()[i] = nntrainer::compute_fp32_to_fp16(value);
  }
  return tensor;
}

/** @brief Values as stored (FP16 rounding for UINT16). */
std::vector<float> stored(const std::vector<float> &values,
                          nntrainer::Tdatatype dtype) {
  if (dtype == nntrainer::Tdatatype::FP32)
    return values;
  std::vector<float> out(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    out[i] = nntrainer::compute_fp16_to_fp32(
      nntrainer::compute_fp32_to_fp16(values[i]));
  return out;
}

/** @brief Attention probabilities P[h][j][i] (i <= j) in double. */
std::vector<double> referenceAttention(const std::vector<float> &q,
                                       const std::vector<float> &k,
                                       unsigned int t, unsigned int hq,
                                       unsigned int hkv, unsigned int d) {
  const unsigned int gqa = hq / hkv;
  std::vector<double> p(static_cast<size_t>(hq) * t * t, 0.0);
  for (unsigned int h = 0; h < hq; ++h)
    for (unsigned int j = 0; j < t; ++j) {
      std::vector<double> s(j + 1);
      double top = -1e300;
      for (unsigned int i = 0; i <= j; ++i) {
        double dot = 0.0;
        for (unsigned int x = 0; x < d; ++x)
          dot += static_cast<double>(q[(j * hq + h) * d + x]) *
                 k[(i * hkv + h / gqa) * d + x];
        s[i] = dot / std::sqrt(static_cast<double>(d));
        top = std::max(top, s[i]);
      }
      double z = 0.0;
      for (unsigned int i = 0; i <= j; ++i)
        z += std::exp(s[i] - top);
      for (unsigned int i = 0; i <= j; ++i)
        p[(static_cast<size_t>(h) * t + j) * t + i] = std::exp(s[i] - top) / z;
    }
  return p;
}

/** @brief L2 norm of row i of KV head g. */
double headNorm(const std::vector<float> &x, unsigned int i, unsigned int g,
                unsigned int hkv, unsigned int d) {
  double sum = 0.0;
  for (unsigned int c = 0; c < d; ++c) {
    const double v = x[(i * hkv + g) * d + c];
    sum += v * v;
  }
  return std::sqrt(sum);
}

/** @brief Softmax over (score + bias) then the expectation of v (d = 1). */
double biasedScalarAttention(const std::vector<double> &score,
                             const std::vector<double> &bias,
                             const std::vector<double> &v) {
  double top = -1e300;
  for (size_t i = 0; i < score.size(); ++i)
    top = std::max(top, score[i] + bias[i]);
  double num = 0.0, z = 0.0;
  for (size_t i = 0; i < score.size(); ++i) {
    const double p = std::exp(score[i] + bias[i] - top);
    num += p * v[i];
    z += p;
  }
  return num / z;
}

/** @brief A configured layer state, as CausalLM::run() would set it up. */
ProbKVState makeState(ProbKVConfig config, unsigned int layer,
                      unsigned int layers, unsigned int prompt,
                      const std::string &session = "1") {
  config.validate();
  ProbKVState state;
  state.consumeProperty("probkv", config.serialize());
  state.consumeProperty("probkv_run", std::to_string(layer) + "," +
                                        std::to_string(layers) + "," +
                                        std::to_string(prompt) + "," + session);
  return state;
}

ProbKVConfig
config(std::initializer_list<std::pair<const char *, const char *>> settings) {
  ProbKVConfig c;
  for (const auto &[key, value] : settings)
    c.set(key, value);
  return c;
}

/** @brief Original row index of each compacted row of KV head g. */
std::vector<unsigned int> keptRows(const std::vector<float> &original,
                                   const nntrainer::Tensor &cache,
                                   unsigned int g, size_t kept, unsigned int t,
                                   unsigned int hkv, unsigned int d) {
  const ProbKVView view = ProbKVView::of(cache);
  std::vector<unsigned int> rows;
  for (size_t r = 0; r < kept; ++r)
    for (unsigned int i = 0; i < t; ++i) {
      bool same = true;
      for (unsigned int c = 0; c < d && same; ++c)
        same = view.at(r, g * d + c) == original[(i * hkv + g) * d + c];
      if (same) {
        rows.push_back(i);
        break;
      }
    }
  return rows;
}

} // namespace

/**
 * @brief Settings survive the property round trip; invalid ones are rejected.
 */
TEST(ProbKVTest, ConfigParseValidateRoundTrip) {
  auto c = config({{"ratio", "0.5"},
                   {"proposal", "\"h2o_h\""},
                   {"tau", "\"inf\""},
                   {"group_alpha", "[0.25,0.75]"},
                   {"seed", "7"},
                   {"mae", "true"}});
  c.validate();
  const auto text = c.serialize();
  EXPECT_EQ(text.find_first_of("= "), std::string::npos);
  const auto back = ProbKVConfig::parse(text);
  EXPECT_EQ(back.serialize(), text);
  EXPECT_TRUE(std::isinf(back.tau));
  EXPECT_EQ(back.proposal, causallm::ProbKVProposal::H2OH);
  EXPECT_EQ(back.group_alpha, (std::vector<double>{0.25, 0.75}));
  EXPECT_EQ(back.seed, 7u);
  EXPECT_TRUE(back.mae);

  auto topk = config({{"ratio", "0.9"}, {"mode", "topk"}});
  topk.validate();
  EXPECT_EQ(topk.scope, causallm::ProbKVScope::Head);
  EXPECT_TRUE(std::isinf(topk.tau));
  EXPECT_EQ(ProbKVConfig::parse(topk.serialize()).serialize(),
            topk.serialize());

  const std::vector<std::vector<std::pair<const char *, const char *>>> bad = {
    {{"mode", "sample"}},
    {{"ratio", "1"}},
    {{"ratio", "-0.1"}},
    {{"ratio", "0.5"}, {"tau", "0"}},
    {{"ratio", "0.5"}, {"mode", "topk"}, {"tau", "1"}},
    {{"ratio", "0.5"}, {"mode", "topk"}, {"scope", "global"}},
    {{"ratio", "0.5"}, {"group_alpha", "[0.6,0.6]"}},
    {{"ratio", "0.5"}, {"uniform_mix", "1.5"}},
    {{"ratio", "0.5"}, {"snapkv_kernel", "4"}},
    {{"ratio", "0.5"}, {"streaming_sink_mass", "1"}},
  };
  for (const auto &settings : bad) {
    ProbKVConfig c2;
    EXPECT_THROW(
      {
        for (const auto &[key, value] : settings)
          c2.set(key, value);
        c2.validate();
      },
      std::invalid_argument);
  }
  ProbKVConfig c3;
  EXPECT_THROW(c3.set("proposal", "foo"), std::invalid_argument);
  EXPECT_THROW(c3.set("unknown", "1"), std::invalid_argument);
  EXPECT_THROW(c3.set("ratio", "0.5x"), std::invalid_argument);
}

/**
 * @brief Fig. 1: ground truth 2.1, uncorrected 1.875, corrected ~2.18, and
 * the additive-bias softmax equals Alg. 1.
 */
TEST(ProbKVTest, Fig1ImportanceAttention) {
  // d = 1, q = 1 and k_i = log p~_i give exp(q k_i) = p~_i exactly.
  const std::vector<double> ptilde = {10, 20, 40, 10, 20};
  const std::vector<float> v = {2.0f, 3.0f, 1.0f, 5.0f, 2.0f};
  std::vector<float> k(5);
  for (size_t i = 0; i < 5; ++i)
    k[i] = static_cast<float>(std::log(ptilde[i]));
  const float q = 1.0f;
  const std::vector<double> p_t = {0.3, 0.1, 0.4, 0.2};
  std::vector<double> pi(4);
  double z = 0.0;
  for (size_t i = 0; i < 4; ++i)
    z += p_t[i] * std::fabs(v[i]);
  for (size_t i = 0; i < 4; ++i)
    pi[i] = p_t[i] * std::fabs(v[i]) / z;
  const std::vector<uint64_t> counts = {3, 0, 3, 14};
  const uint64_t m = 20;

  float out = 0.0f;
  causallm::importanceAttention(&q, k.data(), v.data(), 5, 4, 1,
                                std::vector<double>(4, 1.0), &out);
  EXPECT_NEAR(out, 2.1, 1e-5);
  causallm::importanceAttention(&q, k.data(), v.data(), 5, 4, 1,
                                {1.0, 0.0, 1.0, 1.0}, &out);
  EXPECT_NEAR(out, 1.875, 1e-5);

  std::vector<double> w(4, 0.0), bias(5, 0.0), score(5), value(5);
  for (size_t i = 0; i < 4; ++i) {
    if (counts[i]) {
      const float lw = causallm::logWeight(counts[i], m, pi[i], 1.0);
      w[i] = std::exp(static_cast<double>(lw));
      bias[i] = lw;
    } else {
      bias[i] = -std::numeric_limits<double>::infinity();
    }
  }
  causallm::importanceAttention(&q, k.data(), v.data(), 5, 4, 1, w, &out);
  EXPECT_NEAR(out, 166.5 / 76.35, 1e-4);
  for (size_t i = 0; i < 5; ++i) {
    score[i] = k[i];
    value[i] = v[i];
  }
  EXPECT_NEAR(biasedScalarAttention(score, bias, value), 166.5 / 76.35, 1e-4);
}

/**
 * @brief softmax(s + log w) with -inf for evicted rows equals Alg. 1.
 */
TEST(ProbKVTest, BiasSoftmaxMatchesAlgorithm1) {
  const unsigned int t = 40, n = 43, d = 16;
  const auto q = randomValues(d, 1);
  const auto k = randomValues(n * d, 2);
  const auto v = randomValues(n * d, 3);
  auto pi_raw = randomValues(t, 4);
  std::vector<double> pi(t);
  double sum = 0.0;
  for (unsigned int i = 0; i < t; ++i)
    sum += pi[i] = std::fabs(pi_raw[i]) + 0.01;
  for (auto &p : pi)
    p /= sum;
  std::mt19937_64 rng(5);
  const uint64_t m = 60;
  const auto counts = causallm::sampleCounts(pi.data(), t, m, rng);

  std::vector<double> w(t, 0.0);
  std::vector<float> bias(n, 0.0f);
  for (unsigned int i = 0; i < t; ++i) {
    bias[i] =
      counts[i] ? causallm::logWeight(counts[i], m, pi[i], 1.0) : kNegInf;
    w[i] = counts[i] ? std::exp(static_cast<double>(bias[i])) : 0.0;
  }
  std::vector<float> ref(d);
  causallm::importanceAttention(q.data(), k.data(), v.data(), n, t, d, w,
                                ref.data());

  // The decode kernels: scaled scores, then the bias, then softmax and V.
  std::vector<double> s(n);
  double top = -1e300;
  for (unsigned int i = 0; i < n; ++i) {
    float dot = 0.0f;
    for (unsigned int x = 0; x < d; ++x)
      dot += q[x] * k[i * d + x];
    s[i] = dot / std::sqrt(static_cast<float>(d)) + bias[i];
    top = std::max(top, s[i]);
  }
  std::vector<double> out(d, 0.0);
  double z = 0.0;
  for (unsigned int i = 0; i < n; ++i) {
    const double p = std::exp(s[i] - top);
    z += p;
    for (unsigned int x = 0; x < d; ++x)
      out[x] += p * v[i * d + x];
  }
  for (unsigned int x = 0; x < d; ++x)
    EXPECT_NEAR(out[x] / z, ref[x], 1e-5);
}

/**
 * @brief Eq. 10: tau 1 is Alg. 1, tau 2 halves the log weight, inf is none.
 */
TEST(ProbKVTest, WeightTemperature) {
  const double base = std::log(3.0 / (20 * 0.25));
  EXPECT_NEAR(causallm::logWeight(3, 20, 0.25, 1.0), base, 1e-6);
  EXPECT_NEAR(causallm::logWeight(3, 20, 0.25, 2.0), base / 2, 1e-6);
  EXPECT_NEAR(causallm::logWeight(3, 20, 0.25, 0.5), base * 2, 1e-6);
  EXPECT_EQ(
    causallm::logWeight(3, 20, 0.25, std::numeric_limits<double>::infinity()),
    0.0f);
}

/**
 * @brief Multinomial counts: total m, seed-determined, frequencies near pi.
 */
TEST(ProbKVTest, SampleCountsMultinomial) {
  const unsigned int t = 50;
  const auto raw = randomValues(t, 11);
  std::vector<double> pi(t);
  double sum = 0.0;
  for (unsigned int i = 0; i < t; ++i)
    sum += pi[i] = std::fabs(raw[i]) + 0.001;
  for (auto &p : pi)
    p /= sum;
  const uint64_t m = 1000000;
  std::mt19937_64 a(42), b(42);
  const auto c1 = causallm::sampleCounts(pi.data(), t, m, a);
  const auto c2 = causallm::sampleCounts(pi.data(), t, m, b);
  EXPECT_EQ(c1, c2);
  uint64_t total = 0;
  for (unsigned int i = 0; i < t; ++i) {
    total += c1[i];
    const double sd = std::sqrt(pi[i] * (1 - pi[i]) / m);
    EXPECT_NEAR(static_cast<double>(c1[i]) / m, pi[i], 4 * sd + 1e-9);
  }
  EXPECT_EQ(total, m);

  // A huge m with a peaked proposal stays exact.
  std::vector<double> peaked(t, 0.1 / (t - 1));
  peaked[0] = 0.9;
  const uint64_t huge = causallm::kProbKVMaxSamples;
  std::mt19937_64 c(1);
  const auto big = causallm::sampleCounts(peaked.data(), t, huge, c);
  uint64_t big_total = 0;
  for (auto x : big)
    big_total += x;
  EXPECT_EQ(big_total, huge);
}

/**
 * @brief Prop. 1 / O(1/m): the corrected estimate converges to E_p[V].
 */
TEST(ProbKVTest, SnisConvergesWithSamples) {
  const unsigned int t = 200, n = 205, d = 8;
  const auto q = randomValues(d, 21, 2.0f);
  const auto k = randomValues(n * d, 22, 2.0f);
  const auto v = randomValues(n * d, 23);
  std::vector<float> truth(d);
  causallm::importanceAttention(q.data(), k.data(), v.data(), n, t, d,
                                std::vector<double>(t, 1.0), truth.data());
  // Proposal: a uniform-mixed version of the eviction-time attention.
  std::vector<double> pi(t, 0.5 / t);
  {
    std::vector<double> s(t);
    double top = -1e300, z = 0.0;
    for (unsigned int i = 0; i < t; ++i) {
      double dot = 0.0;
      for (unsigned int x = 0; x < d; ++x)
        dot += q[x] * k[i * d + x];
      top = std::max(top, s[i] = dot / std::sqrt(static_cast<double>(d)));
    }
    for (auto &x : s)
      z += x = std::exp(x - top);
    for (unsigned int i = 0; i < t; ++i)
      pi[i] += 0.5 * s[i] / z;
  }
  // Errors are judged against the value scale; the output itself is a mean
  // of signed values and can sit near zero.
  double scale = 0.0;
  for (auto x : v)
    scale += std::fabs(x) / v.size();

  std::vector<double> mean_error;
  for (uint64_t m : {100ull, 10000ull, 1000000ull}) {
    double error = 0.0;
    for (unsigned int seed = 0; seed < 20; ++seed) {
      std::mt19937_64 rng(seed);
      const auto counts = causallm::sampleCounts(pi.data(), t, m, rng);
      std::vector<double> w(t, 0.0);
      for (unsigned int i = 0; i < t; ++i)
        if (counts[i])
          w[i] = std::exp(
            static_cast<double>(causallm::logWeight(counts[i], m, pi[i], 1.0)));
      std::vector<float> out(d);
      causallm::importanceAttention(q.data(), k.data(), v.data(), n, t, d, w,
                                    out.data());
      for (unsigned int x = 0; x < d; ++x)
        error += std::fabs(out[x] - truth[x]) / d / 20;
    }
    mean_error.push_back(error);
  }
  EXPECT_LT(mean_error[1], mean_error[0]);
  EXPECT_LT(mean_error[2], mean_error[1]);
  // O(1/sqrt(m)) RMSE: 10^4 times more samples cut the error ~100x.
  EXPECT_LT(mean_error[2], mean_error[0] / 30);
  EXPECT_LT(mean_error[2], 1e-2 * scale);
}

/**
 * @brief Sec. 5 GQA mixture: normalized and covers every head's support.
 */
TEST(ProbKVTest, GroupProposalMixture) {
  const unsigned int hq = 4, hkv = 2, t = 5;
  std::vector<double> head = {0.2, 0.2, 0.2, 0.2, 0.2, //
                              1.0, 0.0, 0.0, 0.0, 0.0, //
                              0.0, 0.5, 0.5, 0.0, 0.0, //
                              0.1, 0.1, 0.1, 0.1, 0.6};
  const auto uniform = causallm::mixGroupProposal(head, hq, hkv, t, {});
  const auto weighted =
    causallm::mixGroupProposal(head, hq, hkv, t, {0.25, 0.75});
  for (unsigned int g = 0; g < hkv; ++g) {
    double su = 0.0, sw = 0.0;
    for (unsigned int i = 0; i < t; ++i) {
      const double a = head[(2 * g) * t + i], b = head[(2 * g + 1) * t + i];
      EXPECT_NEAR(uniform[g * t + i], 0.5 * (a + b), 1e-12);
      EXPECT_NEAR(weighted[g * t + i], 0.25 * a + 0.75 * b, 1e-12);
      EXPECT_GE(weighted[g * t + i], 0.25 * a - 1e-15);
      EXPECT_GE(weighted[g * t + i], 0.75 * b - 1e-15);
      su += uniform[g * t + i];
      sw += weighted[g * t + i];
    }
    EXPECT_NEAR(su, 1.0, 1e-12);
    EXPECT_NEAR(sw, 1.0, 1e-12);
  }
  EXPECT_THROW(causallm::mixGroupProposal(head, hq, hkv, t, {1.0}),
               std::invalid_argument);
}

/**
 * @brief Eq. 25 matches Monte-Carlo, and Alg. 2 lands within 0.5 entries.
 */
TEST(ProbKVTest, ExpectedSamplesBinarySearch) {
  std::vector<std::vector<double>> proposals;
  proposals.push_back(std::vector<double>(100, 0.01));
  {
    std::vector<double> zipf(500);
    double z = 0.0;
    for (unsigned int i = 0; i < zipf.size(); ++i)
      z += zipf[i] = 1.0 / (i + 1);
    for (auto &x : zipf)
      x /= z;
    proposals.push_back(zipf);
  }
  {
    std::vector<double> peaked(50, 0.1 / 49);
    peaked[7] = 0.9;
    proposals.push_back(peaked);
  }
  for (const auto &pi : proposals) {
    const auto logs = causallm::logComplement(pi.data(), pi.size());
    auto expected = [&](uint64_t m) {
      return causallm::expectedUnique(logs.data(), logs.size(), m);
    };
    EXPECT_EQ(expected(0), 0.0);
    for (uint64_t m = 1; m < 50; ++m)
      EXPECT_GE(expected(m + 1), expected(m));
    for (double r : {0.1, 0.3, 0.5, 0.7, 0.9}) {
      const double target = (1.0 - r) * pi.size();
      const uint64_t m = causallm::solveSamples(expected, target);
      EXPECT_LE(std::fabs(expected(m) - target), 0.5)
        << "r=" << r << " t=" << pi.size();
    }
  }
  // Monte-Carlo check of Eq. 25 on the uniform proposal.
  const auto &uniform = proposals[0];
  const auto logs = causallm::logComplement(uniform.data(), uniform.size());
  const uint64_t m = causallm::solveSamples(
    [&](uint64_t x) {
      return causallm::expectedUnique(logs.data(), logs.size(), x);
    },
    50.0);
  double unique = 0.0;
  for (unsigned int seed = 0; seed < 400; ++seed) {
    std::mt19937_64 rng(seed);
    const auto c = causallm::sampleCounts(uniform.data(), 100, m, rng);
    unique += std::count_if(c.begin(), c.end(), [](uint64_t x) { return x; });
  }
  EXPECT_NEAR(unique / 400,
              causallm::expectedUnique(logs.data(), logs.size(), m), 0.01 * 50);
}

/**
 * @brief Eq. 34: one m across layers meets the mean ratio, and sharper heads
 * keep fewer entries.
 */
TEST(ProbKVTest, GlobalSamplesAcrossHeads) {
  const unsigned int layers = 3, t = 64, hq = 4, hkv = 2, d = 8, cap = 72;
  causallm::ProbKVRegistry registry;
  std::vector<ProbKVState> states;
  std::vector<nntrainer::Tensor> keys, values;
  states.reserve(layers);
  for (unsigned int l = 0; l < layers; ++l) {
    // Layer 0 has flat attention, layer 2 a sharp one.
    const float sharp = 0.2f + 2.0f * l;
    keys.push_back(
      makeTensor(randomValues(t * hkv * d, 100 + l, sharp), cap, hkv * d));
    values.push_back(
      makeTensor(randomValues(t * hkv * d, 200 + l), cap, hkv * d));
    states.push_back(
      makeState(config({{"ratio", "0.5"}, {"proposal", "h2o"}}), l, layers, t));
  }
  for (unsigned int l = 0; l < layers; ++l) {
    const auto q =
      makeTensor(randomValues(t * hq * d, 300 + l, 2.0f), t, hq * d);
    states[l].checkStep(0, t, cap, 4096);
    states[l].finishPrefill(q, keys[l], values[l], hq, hkv, d, registry);
    if (l + 1 < layers) {
      EXPECT_EQ(states[l].rows(), 0u); // waiting for the global solve
    }
  }
  EXPECT_GT(registry.lastSamples(), 0u);
  size_t kept = 0;
  for (const auto &s : states) {
    kept += s.keptTotal();
    EXPECT_EQ(s.removed(), t - s.rows());
  }
  const double effective = 1.0 - static_cast<double>(kept) / (layers * hkv * t);
  EXPECT_NEAR(effective, 0.5, 0.05);
  EXPECT_LT(states[2].keptTotal(), states[0].keptTotal());
}

/**
 * @brief App. H proposals equal an explicit softmax-matrix reference, for
 * FP32 and FP16-bit caches; the harmonic prior is a distribution.
 */
TEST(ProbKVTest, ProposalsMatchReference) {
  const unsigned int t = 12, hq = 4, hkv = 2, d = 8, gqa = hq / hkv;
  const auto prior = causallm::harmonicPrior(t);
  double total = 0.0;
  for (auto h : prior)
    total += h;
  EXPECT_NEAR(total, 1.0, 1e-12);
  EXPECT_EQ(std::max_element(prior.begin(), prior.end()) - prior.begin(),
            static_cast<long>(t - 1));

  const auto q = randomValues(t * hq * d, 31, 2.0f);
  const auto k_raw = randomValues(t * hkv * d, 32, 2.0f);
  const auto v_raw = randomValues(t * hkv * d, 33);
  for (auto dtype :
       {nntrainer::Tdatatype::FP32, nntrainer::Tdatatype::UINT16}) {
    const auto k = stored(k_raw, dtype), v = stored(v_raw, dtype);
    const auto key = makeTensor(k_raw, t, hkv * d, dtype);
    const auto value = makeTensor(v_raw, t, hkv * d, dtype);
    const auto p = referenceAttention(q, k, t, hq, hkv, d);
    for (const char *name : {"min", "min_h", "h2o", "h2o_h"}) {
      auto c = config({{"ratio", "0.5"}, {"proposal", name}});
      c.validate();
      const auto got = causallm::buildHeadProposals(
        c, q.data(), ProbKVView::of(key), ProbKVView::of(value), t, hq, hkv, d);
      const std::string kind = name;
      for (unsigned int h = 0; h < hq; ++h) {
        std::vector<double> ref(t, 0.0);
        for (unsigned int i = 0; i < t; ++i) {
          for (unsigned int j = i; j < t; ++j) {
            if ((kind == "min") && j != t - 1)
              continue;
            const double w =
              (kind == "min_h" || kind == "h2o_h") ? prior[j] : 1.0;
            ref[i] += w * p[(static_cast<size_t>(h) * t + j) * t + i];
          }
          if (kind == "min" || kind == "min_h")
            ref[i] *= headNorm(v, i, h / gqa, hkv, d);
        }
        double z = 0.0;
        for (auto x : ref)
          z += x;
        for (unsigned int i = 0; i < t; ++i)
          EXPECT_NEAR(got[h * t + i], ref[i] / z, 1e-5 * (ref[i] / z) + 1e-7)
            << kind << " h=" << h << " i=" << i;
      }
    }
  }
}

/**
 * @brief App. D proposals (K-norm, TOVA, SnapKV, StreamingLLM), and their
 * top-k is the deterministic method's selection.
 */
TEST(ProbKVTest, ExistingMethodProposals) {
  const unsigned int t = 12, hq = 4, hkv = 2, d = 8, gqa = hq / hkv;
  const auto q = randomValues(t * hq * d, 41, 2.0f);
  const auto k = randomValues(t * hkv * d, 42, 2.0f);
  const auto v = randomValues(t * hkv * d, 43);
  const auto key = makeTensor(k, t, hkv * d), value = makeTensor(v, t, hkv * d);
  const auto p = referenceAttention(q, k, t, hq, hkv, d);
  auto build = [&](ProbKVConfig c) {
    c.validate();
    return causallm::buildHeadProposals(c, q.data(), ProbKVView::of(key),
                                        ProbKVView::of(value), t, hq, hkv, d);
  };

  // K-norm: softmax(-||k||), keeps the lowest key norms.
  const auto knorm = build(config({{"ratio", "0.5"}, {"proposal", "knorm"}}));
  for (unsigned int h = 0; h < hq; ++h) {
    double z = 0.0;
    for (unsigned int i = 0; i < t; ++i)
      z += std::exp(-headNorm(k, i, h / gqa, hkv, d));
    for (unsigned int i = 0; i < t; ++i)
      EXPECT_NEAR(knorm[h * t + i],
                  std::exp(-headNorm(k, i, h / gqa, hkv, d)) / z, 1e-9);
    const auto keep = causallm::topkKeep(knorm.data() + h * t, t, 4);
    std::vector<unsigned int> order(t);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
      return headNorm(k, a, h / gqa, hkv, d) < headNorm(k, b, h / gqa, hkv, d);
    });
    order.resize(4);
    std::sort(order.begin(), order.end());
    EXPECT_EQ(keep, order);
  }

  // TOVA: the last query's distribution averaged over every head.
  const auto tova = build(config({{"ratio", "0.5"}, {"proposal", "tova"}}));
  for (unsigned int i = 0; i < t; ++i) {
    double mean = 0.0;
    for (unsigned int h = 0; h < hq; ++h)
      mean += p[(static_cast<size_t>(h) * t + t - 1) * t + i] / hq;
    for (unsigned int h = 0; h < hq; ++h)
      EXPECT_NEAR(tova[h * t + i], mean, 1e-6);
  }

  // SnapKV: window mean, avg pool, window keys at the max.
  const unsigned int window = 4, kernel = 3, prefix = t - window;
  const auto snap = build(config({{"ratio", "0.5"},
                                  {"proposal", "snapkv"},
                                  {"snapkv_window", "4"},
                                  {"snapkv_kernel", "3"}}));
  for (unsigned int h = 0; h < hq; ++h) {
    std::vector<double> mean(prefix, 0.0), ref(t);
    for (unsigned int i = 0; i < prefix; ++i)
      for (unsigned int j = prefix; j < t; ++j)
        mean[i] += p[(static_cast<size_t>(h) * t + j) * t + i] / window;
    double best = 0.0;
    for (int i = 0; i < static_cast<int>(prefix); ++i) {
      double acc = 0.0;
      for (int o = -1; o <= 1; ++o)
        if (i + o >= 0 && i + o < static_cast<int>(prefix))
          acc += mean[i + o];
      ref[i] = acc / kernel;
      best = std::max(best, ref[i]);
    }
    for (unsigned int i = prefix; i < t; ++i)
      ref[i] = best;
    double z = 0.0;
    for (auto x : ref)
      z += x;
    for (unsigned int i = 0; i < t; ++i)
      EXPECT_NEAR(snap[h * t + i], ref[i] / z, 1e-6);
  }

  // StreamingLLM: sinks + a monotone ramp; top-k = sinks and the last entries.
  const auto stream =
    build(config({{"ratio", "0.5"}, {"proposal", "streaming"}}));
  for (unsigned int i = 0; i < t; ++i)
    EXPECT_NEAR(stream[i], 0.5 * (i + 1) / 78.0 + (i < 4 ? 0.125 : 0.0), 1e-12);
  EXPECT_EQ(causallm::topkKeep(stream.data(), t, 6),
            (std::vector<unsigned int>{0, 1, 2, 3, 10, 11}));
}

/**
 * @brief App. D.1: proposal temperature and uniform mixture.
 */
TEST(ProbKVTest, ProposalAdjustment) {
  const unsigned int t = 12, hq = 4, hkv = 2, d = 8;
  const auto q = randomValues(t * hq * d, 51, 2.0f);
  const auto k = randomValues(t * hkv * d, 52, 2.0f);
  const auto v = randomValues(t * hkv * d, 53);
  const auto key = makeTensor(k, t, hkv * d), value = makeTensor(v, t, hkv * d);
  const auto p = referenceAttention(q, k, t, hq, hkv, d);

  auto c = config(
    {{"ratio", "0.5"}, {"proposal", "h2o"}, {"proposal_temperature", "0.5"}});
  c.validate();
  const auto tempered = causallm::buildHeadProposals(
    c, q.data(), ProbKVView::of(key), ProbKVView::of(value), t, hq, hkv, d);
  for (unsigned int h = 0; h < hq; ++h) {
    std::vector<double> score(t, 0.0);
    for (unsigned int j = 0; j < t; ++j)
      for (unsigned int i = 0; i <= j; ++i)
        score[i] += p[(static_cast<size_t>(h) * t + j) * t + i];
    double z = 0.0;
    for (auto s : score)
      z += std::exp(s / 0.5);
    for (unsigned int i = 0; i < t; ++i)
      EXPECT_NEAR(tempered[h * t + i], std::exp(score[i] / 0.5) / z, 1e-6);
  }

  auto plain = config({{"ratio", "0.5"}, {"proposal", "h2o"}});
  auto mixed =
    config({{"ratio", "0.5"}, {"proposal", "h2o"}, {"uniform_mix", "0.6"}});
  plain.validate();
  mixed.validate();
  const auto a = causallm::buildHeadProposals(
    plain, q.data(), ProbKVView::of(key), ProbKVView::of(value), t, hq, hkv, d);
  const auto b = causallm::buildHeadProposals(
    mixed, q.data(), ProbKVView::of(key), ProbKVView::of(value), t, hq, hkv, d);
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(b[i], 0.6 * a[i] + 0.4 / t, 1e-12);
    EXPECT_GE(b[i], 0.4 / t - 1e-15);
  }
}

/**
 * @brief Top-k baseline: k = floor(t (1 - r)), ties to the lower index, and
 * a uniform k needs no decode bias.
 */
TEST(ProbKVTest, TopKBaseline) {
  const std::vector<double> score = {0.1, 0.3, 0.3, 0.2, 0.3};
  EXPECT_EQ(causallm::topkKeep(score.data(), 5, 2),
            (std::vector<unsigned int>{1, 2}));
  EXPECT_EQ(causallm::topkKeep(score.data(), 5, 9).size(), 5u);

  const unsigned int t = 30, hq = 4, hkv = 2, d = 8, cap = 34;
  auto key = makeTensor(randomValues(t * hkv * d, 61), cap, hkv * d);
  auto value = makeTensor(randomValues(t * hkv * d, 62), cap, hkv * d);
  const auto q = makeTensor(randomValues(t * hq * d, 63), t, hq * d);
  causallm::ProbKVRegistry registry;
  auto state = makeState(
    config({{"ratio", "0.75"}, {"mode", "topk"}, {"proposal", "h2o"}}), 0, 1,
    t);
  state.checkStep(0, t, cap, 4096);
  state.finishPrefill(q, key, value, hq, hkv, d, registry);
  EXPECT_EQ(state.rows(), 7u); // floor(30 * 0.25)
  EXPECT_EQ(state.removed(), 23u);
  EXPECT_EQ(state.biasRows(), 0u);
  EXPECT_EQ(state.keptTotal(), 14u);
}

/**
 * @brief Per-head packing, -inf padding, and decode attention through
 * addBias() equal Alg. 1 on the kept sets plus the decode row.
 */
TEST(ProbKVTest, CompactionAndBiasLayout) {
  const unsigned int t = 32, hq = 4, hkv = 2, d = 8, gqa = hq / hkv, cap = 40;
  for (auto dtype :
       {nntrainer::Tdatatype::FP32, nntrainer::Tdatatype::UINT16}) {
    const auto k_raw = randomValues(t * hkv * d, 71, 2.0f);
    const auto v_raw = randomValues(t * hkv * d, 72);
    const auto k = stored(k_raw, dtype), v = stored(v_raw, dtype);
    auto key = makeTensor(k_raw, cap, hkv * d, dtype);
    auto value = makeTensor(v_raw, cap, hkv * d, dtype);
    const auto q = makeTensor(randomValues(t * hq * d, 73, 2.0f), t, hq * d);
    causallm::ProbKVRegistry registry;
    auto state = makeState(
      config({{"ratio", "0.5"}, {"scope", "head"}, {"proposal", "h2o"}}), 0, 1,
      t);
    state.checkStep(0, t, cap, 4096);
    state.finishPrefill(q, key, value, hq, hkv, d, registry);

    const unsigned int rows = state.rows();
    ASSERT_GT(rows, 0u);
    ASSERT_LT(rows, t);
    EXPECT_EQ(state.removed(), t - rows);
    EXPECT_EQ(state.biasRows(), rows); // corrected sampling always biases

    // A decode token at position t lands on physical row R.
    EXPECT_NO_THROW(state.checkStep(t, 1, cap, 4096));
    EXPECT_THROW(state.checkStep(t + 2, 1, cap, 4096), std::invalid_argument);
    const auto k_new = randomValues(hkv * d, 74, 2.0f);
    const auto v_new = randomValues(hkv * d, 75);
    const auto k_new_s = stored(k_new, dtype), v_new_s = stored(v_new, dtype);
    for (unsigned int c = 0; c < hkv * d; ++c) {
      if (dtype == nntrainer::Tdatatype::FP32) {
        key.getData<float>()[rows * hkv * d + c] = k_new[c];
        value.getData<float>()[rows * hkv * d + c] = v_new[c];
      } else {
        key.getData<uint16_t>()[rows * hkv * d + c] =
          nntrainer::compute_fp32_to_fp16(k_new[c]);
        value.getData<uint16_t>()[rows * hkv * d + c] =
          nntrainer::compute_fp32_to_fp16(v_new[c]);
      }
    }
    const auto qd = randomValues(hq * d, 76, 2.0f);
    const ProbKVView kv = ProbKVView::of(key), vv = ProbKVView::of(value);

    // Per-row scores [row * Hq + h] for rows [0, R], then the bias.
    const unsigned int n = rows + 1;
    std::vector<float> scores(n * hq);
    for (unsigned int r = 0; r < n; ++r)
      for (unsigned int h = 0; h < hq; ++h) {
        float dot = 0.0f;
        for (unsigned int x = 0; x < d; ++x)
          dot += qd[h * d + x] * kv.at(r, (h / gqa) * d + x);
        scores[r * hq + h] = dot / std::sqrt(static_cast<float>(d));
      }
    state.addBias(scores.data(), n, hq, gqa);

    size_t kept_sum = 0;
    for (unsigned int g = 0; g < hkv; ++g) {
      size_t kept = 0;
      while (kept < rows && state.bias(g)[kept] != kNegInf)
        ++kept;
      for (size_t r = kept; r < rows; ++r)
        EXPECT_EQ(state.bias(g)[r], kNegInf);
      kept_sum += kept;
      const auto original = keptRows(k, key, g, kept, t, hkv, d);
      ASSERT_EQ(original.size(), kept);
      EXPECT_TRUE(std::is_sorted(original.begin(), original.end()));

      for (unsigned int h = g * gqa; h < (g + 1) * gqa; ++h) {
        // Alg. 1 on the original cache: w = exp(bias) on kept, 0 elsewhere.
        std::vector<float> kk((t + 1) * d), vk((t + 1) * d);
        for (unsigned int i = 0; i < t; ++i)
          for (unsigned int x = 0; x < d; ++x) {
            kk[i * d + x] = k[(i * hkv + g) * d + x];
            vk[i * d + x] = v[(i * hkv + g) * d + x];
          }
        for (unsigned int x = 0; x < d; ++x) {
          kk[t * d + x] = k_new_s[g * d + x];
          vk[t * d + x] = v_new_s[g * d + x];
        }
        std::vector<double> w(t, 0.0);
        for (size_t r = 0; r < kept; ++r)
          w[original[r]] = std::exp(static_cast<double>(state.bias(g)[r]));
        std::vector<float> ref(d);
        causallm::importanceAttention(qd.data() + h * d, kk.data(), vk.data(),
                                      t + 1, t, d, w, ref.data());

        double top = -1e300;
        for (unsigned int r = 0; r < n; ++r)
          top = std::max(top, static_cast<double>(scores[r * hq + h]));
        std::vector<double> out(d, 0.0);
        double z = 0.0;
        for (unsigned int r = 0; r < n; ++r) {
          const double e = std::exp(scores[r * hq + h] - top);
          z += e;
          for (unsigned int x = 0; x < d; ++x)
            out[x] += e * vv.at(r, g * d + x);
        }
        for (unsigned int x = 0; x < d; ++x)
          EXPECT_NEAR(out[x] / z, ref[x], 1e-4);
      }
    }
    EXPECT_EQ(kept_sum, state.keptTotal());
  }
}

/**
 * @brief releaseRows() keeps the rows before the cut and frees whole pages.
 */
TEST(ProbKVTest, ReleaseRowsKeepsKeptData) {
  const size_t rows = 4096, width = 64;
  const auto values = randomValues(rows * width, 81);
  auto cache = makeTensor(values, rows, width);
  const size_t released = causallm::releaseRows(cache, 100);
  for (size_t i = 0; i < 100 * width; ++i)
    ASSERT_EQ(cache.getData<float>()[i], values[i]);
#if defined(__linux__) || defined(__ANDROID__)
  EXPECT_GT(released, 0u);
  // The last row lies in a released page and reads as zero; writes fault in.
  EXPECT_EQ(cache.getData<float>()[(rows - 1) * width], 0.0f);
  cache.getData<float>()[(rows - 1) * width] = 3.0f;
  EXPECT_EQ(cache.getData<float>()[(rows - 1) * width], 3.0f);
#else
  (void)released;
#endif
}

/**
 * @brief Sec. 6 MAE diagnostic equals Eq. 11 computed independently, and
 * without correction the corrected error is the uncorrected one.
 */
TEST(ProbKVTest, MaeDiagnostic) {
  const unsigned int t = 48, hq = 4, hkv = 2, d = 8, gqa = hq / hkv, cap = 52;
  for (const char *tau : {"1", "\"inf\""}) {
    const auto k = randomValues(t * hkv * d, 91, 2.0f);
    const auto v = randomValues(t * hkv * d, 92);
    auto key = makeTensor(k, cap, hkv * d);
    auto value = makeTensor(v, cap, hkv * d);
    const auto q = makeTensor(randomValues(t * hq * d, 93, 2.0f), t, hq * d);
    causallm::ProbKVRegistry registry;
    auto state = makeState(config({{"ratio", "0.3"},
                                   {"scope", "head"},
                                   {"proposal", "h2o"},
                                   {"tau", tau},
                                   {"mae", "true"}}),
                           0, 1, t);
    state.checkStep(0, t, cap, 4096);
    state.finishPrefill(q, key, value, hq, hkv, d, registry);
    ASSERT_TRUE(state.maeArmed());
    const unsigned int rows = state.rows();
    const auto k_new = randomValues(hkv * d, 94, 2.0f);
    const auto v_new = randomValues(hkv * d, 95);
    std::copy(k_new.begin(), k_new.end(),
              key.getData<float>() + rows * hkv * d);
    std::copy(v_new.begin(), v_new.end(),
              value.getData<float>() + rows * hkv * d);
    const auto qd_values = randomValues(hq * d, 96, 2.0f);
    const auto qd = makeTensor(qd_values, 1, hq * d);

    // Independent Eq. 11 from the original rows.
    const auto &pi = state.groupProposals();
    double unc = 0.0, cor = 0.0, top = 0.0;
    for (unsigned int g = 0; g < hkv; ++g) {
      size_t kept = 0;
      while (kept < rows && state.bias(g)[kept] != kNegInf)
        ++kept;
      const auto keep = keptRows(k, key, g, kept, t, hkv, d);
      ASSERT_EQ(keep.size(), kept);
      const auto topk =
        causallm::topkKeep(pi.data() + g * t, t, static_cast<unsigned>(kept));
      std::vector<float> kk((t + 1) * d), vk((t + 1) * d);
      for (unsigned int i = 0; i < t; ++i)
        for (unsigned int x = 0; x < d; ++x) {
          kk[i * d + x] = k[(i * hkv + g) * d + x];
          vk[i * d + x] = v[(i * hkv + g) * d + x];
        }
      for (unsigned int x = 0; x < d; ++x) {
        kk[t * d + x] = k_new[g * d + x];
        vk[t * d + x] = v_new[g * d + x];
      }
      std::vector<double> ones(t, 1.0), wu(t, 0.0), wc(t, 0.0), wt(t, 0.0);
      for (size_t r = 0; r < kept; ++r) {
        wu[keep[r]] = 1.0;
        wc[keep[r]] = std::exp(static_cast<double>(state.bias(g)[r]));
        wt[topk[r]] = 1.0;
      }
      for (unsigned int h = g * gqa; h < (g + 1) * gqa; ++h) {
        std::vector<float> o(d), ou(d), oc(d), ot(d);
        const float *qh = qd_values.data() + h * d;
        causallm::importanceAttention(qh, kk.data(), vk.data(), t + 1, t, d,
                                      ones, o.data());
        causallm::importanceAttention(qh, kk.data(), vk.data(), t + 1, t, d, wu,
                                      ou.data());
        causallm::importanceAttention(qh, kk.data(), vk.data(), t + 1, t, d, wc,
                                      oc.data());
        causallm::importanceAttention(qh, kk.data(), vk.data(), t + 1, t, d, wt,
                                      ot.data());
        for (unsigned int x = 0; x < d; ++x) {
          unc += std::fabs(o[x] - ou[x]) / d / hq;
          cor += std::fabs(o[x] - oc[x]) / d / hq;
          top += std::fabs(o[x] - ot[x]) / d / hq;
        }
      }
    }
    const auto got = state.maeReport(qd, key, value);
    EXPECT_FALSE(state.maeArmed());
    EXPECT_NEAR(got[0], unc, 1e-5);
    EXPECT_NEAR(got[1], cor, 1e-5);
    EXPECT_NEAR(got[2], top, 1e-5);
    if (std::string(tau) != "1") {
      EXPECT_NEAR(got[1], got[0], 1e-7);
    }
  }
}
