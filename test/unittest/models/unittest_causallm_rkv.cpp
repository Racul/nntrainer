// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_causallm_rkv.cpp
 * @date   18 November 2026
 * @brief  Unit tests for R-KV redundancy-aware KV cache compression
 * @see    Cai et al., "R-KV: Redundancy-aware KV Cache Compression for
 *         Reasoning Models", NeurIPS 2025, arXiv 2505.24133
 * @bug    No known bugs except for NYI items
 */

#include <causallm_test_utils.h>
#include <qwen3_causallm.h>
#include <rkv.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using causallm::RKVConfig;
using causallm::RKVGqaPool;
using causallm::RKVHeadSelect;
using causallm::RKVRetainAxis;
using causallm::RKVState;
using causallm::RKVView;

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

/** @brief A view over host values laid out as [rows, width] FP32. */
RKVView viewOf(const std::vector<float> &values, size_t width) {
  RKVView view;
  view.data = reinterpret_cast<const unsigned char *>(values.data());
  view.dtype = nntrainer::Tdatatype::FP32;
  view.width = width;
  return view;
}

/** @brief Element of a [rows, width] cache tensor as FP32. */
float cacheAt(const nntrainer::Tensor &tensor, size_t row, size_t col) {
  return RKVView::of(tensor).at(row, col);
}

/** @brief F.max_pool1d(kernel, stride 1, padding kernel // 2) in double. */
std::vector<double> referencePool(const std::vector<double> &in,
                                  unsigned int kernel) {
  const int n = static_cast<int>(in.size());
  const int half = static_cast<int>(kernel / 2);
  std::vector<double> out(n);
  for (int c = 0; c < n; ++c) {
    double top = -std::numeric_limits<double>::infinity();
    for (int i = c - half; i <= c + half; ++i)
      if (i >= 0 && i < n)
        top = std::max(top, in[i]);
    out[c] = top;
  }
  return out;
}

/**
 * @brief r1_kv.py importance with compute_attention_scores() in double:
 * explicit logits, group max/mean, softmax over the candidates, mean over
 * the observations, then the max-pool. q: [a][Hq * D], k: [rows][Hkv * D].
 */
std::vector<double> referenceImportance(const std::vector<float> &q,
                                        const std::vector<float> &k,
                                        unsigned int a, unsigned int heads_q,
                                        unsigned int heads_kv, unsigned int dim,
                                        unsigned int cand, RKVGqaPool pool,
                                        unsigned int kernel) {
  const unsigned int gqa = heads_q / heads_kv;
  std::vector<double> out;
  for (unsigned int g = 0; g < heads_kv; ++g) {
    std::vector<double> mean(cand, 0.0);
    for (unsigned int s = 0; s < a; ++s) {
      std::vector<double> logit(cand);
      for (unsigned int c = 0; c < cand; ++c) {
        double pooled = pool == RKVGqaPool::Max
                          ? -std::numeric_limits<double>::infinity()
                          : 0.0;
        for (unsigned int r = 0; r < gqa; ++r) {
          const unsigned int h = g * gqa + r;
          double dot = 0.0;
          for (unsigned int d = 0; d < dim; ++d)
            dot += static_cast<double>(q[(s * heads_q + h) * dim + d]) *
                   k[(c * heads_kv + g) * dim + d];
          dot /= std::sqrt(static_cast<double>(dim));
          pooled = pool == RKVGqaPool::Max ? std::max(pooled, dot)
                                           : pooled + dot / gqa;
        }
        logit[c] = pooled;
      }
      const double top = *std::max_element(logit.begin(), logit.end());
      double total = 0.0;
      for (auto &v : logit) {
        v = std::exp(v - top);
        total += v;
      }
      for (unsigned int c = 0; c < cand; ++c)
        mean[c] += logit[c] / total / a;
    }
    const auto pooled = referencePool(mean, kernel);
    out.insert(out.end(), pooled.begin(), pooled.end());
  }
  return out;
}

/**
 * @brief cal_similarity() ported as is, with the n x n matrix. beta
 * generalizes retain_direction="last"; the paper axis drops the retained
 * entries from token i's own row mean (Eq. 6, Alg. 1).
 */
std::vector<double> referenceRedundancy(const std::vector<float> &k,
                                        unsigned int n, unsigned int heads_kv,
                                        unsigned int dim, float threshold,
                                        unsigned int beta, RKVRetainAxis axis) {
  std::vector<double> out;
  for (unsigned int g = 0; g < heads_kv; ++g) {
    std::vector<std::vector<float>> unit(n, std::vector<float>(dim));
    for (unsigned int i = 0; i < n; ++i) {
      double squares = 0.0;
      for (unsigned int d = 0; d < dim; ++d) {
        const double v = k[(i * heads_kv + g) * dim + d];
        squares += v * v;
      }
      const float den = static_cast<float>(std::sqrt(squares)) + 1e-8f;
      for (unsigned int d = 0; d < dim; ++d)
        unit[i][d] = k[(i * heads_kv + g) * dim + d] / den;
    }
    std::vector<std::vector<double>> sim(n, std::vector<double>(n, 0.0));
    for (unsigned int i = 0; i < n; ++i)
      for (unsigned int j = 0; j < n; ++j) {
        if (i == j)
          continue;
        double dot = 0.0;
        for (unsigned int d = 0; d < dim; ++d)
          dot += static_cast<double>(unit[i][d]) * unit[j][d];
        sim[i][j] = dot;
      }
    auto zeroed = sim;
    std::vector<double> mean(n, 0.0);
    for (unsigned int i = 0; i < n; ++i) {
      std::vector<unsigned int> retained;
      for (unsigned int j = n; j-- > 0 && retained.size() < beta;)
        if (sim[i][j] > threshold)
          retained.push_back(j);
      if (axis == RKVRetainAxis::Code) {
        if (retained.empty())
          zeroed[i][0] = 0.0;
        for (auto j : retained)
          zeroed[i][j] = 0.0;
      } else {
        double row = 0.0;
        for (unsigned int j = 0; j < n; ++j)
          row += sim[i][j];
        for (auto j : retained)
          row -= sim[i][j];
        mean[i] = row / n;
      }
    }
    if (axis == RKVRetainAxis::Code)
      for (unsigned int j = 0; j < n; ++j) {
        double column = 0.0;
        for (unsigned int i = 0; i < n; ++i)
          column += zeroed[i][j];
        mean[j] = column / n;
      }
    const double top = *std::max_element(mean.begin(), mean.end());
    double total = 0.0;
    for (auto &v : mean) {
      v = std::exp(v - top);
      total += v;
    }
    for (auto v : mean)
      out.push_back(v / total);
  }
  return out;
}

/** @brief Top-count of z (ties: smaller index), ascending, plus [cand, n). */
std::vector<unsigned int> referenceTop(const std::vector<double> &z,
                                       unsigned int count, unsigned int n) {
  std::vector<unsigned int> order(z.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(),
                   [&](unsigned int x, unsigned int y) { return z[x] > z[y]; });
  order.resize(count);
  std::sort(order.begin(), order.end());
  for (unsigned int r = static_cast<unsigned int>(z.size()); r < n; ++r)
    order.push_back(r);
  return order;
}

/** @brief The 32-bit LCG of rkv_golden.py: small integers / 8. */
class GoldenLcg {
public:
  explicit GoldenLcg(uint32_t seed) : state_(seed) {}
  float value() {
    state_ = state_ * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int>((state_ >> 16) % 17) - 8) / 8.0f;
  }

private:
  uint32_t state_;
};

/* Golden values of the official code (R-KV a329f295, HuggingFace/rkv) in
 * FP32 on the CPU, from rkv_golden.py (torch 2.11.0): Hq 4, Hkv 2, D 8,
 * n 24, window 4, budget 12, kernel 7, lambda 0.1, R1KV(retain "last").
 * The same script on CUDA agrees within 7.5e-9. */
constexpr unsigned int kGHq = 4, kGHkv = 2, kGD = 8, kGN = 24, kGA = 4,
                       kGBudget = 12, kGKernel = 7;
constexpr unsigned int kGCand = kGN - kGA;

const float kGoldenImportanceMax[] = {
  0.0551692732f, 0.0583490841f, 0.0583490841f, 0.0583490841f, 0.0583490841f,
  0.0583490841f, 0.0583490841f, 0.0583490841f, 0.0565815084f, 0.0565815084f,
  0.0565815084f, 0.0565815084f, 0.0565815084f, 0.060087014f,  0.060087014f,
  0.060087014f,  0.060087014f,  0.060087014f,  0.060087014f,  0.060087014f,
  0.0559940264f, 0.0559940264f, 0.0559940264f, 0.0559940264f, 0.0559940264f,
  0.0559940264f, 0.0527123064f, 0.0527123064f, 0.0527123064f, 0.0527123064f,
  0.0573234931f, 0.0573234931f, 0.0577425994f, 0.0577425994f, 0.0657076612f,
  0.0657076612f, 0.0657076612f, 0.0657076612f, 0.0657076612f, 0.0657076612f,
};
const float kGoldenImportanceMean[] = {
  0.0535249971f, 0.0593309328f, 0.0593309328f, 0.0593309328f, 0.0593309328f,
  0.0593309328f, 0.0593309328f, 0.0593309328f, 0.0622119457f, 0.0622119457f,
  0.0622119457f, 0.0622119457f, 0.0622119457f, 0.0622119457f, 0.0622119457f,
  0.0566594489f, 0.0566594489f, 0.0554884374f, 0.0554884374f, 0.0533123314f,
  0.05370792f,   0.05370792f,   0.05370792f,   0.05370792f,   0.05370792f,
  0.05370792f,   0.05370792f,   0.05370792f,   0.05370792f,   0.05370792f,
  0.0547742099f, 0.0547742099f, 0.0555340908f, 0.0555340908f, 0.0611305274f,
  0.0611305274f, 0.0611305274f, 0.0611305274f, 0.0611305274f, 0.0611305274f,
};
const float kGoldenRedundancy05[] = {
  0.0435269512f, 0.041384317f,  0.0415204167f, 0.0396998413f, 0.0395892821f,
  0.0404793732f, 0.0436812788f, 0.0413173363f, 0.0425155573f, 0.0396998413f,
  0.0443170033f, 0.0434505418f, 0.0401273854f, 0.0396198221f, 0.0421963483f,
  0.0432353802f, 0.0441488139f, 0.0448784493f, 0.0379601382f, 0.039081879f,
  0.0402749851f, 0.0415204167f, 0.040987879f,  0.0447867103f, 0.0406128541f,
  0.0388536528f, 0.0429882184f, 0.0426631607f, 0.0483990796f, 0.0471486859f,
  0.0376888439f, 0.0364554226f, 0.0406221263f, 0.0381349474f, 0.0468675531f,
  0.0388986431f, 0.0471486859f, 0.0428324975f, 0.0359083563f, 0.0415729694f,
  0.0449762829f, 0.0379158594f, 0.0445442796f, 0.0366689898f, 0.0393390842f,
  0.0436833687f, 0.0480904207f, 0.0379860327f,
};
const float kGoldenRedundancy02[] = {
  0.0439046659f, 0.0419486761f, 0.0429203734f, 0.0410384126f, 0.040135216f,
  0.0401365422f, 0.0444079079f, 0.0419788212f, 0.0421554819f, 0.0410384126f,
  0.0439416729f, 0.0450443998f, 0.0406746008f, 0.0401598439f, 0.0413321964f,
  0.0439416729f, 0.0457691029f, 0.0436020233f, 0.0376941599f, 0.0381985828f,
  0.0391319133f, 0.0395653322f, 0.0415511355f, 0.0397288539f, 0.0399990678f,
  0.0384659804f, 0.0435079336f, 0.0436518304f, 0.0490160696f, 0.0466782451f,
  0.0373127945f, 0.0384659804f, 0.0417997837f, 0.0389602855f, 0.0479814559f,
  0.0398578569f, 0.0466782451f, 0.0424051248f, 0.036685504f,  0.04190211f,
  0.0445275195f, 0.037743438f,  0.0466782451f, 0.0368309654f, 0.0378775187f,
  0.0427494831f, 0.0438031331f, 0.0364213772f,
};
const float kGoldenScore[] = {
  -0.0336573273f, -0.0314109772f, -0.0315334685f, -0.0298949461f,
  -0.0297954436f, -0.0305965263f, -0.0334782414f, -0.0313506946f,
  -0.0326058492f, -0.0300717037f, -0.0342271514f, -0.0334473364f,
  -0.0304564927f, -0.0296491366f, -0.0319680125f, -0.0329031423f,
  -0.0337252319f, -0.0343819037f, -0.0281554218f, -0.0291649904f,
  -0.0309521668f, -0.0293688849f, -0.0330899954f, -0.0327974409f,
  -0.0379597694f, -0.036834415f,  -0.0286487304f, -0.0275386497f,
  -0.0312886834f, -0.0290502235f, -0.0364484452f, -0.0292764269f,
  -0.0366595574f, -0.0327749886f, -0.0257467534f, -0.0308449063f,
  -0.0339078866f, -0.0275535081f, -0.0335190818f, -0.026431324f,
};
const std::vector<std::vector<unsigned int>> kGoldenKeepR1KV = {
  {3, 4, 5, 9, 12, 13, 18, 19, 20, 21, 22, 23},
  {1, 6, 7, 9, 11, 14, 17, 19, 20, 21, 22, 23},
};
/* Head 0 ties at the boundary (max-pool plateau); torch kept the smallest
 * index there, as rkvSelect does. */
const std::vector<std::vector<unsigned int>> kGoldenKeepSnapKV = {
  {1, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
  {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
};

/** @brief Golden inputs: q [window][Hq * D], k and v [n][Hkv * D]. */
struct GoldenInputs {
  std::vector<float> q, k, v;
};

GoldenInputs goldenInputs() {
  GoldenLcg rng(2026);
  GoldenInputs in;
  in.q.resize(kGA * kGHq * kGD);
  in.k.resize(kGN * kGHkv * kGD);
  in.v.resize(kGN * kGHkv * kGD);
  for (unsigned int s = 0; s < kGA; ++s)
    for (unsigned int h = 0; h < kGHq; ++h)
      for (unsigned int d = 0; d < kGD; ++d)
        in.q[(s * kGHq + h) * kGD + d] = rng.value();
  for (auto *data : {&in.k, &in.v})
    for (unsigned int r = 0; r < kGN; ++r)
      for (unsigned int g = 0; g < kGHkv; ++g)
        for (unsigned int d = 0; d < kGD; ++d)
          (*data)[(r * kGHkv + g) * kGD + d] = rng.value();
  const unsigned int copies[][3] = {{0, 9, 3}, {0, 15, 10}, {0, 21, 2},
                                    {1, 7, 1}, {1, 12, 5},  {1, 18, 12}};
  for (const auto &c : copies)
    for (unsigned int d = 0; d < kGD; ++d)
      in.k[(c[1] * kGHkv + c[0]) * kGD + d] =
        in.k[(c[2] * kGHkv + c[0]) * kGD + d];
  return in;
}

/** @brief Serialized config from JSON-style settings. */
std::string
configOf(const std::vector<std::pair<std::string, std::string>> &settings) {
  RKVConfig config;
  for (const auto &item : settings)
    config.set(item.first, item.second);
  config.validate();
  return config.serialize();
}

/**
 * @brief A fake layer cache driven like mha_core: K/V of each step are
 * written at the physical row (position - removed) before the attention,
 * then afterAttention() runs. V encodes (position, head) so every physical
 * row can be traced back.
 */
struct FakeLayer {
  unsigned int heads_q, heads_kv, dim, capacity;
  nntrainer::Tensor key, value;
  RKVState state;
  unsigned int max_rows = 0;

  FakeLayer(unsigned int hq, unsigned int hkv, unsigned int d, unsigned int cap,
            nntrainer::Tdatatype dtype) :
    heads_q(hq),
    heads_kv(hkv),
    dim(d),
    capacity(cap),
    key(makeTensor({}, cap, hkv * d, dtype)),
    value(makeTensor({}, cap, hkv * d, dtype)) {}

  static float keyValue(unsigned int pos, unsigned int g, unsigned int d) {
    GoldenLcg rng(pos * 131u + g * 17u + d);
    return rng.value() + (d == 0 ? 2.0f : 0.0f);
  }

  void step(unsigned int from, unsigned int step, const std::vector<float> &q) {
    state.checkStep(from, step, capacity, 1u << 20);
    const size_t row = from - state.removed();
    for (unsigned int s = 0; s < step; ++s)
      for (unsigned int g = 0; g < heads_kv; ++g)
        for (unsigned int d = 0; d < dim; ++d) {
          const size_t at = (row + s) * heads_kv * dim + g * dim + d;
          const float kv = keyValue(from + s, g, d);
          const float vv =
            d == 0 ? static_cast<float>(from + s) : static_cast<float>(g);
          if (key.getDataType() == nntrainer::Tdatatype::FP32) {
            key.getData<float>()[at] = kv;
            value.getData<float>()[at] = vv;
          } else {
            key.getData<uint16_t>()[at] = nntrainer::compute_fp32_to_fp16(kv);
            value.getData<uint16_t>()[at] = nntrainer::compute_fp32_to_fp16(vv);
          }
        }
    if (from)
      max_rows = std::max<unsigned int>(max_rows, row + step);
    const auto qt = makeTensor(q, step, heads_q * dim);
    state.afterAttention(qt, key, value, heads_q, heads_kv, dim, from, step);
  }

  /** @brief Every physical row holds the K/V of its recorded position. */
  void expectRowsMatchPositions() const {
    const auto &pos = state.positions();
    ASSERT_EQ(pos.size(), heads_kv);
    for (unsigned int g = 0; g < heads_kv; ++g) {
      ASSERT_EQ(pos[g].size(), state.rows());
      for (unsigned int r = 0; r < state.rows(); ++r) {
        if (r) {
          ASSERT_LT(pos[g][r - 1], pos[g][r]);
        }
        ASSERT_EQ(cacheAt(value, r, g * dim), static_cast<float>(pos[g][r]))
          << "head " << g << " row " << r;
        ASSERT_EQ(cacheAt(value, r, g * dim + 1), static_cast<float>(g));
        for (unsigned int d = 0; d < dim; ++d)
          ASSERT_EQ(cacheAt(key, r, g * dim + d), keyValue(pos[g][r], g, d));
      }
    }
  }
};

} // namespace

/**
 * @brief Settings round-trip through the layer property and bad settings are
 * rejected.
 */
TEST(RKVTest, ConfigParseValidateRoundTrip) {
  RKVConfig config;
  config.set("ratio", "0.9");
  config.set("window", "16");
  config.set("buffer", "64");
  config.set("kernel", "5");
  config.set("gqa_pool", "\"mean\"");
  config.set("threshold", "0.25");
  config.set("retain", "2");
  config.set("retain_axis", "paper");
  config.set("lambda", "0.3");
  config.set("heads", "shared");
  config.set("decode_compress", "false");
  config.validate();
  const auto parsed = RKVConfig::parse(config.serialize());
  EXPECT_EQ(parsed.serialize(), config.serialize());
  EXPECT_DOUBLE_EQ(parsed.ratio, 0.9);
  EXPECT_EQ(parsed.budget, 0u);
  EXPECT_EQ(parsed.window, 16u);
  EXPECT_EQ(parsed.buffer, 64u);
  EXPECT_EQ(parsed.kernel, 5u);
  EXPECT_EQ(parsed.gqa_pool, RKVGqaPool::Mean);
  EXPECT_DOUBLE_EQ(parsed.threshold, 0.25);
  EXPECT_EQ(parsed.retain, 2u);
  EXPECT_EQ(parsed.retain_axis, RKVRetainAxis::Paper);
  EXPECT_DOUBLE_EQ(parsed.lambda, 0.3);
  EXPECT_EQ(parsed.heads, RKVHeadSelect::Shared);
  EXPECT_FALSE(parsed.decode_compress);
  // The property value must stay one token for getKeyValue().
  EXPECT_EQ(config.serialize().find('='), std::string::npos);
  EXPECT_EQ(config.serialize().find(' '), std::string::npos);

  RKVConfig defaults;
  defaults.set("ratio", "0.9");
  EXPECT_EQ(defaults.describe(2044),
            "ratio=0.9 budget=204 window=8 buffer=128 kernel=7 lambda=0.1 "
            "threshold=0.5 retain=1 axis=code heads=per_kv_head pool=max "
            "decode=1 prompt=2044");
  RKVConfig fixed;
  fixed.set("budget", "256");
  EXPECT_EQ(RKVConfig::parse(fixed.serialize()).budget, 256u);
  EXPECT_EQ(fixed.describe(2044).rfind("budget=256 window=8", 0), 0u);
  RKVConfig keep;
  keep.set("ratio", "0");
  EXPECT_TRUE(keep.keepOnly());
  EXPECT_EQ(keep.describe(2044), "keep prompt=2044");
  EXPECT_FALSE(RKVConfig().enabled());

  const std::vector<std::vector<std::pair<std::string, std::string>>> bad = {
    {},
    {{"ratio", "1"}},
    {{"ratio", "-0.1"}},
    {{"ratio", "0.5"}, {"budget", "64"}},
    {{"budget", "0"}},
    {{"budget", "8"}},
    {{"ratio", "0.5"}, {"window", "0"}},
    {{"ratio", "0.5"}, {"buffer", "0"}},
    {{"ratio", "0.5"}, {"kernel", "4"}},
    {{"ratio", "0.5"}, {"lambda", "1.5"}},
    {{"ratio", "0.5"}, {"lambda", "-0.1"}},
    {{"ratio", "0.5"}, {"threshold", "1.5"}},
    {{"ratio", "0.5"}, {"retain", "0"}},
    {{"ratio", "0.5"}, {"retain", "-1"}},
    {{"ratio", "0.5"}, {"gqa_pool", "sum"}},
    {{"ratio", "0.5"}, {"retain_axis", "row"}},
    {{"ratio", "0.5"}, {"heads", "all"}},
    {{"ratio", "0.5"}, {"decode_compress", "yes"}},
    {{"ratio", "0.5"}, {"window_size", "8"}},
    {{"ratio", "abc"}},
  };
  for (const auto &settings : bad)
    EXPECT_THROW(configOf(settings), std::invalid_argument)
      << (settings.empty() ? std::string("{}") : settings.back().first);
}

/**
 * @brief B = max(window + 1, floor(t (1 - r))); a fixed budget ignores t.
 */
TEST(RKVTest, BudgetFromRatio) {
  const unsigned int prompts[] = {9, 508, 1020, 2044};
  const double ratios[] = {0.25, 0.5, 0.9};
  const unsigned int expected[4][3] = {
    {9, 9, 9}, {381, 254, 50}, {765, 510, 101}, {1533, 1022, 204}};
  for (int p = 0; p < 4; ++p)
    for (int r = 0; r < 3; ++r) {
      RKVConfig config;
      config.ratio = ratios[r];
      EXPECT_EQ(config.budgetFor(prompts[p]), expected[p][r])
        << prompts[p] << " " << ratios[r];
    }
  RKVConfig fixed;
  fixed.budget = 300;
  EXPECT_EQ(fixed.budgetFor(9), 300u);
  EXPECT_EQ(fixed.budgetFor(2044), 300u);
}

/**
 * @brief The prefill keeps its last window queries; decode steps push one
 * query each and the oldest drops out.
 */
TEST(RKVTest, ObservationRing) {
  constexpr unsigned int hq = 2, hkv = 1, dim = 4, width = hq * dim;
  const auto config = configOf({{"budget", "1000"}, {"window", "8"}});
  const auto q = randomValues(23 * width, 7);
  FakeLayer layer(hq, hkv, dim, 64, nntrainer::Tdatatype::FP32);
  layer.state.consumeProperty("rkv", config);
  layer.state.consumeProperty("rkv_run", "0,20");
  layer.step(0, 20, std::vector<float>(q.begin(), q.begin() + 20 * width));
  auto obs = layer.state.observations();
  EXPECT_EQ(obs,
            std::vector<float>(q.begin() + 12 * width, q.begin() + 20 * width));
  for (unsigned int p = 20; p < 23; ++p)
    layer.step(
      p, 1,
      std::vector<float>(q.begin() + p * width, q.begin() + (p + 1) * width));
  obs = layer.state.observations();
  EXPECT_EQ(obs,
            std::vector<float>(q.begin() + 15 * width, q.begin() + 23 * width));
  EXPECT_EQ(layer.state.events(), 0u);

  FakeLayer short_layer(hq, hkv, dim, 64, nntrainer::Tdatatype::FP32);
  short_layer.state.consumeProperty("rkv", config);
  short_layer.state.consumeProperty("rkv_run", "0,5");
  short_layer.step(0, 5, std::vector<float>(q.begin(), q.begin() + 5 * width));
  EXPECT_EQ(short_layer.state.observations(),
            std::vector<float>(q.begin(), q.begin() + 5 * width));
}

/**
 * @brief Importance equals the explicit-logit reference for both GQA pools,
 * and the two pools differ.
 */
TEST(RKVTest, ImportanceMatchesReference) {
  constexpr unsigned int hq = 4, hkv = 2, dim = 16, n = 40, a = 8;
  constexpr unsigned int cand = n - a;
  const auto q = randomValues(a * hq * dim, 11, 2.0f);
  const auto k = randomValues(n * hkv * dim, 12, 2.0f);
  for (unsigned int kernel : {1u, 7u}) {
    std::vector<float> by_pool[2];
    for (auto pool : {RKVGqaPool::Max, RKVGqaPool::Mean}) {
      std::vector<float> out;
      causallm::rkvImportance(q.data(), a, hq, viewOf(k, hkv * dim), cand, hkv,
                              dim, pool, kernel, out);
      const auto ref =
        referenceImportance(q, k, a, hq, hkv, dim, cand, pool, kernel);
      ASSERT_EQ(out.size(), ref.size());
      for (size_t i = 0; i < out.size(); ++i)
        EXPECT_NEAR(out[i], ref[i], 1e-5 * std::abs(ref[i]) + 1e-9) << i;
      by_pool[pool == RKVGqaPool::Max ? 0 : 1] = out;
    }
    EXPECT_NE(by_pool[0], by_pool[1]);
  }
}

/**
 * @brief The max-pool matches F.max_pool1d with -inf padding, also when the
 * row is shorter than the kernel.
 */
TEST(RKVTest, MaxPoolMatchesTorch) {
  const float in[] = {1.0f, 3.0f, 2.0f, 0.0f, 5.0f, -1.0f, -2.0f, 4.0f};
  float out[8];
  causallm::rkvMaxPool1d(in, 8, 1, out);
  EXPECT_EQ(std::vector<float>(out, out + 8), std::vector<float>(in, in + 8));
  causallm::rkvMaxPool1d(in, 8, 3, out);
  EXPECT_EQ(std::vector<float>(out, out + 8),
            (std::vector<float>{3, 3, 3, 5, 5, 5, 4, 4}));
  causallm::rkvMaxPool1d(in, 8, 7, out);
  EXPECT_EQ(std::vector<float>(out, out + 8),
            (std::vector<float>{3, 5, 5, 5, 5, 5, 5, 5}));
  const float neg[] = {-3.0f, -1.0f};
  causallm::rkvMaxPool1d(neg, 2, 7, out);
  EXPECT_EQ(std::vector<float>(out, out + 2), (std::vector<float>{-1, -1}));
  causallm::rkvMaxPool1d(neg, 1, 7, out);
  EXPECT_EQ(out[0], -3.0f);
}

/**
 * @brief The O(n D) redundancy equals cal_similarity() with the n x n
 * matrix: random keys, exact copies and pairs on both sides of T, rows
 * without a similar key (the S[i, 0] fallback), and several thresholds.
 */
TEST(RKVTest, RedundancyMatchesFullMatrix) {
  constexpr unsigned int hkv = 2, dim = 16;
  for (unsigned int n : {1u, 2u, 9u, 70u, 300u}) {
    auto k = randomValues(n * hkv * dim, 100 + n);
    // Exact copies and a pair just above / below T = 0.5.
    if (n >= 9) {
      for (unsigned int d = 0; d < dim; ++d) {
        k[(7 * hkv + 0) * dim + d] = k[(2 * hkv + 0) * dim + d];
        k[(8 * hkv + 1) * dim + d] = k[(1 * hkv + 1) * dim + d];
      }
      for (const auto &pair : {std::pair<unsigned int, float>{5, 0.52f},
                               std::pair<unsigned int, float>{6, 0.48f}}) {
        // Row pair.first gets cosine pair.second with row 0 in head 0.
        std::vector<float> base(dim), other(dim);
        for (unsigned int d = 0; d < dim; ++d) {
          base[d] = d == 0 ? 1.0f : 0.0f;
          other[d] = d == 1 ? 1.0f : 0.0f;
        }
        const float c = pair.second, s = std::sqrt(1.0f - c * c);
        for (unsigned int d = 0; d < dim; ++d) {
          k[(0 * hkv + 0) * dim + d] = base[d];
          k[(pair.first * hkv + 0) * dim + d] = c * base[d] + s * other[d];
        }
      }
    }
    for (float threshold : {0.5f, 0.95f, 0.1f, -0.2f})
      for (unsigned int beta : {1u, 2u})
        for (auto axis : {RKVRetainAxis::Code, RKVRetainAxis::Paper}) {
          std::vector<float> out;
          causallm::rkvRedundancy(viewOf(k, hkv * dim), n, hkv, dim, threshold,
                                  beta, axis, out);
          const auto ref =
            referenceRedundancy(k, n, hkv, dim, threshold, beta, axis);
          ASSERT_EQ(out.size(), ref.size());
          double worst = 0.0;
          for (size_t i = 0; i < out.size(); ++i)
            worst = std::max(worst, std::abs(out[i] - ref[i]));
          EXPECT_LE(worst, 1e-6)
            << "n=" << n << " T=" << threshold << " beta=" << beta;
        }
  }
}

/**
 * @brief Code and paper axes follow their own definitions and differ: on the
 * code axis the newest of three copies is the least redundant.
 */
TEST(RKVTest, RedundancyRetainAxes) {
  constexpr unsigned int n = 20, hkv = 1, dim = 64;
  auto k = randomValues(n * dim, 5);
  for (unsigned int copy : {8u, 14u})
    for (unsigned int d = 0; d < dim; ++d)
      k[copy * dim + d] = k[2 * dim + d];
  std::vector<float> code, paper;
  for (unsigned int beta : {1u, 2u}) {
    causallm::rkvRedundancy(viewOf(k, dim), n, hkv, dim, 0.9f, beta,
                            RKVRetainAxis::Code, code);
    causallm::rkvRedundancy(viewOf(k, dim), n, hkv, dim, 0.9f, beta,
                            RKVRetainAxis::Paper, paper);
    const auto code_ref =
      referenceRedundancy(k, n, hkv, dim, 0.9f, beta, RKVRetainAxis::Code);
    const auto paper_ref =
      referenceRedundancy(k, n, hkv, dim, 0.9f, beta, RKVRetainAxis::Paper);
    for (unsigned int j = 0; j < n; ++j) {
      EXPECT_NEAR(code[j], code_ref[j], 1e-6);
      EXPECT_NEAR(paper[j], paper_ref[j], 1e-6);
    }
    EXPECT_NE(code, paper);
  }
  // beta = 1, code axis: rows 2 and 8 retain 14, row 14 retains 8.
  causallm::rkvRedundancy(viewOf(k, dim), n, hkv, dim, 0.9f, 1,
                          RKVRetainAxis::Code, code);
  EXPECT_LT(code[14], code[8]);
  EXPECT_LT(code[8], code[2]);
  causallm::rkvRedundancy(viewOf(k, dim), n, hkv, dim, 0.9f, 1,
                          RKVRetainAxis::Paper, paper);
  EXPECT_NEAR(paper[14], paper[8], 1e-7);
  EXPECT_NEAR(paper[8], paper[2], 1e-7);
}

/**
 * @brief Per KV head: Z = lambda I - (1 - lambda) R, top (B - window) of the
 * candidates (ties: smaller index), the window, ascending.
 */
TEST(RKVTest, SelectTopKPerHead) {
  constexpr unsigned int hkv = 3, n = 50, window = 6, budget = 20;
  constexpr unsigned int cand = n - window;
  auto importance = randomValues(hkv * cand, 21);
  auto redundancy = randomValues(hkv * n, 22);
  for (auto &v : importance)
    v = std::abs(v);
  for (auto &v : redundancy)
    v = std::abs(v);
  // Equal top scores on head 1 at rows 3 and 30.
  importance[cand + 3] = importance[cand + 30] = 5.0f;
  redundancy[n + 3] = redundancy[n + 30] = 0.0f;
  for (double lambda : {0.1, 0.5}) {
    const auto keep =
      causallm::rkvSelect(importance, redundancy, n, window, hkv, budget,
                          lambda, RKVHeadSelect::PerKVHead);
    ASSERT_EQ(keep.size(), hkv);
    for (unsigned int g = 0; g < hkv; ++g) {
      std::vector<double> z(cand);
      for (unsigned int c = 0; c < cand; ++c)
        z[c] = static_cast<float>(lambda) * importance[g * cand + c] -
               static_cast<float>(1.0 - lambda) * redundancy[g * n + c];
      EXPECT_EQ(keep[g], referenceTop(z, budget - window, n)) << g;
    }
  }
  // Exactly one of the tied rows fits: the smaller index wins.
  auto tied_i = std::vector<float>(hkv * cand, 0.0f);
  auto tied_r = std::vector<float>(hkv * n, 0.0f);
  for (unsigned int c = 0; c < budget - window - 1; ++c)
    tied_i[c] = 1.0f;
  tied_i[30] = tied_i[35] = 0.5f;
  const auto keep = causallm::rkvSelect(tied_i, tied_r, n, window, hkv, budget,
                                        1.0, RKVHeadSelect::PerKVHead);
  EXPECT_NE(std::find(keep[0].begin(), keep[0].end(), 30u), keep[0].end());
  EXPECT_EQ(std::find(keep[0].begin(), keep[0].end(), 35u), keep[0].end());
}

/**
 * @brief Shared selection (Alg. 1): every head keeps the top of the
 * head-mean Z.
 */
TEST(RKVTest, SelectSharedHeads) {
  constexpr unsigned int hkv = 4, n = 40, window = 4, budget = 16;
  constexpr unsigned int cand = n - window;
  const auto importance = randomValues(hkv * cand, 31);
  const auto redundancy = randomValues(hkv * n, 32);
  const auto keep = causallm::rkvSelect(importance, redundancy, n, window, hkv,
                                        budget, 0.1, RKVHeadSelect::Shared);
  std::vector<double> mean(cand, 0.0);
  for (unsigned int c = 0; c < cand; ++c) {
    for (unsigned int g = 0; g < hkv; ++g)
      mean[c] += 0.1f * importance[g * cand + c] - 0.9f * redundancy[g * n + c];
    mean[c] /= hkv;
  }
  const auto expected = referenceTop(mean, budget - window, n);
  for (unsigned int g = 0; g < hkv; ++g)
    EXPECT_EQ(keep[g], expected);
  const auto per_head =
    causallm::rkvSelect(importance, redundancy, n, window, hkv, budget, 0.1,
                        RKVHeadSelect::PerKVHead);
  EXPECT_NE(per_head, keep);
}

/**
 * @brief The window rows are kept even when their redundancy is the worst.
 */
TEST(RKVTest, SelectKeepsWindow) {
  constexpr unsigned int hkv = 2, n = 30, window = 5, budget = 10;
  constexpr unsigned int cand = n - window;
  std::vector<float> importance(hkv * cand, 0.0f);
  std::vector<float> redundancy(hkv * n, 0.0f);
  for (unsigned int g = 0; g < hkv; ++g)
    for (unsigned int r = cand; r < n; ++r)
      redundancy[g * n + r] = 1e6f;
  for (double lambda : {0.0, 0.1, 1.0}) {
    const auto keep =
      causallm::rkvSelect(importance, redundancy, n, window, hkv, budget,
                          lambda, RKVHeadSelect::PerKVHead);
    for (unsigned int g = 0; g < hkv; ++g) {
      ASSERT_EQ(keep[g].size(), budget);
      for (unsigned int r = cand; r < n; ++r)
        EXPECT_NE(std::find(keep[g].begin(), keep[g].end(), r), keep[g].end());
    }
  }
  // Nothing to drop: every row stays.
  const auto all =
    causallm::rkvSelect(importance, redundancy, budget, window, hkv, budget,
                        0.1, RKVHeadSelect::PerKVHead);
  std::vector<unsigned int> rows(budget);
  std::iota(rows.begin(), rows.end(), 0u);
  EXPECT_EQ(all[0], rows);
}

/**
 * @brief lambda = 1 is the repository's SnapKV: top (B - window) of the
 * pooled importance plus the window, without the redundancy.
 */
TEST(RKVTest, LambdaOneEqualsSnapKV) {
  constexpr unsigned int hq = 8, hkv = 2, dim = 16, n = 64, a = 8, budget = 24;
  constexpr unsigned int cand = n - a;
  const auto q = randomValues(a * hq * dim, 41, 2.0f);
  const auto k = randomValues(n * hkv * dim, 42, 2.0f);
  std::vector<float> importance;
  causallm::rkvImportance(q.data(), a, hq, viewOf(k, hkv * dim), cand, hkv, dim,
                          RKVGqaPool::Max, 7, importance);
  const auto keep = causallm::rkvSelect(importance, {}, n, a, hkv, budget, 1.0,
                                        RKVHeadSelect::PerKVHead);
  const auto ref =
    referenceImportance(q, k, a, hq, hkv, dim, cand, RKVGqaPool::Max, 7);
  for (unsigned int g = 0; g < hkv; ++g) {
    // Valid top-k of the reference scores (plateaus of the max-pool tie).
    std::vector<bool> kept(cand, false);
    for (unsigned int r = 0; r < budget - a; ++r)
      kept[keep[g][r]] = true;
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (unsigned int c = 0; c < cand; ++c) {
      if (kept[c])
        low = std::min(low, ref[g * cand + c]);
      else
        high = std::max(high, ref[g * cand + c]);
    }
    EXPECT_GE(low, high - 1e-9);
    for (unsigned int r = cand; r < n; ++r)
      EXPECT_EQ(keep[g][r - cand + budget - a], r);
  }
}

/**
 * @brief lambda = 0 ranks by -R alone and never reads the importance.
 */
TEST(RKVTest, LambdaZeroUsesRedundancyOnly) {
  constexpr unsigned int hkv = 2, n = 40, window = 8, budget = 16;
  const auto redundancy = randomValues(hkv * n, 51);
  const auto keep = causallm::rkvSelect({}, redundancy, n, window, hkv, budget,
                                        0.0, RKVHeadSelect::PerKVHead);
  for (unsigned int g = 0; g < hkv; ++g) {
    std::vector<double> z(n - window);
    for (unsigned int c = 0; c < n - window; ++c)
      z[c] = -redundancy[g * n + c];
    EXPECT_EQ(keep[g], referenceTop(z, budget - window, n));
  }
  EXPECT_THROW(causallm::rkvSelect({}, redundancy, n, window, hkv, budget, 0.1,
                                   RKVHeadSelect::PerKVHead),
               std::invalid_argument);
}

/**
 * @brief Compression after the prefill and at decode positions divisible by
 * the buffer; rows stay within B + buffer and every physical row maps to its
 * absolute position (cacheRow(u) = u - removed).
 */
TEST(RKVTest, CompressionSchedule) {
  constexpr unsigned int hq = 4, hkv = 2, dim = 4, width = hq * dim;
  struct Case {
    unsigned int prompt, budget, buffer, decode;
    bool decode_compress;
  };
  for (const auto &c :
       {Case{300, 60, 16, 100, true}, Case{300, 60, 16, 100, false},
        Case{40, 60, 16, 60, true}}) {
    for (auto dtype :
         {nntrainer::Tdatatype::FP32, nntrainer::Tdatatype::UINT16}) {
      FakeLayer layer(hq, hkv, dim, 512, dtype);
      layer.state.consumeProperty(
        "rkv", configOf({{"budget", std::to_string(c.budget)},
                         {"buffer", std::to_string(c.buffer)},
                         {"decode_compress", c.decode_compress ? "1" : "0"}}));
      layer.state.consumeProperty("rkv_run", "3," + std::to_string(c.prompt));
      const auto q = randomValues((c.prompt + c.decode) * width, 61);
      unsigned int rows = c.prompt, events = 0;
      layer.step(0, c.prompt,
                 std::vector<float>(q.begin(), q.begin() + c.prompt * width));
      if (rows > c.budget) {
        rows = c.budget;
        ++events;
      }
      EXPECT_EQ(layer.state.events(), events);
      EXPECT_EQ(layer.state.rows(), rows);
      layer.expectRowsMatchPositions();
      for (unsigned int p = c.prompt; p < c.prompt + c.decode; ++p) {
        layer.step(p, 1,
                   std::vector<float>(q.begin() + p * width,
                                      q.begin() + (p + 1) * width));
        ++rows;
        if (c.decode_compress && p % c.buffer == 0 && rows > c.budget) {
          rows = c.budget;
          ++events;
        }
        ASSERT_EQ(layer.state.events(), events) << p;
        ASSERT_EQ(layer.state.rows(), rows) << p;
        ASSERT_EQ(layer.state.removed(), p + 1 - rows);
        // The newest token sits at cacheRow(p) and is always kept.
        for (unsigned int g = 0; g < hkv; ++g)
          ASSERT_EQ(cacheAt(layer.value, p - layer.state.removed(), g * dim),
                    static_cast<float>(p));
      }
      layer.expectRowsMatchPositions();
      if (c.decode_compress && c.prompt > c.budget) {
        EXPECT_LE(layer.max_rows, c.budget + c.buffer);
      }
      EXPECT_EQ(layer.state.budget(), c.budget);
    }
  }
  // Steps must be one full prefill and then single consecutive tokens.
  FakeLayer layer(hq, hkv, dim, 64, nntrainer::Tdatatype::FP32);
  layer.state.consumeProperty("rkv", configOf({{"budget", "16"}}));
  layer.state.consumeProperty("rkv_run", "0,10");
  EXPECT_THROW(layer.state.checkStep(0, 5, 64, 1000), std::invalid_argument);
  layer.state.checkStep(0, 10, 64, 1000);
  EXPECT_THROW(layer.state.checkStep(11, 1, 64, 1000), std::invalid_argument);
  EXPECT_THROW(layer.state.checkStep(10, 2, 64, 1000), std::invalid_argument);
}

/**
 * @brief Compaction moves each head's kept slices to the front and leaves
 * the other head's bytes alone.
 */
TEST(RKVTest, CompactionMovesHeadSlices) {
  constexpr unsigned int n = 10, hkv = 2, dim = 4, width = hkv * dim;
  std::vector<float> data(n * width);
  for (unsigned int r = 0; r < n; ++r)
    for (unsigned int g = 0; g < hkv; ++g)
      for (unsigned int d = 0; d < dim; ++d)
        data[r * width + g * dim + d] = r * 100.0f + g * 10.0f + d;
  const std::vector<std::vector<unsigned int>> keep = {{0, 3, 8, 9},
                                                       {2, 5, 8, 9}};
  auto moved = data;
  causallm::rkvCompact(reinterpret_cast<unsigned char *>(moved.data()),
                       width * sizeof(float), dim * sizeof(float), keep);
  for (unsigned int g = 0; g < hkv; ++g)
    for (unsigned int r = 0; r < 4; ++r)
      for (unsigned int d = 0; d < dim; ++d)
        EXPECT_EQ(moved[r * width + g * dim + d],
                  data[keep[g][r] * width + g * dim + d]);
  // Rows past the kept count are untouched (released later).
  for (unsigned int r = 4; r < n; ++r)
    for (unsigned int c = 0; c < width; ++c)
      EXPECT_EQ(moved[r * width + c], data[r * width + c]);
  EXPECT_THROW(causallm::rkvCompact(
                 reinterpret_cast<unsigned char *>(moved.data()),
                 width * sizeof(float), dim * sizeof(float), {{3, 2}, {0, 1}}),
               std::invalid_argument);
}

/**
 * @brief Releasing the tail keeps the kept rows and the released rows can be
 * written again.
 */
TEST(RKVTest, ReleaseRowsKeepsKeptData) {
  constexpr size_t rows = 4096, width = 512;
  auto cache = makeTensor({}, rows, width, nntrainer::Tdatatype::UINT16);
  uint16_t *data = cache.getData<uint16_t>();
  for (size_t i = 0; i < rows * width; ++i)
    data[i] = static_cast<uint16_t>(i % 60000 + 1);
  const size_t kept = 1000;
  const size_t released = causallm::rkvReleaseRows(cache, kept);
#if defined(_WIN32)
  EXPECT_EQ(released, 0u);
#else
  EXPECT_GT(released, 0u);
#endif
  for (size_t i = 0; i < kept * width; ++i)
    ASSERT_EQ(data[i], static_cast<uint16_t>(i % 60000 + 1));
  for (size_t i = kept * width; i < rows * width; ++i)
    data[i] = 7;
  EXPECT_EQ(data[rows * width - 1], 7);
}

/**
 * @brief Importance, redundancy, Z and the kept rows equal the official
 * R1KV.update_kv on its first compression, from FP32 and FP16 caches, through
 * the pure functions and through RKVState.
 */
TEST(RKVTest, R1KVGolden) {
  const auto in = goldenInputs();
  for (auto dtype :
       {nntrainer::Tdatatype::FP32, nntrainer::Tdatatype::UINT16}) {
    const auto key = makeTensor(in.k, kGN, kGHkv * kGD, dtype);
    const auto view = RKVView::of(key);
    std::vector<float> imax, imean, red;
    causallm::rkvImportance(in.q.data(), kGA, kGHq, view, kGCand, kGHkv, kGD,
                            RKVGqaPool::Max, kGKernel, imax);
    causallm::rkvImportance(in.q.data(), kGA, kGHq, view, kGCand, kGHkv, kGD,
                            RKVGqaPool::Mean, kGKernel, imean);
    causallm::rkvRedundancy(view, kGN, kGHkv, kGD, 0.5f, 1, RKVRetainAxis::Code,
                            red);
    ASSERT_EQ(imax.size(), kGHkv * kGCand);
    for (size_t i = 0; i < imax.size(); ++i) {
      EXPECT_NEAR(imax[i], kGoldenImportanceMax[i], 1e-5 * imax[i]) << i;
      EXPECT_NEAR(imean[i], kGoldenImportanceMean[i], 1e-5 * imean[i]) << i;
    }
    for (unsigned int g = 0; g < kGHkv; ++g)
      for (unsigned int c = 0; c < kGCand; ++c) {
        const float z = imax[g * kGCand + c] * 0.1f -
                        red[g * kGN + c] * static_cast<float>(1.0 - 0.1);
        EXPECT_NEAR(z, kGoldenScore[g * kGCand + c], 1e-6) << g << " " << c;
      }
    EXPECT_EQ(causallm::rkvSelect(imax, red, kGN, kGA, kGHkv, kGBudget, 0.1,
                                  RKVHeadSelect::PerKVHead),
              kGoldenKeepR1KV);
    EXPECT_EQ(causallm::rkvSelect(imax, red, kGN, kGA, kGHkv, kGBudget, 1.0,
                                  RKVHeadSelect::PerKVHead),
              kGoldenKeepSnapKV);

    // The same compression through the layer state (prefill of 24 rows
    // whose last 4 queries are the observations).
    std::vector<float> prompt_q((kGN - kGA) * kGHq * kGD, 0.0f);
    prompt_q.insert(prompt_q.end(), in.q.begin(), in.q.end());
    auto cache_k = makeTensor(in.k, 32, kGHkv * kGD, dtype);
    auto cache_v = makeTensor(in.v, 32, kGHkv * kGD, dtype);
    RKVState state;
    state.consumeProperty(
      "rkv", configOf({{"budget", "12"}, {"window", "4"}, {"kernel", "7"}}));
    state.consumeProperty("rkv_run", "0,24");
    state.checkStep(0, kGN, 32, 1000);
    state.afterAttention(makeTensor(prompt_q, kGN, kGHq * kGD), cache_k,
                         cache_v, kGHq, kGHkv, kGD, 0, kGN);
    EXPECT_EQ(state.events(), 1u);
    EXPECT_EQ(state.removed(), kGN - kGBudget);
    ASSERT_EQ(state.positions(), kGoldenKeepR1KV);
    for (unsigned int g = 0; g < kGHkv; ++g)
      for (unsigned int r = 0; r < kGBudget; ++r)
        for (unsigned int d = 0; d < kGD; ++d) {
          const unsigned int src = kGoldenKeepR1KV[g][r];
          EXPECT_EQ(cacheAt(cache_k, r, g * kGD + d),
                    in.k[(src * kGHkv + g) * kGD + d]);
          EXPECT_EQ(cacheAt(cache_v, r, g * kGD + d),
                    in.v[(src * kGHkv + g) * kGD + d]);
        }
  }
}

/**
 * @brief The redundancy equals cal_similarity() at two thresholds.
 */
TEST(RKVTest, CalSimilarityGolden) {
  const auto in = goldenInputs();
  for (const auto &item :
       {std::pair<float, const float *>{0.5f, kGoldenRedundancy05},
        std::pair<float, const float *>{0.2f, kGoldenRedundancy02}}) {
    std::vector<float> red;
    causallm::rkvRedundancy(viewOf(in.k, kGHkv * kGD), kGN, kGHkv, kGD,
                            item.first, 1, RKVRetainAxis::Code, red);
    ASSERT_EQ(red.size(), kGHkv * kGN);
    for (size_t i = 0; i < red.size(); ++i)
      EXPECT_NEAR(red[i], item.second[i], 1e-5 * item.second[i])
        << item.first << " " << i;
  }
}

namespace {

/**
 * @brief Tiny Qwen3 CausalLM adapter (inference shared by CausalLMTestAdapter)
 */
using TinyQwen3CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

/** @brief A 14-token prompt of the tiny word-level tokenizer. */
const char *kTinyPrompt =
  "hello world tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10 tok11 tok12 tok13 "
  "tok14";
constexpr unsigned int kTinyPromptTokens = 14;

/** @brief Capture std::clog for the lifetime of the object. */
class ClogCapture {
public:
  ClogCapture() : old_(std::clog.rdbuf(text_.rdbuf())) {}
  ~ClogCapture() { std::clog.rdbuf(old_); }
  std::string str() const { return text_.str(); }

private:
  std::ostringstream text_;
  std::streambuf *old_;
};

/**
 * @brief Two-layer tiny Qwen3 (GQA 2) with random weights, a 14-token
 * prompt and 6 generated tokens (EOS banned), run through CausalLM::run().
 */
std::string runTinyQwen3(const std::string &test_name,
                         const causallm::json &extra_nntr,
                         std::string *log = nullptr,
                         TinyQwen3CausalLM **keep = nullptr) {
  static std::vector<std::unique_ptr<TinyQwen3CausalLM>> alive;
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "RKVModelTest", test_name, "Qwen3_FP32");
  causallm::json cfg = {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 64},
    {"model_type", "qwen3"},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 2},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
  auto nntr = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, causallm_test::makeTinyFp32DataType());
  nntr["init_seq_len"] = kTinyPromptTokens + 1;
  nntr["max_seq_len"] = 32;
  nntr["num_to_generate"] = 6;
  nntr["bad_word_ids"] = std::vector<unsigned int>{31};
  for (const auto &item : extra_nntr.items())
    nntr[item.key()] = item.value();

  auto generation = causallm_test::makeTinyGenerationConfig();
  auto model = std::make_unique<TinyQwen3CausalLM>(cfg, generation, nntr);
  model->initializeModel();
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  model->forEachLayer([&](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
    for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
      auto &weight = context.getWeight(i);
      if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
        continue;
      float *data = weight.getData<float>();
      for (size_t k = 0; k < weight.size(); ++k)
        data[k] =
          causallm_test::isRmsNormLayerType(layer.getType()) ? 1.0f : dist(rng);
    }
  });
  {
    ClogCapture capture;
    model->runPrompt(kTinyPrompt);
    if (log)
      *log = capture.str();
  }
  const std::string text = model->getOutputText(0);
  if (keep) {
    *keep = model.get();
    alive.push_back(std::move(model));
  }
  return text;
}

/** @brief Values of "key=<v>" on log lines starting with prefix. */
std::vector<std::string> logValues(const std::string &log,
                                   const std::string &prefix,
                                   const std::string &key) {
  std::vector<std::string> out;
  std::stringstream stream(log);
  std::string line;
  while (std::getline(stream, line)) {
    if (line.rfind(prefix, 0) != 0)
      continue;
    const auto at = line.find(" " + key + "=");
    if (at == std::string::npos)
      continue;
    const auto begin = at + key.size() + 2;
    out.push_back(line.substr(begin, line.find(' ', begin) - begin));
  }
  return out;
}

} // namespace

/**
 * @brief ratio 0 (keep: page release only) and a budget that covers the run
 * take the R-KV path without compressing and generate what the unchanged
 * path does.
 */
TEST(RKVModelTest, KeepAndCoveringBudgetMatchBaseline) {
  std::string log;
  const auto base = runTinyQwen3("Base", causallm::json::object(), &log);
  EXPECT_FALSE(base.empty());
  EXPECT_EQ(log.find("[R-KV"), std::string::npos);

  const auto keep = runTinyQwen3("Keep", {{"rkv", {{"ratio", 0}}}}, &log);
  EXPECT_EQ(keep, base);
  EXPECT_NE(log.find("[R-KV] keep prompt=14"), std::string::npos) << log;
  EXPECT_EQ(log.find("[R-KV layer="), std::string::npos);

  const auto covered = runTinyQwen3(
    "Covered", {{"rkv", {{"budget", 31}, {"window", 4}, {"buffer", 2}}}}, &log);
  EXPECT_EQ(covered, base);
  EXPECT_NE(log.find("[R-KV] budget=31 window=4"), std::string::npos) << log;
  EXPECT_EQ(log.find("[R-KV layer="), std::string::npos) << log;
}

/**
 * @brief Prefill and decode compression under every option: each layer logs
 * the scheduled events and keeps the budget.
 */
TEST(RKVModelTest, CompressesAndGenerates) {
  const causallm::json base = {{"budget", 8}, {"window", 4}, {"buffer", 2}};
  std::vector<causallm::json> settings;
  for (double lambda : {0.0, 0.1, 1.0})
    for (const char *heads : {"per_kv_head", "shared"}) {
      auto s = base;
      s["lambda"] = lambda;
      s["heads"] = heads;
      settings.push_back(s);
    }
  for (const auto &extra :
       {causallm::json{{"retain_axis", "paper"}, {"retain", 2}},
        causallm::json{{"gqa_pool", "mean"}, {"threshold", 0.2}},
        causallm::json{{"decode_compress", false}},
        causallm::json{{"kernel", 1}}}) {
    auto s = base;
    for (const auto &item : extra.items())
      s[item.key()] = item.value();
    settings.push_back(s);
  }
  auto ratio = causallm::json{{"ratio", 0.5}, {"window", 4}, {"buffer", 4}};
  settings.push_back(ratio);

  for (const auto &rkv : settings) {
    std::string log, text;
    ASSERT_NO_THROW(text = runTinyQwen3("Compress", {{"rkv", rkv}}, &log))
      << rkv.dump();
    EXPECT_FALSE(text.empty()) << rkv.dump();
    const unsigned int budget =
      rkv.contains("budget") ? rkv["budget"].get<unsigned int>() : 7u;
    const unsigned int buffer = rkv["buffer"].get<unsigned int>();
    const bool decode = rkv.value("decode_compress", true);
    for (const char *layer : {"[R-KV layer=0]", "[R-KV layer=1]"}) {
      const auto events = logValues(log, layer, "event");
      const auto pos = logValues(log, layer, "pos");
      const auto rows = logValues(log, layer, "rows");
      const auto kept = logValues(log, layer, "kept");
      ASSERT_FALSE(events.empty()) << rkv.dump() << "\n" << log;
      EXPECT_EQ(events[0], "prefill");
      EXPECT_EQ(pos[0], std::to_string(kTinyPromptTokens - 1));
      EXPECT_EQ(rows[0], std::to_string(kTinyPromptTokens));
      for (size_t e = 0; e < events.size(); ++e) {
        EXPECT_EQ(kept[e], std::to_string(budget)) << rkv.dump();
        if (e == 0)
          continue;
        EXPECT_EQ(events[e], "decode");
        const unsigned int p = std::stoul(pos[e]);
        EXPECT_EQ(p % buffer, 0u) << rkv.dump();
        EXPECT_LE(std::stoul(rows[e]), budget + buffer);
      }
      // Decode positions 14.. at every buffer multiple while rows > budget.
      size_t expected = 1;
      if (decode) {
        unsigned int r = budget;
        const unsigned int last =
          events.size() > 1 ? std::stoul(pos.back()) : kTinyPromptTokens;
        for (unsigned int p = kTinyPromptTokens; p <= last; ++p)
          if (++r > budget && p % buffer == 0) {
            r = budget;
            ++expected;
          }
        EXPECT_GT(events.size(), 1u) << rkv.dump();
      }
      EXPECT_EQ(events.size(), expected) << rkv.dump() << "\n" << log;
    }
    EXPECT_EQ(logValues(log, "[R-KV layer=0]", "pos"),
              logValues(log, "[R-KV layer=1]", "pos"));
  }
}

/**
 * @brief The compressed cache gives the same tokens on the flash (gemm) arm
 * and on the per-row arm, and compression reaches the output.
 */
TEST(RKVModelTest, SameOnFlashAndPerRowArms) {
  const auto base_flash =
    runTinyQwen3("ArmsBaseFlash", {{"use_flash_attention", true}});
  const auto base_row =
    runTinyQwen3("ArmsBaseRow", {{"use_flash_attention", false}});
  ASSERT_EQ(base_flash, base_row) << "the two arms already differ";
  const causallm::json rkv = {{"budget", 6}, {"window", 2}, {"buffer", 2}};
  const auto flash =
    runTinyQwen3("ArmsFlash", {{"rkv", rkv}, {"use_flash_attention", true}});
  const auto row =
    runTinyQwen3("ArmsRow", {{"rkv", rkv}, {"use_flash_attention", false}});
  EXPECT_EQ(flash, row);
  EXPECT_NE(flash, base_flash);
}

/**
 * @brief Continued runs, serialized KV caches and bad settings are rejected.
 */
TEST(RKVModelTest, RejectsUnsupportedRuns) {
  TinyQwen3CausalLM *model = nullptr;
  runTinyQwen3("Rejects", {{"rkv", {{"ratio", 0.5}, {"window", 4}}}}, nullptr,
               &model);
  ASSERT_NE(model, nullptr);
  EXPECT_THROW(model->runPrompt(kTinyPrompt), std::invalid_argument);
  EXPECT_THROW(model->save_kvcache("/nonexistent/rkv.bin", 1),
               std::invalid_argument);
  EXPECT_THROW(model->load_kvcache("/nonexistent/rkv.bin", 1),
               std::invalid_argument);
  EXPECT_THROW(
    runTinyQwen3("RejectsWindow", {{"rkv", {{"ratio", 0.5}, {"window", 0}}}}),
    std::invalid_argument);
  EXPECT_THROW(
    runTinyQwen3("RejectsKey", {{"rkv", {{"ratio", 0.5}, {"windows", 4}}}}),
    std::invalid_argument);
  EXPECT_THROW(runTinyQwen3("RejectsType", {{"rkv", 0.5}}),
               std::invalid_argument);
}
