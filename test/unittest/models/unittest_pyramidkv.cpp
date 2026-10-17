// SPDX-License-Identifier: Apache-2.0
/** @file unittest_pyramidkv.cpp
 * @brief PyramidKV layer schedule, Tensor selection and session guards
 */
#include <gtest/gtest.h>
#include <pyramidkv.h>

#include <array>

namespace {
using causallm::PyramidKVConfig;
using causallm::PyramidKVState;
using nntrainer::Tdatatype;
using nntrainer::Tensor;
using nntrainer::Tformat;

void configure(PyramidKVState &state, unsigned int layer = 0,
               unsigned int budget = 8, unsigned int prompt = 16) {
  for (const auto &p :
       {"pyramidkv_budget=" + std::to_string(budget),
        std::string("pyramidkv_window=4"), std::string("pyramidkv_kernel=3"),
        std::string("pyramidkv_beta=4"), std::string("pyramidkv_layers=2"),
        "pyramidkv_layer=" + std::to_string(layer),
        "pyramidkv_prompt_length=" + std::to_string(prompt)}) {
    const auto split = p.find('=');
    ASSERT_TRUE(state.consumeProperty(p.substr(0, split), p.substr(split + 1)));
  }
}

TEST(PyramidKV, OfficialIntegerScheduleAndShortPrompt_p) {
  const PyramidKVConfig config{256, 64, 5, 20};
  EXPECT_EQ(config.retained(4096, 0, 28), 439u);
  EXPECT_EQ(config.retained(4096, 27, 28), 88u);
  EXPECT_EQ(config.retained(128, 0, 28), 128u);
  EXPECT_EQ(config.retained(300, 27, 28), 256u);
  EXPECT_EQ(config.retained(384, 0, 28), 384u);
  EXPECT_EQ(config.retained(384, 27, 28), 141u);
  EXPECT_EQ(config.retained(4096, 0, 1), 256u);
}

TEST(PyramidKV, TensorSelectionAndDecodeAppend_p) {
  for (const auto dtype : {Tdatatype::FP32, Tdatatype::UINT16}) {
    for (unsigned int layer = 0; layer < 2; ++layer) {
      PyramidKVState state;
      for (unsigned int session = 0; session < 2; ++session) {
        configure(state, layer);
        Tensor k(1, 1, 32, 2, Tformat::NCHW, dtype);
        Tensor v(1, 1, 32, 2, Tformat::NCHW, dtype);
        Tensor q(1, 1, 16, 4);
        for (unsigned int i = 0; i < 32; ++i) {
          if (dtype == Tdatatype::FP32) {
            k.getData<float>()[i] = i;
            v.getData<float>()[i] = 1000 + i;
          } else {
            k.getData<uint16_t>()[i] = i;
            v.getData<uint16_t>()[i] = 1000 + i;
          }
        }
        // Per-head ramps make the selected prefix differ across KV heads.
        // Zero padding and average pooling must affect the boundary ranks.
        std::array<std::vector<float>, 2> reference;
        for (unsigned int h = 0; h < 2; ++h) {
          std::vector<std::pair<float, unsigned int>> scores;
          for (int j = 0; j < 12; ++j) {
            float sum = 0;
            for (int n = j - 1; n <= j + 1; ++n)
              if (n >= 0 && n < 12)
                sum += h ? 12 - n : n + 1;
            scores.emplace_back(-sum, j);
          }
          std::sort(scores.begin(), scores.end());
          scores.resize(layer ? 1 : 7);
          std::vector<unsigned int> ids;
          for (const auto &s : scores)
            ids.push_back(s.second);
          for (unsigned int j = 12; j < 16; ++j)
            ids.push_back(j);
          std::sort(ids.begin(), ids.end());
          for (const auto id : ids)
            reference[h].push_back(id * 2 + h);
        }
        state.finishPrefill(q, k, v, 4, 2, 1,
                            [](Tensor &window_q, Tensor &, Tensor &p,
                               unsigned int prefix, unsigned int window) {
                              EXPECT_EQ(window_q.height(), 4u);
                              EXPECT_EQ(prefix, 12u);
                              size_t row = 0;
                              p.setValue(0.0f);
                              for (unsigned int i = 0; i < window; ++i) {
                                for (unsigned int j = 0; j < prefix; ++j)
                                  for (unsigned int h = 0; h < 4; ++h)
                                    p.getData<float>()[(row + j) * 4 + h] =
                                      (h / 2 ? 12 - j : j + 1) / 200.0f;
                                row += prefix + i + 1;
                              }
                            });
        const auto kept = layer ? 5u : 11u;
        EXPECT_EQ(state.physicalPosition(16), kept);
        for (unsigned int h = 0; h < 2; ++h)
          for (unsigned int i = 0; i < kept; ++i) {
            const auto offset = i * 2 + h;
            EXPECT_FLOAT_EQ(dtype == Tdatatype::FP32
                              ? k.getData<float>()[offset]
                              : k.getData<uint16_t>()[offset],
                            reference[h][i]);
            EXPECT_FLOAT_EQ(dtype == Tdatatype::FP32
                              ? v.getData<float>()[offset]
                              : v.getData<uint16_t>()[offset],
                            1000 + reference[h][i]);
          }
        Tensor decode(1, 1, 1, 4);
        for (unsigned int step = 0; step < 5; ++step)
          state.decodeStep(
            decode, k, v, 16 + step, 32, 4, 2, 1,
            [](Tensor &key, Tensor &value) {
              key.setValue(17);
              value.setValue(23);
            },
            [&](Tensor &key, Tensor &value, Tensor &p, unsigned int from) {
              EXPECT_EQ(from, kept + step);
              EXPECT_EQ(key.height(), kept + step + 1);
              EXPECT_EQ(value.height(), key.height());
              EXPECT_EQ(p.height(), key.height());
            });
        EXPECT_EQ(state.physicalPosition(21), kept + 5);
      }
    }
  }
}

TEST(PyramidKV, UncompressedPromptSkipsScoring_p) {
  PyramidKVState state;
  configure(state, 0, 32);
  Tensor q(1, 1, 16, 4), k(1, 1, 32, 2), v(1, 1, 32, 2);
  state.finishPrefill(q, k, v, 4, 2, 1,
                      [](Tensor &, Tensor &, Tensor &, unsigned int,
                         unsigned int) { ADD_FAILURE(); });
  EXPECT_EQ(state.physicalPosition(16), 16u);
}

TEST(PyramidKV, SharedKVHeadUsesBothQueryHeads_p) {
  PyramidKVState state;
  for (const auto &entry :
       std::vector<std::pair<std::string, std::string>>{{"budget", "4"},
                                                        {"window", "2"},
                                                        {"kernel", "1"},
                                                        {"beta", "4"},
                                                        {"layers", "1"},
                                                        {"layer", "0"},
                                                        {"prompt_length", "8"}})
    state.consumeProperty("pyramidkv_" + entry.first, entry.second);
  Tensor q(1, 1, 8, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
  for (unsigned int i = 0; i < 8; ++i) {
    k.getData<float>()[i] = i;
    v.getData<float>()[i] = 100 + i;
  }
  state.finishPrefill(
    q, k, v, 2, 1, 1,
    [](Tensor &, Tensor &, Tensor &p, unsigned int, unsigned int) {
      p.setValue(0.0f);
      for (const size_t row : {0u, 7u}) {
        p.getData<float>()[row * 2] = 0.3f;
        p.getData<float>()[(row + 4) * 2] = 0.2f;
        p.getData<float>()[(row + 4) * 2 + 1] = 0.2f;
        p.getData<float>()[(row + 5) * 2 + 1] = 0.8f;
      }
    });
  for (unsigned int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(k.getData<float>()[i], 4 + i);
    EXPECT_FLOAT_EQ(v.getData<float>()[i], 104 + i);
  }
}

TEST(PyramidKV, InvalidNumbers_n) {
  for (const auto *s : {"", "-1", "1.5", "true", "4294967296"})
    EXPECT_THROW(PyramidKVConfig::parseUnsigned(s), std::exception);
}

TEST(PyramidKV, InvalidSettingsAndLayers_n) {
  for (const auto &cfg :
       {PyramidKVConfig{8, 8, 3, 4}, {8, 0, 3, 4}, {8, 4, 2, 4}, {8, 4, 3, 0}})
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
  const PyramidKVConfig cfg{8, 4, 3, 4};
  EXPECT_THROW(cfg.retained(16, 2, 2), std::invalid_argument);
  EXPECT_THROW(cfg.retained(16, 0, 0), std::invalid_argument);
}

TEST(PyramidKV, InvalidSequence_n) {
  PyramidKVState state;
  configure(state);
  EXPECT_THROW(state.checkStep(0, 8, 32, 32), std::invalid_argument);
  EXPECT_THROW(state.checkStep(1, 16, 32, 32), std::invalid_argument);
  EXPECT_THROW(state.checkStep(0, 16, 8, 32), std::out_of_range);
  EXPECT_THROW(state.checkStep(0, 16, 32, 8), std::out_of_range);
}

TEST(PyramidKV, InvalidCacheAndQuery_n) {
  PyramidKVState state;
  configure(state);
  Tensor q(1, 1, 16, 4), k(1, 1, 32, 2), v(1, 1, 32, 3);
  auto score = [](Tensor &, Tensor &, Tensor &, unsigned int, unsigned int) {};
  EXPECT_THROW(state.finishPrefill(q, k, v, 4, 2, 1, score),
               std::invalid_argument);
  EXPECT_THROW(state.finishPrefill(q, k, k, 3, 2, 1, score),
               std::invalid_argument);
  Tensor bad_query(1, 1, 8, 4);
  EXPECT_THROW(state.finishPrefill(bad_query, k, k, 4, 2, 1, score),
               std::invalid_argument);
}

TEST(PyramidKV, InvalidProbability_n) {
  PyramidKVState state;
  configure(state);
  Tensor q(1, 1, 16, 4), k(1, 1, 32, 2), v(1, 1, 32, 2);
  EXPECT_THROW(state.finishPrefill(
                 q, k, v, 4, 2, 1,
                 [](Tensor &, Tensor &, Tensor &p, unsigned int, unsigned int) {
                   p.setValue(std::numeric_limits<float>::quiet_NaN());
                 }),
               std::invalid_argument);
}
} // namespace
