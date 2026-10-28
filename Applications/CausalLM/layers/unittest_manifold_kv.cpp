// SPDX-License-Identifier: Apache-2.0
/** @file unittest_manifold_kv.cpp
 * @brief Geometric selection and independent RoPE/attention reference tests.
 */
#include <gtest/gtest.h>
#include <layer_context.h>
#include <manifold_kv.h>
#include <mha_core.h>
#include <var_grad.h>

#include <array>

namespace {
using nntrainer::Tensor;
using nntrainer::TensorDim;
using DType = TensorDim::DataType;

causallm::ManifoldKVState policy(unsigned int budget, unsigned int window = 0) {
  causallm::ManifoldKVState state;
  state.consumeProperty("manifold_kv_budget", std::to_string(budget));
  state.consumeProperty("manifold_kv_window_size", std::to_string(window));
  return state;
}

float half(float x) {
  return nntrainer::compute_fp16_to_fp32(nntrainer::compute_fp32_to_fp16(x));
}

// Scalar reference operates on original token IDs and never calls the policy's
// scoring, compaction or logical-to-physical position helpers.
void checkAttention(unsigned int window, bool fp16) {
  const auto dtype = fp16 ? DType::FP16 : DType::FP32;
  const TensorDim qdim(1, 1, 8, 16, {TensorDim::Format::NCHW, dtype});
  const TensorDim kdim(1, 1, 8, 8, {TensorDim::Format::NCHW, dtype});
  const TensorDim cdim(1, 1, 7, 8, {TensorDim::Format::NCHW, DType::UINT16});
  causallm::MHACoreLayer layer;
  layer.setProperty({"num_heads=4", "num_heads_kv=2", "max_timestep=16",
                     "max_position_embeddings=16", "rope_theta=10000",
                     "is_causal=true", "manifold_kv_budget=3",
                     "manifold_kv_window_size=" + std::to_string(window),
                     "manifold_kv_prompt_length=7"});
  nntrainer::InitLayerContext init({qdim, kdim, kdim, cdim, cdim}, {false},
                                   false, "manifold_attention", "", 0.0f,
                                   {"NCHW", "FP32", fp16 ? "FP16" : "FP32"});
  layer.finalize(init);
  std::vector<nntrainer::Var_Grad> inputs;
  for (const auto &dim : {qdim, kdim, kdim, cdim, cdim})
    inputs.emplace_back(dim, nntrainer::Initializer::ZEROS, false, true,
                        "input");
  nntrainer::Var_Grad output(qdim, nntrainer::Initializer::ZEROS, false, true,
                             "output");
  nntrainer::RunLayerContext run(
    "manifold_attention", false, 0, false, 1, nullptr, false, {},
    {&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]}, {&output},
    {});
  std::vector<std::array<float, 16>> queries(11);
  std::vector<std::array<float, 8>> keys(11), values(11);
  for (unsigned int t = 0; t < 11; ++t) {
    for (unsigned int d = 0; d < 16; ++d) {
      const auto x = std::sin((t + 1) * (d + 1) * 0.17f);
      queries[t][d] = fp16 ? half(x) : x;
    }
    for (unsigned int d = 0; d < 8; ++d) {
      const auto k = std::cos((t + 1) * (d + 1) * 0.31f) * (1 + t % 3);
      const auto v = std::sin((t + 1) * (d + 1) * 0.23f);
      keys[t][d] = fp16 ? half(k) : k;
      values[t][d] = half(v);
    }
  }
  auto rotate = [](const float *x, unsigned int position, bool round) {
    std::array<float, 4> y;
    for (unsigned int d = 0; d < 2; ++d) {
      const float angle = position / std::pow(10000.0f, d / 2.0f);
      y[d] = x[d] * std::cos(angle) - x[d + 2] * std::sin(angle);
      y[d + 2] = x[d] * std::sin(angle) + x[d + 2] * std::cos(angle);
    }
    if (round)
      for (auto &v : y)
        v = half(v);
    return y;
  };
  std::array<std::vector<unsigned int>, 2> retained;
  for (unsigned int h = 0; h < 2; ++h) {
    std::array<double, 7> scores{};
    const unsigned int width = window ? window : 7;
    for (unsigned int start = 0; start < 7; start += width) {
      const auto end = std::min(7u, start + width);
      std::array<double, 4> centroid{};
      for (unsigned int t = start; t < end; ++t) {
        const auto k = rotate(keys[t].data() + h * 4, t, true);
        for (unsigned int d = 0; d < 4; ++d)
          centroid[d] += k[d] / static_cast<double>(end - start);
      }
      for (unsigned int t = start; t < end; ++t) {
        const auto k = rotate(keys[t].data() + h * 4, t, true);
        for (unsigned int d = 0; d < 4; ++d)
          scores[t] += std::pow(k[d] - centroid[d], 2);
      }
    }
    retained[h] = {0, 1, 2, 3, 4, 5, 6};
    std::sort(retained[h].begin(), retained[h].end(), [&](auto a, auto b) {
      return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    retained[h].resize(3);
    std::sort(retained[h].begin(), retained[h].end());
  }
  auto write = [&](Tensor &tensor, size_t i, float value) {
    if (fp16) {
#ifdef ENABLE_FP16
      tensor.getData<_FP16>()[i] = static_cast<_FP16>(value);
#endif
    } else
      tensor.getData<float>()[i] = value;
  };
  for (unsigned int from = 0; from < 11;) {
    const auto rows = from ? 1u : 7u;
    for (unsigned int r = 0; r < rows; ++r) {
      for (unsigned int d = 0; d < 16; ++d)
        write(inputs[0].getVariableRef(), r * 16 + d, queries[from + r][d]);
      for (unsigned int d = 0; d < 8; ++d) {
        write(inputs[1].getVariableRef(), r * 8 + d, keys[from + r][d]);
        write(inputs[2].getVariableRef(), r * 8 + d, values[from + r][d]);
      }
    }
    layer.incremental_forwarding(run, from, from + rows, false);
    for (unsigned int r = 0; r < rows; ++r) {
      const auto t = from + r;
      for (unsigned int h = 0; h < 4; ++h) {
        std::vector<unsigned int> context;
        if (!from) {
          for (unsigned int j = 0; j <= t; ++j)
            context.push_back(j);
        } else {
          context = retained[h / 2];
          for (unsigned int j = 7; j <= t; ++j)
            context.push_back(j);
        }
        const auto q = rotate(queries[t].data() + h * 4, t, false);
        std::vector<double> weights;
        for (const auto j : context) {
          const auto k = rotate(keys[j].data() + (h / 2) * 4, j, true);
          double score = 0;
          for (unsigned int d = 0; d < 4; ++d)
            score += q[d] * k[d] / 2.0;
          weights.push_back(score);
        }
        const auto maximum = *std::max_element(weights.begin(), weights.end());
        double denominator = 0;
        for (auto &w : weights) {
          w = std::exp(w - maximum);
          denominator += w;
        }
        for (unsigned int d = 0; d < 4; ++d) {
          double expected = 0;
          for (size_t j = 0; j < context.size(); ++j)
            expected +=
              weights[j] / denominator * values[context[j]][(h / 2) * 4 + d];
          float actual = 0;
          if (fp16) {
#ifdef ENABLE_FP16
            actual = static_cast<float>(
              output.getVariableRef().getData<_FP16>()[r * 16 + h * 4 + d]);
#endif
          } else
            actual =
              output.getVariableRef().getData<float>()[r * 16 + h * 4 + d];
          EXPECT_NEAR(actual, expected, fp16 ? 1e-3 : 2e-4)
            << "token=" << t << " head=" << h;
        }
      }
    }
    for (unsigned int h = 0; h < 2; ++h) {
      auto ids = retained[h];
      for (unsigned int t = 7; t < from + rows; ++t)
        ids.push_back(t);
      for (size_t r = 0; r < ids.size(); ++r) {
        const auto k = rotate(keys[ids[r]].data() + h * 4, ids[r], true);
        for (unsigned int d = 0; d < 4; ++d) {
          EXPECT_EQ(
            inputs[3].getVariableRef().getData<uint16_t>()[r * 8 + h * 4 + d],
            nntrainer::compute_fp32_to_fp16(k[d]));
          EXPECT_EQ(
            inputs[4].getVariableRef().getData<uint16_t>()[r * 8 + h * 4 + d],
            nntrainer::compute_fp32_to_fp16(values[ids[r]][h * 4 + d]));
        }
      }
    }
    from += rows;
  }
  EXPECT_THROW(layer.incremental_forwarding(run, 11, 11, false),
               std::invalid_argument);
  EXPECT_THROW(layer.incremental_forwarding(run, 11, 12, true),
               std::invalid_argument);
  EXPECT_THROW(layer.incremental_forwarding(run, 11, 12, false),
               std::out_of_range);
}
} // namespace

TEST(ManifoldKV, global_centroid_retains_magnitude_outliers_per_head_p) {
  Tensor keys(TensorDim(1, 1, 4, 2), true);
  const std::array<float, 8> data{1, 10, 1, 1, 1, 1, 10, 1};
  std::copy(data.begin(), data.end(), keys.getData<float>());
  EXPECT_EQ(policy(1).select(keys, 2, 1),
            (std::vector<std::vector<unsigned int>>{{3}, {0}}));
}

TEST(ManifoldKV, local_centroids_handle_final_partial_window_p) {
  Tensor keys(TensorDim(1, 1, 5, 1), true);
  const std::array<float, 5> data{0, 2, 100, 110, 1000};
  std::copy(data.begin(), data.end(), keys.getData<float>());
  EXPECT_EQ(policy(2, 2).select(keys, 1, 1)[0],
            (std::vector<unsigned int>{2, 3}));
  EXPECT_EQ(policy(1).select(keys, 1, 1)[0], (std::vector<unsigned int>{4}));
}

TEST(ManifoldKV, equal_scores_and_budget_above_length_preserve_order_p) {
  Tensor keys(TensorDim(1, 1, 4, 2), true);
  keys.setZero();
  EXPECT_EQ(policy(2).select(keys, 1, 2)[0], (std::vector<unsigned int>{0, 1}));
  EXPECT_EQ(policy(8).select(keys, 1, 2)[0],
            (std::vector<unsigned int>{0, 1, 2, 3}));
}

TEST(ManifoldKV, attention_matches_original_position_reference_p) {
  checkAttention(0, false);
  checkAttention(3, false);
#ifdef ENABLE_FP16
  checkAttention(0, true);
  checkAttention(3, true);
#endif
}

TEST(ManifoldKV, malformed_numbers_n) {
  for (const std::string x : {"", "-1", "+1", "1.5", "true", "4294967296"})
    EXPECT_THROW(causallm::ManifoldKVConfig::parseUnsigned(x), std::exception);
}

TEST(ManifoldKV, invalid_capacity_n) {
  EXPECT_THROW((causallm::ManifoldKVConfig{0, 0}.capacity(8, 2)),
               std::invalid_argument);
  EXPECT_THROW((causallm::ManifoldKVConfig{9, 0}.capacity(8, 2)),
               std::invalid_argument);
}

TEST(ManifoldKV, missing_prompt_or_partial_prefill_n) {
  auto state = policy(2);
  EXPECT_THROW(state.checkStep(0, 7, 4, 16), std::invalid_argument);
  state.consumeProperty("manifold_kv_prompt_length", "7");
  EXPECT_THROW(state.checkStep(0, 3, 4, 16), std::invalid_argument);
}

TEST(ManifoldKV, skipped_rewound_and_multi_token_decode_n) {
  auto state = policy(2);
  state.consumeProperty("manifold_kv_prompt_length", "7");
  state.finishStep(0, 7);
  EXPECT_THROW(state.checkStep(8, 1, 8, 16), std::invalid_argument);
  EXPECT_THROW(state.checkStep(0, 7, 8, 16), std::invalid_argument);
  EXPECT_THROW(state.checkStep(7, 2, 8, 16), std::invalid_argument);
  EXPECT_THROW(state.physicalPosition(3), std::invalid_argument);
}

TEST(ManifoldKV, exhausted_cache_and_position_limit_n) {
  auto state = policy(2);
  state.consumeProperty("manifold_kv_prompt_length", "7");
  EXPECT_THROW(state.checkStep(0, 7, 1, 16), std::out_of_range);
  EXPECT_THROW(state.checkStep(0, 7, 4, 6), std::invalid_argument);
  state.finishStep(0, 7);
  EXPECT_THROW(state.checkStep(7, 1, 2, 16), std::out_of_range);
  EXPECT_THROW(state.checkStep(7, 1, 4, 7), std::invalid_argument);
}

TEST(ManifoldKV, invalid_head_geometry_n) {
  Tensor keys(TensorDim(1, 1, 4, 2), true);
  EXPECT_THROW(policy(2).select(keys, 0, 2), std::invalid_argument);
  EXPECT_THROW(policy(2).select(keys, 2, 2), std::invalid_argument);
}

TEST(ManifoldKV, nonfinite_keys_and_score_overflow_n) {
  Tensor keys(TensorDim(1, 1, 4, 1), true);
  for (float x : {std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::quiet_NaN(),
                  std::numeric_limits<float>::max()}) {
    keys.setZero();
    keys.getData<float>()[0] = x;
    EXPECT_THROW(policy(2).select(keys, 1, 1), std::invalid_argument);
  }
}

TEST(ManifoldKV, mismatched_destination_does_not_modify_cache_n) {
  Tensor keys(TensorDim(1, 1, 4, 2), true), values(TensorDim(1, 1, 4, 2), true);
  Tensor destination(TensorDim(1, 1, 1, 2), true);
  destination.setValue(123.0f);
  EXPECT_THROW(policy(2).compress(keys, values, destination, destination, 1, 2),
               std::invalid_argument);
  EXPECT_EQ(destination.getData<float>()[0], 123.0f);
}
