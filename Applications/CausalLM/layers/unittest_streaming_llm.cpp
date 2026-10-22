// SPDX-License-Identifier: Apache-2.0
/** @file unittest_streaming_llm.cpp
 * @brief StreamingLLM cache and independent attention-reference tests.
 */
#include <fp16.h>
#include <gtest/gtest.h>
#include <layer_context.h>
#include <mha_core.h>
#include <streaming_llm.h>
#include <var_grad.h>

#include <array>
#include <cmath>
#include <vector>

namespace {
float half(float x) {
  return nntrainer::compute_fp16_to_fp32(nntrainer::compute_fp32_to_fp16(x));
}

// Deliberately build the retained token list from original token indices,
// rather than using the implementation's eviction or position helpers.
std::vector<unsigned int>
cacheTokenIndices(unsigned int seen, unsigned int sinks, unsigned int recent) {
  std::vector<unsigned int> retained;
  for (unsigned int i = 0; i < seen; ++i)
    if (i < sinks || seen <= sinks + recent || i >= seen - recent)
      retained.push_back(i);
  return retained;
}

// The authors' eval_long_ppl.py attends to the retained past and the current
// token, then trims the returned raw K/V cache. Evicting first changes both
// the attention context and the query's RoPE position by one at capacity.
std::vector<float>
attentionReference(unsigned int token, unsigned int sinks, unsigned int recent,
                   const std::vector<std::array<float, 16>> &q,
                   const std::vector<std::array<float, 8>> &k,
                   const std::vector<std::array<float, 8>> &v) {
  auto retained = cacheTokenIndices(token, sinks, recent);
  retained.push_back(token);
  auto rotate = [](const float *x, unsigned int pos, bool round_input,
                   bool round_output) {
    std::array<float, 4> result;
    for (unsigned int d = 0; d < 2; ++d) {
      const float angle = pos / std::pow(10000.0f, d / 2.0f);
      const float a = round_input ? half(x[d]) : x[d];
      const float b = round_input ? half(x[d + 2]) : x[d + 2];
      result[d] = a * std::cos(angle) - b * std::sin(angle);
      result[d + 2] = a * std::sin(angle) + b * std::cos(angle);
    }
    if (round_output)
      for (auto &x : result)
        x = half(x);
    return result;
  };
  std::vector<float> out(16, 0);
  for (unsigned int h = 0; h < 4; ++h) {
    auto query =
      rotate(q[token].data() + h * 4, retained.size() - 1, false, false);
    std::vector<double> scores;
    for (unsigned int j = 0; j < retained.size(); ++j) {
      auto key = rotate(k[retained[j]].data() + (h / 2) * 4, j, true, true);
      double score = 0;
      for (unsigned int d = 0; d < 4; ++d)
        score += query[d] * key[d] / 2.0;
      scores.push_back(score);
    }
    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0;
    for (auto &score : scores) {
      score = std::exp(score - maximum);
      denominator += score;
    }
    for (unsigned int j = 0; j < retained.size(); ++j)
      for (unsigned int d = 0; d < 4; ++d)
        out[h * 4 + d] +=
          scores[j] / denominator * half(v[retained[j]][(h / 2) * 4 + d]);
  }
  return out;
}

void checkAttention(unsigned int chunk, unsigned int sinks,
                    unsigned int recent) {
  using nntrainer::TensorDim;
  const TensorDim qdim(1, 1, 8, 16);
  const TensorDim kdim(1, 1, 8, 8);
  const TensorDim cache_dim(1, 1, sinks + recent, 8,
                            TensorDim::TensorType(TensorDim::Format::NCHW,
                                                  TensorDim::DataType::UINT16));
  causallm::MHACoreLayer layer;
  layer.setProperty({"num_heads=4", "num_heads_kv=2", "max_timestep=8",
                     "max_position_embeddings=8", "rope_theta=10000",
                     "is_causal=true",
                     "streaming_llm_sink_tokens=" + std::to_string(sinks),
                     "streaming_llm_recent_tokens=" + std::to_string(recent)});
  nntrainer::InitLayerContext init({qdim, kdim, kdim, cache_dim, cache_dim},
                                   {false}, false, "streaming_attention");
  layer.finalize(init);
  std::vector<nntrainer::Var_Grad> inputs;
  for (const auto &dim : {qdim, kdim, kdim, cache_dim, cache_dim})
    inputs.emplace_back(dim, nntrainer::Initializer::ZEROS, false, true,
                        "input");
  nntrainer::Var_Grad output(qdim, nntrainer::Initializer::ZEROS, false, true,
                             "output");
  nntrainer::RunLayerContext run(
    "streaming_attention", false, 0, false, 1, nullptr, false, {},
    {&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4]}, {&output},
    {});
  std::vector<std::array<float, 16>> queries(37);
  std::vector<std::array<float, 8>> keys(37), values(37);
  for (unsigned int t = 0; t < queries.size(); ++t) {
    for (unsigned int i = 0; i < 16; ++i)
      queries[t][i] = std::sin((t + 1) * (i + 1) * 0.17f);
    for (unsigned int i = 0; i < 8; ++i) {
      keys[t][i] = std::cos((t + 1) * (i + 1) * 0.11f);
      values[t][i] = std::sin((t + 1) * (i + 1) * 0.23f);
    }
  }
  for (unsigned int t = 0; t < queries.size();) {
    const unsigned int rows = std::min<unsigned int>(chunk, queries.size() - t);
    for (unsigned int row = 0; row < rows; ++row) {
      std::copy(queries[t + row].begin(), queries[t + row].end(),
                inputs[0].getVariableRef().getData<float>() + row * 16);
      std::copy(keys[t + row].begin(), keys[t + row].end(),
                inputs[1].getVariableRef().getData<float>() + row * 8);
      std::copy(values[t + row].begin(), values[t + row].end(),
                inputs[2].getVariableRef().getData<float>() + row * 8);
    }
    layer.incremental_forwarding(run, t, t + rows, false);
    for (unsigned int row = 0; row < rows; ++row) {
      auto expected =
        attentionReference(t + row, sinks, recent, queries, keys, values);
      for (unsigned int i = 0; i < 16; ++i)
        EXPECT_NEAR(output.getVariableRef().getData<float>()[row * 16 + i],
                    expected[i], 2e-4f)
          << "token=" << t + row << " feature=" << i;
    }
    t += rows;
    // Persisted K must stay unrotated, and K/V must describe exactly the same
    // original tokens after every forward, including the first overflow.
    const auto retained = cacheTokenIndices(t, sinks, recent);
    for (unsigned int row = 0; row < retained.size(); ++row)
      for (unsigned int i = 0; i < 8; ++i) {
        EXPECT_EQ(inputs[3].getVariableRef().getData<uint16_t>()[row * 8 + i],
                  nntrainer::compute_fp32_to_fp16(keys[retained[row]][i]));
        EXPECT_EQ(inputs[4].getVariableRef().getData<uint16_t>()[row * 8 + i],
                  nntrainer::compute_fp32_to_fp16(values[retained[row]][i]));
      }
  }
  EXPECT_THROW(layer.incremental_forwarding(run, 37, 37, false),
               std::invalid_argument);
  EXPECT_THROW(layer.incremental_forwarding(run, 37, 38, true),
               std::invalid_argument);
  // A new session at position zero must ignore every stale row.
  layer.incremental_forwarding(run, 0, 1, false);
  for (unsigned int i = 0; i < 16; ++i)
    EXPECT_NEAR(
      output.getVariableRef().getData<float>()[i],
      half(inputs[2].getVariableRef().getData<float>()[(i / 8) * 4 + i % 4]),
      2e-4f);
  layer.setProperty({"max_timestep=" + std::to_string(sinks + recent - 1)});
  EXPECT_THROW(layer.incremental_forwarding(run, 1, 2, false),
               std::invalid_argument);
}
} // namespace

TEST(StreamingLLM, decode_matches_cache_local_rope_reference) {
  checkAttention(1, 2, 4);
}
TEST(StreamingLLM, chunks_crossing_capacity_match_reference) {
  checkAttention(8, 2, 4);
}
TEST(StreamingLLM, window_attention_ablation_matches_reference) {
  checkAttention(3, 0, 6);
}
TEST(StreamingLLM, paper_figure_four_attends_before_eviction) {
  // Fig. 4: past [0,1,2,3,6,7,8] and current token 9 occupy positions 0..7.
  EXPECT_EQ(cacheTokenIndices(9, 4, 3),
            (std::vector<unsigned int>{0, 1, 2, 3, 6, 7, 8}));
  checkAttention(1, 4, 3);
}
TEST(StreamingLLM, cache_can_fill_entire_model_position_window) {
  // A full eight-row cache needs a ninth scratch/LUT row for the next query.
  checkAttention(8, 2, 6);
}
TEST(StreamingLLM, single_recent_token_preserves_sinks) {
  checkAttention(8, 4, 1);
}

TEST(StreamingLLM, preserves_sink_and_recent_tokens) {
  causallm::StreamingLLMConfig config{4, 3};
  std::array<unsigned int, 8> keys{}, values{};
  for (unsigned int t = 0; t < 1000; ++t) {
    const auto slot = std::min(t, config.capacity());
    keys[slot] = t;
    values[slot] = t + 1000;
    const auto retained =
      config.evict(slot + 1, keys.data(), sizeof(unsigned int));
    EXPECT_EQ(config.evict(slot + 1, values.data(), sizeof(unsigned int)),
              retained);
    for (unsigned int i = 0; i < retained; ++i) {
      const auto expected = t < 7 || i < 4 ? i : t - (retained - 1 - i);
      EXPECT_EQ(keys[i], expected);
      EXPECT_EQ(values[i], expected + 1000);
    }
  }
}

TEST(StreamingLLM, rejects_invalid_configuration_and_steps) {
  for (const std::string value : {"-1", "+1", "1.5", "true", "", "4294967296"})
    EXPECT_THROW(causallm::StreamingLLMConfig::parseUnsigned(value),
                 std::exception);
  causallm::StreamingLLMConfig config{UINT_MAX, 1};
  EXPECT_THROW(config.capacity(), std::invalid_argument);
  config = {UINT_MAX - 1, 1};
  EXPECT_THROW(config.attentionCapacity(), std::invalid_argument);
  config = {4, 3};
  std::array<unsigned int, 7> keys{}, values{};
  EXPECT_THROW(config.evict(0, keys.data(), sizeof(unsigned int)),
               std::invalid_argument);
  EXPECT_THROW(config.evict(7, nullptr, sizeof(unsigned int)),
               std::invalid_argument);
  EXPECT_THROW(config.evict(7, values.data(), 0), std::invalid_argument);
}
