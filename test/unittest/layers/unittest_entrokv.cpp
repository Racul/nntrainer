// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_entrokv.cpp
 * @brief Tensor-backed EntroKV allocation, compaction and decode invariants.
 */

#include <entrokv.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {
using causallm::EntroKVConfig;
using causallm::EntroKVState;
using DType = ml::train::TensorDim::DataType;

/** @brief Two GQA groups: one concentrated, the other uniform. */
nntrainer::Tensor attention(unsigned int from = 0, unsigned int rows = 8) {
  nntrainer::Tensor p(
    1, 1, EntroKVState::triangle(from + rows) - EntroKVState::triangle(from),
    4);
  p.setValue(0.0f);
  for (unsigned int q = from; q < from + rows; ++q) {
    const size_t base =
      (EntroKVState::triangle(q) - EntroKVState::triangle(from)) * 4;
    for (unsigned int h = 0; h < 4; ++h) {
      if (h < 2)
        p.getData<float>()[base + std::min(4u, q) * 4 + h] = 1.0f;
      else
        for (unsigned int j = 0; j <= q; ++j)
          p.getData<float>()[base + j * 4 + h] = 1.0f / (q + 1);
    }
  }
  return p;
}

/** @brief Encode the source row, head and component in each K/V value. */
template <typename T>
void labelCache(nntrainer::Tensor &k, nntrainer::Tensor &v) {
  for (unsigned int row = 0; row < k.height(); ++row)
    for (unsigned int h = 0; h < 2; ++h)
      for (unsigned int d = 0; d < 2; ++d) {
        const size_t offset = row * 4 + h * 2 + d;
        k.getData<T>()[offset] = static_cast<T>(row * 100 + h * 10 + d + 1);
        v.getData<T>()[offset] = static_cast<T>(row * 100 + h * 10 + d + 2001);
      }
}

EntroKVConfig adaptiveConfig() {
  EntroKVConfig cfg;
  cfg.beta = 1.0;
  cfg.window = 2;
  cfg.kernel = 1;
  return cfg;
}
} // namespace

TEST(EntroKV, validates_configuration_and_does_not_invent_calibration) {
  EntroKVState disabled;
  disabled.configure({}, 0);
  EXPECT_FALSE(disabled.enabled());
  auto cfg = adaptiveConfig();
  cfg.alpha = 0.5;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
  cfg.baseline_entropy = 0.45;
  cfg.calibrated = true;
  EXPECT_NO_THROW(cfg.validate());
  cfg.beta = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
  cfg.beta = 1.01;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
  cfg.beta = 0.5;
  cfg.kernel = 2;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
  EXPECT_THROW(EntroKVConfig::integer("-1"), std::invalid_argument);
  EXPECT_THROW(EntroKVConfig::integer("4294967296"), std::invalid_argument);
  EXPECT_THROW(EntroKVConfig::real("0.5x"), std::invalid_argument);
}

TEST(EntroKV, entropy_changes_head_quotas_and_preserves_matching_kv_rows) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  EXPECT_NO_THROW(state.checkStep(0, 8, 12, 12));
  auto first = attention(0, 3);
  auto last = attention(3, 5);
  state.observe(first, 0, 3, 4, 2);
  state.observe(last, 3, 5, 4, 2);
  nntrainer::Tensor k(1, 1, 12, 4), v(1, 1, 12, 4);
  labelCache<float>(k, v);
  state.finishPrefill(k, v, 2, 2);
  ASSERT_EQ(state.quotas(), (std::vector<unsigned int>{2, 7}));
  EXPECT_NEAR(state.entropies()[0], 0.0, 1e-7);
  EXPECT_NEAR(state.entropies()[1],
              (std::log(7.0) + std::log(8.0)) / (2 * std::log(8.0)), 1e-6);
  EXPECT_EQ(state.retainedSpan(), 7u);
  const std::vector<unsigned int> high{0, 1, 2, 3, 4, 6, 7};
  for (unsigned int row = 0; row < 7; ++row) {
    for (unsigned int d = 0; d < 2; ++d) {
      EXPECT_FLOAT_EQ(k.getData<float>()[row * 4 + 2 + d],
                      high[row] * 100 + 10 + d + 1);
      EXPECT_FLOAT_EQ(v.getData<float>()[row * 4 + 2 + d],
                      high[row] * 100 + 10 + d + 2001);
      if (row < 2) {
        EXPECT_FLOAT_EQ(k.getData<float>()[row * 4 + d],
                        (6 + row) * 100 + d + 1);
        EXPECT_FLOAT_EQ(v.getData<float>()[row * 4 + d],
                        (6 + row) * 100 + d + 2001);
      } else {
        EXPECT_FLOAT_EQ(k.getData<float>()[row * 4 + d], 0.0f);
        EXPECT_FLOAT_EQ(v.getData<float>()[row * 4 + d], 0.0f);
      }
    }
  }
  // Allocation remains at capacity, and the logical RoPE origin stays at 8.
  EXPECT_EQ(k.height(), 12u);
  for (unsigned int logical = 8; logical < 11; ++logical) {
    EXPECT_NO_THROW(state.checkStep(logical, 1, 12, 12));
    EXPECT_EQ(state.physicalPosition(logical), logical - 1);
    nntrainer::Tensor scores(1, 1, state.physicalPosition(logical) + 1, 4);
    scores.setValue(0.0f);
    state.maskDecode(scores, 4);
    for (unsigned int j = 0; j <= state.physicalPosition(logical); ++j)
      for (unsigned int h = 0; h < 4; ++h)
        EXPECT_EQ(std::isinf(scores.getData<float>()[j * 4 + h]),
                  h < 2 && j >= 2 && j < 7);
    state.finishDecode();
  }
  EXPECT_THROW(state.checkStep(8, 1, 12, 12), std::invalid_argument);
  EXPECT_THROW(state.checkStep(11, 2, 12, 12), std::invalid_argument);
  EXPECT_THROW(state.checkStep(11, 1, 12, 11), std::invalid_argument);
  EXPECT_THROW(state.configure(adaptiveConfig(), 8), std::invalid_argument);
}

TEST(EntroKV, topk_is_deterministic_and_preserves_uint16_payloads) {
  auto cfg = adaptiveConfig();
  cfg.alpha = 0.0;
  cfg.calibrated = true;
  cfg.baseline_entropy = 1.0;
  cfg.beta = 0.625;
  EntroKVState state;
  state.configure(cfg, 8);
  auto p = attention();
  state.observe(p, 0, 8, 4, 2);
  const auto type = ml::train::TensorDim::TensorType{
    ml::train::TensorDim::Format::NCHW, DType::UINT16};
  nntrainer::Tensor k(1, 1, 12, 4, type), v(1, 1, 12, 4, type);
  labelCache<uint16_t>(k, v);
  state.finishPrefill(k, v, 2, 2);
  ASSERT_EQ(state.quotas(), (std::vector<unsigned int>{5, 5}));
  const std::vector<std::vector<unsigned int>> kept{{0, 1, 4, 6, 7},
                                                    {0, 1, 2, 6, 7}};
  for (unsigned int h = 0; h < 2; ++h)
    for (unsigned int row = 0; row < 5; ++row)
      for (unsigned int d = 0; d < 2; ++d) {
        EXPECT_EQ(k.getData<uint16_t>()[row * 4 + h * 2 + d],
                  kept[h][row] * 100 + h * 10 + d + 1);
        EXPECT_EQ(v.getData<uint16_t>()[row * 4 + h * 2 + d],
                  kept[h][row] * 100 + h * 10 + d + 2001);
      }
}

TEST(EntroKV, full_budget_and_single_token_prefill_keep_original_positions) {
  for (unsigned int length : {1u, 8u}) {
    auto cfg = adaptiveConfig();
    cfg.alpha = 0.0;
    cfg.baseline_entropy = 1.0;
    cfg.calibrated = true;
    EntroKVState state;
    state.configure(cfg, length);
    auto p = attention(0, length);
    state.observe(p, 0, length, 4, 2);
    nntrainer::Tensor k(1, 1, 12, 4), v(1, 1, 12, 4);
    labelCache<float>(k, v);
    state.finishPrefill(k, v, 2, 2);
    EXPECT_EQ(state.retainedSpan(), length);
    EXPECT_EQ(state.physicalPosition(length), length);
    for (unsigned int row = 0; row < length; ++row)
      for (unsigned int h = 0; h < 2; ++h)
        EXPECT_FLOAT_EQ(k.getData<float>()[row * 4 + h * 2],
                        row * 100 + h * 10 + 1);
  }
}

TEST(EntroKV, pooling_retains_neighbors_and_gqa_averages_query_head_entropy) {
  auto cfg = adaptiveConfig();
  cfg.alpha = 0.0;
  cfg.calibrated = true;
  cfg.baseline_entropy = 1.0;
  cfg.beta = 0.625;
  cfg.kernel = 3;
  EntroKVState state;
  state.configure(cfg, 8);
  auto p = attention();
  state.observe(p, 0, 8, 4, 2);
  nntrainer::Tensor k(1, 1, 12, 4), v(1, 1, 12, 4);
  labelCache<float>(k, v);
  state.finishPrefill(k, v, 2, 2);
  const std::vector<unsigned int> kept{3, 4, 5, 6, 7};
  for (unsigned int row = 0; row < kept.size(); ++row)
    EXPECT_FLOAT_EQ(k.getData<float>()[row * 4], kept[row] * 100 + 1);

  // Opposite one-hot query heads both have zero entropy. Entropy of their
  // averaged attention would be nonzero, so this detects the wrong GQA order.
  for (unsigned int q = 6; q < 8; ++q) {
    const size_t base = EntroKVState::triangle(q) * 4;
    p.getData<float>()[base + 4 * 4 + 1] = 0.0f;
    p.getData<float>()[base + 1] = 1.0f;
  }
  EntroKVState grouped;
  grouped.configure(adaptiveConfig(), 8);
  grouped.observe(p, 0, 8, 4, 2);
  labelCache<float>(k, v);
  grouped.finishPrefill(k, v, 2, 2);
  EXPECT_NEAR(grouped.entropies()[0], 0.0, 1e-7);
  EXPECT_EQ(grouped.quotas()[0], 2u);
}

TEST(EntroKV, rejects_chunked_or_incomplete_prefill_and_invalid_probabilities) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  EXPECT_THROW(state.checkStep(0, 4, 12, 12), std::invalid_argument);
  EXPECT_THROW(state.checkStep(1, 8, 12, 12), std::invalid_argument);
  EXPECT_THROW(state.checkStep(0, 8, 7, 12), std::invalid_argument);
  nntrainer::Tensor k(1, 1, 12, 4), v(1, 1, 12, 4);
  EXPECT_THROW(state.finishPrefill(k, v, 2, 2), std::invalid_argument);
  auto p = attention();
  EXPECT_THROW(state.observe(p, 0, 8, 4, 3), std::invalid_argument);
  p.getData<float>()[EntroKVState::triangle(7) * 4] =
    std::numeric_limits<float>::quiet_NaN();
  EXPECT_THROW(state.observe(p, 0, 8, 4, 2), std::runtime_error);
}

TEST(EntroKV, rejects_allocation_values_outside_their_ranges) {
  for (double value : {-0.1, 1.1, std::numeric_limits<double>::infinity()}) {
    auto cfg = adaptiveConfig();
    cfg.beta = value;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg = adaptiveConfig();
    cfg.alpha = value;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg = adaptiveConfig();
    cfg.baseline_entropy = value;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
  }
}

TEST(EntroKV, rejects_zero_window_and_invalid_pooling_kernel) {
  auto cfg = adaptiveConfig();
  cfg.window = 0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
  for (unsigned int kernel : {0u, 2u, 4u}) {
    cfg = adaptiveConfig();
    cfg.kernel = kernel;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
  }
}

TEST(EntroKV, rejects_empty_enabled_prompt_and_malformed_runtime_numbers) {
  EntroKVState state;
  EXPECT_THROW(state.configure(adaptiveConfig(), 0), std::invalid_argument);
  for (const char *value : {"", "1.5", "1x", "-1", "+1"})
    EXPECT_THROW(EntroKVConfig::integer(value), std::invalid_argument);
  EXPECT_THROW(EntroKVConfig::real("nan"), std::invalid_argument);
  EXPECT_THROW(EntroKVConfig::real("inf"), std::invalid_argument);
}

TEST(EntroKV, rejects_observation_ranges_outside_the_prompt) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  auto p = attention();
  EXPECT_THROW(state.observe(p, 9, 1, 4, 2), std::invalid_argument);
  EXPECT_THROW(state.observe(p, 7, 2, 4, 2), std::invalid_argument);
  EXPECT_THROW(state.observe(p, 0, 8, 0, 2), std::invalid_argument);
  EXPECT_THROW(state.observe(p, 0, 8, 4, 0), std::invalid_argument);
}

TEST(EntroKV, rejects_wrong_observation_buffer_size_or_dtype) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  nntrainer::Tensor short_scores(1, 1, 1, 4);
  EXPECT_THROW(state.observe(short_scores, 0, 8, 4, 2), std::invalid_argument);
  nntrainer::Tensor integer_scores(
    1, 1, 36, 4, ml::train::TensorDim::Format::NCHW, DType::UINT16);
  EXPECT_THROW(state.observe(integer_scores, 0, 8, 4, 2),
               std::invalid_argument);
}

TEST(EntroKV, rejects_cache_shape_or_key_value_mismatch_before_compaction) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  auto p = attention();
  state.observe(p, 0, 8, 4, 2);
  nntrainer::Tensor k(1, 1, 12, 4), different(1, 1, 11, 4);
  EXPECT_THROW(state.finishPrefill(k, different, 2, 2), std::invalid_argument);
  nntrainer::Tensor small_k(1, 1, 7, 4), small_v(1, 1, 7, 4);
  EXPECT_THROW(state.finishPrefill(small_k, small_v, 2, 2),
               std::invalid_argument);
  nntrainer::Tensor batched_k(2, 1, 12, 4), batched_v(2, 1, 12, 4);
  EXPECT_THROW(state.finishPrefill(batched_k, batched_v, 2, 2),
               std::invalid_argument);
  EXPECT_THROW(state.finishPrefill(k, k, 2, 0), std::invalid_argument);
}

TEST(EntroKV, rejects_unsupported_cache_storage) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  auto p = attention();
  state.observe(p, 0, 8, 4, 2);
  nntrainer::Tensor k(1, 1, 12, 4, ml::train::TensorDim::Format::NCHW,
                      DType::UINT8);
  nntrainer::Tensor v(1, 1, 12, 4, ml::train::TensorDim::Format::NCHW,
                      DType::UINT8);
  EXPECT_THROW(state.finishPrefill(k, v, 2, 2), std::invalid_argument);
}

TEST(EntroKV, rejects_decode_masks_before_prefill_or_with_wrong_geometry) {
  EntroKVState state;
  state.configure(adaptiveConfig(), 8);
  nntrainer::Tensor scores(1, 1, 8, 4);
  EXPECT_THROW(state.maskDecode(scores, 4), std::invalid_argument);
  auto p = attention();
  state.observe(p, 0, 8, 4, 2);
  nntrainer::Tensor k(1, 1, 12, 4), v(1, 1, 12, 4);
  labelCache<float>(k, v);
  state.finishPrefill(k, v, 2, 2);
  EXPECT_THROW(state.maskDecode(scores, 3), std::invalid_argument);
  nntrainer::Tensor short_scores(1, 1, 1, 4);
  EXPECT_THROW(state.maskDecode(short_scores, 4), std::invalid_argument);
}

TEST(EntroKV, uncompressed_transition_preserves_logical_positions) {
  auto cfg = adaptiveConfig();
  cfg.alpha = 0.0;
  cfg.baseline_entropy = 1.0;
  cfg.calibrated = true;
  EntroKVState state;
  state.configure(cfg, 8);
  ASSERT_TRUE(state.keepsFullCache());
  state.checkStep(0, 8, 12, 12);
  state.finishUncompressed(2);
  EXPECT_EQ(state.quotas(), (std::vector<unsigned int>{8, 8}));
  EXPECT_TRUE(state.entropies().empty());
  for (unsigned int logical = 8; logical < 11; ++logical) {
    EXPECT_NO_THROW(state.checkStep(logical, 1, 12, 12));
    EXPECT_EQ(state.physicalPosition(logical), logical);
    state.finishDecode();
  }
  EXPECT_THROW(state.finishUncompressed(2), std::invalid_argument);
  EntroKVState compressible;
  compressible.configure(adaptiveConfig(), 8);
  EXPECT_FALSE(compressible.keepsFullCache());
  EXPECT_THROW(compressible.finishUncompressed(2), std::invalid_argument);
}
