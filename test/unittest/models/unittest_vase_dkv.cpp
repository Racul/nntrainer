// SPDX-License-Identifier: Apache-2.0
/** @brief VaSEDKV scoring, allocation, position and model integration tests. */
#include <causallm_test_utils.h>
#include <gtest/gtest.h>
#include <layer_context.h>
#include <qwen3_causallm.h>
#include <vase_dkv.h>

namespace {
using nntrainer::Tdatatype;
using nntrainer::Tensor;

void halfWrite(Tensor &t, size_t index, float x) {
  reinterpret_cast<uint16_t *>(t.getData())[index] =
    nntrainer::compute_fp32_to_fp16(x);
}
float halfRead(const Tensor &t, size_t index) {
  return nntrainer::compute_fp16_to_fp32(
    reinterpret_cast<const uint16_t *>(t.getData())[index]);
}

TEST(VaSEDKV, PhysicalAllocationAndPerHeadKVGather_p) {
  causallm::CompactKVCache cache;
  cache.reset(8);
  cache.allocate(8, 2);
  cache.live = 8;
  for (unsigned int i = 0; i < 16; ++i) {
    halfWrite(cache.key, i, i);
    halfWrite(cache.value, i, 100 + i);
  }
  auto old_owner = cache.key.getMemoryData();
  const auto old_bytes = cache.bytes();
  cache.compact({{1, 6}, {2, 7}}, 1, 3);
  EXPECT_EQ(cache.live, 2u);
  EXPECT_EQ(cache.key.height(), 3u);
  EXPECT_LT(cache.bytes(), old_bytes);
  EXPECT_NE(cache.key.getMemoryData(), old_owner);
  EXPECT_EQ(old_owner.use_count(), 1);
  EXPECT_FLOAT_EQ(halfRead(cache.key, 0), 2);
  EXPECT_FLOAT_EQ(halfRead(cache.key, 1), 5);
  EXPECT_FLOAT_EQ(halfRead(cache.key, 2), 12);
  EXPECT_FLOAT_EQ(halfRead(cache.key, 3), 15);
  EXPECT_FLOAT_EQ(halfRead(cache.value, 0), 102);
  EXPECT_FLOAT_EQ(halfRead(cache.value, 3), 115);
  cache.reset(2);
  EXPECT_EQ(cache.live, 0u);
  EXPECT_EQ(cache.next, 0u);
}

TEST(VaSEDKV, AbsolutePositionIndependentOfStorageLength_p) {
  causallm::CompactKVCache cache;
  cache.reset(8);
  cache.allocate(4, 1);
  cache.live = 2;
  cache.next = 100;
  EXPECT_NO_THROW(cache.checkStep(100, 1, 102));
  cache.next = 101;
  EXPECT_NO_THROW(cache.checkStep(101, 1, 102));
}

TEST(VaSEDKV, InvalidScalarSettings_n) {
  for (const auto *s : {"", "-1", "1.5", "true", "4294967296"})
    EXPECT_THROW(causallm::kvUnsigned(s), std::exception);
  for (const auto *s : {"", "0", "-0.1", "1.01", "nan", "inf", "0.2junk"})
    EXPECT_THROW(causallm::kvRatio(s), std::exception);
}

TEST(VaSEDKV, InvalidPrefillResumeChunkAndLimit_n) {
  causallm::CompactKVCache cache;
  EXPECT_THROW(cache.reset(0), std::invalid_argument);
  cache.reset(8);
  EXPECT_THROW(cache.checkStep(0, 4, 16), std::invalid_argument);
  EXPECT_THROW(cache.checkStep(1, 8, 16), std::invalid_argument);
  EXPECT_THROW(cache.checkStep(0, 8, 7), std::out_of_range);
  cache.next = 8;
  EXPECT_THROW(cache.checkStep(8, 2, 16), std::invalid_argument);
  EXPECT_THROW(cache.checkStep(8, 1, 8), std::out_of_range);
}

TEST(VaSEDKV, InvalidGatherLeavesStorageIntact_n) {
  causallm::CompactKVCache cache;
  cache.allocate(8, 2);
  cache.live = 8;
  auto owner = cache.key.getMemoryData();
  EXPECT_THROW(cache.compact({{2, 1}, {1, 2}}, 1, 4), std::invalid_argument);
  EXPECT_THROW(cache.compact({{1, 1}, {1, 2}}, 1, 4), std::invalid_argument);
  EXPECT_THROW(cache.compact({{1}, {2, 3}}, 1, 4), std::invalid_argument);
  EXPECT_THROW(cache.compact({{1, 8}, {1, 2}}, 1, 4), std::out_of_range);
  EXPECT_THROW(cache.compact({{1, 2}, {1, 2}}, 1, 1), std::invalid_argument);
  EXPECT_EQ(cache.key.getMemoryData(), owner);
}

TEST(VaSEDKV, InvalidTopKAndHeadGeometry_n) {
  EXPECT_THROW(causallm::kvTop({1, 2}, 3), std::invalid_argument);
  EXPECT_THROW(causallm::kvTop({std::numeric_limits<double>::quiet_NaN()}, 1),
               std::invalid_argument);
  causallm::CompactKVCache cache;
  cache.allocate(4, 2);
  cache.live = 4;
  std::vector<float> k, v;
  EXPECT_THROW(cache.head(2, 2, 1, k, v), std::invalid_argument);
  EXPECT_THROW(cache.head(0, 3, 1, k, v), std::invalid_argument);
}
TEST(VaSEDKV, ProjectedProductValueAwarenessAndSinks_p) {
  const auto s = causallm::VaSEDKVState::scores(
    {1, 1, 1, 1, 2, 1}, {1, 1, 1, 1, 1, 3}, {1}, 6, 1, 1);
  EXPECT_NEAR(s[4], 4.0 / 17, 1e-12);
  EXPECT_NEAR(s[5], 9.0 / 17, 1e-12);
  EXPECT_TRUE(std::isinf(s[0]));
  EXPECT_EQ(causallm::kvTop(s, 5), (std::vector<unsigned int>{0, 1, 2, 3, 5}));
}

TEST(VaSEDKV, NoPrefillEvictionPeriodicDecodeAndRecency_p) {
  causallm::VaSEDKVState state;
  state.config.budget = 8;
  state.config.buffer = 2;
  state.config.validate();
  state.cache.reset(10);
  state.cache.allocate(10, 1);
  state.rng.seed(42);
  for (unsigned int t = 0; t < 10; ++t) {
    halfWrite(state.cache.key, t, 1);
    halfWrite(state.cache.value, t, t + 1);
  }
  Tensor q(1, 1, 10, 1);
  state.afterAttention(q, 1, 1, 1, 10, 4);
  EXPECT_EQ(state.evictions, 0u);
  EXPECT_EQ(state.cache.live, 10u);
  for (unsigned int t = 10; t < 12; ++t) {
    halfWrite(state.cache.key, state.cache.live, 1);
    halfWrite(state.cache.value, state.cache.live, t + 1);
    state.afterAttention(q, 1, 1, 1, 1, 4);
    EXPECT_EQ(state.evictions, t == 10 ? 0u : 1u);
  }
  EXPECT_EQ(state.cache.live, 8u);
  EXPECT_EQ(state.cache.key.height(), 10u);
  EXPECT_EQ(state.cache.next, 12u);
  for (unsigned int t = 0; t < 4; ++t)
    EXPECT_FLOAT_EQ(halfRead(state.cache.value, t), t + 1);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 4), 9);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 5), 10);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 6), 11);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 7), 12);
  for (unsigned int t = 12; t < 14; ++t) {
    EXPECT_NO_THROW(state.cache.checkStep(t, 1, 32));
    halfWrite(state.cache.key, state.cache.live, 1);
    halfWrite(state.cache.value, state.cache.live, t + 1);
    state.afterAttention(q, 1, 1, 1, 1, 4);
  }
  EXPECT_EQ(state.evictions, 2u);
  EXPECT_EQ(state.cache.live, 8u);
}

TEST(VaSEDKV, ResamplesProjectionAndReproducesSeed_p) {
  causallm::VaSEDKVState a, b;
  a.rng.seed(42);
  b.rng.seed(42);
  const auto first = a.projection(8);
  EXPECT_EQ(first, b.projection(8));
  EXPECT_NE(first, a.projection(8));
}

TEST(VaSEDKV, InvalidBudgetBufferRank_n) {
  for (const auto &cfg : {causallm::VaSEDKVConfig{8, 0, 20, 42},
                          {8, 8, 20, 42},
                          {8, 3, 20, 42},
                          {8, 2, 0, 42}})
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(VaSEDKV, InvalidProjectionGeometry_n) {
  EXPECT_THROW(causallm::VaSEDKVState::scores({1}, {1}, {}, 1, 1, 1),
               std::invalid_argument);
  EXPECT_THROW(causallm::VaSEDKVState::scores({1}, {}, {1}, 1, 1, 1),
               std::invalid_argument);
}

TEST(VaSEDKV, UnknownPropertyAndNonfiniteProjection_n) {
  causallm::VaSEDKVState state;
  EXPECT_FALSE(state.enabled());
  EXPECT_THROW(state.consumeProperty("vase_dkv_typo", "1"),
               std::invalid_argument);
  EXPECT_THROW(causallm::VaSEDKVState::scores(
                 {std::numeric_limits<float>::infinity()}, {1}, {1}, 1, 1, 1),
               std::invalid_argument);
}
using TinyQwen3 = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;
TEST(VaSEDKV, TinyModelActivationDisabledAndUnprunedEquivalence_p) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("VaSEDKV", "activation", "FP32");
  std::string baseline;
  for (const int mode : {-1, 0, 1, 2}) {
    causallm::json cfg = {{"model_type", "qwen3"},
                          {"architectures", {"Qwen3ForCausalLM"}},
                          {"bos_token_id", 0},
                          {"eos_token_id", {31}},
                          {"head_dim", 8},
                          {"hidden_size", 64},
                          {"intermediate_size", 64},
                          {"is_causal", true},
                          {"max_position_embeddings", 16},
                          {"num_attention_heads", 8},
                          {"num_hidden_layers", 1},
                          {"num_key_value_heads", 4},
                          {"rms_norm_eps", 1e-5},
                          {"rope_theta", 10000},
                          {"tie_word_embeddings", true},
                          {"vocab_size", 32}};
    auto generation = causallm_test::makeTinyGenerationConfig();
    auto nntr = causallm_test::makeTinyNntrainerConfig(
      files.tokenizer_path, causallm_test::makeTinyFp32DataType());
    nntr["init_seq_len"] = 7;
    nntr["max_seq_len"] = 16;
    nntr["num_to_generate"] = 3;

    if (mode == 0)
      nntr["vase_dkv"] = {{"enabled", false}};
    if (mode == 1)
      nntr["vase_dkv"] = {{"enabled", true}, {"budget", 16}, {"buffer", 2}};
    if (mode == 2)
      nntr["vase_dkv"] = {{"enabled", true}, {"budget", 6}, {"buffer", 2}};
    TinyQwen3 model(cfg, generation, nntr);
    model.initializeModel();
    model.forEachLayer(
      [](ml::train::Layer &layer, nntrainer::RunLayerContext &ctx, void *) {
        for (unsigned int i = 0; i < ctx.getNumWeights(); ++i) {
          auto &w = ctx.getWeight(i);
          if (w.getDataType() != Tdatatype::FP32)
            continue;
          w.setValue(0.0f);
          if (causallm_test::isRmsNormLayerType(layer.getType()))
            w.setValue(1.0f);
          else if (layer.getName() == "embedding0")
            for (unsigned int token = 0; token < 31; ++token)
              w.setValue(0, 0, token, token % 8, token == 4 ? 2.0f : 1.0f);
          else if (layer.getName() == "layer0_wq" ||
                   layer.getName() == "layer0_wk" ||
                   layer.getName() == "layer0_wv" ||
                   layer.getName() == "layer0_attention_out")
            for (unsigned int x = 0; x < std::min(w.height(), w.width()); ++x)
              w.setValue(0, 0, x, x, 0.25f);
        }
      });

    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(model.runPrompt("hello tok4 hello tok4 hello tok4"));
    const auto log = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(model.hasRun());
    if (mode == -1)
      baseline = model.getOutputText();
    if (mode != 2) {
      EXPECT_EQ(model.getOutputText(), baseline);
    }
    EXPECT_FALSE(model.getOutputText().empty());
    if (mode <= 0) {
      EXPECT_EQ(log.find("[VaSE-DKV"), std::string::npos);
    } else {
      EXPECT_NE(log.find("[VaSE-DKV layer=0]"), std::string::npos);
      EXPECT_THROW(model.runPrompt("hello"), std::invalid_argument);
    }
  }
}
TEST(VaSEDKV, TinyModelRejectsSkipPrefillAndResume_n) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("VaSEDKV", "guards", "FP32");
  causallm::json cfg = {{"model_type", "qwen3"},
                        {"architectures", {"Qwen3ForCausalLM"}},
                        {"bos_token_id", 0},
                        {"eos_token_id", {31}},
                        {"head_dim", 8},
                        {"hidden_size", 64},
                        {"intermediate_size", 64},
                        {"is_causal", true},
                        {"max_position_embeddings", 16},
                        {"num_attention_heads", 8},
                        {"num_hidden_layers", 1},
                        {"num_key_value_heads", 4},
                        {"rms_norm_eps", 1e-5},
                        {"rope_theta", 10000},
                        {"tie_word_embeddings", true},
                        {"vocab_size", 32}};
  auto generation = causallm_test::makeTinyGenerationConfig();
  auto nntr = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, causallm_test::makeTinyFp32DataType());
  nntr["init_seq_len"] = 7;
  nntr["max_seq_len"] = 16;
  nntr["num_to_generate"] = 3;

  nntr["vase_dkv"] = {{"enabled", true}, {"budget", 6}, {"buffer", 2}};
  nntr["skip_prefill"] = true;
  EXPECT_THROW(TinyQwen3 model(cfg, generation, nntr), std::invalid_argument);
  nntr["skip_prefill"] = false;
  TinyQwen3 model(cfg, generation, nntr);
  EXPECT_THROW(model.setSessionTokenPosition(1), std::invalid_argument);
  EXPECT_THROW(model.save_kvcache("unused", 1), std::invalid_argument);
}
} // namespace
