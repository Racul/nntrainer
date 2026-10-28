// SPDX-License-Identifier: Apache-2.0
/** @brief ManifoldKV scoring, allocation, position and model integration tests.
 */
#include <causallm_test_utils.h>
#include <gtest/gtest.h>
#include <layer_context.h>
#include <manifoldkv.h>
#include <qwen3_causallm.h>

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

TEST(ManifoldKV, PhysicalAllocationAndPerHeadKVGather_p) {
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

TEST(ManifoldKV, AbsolutePositionIndependentOfStorageLength_p) {
  causallm::CompactKVCache cache;
  cache.reset(8);
  cache.allocate(4, 1);
  cache.live = 2;
  cache.next = 100;
  EXPECT_NO_THROW(cache.checkStep(100, 1, 102));
  cache.next = 101;
  EXPECT_NO_THROW(cache.checkStep(101, 1, 102));
}

TEST(ManifoldKV, InvalidScalarSettings_n) {
  for (const auto *s : {"", "-1", "1.5", "true", "4294967296"})
    EXPECT_THROW(causallm::kvUnsigned(s), std::exception);
  for (const auto *s : {"", "0", "-0.1", "1.01", "nan", "inf", "0.2junk"})
    EXPECT_THROW(causallm::kvRatio(s), std::exception);
}

TEST(ManifoldKV, InvalidPrefillResumeChunkAndLimit_n) {
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

TEST(ManifoldKV, InvalidGatherLeavesStorageIntact_n) {
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

TEST(ManifoldKV, InvalidTopKAndHeadGeometry_n) {
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
TEST(ManifoldKV, RawCentroidOutlierAndLocalWindow_p) {
  const std::vector<float> keys{0, 0, 0, 8};
  const auto global = causallm::ManifoldKVState::scores(keys, 4, 1, 0);
  EXPECT_EQ(global, (std::vector<double>{2, 2, 2, 6}));
  EXPECT_EQ(causallm::kvTop(global, 2), (std::vector<unsigned int>{0, 3}));
  const auto local = causallm::ManifoldKVState::scores(keys, 4, 1, 2);
  EXPECT_EQ(local, (std::vector<double>{0, 0, 4, 4}));
  EXPECT_EQ(causallm::kvTop(local, 2), (std::vector<unsigned int>{2, 3}));
}

TEST(ManifoldKV, PrefillCompressionAndTwoDecodeSteps_p) {
  causallm::ManifoldKVState state;
  state.config.keep_ratio = .25;
  state.cache.reset(8);
  state.cache.allocate(8, 1);
  for (unsigned int i = 0; i < 8; ++i) {
    halfWrite(state.cache.key, i, i == 7 ? 8 : 0);
    halfWrite(state.cache.value, i, 10 + i);
  }
  Tensor q(1, 1, 8, 1);
  state.afterAttention(q, 1, 1, 1, 8, 2);
  EXPECT_EQ(state.cache.key.height(), 4u);
  EXPECT_EQ(state.cache.live, 2u);
  EXPECT_EQ(state.cache.next, 8u);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 0), 10);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 1), 17);
  for (unsigned int logical = 8; logical < 10; ++logical) {
    EXPECT_NO_THROW(state.cache.checkStep(logical, 1, 10));
    halfWrite(state.cache.key, state.cache.live, 1);
    halfWrite(state.cache.value, state.cache.live, logical);
    state.afterAttention(q, 1, 1, 1, 1, 2);
  }
  EXPECT_EQ(state.cache.live, 4u);
  EXPECT_EQ(state.cache.next, 10u);
}

TEST(ManifoldKV, InvalidRatioAndUnknownProperty_n) {
  causallm::ManifoldKVConfig config;
  config.keep_ratio = 0;
  EXPECT_THROW(config.validate(), std::invalid_argument);
  config.keep_ratio = std::numeric_limits<double>::infinity();
  EXPECT_THROW(config.validate(), std::invalid_argument);
  causallm::ManifoldKVState state;
  EXPECT_FALSE(state.enabled());
  EXPECT_THROW(state.consumeProperty("manifoldkv_typo", "1"),
               std::invalid_argument);
}

TEST(ManifoldKV, InvalidKeyDimensions_n) {
  EXPECT_THROW(causallm::ManifoldKVState::scores({}, 0, 1, 0),
               std::invalid_argument);
  EXPECT_THROW(causallm::ManifoldKVState::scores({1}, 1, 2, 0),
               std::invalid_argument);
}
using TinyQwen3 = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;
TEST(ManifoldKV, TinyModelActivationDisabledAndUnprunedEquivalence_p) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("ManifoldKV", "activation", "FP32");
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
      nntr["manifoldkv"] = {{"enabled", false}};
    if (mode == 1)
      nntr["manifoldkv"] = {{"enabled", true}, {"keep_ratio", 1.0}};
    if (mode == 2)
      nntr["manifoldkv"] = {{"enabled", true}, {"keep_ratio", .5}};
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
      EXPECT_EQ(log.find("[ManifoldKV"), std::string::npos);
    } else {
      EXPECT_NE(log.find("[ManifoldKV layer=0]"), std::string::npos);
      EXPECT_THROW(model.runPrompt("hello"), std::invalid_argument);
    }
  }
}
TEST(ManifoldKV, TinyModelRejectsSkipPrefillAndResume_n) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("ManifoldKV", "guards", "FP32");
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

  nntr["manifoldkv"] = {{"enabled", true}, {"keep_ratio", .5}};
  nntr["skip_prefill"] = true;
  EXPECT_THROW(TinyQwen3 model(cfg, generation, nntr), std::invalid_argument);
  nntr["skip_prefill"] = false;
  TinyQwen3 model(cfg, generation, nntr);
  EXPECT_THROW(model.setSessionTokenPosition(1), std::invalid_argument);
  EXPECT_THROW(model.save_kvcache("unused", 1), std::invalid_argument);
}
} // namespace
