// SPDX-License-Identifier: Apache-2.0
/** @brief DropKV scoring, allocation, position and model integration tests. */
#include <causallm_test_utils.h>
#include <dropkv.h>
#include <gtest/gtest.h>
#include <layer_context.h>
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

TEST(DropKV, PhysicalAllocationAndPerHeadKVGather_p) {
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

TEST(DropKV, AbsolutePositionIndependentOfStorageLength_p) {
  causallm::CompactKVCache cache;
  cache.reset(8);
  cache.allocate(4, 1);
  cache.live = 2;
  cache.next = 100;
  EXPECT_NO_THROW(cache.checkStep(100, 1, 102));
  cache.next = 101;
  EXPECT_NO_THROW(cache.checkStep(101, 1, 102));
}

TEST(DropKV, InvalidScalarSettings_n) {
  for (const auto *s : {"", "-1", "1.5", "true", "4294967296"})
    EXPECT_THROW(causallm::kvUnsigned(s), std::exception);
  for (const auto *s : {"", "0", "-0.1", "1.01", "nan", "inf", "0.2junk"})
    EXPECT_THROW(causallm::kvRatio(s), std::exception);
}

TEST(DropKV, InvalidPrefillResumeChunkAndLimit_n) {
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

TEST(DropKV, InvalidGatherLeavesStorageIntact_n) {
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

TEST(DropKV, InvalidTopKAndHeadGeometry_n) {
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
TEST(DropKV, SquaredPerturbationAndPostScoreGQA_p) {
  const auto scores =
    causallm::DropKVState::scores({0}, {0, 0, 0}, {0, 2, 10}, 3, 1, 1, 1);
  const double ratio = (1.0 / 3) / (2.0 / 3 + 1e-6);
  EXPECT_NEAR(scores[0], ratio * ratio * 16, 1e-5);
  EXPECT_NEAR(scores[1], ratio * ratio * 4, 1e-5);
  EXPECT_NEAR(scores[2], ratio * ratio * 36, 1e-5);
  const auto gqa =
    causallm::DropKVState::scores({0, 0}, {0, 0, 0}, {0, 2, 10}, 3, 1, 2, 1);
  for (size_t i = 0; i < scores.size(); ++i)
    EXPECT_NEAR(gqa[i], scores[i], 1e-5);

  // Distinct heads distinguish mean-of-loss from loss-of-mean probabilities.
  // Q=+/-log(2), K=[1,0], V=[0,2] gives P=[2/3,1/3] and [1/3,2/3].
  const float log_two = std::log(2.0f);
  const auto distinct = causallm::DropKVState::scores(
    {log_two, -log_two}, {1, 0}, {0, 2}, 2, 1, 2, 1);
  const double high = (2.0 / 3) / (1.0 / 3 + 1e-6);
  const double low = (1.0 / 3) / (2.0 / 3 + 1e-6);
  const double expected = (high * high * 4.0 / 9 + low * low * 16.0 / 9) / 2;
  EXPECT_NEAR(distinct[0], expected, 1e-5);
  EXPECT_NEAR(distinct[1], expected, 1e-5);
  const auto pre_reduced =
    causallm::DropKVState::scores({0}, {1, 0}, {0, 2}, 2, 1, 1, 1);
  EXPECT_GT(std::abs(distinct[0] - pre_reduced[0]), .05);
}

TEST(DropKV, CausalQueryWindowAndOfficialAveragePool_p) {
  const auto s =
    causallm::DropKVState::scores({0, 0}, {0, 0, 0}, {0, 2, 10}, 3, 1, 1, 2);
  const auto last =
    causallm::DropKVState::scores({0}, {0, 0, 0}, {0, 2, 10}, 3, 1, 1, 1);
  const double first_ratio = .5 / (.5 + 1e-6);
  EXPECT_NEAR(s[0] - last[0], first_ratio * first_ratio, 1e-5);
  EXPECT_NEAR(s[1] - last[1], first_ratio * first_ratio, 1e-5);
  EXPECT_NEAR(s[2], last[2], 1e-5); // Future value must not affect first query.
  EXPECT_EQ(causallm::DropKVState::pool({3, 0, 6}, 3, "avg"),
            (std::vector<double>{1, 3, 2}));
  EXPECT_EQ(causallm::DropKVState::pool({3, 0, 6}, 3),
            (std::vector<double>{3, 6, 6}));
}

TEST(DropKV, ProtectedRecentWindowAndPhysicalShrink_p) {
  causallm::DropKVState state;
  state.config.keep_ratio = .25;
  state.config.window = 2;
  state.config.kernel = 1;
  state.cache.reset(8);
  state.cache.allocate(8, 1);
  for (unsigned int t = 0; t < 8; ++t) {
    halfWrite(state.cache.key, t, 0);
    halfWrite(state.cache.value, t, t + 1);
  }
  Tensor q(1, 1, 8, 2);
  q.setValue(0);
  state.afterAttention(q, 2, 1, 1, 8, 2);
  EXPECT_EQ(state.cache.live, 2u);
  EXPECT_EQ(state.cache.key.height(), 4u);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 0), 7);
  EXPECT_FLOAT_EQ(halfRead(state.cache.value, 1), 8);
}

TEST(DropKV, InvalidWindowKernelAndRatio_n) {
  for (const auto &cfg :
       {causallm::DropKVConfig{0, 8, 11}, {.5, 0, 11}, {.5, 8, 2}})
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
  EXPECT_THROW(causallm::DropKVState::pool({1}, 0), std::invalid_argument);
}

TEST(DropKV, InvalidScoringGeometry_n) {
  EXPECT_THROW(causallm::DropKVState::scores({}, {0}, {1}, 1, 1, 1, 1),
               std::invalid_argument);
  EXPECT_THROW(causallm::DropKVState::scores({0}, {0}, {1}, 1, 1, 1, 2),
               std::invalid_argument);
}

TEST(DropKV, UnknownPropertyAndNonfiniteLogits_n) {
  causallm::DropKVState state;
  EXPECT_FALSE(state.enabled());
  EXPECT_THROW(state.consumeProperty("dropkv_typo", "1"),
               std::invalid_argument);
  EXPECT_THROW(
    causallm::DropKVState::scores({std::numeric_limits<float>::infinity()}, {1},
                                  {1}, 1, 1, 1, 1),
    std::invalid_argument);
}
using TinyQwen3 = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;
TEST(DropKV, TinyModelActivationDisabledAndUnprunedEquivalence_p) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("DropKV", "activation", "FP32");
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
      nntr["dropkv"] = {{"enabled", false}};
    if (mode == 1)
      nntr["dropkv"] = {
        {"enabled", true}, {"keep_ratio", 1.0}, {"window", 2}, {"kernel", 1}};
    if (mode == 2)
      nntr["dropkv"] = {
        {"enabled", true}, {"keep_ratio", .5}, {"window", 2}, {"kernel", 1}};
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
      EXPECT_EQ(log.find("[DropKV"), std::string::npos);
    } else {
      EXPECT_NE(log.find("[DropKV layer=0]"), std::string::npos);
      EXPECT_THROW(model.runPrompt("hello"), std::invalid_argument);
    }
  }
}
TEST(DropKV, TinyModelRejectsSkipPrefillAndResume_n) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("DropKV", "guards", "FP32");
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

  nntr["dropkv"] = {
    {"enabled", true}, {"keep_ratio", .5}, {"window", 2}, {"kernel", 1}};
  nntr["skip_prefill"] = true;
  EXPECT_THROW(TinyQwen3 model(cfg, generation, nntr), std::invalid_argument);
  nntr["skip_prefill"] = false;
  TinyQwen3 model(cfg, generation, nntr);
  EXPECT_THROW(model.setSessionTokenPosition(1), std::invalid_argument);
  EXPECT_THROW(model.save_kvcache("unused", 1), std::invalid_argument);
}
} // namespace
