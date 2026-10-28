// SPDX-License-Identifier: Apache-2.0
/** @file unittest_causallm_manifold_kv.cpp
 * @brief ManifoldKV prompt semantics, physical allocation and session tests.
 */
#include <causallm_test_utils.h>
#include <gtest/gtest.h>
#include <mha_core.h>
#include <qwen3_causallm.h>

namespace {
using Model = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

class ManifoldKVModel : public ::testing::Test {
protected:
  static void initialize(Model &model) {
    model.initializeModel();
    model.forEachLayer(
      [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
        if (layer.getName() == "output_of_causallm")
          return;
        for (unsigned int i = 0; i < context.getNumWeights(); ++i)
          context.getWeight(i).setValue(
            causallm_test::isRmsNormLayerType(layer.getType()) ? 1.0f : 0.01f);
      });
  }

  void SetUp() override {
    files = causallm_test::makeTinyCausalLMFiles(
      "ManifoldKV",
      ::testing::UnitTest::GetInstance()->current_test_info()->name(), "FP32");
    cfg = {{"model_type", "qwen3"},
           {"architectures", {"Qwen3ForCausalLM"}},
           {"bos_token_id", 0},
           {"eos_token_id", causallm::json::array()},
           {"head_dim", 8},
           {"hidden_size", 64},
           {"intermediate_size", 64},
           {"is_causal", true},
           {"max_position_embeddings", 32},
           {"num_attention_heads", 8},
           {"num_hidden_layers", 2},
           {"num_key_value_heads", 4},
           {"rms_norm_eps", 1e-5},
           {"rope_theta", 10000},
           {"tie_word_embeddings", true},
           {"vocab_size", 32}};
    generation = causallm_test::makeTinyGenerationConfig();
    generation["eos_token_id"] = causallm::json::array();
    nntr = causallm_test::makeTinyNntrainerConfig(
      files.tokenizer_path, causallm_test::makeTinyFp32DataType());
    nntr["init_seq_len"] = 12;
    nntr["max_seq_len"] = 32;
    nntr["num_to_generate"] = 4;
    nntr["manifold_kv"] = {{"budget", 3}};
  }
  void TearDown() override { std::filesystem::remove_all(files.dir); }
  causallm_test::TinyCausalLMFiles files;
  causallm::json cfg, generation, nntr;
  const std::string prompt = "hello world tok3 tok4 tok5 tok6 tok7";
};
} // namespace

TEST_F(ManifoldKVModel,
       compressed_prompt_decode_and_reset_use_small_allocations_p) {
  for (unsigned int window : {0u, 3u}) {
    nntr["manifold_kv"]["window_size"] = window;
    std::vector<std::string> types{"FP32-FP32"};
#ifdef ENABLE_FP16
    types.push_back("FP32-FP16");
#endif
    for (const auto &type : types) {
      nntr["model_tensor_type"] = type;
      Model model(cfg, generation, nntr);
      initialize(model);
      EXPECT_EQ(model.getKVCacheRows(0), 7u);
      EXPECT_EQ(model.getKVCacheRows(1), 7u);
      model.runPrompt(prompt);
      const auto metrics = model.getPerformanceMetrics();
      EXPECT_EQ(metrics.prefill_tokens, 7u);
      EXPECT_EQ(metrics.generation_tokens, 4u);
      EXPECT_EQ(metrics.prompt_truncated_tokens, 0u);
      EXPECT_EQ(model.getSessionTokenPosition(), 11u);
      EXPECT_EQ(model.getKvLen(), 7);
      EXPECT_TRUE(model.hasRun());
      model.setSessionTokenPosition(0);
      EXPECT_EQ(model.getKvLen(), 0);
      model.runPrompt("hello");
      EXPECT_EQ(model.getSessionTokenPosition(), 5u);
      EXPECT_EQ(model.getKvLen(), 5);
    }
  }
}

TEST_F(ManifoldKVModel, compression_preserves_dense_prefill_logits_p) {
  auto dense_config = nntr;
  dense_config.erase("manifold_kv");
  Model dense(cfg, generation, dense_config);
  initialize(dense);
  dense.saveWeight(files.weight_path.string());
  Model compressed(cfg, generation, nntr);
  initialize(compressed);
  compressed.loadWeight(files.weight_path.string());
  compressed.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      if (layer.getType() == causallm::MHACoreLayer::type)
        layer.setProperty({"manifold_kv_prompt_length=7"});
    });
  const std::vector<unsigned int> ids{1, 2, 3, 4, 5, 6, 7};
  const auto expected = dense.prefillLogitsFromIds(ids);
  const auto actual = compressed.prefillLogitsFromIds(ids);
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actual[i]));
    EXPECT_NEAR(actual[i], expected[i], 1e-4f);
  }
}

TEST_F(ManifoldKVModel, no_eviction_decode_matches_dense_with_twelve_layers_p) {
  cfg["num_hidden_layers"] = 12;
  nntr["manifold_kv"]["budget"] = 8;
  auto dense_config = nntr;
  dense_config.erase("manifold_kv");
  Model dense(cfg, generation, dense_config);
  initialize(dense);
  dense.saveWeight(files.weight_path.string());
  const auto expected = dense.greedyGenerateFromIds({1, 2, 3, 4, 5, 6, 7}, 4);
  Model compressed(cfg, generation, nntr);
  initialize(compressed);
  compressed.loadWeight(files.weight_path.string());
  compressed.runPrompt(prompt);
  ASSERT_EQ(compressed.getPerformanceMetrics().generation_tokens,
            expected.size());
  for (size_t i = 0; i < expected.size(); ++i)
    EXPECT_EQ(compressed.tokenAt(i), expected[i]);
}

TEST_F(ManifoldKVModel, invalid_json_settings_n) {
  for (const auto &settings :
       std::vector<causallm::json>{{{"budget", 0}},
                                   {{"budget", -1}},
                                   {{"budget", 1.5}},
                                   {{"budget", 33}},
                                   {{"budget", "3"}},
                                   {{"budget", 4294967296ULL}},
                                   {{"budget", 3}, {"window_size", -1}},
                                   {{"unknown", 3}},
                                   "invalid"}) {
    nntr["manifold_kv"] = settings;
    EXPECT_THROW(Model(cfg, generation, nntr), std::exception);
  }
}

TEST_F(ManifoldKVModel, incompatible_execution_settings_n) {
  for (const auto &setting : std::vector<causallm::json>{
         {{"batch_size", 2}},
         {{"skip_prefill", true}},
         {{"fsu", true}},
         {{"streaming_llm", {{"sink_tokens", 2}, {"recent_tokens", 4}}}},
         {{"snapkv", {{"budget", 3}}}}}) {
    auto invalid = nntr;
    invalid.update(setting);
    EXPECT_THROW(Model(cfg, generation, invalid), std::exception);
  }
  cfg["model_type"] = "qwen2";
  EXPECT_THROW(Model(cfg, generation, nntr), std::invalid_argument);
}

TEST_F(ManifoldKVModel, empty_oversized_and_position_exhausted_prompts_n) {
  Model model(cfg, generation, nntr);
  initialize(model);
  EXPECT_THROW(model.runPrompt(""), std::invalid_argument);
  EXPECT_THROW(model.runPrompt(prompt + " tok8 tok9 tok10 tok11 tok12 tok13"),
               std::invalid_argument);
  cfg["max_position_embeddings"] = 7;
  Model short_positions(cfg, generation, nntr);
  initialize(short_positions);
  EXPECT_THROW(short_positions.runPrompt(prompt), std::invalid_argument);
}

TEST_F(ManifoldKVModel, continuation_and_reposition_require_reset_n) {
  Model model(cfg, generation, nntr);
  initialize(model);
  model.runPrompt(prompt);
  EXPECT_THROW(model.runPrompt("hello"), std::invalid_argument);
  EXPECT_THROW(model.setSessionTokenPosition(2), std::invalid_argument);
  EXPECT_EQ(model.getSessionTokenPosition(), 11u);
  EXPECT_EQ(model.getKvLen(), 7);
}

TEST_F(ManifoldKVModel, compressed_cache_serialization_n) {
  Model model(cfg, generation, nntr);
  initialize(model);
  EXPECT_THROW(model.save_kvcache("unused", 1), std::invalid_argument);
  EXPECT_THROW(model.load_kvcache("unused", 1), std::invalid_argument);
  EXPECT_THROW(model.setPrecomputedKVCache("unused", 1), std::invalid_argument);
  EXPECT_THROW(model.resumeFromKVCacheFile("unused", 1), std::invalid_argument);
}

TEST_F(ManifoldKVModel, uninitialized_inference_n) {
  Model model(cfg, generation, nntr);
  EXPECT_THROW(model.runPrompt(prompt), std::runtime_error);
}
