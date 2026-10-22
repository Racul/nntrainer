// SPDX-License-Identifier: Apache-2.0
/** @file unittest_causallm_streaming_llm.cpp
 * @brief StreamingLLM long-prompt, continuation and allocation integration.
 */
#include <causallm_test_utils.h>
#include <gtest/gtest.h>
#include <qwen3_causallm.h>

#include <filesystem>

namespace {
using Model = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

causallm::json modelConfig() {
  return {{"model_type", "qwen3"},
          {"architectures", {"Qwen3ForCausalLM"}},
          {"bos_token_id", 0},
          {"eos_token_id", {31}},
          {"head_dim", 8},
          {"hidden_size", 64},
          {"intermediate_size", 64},
          {"is_causal", true},
          {"max_position_embeddings", 8},
          {"num_attention_heads", 8},
          {"num_hidden_layers", 2},
          {"num_key_value_heads", 4},
          {"rms_norm_eps", 1e-5},
          {"rope_theta", 10000},
          {"tie_word_embeddings", true},
          {"vocab_size", 32}};
}

class StreamingLLMModel : public ::testing::Test {
protected:
  void SetUp() override {
    files = causallm_test::makeTinyCausalLMFiles(
      "StreamingLLM",
      ::testing::UnitTest::GetInstance()->current_test_info()->name(), "FP32");
    cfg = modelConfig();
    generation = causallm_test::makeTinyGenerationConfig();
    nntr = causallm_test::makeTinyNntrainerConfig(
      files.tokenizer_path, causallm_test::makeTinyFp32DataType());
    nntr["streaming_llm"] = {{"sink_tokens", 2}, {"recent_tokens", 4}};
    nntr["num_to_generate"] = 12;
    // Remove EOS so the tiny model deterministically exercises the whole
    // decode budget, independently of its untrained weights.
    generation["eos_token_id"] = causallm::json::array();
    cfg["eos_token_id"] = causallm::json::array();
  }
  void TearDown() override { std::filesystem::remove_all(files.dir); }
  causallm_test::TinyCausalLMFiles files;
  causallm::json cfg, generation, nntr;
};
} // namespace

TEST_F(StreamingLLMModel,
       long_prompt_decode_and_continuation_use_bounded_cache) {
  Model model(cfg, generation, nntr);
  model.initializeModel();
  ASSERT_EQ(model.getKVCacheRows(0), 6u);
  ASSERT_EQ(model.getKVCacheRows(1), 6u);
  std::string prompt;
  for (unsigned int i = 0; i < 25; ++i)
    prompt += "hello ";
  model.runPrompt(prompt);
  auto metrics = model.getPerformanceMetrics();
  EXPECT_EQ(metrics.prefill_tokens, 25u);
  EXPECT_EQ(metrics.generation_tokens, 12u);
  EXPECT_EQ(metrics.prompt_truncated_tokens, 0u);
  EXPECT_EQ(model.getSessionTokenPosition(), 37u);
  EXPECT_EQ(model.getKvLen(), 6);
  EXPECT_TRUE(model.hasRun());
  model.runPrompt("world hello");
  EXPECT_EQ(model.getSessionTokenPosition(), 51u);
  EXPECT_EQ(model.getPerformanceMetrics().prefill_tokens, 2u);
  model.setSessionTokenPosition(0);
  EXPECT_EQ(model.getKvLen(), 0);
  model.runPrompt("hello");
  EXPECT_EQ(model.getSessionTokenPosition(), 13u);
  EXPECT_THROW(model.setSessionTokenPosition(3), std::invalid_argument);
  EXPECT_THROW(model.save_kvcache("unused", 1), std::invalid_argument);
  EXPECT_THROW(model.load_kvcache("unused", 1), std::invalid_argument);
  EXPECT_THROW(model.setPrecomputedKVCache("unused", 1), std::invalid_argument);
}

TEST_F(StreamingLLMModel, chunk_sizes_produce_same_tokens_after_eviction) {
  Model first(cfg, generation, nntr);
  first.initializeModel();
  first.saveWeight(files.weight_path.string());
  nntr["init_seq_len"] = 1;
  Model second(cfg, generation, nntr);
  second.initializeModel();
  second.loadWeight(files.weight_path.string());
  const std::string prompt =
    "hello world tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10 tok11";
  first.runPrompt(prompt);
  second.runPrompt(prompt);
  EXPECT_EQ(first.getOutputText(), second.getOutputText());
  EXPECT_EQ(first.getSessionTokenPosition(), second.getSessionTokenPosition());
}

TEST_F(StreamingLLMModel, invalid_settings_fail_during_construction) {
  for (const auto &settings :
       std::vector<causallm::json>{{{"sink_tokens", -1}},
                                   {{"recent_tokens", 0}},
                                   {{"recent_tokens", 1.5}},
                                   {{"sink_tokens", 7}, {"recent_tokens", 2}},
                                   {{"typo", 1}},
                                   "invalid"}) {
    nntr["streaming_llm"] = settings;
    EXPECT_THROW(Model(cfg, generation, nntr), std::exception);
  }
  nntr["streaming_llm"] = {{"sink_tokens", 2}, {"recent_tokens", 4}};
  nntr["batch_size"] = 2;
  EXPECT_THROW(Model(cfg, generation, nntr), std::invalid_argument);
  nntr["batch_size"] = 1;
  nntr["skip_prefill"] = true;
  EXPECT_THROW(Model(cfg, generation, nntr), std::invalid_argument);
}

#ifdef ENABLE_FP16
TEST_F(StreamingLLMModel, fp16_activations_stream_beyond_position_table) {
  nntr["model_tensor_type"] = "FP32-FP16";
  Model model(cfg, generation, nntr);
  model.initializeModel();
  model.runPrompt("hello world tok3 tok4 tok5 tok6 tok7 tok8 tok9 tok10 tok11");
  EXPECT_EQ(model.getPerformanceMetrics().prefill_tokens, 11u);
  EXPECT_EQ(model.getPerformanceMetrics().generation_tokens, 12u);
  EXPECT_EQ(model.getSessionTokenPosition(), 23u);
  EXPECT_EQ(model.getKvLen(), 6);
}
#endif
