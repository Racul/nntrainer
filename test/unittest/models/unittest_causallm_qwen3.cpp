// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_causallm_qwen3.cpp
 * @date   15 May 2026
 * @brief  Tiny Qwen3 model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <capkv.h>
#include <causallm_test_utils.h>
#include <fp16.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_causallm.h>
#include <qwen3_embedding.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

/**
 * @brief Helpers for tiny Qwen3 model tests
 */
namespace {

/**
 * @brief Tiny Qwen3 CausalLM adapter (inference shared by CausalLMTestAdapter)
 */
using TinyQwen3CausalLM =
  causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

/**
 * @brief Populate deterministic tiny Qwen3 weights for golden token tests
 */
void setupQwen3DeterministicWeights(TinyQwen3CausalLM &model) {
  model.forEachLayer(
    [](ml::train::Layer &layer, nntrainer::RunLayerContext &context, void *) {
      if (layer.getName() == "output_of_causallm")
        return;

      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (causallm_test::isRmsNormLayerType(layer.getType())) {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0.0f);
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    });
}

/**
 * @brief Files generated for one tiny Qwen3 embedding test invocation
 */
struct TinyQwen3EmbeddingFiles {
  std::filesystem::path dir;            /**< Temporary test directory */
  std::filesystem::path tokenizer_path; /**< Tiny tokenizer.json path */
  std::filesystem::path modules_path;   /**< Tiny modules.json path */
  std::filesystem::path weight_path;    /**< Tiny model weight path */
};

/**
 * @brief Tiny Qwen3 Embedding adapter for model-level tests
 */
class TinyQwen3Embedding final : public causallm::Qwen3Embedding {
public:
  /**
   * @brief Construct a tiny Qwen3 Embedding test adapter
   */
  TinyQwen3Embedding(causallm::json &cfg, causallm::json &generation_cfg,
                     causallm::json &nntr_cfg) :
    causallm::Transformer(cfg, generation_cfg, nntr_cfg,
                          causallm::ModelType::EMBEDDING),
    causallm::Qwen3Embedding(cfg, generation_cfg, nntr_cfg) {}

  /**
   * @brief Initialize the tiny Qwen3 embedding model
   */
  void initializeModel() { initialize(); }

  /**
   * @brief Save tiny Qwen3 embedding weights with dtype conversion
   */
  void saveWeightWithDtype(
    const std::string &path,
    const std::map<std::string, ml::train::TensorDim::DataType>
      &layer_dtype_map) {
    save_weight(path, ml::train::TensorDim::DataType::NONE, layer_dtype_map);
  }

  /**
   * @brief Load tiny Qwen3 embedding model weights
   */
  void loadWeight(const std::string &path) { load_weight(path); }

  /**
   * @brief Set deterministic tiny Qwen3 embedding weights
   */
  void setDeterministicWeights() {
    auto set_weights = [](ml::train::Layer &layer,
                          nntrainer::RunLayerContext &context, void *) {
      for (unsigned int i = 0; i < context.getNumWeights(); ++i) {
        auto &weight = context.getWeight(i);
        if (weight.getDataType() != ml::train::TensorDim::DataType::FP32)
          continue;

        weight.setValue(0.0f);
        if (causallm_test::isRmsNormLayerType(layer.getType())) {
          weight.setValue(1.0f);
        } else if (layer.getName() == "embedding0") {
          weight.setValue(0, 0, 1, 0, 1.0f);
          weight.setValue(0, 0, 4, 0, 2.0f);
        }
      }
    };

    model->forEachLayer(set_weights, nullptr);
  }

  /**
   * @brief Encode one prompt and copy the embedding output
   */
  std::vector<float> encodePrompt(const std::string &prompt) {
    auto output = encode(prompt);
    std::vector<float> embedding(output[0], output[0] + BATCH_SIZE * DIM);
    for (auto &out : output)
      delete[] out;
    return embedding;
  }
};

/**
 * @brief Write a string into a file
 */
void writeTextFile(const std::filesystem::path &path,
                   const std::string &content) {
  std::ofstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("failed to open " + path.string());

  file << content;
  if (!file.good())
    throw std::runtime_error("failed to write " + path.string());
}

/**
 * @brief Make tiny SentenceTransformer module config files
 */
std::filesystem::path
writeTinyQwen3EmbeddingModules(const std::filesystem::path &dir) {
  auto modules_path = dir / "modules.json";
  auto pooling_dir = dir / "1_Pooling";
  std::filesystem::create_directories(pooling_dir);

  writeTextFile(modules_path, R"([
  {
    "idx": 0,
    "name": "0",
    "path": "",
    "type": "sentence_transformers.models.Transformer"
  },
  {
    "idx": 1,
    "name": "1",
    "path": "1_Pooling",
    "type": "sentence_transformers.models.Pooling"
  }
])");

  writeTextFile(pooling_dir / "config.json", R"({
  "word_embedding_dimension": 64,
  "pooling_mode_cls_token": false,
  "pooling_mode_mean_tokens": false,
  "pooling_mode_max_tokens": false,
  "pooling_mode_mean_sqrt_len_tokens": false,
  "pooling_mode_weightedmean_tokens": false,
  "pooling_mode_lasttoken": true,
  "include_prompt": true
})");

  return modules_path;
}

/**
 * @brief Make the tiny Qwen3 model config
 */
causallm::json makeTinyQwen3Config() {
  return {
    {"architectures", {"Qwen3ForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Make the tiny Qwen3 embedding model config
 */
causallm::json makeTinyQwen3EmbeddingConfig() {
  auto cfg = makeTinyQwen3Config();
  cfg["vocab_size"] = 33;
  return cfg;
}

/**
 * @brief Make tiny Qwen3 embedding test files
 */
TinyQwen3EmbeddingFiles makeQwen3EmbeddingFiles() {
  const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
  std::string suite_name = "Qwen3EmbeddingTinyModelTest";
  std::string test_name = "Unknown";

  if (info != nullptr) {
    suite_name = info->test_suite_name();
    test_name = info->name();
  }

  auto files = causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                    "Qwen3Embedding_Q40_FP32");

  return {
    files.dir,
    files.tokenizer_path,
    writeTinyQwen3EmbeddingModules(files.dir),
    files.dir / "qwen3_embedding_tiny.bin",
  };
}

/**
 * @brief Make the tiny Qwen3 embedding nntrainer config
 */
causallm::json makeTinyQwen3EmbeddingNntrainerConfig(
  const TinyQwen3EmbeddingFiles &files,
  const causallm_test::TinyCausalLMDataType &data_type) {
  auto cfg =
    causallm_test::makeTinyNntrainerConfig(files.tokenizer_path, data_type);
  cfg["model_type"] = "Embedding";
  cfg["module_config_path"] = files.modules_path.string();
  return cfg;
}

/**
 * @brief Make the tiny Qwen3 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3LayerDtypeMap(const causallm_test::TinyCausalLMDataType &data_type) {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;

  if (data_type.embedding_dtype != "FP32")
    dtype_map["embedding0"] =
      causallm_test::toTensorDataType(data_type.embedding_dtype);

  if (data_type.fc_layer_dtype != "FP32") {
    const auto dtype =
      causallm_test::toTensorDataType(data_type.fc_layer_dtype);
    dtype_map["layer0_wq"] = dtype;
    dtype_map["layer0_wk"] = dtype;
    dtype_map["layer0_wv"] = dtype;
    dtype_map["layer0_attention_out"] = dtype;
    dtype_map["layer0_ffn_up"] = dtype;
    dtype_map["layer0_ffn_gate"] = dtype;
    dtype_map["layer0_ffn_down"] = dtype;
  }

  if (data_type.lmhead_dtype != "FP32")
    dtype_map["output_of_causallm"] =
      causallm_test::toTensorDataType(data_type.lmhead_dtype);

  return dtype_map;
}

/**
 * @brief Make the tiny Qwen3 embedding Q4_0 layer dtype map
 */
std::map<std::string, ml::train::TensorDim::DataType>
makeQwen3EmbeddingQ40LayerDtypeMap() {
  std::map<std::string, ml::train::TensorDim::DataType> dtype_map;
  const auto dtype = ml::train::TensorDim::DataType::Q4_0;

  dtype_map["embedding0"] = dtype;
  dtype_map["layer0_wq"] = dtype;
  dtype_map["layer0_wk"] = dtype;
  dtype_map["layer0_wv"] = dtype;
  dtype_map["layer0_attention_out"] = dtype;
  dtype_map["layer0_ffn_up"] = dtype;
  dtype_map["layer0_ffn_gate"] = dtype;
  dtype_map["layer0_ffn_down"] = dtype;

  return dtype_map;
}

/**
 * @brief Create a loaded tiny Qwen3 embedding model
 */
std::unique_ptr<TinyQwen3Embedding>
makeLoadedQwen3Embedding(const TinyQwen3EmbeddingFiles &files) {
  const auto fp32_data_type = causallm_test::makeTinyFp32DataType();
  const auto q40_data_type = causallm_test::makeTinyQ40Fp32DataType();
  auto source_model_cfg = makeTinyQwen3EmbeddingConfig();
  auto source_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto source_nntr_cfg =
    makeTinyQwen3EmbeddingNntrainerConfig(files, fp32_data_type);

  TinyQwen3Embedding source(source_model_cfg, source_generation_cfg,
                            source_nntr_cfg);
  source.initializeModel();
  source.setDeterministicWeights();
  source.saveWeightWithDtype(files.weight_path.string(),
                             makeQwen3EmbeddingQ40LayerDtypeMap());

  auto loaded_model_cfg = makeTinyQwen3EmbeddingConfig();
  auto loaded_generation_cfg = causallm_test::makeTinyGenerationConfig();
  auto loaded_nntr_cfg =
    makeTinyQwen3EmbeddingNntrainerConfig(files, q40_data_type);
  auto loaded = std::make_unique<TinyQwen3Embedding>(
    loaded_model_cfg, loaded_generation_cfg, loaded_nntr_cfg);
  loaded->initializeModel();
  loaded->loadWeight(files.weight_path.string());

  return loaded;
}

/**
 * @brief Make the expected tiny Qwen3 prefill logits
 */
std::vector<float> makeExpectedQwen3Logits() {
  std::vector<float> logits(32, 0.0f);
  logits[1] = 7.99936008f;
  logits[4] = 15.99872017f;
  return logits;
}

/**
 * @brief Make a Qwen3 tiny CausalLM test case
 */
causallm_test::TinyCausalLMCase
makeQwen3Case(const causallm_test::TinyCausalLMDataType &data_type) {
  return {
    "Qwen3_" + data_type.name,
    data_type,
    {"hello tok4", makeExpectedQwen3Logits(),
     data_type.name == "FP32"       ? 1e-4f
     : data_type.name == "Q40_FP16" ? 2e-3f
                                    : 1e-3f},
    makeTinyQwen3Config,
    makeQwen3LayerDtypeMap,
    [](causallm::json &cfg, causallm::json &generation_cfg,
       causallm::json &nntr_cfg) {
      return std::make_unique<TinyQwen3CausalLM>(cfg, generation_cfg, nntr_cfg);
    },
    [](causallm_test::TinyCausalLMRunner &runner) {
      setupQwen3DeterministicWeights(static_cast<TinyQwen3CausalLM &>(runner));
    },
  };
}

/**
 * @brief Parameterized fixture for tiny CausalLM model cases
 */
class CausalLMTinyModelTest
  : public ::testing::TestWithParam<causallm_test::TinyCausalLMCase> {
protected:
  /**
   * @brief Make test files for the current parameterized case
   */
  causallm_test::TinyCausalLMFiles makeFiles() const {
    const auto *info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string suite_name = "CausalLMTinyModelTest";
    std::string test_name = "Unknown";

    if (info != nullptr) {
      suite_name = info->test_suite_name();
      test_name = info->name();
    }

    return causallm_test::makeTinyCausalLMFiles(suite_name, test_name,
                                                GetParam().name);
  }
};

/**
 * @brief Test that greedy generation chooses the argmax logit
 */
TEST_P(CausalLMTinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  const auto files = makeFiles();
  auto config =
    causallm_test::makeTinyCausalLMConfig(GetParam(), files.tokenizer_path);
  auto model =
    GetParam().create_model(config.model, config.generation, config.nntrainer);

  causallm_test::expectGreedyGenerationSelectsArgmax(*model);
}

/**
 * @brief Test that a save/load round-trip preserves logits
 */
TEST_P(CausalLMTinyModelTest, WeightRoundTripProducesSameLogits) {
  const auto files = makeFiles();
  causallm_test::expectWeightRoundTripProducesSameLogits(GetParam(), files);
}

/**
 * @brief Test that a prompt produces the expected golden logits
 */
TEST_P(CausalLMTinyModelTest, PromptProducesExpectedLogits) {
  const auto files = makeFiles();
  causallm_test::expectPromptProducesExpectedLogits(GetParam(), files);
}

INSTANTIATE_TEST_SUITE_P(
  Qwen3, CausalLMTinyModelTest,
  ::testing::Values(makeQwen3Case(causallm_test::makeTinyFp32DataType()),
                    makeQwen3Case(causallm_test::makeTinyQ40Fp32DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });

#ifdef ENABLE_FP16
INSTANTIATE_TEST_SUITE_P(
  Qwen3Fp16, CausalLMTinyModelTest,
  ::testing::Values(makeQwen3Case(causallm_test::makeTinyQ40Fp16DataType())),
  [](const ::testing::TestParamInfo<causallm_test::TinyCausalLMCase> &info) {
    return info.param.name;
  });
#endif

/**
 * @brief Test that a tiny Qwen3 embedding model can save/load and encode
 */
TEST(Qwen3EmbeddingTinyModelTest,
     WeightRoundTripEncodesPromptWithQ40VocabRemainder) {
  const auto files = makeQwen3EmbeddingFiles();
  auto model = makeLoadedQwen3Embedding(files);

  std::vector<float> embedding;
  ASSERT_NO_THROW(embedding = model->encodePrompt("hello tok4"));
  ASSERT_EQ(embedding.size(), 64u);

  bool has_non_zero_value = false;
  for (float value : embedding) {
    EXPECT_TRUE(std::isfinite(value));
    has_non_zero_value = has_non_zero_value || std::abs(value) > 1e-5f;
  }
  EXPECT_TRUE(has_non_zero_value);
}

// CapKV reference: CapKV/cap_press.py (CapPress.score) and record_mixin.py at
// cf9c894, evaluated in float64 for Hq=4, Hkv=2, D=4, L=10, sink 2, tau 5,
// 8 future positions, RoPE theta 10000 and 6 kept entries. Inputs are
// multiples of 1/16, exact in FP16. Rows are positions; columns heads x D.
constexpr unsigned int kCapHeadsQ = 4, kCapHeadsKV = 2, kCapDim = 4,
                       kCapLength = 10, kCapSink = 2, kCapKeep = 6,
                       kCapFuture = 8;
constexpr float kCapTheta = 10000.0f;
constexpr unsigned int kCapRowMajor =
  static_cast<unsigned int>(ml::train::TensorDim::StorageOrder::ROW_MAJOR);

/** Pre-RoPE queries (after q_norm), [L, Hq * D] */
const std::vector<float> kCapQuery = {
  -1.125f,  -1.125f,  -0.25f,   -0.4375f, 0.875f,   0.6875f,  -0.3125f,
  -2.0f,    0.3125f,  -1.25f,   0.375f,   0.3125f,  0.125f,   1.25f,
  1.125f,   -0.25f,   -1.375f,  -1.6875f, 0.5625f,  0.8125f,  0.625f,
  -1.5625f, -0.3125f, 1.875f,   0.75f,    -0.5625f, -0.1875f, 0.1875f,
  1.375f,   1.5625f,  0.9375f,  -0.8125f, -0.625f,  0.0625f,  -0.5f,
  0.25f,    0.4375f,  0.125f,   0.625f,   0.4375f,  -0.125f,  0.8125f,
  -0.3125f, 0.0625f,  0.5f,     2.0f,     -1.5f,    -1.5625f, -0.6875f,
  0.875f,   1.0625f,  0.1875f,  -0.25f,   -0.375f,  0.5625f,  -0.375f,
  -0.4375f, 0.75f,    1.5f,     2.0f,     -1.5f,    -1.25f,   1.8125f,
  -0.5625f, -0.5625f, 0.9375f,  1.125f,   1.3125f,  -1.5f,    2.0f,
  -0.5f,    0.3125f,  -1.625f,  -0.5625f, -0.5f,    -0.5f,    -1.0625f,
  1.125f,   -0.125f,  0.8125f,  -0.0625f, 0.6875f,  -0.8125f, 0.0f,
  0.8125f,  -0.375f,  1.0625f,  0.375f,   -0.25f,   2.0f,     -1.875f,
  -0.0625f, -1.0625f, -0.9375f, 0.0625f,  0.6875f,  1.625f,   -1.375f,
  0.375f,   0.5f,     -2.0f,    -1.6875f, -0.25f,   0.25f,    0.25f,
  0.0625f,  0.3125f,  0.4375f,  0.4375f,  -0.875f,  0.8125f,  -0.9375f,
  -0.25f,   -2.0f,    -0.0625f, -0.0625f, 0.9375f,  0.375f,   -0.9375f,
  -0.0625f, -0.625f,  -0.4375f, 1.9375f,  -0.375f,  0.125f,   1.1875f,
  0.9375f,  1.375f,   -0.875f,  -0.4375f, -0.8125f, 0.5625f,  0.625f,
  0.4375f,  1.9375f,  -1.0625f, -0.0625f, 0.125f,   -0.625f,  -1.25f,
  -0.125f,  -1.0625f, -0.3125f, 0.25f,    -0.25f,   0.125f,   0.25f,
  1.1875f,  0.3125f,  0.375f,   -0.1875f, -0.125f,  0.1875f,  -0.75f,
  -1.875f,  0.125f,   -0.6875f, 0.3125f,  0.1875f,  0.375f};

/** Cached post-RoPE keys, [L, Hkv * D] */
const std::vector<float> kCapKey = {
  0.9375f,  -1.0f,    0.3125f,  -0.125f, 1.0f,     -0.5f,    0.75f,    0.625f,
  0.3125f,  0.1875f,  -0.125f,  0.375f,  -0.3125f, -1.0625f, 0.25f,    -1.375f,
  -0.0625f, -0.375f,  -1.5625f, -0.375f, 1.0625f,  0.125f,   1.5625f,  0.8125f,
  -1.0625f, -0.75f,   1.5f,     0.25f,   0.6875f,  -1.0f,    -1.1875f, -1.1875f,
  -0.5625f, 0.5625f,  0.4375f,  0.375f,  0.1875f,  0.875f,   -0.3125f, 1.0f,
  -0.0625f, -1.0625f, 0.375f,   0.625f,  0.6875f,  -1.1875f, 0.625f,   -0.5625f,
  1.1875f,  0.125f,   0.9375f,  0.8125f, -1.25f,   1.0f,     -0.5f,    -1.3125f,
  0.9375f,  1.3125f,  -0.6875f, 0.5625f, 0.3125f,  -0.0625f, -1.0625f, -0.125f,
  -0.3125f, 1.5625f,  0.1875f,  0.375f,  -0.125f,  0.625f,   -0.3125f, -1.3125f,
  -0.125f,  -0.5625f, -0.5f,    0.875f,  -0.875f,  0.8125f,  1.125f,   -1.75f};

/** Cached values, [L, Hkv * D] */
const std::vector<float> kCapValue = {
  -1.5f,    -1.75f,   0.0625f,  -1.0625f, 1.4375f, -0.125f, 0.5625f,
  -0.1875f, -0.0625f, 1.0625f,  0.9375f,  -0.5f,   -0.875f, 0.125f,
  1.875f,   0.375f,   -0.0625f, -0.8125f, -0.5f,   -0.375f, 0.5625f,
  -1.125f,  -0.125f,  -0.1875f, -0.0625f, 1.375f,  0.625f,  -0.5f,
  -0.375f,  -1.3125f, 1.6875f,  2.0f,     -0.25f,  0.6875f, 0.5625f,
  0.25f,    -2.0f,    0.5625f,  -0.5625f, -0.875f, 0.0f,    1.1875f,
  0.375f,   -0.1875f, -1.125f,  -0.3125f, 0.9375f, -1.125f, 0.5625f,
  0.8125f,  -0.1875f, -0.75f,   -0.0625f, -1.0f,   0.0625f, -0.5f,
  -0.8125f, 0.8125f,  -0.625f,  1.3125f,  1.4375f, -0.625f, 1.0f,
  -0.4375f, -0.625f,  -0.1875f, -0.5f,    0.8125f, 0.125f,  -1.0625f,
  -0.125f,  -1.0f,    -1.1875f, 0.5f,     -1.25f,  -1.375f, 1.5625f,
  -0.0625f, -0.4375f, -0.25f};

/** CapPress scores of positions [sink, L) per KV head */
const std::vector<std::vector<double>> kCapScores = {
  {0.00614682035, 0.0289692234, 0.0152041005, 0.235346844, 0.0919894017,
   0.383203907, 0.111414843, 0.418658817},
  {0.372154804, 0.00889223886, 0.746616123, 0.0158264325, 0.0217374553,
   0.00979238041, 0.0233579577, 0.0428348015}};

/** topk(6) of the padded scores, sorted */
const std::vector<std::vector<unsigned int>> kCapSelected = {
  {0, 1, 5, 7, 8, 9}, {0, 1, 2, 4, 8, 9}};

/**
 * @brief Rows of one KV head, [L, D]
 */
std::vector<float> capHead(const std::vector<float> &rows, unsigned int head) {
  std::vector<float> out;
  for (unsigned int r = 0; r < kCapLength; ++r)
    for (unsigned int c = 0; c < kCapDim; ++c)
      out.push_back(rows[(r * kCapHeadsKV + head) * kCapDim + c]);
  return out;
}

/**
 * @brief CapPress query anchors of every query head
 */
std::vector<float> capAnchors() {
  std::vector<double> sum(kCapHeadsQ * kCapDim, 0.0);
  for (unsigned int r = kCapSink; r < kCapLength; ++r)
    for (unsigned int i = 0; i < sum.size(); ++i)
      sum[i] += kCapQuery[r * sum.size() + i];
  return causallm::CapKVState::anchors(sum, kCapLength - kCapSink, kCapHeadsQ,
                                       kCapDim, kCapTheta, kCapLength,
                                       kCapFuture);
}

/**
 * @brief Test that CapKV scores and selection match the official CapPress
 */
TEST(CapKVTest, ScoresAndSelectionMatchOfficialCapPress) {
  const auto anchors = capAnchors();
  const unsigned int group = kCapHeadsQ / kCapHeadsKV;
  for (unsigned int head = 0; head < kCapHeadsKV; ++head) {
    const auto keys = capHead(kCapKey, head);
    const auto values = capHead(kCapValue, head);
    const auto scores = causallm::CapKVState::scoreKVHead(
      anchors.data() + head * group * kCapDim, group, keys.data(),
      values.data(), kCapLength, kCapDim, kCapSink, 5.0, kCapRowMajor);
    ASSERT_EQ(scores.size(), kCapScores[head].size());
    for (size_t i = 0; i < scores.size(); ++i)
      EXPECT_NEAR(scores[i], kCapScores[head][i],
                  1e-5 + 1e-4 * kCapScores[head][i])
        << "head " << head << " position " << i + kCapSink;
    EXPECT_EQ(causallm::CapKVState::select(scores, kCapSink, kCapKeep),
              kCapSelected[head]);
  }
}

/**
 * @brief Test that capacity proxies match the official CapacityLoggerMixin
 */
TEST(CapKVTest, CapacityMatchesOfficialLogger) {
  causallm::CapKVCapacity pre, post;
  for (unsigned int head = 0; head < kCapHeadsKV; ++head) {
    const auto keys = capHead(kCapKey, head);
    const auto values = capHead(kCapValue, head);
    const auto all = causallm::CapKVState::capacity(
      keys.data(), values.data(), kCapLength, kCapDim, kCapRowMajor);
    std::vector<float> kept_keys, kept_values;
    for (unsigned int position : kCapSelected[head])
      for (unsigned int c = 0; c < kCapDim; ++c) {
        kept_keys.push_back(keys[position * kCapDim + c]);
        kept_values.push_back(values[position * kCapDim + c]);
      }
    const auto kept = causallm::CapKVState::capacity(
      kept_keys.data(), kept_values.data(), kCapKeep, kCapDim, kCapRowMajor);
    pre.k += all.k / kCapHeadsKV;
    pre.u += all.u / kCapHeadsKV;
    pre.ku += all.ku / kCapHeadsKV;
    post.k += kept.k / kCapHeadsKV;
    post.u += kept.u / kCapHeadsKV;
    post.ku += kept.ku / kCapHeadsKV;
  }
  EXPECT_NEAR(pre.k, 8.225656588827503, 1e-4);
  EXPECT_NEAR(pre.u, 8.737974207375704, 1e-4);
  EXPECT_NEAR(pre.ku, 3.569517577311238, 1e-4);
  EXPECT_NEAR(post.k, 6.477104788471239, 1e-4);
  EXPECT_NEAR(post.u, 7.24645425628186, 1e-4);
  EXPECT_NEAR(post.ku, 3.209898469901547, 1e-4);
}

/**
 * @brief Fill a [1, 1, rows, width] cache tensor of the given storage type
 */
nntrainer::Tensor capCache(const std::vector<float> &values, unsigned int rows,
                           nntrainer::Tdatatype dtype) {
  const unsigned int width = kCapHeadsKV * kCapDim;
  nntrainer::Tensor cache(
    nntrainer::TensorDim(1, 1, rows, width, {nntrainer::Tformat::NCHW, dtype}),
    true);
  for (size_t i = 0; i < static_cast<size_t>(rows) * width; ++i) {
    const float value = i < values.size() ? values[i] : 0.0f;
    if (dtype == nntrainer::Tdatatype::FP32)
      cache.getData<float>()[i] = value;
    else
      cache.getData<uint16_t>()[i] = nntrainer::compute_fp32_to_fp16(value);
  }
  return cache;
}

/**
 * @brief Read one element of a cache tensor as FP32
 */
float capRead(const nntrainer::Tensor &cache, size_t index) {
  return cache.getDataType() == nntrainer::Tdatatype::FP32
           ? cache.getData<float>()[index]
           : nntrainer::compute_fp16_to_fp32(cache.getData<uint16_t>()[index]);
}

/**
 * @brief Start a CapKV session as CausalLM::run() does for one layer
 */
causallm::CapKVState capSession(const std::string &budget) {
  causallm::CapKVState state;
  for (const auto &[key, value] :
       std::vector<std::pair<std::string, std::string>>{
         {"capkv_budget", budget},
         {"capkv_sink", std::to_string(kCapSink)},
         {"capkv_tau", "5"},
         {"capkv_future", std::to_string(kCapFuture)},
         {"capkv_layer", "0"},
         {"capkv_prompt_length", std::to_string(kCapLength)}})
    EXPECT_TRUE(state.consumeProperty(key, value));
  return state;
}

/**
 * @brief Test that prefill compaction keeps the selected rows in place and
 * decode rows follow the kept prefix, for FP32 and FP16-bit caches
 */
TEST(CapKVTest, CompactsPromptCacheAndMapsDecodeRows) {
  for (auto dtype :
       {nntrainer::Tdatatype::FP32, nntrainer::Tdatatype::UINT16}) {
    const unsigned int rows = kCapLength + 2, width = kCapHeadsKV * kCapDim;
    auto key = capCache(kCapKey, rows, dtype);
    auto value = capCache(kCapValue, rows, dtype);
    nntrainer::Tensor query(
      nntrainer::TensorDim(1, 1, kCapLength, kCapHeadsQ * kCapDim), true);
    std::copy(kCapQuery.begin(), kCapQuery.end(), query.getData<float>());

    auto state = capSession(std::to_string(kCapKeep));
    ASSERT_NO_THROW(state.checkStep(0, kCapLength, rows, 40960));
    state.observeQueries(query, kCapHeadsQ, kCapDim);
    state.finishPrefill(key, value, kCapHeadsQ, kCapHeadsKV, kCapDim,
                        kCapTheta);
    state.advance(0, kCapLength);
    EXPECT_EQ(state.removed(), kCapLength - kCapKeep);

    for (unsigned int head = 0; head < kCapHeadsKV; ++head)
      for (unsigned int row = 0; row < kCapKeep; ++row)
        for (unsigned int c = 0; c < kCapDim; ++c) {
          const size_t original =
            kCapSelected[head][row] * width + head * kCapDim + c;
          const size_t moved = row * width + head * kCapDim + c;
          EXPECT_EQ(capRead(key, moved), kCapKey[original]);
          EXPECT_EQ(capRead(value, moved), kCapValue[original]);
        }
    // The next token is position L at physical row kept; steps stay ordered.
    EXPECT_NO_THROW(state.checkStep(kCapLength, 1, rows, 40960));
    EXPECT_THROW(state.checkStep(kCapLength + 1, 1, rows, 40960),
                 std::invalid_argument);
    EXPECT_THROW(state.checkStep(kCapLength, 1, kCapKeep, 40960),
                 std::out_of_range);
  }
}

/**
 * @brief Test the kept count, short prompts and rejected settings
 */
TEST(CapKVTest, KeepsShortPromptsAndRejectsInvalidSettings) {
  causallm::CapKVConfig config;
  config.ratio = 0.5;
  EXPECT_EQ(config.keep(8188), 4094u); // int(L * (1 - ratio))
  EXPECT_EQ(config.keep(5), 2u);
  config.ratio = 0.0;
  config.budget = 6;
  EXPECT_EQ(config.keep(4), 4u);
  config.ratio = 0.5;
  EXPECT_THROW(config.validate(), std::invalid_argument);
  config.budget = 0;
  config.ratio = 1.0;
  EXPECT_THROW(config.validate(), std::invalid_argument);
  config.ratio = 0.5;
  config.tau = -1.0;
  EXPECT_THROW(config.validate(), std::invalid_argument);
  config.tau = 5.0;
  config.future = 0;
  EXPECT_THROW(config.validate(), std::invalid_argument);
  EXPECT_THROW(causallm::CapKVConfig::parseNumber("0.5x"),
               std::invalid_argument);
  EXPECT_THROW(causallm::CapKVConfig::parseUnsigned("-1"),
               std::invalid_argument);
  EXPECT_THROW(causallm::CapKVConfig::parseBool("yes"), std::invalid_argument);

  causallm::CapKVState unrelated;
  EXPECT_FALSE(unrelated.consumeProperty("cache_index", "3"));
  EXPECT_FALSE(unrelated.enabled());

  const unsigned int rows = kCapLength + 2;
  nntrainer::Tensor query(
    nntrainer::TensorDim(1, 1, kCapLength, kCapHeadsQ * kCapDim), true);
  std::copy(kCapQuery.begin(), kCapQuery.end(), query.getData<float>());

  // A budget covering the prompt keeps the cache bit for bit.
  auto key = capCache(kCapKey, rows, nntrainer::Tdatatype::UINT16);
  auto value = capCache(kCapValue, rows, nntrainer::Tdatatype::UINT16);
  auto whole = capSession(std::to_string(kCapLength));
  whole.observeQueries(query, kCapHeadsQ, kCapDim);
  whole.finishPrefill(key, value, kCapHeadsQ, kCapHeadsKV, kCapDim, kCapTheta);
  EXPECT_EQ(whole.removed(), 0u);
  for (size_t i = 0; i < kCapKey.size(); ++i)
    EXPECT_EQ(capRead(key, i), kCapKey[i]);

  // Keeping only the sinks is refused.
  auto sinks = capSession(std::to_string(kCapSink));
  sinks.observeQueries(query, kCapHeadsQ, kCapDim);
  EXPECT_THROW(sinks.finishPrefill(key, value, kCapHeadsQ, kCapHeadsKV, kCapDim,
                                   kCapTheta),
               std::invalid_argument);
}

} // namespace
