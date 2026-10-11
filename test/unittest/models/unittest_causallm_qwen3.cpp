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

#include <causallm_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <mha_core.h>
#include <qwen3_causallm.h>
#include <qwen3_embedding.h>
#include <snapkv.h>

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

/** @brief Inspect cache ownership without changing the production interface. */
class SnapKVQwen3 : public TinyQwen3CausalLM {
public:
  SnapKVQwen3(causallm::json &cfg, causallm::json &generation,
              causallm::json &nntr) :
    causallm::Transformer(cfg, generation, nntr, causallm::ModelType::CAUSALLM),
    TinyQwen3CausalLM(cfg, generation, nntr) {}

  using causallm::CausalLM::kv_cache;
  using causallm::CausalLM::shrinkSnapKVCache;
  std::vector<std::weak_ptr<nntrainer::MemoryData>> original_storage;

  void allocateAndBindKVCache() override {
    causallm::CausalLM::allocateAndBindKVCache();
    if (original_storage.empty()) {
      for (unsigned int i = 0; i < kv_cache.getNumLayers(); ++i) {
        original_storage.emplace_back(kv_cache.getKeyCache(i).getMemoryData());
        original_storage.emplace_back(
          kv_cache.getValueCache(i).getMemoryData());
      }
    }
  }
};

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
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
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
        if (layer.getType() == "rms_norm" ||
            layer.getType() == "reshaped_rms_norm") {
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

/** @brief Golden data from the cloned SnapKVCluster.update_kv(), not our
 * selector. Reference e216ddc84c5bd210378cbdbbba12ba02102aa640, torch 2.14.1
 * CPU, seed 17; Hq=4, Hkv=2, L=10, D=4, W=2, C=5, kernel=3. Q=randn*0.4;
 * K[p,h]=[p,p*p%7,2*h+1,(p+h)%3]. Probabilities below are the final two causal
 * softmax rows in NNTrainer's token/head layout. Expected positions were read
 * from the official helper's gathered V.
 */
TEST(Qwen3SnapKVTest, MatchesOfficialHeadwiseSelection) {
  using nntrainer::Tensor;
  using nntrainer::TensorDim;
  const float observation[] = {
    0.11693424f,   0.0368258059f, 0.163079813f,  0.163869143f,  0.102730542f,
    0.0509713925f, 0.155099317f,  0.185908049f,  0.0843179077f, 0.0748813078f,
    0.107150026f,  0.196744025f,  0.119671091f,  0.0720876977f, 0.115636505f,
    0.12293461f,   0.108771719f,  0.0968498811f, 0.114710264f,  0.114322685f,
    0.09236449f,   0.138105094f,  0.0826575086f, 0.0991728827f, 0.135626122f,
    0.129050925f,  0.0930427685f, 0.0507952049f, 0.127537787f,  0.168291703f,
    0.0962692648f, 0.0387202501f, 0.112046115f,  0.232936189f,  0.0723544657f,
    0.0275331493f, 0.15030849f,   0.189281628f,  0.301897615f,  0.152689293f,
    0.152504772f,  0.155468896f,  0.224180952f,  0.120787695f,  0.256905556f,
    0.101153478f,  0.146943405f,  0.0626510978f, 0.112056784f,  0.105231784f,
    0.106030345f,  0.0948117673f, 0.088235423f,  0.0971137881f, 0.07798592f,
    0.0931071043f, 0.115355425f,  0.0709932521f, 0.0506307818f, 0.059950836f,
    0.0390487947f, 0.082981661f,  0.0361861028f, 0.112625122f,  0.0238625817f,
    0.0860428885f, 0.0263617896f, 0.137297422f,  0.0242112596f, 0.0706724301f,
    0.0169519763f, 0.109744117f,  0.0375108905f, 0.041060254f,  0.0128311403f,
    0.0563355349f};
  const unsigned int expected[2][4][5] = {
    {{1, 2, 3, 8, 9}, {6, 5, 1, 8, 9}, {1, 2, 0, 8, 9}, {1, 2, 3, 8, 9}},
    {{1, 3, 2, 8, 9}, {7, 6, 1, 8, 9}, {1, 0, 2, 8, 9}, {1, 0, 2, 8, 9}}};
  unsigned int mode = 0;
  for (const auto *pooling : {"avgpool", "maxpool"}) {
    SCOPED_TRACE(pooling);
    Tensor probs(TensorDim(1, 1, 55, 4), true);
    probs.setZero();
    std::copy(std::begin(observation), std::end(observation),
              probs.getData<float>() + (8 * 9 / 2) * 4);
    Tensor key(TensorDim(1, 1, 10, 8), true);
    Tensor value(key.getDim(), true);
    for (unsigned int p = 0; p < 10; ++p) {
      for (unsigned int h = 0; h < 2; ++h) {
        const float k[] = {float(p), float(p * p % 7), float(h * 2 + 1),
                           float((p + h) % 3)};
        for (unsigned int d = 0; d < 4; ++d) {
          key.getData<float>()[p * 8 + h * 4 + d] = k[d];
          value.getData<float>()[p * 8 + h * 4 + d] = p * 100 + h * 10 + d;
        }
      }
    }
    causallm::SnapKVState state;
    state.consumeProperty("snapkv_budget", "5");
    state.consumeProperty("snapkv_window", "2");
    state.consumeProperty("snapkv_kernel", "3");
    state.consumeProperty("snapkv_pooling", pooling);
    state.consumeProperty("snapkv_prompt_length", "10");
    state.finishStep(probs, key, value, 0, 10, 4, 2, 4);
    // Packed Hq cache is staged in the host slab until post-prefill rebinding.
    ASSERT_GE(key.size(), 5u * 4 * 4);
    EXPECT_EQ(state.physicalPosition(10), 5u);
    for (unsigned int h = 0; h < 4; ++h) {
      for (unsigned int row = 0; row < 5; ++row) {
        const auto p = expected[mode][h][row];
        const float k[] = {float(p), float(p * p % 7), float((h / 2) * 2 + 1),
                           float((p + h / 2) % 3)};
        for (unsigned int d = 0; d < 4; ++d) {
          EXPECT_FLOAT_EQ(key.getData<float>()[row * 16 + h * 4 + d], k[d]);
          EXPECT_FLOAT_EQ(value.getData<float>()[row * 16 + h * 4 + d],
                          p * 100 + (h / 2) * 10 + d);
        }
      }
    }
    ++mode;
  }
}

/** @brief Compare actual prefill/decode attention to the official compressed
 * cache. Same seed/model-free fixture as MatchesOfficialHeadwiseSelection.
 * Reference decode is softmax(Q @ cat(K_compressed, repeat_kv(K_new)) /
 * sqrt(D)) @ V. RoPE is disabled here: K/Q represent already rotated vectors;
 * absolute positions and physical bounds are separately exercised by the model
 * test.
 */
TEST(Qwen3SnapKVTest, DecodeUsesIndependentCompressedHeads) {
  using nntrainer::Tensor;
  using nntrainer::TensorDim;
  const float q_observation[] = {
    0.0607655942f,  -0.0680129752f, -0.0240471847f,  -0.251757085f,
    0.408073753f,   0.0595740862f,  0.14272134f,     0.182483673f,
    -0.173016176f,  -0.0842638165f, -0.00981722772f, 0.156931907f,
    -0.456692606f,  0.39762333f,    0.183313608f,    0.311437339f,
    -0.533793688f,  0.507006526f,   0.36260125f,     0.055799406f,
    -0.236036062f,  -0.233016536f,  0.550661206f,    0.0754720643f,
    -0.710325122f,  0.0191251338f,  -0.164085746f,   0.09593115f,
    -0.0293708742f, -0.432455301f,  0.334746331f,    -0.00691506127f};
  const float q_decode[] = {
    0.212234959f, 0.107677519f,  0.414767563f, -0.881746113f,
    0.257902771f, -0.383747399f, 0.178663597f, 0.114914536f,
    0.319978356f, 0.0727683827f, -0.60705024f, -0.251730323f,
    0.10960336f,  0.531924665f,  0.374941677f, 0.0257930346f};
  const float prefill_reference[] = {
    286.482422f, 287.482391f, 288.482361f, 289.482361f,
    345.20047f,  346.20047f,  347.200439f, 348.200439f,
    215.400146f, 216.400162f, 217.400146f, 218.400162f,
    432.451294f, 433.451294f, 434.451294f, 435.451324f};
  const float decode_reference[] = {
    698.380859f, 699.380859f, 700.380859f, 701.380859f,
    735.131226f, 736.131165f, 737.131165f, 738.131165f,
    741.724426f, 742.724487f, 743.724426f, 744.724426f,
    636.917786f, 637.917725f, 638.917786f, 639.917725f};
  Tensor query(TensorDim(1, 1, 10, 16), true);
  query.setZero();
  std::copy(std::begin(q_observation), std::end(q_observation),
            query.getData<float>() + 8 * 16);
  Tensor key(TensorDim(1, 1, 10, 8), true);
  Tensor value(key.getDim(), true);
  for (unsigned int p = 0; p < 10; ++p) {
    for (unsigned int h = 0; h < 2; ++h) {
      const float k[] = {float(p), float(p * p % 7), float(h * 2 + 1),
                         float((p + h) % 3)};
      for (unsigned int d = 0; d < 4; ++d) {
        key.getData<float>()[p * 8 + h * 4 + d] = k[d];
        value.getData<float>()[p * 8 + h * 4 + d] = p * 100 + h * 10 + d;
      }
    }
  }
  Tensor cache_key(TensorDim(1, 1, 16, 8), true);
  Tensor cache_value(cache_key.getDim(), true);
  Tensor output(query.getDim(), true);
  causallm::MHACoreLayer layer;
  layer.setProperty({"num_heads=4", "num_heads_kv=2", "max_timestep=16",
                     "use_rope=false", "snapkv_budget=5", "snapkv_window=2",
                     "snapkv_kernel=3", "snapkv_pooling=avgpool",
                     "snapkv_prompt_length=10"});
  nntrainer::InitLayerContext init({query.getDim(), key.getDim(),
                                    value.getDim(), cache_key.getDim(),
                                    cache_value.getDim()},
                                   {true}, false, "snapkv_attention");
  layer.finalize(init);
  auto cache_dim = cache_key.getDim();
  auto step_dim = cache_dim;
  step_dim.height(10);
  layer.one_batch_incremental_forwarding(0, 0, 0, 10, query, key, value, output,
                                         cache_key, cache_value, cache_dim,
                                         step_dim, cache_dim, step_dim);
  for (unsigned int d = 0; d < 16; ++d)
    EXPECT_NEAR(output.getData<float>()[9 * 16 + d], prefill_reference[d],
                0.001f);

  // Host adopts the packed C x Hq prefix and reserves one decode row.
  Tensor small_key(TensorDim(1, 1, 6, 16), true);
  Tensor small_value(small_key.getDim(), true);
  std::copy_n(cache_key.getData<float>(), 5 * 16, small_key.getData<float>());
  std::copy_n(cache_value.getData<float>(), 5 * 16,
              small_value.getData<float>());
  Tensor next_query(TensorDim(1, 1, 1, 16), true);
  Tensor next_key(TensorDim(1, 1, 1, 8), true);
  Tensor next_value(next_key.getDim(), true);
  Tensor next_output(next_query.getDim(), true);
  std::copy(std::begin(q_decode), std::end(q_decode),
            next_query.getData<float>());
  for (unsigned int h = 0; h < 2; ++h) {
    const float k[] = {10.0f, 2.0f, float(h * 2 + 1), float((10 + h) % 3)};
    for (unsigned int d = 0; d < 4; ++d) {
      next_key.getData<float>()[h * 4 + d] = k[d];
      next_value.getData<float>()[h * 4 + d] = 1000 + h * 10 + d;
    }
  }
  layer.setCacheIndex(10);
  cache_dim = small_key.getDim();
  step_dim = cache_dim;
  step_dim.height(1);
  ASSERT_NO_THROW(layer.one_batch_incremental_forwarding(
    0, 10, 10, 11, next_query, next_key, next_value, next_output, small_key,
    small_value, cache_dim, step_dim, cache_dim, step_dim));
  // Only 6 rows are allocated: reading the original logical length 11 would
  // exceed this buffer. Hq=4/gqa_size=1 must consume different head histories.
  for (unsigned int d = 0; d < 16; ++d)
    EXPECT_NEAR(next_output.getData<float>()[d], decode_reference[d], 0.001f);
}

/** @brief New decode K/V repeat to the independently selected query heads. */
TEST(Qwen3SnapKVTest, RepeatsDecodeHeadsWithoutOverwritingSources) {
  nntrainer::TensorDim dim(1, 1, 1, 24);
  dim.setDataType(nntrainer::Tdatatype::UINT16);
  nntrainer::Tensor token(dim, true);
  token.setZero();
  for (unsigned int i = 0; i < 8; ++i)
    token.getData<uint16_t>()[i] = 100 + i;
  causallm::SnapKVState::repeatKV(token, 2, 4);
  for (unsigned int h = 0; h < 6; ++h)
    for (unsigned int d = 0; d < 4; ++d)
      EXPECT_EQ(token.getData<uint16_t>()[h * 4 + d], 100 + (h / 3) * 4 + d);
}

/** @brief Shrinking retains K/V bits and releases every old owning buffer. */
TEST(Qwen3SnapKVTest, ReplacesStorageAndRebindsCacheViews) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Qwen3SnapKVTest", "ReplacesStorageAndRebindsCacheViews", "FP32");
  auto cfg = makeTinyQwen3Config();
  cfg["model_type"] = "qwen3";
  cfg["num_hidden_layers"] = 2;
  auto generation = causallm_test::makeTinyGenerationConfig();
  auto nntr = causallm_test::makeTinyNntrainerConfig(
    files.tokenizer_path, causallm_test::makeTinyFp32DataType());
  nntr["num_to_generate"] = 3;
  nntr["snapkv"] = {{"budget", 2}, {"window", 1}, {"kernel", 1}};
  SnapKVQwen3 model(cfg, generation, nntr);
  model.initializeModel();
  model.allocateAndBindKVCache();
  for (unsigned int i = 0; i < 2; ++i) {
    for (auto *cache :
         {&model.kv_cache.getKeyCache(i), &model.kv_cache.getValueCache(i)}) {
      for (unsigned int j = 0; j < cache->size(); ++j)
        cache->getData<uint16_t>()[j] = static_cast<uint16_t>(i * 1000 + j);
    }
  }
  model.shrinkSnapKVCache();
  EXPECT_EQ(model.kv_cache.getMaxSeqLen(), 5u);
  EXPECT_EQ(model.kv_cache.getPosition(), 2u);
  for (const auto &old : model.original_storage)
    EXPECT_TRUE(old.expired());
  for (unsigned int i = 0; i < 2; ++i) {
    for (auto *cache :
         {&model.kv_cache.getKeyCache(i), &model.kv_cache.getValueCache(i)}) {
      EXPECT_EQ(cache->bytes(), 5u * 8 * 8 * sizeof(uint16_t));
      for (unsigned int j = 0; j < 2 * cache->width(); ++j)
        EXPECT_EQ(cache->getData<uint16_t>()[j], i * 1000 + j);
    }
  }
  unsigned int cache_views = 0;
  model.forEachLayer([&cache_views](ml::train::Layer &layer,
                                    nntrainer::RunLayerContext &context,
                                    void *) {
    if (layer.getName().find("cache_k_l") == 0 ||
        layer.getName().find("cache_v_l") == 0) {
      EXPECT_EQ(context.getInput(0).height(), 5u);
      EXPECT_EQ(context.getOutput(0).height(), 5u);
      cache_views += 2;
    } else if (layer.getType() == "mha_core") {
      EXPECT_EQ(context.getInput(3).height(), 5u);
      EXPECT_EQ(context.getInput(4).height(), 5u);
      cache_views += 2;
    }
  });
  EXPECT_EQ(cache_views, 12u);
}

/** @brief Decode beyond physical capacity using unchanged absolute positions.
 */
TEST(Qwen3SnapKVTest, RunsWithShrunkStorageAndPreservesDisabledPath) {
  const auto files = causallm_test::makeTinyCausalLMFiles(
    "Qwen3SnapKVTest", "RunsWithShrunkStorageAndPreservesDisabledPath", "FP32");
  for (const int budget : {-1, 0, 8, 2}) {
    SCOPED_TRACE(budget);
    auto cfg = makeTinyQwen3Config();
    cfg["model_type"] = "qwen3";
    cfg["num_hidden_layers"] = 2;
    auto generation = causallm_test::makeTinyGenerationConfig();
    auto nntr = causallm_test::makeTinyNntrainerConfig(
      files.tokenizer_path, causallm_test::makeTinyFp32DataType());
    nntr["init_seq_len"] = 4;
    nntr["num_to_generate"] = 3;
    if (budget >= 0)
      nntr["snapkv"] = {{"budget", budget}, {"window", 1}, {"kernel", 1}};
    SnapKVQwen3 model(cfg, generation, nntr);
    model.initializeModel();
    setupQwen3DeterministicWeights(model);
    ASSERT_NO_THROW(model.runPrompt("hello tok4 hello tok4"));
    EXPECT_TRUE(model.hasRun());
    EXPECT_EQ(model.kv_cache.getMaxSeqLen(), budget == 2 ? 5u : 8u);
    EXPECT_EQ(model.kv_cache.getKeyCache(0).width(),
              budget == 2 ? 8u * 8 : 4u * 8);
    for (const auto &old : model.original_storage)
      EXPECT_EQ(old.expired(), budget == 2);
    for (size_t i = 5; i < 8; ++i)
      EXPECT_EQ(model.tokenAt(i), 4u);
  }
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

} // namespace
