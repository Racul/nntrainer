// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_obcache.cpp
 * @brief OBCache scoring, Tensor compaction and tiny-model integration
 */
#include <causallm_test_utils.h>
#include <gtest/gtest.h>
#include <layer.h>
#include <layer_context.h>
#include <obcache.h>
#include <qwen3_causallm.h>

#include <array>

namespace {
using causallm::OBCacheConfig;
using causallm::OBCacheState;
using nntrainer::Tdatatype;
using nntrainer::Tensor;
using nntrainer::Tformat;

void configure(OBCacheState &state, const std::string &score = "key",
               unsigned int budget = 4, unsigned int prompt = 6,
               unsigned int kernel = 1) {
  for (const auto &p :
       {"obcache_budget=" + std::to_string(budget),
        std::string("obcache_window=2"),
        "obcache_kernel=" + std::to_string(kernel), "obcache_score=" + score,
        "obcache_prompt_length=" + std::to_string(prompt)}) {
    const auto split = p.find('=');
    ASSERT_TRUE(state.consumeProperty(p.substr(0, split), p.substr(split + 1)));
  }
}

void write(Tensor &tensor, size_t i, float value) {
  if (tensor.getDataType() == Tdatatype::FP32)
    tensor.getData<float>()[i] = value;
  else
    tensor.getData<uint16_t>()[i] = nntrainer::compute_fp32_to_fp16(value);
}

float read(const Tensor &tensor, size_t i) {
  return tensor.getDataType() == Tdatatype::FP32
           ? tensor.getData<float>()[i]
           : nntrainer::compute_fp16_to_fp32(tensor.getData<uint16_t>()[i]);
}

// Independent kernel outputs isolate pre_redc and equations (5)-(7).
// Mean A = [.3, .1, .1, .1, .4], mean Z = [0, 1, 3, 0, 0].
// With V = [1, 30, 2, 100, 0], O = 13.5. Each observation votes:
// key: [0, 2.7225, 11.9025, 0]; value: [.09, 9, .04, 100];
// joint: [.09, 21.6225, 10.5625, 100]. Protected tokens never compete.
void syntheticRow(Tensor &q, Tensor &, Tensor &z, Tensor &a, unsigned int pos) {
  EXPECT_EQ(q.height(), 1u);
  EXPECT_EQ(z.height(), pos + 1);
  EXPECT_EQ(a.width(), 2u);
  z.setValue(0.0f);
  a.setValue(0.0f);
  const std::array<float, 4> p0{.4f, .1f, .1f, .1f};
  const std::array<float, 4> p1{.2f, .1f, .1f, .1f};
  const std::array<float, 4> z0{2, 1, 3, 0};
  const std::array<float, 4> z1{-2, 1, 3, 0};
  for (size_t j = 0; j < 4; ++j) {
    a.getData<float>()[j * 2] = p0[j];
    a.getData<float>()[j * 2 + 1] = p1[j];
    z.getData<float>()[j * 2] = z0[j];
    z.getData<float>()[j * 2 + 1] = z1[j];
  }
  for (size_t j = 4; j <= pos; ++j) {
    a.getData<float>()[j * 2] = .3f / (pos - 3);
    a.getData<float>()[j * 2 + 1] = .5f / (pos - 3);
  }
}

TEST(OBCache, OutputAwareScoresAndPreReduction_p) {
  for (const auto *score : {"key", "value", "joint"}) {
    for (const auto dtype : {Tdatatype::FP32, Tdatatype::UINT16}) {
      OBCacheState state;
      configure(state, score);
      Tensor q(1, 1, 6, 2);
      Tensor k(1, 1, 16, 1, Tformat::NCHW, dtype);
      Tensor v(1, 1, 16, 1, Tformat::NCHW, dtype);
      const std::array<float, 6> values{1, 30, 2, 100, 0, 0};
      for (size_t j = 0; j < 6; ++j) {
        write(k, j, 10 + j);
        write(v, j, values[j]);
      }
      state.finishPrefill(q, k, v, 2, 1, 1, syntheticRow);
      const auto selected = std::string(score) == "key"
                              ? std::array<unsigned int, 4>{1, 2, 4, 5}
                              : std::array<unsigned int, 4>{1, 3, 4, 5};
      EXPECT_EQ(state.physicalPosition(6), 4u);
      for (size_t j = 0; j < selected.size(); ++j) {
        EXPECT_FLOAT_EQ(read(k, j), 10 + selected[j]);
        EXPECT_FLOAT_EQ(read(v, j), values[selected[j]]);
      }
    }
  }
}

TEST(OBCache, ProtectedValuesContributeToOutput_p) {
  OBCacheState state;
  configure(state);
  Tensor q(1, 1, 6, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
  const std::array<float, 6> values{1, 30, 2, 100, -30, -30};
  for (size_t j = 0; j < 6; ++j) {
    write(k, j, j);
    write(v, j, values[j]);
  }
  // Protected values change O from 13.5 to 1.5. Token 1 now wins
  // (8.1225 vs .0225); omitting protected values would select token 2.
  configure(state, "key", 3);
  state.finishPrefill(q, k, v, 2, 1, 1, syntheticRow);
  EXPECT_FLOAT_EQ(read(k, 0), 1);
  EXPECT_FLOAT_EQ(read(k, 1), 4);
  EXPECT_FLOAT_EQ(read(k, 2), 5);
}

TEST(OBCache, PerHeadCompactionAndConsecutiveDecode_p) {
  for (const auto dtype : {Tdatatype::FP32, Tdatatype::UINT16}) {
    OBCacheState state;
    for (unsigned int session = 0; session < 2; ++session) {
      configure(state, "value");
      Tensor q(1, 1, 6, 4);
      Tensor k(1, 1, 16, 2, Tformat::NCHW, dtype);
      Tensor v(1, 1, 16, 2, Tformat::NCHW, dtype);
      for (size_t j = 0; j < 6; ++j)
        for (size_t h = 0; h < 2; ++h) {
          write(k, j * 2 + h, 10 * j + h);
          write(v, j * 2 + h, h ? 6 - j : j + 1);
        }
      state.finishPrefill(
        q, k, v, 4, 2, 1,
        [](Tensor &, Tensor &, Tensor &z, Tensor &a, unsigned int pos) {
          z.setValue(0.0f);
          a.setValue(1.0f / (pos + 1));
        });
      const std::array<std::array<size_t, 4>, 2> selected{
        {{2, 3, 4, 5}, {0, 1, 4, 5}}};
      for (size_t h = 0; h < 2; ++h)
        for (size_t j = 0; j < 4; ++j) {
          EXPECT_FLOAT_EQ(read(k, j * 2 + h), 10 * selected[h][j] + h);
          EXPECT_FLOAT_EQ(read(v, j * 2 + h),
                          h ? 6 - selected[h][j] : selected[h][j] + 1);
        }
      Tensor decode(1, 1, 1, 4);
      for (unsigned int step = 0; step < 4; ++step) {
        state.decodeStep(
          decode, k, v, 6 + step, 16, 4, 2, 1,
          [&](Tensor &key, Tensor &value) {
            EXPECT_EQ(key.height(), 1u);
            for (size_t h = 0; h < 2; ++h) {
              write(key, h, 60 + step * 10 + h);
              write(value, h, 20 + step * 2 + h);
            }
          },
          [&](Tensor &key, Tensor &value, Tensor &p, unsigned int from) {
            EXPECT_EQ(from, 4 + step);
            EXPECT_EQ(key.height(), from + 1);
            EXPECT_EQ(value.height(), key.height());
            EXPECT_EQ(p.height(), key.height());
            for (size_t h = 0; h < 2; ++h) {
              EXPECT_FLOAT_EQ(read(key, from * 2 + h), 60 + step * 10 + h);
              EXPECT_FLOAT_EQ(read(value, from * 2 + h), 20 + step * 2 + h);
            }
          });
      }
      EXPECT_EQ(state.physicalPosition(10), 8u);
      EXPECT_THROW(state.checkStep(0, 6, 16, 16), std::invalid_argument);
      EXPECT_THROW(state.checkStep(9, 1, 16, 16), std::invalid_argument);
      EXPECT_THROW(state.checkStep(10, 2, 16, 16), std::invalid_argument);
    }
  }
}

TEST(OBCache, PoolingAndStableTies_p) {
  for (const auto *score : {"key", "value"}) {
    OBCacheState state;
    configure(state, score, 3, 6, 3);
    Tensor q(1, 1, 6, 1), k(1, 1, 16, 1), v(1, 1, 16, 1);
    for (size_t j = 0; j < 6; ++j) {
      write(k, j, j);
      write(v, j, j + 1);
    }
    state.finishPrefill(
      q, k, v, 1, 1, 1,
      [](Tensor &, Tensor &, Tensor &z, Tensor &a, unsigned int pos) {
        z.setValue(0.0f);
        a.setValue(1.0f / (pos + 1));
      });
    // Value pooling [5,14,29,25]/3 prefers position 2 over boundary 3.
    // Zero raw logits make all key scores zero; ties prefer position 0.
    EXPECT_FLOAT_EQ(read(k, 0), std::string(score) == "key" ? 0 : 2);
    EXPECT_FLOAT_EQ(read(k, 1), 4);
    EXPECT_FLOAT_EQ(read(k, 2), 5);
  }
}

TEST(OBCache, BudgetBoundarySkipsScoring_p) {
  for (const auto budget : {6u, 7u}) {
    OBCacheState state;
    configure(state, "key", budget);
    Tensor q(1, 1, 6, 1), k(1, 1, 16, 1), v(1, 1, 16, 1);
    state.finishPrefill(q, k, v, 1, 1, 1,
                        [](Tensor &, Tensor &, Tensor &, Tensor &,
                           unsigned int) { ADD_FAILURE(); });
    EXPECT_EQ(state.physicalPosition(6), 6u);
    EXPECT_NO_THROW(state.checkStep(6, 1, 16, 16));
  }
}

TEST(OBCache, InvalidConfigurationAndSteps_n) {
  OBCacheState off;
  EXPECT_FALSE(off.enabled());
  EXPECT_FALSE(off.consumeProperty("unrelated", "text"));
  for (const auto *s : {"", "-1", "1.5", "true", "4294967296"})
    EXPECT_THROW(OBCacheConfig::parseUnsigned(s), std::exception);
  for (const auto &cfg : {OBCacheConfig{4, 4, 3, "key"},
                          {4, 0, 3, "key"},
                          {4, 2, 2, "key"},
                          {4, 2, 3, "attention"}})
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
  OBCacheState state;
  configure(state);
  EXPECT_THROW(state.checkStep(0, 3, 16, 16), std::invalid_argument);
  EXPECT_THROW(state.checkStep(1, 6, 16, 16), std::invalid_argument);
  EXPECT_THROW(state.checkStep(0, 6, 4, 16), std::out_of_range);
  EXPECT_THROW(state.checkStep(0, 6, 16, 4), std::out_of_range);
  EXPECT_THROW(state.consumeProperty("obcache_unknown", "1"),
               std::invalid_argument);
}

TEST(OBCache, InvalidTensorAndNonfiniteScore_n) {
  OBCacheState state;
  configure(state);
  Tensor q(1, 1, 6, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
  Tensor bad_value(1, 1, 16, 2);
  EXPECT_THROW(state.finishPrefill(q, k, bad_value, 2, 1, 1, syntheticRow),
               std::invalid_argument);
  EXPECT_THROW(state.finishPrefill(q, k, v, 3, 2, 1, syntheticRow),
               std::invalid_argument);
  Tensor short_query(1, 1, 3, 2);
  EXPECT_THROW(state.finishPrefill(short_query, k, v, 2, 1, 1, syntheticRow),
               std::invalid_argument);
  k.setValue(17.0f);
  v.setValue(23.0f);
  EXPECT_THROW(state.finishPrefill(
                 q, k, v, 2, 1, 1,
                 [](Tensor &, Tensor &, Tensor &z, Tensor &a, unsigned int) {
                   z.setValue(std::numeric_limits<float>::infinity());
                   a.setValue(0.2f);
                 }),
               std::invalid_argument);
  EXPECT_FLOAT_EQ(read(k, 0), 17);
  EXPECT_FLOAT_EQ(read(v, 0), 23);
}

TEST(OBCache, InvalidProbabilities_n) {
  for (const float bad : {-0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    OBCacheState state;
    configure(state);
    Tensor q(1, 1, 6, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
    k.setValue(17.0f);
    v.setValue(0.0f);
    EXPECT_THROW(state.finishPrefill(q, k, v, 2, 1, 1,
                                     [bad](Tensor &, Tensor &, Tensor &z,
                                           Tensor &a, unsigned int) {
                                       z.setValue(0.0f);
                                       a.setValue(bad);
                                     }),
                 std::invalid_argument);
    EXPECT_FLOAT_EQ(read(k, 0), 17);
    EXPECT_NO_THROW(state.checkStep(0, 6, 16, 16));
  }
}

TEST(OBCache, NonfiniteValues_n) {
  for (const float bad : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
    OBCacheState state;
    configure(state);
    Tensor q(1, 1, 6, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
    k.setValue(17.0f);
    v.setValue(bad);
    EXPECT_THROW(state.finishPrefill(q, k, v, 2, 1, 1,
                                     [](Tensor &, Tensor &, Tensor &z,
                                        Tensor &a, unsigned int pos) {
                                       z.setValue(1.0f);
                                       a.setValue(1.0f / (pos + 1));
                                     }),
                 std::invalid_argument);
    EXPECT_FLOAT_EQ(read(k, 0), 17);
  }
}

TEST(OBCache, InvalidDecodeDoesNotWriteCache_n) {
  OBCacheState state;
  configure(state, "key", 6);
  Tensor prefill(1, 1, 6, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
  k.setValue(17.0f);
  v.setValue(23.0f);
  state.finishPrefill(prefill, k, v, 2, 1, 1,
                      [](Tensor &, Tensor &, Tensor &, Tensor &, unsigned int) {
                        ADD_FAILURE();
                      });
  auto store = [](Tensor &, Tensor &) { ADD_FAILURE(); };
  auto attend = [](Tensor &, Tensor &, Tensor &, unsigned int) {
    ADD_FAILURE();
  };
  Tensor decode(1, 1, 1, 2), multirow(1, 1, 2, 2), narrow(1, 1, 1, 1);
  EXPECT_THROW(state.decodeStep(decode, k, v, 7, 16, 2, 1, 1, store, attend),
               std::invalid_argument);
  EXPECT_THROW(state.decodeStep(decode, k, v, 6, 6, 2, 1, 1, store, attend),
               std::out_of_range);
  EXPECT_THROW(state.decodeStep(multirow, k, v, 6, 16, 2, 1, 1, store, attend),
               std::invalid_argument);
  EXPECT_THROW(state.decodeStep(narrow, k, v, 6, 16, 2, 1, 1, store, attend),
               std::invalid_argument);
  EXPECT_FLOAT_EQ(read(k, 0), 17);
  EXPECT_FLOAT_EQ(read(v, 0), 23);
  EXPECT_NO_THROW(state.checkStep(6, 1, 16, 16));
}

TEST(OBCache, UnsupportedCacheTypeLayoutAndBatch_n) {
  OBCacheState state;
  configure(state);
  Tensor q(1, 1, 6, 2), k(1, 1, 16, 1), v(1, 1, 16, 1);
  Tensor half(1, 1, 16, 1, Tformat::NCHW, Tdatatype::UINT16);
  Tensor nhwc(1, 1, 16, 1, Tformat::NHWC, Tdatatype::FP32);
  Tensor batch(2, 1, 16, 1);
  EXPECT_THROW(state.finishPrefill(q, k, half, 2, 1, 1, syntheticRow),
               std::invalid_argument);
  EXPECT_THROW(state.finishPrefill(q, nhwc, nhwc, 2, 1, 1, syntheticRow),
               std::invalid_argument);
  EXPECT_THROW(state.finishPrefill(q, batch, batch, 2, 1, 1, syntheticRow),
               std::invalid_argument);
}

using TinyQwen3 = causallm_test::CausalLMTestAdapter<causallm::Qwen3CausalLM>;

TEST(OBCache, TinyModelActivationAndDisabledEquivalence_p) {
  const auto files =
    causallm_test::makeTinyCausalLMFiles("OBCache", "activation", "FP32");
  std::string baseline;
  for (int budget : {-1, 0, 8, 3}) {
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
    if (budget >= 0)
      nntr["obcache"] = {{"budget", budget}, {"window", 2}, {"kernel", 1}};
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
    const auto marker = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(model.hasRun());
    if (budget < 0) {
      baseline = model.getOutputText();
      EXPECT_FALSE(baseline.empty());
    }
    if (budget != 3)
      EXPECT_EQ(model.getOutputText(), baseline);
    else
      EXPECT_FALSE(model.getOutputText().empty());
    if (budget <= 0)
      EXPECT_EQ(marker.find("[OBCache"), std::string::npos);
    else {
      EXPECT_NE(marker.find("[OBCache layer=0] prompt=6 retained=" +
                            std::to_string(budget == 3 ? 3 : 6)),
                std::string::npos);
      EXPECT_THROW(model.runPrompt("hello"), std::invalid_argument);
    }
  }
}
} // namespace
