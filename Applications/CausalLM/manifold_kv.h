// SPDX-License-Identifier: Apache-2.0
/**
 * @file manifold_kv.h
 * @brief ManifoldKV prompt selection by distance from key centroids.
 * @see https://proceedings.mlr.press/v306/datta26b.html
 */
#ifndef __CAUSALLM_MANIFOLD_KV_H__
#define __CAUSALLM_MANIFOLD_KV_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fp16.h>
#include <tensor.h>

namespace causallm {

/** @brief Fixed prompt budget per KV head; zero window uses a global centroid.
 */
struct ManifoldKVConfig {
  unsigned int budget = 0;
  unsigned int window_size = 0;

  bool enabled() const { return budget != 0; }

  /** @brief Parse JSON integers and layer properties without narrowing. */
  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("ManifoldKV expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::out_of_range("ManifoldKV integer is out of range");
    return static_cast<unsigned int>(number);
  }

  /** @brief Reserve retained prompt rows and the configured decode allowance.
   */
  unsigned int capacity(unsigned int max_sequence, int generation_limit) const {
    if (!budget || budget > max_sequence)
      throw std::invalid_argument("ManifoldKV budget must fit max_seq_len");
    const auto room = max_sequence - budget;
    return budget +
           (generation_limit > 0
              ? std::min(room, static_cast<unsigned int>(generation_limit))
              : room);
  }
};

/**
 * @brief One-shot, layer-local prompt compression with fixed per-KV-head
 * budgets. Cached keys already carry their original RoPE positions. Generated
 * tokens are appended without rescoring or eviction. No sinks, pooling or
 * recent-token reservation are added to the paper's geometric scorer.
 */
class ManifoldKVState {
public:
  bool enabled() const { return config_.enabled(); }

  /** @brief Set configuration; prompt_length starts a fresh compression
   * session. */
  bool consumeProperty(const std::string &key, const std::string &value) {
    if (key != "manifold_kv_budget" && key != "manifold_kv_window_size" &&
        key != "manifold_kv_prompt_length")
      return false;
    const auto number = ManifoldKVConfig::parseUnsigned(value);
    if (key == "manifold_kv_budget")
      config_.budget = number;
    else if (key == "manifold_kv_window_size")
      config_.window_size = number;
    else {
      if (!enabled() || !number)
        throw std::invalid_argument(
          "ManifoldKV requires a budget and a prompt");
      prompt_length_ = number;
      next_position_ = 0;
    }
    return true;
  }

  /** @brief Map the original token position to a compacted cache row. */
  unsigned int physicalPosition(unsigned int logical) const {
    if (!prompt_length_ || (logical && logical < prompt_length_))
      throw std::invalid_argument(
        "ManifoldKV cannot address an unfinished prompt");
    return logical ? logical - (prompt_length_ -
                                std::min(prompt_length_, config_.budget))
                   : 0;
  }

  /** @brief Validate the entire step before writing cache memory or applying
   * RoPE. */
  void checkStep(unsigned int from, unsigned int step, unsigned int capacity,
                 unsigned int position_limit) const {
    if (!enabled() || !prompt_length_ || !step || from != next_position_ ||
        step != (from ? 1u : prompt_length_) || from > position_limit ||
        step > position_limit - from)
      throw std::invalid_argument("ManifoldKV requires one full prefill "
                                  "followed by consecutive single tokens");
    const auto retained = from ? physicalPosition(from) : 0u;
    const auto added = from ? step : std::min(step, config_.budget);
    if (retained > capacity || added > capacity - retained)
      throw std::out_of_range("ManifoldKV cache capacity exceeded");
  }

  /** @brief Commit bookkeeping after successful attention and cache writes. */
  void finishStep(unsigned int from, unsigned int step) {
    next_position_ = from + step;
  }

  /**
   * @brief Score each KV head independently and return chronological keep
   * indices. Squared L2 distances have the same ordering as L2 distances.
   * Windowed mode computes independent centroids over consecutive,
   * non-overlapping windows, including the final partial window, as in the
   * paper's Algorithm 2.
   */
  std::vector<std::vector<unsigned int>> select(const nntrainer::Tensor &keys,
                                                unsigned int heads,
                                                unsigned int dim) const {
    using DType = nntrainer::TensorDim::DataType;
    const auto dtype = keys.getDataType();
    if (!enabled() || !heads || !dim || keys.batch() != 1 ||
        keys.channel() != 1 || !keys.height() ||
        static_cast<size_t>(heads) * dim != keys.width() ||
        (dtype != DType::FP32 && dtype != DType::FP16 &&
         dtype != DType::UINT16))
      throw std::invalid_argument("Invalid ManifoldKV key geometry or dtype");
    const unsigned int length = keys.height();
    const auto keep = std::min(length, config_.budget);
    const auto window = config_.window_size ? config_.window_size : length;
    auto read = [&](size_t i) {
      const float x =
        dtype == DType::FP32
          ? keys.getData<float>()[i]
          : nntrainer::compute_fp16_to_fp32(keys.getData<uint16_t>()[i]);
      if (!std::isfinite(x))
        throw std::invalid_argument("ManifoldKV keys must be finite");
      return x;
    };
    std::vector<std::vector<unsigned int>> selected(heads);
    std::vector<float> centroid(dim), scores(length);
    std::vector<unsigned int> indices(length);
    for (unsigned int head = 0; head < heads; ++head) {
      for (unsigned int start = 0; start < length;) {
        const auto count = std::min(window, length - start);
        std::fill(centroid.begin(), centroid.end(), 0.0f);
        for (unsigned int row = start; row < start + count; ++row)
          for (unsigned int d = 0; d < dim; ++d)
            centroid[d] +=
              read(static_cast<size_t>(row) * keys.width() + head * dim + d);
        for (auto &x : centroid)
          x /= count;
        for (unsigned int row = start; row < start + count; ++row) {
          float score = 0.0f;
          for (unsigned int d = 0; d < dim; ++d) {
            const float delta =
              read(static_cast<size_t>(row) * keys.width() + head * dim + d) -
              centroid[d];
            score += delta * delta;
          }
          if (!std::isfinite(score))
            throw std::invalid_argument("ManifoldKV score overflow");
          scores[row] = score;
        }
        start += count;
      }
      std::iota(indices.begin(), indices.end(), 0u);
      std::partial_sort(indices.begin(), indices.begin() + keep, indices.end(),
                        [&](unsigned int a, unsigned int b) {
                          return scores[a] == scores[b] ? a < b
                                                        : scores[a] > scores[b];
                        });
      selected[head].assign(indices.begin(), indices.begin() + keep);
      std::sort(selected[head].begin(), selected[head].end());
    }
    return selected;
  }

  /** @brief Copy selected K/V head slices bitwise into the persistent cache. */
  void compress(const nntrainer::Tensor &keys, const nntrainer::Tensor &values,
                nntrainer::Tensor &cache_key, nntrainer::Tensor &cache_value,
                unsigned int heads, unsigned int dim) const {
    if (keys.getDim() != values.getDim() ||
        cache_key.getDim() != cache_value.getDim() ||
        keys.getDataType() != cache_key.getDataType() ||
        cache_key.batch() != 1 || cache_key.channel() != 1 ||
        cache_key.width() != keys.width() ||
        cache_key.height() < std::min<size_t>(keys.height(), config_.budget))
      throw std::invalid_argument("Invalid ManifoldKV destination cache");
    const auto selected = select(keys, heads, dim);
    const size_t element =
      keys.getDataType() == nntrainer::TensorDim::DataType::FP32
        ? sizeof(float)
        : sizeof(uint16_t);
    const size_t row_bytes = keys.width() * element;
    const size_t head_bytes = dim * element;
    for (unsigned int head = 0; head < heads; ++head)
      for (size_t row = 0; row < selected[head].size(); ++row)
        for (unsigned int plane = 0; plane < 2; ++plane) {
          const auto &source = plane ? values : keys;
          auto &destination = plane ? cache_value : cache_key;
          std::memmove(destination.getData<unsigned char>() + row * row_bytes +
                         head * head_bytes,
                       source.getData<unsigned char>() +
                         selected[head][row] * row_bytes + head * head_bytes,
                       head_bytes);
        }
  }

private:
  ManifoldKVConfig config_;
  unsigned int prompt_length_ = 0;
  unsigned int next_position_ = 0;
};

} // namespace causallm
#endif // __CAUSALLM_MANIFOLD_KV_H__
