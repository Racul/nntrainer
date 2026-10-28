// SPDX-License-Identifier: Apache-2.0
/**
 * @file compact_kv.h
 * @brief Layer-owned, physically compact half-precision KV storage.
 */
#ifndef __CAUSALLM_COMPACT_KV_H__
#define __CAUSALLM_COMPACT_KV_H__

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

inline unsigned int kvUnsigned(const std::string &s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("KV setting requires an unsigned integer");
  const auto n = std::stoull(s);
  if (n > std::numeric_limits<unsigned int>::max())
    throw std::out_of_range("KV setting exceeds unsigned int");
  return static_cast<unsigned int>(n);
}

inline double kvRatio(const std::string &s) {
  size_t end = 0;
  const double n = std::stod(s, &end);
  if (end != s.size() || !std::isfinite(n) || n <= 0 || n > 1)
    throw std::invalid_argument("KV keep_ratio must be in (0, 1]");
  return n;
}

inline void kvFinite(const std::vector<float> &values) {
  for (const auto x : values)
    if (!std::isfinite(x))
      throw std::invalid_argument("Non-finite compressed KV input");
}

/** @brief Stable top-k, followed by chronological physical storage. */
inline std::vector<unsigned int> kvTop(const std::vector<double> &scores,
                                       unsigned int keep) {
  if (keep > scores.size())
    throw std::invalid_argument("KV budget exceeds candidate count");
  for (const auto s : scores)
    if (std::isnan(s))
      throw std::invalid_argument("Non-finite KV importance score");
  std::vector<unsigned int> ids(scores.size());
  std::iota(ids.begin(), ids.end(), 0u);
  std::partial_sort(ids.begin(), ids.begin() + keep, ids.end(),
                    [&](unsigned int a, unsigned int b) {
                      return scores[a] == scores[b] ? a < b
                                                    : scores[a] > scores[b];
                    });
  ids.resize(keep);
  std::sort(ids.begin(), ids.end());
  return ids;
}

/**
 * @brief Owns storage independently of graph input aliases.
 * @details Graph cache inputs are one-row binding placeholders. Attention
 * uses these owned tensors exclusively when compression is enabled. Replacing
 * them releases the old full-prompt allocation after all attention views die.
 * Absolute RoPE positions live in next; live is the physical cache length.
 */
class CompactKVCache {
public:
  nntrainer::Tensor key, value;
  unsigned int prompt = 0, next = 0, live = 0, layer = 0;

  void reset(unsigned int length) {
    if (!length)
      throw std::invalid_argument("Compressed KV requires a nonempty prompt");
    prompt = length;
    next = live = 0;
    key = nntrainer::Tensor();
    value = nntrainer::Tensor();
  }

  void checkStep(unsigned int from, unsigned int step,
                 unsigned int limit) const {
    if (!prompt || from != next || !step || step != (next ? 1u : prompt))
      throw std::invalid_argument(
        "Compressed KV requires full fresh prefill then consecutive decode");
    if (from >= limit || step > limit - from)
      throw std::out_of_range("Compressed KV exceeds logical sequence limit");
  }

  void allocate(unsigned int rows, unsigned int width) {
    if (!rows || !width)
      throw std::invalid_argument("Empty compressed KV allocation");
#ifdef ENABLE_FP16
    const auto dtype = ml::train::TensorDim::DataType::FP16;
#else
    const auto dtype = ml::train::TensorDim::DataType::UINT16;
#endif
    key = nntrainer::Tensor(1, 1, rows, width,
                            ml::train::TensorDim::Format::NCHW, dtype);
    value = nntrainer::Tensor(1, 1, rows, width,
                              ml::train::TensorDim::Format::NCHW, dtype);
  }

  /** @brief Materialize one head once, outside the scoring inner loops. */
  void head(unsigned int h, unsigned int heads, unsigned int dim,
            std::vector<float> &k, std::vector<float> &v) const {
    if (!heads || h >= heads || !dim || key.width() != heads * dim ||
        live > key.height())
      throw std::invalid_argument("Invalid compressed KV head geometry");
    k.resize(static_cast<size_t>(live) * dim);
    v.resize(k.size());
    const auto *kp = reinterpret_cast<const uint16_t *>(key.getData());
    const auto *vp = reinterpret_cast<const uint16_t *>(value.getData());
    for (unsigned int t = 0; t < live; ++t)
      for (unsigned int j = 0; j < dim; ++j) {
        const size_t src = static_cast<size_t>(t) * key.width() + h * dim + j;
        const size_t dst = static_cast<size_t>(t) * dim + j;
        k[dst] = nntrainer::compute_fp16_to_fp32(kp[src]);
        v[dst] = nntrainer::compute_fp16_to_fp32(vp[src]);
      }
    kvFinite(k);
    kvFinite(v);
  }

  /** @brief Gather identical K/V indices per head into a smaller allocation. */
  void compact(const std::vector<std::vector<unsigned int>> &ids,
               unsigned int dim, unsigned int capacity) {
    if (ids.empty() || !dim || key.width() != ids.size() * dim ||
        ids.front().empty() || capacity < ids.front().size())
      throw std::invalid_argument("Invalid compressed KV selection geometry");
    const auto keep = ids.front().size();
    for (const auto &head_ids : ids) {
      if (head_ids.size() != keep ||
          !std::is_sorted(head_ids.begin(), head_ids.end()) ||
          std::adjacent_find(head_ids.begin(), head_ids.end()) !=
            head_ids.end())
        throw std::invalid_argument("KV indices must be unique and ordered");
      for (const auto id : head_ids)
        if (id >= live)
          throw std::out_of_range("KV selection exceeds live rows");
    }
    auto old_key = key;
    auto old_value = value;
    allocate(capacity, old_key.width());
    const auto *kp = reinterpret_cast<const uint16_t *>(old_key.getData());
    const auto *vp = reinterpret_cast<const uint16_t *>(old_value.getData());
    auto *ko = reinterpret_cast<uint16_t *>(key.getData());
    auto *vo = reinterpret_cast<uint16_t *>(value.getData());
    for (size_t h = 0; h < ids.size(); ++h)
      for (size_t t = 0; t < keep; ++t) {
        const size_t src =
          static_cast<size_t>(ids[h][t]) * key.width() + h * dim;
        const size_t dst = t * key.width() + h * dim;
        std::memcpy(ko + dst, kp + src, dim * sizeof(uint16_t));
        std::memcpy(vo + dst, vp + src, dim * sizeof(uint16_t));
      }
    live = static_cast<unsigned int>(keep);
  }

  void reserve(unsigned int heads, unsigned int dim, unsigned int capacity) {
    std::vector<std::vector<unsigned int>> ids(heads,
                                               std::vector<unsigned int>(live));
    for (auto &head_ids : ids)
      std::iota(head_ids.begin(), head_ids.end(), 0u);
    compact(ids, dim, capacity);
  }

  size_t bytes() const {
    return static_cast<size_t>(key.height()) * key.width() * 2 *
           sizeof(uint16_t);
  }
};
} // namespace causallm
#endif
