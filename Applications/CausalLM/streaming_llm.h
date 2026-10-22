// SPDX-License-Identifier: Apache-2.0
/**
 * @file streaming_llm.h
 * @brief StreamingLLM attention-sink and recent-token cache policy.
 * @see https://arxiv.org/abs/2309.17453
 */
#ifndef __STREAMING_LLM_H__
#define __STREAMING_LLM_H__

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace causallm {

/**
 * @brief A disabled policy has zero recent tokens. An enabled cache preserves
 * sink_tokens and recent_tokens after each attention step. The next query
 * attends to those past tokens and itself before eviction.
 */
struct StreamingLLMConfig {
  unsigned int sink_tokens = 4;
  unsigned int recent_tokens = 0;

  bool enabled() const { return recent_tokens != 0; }

  unsigned int capacity() const {
    if (sink_tokens > std::numeric_limits<unsigned int>::max() - recent_tokens)
      throw std::invalid_argument("StreamingLLM cache capacity overflows");
    return sink_tokens + recent_tokens;
  }

  /** @brief Bounded attention scratch includes one incoming token. */
  unsigned int attentionCapacity() const {
    const auto rows = capacity();
    if (rows == std::numeric_limits<unsigned int>::max())
      throw std::invalid_argument("StreamingLLM attention capacity overflows");
    return rows + 1;
  }

  /** @brief Strict unsigned parsing for JSON numbers and layer properties. */
  static unsigned int parseUnsigned(const std::string &value) {
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument("StreamingLLM expects an unsigned integer");
    const auto number = std::stoull(value);
    if (number > std::numeric_limits<unsigned int>::max())
      throw std::invalid_argument("StreamingLLM integer is out of range");
    return static_cast<unsigned int>(number);
  }

  /**
   * @brief Compact a scratch cache after attention and return retained rows.
   * The allocation must hold length rows. Preserve the initial sink rows and
   * newest recent rows in chronological order. Call separately for raw keys
   * and values; rotated keys are never persisted.
   */
  unsigned int evict(unsigned int length, void *cache, size_t row_bytes) const {
    if (!enabled() || !length || !cache || !row_bytes)
      throw std::invalid_argument("Invalid StreamingLLM cache eviction");
    const unsigned int cap = capacity();
    if (length > std::numeric_limits<size_t>::max() / row_bytes)
      throw std::invalid_argument("StreamingLLM cache byte size overflows");
    if (length > cap) {
      auto *bytes = static_cast<unsigned char *>(cache);
      std::memmove(bytes + sink_tokens * row_bytes,
                   bytes + (length - recent_tokens) * row_bytes,
                   recent_tokens * row_bytes);
    }
    return std::min(length, cap);
  }
};

} // namespace causallm
#endif // __STREAMING_LLM_H__
