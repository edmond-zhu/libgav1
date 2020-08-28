/*
 * Copyright 2020 The libgav1 Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LIBGAV1_SRC_DSP_X86_COMMON_AVX2_H_
#define LIBGAV1_SRC_DSP_X86_COMMON_AVX2_H_

#include "src/utils/compiler_attributes.h"
#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_AVX2

#include <immintrin.h>

#include <cassert>
#include <cstdint>

namespace libgav1 {
namespace dsp {

//------------------------------------------------------------------------------
// Load functions.

inline __m256i LoadAligned32(const void* a) {
  assert((reinterpret_cast<uintptr_t>(a) & 0x1f) == 0);
  return _mm256_load_si256(static_cast<const __m256i*>(a));
}

inline __m256i LoadUnaligned32(const void* a) {
  return _mm256_loadu_si256(static_cast<const __m256i*>(a));
}

//------------------------------------------------------------------------------
// Store functions.

inline void StoreAligned32(void* a, const __m256i v) {
  assert((reinterpret_cast<uintptr_t>(a) & 0x1f) == 0);
  _mm256_store_si256(static_cast<__m256i*>(a), v);
}

inline void StoreUnaligned32(void* a, const __m256i v) {
  _mm256_storeu_si256(static_cast<__m256i*>(a), v);
}

}  // namespace dsp
}  // namespace libgav1

#endif  // LIBGAV1_ENABLE_AVX2
#endif  // LIBGAV1_SRC_DSP_X86_COMMON_AVX2_H_
