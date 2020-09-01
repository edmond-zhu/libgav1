// Copyright 2020 The libgav1 Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "src/dsp/loop_restoration.h"
#include "src/utils/cpu.h"

#if LIBGAV1_TARGETING_AVX2 && LIBGAV1_MAX_BITDEPTH >= 10
#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/dsp/common.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/x86/common_avx2.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace {

inline void WienerHorizontalClip(const __m256i s[2],
                                 int16_t* const wiener_buffer) {
  constexpr int offset =
      1 << (10 + kWienerFilterBits - kInterRoundBitsHorizontal - 1);
  constexpr int limit = (offset << 2) - 1;
  const __m256i offsets = _mm256_set1_epi16(-offset);
  const __m256i limits = _mm256_set1_epi16(limit - offset);
  const __m256i round = _mm256_set1_epi32(1 << (kInterRoundBitsHorizontal - 1));
  const __m256i sum0 = _mm256_add_epi32(s[0], round);
  const __m256i sum1 = _mm256_add_epi32(s[1], round);
  const __m256i rounded_sum0 =
      _mm256_srai_epi32(sum0, kInterRoundBitsHorizontal);
  const __m256i rounded_sum1 =
      _mm256_srai_epi32(sum1, kInterRoundBitsHorizontal);
  const __m256i rounded_sum = _mm256_packs_epi32(rounded_sum0, rounded_sum1);
  const __m256i d0 = _mm256_max_epi16(rounded_sum, offsets);
  const __m256i d1 = _mm256_min_epi16(d0, limits);
  StoreAligned32(wiener_buffer, d1);
}

inline void WienerHorizontalTap7Kernel(const __m256i s[7],
                                       const __m256i filter[2],
                                       int16_t* const wiener_buffer) {
  const __m256i s06 = _mm256_add_epi16(s[0], s[6]);
  const __m256i s15 = _mm256_add_epi16(s[1], s[5]);
  const __m256i s24 = _mm256_add_epi16(s[2], s[4]);
  const __m256i ss0 = _mm256_unpacklo_epi16(s06, s15);
  const __m256i ss1 = _mm256_unpackhi_epi16(s06, s15);
  const __m256i ss2 = _mm256_unpacklo_epi16(s24, s[3]);
  const __m256i ss3 = _mm256_unpackhi_epi16(s24, s[3]);
  __m256i madds[4];
  madds[0] = _mm256_madd_epi16(ss0, filter[0]);
  madds[1] = _mm256_madd_epi16(ss1, filter[0]);
  madds[2] = _mm256_madd_epi16(ss2, filter[1]);
  madds[3] = _mm256_madd_epi16(ss3, filter[1]);
  madds[0] = _mm256_add_epi32(madds[0], madds[2]);
  madds[1] = _mm256_add_epi32(madds[1], madds[3]);
  WienerHorizontalClip(madds, wiener_buffer);
}

inline void WienerHorizontalTap5Kernel(const __m256i s[5], const __m256i filter,
                                       int16_t* const wiener_buffer) {
  const __m256i s04 = _mm256_add_epi16(s[0], s[4]);
  const __m256i s13 = _mm256_add_epi16(s[1], s[3]);
  const __m256i s2d = _mm256_add_epi16(s[2], s[2]);
  const __m256i s0m = _mm256_sub_epi16(s04, s2d);
  const __m256i s1m = _mm256_sub_epi16(s13, s2d);
  const __m256i ss0 = _mm256_unpacklo_epi16(s0m, s1m);
  const __m256i ss1 = _mm256_unpackhi_epi16(s0m, s1m);
  __m256i madds[2];
  madds[0] = _mm256_madd_epi16(ss0, filter);
  madds[1] = _mm256_madd_epi16(ss1, filter);
  const __m256i s2_lo = _mm256_unpacklo_epi16(s[2], _mm256_setzero_si256());
  const __m256i s2_hi = _mm256_unpackhi_epi16(s[2], _mm256_setzero_si256());
  const __m256i s2x128_lo = _mm256_slli_epi32(s2_lo, 7);
  const __m256i s2x128_hi = _mm256_slli_epi32(s2_hi, 7);
  madds[0] = _mm256_add_epi32(madds[0], s2x128_lo);
  madds[1] = _mm256_add_epi32(madds[1], s2x128_hi);
  WienerHorizontalClip(madds, wiener_buffer);
}

inline void WienerHorizontalTap3Kernel(const __m256i s[3], const __m256i filter,
                                       int16_t* const wiener_buffer) {
  const __m256i s02 = _mm256_add_epi16(s[0], s[2]);
  const __m256i ss0 = _mm256_unpacklo_epi16(s02, s[1]);
  const __m256i ss1 = _mm256_unpackhi_epi16(s02, s[1]);
  __m256i madds[2];
  madds[0] = _mm256_madd_epi16(ss0, filter);
  madds[1] = _mm256_madd_epi16(ss1, filter);
  WienerHorizontalClip(madds, wiener_buffer);
}

inline void WienerHorizontalTap7(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m256i* const coefficients,
                                 int16_t** const wiener_buffer) {
  __m256i filter[2];
  filter[0] = _mm256_shuffle_epi32(*coefficients, 0x0);
  filter[1] = _mm256_shuffle_epi32(*coefficients, 0x55);
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i s[7];
      s[0] = LoadUnaligned32(src + x + 0);
      s[1] = LoadUnaligned32(src + x + 1);
      s[2] = LoadUnaligned32(src + x + 2);
      s[3] = LoadUnaligned32(src + x + 3);
      s[4] = LoadUnaligned32(src + x + 4);
      s[5] = LoadUnaligned32(src + x + 5);
      s[6] = LoadUnaligned32(src + x + 6);
      WienerHorizontalTap7Kernel(s, filter, *wiener_buffer + x);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap5(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m256i* const coefficients,
                                 int16_t** const wiener_buffer) {
  const __m256i filter =
      _mm256_shuffle_epi8(*coefficients, _mm256_set1_epi32(0x05040302));
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i s[5];
      s[0] = LoadUnaligned32(src + x + 0);
      s[1] = LoadUnaligned32(src + x + 1);
      s[2] = LoadUnaligned32(src + x + 2);
      s[3] = LoadUnaligned32(src + x + 3);
      s[4] = LoadUnaligned32(src + x + 4);
      WienerHorizontalTap5Kernel(s, filter, *wiener_buffer + x);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap3(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m256i* const coefficients,
                                 int16_t** const wiener_buffer) {
  const auto filter = _mm256_shuffle_epi32(*coefficients, 0x55);
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i s[3];
      s[0] = LoadUnaligned32(src + x + 0);
      s[1] = LoadUnaligned32(src + x + 1);
      s[2] = LoadUnaligned32(src + x + 2);
      WienerHorizontalTap3Kernel(s, filter, *wiener_buffer + x);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap1(const uint16_t* src,
                                 const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      const __m256i s0 = LoadUnaligned32(src + x);
      const __m256i d0 = _mm256_slli_epi16(s0, 4);
      StoreAligned32(*wiener_buffer + x, d0);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline __m256i WienerVertical7(const __m256i a[4], const __m256i filter[4]) {
  const __m256i madd0 = _mm256_madd_epi16(a[0], filter[0]);
  const __m256i madd1 = _mm256_madd_epi16(a[1], filter[1]);
  const __m256i madd2 = _mm256_madd_epi16(a[2], filter[2]);
  const __m256i madd3 = _mm256_madd_epi16(a[3], filter[3]);
  const __m256i madd01 = _mm256_add_epi32(madd0, madd1);
  const __m256i madd23 = _mm256_add_epi32(madd2, madd3);
  const __m256i sum = _mm256_add_epi32(madd01, madd23);
  return _mm256_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m256i WienerVertical5(const __m256i a[3], const __m256i filter[3]) {
  const __m256i madd0 = _mm256_madd_epi16(a[0], filter[0]);
  const __m256i madd1 = _mm256_madd_epi16(a[1], filter[1]);
  const __m256i madd2 = _mm256_madd_epi16(a[2], filter[2]);
  const __m256i madd01 = _mm256_add_epi32(madd0, madd1);
  const __m256i sum = _mm256_add_epi32(madd01, madd2);
  return _mm256_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m256i WienerVertical3(const __m256i a[2], const __m256i filter[2]) {
  const __m256i madd0 = _mm256_madd_epi16(a[0], filter[0]);
  const __m256i madd1 = _mm256_madd_epi16(a[1], filter[1]);
  const __m256i sum = _mm256_add_epi32(madd0, madd1);
  return _mm256_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m256i WienerVerticalClip(const __m256i s[2]) {
  const __m256i d = _mm256_packus_epi32(s[0], s[1]);
  return _mm256_min_epu16(d, _mm256_set1_epi16(1023));
}

inline __m256i WienerVerticalFilter7(const __m256i a[7],
                                     const __m256i filter[2]) {
  const __m256i round = _mm256_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m256i b[4], c[2];
  b[0] = _mm256_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm256_unpacklo_epi16(a[2], a[3]);
  b[2] = _mm256_unpacklo_epi16(a[4], a[5]);
  b[3] = _mm256_unpacklo_epi16(a[6], round);
  c[0] = WienerVertical7(b, filter);
  b[0] = _mm256_unpackhi_epi16(a[0], a[1]);
  b[1] = _mm256_unpackhi_epi16(a[2], a[3]);
  b[2] = _mm256_unpackhi_epi16(a[4], a[5]);
  b[3] = _mm256_unpackhi_epi16(a[6], round);
  c[1] = WienerVertical7(b, filter);
  return WienerVerticalClip(c);
}

inline __m256i WienerVerticalFilter5(const __m256i a[5],
                                     const __m256i filter[3]) {
  const __m256i round = _mm256_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m256i b[3], c[2];
  b[0] = _mm256_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm256_unpacklo_epi16(a[2], a[3]);
  b[2] = _mm256_unpacklo_epi16(a[4], round);
  c[0] = WienerVertical5(b, filter);
  b[0] = _mm256_unpackhi_epi16(a[0], a[1]);
  b[1] = _mm256_unpackhi_epi16(a[2], a[3]);
  b[2] = _mm256_unpackhi_epi16(a[4], round);
  c[1] = WienerVertical5(b, filter);
  return WienerVerticalClip(c);
}

inline __m256i WienerVerticalFilter3(const __m256i a[3],
                                     const __m256i filter[2]) {
  const __m256i round = _mm256_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m256i b[2], c[2];
  b[0] = _mm256_unpacklo_epi16(a[0], a[1]);
  b[1] = _mm256_unpacklo_epi16(a[2], round);
  c[0] = WienerVertical3(b, filter);
  b[0] = _mm256_unpackhi_epi16(a[0], a[1]);
  b[1] = _mm256_unpackhi_epi16(a[2], round);
  c[1] = WienerVertical3(b, filter);
  return WienerVerticalClip(c);
}

inline __m256i WienerVerticalTap7Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m256i filter[2], __m256i a[7]) {
  a[0] = LoadAligned32(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned32(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned32(wiener_buffer + 2 * wiener_stride);
  a[3] = LoadAligned32(wiener_buffer + 3 * wiener_stride);
  a[4] = LoadAligned32(wiener_buffer + 4 * wiener_stride);
  a[5] = LoadAligned32(wiener_buffer + 5 * wiener_stride);
  a[6] = LoadAligned32(wiener_buffer + 6 * wiener_stride);
  return WienerVerticalFilter7(a, filter);
}

inline __m256i WienerVerticalTap5Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m256i filter[3], __m256i a[5]) {
  a[0] = LoadAligned32(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned32(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned32(wiener_buffer + 2 * wiener_stride);
  a[3] = LoadAligned32(wiener_buffer + 3 * wiener_stride);
  a[4] = LoadAligned32(wiener_buffer + 4 * wiener_stride);
  return WienerVerticalFilter5(a, filter);
}

inline __m256i WienerVerticalTap3Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m256i filter[2], __m256i a[3]) {
  a[0] = LoadAligned32(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned32(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned32(wiener_buffer + 2 * wiener_stride);
  return WienerVerticalFilter3(a, filter);
}

inline void WienerVerticalTap7Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m256i filter[2], __m256i d[2]) {
  __m256i a[8];
  d[0] = WienerVerticalTap7Kernel(wiener_buffer, wiener_stride, filter, a);
  a[7] = LoadAligned32(wiener_buffer + 7 * wiener_stride);
  d[1] = WienerVerticalFilter7(a + 1, filter);
}

inline void WienerVerticalTap5Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m256i filter[3], __m256i d[2]) {
  __m256i a[6];
  d[0] = WienerVerticalTap5Kernel(wiener_buffer, wiener_stride, filter, a);
  a[5] = LoadAligned32(wiener_buffer + 5 * wiener_stride);
  d[1] = WienerVerticalFilter5(a + 1, filter);
}

inline void WienerVerticalTap3Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m256i filter[2], __m256i d[2]) {
  __m256i a[4];
  d[0] = WienerVerticalTap3Kernel(wiener_buffer, wiener_stride, filter, a);
  a[3] = LoadAligned32(wiener_buffer + 3 * wiener_stride);
  d[1] = WienerVerticalFilter3(a + 1, filter);
}

inline void WienerVerticalTap7(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[4], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m256i c = _mm256_broadcastq_epi64(LoadLo8(coefficients));
  __m256i filter[4];
  filter[0] = _mm256_shuffle_epi32(c, 0x0);
  filter[1] = _mm256_shuffle_epi32(c, 0x55);
  filter[2] = _mm256_shuffle_epi8(c, _mm256_set1_epi32(0x03020504));
  filter[3] =
      _mm256_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[0]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i d[2];
      WienerVerticalTap7Kernel2(wiener_buffer + x, width, filter, d);
      StoreUnaligned32(dst + x, d[0]);
      StoreUnaligned32(dst + dst_stride + x, d[1]);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m256i a[7];
      const __m256i d =
          WienerVerticalTap7Kernel(wiener_buffer + x, width, filter, a);
      StoreUnaligned32(dst + x, d);
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap5(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[3], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m256i c = _mm256_broadcastq_epi64(LoadLo8(coefficients));
  __m256i filter[3];
  filter[0] = _mm256_shuffle_epi32(c, 0x0);
  filter[1] = _mm256_shuffle_epi8(c, _mm256_set1_epi32(0x03020504));
  filter[2] =
      _mm256_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[0]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i d[2];
      WienerVerticalTap5Kernel2(wiener_buffer + x, width, filter, d);
      StoreUnaligned32(dst + x, d[0]);
      StoreUnaligned32(dst + dst_stride + x, d[1]);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m256i a[5];
      const __m256i d =
          WienerVerticalTap5Kernel(wiener_buffer + x, width, filter, a);
      StoreUnaligned32(dst + x, d);
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap3(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[2], uint16_t* dst,
                               const ptrdiff_t dst_stride) {
  __m256i filter[2];
  filter[0] =
      _mm256_set1_epi32(*reinterpret_cast<const int32_t*>(coefficients));
  filter[1] =
      _mm256_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[0]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m256i d[2][2];
      WienerVerticalTap3Kernel2(wiener_buffer + x, width, filter, d[0]);
      StoreUnaligned32(dst + x, d[0][0]);
      StoreUnaligned32(dst + dst_stride + x, d[0][1]);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m256i a[3];
      const __m256i d =
          WienerVerticalTap3Kernel(wiener_buffer + x, width, filter, a);
      StoreUnaligned32(dst + x, d);
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap1Kernel(const int16_t* const wiener_buffer,
                                     uint16_t* const dst) {
  const __m256i a = LoadAligned32(wiener_buffer);
  const __m256i b = _mm256_add_epi16(a, _mm256_set1_epi16(8));
  const __m256i c = _mm256_srai_epi16(b, 4);
  const __m256i d = _mm256_max_epi16(c, _mm256_setzero_si256());
  const __m256i e = _mm256_min_epi16(d, _mm256_set1_epi16(1023));
  StoreUnaligned32(dst, e);
}

inline void WienerVerticalTap1(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               uint16_t* dst, const ptrdiff_t dst_stride) {
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      WienerVerticalTap1Kernel(wiener_buffer + x, dst + x);
      WienerVerticalTap1Kernel(wiener_buffer + width + x, dst + dst_stride + x);
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      WienerVerticalTap1Kernel(wiener_buffer + x, dst + x);
      x += 16;
    } while (x < width);
  }
}

void WienerFilter_AVX2(const RestorationUnitInfo& restoration_info,
                       const void* const source, const void* const top_border,
                       const void* const bottom_border, const ptrdiff_t stride,
                       const int width, const int height,
                       RestorationBuffer* const restoration_buffer,
                       void* const dest) {
  const int16_t* const number_leading_zero_coefficients =
      restoration_info.wiener_info.number_leading_zero_coefficients;
  const int number_rows_to_skip = std::max(
      static_cast<int>(number_leading_zero_coefficients[WienerInfo::kVertical]),
      1);
  const ptrdiff_t wiener_stride = Align(width, 16);
  int16_t* const wiener_buffer_vertical = restoration_buffer->wiener_buffer;
  // The values are saturated to 13 bits before storing.
  int16_t* wiener_buffer_horizontal =
      wiener_buffer_vertical + number_rows_to_skip * wiener_stride;

  // horizontal filtering.
  // Over-reads up to 15 - |kRestorationHorizontalBorder| values.
  const int height_horizontal =
      height + kWienerFilterTaps - 1 - 2 * number_rows_to_skip;
  const int height_extra = (height_horizontal - height) >> 1;
  assert(height_extra <= 2);
  const auto* const src = static_cast<const uint16_t*>(source);
  const auto* const top = static_cast<const uint16_t*>(top_border);
  const auto* const bottom = static_cast<const uint16_t*>(bottom_border);
  const __m128i c =
      LoadLo8(restoration_info.wiener_info.filter[WienerInfo::kHorizontal]);
  const __m256i coefficients_horizontal = _mm256_broadcastq_epi64(c);
  if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 0) {
    WienerHorizontalTap7(top + (2 - height_extra) * stride - 3, stride,
                         wiener_stride, height_extra, &coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap7(src - 3, stride, wiener_stride, height,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap7(bottom - 3, stride, wiener_stride, height_extra,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 1) {
    WienerHorizontalTap5(top + (2 - height_extra) * stride - 2, stride,
                         wiener_stride, height_extra, &coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap5(src - 2, stride, wiener_stride, height,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap5(bottom - 2, stride, wiener_stride, height_extra,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 2) {
    // The maximum over-reads happen here.
    WienerHorizontalTap3(top + (2 - height_extra) * stride - 1, stride,
                         wiener_stride, height_extra, &coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap3(src - 1, stride, wiener_stride, height,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap3(bottom - 1, stride, wiener_stride, height_extra,
                         &coefficients_horizontal, &wiener_buffer_horizontal);
  } else {
    assert(number_leading_zero_coefficients[WienerInfo::kHorizontal] == 3);
    WienerHorizontalTap1(top + (2 - height_extra) * stride, stride,
                         wiener_stride, height_extra,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap1(src, stride, wiener_stride, height,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap1(bottom, stride, wiener_stride, height_extra,
                         &wiener_buffer_horizontal);
  }

  // vertical filtering.
  // Over-writes up to 15 values.
  const int16_t* const filter_vertical =
      restoration_info.wiener_info.filter[WienerInfo::kVertical];
  auto* dst = static_cast<uint16_t*>(dest);
  if (number_leading_zero_coefficients[WienerInfo::kVertical] == 0) {
    // Because the top row of |source| is a duplicate of the second row, and the
    // bottom row of |source| is a duplicate of its above row, we can duplicate
    // the top and bottom row of |wiener_buffer| accordingly.
    memcpy(wiener_buffer_horizontal, wiener_buffer_horizontal - wiener_stride,
           sizeof(*wiener_buffer_horizontal) * wiener_stride);
    memcpy(restoration_buffer->wiener_buffer,
           restoration_buffer->wiener_buffer + wiener_stride,
           sizeof(*restoration_buffer->wiener_buffer) * wiener_stride);
    WienerVerticalTap7(wiener_buffer_vertical, wiener_stride, height,
                       filter_vertical, dst, stride);
  } else if (number_leading_zero_coefficients[WienerInfo::kVertical] == 1) {
    WienerVerticalTap5(wiener_buffer_vertical + wiener_stride, wiener_stride,
                       height, filter_vertical + 1, dst, stride);
  } else if (number_leading_zero_coefficients[WienerInfo::kVertical] == 2) {
    WienerVerticalTap3(wiener_buffer_vertical + 2 * wiener_stride,
                       wiener_stride, height, filter_vertical + 2, dst, stride);
  } else {
    assert(number_leading_zero_coefficients[WienerInfo::kVertical] == 3);
    WienerVerticalTap1(wiener_buffer_vertical + 3 * wiener_stride,
                       wiener_stride, height, dst, stride);
  }
}

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth10);
  assert(dsp != nullptr);
#if DSP_ENABLED_10BPP_AVX2(WienerFilter)
  dsp->loop_restorations[0] = WienerFilter_AVX2;
#endif
}

}  // namespace

void LoopRestorationInit10bpp_AVX2() { Init10bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else  // !(LIBGAV1_TARGETING_AVX2 && LIBGAV1_MAX_BITDEPTH >= 10)
namespace libgav1 {
namespace dsp {

void LoopRestorationInit10bpp_AVX2() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_TARGETING_AVX2 && LIBGAV1_MAX_BITDEPTH >= 10
