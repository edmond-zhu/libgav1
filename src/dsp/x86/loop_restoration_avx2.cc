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

#if LIBGAV1_ENABLE_AVX2
#include <smmintrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/dsp/common.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

inline void WienerHorizontalTap7Kernel(const __m128i s[2],
                                       const __m128i filter[4],
                                       int16_t* const wiener_buffer) {
  const int limit =
      (1 << (8 + 1 + kWienerFilterBits - kInterRoundBitsHorizontal)) - 1;
  const int offset =
      1 << (8 + kWienerFilterBits - kInterRoundBitsHorizontal - 1);
  const __m128i offsets = _mm_set1_epi16(-offset);
  const __m128i limits = _mm_set1_epi16(limit - offset);
  const __m128i round = _mm_set1_epi16(1 << (kInterRoundBitsHorizontal - 1));
  const auto s01 = _mm_alignr_epi8(s[1], s[0], 1);
  const auto s23 = _mm_alignr_epi8(s[1], s[0], 5);
  const auto s45 = _mm_alignr_epi8(s[1], s[0], 9);
  const auto s67 = _mm_alignr_epi8(s[1], s[0], 13);
  const __m128i madd01 = _mm_maddubs_epi16(s01, filter[0]);
  const __m128i madd23 = _mm_maddubs_epi16(s23, filter[1]);
  const __m128i madd45 = _mm_maddubs_epi16(s45, filter[2]);
  const __m128i madd67 = _mm_maddubs_epi16(s67, filter[3]);
  const __m128i madd0123 = _mm_add_epi16(madd01, madd23);
  const __m128i madd4567 = _mm_add_epi16(madd45, madd67);
  // The sum range here is [-128 * 255, 90 * 255].
  const __m128i madd = _mm_add_epi16(madd0123, madd4567);
  const __m128i sum = _mm_add_epi16(madd, round);
  const __m128i rounded_sum0 = _mm_srai_epi16(sum, kInterRoundBitsHorizontal);
  // Calculate scaled down offset correction, and add to sum here to prevent
  // signed 16 bit outranging.
  const __m128i s_3x128 =
      _mm_slli_epi16(_mm_srli_epi16(s23, 8), 7 - kInterRoundBitsHorizontal);
  const __m128i rounded_sum1 = _mm_add_epi16(rounded_sum0, s_3x128);
  const __m128i d0 = _mm_max_epi16(rounded_sum1, offsets);
  const __m128i d1 = _mm_min_epi16(d0, limits);
  StoreAligned16(wiener_buffer, d1);
}

inline void WienerHorizontalTap5Kernel(const __m128i s[2],
                                       const __m128i filter[3],
                                       int16_t* const wiener_buffer) {
  const int limit =
      (1 << (8 + 1 + kWienerFilterBits - kInterRoundBitsHorizontal)) - 1;
  const int offset =
      1 << (8 + kWienerFilterBits - kInterRoundBitsHorizontal - 1);
  const __m128i offsets = _mm_set1_epi16(-offset);
  const __m128i limits = _mm_set1_epi16(limit - offset);
  const __m128i round = _mm_set1_epi16(1 << (kInterRoundBitsHorizontal - 1));
  const auto s01 = _mm_alignr_epi8(s[1], s[0], 1);
  const auto s23 = _mm_alignr_epi8(s[1], s[0], 5);
  const auto s45 = _mm_alignr_epi8(s[1], s[0], 9);
  const __m128i madd01 = _mm_maddubs_epi16(s01, filter[0]);
  const __m128i madd23 = _mm_maddubs_epi16(s23, filter[1]);
  const __m128i madd45 = _mm_maddubs_epi16(s45, filter[2]);
  const __m128i madd0123 = _mm_add_epi16(madd01, madd23);
  // The sum range here is [-128 * 255, 90 * 255].
  const __m128i madd = _mm_add_epi16(madd0123, madd45);
  const __m128i sum = _mm_add_epi16(madd, round);
  const __m128i rounded_sum0 = _mm_srai_epi16(sum, kInterRoundBitsHorizontal);
  // Calculate scaled down offset correction, and add to sum here to prevent
  // signed 16 bit outranging.
  const __m128i s_3x128 =
      _mm_srli_epi16(_mm_slli_epi16(s23, 8), kInterRoundBitsHorizontal + 1);
  const __m128i rounded_sum1 = _mm_add_epi16(rounded_sum0, s_3x128);
  const __m128i d0 = _mm_max_epi16(rounded_sum1, offsets);
  const __m128i d1 = _mm_min_epi16(d0, limits);
  StoreAligned16(wiener_buffer, d1);
}

inline void WienerHorizontalTap3Kernel(const __m128i s[2],
                                       const __m128i filter[2],
                                       int16_t* const wiener_buffer) {
  const int limit =
      (1 << (8 + 1 + kWienerFilterBits - kInterRoundBitsHorizontal)) - 1;
  const int offset =
      1 << (8 + kWienerFilterBits - kInterRoundBitsHorizontal - 1);
  const __m128i offsets = _mm_set1_epi16(-offset);
  const __m128i limits = _mm_set1_epi16(limit - offset);
  const __m128i round = _mm_set1_epi16(1 << (kInterRoundBitsHorizontal - 1));
  const auto s01 = _mm_alignr_epi8(s[1], s[0], 1);
  const auto s23 = _mm_alignr_epi8(s[1], s[0], 5);
  const __m128i madd01 = _mm_maddubs_epi16(s01, filter[0]);
  const __m128i madd23 = _mm_maddubs_epi16(s23, filter[1]);
  // The sum range here is [-128 * 255, 90 * 255].
  const __m128i madd = _mm_add_epi16(madd01, madd23);
  const __m128i sum = _mm_add_epi16(madd, round);
  const __m128i rounded_sum0 = _mm_srai_epi16(sum, kInterRoundBitsHorizontal);
  // Calculate scaled down offset correction, and add to sum here to prevent
  // signed 16 bit outranging.
  const __m128i s_3x128 =
      _mm_slli_epi16(_mm_srli_epi16(s01, 8), 7 - kInterRoundBitsHorizontal);
  const __m128i rounded_sum1 = _mm_add_epi16(rounded_sum0, s_3x128);
  const __m128i d0 = _mm_max_epi16(rounded_sum1, offsets);
  const __m128i d1 = _mm_min_epi16(d0, limits);
  StoreAligned16(wiener_buffer, d1);
}

inline void WienerHorizontalTap7(const uint8_t* src, const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m128i coefficients,
                                 int16_t** const wiener_buffer) {
  __m128i filter[4];
  filter[0] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x0200));
  filter[1] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x0604));
  filter[2] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x0204));
  filter[3] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x8000));
  for (int y = height; y != 0; --y) {
    const __m128i s0 = LoadUnaligned16(src);
    __m128i ss[4];
    ss[0] = _mm_unpacklo_epi8(s0, s0);
    ss[1] = _mm_unpackhi_epi8(s0, s0);
    ptrdiff_t x = 0;
    do {
      const __m128i s1 = LoadUnaligned16(src + x + 16);
      ss[2] = _mm_unpacklo_epi8(s1, s1);
      ss[3] = _mm_unpackhi_epi8(s1, s1);
      WienerHorizontalTap7Kernel(ss + 0, filter, *wiener_buffer + x + 0);
      WienerHorizontalTap7Kernel(ss + 1, filter, *wiener_buffer + x + 8);
      ss[0] = ss[2];
      ss[1] = ss[3];
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap5(const uint8_t* src, const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m128i coefficients,
                                 int16_t** const wiener_buffer) {
  __m128i filter[3];
  filter[0] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x0402));
  filter[1] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x0406));
  filter[2] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x8002));
  for (int y = height; y != 0; --y) {
    const __m128i s0 = LoadUnaligned16(src);
    __m128i ss[4];
    ss[0] = _mm_unpacklo_epi8(s0, s0);
    ss[1] = _mm_unpackhi_epi8(s0, s0);
    ptrdiff_t x = 0;
    do {
      const __m128i s1 = LoadUnaligned16(src + x + 16);
      ss[2] = _mm_unpacklo_epi8(s1, s1);
      ss[3] = _mm_unpackhi_epi8(s1, s1);
      WienerHorizontalTap5Kernel(ss + 0, filter, *wiener_buffer + x + 0);
      WienerHorizontalTap5Kernel(ss + 1, filter, *wiener_buffer + x + 8);
      ss[0] = ss[2];
      ss[1] = ss[3];
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap3(const uint8_t* src, const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 const __m128i coefficients,
                                 int16_t** const wiener_buffer) {
  __m128i filter[2];
  filter[0] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x0604));
  filter[1] = _mm_shuffle_epi8(coefficients, _mm_set1_epi16(0x8004));
  for (int y = height; y != 0; --y) {
    const __m128i s0 = LoadUnaligned16(src);
    __m128i ss[4];
    ss[0] = _mm_unpacklo_epi8(s0, s0);
    ss[1] = _mm_unpackhi_epi8(s0, s0);
    ptrdiff_t x = 0;
    do {
      const __m128i s1 = LoadUnaligned16(src + x + 16);
      ss[2] = _mm_unpacklo_epi8(s1, s1);
      ss[3] = _mm_unpackhi_epi8(s1, s1);
      WienerHorizontalTap3Kernel(ss + 0, filter, *wiener_buffer + x + 0);
      WienerHorizontalTap3Kernel(ss + 1, filter, *wiener_buffer + x + 8);
      ss[0] = ss[2];
      ss[1] = ss[3];
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline void WienerHorizontalTap1(const uint8_t* src, const ptrdiff_t src_stride,
                                 const ptrdiff_t width, const int height,
                                 int16_t** const wiener_buffer) {
  for (int y = height; y != 0; --y) {
    ptrdiff_t x = 0;
    do {
      const __m128i s = LoadUnaligned16(src + x);
      const __m128i s0 = _mm_unpacklo_epi8(s, _mm_setzero_si128());
      const __m128i s1 = _mm_unpackhi_epi8(s, _mm_setzero_si128());
      const __m128i d0 = _mm_slli_epi16(s0, 4);
      const __m128i d1 = _mm_slli_epi16(s1, 4);
      StoreAligned16(*wiener_buffer + x + 0, d0);
      StoreAligned16(*wiener_buffer + x + 8, d1);
      x += 16;
    } while (x < width);
    src += src_stride;
    *wiener_buffer += width;
  }
}

inline __m128i WienerVertical7(const __m128i a[2], const __m128i filter[2]) {
  const __m128i round = _mm_set1_epi32(1 << (kInterRoundBitsVertical - 1));
  const __m128i madd0 = _mm_madd_epi16(a[0], filter[0]);
  const __m128i madd1 = _mm_madd_epi16(a[1], filter[1]);
  const __m128i sum0 = _mm_add_epi32(round, madd0);
  const __m128i sum1 = _mm_add_epi32(sum0, madd1);
  return _mm_srai_epi32(sum1, kInterRoundBitsVertical);
}

inline __m128i WienerVertical5(const __m128i a[2], const __m128i filter[2]) {
  const __m128i madd0 = _mm_madd_epi16(a[0], filter[0]);
  const __m128i madd1 = _mm_madd_epi16(a[1], filter[1]);
  const __m128i sum = _mm_add_epi32(madd0, madd1);
  return _mm_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m128i WienerVertical3(const __m128i a, const __m128i filter) {
  const __m128i round = _mm_set1_epi32(1 << (kInterRoundBitsVertical - 1));
  const __m128i madd = _mm_madd_epi16(a, filter);
  const __m128i sum = _mm_add_epi32(round, madd);
  return _mm_srai_epi32(sum, kInterRoundBitsVertical);
}

inline __m128i WienerVerticalFilter7(const __m128i a[7],
                                     const __m128i filter[2]) {
  __m128i b[2];
  const __m128i a06 = _mm_add_epi16(a[0], a[6]);
  const __m128i a15 = _mm_add_epi16(a[1], a[5]);
  const __m128i a24 = _mm_add_epi16(a[2], a[4]);
  b[0] = _mm_unpacklo_epi16(a06, a15);
  b[1] = _mm_unpacklo_epi16(a24, a[3]);
  const __m128i sum0 = WienerVertical7(b, filter);
  b[0] = _mm_unpackhi_epi16(a06, a15);
  b[1] = _mm_unpackhi_epi16(a24, a[3]);
  const __m128i sum1 = WienerVertical7(b, filter);
  return _mm_packs_epi32(sum0, sum1);
}

inline __m128i WienerVerticalFilter5(const __m128i a[5],
                                     const __m128i filter[2]) {
  const __m128i round = _mm_set1_epi16(1 << (kInterRoundBitsVertical - 1));
  __m128i b[2];
  const __m128i a04 = _mm_add_epi16(a[0], a[4]);
  const __m128i a13 = _mm_add_epi16(a[1], a[3]);
  b[0] = _mm_unpacklo_epi16(a04, a13);
  b[1] = _mm_unpacklo_epi16(a[2], round);
  const __m128i sum0 = WienerVertical5(b, filter);
  b[0] = _mm_unpackhi_epi16(a04, a13);
  b[1] = _mm_unpackhi_epi16(a[2], round);
  const __m128i sum1 = WienerVertical5(b, filter);
  return _mm_packs_epi32(sum0, sum1);
}

inline __m128i WienerVerticalFilter3(const __m128i a[3], const __m128i filter) {
  __m128i b;
  const __m128i a02 = _mm_add_epi16(a[0], a[2]);
  b = _mm_unpacklo_epi16(a02, a[1]);
  const __m128i sum0 = WienerVertical3(b, filter);
  b = _mm_unpackhi_epi16(a02, a[1]);
  const __m128i sum1 = WienerVertical3(b, filter);
  return _mm_packs_epi32(sum0, sum1);
}

inline __m128i WienerVerticalTap7Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m128i filter[2], __m128i a[7]) {
  a[0] = LoadAligned16(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned16(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned16(wiener_buffer + 2 * wiener_stride);
  a[3] = LoadAligned16(wiener_buffer + 3 * wiener_stride);
  a[4] = LoadAligned16(wiener_buffer + 4 * wiener_stride);
  a[5] = LoadAligned16(wiener_buffer + 5 * wiener_stride);
  a[6] = LoadAligned16(wiener_buffer + 6 * wiener_stride);
  return WienerVerticalFilter7(a, filter);
}

inline __m128i WienerVerticalTap5Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m128i filter[2], __m128i a[5]) {
  a[0] = LoadAligned16(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned16(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned16(wiener_buffer + 2 * wiener_stride);
  a[3] = LoadAligned16(wiener_buffer + 3 * wiener_stride);
  a[4] = LoadAligned16(wiener_buffer + 4 * wiener_stride);
  return WienerVerticalFilter5(a, filter);
}

inline __m128i WienerVerticalTap3Kernel(const int16_t* wiener_buffer,
                                        const ptrdiff_t wiener_stride,
                                        const __m128i filter, __m128i a[3]) {
  a[0] = LoadAligned16(wiener_buffer + 0 * wiener_stride);
  a[1] = LoadAligned16(wiener_buffer + 1 * wiener_stride);
  a[2] = LoadAligned16(wiener_buffer + 2 * wiener_stride);
  return WienerVerticalFilter3(a, filter);
}

inline void WienerVerticalTap7Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m128i filter[2], __m128i d[2]) {
  __m128i a[8];
  d[0] = WienerVerticalTap7Kernel(wiener_buffer, wiener_stride, filter, a);
  a[7] = LoadAligned16(wiener_buffer + 7 * wiener_stride);
  d[1] = WienerVerticalFilter7(a + 1, filter);
}

inline void WienerVerticalTap5Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m128i filter[2], __m128i d[2]) {
  __m128i a[6];
  d[0] = WienerVerticalTap5Kernel(wiener_buffer, wiener_stride, filter, a);
  a[5] = LoadAligned16(wiener_buffer + 5 * wiener_stride);
  d[1] = WienerVerticalFilter5(a + 1, filter);
}

inline void WienerVerticalTap3Kernel2(const int16_t* wiener_buffer,
                                      const ptrdiff_t wiener_stride,
                                      const __m128i filter, __m128i d[2]) {
  __m128i a[4];
  d[0] = WienerVerticalTap3Kernel(wiener_buffer, wiener_stride, filter, a);
  a[3] = LoadAligned16(wiener_buffer + 3 * wiener_stride);
  d[1] = WienerVerticalFilter3(a + 1, filter);
}

inline void WienerVerticalTap7(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[4], uint8_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m128i c = LoadLo8(coefficients);
  __m128i filter[2];
  filter[0] = _mm_shuffle_epi32(c, 0x0);
  filter[1] = _mm_shuffle_epi32(c, 0x55);
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m128i d[2][2];
      WienerVerticalTap7Kernel2(wiener_buffer + x + 0, width, filter, d[0]);
      WienerVerticalTap7Kernel2(wiener_buffer + x + 8, width, filter, d[1]);
      StoreAligned16(dst + x, _mm_packus_epi16(d[0][0], d[1][0]));
      StoreAligned16(dst + dst_stride + x, _mm_packus_epi16(d[0][1], d[1][1]));
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m128i a[7];
      const __m128i d0 =
          WienerVerticalTap7Kernel(wiener_buffer + x + 0, width, filter, a);
      const __m128i d1 =
          WienerVerticalTap7Kernel(wiener_buffer + x + 8, width, filter, a);
      StoreAligned16(dst + x, _mm_packus_epi16(d0, d1));
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap5(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[3], uint8_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m128i c = Load4(coefficients);
  __m128i filter[2];
  filter[0] = _mm_shuffle_epi32(c, 0);
  filter[1] =
      _mm_set1_epi32((1 << 16) | static_cast<uint16_t>(coefficients[2]));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m128i d[2][2];
      WienerVerticalTap5Kernel2(wiener_buffer + x + 0, width, filter, d[0]);
      WienerVerticalTap5Kernel2(wiener_buffer + x + 8, width, filter, d[1]);
      StoreAligned16(dst + x, _mm_packus_epi16(d[0][0], d[1][0]));
      StoreAligned16(dst + dst_stride + x, _mm_packus_epi16(d[0][1], d[1][1]));
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m128i a[5];
      const __m128i d0 =
          WienerVerticalTap5Kernel(wiener_buffer + x + 0, width, filter, a);
      const __m128i d1 =
          WienerVerticalTap5Kernel(wiener_buffer + x + 8, width, filter, a);
      StoreAligned16(dst + x, _mm_packus_epi16(d0, d1));
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap3(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               const int16_t coefficients[2], uint8_t* dst,
                               const ptrdiff_t dst_stride) {
  const __m128i filter =
      _mm_set1_epi32(*reinterpret_cast<const int32_t*>(coefficients));
  for (int y = height >> 1; y > 0; --y) {
    ptrdiff_t x = 0;
    do {
      __m128i d[2][2];
      WienerVerticalTap3Kernel2(wiener_buffer + x + 0, width, filter, d[0]);
      WienerVerticalTap3Kernel2(wiener_buffer + x + 8, width, filter, d[1]);
      StoreAligned16(dst + x, _mm_packus_epi16(d[0][0], d[1][0]));
      StoreAligned16(dst + dst_stride + x, _mm_packus_epi16(d[0][1], d[1][1]));
      x += 16;
    } while (x < width);
    dst += 2 * dst_stride;
    wiener_buffer += 2 * width;
  }

  if ((height & 1) != 0) {
    ptrdiff_t x = 0;
    do {
      __m128i a[3];
      const __m128i d0 =
          WienerVerticalTap3Kernel(wiener_buffer + x + 0, width, filter, a);
      const __m128i d1 =
          WienerVerticalTap3Kernel(wiener_buffer + x + 8, width, filter, a);
      StoreAligned16(dst + x, _mm_packus_epi16(d0, d1));
      x += 16;
    } while (x < width);
  }
}

inline void WienerVerticalTap1Kernel(const int16_t* const wiener_buffer,
                                     uint8_t* const dst) {
  const __m128i a0 = LoadAligned16(wiener_buffer + 0);
  const __m128i a1 = LoadAligned16(wiener_buffer + 8);
  const __m128i b0 = _mm_add_epi16(a0, _mm_set1_epi16(8));
  const __m128i b1 = _mm_add_epi16(a1, _mm_set1_epi16(8));
  const __m128i c0 = _mm_srai_epi16(b0, 4);
  const __m128i c1 = _mm_srai_epi16(b1, 4);
  const __m128i d = _mm_packus_epi16(c0, c1);
  StoreAligned16(dst, d);
}

inline void WienerVerticalTap1(const int16_t* wiener_buffer,
                               const ptrdiff_t width, const int height,
                               uint8_t* dst, const ptrdiff_t dst_stride) {
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
  const auto* const src = static_cast<const uint8_t*>(source);
  const auto* const top = static_cast<const uint8_t*>(top_border);
  const auto* const bottom = static_cast<const uint8_t*>(bottom_border);
  const __m128i c =
      LoadLo8(restoration_info.wiener_info.filter[WienerInfo::kHorizontal]);
  // In order to keep the horizontal pass intermediate values within 16 bits we
  // offset |filter[3]| by 128. The 128 offset will be added back in the loop.
  const __m128i coefficients_horizontal =
      _mm_sub_epi16(c, _mm_setr_epi16(0, 0, 0, 128, 0, 0, 0, 0));
  if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 0) {
    WienerHorizontalTap7(top + (2 - height_extra) * stride - 3, stride,
                         wiener_stride, height_extra, coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap7(src - 3, stride, wiener_stride, height,
                         coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap7(bottom - 3, stride, wiener_stride, height_extra,
                         coefficients_horizontal, &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 1) {
    WienerHorizontalTap5(top + (2 - height_extra) * stride - 2, stride,
                         wiener_stride, height_extra, coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap5(src - 2, stride, wiener_stride, height,
                         coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap5(bottom - 2, stride, wiener_stride, height_extra,
                         coefficients_horizontal, &wiener_buffer_horizontal);
  } else if (number_leading_zero_coefficients[WienerInfo::kHorizontal] == 2) {
    // The maximum over-reads happen here.
    WienerHorizontalTap3(top + (2 - height_extra) * stride - 1, stride,
                         wiener_stride, height_extra, coefficients_horizontal,
                         &wiener_buffer_horizontal);
    WienerHorizontalTap3(src - 1, stride, wiener_stride, height,
                         coefficients_horizontal, &wiener_buffer_horizontal);
    WienerHorizontalTap3(bottom - 1, stride, wiener_stride, height_extra,
                         coefficients_horizontal, &wiener_buffer_horizontal);
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
  auto* dst = static_cast<uint8_t*>(dest);
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

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
#if DSP_ENABLED_8BPP_AVX2(WienerFilter)
  dsp->loop_restorations[0] = WienerFilter_AVX2;
#endif
}

}  // namespace
}  // namespace low_bitdepth

void LoopRestorationInit_AVX2() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else  // !LIBGAV1_ENABLE_AVX2
namespace libgav1 {
namespace dsp {

void LoopRestorationInit_AVX2() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_AVX2
