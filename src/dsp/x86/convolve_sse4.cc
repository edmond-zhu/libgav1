// Copyright 2019 The libgav1 Authors
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

#include "src/dsp/convolve.h"
#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_SSE4_1
#include <smmintrin.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/dsp/x86/common_sse4.h"
#include "src/utils/common.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

// TODO(slavarnway): Move to common neon/sse4 file.
int GetNumTapsInFilter(const int filter_index) {
  if (filter_index < 2) {
    // Despite the names these only use 6 taps.
    // kInterpolationFilterEightTap
    // kInterpolationFilterEightTapSmooth
    return 6;
  }

  if (filter_index == 2) {
    // kInterpolationFilterEightTapSharp
    return 8;
  }

  if (filter_index == 3) {
    // kInterpolationFilterBilinear
    return 2;
  }

  assert(filter_index > 3);
  // For small sizes (width/height <= 4) the large filters are replaced with 4
  // tap options.
  // If the original filters were |kInterpolationFilterEightTap| or
  // |kInterpolationFilterEightTapSharp| then it becomes
  // |kInterpolationFilterSwitchable|.
  // If it was |kInterpolationFilterEightTapSmooth| then it becomes an unnamed 4
  // tap filter.
  return 4;
}

constexpr int kSubPixelMask = (1 << kSubPixelBits) - 1;
constexpr int kHorizontalOffset = 3;

// Multiply every entry in |src[]| by the corresponding entry in |taps[]| and
// sum. The filters in |taps[]| are pre-shifted by 1. This prevents the final
// sum from outranging int16_t.
template <int filter_index>
__m128i SumOnePassTaps(const __m128i* const src, const __m128i* const taps) {
  __m128i sum;
  if (filter_index < 2) {
    // 6 taps.
    const __m128i v_madd_21 = _mm_maddubs_epi16(src[0], taps[0]);  // k2k1
    const __m128i v_madd_43 = _mm_maddubs_epi16(src[1], taps[1]);  // k4k3
    const __m128i v_madd_65 = _mm_maddubs_epi16(src[2], taps[2]);  // k6k5
    sum = _mm_add_epi16(v_madd_21, v_madd_43);
    sum = _mm_add_epi16(sum, v_madd_65);
  } else if (filter_index == 2) {
    // 8 taps.
    const __m128i v_madd_10 = _mm_maddubs_epi16(src[0], taps[0]);  // k1k0
    const __m128i v_madd_32 = _mm_maddubs_epi16(src[1], taps[1]);  // k3k2
    const __m128i v_madd_54 = _mm_maddubs_epi16(src[2], taps[2]);  // k5k4
    const __m128i v_madd_76 = _mm_maddubs_epi16(src[3], taps[3]);  // k7k6
    const __m128i v_sum_3210 = _mm_add_epi16(v_madd_10, v_madd_32);
    const __m128i v_sum_7654 = _mm_add_epi16(v_madd_54, v_madd_76);
    sum = _mm_add_epi16(v_sum_7654, v_sum_3210);
  } else if (filter_index == 3) {
    // 2 taps.
    sum = _mm_maddubs_epi16(src[0], taps[0]);  // k4k3
  } else {
    // 4 taps.
    const __m128i v_madd_32 = _mm_maddubs_epi16(src[0], taps[0]);  // k3k2
    const __m128i v_madd_54 = _mm_maddubs_epi16(src[1], taps[1]);  // k5k4
    sum = _mm_add_epi16(v_madd_32, v_madd_54);
  }
  return sum;
}

template <int filter_index>
__m128i SumHorizontalTaps(const uint8_t* const src,
                          const __m128i* const v_tap) {
  __m128i v_src[4];
  const __m128i src_long = LoadUnaligned16(src);
  const __m128i src_long_dup_lo = _mm_unpacklo_epi8(src_long, src_long);
  const __m128i src_long_dup_hi = _mm_unpackhi_epi8(src_long, src_long);

  if (filter_index < 2) {
    // 6 taps.
    v_src[0] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 3);   // _21
    v_src[1] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 7);   // _43
    v_src[2] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 11);  // _65
  } else if (filter_index == 2) {
    // 8 taps.
    v_src[0] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 1);   // _10
    v_src[1] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 5);   // _32
    v_src[2] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 9);   // _54
    v_src[3] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 13);  // _76
  } else if (filter_index == 3) {
    // 2 taps.
    v_src[0] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 7);  // _43
  } else if (filter_index > 3) {
    // 4 taps.
    v_src[0] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 5);  // _32
    v_src[1] = _mm_alignr_epi8(src_long_dup_hi, src_long_dup_lo, 9);  // _54
  }
  const __m128i sum = SumOnePassTaps<filter_index>(v_src, v_tap);
  return sum;
}

template <int filter_index>
__m128i SimpleHorizontalTaps(const uint8_t* const src,
                             const __m128i* const v_tap) {
  __m128i sum = SumHorizontalTaps<filter_index>(src, v_tap);

  // Normally the Horizontal pass does the downshift in two passes:
  // kInterRoundBitsHorizontal - 1 and then (kFilterBits -
  // kInterRoundBitsHorizontal). Each one uses a rounding shift. Combining them
  // requires adding the rounding offset from the skipped shift.
  constexpr int first_shift_rounding_bit = 1 << (kInterRoundBitsHorizontal - 2);

  sum = _mm_add_epi16(sum, _mm_set1_epi16(first_shift_rounding_bit));
  sum = RightShiftWithRounding_S16(sum, kFilterBits - 1);
  return _mm_packus_epi16(sum, sum);
}

template <int filter_index>
__m128i HorizontalTaps8To16(const uint8_t* const src,
                            const __m128i* const v_tap) {
  const __m128i sum = SumHorizontalTaps<filter_index>(src, v_tap);

  return RightShiftWithRounding_S16(sum, kInterRoundBitsHorizontal - 1);
}

template <int filter_index>
__m128i SumHorizontalTaps2x2(const uint8_t* src, const ptrdiff_t src_stride,
                             const __m128i* const v_tap) {
  const __m128i input0 = LoadLo8(&src[2]);
  const __m128i input1 = LoadLo8(&src[2 + src_stride]);

  if (filter_index == 3) {
    // 03 04 04 05 05 06 06 07 ....
    const __m128i input0_dup =
        _mm_srli_si128(_mm_unpacklo_epi8(input0, input0), 3);
    // 13 14 14 15 15 16 16 17 ....
    const __m128i input1_dup =
        _mm_srli_si128(_mm_unpacklo_epi8(input1, input1), 3);
    const __m128i v_src_43 = _mm_unpacklo_epi64(input0_dup, input1_dup);
    const __m128i v_sum_43 = _mm_maddubs_epi16(v_src_43, v_tap[0]);  // k4k3
    return v_sum_43;
  }

  // 02 03 03 04 04 05 05 06 06 07 ....
  const __m128i input0_dup =
      _mm_srli_si128(_mm_unpacklo_epi8(input0, input0), 1);
  // 12 13 13 14 14 15 15 16 16 17 ....
  const __m128i input1_dup =
      _mm_srli_si128(_mm_unpacklo_epi8(input1, input1), 1);
  // 04 05 05 06 06 07 07 08 ...
  const __m128i input0_dup_54 = _mm_srli_si128(input0_dup, 4);
  // 14 15 15 16 16 17 17 18 ...
  const __m128i input1_dup_54 = _mm_srli_si128(input1_dup, 4);
  const __m128i v_src_32 = _mm_unpacklo_epi64(input0_dup, input1_dup);
  const __m128i v_src_54 = _mm_unpacklo_epi64(input0_dup_54, input1_dup_54);
  const __m128i v_madd_32 = _mm_maddubs_epi16(v_src_32, v_tap[0]);  // k3k2
  const __m128i v_madd_54 = _mm_maddubs_epi16(v_src_54, v_tap[1]);  // k5k4
  const __m128i v_sum_5432 = _mm_add_epi16(v_madd_54, v_madd_32);
  return v_sum_5432;
}

template <int filter_index>
__m128i SimpleHorizontalTaps2x2(const uint8_t* src, const ptrdiff_t src_stride,
                                const __m128i* const v_tap) {
  __m128i sum = SumHorizontalTaps2x2<filter_index>(src, src_stride, v_tap);

  // Normally the Horizontal pass does the downshift in two passes:
  // kInterRoundBitsHorizontal - 1 and then (kFilterBits -
  // kInterRoundBitsHorizontal). Each one uses a rounding shift. Combining them
  // requires adding the rounding offset from the skipped shift.
  constexpr int first_shift_rounding_bit = 1 << (kInterRoundBitsHorizontal - 2);

  sum = _mm_add_epi16(sum, _mm_set1_epi16(first_shift_rounding_bit));
  sum = RightShiftWithRounding_S16(sum, kFilterBits - 1);
  return _mm_packus_epi16(sum, sum);
}

template <int filter_index>
__m128i HorizontalTaps8To16_2x2(const uint8_t* src, const ptrdiff_t src_stride,
                                const __m128i* const v_tap) {
  const __m128i sum =
      SumHorizontalTaps2x2<filter_index>(src, src_stride, v_tap);

  return RightShiftWithRounding_S16(sum, kInterRoundBitsHorizontal - 1);
}

template <int num_taps, int step, int filter_index, bool is_2d = false,
          bool is_compound = false>
void FilterHorizontal(const uint8_t* src, const ptrdiff_t src_stride,
                      void* const dest, const ptrdiff_t pred_stride,
                      const int width, const int height,
                      const __m128i* const v_tap) {
  auto* dest8 = static_cast<uint8_t*>(dest);
  auto* dest16 = static_cast<uint16_t*>(dest);

  // 4 tap filters are never used when width > 4.
  if (num_taps != 4 && width > 4) {
    int y = 0;
    do {
      int x = 0;
      do {
        if (is_2d || is_compound) {
          const __m128i v_sum =
              HorizontalTaps8To16<filter_index>(&src[x], v_tap);
          if (is_2d) {
            StoreAligned16(&dest16[x], v_sum);
          } else {
            StoreUnaligned16(&dest16[x], v_sum);
          }
        } else {
          const __m128i result =
              SimpleHorizontalTaps<filter_index>(&src[x], v_tap);
          StoreLo8(&dest8[x], result);
        }
        x += step;
      } while (x < width);
      src += src_stride;
      dest8 += pred_stride;
      dest16 += pred_stride;
    } while (++y < height);
    return;
  }

  // Horizontal passes only needs to account for |num_taps| 2 and 4 when
  // |width| <= 4.
  assert(width <= 4);
  assert(num_taps <= 4);
  if (num_taps <= 4) {
    if (width == 4) {
      int y = 0;
      do {
        if (is_2d || is_compound) {
          const __m128i v_sum = HorizontalTaps8To16<filter_index>(src, v_tap);
          StoreLo8(dest16, v_sum);
        } else {
          const __m128i result = SimpleHorizontalTaps<filter_index>(src, v_tap);
          Store4(&dest8[0], result);
        }
        src += src_stride;
        dest8 += pred_stride;
        dest16 += pred_stride;
      } while (++y < height);
      return;
    }

    if (!is_compound) {
      int y = 0;
      do {
        if (is_2d) {
          const __m128i sum =
              HorizontalTaps8To16_2x2<filter_index>(src, src_stride, v_tap);
          Store4(&dest16[0], sum);
          dest16 += pred_stride;
          Store4(&dest16[0], _mm_srli_si128(sum, 8));
          dest16 += pred_stride;
        } else {
          const __m128i sum =
              SimpleHorizontalTaps2x2<filter_index>(src, src_stride, v_tap);
          Store2(dest8, sum);
          dest8 += pred_stride;
          Store2(dest8, _mm_srli_si128(sum, 4));
          dest8 += pred_stride;
        }

        src += src_stride << 1;
        y += 2;
      } while (y < height - 1);

      // The 2d filters have an odd |height| because the horizontal pass
      // generates context for the vertical pass.
      if (is_2d) {
        assert(height % 2 == 1);
        __m128i sum;
        const __m128i input = LoadLo8(&src[2]);
        if (filter_index == 3) {
          // 03 04 04 05 05 06 06 07 ....
          const __m128i v_src_43 =
              _mm_srli_si128(_mm_unpacklo_epi8(input, input), 3);
          sum = _mm_maddubs_epi16(v_src_43, v_tap[0]);  // k4k3
        } else {
          // 02 03 03 04 04 05 05 06 06 07 ....
          const __m128i v_src_32 =
              _mm_srli_si128(_mm_unpacklo_epi8(input, input), 1);
          // 04 05 05 06 06 07 07 08 ...
          const __m128i v_src_54 = _mm_srli_si128(v_src_32, 4);
          const __m128i v_madd_32 =
              _mm_maddubs_epi16(v_src_32, v_tap[0]);  // k3k2
          const __m128i v_madd_54 =
              _mm_maddubs_epi16(v_src_54, v_tap[1]);  // k5k4
          sum = _mm_add_epi16(v_madd_54, v_madd_32);
        }
        sum = RightShiftWithRounding_S16(sum, kInterRoundBitsHorizontal - 1);
        Store4(dest16, sum);
      }
    }
  }
}

template <int num_taps, bool is_2d_vertical = false>
LIBGAV1_ALWAYS_INLINE void SetupTaps(const __m128i* const filter,
                                     __m128i* v_tap) {
  if (num_taps == 8) {
    v_tap[0] = _mm_shufflelo_epi16(*filter, 0x0);   // k1k0
    v_tap[1] = _mm_shufflelo_epi16(*filter, 0x55);  // k3k2
    v_tap[2] = _mm_shufflelo_epi16(*filter, 0xaa);  // k5k4
    v_tap[3] = _mm_shufflelo_epi16(*filter, 0xff);  // k7k6
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
      v_tap[1] = _mm_cvtepi8_epi16(v_tap[1]);
      v_tap[2] = _mm_cvtepi8_epi16(v_tap[2]);
      v_tap[3] = _mm_cvtepi8_epi16(v_tap[3]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
      v_tap[1] = _mm_unpacklo_epi64(v_tap[1], v_tap[1]);
      v_tap[2] = _mm_unpacklo_epi64(v_tap[2], v_tap[2]);
      v_tap[3] = _mm_unpacklo_epi64(v_tap[3], v_tap[3]);
    }
  } else if (num_taps == 6) {
    const __m128i adjusted_filter = _mm_srli_si128(*filter, 1);
    v_tap[0] = _mm_shufflelo_epi16(adjusted_filter, 0x0);   // k2k1
    v_tap[1] = _mm_shufflelo_epi16(adjusted_filter, 0x55);  // k4k3
    v_tap[2] = _mm_shufflelo_epi16(adjusted_filter, 0xaa);  // k6k5
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
      v_tap[1] = _mm_cvtepi8_epi16(v_tap[1]);
      v_tap[2] = _mm_cvtepi8_epi16(v_tap[2]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
      v_tap[1] = _mm_unpacklo_epi64(v_tap[1], v_tap[1]);
      v_tap[2] = _mm_unpacklo_epi64(v_tap[2], v_tap[2]);
    }
  } else if (num_taps == 4) {
    v_tap[0] = _mm_shufflelo_epi16(*filter, 0x55);  // k3k2
    v_tap[1] = _mm_shufflelo_epi16(*filter, 0xaa);  // k5k4
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
      v_tap[1] = _mm_cvtepi8_epi16(v_tap[1]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
      v_tap[1] = _mm_unpacklo_epi64(v_tap[1], v_tap[1]);
    }
  } else {  // num_taps == 2
    const __m128i adjusted_filter = _mm_srli_si128(*filter, 1);
    v_tap[0] = _mm_shufflelo_epi16(adjusted_filter, 0x55);  // k4k3
    if (is_2d_vertical) {
      v_tap[0] = _mm_cvtepi8_epi16(v_tap[0]);
    } else {
      v_tap[0] = _mm_unpacklo_epi64(v_tap[0], v_tap[0]);
    }
  }
}

template <int num_taps, bool is_compound>
__m128i SimpleSum2DVerticalTaps(const __m128i* const src,
                                const __m128i* const taps) {
  __m128i sum_lo = _mm_madd_epi16(_mm_unpacklo_epi16(src[0], src[1]), taps[0]);
  __m128i sum_hi = _mm_madd_epi16(_mm_unpackhi_epi16(src[0], src[1]), taps[0]);
  if (num_taps >= 4) {
    __m128i madd_lo =
        _mm_madd_epi16(_mm_unpacklo_epi16(src[2], src[3]), taps[1]);
    __m128i madd_hi =
        _mm_madd_epi16(_mm_unpackhi_epi16(src[2], src[3]), taps[1]);
    sum_lo = _mm_add_epi32(sum_lo, madd_lo);
    sum_hi = _mm_add_epi32(sum_hi, madd_hi);
    if (num_taps >= 6) {
      madd_lo = _mm_madd_epi16(_mm_unpacklo_epi16(src[4], src[5]), taps[2]);
      madd_hi = _mm_madd_epi16(_mm_unpackhi_epi16(src[4], src[5]), taps[2]);
      sum_lo = _mm_add_epi32(sum_lo, madd_lo);
      sum_hi = _mm_add_epi32(sum_hi, madd_hi);
      if (num_taps == 8) {
        madd_lo = _mm_madd_epi16(_mm_unpacklo_epi16(src[6], src[7]), taps[3]);
        madd_hi = _mm_madd_epi16(_mm_unpackhi_epi16(src[6], src[7]), taps[3]);
        sum_lo = _mm_add_epi32(sum_lo, madd_lo);
        sum_hi = _mm_add_epi32(sum_hi, madd_hi);
      }
    }
  }

  if (is_compound) {
    return _mm_packs_epi32(
        RightShiftWithRounding_S32(sum_lo, kInterRoundBitsCompoundVertical - 1),
        RightShiftWithRounding_S32(sum_hi,
                                   kInterRoundBitsCompoundVertical - 1));
  }

  return _mm_packs_epi32(
      RightShiftWithRounding_S32(sum_lo, kInterRoundBitsVertical - 1),
      RightShiftWithRounding_S32(sum_hi, kInterRoundBitsVertical - 1));
}

template <int num_taps, bool is_compound = false>
void Filter2DVertical(const uint16_t* src, void* const dst,
                      const ptrdiff_t dst_stride, const int width,
                      const int height, const __m128i* const taps) {
  assert(width >= 8);
  constexpr int next_row = num_taps - 1;
  // The Horizontal pass uses |width| as |stride| for the intermediate buffer.
  const ptrdiff_t src_stride = width;

  auto* dst8 = static_cast<uint8_t*>(dst);
  auto* dst16 = static_cast<uint16_t*>(dst);

  int x = 0;
  do {
    __m128i srcs[8];
    const uint16_t* src_x = src + x;
    srcs[0] = LoadAligned16(src_x);
    src_x += src_stride;
    if (num_taps >= 4) {
      srcs[1] = LoadAligned16(src_x);
      src_x += src_stride;
      srcs[2] = LoadAligned16(src_x);
      src_x += src_stride;
      if (num_taps >= 6) {
        srcs[3] = LoadAligned16(src_x);
        src_x += src_stride;
        srcs[4] = LoadAligned16(src_x);
        src_x += src_stride;
        if (num_taps == 8) {
          srcs[5] = LoadAligned16(src_x);
          src_x += src_stride;
          srcs[6] = LoadAligned16(src_x);
          src_x += src_stride;
        }
      }
    }

    int y = 0;
    do {
      srcs[next_row] = LoadAligned16(src_x);
      src_x += src_stride;

      const __m128i sum =
          SimpleSum2DVerticalTaps<num_taps, is_compound>(srcs, taps);
      if (is_compound) {
        StoreUnaligned16(dst16 + x + y * dst_stride, sum);
      } else {
        StoreLo8(dst8 + x + y * dst_stride, _mm_packus_epi16(sum, sum));
      }

      srcs[0] = srcs[1];
      if (num_taps >= 4) {
        srcs[1] = srcs[2];
        srcs[2] = srcs[3];
        if (num_taps >= 6) {
          srcs[3] = srcs[4];
          srcs[4] = srcs[5];
          if (num_taps == 8) {
            srcs[5] = srcs[6];
            srcs[6] = srcs[7];
          }
        }
      }
    } while (++y < height);
    x += 8;
  } while (x < width);
}

// Take advantage of |src_stride| == |width| to process two rows at a time.
template <int num_taps, bool is_compound = false>
void Filter2DVertical4xH(const uint16_t* src, void* const dst,
                         const ptrdiff_t dst_stride, const int height,
                         const __m128i* const taps) {
  auto* dst8 = static_cast<uint8_t*>(dst);
  auto* dst16 = static_cast<uint16_t*>(dst);

  __m128i srcs[9];
  srcs[0] = LoadAligned16(src);
  src += 8;
  if (num_taps >= 4) {
    srcs[2] = LoadAligned16(src);
    src += 8;
    srcs[1] = _mm_unpacklo_epi64(_mm_srli_si128(srcs[0], 8), srcs[2]);
    if (num_taps >= 6) {
      srcs[4] = LoadAligned16(src);
      src += 8;
      srcs[3] = _mm_unpacklo_epi64(_mm_srli_si128(srcs[2], 8), srcs[4]);
      if (num_taps == 8) {
        srcs[6] = LoadAligned16(src);
        src += 8;
        srcs[5] = _mm_unpacklo_epi64(_mm_srli_si128(srcs[4], 8), srcs[6]);
      }
    }
  }

  int y = 0;
  do {
    srcs[num_taps] = LoadAligned16(src);
    src += 8;
    srcs[num_taps - 1] = _mm_unpacklo_epi64(
        _mm_srli_si128(srcs[num_taps - 2], 8), srcs[num_taps]);

    const __m128i sum =
        SimpleSum2DVerticalTaps<num_taps, is_compound>(srcs, taps);
    if (is_compound) {
      StoreUnaligned16(dst16, sum);
      dst16 += 4 << 1;
    } else {
      const __m128i results = _mm_packus_epi16(sum, sum);
      Store4(dst8, results);
      dst8 += dst_stride;
      Store4(dst8, _mm_srli_si128(results, 4));
      dst8 += dst_stride;
    }

    srcs[0] = srcs[2];
    if (num_taps >= 4) {
      srcs[1] = srcs[3];
      srcs[2] = srcs[4];
      if (num_taps >= 6) {
        srcs[3] = srcs[5];
        srcs[4] = srcs[6];
        if (num_taps == 8) {
          srcs[5] = srcs[7];
          srcs[6] = srcs[8];
        }
      }
    }
    y += 2;
  } while (y < height);
}

// Take advantage of |src_stride| == |width| to process four rows at a time.
template <int num_taps>
void Filter2DVertical2xH(const uint16_t* src, void* const dst,
                         const ptrdiff_t dst_stride, const int height,
                         const __m128i* const taps) {
  constexpr int next_row = (num_taps < 6) ? 4 : 8;

  auto* dst8 = static_cast<uint8_t*>(dst);

  __m128i srcs[9];
  srcs[0] = LoadAligned16(src);
  src += 8;
  if (num_taps >= 6) {
    srcs[4] = LoadAligned16(src);
    src += 8;
    srcs[1] = _mm_alignr_epi8(srcs[4], srcs[0], 4);
    if (num_taps == 8) {
      srcs[2] = _mm_alignr_epi8(srcs[4], srcs[0], 8);
      srcs[3] = _mm_alignr_epi8(srcs[4], srcs[0], 12);
    }
  }

  int y = 0;
  do {
    srcs[next_row] = LoadAligned16(src);
    src += 8;
    if (num_taps == 2) {
      srcs[1] = _mm_alignr_epi8(srcs[4], srcs[0], 4);
    } else if (num_taps == 4) {
      srcs[1] = _mm_alignr_epi8(srcs[4], srcs[0], 4);
      srcs[2] = _mm_alignr_epi8(srcs[4], srcs[0], 8);
      srcs[3] = _mm_alignr_epi8(srcs[4], srcs[0], 12);
    } else if (num_taps == 6) {
      srcs[2] = _mm_alignr_epi8(srcs[4], srcs[0], 8);
      srcs[3] = _mm_alignr_epi8(srcs[4], srcs[0], 12);
      srcs[5] = _mm_alignr_epi8(srcs[8], srcs[4], 4);
    } else if (num_taps == 8) {
      srcs[5] = _mm_alignr_epi8(srcs[8], srcs[4], 4);
      srcs[6] = _mm_alignr_epi8(srcs[8], srcs[4], 8);
      srcs[7] = _mm_alignr_epi8(srcs[8], srcs[4], 12);
    }

    const __m128i sum =
        SimpleSum2DVerticalTaps<num_taps, /*is_compound=*/false>(srcs, taps);
    const __m128i results = _mm_packus_epi16(sum, sum);

    Store2(dst8, results);
    dst8 += dst_stride;
    Store2(dst8, _mm_srli_si128(results, 2));
    // When |height| <= 4 the taps are restricted to 2 and 4 tap variants.
    // Therefore we don't need to check this condition when |height| > 4.
    if (num_taps <= 4 && height == 2) return;
    dst8 += dst_stride;
    Store2(dst8, _mm_srli_si128(results, 4));
    dst8 += dst_stride;
    Store2(dst8, _mm_srli_si128(results, 6));
    dst8 += dst_stride;

    srcs[0] = srcs[4];
    if (num_taps == 6) {
      srcs[1] = srcs[5];
      srcs[4] = srcs[8];
    } else if (num_taps == 8) {
      srcs[1] = srcs[5];
      srcs[2] = srcs[6];
      srcs[3] = srcs[7];
      srcs[4] = srcs[8];
    }

    y += 4;
  } while (y < height);
}

template <bool is_2d = false, bool is_compound = false>
LIBGAV1_ALWAYS_INLINE void DoHorizontalPass(
    const uint8_t* const src, const ptrdiff_t src_stride, void* const dst,
    const ptrdiff_t dst_stride, const int width, const int height,
    const int subpixel, const int filter_index) {
  const int filter_id = (subpixel >> 6) & kSubPixelMask;
  assert(filter_id != 0);
  __m128i v_tap[4];
  const __m128i v_horizontal_filter =
      LoadLo8(kHalfSubPixelFilters[filter_index][filter_id]);

  if (filter_index == 2) {  // 8 tap.
    SetupTaps<8>(&v_horizontal_filter, v_tap);
    FilterHorizontal<8, 8, 2, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 1) {  // 6 tap.
    SetupTaps<6>(&v_horizontal_filter, v_tap);
    FilterHorizontal<6, 8, 1, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 0) {  // 6 tap.
    SetupTaps<6>(&v_horizontal_filter, v_tap);
    FilterHorizontal<6, 8, 0, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 4) {  // 4 tap.
    SetupTaps<4>(&v_horizontal_filter, v_tap);
    FilterHorizontal<4, 8, 4, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else if (filter_index == 5) {  // 4 tap.
    SetupTaps<4>(&v_horizontal_filter, v_tap);
    FilterHorizontal<4, 8, 5, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  } else {  // 2 tap.
    SetupTaps<2>(&v_horizontal_filter, v_tap);
    FilterHorizontal<2, 8, 3, is_2d, is_compound>(
        src, src_stride, dst, dst_stride, width, height, v_tap);
  }
}

void Convolve2D_SSE4_1(const void* const reference,
                       const ptrdiff_t reference_stride,
                       const int horizontal_filter_index,
                       const int vertical_filter_index, const int subpixel_x,
                       const int subpixel_y, const int width, const int height,
                       void* prediction, const ptrdiff_t pred_stride) {
  const int horiz_filter_index = GetFilterIndex(horizontal_filter_index, width);
  const int vert_filter_index = GetFilterIndex(vertical_filter_index, height);
  const int vertical_taps = GetNumTapsInFilter(vert_filter_index);

  // The output of the horizontal filter is guaranteed to fit in 16 bits.
  alignas(16) uint16_t
      intermediate_result[kMaxSuperBlockSizeInPixels *
                          (kMaxSuperBlockSizeInPixels + kSubPixelTaps - 1)];
  const int intermediate_height = height + vertical_taps - 1;

  const ptrdiff_t src_stride = reference_stride;
  const auto* src = static_cast<const uint8_t*>(reference) -
                    (vertical_taps / 2 - 1) * src_stride - kHorizontalOffset;

  DoHorizontalPass</*is_2d=*/true>(src, src_stride, intermediate_result, width,
                                   width, intermediate_height, subpixel_x,
                                   horiz_filter_index);

  // Vertical filter.
  auto* dest = static_cast<uint8_t*>(prediction);
  const ptrdiff_t dest_stride = pred_stride;
  const int filter_id = ((subpixel_y & 1023) >> 6) & kSubPixelMask;
  assert(filter_id != 0);

  __m128i taps[4];
  const __m128i v_filter =
      LoadLo8(kHalfSubPixelFilters[vert_filter_index][filter_id]);

  if (vertical_taps == 8) {
    SetupTaps<8, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 2) {
      Filter2DVertical2xH<8>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else if (width == 4) {
      Filter2DVertical4xH<8>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else {
      Filter2DVertical<8>(intermediate_result, dest, dest_stride, width, height,
                          taps);
    }
  } else if (vertical_taps == 6) {
    SetupTaps<6, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 2) {
      Filter2DVertical2xH<6>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else if (width == 4) {
      Filter2DVertical4xH<6>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else {
      Filter2DVertical<6>(intermediate_result, dest, dest_stride, width, height,
                          taps);
    }
  } else if (vertical_taps == 4) {
    SetupTaps<4, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 2) {
      Filter2DVertical2xH<4>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else if (width == 4) {
      Filter2DVertical4xH<4>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else {
      Filter2DVertical<4>(intermediate_result, dest, dest_stride, width, height,
                          taps);
    }
  } else {  // |vertical_taps| == 2
    SetupTaps<2, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 2) {
      Filter2DVertical2xH<2>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else if (width == 4) {
      Filter2DVertical4xH<2>(intermediate_result, dest, dest_stride, height,
                             taps);
    } else {
      Filter2DVertical<2>(intermediate_result, dest, dest_stride, width, height,
                          taps);
    }
  }
}

// The 1D compound shift is always |kInterRoundBitsHorizontal|, even for 1D
// Vertical calculations.
__m128i Compound1DShift(const __m128i sum) {
  return RightShiftWithRounding_S16(sum, kInterRoundBitsHorizontal - 1);
}

template <int filter_index>
__m128i SumVerticalTaps(const __m128i* const srcs, const __m128i* const v_tap) {
  __m128i v_src[4];

  if (filter_index < 2) {
    // 6 taps.
    v_src[0] = _mm_unpacklo_epi8(srcs[0], srcs[1]);
    v_src[1] = _mm_unpacklo_epi8(srcs[2], srcs[3]);
    v_src[2] = _mm_unpacklo_epi8(srcs[4], srcs[5]);
  } else if (filter_index == 2) {
    // 8 taps.
    v_src[0] = _mm_unpacklo_epi8(srcs[0], srcs[1]);
    v_src[1] = _mm_unpacklo_epi8(srcs[2], srcs[3]);
    v_src[2] = _mm_unpacklo_epi8(srcs[4], srcs[5]);
    v_src[3] = _mm_unpacklo_epi8(srcs[6], srcs[7]);
  } else if (filter_index == 3) {
    // 2 taps.
    v_src[0] = _mm_unpacklo_epi8(srcs[0], srcs[1]);
  } else if (filter_index > 3) {
    // 4 taps.
    v_src[0] = _mm_unpacklo_epi8(srcs[0], srcs[1]);
    v_src[1] = _mm_unpacklo_epi8(srcs[2], srcs[3]);
  }
  const __m128i sum = SumOnePassTaps<filter_index>(v_src, v_tap);
  return sum;
}

template <int filter_index, bool is_compound = false>
void FilterVertical(const uint8_t* src, const ptrdiff_t src_stride,
                    void* const dst, const ptrdiff_t dst_stride,
                    const int width, const int height,
                    const __m128i* const v_tap) {
  const int num_taps = GetNumTapsInFilter(filter_index);
  const int next_row = num_taps - 1;
  auto* dst8 = static_cast<uint8_t*>(dst);
  auto* dst16 = static_cast<uint16_t*>(dst);
  assert(width >= 8);

  int x = 0;
  do {
    const uint8_t* src_x = src + x;
    __m128i srcs[8];
    srcs[0] = LoadLo8(src_x);
    src_x += src_stride;
    if (num_taps >= 4) {
      srcs[1] = LoadLo8(src_x);
      src_x += src_stride;
      srcs[2] = LoadLo8(src_x);
      src_x += src_stride;
      if (num_taps >= 6) {
        srcs[3] = LoadLo8(src_x);
        src_x += src_stride;
        srcs[4] = LoadLo8(src_x);
        src_x += src_stride;
        if (num_taps == 8) {
          srcs[5] = LoadLo8(src_x);
          src_x += src_stride;
          srcs[6] = LoadLo8(src_x);
          src_x += src_stride;
        }
      }
    }

    int y = 0;
    do {
      srcs[next_row] = LoadLo8(src_x);
      src_x += src_stride;

      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      if (is_compound) {
        const __m128i results = Compound1DShift(sums);
        StoreUnaligned16(dst16 + x + y * dst_stride, results);
      } else {
        const __m128i results =
            RightShiftWithRounding_S16(sums, kFilterBits - 1);
        StoreLo8(dst8 + x + y * dst_stride, _mm_packus_epi16(results, results));
      }

      srcs[0] = srcs[1];
      if (num_taps >= 4) {
        srcs[1] = srcs[2];
        srcs[2] = srcs[3];
        if (num_taps >= 6) {
          srcs[3] = srcs[4];
          srcs[4] = srcs[5];
          if (num_taps == 8) {
            srcs[5] = srcs[6];
            srcs[6] = srcs[7];
          }
        }
      }
    } while (++y < height);
    x += 8;
  } while (x < width);
}

template <int filter_index, bool is_compound = false>
void FilterVertical4xH(const uint8_t* src, const ptrdiff_t src_stride,
                       void* const dst, const ptrdiff_t dst_stride,
                       const int height, const __m128i* const v_tap) {
  const int num_taps = GetNumTapsInFilter(filter_index);
  auto* dst8 = static_cast<uint8_t*>(dst);
  auto* dst16 = static_cast<uint16_t*>(dst);

  __m128i srcs[9];

  if (num_taps == 2) {
    srcs[2] = _mm_setzero_si128();
    // 00 01 02 03
    srcs[0] = Load4(src);
    src += src_stride;

    int y = 0;
    do {
      // 10 11 12 13
      const __m128i a = Load4(src);
      // 00 01 02 03 10 11 12 13
      srcs[0] = _mm_unpacklo_epi32(srcs[0], a);
      src += src_stride;
      // 20 21 22 23
      srcs[2] = Load4(src);
      src += src_stride;
      // 10 11 12 13 20 21 22 23
      srcs[1] = _mm_unpacklo_epi32(a, srcs[2]);

      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      if (is_compound) {
        const __m128i results = Compound1DShift(sums);
        StoreUnaligned16(dst16, results);
        dst16 += 4 << 1;
      } else {
        const __m128i results_16 =
            RightShiftWithRounding_S16(sums, kFilterBits - 1);
        const __m128i results = _mm_packus_epi16(results_16, results_16);
        Store4(dst8, results);
        dst8 += dst_stride;
        Store4(dst8, _mm_srli_si128(results, 4));
        dst8 += dst_stride;
      }

      srcs[0] = srcs[2];
      y += 2;
    } while (y < height);
  } else if (num_taps == 4) {
    srcs[4] = _mm_setzero_si128();
    // 00 01 02 03
    srcs[0] = Load4(src);
    src += src_stride;
    // 10 11 12 13
    const __m128i a = Load4(src);
    // 00 01 02 03 10 11 12 13
    srcs[0] = _mm_unpacklo_epi32(srcs[0], a);
    src += src_stride;
    // 20 21 22 23
    srcs[2] = Load4(src);
    src += src_stride;
    // 10 11 12 13 20 21 22 23
    srcs[1] = _mm_unpacklo_epi32(a, srcs[2]);

    int y = 0;
    do {
      // 30 31 32 33
      const __m128i b = Load4(src);
      // 20 21 22 23 30 31 32 33
      srcs[2] = _mm_unpacklo_epi32(srcs[2], b);
      src += src_stride;
      // 40 41 42 43
      srcs[4] = Load4(src);
      src += src_stride;
      // 30 31 32 33 40 41 42 43
      srcs[3] = _mm_unpacklo_epi32(b, srcs[4]);

      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      if (is_compound) {
        const __m128i results = Compound1DShift(sums);
        StoreUnaligned16(dst16, results);
        dst16 += 4 << 1;
      } else {
        const __m128i results_16 =
            RightShiftWithRounding_S16(sums, kFilterBits - 1);
        const __m128i results = _mm_packus_epi16(results_16, results_16);
        Store4(dst8, results);
        dst8 += dst_stride;
        Store4(dst8, _mm_srli_si128(results, 4));
        dst8 += dst_stride;
      }

      srcs[0] = srcs[2];
      srcs[1] = srcs[3];
      srcs[2] = srcs[4];
      y += 2;
    } while (y < height);
  } else if (num_taps == 6) {
    srcs[6] = _mm_setzero_si128();
    // 00 01 02 03
    srcs[0] = Load4(src);
    src += src_stride;
    // 10 11 12 13
    const __m128i a = Load4(src);
    // 00 01 02 03 10 11 12 13
    srcs[0] = _mm_unpacklo_epi32(srcs[0], a);
    src += src_stride;
    // 20 21 22 23
    srcs[2] = Load4(src);
    src += src_stride;
    // 10 11 12 13 20 21 22 23
    srcs[1] = _mm_unpacklo_epi32(a, srcs[2]);
    // 30 31 32 33
    const __m128i b = Load4(src);
    // 20 21 22 23 30 31 32 33
    srcs[2] = _mm_unpacklo_epi32(srcs[2], b);
    src += src_stride;
    // 40 41 42 43
    srcs[4] = Load4(src);
    src += src_stride;
    // 30 31 32 33 40 41 42 43
    srcs[3] = _mm_unpacklo_epi32(b, srcs[4]);

    int y = 0;
    do {
      // 50 51 52 53
      const __m128i c = Load4(src);
      // 40 41 42 43 50 51 52 53
      srcs[4] = _mm_unpacklo_epi32(srcs[4], c);
      src += src_stride;
      // 60 61 62 63
      srcs[6] = Load4(src);
      src += src_stride;
      // 50 51 52 53 60 61 62 63
      srcs[5] = _mm_unpacklo_epi32(c, srcs[6]);

      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      if (is_compound) {
        const __m128i results = Compound1DShift(sums);
        StoreUnaligned16(dst16, results);
        dst16 += 4 << 1;
      } else {
        const __m128i results_16 =
            RightShiftWithRounding_S16(sums, kFilterBits - 1);
        const __m128i results = _mm_packus_epi16(results_16, results_16);
        Store4(dst8, results);
        dst8 += dst_stride;
        Store4(dst8, _mm_srli_si128(results, 4));
        dst8 += dst_stride;
      }

      srcs[0] = srcs[2];
      srcs[1] = srcs[3];
      srcs[2] = srcs[4];
      srcs[3] = srcs[5];
      srcs[4] = srcs[6];
      y += 2;
    } while (y < height);
  } else if (num_taps == 8) {
    srcs[8] = _mm_setzero_si128();
    // 00 01 02 03
    srcs[0] = Load4(src);
    src += src_stride;
    // 10 11 12 13
    const __m128i a = Load4(src);
    // 00 01 02 03 10 11 12 13
    srcs[0] = _mm_unpacklo_epi32(srcs[0], a);
    src += src_stride;
    // 20 21 22 23
    srcs[2] = Load4(src);
    src += src_stride;
    // 10 11 12 13 20 21 22 23
    srcs[1] = _mm_unpacklo_epi32(a, srcs[2]);
    // 30 31 32 33
    const __m128i b = Load4(src);
    // 20 21 22 23 30 31 32 33
    srcs[2] = _mm_unpacklo_epi32(srcs[2], b);
    src += src_stride;
    // 40 41 42 43
    srcs[4] = Load4(src);
    src += src_stride;
    // 30 31 32 33 40 41 42 43
    srcs[3] = _mm_unpacklo_epi32(b, srcs[4]);
    // 50 51 52 53
    const __m128i c = Load4(src);
    // 40 41 42 43 50 51 52 53
    srcs[4] = _mm_unpacklo_epi32(srcs[4], c);
    src += src_stride;
    // 60 61 62 63
    srcs[6] = Load4(src);
    src += src_stride;
    // 50 51 52 53 60 61 62 63
    srcs[5] = _mm_unpacklo_epi32(c, srcs[6]);

    int y = 0;
    do {
      // 70 71 72 73
      const __m128i d = Load4(src);
      // 60 61 62 63 70 71 72 73
      srcs[6] = _mm_unpacklo_epi32(srcs[6], d);
      src += src_stride;
      // 80 81 82 83
      srcs[8] = Load4(src);
      src += src_stride;
      // 70 71 72 73 80 81 82 83
      srcs[7] = _mm_unpacklo_epi32(d, srcs[8]);

      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      if (is_compound) {
        const __m128i results = Compound1DShift(sums);
        StoreUnaligned16(dst16, results);
        dst16 += 4 << 1;
      } else {
        const __m128i results_16 =
            RightShiftWithRounding_S16(sums, kFilterBits - 1);
        const __m128i results = _mm_packus_epi16(results_16, results_16);
        Store4(dst8, results);
        dst8 += dst_stride;
        Store4(dst8, _mm_srli_si128(results, 4));
        dst8 += dst_stride;
      }

      srcs[0] = srcs[2];
      srcs[1] = srcs[3];
      srcs[2] = srcs[4];
      srcs[3] = srcs[5];
      srcs[4] = srcs[6];
      srcs[5] = srcs[7];
      srcs[6] = srcs[8];
      y += 2;
    } while (y < height);
  }
}

template <int filter_index, bool negative_outside_taps = false>
void FilterVertical2xH(const uint8_t* src, const ptrdiff_t src_stride,
                       void* const dst, const ptrdiff_t dst_stride,
                       const int height, const __m128i* const v_tap) {
  const int num_taps = GetNumTapsInFilter(filter_index);
  auto* dst8 = static_cast<uint8_t*>(dst);

  __m128i srcs[9];

  if (num_taps == 2) {
    srcs[2] = _mm_setzero_si128();
    // 00 01
    srcs[0] = Load2(src);
    src += src_stride;

    int y = 0;
    do {
      // 00 01 10 11
      srcs[0] = Load2<1>(src, srcs[0]);
      src += src_stride;
      // 00 01 10 11 20 21
      srcs[0] = Load2<2>(src, srcs[0]);
      src += src_stride;
      // 00 01 10 11 20 21 30 31
      srcs[0] = Load2<3>(src, srcs[0]);
      src += src_stride;
      // 40 41
      srcs[2] = Load2<0>(src, srcs[2]);
      src += src_stride;
      // 00 01 10 11 20 21 30 31 40 41
      const __m128i srcs_0_2 = _mm_unpacklo_epi64(srcs[0], srcs[2]);
      // 10 11 20 21 30 31 40 41
      srcs[1] = _mm_srli_si128(srcs_0_2, 2);
      // This uses srcs[0]..srcs[1].
      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      const __m128i results_16 =
          RightShiftWithRounding_S16(sums, kFilterBits - 1);
      const __m128i results = _mm_packus_epi16(results_16, results_16);

      Store2(dst8, results);
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 2));
      if (height == 2) return;
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 4));
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 6));
      dst8 += dst_stride;

      srcs[0] = srcs[2];
      y += 4;
    } while (y < height);
  } else if (num_taps == 4) {
    srcs[4] = _mm_setzero_si128();

    // 00 01
    srcs[0] = Load2(src);
    src += src_stride;
    // 00 01 10 11
    srcs[0] = Load2<1>(src, srcs[0]);
    src += src_stride;
    // 00 01 10 11 20 21
    srcs[0] = Load2<2>(src, srcs[0]);
    src += src_stride;

    int y = 0;
    do {
      // 00 01 10 11 20 21 30 31
      srcs[0] = Load2<3>(src, srcs[0]);
      src += src_stride;
      // 40 41
      srcs[4] = Load2<0>(src, srcs[4]);
      src += src_stride;
      // 40 41 50 51
      srcs[4] = Load2<1>(src, srcs[4]);
      src += src_stride;
      // 40 41 50 51 60 61
      srcs[4] = Load2<2>(src, srcs[4]);
      src += src_stride;
      // 00 01 10 11 20 21 30 31 40 41 50 51 60 61
      const __m128i srcs_0_4 = _mm_unpacklo_epi64(srcs[0], srcs[4]);
      // 10 11 20 21 30 31 40 41
      srcs[1] = _mm_srli_si128(srcs_0_4, 2);
      // 20 21 30 31 40 41 50 51
      srcs[2] = _mm_srli_si128(srcs_0_4, 4);
      // 30 31 40 41 50 51 60 61
      srcs[3] = _mm_srli_si128(srcs_0_4, 6);

      // This uses srcs[0]..srcs[3].
      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      const __m128i results_16 =
          RightShiftWithRounding_S16(sums, kFilterBits - 1);
      const __m128i results = _mm_packus_epi16(results_16, results_16);

      Store2(dst8, results);
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 2));
      if (height == 2) return;
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 4));
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 6));
      dst8 += dst_stride;

      srcs[0] = srcs[4];
      y += 4;
    } while (y < height);
  } else if (num_taps == 6) {
    // During the vertical pass the number of taps is restricted when
    // |height| <= 4.
    assert(height > 4);
    srcs[8] = _mm_setzero_si128();

    // 00 01
    srcs[0] = Load2(src);
    src += src_stride;
    // 00 01 10 11
    srcs[0] = Load2<1>(src, srcs[0]);
    src += src_stride;
    // 00 01 10 11 20 21
    srcs[0] = Load2<2>(src, srcs[0]);
    src += src_stride;
    // 00 01 10 11 20 21 30 31
    srcs[0] = Load2<3>(src, srcs[0]);
    src += src_stride;
    // 40 41
    srcs[4] = Load2(src);
    src += src_stride;
    // 00 01 10 11 20 21 30 31 40 41 50 51 60 61
    const __m128i srcs_0_4x = _mm_unpacklo_epi64(srcs[0], srcs[4]);
    // 10 11 20 21 30 31 40 41
    srcs[1] = _mm_srli_si128(srcs_0_4x, 2);

    int y = 0;
    do {
      // 40 41 50 51
      srcs[4] = Load2<1>(src, srcs[4]);
      src += src_stride;
      // 40 41 50 51 60 61
      srcs[4] = Load2<2>(src, srcs[4]);
      src += src_stride;
      // 40 41 50 51 60 61 70 71
      srcs[4] = Load2<3>(src, srcs[4]);
      src += src_stride;
      // 80 81
      srcs[8] = Load2<0>(src, srcs[8]);
      src += src_stride;
      // 00 01 10 11 20 21 30 31 40 41 50 51 60 61
      const __m128i srcs_0_4 = _mm_unpacklo_epi64(srcs[0], srcs[4]);
      // 20 21 30 31 40 41 50 51
      srcs[2] = _mm_srli_si128(srcs_0_4, 4);
      // 30 31 40 41 50 51 60 61
      srcs[3] = _mm_srli_si128(srcs_0_4, 6);
      const __m128i srcs_4_8 = _mm_unpacklo_epi64(srcs[4], srcs[8]);
      // 50 51 60 61 70 71 80 81
      srcs[5] = _mm_srli_si128(srcs_4_8, 2);

      // This uses srcs[0]..srcs[5].
      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      const __m128i results_16 =
          RightShiftWithRounding_S16(sums, kFilterBits - 1);
      const __m128i results = _mm_packus_epi16(results_16, results_16);

      Store2(dst8, results);
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 2));
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 4));
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 6));
      dst8 += dst_stride;

      srcs[0] = srcs[4];
      srcs[1] = srcs[5];
      srcs[4] = srcs[8];
      y += 4;
    } while (y < height);
  } else if (num_taps == 8) {
    // During the vertical pass the number of taps is restricted when
    // |height| <= 4.
    assert(height > 4);
    srcs[8] = _mm_setzero_si128();
    // 00 01
    srcs[0] = Load2(src);
    src += src_stride;
    // 00 01 10 11
    srcs[0] = Load2<1>(src, srcs[0]);
    src += src_stride;
    // 00 01 10 11 20 21
    srcs[0] = Load2<2>(src, srcs[0]);
    src += src_stride;
    // 00 01 10 11 20 21 30 31
    srcs[0] = Load2<3>(src, srcs[0]);
    src += src_stride;
    // 40 41
    srcs[4] = Load2(src);
    src += src_stride;
    // 40 41 50 51
    srcs[4] = Load2<1>(src, srcs[4]);
    src += src_stride;
    // 40 41 50 51 60 61
    srcs[4] = Load2<2>(src, srcs[4]);
    src += src_stride;

    // 00 01 10 11 20 21 30 31 40 41 50 51 60 61
    const __m128i srcs_0_4 = _mm_unpacklo_epi64(srcs[0], srcs[4]);
    // 10 11 20 21 30 31 40 41
    srcs[1] = _mm_srli_si128(srcs_0_4, 2);
    // 20 21 30 31 40 41 50 51
    srcs[2] = _mm_srli_si128(srcs_0_4, 4);
    // 30 31 40 41 50 51 60 61
    srcs[3] = _mm_srli_si128(srcs_0_4, 6);

    int y = 0;
    do {
      // 40 41 50 51 60 61 70 71
      srcs[4] = Load2<3>(src, srcs[4]);
      src += src_stride;
      // 80 81
      srcs[8] = Load2<0>(src, srcs[8]);
      src += src_stride;
      // 80 81 90 91
      srcs[8] = Load2<1>(src, srcs[8]);
      src += src_stride;
      // 80 81 90 91 a0 a1
      srcs[8] = Load2<2>(src, srcs[8]);
      src += src_stride;

      // 40 41 50 51 60 61 70 71 80 81 90 91 a0 a1
      const __m128i srcs_4_8 = _mm_unpacklo_epi64(srcs[4], srcs[8]);
      // 50 51 60 61 70 71 80 81
      srcs[5] = _mm_srli_si128(srcs_4_8, 2);
      // 60 61 70 71 80 81 90 91
      srcs[6] = _mm_srli_si128(srcs_4_8, 4);
      // 70 71 80 81 90 91 a0 a1
      srcs[7] = _mm_srli_si128(srcs_4_8, 6);

      // This uses srcs[0]..srcs[7].
      const __m128i sums = SumVerticalTaps<filter_index>(srcs, v_tap);
      const __m128i results_16 =
          RightShiftWithRounding_S16(sums, kFilterBits - 1);
      const __m128i results = _mm_packus_epi16(results_16, results_16);

      Store2(dst8, results);
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 2));
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 4));
      dst8 += dst_stride;
      Store2(dst8, _mm_srli_si128(results, 6));
      dst8 += dst_stride;

      srcs[0] = srcs[4];
      srcs[1] = srcs[5];
      srcs[2] = srcs[6];
      srcs[3] = srcs[7];
      srcs[4] = srcs[8];
      y += 4;
    } while (y < height);
  }
}

void ConvolveVertical_SSE4_1(const void* const reference,
                             const ptrdiff_t reference_stride,
                             const int /*horizontal_filter_index*/,
                             const int vertical_filter_index,
                             const int /*subpixel_x*/, const int subpixel_y,
                             const int width, const int height,
                             void* prediction, const ptrdiff_t pred_stride) {
  const int filter_index = GetFilterIndex(vertical_filter_index, height);
  const int vertical_taps = GetNumTapsInFilter(filter_index);
  const ptrdiff_t src_stride = reference_stride;
  const auto* src = static_cast<const uint8_t*>(reference) -
                    (vertical_taps / 2 - 1) * src_stride;
  auto* dest = static_cast<uint8_t*>(prediction);
  const ptrdiff_t dest_stride = pred_stride;
  const int filter_id = (subpixel_y >> 6) & kSubPixelMask;
  assert(filter_id != 0);

  __m128i taps[4];
  const __m128i v_filter =
      LoadLo8(kHalfSubPixelFilters[filter_index][filter_id]);

  if (filter_index < 2) {  // 6 tap.
    SetupTaps<6>(&v_filter, taps);
    if (width == 2) {
      FilterVertical2xH<0>(src, src_stride, dest, dest_stride, height, taps);
    } else if (width == 4) {
      FilterVertical4xH<0>(src, src_stride, dest, dest_stride, height, taps);
    } else {
      FilterVertical<0>(src, src_stride, dest, dest_stride, width, height,
                        taps);
    }
  } else if (filter_index == 2) {  // 8 tap.
    SetupTaps<8>(&v_filter, taps);
    if (width == 2) {
      FilterVertical2xH<2>(src, src_stride, dest, dest_stride, height, taps);
    } else if (width == 4) {
      FilterVertical4xH<2>(src, src_stride, dest, dest_stride, height, taps);
    } else {
      FilterVertical<2>(src, src_stride, dest, dest_stride, width, height,
                        taps);
    }
  } else if (filter_index == 3) {  // 2 tap.
    SetupTaps<2>(&v_filter, taps);
    if (width == 2) {
      FilterVertical2xH<3>(src, src_stride, dest, dest_stride, height, taps);
    } else if (width == 4) {
      FilterVertical4xH<3>(src, src_stride, dest, dest_stride, height, taps);
    } else {
      FilterVertical<3>(src, src_stride, dest, dest_stride, width, height,
                        taps);
    }
  } else if (filter_index == 4) {  // 4 tap.
    SetupTaps<4>(&v_filter, taps);
    if (width == 2) {
      FilterVertical2xH<4>(src, src_stride, dest, dest_stride, height, taps);
    } else if (width == 4) {
      FilterVertical4xH<4>(src, src_stride, dest, dest_stride, height, taps);
    } else {
      FilterVertical<4>(src, src_stride, dest, dest_stride, width, height,
                        taps);
    }
  } else {
    // TODO(slavarnway): Investigate adding |filter_index| == 1 special cases.
    // See convolve_neon.cc
    SetupTaps<4>(&v_filter, taps);

    if (width == 2) {
      FilterVertical2xH<5>(src, src_stride, dest, dest_stride, height, taps);
    } else if (width == 4) {
      FilterVertical4xH<5>(src, src_stride, dest, dest_stride, height, taps);
    } else {
      FilterVertical<5>(src, src_stride, dest, dest_stride, width, height,
                        taps);
    }
  }
}

void ConvolveCompoundCopy_SSE4(
    const void* const reference, const ptrdiff_t reference_stride,
    const int /*horizontal_filter_index*/, const int /*vertical_filter_index*/,
    const int /*subpixel_x*/, const int /*subpixel_y*/, const int width,
    const int height, void* prediction, const ptrdiff_t pred_stride) {
  const auto* src = static_cast<const uint8_t*>(reference);
  const ptrdiff_t src_stride = reference_stride;
  auto* dest = static_cast<uint16_t*>(prediction);
  constexpr int kRoundBitsVertical =
      kInterRoundBitsVertical - kInterRoundBitsCompoundVertical;
  if (width >= 16) {
    int y = height;
    do {
      int x = 0;
      do {
        const __m128i v_src = LoadUnaligned16(&src[x]);
        const __m128i v_src_ext_lo = _mm_cvtepu8_epi16(v_src);
        const __m128i v_src_ext_hi =
            _mm_cvtepu8_epi16(_mm_srli_si128(v_src, 8));
        const __m128i v_dest_lo =
            _mm_slli_epi16(v_src_ext_lo, kRoundBitsVertical);
        const __m128i v_dest_hi =
            _mm_slli_epi16(v_src_ext_hi, kRoundBitsVertical);
        // TODO(slavarnway): Investigate using aligned stores.
        StoreUnaligned16(&dest[x], v_dest_lo);
        StoreUnaligned16(&dest[x + 8], v_dest_hi);
        x += 16;
      } while (x < width);
      src += src_stride;
      dest += pred_stride;
    } while (--y != 0);
  } else if (width == 8) {
    int y = height;
    do {
      const __m128i v_src = LoadLo8(&src[0]);
      const __m128i v_src_ext = _mm_cvtepu8_epi16(v_src);
      const __m128i v_dest = _mm_slli_epi16(v_src_ext, kRoundBitsVertical);
      StoreUnaligned16(&dest[0], v_dest);
      src += src_stride;
      dest += pred_stride;
    } while (--y != 0);
  } else { /* width == 4 */
    int y = height;
    do {
      const __m128i v_src0 = Load4(&src[0]);
      const __m128i v_src1 = Load4(&src[src_stride]);
      const __m128i v_src = _mm_unpacklo_epi32(v_src0, v_src1);
      const __m128i v_src_ext = _mm_cvtepu8_epi16(v_src);
      const __m128i v_dest = _mm_slli_epi16(v_src_ext, kRoundBitsVertical);
      StoreLo8(&dest[0], v_dest);
      StoreHi8(&dest[pred_stride], v_dest);
      src += src_stride * 2;
      dest += pred_stride * 2;
      y -= 2;
    } while (y != 0);
  }
}

void ConvolveCompoundVertical_SSE4_1(
    const void* const reference, const ptrdiff_t reference_stride,
    const int /*horizontal_filter_index*/, const int vertical_filter_index,
    const int /*subpixel_x*/, const int subpixel_y, const int width,
    const int height, void* prediction, const ptrdiff_t /*pred_stride*/) {
  const int filter_index = GetFilterIndex(vertical_filter_index, height);
  const int vertical_taps = GetNumTapsInFilter(filter_index);
  const ptrdiff_t src_stride = reference_stride;
  const auto* src = static_cast<const uint8_t*>(reference) -
                    (vertical_taps / 2 - 1) * src_stride;
  auto* dest = static_cast<uint16_t*>(prediction);
  const int filter_id = (subpixel_y >> 6) & kSubPixelMask;
  assert(filter_id != 0);

  __m128i taps[4];
  const __m128i v_filter =
      LoadLo8(kHalfSubPixelFilters[filter_index][filter_id]);

  if (filter_index < 2) {  // 6 tap.
    SetupTaps<6>(&v_filter, taps);
    if (width == 4) {
      FilterVertical4xH<0, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps);
    } else {
      FilterVertical<0, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps);
    }
  } else if (filter_index == 2) {  // 8 tap.
    SetupTaps<8>(&v_filter, taps);

    if (width == 4) {
      FilterVertical4xH<2, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps);
    } else {
      FilterVertical<2, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps);
    }
  } else if (filter_index == 3) {  // 2 tap.
    SetupTaps<2>(&v_filter, taps);

    if (width == 4) {
      FilterVertical4xH<3, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps);
    } else {
      FilterVertical<3, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps);
    }
  } else if (filter_index == 4) {  // 4 tap.
    SetupTaps<4>(&v_filter, taps);

    if (width == 4) {
      FilterVertical4xH<4, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps);
    } else {
      FilterVertical<4, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps);
    }
  } else {
    SetupTaps<4>(&v_filter, taps);

    if (width == 4) {
      FilterVertical4xH<5, /*is_compound=*/true>(src, src_stride, dest, 4,
                                                 height, taps);
    } else {
      FilterVertical<5, /*is_compound=*/true>(src, src_stride, dest, width,
                                              width, height, taps);
    }
  }
}

void ConvolveHorizontal_SSE4_1(const void* const reference,
                               const ptrdiff_t reference_stride,
                               const int horizontal_filter_index,
                               const int /*vertical_filter_index*/,
                               const int subpixel_x, const int /*subpixel_y*/,
                               const int width, const int height,
                               void* prediction, const ptrdiff_t pred_stride) {
  const int filter_index = GetFilterIndex(horizontal_filter_index, width);
  // Set |src| to the outermost tap.
  const auto* src = static_cast<const uint8_t*>(reference) - kHorizontalOffset;
  auto* dest = static_cast<uint8_t*>(prediction);

  DoHorizontalPass(src, reference_stride, dest, pred_stride, width, height,
                   subpixel_x, filter_index);
}

void ConvolveCompoundHorizontal_SSE4_1(
    const void* const reference, const ptrdiff_t reference_stride,
    const int horizontal_filter_index, const int /*vertical_filter_index*/,
    const int subpixel_x, const int /*subpixel_y*/, const int width,
    const int height, void* prediction, const ptrdiff_t /*pred_stride*/) {
  const int filter_index = GetFilterIndex(horizontal_filter_index, width);
  const auto* src = static_cast<const uint8_t*>(reference) - kHorizontalOffset;
  auto* dest = static_cast<uint16_t*>(prediction);

  DoHorizontalPass</*is_2d=*/false, /*is_compound=*/true>(
      src, reference_stride, dest, width, width, height, subpixel_x,
      filter_index);
}

void ConvolveCompound2D_SSE4_1(
    const void* const reference, const ptrdiff_t reference_stride,
    const int horizontal_filter_index, const int vertical_filter_index,
    const int subpixel_x, const int subpixel_y, const int width,
    const int height, void* prediction, const ptrdiff_t /*pred_stride*/) {
  // The output of the horizontal filter, i.e. the intermediate_result, is
  // guaranteed to fit in int16_t.
  alignas(16) uint16_t
      intermediate_result[kMaxSuperBlockSizeInPixels *
                          (kMaxSuperBlockSizeInPixels + kSubPixelTaps - 1)];

  // Horizontal filter.
  // Filter types used for width <= 4 are different from those for width > 4.
  // When width > 4, the valid filter index range is always [0, 3].
  // When width <= 4, the valid filter index range is always [4, 5].
  // Similarly for height.
  const int horiz_filter_index = GetFilterIndex(horizontal_filter_index, width);
  const int vert_filter_index = GetFilterIndex(vertical_filter_index, height);
  const int vertical_taps = GetNumTapsInFilter(vert_filter_index);
  const int intermediate_height = height + vertical_taps - 1;
  const ptrdiff_t src_stride = reference_stride;
  const auto* const src = static_cast<const uint8_t*>(reference) -
                          (vertical_taps / 2 - 1) * src_stride -
                          kHorizontalOffset;

  DoHorizontalPass</*is_2d=*/true, /*is_compound=*/true>(
      src, src_stride, intermediate_result, width, width, intermediate_height,
      subpixel_x, horiz_filter_index);

  // Vertical filter.
  auto* dest = static_cast<uint16_t*>(prediction);
  const int filter_id = ((subpixel_y & 1023) >> 6) & kSubPixelMask;
  assert(filter_id != 0);

  const ptrdiff_t dest_stride = width;
  __m128i taps[4];
  const __m128i v_filter =
      LoadLo8(kHalfSubPixelFilters[vert_filter_index][filter_id]);

  if (vertical_taps == 8) {
    SetupTaps<8, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 4) {
      Filter2DVertical4xH<8, /*is_compound=*/true>(intermediate_result, dest,
                                                   dest_stride, height, taps);
    } else {
      Filter2DVertical<8, /*is_compound=*/true>(
          intermediate_result, dest, dest_stride, width, height, taps);
    }
  } else if (vertical_taps == 6) {
    SetupTaps<6, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 4) {
      Filter2DVertical4xH<6, /*is_compound=*/true>(intermediate_result, dest,
                                                   dest_stride, height, taps);
    } else {
      Filter2DVertical<6, /*is_compound=*/true>(
          intermediate_result, dest, dest_stride, width, height, taps);
    }
  } else if (vertical_taps == 4) {
    SetupTaps<4, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 4) {
      Filter2DVertical4xH<4, /*is_compound=*/true>(intermediate_result, dest,
                                                   dest_stride, height, taps);
    } else {
      Filter2DVertical<4, /*is_compound=*/true>(
          intermediate_result, dest, dest_stride, width, height, taps);
    }
  } else {  // |vertical_taps| == 2
    SetupTaps<2, /*is_2d_vertical=*/true>(&v_filter, taps);
    if (width == 4) {
      Filter2DVertical4xH<2, /*is_compound=*/true>(intermediate_result, dest,
                                                   dest_stride, height, taps);
    } else {
      Filter2DVertical<2, /*is_compound=*/true>(
          intermediate_result, dest, dest_stride, width, height, taps);
    }
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
  dsp->convolve[0][0][0][1] = ConvolveHorizontal_SSE4_1;
  dsp->convolve[0][0][1][0] = ConvolveVertical_SSE4_1;
  dsp->convolve[0][0][1][1] = Convolve2D_SSE4_1;

  dsp->convolve[0][1][0][0] = ConvolveCompoundCopy_SSE4;
  dsp->convolve[0][1][0][1] = ConvolveCompoundHorizontal_SSE4_1;
  dsp->convolve[0][1][1][0] = ConvolveCompoundVertical_SSE4_1;
  dsp->convolve[0][1][1][1] = ConvolveCompound2D_SSE4_1;
}

}  // namespace
}  // namespace low_bitdepth

void ConvolveInit_SSE4_1() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1

#else   // !LIBGAV1_ENABLE_SSE4_1
namespace libgav1 {
namespace dsp {

void ConvolveInit_SSE4_1() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_SSE4_1
