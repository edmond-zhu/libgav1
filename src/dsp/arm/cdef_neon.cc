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

#include "src/dsp/cdef.h"
#include "src/utils/cpu.h"

#if LIBGAV1_ENABLE_NEON

#include <arm_neon.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "src/dsp/arm/common_neon.h"
#include "src/dsp/constants.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {
namespace low_bitdepth {
namespace {

#include "src/dsp/cdef.inc"

// Expand |a| to int8x16_t, left shift it by |shift| and sum the low
// and high values with |b| and |c| respectively.
// Used to calculate |partial[0][i + j]| and |partial[4][7 + i - j]|. The input
// is |src[j]| and it is being added to |partial[]| based on the above indices.
template <int shift, bool is_partial4 = false>
void AddPartial0(uint8x8_t a, uint16x8_t* b, uint16x8_t* c) {
  // Allow Left/RightShift() to compile when |shift| is out of range.
  constexpr int safe_shift = (shift > 0) ? shift : 1;
  if (is_partial4) a = vrev64_u8(a);
  if (shift == 0) {
    *b = vaddw_u8(*b, a);
  } else {
    *b = vaddw_u8(*b, LeftShift<safe_shift * 8>(a));
    *c = vaddw_u8(*c, RightShift<(8 - safe_shift) * 8>(a));
  }
}

// |[i + j / 2]| effectively adds two values to the same index:
// partial[1][0 + 0 / 2] += src[j]
// partial[1][0 + 1 / 2] += src[j + 1]
// Pairwise add |a| to generate 4 values. Shift/extract as necessary to add
// these values to |b| and |c|.
//
// |partial[3]| is the same with one exception: The values are reversed.
// |i| |j| |3 + i - j / 2|
//  0   0   3
//  0   1   3
//  0   2   2
//  < ... >
//  0   7   0
//
//  1   0   4
//  < ... >
//  1   7   1
// Used to calculate |partial[1][i + j / 2]| and |partial[3][3 + i - j / 2]|.
template <int shift, bool is_partial3 = false>
void AddPartial1(uint8x8_t a, uint16x8_t* b, uint16x8_t* c) {
  // Allow vextq_u16() to compile when |shift| is out of range.
  constexpr int safe_shift = (shift > 0) ? shift : 1;
  const uint16x4_t zero4 = vdup_n_u16(0);
  const uint16x8_t zero8 = vdupq_n_u16(0);
  if (is_partial3) a = vrev64_u8(a);
  const uint16x4_t paired = vpaddl_u8(a);
  const uint16x8_t extended = vcombine_u16(paired, zero4);
  if (shift == 0) {
    *b = vaddq_u16(*b, extended);
  } else if (shift == 4) {
    *b = vaddq_u16(*b, vcombine_u16(zero4, paired));
  } else {
    const uint16x8_t shifted_b = vextq_u16(zero8, extended, 8 - safe_shift);
    *b = vaddq_u16(*b, shifted_b);
    if (shift > 4) {
      // Split |paired| between |b| and |c|.
      const uint16x8_t shifted_c = vextq_u16(extended, zero8, 8 - safe_shift);
      *c = vaddq_u16(*c, shifted_c);
    }
  }
}

// Simple add starting at [3] and stepping back every other row.
// Used to calculate |partial[5][3 - i / 2 + j]|.
template <int shift>
void AddPartial5(const uint8x8_t a, uint16x8_t* b, uint16x8_t* c) {
  // Allow Left/RightShift() to compile when |shift| is out of range.
  constexpr int safe_shift = (shift < 6) ? shift : 1;
  if (shift > 5) {
    *b = vaddw_u8(*b, a);
  } else {
    *b = vaddw_u8(*b, LeftShift<(3 - (safe_shift / 2)) * 8>(a));
    *c = vaddw_u8(*c, RightShift<(5 + (safe_shift / 2)) * 8>(a));
  }
}

// Simple add.
// Used to calculate |partial[6][j]|
void AddPartial6(const uint8x8_t a, uint16x8_t* b) { *b = vaddw_u8(*b, a); }

// Simple add starting at [0] and stepping forward every other row.
// Used to calculate |partial[7][i / 2 + j]|.
template <int shift>
void AddPartial7(const uint8x8_t a, uint16x8_t* b, uint16x8_t* c) {
  // Allow Left/RightShift() to compile when |shift| is out of range.
  constexpr int safe_shift = (shift > 1) ? shift : 2;
  if (shift < 2) {
    *b = vaddw_u8(*b, a);
  } else {
    *b = vaddw_u8(*b, LeftShift<(safe_shift / 2) * 8>(a));
    *c = vaddw_u8(*c, RightShift<(8 - (safe_shift / 2)) * 8>(a));
  }
}

template <int value>
void AddPartial(uint8x8_t source, uint16x8_t dest_lo[8], uint16x8_t dest_hi[8],
                uint16_t dest_2[8]) {
  AddPartial0<value>(source, &dest_lo[0], &dest_hi[0]);
  AddPartial1<value>(source, &dest_lo[1], &dest_hi[1]);
  dest_2[value] = SumVector(source);
  AddPartial1<value, /*is_partial3=*/true>(source, &dest_lo[3], &dest_hi[3]);
  AddPartial0<value, /*is_partial4=*/true>(source, &dest_lo[4], &dest_hi[4]);
  AddPartial5<value>(source, &dest_lo[5], &dest_hi[5]);
  AddPartial6(source, &dest_lo[6]);
  AddPartial7<value>(source, &dest_lo[7], &dest_hi[7]);
}

uint32x4_t Square(uint16x4_t a) { return vmull_u16(a, a); }

uint32x4_t SquareAccumulate(uint32x4_t a, uint16x4_t b) {
  return vmlal_u16(a, b, b);
}

// |cost[0]| and |cost[4]| square the input and sum with the corresponding
// element from the other end of the vector:
// |kCdefDivisionTable[]| element:
// cost[0] += (Square(partial[0][i]) + Square(partial[0][14 - i])) *
//             kCdefDivisionTable[i + 1];
// cost[0] += Square(partial[0][7]) * kCdefDivisionTable[8];
// Because everything is being summed into a single value the distributive
// property allows us to mirror the division table and accumulate once.
uint32_t Cost0Or4(const uint16x8_t a, const uint16x8_t b,
                  const uint32x4_t division_table[4]) {
  uint32x4_t c = vmulq_u32(Square(vget_low_u16(a)), division_table[0]);
  c = vmlaq_u32(c, Square(vget_high_u16(a)), division_table[1]);
  c = vmlaq_u32(c, Square(vget_low_u16(b)), division_table[2]);
  c = vmlaq_u32(c, Square(vget_high_u16(b)), division_table[3]);
  return SumVector(c);
}

// |cost[2]| and |cost[6]| square the input and accumulate:
// cost[2] += Square(partial[2][i])
uint32_t SquareAccumulate(const uint16x8_t a) {
  uint32x4_t c = Square(vget_low_u16(a));
  c = SquareAccumulate(c, vget_high_u16(a));
  c = vmulq_n_u32(c, kCdefDivisionTable[7]);
  return SumVector(c);
}

uint32_t CostOdd(const uint16x8_t a, const uint16x8_t b, const uint32x4_t mask,
                 const uint32x4_t division_table[2]) {
  // Remove elements 0-2.
  uint32x4_t c = vandq_u32(mask, Square(vget_low_u16(a)));
  c = vaddq_u32(c, Square(vget_high_u16(a)));
  c = vmulq_n_u32(c, kCdefDivisionTable[7]);

  c = vmlaq_u32(c, Square(vget_low_u16(a)), division_table[0]);
  c = vmlaq_u32(c, Square(vget_low_u16(b)), division_table[1]);
  return SumVector(c);
}

void CdefDirection_NEON(const void* const source, ptrdiff_t stride,
                        int* const direction, int* const variance) {
  assert(direction != nullptr);
  assert(variance != nullptr);
  const auto* src = static_cast<const uint8_t*>(source);
  uint32_t cost[8];
  uint16x8_t partial_lo[8], partial_hi[8];
  uint16_t partial_2[8];

  for (int i = 0; i < 8; ++i) {
    partial_lo[i] = partial_hi[i] = vdupq_n_u16(0);
  }

  AddPartial<0>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<1>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<2>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<3>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<4>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<5>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<6>(vld1_u8(src), partial_lo, partial_hi, partial_2);
  src += stride;
  AddPartial<7>(vld1_u8(src), partial_lo, partial_hi, partial_2);

  partial_lo[2] = vld1q_u16(partial_2);

  cost[2] = SquareAccumulate(partial_lo[2]);
  cost[6] = SquareAccumulate(partial_lo[6]);

  const uint32x4_t division_table[4] = {
      vld1q_u32(kCdefDivisionTable), vld1q_u32(kCdefDivisionTable + 4),
      vld1q_u32(kCdefDivisionTable + 8), vld1q_u32(kCdefDivisionTable + 12)};

  cost[0] = Cost0Or4(partial_lo[0], partial_hi[0], division_table);
  cost[4] = Cost0Or4(partial_lo[4], partial_hi[4], division_table);

  const uint32x4_t division_table_odd[2] = {
      vld1q_u32(kCdefDivisionTableOdd), vld1q_u32(kCdefDivisionTableOdd + 4)};

  const uint32x4_t element_3_mask = {0, 0, 0, static_cast<uint32_t>(-1)};

  cost[1] =
      CostOdd(partial_lo[1], partial_hi[1], element_3_mask, division_table_odd);
  cost[3] =
      CostOdd(partial_lo[3], partial_hi[3], element_3_mask, division_table_odd);
  cost[5] =
      CostOdd(partial_lo[5], partial_hi[5], element_3_mask, division_table_odd);
  cost[7] =
      CostOdd(partial_lo[7], partial_hi[7], element_3_mask, division_table_odd);

  uint32_t best_cost = 0;
  *direction = 0;
  for (int i = 0; i < 8; ++i) {
    if (cost[i] > best_cost) {
      best_cost = cost[i];
      *direction = i;
    }
  }
  *variance = (best_cost - cost[(*direction + 4) & 7]) >> 10;
}

// -------------------------------------------------------------------------
// CdefFilter

// Load 4 vectors based on the given |direction|.
void LoadDirection(const uint16_t* const src, const ptrdiff_t stride,
                   uint16x8_t* output, const int direction) {
  // Each |direction| describes a different set of source values. Expand this
  // set by negating each set. For |direction| == 0 this gives a diagonal line
  // from top right to bottom left. The first value is y, the second x. Negative
  // y values move up.
  //    a       b         c       d
  // {-1, 1}, {1, -1}, {-2, 2}, {2, -2}
  //         c
  //       a
  //     0
  //   b
  // d
  const int y_0 = kCdefDirections[direction][0][0];
  const int x_0 = kCdefDirections[direction][0][1];
  const int y_1 = kCdefDirections[direction][1][0];
  const int x_1 = kCdefDirections[direction][1][1];
  output[0] = vld1q_u16(src + y_0 * stride + x_0);
  output[1] = vld1q_u16(src - y_0 * stride - x_0);
  output[2] = vld1q_u16(src + y_1 * stride + x_1);
  output[3] = vld1q_u16(src - y_1 * stride - x_1);
}

// Load 4 vectors based on the given |direction|. Use when |block_width| == 4 to
// do 2 rows at a time.
void LoadDirection4(const uint16_t* const src, const ptrdiff_t stride,
                    uint16x8_t* output, const int direction) {
  const int y_0 = kCdefDirections[direction][0][0];
  const int x_0 = kCdefDirections[direction][0][1];
  const int y_1 = kCdefDirections[direction][1][0];
  const int x_1 = kCdefDirections[direction][1][1];
  output[0] = vcombine_u16(vld1_u16(src + y_0 * stride + x_0),
                           vld1_u16(src + y_0 * stride + stride + x_0));
  output[1] = vcombine_u16(vld1_u16(src - y_0 * stride - x_0),
                           vld1_u16(src - y_0 * stride + stride - x_0));
  output[2] = vcombine_u16(vld1_u16(src + y_1 * stride + x_1),
                           vld1_u16(src + y_1 * stride + stride + x_1));
  output[3] = vcombine_u16(vld1_u16(src - y_1 * stride - x_1),
                           vld1_u16(src - y_1 * stride + stride - x_1));
}

int16x8_t Constrain(const uint16x8_t pixel, const uint16x8_t reference,
                    const uint16x8_t threshold, const int16x8_t damping) {
  // If reference > pixel, the difference will be negative, so covert to 0 or
  // -1.
  const uint16x8_t sign = vcgtq_u16(reference, pixel);
  const uint16x8_t abs_diff = vabdq_u16(pixel, reference);
  const uint16x8_t shifted_diff = vshlq_u16(abs_diff, damping);
  // For bitdepth == 8, the threshold range is [0, 15] and the damping range is
  // [3, 6]. If pixel == kCdefLargeValue(0x4000), shifted_diff will always be
  // larger than threshold. Subtract using saturation will return 0 when pixel
  // == kCdefLargeValue.
  static_assert(kCdefLargeValue == 0x4000, "Invalid kCdefLargeValue");
  const uint16x8_t thresh_minus_shifted_diff =
      vqsubq_u16(threshold, shifted_diff);
  const uint16x8_t clamp_abs_diff =
      vminq_u16(thresh_minus_shifted_diff, abs_diff);
  // Restore the sign.
  return vreinterpretq_s16_u16(
      vsubq_u16(veorq_u16(clamp_abs_diff, sign), sign));
}

template <int width, bool enable_primary = true, bool enable_secondary = true>
void CdefFilter_NEON(const uint16_t* src, const ptrdiff_t src_stride,
                     const int height, const int primary_strength,
                     const int secondary_strength, const int damping,
                     const int direction, void* dest,
                     const ptrdiff_t dst_stride) {
  static_assert(width == 8 || width == 4, "");
  static_assert(enable_primary || enable_secondary, "");
  constexpr bool clipping_required = enable_primary && enable_secondary;
  auto* dst = static_cast<uint8_t*>(dest);
  const uint16x8_t cdef_large_value_mask =
      vdupq_n_u16(static_cast<uint16_t>(~kCdefLargeValue));
  const uint16x8_t primary_threshold = vdupq_n_u16(primary_strength);
  const uint16x8_t secondary_threshold = vdupq_n_u16(secondary_strength);

  int16x8_t primary_damping_shift, secondary_damping_shift;

  // FloorLog2() requires input to be > 0.
  // 8-bit damping range: Y: [3, 6], UV: [2, 5].
  if (enable_primary) {
    // primary_strength: [0, 15] -> FloorLog2: [0, 3] so a clamp is necessary
    // for UV filtering.
    primary_damping_shift =
        vdupq_n_s16(-std::max(0, damping - FloorLog2(primary_strength)));
  }
  if (enable_secondary) {
    // secondary_strength: [0, 4] -> FloorLog2: [0, 2] so no clamp to 0 is
    // necessary.
    assert(damping - FloorLog2(secondary_strength) >= 0);
    secondary_damping_shift =
        vdupq_n_s16(-(damping - FloorLog2(secondary_strength)));
  }

  const int primary_tap_0 = kCdefPrimaryTaps[primary_strength & 1][0];
  const int primary_tap_1 = kCdefPrimaryTaps[primary_strength & 1][1];

  int y = height;
  do {
    uint16x8_t pixel;
    if (width == 8) {
      pixel = vld1q_u16(src);
    } else {
      pixel = vcombine_u16(vld1_u16(src), vld1_u16(src + src_stride));
    }

    uint16x8_t min = pixel;
    uint16x8_t max = pixel;
    int16x8_t sum;

    if (enable_primary) {
      // Primary |direction|.
      uint16x8_t primary_val[4];
      if (width == 8) {
        LoadDirection(src, src_stride, primary_val, direction);
      } else {
        LoadDirection4(src, src_stride, primary_val, direction);
      }

      if (clipping_required) {
        min = vminq_u16(min, primary_val[0]);
        min = vminq_u16(min, primary_val[1]);
        min = vminq_u16(min, primary_val[2]);
        min = vminq_u16(min, primary_val[3]);

        // The source is 16 bits, however, we only really care about the lower
        // 8 bits.  The upper 8 bits contain the "large" flag.  After the final
        // primary max has been calculated, zero out the upper 8 bits.  Use this
        // to find the "16 bit" max.
        const uint8x16_t max_p01 =
            vmaxq_u8(vreinterpretq_u8_u16(primary_val[0]),
                     vreinterpretq_u8_u16(primary_val[1]));
        const uint8x16_t max_p23 =
            vmaxq_u8(vreinterpretq_u8_u16(primary_val[2]),
                     vreinterpretq_u8_u16(primary_val[3]));
        const uint16x8_t max_p =
            vreinterpretq_u16_u8(vmaxq_u8(max_p01, max_p23));
        max = vmaxq_u16(max, vandq_u16(max_p, cdef_large_value_mask));
      }

      sum = Constrain(primary_val[0], pixel, primary_threshold,
                      primary_damping_shift);
      sum = vmulq_n_s16(sum, primary_tap_0);
      sum = vmlaq_n_s16(sum,
                        Constrain(primary_val[1], pixel, primary_threshold,
                                  primary_damping_shift),
                        primary_tap_0);
      sum = vmlaq_n_s16(sum,
                        Constrain(primary_val[2], pixel, primary_threshold,
                                  primary_damping_shift),
                        primary_tap_1);
      sum = vmlaq_n_s16(sum,
                        Constrain(primary_val[3], pixel, primary_threshold,
                                  primary_damping_shift),
                        primary_tap_1);
    } else {
      sum = vdupq_n_s16(0);
    }

    if (enable_secondary) {
      // Secondary |direction| values (+/- 2). Clamp |direction|.
      uint16x8_t secondary_val[8];
      if (width == 8) {
        LoadDirection(src, src_stride, secondary_val, direction + 2);
        LoadDirection(src, src_stride, secondary_val + 4, direction - 2);
      } else {
        LoadDirection4(src, src_stride, secondary_val, direction + 2);
        LoadDirection4(src, src_stride, secondary_val + 4, direction - 2);
      }

      if (clipping_required) {
        min = vminq_u16(min, secondary_val[0]);
        min = vminq_u16(min, secondary_val[1]);
        min = vminq_u16(min, secondary_val[2]);
        min = vminq_u16(min, secondary_val[3]);
        min = vminq_u16(min, secondary_val[4]);
        min = vminq_u16(min, secondary_val[5]);
        min = vminq_u16(min, secondary_val[6]);
        min = vminq_u16(min, secondary_val[7]);

        const uint8x16_t max_s01 =
            vmaxq_u8(vreinterpretq_u8_u16(secondary_val[0]),
                     vreinterpretq_u8_u16(secondary_val[1]));
        const uint8x16_t max_s23 =
            vmaxq_u8(vreinterpretq_u8_u16(secondary_val[2]),
                     vreinterpretq_u8_u16(secondary_val[3]));
        const uint8x16_t max_s45 =
            vmaxq_u8(vreinterpretq_u8_u16(secondary_val[4]),
                     vreinterpretq_u8_u16(secondary_val[5]));
        const uint8x16_t max_s67 =
            vmaxq_u8(vreinterpretq_u8_u16(secondary_val[6]),
                     vreinterpretq_u8_u16(secondary_val[7]));
        const uint16x8_t max_s = vreinterpretq_u16_u8(
            vmaxq_u8(vmaxq_u8(max_s01, max_s23), vmaxq_u8(max_s45, max_s67)));
        max = vmaxq_u16(max, vandq_u16(max_s, cdef_large_value_mask));
      }

      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[0], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap0);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[1], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap0);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[2], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap1);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[3], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap1);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[4], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap0);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[5], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap0);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[6], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap1);
      sum = vmlaq_n_s16(sum,
                        Constrain(secondary_val[7], pixel, secondary_threshold,
                                  secondary_damping_shift),
                        kCdefSecondaryTap1);
    }
    // Clip3(pixel + ((8 + sum - (sum < 0)) >> 4), min, max))
    const int16x8_t sum_lt_0 = vshrq_n_s16(sum, 15);
    sum = vaddq_s16(sum, sum_lt_0);
    int16x8_t result = vrsraq_n_s16(vreinterpretq_s16_u16(pixel), sum, 4);
    if (clipping_required) {
      result = vminq_s16(result, vreinterpretq_s16_u16(max));
      result = vmaxq_s16(result, vreinterpretq_s16_u16(min));
    }

    const uint8x8_t dst_pixel = vqmovun_s16(result);
    if (width == 8) {
      src += src_stride;
      vst1_u8(dst, dst_pixel);
      dst += dst_stride;
      --y;
    } else {
      src += 2 * src_stride;
      StoreLo4(dst, dst_pixel);
      dst += dst_stride;
      StoreHi4(dst, dst_pixel);
      dst += dst_stride;
      y -= 2;
    }
  } while (y != 0);
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(kBitdepth8);
  assert(dsp != nullptr);
  dsp->cdef_direction = CdefDirection_NEON;
  dsp->cdef_filters[0][0] = CdefFilter_NEON<4>;
  dsp->cdef_filters[0][1] =
      CdefFilter_NEON<4, /*enable_primary=*/true, /*enable_secondary=*/false>;
  dsp->cdef_filters[0][2] = CdefFilter_NEON<4, /*enable_primary=*/false>;
  dsp->cdef_filters[1][0] = CdefFilter_NEON<8>;
  dsp->cdef_filters[1][1] =
      CdefFilter_NEON<8, /*enable_primary=*/true, /*enable_secondary=*/false>;
  dsp->cdef_filters[1][2] = CdefFilter_NEON<8, /*enable_primary=*/false>;
}

}  // namespace
}  // namespace low_bitdepth

void CdefInit_NEON() { low_bitdepth::Init8bpp(); }

}  // namespace dsp
}  // namespace libgav1
#else   // !LIBGAV1_ENABLE_NEON
namespace libgav1 {
namespace dsp {

void CdefInit_NEON() {}

}  // namespace dsp
}  // namespace libgav1
#endif  // LIBGAV1_ENABLE_NEON
