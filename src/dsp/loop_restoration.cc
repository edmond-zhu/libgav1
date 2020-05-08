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

#include "src/dsp/loop_restoration.h"

#include <algorithm>  // std::max
#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/dsp/common.h"
#include "src/dsp/dsp.h"
#include "src/utils/common.h"
#include "src/utils/constants.h"

namespace libgav1 {
namespace dsp {

// Section 7.17.3.
// a2: range [1, 256].
// if (z >= 255)
//   a2 = 256;
// else if (z == 0)
//   a2 = 1;
// else
//   a2 = ((z << kSgrProjSgrBits) + (z >> 1)) / (z + 1);
const int kXByXPlus1[256] = {
    1,   128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239,
    240, 241, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247,
    248, 248, 248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250,
    250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252,
    252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253,
    253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
    253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    256};

// a2 = ((z << kSgrProjSgrBits) + (z >> 1)) / (z + 1);
// sgr_ma2 = 256 - a2
const uint8_t kSgrMa2Lookup[256] = {
    255, 128, 85, 64, 51, 43, 37, 32, 28, 26, 23, 21, 20, 18, 17, 16, 15, 14,
    13,  13,  12, 12, 11, 11, 10, 10, 9,  9,  9,  9,  8,  8,  8,  8,  7,  7,
    7,   7,   7,  6,  6,  6,  6,  6,  6,  6,  5,  5,  5,  5,  5,  5,  5,  5,
    5,   5,   4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
    4,   3,   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
    3,   3,   3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  2,  2,  2,  2,  2,  2,
    2,   2,   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,   2,   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,   2,   2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
    2,   2,   2,  2,  2,  2,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,   1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,   1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,   1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,   1,   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
    1,   1,   1,  0};

namespace {

struct SgrBuffer {
  // The arrays flt0 and flt1 in Section 7.17.2, the outputs of the box
  // filter process in pass 0 and pass 1.
  int32_t box_filter_process_output[2][kMaxBoxFilterProcessOutputPixels];
  // The 2d arrays A and B in Section 7.17.3, the intermediate results in
  // the box filter process. Reused for pass 0 and pass 1.
  uint32_t box_filter_process_intermediate[2]
                                          [kBoxFilterProcessIntermediatePixels];
};

constexpr int kOneByX[25] = {
    4096, 2048, 1365, 1024, 819, 683, 585, 512, 455, 410, 372, 341, 315,
    293,  273,  256,  241,  228, 216, 205, 195, 186, 178, 171, 164,
};

template <int bitdepth, typename Pixel>
struct LoopRestorationFuncs_C {
  LoopRestorationFuncs_C() = delete;

  static void SelfGuidedFilter(const void* source, void* dest,
                               const RestorationUnitInfo& restoration_info,
                               ptrdiff_t source_stride, ptrdiff_t dest_stride,
                               int width, int height,
                               RestorationBuffer* buffer);
  static void WienerFilter(const void* source, void* dest,
                           const RestorationUnitInfo& restoration_info,
                           ptrdiff_t source_stride, ptrdiff_t dest_stride,
                           int width, int height, RestorationBuffer* buffer);
  static void BoxFilterPreProcess(const RestorationUnitInfo& restoration_info,
                                  const Pixel* src, ptrdiff_t stride, int width,
                                  int height, int pass, SgrBuffer* buffer);
  static void BoxFilterProcess(const RestorationUnitInfo& restoration_info,
                               const Pixel* src, ptrdiff_t stride, int width,
                               int height, SgrBuffer* buffer);
};

// Note: range of wiener filter coefficients.
// Wiener filter coefficients are symmetric, and their sum is 1 (128).
// The range of each coefficient:
// filter[0] = filter[6], 4 bits, min = -5, max = 10.
// filter[1] = filter[5], 5 bits, min = -23, max = 8.
// filter[2] = filter[4], 6 bits, min = -17, max = 46.
// filter[3] = 128 - (filter[0] + filter[1] + filter[2]) * 2.
// The difference from libaom is that in libaom:
// filter[3] = 0 - (filter[0] + filter[1] + filter[2]) * 2.
// Thus in libaom's computation, an offset of 128 is needed for filter[3].
inline void PopulateWienerCoefficients(
    const RestorationUnitInfo& restoration_info, int direction,
    int16_t* const filter) {
  filter[3] = 128;
  for (int i = 0; i < 3; ++i) {
    const int16_t coeff = restoration_info.wiener_info.filter[direction][i];
    filter[i] = coeff;
    filter[3] -= MultiplyBy2(coeff);
  }
}

inline int CountZeroCoefficients(const int16_t* const filter) {
  int number_zero_coefficients = 0;
  if (filter[0] == 0) {
    number_zero_coefficients++;
    if (filter[1] == 0) {
      number_zero_coefficients++;
      if (filter[2] == 0) {
        number_zero_coefficients++;
      }
    }
  }
  return number_zero_coefficients;
}

template <typename Pixel>
inline int WienerHorizontal(const Pixel* const source,
                            const int16_t* const filter,
                            const int number_zero_coefficients, int sum) {
  constexpr int kCenterTap = (kSubPixelTaps - 1) / 2;
  for (int k = number_zero_coefficients; k < kCenterTap; ++k) {
    sum += filter[k] * (source[k] + source[kSubPixelTaps - 2 - k]);
  }
  return sum;
}

inline int WienerVertical(const uint16_t* const source,
                          const int16_t* const filter, const int width,
                          const int number_zero_coefficients, int sum) {
  constexpr int kCenterTap = (kSubPixelTaps - 1) / 2;
  for (int k = number_zero_coefficients; k < kCenterTap; ++k) {
    sum += filter[k] *
           (source[k * width] + source[(kSubPixelTaps - 2 - k) * width]);
  }
  return sum;
}

// Note: bit range for wiener filter.
// Wiener filter process first applies horizontal filtering to input pixels,
// followed by rounding with predefined bits (dependent on bitdepth).
// Then vertical filtering is applied, followed by rounding (dependent on
// bitdepth).
// The process is the same as convolution:
// <input> --> <horizontal filter> --> <rounding 0> --> <vertical filter>
// --> <rounding 1>
// By design:
// (a). horizontal/vertical filtering adds 7 bits to input.
// (b). The output of first rounding fits into 16 bits.
// (c). The output of second rounding fits into 16 bits.
// If input bitdepth > 8, the accumulator of the horizontal filter is larger
// than 16 bit and smaller than 32 bits.
// The accumulator of the vertical filter is larger than 16 bits and smaller
// than 32 bits.
template <int bitdepth, typename Pixel>
void LoopRestorationFuncs_C<bitdepth, Pixel>::WienerFilter(
    const void* const source, void* const dest,
    const RestorationUnitInfo& restoration_info, ptrdiff_t source_stride,
    ptrdiff_t dest_stride, int width, int height,
    RestorationBuffer* const buffer) {
  constexpr int kCenterTap = (kSubPixelTaps - 1) / 2;
  constexpr int kRoundBitsHorizontal = (bitdepth == 12)
                                           ? kInterRoundBitsHorizontal12bpp
                                           : kInterRoundBitsHorizontal;
  constexpr int kRoundBitsVertical =
      (bitdepth == 12) ? kInterRoundBitsVertical12bpp : kInterRoundBitsVertical;
  const int limit =
      (1 << (bitdepth + 1 + kWienerFilterBits - kRoundBitsHorizontal)) - 1;
  int16_t filter_horizontal[kSubPixelTaps / 2];
  int16_t filter_vertical[kSubPixelTaps / 2];
  PopulateWienerCoefficients(restoration_info, WienerInfo::kHorizontal,
                             filter_horizontal);
  PopulateWienerCoefficients(restoration_info, WienerInfo::kVertical,
                             filter_vertical);
  const int number_zero_coefficients_horizontal =
      CountZeroCoefficients(filter_horizontal);
  const int number_zero_coefficients_vertical =
      CountZeroCoefficients(filter_vertical);

  // horizontal filtering.
  const auto* src = static_cast<const Pixel*>(source);
  src -= (kCenterTap - number_zero_coefficients_vertical) * source_stride +
         kCenterTap;
  auto* wiener_buffer =
      buffer->wiener_buffer + number_zero_coefficients_vertical * width;
  const int horizontal_rounding = 1 << (bitdepth + kWienerFilterBits - 1);
  int y = height + kSubPixelTaps - 2 - 2 * number_zero_coefficients_vertical;

  if (number_zero_coefficients_horizontal == 0) {
    do {
      int x = 0;
      do {
        // sum fits into 16 bits only when bitdepth = 8.
        int sum = horizontal_rounding;
        sum = WienerHorizontal<Pixel>(src + x, filter_horizontal, 0, sum);
        sum += filter_horizontal[kCenterTap] * src[x + kCenterTap];
        const int rounded_sum =
            RightShiftWithRounding(sum, kRoundBitsHorizontal);
        wiener_buffer[x] = static_cast<uint16_t>(Clip3(rounded_sum, 0, limit));
      } while (++x < width);
      src += source_stride;
      wiener_buffer += width;
    } while (--y != 0);
  } else if (number_zero_coefficients_horizontal == 1) {
    do {
      int x = 0;
      do {
        // sum fits into 16 bits only when bitdepth = 8.
        int sum = horizontal_rounding;
        sum = WienerHorizontal<Pixel>(src + x, filter_horizontal, 1, sum);
        sum += filter_horizontal[kCenterTap] * src[x + kCenterTap];
        const int rounded_sum =
            RightShiftWithRounding(sum, kRoundBitsHorizontal);
        wiener_buffer[x] = static_cast<uint16_t>(Clip3(rounded_sum, 0, limit));
      } while (++x < width);
      src += source_stride;
      wiener_buffer += width;
    } while (--y != 0);
  } else if (number_zero_coefficients_horizontal == 2) {
    do {
      int x = 0;
      do {
        // sum fits into 16 bits only when bitdepth = 8.
        int sum = horizontal_rounding;
        sum = WienerHorizontal<Pixel>(src + x, filter_horizontal, 2, sum);
        sum += filter_horizontal[kCenterTap] * src[x + kCenterTap];
        const int rounded_sum =
            RightShiftWithRounding(sum, kRoundBitsHorizontal);
        wiener_buffer[x] = static_cast<uint16_t>(Clip3(rounded_sum, 0, limit));
      } while (++x < width);
      src += source_stride;
      wiener_buffer += width;
    } while (--y != 0);
  } else {
    do {
      int x = 0;
      do {
        // sum fits into 16 bits only when bitdepth = 8.
        int sum = horizontal_rounding;
        sum += filter_horizontal[kCenterTap] * src[x + kCenterTap];
        const int rounded_sum =
            RightShiftWithRounding(sum, kRoundBitsHorizontal);
        wiener_buffer[x] = static_cast<uint16_t>(Clip3(rounded_sum, 0, limit));
      } while (++x < width);
      src += source_stride;
      wiener_buffer += width;
    } while (--y != 0);
  }

  // vertical filtering.
  const int vertical_rounding = -(1 << (bitdepth + kRoundBitsVertical - 1));
  auto* dst = static_cast<Pixel*>(dest);
  wiener_buffer = buffer->wiener_buffer;
  y = height;

  if (number_zero_coefficients_vertical == 0) {
    do {
      int x = 0;
      do {
        // sum needs 32 bits.
        int sum = vertical_rounding;
        sum = WienerVertical(wiener_buffer + x, filter_vertical, width, 0, sum);
        sum +=
            filter_vertical[kCenterTap] * wiener_buffer[kCenterTap * width + x];
        const int rounded_sum = RightShiftWithRounding(sum, kRoundBitsVertical);
        dst[x] = static_cast<Pixel>(Clip3(rounded_sum, 0, (1 << bitdepth) - 1));
      } while (++x < width);
      dst += dest_stride;
      wiener_buffer += width;
    } while (--y != 0);
  } else if (number_zero_coefficients_vertical == 1) {
    do {
      int x = 0;
      do {
        // sum needs 32 bits.
        int sum = vertical_rounding;
        sum = WienerVertical(wiener_buffer + x, filter_vertical, width, 1, sum);
        sum +=
            filter_vertical[kCenterTap] * wiener_buffer[kCenterTap * width + x];
        const int rounded_sum = RightShiftWithRounding(sum, kRoundBitsVertical);
        dst[x] = static_cast<Pixel>(Clip3(rounded_sum, 0, (1 << bitdepth) - 1));
      } while (++x < width);
      dst += dest_stride;
      wiener_buffer += width;
    } while (--y != 0);
  } else if (number_zero_coefficients_vertical == 2) {
    do {
      int x = 0;
      do {
        // sum needs 32 bits.
        int sum = vertical_rounding;
        sum = WienerVertical(wiener_buffer + x, filter_vertical, width, 2, sum);
        sum +=
            filter_vertical[kCenterTap] * wiener_buffer[kCenterTap * width + x];
        const int rounded_sum = RightShiftWithRounding(sum, kRoundBitsVertical);
        dst[x] = static_cast<Pixel>(Clip3(rounded_sum, 0, (1 << bitdepth) - 1));
      } while (++x < width);
      dst += dest_stride;
      wiener_buffer += width;
    } while (--y != 0);
  } else {
    do {
      int x = 0;
      do {
        // sum needs 32 bits.
        int sum = vertical_rounding;
        sum +=
            filter_vertical[kCenterTap] * wiener_buffer[kCenterTap * width + x];
        const int rounded_sum = RightShiftWithRounding(sum, kRoundBitsVertical);
        dst[x] = static_cast<Pixel>(Clip3(rounded_sum, 0, (1 << bitdepth) - 1));
      } while (++x < width);
      dst += dest_stride;
      wiener_buffer += width;
    } while (--y != 0);
  }
}

//------------------------------------------------------------------------------
// SGR

template <typename Pixel>
inline void UpdateSum(const Pixel* src, const ptrdiff_t stride, const int size,
                      uint32_t* const a, uint32_t* const b) {
  int y = size;
  do {
    const Pixel source0 = src[0];
    const Pixel source1 = src[size];
    *a -= source0 * source0;
    *a += source1 * source1;
    *b -= source0;
    *b += source1;
    src += stride;
  } while (--y != 0);
}

template <int bitdepth>
inline void CalculateIntermediate(const uint32_t s, uint32_t a,
                                  const uint32_t b, const uint32_t n,
                                  const int x,
                                  uint32_t* intermediate_result[2]) {
  // a: before shift, max is 25 * (2^(bitdepth) - 1) * (2^(bitdepth) - 1).
  // since max bitdepth = 12, max < 2^31.
  // after shift, a < 2^16 * n < 2^22 regardless of bitdepth
  a = RightShiftWithRounding(a, (bitdepth - 8) << 1);
  // b: max is 25 * (2^(bitdepth) - 1). If bitdepth = 12, max < 2^19.
  // d < 2^8 * n < 2^14 regardless of bitdepth
  const uint32_t d = RightShiftWithRounding(b, bitdepth - 8);
  // p: Each term in calculating p = a * n - b * b is < 2^16 * n^2 < 2^28,
  // and p itself satisfies p < 2^14 * n^2 < 2^26.
  // This bound on p is due to:
  // https://en.wikipedia.org/wiki/Popoviciu's_inequality_on_variances
  // Note: Sometimes, in high bitdepth, we can end up with a*n < b*b.
  // This is an artifact of rounding, and can only happen if all pixels
  // are (almost) identical, so in this case we saturate to p=0.
  const uint32_t p = (a * n < d * d) ? 0 : a * n - d * d;
  // p * s < (2^14 * n^2) * round(2^20 / (n^2 * scale)) < 2^34 / scale <
  // 2^32 as long as scale >= 4. So p * s fits into a uint32_t, and z < 2^12
  // (this holds even after accounting for the rounding in s)
  const uint32_t z = RightShiftWithRounding(p * s, kSgrProjScaleBits);
  // a2: range [1, 256].
  uint32_t a2 = kXByXPlus1[std::min(z, 255u)];
  const uint32_t one_over_n = kOneByX[n - 1];
  // (kSgrProjSgrBits - a2) < 2^8, b < 2^(bitdepth) * n,
  // one_over_n = round(2^12 / n)
  // => the product here is < 2^(20 + bitdepth) <= 2^32,
  // and b is set to a value < 2^(8 + bitdepth).
  // This holds even with the rounding in one_over_n and in the overall
  // result, as long as (kSgrProjSgrBits - a2) is strictly less than 2^8.
  const uint32_t b2 = ((1 << kSgrProjSgrBits) - a2) * b * one_over_n;
  intermediate_result[0][x] = a2;
  intermediate_result[1][x] =
      RightShiftWithRounding(b2, kSgrProjReciprocalBits);
}

template <int bitdepth, typename Pixel>
void LoopRestorationFuncs_C<bitdepth, Pixel>::BoxFilterPreProcess(
    const RestorationUnitInfo& restoration_info, const Pixel* src,
    const ptrdiff_t stride, int width, int height, int pass,
    SgrBuffer* const buffer) {
  const int sgr_proj_index = restoration_info.sgr_proj_info.index;
  const uint8_t radius = kSgrProjParams[sgr_proj_index][pass * 2];
  assert(radius != 0);
  const uint32_t n = (2 * radius + 1) * (2 * radius + 1);
  const uint32_t s = kSgrScaleParameter[sgr_proj_index][pass];
  assert(s != 0);
  const ptrdiff_t array_stride =
      kRestorationUnitWidthWithBorders + kRestorationPadding;
  // The size of the intermediate result buffer is the size of the filter area
  // plus horizontal (3) and vertical (3) padding. The processing start point
  // is the filter area start point -1 row and -1 column. Therefore we need to
  // set offset and use the intermediate_result as the start point for
  // processing.
  const ptrdiff_t intermediate_buffer_offset =
      kRestorationBorder * array_stride + kRestorationBorder;
  uint32_t* intermediate_result[2] = {
      buffer->box_filter_process_intermediate[0] + intermediate_buffer_offset -
          array_stride,
      buffer->box_filter_process_intermediate[1] + intermediate_buffer_offset -
          array_stride};

  // Calculate intermediate results, including one-pixel border, for example,
  // if unit size is 64x64, we calculate 66x66 pixels.
  const int step = (pass == 0) ? 2 : 1;
  const ptrdiff_t intermediate_stride = step * array_stride;
  src -= (radius + 1) * stride + radius + 1;
  for (int y = -1; y <= height; y += step) {
    uint32_t a = 0;
    uint32_t b = 0;
    for (int dy = 0; dy <= 2 * radius; ++dy) {
      for (int dx = 0; dx <= 2 * radius; ++dx) {
        const Pixel source = src[dy * stride + dx];
        a += source * source;
        b += source;
      }
    }
    CalculateIntermediate<bitdepth>(s, a, b, n, -1, intermediate_result);
    for (int x = 0; x <= width; ++x) {
      UpdateSum<Pixel>(src + x, stride, 2 * radius + 1, &a, &b);
      CalculateIntermediate<bitdepth>(s, a, b, n, x, intermediate_result);
    }
    intermediate_result[0] += intermediate_stride;
    intermediate_result[1] += intermediate_stride;
    src += step * stride;
  }
}

template <int bitdepth, typename Pixel>
void LoopRestorationFuncs_C<bitdepth, Pixel>::BoxFilterProcess(
    const RestorationUnitInfo& restoration_info, const Pixel* src,
    ptrdiff_t stride, int width, int height, SgrBuffer* const buffer) {
  const int sgr_proj_index = restoration_info.sgr_proj_info.index;
  const ptrdiff_t filtered_output_stride = width;
  const ptrdiff_t intermediate_stride =
      kRestorationUnitWidthWithBorders + kRestorationPadding;
  const ptrdiff_t intermediate_buffer_offset =
      kRestorationBorder * intermediate_stride + kRestorationBorder;

  for (int pass = 0; pass < 2; ++pass) {
    const uint8_t radius = kSgrProjParams[sgr_proj_index][pass * 2];
    if (radius == 0) continue;
    LoopRestorationFuncs_C<bitdepth, Pixel>::BoxFilterPreProcess(
        restoration_info, src, stride, width, height, pass, buffer);

    const Pixel* src_ptr = src;
    // Set intermediate buffer start point to the actual start point of
    // filtering.
    const uint32_t* array_start[2] = {
        buffer->box_filter_process_intermediate[0] + intermediate_buffer_offset,
        buffer->box_filter_process_intermediate[1] +
            intermediate_buffer_offset};
    int* filtered_output = buffer->box_filter_process_output[pass];
    for (int y = 0; y < height; ++y) {
      const int shift = (pass == 0 && (y & 1) != 0) ? 4 : 5;
      // array_start[0]: range [1, 256].
      // array_start[1] < 2^20.
      for (int x = 0; x < width; ++x) {
        uint32_t a, b;
        if (pass == 0) {
          if ((y & 1) == 0) {
            a = 5 * (array_start[0][-intermediate_stride + x - 1] +
                     array_start[0][-intermediate_stride + x + 1] +
                     array_start[0][intermediate_stride + x - 1] +
                     array_start[0][intermediate_stride + x + 1]) +
                6 * (array_start[0][-intermediate_stride + x] +
                     array_start[0][intermediate_stride + x]);
            b = 5 * (array_start[1][-intermediate_stride + x - 1] +
                     array_start[1][-intermediate_stride + x + 1] +
                     array_start[1][intermediate_stride + x - 1] +
                     array_start[1][intermediate_stride + x + 1]) +
                6 * (array_start[1][-intermediate_stride + x] +
                     array_start[1][intermediate_stride + x]);
          } else {
            a = 5 * (array_start[0][x - 1] + array_start[0][x + 1]) +
                6 * array_start[0][x];
            b = 5 * (array_start[1][x - 1] + array_start[1][x + 1]) +
                6 * array_start[1][x];
          }
        } else {
          a = 3 * (array_start[0][-intermediate_stride + x - 1] +
                   array_start[0][-intermediate_stride + x + 1] +
                   array_start[0][intermediate_stride + x - 1] +
                   array_start[0][intermediate_stride + x + 1]) +
              4 * (array_start[0][-intermediate_stride + x] +
                   array_start[0][x - 1] + array_start[0][x] +
                   array_start[0][x + 1] +
                   array_start[0][intermediate_stride + x]);
          b = 3 * (array_start[1][-intermediate_stride + x - 1] +
                   array_start[1][-intermediate_stride + x + 1] +
                   array_start[1][intermediate_stride + x - 1] +
                   array_start[1][intermediate_stride + x + 1]) +
              4 * (array_start[1][-intermediate_stride + x] +
                   array_start[1][x - 1] + array_start[1][x] +
                   array_start[1][x + 1] +
                   array_start[1][intermediate_stride + x]);
        }
        // v < 2^32. All intermediate calculations are positive.
        const uint32_t v = a * src_ptr[x] + b;
        filtered_output[x] = RightShiftWithRounding(
            v, kSgrProjSgrBits + shift - kSgrProjRestoreBits);
      }
      src_ptr += stride;
      array_start[0] += intermediate_stride;
      array_start[1] += intermediate_stride;
      filtered_output += filtered_output_stride;
    }
  }
}

// Assume box_filter_process_output[2] are allocated before calling
// this function. Their sizes are width * height, stride equals width.
template <int bitdepth, typename Pixel>
void LoopRestorationFuncs_C<bitdepth, Pixel>::SelfGuidedFilter(
    const void* const source, void* const dest,
    const RestorationUnitInfo& restoration_info, ptrdiff_t source_stride,
    ptrdiff_t dest_stride, int width, int height,
    RestorationBuffer* const /*buffer*/) {
  SgrBuffer buffer;
  const int w0 = restoration_info.sgr_proj_info.multiplier[0];
  const int w1 = restoration_info.sgr_proj_info.multiplier[1];
  const int w2 = (1 << kSgrProjPrecisionBits) - w0 - w1;
  const int index = restoration_info.sgr_proj_info.index;
  const int radius_pass_0 = kSgrProjParams[index][0];
  const int radius_pass_1 = kSgrProjParams[index][2];
  const ptrdiff_t array_stride = width;
  const int* box_filter_process_output[2] = {
      buffer.box_filter_process_output[0], buffer.box_filter_process_output[1]};
  const auto* src = static_cast<const Pixel*>(source);
  auto* dst = static_cast<Pixel*>(dest);
  LoopRestorationFuncs_C<bitdepth, Pixel>::BoxFilterProcess(
      restoration_info, src, source_stride, width, height, &buffer);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int u = src[x] << kSgrProjRestoreBits;
      int v = w1 * u;
      if (radius_pass_0 != 0) {
        v += w0 * box_filter_process_output[0][x];
      } else {
        v += w0 * u;
      }
      if (radius_pass_1 != 0) {
        v += w2 * box_filter_process_output[1][x];
      } else {
        v += w2 * u;
      }
      // if radius_pass_0 == 0 and radius_pass_1 == 0, the range of v is:
      // bits(u) + bits(w0/w1/w2) + 2 = bitdepth + 13.
      // Then, range of s is bitdepth + 2. This is a rough estimation, taking
      // the maximum value of each element.
      const int s = RightShiftWithRounding(
          v, kSgrProjRestoreBits + kSgrProjPrecisionBits);
      dst[x] = static_cast<Pixel>(Clip3(s, 0, (1 << bitdepth) - 1));
    }
    src += source_stride;
    dst += dest_stride;
    box_filter_process_output[0] += array_stride;
    box_filter_process_output[1] += array_stride;
  }
}

void Init8bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(8);
  assert(dsp != nullptr);
#if LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
  dsp->loop_restorations[0] = LoopRestorationFuncs_C<8, uint8_t>::WienerFilter;
  dsp->loop_restorations[1] =
      LoopRestorationFuncs_C<8, uint8_t>::SelfGuidedFilter;
#else  // !LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
  static_cast<void>(dsp);
#ifndef LIBGAV1_Dsp8bpp_WienerFilter
  dsp->loop_restorations[0] = LoopRestorationFuncs_C<8, uint8_t>::WienerFilter;
#endif
#ifndef LIBGAV1_Dsp8bpp_SelfGuidedFilter
  dsp->loop_restorations[1] =
      LoopRestorationFuncs_C<8, uint8_t>::SelfGuidedFilter;
#endif
#endif  // LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
}

#if LIBGAV1_MAX_BITDEPTH >= 10

void Init10bpp() {
  Dsp* const dsp = dsp_internal::GetWritableDspTable(10);
  assert(dsp != nullptr);
#if LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
  dsp->loop_restorations[0] =
      LoopRestorationFuncs_C<10, uint16_t>::WienerFilter;
  dsp->loop_restorations[1] =
      LoopRestorationFuncs_C<10, uint16_t>::SelfGuidedFilter;
#else  // !LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
  static_cast<void>(dsp);
#ifndef LIBGAV1_Dsp10bpp_WienerFilter
  dsp->loop_restorations[0] =
      LoopRestorationFuncs_C<10, uint16_t>::WienerFilter;
#endif
#ifndef LIBGAV1_Dsp10bpp_SelfGuidedFilter
  dsp->loop_restorations[1] =
      LoopRestorationFuncs_C<10, uint16_t>::SelfGuidedFilter;
#endif
#endif  // LIBGAV1_ENABLE_ALL_DSP_FUNCTIONS
}

#endif  // LIBGAV1_MAX_BITDEPTH >= 10
}  // namespace

void LoopRestorationInit_C() {
  Init8bpp();
#if LIBGAV1_MAX_BITDEPTH >= 10
  Init10bpp();
#endif
  // Local functions that may be unused depending on the optimizations
  // available.
  static_cast<void>(CountZeroCoefficients);
  static_cast<void>(PopulateWienerCoefficients);
  static_cast<void>(WienerVertical);
}

}  // namespace dsp
}  // namespace libgav1
