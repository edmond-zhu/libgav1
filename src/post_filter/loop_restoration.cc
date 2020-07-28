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
#include "src/post_filter.h"
#include "src/utils/blocking_counter.h"

namespace libgav1 {

template <typename Pixel>
void PostFilter::ApplyLoopRestorationForOneRow(
    const Pixel* src_buffer, const ptrdiff_t stride, const Plane plane,
    const int plane_height, const int plane_width, const int unit_y,
    const int unit_row, const int current_process_unit_height,
    const int plane_unit_size, Pixel* dst_buffer) {
  const int num_horizontal_units =
      restoration_info_->num_horizontal_units(static_cast<Plane>(plane));
  const RestorationUnitInfo* const restoration_info =
      restoration_info_->loop_restoration_info(static_cast<Plane>(plane),
                                               unit_row * num_horizontal_units);
  const bool in_place = DoCdef() || thread_pool_ != nullptr;
  const Pixel* border = nullptr;
  src_buffer += unit_y * stride;
  if (in_place) {
    assert(loop_restoration_border_.stride(plane) ==
           static_cast<int>(sizeof(Pixel) * stride));
    const int border_unit_y = std::max(
        RightShiftWithCeiling(unit_y, 4 - subsampling_y_[plane]) - 4, 0);
    border =
        reinterpret_cast<const Pixel*>(loop_restoration_border_.data(plane)) +
        border_unit_y * stride;
  }
  int unit_column = 0;
  int column = 0;
  do {
    const int current_process_unit_width =
        std::min(plane_unit_size, plane_width - column);
    const Pixel* src = src_buffer + column;
    unit_column = std::min(unit_column, num_horizontal_units - 1);
    if (restoration_info[unit_column].type == kLoopRestorationTypeNone) {
      Pixel* dst = dst_buffer + column;
      if (in_place) {
        int k = current_process_unit_height;
        do {
          memmove(dst, src, current_process_unit_width * sizeof(Pixel));
          src += stride;
          dst += stride;
        } while (--k != 0);
      } else {
        CopyPlane(src, stride, current_process_unit_width,
                  current_process_unit_height, dst, stride);
      }
    } else {
      const Pixel* top_border = src - kRestorationVerticalBorder * stride;
      const Pixel* bottom_border = src + current_process_unit_height * stride;
      const bool frame_bottom_border =
          (unit_y + current_process_unit_height >= plane_height);
      if (in_place && (unit_y != 0 || !frame_bottom_border)) {
        const Pixel* loop_restoration_border = border + column;
        if (unit_y != 0) {
          top_border = loop_restoration_border;
          loop_restoration_border += 4 * stride;
        }
        if (!frame_bottom_border) {
          bottom_border =
              loop_restoration_border + kRestorationVerticalBorder * stride;
        }
      }
      RestorationBuffer restoration_buffer;
      const LoopRestorationType type = restoration_info[unit_column].type;
      assert(type == kLoopRestorationTypeSgrProj ||
             type == kLoopRestorationTypeWiener);
      const dsp::LoopRestorationFunc restoration_func =
          dsp_.loop_restorations[type - 2];
      restoration_func(restoration_info[unit_column], src, top_border,
                       bottom_border, stride, current_process_unit_width,
                       current_process_unit_height, &restoration_buffer,
                       dst_buffer + column);
    }
    ++unit_column;
    column += plane_unit_size;
  } while (column < plane_width);
}

template <typename Pixel>
void PostFilter::ApplyLoopRestorationSingleThread(const int row4x4_start,
                                                  const int sb4x4) {
  assert(row4x4_start >= 0);
  assert(DoRestoration());
  for (int plane = 0; plane < planes_; ++plane) {
    if (loop_restoration_.type[plane] == kLoopRestorationTypeNone) {
      continue;
    }
    const ptrdiff_t stride = frame_buffer_.stride(plane) / sizeof(Pixel);
    const int unit_height_offset =
        kRestorationUnitOffset >> subsampling_y_[plane];
    const int plane_height = SubsampledValue(height_, subsampling_y_[plane]);
    const int plane_width =
        SubsampledValue(upscaled_width_, subsampling_x_[plane]);
    const int plane_unit_size = 1 << loop_restoration_.unit_size_log2[plane];
    const int plane_process_unit_height =
        kRestorationUnitHeight >> subsampling_y_[plane];
    int y = (row4x4_start == 0)
                ? 0
                : (MultiplyBy4(row4x4_start) >> subsampling_y_[plane]) -
                      unit_height_offset;
    int expected_height = plane_process_unit_height -
                          ((row4x4_start == 0) ? unit_height_offset : 0);
    int current_process_unit_height;
    for (int sb_y = 0; sb_y < sb4x4;
         sb_y += 16, y += current_process_unit_height) {
      if (y >= plane_height) break;
      const int unit_row = std::min(
          (y + unit_height_offset) >> loop_restoration_.unit_size_log2[plane],
          restoration_info_->num_vertical_units(static_cast<Plane>(plane)) - 1);
      current_process_unit_height = std::min(expected_height, plane_height - y);
      expected_height = plane_process_unit_height;
      ApplyLoopRestorationForOneRow<Pixel>(
          reinterpret_cast<Pixel*>(superres_buffer_[plane]), stride,
          static_cast<Plane>(plane), plane_height, plane_width, y, unit_row,
          current_process_unit_height, plane_unit_size,
          reinterpret_cast<Pixel*>(loop_restoration_buffer_[plane]) +
              y * stride);
    }
  }
}

// Multi-thread version of loop restoration, based on a moving window of
// |window_buffer_height_| rows. Inside the moving window, we create a filtering
// job for each row and each filtering job is submitted to the thread pool. Each
// free thread takes one job from the thread pool and completes filtering until
// all jobs are finished.
template <typename Pixel>
void PostFilter::ApplyLoopRestorationThreaded() {
  for (int plane = kPlaneY; plane < planes_; ++plane) {
    if (loop_restoration_.type[plane] == kLoopRestorationTypeNone) {
      continue;
    }
    auto* const src_buffer = reinterpret_cast<Pixel*>(superres_buffer_[plane]);
    const ptrdiff_t stride = frame_buffer_.stride(plane) / sizeof(Pixel);
    const int unit_height_offset =
        kRestorationUnitOffset >> subsampling_y_[plane];
    const int plane_height = SubsampledValue(height_, subsampling_y_[plane]);
    const int plane_width =
        SubsampledValue(upscaled_width_, subsampling_x_[plane]);
    const int plane_unit_size = 1 << loop_restoration_.unit_size_log2[plane];
    const int plane_process_unit_height_log2 = 6 - subsampling_y_[plane];
    PostFilter::ExtendFrame<Pixel>(
        src_buffer, plane_width, plane_height, stride,
        kRestorationHorizontalBorder, kRestorationHorizontalBorder,
        kRestorationVerticalBorder, kRestorationVerticalBorder);
    const int num_workers = thread_pool_->num_threads();
    for (int y = 0; y < plane_height; y += window_buffer_height_) {
      const int actual_window_height =
          std::min(window_buffer_height_ - ((y == 0) ? unit_height_offset : 0),
                   plane_height - y);
      int vertical_units_per_window = RightShiftWithCeiling(
          actual_window_height, plane_process_unit_height_log2);
      if (y == 0) {
        // The first row of loop restoration processing units is not 64x64, but
        // 64x56 (|unit_height_offset| = 8 rows less than other restoration
        // processing units). For u/v with subsampling, the size is halved. To
        // compute the number of vertical units per window, we need to take a
        // special handling for it.
        const int height_without_first_unit =
            actual_window_height -
            std::min(
                actual_window_height,
                (1 << plane_process_unit_height_log2) - unit_height_offset);
        vertical_units_per_window =
            RightShiftWithCeiling(height_without_first_unit,
                                  plane_process_unit_height_log2) +
            1;
      }
      const int jobs_for_threadpool =
          vertical_units_per_window * num_workers / (num_workers + 1);
      assert(jobs_for_threadpool < vertical_units_per_window);
      BlockingCounter pending_jobs(jobs_for_threadpool);
      int job_count = 0;
      int current_process_unit_height;
      for (int unit_y = y; unit_y < y + actual_window_height;
           unit_y += current_process_unit_height) {
        const int expected_height = (1 << plane_process_unit_height_log2) -
                                    ((unit_y == 0) ? unit_height_offset : 0);
        current_process_unit_height =
            std::min(expected_height, plane_height - unit_y);
        const int unit_row = std::min(
            (unit_y + unit_height_offset) >>
                loop_restoration_.unit_size_log2[plane],
            restoration_info_->num_vertical_units(static_cast<Plane>(plane)) -
                1);
        Pixel* const dst =
            reinterpret_cast<Pixel*>(loop_restoration_buffer_[plane]) +
            unit_y * stride;
        if (job_count < jobs_for_threadpool) {
          thread_pool_->Schedule([this, src_buffer, plane, plane_height,
                                  plane_width, unit_y, unit_row,
                                  current_process_unit_height, plane_unit_size,
                                  dst, stride, &pending_jobs]() {
            ApplyLoopRestorationForOneRow<Pixel>(
                src_buffer, stride, static_cast<Plane>(plane), plane_height,
                plane_width, unit_y, unit_row, current_process_unit_height,
                plane_unit_size, dst);
            pending_jobs.Decrement();
          });
        } else {
          ApplyLoopRestorationForOneRow<Pixel>(
              src_buffer, stride, static_cast<Plane>(plane), plane_height,
              plane_width, unit_y, unit_row, current_process_unit_height,
              plane_unit_size, dst);
        }
        ++job_count;
      }
      // Wait for all jobs of current window to finish.
      pending_jobs.Wait();
      if (y == 0) y -= unit_height_offset;
    }
  }
}

void PostFilter::ApplyLoopRestoration(const int row4x4_start, const int sb4x4) {
#if LIBGAV1_MAX_BITDEPTH >= 10
  if (bitdepth_ >= 10) {
    ApplyLoopRestorationSingleThread<uint16_t>(row4x4_start, sb4x4);
    return;
  }
#endif
  ApplyLoopRestorationSingleThread<uint8_t>(row4x4_start, sb4x4);
}

void PostFilter::ApplyLoopRestoration() {
  assert(threaded_window_buffer_ != nullptr);
#if LIBGAV1_MAX_BITDEPTH >= 10
  if (bitdepth_ >= 10) {
    ApplyLoopRestorationThreaded<uint16_t>();
    return;
  }
#endif
  ApplyLoopRestorationThreaded<uint8_t>();
}

}  // namespace libgav1
