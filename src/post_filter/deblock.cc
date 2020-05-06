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
#include <atomic>

#include "src/post_filter.h"
#include "src/utils/blocking_counter.h"

namespace libgav1 {
namespace {

// 7.14.5.
void ComputeDeblockFilterLevelsHelper(
    const ObuFrameHeader& frame_header, int segment_id, int level_index,
    const int8_t delta_lf[kFrameLfCount],
    uint8_t deblock_filter_levels[kNumReferenceFrameTypes][2]) {
  const int delta = delta_lf[frame_header.delta_lf.multi ? level_index : 0];
  uint8_t level = Clip3(frame_header.loop_filter.level[level_index] + delta, 0,
                        kMaxLoopFilterValue);
  const auto feature = static_cast<SegmentFeature>(
      kSegmentFeatureLoopFilterYVertical + level_index);
  level =
      Clip3(level + frame_header.segmentation.feature_data[segment_id][feature],
            0, kMaxLoopFilterValue);
  if (!frame_header.loop_filter.delta_enabled) {
    static_assert(sizeof(deblock_filter_levels[0][0]) == 1, "");
    memset(deblock_filter_levels, level, kNumReferenceFrameTypes * 2);
    return;
  }
  assert(frame_header.loop_filter.delta_enabled);
  const int shift = level >> 5;
  deblock_filter_levels[kReferenceFrameIntra][0] = Clip3(
      level +
          LeftShift(frame_header.loop_filter.ref_deltas[kReferenceFrameIntra],
                    shift),
      0, kMaxLoopFilterValue);
  // deblock_filter_levels[kReferenceFrameIntra][1] is never used. So it does
  // not have to be populated.
  for (int reference_frame = kReferenceFrameIntra + 1;
       reference_frame < kNumReferenceFrameTypes; ++reference_frame) {
    for (int mode_id = 0; mode_id < 2; ++mode_id) {
      deblock_filter_levels[reference_frame][mode_id] = Clip3(
          level +
              LeftShift(frame_header.loop_filter.ref_deltas[reference_frame] +
                            frame_header.loop_filter.mode_deltas[mode_id],
                        shift),
          0, kMaxLoopFilterValue);
    }
  }
}

}  // namespace

void PostFilter::ComputeDeblockFilterLevels(
    const int8_t delta_lf[kFrameLfCount],
    uint8_t deblock_filter_levels[kMaxSegments][kFrameLfCount]
                                 [kNumReferenceFrameTypes][2]) const {
  if (!DoDeblock()) return;
  for (int segment_id = 0;
       segment_id < (frame_header_.segmentation.enabled ? kMaxSegments : 1);
       ++segment_id) {
    int level_index = 0;
    for (; level_index < 2; ++level_index) {
      ComputeDeblockFilterLevelsHelper(
          frame_header_, segment_id, level_index, delta_lf,
          deblock_filter_levels[segment_id][level_index]);
    }
    for (; level_index < kFrameLfCount; ++level_index) {
      if (frame_header_.loop_filter.level[level_index] != 0) {
        ComputeDeblockFilterLevelsHelper(
            frame_header_, segment_id, level_index, delta_lf,
            deblock_filter_levels[segment_id][level_index]);
      }
    }
  }
}

void PostFilter::InitDeblockFilterParams() {
  const int8_t sharpness = frame_header_.loop_filter.sharpness;
  assert(0 <= sharpness && sharpness < 8);
  const int shift = DivideBy4(sharpness + 3);  // ceil(sharpness / 4.0)
  for (int level = 0; level <= kMaxLoopFilterValue; ++level) {
    uint8_t limit = level >> shift;
    if (sharpness > 0) {
      limit = Clip3(limit, 1, 9 - sharpness);
    } else {
      limit = std::max(limit, static_cast<uint8_t>(1));
    }
    inner_thresh_[level] = limit;
    outer_thresh_[level] = 2 * (level + 2) + limit;
    hev_thresh_[level] = level >> 4;
  }
}

void PostFilter::GetDeblockFilterParams(uint8_t level, int* outer_thresh,
                                        int* inner_thresh,
                                        int* hev_thresh) const {
  *outer_thresh = outer_thresh_[level];
  *inner_thresh = inner_thresh_[level];
  *hev_thresh = hev_thresh_[level];
}

template <LoopFilterType type>
bool PostFilter::GetDeblockFilterEdgeInfo(const Plane plane, int row4x4,
                                          int column4x4,
                                          const int8_t subsampling_x,
                                          const int8_t subsampling_y,
                                          uint8_t* level, int* step,
                                          int* filter_length) const {
  row4x4 = GetDeblockPosition(row4x4, subsampling_y);
  column4x4 = GetDeblockPosition(column4x4, subsampling_x);
  const BlockParameters* bp = block_parameters_.Find(row4x4, column4x4);
  const TransformSize transform_size =
      (plane == kPlaneY) ? inter_transform_sizes_[row4x4][column4x4]
                         : bp->uv_transform_size;
  *step = (type == kLoopFilterTypeHorizontal) ? kTransformHeight[transform_size]
                                              : kTransformWidth[transform_size];
  if ((type == kLoopFilterTypeHorizontal && row4x4 == subsampling_y) ||
      (type == kLoopFilterTypeVertical && column4x4 == subsampling_x)) {
    return false;
  }

  const int filter_id = kDeblockFilterLevelIndex[plane][type];
  const uint8_t level_this = bp->deblock_filter_level[filter_id];
  const int row4x4_prev = (type == kLoopFilterTypeHorizontal)
                              ? row4x4 - (1 << subsampling_y)
                              : row4x4;
  const int column4x4_prev = (type == kLoopFilterTypeHorizontal)
                                 ? column4x4
                                 : column4x4 - (1 << subsampling_x);
  assert(row4x4_prev >= 0 && column4x4_prev >= 0);
  const BlockParameters* bp_prev =
      block_parameters_.Find(row4x4_prev, column4x4_prev);
  const uint8_t level_prev = bp_prev->deblock_filter_level[filter_id];
  *level = level_this;
  if (level_this == 0) {
    if (level_prev == 0) return false;
    *level = level_prev;
  }

  const BlockSize size =
      kPlaneResidualSize[bp->size][subsampling_x][subsampling_y];
  const int prediction_masks = (type == kLoopFilterTypeHorizontal)
                                   ? kBlockHeightPixels[size] - 1
                                   : kBlockWidthPixels[size] - 1;
  const int pixel_position = MultiplyBy4((type == kLoopFilterTypeHorizontal)
                                             ? row4x4 >> subsampling_y
                                             : column4x4 >> subsampling_x);
  const bool is_border = (pixel_position & prediction_masks) == 0;
  const bool skip = bp->skip && bp->is_inter;
  const bool skip_prev = bp_prev->skip && bp_prev->is_inter;
  if (!skip || !skip_prev || is_border) {
    const TransformSize transform_size_prev =
        (plane == kPlaneY) ? inter_transform_sizes_[row4x4_prev][column4x4_prev]
                           : bp_prev->uv_transform_size;
    const int step_prev = (type == kLoopFilterTypeHorizontal)
                              ? kTransformHeight[transform_size_prev]
                              : kTransformWidth[transform_size_prev];
    *filter_length = std::min(*step, step_prev);
    return true;
  }
  return false;
}

void PostFilter::HorizontalDeblockFilter(Plane plane, int row4x4_start,
                                         int column4x4_start) {
  const int8_t subsampling_x = subsampling_x_[plane];
  const int8_t subsampling_y = subsampling_y_[plane];
  const int column_step = 1 << subsampling_x;
  const size_t src_step = MultiplyBy4(pixel_size_);
  const ptrdiff_t src_stride = frame_buffer_.stride(plane);
  uint8_t* src = GetSourceBuffer(plane, row4x4_start, column4x4_start);
  const LoopFilterType type = kLoopFilterTypeHorizontal;
  int row_step;
  uint8_t level;
  int filter_length;

  for (int column4x4 = 0; MultiplyBy4(column4x4_start + column4x4) < width_ &&
                          column4x4 < kNum4x4InLoopFilterMaskUnit;
       column4x4 += column_step, src += src_step) {
    uint8_t* src_row = src;
    for (int row4x4 = 0; MultiplyBy4(row4x4_start + row4x4) < height_ &&
                         row4x4 < kNum4x4InLoopFilterMaskUnit;
         row4x4 += row_step) {
      const bool need_filter =
          GetDeblockFilterEdgeInfo<kLoopFilterTypeHorizontal>(
              plane, row4x4_start + row4x4, column4x4_start + column4x4,
              subsampling_x, subsampling_y, &level, &row_step, &filter_length);
      if (need_filter) {
        int outer_thresh;
        int inner_thresh;
        int hev_thresh;
        GetDeblockFilterParams(level, &outer_thresh, &inner_thresh,
                               &hev_thresh);
        const dsp::LoopFilterSize size =
            GetLoopFilterSize(plane, filter_length);
        const dsp::LoopFilterFunc filter_func = dsp_.loop_filters[size][type];
        filter_func(src_row, src_stride, outer_thresh, inner_thresh,
                    hev_thresh);
      }
      // TODO(chengchen): use shifts instead of multiplication.
      src_row += row_step * src_stride;
      row_step = DivideBy4(row_step << subsampling_y);
    }
  }
}

void PostFilter::VerticalDeblockFilter(Plane plane, int row4x4_start,
                                       int column4x4_start) {
  const int8_t subsampling_x = subsampling_x_[plane];
  const int8_t subsampling_y = subsampling_y_[plane];
  const int row_step = 1 << subsampling_y;
  const ptrdiff_t row_stride = MultiplyBy4(frame_buffer_.stride(plane));
  const ptrdiff_t src_stride = frame_buffer_.stride(plane);
  uint8_t* src = GetSourceBuffer(plane, row4x4_start, column4x4_start);
  const LoopFilterType type = kLoopFilterTypeVertical;
  int column_step;
  uint8_t level;
  int filter_length;

  for (int row4x4 = 0; MultiplyBy4(row4x4_start + row4x4) < height_ &&
                       row4x4 < kNum4x4InLoopFilterMaskUnit;
       row4x4 += row_step, src += row_stride) {
    uint8_t* src_row = src;
    for (int column4x4 = 0; MultiplyBy4(column4x4_start + column4x4) < width_ &&
                            column4x4 < kNum4x4InLoopFilterMaskUnit;
         column4x4 += column_step) {
      const bool need_filter =
          GetDeblockFilterEdgeInfo<kLoopFilterTypeVertical>(
              plane, row4x4_start + row4x4, column4x4_start + column4x4,
              subsampling_x, subsampling_y, &level, &column_step,
              &filter_length);
      if (need_filter) {
        int outer_thresh;
        int inner_thresh;
        int hev_thresh;
        GetDeblockFilterParams(level, &outer_thresh, &inner_thresh,
                               &hev_thresh);
        const dsp::LoopFilterSize size =
            GetLoopFilterSize(plane, filter_length);
        const dsp::LoopFilterFunc filter_func = dsp_.loop_filters[size][type];
        filter_func(src_row, src_stride, outer_thresh, inner_thresh,
                    hev_thresh);
      }
      src_row += column_step * pixel_size_;
      column_step = DivideBy4(column_step << subsampling_x);
    }
  }
}

void PostFilter::ApplyDeblockFilterForOneSuperBlockRow(int row4x4_start,
                                                       int sb4x4) {
  assert(row4x4_start >= 0);
  assert(DoDeblock());
  for (int plane = kPlaneY; plane < planes_; ++plane) {
    if (plane != kPlaneY && frame_header_.loop_filter.level[plane + 1] == 0) {
      continue;
    }

    for (int y = 0; y < sb4x4; y += 16) {
      const int row4x4 = row4x4_start + y;
      if (row4x4 >= frame_header_.rows4x4) break;
      int column4x4;
      for (column4x4 = 0; column4x4 < frame_header_.columns4x4;
           column4x4 += kNum4x4InLoopFilterMaskUnit) {
        // First apply vertical filtering
        VerticalDeblockFilter(static_cast<Plane>(plane), row4x4, column4x4);

        // Delay one superblock to apply horizontal filtering.
        if (column4x4 != 0) {
          HorizontalDeblockFilter(static_cast<Plane>(plane), row4x4,
                                  column4x4 - kNum4x4InLoopFilterMaskUnit);
        }
      }
      // Horizontal filtering for the last 64x64 block.
      HorizontalDeblockFilter(static_cast<Plane>(plane), row4x4,
                              column4x4 - kNum4x4InLoopFilterMaskUnit);
    }
  }
}

void PostFilter::DeblockFilterWorker(int jobs_per_plane, const Plane* planes,
                                     int num_planes,
                                     std::atomic<int>* job_counter,
                                     DeblockFilter deblock_filter) {
  const int total_jobs = jobs_per_plane * num_planes;
  int job_index;
  while ((job_index = job_counter->fetch_add(1, std::memory_order_relaxed)) <
         total_jobs) {
    const Plane plane = planes[job_index / jobs_per_plane];
    const int row_unit = job_index % jobs_per_plane;
    const int row4x4 = row_unit * kNum4x4InLoopFilterMaskUnit;
    for (int column4x4 = 0, column_unit = 0;
         column4x4 < frame_header_.columns4x4;
         column4x4 += kNum4x4InLoopFilterMaskUnit, ++column_unit) {
      (this->*deblock_filter)(plane, row4x4, column4x4);
    }
  }
}

void PostFilter::ApplyDeblockFilterThreaded() {
  const int jobs_per_plane = DivideBy16(frame_header_.rows4x4 + 15);
  const int num_workers = thread_pool_->num_threads();
  std::array<Plane, kMaxPlanes> planes;
  planes[0] = kPlaneY;
  int num_planes = 1;
  for (int plane = kPlaneU; plane < planes_; ++plane) {
    if (frame_header_.loop_filter.level[plane + 1] != 0) {
      planes[num_planes++] = static_cast<Plane>(plane);
    }
  }
  // The vertical filters are not dependent on each other. So simply schedule
  // them for all possible rows.
  //
  // The horizontal filter for a row/column depends on the vertical filter being
  // finished for the blocks to the top and to the right. To work around
  // this synchronization, we simply wait for the vertical filter to finish for
  // all rows. Now, the horizontal filters can also be scheduled
  // unconditionally similar to the vertical filters.
  //
  // The only synchronization involved is to know when the each directional
  // filter is complete for the entire frame.
  for (auto& type : {kLoopFilterTypeVertical, kLoopFilterTypeHorizontal}) {
    const DeblockFilter deblock_filter = deblock_filter_func_[type];
    std::atomic<int> job_counter(0);
    BlockingCounter pending_workers(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      thread_pool_->Schedule([this, jobs_per_plane, &planes, num_planes,
                              &job_counter, deblock_filter,
                              &pending_workers]() {
        DeblockFilterWorker(jobs_per_plane, planes.data(), num_planes,
                            &job_counter, deblock_filter);
        pending_workers.Decrement();
      });
    }
    // Run the jobs on the current thread.
    DeblockFilterWorker(jobs_per_plane, planes.data(), num_planes, &job_counter,
                        deblock_filter);
    // Wait for the threadpool jobs to finish.
    pending_workers.Wait();
  }
}

void PostFilter::ApplyDeblockFilter(LoopFilterType loop_filter_type,
                                    int row4x4_start, int column4x4_start,
                                    int column4x4_end, int sb4x4) {
  assert(row4x4_start >= 0);
  assert(DoDeblock());

  column4x4_end = std::min(column4x4_end, frame_header_.columns4x4);
  if (column4x4_start >= column4x4_end) return;

  const DeblockFilter deblock_filter = deblock_filter_func_[loop_filter_type];
  const int sb_height4x4 =
      std::min(sb4x4, frame_header_.rows4x4 - row4x4_start);
  for (int plane = kPlaneY; plane < planes_; ++plane) {
    if (plane != kPlaneY && frame_header_.loop_filter.level[plane + 1] == 0) {
      continue;
    }

    for (int y = 0; y < sb_height4x4; y += kNum4x4InLoopFilterMaskUnit) {
      const int row4x4 = row4x4_start + y;
      for (int column4x4 = column4x4_start; column4x4 < column4x4_end;
           column4x4 += kNum4x4InLoopFilterMaskUnit) {
        (this->*deblock_filter)(static_cast<Plane>(plane), row4x4, column4x4);
      }
    }
  }
}

}  // namespace libgav1
