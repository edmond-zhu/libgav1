/*
 * Copyright 2019 The libgav1 Authors
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

#ifndef LIBGAV1_SRC_DECODER_IMPL_H_
#define LIBGAV1_SRC_DECODER_IMPL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "src/buffer_pool.h"
#include "src/dsp/constants.h"
#include "src/frame_scratch_buffer.h"
#include "src/gav1/decoder_buffer.h"
#include "src/gav1/decoder_settings.h"
#include "src/gav1/status_code.h"
#include "src/loop_filter_mask.h"
#include "src/obu_parser.h"
#include "src/residual_buffer_pool.h"
#include "src/symbol_decoder_context.h"
#include "src/threading_strategy.h"
#include "src/tile.h"
#include "src/utils/array_2d.h"
#include "src/utils/block_parameters_holder.h"
#include "src/utils/compiler_attributes.h"
#include "src/utils/constants.h"
#include "src/utils/memory.h"
#include "src/utils/queue.h"
#include "src/utils/segmentation_map.h"
#include "src/utils/types.h"

namespace libgav1 {

struct DecoderState {
  // Section 7.20. Updates frames in the reference_frame array with
  // |current_frame|, based on the |refresh_frame_flags| bitmask.
  void UpdateReferenceFrames(const RefCountedBufferPtr& current_frame,
                             int refresh_frame_flags);

  // Clears all the reference frames.
  void ClearReferenceFrames();

  // reference_valid and reference_frame_id are used only if
  // sequence_header_.frame_id_numbers_present is true.
  // The reference_valid array is indexed by a reference picture slot number.
  // A value (boolean) in the array signifies whether the corresponding
  // reference picture slot is valid for use as a reference picture.
  std::array<bool, kNumReferenceFrameTypes> reference_valid = {};
  std::array<uint16_t, kNumReferenceFrameTypes> reference_frame_id = {};
  // A valid value of current_frame_id is an unsigned integer of at most 16
  // bits. -1 indicates current_frame_id is not initialized.
  int current_frame_id = -1;
  // The RefOrderHint array variable in the spec.
  std::array<uint8_t, kNumReferenceFrameTypes> reference_order_hint = {};
  // The OrderHint variable in the spec. Its value comes from either the
  // order_hint syntax element in the uncompressed header (if
  // show_existing_frame is false) or RefOrderHint[ frame_to_show_map_idx ]
  // (if show_existing_frame is true and frame_type is KEY_FRAME). See Section
  // 5.9.2 and Section 7.4.
  //
  // NOTE: When show_existing_frame is false, it is often more convenient to
  // just use the order_hint field of the frame header as OrderHint. So this
  // field is mainly used to update the reference_order_hint array in
  // UpdateReferenceFrames().
  uint8_t order_hint = 0;
  // reference_frame_sign_bias[i] (a boolean) specifies the intended direction
  // of the motion vector in time for each reference frame.
  // * |false| indicates that the reference frame is a forwards reference (i.e.
  //   the reference frame is expected to be output before the current frame);
  // * |true| indicates that the reference frame is a backwards reference.
  // Note: reference_frame_sign_bias[0] (for kReferenceFrameIntra) is not used.
  std::array<bool, kNumReferenceFrameTypes> reference_frame_sign_bias = {};
  std::array<RefCountedBufferPtr, kNumReferenceFrameTypes> reference_frame;
};

struct TemporalUnit;

struct EncodedFrame {
  EncodedFrame(ObuParser* const obu, const DecoderState& state,
               TemporalUnit* const temporal_unit,
               const RefCountedBufferPtr& frame, int position_in_temporal_unit)
      : sequence_header(obu->sequence_header()),
        frame_header(obu->frame_header()),
        state(state),
        temporal_unit(*temporal_unit),
        frame(frame),
        position_in_temporal_unit(position_in_temporal_unit) {
    obu->MoveTileGroups(&tile_groups);
    frame->MarkFrameAsStarted();
  }

  const ObuSequenceHeader sequence_header;
  const ObuFrameHeader frame_header;
  Vector<ObuTileGroup> tile_groups;
  DecoderState state;
  TemporalUnit& temporal_unit;
  RefCountedBufferPtr frame;
  const int position_in_temporal_unit;
};

struct TemporalUnit : public Allocable {
  // The default constructor is invoked by the Queue<TemporalUnit>::Init()
  // method. Queue<> does not use the default-constructed elements, so it is
  // safe for the default constructor to not initialize the members.
  TemporalUnit() = default;
  TemporalUnit(const uint8_t* data, size_t size, int64_t user_private_data,
               void* buffer_private_data)
      : data(data),
        size(size),
        user_private_data(user_private_data),
        buffer_private_data(buffer_private_data),
        decoded(false),
        status(kStatusOk),
        has_displayable_frame(false),
        decoded_count(0) {}

  const uint8_t* data;
  size_t size;
  int64_t user_private_data;
  void* buffer_private_data;

  // The following members are used only in frame parallel mode.
  bool decoded;
  StatusCode status;
  bool has_displayable_frame;
  RefCountedBufferPtr output_frame;
  int output_frame_position;
  Vector<EncodedFrame> frames;
  size_t decoded_count;
};

class DecoderImpl : public Allocable {
 public:
  // The constructor saves a const reference to |*settings|. Therefore
  // |*settings| must outlive the DecoderImpl object. On success, |*output|
  // contains a pointer to the newly-created DecoderImpl object. On failure,
  // |*output| is not modified.
  static StatusCode Create(const DecoderSettings* settings,
                           std::unique_ptr<DecoderImpl>* output);
  ~DecoderImpl();
  StatusCode EnqueueFrame(const uint8_t* data, size_t size,
                          int64_t user_private_data, void* buffer_private_data);
  StatusCode DequeueFrame(const DecoderBuffer** out_ptr);
  int GetMaxAllowedFrames() const { return frame_threads_; }
  static constexpr int GetMaxBitdepth() {
    static_assert(LIBGAV1_MAX_BITDEPTH == 8 || LIBGAV1_MAX_BITDEPTH == 10,
                  "LIBGAV1_MAX_BITDEPTH must be 8 or 10.");
    return LIBGAV1_MAX_BITDEPTH;
  }

 private:
  explicit DecoderImpl(const DecoderSettings* settings);
  StatusCode Init();

  bool AllocateCurrentFrame(RefCountedBuffer* current_frame,
                            const ColorConfig& color_config,
                            const ObuFrameHeader& frame_header, int left_border,
                            int right_border, int top_border,
                            int bottom_border);
  void ReleaseOutputFrame();

  // Decodes all the frames contained in the given temporal unit. Used only in
  // non frame parallel mode.
  StatusCode DecodeTemporalUnit(const TemporalUnit& temporal_unit,
                                const DecoderBuffer** out_ptr);
  // Used only in frame parallel mode. EnqueueFrame pushes the last enqueued
  // temporal unit into |temporal_units_| and this function will do the OBU
  // parsing for the last temporal unit that was pushed into the queue and
  // enqueue the frames for decoding.
  StatusCode ParseAndEnqueue();
  // Decodes the |encoded_frame| and updates the
  // |encoded_frame->temporal_unit|'s parameters if the decoded frame is a
  // displayable frame. Used only in frame parallel mode.
  StatusCode DecodeFrame(EncodedFrame* encoded_frame);

  // Populates |buffer_| with values from |frame|. Adds a reference to |frame|
  // in |output_frame_|.
  StatusCode CopyFrameToOutputBuffer(const RefCountedBufferPtr& frame);
  StatusCode DecodeTiles(const ObuSequenceHeader& sequence_header,
                         const ObuFrameHeader& frame_header,
                         const Vector<ObuTileGroup>& tile_groups,
                         const DecoderState& state,
                         FrameScratchBuffer* frame_scratch_buffer,
                         RefCountedBuffer* current_frame);
  // Sets the current frame's segmentation map for two cases. The third case
  // is handled in Tile::DecodeBlock().
  void SetCurrentFrameSegmentationMap(const ObuFrameHeader& frame_header,
                                      const SegmentationMap* prev_segment_ids,
                                      RefCountedBuffer* current_frame);
  // Applies film grain synthesis to the |displayable_frame| and stores the film
  // grain applied frame into |film_grain_frame|. Returns kStatusOk on success.
  StatusCode ApplyFilmGrain(const ObuSequenceHeader& sequence_header,
                            const ObuFrameHeader& frame_header,
                            const RefCountedBufferPtr& displayable_frame,
                            RefCountedBufferPtr* film_grain_frame);

  bool FrameParallel() const { return frame_threads_ > 1; }

  Queue<TemporalUnit> temporal_units_;
  DecoderState state_;
  ThreadingStrategy threading_strategy_;

  DecoderBuffer buffer_ = {};
  // output_frame_ holds a reference to the output frame on behalf of buffer_.
  RefCountedBufferPtr output_frame_;

  BufferPool buffer_pool_;
  WedgeMaskArray wedge_masks_;
  FrameScratchBufferPool frame_scratch_buffer_pool_;
  int frame_threads_ = 1;

  // Used to synchronize the accesses into |temporal_units_| in order to update
  // the "decoded" state of an temporal unit.
  std::mutex mutex_;
  std::unique_ptr<ThreadPool> frame_thread_pool_;

  ObuSequenceHeader sequence_header_ = {};
  // If true, sequence_header is valid.
  bool has_sequence_header_ = false;

#if defined(ENABLE_FRAME_PARALLEL)
  // TODO(b/142583029): A copy of the DecoderSettings is made to facilitate the
  // development of frame parallel mode behind a compile time flag.
  DecoderSettings settings_;
#else
  const DecoderSettings& settings_;
#endif
};

}  // namespace libgav1

#endif  // LIBGAV1_SRC_DECODER_IMPL_H_
