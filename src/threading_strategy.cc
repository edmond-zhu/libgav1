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

#include "src/threading_strategy.h"

#include <algorithm>
#include <cassert>

#include "src/frame_scratch_buffer.h"
#include "src/utils/constants.h"
#include "src/utils/logging.h"

namespace libgav1 {

bool ThreadingStrategy::Reset(const ObuFrameHeader& frame_header,
                              int thread_count) {
  assert(thread_count > 0);
  frame_parallel_ = false;

  if (thread_count == 1) {
    thread_pool_.reset(nullptr);
    tile_thread_count_ = 0;
    max_tile_index_for_row_threads_ = 0;
    return true;
  }

  // We do work in the current thread, so it is sufficient to create
  // |thread_count|-1 threads in the threadpool.
  thread_count = std::min(thread_count, static_cast<int>(kMaxThreads)) - 1;

  if (thread_pool_ == nullptr || thread_pool_->num_threads() != thread_count) {
    thread_pool_ = ThreadPool::Create("libgav1", thread_count);
    if (thread_pool_ == nullptr) {
      LIBGAV1_DLOG(ERROR, "Failed to create a thread pool with %d threads.",
                   thread_count);
      tile_thread_count_ = 0;
      max_tile_index_for_row_threads_ = 0;
      return false;
    }
  }

  // Prefer tile threads first (but only if there is more than one tile).
  const int tile_count = frame_header.tile_info.tile_count;
  if (tile_count > 1) {
    // We want 1 + tile_thread_count_ <= tile_count because the current thread
    // is also used to decode tiles. This is equivalent to
    // tile_thread_count_ <= tile_count - 1.
    tile_thread_count_ = std::min(thread_count, tile_count - 1);
    thread_count -= tile_thread_count_;
    if (thread_count == 0) {
      max_tile_index_for_row_threads_ = 0;
      return true;
    }
  } else {
    tile_thread_count_ = 0;
  }

#if defined(__ANDROID__)
  // Assign the remaining threads for each Tile. The heuristic used here is that
  // we will assign two threads for each Tile. So for example, if |thread_count|
  // is 2, for a stream with 2 tiles the first tile would get both the threads
  // and the second tile would have row multi-threading turned off. This
  // heuristic is based on the fact that row multi-threading is fast enough only
  // when there are at least two threads to do the decoding (since one thread
  // always does the parsing).
  //
  // This heuristic might stop working when SIMD optimizations make the decoding
  // much faster and the parsing thread is only as fast as the decoding threads.
  // So we will have to revisit this later to make sure that this is still
  // optimal.
  //
  // Note that while this heuristic significantly improves performance on high
  // end devices (like the Pixel 3), there are some performance regressions in
  // some lower end devices (in some cases) and that needs to be revisited as we
  // bring in more optimizations. Overall, the gains because of this heuristic
  // seems to be much larger than the regressions.
  for (int i = 0; i < tile_count; ++i) {
    max_tile_index_for_row_threads_ = i + 1;
    thread_count -= 2;
    if (thread_count <= 0) break;
  }
#else   // !defined(__ANDROID__)
  // Assign the remaining threads to each Tile.
  for (int i = 0; i < tile_count; ++i) {
    const int count = thread_count / tile_count +
                      static_cast<int>(i < thread_count % tile_count);
    if (count == 0) {
      // Once we see a 0 value, all subsequent values will be 0 since it is
      // supposed to be assigned in a round-robin fashion.
      break;
    }
    max_tile_index_for_row_threads_ = i + 1;
  }
#endif  // defined(__ANDROID__)
  return true;
}

bool ThreadingStrategy::Reset(int thread_count) {
  assert(thread_count > 0);
  frame_parallel_ = true;

  // In frame parallel mode, we simply access the underlying |thread_pool_|
  // directly. So ensure all the other threadpool getter functions return
  // nullptr. Also, superblock row multithreading is always disabled in frame
  // parallel mode.
  tile_thread_count_ = 0;
  max_tile_index_for_row_threads_ = 0;

  if (thread_pool_ == nullptr || thread_pool_->num_threads() != thread_count) {
    thread_pool_ = ThreadPool::Create("libgav1-fp", thread_count);
    if (thread_pool_ == nullptr) {
      LIBGAV1_DLOG(ERROR, "Failed to create a thread pool with %d threads.",
                   thread_count);
      return false;
    }
  }
  return true;
}

bool InitializeThreadPoolsForFrameParallel(
    int thread_count, std::unique_ptr<ThreadPool>* const frame_thread_pool,
    FrameScratchBufferPool* const frame_scratch_buffer_pool) {
  // TODO(b/142583029): For now, we set frame threads to 2 and distribute the
  // rest of the threads to be used as tile threads. Eventually this will be
  // replaced by a proper threading model that combines frame threads and tile
  // threads.
  static constexpr int kFrameThreads = 2;
  const int frame_threads = kFrameThreads;
  *frame_thread_pool = ThreadPool::Create(frame_threads);
  if (*frame_thread_pool == nullptr) {
    LIBGAV1_DLOG(ERROR, "Failed to create frame thread pool with %d threads.",
                 frame_threads);
    return false;
  }
  int remaining_threads = thread_count - frame_threads;
  if (remaining_threads == 0) return true;
  int threads_per_frame = remaining_threads / frame_threads;
  assert(frame_threads <= kFrameThreads);
  std::unique_ptr<FrameScratchBuffer> frame_scratch_buffers[kFrameThreads] = {};
  const int extra_threads_for_first_buffer = remaining_threads % frame_threads;
  // Create the tile thread pools.
  for (int i = 0; i < frame_threads && remaining_threads > 0; ++i) {
    frame_scratch_buffers[i] = frame_scratch_buffer_pool->Get();
    if (frame_scratch_buffers[i] == nullptr) {
      return false;
    }
    // If the number of tile threads cannot be divided equally amongst all the
    // frame threads, simply assign the extra to the first frame thread.
    // TODO(vigneshv): This can be improved by splitting the extra threads
    // across frame in a round robin fashion.
    const int current_frame_thread_count =
        threads_per_frame + ((i == 0) ? extra_threads_for_first_buffer : 0);
    if (!frame_scratch_buffers[i]->threading_strategy.Reset(
            current_frame_thread_count)) {
      return false;
    }
    remaining_threads -= current_frame_thread_count;
  }
  for (auto& frame_scratch_buffer : frame_scratch_buffers) {
    if (frame_scratch_buffer == nullptr) break;
    frame_scratch_buffer_pool->Release(std::move(frame_scratch_buffer));
  }
  return true;
}

}  // namespace libgav1
