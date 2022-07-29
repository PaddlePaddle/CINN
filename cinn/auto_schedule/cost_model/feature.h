// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <vector>

#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

/* Loop feature enums */
enum class LoopPositionFeatureEnum : int { kNone; kInnerMost; kMiddle; kOuterMost; };

enum class LoopReduceFeatureEnum : int { kNone; kIsReduce; kNonReduce; kMix; };

enum class ForOptimizeFeatureEnum : int { kNone; kVectorize; kParallel; kUnroll; };

/* Buffer memory feature enums */
enum class BufferOperationFeatureEnum : int {
  kUnknown; kRead; kWrite; kReadWrite;  // read and write at same time, such as a[i] = a[i + 1]
  kAlloc;
  kDelete;
};

class LoopBlockFeature {
 public:
  /* Arithmetic features */
  int float_add_or_sub   = 0;
  int float_mul          = 0;
  int float_div_or_mod   = 0;
  int float_cmp          = 0;
  int float_multiply_add = 0;  // number of float Multiply-add ops
  int float_math_func    = 0;
  int float_other_call   = 0;  // like simple assign, cast, etc.

  int int_add_or_sub   = 0;
  int int_mul          = 0;
  int int_div_or_mod   = 0;
  int int_cmp          = 0;
  int int_multiply_add = 0;  // number of float Multiply-add ops
  int int_math_func    = 0;
  int int_other_call   = 0;  // like simple assign, cast, etc.

  int bool_op   = 0;
  int select_op = 0;

  /* Buffer memory features */

  // Buffer operation types
  std::vector<BufferOperationFeatureEnum> buffer_op_types;
  // The size of buffer operation, the vector size must be same as buffer_op_types
  std::vector<int> buffer_op_size;

  /* Loop type features */
  LoopPositionFeatureEnum loop_pos_type  = LoopPositionEnum::kNone;
  LoopReduceFeatureEnum loop_reduce_type = LoopReduceEnum::kNone;
  ForOptimizeFeatureEnum loop_opt_type   = ForOptimizeKindEnum::kNone;

  /* Thread features if loop is optimized by GPU or CPU parallelism.
   * Useless in other cases.
   */
  int len_blockIdx_x  = 1;
  int len_blockIdx_y  = 1;
  int len_blockIdx_z  = 1;
  int len_threadIdx_x = 1;
  int len_threadIdx_y = 1;
  int len_threadIdx_z = 1;
  int len_vthread     = 1;  // length of virtual thread

  // Number to indicate the loop block inside current one
  int num_sub_loops = 0;
};

/**
 * Feature of Expr. It is used in CostModel
 */
class Feature {
 public:
  Feature();
  std::vector<float> ToVector();

 public:
  // We treat a computation feature to be encoded as variable-length vector.
  // The root compute block is not a loop, but we treat it as a size-1 loop.
  // Blocks are encoded like a stack. Each LoopBlockFeature contains a
  // num_sub_loops to indicate the next level sub-loop-block it contains.
  //
  // For example, code like:
  //
  // some_compute_0
  // loop1 {
  //   some_compute_1
  //   loop2 {
  //     some_compute_2
  //   }
  // }
  //
  // loop3 {
  //   some_compute_3
  // }
  //
  // We go through the code and push loops into stack, then the features are encoded as
  // [loop_block_feature_0, loop_block_feature_1, loop_block_feature_2, loop_block_feature_3]
  // where loop_block_feature_i stores the features of some_compute_i (such
  // as number of arithmetic operations)
  //
  // loop_block_feature_0.num_sub_loops = 2
  // loop_block_feature_1.num_sub_loops = 1
  // loop_block_feature_2.num_sub_loops = 0
  // loop_block_feature_3.num_sub_loops = 0
  std::vector<LoopBlockFeature> stack_encoded_feature;
};

}  // namespace auto_schedule
}  // namespace cinn
