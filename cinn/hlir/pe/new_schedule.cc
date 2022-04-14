// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/hlir/pe/new_schedule.h"

#include <absl/container/flat_hash_map.h>
#include <isl/cpp.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <utility>

#include "cinn/common/cas.h"
#include "cinn/hlir/pe/load_x86_params.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/poly/isl_utils.h"

namespace cinn {
namespace hlir {
namespace pe {

void NewScheduleInjectiveCPU(ir::IRSchedule &ir_sch,
                             const std::vector<int> &output_shape,
                             const common::Target &target,
                             bool vectorizable) {
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops      = ir_sch.GetLoops(all_blocks[0]);
  int dims        = output_shape.size();
  int factor      = GetBasicFactor(GetTensor(all_blocks[0])->type(), target);
  auto fused      = loops[0];
  if (dims >= 5) {
    fused = ir_sch.Fuse({loops[0], loops[1], loops[2]});
    dims  = dims - 2;
  } else if (dims >= 3) {
    fused = ir_sch.Fuse({loops[0], loops[1]});
    dims  = dims - 1;
  }
  ir_sch.Parallel(fused);

  if (vectorizable) {
    auto all_blocks = ir_sch.GetAllBlocks();
    auto loops      = ir_sch.GetLoops(all_blocks[0]);
    int last_shape  = ir::GetLoopExtent(loops[dims - 1]);
    factor          = GetVectorizeFactor(last_shape, factor);
    auto splited    = ir_sch.Split(loops[dims - 1], {-1, factor});
    ir_sch.Vectorize(splited[1], factor);
    if (dims == 1) {
      ir_sch.Parallel(splited[0]);
    }
  }
}

void NewCudaScheduleInjective(ir::IRSchedule &ir_sch,
                              const std::vector<int> &output_shape,
                              const common::Target &target) {
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops      = ir_sch.GetLoops(all_blocks[0]);
  auto fused      = ir_sch.Fuse(loops);

  int num_thread        = target.max_num_threads();
  int num_block         = 1024;
  int vector_width      = 1;
  int prod_size         = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
  bool need_block_split = prod_size > num_thread * num_block * vector_width ? true : false;
  if (need_block_split) {
    auto splited = ir_sch.Split(fused, {num_block, num_thread, -1});
    ir_sch.Bind(splited[0], "blockIdx.x");
    ir_sch.Bind(splited[1], "threadIdx.x");
  } else {
    if (prod_size > num_thread) {
      auto splited = ir_sch.Split(fused, {-1, num_thread});
      ir_sch.Bind(splited[0], "blockIdx.x");
      ir_sch.Bind(splited[1], "threadIdx.x");
    } else {
      ir_sch.Bind(fused, "blockIdx.x");
    }
  }
  LOG(INFO) << "After NewCudaScheduleInjective, new ir is : " << ir_sch.GetModule().GetExprs().at(0);
}

void NewCudaScheduleMul(ir::IRSchedule &ir_sch, const std::vector<int> &output_shape, const common::Target &target) {
  auto all_blocks = ir_sch.GetAllBlocks();
  auto loops      = ir_sch.GetLoops(all_blocks.back());
  auto splited    = ir_sch.Split(loops[1], {-1, 2});
  all_blocks      = ir_sch.GetAllBlocks();
  loops           = ir_sch.GetLoops(all_blocks.back());
  ir_sch.Bind(loops[0], "blockIdx.x");
  ir_sch.Bind(loops[1], "threadIdx.x");
}

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
