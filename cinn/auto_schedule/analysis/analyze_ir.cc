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

#include "cinn/auto_schedule/analysis/analyze_ir.h"

#include "cinn/ir/buffer.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace auto_schedule {

void AnalyzeScheduleBlockReadWriteBuffer(ir::ScheduleBlock* sche_block) {
  if (!sche_block->read_buffers.empty() || !sche_block->write_buffers.empty()) {
    return;
  }

  std::set<ir::Expr> load_tensors = ir::CollectLoadTensors(sche_block->body, [&](const Expr* x) { return true; });
  for (const ir::Expr& e : load_tensors) {
    ir::Tensor t = e.as_tensor_ref();
    sche_block->read_buffers.emplace_back(ir::BufferRange(t->buffer, t->axis_with_reduce()));
  }

  std::set<ir::Expr> store_tensors = ir::CollectStoreTensors(sche_block->body, [&](const Expr* x) { return true; });
  for (const ir::Expr& e : store_tensors) {
    ir::Tensor t = e.as_tensor_ref();
    sche_block->write_buffers.emplace_back(ir::BufferRange(t->buffer, t->axis_with_reduce()));
  }

  auto buffer_range_cmp = [](const Expr& lhs, const Expr& rhs) {
    return lhs.As<ir::_BufferRange_>()->buffer.as_buffer_ref() < rhs.As<ir::_BufferRange_>()->buffer.as_buffer_ref();
  };
  sort(sche_block->read_buffers.begin(), sche_block->read_buffers.end(), buffer_range_cmp);
  sort(sche_block->write_buffers.begin(), sche_block->write_buffers.end(), buffer_range_cmp);
}

}  // namespace auto_schedule
}  // namespace cinn
