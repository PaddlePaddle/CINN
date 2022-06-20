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

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_inline.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cinn/auto_schedule/analysis/analyze_ir.h"
#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/common/target.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

AutoInline::AutoInline(const common::Target& target) : AutoGenRule(target) {}

bool AutoInline::CanInlineIntoConsumer(const ir::ScheduleBlockRealize& sche_block_realize) const {
  const ir::ScheduleBlock* sche_block = sche_block_realize.schedule_block.As<ir::ScheduleBlock>();
  ir::Expr compute_body               = sche_block->body;
  ir::Expr root                       = ir_schedule_->GetRootBlock(ir::Expr(&sche_block_realize));
  // 1. Check the schedule block to be inlined is not a reduce tensor.
  std::set<ir::Expr> find_store =
      ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Store>(); });
  if (find_store.size() != 1UL) {
    return false;
  }
  Expr tensor = (*find_store.begin()).As<ir::Store>()->tensor;
  if (tensor.as_tensor_ref()->is_reduce_tensor()) {
    return false;
  }
  // 2. Check this schedule block is the only writer of the tensor.
  find_store = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
    return x->As<ir::Store>() && (x->As<ir::Store>()->tensor).as_tensor_ref()->name == tensor.as_tensor_ref()->name;
  });
  if (find_store.size() != 1UL) {
    return false;
  }
  // 3. Check there is no overlap between the buffers the schedule block reads and writes.
  std::set<ir::Expr> find_load = ir::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Load>() && x->As<ir::Load>()->tensor == tensor; });
  if (!find_load.empty()) {
    return false;
  }

  ir::Expr store = *(find_store.begin());

  ir::ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(), store);
  return inliner.BodyPatternAllowInline();
}

AutoInlineType AutoInline::AnalyzeInlineType(const ir::ScheduleBlockRealize& sche_block_realize) const {
  const ir::ScheduleBlock* sche_block = sche_block_realize.schedule_block.As<ir::ScheduleBlock>();

  // First, we inline if the block has only 1 write buffer
  if (sche_block->write_buffers.size() != 1) {
    return AutoInlineType::kCannotInline;
  }

  // write_buffers.size() = 1 and read_buffers is empty, means const
  // we can inline to consumer
  if (sche_block->read_buffers.empty()) {
    return AutoInlineType::kInlineIntoConsumer;
  }

  std::unordered_set<ir::IrNodeTy> no_inline_node_types = {ir::IrNodeTy::Power, ir::IrNodeTy::IfThenElse};
  if (ContainsNodeType(sche_block->body, no_inline_node_types)) {
    return AutoInlineType::kCannotInline;
  }

  // InlineIntoConsumer other than above situations
  if (CanInlineIntoConsumer(sche_block_realize)) {
    return AutoInlineType::kInlineIntoConsumer;
  }

  // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
  // so we just do kInlineIntoConsumer here. Add CanInlineIntoConsumer
  // once ReverseComputeInline is ready.
  return AutoInlineType::kCannotInline;
}

RuleApplyType AutoInline::Init(const ir::ModuleExpr& mod_expr) {
  ir_schedule_        = std::make_unique<ir::IRSchedule>(mod_expr);
  all_block_realizes_ = ir_schedule_->GetAllBlocks();
  apply_indices_and_type_.clear();
  num_applicable_ = 0;

  for (size_t i = 0; i < all_block_realizes_.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize = all_block_realizes_[i].As<ir::ScheduleBlockRealize>();
    AnalyzeScheduleBlockReadWriteBuffer(sche_block_realize->schedule_block.As<ir::ScheduleBlock>());
    AutoInlineType type = AnalyzeInlineType(*sche_block_realize);
    if (type != AutoInlineType::kCannotInline) {
      ++num_applicable_;
      apply_indices_and_type_.push_back({i, type});
    }
  }

  return num_applicable_ > 0 ? RuleApplyType::kApply : RuleApplyType::kCannotApply;
}

ir::ModuleExpr AutoInline::Apply(int index) {
  CHECK(ir_schedule_ != nullptr) << "Run AutoInline::Apply without Init";
  CHECK(num_applicable_ > 0 && apply_indices_and_type_.size() == num_applicable_)
      << "AutoInline::Apply pre-condition doesn't meet";
  CHECK(num_applicable_ > index)
      << "Invalid index for AutoInline::Apply, the index needs 0 <= index && index < NumberApplicable()";

  int apply_index     = apply_indices_and_type_[index].first;
  AutoInlineType type = apply_indices_and_type_[index].second;
  if (type == AutoInlineType::kInlineIntoConsumer) {
    ir_schedule_->ComputeInline(all_block_realizes_[apply_index]);
  } else if (type == AutoInlineType::kInlineIntoProducer) {
    // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
    // so we just do kInlineIntoConsumer here. Add CanInlineIntoConsumer
    // once ReverseComputeInline is ready.
    // ir_schedule_->ReverseComputeInline(all_block_realizes_[apply_index]);
  }

  // Make sure re-apply the AutoInline won't be error.
  apply_indices_and_type_[index].second = AutoInlineType::kCannotInline;
  return ir_schedule_->GetModule();
}

std::string AutoInline::GetRuleName() const { return "AutoInline"; }

AutoGenRule* AutoInline::NewPointer() const { return new AutoInline(*target_); }

}  // namespace auto_schedule
}  // namespace cinn
