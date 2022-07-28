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
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

AutoInline::AutoInline(const common::Target& target, const std::unordered_set<std::string>& no_inline_output_names)
    : AutoGenRule(target), no_inline_output_names_(no_inline_output_names) {}

bool AutoInline::CanInlineIntoConsumer(const Expr& sche_block_realize_expr) const {
  const ir::ScheduleBlockRealize* sche_block_realize = sche_block_realize_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sche_block                = sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
  ir::Expr compute_body                              = sche_block->body;
  ir::Expr root                                      = ir_schedule_->GetRootBlock(sche_block_realize_expr);
  // Check the schedule block to be inlined is not a reduce tensor.
  std::set<ir::Expr> find_store =
      ir::CollectIRNodesWithoutTensor(compute_body, [&](const Expr* x) { return x->As<ir::Store>(); });
  if (find_store.size() != 1UL) {
    return false;
  }
  ir::Expr tensor_expr = (*find_store.begin()).As<ir::Store>()->tensor;
  ir::Tensor tensor    = tensor_expr.as_tensor_ref();
  if (tensor->is_reduce_tensor()) {
    return false;
  }

  if (no_inline_output_names_.find(tensor->name) != no_inline_output_names_.end()) {
    return false;
  }

  // write_buffers.size() = 1 and read_buffers is empty, means const
  // we can inline to consumer
  if (sche_block->read_buffers.empty()) {
    return true;
  }

  // Check this schedule block is the only writer of the tensor.
  find_store = ir::CollectIRNodesWithoutTensor(root, [&](const Expr* x) {
    return x->As<ir::Store>() && (x->As<ir::Store>()->tensor).as_tensor_ref()->name == tensor->name;
  });
  if (find_store.size() != 1UL) {
    return false;
  }
  // Check there is no overlap between the buffers the schedule block reads and writes.
  std::set<ir::Expr> find_load = ir::CollectIRNodesWithoutTensor(
      compute_body, [&](const Expr* x) { return x->As<ir::Load>() && x->As<ir::Load>()->tensor == tensor_expr; });
  if (!find_load.empty()) {
    return false;
  }

  ir::Expr store = *(find_store.begin());

  ir::ComputeInliner inliner(store.As<ir::Store>()->tensor.as_tensor_ref(), store);
  if (!inliner.BodyPatternAllowInline()) {
    return false;
  }

  ir::LeafBlockRemovalPlan remove_plan(sche_block_realize_expr, &inliner.src_stmt, &inliner.tgt_stmt);
  remove_plan(&root);
  if (!inliner.src_stmt.defined() || !inliner.tgt_stmt.defined()) {
    return false;
  }

  VLOG(6) << "Found store Expr " << store << ", which CanInlineIntoConsumer";
  return true;
}

AutoInlineType AutoInline::AnalyzeInlineType(const Expr& sche_block_realize_expr) const {
  const ir::ScheduleBlockRealize* sche_block_realize = sche_block_realize_expr.As<ir::ScheduleBlockRealize>();
  const ir::ScheduleBlock* sche_block                = sche_block_realize->schedule_block.As<ir::ScheduleBlock>();

  // Inline if the block has only 1 write buffer
  if (sche_block->write_buffers.size() != 1) {
    return AutoInlineType::kCannotInline;
  }

  std::unordered_set<ir::IrNodeTy> no_inline_node_types = {ir::IrNodeTy::Power, ir::IrNodeTy::IfThenElse};
  if (ContainsNodeType(sche_block->body, no_inline_node_types)) {
    return AutoInlineType::kCannotInline;
  }

  // InlineIntoConsumer other than above situations
  if (CanInlineIntoConsumer(sche_block_realize_expr)) {
    return AutoInlineType::kInlineIntoConsumer;
  }

  // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
  // so we just do kInlineIntoConsumer here. Add CanInlineIntoProducer
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
    AutoInlineType type = AnalyzeInlineType(all_block_realizes_[i]);
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
    VLOG(6) << "Apply ComputeInline on " << all_block_realizes_[apply_index];
    ir_schedule_->ComputeInline(all_block_realizes_[apply_index]);
    VLOG(6) << "After ComputeInline: " << all_block_realizes_[apply_index];

  } else if (type == AutoInlineType::kInlineIntoProducer) {
    // TODO(zhhsplendid): We don't have ReverseComputeInline in IRSchedule now,
    // so we just do kInlineIntoConsumer here. Add CanInlineIntoConsumer
    // once ReverseComputeInline is ready.

    // ir_schedule_->ReverseComputeInline(all_block_realizes_[apply_index]);
  }

  // Make sure re-apply the AutoInline won't be error.
  // AutoInline changes the read and write buffers of schedule blocks,
  // we need to re-analyze
  all_block_realizes_ = ir_schedule_->GetAllBlocks();
  for (size_t i = 0; i < all_block_realizes_.size(); ++i) {
    ir::ScheduleBlockRealize* sche_block_realize = all_block_realizes_[i].As<ir::ScheduleBlockRealize>();
    ir::ScheduleBlock* sche_block                = sche_block_realize->schedule_block.As<ir::ScheduleBlock>();
    sche_block->read_buffers                     = {};
    sche_block->write_buffers                    = {};
    AnalyzeScheduleBlockReadWriteBuffer(sche_block);
  }
  return ir_schedule_->GetModule();
}

std::string AutoInline::GetRuleName() const { return "AutoInline"; }

AutoGenRule* AutoInline::NewPointer() const { return new AutoInline(*target_, no_inline_output_names_); }

}  // namespace auto_schedule
}  // namespace cinn
