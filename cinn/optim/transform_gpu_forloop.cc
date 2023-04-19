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

#include "cinn/optim/transform_gpu_forloop.h"

#include <algorithm>
#include <map>
#include <stack>
#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/cas.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/ir_simplify.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/poly/isl_utils.h"
#include "cinn/poly/stage.h"
#include "cinn/runtime/intrinsic.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {

/**
 * 1. Determine the grid and block dimensions.
 * It takes the domains like `[0, 20]` or `[0, min(20, M/2)]`, the domain should have a integer right bound.
 *
 * 2. Replace the grid/thread iterators with something like `threadIdx.x`, `threadIdx.y`.
 *
 * 3. Remove the forloops owning the gpu axis.
 *   1. if the extent is an IntImm, just remove this forloop.
 *   2. if the extent is a Min, replace the forloop with an IfThenElse, with forloop's condition, new check will add (if
 * the min of forloop is not zero).
 *
 * @param expr The expression to mutate.
 */
void RemoveGpuForloopsAxis(Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const ir::For *op, Expr *expr) override {
      switch (op->for_type()) {
        case ir::ForType::GPUBlock:
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          break;
        case ir::ForType::GPUThread:
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          break;
        default:
          auto *node = expr->As<ir::For>();
          IRMutator<>::Visit(&node->body, &node->body);
          break;
      }
    }

    bool NeedToReplaceForloopWithIfThenElse(const ir::For *n) const { return true; }

    void ReplaceForloopWithIfThenElse(Expr *expr) {
      auto *for_n      = expr->As<ir::For>();
      auto *poly_for_n = expr->As<ir::PolyFor>();
      CHECK(for_n || poly_for_n);

      Expr condition;

      auto condition_append = [&](Expr new_cond) {
        if (condition.defined()) {
          condition = ir::And::Make(condition, new_cond);
        } else {
          condition = new_cond;
        }
      };

      if (for_n) {
        // for(i, 2, 100);
        //        ^
        if (for_n->min != common::make_const(0)) {
          condition_append(ir::GE::Make(for_n->loop_var, for_n->min));
        }

        // for(i, 2, min(M/2, 20)
        //            ^
        condition_append(ir::LT::Make(for_n->loop_var, for_n->extent));
      } else {
        if (poly_for_n->init != common::make_const(0)) {
          condition_append(ir::GE::Make(poly_for_n->iterator, poly_for_n->init));
        }

        condition_append(poly_for_n->condition);
      }

      CHECK(condition.defined());

      VLOG(3) << "GPU replacing\n" << *expr;
      VLOG(3) << "\nto\n";
      auto if_n = ir::IfThenElse::Make(condition, for_n->body);
      VLOG(3) << if_n;
      *expr = if_n;
    }

    void Visit(const ir::PolyFor *op, Expr *expr) override {
      const auto msg = "PolyFor is not allowed for GPU, only For nodes are allowed";
      CHECK(op->for_type() != ir::ForType::GPUBlock) << msg;
      CHECK(op->for_type() != ir::ForType::GPUThread) << msg;
      CHECK(op->for_type() != ir::ForType::GPULane) << msg;
    }
  };

  Mutator mutator;
  mutator(expr);
}

/**
 * The generated __syncthreads call will be wrapped with a `if (xxxx == 0) { }`, this is the problem of isl AST output,
 * drop it to make it run in all the threads.
 */
void CudaSyncThreadsDropIfThenElse(Expr *expr) {
  struct Mutator : public ir::IRMutator<> {
    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

    void Visit(const ir::IfThenElse *op, Expr *expr) override {
      blocked_statement_stack.push_back(expr);
      ir::IRMutator<>::Visit(op, expr);
      blocked_statement_stack.pop_back();
    }

    void Visit(const ir::Call *op, Expr *expr) override {
      if (op->name == runtime::intrinsic::cuda_sync_threads) {
        if (!blocked_statement_stack.empty()) {
          auto *last_for = blocked_statement_stack.back()->As<ir::IfThenElse>();
          if (auto *eq_n = last_for->condition.As<ir::EQ>()) {
            if (eq_n->b() == common::make_const(0)) {
              *blocked_statement_stack.back() = *expr;
            }
          }
        }
      }
    }

    // Collect all the statements with Block(include Block) to the statement.
    std::vector<ir::Expr *> blocked_statement_stack;
  };

  Mutator()(expr);
}

// replace var to block/thread.
class ReplaceLoopVarToGpu : public ir::IRMutator<> {
 public:
  void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::For *op, Expr *expr) override {
    auto for_ir = expr->As<ir::For>();
    CHECK(for_ir);

    auto bind_info = for_ir->bind_info();

    std::string var_name = "";
    if (bind_info.offset == 0)
      var_name = "x";
    else if (bind_info.offset == 1)
      var_name = "y";
    else if (bind_info.offset == 2)
      var_name = "z";
    if (for_ir->is_gpu_block_binded()) {
      var_name = "blockIdx." + var_name;
      optim::ReplaceVarWithExpr(expr, op->loop_var, ir::Expr(ir::Var(var_name)));
    } else if (for_ir->is_gpu_thread_binded()) {
      var_name = "threadIdx." + var_name;
      optim::ReplaceVarWithExpr(expr, op->loop_var, ir::Expr(ir::Var(var_name)));
    }

    ir::IRMutator<>::Visit(&for_ir->body, &for_ir->body);
  }
  void Visit(const ir::PolyFor *op, Expr *expr) override { LOG(FATAL) << "Unkown PolyFor!"; }
};

class ReplaceIndexToBindExpr : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::ScheduleBlockRealize *op, Expr *expr) override {
    auto *schedule_block_realize = expr->As<ir::ScheduleBlockRealize>();
    CHECK(schedule_block_realize->schedule_block.As<ir::ScheduleBlock>());
    auto iter_values = schedule_block_realize->iter_values;
    auto body_copy   = schedule_block_realize->schedule_block.As<ir::ScheduleBlock>()->body;
    auto iter_vars   = schedule_block_realize->schedule_block.As<ir::ScheduleBlock>()->iter_vars;

    CHECK_EQ(iter_values.size(), iter_vars.size());
    for (int idx = 0; idx < iter_values.size(); ++idx) {
      ReplaceVarWithExpr(&body_copy, iter_vars[idx], iter_values[idx]);
    }
    ir::IRMutator<>::Visit(&body_copy, &body_copy);
  }
};

// update buffer size
using LOOPS = std::vector<ir::Expr>;
class CollectTensorLoopVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    store_buffer_loop_map_[op->tensor.as_tensor_ref()->name] = loops_;
    stores_.push_back(*expr);
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    if (op->is_addr_scalar()) {
      return;
    }

    auto tensor = op->tensor.as_tensor_ref();
    if (load_buffer_loop_map_.count(tensor->name)) {
      load_buffer_loop_map_[tensor->name].push_back(loops_);
    } else {
      load_buffer_loop_map_[tensor->name] = {loops_};
    }
    loads_.push_back(*expr);
  }

  void Visit(const ir::For *op, Expr *expr) override {
    loops_.push_back(*expr);
    IRMutator::Visit(op, expr);
    loops_.pop_back();
  }

  void Visit(const ir::PolyFor *op, Expr *expr) override { LOG(FATAL) << "Unkown PolyFor!"; }

 public:
  std::vector<ir::Expr> stores_;
  std::vector<ir::Expr> loads_;
  std::vector<ir::Expr> loops_;

  std::unordered_map<std::string, LOOPS> store_buffer_loop_map_;
  std::unordered_map<std::string, std::vector<LOOPS>> load_buffer_loop_map_;
};

void UpdateBufferSizePass(ir::Expr *expr) {
  CollectTensorLoopVisitor collect_tensor_loop_visitor;
  collect_tensor_loop_visitor(expr);

  auto &stores          = collect_tensor_loop_visitor.stores_;
  auto &loads           = collect_tensor_loop_visitor.loads_;
  auto &store_loops_map = collect_tensor_loop_visitor.store_buffer_loop_map_;
  auto &load_loops_map  = collect_tensor_loop_visitor.load_buffer_loop_map_;

  for (auto expr : stores) {
    auto store = expr.As<ir::Store>();

    auto store_tensor = store->tensor.as_tensor_ref();
    if (!store_tensor->buffer.defined()) {
      continue;
    }

    if (store_tensor->buffer->memory_type == ir::MemoryType::Heap) {
      continue;
    }

    auto store_loops  = store_loops_map[store_tensor->name];
    auto load_loops_v = load_loops_map[store_tensor->name];

    int count = 0;
    for (auto load_loops : load_loops_v) {
      for (int idx = 0; idx < std::min(store_loops.size(), load_loops.size()); ++idx) {
        if (store_loops[idx] != load_loops[idx]) {
          count = std::max(count, idx);
          break;
        }
      }
    }
    for (int idx = 0; idx < count; ++idx) {
      auto loop_expr = store_loops[idx];
      auto loop_ir   = loop_expr.As<ir::For>();
      auto loop_var  = loop_ir->loop_var;

      optim::ReplaceVarWithExpr(&expr, loop_var, ir::Expr(0));
      for (auto tmp : loads) {
        auto load = tmp.As<ir::Load>();
        if (load->tensor.as_tensor_ref()->name != store_tensor->name) {
          continue;
        }
        optim::ReplaceVarWithExpr(&tmp, loop_var, ir::Expr(0));
      }
    }
  }
}

class SharedBufferVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store = expr->As<ir::Store>();
    if (!store->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (store->tensor.as_tensor_ref()->buffer->memory_type == ir::MemoryType::GPUShared) {
      for (auto axis : gpu_axis) {
        optim::ReplaceVarWithExpr(expr, ir::Var(axis), ir::Expr(0));
      }
    }
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (load->is_addr_scalar()) {
      return;
    }
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type == ir::MemoryType::GPUShared) {
      for (auto axis : gpu_axis) {
        optim::ReplaceVarWithExpr(expr, ir::Var(axis), ir::Expr(0));
      }
    }
  }

  const std::vector<std::string> gpu_axis = {"blockIdx.x", "blockIdx.y", "blockIdx.z"};
};

class ResizeBufferSizeVisitor : public ir::IRMutator<> {
 public:
  void operator()(ir::Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store *op, Expr *expr) override {
    auto store        = expr->As<ir::Store>();
    auto store_tensor = store->tensor.as_tensor_ref();
    if (!store_tensor->buffer.defined()) {
      return;
    }

    if (store_tensor->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }

    auto &indices = store->indices;
    auto &shape   = store_tensor->shape;
    auto &buffer  = store_tensor->buffer->shape;

    shape.clear();
    buffer.clear();
    // CHECK_EQ(shape.size(), indices.size());
    for (int idx = 0; idx < indices.size(); ++idx) {
      shape.push_back(ir::Expr(BuffeSize(indices[idx])));
      buffer.push_back(shape.back());
    }
  }

  void Visit(const ir::Load *op, Expr *expr) override {
    auto load = expr->As<ir::Load>();
    if (!load->tensor.as_tensor_ref()->buffer.defined()) {
      return;
    }

    if (load->tensor.as_tensor_ref()->buffer->memory_type == ir::MemoryType::Heap) {
      return;
    }

    load->tensor.as_tensor_ref()->shape = load->tensor.as_tensor_ref()->buffer->shape;
  }

  void Visit(const ir::For *op, Expr *expr) override {
    CHECK(expr->As<ir::For>());
    auto for_ir   = expr->As<ir::For>();
    auto var_name = for_ir->loop_var->name;
    auto extent_i = for_ir->extent;

    if (extent_i.is_constant()) loop_2_extent_[var_name] = extent_i.as_int32();
  }

  int BuffeSize(ir::Expr indice) {
    auto copy = IRCopy(indice);
    auto vars = ir::CollectIRNodesInOrder(copy, [](const ir::Expr *expr) { return expr->As<ir::_Var_>(); });

    int max_range = 1;
    // using recursion funcitons index range.
    std::function<void(int, ir::Expr)> compute_range = [&](const int deep, ir::Expr index) {
      auto var = vars[deep].as_var_ref();
      CHECK(loop_2_extent_.count(var->name)) << var->name;
      auto extent = loop_2_extent_.find(var->name)->second;

      for (int idx = 0; idx < extent; ++idx) {
        auto tmp = IRCopy(index);
        ReplaceVarWithExpr(&tmp, var, Expr(idx));

        if (deep == vars.size() - 1) {
          auto simplify = common::AutoSimplify(tmp);
          auto range    = common::AutoSimplify(simplify);
          CHECK(range.is_constant());
          max_range = std::max(max_range, range.as_int32() + 1);
        } else {
          compute_range(deep + 1, tmp);
        }
      }
    };

    if (vars.size()) compute_range(0, copy);
    return max_range;
  }

  std::unordered_map<std::string, int> loop_2_extent_;
};

void OptimizeExprGPU(Expr *expr) {
  VLOG(3) << "Before Optimize Expr:\n" << *expr;
  // replace var name with block/thread
  ReplaceLoopVarToGpu replace_loop_var_to_gpu;
  replace_loop_var_to_gpu(expr);

  // replace var to bind expr
  ReplaceIndexToBindExpr replace_index_to_bind_expr;
  replace_index_to_bind_expr(expr);

  // resize buffer size
  UpdateBufferSizePass(expr);

  // resize shared buffer size
  SharedBufferVisitor shared_buffer_visitor;
  shared_buffer_visitor(expr);

  ResizeBufferSizeVisitor resize_buffer_size_visitor;
  resize_buffer_size_visitor(expr);
  VLOG(3) << "After Optimize Expr: \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
