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
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
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

void MarkGpuForloop(const std::string &statement,
                    const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
                    const std::set<std::string> gpu_launch_axis,
                    std::map<std::string, ir::Tensor> *global_tensor_map,
                    std::unordered_map<std::string, std::vector<Expr>> &resized_buffer_cache,
                    Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    const std::string &statement;
    const std::map<std::string, poly::StageForloopInfo> forloop_infos;
    std::map<std::string, ir::Tensor> *global_tensor_map;
    std::set<std::string> gpu_launch_axis;
    std::unordered_map<std::string, std::vector<Expr>> &resized_buffer_cache;
    /**
     * @param statement the tuple name.
     * @param forloop_infos the axis.
     */
    Mutator(const std::string &statement,
            const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
            std::set<std::string> gpu_launch_axis,
            std::map<std::string, ir::Tensor> *global_tensor_map,
            std::unordered_map<std::string, std::vector<Expr>> &resized_buffer_cache)
        : statement(statement),
          forloop_infos(forloop_infos),
          gpu_launch_axis(gpu_launch_axis),
          global_tensor_map(global_tensor_map),
          resized_buffer_cache(resized_buffer_cache) {}

    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    // Mark the specific store.
    void Visit(const ir::Store *op, Expr *expr) override {
      auto *tensor = op->tensor.As<ir::_Tensor_>();
      if (tensor->name == statement) {
        if (tensor->buffer.defined()) {
          MarkForloop(tensor->buffer->name);
        } else {
          MarkForloop("not_defined");
        }
        // if compute A bind threadIdx.x, and set launch grid param `threadIdx.x` to 32 and
        // compute B bind threadIdx.y, blockIdx.x and set launch grid param to 32, 16.
        // assume B has a store operation buf[blockIdx.x * 32 + threadIdx.y] = ...,
        // if A is computed in B's, then the generated fused kernel has 0 < threadIdx.x < 32,
        // 0 < threadIdx.y < 32, 0 < blockIdx.x < 16. we should limit the store operation in B
        // to run only once, simply let the threadIdx.x = 0 thread to run it.
        // compute_A {
        //   for (threadIdx.x, 0, 32) {
        //     buf_A[threadIdx.x] = .....;
        //   }
        // }
        // compute_B {
        //  for (blockIdx.x, 0, 16} {
        //    for (threadIdx.y, 0, 32) {
        //      buf_B[block_Idx.x * 32 + threadIdx.y] = buf_A[threadIdx.y];
        //    }
        //  }
        // }
        //
        // compute_A_in_B {
        //  for (blockIdx.x, 0, 16} {
        //    for (threadIdx.y, 0, 32) {
        //      for (threadIdx.x, 0, 32) {
        //        buf_A[threadIdx.x] = .....;
        //      }
        //      // when launched the store operation will run 32 times
        //      buf_B[block_Idx.x * 32 + threadIdx.y] = buf_A[threadIdx.y];
        //    }
        //  }
        // }
        //
        // compute_A_in_B {
        //  for (blockIdx.x, 0, 16} {
        //    for (threadIdx.y, 0, 32) {
        //      for (threadIdx.x, 0, 32) {
        //        buf_A[threadIdx.x] = .....;
        //      }
        //      if (threadIdx.x == 0) {
        //        // make sure only one thread runs this store operation
        //        buf_B[block_Idx.x * 32 + threadIdx.y] = buf_A[threadIdx.y];
        //      }
        //    }
        //  }
        // }
        std::set<std::string> gpu_axis = {
            "blockIdx.x", "blockIdx.y", "blockIdx.z", "threadIdx.x", "threadIdx.y", "threadIdx.z"};
        std::set<std::string> loop_gpu_axis;
        for (auto *expr : forloop_stack) {
          auto *for_     = expr->As<ir::For>();
          auto *poly_for = expr->As<ir::PolyFor>();
          Var axis_var   = for_ ? for_->loop_var : poly_for->iterator;
          if (gpu_axis.find(axis_var->name) != gpu_axis.end()) {
            loop_gpu_axis.insert(axis_var->name);
          }
        }
        Expr condition;
        auto condition_append = [&](Expr new_cond) {
          if (condition.defined()) {
            condition = ir::And::Make(condition, new_cond);
          } else {
            condition = new_cond;
          }
        };
        for (auto &axis : gpu_launch_axis) {
          if (loop_gpu_axis.find(axis) == loop_gpu_axis.end()) {
            Var cuda_var(axis);
            condition_append(ir::EQ::Make(Expr(cuda_var), Expr(0)));
          }
        }
        if (condition.defined()) {
          *expr = ir::IfThenElse::Make(condition, *expr);
        }
      }
    }

    void MarkForloop(const std::string &tensor_name) {
      // start from 0, threadIdx.x
      VLOG(3) << "input tensor_name=" << tensor_name;
      std::map<std::string, int> loop2extent;
      for (auto *expr : forloop_stack) {
        auto *for_     = expr->As<ir::For>();
        auto *poly_for = expr->As<ir::PolyFor>();
        Var axis_var   = for_ ? for_->loop_var : poly_for->iterator;
        auto loop_name = axis_var->name;
        Expr extent    = for_ ? for_->extent : poly_for->ExtractExtent();
        if (!extent.defined() || !extent.is_constant()) {
          continue;
        }
        int extent_i           = extent.get_constant();
        loop2extent[loop_name] = extent_i;
        VLOG(3) << "collect loop=" << loop_name << ", extent=" << extent_i;
      }

      for (auto *expr : forloop_stack) {
        auto *for_     = expr->As<ir::For>();
        auto *poly_for = expr->As<ir::PolyFor>();
        Var axis_var   = for_ ? for_->loop_var : poly_for->iterator;
        auto it        = forloop_infos.find(axis_var->name);
        VLOG(3) << "loop var=" << axis_var->name << ", content:\n" << *expr;
        std::string iterator_name;
        if (it != forloop_infos.end()) {
          if (for_) {
            for_->set_for_type(it->second.for_type);
            for_->device_api = it->second.device;
            iterator_name    = for_->loop_var->name;
            VLOG(2) << "iterator_name=" << iterator_name << ",body:\n" << for_->body;
          } else {
            poly_for->set_for_type(it->second.for_type);
            poly_for->device_api = it->second.device;
            iterator_name        = poly_for->iterator->name;
            VLOG(2) << "In this poly_for loop, condition is : " << poly_for->condition;
            VLOG(2) << "In this poly_for loop, body is : " << poly_for->body;
          }

          auto &forloop_info = forloop_infos.at(iterator_name);
          if (it->second.for_type == ir::ForType::GPUThread) {
            Var cuda_var(backends::cuda_thread_axis_name(forloop_info.offset));
            Expr var_expr(cuda_var);
            VLOG(2) << "gpu replacing var " << axis_var->name << " to " << cuda_var->name;
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
            Expr extent = for_ ? for_->extent : poly_for->ExtractExtent();
            VLOG(2) << "gpu replacing var " << cuda_var->name << " to Expr(0)";
            optim::CUDAReplaceIndexOfCachePass(
                expr, var_expr, ir::Expr(0), global_tensor_map, resized_buffer_cache, false, extent, "", loop2extent);
          } else if (it->second.for_type == ir::ForType::GPUBlock) {
            Var cuda_var(backends::cuda_block_axis_name(forloop_info.offset));
            Expr var_expr(cuda_var);
            VLOG(2) << "gpu replacing var " << axis_var->name << " to " << cuda_var->name;
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
            VLOG(2) << "After that, expr is : " << *expr;
            Expr extent = for_ ? for_->extent : poly_for->ExtractExtent();
            VLOG(2) << "gpu replacing var " << cuda_var->name << " to Expr(0)";
            optim::CUDAReplaceIndexOfCachePass(
                expr, var_expr, ir::Expr(0), global_tensor_map, resized_buffer_cache, true, extent, "", loop2extent);
            VLOG(2) << "After that, expr is : " << *expr;
          } else if (it->second.for_type == ir::ForType::Default) {
            Expr extent = for_ ? for_->extent : poly_for->ExtractExtent();
            VLOG(2) << "ComputeAt replacing var " << axis_var->name << " to Expr(0) in tensor " << tensor_name;
            optim::CUDAReplaceIndexOfCachePass(expr,
                                               axis_var,
                                               ir::Expr(0),
                                               global_tensor_map,
                                               resized_buffer_cache,
                                               false,
                                               extent,
                                               tensor_name,
                                               loop2extent);
          } else {
            CINN_NOT_IMPLEMENTED
          }
        }
      }
    }

    void Visit(const ir::For *op, Expr *expr) override {
      forloop_stack.push_back(expr);
      IRMutator::Visit(op, expr);
      forloop_stack.pop_back();
    }
    void Visit(const ir::PolyFor *op, Expr *expr) override {
      forloop_stack.push_back(expr);
      IRMutator::Visit(op, expr);
      forloop_stack.pop_back();
    }

    std::vector<Expr *> forloop_stack;
  };

  Mutator mutator(statement, forloop_infos, gpu_launch_axis, global_tensor_map, resized_buffer_cache);
  mutator(expr);
}

void TransformGpuForloops(const forloop_infos_t &forloop_infos,
                          const std::vector<std::string> &traverse_order,
                          std::map<std::string, ir::Tensor> *global_tensor_map,
                          std::unordered_map<std::string, std::vector<Expr>> &resized_buffer_cache,
                          Expr *expr) {
  VLOG(3) << "traverse_order=" << utils::Join(traverse_order, ",");
  std::set<std::string> gpu_launch_axis;
  for (auto &i : traverse_order) {
    if (forloop_infos.count(i) == 0) continue;
    for (auto &f : forloop_infos.at(i)) {
      if (f.second.for_type == ir::ForType::GPUThread) {
        gpu_launch_axis.insert(backends::cuda_thread_axis_name(f.second.offset));
      } else if (f.second.for_type == ir::ForType::GPUBlock) {
        gpu_launch_axis.insert(backends::cuda_block_axis_name(f.second.offset));
      }
    }
  }
  for (auto &i : traverse_order) {
    if (forloop_infos.count(i) == 0) continue;
    MarkGpuForloop(i, forloop_infos.at(i), gpu_launch_axis, global_tensor_map, resized_buffer_cache, expr);
  }
}

ir::CudaAxisInfo GatherAxisInfoFromStages(const std::vector<poly::Stage *> &stage_group) {
  std::map<std::pair<ir::ForType, uint8_t>, int> gpu_axis_range;
  ir::CudaAxisInfo info;
  for (auto *stage : stage_group) {
    if (stage->IfCudaBind()) info.set_valid(true);
    for (auto &item : stage->forloop_infos()) {
      if (item.first < 0) continue;
      int level = poly::isl_get_original_axes_from_optimized_level(stage->transformed_domain().get(), item.first);
      auto _min_val_max_val_ = poly::isl_set_get_axis_range(stage->transformed_domain().get(), level);
      auto &min_val          = std::get<0>(_min_val_max_val_);
      auto &max_val          = std::get<1>(_min_val_max_val_);
      auto key               = std::make_pair(item.second.for_type, item.second.offset);
      gpu_axis_range[key]    = std::max(max_val.get_num_si() + 1, static_cast<int64_t>(gpu_axis_range[key]));
    }
  }
  for (auto &item : gpu_axis_range) {
    switch (item.first.first) {
      case ir::ForType::GPUBlock:
        info.set_grid_dim(item.first.second, item.second);
        break;
      case ir::ForType::GPUThread:
        info.set_block_dim(item.first.second, item.second);
        break;
      case ir::ForType::Default:
        break;
      default:
        CINN_NOT_IMPLEMENTED
    }
  }

  return info;
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

void OptimizeExprGPU(Expr *expr) {
  VLOG(3) << "input Expr:\n" << *expr;
  std::vector<std::string> tensor_traverse_order;
  std::map<std::string, ir::Tensor> global_tensor_map;
  auto find_expr = ir::CollectIRNodesWithoutTensor(*expr, [&](const Expr *x) {
    if (x->As<ir::Store>() && global_tensor_map.count(x->As<ir::Store>()->tensor.as_tensor_ref()->name) == 0) {
      tensor_traverse_order.push_back(x->As<ir::Store>()->tensor.as_tensor_ref()->name);
      global_tensor_map[x->As<ir::Store>()->tensor.as_tensor_ref()->name] = x->As<ir::Store>()->tensor.as_tensor_ref();
    }
    return x->As<ir::Store>();
  });
  std::reverse(tensor_traverse_order.begin(), tensor_traverse_order.end());
  forloop_infos_t forloop_infos;

  // Add loops bound by gpu axis in for_loop_info.
  auto find_bind_forloops = ir::CollectIRNodesWithoutTensor(
      *expr, [&](const Expr *x) { return x->As<ir::For>() && x->As<ir::For>()->bind_info().valid(); });
  for (auto &for_loop : find_bind_forloops) {
    VLOG(3) << "format bind loop:\n" << for_loop;
    std::string for_iter = for_loop.As<ir::For>()->loop_var->name;
    poly::StageForloopInfo for_loop_info;
    for_loop_info.for_type = for_loop.As<ir::For>()->bind_info().for_type;
    for_loop_info.offset   = for_loop.As<ir::For>()->bind_info().offset;
    for_loop_info.device   = for_loop.As<ir::For>()->bind_info().device;
    auto find_store        = ir::CollectIRNodesWithoutTensor(for_loop, [&](const Expr *x) {
      if (x->As<ir::Store>()) {
        forloop_infos[x->As<ir::Store>()->tensor.as_tensor_ref()->name][for_iter] = for_loop_info;
        VLOG(3) << "Found tensor=" << x->As<ir::Store>()->tensor.as_tensor_ref()->name << " in for_iter=" << for_iter;
      }
      return x->As<ir::Store>();
    });
  }

  // Add loops not bound by gpu axis in for_loop_info, used for recalculating the size of cache buffer after ComputeAt.
  auto find_no_bind_forloops = ir::CollectIRNodesWithoutTensor(
      *expr, [&](const Expr *x) { return x->As<ir::For>() && !x->As<ir::For>()->bind_info().valid(); });
  for (auto &for_loop : find_no_bind_forloops) {
    VLOG(3) << "format no bind loop:\n" << for_loop;
  }

  auto find_schedule_block_realize =
      ir::CollectIRNodesWithoutTensor(*expr, [&](const Expr *x) { return x->As<ir::ScheduleBlockRealize>(); });
  for (auto &schedule_block_expr : find_schedule_block_realize) {
    auto schedule_block_realize            = schedule_block_expr.As<ir::ScheduleBlockRealize>();
    auto schedule_block                    = schedule_block_realize->schedule_block.As<ir::ScheduleBlock>();
    auto iter_vars                         = schedule_block->iter_vars;
    auto iter_values                       = schedule_block_realize->iter_values;
    auto compute_at_extra_var_iter         = schedule_block->attrs.find(ir::attr::compute_at_extra_var);
    auto reverse_compute_at_extra_var_iter = schedule_block->attrs.find(ir::attr::reverse_compute_at_extra_var);
    if (compute_at_extra_var_iter == schedule_block->attrs.end() &&
        reverse_compute_at_extra_var_iter == schedule_block->attrs.end())
      continue;
    std::vector<std::string> compute_at_extra_var, reverse_compute_at_extra_var;
    // Get the names of vars that cannot be simplified after ComputeAt.
    // The loop corresponding to these axes is related to the size of the cache buffer.
    if (compute_at_extra_var_iter != schedule_block->attrs.end()) {
      compute_at_extra_var =
          utils::Split(absl::get<std::string>(schedule_block->attrs.at(ir::attr::compute_at_extra_var)), ",");
    }
    // Get the names of vars that cannot be simplified after ReverseComputeAt.
    // The loop corresponding to these axes is related to the size of the cache buffer.
    if (reverse_compute_at_extra_var_iter != schedule_block->attrs.end()) {
      reverse_compute_at_extra_var =
          utils::Split(absl::get<std::string>(schedule_block->attrs.at(ir::attr::reverse_compute_at_extra_var)), ",");
    }

    auto temp_buffer_store = ir::CollectIRNodesWithoutTensor(schedule_block_expr, [&](const Expr *x) {
      return (x->As<ir::Store>() &&
              utils::Endswith(x->As<ir::Store>()->tensor.as_tensor()->buffer->name, "temp_buffer"));
    });
    auto temp_buffer_load  = ir::CollectIRNodesWithoutTensor(schedule_block_expr, [&](const Expr *x) {
      return (x->As<ir::Load>() && utils::Endswith(x->As<ir::Load>()->tensor.as_tensor()->buffer->name, "temp_buffer"));
    });

    auto find_for_loop_info_func =
        [&](const std::set<ir::Expr> &temp_buffer_ops, const std::vector<std::string> &extra_var, bool is_load) {
          // Traversing the operation nodes(Load or Store node) of the temp buffer.
          for (auto &buf_op : temp_buffer_ops) {
            std::vector<cinn::ir::Expr> indices;
            if (is_load) {
              CHECK(buf_op.As<ir::Load>());
              indices = buf_op.As<ir::Load>()->indices;
            } else {
              CHECK(buf_op.As<ir::Store>());
              indices = buf_op.As<ir::Store>()->indices;
            }

            // 'iter_vars' are stored in ScheduleBlock node, through it, we can find the corresponding iter_values in
            // the information of ScheduleBlockRealize.
            // 'indices' are stored in Load and Store node, means subscript for
            // operating the temp buffer.
            // For each indice that operates the temp buffer, we find an iter_var with the
            // same name, then we can find the corresponding iter_values and loops.
            for (int i = 0; i < iter_vars.size(); ++i) {
              for (auto &ind : indices) {
                if (ind.as_var_ref()->name == iter_vars[i]->name) {
                  auto iter_value     = iter_values.at(i);
                  auto vars_in_indice = ir::CollectIRNodes(iter_value, [&](const Expr *x) { return x->as_var(); });
                  for (auto &var_in_indice : vars_in_indice) {
                    std::string var_name = var_in_indice.as_var_ref()->name;

                    bool is_in_extra_var = false;
                    // Filter out vars that cannot be eliminated,
                    // these vars are stored in schedule_block->attrs with name "[reverse_]compute_at_extra_var",
                    // and are calculated and stored when calling the [Reverse]ComputeAt primitive.
                    for (const auto &ev : extra_var) {
                      if (var_name.find(ev) != std::string::npos) {
                        is_in_extra_var = true;
                        break;
                      }
                    }
                    // find the loops that can be eliminated, which are the outer loops of caculation of the temp
                    // buffer. These loops are in one thread and do not need to participate in the calculation of the
                    // temp buffer.
                    auto find_loop_with_var_name = std::find_if(
                        find_no_bind_forloops.begin(), find_no_bind_forloops.end(), [&var_name](const ir::Expr &e) {
                          return e.As<ir::For>()->loop_var->name == var_name;
                        });
                    if (!is_in_extra_var && find_loop_with_var_name != find_no_bind_forloops.end()) {
                      poly::StageForloopInfo for_loop_info;
                      for_loop_info.for_type = find_loop_with_var_name->As<ir::For>()->bind_info().for_type;
                      for_loop_info.offset   = find_loop_with_var_name->As<ir::For>()->bind_info().offset;
                      for_loop_info.device   = find_loop_with_var_name->As<ir::For>()->bind_info().device;
                      if (is_load) {
                        forloop_infos[buf_op.As<ir::Load>()->tensor.as_tensor_ref()->name][var_name] = for_loop_info;
                      } else {
                        forloop_infos[buf_op.As<ir::Store>()->tensor.as_tensor_ref()->name][var_name] = for_loop_info;
                      }
                    }
                  }
                }
              }
            }
          }
        };

    find_for_loop_info_func(temp_buffer_store, compute_at_extra_var, false);
    find_for_loop_info_func(temp_buffer_load, reverse_compute_at_extra_var, true);
  }

  std::unordered_map<std::string, std::vector<Expr>> resized_buffer_cache;
  TransformGpuForloops(forloop_infos, tensor_traverse_order, &global_tensor_map, resized_buffer_cache, expr);
  VLOG(3) << "After TransformGpuForloops Expr: \n" << *expr;
}

}  // namespace optim
}  // namespace cinn
