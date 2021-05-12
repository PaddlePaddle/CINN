#include "cinn/optim/transform_gpu_forloop.h"

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
#include "cinn/poly/stage.h"
#include "cinn/runtime/intrinsic.h"

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
    void operator()(Expr *expr) {
      if (!expr->As<ir::_LoweredFunc_>()) {
        LOG(ERROR) << "The outermost should be a _LoweredFunc_ node, so that we can register "
                      "the GPU kernal dimension information there.";
        return;
      }
      cur_func_ = expr->As<ir::_LoweredFunc_>();
      ir::IRMutator<>::Visit(expr, expr);
    }

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

    ir::_LoweredFunc_ *cur_func_{};
  };

  Mutator mutator;
  mutator(expr);
}

void MarkGpuForloop(const std::string &statement,
                    const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
                    std::map<std::string, ir::Tensor> *global_tensor_map,
                    std::unordered_set<std::string> &resized_buffer,
                    Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    const std::string &statement;
    const std::map<std::string, poly::StageForloopInfo> forloop_infos;
    std::map<std::string, ir::Tensor> *global_tensor_map;
    std::unordered_set<std::string> &resized_buffer;
    /**
     * @param statement the tuple name.
     * @param forloop_infos the axis.
     */
    Mutator(const std::string &statement,
            const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
            std::map<std::string, ir::Tensor> *global_tensor_map,
            std::unordered_set<std::string> &resized_buffer)
        : statement(statement),
          forloop_infos(forloop_infos),
          global_tensor_map(global_tensor_map),
          resized_buffer(resized_buffer) {}

    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    // Mark the specific store.
    void Visit(const ir::Store *op, Expr *expr) override {
      auto *tensor = op->tensor.As<ir::_Tensor_>();
      if (tensor->name == statement) {
        MarkForloop();
      }
    }

    void MarkForloop() {
      // start from 0, threadIdx.x
      for (auto *expr : forloop_stack) {
        auto *for_     = expr->As<ir::For>();
        auto *poly_for = expr->As<ir::PolyFor>();
        Var axis_var   = for_ ? for_->loop_var : poly_for->iterator;
        auto it        = forloop_infos.find(axis_var->name);
        std::string iterator_name;
        if (it != forloop_infos.end()) {
          if (for_) {
            for_->set_for_type(it->second.for_type);
            for_->device_api = it->second.device;
            iterator_name    = for_->loop_var->name;
          } else {
            poly_for->set_for_type(it->second.for_type);
            poly_for->device_api = it->second.device;
            iterator_name        = poly_for->iterator->name;
          }

          auto &forloop_info = forloop_infos.at(iterator_name);
          VLOG(2) << "Statement of for loop is : " << statement;
          if (it->second.for_type == ir::ForType::GPUThread) {
            Var cuda_var(backends::cuda_thread_axis_name(forloop_info.offset));
            Expr var_expr(cuda_var);
            VLOG(2) << "gpu replacing var " << axis_var->name << " to " << cuda_var->name;
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
            Expr extent = for_ ? for_->extent : poly_for->ExtractExtent();
            VLOG(2) << "gpu replacing var " << cuda_var->name << " to Expr(0)";
            optim::CUDAReplaceIndexOfCachePass(
                expr, var_expr, ir::Expr(0), global_tensor_map, resized_buffer, false, extent);
          } else if (it->second.for_type == ir::ForType::GPUBlock) {
            Var cuda_var(backends::cuda_block_axis_name(forloop_info.offset));
            Expr var_expr(cuda_var);
            VLOG(2) << "gpu replacing var " << axis_var->name << " to " << cuda_var->name;
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
            Expr extent = for_ ? for_->extent : poly_for->ExtractExtent();
            VLOG(3) << "gpu replacing var " << cuda_var->name << " to Expr(0)";
            optim::CUDAReplaceIndexOfCachePass(
                expr, var_expr, ir::Expr(0), global_tensor_map, resized_buffer, true, extent);
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

  Mutator mutator(statement, forloop_infos, global_tensor_map, resized_buffer);
  mutator(expr);
}

void TransformGpuForloops(const forloop_infos_t &forloop_infos,
                          std::map<std::string, ir::Tensor> *global_tensor_map,
                          std::unordered_set<std::string> &resized_buffer,
                          Expr *expr) {
  for (auto &item : forloop_infos) {
    MarkGpuForloop(item.first, item.second, global_tensor_map, resized_buffer, expr);
  }
}

ir::CudaAxisInfo GatherAxisInfoFromStages(const std::vector<poly::Stage *> &stage_group) {
  std::map<std::pair<ir::ForType, uint8_t>, int> gpu_axis_range;
  for (auto *stage : stage_group) {
    for (auto &item : stage->forloop_infos()) {
      int level               = stage->GetTransformedLevel(item.first);
      auto [min_val, max_val] = poly::isl_set_get_axis_range(stage->transformed_domain().get(), level);
      auto key                = std::make_pair(item.second.for_type, item.second.offset);
      gpu_axis_range[key]     = std::max(max_val.get_num_si() + 1, static_cast<int64_t>(gpu_axis_range[key]));
    }
  }

  ir::CudaAxisInfo info;
  for (auto &item : gpu_axis_range) {
    switch (item.first.first) {
      case ir::ForType::GPUBlock:
        info.set_grid_dim(item.first.second, item.second);
        break;
      case ir::ForType::GPUThread:
        info.set_block_dim(item.first.second, item.second);
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

}  // namespace optim
}  // namespace cinn
