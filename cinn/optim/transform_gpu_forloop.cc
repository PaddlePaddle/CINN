#include "cinn/optim/transform_gpu_forloop.h"

#include <map>
#include <stack>
#include <string>
#include <vector>

#include "cinn/backends/cuda_util.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/poly/stage.h"

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
 *   3. if the same gpu axis exists in multiple forloop, remain the condition of each forloop.
 *
 * @param expr The expression to mutate.
 */
void RemoveGpuForloopsAxis(Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    std::map<std::string, int> gpu_axis_num_forloops;

    void operator()(Expr *expr) {
      if (!expr->As<ir::_LoweredFunc_>()) {
        LOG(ERROR) << "The outermost should be a _LoweredFunc_ node, so that we can register "
                      "the GPU kernal dimension information there.";
        return;
      }

      CountGpuAxisForloops(expr);

      cur_func_ = expr->As<ir::_LoweredFunc_>();
      ir::IRMutator<>::Visit(expr, expr);
    }

   private:
    void Visit(const ir::For *op, Expr *expr) override {
      LOG(INFO) << "processing: \n" << *expr;
      int dim = 0;
      switch (op->for_type()) {
        case ir::ForType::GPUBlock:
          dim = GpuAxisGetExtent(op->extent);
          CHECK_GT(dim, 0) << "Invalid dimension found " << dim;
          // TODO(Superjomn) Support multiple block dimensions
          cur_func_->gpu_grid_dims.push_back(dim);
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          cur_func_->gpu_grid_dims.resize(3, 1);
          break;
        case ir::ForType::GPUThread:
          dim = GpuAxisGetExtent(op->extent);
          CHECK_GT(dim, 0) << "Invalid dimension found " << dim;
          // TODO(Superjomn) Support multiple block dimensions
          cur_func_->gpu_block_dims.push_back(dim);
          if (NeedToReplaceForloopWithIfThenElse(op)) {
            ReplaceForloopWithIfThenElse(expr);
          } else {
            *expr = op->body;
          }
          IRMutator<>::Visit(expr, expr);
          cur_func_->gpu_block_dims.resize(3, 1);
          break;
        default:
          auto *node = expr->As<ir::For>();
          IRMutator<>::Visit(&node->body, &node->body);
          break;
      }
    }

    void CountGpuAxisForloops(Expr *expr) {
      auto fors = ir::CollectIRNodes(*expr, [](const Expr *x) {
        auto *for_n = x->As<ir::For>();
        return for_n && (for_n->for_type() == ir::ForType::GPUBlock || for_n->for_type() == ir::ForType::GPUThread);
      });

      for (auto &forn : fors) {
        gpu_axis_num_forloops[forn.As<ir::For>()->loop_var->name]++;
      }
    }

    bool NeedToReplaceForloopWithIfThenElse(const ir::For *n) const {
      // We need something like: `threadIdx.x > j condition`.
      if (n->min != common::make_const(0)) {
        return true;
      }

      if (n->extent.As<ir::Min>()) return true;

      // if a gpu axis exists in multiple forloops, the condition should remain.
      auto it = gpu_axis_num_forloops.find(n->loop_var->name);
      return it != gpu_axis_num_forloops.end() && it->second > 1;
    }

    void ReplaceForloopWithIfThenElse(Expr *expr) {
      auto *for_n = expr->As<ir::For>();
      CHECK(for_n);

      Expr condition;

      auto condition_append = [&](Expr new_cond) {
        if (condition.defined()) {
          condition = ir::And::Make(condition, new_cond);
        } else {
          condition = new_cond;
        }
      };

      // for(i, 2, 100);
      //        ^
      if (for_n->min != common::make_const(0)) {
        condition_append(ir::GE::Make(for_n->loop_var, for_n->min));
      }

      // for(i, 2, min(M/2, 20)
      //            ^
      condition_append(ir::LT::Make(for_n->loop_var, for_n->extent));

      CHECK(condition.defined());

      VLOG(3) << "GPU replacing\n" << *expr;
      VLOG(3) << "\nto\n";
      auto if_n = ir::IfThenElse::Make(condition, for_n->body);
      VLOG(3) << if_n;
      *expr = if_n;
    }

    int GpuAxisGetExtent(Expr v) {
      auto *v_int = v.As<ir::IntImm>();
      auto *v_min = v.As<ir::Min>();
      CHECK(v_int || v_min) << "We deduce the GPU block or grid dimensions from the domain of GPU Axis, can only "
                               "accept IntImm or Min node with a IntImm nodes, but get "
                            << v;
      if (v_int) return v_int->value;

      // min
      auto *min_a_int = v_min->a().As<ir::IntImm>();
      auto *min_b_int = v_min->b().As<ir::IntImm>();
      CHECK(min_a_int || min_b_int) << "At least one operand of Min should be IntImm so that we can deduce the "
                                       "dimension";
      if (min_a_int) return min_a_int->value;
      return min_b_int->value;
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

/**
 * This replace the axis and vars in for if its iterators equals \p iterator.
 * @param for_expr
 * @param iterator
 * @param to
 */
void ReplaceForAxis(Expr *for_expr, Var iterator, Var to);

void MarkGpuForloop(const std::string &statement,
                    const std::map<std::string, poly::StageForloopInfo> &forloop_infos,
                    Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    const std::string &statement;
    const std::map<std::string, poly::StageForloopInfo> forloop_infos;

    Mutator(const std::string &statement, const std::map<std::string, poly::StageForloopInfo> &forloop_infos)
        : statement(statement), forloop_infos(forloop_infos) {}

    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
    void Visit(const ir::Store *op, Expr *expr) override {
      auto *tensor = op->tensor.As<ir::_Tensor_>();
      if (tensor->name == statement) {
        MarkForloop();
      }
    }
    void Visit(const ir::_LoweredFunc_ *op, Expr *expr) override { ir::IRMutator<>::Visit(op, expr); }

    void MarkForloop() {
      // start from 0, threadIdx.x
      int thread_level = 0;
      int block_level  = 0;
      for (auto *expr : forloops) {
        auto *for_n      = expr->As<ir::For>();
        auto *poly_for_n = expr->As<ir::PolyFor>();
        Var axis_var     = for_n ? for_n->loop_var : poly_for_n->iterator;
        auto it          = forloop_infos.find(axis_var->name);
        if (it != forloop_infos.end()) {
          if (for_n) {
            for_n->set_for_type(it->second.for_type);
            for_n->device_api = it->second.device;
          } else {
            poly_for_n->set_for_type(it->second.for_type);
            poly_for_n->device_api = it->second.device;
          }

          if (it->second.for_type == ir::ForType::GPUThread) {
            Var cuda_var(backends::cuda_thread_axis_name(thread_level++));
            Expr var_expr(cuda_var);
            VLOG(3) << "gpu replacing var " << axis_var << " to " << var_expr;
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
          } else if (it->second.for_type == ir::ForType::GPUBlock) {
            Var cuda_var(backends::cuda_block_axis_name(block_level++));
            Expr var_expr(cuda_var);
            VLOG(3) << "gpu replacing var " << axis_var << " to " << var_expr;
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
          } else if (it->second.for_type == ir::ForType::GPULane) {
            NOT_IMPLEMENTED
          }
        }
      }
    }

    void Visit(const ir::For *op, Expr *expr) override {
      forloops.push_back(expr);
      IRMutator::Visit(op, expr);
      forloops.pop_back();
    }
    void Visit(const ir::PolyFor *op, Expr *expr) override {
      forloops.push_back(expr);
      IRMutator::Visit(op, expr);
      forloops.pop_back();
    }

    std::vector<Expr *> forloops;
  };

  Mutator mutator(statement, forloop_infos);
  mutator(expr);
}

void TransformGpuForloop(const forloop_infos_t &forloop_infos, Expr *expr) {
  for (auto &item : forloop_infos) {
    MarkGpuForloop(item.first, item.second, expr);
  }
}

}  // namespace optim
}  // namespace cinn
