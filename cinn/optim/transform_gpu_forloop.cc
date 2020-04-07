#include "cinn/optim/transform_gpu_forloop.h"

#include <map>
#include <stack>
#include <string>

#include "cinn/ir/ir_mutator.h"
#include "cinn/optim/replace_var_with_expr.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace optim {

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
          cur_func_->gpu_block_dims.push_back(op->extent.as_int32());
          *expr = op->body;
          IRMutator<>::Visit(expr, expr);
          break;
        case ir::ForType::GPUThread:
          cur_func_->gpu_thread_dims.push_back(op->extent.as_int32());
          *expr = op->body;
          IRMutator<>::Visit(expr, expr);
          break;
        default:
          auto *node = expr->As<ir::For>();
          IRMutator<>::Visit(&node->body, &node->body);
          break;
      }
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
                    Expr *expr) {
  struct Mutator : public ir::IRMutator<Expr *> {
    const std::string &statement;
    const std::map<std::string, poly::StageForloopInfo> forloop_infos;

    Mutator(const std::string &statement, const std::map<std::string, poly::StageForloopInfo> &forloop_infos)
        : statement(statement), forloop_infos(forloop_infos) {}

    void operator()(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

   private:
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
    void Visit(const ir::Store *op, Expr *expr) override {
      auto *tensor = op->tensor.As<ir::_Tensor_>();
      if (tensor->name == statement) {
        MarkForloop();
      }
    }

    void MarkForloop() {
      // start from 0, threadIdx.x
      int thread_level = 0;
      int block_level  = 0;
      for (auto *expr : forloops) {
        auto *for_     = expr->As<ir::For>();
        auto *poly_for = expr->As<ir::PolyFor>();
        Var axis_var   = for_ ? for_->loop_var : poly_for->iterator;
        auto it        = forloop_infos.find(axis_var->name);
        if (it != forloop_infos.end()) {
          if (for_) {
            for_->set_for_type(it->second.for_type);
            for_->device_api = it->second.device;
          } else {
            poly_for->set_for_type(it->second.for_type);
            poly_for->device_api = it->second.device;
          }

          if (it->second.for_type == ir::ForType::GPUThread) {
            Var cuda_var(cuda_thread_axis_name(thread_level++));
            Expr var_expr(cuda_var);
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
          } else if (it->second.for_type == ir::ForType::GPUBlock) {
            Var cuda_var(cuda_block_axis_name(block_level++));
            Expr var_expr(cuda_var);
            optim::ReplaceVarWithExpr(expr, axis_var, var_expr);
          } else if (it->second.for_type == ir::ForType::GPULane) {
            NOT_IMPLEMENTED
          }
        }
      }
    }

    std::string cuda_thread_axis_name(int level) {
      switch (level) {
        case 0:
          return "threadIdx.x";
          break;
        case 1:
          return "threadIdx.y";
          break;
        case 2:
          return "threadIdx.z";
          break;
      }
      return "";
    }

    std::string cuda_block_axis_name(int level) {
      switch (level) {
        case 0:
          return "blockIdx.x";
          break;
        case 1:
          return "blockIdx.y";
          break;
        case 2:
          return "blockIdx.z";
          break;
      }
      return "";
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
