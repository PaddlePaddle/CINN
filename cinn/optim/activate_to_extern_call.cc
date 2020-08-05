#include "cinn/optim/activate_to_extern_call.h"

#include "cinn/cinn.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace optim {

void ActivateToExternCall(Expr *e) {
  struct Mutator : ir::IRMutator<Expr *> {
    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Activate *op, Expr *expr) override {
      auto *node = expr->As<ir::Activate>();
      operator()(&node->operand(0));

      switch (node->kind) {
#define __(code__, func__)                                 \
  case ir::Activate::Kind::code__:                         \
    *expr = lang::CallExtern(#func__, {node->operand(0)}); \
    break;
        __(kTanh, cinn_cpu_tanh_fp32)
        __(kCeil, cinn_cpu_ceil_fp32)
        __(kFloor, cinn_cpu_floor_fp32)
        __(kExp, cinn_cpu_exp_fp32)
#undef __
        default:
          CINN_NOT_IMPLEMENTED
      }
    }
  };

  Mutator()(e);
}

}  // namespace optim
}  // namespace cinn
