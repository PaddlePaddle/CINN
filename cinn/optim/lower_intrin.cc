#include "cinn/optim/lower_intrin.h"

#include <string>

#include "cinn/backends/llvm/llvm_intrin_rule.h"
#include "cinn/cinn.h"
#include "cinn/ir/intrinsic_ops.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/registry.h"

namespace cinn {
namespace optim {

void LowerIntrin(Expr *e, Target target) {
  if (target.arch == Target::Arch::X86) {
    codegen::RegisterCpuIntrinRule();
  }
  struct Mutator : ir::IRMutator<Expr *> {
    Target target;

    explicit Mutator(Target target) : target(target) {}

    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Call *op, Expr *expr) override {
      auto *node = expr->As<ir::Call>();
      CHECK(node);

      if (target.arch == Target::Arch::X86) {
        LowerCpuIntrisicOp(node, expr);
      }
    }

    void LowerCpuIntrisicOp(ir::Call *node, Expr *expr) {
      if (kIntrinsicCalls.count(node->name)) {
        CHECK(!node->name.empty());
        auto *func_ptr = ir::Registry::Get("lower_cpu_intrinsic_" + node->name);
        CHECK(func_ptr) << "find no rule to lower cpu intrinsic for "
                        << "lower_cpu_intrinsic_" + node->name;
        Expr ret = (*func_ptr)(Expr(node));
        if (!ret.same_as(*expr)) {
          ir::IRMutator<>::Visit(&ret, &ret);
        }
        *expr = ret;
      }
    }
  };

  Mutator m(target);
  m(e);
}

}  // namespace optim
}  // namespace cinn
