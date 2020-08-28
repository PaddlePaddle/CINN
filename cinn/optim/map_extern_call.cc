#include "cinn/optim/map_extern_call.h"

#include "cinn/cinn.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace optim {

void MapExternCall(Expr *e, Target target) {
  struct Mutator : ir::IRMutator<Expr *> {
    Target target;

    Mutator(Target target) : target(target) { LOG(INFO) << "target: " << target; }

    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Call *op, Expr *expr) override {
      auto *node = expr->As<ir::Call>();
      CHECK(node);

      if (target.arch == Target::Arch::NVGPU) {
        DealWithNvGpuIntrisics(node, expr);
      } else {
        DealWithCpuIntrisics(node, expr);
      }
    }

    void DealWithCpuIntrisics(ir::Call *node, Expr *expr) {
      if (kExternFp32Calls.count(node->name)) {
        CHECK_GE(node->read_args.size(), 1UL);
        CHECK_EQ(node->read_args.front().type(), Float(32));
        *expr = lang::CallExtern("cinn_cpu_" + node->name + "_fp32", node->read_args);
      } else if (kExternInt64Calls.count(node->name)) {
        CHECK_GE(node->read_args.size(), 1UL);
        CHECK_EQ(node->read_args.front().type(), Int(64));
        *expr = lang::CallExtern("cinn_cpu_" + node->name + "_int64", node->read_args);
      }
    }

    void DealWithNvGpuIntrisics(ir::Call *node, Expr *expr) {
      if (kExternFp32Calls.count(node->name)) {
        CHECK_GE(node->read_args.size(), 1UL);
        CHECK_EQ(node->read_args.front().type(), Float(32));
        *expr = lang::CallExtern("cinn_nvgpu_" + node->name + "_fp32", node->read_args);
      }
      // TODO(Superjomn) deal with int64 intrisics.
    }
  };

  Mutator m(target);
  m(e);
}

}  // namespace optim
}  // namespace cinn
