#include "cinn/optim/extern_call_process.h"

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

struct ExternCallMultiOutputShallowStoreMutator : public ir::IRMutator<> {
  void operator()(Expr* e) { ir::IRMutator<>::Visit(e, e); }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* call = op->value.As<ir::Call>();
    if (call && call->is_extern_call() && !call->write_args.empty()) {
      *expr = op->value;
    }
  }
};

}  // namespace

void ExternCallMultiOutputShallowStore(Expr* e) { ExternCallMultiOutputShallowStoreMutator()(e); }

}  // namespace optim
}  // namespace cinn
