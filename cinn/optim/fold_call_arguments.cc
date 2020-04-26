#include "cinn/optim/fold_call_arguments.h"
#include <vector>
#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

struct FoldCallArgumentsMutator : public ir::IRMutator<> {
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::Store* op, Expr* expr) override {
    auto* node = expr->As<ir::Store>();
    if (node->value.As<ir::Call>()) {
      auto* call = node->value.As<ir::Call>();
      switch (call->call_type) {
        case ir::Call::CallType::CINN:
          MutateCall(call);
          *expr = node->value;
          break;
        case ir::Call::CallType::Intrinsic:
          break;
        default:
          NOT_IMPLEMENTED
      }
    }
  }

  void MutateCall(ir::Call* call) {
    std::vector<Expr> read_args;
    std::vector<Expr> write_args;
    for (auto& arg : call->read_args) {
      if (arg.as_tensor()) {
        read_args.push_back(arg.as_tensor()->buffer);
      } else {
        read_args.push_back(arg);
      }
    }

    for (auto& arg : call->write_args) {
      if (arg.as_tensor()) {
        write_args.push_back(arg.as_tensor()->buffer);
      } else {
        write_args.push_back(arg);
      }
    }

    call->read_args  = read_args;
    call->write_args = write_args;
  }
};

}  // namespace

void FoldCallArguments(Expr* expr) { FoldCallArgumentsMutator()(expr); }

}  // namespace optim
}  // namespace cinn
