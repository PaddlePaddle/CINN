#include "cinn/optim/call_arg_list_to_pod_value.h"

#include <string>
#include <tuple>
#include <vector>

#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace optim {

namespace {

struct CallArgListToPodValueMutator : ir::IRMutator<> {
  void operator()(Expr* e) { ir::IRMutator<>::Visit(e, e); }

 private:
  void Visit(const ir::Call* op, Expr* expr) override {
    if (op->is_cinn_call()) {
      auto [oprs, args] = pack_arg_exprs(op);  // NOLINT

      Var pod_array_var(Context::Global().NewName("_pod_arr"),
                        type_of<cinn_pod_value_t>().with_lanes(op->total_args_count()));

      // Declare pod_array.
      oprs.push_back(ir::Let::Make(pod_array_var, Expr()));
      oprs.push_back(ir::intrinsics::ArgsConstruct::Make(pod_array_var, args));

      auto new_call = ir::Call::Make(Void(),
                                     op->name,
                                     {pod_array_var, common::make_const(Int(32), args.size())},
                                     {},
                                     ir::CallType::CINN,
                                     op->func,
                                     op->value_index);

      oprs.push_back(new_call);

      *expr = ir::Block::Make(oprs);
    }
  }

  std::tuple<std::vector<Expr> /*oprs*/, std::vector<Expr> /*args*/> pack_arg_exprs(const ir::Call* op) {
    std::vector<Expr> exprs;
    std::vector<Expr> args;

    auto pack_arg = [&](const Expr& arg) {
      Var pod_var(Context::Global().NewName("_pod_val_"), type_of<cinn_pod_value_t>());

      // declare the array.
      exprs.push_back(ir::Let::Make(pod_var, Expr()));

      auto pod_val_addr_expr = ir::intrinsics::GetAddr::Make(pod_var);

      Expr cast;
      if (arg.As<ir::_Buffer_>()) {
        cast = runtime::IntrinsicCall(
            Void(), runtime::intrisic::buffer_p_to_cinn_pod_value_repr, {arg}, {pod_val_addr_expr});

      } else if (arg.type() == type_of<float>()) {
        cast =
            runtime::IntrinsicCall(Void(), runtime::intrisic::float_to_cinn_pod_value_repr, {arg}, {pod_val_addr_expr});
      } else if (arg.type() == type_of<int32_t>()) {
        cast =
            runtime::IntrinsicCall(Void(), runtime::intrisic::int32_to_cinn_pod_value_repr, {arg}, {pod_val_addr_expr});
      } else {
        CINN_NOT_IMPLEMENTED
      }

      exprs.push_back(cast);
      args.push_back(pod_val_addr_expr);
    };

    for (auto& arg : op->read_args) {
      pack_arg(arg);
    }
    for (auto& arg : op->write_args) {
      pack_arg(arg);
    }

    return std::make_tuple(exprs, args);
  }
};

}  // namespace

void CallArgListToPodValue(Expr* e) { CallArgListToPodValueMutator()(e); }

}  // namespace optim
}  // namespace cinn
