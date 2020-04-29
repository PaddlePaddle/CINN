#include "cinn/optim/call_arg_list_to_pod_value.h"
#include <string>
#include <tuple>
#include <vector>
#include "cinn/common/ir.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/runtime/intrinsic.h"

namespace cinn {
namespace optim {

namespace {

struct CallArgListToPodValueMutator : ir::IRMutator<> {
  void operator()(Expr* e) { ir::IRMutator<>::Visit(e, e); }

 private:
  void Visit(const ir::Call* op, Expr* expr) override {
    if (op->call_type == ir::Call::CallType::CINN) {
      auto [oprs, args] = pack_arg_exprs(op);  // NOLINT

      Var pod_array_var(Context::Global().NewName("_pod_arr"), type_of<cinn_pod_value_t*>());
      oprs.push_back(ir::Call::Make(
          Void(), runtime::pod_values_to_array_repr, args, {pod_array_var}, ir::Call::CallType::Intrinsic));

      auto new_call = ir::Call::Make(Void(),
                                     op->name,
                                     {pod_array_var, common::make_const(Int(32), args.size())},
                                     {},
                                     ir::Call::CallType::CINN,
                                     op->func,
                                     op->value_index,
                                     op->tensor);

      oprs.push_back(new_call);

      *expr = ir::Block::Make(oprs);
    }
  }

  std::tuple<std::vector<Expr> /*oprs*/, std::vector<Expr> /*args*/> pack_arg_exprs(const ir::Call* op) {
    std::vector<Expr> exprs;
    std::vector<Expr> args;

    auto pack_arg = [&](const Expr& arg) {
      Var new_var(Context::Global().NewName("_pod_val_"), type_of<cinn_pod_value_t>());
      exprs.push_back(ir::Let::Make(new_var, Expr()));

      Expr cast;
      if (arg.type() == type_of<float>()) {
        auto casted_arg = ir::Call::Make(
            type_of<cinn_pod_value_t*>(), runtime::get_address_repr, {new_var}, {}, ir::Call::CallType::Intrinsic);
        cast = ir::Call::Make(
            Void(), runtime::float_to_cinn_pod_value_repr, {arg}, {casted_arg}, ir::Call::CallType::Intrinsic);
      } else if (arg.type() == type_of<cinn_buffer_t>()) {
        cast = ir::Call::Make(
            Void(), runtime::buffer_p_to_cinn_pod_value_repr, {arg}, {new_var}, ir::Call::CallType::Intrinsic);
      } else {
        NOT_IMPLEMENTED
      }

      exprs.push_back(cast);

      args.push_back(Expr(new_var));
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
