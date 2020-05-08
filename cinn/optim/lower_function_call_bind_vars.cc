#include "cinn/optim/lower_function_call_bind_vars.h"

#include <string>
#include <vector>

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

namespace {

struct LowerFunctionCallBindVarsMutator : public ir::IRMutator<> {
  LowerFunctionCallBindVarsMutator() = default;

  void operator()(Expr* m) {
    m_ = m->as_module();
    Expr module(m->get());
    ir::IRMutator<>::Visit(&module, &module);
  }

 private:
  void Visit(const ir::Call* op, Expr* expr) {
    auto* node = expr->As<ir::Call>();
    if (op->call_type == ir::Call::CallType::CINN) {
      const std::string& target = op->name;
      auto it                   = std::find_if(m_->functions.begin(), m_->functions.end(), [&](const Expr& x) {
        return x.as_lowered_func()->name == target;
      });
      CHECK(it != m_->functions.end()) << "The called function [" << target << "] is not exist";

      std::vector<Expr> extra_var_args;

      for (auto& arg : (*it).as_lowered_func()->args) {
        if (arg.is_var()) {
          extra_var_args.push_back(arg.var_arg());
        }
      }

      // insert the extra var arguments to the begining of the original call's argument list.
      node->read_args.insert(std::begin(op->read_args), extra_var_args.begin(), extra_var_args.end());
    }

    ir::IRMutator<>::Visit(op, expr);
  }

 private:
  ir::_Module_* m_{};
};

}  // namespace

void LowerFunctionCallBindVars(Expr* m) {
  CHECK(m->as_module());
  LowerFunctionCallBindVarsMutator()(m);
}

}  // namespace optim
}  // namespace cinn
