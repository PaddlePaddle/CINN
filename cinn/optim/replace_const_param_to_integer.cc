#include "cinn/optim/replace_const_param_to_integer.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/poly/ast_gen.h"
#include "cinn/utils/string.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  void Visit(const ir::_Var_* op, Expr* expr) override {
    if (utils::Startswith(op->name, poly::kIslParamConstPrefix)) {
      std::string value = op->name.substr(strlen(poly::kIslParamConstPrefix));
      *expr             = Expr(std::stoi(value));
    }
  }
};

}  // namespace

void ReplaceConstParamToInteger(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim
