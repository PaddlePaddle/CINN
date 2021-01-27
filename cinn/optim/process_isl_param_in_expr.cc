#include "cinn/optim/process_isl_param_in_expr.h"
#include <regex>
#include "cinn/ir/ir_mutator.h"
#include "cinn/poly/ast_gen.h"
#include "cinn/poly/isl_utils.h"

namespace cinn {
namespace optim {

void ProcessIslParamInExpr(Expr* expr) {
  struct Mutator : public ir::IRMutator<> {
    using ir::IRMutator<>::Visit;

    void Visit(const ir::_Var_* op, Expr* e) override {
      if (poly::IsIslConstantParam(op->name)) {
        *e = Expr(poly::IslConstantParamGetId(op->name));
      }
    }
  };

  Mutator mutator;
  mutator.Visit(expr, expr);
}

}  // namespace optim
}  // namespace cinn
