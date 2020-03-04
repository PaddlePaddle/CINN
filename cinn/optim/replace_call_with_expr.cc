#include "cinn/optim/replace_call_with_expr.h"

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

struct ReplaceCallWithExprModifier : public ir::IRMutator<> {
  ReplaceCallWithExprModifier(const std::string &statement, const Expr &candidate)
      : statement_(statement), candidate_(candidate) {}

  void operator()(Expr *e) { IRMutator<>::Visit(e, e); }

 private:
  void Visit(const ir::Call *expr, Expr *op) override {
    auto *node = expr->As<ir::Call>();
    CHECK(!node->name.empty()) << "Call has no name";
    VLOG(2) << "Processing Call node " << node->name;
    if (statement_ != node->name) return;

    Expr expr_candidate = IRCopy(candidate_);

    // Replace the Call node with the expression candidate.
    *op = expr_candidate;
    VLOG(2) << "expr " << *op;
  }

 private:
  std::string statement_;
  const Expr &candidate_;
};

void ReplaceCallWithExpr(Expr *e, const std::string &statement, const Expr &candidate) {
  ReplaceCallWithExprModifier modifier(statement, candidate);
  modifier(e);
}

void ReplaceCallWithExpr(Expr *e,
                         const std::string &statement,
                         const Expr &candidate,
                         const std::map<std::string, Expr> &axis) {
  Expr copied = IRCopy(candidate);
  for (auto &axis : axis) {
    ReplaceVarWithExpr(&copied, Var(axis.first), axis.second);
  }
  // LOG(INFO) << "expression after replaced: " << copied;
  ReplaceCallWithExpr(e, statement, copied);
}

}  // namespace optim
}  // namespace cinn
