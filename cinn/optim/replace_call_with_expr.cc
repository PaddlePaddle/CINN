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
    auto *node = op->As<ir::Call>();
    CHECK(!node->name.empty()) << "Call has no name";
    VLOG(3) << "Processing Call node " << *op;
    if (statement_ != node->name) return;

    Expr expr_candidate = IRCopy(candidate_);
    VLOG(3) << "Original candidate expr: " << candidate_;
    VLOG(3) << "Copied candidate expr: " << expr_candidate;

    // Replace the Call node with the expression candidate.
    *op = expr_candidate;
    VLOG(3) << "expr " << *op;
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
  VLOG(3) << "ReplaceCallWithExpr, original expression: " << candidate;
  Expr copied = IRCopy(candidate);
  VLOG(3) << "ReplaceCallWithExpr, copied expression: " << copied;
  // update the axis in the copied expression.

  // we treat the Store node as the normal statement, the others like Call node has no axis.
  std::map<std::string, Expr> local_axis;
  if (copied.As<ir::Store>()) {
    auto *store = copied.As<ir::Store>();
    for (int i = 0; i < store->indices.size(); i++) {
      auto indice = store->indices[i];
      CHECK(indice.is_var() || indice.is_constant());
      if (!indice.is_constant()) {
        local_axis[indice.as_var()->name] = axis.at(std::to_string(i));
      }
    }
  }

  for (auto &axis : local_axis) {
    LOG(INFO) << "replacing axis: " << axis.first << " " << axis.second;
    ReplaceVarWithExpr(&copied, Var(axis.first), axis.second);
  }
  // LOG(INFO) << "expression after replaced: " << copied;
  ReplaceCallWithExpr(e, statement, copied);
}

}  // namespace optim
}  // namespace cinn
