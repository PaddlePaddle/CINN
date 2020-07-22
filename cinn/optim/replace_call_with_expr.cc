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

void ReplaceIslCallWithExpr(Expr *e,
                            const std::string &statement,
                            const Expr &candidate,
                            const std::map<std::string, Expr> &axis_map) {
  VLOG(3) << "ReplaceCallWithExpr, original expression: " << candidate;
  Expr copied = IRCopy(candidate);
  // update the axis in the copied expression.

  // we treat the Store node as the normal statement, the others like Call node has no axis.
  std::map<std::string, Expr> local_axis;
  if (copied.As<ir::Store>()) {
    auto *store = copied.As<ir::Store>();
    for (int i = 0; i < store->indices.size(); i++) {
      auto indice = store->indices[i];
      CHECK(indice.is_var() || indice.is_constant());
      if (!axis_map.count(std::to_string(i))) continue;
      if (!indice.is_constant()) {
        local_axis[indice.as_var()->name] = axis_map.at(std::to_string(i));
      }
    }
    // the store indices just contains the ones of transform's domain, not the range.
    // e.g. { s[i,j] -> s[i0,i1,j]: i0=i/4 and i1=i%4 }, the store's indices just contains i,j while in the final code,
    // the axis are from the range, that is, there are some new axis not exists in store->indice, i0 and i1.
  }

  for (auto &laxis : local_axis) {
    LOG(INFO) << "replacing axis: " << laxis.first << " " << laxis.second;
    ReplaceVarWithExpr(&copied, Var(laxis.first), laxis.second);
  }
  // replace the remaining axis(in the transform's range)
  for (auto &item : axis_map) {
    if (!local_axis.count(item.first)) {
      ReplaceVarWithExpr(&copied, Var(item.first), item.second);
    }
  }

  // LOG(INFO) << "expression after replaced: " << copied;
  ReplaceCallWithExpr(e, statement, copied);
}

}  // namespace optim
}  // namespace cinn
