#include "cinn/optim/ir_replace.h"

#include <set>

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/optim/ir_copy.h"
#include "cinn/utils/string.h"

namespace cinn {
namespace optim {
using utils::GetStreamCnt;

namespace {

struct IrReplaceMutator : ir::IRMutator<Expr*> {
  std::set<ir::IrNodeTy> valid_nodetys{{ir::IrNodeTy::Broadcast, ir::IrNodeTy::_Var_}};

  IrReplaceMutator(ir::Expr from, Expr to) : from_(from), to_(to), from_repr_(GetStreamCnt(from)) {
    CHECK(valid_nodetys.count(from->node_type())) << "Not valid node type got " << from->node_type();
  }
  void operator()(Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

 private:
  void Visit(const ir::_Var_* op, Expr* expr) override {
    if (op->node_type() == from_->node_type() && from_repr_ == GetStreamCnt(*expr)) {
      *expr = optim::IRCopy(to_);
    }
  }

  void Visit(const ir::Broadcast* op, Expr* expr) override {
    if (op->node_type() == from_->node_type() && from_repr_ == GetStreamCnt(*expr)) {
      *expr = optim::IRCopy(to_);
    }
  }

  std::string from_repr_;
  ir::Expr from_;
  Expr to_;
};

}  // namespace

void IrReplace(ir::Expr* expr, ir::Expr from, ir::Expr to) {
  CHECK(expr);
  IrReplaceMutator(from, to)(expr);
}

}  // namespace optim
}  // namespace cinn
