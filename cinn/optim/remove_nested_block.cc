#include "cinn/optim/remove_nested_block.h"

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

struct NestedBlockRemover : public ir::IRMutator {
  void Visit(const ir::Block* expr, Expr* op) override {
    auto* node = op->As<ir::Block>();

    std::vector<ir::Expr> new_exprs;

    bool detect_nested = false;
    for (auto it = node->stmts.begin(); it != node->stmts.end(); it++) {
      auto* block = it->As<ir::Block>();
      if (block) {
        detect_nested = true;
        new_exprs.insert(std::end(new_exprs), block->stmts.begin(), block->stmts.end());
      } else {
        new_exprs.push_back(*it);
      }
    }

    node->stmts = new_exprs;

    IRMutator::Visit(expr, op);
  }
};

void RemoveNestedBlock(Expr* e) {
  NestedBlockRemover remover;
  auto* p = e->As<ir::Block>();
  CHECK(p);
  remover.Visit(p, e);
}

}  // namespace optim
}  // namespace cinn
