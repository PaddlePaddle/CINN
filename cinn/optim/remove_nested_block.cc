#include "cinn/optim/remove_nested_block.h"

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {

// This will remove the nested blocks, but it will also remove the block outside the forloop's body.
struct NestedBlockSimplifer : public ir::IRMutator<Expr*> {
  void operator()(ir::Expr* expr) { Visit(expr); }

 private:
  void Visit(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  Expr GetExprInsideBlock(Expr op) {
    Expr node = op;
    while (node.As<ir::Block>()) {
      auto& stmts = node.As<ir::Block>()->stmts;
      if (stmts.size() == 1) {
        node = stmts.front();
      } else {
        break;
      }
    }
    return node;
  }

  void Visit(const ir::Block* expr, Expr* op) override {
    auto* node = op->As<ir::Block>();
    if (node->stmts.size() == 1) {
      *op = GetExprInsideBlock(*op);
      IRMutator::Visit(op, op);
    } else {
      IRMutator::Visit(expr, op);
    }
  }
};

struct NestedBlockSimplifer2 : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { Visit(expr, expr); }

 private:
  using ir::IRMutator<>::Visit;

  std::vector<Expr> new_exprs;

  void Visit(const ir::Block* op, Expr* expr) override {
    auto* node = expr->As<ir::Block>();
    bool detect_nested{};
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
  }
};

// add block outside forloop's body.
struct AddBlockToForloop : public ir::IRMutator<> {
  void operator()(ir::Expr* expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::For* expr, Expr* op) override {
    auto* node = op->As<ir::For>();
    if (!node->body.As<ir::Block>()) {
      node->body = ir::Block::Make({node->body});
    }

    ir::IRMutator<>::Visit(expr, op);
  }

  void Visit(const ir::PolyFor* expr, Expr* op) override {
    auto* node = op->As<ir::PolyFor>();
    if (!node->body.As<ir::Block>()) {
      node->body = ir::Block::Make({node->body});
    }

    ir::IRMutator<>::Visit(expr, op);
  }

  void Visit(const ir::_LoweredFunc_* expr, Expr* op) override {
    auto* node = op->As<ir::_LoweredFunc_>();
    if (!node->body.As<ir::Block>()) {
      node->body = ir::Block::Make({node->body});
    }

    ir::IRMutator<>::Visit(expr, op);
  }
};

void RemoveNestedBlock(Expr* e) {
  NestedBlockSimplifer()(e);
  NestedBlockSimplifer2()(e);
  AddBlockToForloop()(e);
}

}  // namespace optim
}  // namespace cinn
