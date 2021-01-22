#include "cinn/optim/simplify_identity_domain_forloop.h"
#include <llvm/ADT/SmallVector.h>
#include "cinn/ir/ir_mutator.h"
#include "cinn/optim/replace_var_with_expr.h"

namespace cinn {
namespace optim {

/**
 * Help to manage the Vars.
 *
 * for (i, 0, 1) {  // status:    [{i}]
 *   for (j0, 0, 1) { // status:   [{i},{j0}]
 *   }
 *   for (j1, 0, 100) { // status: [{i},{j1}]
 *   }
 * }
 */
struct SymbolTable {
  struct Record {
    std::map<Var, Expr> constant_vars;
    std::set<Var> nonconstant_vars;
  };

  llvm::SmallVector<Record, 4> stack;

  Record &PushLevel() {
    stack.emplace_back();
    return stack.back();
  }
  void PopLevel() { stack.pop_back(); }

  Record &GetTop() { return stack.back(); }

  void AddConstantVar(Var var, Expr val) {
    CHECK(!GetTop().nonconstant_vars.count(var));
    GetTop().constant_vars[var] = val;
  }
  void AddNonconstantVar(Var var) {
    CHECK(!GetTop().constant_vars.count(var));
    GetTop().nonconstant_vars.insert(var);
  }

  llvm::SmallVector<std::pair<Var, Expr>, 4> GetAllConstantVars() {
    llvm::SmallVector<std::pair<Var, Expr>, 4> res;

    std::set<Var> nonconstant_vars;  // globally
    for (auto it = std::rbegin(stack); it != std::rend(stack); ++it) {
      for (auto &item : it->constant_vars) {
        if (!nonconstant_vars.count(item.first)) {
          res.push_back(item);
        }
      }

      for (auto &item : it->nonconstant_vars) {
        nonconstant_vars.insert(item);
      }
    }

    return res;
  }

  std::optional<Expr> FindConstantVar(Var var) {
    for (auto it = std::rbegin(stack); it != std::rend(stack); ++it) {
      if (it->constant_vars.count(var))
        return it->constant_vars[var];
      else if (it->nonconstant_vars.count(var))
        break;
    }
  }
};

void SimplifyIdentityDomainForloop(Expr *e) {
  struct Mutator : public ir::IRMutator<> {
    SymbolTable table;

    using ir::IRMutator<>::Visit;

    void Visit(const ir::For *op, Expr *expr) override {
      auto *node = expr->As<ir::For>();

      table.PushLevel();

      if (op->extent.is_constant() && op->extent.as_int32() == 1) {  // to simplify
        table.AddConstantVar(op->loop_var, op->min);
        ir::IRMutator<>::Visit(&node->body, &node->body);

        // remove outer forloop
        *expr = node->body;

      } else {
        table.AddNonconstantVar(op->loop_var);
        ir::IRMutator<>::Visit(&node->body, &node->body);
      }

      table.PopLevel();
    }

    void Visit(const ir::Store *op, Expr *expr) override {
      auto *node = expr->As<ir::Store>();
      for (auto &e : node->indices) {
        for (auto &[var, expr] : table.GetAllConstantVars()) {
          ReplaceVarWithExpr(&e, var, expr);
        }
      }
    }

    void Visit(const ir::Load *op, Expr *expr) override {
      auto *node = expr->As<ir::Store>();
      for (auto &e : node->indices) {
        for (auto &[var, expr] : table.GetAllConstantVars()) {
          ReplaceVarWithExpr(&e, var, expr);
        }
      }
    }
  };

  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace optim
}  // namespace cinn