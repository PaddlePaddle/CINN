#include "cinn/optim/collect_undefined_vars.h"

#include <set>

#include "cinn/ir/ir_mutator.h"

namespace cinn::optim {

namespace {
struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;
  std::vector<std::string> undefined_vars;
  std::set<std::string> defined_vars;
  std::set<std::string> used_vars;

  void CollectVarDef(const std::string& var) {
    CHECK(!defined_vars.count(var)) << "var " << var << " has been defined, please check";
    CHECK(!used_vars.count(var)) << "var " << var << " is wrongly used before definition";
    defined_vars.insert(var);
  }

  void CollectVarUse(const std::string& var) {
    used_vars.insert(var);
    if (defined_vars.count(var) == 0) {
      undefined_vars.push_back(var);
    }
  }

  void Visit(const ir::Let* op, Expr* expr) final {
    Expr symbol = op->symbol;
    auto var    = symbol.as_var_ref();
    CHECK(var.defined());
    CollectVarDef(var->name);
    auto* node = expr->As<ir::Let>();
    Visit(&node->body, &node->body);
  }

  void Visit(const ir::For* op, Expr* expr) final {
    CollectVarDef(op->loop_var->name);
    auto* node = expr->As<ir::For>();
    Visit(&node->min, &node->min);
    Visit(&node->extent, &node->extent);
    Visit(&node->body, &node->body);
  }

  void Visit(const ir::Load* op, Expr* expr) final {
    auto tensor = op->tensor.as_tensor_ref();
    CollectVarUse(tensor->name);
    auto* node = expr->As<ir::Load>();
    for (auto& idx : node->indices) Visit(&idx, &idx);
  }

  void Visit(const ir::Store* op, Expr* expr) final {
    auto tensor = op->tensor.as_tensor_ref();
    CollectVarUse(tensor->name);
    auto* node = expr->As<ir::Store>();
    for (auto& idx : node->indices) Visit(&idx, &idx);
    Visit(&node->value, &node->value);
  }

  void Visit(const ir::_Var_* op, Expr* expr) final {
    CollectVarUse(op->name);
    auto* node = expr->As<ir::_Var_>();
    if (node->is_reduce_axis) {
      Visit(&node->lower_bound, &node->lower_bound);
      Visit(&node->upper_bound, &node->upper_bound);
    }
  }

  void Visit(const ir::Reduce* op, Expr* expr) final {
    for (auto& axis : op->reduce_axis) {
      CollectVarDef(axis->name);
    }
    auto* node = expr->As<ir::Reduce>();
    if (node->init.defined()) Visit(&node->init, &node->init);
    Visit(&node->body, &node->body);
  }
};
}  // namespace

std::vector<std::string> CollectUndefinedVars(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
  return mutator.undefined_vars;
}

}  // namespace cinn::optim
