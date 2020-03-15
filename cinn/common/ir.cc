#include "cinn/common/ir.h"

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"

namespace cinn {
namespace common {

Expr ExpandTo1DIndice(const std::vector<Expr> &shape, const std::vector<Expr> &indices) {
  CHECK_EQ(shape.size(), indices.size());
  Expr res;
  for (int i = 0; i < shape.size(); i++) {
    Expr indice_prod = indices[i];
    for (int j = i + 1; j < shape.size(); j++) {
      indice_prod = indice_prod * shape[j];
    }

    if (res.defined())
      res = res + indice_prod;
    else
      res = indice_prod;
  }

  return res;
}

Expr ExpandTo1DIndice(const std::vector<int> &shape, const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return ExpandTo1DIndice(shape, indices);
}

namespace {

class SubstituteMutator : ir::IRMutator<ir::Expr *> {
 public:
  SubstituteMutator(const std::map<const ir::_Var_ *, Expr> &var_map) {
    for (auto &item : var_map) {
      var_map_[item.first->name] = item.second;
    }
  }

  void operator()(ir::Expr *expr) { Visit(expr); }

 private:
  void Visit(Expr *expr) { ir::IRMutator<>::Visit(expr, expr); }

  void Visit(const ir::_Var_ *op, ir::Expr *expr) override {
    auto it = var_map_.find(op->name);
    if (it == var_map_.end()) return;
    *expr = it->second;
  }

  Expr *expr_{};
  std::map<std::string, Expr> var_map_;
};

}  // namespace

void Substitute(Expr *expr, const std::map<const ir::_Var_ *, Expr> &var_map) {
  SubstituteMutator mutator(var_map);
  mutator(expr);
}

}  // namespace common
}  // namespace cinn
