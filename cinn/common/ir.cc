#include "cinn/common/ir.h"

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

}  // namespace common
}  // namespace cinn
