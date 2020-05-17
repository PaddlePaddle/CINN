#include "cinn/hlir/instruction/shape.h"

#include <string>
#include <unordered_set>

#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace hlir {
namespace instruction {

bool Shape::operator==(const Shape &other) const {
  if (dims_.size() != other.dims_.size()) return false;
  for (int i = 0; i < num_dims(); i++) {
    if (dims_[i] != other.dims_[i]) return false;
  }
  return true;
}

const Shape::dim_t &Shape::operator[](int offset) const {
  CHECK_LT(offset, dims_.size());
  return dims_[offset];
}

Shape::dim_t &Shape::operator[](int offset) {
  CHECK_LT(offset, dims_.size());
  return dims_[offset];
}

bool Shape::operator!=(const Shape &other) { return !(*this == other); }

Shape::Shape(std::initializer_list<dim_t> list) {
  for (auto &x : list) AddDim(x);
}

std::string Shape::to_debug_string() const {
  std::stringstream ss;

  ss << "[";
  if (!dims_.empty()) {
    for (int i = 0; i < dims_.size() - 1; i++) {
      std::visit([&ss](auto const &e) { ss << e << ", "; }, dims_[i]);
    }
    std::visit([&ss](auto const &e) { ss << e; }, dims_.back());
  }
  ss << "]";
  return ss.str();
}

namespace {
struct ShapeDimVisitor {
  ShapeDimVisitor(std::vector<cinn::Expr> &out_shape) : out_shape_(out_shape) {}  // NOLINT

  void operator()(int v) { out_shape_.emplace_back(cinn::Expr(v)); }

  void operator()(cinn::Var v) { out_shape_.emplace_back(cinn::Expr(v)); }

 private:
  std::vector<cinn::Expr> &out_shape_;
};
}  // namespace

std::vector<cinn::Expr> Shape::ToCinnShape() const {
  std::vector<cinn::Expr> cinn_shape;
  ShapeDimVisitor visitor{cinn_shape};
  for (auto &v : dims_) {
    std::visit(visitor, v);
  }
  return cinn_shape;
}

std::vector<cinn::Var> Shape::CollectDynamicDims() const {
  struct Visitor {
    Visitor(std::vector<cinn::Var> &vars) : vars_(vars) {}

    void operator()(int v) {}
    void operator()(cinn::Var v) {
      if (!var_names.count(v->name)) {
        vars_.push_back(v);
        var_names.insert(v->name);
      }
    }

    std::vector<cinn::Var> &vars_;
    std::unordered_set<std::string> var_names;
  };

  std::vector<cinn::Var> res;
  Visitor visitor{res};

  for (auto &dim : dims_) {
    std::visit(visitor, dim);
  }

  return res;
}

}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
