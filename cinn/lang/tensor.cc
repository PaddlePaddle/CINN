#include "cinn/lang/tensor.h"

#include <cstring>

#include "cinn/common/common.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace ir {

namespace detail {

Expr ExpandTo1DIndice(const std::vector<Expr> &shape, const std::vector<Expr> &indices) {
  CHECK_EQ(shape.size(), indices.size());
  Expr res = indices.front() * shape[1];
  for (int i = 1; i < shape.size() - 1; i++) {
    res = res + indices[i] * shape[i + 1];
  }
  if (shape.size() > 1) res = res + indices.back();
  return res;
}

Expr ExpandTo1DIndice(const std::vector<int> &shape, const std::vector<Expr> &indices) {
  std::vector<Expr> shape_;
  for (int v : shape) shape_.push_back(Expr(v));
  return ExpandTo1DIndice(shape, indices);
}

}  // namespace detail

Tensor _Tensor_::Make(const std::string &name, const std::vector<Expr> &shape, FunctionRef fn) {
  CHECK(!shape.empty()) << "Tensor shape is set empty";
  CHECK(!name.empty()) << "Tensor name is set empty";
  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->shape    = shape;
  n->operaion = fn;
  n->InitStage();
  n->InitAxis();
  n->SetDefaultBindedBuffer();
  return Tensor(n);
}

Tensor _Tensor_::Make(const std::string &name,
                      const std::string &tag,
                      const std::vector<Expr> &shape,
                      const std::vector<Var> &axis,
                      Type dtype,
                      const std::map<std::string, IrNodeRef> &attrs,
                      const std::vector<Expr> &body) {
  CHECK(!shape.empty()) << "Tensor shape is set empty";
  CHECK(!name.empty()) << "Tensor name is set empty";

  auto op          = ComputeOp::Make(name, tag, attrs, axis, body, shape);
  auto *compute_op = const_cast<ComputeOp *>(op->As<ComputeOp>());

  CHECK_EQ(axis.size(), shape.size()) << "axis not match the dimension in shape";
  compute_op->axis = axis;

  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->operaion = op;
  n->shape    = shape;
  n->set_type(dtype);
  n->InitStage();
  n->SetDefaultBindedBuffer();
  return Tensor(n);
}

Tensor::Tensor(
    const std::vector<Expr> &shape, const std::vector<Var> &axis, Type dtype, Expr expr, const std::string &name)
    : IrNodeRef(_Tensor_::Make(
          name.empty() ? Context::Global().NewName("tensor") : name, "", shape, axis, dtype, {}, {expr})) {}

size_t Tensor::ndims() const { return operator->()->shape.size(); }

Expr Tensor::operator()(const std::vector<Expr> &indices) const {
  CHECK_EQ(indices.size(), ndims()) << "number of indices not match the dimension";
  auto *node = operator->();
  auto n     = Call::Make(node->type().ElementOf(), node->name, indices, Call::Halide, node->operaion, 0, Expr(*this));
  n->set_type(node->type());
  return n;
}

const char *_Tensor_::operation_type() const {
  if (!operaion.defined()) return "";
  return operaion->As<ir::_Operation_>()->func_type();
}

bool _Tensor_::is_compute_node() const { return std::strcmp(operation_type(), ir::ComputeOp::__func_type__) == 0; }
bool _Tensor_::is_placeholder_node() const {
  return std::strcmp(operation_type(), ir::PlaceholderOp::__func_type__) == 0;
}

void _Tensor_::InitStage() {
  CHECK(!stage) << "Duplicate initialize the poly_element";
  auto *op = operaion->As<_Operation_>();
  if (is_compute_node()) {
    auto &body = op->As<ComputeOp>()->body;
    CHECK_EQ(body.size(), 1UL) << "only support functional programming";
    stage = make_shared<poly::Stage>(GenerateIslDomain(), body.front());
  } else {
    stage = make_shared<poly::Stage>(GenerateIslDomain());
  }
}

void _Tensor_::InitAxis() {
  CHECK(!shape.empty());
  CHECK(axis.empty()) << "duplicate init axis";
  axis = common::GenDefaultAxis(shape.size());
}

isl::set _Tensor_::GenerateIslDomain() {
  CHECK(!shape.empty()) << "shape should be set";
  std::vector<poly::Dim> dims;
  for (int i = 0; i < shape.size(); i++) {
    dims.emplace_back(common::axis_name(i), 0, shape[i].as_int32() - 1);
  }

  poly::Domain domain(isl_ctx_alloc(), name, dims);
  return domain.to_isl();
}
std::vector<Expr *> _Tensor_::expr_fields() {
  std::vector<Expr *> res;
  const char *func_type = operaion->As<ir::_Operation_>()->func_type();
  if (operaion.defined()) {
    if (func_type == ir::ComputeOp::__func_type__) {
      auto *op = operaion->As<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
      for (auto &expr : op->shape) res.push_back(&expr);
    } else if (func_type == ir::PlaceholderOp::__func_type__) {
      auto *op = operaion->As<ir::PlaceholderOp>();
      for (auto &expr : op->shape) res.push_back(&expr);
    } else {
      NOT_IMPLEMENTED
    }
  }
  return res;
}

std::vector<const Expr *> _Tensor_::expr_fields() const {
  std::vector<const Expr *> res;
  const char *func_type = operaion->As<ir::_Operation_>()->func_type();
  if (operaion.defined()) {
    if (is_compute_node()) {
      auto *op = operaion->As<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
      for (auto &expr : op->shape) res.push_back(&expr);
    } else if (is_placeholder_node()) {
      auto *op = operaion->As<ir::PlaceholderOp>();
      for (auto &expr : op->shape) res.push_back(&expr);
    } else {
      LOG(ERROR) << "func_type: " << func_type;
      NOT_IMPLEMENTED
    }
  }
  return res;
}

_Tensor_::~_Tensor_() {
  if (stage) {
    delete stage;
    stage = nullptr;
  }
}

const _Operation_ *Operation::operator->() const { return static_cast<_Operation_ *>(get()); }

Expr _Tensor_::body() const {
  if (is_placeholder_node()) return Expr();
  if (is_compute_node()) return operaion->As<ir::ComputeOp>()->body.front();
  NOT_IMPLEMENTED;
}

Expr _Tensor_::tensor_store_expanded_body() const {
  CHECK(!is_placeholder_node()) << "placeholder should not expand store";
  std::vector<Expr> axis_;
  for (auto &a : axis) axis_.push_back(Expr(a));
  return ir::Store::Make(buffer_var, body(), detail::ExpandTo1DIndice(shape, axis_));
}

}  // namespace ir
}  // namespace cinn
