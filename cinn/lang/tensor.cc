#include "cinn/lang/tensor.h"

#include <cstring>

#include "cinn/common/common.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace ir {

Tensor _Tensor_::Make(const std::string &name, const std::vector<Expr> &shape, FunctionRef fn) {
  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->shape    = shape;
  n->operaion = fn;
  n->InitStage();
  return Tensor(n);
}

Tensor _Tensor_::Make(const std::string &name,
                      const std::string &tag,
                      const std::vector<Expr> &shape,
                      const std::vector<Var> &axis,
                      Type dtype,
                      const std::map<std::string, IrNodeRef> &attrs,
                      const std::vector<Expr> &body) {
  auto op          = ComputeOp::Make(name, tag, attrs, axis, body, shape);
  auto *compute_op = const_cast<ComputeOp *>(op->As<ComputeOp>());
  compute_op->axis = axis;

  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->operaion = op;
  n->shape    = shape;
  n->set_type(dtype);
  n->InitStage();
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
  }
}

const _Operation_ *Operation::operator->() const { return static_cast<_Operation_ *>(get()); }

}  // namespace ir
}  // namespace cinn
