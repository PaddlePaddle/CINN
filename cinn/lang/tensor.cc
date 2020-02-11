#include "cinn/lang/tensor.h"

#include "cinn/common/common.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"
#include "cinn/poly/element.h"

namespace cinn {
namespace ir {

const _Operation_ *Operation::operator->() const { return static_cast<_Operation_ *>(get()); }

Tensor _Tensor_::Make(const std::string &name, const std::vector<Expr> &shape, FunctionRef fn) {
  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->shape    = shape;
  n->operaion = fn;
  n->InitPolyElement();
  return Tensor(n);
}

Tensor _Tensor_::Make(const std::string &name,
                      const std::string &tag,
                      const std::vector<Expr> &shape,
                      const std::vector<Var> &axis,
                      Type dtype,
                      const std::map<std::string, IrNodeRef> &attrs,
                      const std::vector<Expr> &body) {
  auto op          = ComputeOp::Make(name, tag, attrs, axis, body);
  auto *compute_op = const_cast<ComputeOp *>(op->As<ComputeOp>());
  compute_op->axis = axis;

  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->operaion = op;
  n->shape    = shape;
  n->set_type(dtype);
  n->InitPolyElement();
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
  auto n     = Call::Make(node->type().ElementOf(), node->name, indices, Call::Halide, node->operaion);
  n->set_type(node->type());
  return n;
}

void _Tensor_::InitPolyElement() {
  CHECK(!poly_element) << "Duplicate initialize the poly_element";
  poly_element = new poly::Element(GenerateIslDomain());
}

isl::set _Tensor_::GenerateIslDomain() {
  std::vector<poly::Dim> dims;
  for (int i = 0; i < shape.size(); i++) {
    dims.emplace_back(common::axis_name(i), 0, shape[i].as_int32());
  }

  poly::Domain domain(isl_ctx_alloc(), name, dims);
  return domain.to_isl();
}

}  // namespace ir
}  // namespace cinn
