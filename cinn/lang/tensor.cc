#include "cinn/lang/tensor.h"

#include <cstring>

#include "cinn/common/common.h"
#include "cinn/common/ir.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace ir {

Tensor _Tensor_::Make(const std::string &name, const std::vector<Expr> &shape, FunctionRef fn) {
  CHECK(!shape.empty()) << "Tensor shape is set empty";
  CHECK(!name.empty()) << "Tensor name is set empty";
  auto n      = make_shared<_Tensor_>();
  n->name     = name;
  n->shape    = shape;
  n->operaion = fn;
  n->InitStage();
  n->InitAxis();

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
  auto *compute_op = op->as<ComputeOp>();

  CHECK_EQ(axis.size(), shape.size()) << "axis not match the dimension in shape";
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

  if (node->inlined()) {
    VLOG(3) << "detect an inlined tensor and expand it";
    auto *compute_op = node->get_compute_op();
    CHECK(compute_op) << "Inlined Tensor should be generate from a ComputeOp";
    CHECK(compute_op->producer_fn) << "producer_fn field is unset";
    return compute_op->producer_fn(indices);
  } else {
    CHECK(node->buffer.defined()) << utils::StringFormat("Buffer for [%s] should be defined so that it can be sliced",
                                                         node->name.c_str());
    return Load::Make(*this, indices);
  }
}

const char *_Tensor_::operation_type() const {
  if (!operaion.defined()) return "";
  return operaion->as<ir::_Operation_>()->func_type();
}

bool _Tensor_::is_compute_node() const { return std::strcmp(operation_type(), ir::ComputeOp::__func_type__) == 0; }
bool _Tensor_::is_placeholder_node() const {
  return std::strcmp(operation_type(), ir::PlaceholderOp::__func_type__) == 0;
}

ComputeOp *_Tensor_::get_compute_op() const {
  if (!is_compute_node()) return nullptr;
  return operaion->as<ComputeOp>();
}

PlaceholderOp *_Tensor_::get_placeholder_op() const {
  if (!is_placeholder_node()) return nullptr;
  return operaion->as<PlaceholderOp>();
}

void _Tensor_::InitStage() {
  if (inlined()) {
    DropStage();
    return;
  }
  if (is_placeholder_node()) {
    VLOG(2) << "Tensor " << name << " is placeholder, skip stage initialization";
    DropStage();
    return;
  }
  // Avoid duplicate init.
  if (stage_shared) {
    auto &shared_stage = *static_cast<Shared<poly::Stage> *>(stage_shared);
    shared_stage->set_extra_depend_stages(buffer_depended_tensor_names_);
    return;
  }

  stage_shared       = new Shared<poly::Stage>;
  auto &shared_stage = *static_cast<Shared<poly::Stage> *>(stage_shared);
  auto *op           = operaion->as<_Operation_>();
  if (is_compute_node()) {
    auto &body = op->as<ComputeOp>()->body;
    CHECK_EQ(body.size(), 1UL) << "only support functional programming";
    shared_stage = poly::Stage::New(GenerateIslDomain(), body.front());
  } else {
    shared_stage = poly::Stage::New(GenerateIslDomain());
  }

  shared_stage->set_extra_depend_stages(buffer_depended_tensor_names_);
}

void _Tensor_::DropStage() {
  if (stage_shared) {
    delete static_cast<Shared<poly::Stage> *>(stage_shared);
    stage_shared = nullptr;
  }
}

poly::Stage *_Tensor_::stage() {
  if (!stage_shared) return nullptr;
  return (*static_cast<Shared<poly::Stage> *>(stage_shared))->as<poly::Stage>();
}

void _Tensor_::InitAxis() {
  CHECK(!shape.empty());
  CHECK(axis.empty()) << "duplicate init axis";
  axis = common::GenDefaultAxis(shape.size());
}

isl::set _Tensor_::GenerateIslDomain() {
  CHECK(!domain.empty()) << "shape should be set";
  std::vector<poly::Dim> dims;
  for (int i = 0; i < domain.size(); i++) {
    dims.emplace_back(common::axis_name(i), 0, domain[i].as_int32() - 1);
  }

  poly::Domain domain(isl_ctx_alloc(), name, dims);
  return domain.to_isl();
}
std::vector<Expr *> _Tensor_::expr_fields() {
  std::vector<Expr *> res;
  const char *func_type = operaion->as<ir::_Operation_>()->func_type();
  if (operaion.defined()) {
    if (func_type == ir::ComputeOp::__func_type__) {
      auto *op = operaion->as<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
      for (auto &expr : op->shape) res.push_back(&expr);
    } else if (func_type == ir::PlaceholderOp::__func_type__) {
      auto *op = operaion->as<ir::PlaceholderOp>();
      for (auto &expr : op->shape) res.push_back(&expr);
    } else {
      NOT_IMPLEMENTED
    }
  }
  return res;
}

std::vector<const Expr *> _Tensor_::expr_fields() const {
  std::vector<const Expr *> res;
  const char *func_type = operaion->as<ir::_Operation_>()->func_type();
  if (operaion.defined()) {
    if (is_compute_node()) {
      auto *op = operaion->as<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
      for (auto &expr : op->shape) res.push_back(&expr);
    } else if (is_placeholder_node()) {
      auto *op = operaion->as<ir::PlaceholderOp>();
      for (auto &expr : op->shape) res.push_back(&expr);
    } else {
      LOG(ERROR) << "func_type: " << func_type;
      NOT_IMPLEMENTED
    }
  }
  return res;
}

_Tensor_::~_Tensor_() {
  if (stage_shared) {
    delete static_cast<Shared<poly::Stage> *>(stage_shared);
  }
}

const _Operation_ *Operation::operator->() const { return static_cast<_Operation_ *>(get()); }
_Operation_ *Operation::operator->() { return static_cast<_Operation_ *>(get()); }

Expr _Tensor_::body() const {
  if (is_placeholder_node()) return Expr();
  if (is_compute_node()) return operaion->as<ir::ComputeOp>()->body.front();
  NOT_IMPLEMENTED;
}

Expr _Tensor_::tensor_store_expanded_body() {
  CHECK(!is_placeholder_node()) << "placeholder should not expand store";

  CHECK_EQ(shape.size(), axis.size());

  std::vector<Expr> axis_;
  for (int i = 0; i < axis.size(); i++) {
    axis_.push_back(Expr(axis[i]));
  }
  Expr final_body = body();

  auto *reduce_node = body().As<ir::Reduce>();
  if (reduce_node) {
    final_body = reduce_node->body;
    switch (reduce_node->reduce_type) {
      case ir::Reduce::kSum:
        final_body = Tensor(this)(axis_) + final_body;
        break;
      default:
        NOT_IMPLEMENTED
    }
  }

  return ir::Store::Make(Expr(Buffer(this)), final_body, axis_);
}

void _Tensor_::Bind(lang::Buffer &buffer) {
  // Extract the tensors thouse has binded to this buffer.
  buffer_depended_tensor_names_ = buffer.buffer()->binded_tensor_names();

  buffer.buffer()->BindTo(this);
  CHECK(!buffer->binded_tensor_names().empty());
  this->buffer = buffer.buffer();
  CHECK(this->buffer.defined());
  CHECK(!inlined());

  // Reset stage to nullptr to tell others this tensor should be inlined.
  InitStage();
}

void Tensor::ExpandInlined() {
  // Collect all the Calls with Tensors
  // Expand all the uninlined tensor.
}

}  // namespace ir
}  // namespace cinn
