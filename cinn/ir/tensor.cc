#include "cinn/ir/tensor.h"

#include <cstring>

#include "cinn/cinn.h"
#include "cinn/common/arithmatic.h"
#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/ir/operation.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace ir {

Tensor _Tensor_::Make(const std::string &name,
                      Type dtype,
                      const std::vector<Expr> &shape,
                      const std::vector<Expr> &domain,
                      FunctionRef fn,
                      const std::vector<Var> &reduce_axis) {
  CHECK(!name.empty()) << "Tensor name is set empty";
  auto n         = make_shared<_Tensor_>();
  n->name        = name;
  n->shape       = shape;
  n->domain      = domain;
  n->reduce_axis = reduce_axis;
  n->set_type(dtype);
  n->operation = fn;
  n->InitAxis();

  return Tensor(n);
}

size_t Tensor::ndims() const { return operator->()->shape.size(); }

std::set<std::string> _Tensor_::GetDependTensorNames() const {
  std::set<std::string> names;

  auto add_depend_tensors_from_expr = [&](Expr expr) {
    auto tensors =
        CollectIRNodes(expr, [&](const Expr *x) { return x->as_tensor() && x->as_tensor()->name != this->name; });
    for (auto &e : tensors) {
      names.insert(e.as_tensor()->name);
    }
  };

  if (is_compute_node()) {
    add_depend_tensors_from_expr(body());
  } else if (is_call_node()) {
    add_depend_tensors_from_expr(body());
  } else if (is_extern_call_node()) {
    add_depend_tensors_from_expr(body());
  } else if (is_placeholder_node()) {
    return names;
  } else {
    CINN_NOT_IMPLEMENTED
  }

  return names;
}

Expr Tensor::operator()(const std::vector<Expr> &indices) const {
  CHECK(!self()->is_tuple()) << "should extract a specific value from the tuple and operate on that instead";
  auto *node = operator->();

  CHECK_EQ(indices.size(), ndims()) << "number of indices not match the dimension";

  return Load::Make(*this, indices);
}

Expr _Tensor_::inline_expanded(const std::vector<Expr> &indices) {
  CHECK(is_compute_node());
  return get_compute_op()->producer_fn(indices);
}

const char *_Tensor_::operation_type() const {
  if (!operation.defined()) return "";
  return operation->as<ir::_Operation_>()->func_type();
}

bool _Tensor_::is_compute_node() const { return std::strcmp(operation_type(), ir::ComputeOp::__func_type__) == 0; }
bool _Tensor_::is_placeholder_node() const {
  return std::strcmp(operation_type(), ir::PlaceholderOp::__func_type__) == 0;
}
bool _Tensor_::is_call_node() const { return std::strcmp(operation_type(), ir::CallOp::__func_type__) == 0; }
bool _Tensor_::is_extern_call_node() const {
  if (std::strcmp(operation_type(), ir::CallOp::__func_type__) == 0) {
    auto *op   = operation->as<ir::CallOp>();
    auto *call = op->call_expr.As<ir::Call>();
    if (call) {
      return call->is_extern_call();
    }
  }
  return false;
}
bool _Tensor_::is_buffer_shared_node() const {
  return std::strcmp(operation_type(), ir::BufferShareOp::__func_type__) == 0;
}

bool _Tensor_::is_preceding_view_node() const {
  return std::strcmp(operation_type(), ir::PrecedingViewOp::__func_type__) == 0;
}

ComputeOp *_Tensor_::get_compute_op() const {
  if (!is_compute_node()) return nullptr;
  return operation->as<ComputeOp>();
}

PlaceholderOp *_Tensor_::get_placeholder_op() const {
  if (!is_placeholder_node()) return nullptr;
  return operation->as<PlaceholderOp>();
}

void _Tensor_::InitAxis() const {
  // CHECK(!domain_without_reduce_axis().empty());
  axis_ = common::GenDefaultAxis(domain_without_reduce_axis().size());
}

bool _Tensor_::has_expression() const {
  return (!is_placeholder_node()) && (!is_tuple_get()) && (!is_buffer_shared_node());
}

isl::set _Tensor_::GenerateIslDomain() const {
  // include the reduce axis.
  std::vector<poly::Dim> dims;

  if (has_expression()) {
    if (axis_.empty()) InitAxis();
    auto domain = domain_with_reduce_axis();
    CHECK_EQ(axis_with_reduce().size(), domain.size());
    auto _axis_with_reduce = axis_with_reduce();
    for (int i = 0; i < domain.size(); i++) {
      auto dim = domain[i];
      if (dim.is_constant()) {
        dims.emplace_back(_axis_with_reduce[i]->name, 0, dim.as_int32() - 1);
      } else {
        dims.emplace_back(_axis_with_reduce[i]->name, Expr(0), Sub::Make(dim, common::make_const(1)));
      }
    }
  }
  poly::Domain isl_domain(Context::Global().isl_ctx(), name, dims);
  VLOG(1) << "domain: " << isl_domain.__str__();
  return isl_domain.to_isl();
}

std::vector<Expr *> _Tensor_::expr_fields() {
  std::vector<Expr *> res;
  const char *func_type = operation->as<ir::_Operation_>()->func_type();
  if (operation.defined()) {
    if (is_compute_node()) {
      auto *op = operation->as<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
    } else if (is_placeholder_node()) {
      auto *op = operation->as<ir::PlaceholderOp>();
    } else if (is_call_node()) {
      auto *op = operation->as<ir::CallOp>();
      for (auto &expr : op->read_args()) res.push_back(&expr);
    } else if (is_buffer_shared_node()) {
    } else {
      CINN_NOT_IMPLEMENTED
    }
  }

  for (auto &e : shape) {
    res.push_back(&e);
  }
  for (auto &e : domain) {
    res.push_back(&e);
  }
  return res;
}

std::vector<const Expr *> _Tensor_::expr_fields() const {
  std::vector<const Expr *> res;
  const char *func_type = operation->as<ir::_Operation_>()->func_type();
  if (operation.defined()) {
    if (is_compute_node()) {
      auto *op = operation->as<ir::ComputeOp>();
      for (auto &expr : op->body) res.push_back(&expr);
    } else if (is_placeholder_node()) {
      auto *op = operation->as<ir::PlaceholderOp>();
    } else if (is_call_node()) {
      auto *op = operation->as<ir::CallOp>();
      for (auto &expr : op->read_args()) res.push_back(&expr);
    } else if (is_buffer_shared_node()) {
    } else {
      LOG(ERROR) << "func_type: " << func_type;
      CINN_NOT_IMPLEMENTED
    }
  }

  for (auto &e : shape) {
    res.push_back(&e);
  }
  for (auto &e : domain) {
    res.push_back(&e);
  }

  return res;
}

_Tensor_::~_Tensor_() {}

Expr _Tensor_::body() const {
  if (is_placeholder_node()) return Expr();
  if (is_buffer_shared_node()) return Expr();
  if (is_compute_node()) return operation->as<ir::ComputeOp>()->body.front();
  if (is_call_node()) return operation->as<ir::CallOp>()->call_expr;
  CINN_NOT_IMPLEMENTED;
}

Expr *_Tensor_::mutable_body() {
  if (is_placeholder_node()) return nullptr;
  if (is_buffer_shared_node()) return nullptr;
  if (is_compute_node()) return &operation->as<ir::ComputeOp>()->body.front();
  if (is_call_node()) return &operation->as<ir::CallOp>()->call_expr;
  CINN_NOT_IMPLEMENTED
}

ir::Tensor _Tensor_::InitReduction(poly::StageMap stages) const {
  CHECK(contains_reduce_axis()) << "InitReduction only works on a reduce tensor";
  // return if already rexists.
  std::string init_reduce_tensor_name = GenReduceInitTensorNameOf(name);
  if (stages->Lookup(init_reduce_tensor_name)) return stages[this]->LookupCtrlDepend(init_reduce_tensor_name);

  // create a new init tensor.
  auto init_tensor = lang::Compute(
      shape, [=](const std::vector<Expr> &axis) { return GetReduceInitVal(); }, UniqName(init_reduce_tensor_name));
  stages->InsertLazily(init_tensor);
  stages[this]->CtrlDepend(init_tensor);
  stages[this]->ShareBufferWith(stages[init_tensor]);
  return init_tensor;
}

Expr _Tensor_::tensor_store_expanded_body() {
  CHECK(!is_placeholder_node()) << "placeholder should not expand store";

  Expr final_body = body();
  if (shape.empty()) return final_body;

  std::vector<Expr> g_axis = common::GenDefaultAxisAsExpr(shape.size());

  auto *reduce_node = body().As<ir::Reduce>();
  if (reduce_node) {
    final_body = reduce_node->body;
    switch (reduce_node->reduce_type) {
      case ir::Reduce::kSum:
        final_body = Tensor(this)(g_axis) + final_body;
        break;
      case ir::Reduce::kMul:
        final_body = Tensor(this)(g_axis) * final_body;
        break;
      case ir::Reduce::kMax:
        final_body = Max::Make(Tensor(this)(g_axis), final_body);
        break;
      case ir::Reduce::kMin:
        final_body = Min::Make(Tensor(this)(g_axis), final_body);
        break;
      default:
        CINN_NOT_IMPLEMENTED
    }
  }

  if (is_tuple()) return final_body;

  return ir::Store::Make(Expr(Buffer(this)), final_body, g_axis);
}

void _Tensor_::Bind(lang::Buffer &buffer) {
  // CHECK(!inlined()) << "Inlined tensor should bing buffer";
  CHECK(!buffer->type().is_void());
  if (this->buffer.defined()) {
    // remove the old buffer
    if (this->buffer == buffer.buffer()) return;
    this->buffer->Unbind(this);
  }
  // Extract the tensors thouse has binded to this buffer.
  buffer_depended_tensor_names_ = buffer.buffer()->binded_tensor_names();

  buffer.buffer()->BindTo(this);
  CHECK(!buffer->binded_tensor_names().empty());
  this->buffer = buffer.buffer();
  CHECK(this->buffer.defined());
}

void _Tensor_::Bind(const Buffer &buffer) {
  lang::Buffer buf(buffer);
  Bind(buf);
}

void _Tensor_::WithBuffer(const Type &type) {
  Type buf_type = type.is_void() ? type_ : type;
  lang::Buffer buf(buf_type);
  buf->target = common::DefaultHostTarget();
  Bind(buf);
}

void _Tensor_::WithBuffer(const std::string &memory_type, const Type &type) {
  Type buf_type = type.is_void() ? type_ : type;
  lang::Buffer buf(buf_type);
  buf->target = common::DefaultHostTarget();
  Bind(buf);

  if (memory_type == "shared") {
    buf->memory_type = MemoryType::GPUShared;
  } else if (memory_type == "local") {
    buf->memory_type = MemoryType::GPULocal;
  } else {
    LOG(FATAL) << "Not supported memory type " << memory_type;
  }
}

bool _Tensor_::HasSameShapeWith(const Tensor &other) const {
  if (shape.size() != other->shape.size()) return false;

  for (int i = 0; i < shape.size(); i++) {
    Expr dim0 = common::AutoSimplify(shape[i]);
    Expr dim1 = common::AutoSimplify(other->shape[i]);

    if (dim0 != dim1) return false;
  }
  return true;
}

Tensor _Tensor_::TupleGet(int offset) const {
  CHECK(is_tuple());
  auto *call = body().As<ir::Call>();
  CHECK_LT(offset, call->write_args.size());
  auto tensor = call->write_args[offset].as_tensor_ref();
  tensor->WithBuffer();
  return tensor;
}

bool _Tensor_::is_tuple() const {
  if (!has_expression()) return false;
  auto *call = body().As<ir::Call>();
  if (call && call->is_extern_call() && !call->write_args.empty()) return true;
  return false;
}

std::vector<Expr> _Tensor_::domain_with_reduce_axis() const {
  if (reduce_axis.empty()) return domain;
  auto res = domain;
  for (const Var &axis : reduce_axis) {
    CHECK(axis->upper_bound.type().is_int(32)) << axis->upper_bound;
    res.push_back(axis->upper_bound);
  }
  return res;
}

bool operator<(const Tensor &a, const Tensor &b) { return a->name < b->name; }

Tensor::Tensor(const std::string &name,
               Type dtype,
               const std::vector<Expr> &shape,
               const std::vector<Expr> &domain,
               FunctionRef fn,
               const std::vector<Var> &reduce_axis)
    : IrNodeRef(_Tensor_::Make(name, dtype, shape, domain, fn, reduce_axis).self()) {}

bool _Tensor_::is_tuple_get() const {
  return is_call_node() && operation.defined() &&
         operation->as<ir::_Operation_>()->func_type() == ir::CallOp::__func_type__ &&
         operation->as<ir::CallOp>()->is_tuple_get;
}

bool _Tensor_::IsDependOnStatement(std::string_view statement) {
  if (!is_compute_node()) {
    return false;
  }

  auto depend_tensors = DependingTensorNames();
  for (const auto &x : depend_tensors) {
    if (x == statement) return true;
  }
  return false;
}

std::set<std::string> _Tensor_::DependingTensorNames() {
  std::set<std::string> res;
  if (body().defined()) {
    auto depend_tensors = ir::CollectIRNodes(body(), [](const Expr *x) -> bool { return x->as_tensor(); });
    for (const auto &x : depend_tensors) {
      if (x.get() != this) {
        res.insert(x.as_tensor()->name);
      }
    }
  }
  return res;
}

const std::vector<Var> &_Tensor_::axis() const {
  CHECK_EQ(axis_.size(), domain_without_reduce_axis().size());
  return axis_;
}

std::vector<Var> _Tensor_::axis_with_reduce() const {
  auto axis = axis_;
  axis.insert(axis.end(), reduce_axis.begin(), reduce_axis.end());
  return axis;
}

bool _Tensor_::Uses(const Tensor &other) {
  auto loads = ir::CollectIRNodes(body(), [&](const Expr *x) {
    auto *loadn = x->As<ir::Load>();
    if (!loadn) return false;
    return loadn->tensor.as_tensor()->name == other->name;
  });
  return !loads.empty();
}

ir::Tensor _Tensor_::Reshape(const std::vector<Expr> &shape, poly::StageMap stages) const {
  CHECK(!stages[this]->inlined());
  auto op    = BufferShareOp::Make();
  auto n     = make_shared<_Tensor_>();
  auto selft = Tensor(const_cast<ir::_Tensor_ *>(this));

  {
    Expr this_num_elements = Expr(1);
    for (auto &e : this->shape) this_num_elements = this_num_elements * e;

    Expr num_elements = Expr(1);
    for (auto &e : shape) num_elements = num_elements * e;

    CHECK(MathIsZero(this_num_elements - num_elements)) << "number of elements mismatch";
  }

  n->name   = Context::Global().NewName(name + "_reshape");
  n->shape  = shape;
  n->domain = shape;
  n->set_type(type());
  n->operation = op;
  n->InitAxis();

  auto t = Tensor(n);
  stages->InsertLazily(t);

  stages[n]->ShareBufferWith(stages[this]);
  stages[n]->CtrlDepend(selft);
  return t;
}

ir::Tensor _Tensor_::ReshapeCopied(const std::vector<Expr> &shape, poly::StageMap stages) const {
  auto t      = ir::Tensor(const_cast<ir::_Tensor_ *>(this));
  auto copied = Compute(
      domain,
      [=](const std::vector<Expr> &axis) { return t(axis); },
      Context::Global().NewName(this->name + "_copied"));
  stages->InsertLazily(copied);
  auto res = copied->Reshape(shape, stages);
  stages->InsertLazily(res);
  return res;
}

Shared<poly::Stage> CreateStage(Tensor tensor) {
  auto isl_domain = tensor->GenerateIslDomain();
  return poly::Stage::New(isl_domain, tensor->body(), tensor.self());
}

std::string GenReduceInitTensorNameOf(const std::string &tensor_name) { return tensor_name + "__reduce_init"; }

bool _Tensor_::is_reduce_sum() const {
  if (!contains_reduce_axis()) return false;
  return body().As<ir::Reduce>() && body().As<ir::Reduce>()->reduce_type == ir::Reduce::ReduceType::kSum;
}
bool _Tensor_::is_reduce_mul() const {
  if (!contains_reduce_axis()) return false;
  return body().As<ir::Reduce>() && body().As<ir::Reduce>()->reduce_type == ir::Reduce::ReduceType::kMul;
}

Expr _Tensor_::GetReduceInitVal() const {
  CHECK(is_reduce_tensor());
  return body().As<ir::Reduce>()->init;
}

}  // namespace ir
}  // namespace cinn
