#include "cinn/ir/ir.h"

#include "cinn/common/pod_value.h"
#include "cinn/ir/ir_visitor.h"
#include "cinn/lang/tensor.h"

namespace cinn {

namespace common {

PODValue::operator ir::Var() const {
  if (type_code_ == TypeCode<std::nullptr_t>()) return ir::Var();
  void *handle = *this;
  return ir::Var(static_cast<ir::IrNode *>(handle));
}
PODValue::operator ir::Expr() const {
  if (type_code_ == TypeCode<std::nullptr_t>()) return ir::Expr();
  void *handle = *this;
  return ir::Expr(static_cast<ir::IrNode *>(handle));
}

}  // namespace common

namespace ir {
using common::make_shared;

Expr Cast::Make(Type t, Expr v) {
  auto node = make_shared<Cast>(t, v);
  return Expr(node);
}

Expr Add::Make(Expr a, Expr b) {
  auto node = make_shared<Add>(a, b);
  return Expr(node);
}

Expr Sub::Make(Expr a, Expr b) {
  auto node = make_shared<Sub>(a, b);
  return Expr(node);
}

Expr Mul::Make(Expr a, Expr b) {
  auto node = make_shared<Mul>(a, b);
  return Expr(node);
}

Expr Div::Make(Expr a, Expr b) {
  auto node = make_shared<Div>(a, b);
  return Expr(node);
}

Expr Mod::Make(Expr a, Expr b) {
  auto node = make_shared<Mod>(a, b);
  return Expr(node);
}

Expr Min::Make(Expr a, Expr b) {
  auto node = make_shared<Min>(a, b);
  return Expr(node);
}

Expr Max::Make(Expr a, Expr b) {
  auto node = make_shared<Max>(a, b);
  return Expr(node);
}

Expr EQ::Make(Expr a, Expr b) {
  auto node = make_shared<EQ>(a, b);
  return Expr(node);
}

Expr NE::Make(Expr a, Expr b) {
  auto node = make_shared<NE>(a, b);
  return Expr(node);
}

Expr LT::Make(Expr a, Expr b) {
  auto node = make_shared<LT>(a, b);
  return Expr(node);
}

Expr LE::Make(Expr a, Expr b) {
  auto node = make_shared<LE>(a, b);
  return Expr(node);
}

Expr GT::Make(Expr a, Expr b) {
  auto node = make_shared<GT>(a, b);
  return Expr(node);
}

Expr GE::Make(Expr a, Expr b) {
  auto node = make_shared<GE>(a, b);
  return Expr(node);
}

Expr And::Make(Expr a, Expr b) {
  auto node = make_shared<And>(a, b);
  return Expr(node);
}

Expr Or::Make(Expr a, Expr b) {
  auto node = make_shared<Or>(a, b);
  return Expr(node);
}

Expr Not::Make(Expr v) {
  auto node = make_shared<Not>(v);
  return Expr(node);
}

Expr Variable::Make(const std::string &name, const Type &type) {
  auto node = new Variable(name, type);
  return Expr(node);
}

For::For(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body) {
  CHECK(min.defined());
  CHECK(extent.defined());
  CHECK(body.defined());

  this->min        = std::move(min);
  this->extent     = std::move(extent);
  this->for_type   = std::move(for_type);
  this->device_api = device_api;
  this->body       = std::move(body);
}

Stmt For::Make(Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Stmt body) {
  auto node = make_shared<For>(min, extent, for_type, device_api, body);
  return Stmt(node);
}

Stmt Block::Make(const std::vector<Stmt> &stmts) {
  auto node   = make_shared<Block>();
  node->stmts = stmts;
  return Stmt(node);
}

Stmt IfThenElse::Make(Expr condition, Stmt true_case, Stmt false_case) {
  auto node = make_shared<IfThenElse>(condition, true_case, false_case);
  return Stmt(node);
}

Stmt Store::Make(Var buffer_var, Expr value, Expr index) {
  auto node        = make_shared<Store>();
  node->buffer_var = buffer_var;
  node->value      = value;
  node->index      = index;
  return Stmt(node);
}

Stmt Alloc::Make(Var buffer_var, Type type, const std::vector<Expr> &extents, Expr condition, Stmt body) {
  auto node        = make_shared<Alloc>();
  node->buffer_var = buffer_var;
  node->type       = type;
  node->extents    = extents;
  node->condition  = condition;
  node->body       = body;
  return Stmt(node);
}

int32_t Alloc::ConstantAllocationSize() const {
  auto *var = buffer_var.As<Variable>();
  CHECK(var);
  return ConstantAllocationSize(var->name, extents);
}

int32_t Alloc::ConstantAllocationSize(const std::string &name, const std::vector<Expr> &extents) {
  int32_t res{1};
  for (auto &e : extents) {
    auto *p = e.As<IntImm>();
    CHECK(p) << "extent should be IntImm";
    res *= p->value;
  }
  return res;
}

Stmt Free::Make(Var var) {
  auto node = make_shared<Free>();
  node->var = var;
  return Stmt(node);
}

void _Range_::Accept(IrVisitor *v) const { v->Visit(this); }

Range::Range(_Range_ *n) : IrNodeRef(n) {}

void _IterVar_::Accept(IrVisitor *v) const { v->Visit(this); }

IterVar _IterVar_::Make(Range dom, Var var, IterVarType iter_type, const std::string &thread_tag) {
  auto node        = common::make_shared<_IterVar_>();
  node->dom        = dom;
  node->var        = var;
  node->iter_type  = iter_type;
  node->thread_tag = thread_tag;
  return IterVar(IrNodeRef(node));
}

Expr Call::Make(Type type,
                const std::string &name,
                const std::vector<Expr> &args,
                Call::CallType call_type,
                FunctionRef func,
                int value_index) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined());
  }
  if (call_type == Halide) {
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args[i].type().is_int());
    }
  }

  auto node         = common::make_shared<Call>(type);
  node->name        = name;
  node->args        = args;
  node->call_type   = call_type;
  node->func        = func;
  node->value_index = value_index;
  node->set_type(type);
  return Expr(node);
}

void _Tensor_::Accept(IrVisitor *v) const { v->Visit(this); }

lang::Tensor _Tensor_::Make(const std::vector<Expr> &shape,
                            const std::vector<Var> &iterators,
                            Type dtype,
                            ir::Expr expr) {
  CHECK_EQ(shape.size(), iterators.size()) << "dimension of the shape and the iterators should match";
  auto n       = common::make_shared<_Tensor_>();
  n->dtype     = dtype;
  n->shape     = shape;
  n->expr      = expr;
  n->iterators = iterators;
  return lang::Tensor(n);
}

}  // namespace ir

namespace common {

template <>
void PODValue::Set<ir::Var>(ir::Var v) {
  type_code_      = TypeCode<ir::Var>();
  value_.v_handle = v.ptr();
}
template <>
void PODValue::Set<ir::Expr>(ir::Expr v) {
  type_code_      = TypeCode<ir::Expr>();
  value_.v_handle = v.ptr();
}
template <>
Value ToValue<ir::Expr>(ir::Expr v) {
  Value val;
  val.v_handle = v.ptr();
  return val;
}
template <>
Value ToValue<ir::Var>(ir::Var v) {
  Value val;
  val.v_handle = v.ptr();
  return val;
}

}  // namespace common
}  // namespace cinn
