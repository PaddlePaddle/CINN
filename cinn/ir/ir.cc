#include "cinn/ir/ir.h"

#include <map>
#include <string>
#include <vector>

#include "cinn/common/pod_value.h"
#include "cinn/ir/ir_printer.h"
#include "cinn/ir/ir_visitor.h"

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

void Cast::Accept(IRVisitor *v) const { v->IRVisitorBase::Visit(&this->v); }

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

Expr Minus::Make(Expr a) {
  auto node = make_shared<Minus>(a);
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

Expr Let::Make(Expr value, Expr body) {
  auto *n = make_shared<Let>();
  CHECK(value.type().valid());
  CHECK(body.type().valid());
  n->value = value;
  n->body  = body;
  n->set_type(n->value->type());
  return Expr(n);
}

Expr _Var_::Make(const std::string &name, const Type &type) {
  auto node = new _Var_(name, type);
  return Expr(node);
}

Expr _Var_::Make(Expr lower_bound, Expr upper_bound, const std::string &name) {
  auto *n           = make_shared<_Var_>();
  n->lower_bound    = lower_bound;
  n->upper_bound    = upper_bound;
  n->is_reduce_axis = true;
  n->name           = name;
  n->set_type(lower_bound.type());
  return Expr(n);
}

Expr _Var_::Copy() const {
  auto *n           = make_shared<_Var_>();
  n->name           = name;
  n->is_reduce_axis = is_reduce_axis;
  n->lower_bound    = lower_bound;
  n->upper_bound    = upper_bound;
  n->set_type(type());
  return Expr(n);
}

Expr For::Make(Var loop_var, Expr min, Expr extent, ForType for_type, DeviceAPI device_api, Expr body) {
  auto node = make_shared<For>();
  CHECK(loop_var.defined());
  CHECK(min.defined());
  CHECK(extent.defined());
  node->loop_var   = loop_var;
  node->min        = min;
  node->extent     = extent;
  node->for_type   = for_type;
  node->device_api = device_api;
  node->body       = body;
  return Expr(node);
}

std::vector<Expr *> For::expr_fields() { return {&min, &extent, &body}; }
std::vector<const Expr *> For::expr_fields() const { return {&min, &extent, &body}; }

Expr Block::Make(const std::vector<Expr> &stmts) {
  auto node   = make_shared<Block>();
  node->stmts = stmts;
  return Expr(node);
}
std::vector<Expr *> Block::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : stmts) res.push_back(&x);
  return res;
}
std::vector<const Expr *> Block::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : stmts) res.push_back(&x);
  return res;
}

Expr IfThenElse::Make(Expr condition, Expr true_case, Expr false_case) {
  auto node = make_shared<IfThenElse>(condition, true_case, false_case);
  return Expr(node);
}

IfThenElse::IfThenElse(Expr condition, Expr true_case, Expr false_case)
    : ExprNode(Type()), condition(condition), true_case(true_case), false_case(false_case) {
  CHECK(condition.defined());
  CHECK(true_case.defined());
}
std::vector<Expr *> IfThenElse::expr_fields() { return {&condition, &true_case, &false_case}; }
std::vector<const Expr *> IfThenElse::expr_fields() const { return {&condition, &true_case, &false_case}; }

Expr Store::Make(Expr tensor, Expr value, Expr index) {
  CHECK(tensor.As<_Tensor_>()) << "tensor should be _Tensor_ type";
  auto node    = make_shared<Store>();
  node->tensor = tensor;
  node->value  = value;
  node->index  = index;
  node->set_type(tensor->type().ElementOf().with_lanes(index->type().lanes()));
  return Expr(node);
}

Expr Alloc::Make(Var buffer_var, Type type, const std::vector<Expr> &extents, Expr condition, Expr body) {
  auto node        = make_shared<Alloc>();
  node->buffer_var = buffer_var;
  node->extents    = extents;
  node->condition  = condition;
  node->body       = body;
  node->set_type(type);
  return Expr(node);
}

int32_t Alloc::ConstantAllocationSize() const {
  auto *var = buffer_var.As<_Var_>();
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
std::vector<Expr *> Alloc::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : extents) res.push_back(&x);
  res.push_back(&condition);
  res.push_back(&body);
  return res;
}
std::vector<const Expr *> Alloc::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : extents) res.push_back(&x);
  res.push_back(&condition);
  res.push_back(&body);
  return res;
}

Expr Free::Make(Var var) {
  auto node = make_shared<Free>();
  node->var = var;
  return Expr(node);
}

void _Range_::Accept(IRVisitor *v) const { v->Visit(this); }

Range::Range(_Range_ *n) : IrNodeRef(n) {}

void _IterVar_::Accept(IRVisitor *v) const { v->Visit(this); }

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
                int value_index,
                Expr tensor) {
  for (size_t i = 0; i < args.size(); ++i) {
    CHECK(args[i].defined());
  }
  if (call_type == Halide) {
    for (size_t i = 0; i < args.size(); ++i) {
      CHECK(args[i].type().is_int()) << "get type " << args[i].type();
    }
  }

  auto node         = common::make_shared<Call>(type);
  node->name        = name;
  node->args        = args;
  node->call_type   = call_type;
  node->func        = func;
  node->value_index = value_index;
  node->tensor      = tensor;
  node->set_type(type);
  return Expr(node);
}
std::vector<Expr *> Call::expr_fields() {
  std::vector<Expr *> res;
  for (auto &x : args) res.push_back(&x);
  res.push_back(&tensor);
  return res;
}
std::vector<const Expr *> Call::expr_fields() const {
  std::vector<const Expr *> res;
  for (auto &x : args) res.push_back(&x);
  res.push_back(&tensor);
  return res;
}

Expr PolyFor::Make(
    Var iterator, Expr init_val, Expr condition, Expr inc, ForType for_type, DeviceAPI device_api, Expr body) {
  auto n        = make_shared<PolyFor>();
  n->iterator   = iterator;
  n->init       = init_val;
  n->condition  = condition;
  n->inc        = inc;
  n->for_type   = for_type;
  n->device_api = device_api;
  n->body       = body;
  return Expr(n);
}
std::vector<Expr *> PolyFor::expr_fields() { return {&init, &condition, &inc, &body}; }
std::vector<const Expr *> PolyFor::expr_fields() const { return {&init, &condition, &inc, &body}; }

Expr PolyFor::extent() const {
  auto nodes = CollectIRNodes(condition, [&](const Expr *e) {
    return e->As<NE>() ||   //
           e->As<EQ>() ||   //
           e->As<Min>() ||  //
           e->As<Max>();
  });

  if (nodes.empty()) {
    return Expr();
  }

  auto *le_n = condition.As<LE>();
  auto *lt_n = condition.As<LT>();
  if (!(le_n || lt_n)) return Expr();

  if (le_n) {
    if (le_n->a != Expr(iterator)) return Expr();
    auto *le_b_int = le_n->b.As<IntImm>();
    if (le_b_int) return Expr(make_shared<IntImm>(Int(32), le_b_int->value + 1));
    return Add::Make(le_n->b, Expr(1));
  }

  if (lt_n) {
    if (lt_n->a != Expr(iterator)) return Expr();
    return lt_n->b;
  }
  return Expr();
}

bool Var::operator==(const Var &o) const { return o->name == operator->()->name; }
bool Var::operator!=(const Var &o) const { return !(*this == o); }

Var &Var::operator=(_Var_ *x) {
  *this = Var(x);
  return *this;
}

Var &Var::operator=(const _Var_ *x) {
  *this = x->Copy();
  return *this;
}

Expr Load::Make(Expr tensor, Expr index) {
  CHECK(tensor.As<ir::_Tensor_>()) << "Load's address should be a tensor";
  CHECK(tensor->type().valid());
  CHECK(index.type().is_int(32));
  auto node    = make_shared<Load>();
  node->tensor = tensor;
  node->index  = index;
  node->set_type(tensor->type().ElementOf().with_lanes(index.type().lanes()));
  return Expr(node);
}

Expr Ramp::Make(Expr base, Expr stride, int lanes) {
  CHECK(base.defined());
  CHECK(stride.defined());
  CHECK(base.type().valid());
  CHECK(stride.type().valid());
  CHECK_EQ(stride.type(), Int(32));
  CHECK_GT(lanes, 0);

  auto *n   = make_shared<Ramp>();
  n->base   = base;
  n->stride = stride;
  n->lanes  = lanes;
  Type type(base.type().type(), base.type().bits(), lanes);
  n->set_type(type);
  return Expr(n);
}

Expr Broadcast::Make(Expr value, int lanes) {
  CHECK(value.defined());
  CHECK(value.type().valid());

  auto *n  = make_shared<Broadcast>();
  n->value = value;
  n->lanes = lanes;

  Type type(value.type().type(), value.type().bits(), lanes);
  n->set_type(type);

  return Expr(n);
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
