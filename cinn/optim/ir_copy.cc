#include "cinn/optim/ir_copy.h"

#include <memory>
#include <string>
#include <vector>

#include "cinn/common/common.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn {
namespace optim {
using namespace ir;  // NOLINT

struct IRCopyVisitor : public ir::IRVisitorBase<Expr> {
  Expr Visit(const Expr* op) override { return IRVisitorBase::Visit(op); }

 protected:
  // The methods of ir nodes follows the order defined in node.h

  Expr Visit(const ir::IntImm* op) override { return Expr(make_shared<IntImm>(op->type(), op->value)); }
  Expr Visit(const ir::UIntImm* op) override { return Expr(make_shared<UIntImm>(op->type(), op->value)); }
  Expr Visit(const ir::FloatImm* op) override { return Expr(make_shared<FloatImm>(op->type(), op->value)); }

  Expr Visit(const ir::Cast* op) override {
    auto v = Visit(&op->v());
    return Cast::Make(op->type(), v);
  }

  Expr Visit(const ir::PolyFor* op) override {
    auto init      = Visit(&op->init);
    auto condition = Visit(&op->condition);
    auto inc       = Visit(&op->inc);
    auto body      = Visit(&op->body);
    return PolyFor::Make(op->iterator, init, condition, inc, op->for_type, op->device_api, body);
  }

  Expr Visit(const Select* op) override {
    auto condition   = Visit(&op->condition);
    auto true_value  = Visit(&op->true_value);
    auto false_value = Visit(&op->false_value);
    return Select::Make(condition, true_value, false_value);
  }

  Expr Visit(const IfThenElse* op) override {
    auto condition = Visit(&op->condition);
    auto true_case = Visit(&op->true_case);
    Expr false_case;
    if (op->false_case.defined()) Visit(&op->false_case);
    return IfThenElse::Make(condition, true_case, false_case);
  }

  Expr Visit(const Block* op) override {
    std::vector<Expr> stmts;
    for (auto& s : op->stmts) {
      stmts.push_back(Visit(&s));
    }
    return Block::Make(stmts);
  }

  Expr Visit(const Call* op) override {
    auto args = Visit(op->args);
    return Call::Make(op->type(), op->name, args, op->call_type);
  }

  Expr Visit(const _Var_* op) override {
    auto* n = make_shared<_Var_>();

    n->name           = op->name;
    n->is_reduce_axis = op->is_reduce_axis;
    n->set_type(op->type());

    if (n->is_reduce_axis) {
      auto lower_bound = Visit(&op->lower_bound);
      auto upper_bound = Visit(&op->upper_bound);
      n->lower_bound   = lower_bound;
      n->upper_bound   = upper_bound;
    }
    return Expr(n);
  }

  Expr Visit(const Load* op) override {
    auto tensor = Visit(&op->tensor);
    std::vector<Expr> indices;
    for (auto& idx : op->indices) {
      indices.push_back(Visit(&idx));
    }
    return Load::Make(tensor, indices);
  }

  Expr Visit(const Store* op) override {
    auto tensor = Visit(&op->tensor);
    auto value  = Visit(&op->value);
    std::vector<Expr> indices;
    for (auto& idx : op->indices) indices.push_back(Visit(&idx));

    return Store::Make(tensor, value, indices);
  }

  Expr Visit(const Alloc* op) override {
    auto extents   = Visit(op->extents);
    auto condition = Visit(&op->condition);
    auto body      = Visit(&op->body);

    return Alloc::Make(op->buffer_var, op->type(), extents, condition, body);
  }

  Expr Visit(const Free* op) override { return Free::Make(op->var); }

  Expr Visit(const _Buffer_* op) override {
    auto shape         = Visit(op->shape);
    auto strides       = Visit(op->strides);
    auto name          = op->name;
    auto scope         = op->scope;
    int data_alignment = op->data_alignment;
    auto elem_offset   = Visit(&op->elem_offset);
    int offset_factor  = op->offset_factor;
    Target target      = op->target;

    auto new_node            = _Buffer_::Make(name);
    new_node->shape          = shape;
    new_node->strides        = strides;
    new_node->name           = name;
    new_node->scope          = scope;
    new_node->data_alignment = data_alignment;
    new_node->elem_offset    = elem_offset;
    new_node->offset_factor  = offset_factor;
    new_node->target         = target;
    new_node->set_type(op->type());
    op->CopyMeta(new_node.As<ir::_Buffer_>());
    return Expr(ir::Buffer(new_node));
  }

  Expr Visit(const _Tensor_* op) override {
    auto shape       = Visit(op->shape);
    auto domain      = Visit(op->domain);
    auto axis        = op->axis;
    auto buffer_expr = Expr(op->buffer);
    // TODO(Superjomn) copy the operation.
    auto operaion    = op->operaion;
    auto name        = op->name;
    auto buffer      = Visit(&buffer_expr);
    auto tensor      = make_shared<_Tensor_>();
    tensor->domain   = domain;
    tensor->shape    = shape;
    tensor->axis     = axis;
    tensor->operaion = operaion;
    tensor->name     = name;
    tensor->buffer   = ir::Buffer(buffer.As<_Buffer_>());
    return tensor;
  }

  Expr Visit(const For* op) override {
    auto extent = Visit(&op->extent);
    auto min    = Visit(&op->min);
    auto body   = Visit(&op->body);

    return ir::For::Make(op->loop_var, min, extent, op->for_type, op->device_api, body);
  }

  Expr Visit(const _Range_* op) override {
    LOG(FATAL) << "not implemented";
    return Expr();
  }

  Expr Visit(const Module* op) override {
    LOG(FATAL) << "not implemented";
    return Expr();
  }

  Expr Visit(const _LoweredFunc_* op) override {
    auto name = op->name;
    auto args = op->args;
    auto body = Visit(&op->body);

    return _LoweredFunc_::Make(name, args, body);
  }

  Expr Visit(const _IterVar_* op) override {
    LOG(FATAL) << "not implemented";
    return Expr();
  }

  Expr Visit(const Let* op) override {
    auto value = Visit(&op->value);
    auto body  = Visit(&op->body);
    return Let::Make(value, body);
  }

  Expr Visit(const Reduce* op) override {
    auto init = Visit(&op->init);
    auto body = Visit(&op->body);
    return Reduce::Make(op->reduce_type, init, body);
  }

  Expr Visit(const Ramp* op) override {
    auto base   = Visit(&op->base);
    auto stride = Visit(&op->stride);
    int lanes   = op->lanes;
    return Ramp::Make(base, stride, lanes);
  }

  Expr Visit(const Broadcast* op) override {
    auto value = Visit(&op->value);
    int lanes  = op->lanes;
    CHECK(value.defined());
    CHECK(value.type().valid());

    auto* n  = make_shared<Broadcast>();
    n->value = value;
    n->lanes = lanes;
    return Expr(n);
  }

  Expr Visit(const FracOp* op) override {
    auto a = Visit(&op->a());
    auto b = Visit(&op->b());
    CHECK(a.defined());
    CHECK(b.defined());

    auto* n = make_shared<FracOp>();
    n->a()  = a;
    n->b()  = b;
    return Expr(n);
  }

  Expr Visit(const Power* op) override {
    auto a = Visit(&op->a());
    auto b = Visit(&op->b());
    CHECK(a.defined());
    CHECK(b.defined());

    auto* n = make_shared<Power>();
    n->a()  = a;
    n->b()  = b;
    return Expr(n);
  }

  Expr Visit(const Product* op) override {
    std::vector<Expr> operands;
    for (auto& v : op->operands()) {
      operands.push_back(Visit(&v));
    }
    return Product::Make(operands);
  }

  Expr Visit(const Sum* op) override {
    std::vector<Expr> operands;
    for (auto& v : op->operands()) {
      operands.push_back(Visit(&v));
    }
    return Sum::Make(operands);
  }

#define OP_BINARY_HANDLE(op__)               \
  Expr Visit(const ir::op__* op) override {  \
    auto a = IRVisitorBase::Visit(&op->a()); \
    auto b = IRVisitorBase::Visit(&op->b()); \
    return op__::Make(a, b);                 \
  }
  NODETY_BINARY_OP_FOR_EACH(OP_BINARY_HANDLE)
#undef OP_BINARY_HANDLE

#define OP_UNARY_HANDLE(op__)                \
  Expr Visit(const op__* op) override {      \
    auto v = IRVisitorBase::Visit(&op->v()); \
    return op__::Make(v);                    \
  }
  NODETY_UNARY_OP_FOR_EACH(OP_UNARY_HANDLE)
#undef OP_UNARY_HANDLE

  std::vector<Expr> Visit(const std::vector<Expr>& vs) {
    std::vector<Expr> copied;
    for (auto& e : vs) {
      copied.push_back(Visit(&e));
    }
    return copied;
  }
};

Expr IRCopy(Expr x) {
  IRCopyVisitor visitor;
  return visitor.Visit(&x);
}

}  // namespace optim
}  // namespace cinn
