#include "cinn/ir/ir_mutator.h"

#include "cinn/ir/ir_printer.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace ir {

void IRMutator::Visit(const Expr *expr, Expr *op) { IRVisitorBase::Visit(expr, op); }

#define UNARY_OP_IMPL(op__)                           \
  void IRMutator::Visit(const op__ *expr, Expr *op) { \
    auto *node = op->As<op__>();                      \
    IRVisitorBase::Visit(&node->v, &node->v);         \
  }

#define BINARY_OP_IMPL(op__)                          \
  void IRMutator::Visit(const op__ *expr, Expr *op) { \
    auto *node = op->As<op__>();                      \
    IRVisitorBase::Visit(&node->a, &node->a);         \
    IRVisitorBase::Visit(&node->b, &node->b);         \
  }

NODETY_UNARY_OP_FOR_EACH(UNARY_OP_IMPL)
NODETY_BINARY_OP_FOR_EACH(BINARY_OP_IMPL)

#undef UNARY_OP_IMPL
#undef BINARY_OP_IMPL

void IRMutator::Visit(const IntImm *expr, Expr *op) {}
void IRMutator::Visit(const UIntImm *expr, Expr *op) {}
void IRMutator::Visit(const FloatImm *expr, Expr *op) {}
void IRMutator::Visit(const Cast *expr, Expr *op) {
  auto *node = op->As<Cast>();
  Visit(&node->v, &node->v);
}
void IRMutator::Visit(const For *expr, Expr *op) {
  auto *node = op->As<For>();
  IRVisitorBase::Visit(&node->min, &node->min);
  IRVisitorBase::Visit(&node->extent, &node->extent);
  IRVisitorBase::Visit(&node->body, &node->body);
}
void IRMutator::Visit(const PolyFor *expr, Expr *op) {
  auto *node = op->As<PolyFor>();
  IRVisitorBase::Visit(&node->body, &node->body);
  IRVisitorBase::Visit(&node->condition, &node->condition);
  IRVisitorBase::Visit(&node->inc, &node->inc);
}
void IRMutator::Visit(const Select *expr, Expr *op) {
  auto *node = op->As<Select>();
  IRVisitorBase::Visit(&node->condition, &node->condition);
  IRVisitorBase::Visit(&node->true_value, &node->true_value);
  IRVisitorBase::Visit(&node->false_value, &node->false_value);
}
void IRMutator::Visit(const IfThenElse *expr, Expr *op) {
  auto *node = op->As<IfThenElse>();
  IRVisitorBase::Visit(&node->condition, &node->condition);
  Expr true_case(node->true_case);
  Expr false_case(node->false_case);
  IRVisitorBase::Visit(&node->true_case, &true_case);
  IRVisitorBase::Visit(&node->false_case, &false_case);
}
void IRMutator::Visit(const Block *expr, Expr *op) {
  auto *node = op->As<Block>();
  for (auto &expr : node->stmts) {
    IRVisitorBase::Visit(&expr, &expr);
  }
}
void IRMutator::Visit(const Call *expr, Expr *op) {
  auto *node = op->As<Call>();
  for (auto &expr : node->args) {
    IRVisitorBase::Visit(&expr, &expr);
  }
}
void IRMutator::Visit(const Module *expr, Expr *op) {}
void IRMutator::Visit(const _Var_ *expr, Expr *op) {}
void IRMutator::Visit(const Load *expr, Expr *op) {
  auto *node = op->As<Load>();
  IRVisitorBase::Visit(&node->index, &node->index);
}
void IRMutator::Visit(const Store *expr, Expr *op) {
  auto *node = op->As<Store>();
  IRVisitorBase::Visit(&node->value, &node->value);
  IRVisitorBase::Visit(&node->index, &node->index);
}
void IRMutator::Visit(const Alloc *expr, Expr *op) {
  auto *node = op->As<Alloc>();
  for (auto &e : node->extents) {
    IRVisitorBase::Visit(&e, &e);
  }
  IRVisitorBase::Visit(&node->condition, &node->condition);
  Expr body(node->body);
  IRVisitorBase::Visit(&node->body, &body);
}
void IRMutator::Visit(const Free *expr, Expr *op) {}
void IRMutator::Visit(const _Range_ *expr, Expr *op) {}
void IRMutator::Visit(const _IterVar_ *expr, Expr *op) {}
void IRMutator::Visit(const _Buffer_ *expr, Expr *op) {
  auto *node = op->As<_Buffer_>();

  for (auto &e : node->shape) {
    IRVisitorBase::Visit(&e, &e);
  }
  for (auto &e : node->strides) {
    IRVisitorBase::Visit(&e, &e);
  }
  IRVisitorBase::Visit(&node->elem_offset, &node->elem_offset);
}
void IRMutator::Visit(const _Tensor_ *expr, Expr *op) {
  auto *node = op->As<_Tensor_>();

  for (auto &e : node->shape) {
    IRVisitorBase::Visit(&e, &e);
  }
}

void IRMutator::Visit(const _LoweredFunc_ *expr, Expr *op) {
  auto *node = op->As<_LoweredFunc_>();
  IRVisitorBase::Visit(&node->body, &node->body);
}

}  // namespace ir
}  // namespace cinn
