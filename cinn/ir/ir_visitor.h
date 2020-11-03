#pragma once
#include <functional>
#include <set>

#include "cinn/ir/buffer.h"
#include "cinn/ir/collect_ir_nodes.h"
#include "cinn/ir/intrinsic_ops.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/lowered_func.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace ir {

struct _Tensor_;

/**
 * Base class of all the methods visit the IR tree.
 * @param RetTy return type.
 * @param Args type of the extra arguments passed to the all the methods.
 */
template <typename RetTy = void, typename... Args>
struct IRVisitorBase {
  //! Visit a expression.
  // @{
  virtual RetTy Visit(const ir::Expr* expr, Args... args) {
    CHECK(expr->defined());
    switch (expr->node_type()) {
#define __(op__)           \
  case ir::IrNodeTy::op__: \
    return Visit(expr->As<ir::op__>(), args...);

      NODETY_FORALL(__)

      default:
        LOG(FATAL) << "not supported NodeTy";
#undef __
    }
    return RetTy();
  }
  // @}

 protected:
#define __(op__) virtual RetTy Visit(const ir::op__* op, Args... args) = 0;
  NODETY_FORALL(__)
#undef __
};

/**
 * Base of all the Ir readonly visitor.
 */
struct IRVisitor : public IRVisitorBase<void> {
  IRVisitor() = default;

  void Visit(const Expr* x) { IRVisitorBase::Visit(x); }
#define __m(t__) \
  virtual void Visit(const t__* x) {}
  NODETY_FORALL(__m)
#undef __m
};

// std::set<Expr> CollectIRNodes(Expr expr, std::function<bool(const Expr*)> teller);

bool operator==(Expr a, Expr b);
bool operator!=(Expr a, Expr b);

}  // namespace ir
}  // namespace cinn
