#pragma once
#include "cinn/ir/buffer.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/tensor.h"

namespace cinn {
namespace ir {

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

#define __m(t__) virtual void Visit(const t__* x) = 0;
  NODETY_FORALL(__m)
#undef __m
};

}  // namespace ir
}  // namespace cinn
