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

namespace {

struct IrNodesCollector : public IRVisitor {
  using teller_t  = std::function<bool(const Expr*)>;
  using handler_t = std::function<void(const Expr*)>;

  teller_t teller;
  handler_t handler;

  IrNodesCollector(teller_t&& teller, handler_t&& handler) : teller(teller), handler(handler) {}

  void Visit(const Expr* expr) override {
    if (teller(expr)) handler(expr);

    switch (expr->node_type()) {
#define __(op__)           \
  case ir::IrNodeTy::op__: \
    return IRVisitor::Visit(expr->As<ir::op__>());

      NODETY_FORALL(__)

      default:
        LOG(FATAL) << "not supported NodeTy";
#undef __
    }
  }
};

}  // namespace

std::set<Expr> CollectIRNodes(Expr expr,
                              std::function<bool(const Expr*)> teller,
                              std::function<void(const Expr*)> handler) {
  IrNodesCollector collector(std::move(teller), std::move(handler));
}

}  // namespace ir
}  // namespace cinn
