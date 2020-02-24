#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

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
    return Visit(expr->As<ir::op__>());

      NODETY_FORALL(__)

      default:
        LOG(FATAL) << "not supported NodeTy";
#undef __
    }
  }

#define __m(t__)                       \
  void Visit(const t__* x) override {  \
    for (auto* n : x->expr_fields()) { \
      Visit(n);                        \
    }                                  \
  }

  NODETY_FORALL(__m)
#undef __m
};

}  // namespace

std::set<Expr> CollectIRNodes(Expr expr, std::function<bool(const Expr*)> teller) {
  std::set<Expr> exprs;
  IrNodesCollector::handler_t handler = [&](const Expr* x) {
    auto* call = x->As<Call>();
    exprs.insert(*x);
  };
  IrNodesCollector collector(std::move(teller), std::move(handler));
  collector.Visit(&expr);
  return exprs;
}

}  // namespace ir
}  // namespace cinn
