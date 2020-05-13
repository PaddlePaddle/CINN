#include "cinn/ir/ir_visitor.h"

#include <unordered_set>

#include "cinn/ir/ir_printer.h"
#include "cinn/lang/tensor.h"
#include "cinn/utils/string.h"

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
    if (!expr->defined()) return;
    if (visited_.count(expr->get())) return;
    if (teller(expr)) handler(expr);
    visited_.insert(expr->get());

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
      if (n->defined()) Visit(n);      \
    }                                  \
  }

  NODETY_FORALL(__m)
#undef __m
  std::unordered_set<void*> visited_;
};

}  // namespace

std::set<Expr> CollectIRNodes(Expr expr, std::function<bool(const Expr*)> teller) {
  std::set<Expr> exprs;
  IrNodesCollector::handler_t handler = [&](const Expr* x) { exprs.insert(*x); };
  IrNodesCollector collector(std::move(teller), std::move(handler));
  collector.Visit(&expr);
  return exprs;
}

bool operator==(Expr a, Expr b) {
  if (a.get() == b.get()) return true;
  // TODO(Superjomn) implement with a more accurate one
  return utils::GetStreamCnt(a) == utils::GetStreamCnt(b);
}

bool operator!=(Expr a, Expr b) { return !(a == b); }

}  // namespace ir
}  // namespace cinn
