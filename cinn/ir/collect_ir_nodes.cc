#include "cinn/ir/collect_ir_nodes.h"

#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

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

    if (teller(expr)) {
      handler(expr);
    }
    visited_.insert(expr->get());

    switch (expr->node_type()) {
#define __(op__)                 \
  case ir::IrNodeTy::op__:       \
    Visit(expr->As<ir::op__>()); \
    break;

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
  std::set<void*> visited_;
};

}  // namespace

std::set<Expr> CollectIRNodes(Expr expr, std::function<bool(const Expr*)>&& teller) {
  std::set<Expr> exprs;
  IrNodesCollector::handler_t handler = [&](const Expr* x) { exprs.insert(*x); };
  IrNodesCollector collector(std::move(teller), std::move(handler));
  collector.Visit(&expr);
  return exprs;
}

std::map<std::string, Expr> CollectTensorMap(Expr x, std::function<bool(const Expr*)>&& extra_teller) {
  std::map<std::string, Expr> tensor_map;

  auto tensors = CollectIRNodes(x, [&](const Expr* x) { return x->as_tensor() && extra_teller(x); });
  for (auto& e : tensors) {
    auto* t             = e.as_tensor();
    tensor_map[t->name] = e;
  }
  return tensor_map;
}

}  // namespace ir
}  // namespace cinn
