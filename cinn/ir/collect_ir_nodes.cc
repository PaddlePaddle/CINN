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

std::set<Expr> CollectLoadTensors(Expr x, std::function<bool(const Expr*)>&& teller) {
  struct Mutator : public ir::IRMutator<const Expr*> {
    std::function<bool(const Expr*)> teller;
    std::set<Expr> exprs;
    Mutator(std::function<bool(const Expr*)>&& teller) : teller(std::move(teller)) {}

    void operator()(const Expr* expr) { ir::IRMutator<const Expr*>::Visit(expr, expr); }

    void Visit(const Load* op, const Expr* expr) override {
      if (teller(&op->tensor)) exprs.insert(op->tensor);
    }
  };

  Mutator mutator(std::move(teller));
  mutator(&x);
  return mutator.exprs;
}

std::set<Expr> CollectStoreTensors(Expr x, std::function<bool(const Expr*)>&& teller) {
  struct Mutator : public ir::IRMutator<const Expr*> {
    std::function<bool(const Expr*)> teller;
    std::set<Expr> exprs;
    Mutator(std::function<bool(const Expr*)>&& teller) : teller(std::move(teller)) {}

    void operator()(const Expr* expr) { ir::IRMutator<const Expr*>::Visit(expr, expr); }

    void Visit(const Store* op, const Expr* expr) override {
      if (teller(&op->tensor)) exprs.insert(op->tensor);
    }
  };

  Mutator mutator(std::move(teller));
  mutator(&x);
  return mutator.exprs;
}

std::set<Expr> CollectReferencedTensors(Expr x, const std::function<bool(const Expr*)>& teller) {
  auto handle0 = teller;
  auto handle1 = teller;

  auto ts0 = CollectLoadTensors(x, std::move(handle0));
  auto ts1 = CollectLoadTensors(x, std::move(handle1));

  for (auto& item : ts1) {
    ts0.insert(item);
  }
  return ts0;
}

}  // namespace ir
}  // namespace cinn
