#include "cinn/optim/reduce_expand.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"

namespace cinn::optim {

namespace {

struct Mutator : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

  ir::Tensor tensor;

  void Visit(const ir::Store* op, Expr* expr) {
    tensor = op->tensor.as_tensor_ref();
    ir::IRMutator<>::Visit(op, expr);
  }

  void Visit(const ir::Reduce* op, Expr* expr) {
    auto* node = expr->As<ir::Reduce>();
    CHECK(tensor.defined());

    std::vector<Expr> g_axis = common::GenDefaultAxisAsExpr(tensor->shape.size());

    Expr final_body = node->body;
    switch (node->reduce_type) {
      case ir::Reduce::kSum:
        final_body = tensor(g_axis) + final_body;
        break;
      case ir::Reduce::kMul:
        final_body = tensor(g_axis) * final_body;
        break;
      case ir::Reduce::kMax:
        final_body = ir::Max::Make(tensor(g_axis), final_body);
        break;
      case ir::Reduce::kMin:
        final_body = ir::Min::Make(tensor(g_axis), final_body);
        break;
      default:
        CINN_NOT_IMPLEMENTED
    }
  }
};

}  // namespace

void ExpandReduce(Expr* e) {
  Mutator mutator;
  mutator.Visit(e, e);
}

}  // namespace cinn::optim