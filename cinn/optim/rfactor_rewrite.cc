#include "cinn/optim/rfactor_rewrite.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_operators.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::optim {

/*
// C[i,j] += A(i,k) * B(k,j)

for (i, 10) {
  for (j, 20) {
    C__reduce_init[i, j] = 0
  }
}
for (i, 10) {
  for (j, 20) {
    for (k0_outer, 2) {
      for (k0_inner, (1 + cinn_min(15, (29 + (-16 * k0_outer))))) {
        C[i, j] = (C[i, j] + (A[i, ((16 * k0_outer) + k0_inner)] * B[((16 * k0_outer) + k0_inner), j]))
      }
    }
  }
}

To something like

for (i, 16) {
  for (j, 32) {
    C__reduce_init[i, j] = 0
  }
}
for (i, 16) {
  for (j, 32) {
    for (k0_outer, 3) {
      for (k0_inner, 16) { // init to zero
        C.rt[k0_inner] = 0.f;
      }
      for (k0_inner, 16) {
        C.rt[k0_inner] = C.rt[k0_inner] + (A[i, ((16 * k0_outer) + k0_inner)] * B[((16 * k0_outer) + k0_inner), j]))
        //C[i, j] = (C[i, j] + (A[i, ((16 * k0_outer) + k0_inner)] * B[((16 * k0_outer) + k0_inner), j]))
      }
      for (k0_inner, 16) {
        C[i, j] += C.rt[k0_inner];
      }
    }
  }
}
*/

namespace {

struct Mutator : public ir::IRMutator<> {
  std::string tensor_name;
  Var rfactor_axis;
  poly::StageMap stages;

  Mutator(poly::StageMap stages) : stages(stages) {}

  using ir::IRMutator<>::Visit;

  void Visit(const ir::Store* op, Expr* expr) override {
    ir::Tensor tensor   = op->tensor.as_tensor_ref();
    auto rfactor_axises = stages[tensor]->rfactor_axis();
    if (rfactor_axises.empty()) return;
    CHECK_EQ(rfactor_axises.size(), 1UL) << "Cannot process more than 1 rfactor";

    this->rfactor_axis = Var(*rfactor_axises.begin());

    std::string_view axis_name = *rfactor_axises.begin();

    auto axis_level = std::find_if(forloops.begin(), forloops.end(), [&](const Expr& forloop) {
      return forloop.As<ir::For>() && forloop.As<ir::For>()->loop_var->name == axis_name;
    });

    // get the tensor reference indice
    std::vector<Expr> indice;
    std::vector<Expr> shape;
    for (auto& forloop_expr : forloops) {
      auto* forloop = forloop_expr.As<ir::For>();
      Var loop_var  = forloop->loop_var;
      if (loop_var == rfactor_axis) break;
      indice.push_back(loop_var);
      shape.push_back(forloop->extent);
    }

    RewriteRFactor(std::distance(forloops.begin(), axis_level), indice, shape);
  }

  void Visit(const ir::For* op, Expr* expr) override {
    auto* node = expr->As<ir::For>();
    forloops.push_back(*expr);

    ir::IRMutator<>::Visit(&node->body, &node->body);

    forloops.pop_back();
  }

  void Visit(const ir::PolyFor* op, Expr* expr) override {
    auto* node = expr->As<ir::PolyFor>();
    forloops.push_back(*expr);

    ir::IRMutator<>::Visit(&node->body, &node->body);

    forloops.pop_back();
  }

  void RewriteRFactor(int offset, std::vector<Expr> indices, std::vector<Expr> shape) {
    auto* forloop = forloops[offset].As<ir::For>();
    CHECK(forloop);

    ir::Store* old_store = forloop->body.As<ir::Store>();
    if (auto* block = forloop->body.As<ir::Block>()) {
      CHECK_EQ(block->stmts.size(), 1UL);
      old_store = block->stmts[0].As<ir::Store>();
    }
    CHECK(old_store) << "expr: " << forloop->body;
    LOG(INFO) << "old_store: " << forloop->body;

    Expr tensor = ir::_Tensor_::Make(
        old_store->tensor.as_tensor()->name + "_rfactor", old_store->value.type(), shape, shape, nullptr);

    auto [node_type, right_operand] = common::BinaryArithEqualGetBody(old_store);
    CHECK(right_operand.defined());

    Expr right_operand1 = tensor.as_tensor_ref()(indices);
    switch (node_type) {
      case ir::IrNodeTy::Add:
        right_operand1 = right_operand1 + right_operand;
        break;
      case ir::IrNodeTy::Sub:
        right_operand1 = right_operand1 - right_operand;
        break;
      case ir::IrNodeTy::Mul:
        right_operand1 = right_operand1 * right_operand;
        break;
      case ir::IrNodeTy::Div:
        right_operand1 = right_operand1 / right_operand;
        break;
      case ir::IrNodeTy::Min:
        right_operand1 = ir::Min::Make(right_operand1, right_operand);
        break;
      case ir::IrNodeTy::Max:
        right_operand1 = ir::Max::Make(right_operand1, right_operand);
        break;
      default:
        CINN_NOT_IMPLEMENTED
    }

    // create a new store
    auto new_store = ir::Store::Make(tensor, right_operand1, indices);

    forloop->body = ir::Block::Make({new_store});

    // to append a forloop to assign the origial tensor value
  }

  std::vector<Expr> forloops;
};

}  // namespace

void RFactorRewrite(Expr* e, poly::StageMap stages) {
  Mutator mutator(stages);
  mutator.Visit(e, e);
}

}  // namespace cinn::optim
