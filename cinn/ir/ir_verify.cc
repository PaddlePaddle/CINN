#include "cinn/ir/ir_verify.h"
#include "cinn/ir/ir_mutator.h"
#include "cinn/ir/ir_printer.h"

namespace cinn::ir {

struct IrVerifyVisitor : public ir::IRMutator<> {
  using ir::IRMutator<>::Visit;

#define __(op__)                                    \
  void Visit(const op__ *op, Expr *expr) override { \
    op->Verify();                                   \
    IRMutator::Visit(op, expr);                     \
  }
  NODETY_FORALL(__)
#undef __
};

void IrVerify(Expr e) {
  IrVerifyVisitor visitor;
  visitor.Visit(&e, &e);
}

}  // namespace cinn::ir
