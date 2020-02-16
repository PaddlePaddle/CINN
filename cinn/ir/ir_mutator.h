/**
 * This file implements the IRMutator as the base interface to mutate the IR.
 */
#pragma once

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

class IRMutator : public IRVisitorBase<void, Expr*> {
 public:
#define __(op__) void Visit(const op__* expr, Expr* op) override;
  NODETY_FORALL(__)
#undef __
};

}  // namespace ir
}  // namespace cinn
