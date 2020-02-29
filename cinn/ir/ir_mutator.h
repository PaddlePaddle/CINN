/**
 * This file implements the IRMutator as the base interface to mutate the IR.
 */
#pragma once

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_visitor.h"

namespace cinn {
namespace ir {

//! T might be Expr* or const Expr*
template <typename T = Expr*>
class IRMutator : public IRVisitorBase<void, T> {
 public:
  void Visit(const Expr* expr, T op) override;

#define __(op__) void Visit(const op__* expr, T op) override;
  NODETY_FORALL(__)
#undef __
};

}  // namespace ir
}  // namespace cinn
