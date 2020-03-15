#pragma once

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

void VectorizeLoops(Expr* expr, const Target& target);

namespace detail {

//! Vecorize the \p expr by making the \p var has \p lanes lanes.
void Vectorize(Var var, int lanes, Expr* expr);

}  // namespace detail

}  // namespace optim
}  // namespace cinn
