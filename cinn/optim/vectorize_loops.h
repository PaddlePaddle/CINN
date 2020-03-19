#pragma once

#include "cinn/ir/ir_mutator.h"

namespace cinn {
namespace optim {

/**
 * Vectorize the forloops(For) if its for_type is marked as kVectorize.
 * @param expr
 * @param target
 */
void VectorizeLoops(Expr* expr, const Target& target);

namespace detail {

//! Vecorize the \p expr by making the \p var has \p lanes lanes.
void Vectorize(Var var, int lanes, Expr* expr);

//! Fit the vector's lanes in IR with the device SIMD size.
void FitVectorLanesWithDevice(int bits, Expr *expr);

}  // namespace detail

}  // namespace optim
}  // namespace cinn
