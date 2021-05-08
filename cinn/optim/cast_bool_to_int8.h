#pragma once
#include "cinn/ir/ir.h"

namespace cinn::optim {

/**
 * Cast the expr from bool to Int8 type for llvm codegen, currently used in cpu.
 *
 * e.g.
 *
 * The expression:
 * a = b
 *
 * to
 *
 * a = int8(b)
 */
void CastBoolToInt8(Expr* e, Target target);

}  // namespace cinn::optim
