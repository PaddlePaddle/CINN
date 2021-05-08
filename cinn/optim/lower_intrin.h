#pragma once

#include <set>
#include <string>

#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

static const std::set<std::string> kIntrinsicCalls{
    {"exp",         "exp2",       "sqrt",        "log",         "log2",        "log10", "floor",
     "ceil",        "round",      "trunc",       "cos",         "cosh",        "tan",   "tanh",
     "sin",         "sinh",       "fabs",        "isnan",       "isfinite",    "isinf", "left_shift",
     "right_shift", "bitwise_or", "bitwise_and", "bitwise_xor", "bitwise_not", "fma"}};

/**
 * Map the Call nodes to llvm intrinsic.
 *
 * This will rename the external call with the function in different backends.
 *
 * Notes: only support cpu currently.
 */
void LowerIntrin(Expr *e, Target target);

}  // namespace optim
}  // namespace cinn
