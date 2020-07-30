#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

//! Transform the PolyFor node to For node. This will also separate the PolyFor with Min or Max conditions into two For
//! nodes if \p auto_separate is true.
void TransformPolyForToFor(Expr* expr, bool auto_separate = true);

namespace detail {

void PolyForWithSimpleConditionToFor(Expr* expr);

}  // namespace detail

}  // namespace optim
}  // namespace cinn
