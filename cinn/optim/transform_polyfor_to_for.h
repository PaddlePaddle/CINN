#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace optim {

//! Transform the PolyFor node to For node. This will also separate the PolyFor with Min or Max conditions into two For
//! nodes.
void TransformPolyforToFor(Expr* expr);

namespace detail {

//! Automatically separate the PolyFor with some specific kind of conditions(such as i < min(a, b)) into two For nodes.
//! e.g. PolyFor(i, 0, 100) { PolyFor(j, 0, min(i, 40))}
//!      to
//!      {
//!        PolyFor(i, 0, 40) { PolyFor(j, 0, i) }
//!        PolyFor(i, 40, 100) { PolyFor(j, 0, 40) }
//!      }
void PolyForAutoSeparate(Expr* expr);

void PolyForWithSimpleConditionToFor(Expr* expr);

}  // namespace detail

}  // namespace optim
}  // namespace cinn
