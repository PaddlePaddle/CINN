#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace lang {

//! Reduce sum with reduce axis.
Expr Sum(Expr body);

//! Reduce mul with reduce axis.
Expr Mul(Expr body);

}  // namespace lang
}  // namespace cinn
