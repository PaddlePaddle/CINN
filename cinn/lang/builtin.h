#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace lang {

Expr Sum(Expr body, Var reduce_axis);

}  // namespace lang
}  // namespace cinn
