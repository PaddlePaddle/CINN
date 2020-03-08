#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace lang {

Expr ReduceSum(Expr body, Var axis);

}  // namespace lang
}  // namespace cinn
