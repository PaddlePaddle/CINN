#pragma once
#include "cinn/ir/ir.h"

namespace cinn {
namespace lang {

//! Reduce sum with single reduce varialbe.
Expr Sum(Expr body);

}  // namespace lang
}  // namespace cinn
