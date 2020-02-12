#pragma once
#include <functional>
#include <utility>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/lang/placeholder.h"
#include "cinn/poly/schedule.h"

namespace cinn {
namespace lang {

using ir::Var;
using compute_handle_1_t = std::function<ir::Expr(Var i)>;
using compute_handle_2_t = std::function<ir::Expr(Var i0, Var i1)>;
using compute_handle_3_t = std::function<ir::Expr(Var i0, Var i1, Var i2)>;
using compute_handle_4_t = std::function<ir::Expr(Var i0, Var i1, Var i2, Var i3)>;

template <typename Fn>
ir::Tensor Compute(const std::vector<int>& dims, Fn handle);

}  // namespace lang
}  // namespace cinn
