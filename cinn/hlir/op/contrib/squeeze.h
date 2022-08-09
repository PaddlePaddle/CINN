#pragma once

#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"

namespace cinn {
namespace hlir {
namespace op {

ir::Tensor Squeeze(const ir::Tensor& A, const std::vector<int>& axis, poly::StageMap stages, const std::string& name);

}  // namespace op
}  // namespace hlir
}  // namespace cinn
