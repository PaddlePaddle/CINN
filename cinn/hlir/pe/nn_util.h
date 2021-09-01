#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/context.h"
#include "cinn/common/ir_util.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "cinn/poly/stage.h"

namespace cinn {
namespace hlir {
namespace pe {

ir::Tensor const_matrix(const std::vector<std::vector<float>>& input, const std::string& name);

std::vector<std::vector<std::vector<float>>> get_winograd_val(const int& tile_size, const int& kernel_size);

std::vector<ir::Tensor> winograd_transform_matrices(const int& tile_size, const int& kernel_size);

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
