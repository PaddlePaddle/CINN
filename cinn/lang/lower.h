/**
 * Lower lowerise the statements to LoweredFuncs.
 */

#pragma once
#include <string>
#include <vector>

#include "cinn/ir/function.h"
#include "cinn/ir/ir.h"
#include "cinn/lang/module.h"
#include "cinn/lang/tensor.h"
#include "cinn/poly/schedule.h"

namespace cinn {
namespace lang {
using ir::Tensor;

std::vector<LoweredFunc> Lower(const std::string& name, const std::vector<Tensor>& args);

}  // namespace lang
}  // namespace cinn
