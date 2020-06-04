#pragma once
#include "cinn/cinn.h"
#include "cinn/common/ir_util.h"
#include "cinn/hlir/instruction/instruction.h"

namespace cinn {
namespace hlir {
namespace instruction {
namespace primitive {
using common::make_const;

ir::Tensor Abs(const ir::Tensor& a, const std::string& name);

ir::Tensor Ceil(const ir::Tensor& a, const std::string& name);

ir::Tensor Floor(const ir::Tensor& a, const std::string& name);

ir::Tensor Sign(const ir::Tensor& a, const std::string& name);

ir::Tensor Tanh(const ir::Tensor& a, const std::string& name);

ir::Tensor Exp(const ir::Tensor& a, const std::string& name);

ir::Tensor Sigmoid(const ir::Tensor& a, const std::string& name);

}  // namespace primitive
}  // namespace instruction
}  // namespace hlir
}  // namespace cinn
