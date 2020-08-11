#pragma once

#include "cinn/ir/ir.h"

namespace cinn {
namespace hlir {
namespace pe {

template <typename T>
ir::Tensor Relu(const ir::Tensor& A, T threshold = static_cast<T>(0), const std::string& output_name = "T_Relu_out");

ir::Tensor LeakyRelu(const ir::Tensor& A, double alpha = 0.1, const std::string& output_name = "T_LeakyRelu_out");

}  // namespace pe
}  // namespace hlir
}  // namespace cinn
