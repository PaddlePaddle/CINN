// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cinn/hlir/op/contrib/pool_grad.h"

#include <string>
#include <vector>

#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"

namespace cinn {
namespace hlir {
namespace op {

std::vector<ir::Tensor> Pool2dGrad(const ir::Tensor& in_tensor,
                                   const ir::Tensor& output_tensor,
                                   const ir::Tensor& output_grad,
                                   const std::vector<int>& kernel_size,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   const std::string& pool_type,
                                   bool ceil_mode,
                                   bool exclusive,
                                   bool adaptive,
                                   const std::string& data_format,
                                   const std::string& output_name) {
  CHECK(in_tensor->shape.size() == 4U || in_tensor->shape.size() == 5U)
      << "Pool2dGrad requires in_tensor's rank to be 4 or 5";
  CHECK(output_tensor->shape.size() == 4U || output_tensor->shape.size() == 5U)
      << "Pool2dGrad requires output_tensor's rank to be 4 or 5";
  CHECK(output_grad->shape.size() == 4U || output_grad->shape.size() == 5U)
      << "Pool2dGrad requires output_grad's rank to be 4 or 5";

  CHECK_EQ(kernel_size.size(), 2) << "Pool2dGrad kernel_size should be 2";
  CHECK_EQ(strides.size(), 2) << "Pool2dGrad stride_size should be 2";
  CHECK_EQ(paddings.size(), 4) << "Pool2dGrad padding_size should be 4, which is double as kernel size";

  int height_axis = -1;
  int width_axis  = -1;
  if (data_format == "NCHW") {
    height_axis = 2;
    width_axis  = 3;
  } else if (data_format == "NHWC") {
    height_axis = 1;
    width_axis  = 2;
  } else if (data_format == "AnyLayout") {
    height_axis = 2;
    width_axis  = 3;
  } else {
    LOG(FATAL) << "Unsupported data format: " << data_format << std::endl;
  }
  std::vector<int> hw_axis = {height_axis, width_axis};

  std::vector<Expr> in_grad_shape = in_tensor->shape;
  int ksize                       = kernel_size.size();

  if (pool_type == "max") {
    LOG(ERROR) << "Unimplemented pool_type: " << pool_type;
  } else if (pool_type == "avg") {
    float factor = 1.0f / (kernel_size[0] * kernel_size[1]);
    ir::Expr factor_expr(factor);
    res = Compute(in_grad_shape, [=](const std::vector<ir::Expr>& output) {
      // Find that x * stride <= y + padding < x * stride + kernel
      // the miminal x would be in start
      // the maximal x would be in end
      // Then it construct the mapping for the indices from output_tensor to in_tensor
      std::vector<ir::Expr> start(ksize);
      std::vector<ir::Expr> end(ksize);
      std::vector<ir::Var> vars(ksize);

      std::vector<ir::Expr> indices(output);
      for (int i = 0; i < ksize; ++i) {
        int axis = hw_axis[i];
        start[i] = common::AutoSimplify((output[axis] + paddings[i] - kernel_size[i]) / strides[i] + 1);
        end[i]   = common::AutoSimplify((output[axis] + paddings[i]) / strides[i]);
        vars.emplace_back(ir::Var(start[i], end[i], UniqName("kernel_idx")));
        indices[axis] = vars;
      }

      return lang::ReduceSum(ir::Mul::Make(output_grad(indices), factor_expr), vars);
    } UniqName(output_name));
  } else {
    LOG(ERROR) << "Unrecognized pool_type: " << pool_type;
  }
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn
