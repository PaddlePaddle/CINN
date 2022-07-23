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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"

namespace cinn {
namespace hlir {
namespace op {

std::vector<ir::Tensor> Pool2dGrad(const ir::Tensor &in_tensor,
                                   const ir::Tensor &output_tensor,
                                   const ir::Tensor &output_grad,
                                   const std::vector<int> &kernel_size,
                                   const std::vector<int> &strides,
                                   const std::vector<int> &paddings,
                                   const std::string &pool_type,
                                   bool ceil_mode,
                                   bool exclusive,
                                   bool adaptive,
                                   const std::string &data_format,
                                   const std::string &output_name) {
  CHECK(in_tensor->shape.size() == 4U || in_tensor->shape.size() == 5U)
      << "Pool2dGrad requires in_tensor's rank to be 4 or 5";
  CHECK(output_tensor->shape.size() == 4U || output_tensor->shape.size() == 5U)
      << "Pool2dGrad requires output_tensor's rank to be 4 or 5";
  CHECK(output_grad->shape.size() == 4U || output_grad->shape.size() == 5U)
      << "Pool2dGrad requires output_grad's rank to be 4 or 5";

  CHECK_EQ(kernel_size.size(), 2U) << "Pool2dGrad kernel_size should be 2";
  CHECK_EQ(strides.size(), 2U) << "Pool2dGrad stride_size should be 2";
  CHECK_EQ(paddings.size(), 4U) << "Pool2dGrad padding_size should be 4, which is double as kernel size";

  CHECK(!ceil_mode) << "This is just an example op so we don't support ceil_mode=true";
  CHECK(!exclusive) << "This is just an example op so we don't support exclusive=true";
  CHECK(!adaptive) << "This is just an example op so we don't support adaptive=true";

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

  std::vector<ir::Expr> in_grad_shape  = in_tensor->shape;
  std::vector<ir::Expr> out_grad_shape = output_grad->shape;
  int ksize                            = kernel_size.size();

  std::vector<ir::Var> pool_vars;
  std::vector<ir::Expr> out_grad_pad_front(out_grad_shape.size(), Expr(0));
  std::vector<ir::Expr> out_grad_pad_back(out_grad_shape.size(), Expr(0));
  for (int i = 0; i < ksize; ++i) {
    CHECK(kernel_size[i] >= strides[i]) << "This is just an example op so we require kernel_size > stride";
    CHECK_EQ(kernel_size[i] % strides[i], 0) << "This is just an example op so we require kernel_size % stride == 0";

    int axis = hw_axis[i];
    pool_vars.push_back(ir::Var(kernel_size[i] / strides[i], common::UniqName("pool_grad_idx")));

    out_grad_pad_front[axis] = ir::Expr(kernel_size[i] - 1);
    out_grad_pad_back[axis]  = ir::Expr(kernel_size[i] - 1);
  }
  ir::Tensor padding_out_grad =
      pe::Pad(output_grad, out_grad_pad_front, out_grad_pad_back, 0, common::UniqName("padding_out_grad"));
  if (pool_type == "max") {
    CHECK(false) << "Unimplemented pool_type: " << pool_type;
  } else if (pool_type == "avg") {
    float factor = 1.0f / (kernel_size[0] * kernel_size[1]);
    ir::Expr factor_expr(factor);
    ir::Tensor res = lang::Compute(
        in_grad_shape,
        [=](const std::vector<ir::Expr> &output) {
          // Find that x * stride <= y + padding < x * stride + kernel
          // the miminal x would be in start (inclusive)
          // the maximal x would be in end (inclusive)
          // Then it construct the mapping for the indices from output_tensor to in_tensor

          std::vector<ir::Expr> start(ksize);
          std::vector<ir::Expr> end(ksize);
          // std::vector<ir::Var> vars(ksize);

          std::vector<ir::Expr> indices(output);
          for (int i = 0; i < ksize; ++i) {
            int axis = hw_axis[i];
            start[i] =
                common::AutoSimplify((output[axis] + paddings[i] - kernel_size[i]) / strides[i] + kernel_size[i]);
            // start[i]      = common::AutoSimplify((output[axis] + paddings[i] - kernel_size[i]) / strides[i] + 1);
            // start[i]      = ir::Max::Make(start[i], make_const(Int(32), 0));
            // end[i]        = common::AutoSimplify((output[axis] + paddings[i]) / strides[i]);
            // end[i]        = ir::Min::Make(end[i], out_grad_shape[axis]);

            // vars[i]       = ir::Var(common::AutoSimplify(end[i] - start[i] + 1), common::UniqName("pool_grad_idx"));
            indices[axis] = start[i] + pool_vars[i];
          }

          // return factor_expr;
          // return lang::ReduceSum(output_grad(indices), pool_vars);
          return lang::ReduceSum(ir::Mul::Make(padding_out_grad(indices), factor_expr), pool_vars);
        },
        common::UniqName(output_name));
    return {res, padding_out_grad};
  } else {
    CHECK(false) << "Unrecognized pool_type: " << pool_type;
  }
  return {};
}

std::vector<std::vector<int>> InferShapeForPool2dGrad(const std::vector<std::vector<int>> &inputs_shape,
                                                      const framework::AttrMapType &attrs) {
  CHECK_EQ(inputs_shape.size(), 3UL) << "The number of pool2d_grad's input should be 3";
  CHECK(inputs_shape[0].size() == 4UL || inputs_shape[0].size() == 5UL)
      << "The input's shape size of pool2d_grad should be 4 or 5! Please check again.";

  std::vector<int> kernel_size;
  std::vector<int> stride_size;
  std::vector<int> padding_size;
  std::string pool_type   = "avg";
  bool ceil_mode          = false;
  bool exclusive          = false;
  bool adaptive           = false;
  std::string data_format = "NCHW";

  for (auto &iter : attrs) {
    if (iter.first == "kernel_size") {
      kernel_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "stride_size") {
      stride_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "padding_size") {
      padding_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "ceil_mode") {
      ceil_mode = absl::get<bool>(iter.second);
    } else if (iter.first == "exclusive") {
      exclusive = absl::get<bool>(iter.second);
    } else if (iter.first == "adaptive") {
      adaptive = absl::get<bool>(iter.second);
    } else if (iter.first == "data_format") {
      data_format = absl::get<std::string>(iter.second);
    }
  }

  CHECK_EQ(kernel_size.size(), 2U) << "kernel size for pool2d_grad should be 2.\n";
  CHECK_EQ(stride_size.size(), 2U) << "stride_size size for pool2d_grad should be 2.\n";
  CHECK_EQ(padding_size.size(), 4U) << "padding_size size for pool2d_grad should be 4.\n";

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
    data_format = "NCHW";
  } else {
    LOG(ERROR) << "unsupported data_format: " << data_format << std::endl;
  }

  CHECK_EQ(inputs_shape[0].size(), inputs_shape[1].size())
      << "out_tensor and in_grad of pool2d_grad rank should be same";
  CHECK_EQ(inputs_shape[1].size(), inputs_shape[2].size())
      << "out_tensor and out_grad of pool2d_grad shape should be same";
  for (int i = 0; i < inputs_shape[0].size(); ++i) {
    if (i == height_axis) {
      CHECK_EQ(inputs_shape[1][i],
               (inputs_shape[0][i] - kernel_size[0] + padding_size[0] + padding_size[2]) / stride_size[0] + 1)
          << "out_tensor of pool2d_grad has wrong size";
    } else if (i == width_axis) {
      CHECK_EQ(inputs_shape[1][i],
               (inputs_shape[0][i] - kernel_size[1] + padding_size[1] + padding_size[3]) / stride_size[1] + 1)
          << "out_tensor of pool2d_grad has wrong size";
    } else {
      CHECK_EQ(inputs_shape[0][i], inputs_shape[1][i])
          << "inputs of pool2d_grad should have same batch size and channels";
    }
    CHECK_EQ(inputs_shape[1][i], inputs_shape[2][i])
        << "inputs of pool2d_grad should have same batch size and channels";
  }

  // if (data_format == "AnyLayout") {
  //   data_format = "NCHW";
  // } else if (data_format != "NCHW" && data_format != "NHWC") {
  //   LOG(ERROR) << "unsupported data_format: " << data_format << std::endl;
  // }

  std::vector<std::vector<int>> res = {inputs_shape[0]};
  return res;
}

std::vector<Type> InferDtypeForPool2dGrad(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::shared_ptr<framework::OpStrategy> StrategyForPool2dGrad(const framework::NodeAttr &attrs,
                                                             const std::vector<ir::Tensor> &inputs,
                                                             const std::vector<Type> &out_type,
                                                             const std::vector<std::vector<int>> &output_shapes,
                                                             const Target &target) {
  auto attr_store = attrs.attr_store;
  std::vector<int> kernel_size;   // [kernel_h, kernel_w]
  std::vector<int> stride_size;   // [stride_h, stride_w]
  std::vector<int> padding_size;  // [padding_top, padding_left, padding_bottom, padding_right]
  std::string pool_type   = "avg";
  bool ceil_mode          = false;
  bool exclusive          = false;
  bool adaptive           = false;
  std::string data_format = "NCHW";

  for (auto &iter : attrs.attr_store) {
    if (iter.first == "kernel_size") {
      kernel_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "stride_size") {
      stride_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "padding_size") {
      padding_size = absl::get<std::vector<int>>(iter.second);
    } else if (iter.first == "pool_type") {
      pool_type = absl::get<std::string>(iter.second);
    } else if (iter.first == "ceil_mode") {
      ceil_mode = absl::get<bool>(iter.second);
    } else if (iter.first == "exclusive") {
      exclusive = absl::get<bool>(iter.second);
    } else if (iter.first == "adaptive") {
      adaptive = absl::get<bool>(iter.second);
    } else if (iter.first == "data_format") {
      data_format = absl::get<std::string>(iter.second);
    }
  }

  CHECK_EQ(kernel_size.size(), 2U) << "kernel size for pool2d_grad should be 2.\n";
  CHECK_EQ(stride_size.size(), 2U) << "stride_size size for pool2d_grad should be 2.\n";
  CHECK_EQ(padding_size.size(), 4U) << "padding_size size for pool2d_grad should be 4.\n";

  framework::CINNCompute pool2d_grad_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool2d compute is empty! Please check.\n";
    common::CINNValuePack value_args = args[0];
    Expr in_expr                     = value_args[0];
    Expr out_expr                    = value_args[1];
    Expr out_grad_expr               = value_args[2];
    CHECK(in_expr.as_tensor());
    CHECK(out_expr.as_tensor());
    CHECK(out_grad_expr.as_tensor());

    ir::Tensor in_tensor  = in_expr.as_tensor_ref();
    ir::Tensor out_tensor = out_expr.as_tensor_ref();
    ir::Tensor out_grad   = out_grad_expr.as_tensor_ref();

    std::vector<ir::Tensor> out = Pool2dGrad(in_tensor,
                                             out_tensor,
                                             out_grad,
                                             kernel_size,
                                             stride_size,
                                             padding_size,
                                             pool_type,
                                             ceil_mode,
                                             exclusive,
                                             adaptive,
                                             data_format,
                                             common::UniqName("T_Pool2d_Grad_out"));

    auto stages = CreateStages({in_tensor, out_tensor, out_grad});
    CHECK(out.size() == 2U) << "The size of Pool2dGrad's output should be 1";
    std::vector<common::CINNValue> res;
    for (auto &t : out) {
      stages->InsertLazily(t);
      res.push_back(common::CINNValue(t));
    }
    CHECK(!out_type.empty()) << "Output type of Pool2dGrad is empty! Please check.\n";
    res.push_back(common::CINNValue(stages));
    *ret = common::CINNValuePack{res};
  });

  framework::CINNSchedule pool2d_grad_schedule([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of pool2d_grad schedule is empty! Please check.\n";
    common::CINNValuePack arg_pack = args[0];
    CHECK_EQ(arg_pack.size(), 3UL);
    Expr Out = arg_pack[0];
    CHECK(Out.as_tensor());
    poly::StageMap stages = arg_pack[arg_pack.size() - 1];
    *ret                  = common::CINNValuePack{{common::CINNValue(Out), common::CINNValue(stages)}};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(pool2d_grad_compute, pool2d_grad_schedule, "strategy.pool2d_grad.x86", 1);
  return strategy;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(pool_grad_ops) {
  CINN_REGISTER_OP(pool2d_grad)
      .describe("The gradient of pool2d.")
      .set_num_inputs(3)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForPool2dGrad)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForPool2dGrad))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForPool2dGrad))
      .set_support_level(4);

  return true;
}
