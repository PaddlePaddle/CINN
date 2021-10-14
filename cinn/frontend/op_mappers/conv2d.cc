// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "cinn/backends/cuda_util.h"
#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"

namespace cinn {
namespace frontend {
namespace op_mappers {

inline void ReverseHWData(float* data, std::vector<int> shape) {
  CHECK_EQ(shape.size(), 4UL);
  for (int i = 0; i < shape[0] * shape[1]; i++) {
    int num = shape[2] * shape[3];
    std::reverse(data + (i * num), data + (i * num + num));
  }
}

void ReverseHWVar(const std::string& origin_name, const OpMapperContext& ctx) {
  const auto& name = cinn::utils::TransValidVarName(origin_name);
  CheckVarNameValid(name);
  auto* var = ctx.Scope().FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (ctx.Target().arch == Target::Arch::X86) {
      float* data = tensor->mutable_data<float>(ctx.Target());
      CHECK_EQ(tensor->shape().size(), 4UL)
          << "The y data's shape size of op [conv2d] is not equal to 4! Please check.";
      ReverseHWData(data, tensor->shape().data());
    } else if (ctx.Target().arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
      std::vector<float> data(tensor->shape().numel());
      CUDA_CALL(cudaMemcpy(data.data(),
                           reinterpret_cast<void*>(tensor->mutable_data<float>(ctx.Target())),
                           tensor->shape().numel() * sizeof(float),
                           cudaMemcpyDeviceToHost));
      CHECK_EQ(tensor->shape().size(), 4UL)
          << "The y data's shape size of op [conv2d] is not equal to 4! Please check.";
      ReverseHWData(data.data(), tensor->shape().data());
      CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(tensor->mutable_data<float>(ctx.Target())),
                           data.data(),
                           tensor->shape().numel() * sizeof(float),
                           cudaMemcpyHostToDevice));
#else
      LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
    } else {
      CINN_NOT_IMPLEMENTED
    }
  } else {
    LOG(FATAL) << "No var called [" << name << "] exists";
  }
}

void Conv2dOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
  auto y_name = op_desc.Input("Filter").front();
#ifdef CINN_WITH_CUDNN
  if (ctx.Target().arch == Target::Arch::NVGPU) {
    ReverseHWVar(y_name, ctx);
  }
#endif
  CHECK_EQ(op_desc.Output("Output").size(), 1UL);
  auto out_name = op_desc.Output("Output").front();

  auto strides   = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings  = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups    = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "AnyLayout");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");
  auto x                 = ctx.GetVar(x_name);
  auto y                 = ctx.GetVar(y_name);
  auto out = ctx.Builder()->conv2d(x, y, strides, paddings, dilations, groups, data_format, padding_algorithm);

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

void DepthwiseConv2dOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("Input").size(), 1UL);
  auto x_name = op_desc.Input("Input").front();
  CHECK_EQ(op_desc.Input("Filter").size(), 1UL);
  auto y_name = op_desc.Input("Filter").front();
#ifdef CINN_WITH_CUDNN
  if (ctx.Target().arch == Target::Arch::NVGPU) {
    ReverseHWVar(y_name, ctx);
  }
#endif
  CHECK_EQ(op_desc.Output("Output").size(), 1UL);
  auto out_name = op_desc.Output("Output").front();

  auto strides   = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "strides", {1, 1});
  auto paddings  = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "paddings", {0, 0});
  auto dilations = utils::GetAttrOrDefault<std::vector<int>>(op_desc, "dilations", {1, 1});
  auto groups    = utils::GetAttrOrDefault<int>(op_desc, "groups", 1);

  auto data_format = utils::GetAttrOrDefault<std::string>(op_desc, "data_format", "NCHW");
  if (data_format == "AnyLayout") {
    data_format = "NCHW";
  }

  auto padding_algorithm = utils::GetAttrOrDefault<std::string>(op_desc, "padding_algorithm", "EXPLICIT");
  auto x                 = ctx.GetVar(x_name);
  auto y                 = ctx.GetVar(y_name);
  Variable out;
  if (ctx.Target().arch == Target::Arch::X86) {
    out = ctx.Builder()->conv2d(x, y, strides, paddings, dilations, groups, data_format, padding_algorithm);
  } else {
    out = ctx.Builder()->depthwise_conv2d(x, y, strides, paddings, dilations, groups, data_format, padding_algorithm);
  }

  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgram(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(conv2d) {
  CINN_REGISTER_OP_MAPPER(conv2d, cinn::frontend::op_mappers::Conv2dOpMapper)
  CINN_REGISTER_OP_MAPPER(depthwise_conv2d, cinn::frontend::op_mappers::DepthwiseConv2dOpMapper)
  return true;
}
