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

void MoveData(float* data, int i, int M, int N) {
  float temp = data[i];
  int cur    = i;  // current data index
  int pre    = (cur % M) * N + cur / M;
  while (pre != i) {
    data[cur] = data[pre];
    cur       = pre;
    pre       = (cur % M) * N + cur / M;
  }
  data[cur] = temp;
}

void TransposeData(float* data, int M, int N) {
  for (int i = 0; i < M * N; i++) {
    int next = (i % N) * M + i / N;
    while (next > i)  // next < 1 implies duplicate
      next = (next % N) * M + next / N;
    if (next == i)  // process current ring
      MoveData(data, i, M, N);
  }
}

void TransposeVar(const std::string& origin_name, const OpMapperContext& ctx) {
  const auto& name = cinn::utils::TransValidVarName(origin_name);
  CheckVarNameValid(name);
  auto* var = ctx.scope_->FindVar(name);
  if (var) {
    auto& tensor = absl::get<hlir::framework::Tensor>(*var);
    if (ctx.target_.arch == Target::Arch::X86) {
      float* data = tensor->mutable_data<float>(ctx.target_);
      CHECK_EQ(tensor->shape().size(), 2UL) << "The y data's shape size of op [mul] is not equal to 2! Please check.";
      TransposeData(data, tensor->shape().data()[0], tensor->shape().data()[1]);
    } else if (ctx.target_.arch == Target::Arch::NVGPU) {
#ifdef CINN_WITH_CUDA
      // To use cublas mul api, there is no need to transpose data.
#ifndef CINN_WITH_CUDNN
      std::vector<float> data(tensor->shape().numel());
      CUDA_CALL(cudaMemcpy(data.data(),
                           reinterpret_cast<void*>(tensor->mutable_data<float>(ctx.target_)),
                           tensor->shape().numel() * sizeof(float),
                           cudaMemcpyDeviceToHost));
      CHECK_EQ(tensor->shape().size(), 2UL) << "The y data's shape size of op [mul] is not equal to 2! Please check.";
      TransposeData(data.data(), tensor->shape().data()[0], tensor->shape().data()[1]);
      CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(tensor->mutable_data<float>(ctx.target_)),
                           data.data(),
                           tensor->shape().numel() * sizeof(float),
                           cudaMemcpyHostToDevice));
#endif
#else
      LOG(FATAL) << "To use CUDA backends, you need to set WITH_CUDA ON!";
#endif
    } else {
      CINN_NOT_IMPLEMENTED
    }

    Variable var;
    var.set_id(name);
    std::vector<int> reverse_shape = tensor->shape().data();
    std::reverse(reverse_shape.begin(), reverse_shape.end());
    tensor->shape().SetData(reverse_shape);
    var->shape = tensor->shape().data();
    var->type  = tensor->type();
    ctx.AddVar(name, var, true);
  } else {
    LOG(FATAL) << "No var called [" << name << "] exists";
  }
}

void MulOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  auto x      = ctx.GetVar(x_name);
  TransposeVar(y_name, ctx);
  auto y = ctx.GetVar(y_name);

  auto x_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "x_num_col_dims", 1);
  auto y_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "y_num_col_dims", 1);

  VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
  VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "y shape: " << cinn::utils::Join(y->shape, ",");
  auto out = ctx.builder_->mul(x, y, x_num_col_dims, y_num_col_dims);
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

void MulBiasOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  CHECK_EQ(op_desc.Input("Y").size(), 1UL);
  auto y_name = op_desc.Input("Y").front();
  CHECK_EQ(op_desc.Input("Z").size(), 1UL);
  auto z_name = op_desc.Input("Z").front();

  auto x = ctx.GetVar(x_name);
  TransposeVar(y_name, ctx);
  auto y = ctx.GetVar(y_name);
  auto z = ctx.GetVar(z_name);

  auto x_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "x_num_col_dims", 1);
  auto y_num_col_dims = utils::GetAttrOrDefault<int>(op_desc, "y_num_col_dims", 1);

  VLOG(4) << "Mul x_num_col_dims: " << x_num_col_dims;
  VLOG(4) << "Mul y_num_col_dims: " << y_num_col_dims;
  VLOG(4) << "x shape: " << cinn::utils::Join(x->shape, ",");
  VLOG(4) << "y shape: " << cinn::utils::Join(y->shape, ",");
  VLOG(4) << "z shape: " << cinn::utils::Join(z->shape, ",");
  auto out = ctx.builder_->mulbias(x, y, z, x_num_col_dims, y_num_col_dims);

  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();
  ctx.AddVar(out_name, out);
  ctx.AddVarModelToProgramMap(out_name, out->id);
}

}  // namespace op_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(mul) {
  CINN_REGISTER_OP_MAPPER(mul, cinn::frontend::op_mappers::MulOpMapper)
  CINN_REGISTER_OP_MAPPER(mulbias, cinn::frontend::op_mappers::MulBiasOpMapper)
  return true;
}
