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

#include "fusion_checker.h"

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace hlir {
namespace framework {

FusionChecker::FusionChecker(const Instruction& target_instr,
                             const Graph& graph,
                             const GroupPtr& group,
                             const common::Target& target)
    : target_instr_(target_instr), graph_(graph), group_(group), target_(target) {
  BuildSubInstrution();
}

FusionChecker::BuildSubInstrutions() {
  auto nodes = group_->CollectNodes();
  for (auto node : nodes) {
  }
}

bool FusionChecker::operator()() { return RunChecker(); }

bool FusionChecker::RunChecker() {
  InitInputTensor();
  auto src_tensors = RunSubInstructions();
  auto dst_tensors = RunTargetInstruction();
  for (auto& src : src_tensors) {
    CHECK(dst_tensors.count(src.first));
    CHECK(CheckTensorValue(src.second, dst_tensors[src.first]));
  }
  return true;
}

template <class T>
void FusionChecker::GetRandom(T* data, size_t size) {
  std::default_random_engine engine(time(NULL));
  std::uniform_real_distribution<float> generator(-1, 1);
  for (size_t idx = 0; idx < size; ++idx) {
    *(data++) = generator(engine);
  }
}

template void FusionChecker::GetRandom<float>(float* data, size_t size);

void FusionChecker::InitInputTensor() {
  auto& input_names = group_->input_names;
  auto& dtype_dict  = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict  = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  for (auto& name : input_names) {
    Tensor tensor;
    tensor->Resize(shape_dict.at(name));
    tensor->set_type(dtype_dict.at(name));
    if (tensor->type() == Float(32)) {
      GetRandom(tensor->mutable_data<float>(common::DefaultHostTarget()));
    } else {
      LOG(FATAL) << "Not Support Now!";
    }
    input_tensors[name] = tensor;
  }
}

Tensor FusionChecker::TensorHostToDevice(Tensor& src, void* stream) {
  if (src->get_buffer()->GetTarget() == common::DefaultNVGPUTarget()) {
    return src;
  }

  Tensor dst;
  dst->Resize(dst->shape());
  auto src_data = src->data<void>();
  auto dst_data = dst->mutable_data(common::DefaultNVGPUTarget(), src->type());

#ifdef CINN_WITH_CUDA
  int width = src->type().bits() / 8;
  cudaMemcpyAsync(
      dst_data, src_data, src_data->shape().size() * width, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
  auto st = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  if (st) {
    LOG(FATAL) << "Error Ocurs -> " << cudaGetErrorString(st);
  }
#endif

  return dst;
}

Tensor FusionChecker::TensorDeviceToHost(Tensor& src, void* stream) {
  if (src->get_buffer()->GetTarget() == common::DefaultHostTarget()) {
    return src;
  }
  Tensor dst;
  dst->Resize(dst->shape());
  auto src_data = src->data<void>();
  auto dst_data = dst->mutable_data(common::DefaultHostTarget(), src->type());

#ifdef CINN_WITH_CUDA
  int width = src->type().bits() / 8;
  cudaMemcpyAsync(
      dst_data, src_data, src_data->shape().size() * width, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
  auto st = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  if (st) {
    LOG(FATAL) << "Error Ocurs -> " << cudaGetErrorString(st);
  }
#endif

  return dst;
}

std::unordered_map<Tensor> FusionChecker::RunSubInstructions() {}

std::unordered_map<Tensor> FusionChecker::RunTargetInstruction() {}

bool CheckTensorValue(const Tensor& src, const Tensor& dst) {
  CHECK_EQ(src->get_buffer()->GetTarget(), common::DefaultHostTarget()) << "data is not on host!";
  CHECK_EQ(dst->get_buffer()->GetTarget(), common::DefaultHostTarget()) << "data is not on host!";

  int size      = src->shape().size();
  auto src_data = src->data<float>();
  auto dst_data = dst->data<float>();
  for (int idx = 0; idx < size; ++idx) {
    CHECK_LT(fabsf((*src_data - *dst_data) / *rc_data), 1e-5);
    ++src_data;
    ++dst_data;
  }
  return true;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
