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

FusionChecker::FusionChecker(const Instruction* target_instr,
                             const Graph& graph,
                             const GroupPtr& group,
                             const common::Target& target)
    : target_instr_(target_instr), graph_(graph), group_(group), target_(target) {
  BuildSubInstrution();
}

std::vector<std::string> FusionChecker::OpGetInputNames(const Node* node) {
  std::vector<std::string> res;
  for (auto& i : node->inlinks_in_order()) {
    res.push_back(i->source()->as<NodeData>()->id());
  }
  return res;
}

std::vector<std::string> FusionChecker::OpGetOutputNames(const Node* node) {
  std::vector<std::string> res;
  for (auto& i : node->outlinks_in_order()) {
    res.push_back(i->sink()->as<NodeData>()->id());
  }
  return res;
}

const std::string& GraphCompiler::GetOrGenFullFuncName(const std::string& prefix) {
  // try_emplace only insert once, so the same function
  // can get a consistent name next time
  prefix2full_namemap_.try_emplace(prefix, Context::Global().NewName(prefix));
  return prefix2full_namemap_.at(prefix);
}

FusionChecker::BuildSubInstrutions() {
  auto nodes = group_->CollectNodes();
  for (auto node : nodes) {
    auto instr_name = node->op()->name;

    auto instr =
        std::make_shared<Instruction>(target_, nullptr, OpGetInputNames(node), OpGetOutputNames(node), instr_name);
    std::string op_func_name = GetOrGenFullFuncName(GenOpFuncName(node));
    auto* fn                 = compiler_->Lookup(op_func_name);
    CHECK(fn);
    instr->SetLoweredFunc(fn, op_func_name);
    sub_instrs_.push_back(instr);
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
    input_tensors_[name] = tensor;
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

std::unordered_map<Tensor> FusionChecker::RunSubInstructions(void* stream) {
  auto& dtype_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  std::unordered_map<std::string, Tensor> tensor_map;
  for (auto& input : input_tensors) {
    tensor_map[input.first] = TensorHostToDevice(input.second, stream);
  }

  for (auto& instr : sub_instrs_) {
    std::map<std::string, cinn_pod_value_t> run_args;
    for (auto& arg : instr->GetInArgs()) {
      if (!tensor_map.count(arg)) {
        Tensor tensor;
        tensor->Resize(shape_dict.at(arg));
        tensor->set_type(dtype_dict.at(arg));
        if (tensor->type() == Float(32)) {
          tensor->mutable_data<float>(target_);
        } else {
          LOG(FATAL) << "Not Support Now!";
        }
        input_tensors_[arg] = tensor;
        tensor_map[arg]     = tensor;
      }

      run_args[arg] = cinn_pod_value_t(tensor_map[arg]->buffer());
    }

    for (auto& arg : instr->GetOutArgs()) {
      if (!tensor_map.count(arg)) {
        Tensor tensor;
        tensor->Resize(shape_dict.at(arg));
        tensor->set_type(dtype_dict.at(arg));
        if (tensor->type() == Float(32)) {
          tensor->mutable_data<float>(target_);
        } else {
          LOG(FATAL) << "Not Support Now!";
        }
        input_tensors_[arg] = tensor;
        tensor_map[arg]     = tensor;
      }
      run_args[arg] = cinn_pod_value_t(tensor_map[arg]->buffer());
    }

    instr->run(run_args);
  }

  std::unordered_map<std::string, Tensor> output_tensors;
  for (auto& arg : target_instr_->GetOutArgs()) {
    output_tensors[arg] = TensorDeviceToHost(tensor_map[arg]);
  }
  return output_tensors;
}

std::unordered_map<std::string, Tensor> FusionChecker::RunTargetInstruction() {
  auto& dtype_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  std::map<std::string, cinn_pod_value_t> run_args;
  for (auto& arg : target_instr_->GetInArgs()) {
    if (!tensor_map.count(arg)) {
      Tensor tensor;
      tensor->Resize(shape_dict.at(arg));
      tensor->set_type(dtype_dict.at(arg));
      if (tensor->type() == Float(32)) {
        tensor->mutable_data<float>(target_);
      } else {
        LOG(FATAL) << "Not Support Now!";
      }
      input_tensors_[arg] = tensor;
      tensor_map[arg]     = tensor;
    }

    run_args[arg] = cinn_pod_value_t(tensor_map[arg]->buffer());
  }

  std::unordered_map<std::string, Tensor> output_tensors;
  for (auto& arg : target_instr_->GetOutArgs()) {
    if (!tensor_map.count(arg)) {
      Tensor tensor;
      tensor->Resize(shape_dict.at(arg));
      tensor->set_type(dtype_dict.at(arg));
      if (tensor->type() == Float(32)) {
        tensor->mutable_data<float>(target_);
      } else {
        LOG(FATAL) << "Not Support Now!";
      }
      input_tensors_[arg] = tensor;
      tensor_map[arg]     = tensor;
    }
    run_args[arg]       = cinn_pod_value_t(tensor_map[arg]->buffer());
    output_tensors[arg] = tensor_map[arg];
  }

  target_instr_->run(run_args);
  std::unordered_map<std::string, Tensor> output_tensors;
  for (auto& arg : target_instr_->GetOutArgs()) {
    output_tensors[arg] = TensorDeviceToHost(output_tensors[arg]);
  }
  return output_tensors;
}

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
