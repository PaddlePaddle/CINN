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

#include <time.h>

#include <random>

#ifdef CINN_WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace cinn {
namespace hlir {
namespace framework {

FusionChecker::FusionChecker(Instruction* target_instr,
                             backends::Compiler* compiler,
                             absl::flat_hash_map<std::string, std::string>& prefix2full_namemap,
                             Graph* graph,
                             std::shared_ptr<Graph::Group>& group,
                             common::Target& target)
    : target_instr_(target_instr),
      compiler_(compiler),
      prefix2full_namemap_(prefix2full_namemap),
      graph_(graph),
      group_(group),
      target_(target) {
  BuildSubInstructions();
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

const std::string& FusionChecker::GetOrGenFullFuncName(const std::string& prefix) {
  // try_emplace only insert once, so the same function
  // can get a consistent name next time
  prefix2full_namemap_.try_emplace(prefix, Context::Global().NewName(prefix));
  return prefix2full_namemap_.at(prefix);
}

void FusionChecker::BuildSubInstructions() {
  auto nodes = group_->CollectNodes();
  for (auto node : nodes) {
    auto instr_name = node->op()->name;
    auto instr =
        std::make_shared<Instruction>(target_, nullptr, OpGetInputNames(node), OpGetOutputNames(node), instr_name);
    std::string op_func_name = GetOrGenFullFuncName(GenOpFuncName(node));
    auto* fn                 = compiler_->Lookup(op_func_name);
    CHECK(fn);
    instr->SetLoweredFunc(fn, op_func_name);
    instr->Finalize();
    sub_instrs_.push_back(instr);
  }
}

bool FusionChecker::operator()() { return RunChecker(); }

bool FusionChecker::RunChecker() {
  InitInputTensor();
  auto src_tensors = RunSubInstructions();
  auto dst_tensors = RunTargetInstruction();
  for (auto& src : src_tensors) {
    CHECK(dst_tensors.count(src.first)) << "Can't find output var: " << src.first << " in other set!";
    auto max_diff = CheckTensorValue(src.second, dst_tensors[src.first]);
    LOG(WARNING) << "The max relative diff of output var: " << src.first << " in group: " << group_->group_id
                 << " is: " << max_diff;
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
  auto& dtype_dict  = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
  auto& shape_dict  = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  for (auto& name : input_names) {
    Tensor tensor;
    tensor->Resize(Shape(shape_dict.at(name)));
    tensor->set_type(dtype_dict.at(name));
    if (tensor->type() == Float(32)) {
      GetRandom(tensor->mutable_data<float>(common::DefaultHostTarget()), tensor->shape().numel());
    } else {
      LOG(FATAL) << "Not Support Now!";
    }
    if (target_ == common::DefaultHostTarget()) {
      input_tensors_[name] = tensor;
    } else {
      input_tensors_[name] = TensorHostToDevice(tensor);
    }
  }
}

Tensor FusionChecker::TensorHostToDevice(Tensor& src) {
  if (src->get_buffer()->GetTarget() == common::DefaultNVGPUTarget()) {
    return src;
  }
  Tensor dst;
  dst->Resize(src->shape());
  auto src_data = src->data<void>();
  auto dst_data = dst->mutable_data(common::DefaultNVGPUTarget(), src->type());
#ifdef CINN_WITH_CUDA
  int width = src->type().bits() / 8;
  auto st   = cudaMemcpy(dst_data, src_data, src->shape().numel() * width, cudaMemcpyHostToDevice);
  if (st) {
    LOG(FATAL) << "Error Ocurs -> " << cudaGetErrorString(st);
  }
#endif

  return dst;
}

Tensor FusionChecker::TensorDeviceToHost(Tensor& src) {
  if (src->get_buffer()->GetTarget() == common::DefaultHostTarget()) {
    return src;
  }
  Tensor dst;
  dst->Resize(src->shape());
  auto src_data = src->data<void>();
  auto dst_data = dst->mutable_data(common::DefaultHostTarget(), src->type());

#ifdef CINN_WITH_CUDA
  int width = src->type().bits() / 8;
  auto st   = cudaMemcpy(dst_data, src_data, src->shape().numel() * width, cudaMemcpyDeviceToHost);
  if (st) {
    LOG(FATAL) << "Error Ocurs -> " << cudaGetErrorString(st);
  }
#endif

  return dst;
}

std::unordered_map<std::string, Tensor> FusionChecker::RunSubInstructions() {
  auto& dtype_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
  auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  std::unordered_map<std::string, Tensor> tensor_map;
  for (auto& input : input_tensors_) {
    tensor_map[input.first] = input.second;
  }

  for (auto& instr : sub_instrs_) {
    std::map<std::string, cinn_pod_value_t> run_args;
    auto in_args = instr->GetInArgs().front();
    for (auto& arg : in_args) {
      CHECK(tensor_map.count(arg));
      run_args[arg] = cinn_pod_value_t(tensor_map[arg]->buffer());
    }

    auto out_args = instr->GetOutArgs().front();
    for (auto& arg : out_args) {
      CHECK(!tensor_map.count(arg));
      Tensor tensor;
      tensor->Resize(Shape(shape_dict.at(arg)));
      tensor->set_type(dtype_dict.at(arg));
      if (tensor->type() == Float(32)) {
        tensor->mutable_data<float>(target_);
      } else {
        LOG(FATAL) << "Not Support Now!";
      }
      tensor_map[arg] = tensor;
      run_args[arg]   = cinn_pod_value_t(tensor->buffer());
    }
    instr->Run(&run_args);
  }

  std::unordered_map<std::string, Tensor> output_tensors;
  auto out_args = target_instr_->GetOutArgs().front();
  for (auto& arg : out_args) {
    CHECK(tensor_map.count(arg));
    auto tensor = tensor_map[arg];
    if (target_ == common::DefaultNVGPUTarget()) {
      output_tensors[arg] = std::move(TensorDeviceToHost(tensor));
    } else {
      output_tensors[arg] = tensor;
    }
  }

  return output_tensors;
}

std::unordered_map<std::string, Tensor> FusionChecker::RunTargetInstruction() {
  auto& dtype_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, common::Type>>("inferdtype");
  auto& shape_dict = graph_->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  std::map<std::string, cinn_pod_value_t> run_args;
  auto in_args = target_instr_->GetInArgs().front();
  for (auto& arg : in_args) {
    CHECK(input_tensors_.count(arg)) << "Can't find data in input tensors map";
    run_args[arg] = cinn_pod_value_t(input_tensors_[arg]->buffer());
  }

  std::unordered_map<std::string, Tensor> output_tensors;
  auto out_args = target_instr_->GetOutArgs().front();
  for (auto& arg : out_args) {
    Tensor tensor;
    tensor->Resize(Shape(shape_dict.at(arg)));
    tensor->set_type(dtype_dict.at(arg));
    if (tensor->type() == Float(32)) {
      tensor->mutable_data<float>(target_);
    } else {
      LOG(FATAL) << "Not Support Now!";
    }
    run_args[arg]       = cinn_pod_value_t(tensor->buffer());
    output_tensors[arg] = tensor;
  }

  target_instr_->Run(&run_args, false, nullptr, false);

  if (target_ == common::DefaultNVGPUTarget()) {
    auto out_args = target_instr_->GetOutArgs().front();
    for (auto& arg : out_args) {
      auto tensor         = output_tensors[arg];
      output_tensors[arg] = TensorDeviceToHost(tensor);
    }
  }
  return output_tensors;
}

float FusionChecker::CheckTensorValue(const Tensor& src, const Tensor& dst) {
  CHECK_EQ(src->get_buffer()->GetTarget(), common::DefaultHostTarget()) << "data is not on host!";
  CHECK_EQ(dst->get_buffer()->GetTarget(), common::DefaultHostTarget()) << "data is not on host!";

  int size       = src->shape().numel();
  auto src_data  = src->data<float>();
  auto dst_data  = dst->data<float>();
  float max_diff = 0.0f;
  for (int idx = 0; idx < size; ++idx) {
    auto diff = fabsf(*src_data - *dst_data);
    if (diff > 0.0f) {
      max_diff = std::max(diff / std::max(fabsf(*src_data), fabsf(*dst_data)), max_diff);
    }
    ++src_data;
    ++dst_data;
  }
  return max_diff;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
