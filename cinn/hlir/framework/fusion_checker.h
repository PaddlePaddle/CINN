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

#pragma once

#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

class FusionChecker {
 public:
  FusionChecker(const Instruction* target_instr,
                const Graph& graph_,
                const Graph::Group group,
                const common::Target target);
  bool operator()();

 private:
  bool RunChecker();
  void InitInputTensor();
  template <class T>
  void GetRandom(T* data, size_t size);
  std::unordered_map<Tensor> RunSubInstructions();
  std::unordered_map<Tensor> RunTargetInstruction();
  bool CheckTensorValue(const Tensor& src, const Tensor& dst);
  Tensor FusionChecker::TensorHostToDevice(Tensor& src, void* stream);
  Tensor FusionChecker::TensorDeviceToHost(Tensor& src, void* stream);

  void BuildSubInstrution();
  std::vector<std::string> OpGetOutputNames(const Node* node);
  std::vector<std::string> OpGetInputNames(const Node* node);
  std::string FusionChecker::GenOpFuncName(const Node* node) const { return "fn_" + node->id(); }
  const std::string& GraphCompiler::GetOrGenFullFuncName(const std::string& prefix);

  // input values
  std::unordered_map<Tensor> input_tensors;

  // target and instruction
  Graph graph_;
  Graph::Group group_;
  common::Target target_;

  Instruction* target_instr_;
  std::vector<std::shared_ptr<Instruction>*> sub_instrs_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
