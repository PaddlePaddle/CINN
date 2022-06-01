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
#include <absl/container/flat_hash_map.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "cinn/backends/compiler.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace framework {

class FusionChecker {
 public:
  FusionChecker(Instruction* target_instr,
                backends::Compiler* compiler,
                absl::flat_hash_map<std::string, std::string>& prefix2full_namemap,
                Graph* graph,
                std::shared_ptr<Graph::Group>& group,
                common::Target& target);
  bool operator()();

 private:
  bool RunChecker();
  void InitInputTensor();
  template <class T>
  void GetRandom(T* data, size_t size);
  std::unordered_map<std::string, Tensor> RunSubInstructions();
  std::unordered_map<std::string, Tensor> RunTargetInstruction();

  Tensor TensorHostToDevice(Tensor& src);
  Tensor TensorDeviceToHost(Tensor& src);
  bool CheckTensorValue(const Tensor& src, const Tensor& dst);

  void BuildSubInstructions();
  std::vector<std::string> OpGetOutputNames(const Node* node);
  std::vector<std::string> OpGetInputNames(const Node* node);
  std::string GenOpFuncName(const Node* node) const { return "fn_" + node->id(); }
  const std::string& GetOrGenFullFuncName(const std::string& prefix);

  // input values
  std::unordered_map<std::string, Tensor> input_tensors_;

  // target and instruction
  Graph* graph_;
  common::Target target_;

  backends::Compiler* compiler_;
  std::shared_ptr<Graph::Group> group_;

  void* stream_;
  Instruction* target_instr_;
  std::vector<std::shared_ptr<Instruction>> sub_instrs_;
  absl::flat_hash_map<std::string, std::string> prefix2full_namemap_;
};

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
