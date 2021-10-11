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

#include "cinn/hlir/framework/instruction.h"

#include "cinn/common/test_helper.h"

namespace cinn {
namespace hlir {
namespace framework {

std::vector<cinn_pod_value_t>& Instruction::PreparePodArgs(int i) {
  if (args_cached_.size() > i) return args_cached_[i];
  common::ArgsBuilder builder;
  std::vector<std::string> all_args(in_args_[i].begin(), in_args_[i].end());
  all_args.insert(std::end(all_args), out_args_[i].begin(), out_args_[i].end());

  for (auto& arg : all_args) {
    auto* var = scope_->FindVar(arg);
    CHECK(var) << "Argument [" << arg << "] not found in the scope";

    // TODO(Superjomn) Support other types.
    auto& tensor = absl::get<Tensor>(*var);
    builder.Add(tensor->buffer());
  }

  args_cached_.emplace_back(builder.Build());
  CHECK(args_cached_.size() > i);
  return args_cached_[i];
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
