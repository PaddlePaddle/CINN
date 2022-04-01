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

#include "cinn/frontend/program_pass.h"

#include <unordered_set>

namespace cinn {
namespace frontend {

void ProgramPass::Apply(Program* prog,
                        const std::unordered_set<std::string>& fetch_ids,
                        const common::Target& target,
                        const std::vector<std::string>& passes) {
  std::vector<const ProgramPass*> fpass;
  for (auto& name : passes) {
    const auto* pass = ProgramPassRegistry::Global()->Get(name);
    fpass.push_back(pass);
  }
  for (const auto* pass : fpass) {
    pass->ApplyImpl(prog, fetch_ids, target);
  }
}

void ApplyPass(Program* program, const std::unordered_set<std::string>& fetch_ids, const std::string& pass) {
  auto* reg = Registry<ProgramPassFunctionRegistry>::Global()->Find(pass);
  CHECK(reg) << "Cannot find pass " << pass << " in the registry";
  reg->body(program, fetch_ids);
}

}  // namespace frontend
}  // namespace cinn
