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

#include "cinn/frontend/decomposer_registry.h"

namespace cinn {
namespace frontend {
namespace decomposer {

// conv2d backward data/filter
void conv2d_grad(const Instruction& instr, const DecomposerContext& context) {
  auto& dy = instr->inputs[0];
  auto& x  = instr->inputs[1];
  auto& w  = instr->inputs[2];

  CinnBuilder* builder = context.builder();
  // create backward data
  if (!instr->outputs[0].is_const()) {
    auto dx = builder->Conv(w,
                            dy,
                            instr.GetAttrs<std::vector<int>>("strides"),
                            instr.GetAttrs<std::vector<int>>("paddings"),
                            instr.GetAttrs<std::vector<int>>("dilations"),
                            instr.GetAttrs<int>("groups"),
                            "backward_data",
                            instr.GetAttrs<std::string>("data_format"),
                            instr.GetAttrs<std::string>("padding_algorithm"),
                            x->shape);
    context.MapOutToOrigin(dx, instr->outputs[0]);
  }

  // create backward filter
  auto dw = builder->Conv(x,
                          dy,
                          instr.GetAttrs<std::vector<int>>("strides"),
                          instr.GetAttrs<std::vector<int>>("paddings"),
                          instr.GetAttrs<std::vector<int>>("dilations"),
                          instr.GetAttrs<int>("groups"),
                          "backward_filter",
                          instr.GetAttrs<std::string>("data_format"),
                          instr.GetAttrs<std::string>("padding_algorithm"),
                          w->shape);
  context.MapOutToOrigin(dw, instr->outputs[1]);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(conv2d_grad_decomposer) {
  CINN_DECOMPOSER_REGISTER(conv2d_grad, cinn::frontend::decomposer::conv2d_grad);

  return true;
}
