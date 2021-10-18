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
#include "cinn/frontend/syntax.h"

namespace cinn {
namespace frontend {
namespace decomposer {

void elementwise_add(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 2UL) << " 2 input tensors for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 1UL) << "1 output tensor for " << instr->op_type;
  auto x        = instr->inputs[0];
  auto y        = instr->inputs[1];
  auto output   = instr->outputs[0];
  int axis      = instr.GetAttrs<int>("axis");
  axis          = axis > 0 ? axis : x->shape.size() - y->shape.size();
  auto* builder = context.builder();

  Variable out;
  if (x->shape == y->shape) {
    out = builder->Add(x, y);
  } else {
    std::vector<int> bcast_axes(y->shape.size());
    std::iota(bcast_axes.begin(), bcast_axes.end(), axis);
    auto bcast_y = builder->BroadcastTo(y, x->shape, bcast_axes);
    out          = builder->Add(x, bcast_y);
  }

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(out, output);
}

void elementwise_add_grad(const Instruction& instr, const DecomposerContext& context) {
  CHECK_EQ(instr->inputs.size(), 3UL) << " 3 input tensors for " << instr->op_type;
  CHECK_EQ(instr->outputs.size(), 2UL) << "2 output tensors for " << instr->op_type;
  auto dout     = instr->inputs[0];
  auto dx       = instr->outputs[0];
  auto dy       = instr->outputs[1];
  int axis      = instr.GetAttrs<int>("axis");
  axis          = axis > 0 ? axis : axis + dx->shape.size();
  auto* builder = context.builder();

  auto dx_t = builder->Identity(dout);
  Variable dy_t;
  if (dx->shape == dy->shape) {
    dy_t = builder->Identity(dout);
  } else {
    // e.g., dx.shape = [2, 3, 4, 5], dy.shape = [3, 4], axis = 1, reduce_dims=[0, 3]
    std::vector<int> reduce_dims;
    for (size_t i = 0; i < dx->shape.size(); ++i) {
      if (i < axis || i >= axis + dy->shape.size()) {
        reduce_dims.push_back(i);
      }
    }
    dy_t = builder->Reduce(dout, ReduceKind::kSum, reduce_dims);
  }

  // map the the output of decomposed operator to the original.
  context.MapOutToOrigin(dx_t, dx);
  context.MapOutToOrigin(dy_t, dy);
}

}  // namespace decomposer
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(broadcast) {
  CINN_DECOMPOSER_REGISTER(elementwise_add, cinn::frontend::decomposer::elementwise_add);

  return true;
}

CINN_REGISTER_HELPER(broadcast_grad) {
  CINN_DECOMPOSER_REGISTER(elementwise_add_grad, cinn::frontend::decomposer::elementwise_add_grad);

  return true;
}
