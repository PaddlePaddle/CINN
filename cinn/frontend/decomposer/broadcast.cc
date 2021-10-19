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

void GetReduceDimsForX(const std::vector<int>& dx_shape,
                       const std::vector<int>& dout_shape,
                       std::vector<int>* reduce_dims) {
  // e.g., dx_shape = [4, 1, 3], dout_shape = [4, 2, 3], reduce_dims=[1]
  for (size_t i = 0; i < dout_shape.size(); ++i) {
    if (dx_shape[i] == 1 && dout_shape[i] != 1) {
      reduce_dims->push_back(i);
    }
  }
  VLOG(3) << "The reduce_dims for X: " << utils::Join(*reduce_dims, ",");
}

void GetReduceDimsForY(const std::vector<int>& dy_shape,
                       const std::vector<int>& dout_shape,
                       int axis,
                       std::vector<int>* reduce_dims) {
  // e.g., dy_shape = [3, 1, 4], dx_shape = [2, 3, 4, 4, 5], axis = 1
  // reduce_dims=[0, 2, 4]
  for (size_t i = 0; i < dout_shape.size(); ++i) {
    if (i < axis || i >= axis + dy_shape.size()) {
      reduce_dims->push_back(i);
    } else {
      if (dy_shape[i - axis] == 1 && dout_shape[i] != 1) {
        reduce_dims->push_back(i);
      }
    }
  }
  VLOG(3) << "The reduce_dims for Y: " << utils::Join(*reduce_dims, ",");
}

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
  if (x->shape == output->shape) {
    out = builder->Add(x, y);
  } else {
    // e.g., x.shape = [4, 1, 3], out.shape = [4, 2, 3], bcast_axes_x = [0, 1, 2]
    std::vector<int> bcast_axes_x(x->shape.size());
    std::iota(bcast_axes_x.begin(), bcast_axes_x.end(), 0);
    auto bcast_x = builder->BroadcastTo(x, output->shape, bcast_axes_x);
    // e.g., aixs = 1, y.shape = [2, 3], bcast_axes_y = [1, 2]
    std::vector<int> bcast_axes_y(y->shape.size());
    std::iota(bcast_axes_y.begin(), bcast_axes_y.end(), axis);
    auto bcast_y = builder->BroadcastTo(y, output->shape, bcast_axes_y);
    out          = builder->Add(bcast_x, bcast_y);
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
  axis          = axis > 0 ? axis : dx->shape.size() - dy->shape.size();
  auto* builder = context.builder();

  Variable dx_t;
  if (dx->shape == dout->shape) {
    dx_t = builder->Identity(dout);
  } else {
    std::vector<int> x_reduce_dims;
    GetReduceDimsForX(dx->shape, dout->shape, &x_reduce_dims);
    dx_t = builder->Reduce(dout, ReduceKind::kSum, x_reduce_dims, true);
  }

  Variable dy_t;
  if (dy->shape == dout->shape) {
    dy_t = builder->Identity(dout);
  } else {
    std::vector<int> y_reduce_dims;
    GetReduceDimsForY(dy->shape, dout->shape, axis, &y_reduce_dims);
    auto dy_res = builder->Reduce(dout, ReduceKind::kSum, y_reduce_dims, true);
    dy_t        = builder->Reshape(dy_res, dy->shape);
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
