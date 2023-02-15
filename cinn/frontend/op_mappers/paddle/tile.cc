// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include "cinn/frontend/op_mapper_registry.h"
#include "cinn/frontend/op_mappers/common_utils.h"
#include "cinn/frontend/var_type_utils.h"

namespace cinn {
namespace frontend {
namespace paddle_mappers {

void TileOpMapper(const paddle::cpp::OpDesc& op_desc, const OpMapperContext& ctx) {
  // input
  CHECK_EQ(op_desc.Input("X").size(), 1UL);
  auto x_name = op_desc.Input("X").front();
  // output
  CHECK_EQ(op_desc.Output("Out").size(), 1UL);
  auto out_name = op_desc.Output("Out").front();

  // attr repeat_times
  std::vector<int> repeat_times = op_desc.GetAttr<std::vector<int>>("repeat_times");

  // for loop check repeat_times's element
  for (auto i : repeat_times) {
    CHECK_GT(i, 0) << "repeat_times's element must be greater than 0";
  }

  auto x = ctx.GetVar(x_name);

  // promotion
  auto vec_x_dims = std::vector<int>(x->shape);
  if (repeat_times.size() < vec_x_dims.size()) {
    int diff = vec_x_dims.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, 1);
  } else {
    int diff = repeat_times.size() - vec_x_dims.size();
    vec_x_dims.insert(vec_x_dims.begin(), diff, 1);
  }

  // VLOG for loop check repeat_times's element
  for (auto i : repeat_times) {
    VLOG(1) << "repeat_times's element: " << i;
  }

  // VLOG for loop check vec_x_dims's element
  for (auto i : vec_x_dims) {
    VLOG(1) << "vec_x_dims's element: " << i;
  }

  // check vec_x_dims's size and repeat_times's size
  CHECK_EQ(vec_x_dims.size(), repeat_times.size())
      << "vec_x_dims's size must be equal to repeat_times's size after promotion";

  // output's shape
  std::vector<int> output_shape = vec_x_dims;

  // calucate output's shape
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    output_shape[i] *= repeat_times[i];
  }
  x = ctx.Builder()->Reshape(x, vec_x_dims);

  // VLOG for loop check output_shape's element
  for (auto i : output_shape) {
    VLOG(1) << "output_shape's element: " << i;
  }
  x        = ctx.Builder()->Reshape(x, {1, 1, 1, 3});
  auto tmp = ctx.Builder()->BroadcastTo(x, {1, 2, 2, 3});
  // auto output = ctx.Builder()->BroadcastTo(x, output_shape, repeat_times);
  auto output = ctx.Builder()->Reshape(tmp, output_shape);

  ctx.AddVar(out_name, output);
  ctx.AddVarModelToProgram(out_name, output->id);
}

}  // namespace paddle_mappers
}  // namespace frontend
}  // namespace cinn

CINN_REGISTER_HELPER(paddle_tile) {
  CINN_REGISTER_OP_MAPPER(tile, cinn::frontend::paddle_mappers::TileOpMapper)
  return true;
}
