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

#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn::frontend {

TEST(Decomposer, relu) {
  NetBuilder builder("relu");
  auto x   = builder.CreateInput(Float(32), {20, 10});
  auto out = builder.relu(x);

  std::vector<std::string> input_names = {"X"};
  RunProgram<float>(builder, input_names);
}

TEST(Decomposer, relu_grad) {
  NetBuilder builder("relu_grad");
  auto dout = builder.CreateInput(Float(32), {20, 10});
  auto out  = builder.CreateInput(Float(32), {20, 10});
  auto dx   = builder.relu_grad(dout, out);

  std::vector<std::string> input_names = {"Dout", "Out"};
  RunProgram<float>(builder, input_names);
}

}  // namespace cinn::frontend
