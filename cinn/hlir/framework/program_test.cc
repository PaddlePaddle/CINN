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

#include <gtest/gtest.h>

#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace hlir {
namespace framework {

TEST(Program, ExecuteWithRawArgs) {
  // build fronted program
  frontend::Program prog;
  frontend::Variable a("A");
  frontend::Variable b("B");
  Type t   = Float(32);
  a->shape = {100, 32};
  b->shape = {100, 32};
  a->type  = t;
  b->type  = t;
  auto c   = prog.add(a, b);
  auto d   = prog.add(c, b);
  auto e   = prog.add(c, d);
  ASSERT_EQ(prog.size(), 3UL);
  Target target = common::DefaultHostTarget();

  // transform to graph and run pass
  auto g = std::make_shared<Graph>(prog, target);
  ApplyPass(g.get(), "InferShape");

  // compile to runtime program
  auto scope = BuildScope(target, g);
  GraphCompiler gc(target, scope, g);
  GraphCompiler::CompileOptions options;
  options.with_instantiate_variables = false;
  auto&& compilation_result          = gc.Build(options);
  auto&& program                     = compilation_result.runtime_program;

  // prepare arguments
  std::map<std::string, cinn_pod_value_t> name2podargs;
  for (auto& name_view : scope->var_names()) {
    std::string name({name_view.data(), name_view.size()});
    auto tensor = scope->GetTensor(name);
    // ensure the buffer of tensor is not instantiated
    ASSERT_EQ(tensor->data<float>(), nullptr);

    auto* data = tensor->mutable_data<float>(target);
    for (size_t j = 0; j < tensor->shape().numel(); j++) {
      data[j] = (rand() * 1.f) / RAND_MAX;  // NOLINT
    }
    name2podargs.emplace(name, tensor->buffer());
  }

  // program execute
  program->Execute(&name2podargs);

  // check result
  auto A_data = reinterpret_cast<float*>(cinn_pod_value_to_buffer_p(&name2podargs.at("A"))->memory);
  auto B_data = reinterpret_cast<float*>(cinn_pod_value_to_buffer_p(&name2podargs.at("B"))->memory);
  auto E_data = reinterpret_cast<float*>(cinn_pod_value_to_buffer_p(&name2podargs.at(e->id))->memory);
  for (int i = 0; i < 100 * 32; i++) {
    ASSERT_NEAR(2 * A_data[i] + 3 * B_data[i], E_data[i], 1e-5);
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
