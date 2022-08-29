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

#include <absl/types/any.h>
#include <gtest/gtest.h>

#include <string>

#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/hlir/pe/broadcast.h"
#include "cinn/lang/packed_func.h"
#include "cinn/utils/data_util.h"

namespace cinn {
namespace hlir {
namespace framework {

TEST(Operator, GetAttrs) {
  frontend::Program prog;
  // TODO(Superjomn) Replace with Placeholder here.
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
  auto g        = std::make_shared<Graph>(prog, target);
  ApplyPass(g.get(), "InferShape");

  auto scope = BuildScope(target, g);

  GraphCompiler gc(target, scope, g);
  std::unique_ptr<Program> program = gc.Build();

  auto A = scope->GetTensor("A");
  auto B = scope->GetTensor("B");
  SetRandData<float>(A, target);
  SetRandData<float>(B, target);

  program->Execute();

  auto A_data = A->data<float>();
  auto B_data = B->data<float>();
  auto E_data = scope->GetTensor(e->id)->data<float>();
  for (int i = 0; i < 100 * 32; i++) {
    LOG_FIRST_N(INFO, 3) << "data: " << 2 * A_data[i] << " + " << 3 * B_data[i] << " = " << E_data[i];
    ASSERT_NEAR(2 * A_data[i] + 3 * B_data[i], E_data[i], 1e-5);
  }
}
}  // namespace framework

}  // namespace hlir
}  // namespace cinn
