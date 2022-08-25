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

#include <gtest/gtest.h>

#include <cfloat>

#include "cinn/cinn.h"
#include "cinn/frontend/cinn_builder.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pass/use_pass.h"
#include "cinn/utils/data_util.h"

namespace cinn::frontend {

void RunWithProgram(const Program& program,
                    const Target& target,
                    const std::shared_ptr<hlir::framework::Scope>& scope) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPasses(graph.get(), {"InferShape", "OpFusion"});
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();
}

TEST(TransposeFoldingInput, FoldIntoDotBachedCase1) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y           = builder.CreateInput(Float(32), {4, 5, 6}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto out         = builder.Dot(transpose_x, y);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, FoldIntoDotBachedCase2) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {4, 3, 5}, "X");
  auto y           = builder.CreateInput(Float(32), {4, 6, 5}, "Y");
  auto transpose_y = builder.Transpose(y, {0, 2, 1});
  auto out         = builder.Dot(x, transpose_y);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, FoldIntoDotBachedCase3) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto y           = builder.CreateInput(Float(32), {4, 6, 5}, "Y");
  auto transpose_x = builder.Transpose(x, {0, 2, 1});
  auto transpose_y = builder.Transpose(y, {0, 2, 1});
  auto out         = builder.Dot(transpose_x, transpose_y);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, FoldIntoDotCase1) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 3}, "X");
  auto y           = builder.CreateInput(Float(32), {2, 3}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out         = builder.Dot(x, transpose_y);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, FoldIntoDotCase2) {
  NetBuilder builder("net_builder");
  auto a             = builder.FillConstant<float>({2, 20}, 2.0f, "A");
  auto b             = builder.Transpose(a, {1, 0});
  auto c             = builder.CreateInput(Float(32), {121, 20}, "C");
  auto d             = builder.Matmul(c, b);
  auto x             = builder.FillConstant<float>({2, 20}, 1.0f, "X");
  auto y             = builder.Transpose(x, {1, 0});
  auto z             = builder.CreateInput(Float(32), {121, 20}, "Z");
  auto q             = builder.Matmul(z, y);
  auto out           = builder.Add(d, q);
  auto program       = builder.Build();
  auto target        = common::DefaultTarget();
  auto graph         = std::make_shared<hlir::framework::Graph>(program, target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  auto before_scope = hlir::framework::BuildScope(target, graph);
  before_scope->Var<hlir::framework::Tensor>("C");
  before_scope->Var<hlir::framework::Tensor>("Z");
  SetRandData<float>(before_scope->GetTensor("C"), target);
  SetRandData<float>(before_scope->GetTensor("Z"), target);
  RunWithProgram(program, target, before_scope);
  auto origin_out = GetTensorData<float>(before_scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  auto after_scope = hlir::framework::BuildScope(target, graph);
  after_scope->Var<hlir::framework::Tensor>("C");
  after_scope->Var<hlir::framework::Tensor>("Z");
  after_scope->GetTensor("C")->set_buffer(before_scope->GetTensor("C")->get_buffer());
  after_scope->GetTensor("Z")->set_buffer(before_scope->GetTensor("Z")->get_buffer());
  RunWithProgram(program, target, after_scope);
  auto folded_out = GetTensorData<float>(after_scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, TransposeOutInFetchIds) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 3}, "X");
  auto y           = builder.CreateInput(Float(32), {2, 3}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out         = builder.Dot(x, transpose_y);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {transpose_y->id}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, TransposeOutUsedByOtherInstrs) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 2}, "X");
  auto y           = builder.CreateInput(Float(32), {2, 2}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto dot         = builder.Dot(x, transpose_y);
  auto out         = builder.Add(transpose_y, dot);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFoldingInput, TransposeTwiceWithMatmul) {
  CinnBuilder builder("cinn_builder");
  auto x = builder.CreateInput(Float(32), {2, 20}, "X");
  auto y = builder.CreateInput(Float(32), {10201, 20}, "Y");
  auto z = builder.CreateInput(Float(32), {10201, 2}, "Z");

  auto x_t     = builder.Transpose(x, {1, 0});
  auto x_t_t   = builder.Transpose(x_t, {1, 0});
  auto dot1    = builder.Dot(y, x_t);
  auto dot2    = builder.Dot(z, x_t_t);
  auto program = builder.Build();

  auto target = common::DefaultTarget();
  auto graph  = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope  = hlir::framework::BuildScope(target, graph);

  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  scope->Var<hlir::framework::Tensor>("Z");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  SetRandData<float>(scope->GetTensor("Z"), target);

  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  // The origin Program should be：
  // Program {
  // var_51 = transpose(X, axis=[1,0])
  // var_52 = transpose(var_51, axis=[1,0])
  // var_53 = matmul(Y, var_51)
  // var_54 = matmul(Z, var_52)
  // }
  RunWithProgram(program, target, scope);
  auto origin_out1 = GetTensorData<float>(scope->GetTensor(dot1->id), target);
  auto origin_out2 = GetTensorData<float>(scope->GetTensor(dot2->id), target);

  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  // The program after transpose folding pass should be:
  // Program {
  // var_51 = transpose(X, axis=[1,0])
  // var_53 = matmul(Y, X, trans_b=true)
  // var_54 = matmul(Z, var_51, trans_b=true)
  // }
  // the transpose of x->x_t should retain
  RunWithProgram(program, target, scope);
  auto folded_out1 = GetTensorData<float>(scope->GetTensor(dot1->id), target);
  auto folded_out2 = GetTensorData<float>(scope->GetTensor(dot2->id), target);

  ASSERT_EQ(origin_size - 1, folded_size);
  ASSERT_EQ(origin_out1.size(), folded_out1.size());
  for (size_t i = 0; i < origin_out1.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out1[i], folded_out1[i]);
  }
  ASSERT_EQ(origin_out2.size(), folded_out2.size());
  for (size_t i = 0; i < origin_out2.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out2[i], folded_out2[i]);
  }
}

TEST(TransposeFoldingInput, TransposeWithMultiMamtul) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.CreateInput(Float(32), {2, 2}, "X");
  auto y           = builder.CreateInput(Float(32), {2, 2}, "Y");
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto dot1        = builder.Dot(x, transpose_y);
  auto dot2        = builder.Dot(transpose_y, x);
  auto out         = builder.Add(dot1, dot2);
  auto program     = builder.Build();
  auto target      = common::DefaultTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);
  scope->Var<hlir::framework::Tensor>("X");
  scope->Var<hlir::framework::Tensor>("Y");
  SetRandData<float>(scope->GetTensor("X"), target);
  SetRandData<float>(scope->GetTensor("Y"), target);
  size_t origin_size = program.size();
  VLOG(1) << "Program:\n" << program;
  // Program {
  //   var_60 = transpose(Y, axis=[1,0])
  //   var_61 = matmul(X, var_60)
  //   var_62 = matmul(var_60, X)
  //   var_63 = elementwise_add(var_61, var_62)
  // }
  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ProgramPass::Apply(&program, {}, target, {"TransposeFoldingInput"});
  size_t folded_size = program.size();
  VLOG(1) << "Program:\n" << program;
  // Program {
  //   var_61 = matmul(X, Y, trans_b=true)
  //   var_62 = matmul(Y, X, trans_a=true)
  //   var_63 = elementwise_add(var_61, var_62)
  // }
  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData<float>(scope->GetTensor(out->id), target);
  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

}  // namespace cinn::frontend
