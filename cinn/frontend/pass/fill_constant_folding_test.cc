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

namespace cinn::frontend {

Target GetTarget() {
#ifdef CINN_WITH_CUDA
  return common::DefaultNVGPUTarget();
#else
  return common::DefaultHostTarget();
#endif
}

std::vector<float> GetTensorData(const hlir::framework::Tensor& tensor, Target target) {
  std::vector<float> data;
#ifdef CINN_WITH_CUDA
  data.resize(tensor->shape().numel());
  CUDA_CALL(cudaMemcpy(data.data(),
                       reinterpret_cast<void*>(tensor->mutable_data<float>(target)),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyDeviceToHost));
#else
  for (size_t i = 0; i < tensor->shape().numel(); ++i) {
    data.push_back(tensor->data<float>()[i]);
  }
#endif
  return data;
}

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

TEST(TransposeFolding, FoldTwoFillConstant) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y           = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto transpose_y = builder.Transpose(y, {1, 0});
  auto out         = builder.Add(transpose_x, transpose_y);
  auto program     = builder.Build();
  auto target      = GetTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData(scope->GetTensor(out->id), target);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData(scope->GetTensor(out->id), target);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldTwoFillConstantWithSameOuput) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y       = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto out     = builder.Add(x, y);
  auto program = builder.Build();
  auto target  = GetTarget();
  auto graph   = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope   = hlir::framework::BuildScope(target, graph);

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData(scope->GetTensor(out->id), target);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData(scope->GetTensor(out->id), target);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldThreeFillConstant) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y           = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto z           = builder.FillConstant<float>({32, 32}, 1.0f, "z");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto out         = builder.Add(y, z);
  auto program     = builder.Build();
  auto target      = GetTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData(scope->GetTensor(out->id), target);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData(scope->GetTensor(out->id), target);

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

TEST(TransposeFolding, FoldThreeFillConstantWithOneDiff) {
  CinnBuilder builder("cinn_builder");
  auto x           = builder.FillConstant<float>({32, 32}, 1.0f, "x");
  auto y           = builder.FillConstant<float>({32, 32}, 1.0f, "y");
  auto z           = builder.FillConstant<float>({32, 32}, 0.0f, "z");
  auto transpose_x = builder.Transpose(x, {1, 0});
  auto out         = builder.Add(y, z);
  auto program     = builder.Build();
  auto target      = GetTarget();
  auto graph       = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope       = hlir::framework::BuildScope(target, graph);

  size_t origin_size = program.size();
  VLOG(1) << "Program Before FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto origin_out = GetTensorData(scope->GetTensor(out->id), target);

  ProgramPass::Apply(&program, {}, target, {"FillConstantFolding"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after FillConstantFolding:\n" << program;

  RunWithProgram(program, target, scope);
  auto folded_out = GetTensorData(scope->GetTensor(out->id), target);

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_FLOAT_EQ(origin_out[i], folded_out[i]);
  }
}

}  // namespace cinn::frontend
