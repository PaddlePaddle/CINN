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

void SetInputData(const hlir::framework::Tensor& tensor, Target target) {
#ifdef CINN_WITH_CUDA
  auto* data = tensor->mutable_data<float>(target);
  std::vector<float> host_memory(tensor->shape().numel(), 0);
  for (size_t i = 0; i < tensor->shape().numel(); ++i) {
    host_memory[i] = static_cast<float>(i);
  }
  CUDA_CALL(cudaMemcpy(reinterpret_cast<void*>(data),
                       host_memory.data(),
                       tensor->shape().numel() * sizeof(float),
                       cudaMemcpyHostToDevice));
#else
  auto* data = tensor->mutable_data<float>(target);
  for (size_t j = 0; j < tensor->shape().numel(); j++) {
    data[j] = static_cast<float>(j);  // All random data
  }
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

std::vector<std::vector<float>> RunWithProgram(const Program& program,
                                               const Target& target,
                                               const std::vector<std::string>& input_names,
                                               const std::vector<std::string>& out_ids) {
  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  auto scope = hlir::framework::BuildScope(target, graph);

  for (const auto& in_name : input_names) {
    scope->Var<hlir::framework::Tensor>(in_name);
    SetInputData(scope->GetTensor(in_name), target);
  }

  hlir::framework::ApplyPasses(graph.get(), {"InferShape", "OpFusion"});
  VLOG(1) << "graph:\n" << graph->Visualize();
  hlir::framework::GraphCompiler gc(target, scope, graph);
  auto runtime_program = gc.Build();
  runtime_program->Execute();

  std::vector<std::vector<float>> outputs;
  for (const auto& out_id : out_ids) {
    outputs.emplace_back(GetTensorData(scope->GetTensor(out_id), target));
  }
  return outputs;
}

TEST(TransposeCollapsing, FuseTwoTranspose) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_t     = builder.Transpose(x, {0, 2, 1});
  auto out     = builder.Transpose(x_t, {2, 1, 0});
  auto program = builder.Build();
  auto target  = GetTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_0 = transpose(X, axis=[0,2,1])
  //   var_1 = transpose(var_0, axis=[2,1,0])
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, {out->id});

  ProgramPass::Apply(&program, {}, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_1 = transpose(X, axis=[1,2,0])
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, {out->id});

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseThreeTranspose) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t    = builder.Transpose(x, {0, 2, 1});
  auto x_2t    = builder.Transpose(x_1t, {2, 1, 0});
  auto out     = builder.Transpose(x_2t, {1, 2, 0});
  auto program = builder.Build();
  auto target  = GetTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_4 = transpose(X, axis=[0,2,1])
  //   var_5 = transpose(var_4, axis=[2,1,0])
  //   var_6 = transpose(var_5, axis=[1,2,0])
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, {out->id});

  ProgramPass::Apply(&program, {}, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_6 = transpose(X, axis=[2,0,1])
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, {out->id});

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, RemoveUselessTranspose) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_t     = builder.Transpose(x, {0, 1, 2});
  auto out     = builder.Add(x, x_t);
  auto program = builder.Build();
  auto target  = GetTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_9 = transpose(X, axis=[0,1,2])
  //   var_10 = elementwise_add(X, var_9)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, {out->id});

  ProgramPass::Apply(&program, {}, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_10 = elementwise_add(X, X)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, {out->id});

  ASSERT_EQ(origin_size, folded_size + 1);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTransposeToUseless) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t    = builder.Transpose(x, {0, 2, 1});
  auto x_2t    = builder.Transpose(x_1t, {0, 2, 1});
  auto x_3t    = builder.Transpose(x_2t, {0, 2, 1});
  auto out     = builder.Add(x_3t, x_3t);
  auto program = builder.Build();
  auto target  = GetTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_13 = transpose(X, axis=[0,2,1])
  //   var_14 = transpose(var_13, axis=[0,2,1])
  //   var_15 = transpose(var_14, axis=[0,2,1])
  //   var_16 = elementwise_add(var_15, var_15)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, {out->id});

  ProgramPass::Apply(&program, {}, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_15 = transpose(X, axis=[0,2,1])
  //   var_16 = elementwise_add(var_15, var_15)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, {out->id});

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTransposeWithMultiOutput) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t    = builder.Transpose(x, {0, 2, 1});
  auto x_2t    = builder.Transpose(x_1t, {2, 0, 1});
  auto x_3t    = builder.Transpose(x_2t, {1, 2, 0});
  auto out1    = builder.Sqrt(x_1t);
  auto out2    = builder.Sqrt(x_2t);
  auto out3    = builder.Sqrt(x_3t);
  auto program = builder.Build();
  auto target  = GetTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_18 = transpose(X, axis=[0,2,1])
  //   var_19 = transpose(var_18, axis=[2,0,1])
  //   var_20 = transpose(var_19, axis=[1,2,0])
  //   var_21 = sqrt(var_18)
  //   var_22 = sqrt(var_19)
  //   var_23 = sqrt(var_20)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, {out1->id, out2->id, out3->id});

  ProgramPass::Apply(&program, {}, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_18 = transpose(X, axis=[0,2,1])
  //   var_19 = transpose(X, axis=[1,0,2])
  //   var_20 = transpose(X, axis=[0,2,1])
  //   var_21 = sqrt(var_18)
  //   var_22 = sqrt(var_19)
  //   var_23 = sqrt(var_20)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, {out1->id, out2->id, out3->id});

  ASSERT_EQ(origin_size, folded_size);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

TEST(TransposeCollapsing, FuseTwoSecTranspose) {
  CinnBuilder builder("cinn_builder");
  auto x       = builder.CreateInput(Float(32), {4, 5, 3}, "X");
  auto x_1t    = builder.Transpose(x, {0, 2, 1});
  auto x_2t    = builder.Transpose(x_1t, {2, 0, 1});
  auto out1    = builder.Sqrt(x_2t);
  auto x_3t    = builder.Transpose(out1, {0, 2, 1});
  auto x_4t    = builder.Transpose(x_3t, {2, 0, 1});
  auto out2    = builder.Sqrt(x_4t);
  auto program = builder.Build();
  auto target  = GetTarget();

  size_t origin_size = program.size();
  VLOG(1) << "Program before pass:\n" << program;
  // Program {
  //   var_26 = transpose(X, axis=[0,2,1])
  //   var_27 = transpose(var_26, axis=[2,0,1])
  //   var_28 = sqrt(var_27)
  //   var_29 = transpose(var_28, axis=[0,2,1])
  //   var_30 = transpose(var_29, axis=[2,0,1])
  //   var_31 = sqrt(var_30)
  // }

  auto origin_out = RunWithProgram(program, target, {"X"}, {out1->id, out2->id});

  ProgramPass::Apply(&program, {}, target, {"TransposeCollapsing"});
  size_t folded_size = program.size();
  VLOG(1) << "Program after pass:\n" << program;
  // Program {
  //   var_27 = transpose(X, axis=[1,0,2])
  //   var_28 = sqrt(var_27)
  //   var_30 = transpose(var_28, axis=[1,0,2])
  //   var_31 = sqrt(var_30)
  // }

  auto folded_out = RunWithProgram(program, target, {"X"}, {out1->id, out2->id});

  ASSERT_EQ(origin_size, folded_size + 2);
  ASSERT_EQ(origin_out.size(), folded_out.size());
  for (size_t i = 0; i < origin_out.size(); ++i) {
    ASSERT_EQ(origin_out[i].size(), folded_out[i].size());
    for (size_t j = 0; j < origin_out[i].size(); ++j) {
      ASSERT_FLOAT_EQ(origin_out[i][j], folded_out[i][j]);
    }
  }
}

}  // namespace cinn::frontend
