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

#include "cinn/hlir/framework/parallel_compiler.h"

#include <gtest/gtest.h>

#include "cinn/common/target.h"
#include "cinn/frontend/net_builder.h"
#include "cinn/frontend/optimize.h"
#include "cinn/hlir/framework/graph_compiler.h"

DECLARE_int32(cinn_parallel_compile_size);

namespace cinn {
namespace hlir {
namespace framework {

using namespace frontend;

TEST(ParallelCompilerTest, Add_TEST_0) {
  frontend::NetBuilder builder("Add_TEST_0");
  auto A       = builder.CreateInput(Float(32), {128, 128}, "A");
  auto B       = builder.CreateInput(Float(32), {128, 128}, "B");
  auto C       = builder.Add(A, B);
  auto target  = common::DefaultNVGPUTarget();
  auto program = builder.Build();
  auto graph   = std::make_shared<Graph>(program, target);
  auto scope   = BuildScope(target, graph);

  ParallelCompiler::CompileOptions option;
  ParallelCompiler pc(scope, graph, option, target);
  auto runtime_program = pc();
}

TEST(ParallelCompilerTest, Conv2d_Test_0) {
  frontend::NetBuilder builder("Conv2d_Test_0");
  auto A = builder.CreateInput(Float(32), {1, 64, 112, 112}, "A");
  auto B = builder.CreateInput(Float(32), {64, 64, 3, 3}, "B");
  auto C = builder.CreateInput(Float(32), {1, 64, 56, 56}, "C");
  auto D = builder.Conv2d(A, B, {2, 2}, {1, 1});
  auto E = builder.Add(C, D);

  auto target  = common::DefaultNVGPUTarget();
  auto program = builder.Build();
  auto graph   = Optimize(&program, {}, target);
  auto scope   = BuildScope(target, graph);

  ParallelCompiler::CompileOptions option;
  ParallelCompiler pc(scope, graph, option, target);
  auto runtime_program = pc();
}

TEST(ParallelCompilerTest, Matmul_Test_0) {
  frontend::NetBuilder builder("Matmul_Test_0");
  auto A = builder.CreateInput(Float(32), {64, 128, 128}, "A");
  auto B = builder.CreateInput(Float(32), {64, 128, 128}, "B");
  auto C = builder.CreateInput(Float(32), {64, 128, 128}, "C");
  auto D = builder.Matmul(A, B);
  auto E = builder.Add(C, D);

  auto target  = common::DefaultNVGPUTarget();
  auto program = builder.Build();
  auto graph   = Optimize(&program, {}, target);
  auto scope   = BuildScope(target, graph);

  ParallelCompiler::CompileOptions option;
  ParallelCompiler pc(scope, graph, option, target);
  auto runtime_program = pc();
}

TEST(ParallelCompilerTest, parallel_compile_8) {
  int old_value                    = FLAGS_cinn_parallel_compile_size;
  FLAGS_cinn_parallel_compile_size = 8;

  NetBuilder net_builder("Reduce_Fuse_Broadcast_With_Output");
  auto layer_norm_51__tmp_1 = net_builder.CreateInput(Float(32), {256}, "layer_norm_51__tmp_1");
  auto var_3216             = net_builder.CreateInput(Float(32), {60, 60}, "var_3216");
  auto var_3202             = net_builder.CreateInput(Float(32), {1, 60}, "var_3202");
  auto var_3212             = net_builder.CreateInput(Float(32), {256, 60}, "var_3212");

  auto var_3206         = net_builder.Reshape(layer_norm_51__tmp_1, {256, 1});
  auto composite_tmp_8  = net_builder.FillConstant<float>({256, 1}, 1e-5, "composite_tmp_8");
  auto var_3214         = net_builder.Add(var_3206, composite_tmp_8);
  auto composite_tmp_10 = net_builder.FillConstant<float>({256, 1}, 1.0, "composite_tmp_10");
  auto var_3220         = net_builder.Divide(composite_tmp_10, var_3214);
  auto var_3226         = net_builder.Sqrt(var_3220);
  auto var_3224         = net_builder.Scale(var_3220, -1.0, 0.0, true);
  auto var_3366         = net_builder.BroadcastTo(var_3224, {256, 60});
  auto var_3228         = net_builder.Matmul(var_3366, var_3216);
  auto var_3368         = net_builder.BroadcastTo(var_3202, {256, 60});
  auto var_3236         = net_builder.Multiply(var_3228, var_3212);
  auto var_3244         = net_builder.Multiply(var_3236, var_3368);
  auto var_3252         = net_builder.ReduceSum(var_3244, {1}, true);
  auto var_3232         = net_builder.Scale(var_3226, 0.0166667, 0.0, true);

  auto target  = common::DefaultNVGPUTarget();
  auto program = net_builder.Build();
  auto graph   = Optimize(&program, {var_3252->id, var_3232->id}, target);
  auto scope   = BuildScope(target, graph);

  ParallelCompiler::CompileOptions option;
  ParallelCompiler pc(scope, graph, option, target);
  auto runtime_program             = pc();
  FLAGS_cinn_parallel_compile_size = old_value;
}

TEST(ParallelCompilerTest, parallel_compile_1) {
  int old_value                    = FLAGS_cinn_parallel_compile_size;
  FLAGS_cinn_parallel_compile_size = 1;

  NetBuilder net_builder("Reduce_Fuse_Broadcast_With_Output");
  auto layer_norm_51__tmp_1 = net_builder.CreateInput(Float(32), {256}, "layer_norm_51__tmp_1");
  auto var_3216             = net_builder.CreateInput(Float(32), {60, 60}, "var_3216");
  auto var_3202             = net_builder.CreateInput(Float(32), {1, 60}, "var_3202");
  auto var_3212             = net_builder.CreateInput(Float(32), {256, 60}, "var_3212");

  auto var_3206         = net_builder.Reshape(layer_norm_51__tmp_1, {256, 1});
  auto composite_tmp_8  = net_builder.FillConstant<float>({256, 1}, 1e-5, "composite_tmp_8");
  auto var_3214         = net_builder.Add(var_3206, composite_tmp_8);
  auto composite_tmp_10 = net_builder.FillConstant<float>({256, 1}, 1.0, "composite_tmp_10");
  auto var_3220         = net_builder.Divide(composite_tmp_10, var_3214);
  auto var_3226         = net_builder.Sqrt(var_3220);
  auto var_3224         = net_builder.Scale(var_3220, -1.0, 0.0, true);
  auto var_3366         = net_builder.BroadcastTo(var_3224, {256, 60});
  auto var_3228         = net_builder.Matmul(var_3366, var_3216);
  auto var_3368         = net_builder.BroadcastTo(var_3202, {256, 60});
  auto var_3236         = net_builder.Multiply(var_3228, var_3212);
  auto var_3244         = net_builder.Multiply(var_3236, var_3368);
  auto var_3252         = net_builder.ReduceSum(var_3244, {1}, true);
  auto var_3232         = net_builder.Scale(var_3226, 0.0166667, 0.0, true);

  auto target  = common::DefaultNVGPUTarget();
  auto program = net_builder.Build();
  auto graph   = Optimize(&program, {var_3252->id, var_3232->id}, target);
  auto scope   = BuildScope(target, graph);

  ParallelCompiler::CompileOptions option;
  ParallelCompiler pc(scope, graph, option, target);
  auto runtime_program             = pc();
  FLAGS_cinn_parallel_compile_size = old_value;
}

TEST(ParallelCompilerTest, parallel_compile_0) {
  int old_value                    = FLAGS_cinn_parallel_compile_size;
  FLAGS_cinn_parallel_compile_size = 0;

  NetBuilder net_builder("Reduce_Fuse_Broadcast_With_Output");
  auto layer_norm_51__tmp_1 = net_builder.CreateInput(Float(32), {256}, "layer_norm_51__tmp_1");
  auto var_3216             = net_builder.CreateInput(Float(32), {60, 60}, "var_3216");
  auto var_3202             = net_builder.CreateInput(Float(32), {1, 60}, "var_3202");
  auto var_3212             = net_builder.CreateInput(Float(32), {256, 60}, "var_3212");

  auto var_3206         = net_builder.Reshape(layer_norm_51__tmp_1, {256, 1});
  auto composite_tmp_8  = net_builder.FillConstant<float>({256, 1}, 1e-5, "composite_tmp_8");
  auto var_3214         = net_builder.Add(var_3206, composite_tmp_8);
  auto composite_tmp_10 = net_builder.FillConstant<float>({256, 1}, 1.0, "composite_tmp_10");
  auto var_3220         = net_builder.Divide(composite_tmp_10, var_3214);
  auto var_3226         = net_builder.Sqrt(var_3220);
  auto var_3224         = net_builder.Scale(var_3220, -1.0, 0.0, true);
  auto var_3366         = net_builder.BroadcastTo(var_3224, {256, 60});
  auto var_3228         = net_builder.Matmul(var_3366, var_3216);
  auto var_3368         = net_builder.BroadcastTo(var_3202, {256, 60});
  auto var_3236         = net_builder.Multiply(var_3228, var_3212);
  auto var_3244         = net_builder.Multiply(var_3236, var_3368);
  auto var_3252         = net_builder.ReduceSum(var_3244, {1}, true);
  auto var_3232         = net_builder.Scale(var_3226, 0.0166667, 0.0, true);

  auto target  = common::DefaultNVGPUTarget();
  auto program = net_builder.Build();
  auto graph   = Optimize(&program, {var_3252->id, var_3232->id}, target);
  auto scope   = BuildScope(target, graph);

  ParallelCompiler::CompileOptions option;
  ParallelCompiler pc(scope, graph, option, target);
  auto runtime_program             = pc();
  FLAGS_cinn_parallel_compile_size = old_value;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
