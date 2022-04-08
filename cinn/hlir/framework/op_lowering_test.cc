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

#include "cinn/hlir/framework/op_lowering.h"

#include "cinn/backends/codegen_c_x86.h"
#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/common/target.h"
#include "cinn/frontend/decomposer/test_helper.h"

namespace cinn {
namespace hlir {
namespace framework {

using namespace frontend;

void CodeGen(ir::LoweredFunc& func) {
#ifdef CINN_WITH_CUDA
  auto target = common::DefaultNVGPUTarget();
  Module::Builder builder("Module_Builder", target);
  builder.AddFunction(func);

  auto module                    = builder.Build();
  auto host_module_device_module = backends::SplitCudaAndHostModule(module);
  auto& host_module              = std::get<0>(host_module_device_module);
  auto& device_module            = std::get<1>(host_module_device_module);

  backends::CodeGenCUDA_Dev codegen(target);
  auto source_code = codegen.Compile(builder.Build());
  LOG(INFO) << "compiled code:\n\n\n" << source_code;

  // nv jit compile to ptx
  backends::NVRTC_Compiler compiler;
  auto ptx = compiler(source_code);
  CHECK(!ptx.empty());
#else
  auto target = common::DefaultHostTarget();
  ir::Module::Builder builder("Module_Builder", target);
  builder.AddFunction(func);

  CodeGenCX86 codegen(target, CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "compiled code:\n\n\n" << source_code;
#endif
}

TEST(OP_LOWERING, Elementwise_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Elementwise_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(C, D);
    auto G = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
  }
}

TEST(OP_LOWERING, Elementwise_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Elementwise_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(A, B);
    auto F = net_builder.ElementwiseAdd(E, C);
    auto G = net_builder.ElementwiseAdd(E, D);
    auto H = net_builder.ElementwiseAdd(F, G);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
  }
}

TEST(OP_LOWERING, Broadcast_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Broadcast_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.ElementwiseAdd(C, A);
    auto F = net_builder.ElementwiseAdd(D, B);
    auto G = net_builder.ElementwiseAdd(E, F);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
  }
}

TEST(OP_LOWERING, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w, h}, "A");
    auto B = net_builder.Reduce(A, ReduceKind::kSum, {0, 1});
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");

    auto C = net_builder.Reduce(A, ReduceKind::kSum, {0});
    auto D = net_builder.ElementwiseAdd(B, C);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto D = net_builder.ElementwiseAdd(A, B);
    auto E = net_builder.Reduce(D, ReduceKind::kSum, {1});
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {w}, "D");

    auto E = net_builder.Reduce(A, ReduceKind::kSum, {0});
    auto F = net_builder.ElementwiseAdd(B, C);
    auto G = net_builder.ElementwiseAdd(D, F);
    auto H = net_builder.ElementwiseAdd(E, G);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.Reduce(A, ReduceKind::kSum, {0});
    auto C = net_builder.Reduce(A, ReduceKind::kSum, {0});
    auto D = net_builder.ElementwiseAdd(B, C);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    LOG(INFO) << lowered_func[0];
    CodeGen(lowered_func[0]);
  }
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn
