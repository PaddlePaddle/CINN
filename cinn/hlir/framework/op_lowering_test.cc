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
  Module::Builder builder("module_builder", target);

  builder.AddFunction(func);
  auto module   = builder.Build();
  auto compiler = backends::Compiler::Create(target);

  std::string code = "";
  compiler->Build(module, code);
#else
  auto target = common::DefaultHostTarget();
  ir::Module::Builder builder("Module_Builder", target);
  builder.AddFunction(func);

  CodeGenCX86 codegen(target, CodeGenCX86::Feature::AVX512);
  codegen.SetInlineBuiltinCodes(false);
  auto source_code = codegen.Compile(builder.Build(), CodeGenC::OutputKind::CImpl);
  LOG(INFO) << "compiled code of " << func->name << "is:\n\n\n" << source_code;
#endif
}

TEST(OP_LOWERING, OpaqueOp_TEST_0) {
  NetBuilder net_builder("OpaqueOp_TEST_0");
  {
    auto A = net_builder.CreateInput(Float(32), {9801, 2}, "A");
    auto B = net_builder.Reshape(A, {9801, 2});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, OpaqueOp_TEST_1) {
  NetBuilder net_builder("OpaqueOp_TEST_1");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128, 128}, "B");
    auto C = net_builder.Matmul(A, B);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, OpaqueOp_TEST_2) {
  NetBuilder net_builder("OpaqueOp_TEST_2");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.Matmul(A, A);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, OpaqueOp_TEST_3) {
  NetBuilder net_builder("OpaqueOp_TEST_3");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 256}, "A");
    auto C = net_builder.Split(A, {4}, 1);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

#ifdef CINN_WITH_CUDA
TEST(OP_LOWERING, OpaqueOp_TEST_4) {
  NetBuilder net_builder("OpaqueOp_TEST_4");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128, 128}, "B");
    auto C = net_builder.CreateInput(Float(32), {128, 128}, "C");
    auto D = net_builder.Matmul(A, B);
    auto E = net_builder.Add(C, D);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "MatmulToCublasCustomCallPass");
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}
#endif

TEST(OP_LOWERING, Transform_TEST_0) {
  NetBuilder net_builder("Transform_TEST_0");
  {
    auto A = net_builder.CreateInput(Float(32), {128, 128}, "A");
    auto B = net_builder.CreateInput(Float(32), {128, 128}, "B");
    auto C = net_builder.CreateInput(Float(32), {128, 128}, "C");
    auto D = net_builder.Concat({A, B, C}, 1);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
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
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(C, D);
    auto G = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
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
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);
    auto H = net_builder.Add(F, G);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Elementwise_Test_2) {
  int h = 50, w = 10201;
  NetBuilder net_builder("Elementwise_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h, w}, "D");
    auto E = net_builder.Add(A, B);
    auto F = net_builder.Add(E, C);
    auto G = net_builder.Add(E, D);
    auto H = net_builder.Add(F, G);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w, h}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_1) {
  int c = 32, h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1, 2});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_2) {
  int c = 32, h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_3) {
  int c = 32, h = 16, w = 16;
  NetBuilder net_builder("Reduce_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 1, 2});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {w, h}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_5) {
  int h = 32, w = 768;
  NetBuilder net_builder("Reduce_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_6) {
  int h = 32, w = 2048;
  NetBuilder net_builder("Reduce_Test_6");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_7) {
  int h = 32, w = 512;
  NetBuilder net_builder("Reduce_Test_7");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_8) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_8");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w, w}, "A");
    auto B = net_builder.ReduceSum(A, {1, 2});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_9) {
  int n = 16, c = 128, h = 56, w = 56;
  NetBuilder net_builder("Reduce_Test_9");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0, 2, 3});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Test_10) {
  int n = 16, c = 16, h = 32, w = 32;
  NetBuilder net_builder("Reduce_Test_10");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.ReduceSum(A, {1});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_0) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_0");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");

    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.Add(B, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
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
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_1) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_1");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {1});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
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
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_2) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_2");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.CreateInput(Float(32), {w}, "D");

    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.Add(B, C);
    auto G = net_builder.Add(D, F);
    auto H = net_builder.Add(E, G);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
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
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_3) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_3");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.ReduceSum(A, {0});
    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.Add(B, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
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
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_4) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_4");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);

    auto D = net_builder.ReduceSum(C, {0});
    auto E = net_builder.ReduceSum(C, {0});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_5) {
  int h = 32, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_5");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.Add(A, B);

    auto D = net_builder.ReduceSum(C, {1});
    auto E = net_builder.ReduceSum(C, {1});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_6) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_6");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0});
    auto F = net_builder.ReduceSum(D, {0});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_7) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_7");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {w}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {1});
    auto F = net_builder.ReduceSum(D, {1});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_8) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_8");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {1}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 1});
    auto F = net_builder.ReduceSum(D, {0, 1});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_9) {
  int c = 128, h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_9");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {c, h, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h}, "C");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2});
    auto F = net_builder.ReduceSum(D, {0, 2});
    auto G = net_builder.Add(E, C);
    auto I = net_builder.Add(F, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 5);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_10) {
  int h = 10201, w = 50;
  NetBuilder net_builder("Reduce_Fusion_Test_10");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {w}, "B");
    auto C = net_builder.ReduceSum(A, {0});
    auto D = net_builder.Add(B, C);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_11) {
  int n = 128, c = 128, h = 16, w = 16;
  NetBuilder net_builder("Reduce_Fusion_Test_11");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2, 3});
    auto F = net_builder.ReduceSum(D, {0, 2, 3});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}

TEST(OP_LOWERING, Reduce_Fusion_Test_12) {
  int n = 128, c = 128, h = 112, w = 112;
  NetBuilder net_builder("Reduce_Fusion_Test_12");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2, 3});
    auto F = net_builder.ReduceSum(D, {0, 2, 3});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  auto& dtype_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, Type>>("inferdtype");
  auto& shape_dict = graph->GetMutableAttrs<absl::flat_hash_map<std::string, shape_t>>("infershape");

  OpLowerer op_lowerer(dtype_dict, shape_dict, target);
  for (auto& fusion_op : graph->fusion_groups) {
    auto lowered_func = op_lowerer.Lower(fusion_op);
    CHECK_EQ(lowered_func.size(), 1);
    CodeGen(lowered_func[0]);
  }
}
/*
TODO:exist coredump.
TEST(OP_LOWERING, Reduce_Fusion_Test_13) {
  int n = 8, c = 8, h = 8, w = 8;
  NetBuilder net_builder("Reduce_Fusion_Test_13");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 1, 2});
    auto F = net_builder.ReduceSum(D, {0, 1, 2});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
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
*/

TEST(OP_LOWERING, Reduce_Fusion_Test_14) {
  int n = 8, c = 8, h = 8, w = 8;
  NetBuilder net_builder("Reduce_Fusion_Test_14");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, n, n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, n, n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 3, 4});
    auto F = net_builder.ReduceSum(D, {0, 3, 4});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
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

TEST(OP_LOWERING, Reduce_Fusion_Test_15) {
  int h = 512, w = 32;
  NetBuilder net_builder("Reduce_Fusion_Test_15");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0});
    auto F = net_builder.ReduceSum(D, {0});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
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
TEST(OP_LOWERING, Reduce_Fusion_Test_16) {
  int n = 128, c = 128, h = 28, w = 28;
  NetBuilder net_builder("Reduce_Fusion_Test_16");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {n, c, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {n, c, h, w}, "B");
    auto D = net_builder.Add(A, B);
    auto E = net_builder.ReduceSum(D, {0, 2, 3});
    auto F = net_builder.ReduceSum(D, {0, 2, 3});
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 3);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
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

TEST(OP_LOWERING, Reduce_Fusion_Test_17) {
  int h = 128, w = 768;
  NetBuilder net_builder("Reduce_Fusion_Test_17");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
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

TEST(OP_LOWERING, Reduce_Fusion_Test_18) {
  int h = 128, w = 768;
  NetBuilder net_builder("Reduce_Fusion_Test_18");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {16, h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {16, h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {1});
    auto F = net_builder.ReduceSum(B, {1});
    auto G = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  // hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  // CHECK_EQ(graph->fusion_groups.size(), 1);

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

TEST(OP_LOWERING, Reduce_Fusion_Test_19) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_19");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.Add(E, F);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  // hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  // CHECK_EQ(graph->fusion_groups.size(), 1);

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

TEST(OP_LOWERING, Reduce_Fusion_Test_20) {
  int h = 128, w = 128;
  NetBuilder net_builder("Reduce_Fusion_Test_20");
  // create model
  {
    auto A = net_builder.CreateInput(Float(32), {h, w}, "A");
    auto B = net_builder.CreateInput(Float(32), {h * 2, w}, "B");
    auto C = net_builder.CreateInput(Float(32), {h * 3, w}, "C");
    auto D = net_builder.CreateInput(Float(32), {h * 4, w}, "D");
    auto E = net_builder.ReduceSum(A, {0});
    auto F = net_builder.ReduceSum(B, {0});
    auto G = net_builder.ReduceSum(C, {0});
    auto H = net_builder.ReduceSum(D, {0});
    auto I = net_builder.Add(E, F);
    auto J = net_builder.Add(G, I);
    auto K = net_builder.Add(H, J);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 1);

  // hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
  // CHECK_EQ(graph->fusion_groups.size(), 1);

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

TEST(OP_LOWERING, Reduce_Fusion_Test_21) {
  int h = 128, w = 4;
  NetBuilder net_builder("Reduce_Fusion_Test_21");
  // create model
  {
    auto A0  = net_builder.CreateInput(Float(32), {256, w}, "A0");
    auto B0  = net_builder.CreateInput(Float(32), {256, w}, "B0");
    auto C0  = net_builder.CreateInput(Float(32), {55200, w}, "C0");
    auto D0  = net_builder.CreateInput(Float(32), {2750, w}, "D0");
    auto A1  = net_builder.CreateInput(Float(32), {256, w}, "A1");
    auto B1  = net_builder.CreateInput(Float(32), {256, w}, "B1");
    auto C1  = net_builder.CreateInput(Float(32), {55200, w}, "C1");
    auto D1  = net_builder.CreateInput(Float(32), {2750, w}, "D1");
    auto AA  = net_builder.Add(A0, A1);
    auto BB  = net_builder.Add(B0, B1);
    auto CC  = net_builder.Add(C0, C1);
    auto DD  = net_builder.Add(D0, D1);
    auto E   = net_builder.ReduceSum(AA, {0});
    auto F   = net_builder.ReduceSum(BB, {0});
    auto G   = net_builder.ReduceSum(CC, {0});
    auto H   = net_builder.ReduceSum(DD, {0});
    auto I   = net_builder.Add(E, F);
    auto J   = net_builder.Add(G, I);
    auto K   = net_builder.Add(H, J);
    auto AAA = net_builder.Add(AA, A1);
    auto BBB = net_builder.Add(BB, B1);
    auto CCC = net_builder.Add(CC, C1);
    auto DDD = net_builder.Add(DD, D1);
  }

  auto program = net_builder.Build();
  auto target  = common::DefaultTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 9);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
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

TEST(OP_LOWERING, Reduce_Test_22) {
  int h = 128, w = 4;
  NetBuilder net_builder("Reduce_Test_22");
  // create model
  {
    auto A0  = net_builder.CreateInput(Float(32), {256, w}, "A0");
    auto B0  = net_builder.CreateInput(Float(32), {2052, w}, "B0");
    auto C0  = net_builder.CreateInput(Float(32), {2750, w}, "C0");
    auto D0  = net_builder.CreateInput(Float(32), {2750, w}, "D0");
    auto A1  = net_builder.CreateInput(Float(32), {256, w}, "A1");
    auto B1  = net_builder.CreateInput(Float(32), {2052, w}, "B1");
    auto C1  = net_builder.CreateInput(Float(32), {2750, w}, "C1");
    auto D1  = net_builder.CreateInput(Float(32), {2750, w}, "D1");
    auto AA  = net_builder.ElementwiseAdd(A0, A1);
    auto BB  = net_builder.ElementwiseAdd(B0, B1);
    auto CC  = net_builder.ElementwiseAdd(C0, C1);
    auto DD  = net_builder.ElementwiseAdd(D0, D1);
    auto E   = net_builder.Reduce(AA, ReduceKind::kSum, {0});
    auto F   = net_builder.Reduce(BB, ReduceKind::kSum, {0});
    auto G   = net_builder.Reduce(CC, ReduceKind::kSum, {0});
    auto H   = net_builder.Reduce(DD, ReduceKind::kSum, {0});
    auto I   = net_builder.ElementwiseAdd(E, F);
    auto J   = net_builder.ElementwiseAdd(G, I);
    auto K   = net_builder.ElementwiseAdd(H, J);
    auto AAA = net_builder.ElementwiseAdd(AA, A1);
    auto BBB = net_builder.ElementwiseAdd(BB, B1);
    auto CCC = net_builder.ElementwiseAdd(CC, C1);
    auto DDD = net_builder.ElementwiseAdd(DD, D1);
  }

  auto program = net_builder.Build();
  auto target  = GetTarget();
  RunDecomposer(&program, target);

  auto graph = std::make_shared<hlir::framework::Graph>(program, target);
  hlir::framework::ApplyPass(graph.get(), "OpFusionPass");
  CHECK_EQ(graph->fusion_groups.size(), 9);

  hlir::framework::ApplyPass(graph.get(), "FusionMergePass");
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
