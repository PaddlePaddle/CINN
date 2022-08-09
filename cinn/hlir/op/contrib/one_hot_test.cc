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

// #include "cinn/hlir/op/contrib/one_hot.h"

// #include <glog/logging.h>
// #include <gtest/gtest.h>

// #include <string>
// #include <vector>

// #include "cinn/backends/codegen_c.h"
// #include "cinn/backends/codegen_c_x86.h"
// #include "cinn/backends/codegen_cuda_dev.h"
// #include "cinn/backends/cuda_util.h"
// #include "cinn/common/context.h"
// #include "cinn/lang/lower.h"
// #include "cinn/lang/placeholder.h"
// #include "cinn/poly/stage.h"

#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <string>

#include "cinn/backends/codegen_cuda_dev.h"
#include "cinn/backends/codegen_cuda_host.h"
#include "cinn/backends/codegen_cuda_util.h"
#include "cinn/backends/cuda_util.h"
#include "cinn/backends/llvm/execution_engine.h"
#include "cinn/backends/llvm/runtime_symbol_registry.h"
#include "cinn/backends/llvm/simple_jit.h"
#include "cinn/backends/nvrtc_util.h"
#include "cinn/cinn.h"
#include "cinn/common/target.h"
#include "cinn/common/test_helper.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/use_ops.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/cuda/cuda_module.h"

namespace cinn {
namespace hlir {
namespace op {

// using common::_CINNValuePack_;
// using common::CINNValue;
// using common::CINNValuePack;
// using framework::OpStrategy;
// using framework::shape_t;
// using framework::StrategyFunction;

// void GenOneHotCode(const std::string& device) {
//   common::Context::Global().ResetNameId();

//   common::Target target;
//   if (device == "cuda") {
//     target = common::DefaultNVGPUTarget();
//   } else if (device == "cpu") {
//     target = common::DefaultHostTarget();
//   } else {
//     LOG(FATAL) << "Unknown device: " << device;
//   }

//   int depth                           = 9;
//   int axis                            = -1;
//   const std::string dtype             = "float32";
//   std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};

//   lang::Placeholder<float> indices("indices", indices_shape);
//   lang::Placeholder<float> on_value("on_value", {Expr(1)});
//   lang::Placeholder<float> off_value("off_value", {Expr(1)});
//   ir::Tensor res = OneHot(indices, on_value, off_value, depth, axis, dtype, "test_one_hot");

//   poly::StageMap stages = poly::CreateStages({res});
//   if (device == "cuda") {
//     stages[res]->Bind(0, "blockIdx.x");
//     stages[res]->Bind(1, "threadIdx.y");
//     stages[res]->SetBuffer("global");
//   }
//   std::vector<ir::LoweredFunc> funcs =
//       lang::LowerVec("TestGenerateCodeCpu_OneHot", stages, {res}, {}, {}, nullptr, target, true);

//   VLOG(6) << "Expr before " << device << " codegen:";
//   VLOG(6) << funcs[0]->body;

//   ir::Module::Builder builder("OneHot_Module", target);
//   for (auto& f : funcs) {
//     builder.AddFunction(f);
//   }

//   if (device == "cuda") {
//     backends::CodeGenCUDA_Dev codegen(target);
//     std::string code = codegen.Compile(builder.Build());
//     VLOG(6) << "Cuda Codegen result:";
//     VLOG(6) << code << std::endl;
//   } else {
//     backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
//     codegen.SetInlineBuiltinCodes(false);
//     std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
//     VLOG(6) << "Cpu Codegen result:";
//     VLOG(6) << code << std::endl;
//   }
// }

// TEST(GenerateCode_Cpu, OneHot) { GenOneHotCode("cpu"); }
// #ifdef CINN_WITH_CUDA
// TEST(GenerateCode_Cuda, OneHot) { GenOneHotCode("cuda"); }
// #endif

// TEST(GenerateCode_Cuda, OntHot) {
void GenOneHotCode(std::vector<ir::Expr> indices_shape,
                   int axis,
                   int depth,
                   const std::string& dtype,
                   std::vector<int> out_shape,
                   Type out_type,
                   const std::string& device) {
  // code gen
  auto one_hot = cinn::hlir::framework::Operator::Get("one_hot");
  auto strategy =
      cinn::hlir::framework::Operator::GetAttrs<cinn::hlir::framework::StrategyFunction>("CINNStrategy")[one_hot];

  lang::Placeholder<float> indices("indices", indices_shape);
  lang::Placeholder<float> on_value("on_value", {Expr(1)});
  lang::Placeholder<float> off_value("off_value", {Expr(1)});

  // set attrs
  cinn::hlir::framework::NodeAttr attrs;
  attrs.attr_store["axis"]  = axis;
  attrs.attr_store["depth"] = depth;
  attrs.attr_store["dtype"] = dtype;

  std::vector<ir::Tensor> inputs{indices.tensor(), on_value.tensor(), off_value.tensor()};

  common::Target target;
  if (device == "cuda") {
    target = common::DefaultNVGPUTarget();
  } else if (device == "cpu") {
    target = common::DefaultHostTarget();
  } else {
    LOG(FATAL) << "Unknown device: " << device;
    return;
  }

  auto impl = cinn::hlir::framework::OpStrategy::SelectImpl(strategy(attrs, inputs, {out_type}, {out_shape}, target));

  common::CINNValuePack cinn_input = common::CINNValuePack{{common::CINNValue(indices.tensor()),
                                                            common::CINNValue(on_value.tensor()),
                                                            common::CINNValue(off_value.tensor())}};
  common::CINNValuePack rets       = impl->fcompute(cinn_input);
  rets                             = impl->fschedule(rets);

  // the last element is a StageMap
  for (int i = 0; i < rets->size() - 1; i++) {
    Expr temp = rets[i];
    if (!temp.as_tensor_ref()->buffer.defined()) {
      inputs.push_back(temp.as_tensor_ref());
    }
  }

  auto func = lang::LowerVec("one_hot", rets.back(), inputs, {}, {}, nullptr, target);
  for (auto& f : func) {
    LOG(INFO) << "Test Strategy Codegen:\n" << f;
  }

  ir::Module::Builder builder("OneHot_Module", target);
  for (auto& f : func) {
    builder.AddFunction(f);
  }

  if (device == "cuda") {
    backends::CodeGenCUDA_Dev codegen(target);
    std::string code = codegen.Compile(builder.Build());
    VLOG(6) << "Cuda Codegen result:";
    VLOG(6) << code << std::endl;
  } else {
    backends::CodeGenCX86 codegen(target, backends::CodeGenCX86::Feature::AVX512);
    codegen.SetInlineBuiltinCodes(false);
    std::string code = codegen.Compile(builder.Build(), backends::CodeGenC::OutputKind::CImpl);
    VLOG(6) << "Cpu Codegen result:";
    VLOG(6) << code << std::endl;
  }
}

TEST(OneHotOpTest, OneHotTest_Base) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = -1;
  int depth                           = 9;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 9};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Axis_1) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = 1;
  int depth                           = 9;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 9, 3, 4, 5};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Axis_Ndim) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = 4;
  int depth                           = 9;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 9};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Depth) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = -1;
  int depth                           = 20;
  const std::string dtype             = "float32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 20};
  Type out_type                       = Float(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cuda");
#endif
}

TEST(OneHotOpTest, OneHotTest_Dtype) {
  std::vector<ir::Expr> indices_shape = {Expr(2), Expr(3), Expr(4), Expr(5)};
  int axis                            = -1;
  int depth                           = 9;
  const std::string dtype             = "int32";
  std::vector<int> out_shape          = {2, 3, 4, 5, 9};
  Type out_type                       = Int(32);
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cpu");
#ifdef CINN_WITH_CUDA
  GenOneHotCode(indices_shape, axis, depth, dtype, out_shape, out_type, "cuda");
#endif
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn